#include "mlnd/bdd.hpp"
#include "mlnd/strengthening.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mlnd {
namespace {
std::size_t index(BddIntegrality mode) {
    switch (mode) {
        case BddIntegrality::Continuous: return 0;
        case BddIntegrality::FacilityCapacity: return 1;
        case BddIntegrality::AllDesign: return 2;
    }
    throw std::invalid_argument("Unknown BDD integrality mode");
}
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("BDD vector dimension mismatch");
    long double value = 0;
    for (std::size_t j = 0; j < a.size(); ++j) value += static_cast<long double>(a[j]) * b[j];
    return static_cast<double>(value);
}
bool finite_bound(double value) {
    return std::isfinite(value) && std::abs(value) < .1 * IloInfinity;
}
bool permitted_mip_status(int code) {
    // Exclude numerical / infeasibility / relaxation-failure terminations.
    // Both Concert-normalized and raw CPLEX MIP codes are accepted here.
    return code == IloCplex::Optimal || code == IloCplex::OptimalTol ||
        code == IloCplex::AbortTimeLim || code == IloCplex::AbortDetTimeLim ||
        code == IloCplex::NodeLimFeas || code == IloCplex::NodeLimInfeas ||
        code == IloCplex::SolLim || code == IloCplex::AbortUser ||
        code == CPXMIP_OPTIMAL || code == CPXMIP_TIME_LIM_FEAS ||
        code == CPXMIP_TIME_LIM_INFEAS || code == CPXMIP_ABORT_FEAS ||
        code == CPXMIP_ABORT_INFEAS;
}
}

const char* bdd_integrality_name(BddIntegrality mode) {
    switch (mode) {
        case BddIntegrality::Continuous: return "continuous";
        case BddIntegrality::FacilityCapacity: return "facility_capacity";
        case BddIntegrality::AllDesign: return "all_design";
    }
    throw std::invalid_argument("Unknown BDD integrality mode");
}

BddAnchor bdd_anchor_from_json(const Json& value) {
    BddAnchor anchor;
    anchor.name = value.value("name", value.value("candidate_id", std::string()));
    anchor.design = value.at("design").get<std::vector<double>>();
    anchor.intercept = value.at("intercept").get<double>();
    anchor.coefficients = value.at("coefficients").get<std::vector<double>>();
    anchor.operating_objective = value.at("operating_objective").get<double>();
    anchor.pricing_complete = value.at("pricing_complete").get<bool>();
    return anchor;
}

struct BddFullDomain::Impl {
    struct Variant {
        LinearModel matrix;
        std::unique_ptr<NativeSolver> solver;
        std::size_t calls = 0;
        bool mixed = false;
    };
    Formulation original;
    LinearModel relaxed_design;
    std::array<Variant,3> variants;
    Json setup;

    Impl(const Input& data, const PathCollection& paths, const SolverOptions& supplied)
        : original(data) {
        if (!paths.complete || !paths.support_complete)
            throw std::invalid_argument("BDD requires the complete physical path domain, never a restricted column pool");
        if (supplied.threads != 1)
            throw std::invalid_argument("BDD diagnostic requires one solver thread");
        const double started = wall_time();
        original = build_formulation(data, paths, nullptr, nullptr, nullptr, false, false, supplied.tight_airline_bounds,
                                     formulation_strengthening(supplied));
        if (!original.design_fix_rows.empty()) throw std::logic_error("BDD copied design was accidentally fixed");
        const double record_seconds = wall_time() - started;
        relaxed_design = design_model(data, false, formulation_strengthening(supplied)).matrix;
        setup = {{"scope", "complete_physical_path_domain"}, {"catalogue_complete", true},
            {"support_complete", true}, {"route_scope", paths.scope},
            {"service_paths", paths.service_paths.size()}, {"empty_paths", paths.empty_paths.size()},
            {"initial_segments", paths.initial_segments.size()},
            {"common_matrix_build_seconds", record_seconds}, {"solver_models", Json::array()},
            {"model_size", {{"variables", original.matrix.variables.size()},
                {"linear_rows", original.matrix.rows.size()}, {"cones", original.matrix.cones.size()},
                {"design_variables", original.layout.count}}},
            {"shared_matrix_and_solver_models", true}, {"restricted_column_bound_used", false},
            {"objective", "operating_cost - beta^T copied_design"},
            {"fixed_opening_costs_removed", true}, {"threads", supplied.threads}};
        SolverOptions options = supplied;
        options.gap = 0;
        for (const auto mode : {BddIntegrality::Continuous, BddIntegrality::FacilityCapacity,
                                BddIntegrality::AllDesign}) {
            const double phase = wall_time();
            auto& variant = variants[index(mode)];
            variant.matrix = original.matrix;
            int binaries = 0;
            for (int j = 0; j < original.layout.count; ++j) {
                auto& variable = variant.matrix.variables[j];
                variable.cost = 0; // Later replaced by -beta, never f-beta.
                variable.binary = mode == BddIntegrality::AllDesign ||
                    (mode == BddIntegrality::FacilityCapacity && j < original.layout.assignment);
                binaries += variable.binary;
            }
            variant.mixed = binaries > 0;
            const double copy_seconds = wall_time() - phase;
            const double solver_started = wall_time();
            variant.solver = std::make_unique<NativeSolver>(variant.matrix, options,
                std::string("bdd_") + bdd_integrality_name(mode));
            setup["solver_models"].push_back({{"mode", bdd_integrality_name(mode)},
                {"matrix_copy_seconds", copy_seconds}, {"binary_variables", binaries},
                {"solver_build_seconds", wall_time() - solver_started}});
        }
        setup["shared_setup_seconds"] = wall_time() - started;
    }
};

BddFullDomain::BddFullDomain(const Input& data, const PathCollection& paths, const SolverOptions& options)
    : impl_(std::make_unique<Impl>(data, paths, options)) {}
BddFullDomain::~BddFullDomain() = default;
Json BddFullDomain::setup_metadata() const { return impl_->setup; }

Json BddFullDomain::evaluate(const BddAnchor& anchor, BddIntegrality mode,
                            double seconds, double improvement_relative_tolerance) {
    const double started = wall_time();
    if (!std::isfinite(seconds) || seconds < 0 || !std::isfinite(improvement_relative_tolerance) ||
        improvement_relative_tolerance < 0)
        throw std::invalid_argument("Invalid BDD budget or improvement tolerance");
    const int count = impl_->original.layout.count;
    if (anchor.design.size() != static_cast<std::size_t>(count) ||
        anchor.coefficients.size() != anchor.design.size() || !anchor.pricing_complete ||
        !finite_bound(anchor.intercept) || !finite_bound(anchor.operating_objective) ||
        !std::all_of(anchor.coefficients.begin(), anchor.coefficients.end(), finite_bound))
        throw std::invalid_argument("BDD needs a finite, dimensionally consistent, fully priced ordinary anchor");
    const auto design_residuals = point_residuals(impl_->relaxed_design, anchor.design);
    if (!design_residuals.at("passed").get<bool>())
        throw std::invalid_argument("BDD anchor violates the original relaxed strategic constraints");
    const double anchor_dot = dot(anchor.coefficients, anchor.design);
    const double ordinary_value = anchor.intercept + anchor_dot;
    if (!finite_bound(anchor_dot) || !finite_bound(ordinary_value))
        throw std::invalid_argument("BDD anchor arithmetic is nonfinite or outside solver numeric limits");
    const double anchor_tolerance = 2e-6 * (1 + std::abs(anchor.operating_objective));
    if (ordinary_value > anchor.operating_objective + anchor_tolerance)
        throw std::invalid_argument("Ordinary BDD anchor cut exceeds its supplied feasible operating value");
    auto& variant = impl_->variants[index(mode)];
    bool binary_anchor = true, anchor_in_variant = true;
    for (int j = 0; j < count; ++j) {
        const bool integral = std::abs(anchor.design[j] - std::round(anchor.design[j])) <= 1e-9;
        binary_anchor = binary_anchor && integral;
        if (variant.matrix.variables[j].binary) anchor_in_variant = anchor_in_variant && integral;
    }
    const double improvement_tolerance = improvement_relative_tolerance *
        (1 + std::abs(anchor.operating_objective));
    Json report = {{"mode", bdd_integrality_name(mode)}, {"anchor_name", anchor.name},
        {"anchor_design", anchor.design}, {"coefficients", anchor.coefficients},
        {"ordinary_intercept", anchor.intercept}, {"ordinary_anchor_value", ordinary_value},
        {"anchor_operating_objective", anchor.operating_objective},
        {"ordinary_anchor_slack", anchor.operating_objective - ordinary_value},
        {"anchor_is_integer", binary_anchor}, {"anchor_feasible_for_inner_integrality", anchor_in_variant},
        {"anchor_source_pricing_complete", true}, {"anchor_design_residuals", design_residuals},
        {"integer_anchor_tight_within_tolerance", binary_anchor &&
            std::abs(anchor.operating_objective - ordinary_value) <= anchor_tolerance},
        {"integer_tight_anchor_theorem", "exact valid tight cut implies zero same-slope structural improvement"},
        {"full_path_domain", true}, {"restricted_column_bound_used", false},
        {"budget_seconds", seconds}, {"improvement_tolerance", improvement_tolerance},
        {"solver_model_reused", variant.calls > 0}, {"accepted", false},
        {"valid_global_lower_bound", false}, {"safe_intercept", nullptr},
        {"safe_cut", nullptr}, {"intercept_gain", nullptr},
        {"inner_lower_bound", nullptr}, {"inner_upper_bound", nullptr},
        {"solver_reported_bound", nullptr}, {"solver_reported_objective", nullptr},
        {"validated_original_integer_upper_bound", nullptr},
        {"objective_update_seconds", 0.0}, {"optimize_seconds", 0.0}, {"validation_seconds", 0.0},
        {"certificate", "full_path_cplex_global_bound_with_numerical_margin"}};
    auto finish = [&]() {
        report["wall_seconds"] = wall_time() - started;
        report["budget_overrun_seconds"] = std::max(0.0, report.at("wall_seconds").get<double>() - seconds);
        return report;
    };
    if (seconds <= 0) { report["status"] = "time_limit_before_solve"; return finish(); }
    try {
        const double update_started = wall_time();
        for (int j = 0; j < count; ++j) {
            variant.matrix.variables[j].cost = -anchor.coefficients[j];
            variant.solver->objective.setLinearCoef(variant.solver->variables[j], -anchor.coefficients[j]);
        }
        report["objective_update_seconds"] = wall_time() - update_started;
        const double allowance = std::max(0.0, seconds - (wall_time() - started));
        if (allowance <= 0) { report["status"] = "time_limit_before_solve"; return finish(); }
        ++variant.calls;
        const double optimize_started = wall_time();
        const auto outcome = variant.solver->optimize(allowance);
        report["optimize_seconds"] = wall_time() - optimize_started;
        report["status"] = outcome.status;
        report["solver_status_code"] = outcome.cplex_status;
        report["solver_optimal"] = outcome.optimal;
        report["has_solver_solution"] = outcome.has_solution;
        report["nodes"] = outcome.nodes;
        if (outcome.has_solution) report["solver_reported_objective"] = outcome.objective;
        if (finite_bound(outcome.bound) && (variant.mixed || outcome.optimal))
            report["solver_reported_bound"] = outcome.bound;
        const double validation_started = wall_time();
        bool primal_valid = false;
        double primal_upper = INFINITY;
        if (outcome.has_solution) {
            const auto residuals = point_residuals(variant.matrix, outcome.values);
            const double recomputed = objective_value(variant.matrix, outcome.values);
            const bool objective_matches = std::abs(recomputed - outcome.objective) <=
                2e-7 * (1 + std::abs(recomputed));
            report["matrix_residuals"] = residuals;
            report["recomputed_objective"] = recomputed;
            report["objective_matches"] = objective_matches;
            primal_valid = residuals.at("passed").get<bool>() && objective_matches;
            report["primal_validated"] = primal_valid;
            if (primal_valid) {
                primal_upper = recomputed;
                report["inner_upper_bound"] = recomputed;
                std::vector<double> design(outcome.values.begin(), outcome.values.begin() + count);
                report["inner_design"] = design;
                report["inner_operating_cost"] = recomputed + dot(anchor.coefficients, design);
                // Partial / continuous primal points never become integer UBs,
                // even if they happen to look close to integral numerically.
                if (mode == BddIntegrality::AllDesign &&
                    point_residuals(impl_->original.matrix, outcome.values).at("passed").get<bool>())
                    report["validated_original_integer_upper_bound"] =
                        objective_value(impl_->original.matrix, outcome.values);
            }
        }
        const bool status_valid = variant.mixed ? permitted_mip_status(outcome.cplex_status) :
            (outcome.optimal && outcome.cplex_status == IloCplex::Optimal && primal_valid);
        bool bound_valid = status_valid && finite_bound(outcome.bound) &&
            (!outcome.has_solution || primal_valid);
        std::string rejection = !status_valid ? "solver_status_not_eligible_for_certificate" :
            !finite_bound(outcome.bound) ? "no_finite_global_bound" :
            (outcome.has_solution && !primal_valid) ? "solver_primal_failed_validation" : "";
        double norm = 0;
        for (double beta : anchor.coefficients) norm += std::abs(beta);
        const double margin = 1e-7 * (1 + std::abs(outcome.bound) + norm);
        const double safe = outcome.bound - margin;
        report["solver_bound_margin"] = margin;
        if (bound_valid && !finite_bound(safe)) {
            bound_valid = false; rejection = "nonfinite_conservative_bound";
        }
        if (bound_valid && std::isfinite(primal_upper) && safe > primal_upper + anchor_tolerance) {
            bound_valid = false; rejection = "bound_exceeds_validated_inner_primal";
        }
        if (bound_valid && anchor_in_variant &&
            safe > anchor.operating_objective - anchor_dot + anchor_tolerance) {
            bound_valid = false; rejection = "bound_contradicts_anchor_feasible_cost";
        }
        report["valid_global_lower_bound"] = bound_valid;
        if (bound_valid) {
            report["inner_lower_bound"] = safe;
            report["safe_intercept"] = safe;
            report["intercept_gain"] = safe - anchor.intercept;
            report["new_anchor_value"] = safe + anchor_dot;
            const bool improved = safe > anchor.intercept + improvement_tolerance;
            report["accepted"] = improved;
            if (improved) report["safe_cut"] = {{"intercept", safe}, {"coefficients", anchor.coefficients},
                {"validity_scope", "all_original_integer_designs"},
                {"certificate", "complete_domain_global_lower_bound"},
                {"inner_integrality", bdd_integrality_name(mode)}};
            else rejection = "no_certified_intercept_improvement";
        }
        report["rejection_reason"] = rejection.empty() ? Json(nullptr) : Json(rejection);
        report["validation_seconds"] = wall_time() - validation_started;
    } catch (const IloException& error) {
        report["status"] = "solver_exception"; report["error"] = error.getMessage();
    } catch (const std::exception& error) {
        report["status"] = "exception"; report["error"] = error.what();
    }
    return finish();
}

Json BddFullDomain::evaluate_multiplier(const std::vector<double>& coefficients,
                                      BddIntegrality mode, double seconds) {
    const double started = wall_time();
    const int count = impl_->original.layout.count;
    if (!std::isfinite(seconds) || seconds < 0)
        throw std::invalid_argument("Invalid BDD multiplier budget");
    if (coefficients.size() != static_cast<std::size_t>(count) ||
        !std::all_of(coefficients.begin(), coefficients.end(), finite_bound))
        throw std::invalid_argument("BDD multiplier has invalid dimensions or nonfinite coefficients");
    auto& variant = impl_->variants[index(mode)];
    Json report = {{"mode", bdd_integrality_name(mode)}, {"coefficients", coefficients},
        {"oracle", "arbitrary_multiplier"}, {"ordinary_anchor_required", false},
        {"full_path_domain", true}, {"restricted_column_bound_used", false},
        {"budget_seconds", seconds}, {"solver_model_reused", variant.calls > 0},
        {"accepted", false}, {"valid_global_lower_bound", false}, {"primal_validated", false},
        {"safe_intercept", nullptr}, {"safe_cut", nullptr},
        {"inner_lower_bound", nullptr}, {"inner_upper_bound", nullptr},
        {"inner_design", nullptr}, {"inner_operating_cost", nullptr},
        {"support_affine_plane", nullptr}, {"safe_support_intercept", nullptr},
        {"support_intercept_margin", nullptr},
        {"solver_reported_bound", nullptr}, {"solver_reported_objective", nullptr},
        {"validated_original_integer_upper_bound", nullptr},
        {"objective_update_seconds", 0.0}, {"optimize_seconds", 0.0}, {"validation_seconds", 0.0},
        {"certificate", "full_path_cplex_global_bound_with_numerical_margin"}};
    auto finish = [&]() {
        report["wall_seconds"] = wall_time() - started;
        report["budget_overrun_seconds"] = std::max(0.0, report.at("wall_seconds").get<double>() - seconds);
        return report;
    };
    if (seconds <= 0) { report["status"] = "time_limit_before_solve"; return finish(); }
    try {
        const double update_started = wall_time();
        for (int j = 0; j < count; ++j) {
            variant.matrix.variables[j].cost = -coefficients[j];
            variant.solver->objective.setLinearCoef(variant.solver->variables[j], -coefficients[j]);
        }
        report["objective_update_seconds"] = wall_time() - update_started;
        const double allowance = std::max(0.0, seconds - (wall_time() - started));
        if (allowance <= 0) { report["status"] = "time_limit_before_solve"; return finish(); }
        ++variant.calls;
        const double optimize_started = wall_time();
        const auto outcome = variant.solver->optimize(allowance);
        report["optimize_seconds"] = wall_time() - optimize_started;
        report["status"] = outcome.status;
        report["solver_status_code"] = outcome.cplex_status;
        report["solver_optimal"] = outcome.optimal;
        report["has_solver_solution"] = outcome.has_solution;
        report["nodes"] = outcome.nodes;
        if (outcome.has_solution) report["solver_reported_objective"] = outcome.objective;
        if (finite_bound(outcome.bound) && (variant.mixed || outcome.optimal))
            report["solver_reported_bound"] = outcome.bound;
        const double validation_started = wall_time();
        bool primal_valid = false;
        double primal_upper = INFINITY;
        if (outcome.has_solution) {
            const auto residuals = point_residuals(variant.matrix, outcome.values);
            const double recomputed = objective_value(variant.matrix, outcome.values);
            const bool objective_matches = finite_bound(recomputed) && finite_bound(outcome.objective) &&
                std::abs(recomputed - outcome.objective) <= 2e-7 * (1 + std::abs(recomputed));
            report["matrix_residuals"] = residuals;
            report["recomputed_objective"] = recomputed;
            report["objective_matches"] = objective_matches;
            primal_valid = residuals.at("passed").get<bool>() && objective_matches;
            report["primal_validated"] = primal_valid;
            if (primal_valid) {
                primal_upper = recomputed;
                report["inner_upper_bound"] = recomputed;
                std::vector<double> design(outcome.values.begin(), outcome.values.begin() + count);
                // Sum operating terms directly to avoid cancellation between
                // a large multiplier objective and beta^T z.
                long double operating = impl_->original.matrix.constant;
                for (std::size_t j = count; j < impl_->original.matrix.variables.size(); ++j)
                    operating += static_cast<long double>(impl_->original.matrix.variables[j].cost) * outcome.values[j];
                const double q = static_cast<double>(operating);
                report["inner_design"] = design;
                report["inner_operating_cost"] = q;
                const double identity_error = std::abs(q - dot(coefficients, design) - recomputed);
                report["operating_objective_identity_error"] = identity_error;
                if (finite_bound(q) && identity_error <= 2e-7 * (1 + std::abs(recomputed))) {
                    std::vector<double> slope = design;
                    for (auto& value : slope) value = -value;
                    // A feasible inner point gives an upper affine plane.
                    // Shift its operating-cost intercept upward to avoid
                    // treating floating-point underestimation as a dual cap.
                    const double support_margin = 1e-7 * (1 + std::abs(q));
                    const double safe_support = std::nextafter(q + support_margin,
                        std::numeric_limits<double>::infinity());
                    report["support_intercept_margin"] = safe_support - q;
                    report["safe_support_intercept"] = safe_support;
                    report["support_affine_plane"] = {{"intercept", safe_support}, {"coefficients", slope},
                        {"raw_operating_cost", q}, {"intercept_margin", safe_support - q},
                        {"sense", "upper"}, {"target", "lagrangian_inner_value_as_function_of_beta"},
                        {"formula", "phi(beta_prime) <= safe_support_intercept - beta_prime^T inner_design"},
                        {"inner_integrality", bdd_integrality_name(mode)},
                        {"validity_scope", "same_inner_integrality_domain_only"},
                        {"valid_for_all_design_inner_value", mode == BddIntegrality::AllDesign},
                        {"source_primal_validated", true}, {"is_benders_cut", false}};
                }
                // Relaxed or partially integral variants cannot certify an
                // original integer UB, even when their point looks integral.
                if (mode == BddIntegrality::AllDesign) {
                    const auto original_residuals = point_residuals(impl_->original.matrix, outcome.values);
                    report["original_integer_residuals"] = original_residuals;
                    const double original_value = objective_value(impl_->original.matrix, outcome.values);
                    if (original_residuals.at("passed").get<bool>() && finite_bound(original_value))
                        report["validated_original_integer_upper_bound"] = original_value;
                }
            }
        }
        const bool status_valid = variant.mixed ? permitted_mip_status(outcome.cplex_status) :
            (outcome.optimal && outcome.cplex_status == IloCplex::Optimal && primal_valid);
        bool bound_valid = status_valid && finite_bound(outcome.bound) &&
            (!outcome.has_solution || primal_valid);
        std::string rejection = !status_valid ? "solver_status_not_eligible_for_certificate" :
            !finite_bound(outcome.bound) ? "no_finite_global_bound" :
            (outcome.has_solution && !primal_valid) ? "solver_primal_failed_validation" : "";
        long double norm = 0;
        for (double beta : coefficients) norm += std::abs(beta);
        const double margin = static_cast<double>(1e-7L * (1 + std::abs(outcome.bound) + norm));
        const double safe = outcome.bound - margin;
        report["solver_bound_margin"] = margin;
        if (bound_valid && !finite_bound(safe)) {
            bound_valid = false; rejection = "nonfinite_conservative_bound";
        }
        if (bound_valid && std::isfinite(primal_upper) &&
            safe > primal_upper + 2e-6 * (1 + std::abs(primal_upper))) {
            bound_valid = false; rejection = "bound_exceeds_validated_inner_primal";
        }
        report["valid_global_lower_bound"] = bound_valid;
        if (bound_valid) {
            report["inner_lower_bound"] = safe;
            report["safe_intercept"] = safe;
            // Accepted here means certified, not improved over any previous
            // cut. A caller must test separation at its own anchor/root.
            report["accepted"] = true;
            report["safe_cut"] = {{"intercept", safe}, {"coefficients", coefficients},
                {"validity_scope", "all_original_integer_designs"},
                {"certificate", "complete_domain_global_lower_bound"},
                {"inner_integrality", bdd_integrality_name(mode)}};
        }
        report["rejection_reason"] = rejection.empty() ? Json(nullptr) : Json(rejection);
        report["validation_seconds"] = wall_time() - validation_started;
    } catch (const IloException& error) {
        report["status"] = "solver_exception"; report["error"] = error.getMessage();
    } catch (const std::exception& error) {
        report["status"] = "exception"; report["error"] = error.what();
    }
    return finish();
}

Json compare_bdd_root_bounds(const Input& data, const PathCollection& paths,
        const std::vector<BddAnchor>& anchors, const Json& evaluations,
        const SolverOptions& supplied, double seconds, std::optional<double> total_cost_floor) {
    if (!std::isfinite(seconds) || seconds < 0 || !evaluations.is_array())
        throw std::invalid_argument("Invalid BDD master comparison budget or evaluation records");
    if (total_cost_floor && !std::isfinite(*total_cost_floor))
        throw std::invalid_argument("BDD comparison total-cost floor must be finite and certified");
    const double started = wall_time(), deadline = started + seconds;
    Json report = {{"scope", "same_compact_continuous_master"},
        {"objective", "original_fixed_cost + theta"}, {"fixed_costs_retained", true},
        {"static_analytical_strengthening", true}, {"dynamic_capacity_or_resource_lp_cuts", false},
        {"certified_total_cost_floor", total_cost_floor ? Json(*total_cost_floor) : Json(nullptr)},
        {"total_cost_floor_includes_fixed_costs", total_cost_floor.has_value()},
        {"ordinary_cut_count", anchors.size()}, {"budget_seconds", seconds},
        {"baseline", nullptr}, {"individual_cuts", Json::array()}, {"all_accepted_cuts", nullptr}};
    if (seconds == 0) { report["status"] = "time_limit"; report["wall_seconds"] = 0; return report; }
    try {
        auto base = design_model(data, false, formulation_strengthening(supplied));
        const int theta = base.matrix.variable("theta", 0, IloInfinity, 1);
        report["static_bound_metadata"] = add_master_strengthening(data, paths, base, theta, true);
        if (total_cost_floor) {
            Coefficients floor{{theta, 1}};
            for (int j : base.design_vars)
                if (base.matrix.variables[j].cost != 0)
                    floor.emplace_back(j, base.matrix.variables[j].cost);
            base.matrix.row("certified_continuous_total_cost_floor", 'G', *total_cost_floor, std::move(floor));
        }
        auto row = [&](double intercept, const std::vector<double>& coefficients, const std::string& name) {
            if (coefficients.size() != static_cast<std::size_t>(base.layout.count) || !std::isfinite(intercept) ||
                !std::all_of(coefficients.begin(), coefficients.end(), [](double v) { return std::isfinite(v); }))
                throw std::invalid_argument("BDD comparison cut has invalid dimensions or coefficients");
            Coefficients values{{theta, 1}};
            for (std::size_t j = 0; j < coefficients.size(); ++j)
                if (coefficients[j]) values.emplace_back(static_cast<int>(j), -coefficients[j]);
            return RowRecord{name, 'G', intercept, std::move(values)};
        };
        for (std::size_t i = 0; i < anchors.size(); ++i) {
            if (!anchors[i].pricing_complete) throw std::invalid_argument("Uncertified ordinary comparison cut");
            base.matrix.rows.push_back(row(anchors[i].intercept, anchors[i].coefficients,
                "ordinary_" + std::to_string(i)));
        }
        std::vector<RowRecord> added;
        Json sources = Json::array();
        for (const auto& evaluation : evaluations) if (evaluation.value("accepted", false)) {
            if (!evaluation.value("valid_global_lower_bound", false) ||
                evaluation.value("restricted_column_bound_used", true) ||
                !evaluation.value("full_path_domain", false) || !evaluation.at("safe_cut").is_object())
                throw std::invalid_argument("BDD comparison received an unverified accepted cut");
            const auto& cut = evaluation.at("safe_cut");
            added.push_back(row(cut.at("intercept"), cut.at("coefficients").get<std::vector<double>>(),
                "bdd_" + std::to_string(added.size())));
            sources.push_back({{"anchor_name", evaluation.at("anchor_name")}, {"mode", evaluation.at("mode")},
                {"intercept_gain", evaluation.at("intercept_gain")}});
        }
        report["accepted_cut_count"] = added.size();
        report["master_size"] = {{"variables", base.matrix.variables.size()},
            {"strategic_variables", base.layout.count}, {"operating_variables", 1},
            {"linear_rows_before_bdd", base.matrix.rows.size()}, {"cones", base.matrix.cones.size()}};
        report["record_build_seconds"] = wall_time() - started;
        int remaining_solves = static_cast<int>(added.size()) + 2;
        auto solve = [&](const std::vector<RowRecord>& cuts) {
            const double phase = wall_time();
            Json result = {{"certified", false}, {"lower_bound", nullptr}, {"objective", nullptr},
                {"added_cuts", cuts.size()}, {"solver_build_seconds", 0.0}, {"optimize_seconds", 0.0}};
            const double allowance = std::max(0.0, deadline - phase) / std::max(1, remaining_solves--);
            if (allowance <= 0) { result["status"] = "time_limit"; return result; }
            auto matrix = base.matrix;
            for (const auto& cut : cuts) matrix.rows.push_back(cut);
            SolverOptions options = supplied; options.gap = 0;
            NativeSolver solver(matrix, options, "bdd_master_comparison");
            result["solver_build_seconds"] = wall_time() - phase;
            const double optimize_started = wall_time();
            const auto outcome = solver.optimize(std::max(0.0, allowance - (wall_time() - phase)));
            result["optimize_seconds"] = wall_time() - optimize_started;
            result["status"] = outcome.status;
            result["solver_status_code"] = outcome.cplex_status;
            if (outcome.has_solution) {
                const auto residuals = point_residuals(matrix, outcome.values);
                const double value = objective_value(matrix, outcome.values);
                const bool valid = outcome.optimal && outcome.cplex_status == IloCplex::Optimal &&
                    residuals.at("passed").get<bool>() &&
                    std::abs(value - outcome.objective) <= 2e-7 * (1 + std::abs(value));
                result["residuals"] = residuals;
                result["certified"] = valid;
                if (valid) {
                    result["objective"] = value;
                    result["lower_bound"] = outcome.bound - 1e-8 * (1 + std::abs(outcome.bound));
                }
            }
            result["wall_seconds"] = wall_time() - phase;
            return result;
        };
        report["baseline"] = solve({});
        auto gains = [&](Json result) {
            result["root_objective_change"] = nullptr;
            result["root_gain_lower_bound"] = nullptr;
            if (report.at("baseline").value("certified", false) && result.value("certified", false)) {
                result["root_objective_change"] = result.at("objective").get<double>() -
                    report.at("baseline").at("objective").get<double>();
                // Compare a new lower bound with the old feasible LP objective,
                // not two approximate lower bounds when claiming positive gain.
                result["root_gain_lower_bound"] = std::max(0.0, result.at("lower_bound").get<double>() -
                    report.at("baseline").at("objective").get<double>());
            }
            return result;
        };
        for (std::size_t i = 0; i < added.size(); ++i) {
            auto result = gains(solve({added[i]}));
            result["source"] = sources[i];
            report["individual_cuts"].push_back(std::move(result));
        }
        report["all_accepted_cuts"] = gains(solve(added));
        report["status"] = "completed_diagnostic";
    } catch (const RunBudgetExceeded& error) {
        report["status"] = "construction_time_limit"; report["error"] = error.what();
    } catch (const IloException& error) {
        report["status"] = "solver_exception"; report["error"] = error.getMessage();
    } catch (const std::exception& error) {
        report["status"] = "exception"; report["error"] = error.what();
    }
    report["wall_seconds"] = wall_time() - started;
    report["budget_overrun_seconds"] = std::max(0.0, report.at("wall_seconds").get<double>() - seconds);
    return report;
}
} // namespace mlnd
