#include "mlnd/benders.hpp"
#include "mlnd/benders_control.hpp"
#include "mlnd/pricing.hpp"
#include "mlnd/output.hpp"
#include "mlnd/strengthening.hpp"
#include "mlnd/topology_cuts.hpp"
#include "mlnd/assignment_support_cuts.hpp"
#include "mlnd/branch_joint_cuts.hpp"
#include "mlnd/heuristics.hpp"
#include "mlnd/dual_workspace.hpp"
#include "mlnd/conditional_cut.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
struct Cut {
    bool valid = false;
    double intercept = 0, correction = 0, stationarity_error = 0;
    double raw_anchor = 0, raw_core = 0;
    std::vector<double> coefficients, prices, missing_service, missing_empty;
    bool optimal = false;
    std::string status, method = "explicit_conic_dual";
    Json diagnostics = Json::object();
    Json condition = nullptr;
};
struct Subproblem {
    Json solution, pareto = {{"attempted", false}, {"accepted", false}, {"reason", "disabled"}};
    Json trace = Json::array();
    Cut ordinary, selected, auxiliary;
    std::vector<int> service, empty;
    Json timings = {{"build", 0.0}, {"primal", 0.0}, {"dual", 0.0}, {"pricing", 0.0},
                    {"add_columns", 0.0}, {"pareto", 0.0}, {"read_solution", 0.0},
                    {"validation", 0.0}, {"certificate", 0.0}};
    int iterations = 0;
    double wall = 0;
    std::string reason = "time_limit";
    double continuous_objective = INFINITY, certified_lower_bound = -INFINITY;
    std::vector<double> continuous_design;
    Json continuous_residuals = nullptr, lower_bound_certificate = nullptr, failure = nullptr;
    Json continuous_operating_point = nullptr;
    bool full_bound_valid = false;
    bool conditional_demand = false;
    Json model_size = nullptr;
};
void add_time(Json& times, const char* name, double begin) {
    times[name] = times.value(name, 0.0) + wall_time() - begin;
}
double remaining(double deadline) { return std::max(0.0, deadline - wall_time()); }
double total_demand(const Input& d) { return std::accumulate(d.demand.begin(), d.demand.end(), 0.0); }
double courier_cost(const Input& d) {
    return std::inner_product(d.demand.begin(), d.demand.end(), d.ground_cost.begin(), 0.0);
}
double value(const Cut& c, const std::vector<double>& design) {
    return c.intercept + std::inner_product(c.coefficients.begin(), c.coefficients.end(), design.begin(), 0.0);
}
double gap(double upper, double lower) { return std::max(0.0, upper - lower) / std::max(1.0, std::abs(upper)); }
bool structural_enabled(const SolverOptions& o) {
    return o.assignment_incident_clique || o.shared_rho_joint || o.queue_perspective;
}
const char* continuous_scope(const SolverOptions& o) {
    return structural_enabled(o) ? "structurally_strengthened_continuous_relaxation" : "full_path_continuous_relaxation";
}
Json structural_metadata(const SolverOptions& o) {
    return {{"assignment_incident_clique",o.assignment_incident_clique},
        {"shared_rho_joint",o.shared_rho_joint},{"queue_perspective",o.queue_perspective},
        {"integer_feasible_set_preserved",true},{"master_added_variables",0},
        {"continuous_relaxation",continuous_scope(o)},
        {"integer_recourse_effect","redundant_on_fixed_integer_design"}};
}

// This numeric model check is performed before accepting a native incumbent;
// the repository's physical reconstruction remains an independent final check.
bool feasible_point(const LinearModel& model, const std::vector<double>& values) {
    if (values.size() != model.variables.size()) return false;
    for (std::size_t j = 0; j < values.size(); ++j) {
        const auto& v = model.variables[j];
        if (!std::isfinite(values[j])) return false;
        double tol = 2e-7 * (1 + std::abs(values[j]));
        if (values[j] < v.lower - tol || values[j] > v.upper + tol) return false;
        if (v.binary && std::abs(values[j] - std::round(values[j])) > 1e-7) return false;
    }
    for (const auto& row : model.rows) {
        double lhs = 0, scale = 1 + std::abs(row.rhs);
        for (auto [j, coefficient] : row.coefficients) {
            lhs += coefficient * values[j];
            scale += std::abs(coefficient * values[j]);
        }
        double residual = lhs - row.rhs;
        if (row.sense == 'L') residual = std::max(0.0, residual);
        else if (row.sense == 'G') residual = std::max(0.0, -residual);
        else residual = std::abs(residual);
        if (residual > 2e-7 * scale) return false;
    }
    for (const auto& cone : model.cones) {
        double lhs = values[cone.one] * values[cone.one];
        double rhs = cone.factor * values[cone.up] * values[cone.down];
        if (lhs > rhs + 2e-7 * (1 + std::abs(lhs) + std::abs(rhs))) return false;
    }
    return true;
}

Json feasibility_diagnostics(const LinearModel& model,const std::vector<double>& values) {
    if (values.size()!=model.variables.size()) return {{"passed",false},{"reason","dimension"}};
    if (!std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);}))
        return {{"passed",false},{"reason","nonfinite"}};
    auto report=point_residuals(model,values);
    report["passed"]=feasible_point(model,values);
    double worst=0; Json detail=nullptr;
    auto check=[&](double residual,double tolerance,const std::string& kind,const std::string& name) {
        if (residual/tolerance>worst) {
            worst=residual/tolerance;
            detail={{"kind",kind},{"name",name},{"violation",residual},{"tolerance",tolerance}};
        }
    };
    for (std::size_t j=0;j<values.size();++j) {
        const auto& v=model.variables[j];
        check(std::max({0.0,v.lower-values[j],values[j]-v.upper}),2e-7*(1+std::abs(values[j])),"bound",v.name);
        if (v.binary) check(std::abs(values[j]-std::round(values[j])),1e-7,"integrality",v.name);
    }
    for (const auto& row:model.rows) {
        double lhs=0,scale=1+std::abs(row.rhs);
        for (auto [j,c]:row.coefficients) { lhs+=c*values[j]; scale+=std::abs(c*values[j]); }
        double residual=lhs-row.rhs;
        residual=row.sense=='L'?std::max(0.0,residual):row.sense=='G'?std::max(0.0,-residual):std::abs(residual);
        check(residual,2e-7*scale,"linear",row.name);
    }
    for (const auto& cone:model.cones) {
        const double lhs=values[cone.one]*values[cone.one],rhs=cone.factor*values[cone.up]*values[cone.down];
        check(std::max(0.0,lhs-rhs),2e-7*(1+std::abs(lhs)+std::abs(rhs)),"cone",model.variables[cone.one].name);
    }
    report["worst_tolerance_ratio"]=worst;
    report["worst_constraint"]=detail;
    return report;
}

void validate_design(const Input& d, const std::vector<double>& design, bool fractional = false) {
    const auto constraints = design_model(d, !fractional);
    if (design.size() != static_cast<std::size_t>(constraints.layout.count) ||
        !std::all_of(design.begin(), design.end(), [fractional](double x) {
            return std::isfinite(x) && (fractional ? x >= 0 && x <= 1 : x == 0.0 || x == 1.0); }))
        throw std::invalid_argument("Benders fixed design must be exactly binary");
    if (!feasible_point(constraints.matrix, design))
        throw std::invalid_argument("Benders fixed design violates facility, capacity, assignment or airline constraints");
}

Cut solve_dual(const Input& d, const PathCollection& paths, const Formulation& f,
               const SolverOptions& options, double deadline,
               const std::vector<double>* core = nullptr,
               double anchor_value = 0, double face_tolerance = 0,
               const std::vector<int>& separated_service = {},
               const std::vector<int>& separated_empty = {}, int numerical_profile = 0,
               const std::vector<double>* primal_anchor = nullptr,
               DualRayWorkspace* ray_workspace = nullptr) {
    Cut result;
    const double dual_started = wall_time();
    result.diagnostics = {{"allocated_seconds", remaining(deadline)},
        {"linear_model_build_seconds", 0.0}, {"solver_build_seconds", 0.0},
        {"optimize_seconds", 0.0}, {"certificate_seconds", 0.0}};
    auto finish_dual = [&](const char* reason) {
        result.diagnostics["termination_reason"] = reason;
        result.diagnostics["seconds"] = wall_time() - dual_started;
        result.diagnostics["valid"] = result.valid;
        result.diagnostics["optimal"] = result.optimal;
        result.diagnostics["status"] = result.status;
        return std::move(result);
    };
    if (numerical_profile == 4) result.method = "cone_ray_lp_recovery";
    if (remaining(deadline) <= 0) return finish_dual("deadline_before_model_build");
    const auto& primal = f.matrix;
    const int n = static_cast<int>(primal.variables.size());
    const int m = static_cast<int>(primal.rows.size());
    LinearModel dual;
    dual.minimize = false;
    dual.constant = primal.constant;
    std::vector<int> pi(m), upper(n), lower(n, -1);
    std::vector<Coefficients> stationarity(n);
    std::vector<double> core_rhs(m);
    for (int i = 0; i < m; ++i) core_rhs[i] = primal.rows[i].rhs;
    if (core) {
        if (core->size() != f.design_fix_rows.size()) throw std::runtime_error("Invalid Benders core dimensions");
        for (std::size_t j = 0; j < core->size(); ++j) core_rhs[f.design_fix_rows[j]] = (*core)[j];
    }
    for (int i = 0; i < m; ++i) {
        const auto& row = primal.rows[i];
        pi[i] = dual.variable("pi_" + std::to_string(i), row.sense == 'G' ? 0 : -IloInfinity,
                              row.sense == 'L' ? 0 : IloInfinity, core_rhs[i]);
        for (auto [j, coefficient] : row.coefficients) stationarity[j].emplace_back(pi[i], coefficient);
    }
    for (int j = 0; j < n; ++j) {
        const auto& v = primal.variables[j];
        if (v.lower < 0 || !std::isfinite(v.upper) || v.upper >= IloInfinity)
            throw std::runtime_error("Conic certificate requires finite nonnegative recourse bounds");
        upper[j] = dual.variable("upper_" + std::to_string(j), 0, IloInfinity, -v.upper);
        stationarity[j].emplace_back(upper[j], -1);
        if (v.lower > 0) {
            lower[j] = dual.variable("lower_" + std::to_string(j), 0, IloInfinity, v.lower);
            stationarity[j].emplace_back(lower[j], 1);
        }
    }
    std::vector<std::array<int, 3>> cones;
    for (std::size_t k = 0; k < primal.cones.size(); ++k) {
        const auto& cone = primal.cones[k];
        int up = dual.variable("cone_u_" + std::to_string(k), 0, IloInfinity);
        int down = dual.variable("cone_d_" + std::to_string(k), 0, IloInfinity);
        int one = dual.variable("cone_o_" + std::to_string(k), -IloInfinity, IloInfinity);
        if (primal_anchor && numerical_profile == 4) {
            // Restrict (never relax) the conic dual to the complementary ray
            // suggested by the optimal primal queue point; optimize its
            // nonnegative scale in the LP. Only the unchanged primal/dual gap
            // check can certify this restricted dual solution.
            const double primal_up = primal_anchor->at(cone.up);
            const double primal_down = primal_anchor->at(cone.down);
            if (!(primal_up > 0 && primal_down > 0)) return finish_dual("invalid_cone_recovery_anchor");
            const double a = std::sqrt(cone.factor * primal_down / primal_up);
            const double b = std::nextafter(std::max(std::sqrt(cone.factor * primal_up / primal_down), cone.factor / a), INFINITY);
            const double c = -2 * (primal_anchor->at(cone.one) < 0 ? -1 : 1);
            int scale = dual.variable("cone_ray_scale_" + std::to_string(k), 0, IloInfinity);
            dual.row("cone_ray_up_" + std::to_string(k), 'E', 0, {{up, 1}, {scale, -a}});
            dual.row("cone_ray_down_" + std::to_string(k), 'E', 0, {{down, 1}, {scale, -b}});
            dual.row("cone_ray_one_" + std::to_string(k), 'E', 0, {{one, 1}, {scale, -c}});
        } else dual.cones.push_back({up, down, one, 4.0 / cone.factor});
        stationarity[cone.up].emplace_back(up, 1);
        stationarity[cone.down].emplace_back(down, 1);
        stationarity[cone.one].emplace_back(one, 1);
        cones.push_back({up, down, one});
    }
    for (int j = 0; j < n; ++j)
        dual.row("stationarity_" + std::to_string(j), 'L', primal.variables[j].cost, std::move(stationarity[j]));
    struct Missing { bool empty; int index, variable; Coefficients column; double cost; };
    std::vector<Missing> missing;
    const double total = total_demand(d);
    if (core) {
        auto add_missing = [&](bool empty, int index, Coefficients column, double cost) {
            int u = dual.variable("missing_" + std::to_string(empty) + "_" + std::to_string(index), 0, IloInfinity, -total);
            Coefficients terms{{u, -1}};
            for (auto [row, coefficient] : column) terms.emplace_back(pi[row], coefficient);
            dual.row("missing_stationarity_" + std::to_string(empty) + "_" + std::to_string(index), 'L', cost, std::move(terms));
            missing.push_back({empty, index, u, std::move(column), cost});
        };
        // The auxiliary dual is separated by the same exact label oracle as
        // ordinary CG. Only previously separated columns are materialized;
        // there is no all-missing-catalogue shortcut in Pareto optimization.
        for (int i : separated_service)
            add_missing(false, i, path_column(f, paths.service_paths.at(i)), paths.service_paths.at(i).cost);
        for (int i : separated_empty)
            add_missing(true, i, path_column(f, paths.empty_paths.at(i)), paths.empty_paths.at(i).cost);
        Coefficients face;
        for (int i = 0; i < m; ++i) if (primal.rows[i].rhs != 0) face.emplace_back(pi[i], primal.rows[i].rhs);
        for (int j = 0; j < n; ++j) {
            face.emplace_back(upper[j], -primal.variables[j].upper);
            if (lower[j] >= 0) face.emplace_back(lower[j], primal.variables[j].lower);
        }
        for (const auto& entry : missing) face.emplace_back(entry.variable, -total);
        dual.row("optimal_face", 'G', anchor_value - face_tolerance - primal.constant, std::move(face));
    }
    result.diagnostics["linear_model_build_seconds"] = wall_time() - dual_started;
    result.diagnostics["model_size"] = {{"variables", dual.variables.size()},
        {"linear_rows", dual.rows.size()}, {"cones", dual.cones.size()}};
    if (remaining(deadline) <= 0) return finish_dual("deadline_during_linear_model_build");
    const double solver_build_started = wall_time();
    const bool reuse_ray = options.reuse_dual_lp && ray_workspace && numerical_profile == 4 && !core;
    std::unique_ptr<NativeSolver> cold_solver;
    Json workspace_diagnostics;
    NativeSolver& solver = reuse_ray ? ray_workspace->synchronize(dual,options,workspace_diagnostics)
        : *(cold_solver=std::make_unique<NativeSolver>(dual, options, core ? "pareto_dual" : "benders_dual"));
    if (numerical_profile > 0) {
        solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical, true);
        solver.cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol, 1e-12);
        solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
        solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-9);
        solver.cplex.setParam(IloCplex::Param::Read::Scale, numerical_profile == 1 ? 1 : numerical_profile == 2 ? -1 : 0);
        if (numerical_profile == 3) solver.cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
        if (numerical_profile == 4) solver.cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Dual);
    }
    result.diagnostics["solver_build_seconds"] = wall_time() - solver_build_started;
    if (reuse_ray) result.diagnostics["dual_lp_workspace"] = workspace_diagnostics;
    if (remaining(deadline) <= 0) return finish_dual("deadline_during_solver_build");
    result.diagnostics["solver_time_limit_seconds"] = remaining(deadline);
    const double optimize_started = wall_time();
    auto solved = reuse_ray ? ray_workspace->optimize(remaining(deadline),workspace_diagnostics)
                           : solver.optimize(remaining(deadline));
    if (reuse_ray) result.diagnostics["dual_lp_workspace"] = workspace_diagnostics;
    result.diagnostics["optimize_seconds"] = wall_time() - optimize_started;
    result.diagnostics["solver_status_code"] = solved.cplex_status;
    result.diagnostics["has_solution"] = solved.has_solution;
    result.optimal = solved.optimal;
    result.status = numerical_profile == 4 ? "RecoveredConeRayLP_" + solved.status : solved.status;
    if (!solved.has_solution)
        return finish_dual(solved.status == "AbortTimeLim" ? "solver_time_limit" : "solver_no_solution");
    if (solved.values.size() != dual.variables.size()) return finish_dual("solver_values_dimension");
    if (!std::all_of(solved.values.begin(), solved.values.end(), [](double x) { return std::isfinite(x); }))
        return finish_dual("solver_values_nonfinite");
    if (reuse_ray) {
        const double validation_started=wall_time();
        result.diagnostics["dual_lp_source_residuals"]=feasibility_diagnostics(dual,solved.values);
        result.diagnostics["dual_lp_source_objective"]=objective_value(dual,solved.values);
        const bool source_valid=feasible_point(dual,solved.values);
        result.diagnostics["dual_lp_source_validation_seconds"]=wall_time()-validation_started;
        if (!source_valid)
            return finish_dual("persistent_dual_lp_source_validation_failed");
    }
    const double certificate_started = wall_time();
    result.prices.resize(m);
    for (int i = 0; i < m; ++i) {
        double x = solved.values[pi[i]];
        if (primal.rows[i].sense == 'L') x = std::min(0.0, x);
        if (primal.rows[i].sense == 'G') x = std::max(0.0, x);
        result.prices[i] = x;
    }
    std::vector<double> cone_values(n, 0), upper_prices(n), lower_prices(n, 0), residual(n, 0);
    for (std::size_t k = 0; k < cones.size(); ++k) {
        const auto& source = primal.cones[k];
        auto indices = cones[k];
        double up = std::max(0.0, solved.values[indices[0]]);
        double down = std::max(0.0, solved.values[indices[1]]);
        double one = solved.values[indices[2]];
        const double f = 4.0 / source.factor;
        if (f * up * down < one * one) {
            if (up >= down && up > 0) down = std::nextafter(one * one / (f * up), INFINITY);
            else if (down > 0) up = std::nextafter(one * one / (f * down), INFINITY);
            else up = down = std::nextafter(std::abs(one) / std::sqrt(f), INFINITY);
        }
        cone_values[source.up] += up;
        cone_values[source.down] += down;
        cone_values[source.one] += one;
    }
    double bound_terms = 0;
    for (int j = 0; j < n; ++j) {
        upper_prices[j] = std::max(0.0, solved.values[upper[j]]);
        if (lower[j] >= 0) lower_prices[j] = std::max(0.0, solved.values[lower[j]]);
        bound_terms += -primal.variables[j].upper * upper_prices[j] + primal.variables[j].lower * lower_prices[j];
        residual[j] = -upper_prices[j] + lower_prices[j] + cone_values[j] - primal.variables[j].cost;
    }
    for (int i = 0; i < m; ++i)
        for (auto [j, coefficient] : primal.rows[i].coefficients) residual[j] += coefficient * result.prices[i];
    for (int j = 0; j < n; ++j) {
        double violation = std::max(0.0, residual[j]);
        result.stationarity_error = std::max(result.stationarity_error, violation);
        result.correction += primal.variables[j].upper * violation;
    }
    result.missing_service.resize(paths.service_paths.size(), 0);
    result.missing_empty.resize(paths.empty_paths.size(), 0);
    for (const auto& entry : missing) {
        double price = std::max(0.0, solved.values[entry.variable]);
        double violation = -entry.cost;
        for (auto [row, coefficient] : entry.column) violation += coefficient * result.prices[row];
        price = std::max(price, violation);
        if (price > 0) price = std::nextafter(price, INFINITY);
        (entry.empty ? result.missing_empty : result.missing_service)[entry.index] = price;
        bound_terms -= total * price;
    }
    double rhs_value = 0;
    result.raw_anchor = primal.constant;
    result.raw_core = primal.constant;
    for (int i = 0; i < m; ++i) {
        rhs_value += primal.rows[i].rhs * result.prices[i];
        result.raw_core += core_rhs[i] * solved.values[pi[i]];
        result.raw_anchor += primal.rows[i].rhs * solved.values[pi[i]];
    }
    for (int j = 0; j < n; ++j) {
        double term = -primal.variables[j].upper * solved.values[upper[j]];
        if (lower[j] >= 0) term += primal.variables[j].lower * solved.values[lower[j]];
        result.raw_core += term;
        result.raw_anchor += term;
    }
    for (const auto& entry : missing) {
        result.raw_core -= total * solved.values[entry.variable];
        result.raw_anchor -= total * solved.values[entry.variable];
    }
    result.intercept = primal.constant + rhs_value + bound_terms;
    result.coefficients.resize(f.design_fix_rows.size());
    double coefficient_norm = 0;
    for (std::size_t j = 0; j < f.design_fix_rows.size(); ++j) {
        int row = f.design_fix_rows[j];
        result.coefficients[j] = result.prices[row];
        result.intercept -= primal.rows[row].rhs * result.prices[row];
        coefficient_norm += std::abs(result.prices[row]);
    }
    result.correction += 1e-8 * (1 + courier_cost(d)) + 128 * std::numeric_limits<double>::epsilon()
                         * (1 + std::abs(rhs_value) + std::abs(bound_terms) + coefficient_norm);
    result.intercept -= result.correction;
    result.valid = std::isfinite(result.intercept) && std::isfinite(result.correction)
                   && std::all_of(result.coefficients.begin(), result.coefficients.end(), [](double x) { return std::isfinite(x); });
    result.diagnostics["certificate_seconds"] = wall_time() - certificate_started;
    return finish_dual(!result.valid ? "invalid_dual_certificate" :
                       result.optimal ? "dual_optimal" : "dual_not_optimal");
}

// The final reduced-cost residual is converted into a valid full-model dual
// correction by the label oracle. Its bound includes every omitted column,
// including labels discarded by a proved dominance relation.
bool correct_missing_columns(Cut& cut, const LabelPricingResult& pricing) {
    if (!cut.valid || !pricing.complete ||
        !std::isfinite(pricing.minimum) || !std::isfinite(pricing.min_service) ||
        !std::isfinite(pricing.min_empty) || !std::isfinite(pricing.correction) || pricing.correction < 0) return false;
    const double corrected_intercept = cut.intercept - pricing.correction;
    const double corrected_total = cut.correction + pricing.correction;
    if (!std::isfinite(corrected_intercept) || !std::isfinite(corrected_total)) return false;
    cut.intercept = corrected_intercept;
    cut.correction = corrected_total;
    return true;
}
bool certify(Cut& cut, const LabelPricingResult& pricing, double tolerance) {
    if (!cut.optimal || pricing.minimum < -tolerance || pricing.negative_paths != 0 ||
        !pricing.service.empty() || !pricing.empty.empty()) return false;
    return correct_missing_columns(cut, pricing);
}

Json pricing_record(const LabelPricingResult& pricing) {
    return {{"complete", pricing.complete}, {"minimum_reduced_cost", pricing.minimum},
            {"minimum_service_reduced_cost", pricing.min_service}, {"minimum_empty_reduced_cost", pricing.min_empty},
            {"dual_correction", pricing.correction},
            {"generated_service", pricing.service.size()}, {"generated_empty", pricing.empty.size()},
            {"labels_created", pricing.labels_created}, {"labels_extended", pricing.labels_extended},
            {"labels_dominated", pricing.labels_dominated}, {"terminal_paths", pricing.terminal_paths},
            {"omitted_paths", pricing.omitted_paths}, {"negative_paths", pricing.negative_paths},
            {"resource_pruned", pricing.resource_pruned}, {"excluded_paths", pricing.excluded_paths},
            {"dominance", pricing.dominance}};
}

Subproblem subproblem(const Input& d, PathCollection& paths, const std::vector<double>& design,
                      const SolverOptions& options, double deadline, const std::vector<int>& initial_p,
                      const std::vector<int>& initial_e, const std::vector<double>* core,
                      bool fractional = false, bool free_design = false,
                      bool robust_continuous_recovery = false) {
    double begin = wall_time(), phase = begin;
    Subproblem result;
    if (!free_design) validate_design(d, design, fractional);
    if (free_design && (!fractional || core))
        throw std::invalid_argument("Free-design CG requires continuous variables and no Pareto anchor");
    const bool continuous_recovery = free_design || robust_continuous_recovery;
    if (continuous_recovery && (!fractional || core))
        throw std::invalid_argument("Continuous recovery requires a continuous RMP and no Pareto anchor");
    const bool conditional = options.conditional_demand && !fractional && !free_design;
    result.conditional_demand = conditional;
    // All fractional/free-design strengthening continues to use the original
    // lifted relaxation. An integer-recourse core must stay on its alpha face.
    std::vector<double> face_core;
    if (conditional && core) {
        face_core = *core;
        const DesignLayout layout(d);
        for (int j=layout.assignment;j<layout.airline;++j) face_core[j]=design[j];
        for (int r=0;r<d.n_od;++r) for (int l=0;l<d.n_l;++l) for (int m=0;m<d.assignment_destinations();++m)
            if (design[layout.assignment_index(r,l,m)]==1) {
                face_core[layout.facility+l]=1;
                if (!d.origin_assignment()) face_core[layout.facility+m]=1;
            }
        validate_design(d,face_core,true);
        core = &face_core;
    }
    Formulation f = build_formulation(d, paths, free_design ? nullptr : &design,
                                     options.column_generation ? &initial_p : nullptr,
                                     options.column_generation ? &initial_e : nullptr, fractional, conditional, options.tight_airline_bounds,
                                     formulation_strengthening(options));
    if (free_design) for (auto& variable : f.matrix.variables) variable.binary = false;
    // Relative complete recourse: for every admissible fixed binary design,
    // courier=demand, rho=head=flow=fleet=idle=0 and cone variables equal
    // their opening indicator (one in the baseline formulation) is
    // feasible. Thus the empty-column RMP needs neither artificials nor Phase I.
    std::vector<double> baseline(f.matrix.variables.size(), 0);
    for (std::size_t j = 0; j < baseline.size(); ++j) baseline[j] = f.matrix.variables[j].lower;
    for (std::size_t j = 0; j < design.size(); ++j) baseline[f.design_vars[j]] = design[j];
    for (int r = 0; r < d.n_od; ++r) baseline[f.courier[r]] = d.demand[r];
    for (std::size_t l=0; l<f.matrix.cones.size(); ++l) {
        const auto& cone=f.matrix.cones[l];
        const double indicator=f.strengthening.queue_perspective ? baseline[f.layout.facility+l] : 1;
        baseline[cone.up] = baseline[cone.down] = baseline[cone.one] = indicator;
    }
    if (!feasible_point(f.matrix, baseline)) throw std::runtime_error("All-courier RMP feasibility construction failed");
    if (!fractional) {
        result.solution = solution_json(d, paths, f, baseline);
        result.solution["native_model_validated"] = true;
    }
    const char* primal_role = free_design ? "continuous_rmp" : fractional ? "root_rmp" : "integer_rmp";
    auto solver=std::make_unique<NativeSolver>(f.matrix, options, primal_role);
    const double graph_started=wall_time();
    if(options.progress) options.progress->record({{"event","phase_start"},{"stage","label_graph_construction"},
        {"fractional_design",fractional}});
    LabelPricingOracle oracle(d, paths, deadline);
    if(options.progress) options.progress->record({{"event","phase_end"},{"stage","label_graph_construction"},
        {"seconds",wall_time()-graph_started},{"fractional_design",fractional}});
    add_time(result.timings, "build", phase);
    LabelPricingOptions pricing_options;
    pricing_options.deadline = deadline;
    pricing_options.tolerance = options.pricing_tolerance;
    pricing_options.max_columns = options.cg_max_columns;
    int primal_retries_for_columns = 0, active_primal_profile = 0;
    DualRayWorkspace ray_workspace;
    bool primal_polish_attempted_for_columns = false;
    std::string primal_retry_reason;
    auto primal_qcp_tolerance = [](int profile) {
        return profile == 0 ? 1e-9 : profile == 1 ? 1e-10 : profile == 2 ? 1e-11 : 1e-12;
    };
    auto tighter_primal = [&](int profile) {
        auto recovery=std::make_unique<NativeSolver>(f.matrix,options,
            free_design ? "continuous_rmp_recovery" : fractional ? "root_rmp_recovery" : "integer_rmp_recovery");
        recovery->cplex.setParam(IloCplex::Param::Emphasis::Numerical,true);
        recovery->cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol,primal_qcp_tolerance(profile));
        recovery->cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility,1e-9);
        recovery->cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality,1e-9);
        recovery->cplex.setParam(IloCplex::Param::Read::Scale,profile==1?1:profile==2?-1:0);
        if (profile==3) recovery->cplex.setParam(IloCplex::Param::Preprocessing::Presolve,false);
        return recovery;
    };
    try {
    while (remaining(deadline) > 0) {
        ++result.iterations;
        Json iteration = {{"iteration", result.iterations}, {"service_columns", f.service_indices.size()},
                          {"demand_formulation",conditional?"conditional":"lifted"},
                          {"structural_strengthening",f.strengthening_metadata},
                          {"empty_columns", f.empty_indices.size()}, {"certified", false},
                          {"model_size",{{"variables",f.matrix.variables.size()},{"linear_rows",f.matrix.rows.size()},
                              {"cones",f.matrix.cones.size()}}}};
        auto finish = [&](const std::string& status) {
            iteration["termination_reason"] = status;
            iteration["elapsed_seconds"] = options.progress ? options.progress->elapsed() : wall_time()-begin;
            iteration["subproblem_elapsed_seconds"] = wall_time()-begin;
            if (options.progress) {
                Json event=iteration;
                event["event"]="cg_iteration";
                event["stage"]=primal_role;
                event["bound_scope"]=free_design ? continuous_scope(options) : "fixed_design_recourse";
                options.progress->record(std::move(event));
            }
            result.trace.push_back(iteration);
            result.reason = status;
        };
        phase = wall_time();
        const bool certificate_retry = !primal_retry_reason.empty();
        if (certificate_retry) iteration["primal_retry_reason"] = primal_retry_reason;
        primal_retry_reason.clear();
        auto solved = solver->optimize(certificate_retry ? std::min(20.0,.25*remaining(deadline)) : remaining(deadline));
        iteration["primal_attempts"]=Json::array();
        auto record_primal=[&](int profile,const SolveOutcome& point) {
            Json record={{"numerical_profile",profile},{"status",point.status},{"optimal",point.optimal},
                         {"qcp_convergence_tolerance",primal_qcp_tolerance(profile)},
                         {"has_solution",point.has_solution},{"candidate_available",point.candidate_available},
                         {"candidate_extraction_error",point.candidate_extraction_error.empty() ? Json(nullptr) : Json(point.candidate_extraction_error)},
                         {"seconds",point.seconds},{"objective",point.has_solution?Json(point.objective):Json(nullptr)}};
            if (point.has_solution || point.candidate_available) {
                record["residuals"]=feasibility_diagnostics(f.matrix,point.values);
                record["candidate_objective"]=point.objective;
            }
            iteration["primal_attempts"].push_back(std::move(record));
        };
        record_primal(active_primal_profile,solved);
        iteration["original_rmp_solver_status"] = solved.status;
        iteration["original_rmp_solver_optimal"] = solved.optimal;
        iteration["original_rmp_solver_has_solution"] = solved.has_solution;
        bool polished_primal = false;
        auto recover_candidate = [&](SolveOutcome& point) {
            if (!continuous_recovery || (!point.has_solution && !point.candidate_available)) return false;
            if (!point.has_solution && feasible_point(f.matrix,point.values)) {
                point.has_solution = true;
                point.optimal = false;
                return true;
            }
            if (primal_polish_attempted_for_columns || feasible_point(f.matrix,point.values) ||
                remaining(deadline) <= 0) return false;
            primal_polish_attempted_for_columns = true;
            auto polished = point.values;
            iteration["primal_polish"] = polish_conic_primal(f.matrix, options,
                std::min(20.0, .25 * remaining(deadline)), polished);
            if (iteration.at("primal_polish").at("applied").get<bool>() && feasible_point(f.matrix, polished)) {
                point.values = std::move(polished);
                point.objective = objective_value(f.matrix, point.values);
                // An optimal restricted LP supplies only an original-model
                // feasible point. Its QCP optimality must be established below
                // with the independent original conic dual, never inherited.
                point.has_solution = true;
                point.optimal = false;
                polished_primal = true;
                return true;
            }
            return false;
        };
        bool recovered_primal = recover_candidate(solved);
        // Sparse initial RMPs can be highly degenerate. Retry the unchanged
        // model after fresh extraction when CPLEX's returned point fails our
        // original residual tolerances; a solver status alone is insufficient.
        while (primal_retries_for_columns < 3) {
            if (recovered_primal) break;
            if (solved.has_solution && solved.optimal && feasible_point(f.matrix,solved.values)) break;
            if (remaining(deadline)<=0) break;
            const int profile = ++primal_retries_for_columns;
            auto recovery=tighter_primal(profile);
            auto candidate=recovery->optimize(std::min(20.0,.25*remaining(deadline)));
            record_primal(profile,candidate);
            const bool candidate_recovered = recover_candidate(candidate);
            if (candidate.has_solution && feasible_point(f.matrix,candidate.values)) {
                solved=std::move(candidate); solver=std::move(recovery);
                active_primal_profile=profile;
                recovered_primal=candidate_recovered;
            }
        }
        iteration["same_column_primal_retries"] = primal_retries_for_columns;
        iteration["rmp_seconds"] = wall_time() - phase;
        iteration["rmp_status"] = solved.status;
        iteration["rmp_optimal"] = solved.optimal;
        iteration["rmp_objective"] = solved.has_solution ? Json(solved.objective) : Json(nullptr);
        add_time(result.timings, "primal", phase);
        if (!solved.has_solution) { finish("recourse_" + solved.status); break; }
        phase = wall_time();
        bool feasible = feasible_point(f.matrix, solved.values);
        iteration["rmp_feasible"] = feasible;
        iteration["rmp_residuals"]=feasibility_diagnostics(f.matrix,solved.values);
        const double recomputed_objective = objective_value(f.matrix, solved.values);
        const bool objective_matches = std::isfinite(recomputed_objective) &&
            std::abs(recomputed_objective - solved.objective) <= 2e-7 * (1 + std::abs(recomputed_objective));
        iteration["rmp_recomputed_objective"] = recomputed_objective;
        iteration["rmp_objective_matches"] = objective_matches;
        add_time(result.timings, "validation", phase);
        if (!feasible) { finish("primal_validation_failed"); break; }
        if (!objective_matches) { finish("primal_objective_validation_failed"); break; }
        phase = wall_time();
        Json trial = nullptr;
        if (!fractional) {
            trial = solution_json(d, paths, f, solved.values);
            trial["native_model_validated"] = true;
        }
        add_time(result.timings, "read_solution", phase);
        auto accept_primal = [&]() {
            if (fractional && solved.objective < result.continuous_objective) {
                result.continuous_objective = solved.objective;
                result.continuous_residuals = iteration.at("rmp_residuals");
                result.continuous_design.clear();
                for (int j : f.design_vars)
                    result.continuous_design.push_back(std::clamp(solved.values[j], 0.0, 1.0));
                result.continuous_operating_point={{"rho",Json::array()},{"idle",Json::array()},
                    {"v",Json::array()},{"v_order",d.origin_assignment()?"r_then_departure":"r_then_departure_then_return"},
                    {"assignment_scope",d.assignment_scope()},
                    {"fleet",solved.values[f.fleet]}};
                for (int j:f.rho) result.continuous_operating_point["rho"].push_back(solved.values[j]);
                for (int j:f.idle) result.continuous_operating_point["idle"].push_back(solved.values[j]);
                for (std::size_t j=0;j<f.matrix.variables.size();++j)
                    if (f.matrix.variables[j].name.rfind("v_",0)==0)
                        result.continuous_operating_point["v"].push_back(solved.values[j]);
            }
            if (trial.is_null()) return;
            const double accept_started = wall_time();
            if (trial["objective"].get<double>() < result.solution["objective"].get<double>())
                result.solution = std::move(trial);
            if (options.progress) options.progress->checkpoint(result.solution,"integer_rmp_feasible");
            add_time(result.timings, "read_solution", accept_started);
        };
        // A feasible time-limited RMP remains a primal incumbent, but its
        // multipliers cannot be used as the optimal RMP prices required by CG.
        if (!solved.optimal && !recovered_primal) { accept_primal(); finish("rmp_not_optimal_" + solved.status); break; }
        const double q = solved.objective;
        const double tight_tolerance = options.cg_duality_tolerance * (1 + std::abs(q));
        phase = wall_time();
        Cut dual;
        bool primal_dual_contradiction = false;
        iteration["dual_attempts"] = Json::array();
        const std::array<int,5> dual_profiles = continuous_recovery
            ? std::array<int,5>{4,0,1,2,3} : std::array<int,5>{0,4,1,2,3};
        iteration["dual_profile_order"] = dual_profiles;
        for (int profile : dual_profiles) {
            if (remaining(deadline) <= 0) break;
            // Fresh extraction avoids retaining an unstable barrier state in
            // profiles 0--3. Profile 4 solves a feasible cone-ray LP restriction
            // and may reuse a simplex basis, never a certificate. Every profile must pass the
            // unchanged conservative primal/dual certificate tolerance.
            const double attempt_started = wall_time();
            // A continuous relaxation's conic barrier can spend its entire slice
            // at NumBest while the verified cone-ray LP finishes in 7--9 s.
            // Give that first certificate enough of the remaining budget;
            // its validity and original-RMP gap checks are exactly unchanged.
            const double allowance = continuous_recovery && profile == 4
                ? std::min(30.0, .8 * remaining(deadline)) : std::min(20.0, .25 * remaining(deadline));
            const double attempt_deadline = std::min(deadline, attempt_started + allowance);
            Cut candidate = solve_dual(d, paths, f, options, attempt_deadline, nullptr, 0, 0, {}, {}, profile,
                &solved.values, options.reuse_dual_lp ? &ray_workspace : nullptr);
            const double candidate_gap = candidate.valid ? q - value(candidate, design) : INFINITY;
            iteration["dual_attempts"].push_back({{"numerical_profile", profile}, {"status", candidate.status},
                {"method", candidate.method},
                {"optimal", candidate.optimal}, {"valid", candidate.valid}, {"seconds", wall_time() - attempt_started},
                {"corrected_gap", std::isfinite(candidate_gap) ? Json(candidate_gap) : Json(nullptr)},
                {"correction", candidate.correction}, {"stationarity_error", candidate.stationarity_error}});
            if (candidate.diagnostics.contains("dual_lp_workspace"))
                iteration["dual_attempts"].back()["dual_lp_workspace"]=candidate.diagnostics.at("dual_lp_workspace");
            dual = std::move(candidate);
            if (dual.valid && std::isfinite(candidate_gap) && candidate_gap < -tight_tolerance) {
                // Weak duality applies to every feasible dual, including a
                // cone-ray restriction. A later weaker dual must not hide a
                // contradiction with the current approximate primal point.
                primal_dual_contradiction = true;
                break;
            }
            if (dual.valid && dual.optimal && std::isfinite(candidate_gap) &&
                candidate_gap >= -tight_tolerance && candidate_gap <= tight_tolerance) break;
        }
        iteration["dual_seconds"] = wall_time() - phase;
        iteration["dual_status"] = dual.status;
        iteration["dual_certificate_method"] = dual.method;
        iteration["dual_optimal"] = dual.optimal;
        iteration["dual_diagnostics"] = dual.diagnostics;
        add_time(result.timings, "dual", phase);
        if (primal_dual_contradiction) {
            iteration["primal_rejected_due_to_duality"] = true;
            iteration["rmp_duality_gap"] = q - value(dual, design);
            iteration["duality_tolerance"] = tight_tolerance;
            if (primal_retries_for_columns < 3 && remaining(deadline) > 0) {
                active_primal_profile = ++primal_retries_for_columns;
                primal_retry_reason = "valid_dual_exceeds_primal_objective";
                iteration["next_primal_numerical_profile"] = active_primal_profile;
                finish("primal_reoptimization_required");
                const double rebuild_started = wall_time();
                solver = tighter_primal(active_primal_profile);
                add_time(result.timings, "build", rebuild_started);
                continue;
            }
            finish(remaining(deadline) <= 0 ? "time_limit" : "primal_dual_contradiction_retry_limit");
            break;
        }
        accept_primal();
        if (!dual.valid) { finish(remaining(deadline) <= 0 ? "time_limit" : "dual_unavailable"); break; }
        const double rmp_gap = q - value(dual, design);
        iteration["dual_objective"] = value(dual, design);
        iteration["raw_dual_objective"] = dual.raw_anchor;
        iteration["dual_stationarity_error"] = dual.stationarity_error;
        iteration["dual_numerical_correction"] = dual.correction;
        iteration["rmp_duality_gap"] = rmp_gap;
        iteration["duality_tolerance"] = tight_tolerance;
        if (!dual.optimal) { finish("dual_not_optimal_" + dual.status); break; }
        if (rmp_gap < -tight_tolerance || rmp_gap > tight_tolerance) { finish("rmp_duality_gap_failed"); break; }
        iteration["rmp_optimal"] = true;
        iteration["rmp_optimality_certificate"] = {{"method","original_conic_primal_dual_gap"},
            {"primal_source",polished_primal ? "cone_restriction_lp_primal_only" :
                             recovered_primal ? "validated_numerical_solver_candidate" : "original_qcp"},
            {"primal_feasible",true},{"dual_feasible",true},{"dual_solver_optimal",dual.optimal},
            {"gap",rmp_gap},{"tolerance",tight_tolerance}};
        phase = wall_time();
        LabelPricingResult pricing;
        if(options.progress) options.progress->record({{"event","phase_start"},{"stage","ordinary_label_pricing"},
            {"iteration",result.iterations},{"fractional_design",fractional}});
        if (options.column_generation) pricing = oracle.price(f, dual.prices, {}, {}, pricing_options);
        else pricing.complete = paths.complete; // Full recourse already contains every column.
        iteration["pricing_seconds"] = wall_time() - phase;
        iteration["pricing"] = pricing_record(pricing);
        if(options.progress) options.progress->record({{"event","phase_end"},{"stage","ordinary_label_pricing"},
            {"seconds",iteration.at("pricing_seconds")},{"iteration",result.iterations},
            {"fractional_design",fractional},{"pricing",iteration.at("pricing")}});
        add_time(result.timings, "pricing", phase);
        if (!pricing.complete) { finish("pricing_incomplete"); break; }
        phase = wall_time();
        if (free_design) {
            // Even a nonterminal complete pricing pass gives a full-domain
            // bound: stationarity is repaired for every omitted service flow
            // using sum(flow)<=total demand, and for bounded empty flows.
            // It is not the restricted primal objective and need not be tight.
            Cut full_certificate = dual;
            if (correct_missing_columns(full_certificate, pricing)) {
                const double lower = full_certificate.intercept;
                const double primal_tolerance = options.cg_duality_tolerance *
                    (1 + std::abs(result.continuous_objective));
                if (std::isfinite(result.continuous_objective) &&
                    lower > result.continuous_objective + primal_tolerance) {
                    iteration["rejected_full_lower_bound"] = lower;
                    add_time(result.timings, "certificate", phase);
                    finish("full_bound_primal_dual_contradiction"); break;
                }
                iteration["certified_full_lower_bound"] = lower;
                iteration["full_lower_bound_correction"] = full_certificate.correction;
                if (!result.full_bound_valid || lower > result.certified_lower_bound) {
                    result.full_bound_valid = true;
                    result.certified_lower_bound = lower;
                    result.lower_bound_certificate = {{"method","explicit_conic_dual_with_complete_label_correction"},
                        {"iteration",result.iterations},{"rmp_dual_objective",dual.intercept},
                        {"pricing_correction",pricing.correction},{"total_correction",full_certificate.correction},
                        {"stationarity_error",full_certificate.stationarity_error},
                        {"pricing",pricing_record(pricing)}};
                    if (options.progress) options.progress->record({{"event","continuous_cg_bound"},
                        {"bound_scope",continuous_scope(options)},{"lower_bound",lower},
                        {"bound_valid",true},{"integer_incumbent",false},
                        {"certificate",result.lower_bound_certificate}});
                }
            }
        }
        bool complete = certify(dual, pricing, options.pricing_tolerance);
        add_time(result.timings, "certificate", phase);
        if (complete) {
            const double final_gap = q - value(dual, design);
            iteration["full_dual_objective"] = value(dual, design);
            iteration["full_duality_gap"] = final_gap;
            if (final_gap < -tight_tolerance || final_gap > tight_tolerance) {
                finish("full_duality_gap_failed"); break;
            }
            result.ordinary = dual;
            result.selected = dual;
            iteration["certified"] = true;
            finish("pricing_complete");
            if (core && options.pareto) {
                phase = wall_time();
                const double face = options.cg_duality_tolerance * (1 + std::abs(q));
                const double verify = 2e-7 * (1 + std::abs(q));
                double ordinary_gap = q - value(dual, design);
                result.pareto = {{"attempted", true}, {"accepted", false}, {"reason", "ordinary_anchor_not_tight"},
                                 {"ordinary_anchor_gap", ordinary_gap}, {"face_tolerance", face}, {"verification_tolerance", verify},
                                 {"separation", "exact_label_pricing"}, {"iterations", Json::array()},
                                 {"auxiliary_solves", Json::array()}, {"requested_seconds", options.pareto_call_seconds},
                                 {"allocated_seconds", 0.0}, {"deadline_remaining_seconds", remaining(deadline)},
                                 {"pricing_seconds", 0.0}, {"certificate_seconds", 0.0}, {"timeout", false}};
                if (ordinary_gap >= -verify && ordinary_gap <= face + verify && remaining(deadline) > 0) {
                    // The outer scheduler owns the total and per-call Pareto
                    // budgets. Do not silently shrink its explicit allowance.
                    const double auxiliary_started = wall_time();
                    const double auxiliary_deadline = std::min(deadline,
                        auxiliary_started + std::max(0.0, options.pareto_call_seconds));
                    const double auxiliary_budget = auxiliary_deadline - auxiliary_started;
                    result.pareto["allocated_seconds"] = auxiliary_budget;
                    result.pareto["reason"] = "auxiliary_budget_exhausted_before_build";
                    std::vector<int> separated_p, separated_e;
                    try {
                    while (remaining(auxiliary_deadline) > 0) {
                        Cut auxiliary = solve_dual(d, paths, f, options, auxiliary_deadline, core, q, face,
                                                   separated_p, separated_e);
                        result.pareto["auxiliary_solves"].push_back(auxiliary.diagnostics);
                        result.pareto["auxiliary_status"] = auxiliary.status;
                        result.pareto["reason"] = "auxiliary_" + auxiliary.diagnostics.value("termination_reason", std::string("unavailable"));
                        result.pareto["timeout"] = auxiliary.status == "AbortTimeLim" || remaining(auxiliary_deadline) <= 0;
                        if (!auxiliary.valid || !auxiliary.optimal) break;
                        auto auxiliary_options = pricing_options;
                        auxiliary_options.deadline = auxiliary_deadline;
                        auxiliary_options.excluded_service = separated_p;
                        auxiliary_options.excluded_empty = separated_e;
                        LabelPricingResult check;
                        const double auxiliary_pricing_started=wall_time();
                        if(options.progress) options.progress->record({{"event","phase_start"},{"stage","pareto_label_pricing"},
                            {"iteration",result.pareto.at("iterations").size()+1}});
                        if (options.column_generation)
                            check = oracle.price(f, auxiliary.prices, auxiliary.missing_service,
                                                 auxiliary.missing_empty, auxiliary_options);
                        else check.complete = paths.complete;
                        if(options.progress) options.progress->record({{"event","phase_end"},{"stage","pareto_label_pricing"},
                            {"seconds",wall_time()-auxiliary_pricing_started},
                            {"iteration",result.pareto.at("iterations").size()+1},{"pricing",pricing_record(check)}});
                        result.pareto["pricing_seconds"] = result.pareto.at("pricing_seconds").get<double>()
                            + wall_time() - auxiliary_pricing_started;
                        result.pareto["iterations"].push_back(pricing_record(check));
                        result.pareto["pricing_complete"] = check.complete;
                        result.pareto["reason"] = "pricing_incomplete";
                        if (!check.complete) {
                            result.pareto["timeout"] = remaining(auxiliary_deadline) <= 0;
                            break;
                        }
                        if (!check.service.empty() || !check.empty.empty()) {
                            separated_p.insert(separated_p.end(), check.service.begin(), check.service.end());
                            separated_e.insert(separated_e.end(), check.empty.begin(), check.empty.end());
                            result.pareto["reason"] = "auxiliary_separation_incomplete";
                            continue;
                        }
                        result.pareto["reason"] = "pricing_violation";
                        const double certificate_started = wall_time();
                        if (certify(auxiliary, check, options.pricing_tolerance)) {
                            // Global weak-duality validity is distinct from
                            // tightness on the MW face and core optimality.
                            // Keep a conservatively corrected valid cut even
                            // when it cannot be labelled a Pareto-style cut.
                            double anchor_gap = q - value(auxiliary, design);
                            double gain = value(auxiliary, *core) - value(dual, *core);
                            double core_loss = auxiliary.raw_core - value(auxiliary, *core);
                            const bool weak_duality_consistent = std::isfinite(anchor_gap) && anchor_gap >= -verify;
                            result.pareto["global_cut_valid"] = weak_duality_consistent;
                            if (weak_duality_consistent) result.auxiliary = auxiliary;
                            result.pareto["anchor_gap"] = anchor_gap;
                            result.pareto["core_gain"] = gain;
                            result.pareto["core_optimality_loss"] = core_loss;
                            result.pareto["anchor_check"] = anchor_gap >= -verify && anchor_gap <= face + verify;
                            result.pareto["core_gain_check"] = gain >= -verify;
                            result.pareto["core_optimality_check"] = std::abs(core_loss) <= verify;
                            result.pareto["reason"] = weak_duality_consistent
                                ? "certificate_check_failed" : "auxiliary_primal_dual_contradiction";
                            if (anchor_gap >= -verify && anchor_gap <= face + verify && gain >= -verify && std::abs(core_loss) <= verify) {
                                result.selected = std::move(auxiliary);
                                result.pareto["accepted"] = true;
                                result.pareto["reason"] = "optimal_face";
                            }
                        }
                        result.pareto["certificate_seconds"] = result.pareto.at("certificate_seconds").get<double>()
                            + wall_time() - certificate_started;
                        break;
                    }
                    if (remaining(auxiliary_deadline) <= 0 &&
                        (result.pareto.at("reason") == "auxiliary_separation_incomplete" ||
                         result.pareto.at("reason") == "auxiliary_budget_exhausted_before_build")) {
                        result.pareto["timeout"] = true;
                        result.pareto["reason"] = result.pareto.at("auxiliary_solves").empty()
                            ? "auxiliary_budget_exhausted_before_build" : "auxiliary_separation_time_limit";
                    }
                    } catch (const IloException& exception) {
                        result.pareto["reason"] = "auxiliary_solver_error";
                        result.pareto["error"] = exception.getMessage();
                    } catch (const std::exception& exception) {
                        result.pareto["reason"] = "auxiliary_error";
                        result.pareto["error"] = exception.what();
                    }
                } else if (ordinary_gap >= -verify && ordinary_gap <= face + verify) {
                    result.pareto["reason"] = "deadline_before_auxiliary";
                    result.pareto["timeout"] = true;
                }
                result.pareto["seconds"] = wall_time() - phase;
                result.pareto["dual_seconds"] = 0.0;
                for (const auto& auxiliary : result.pareto.at("auxiliary_solves"))
                    result.pareto["dual_seconds"] = result.pareto.at("dual_seconds").get<double>() + auxiliary.at("seconds").get<double>();
                if (options.progress) options.progress->record({{"event", "pareto_diagnostic"}, {"stage", "pareto"},
                    {"fractional_design", fractional}, {"diagnostics", result.pareto}});
                add_time(result.timings, "pareto", phase);
            }
            break;
        }
        phase = wall_time();
        if (pricing.service.empty() && pricing.empty.empty()) { finish("pricing_stalled"); break; }
        add_columns(f, *solver, d, paths, pricing.service, pricing.empty);
        if (active_primal_profile != 0) {
            // Recovery is local to the unchanged RMP. New columns can change
            // its conditioning, so never retain a restrictive recovery profile
            // for every subsequent CG solve. Fresh extraction restores all
            // NativeSolver defaults and discards the old numerical state.
            const double reset_started = wall_time();
            const int previous_profile = active_primal_profile;
            solver = std::make_unique<NativeSolver>(f.matrix, options,
                free_design ? "continuous_rmp_profile_reset" : fractional ? "root_rmp_profile_reset" : "integer_rmp_profile_reset");
            active_primal_profile = 0;
            iteration["primal_profile_reset"] = {{"from_profile",previous_profile},{"to_profile",0},
                {"reason","column_set_changed"},{"fresh_extraction",true},
                {"qcp_convergence_tolerance",primal_qcp_tolerance(0)},
                {"seconds",wall_time()-reset_started}};
        }
        primal_retries_for_columns = 0;
        primal_polish_attempted_for_columns = false;
        iteration["add_columns_seconds"] = wall_time() - phase;
        add_time(result.timings, "add_columns", phase);
        finish("columns_added");
    }
    } catch (const RunBudgetExceeded& error) {
        if (!continuous_recovery) throw;
        result.reason = "construction_time_limit";
        result.failure = {{"stage",error.stage},{"sizes",error.sizes},{"message",error.what()}};
    } catch (const IloException& error) {
        if (!continuous_recovery) throw;
        result.reason = "solver_exception";
        result.failure = {{"message",error.getMessage()}};
    } catch (const std::exception& error) {
        if (!continuous_recovery) throw;
        result.reason = "exception";
        result.failure = {{"message",error.what()}};
    }
    if (result.reason == "columns_added" && remaining(deadline) <= 0) result.reason = "time_limit";
    result.service = f.service_indices;
    result.empty = f.empty_indices;
    std::size_t nonzeros=0;
    for (const auto& row:f.matrix.rows) nonzeros+=row.coefficients.size();
    result.model_size={{"variables",f.matrix.variables.size()},{"linear_rows",f.matrix.rows.size()},
                       {"cones",f.matrix.cones.size()},{"nonzeros",nonzeros}};
    if (conditional) {
        const double conditioning_started=wall_time();
        auto gate=[&](Cut& cut) {
            if (!cut.valid) return;
            const auto converted=condition_assignment_cut(cut.intercept,cut.coefficients,design,
                                                          f.layout.assignment,f.layout.airline,0.0);
            cut.intercept=converted.intercept;
            cut.coefficients=converted.coefficients;
            cut.correction+=converted.additional_correction;
            cut.condition={{"cut_kind","conditional_dual"},
                {"cut_scope","fixed_assignment_all_facility_designs"},
                {"assignment_fingerprint",converted.assignment_key},{"gate_M",converted.gate_m},
                {"affine_upper_bound",converted.affine_upper_bound},
                {"gate_rounding_margin",converted.gate_rounding_margin},
                {"conditioning_correction",converted.additional_correction},
                {"anchor_before",converted.anchor_value_before},{"anchor_after",converted.anchor_value_after}};
        };
        gate(result.ordinary); gate(result.selected); gate(result.auxiliary);
        result.pareto["scope"]="fixed_assignment_face";
        add_time(result.timings,"conditional_cut",conditioning_started);
    }
    // Pricing, including auxiliary Pareto separation, may append paths after
    // the last primal incumbent was packaged. Those new columns have zero
    // flow in that incumbent and must not alter its physical solution.
    if (!result.solution.is_null()) synchronize_result_paths(d, paths, result.solution);
    result.wall = wall_time() - begin;
    return result;
}

std::vector<double> initial_core(const Input& d) {
    DesignLayout layout(d);
    std::vector<double> core(layout.count, 0);
    std::fill(core.begin() + layout.facility, core.begin() + layout.capacity, 0.5);
    std::fill(core.begin() + layout.capacity, core.begin() + layout.assignment, 0.5 / d.n_levels);
    for (int r = 0; r < d.n_od; ++r)
        for (int l = 0; l < d.n_l; ++l)
            for (int m = 0; m < d.assignment_destinations(); ++m)
                core[layout.assignment_index(r,l,m)] = d.origin_assignment() ? .25 / (d.n_l+1) :
                    (l == m ? .5 : .25) / (d.n_l * d.n_l + 1);
    for (int u = 0; u < d.n_nodes(); ++u)
        for (int v = 0; v < d.n_nodes(); ++v)
            core[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v) ? (u == v ? .25 : .125) : 0;
    return core;
}

std::vector<double> initial_design(const Input& d, const PathCollection& paths) {
    DesignLayout layout(d);
    std::vector<double> design(layout.count, 0);
    std::vector<double> reposition(d.n_l * d.n_l, INFINITY), gains(d.assignment_count(), 0);
    for (const auto& p : paths.empty_paths)
        reposition[p.b[0] * d.n_l + p.b[1]] = std::min(reposition[p.b[0] * d.n_l + p.b[1]], p.cost + d.fleet_cost * p.time);
    for (const auto& p : paths.service_paths) {
        if (p.mode != "single") continue;
        double cost = p.cost + d.fleet_cost * p.time + (p.b[0] == p.b[1] ? 0 : reposition[p.b[1] * d.n_l + p.b[0]]);
        int index = d.assignment_offset(p.first_od,p.b[0],p.b[1]);
        gains[index] = std::max(gains[index], d.ground_cost[p.first_od] - cost);
    }
    std::vector<int> order(gains.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ar = a / d.assignment_pool_count(), br = b / d.assignment_pool_count();
        double sa = gains[a] * d.demand[ar], sb = gains[b] * d.demand[br];
        return std::make_tuple(-sa, -d.demand[ar], a) < std::make_tuple(-sb, -d.demand[br], b);
    });
    std::vector<bool> assigned(d.n_od, false);
    // This is a primal proposal, not a restriction on the global design
    // domain. Each assigned first order must have a single route and every
    // positive-kappa continuation to the proposed assigned pool. Otherwise
    // an absent compulsory branch can force the whole group's head to zero.
    std::set<std::array<int,4>> branches;
    std::map<int,std::set<int>> prefix_branches;
    std::map<std::array<int,3>,std::vector<int>> prefixes;
    for (const auto& p:paths.service_paths) {
        branches.insert({p.b[0],p.b[1],p.first_od,p.second_od});
        prefix_branches[p.segment_id].insert(p.second_od);
    }
    for (const auto& s:paths.initial_segments)
        prefixes[{s.b[0],s.b[1],s.first_od}].push_back(s.id);
    std::vector<std::vector<int>> pools(d.assignment_pool_count());
    auto supported = [&](int l,int b,int first,const std::vector<int>& pool) {
        if (d.effective_delta()<=0) return true;
        if (d.flow_policy=="aggregate") {
            if (!branches.count({l,b,first,-1})) return false;
            for (int second:pool)
                if (d.kappa(first,second)>0 && !branches.count({l,b,first,second})) return false;
            return true;
        }
        for (int segment:prefixes[{l,b,first}]) {
            const auto& available=prefix_branches[segment];
            if (!available.count(-1)) continue;
            bool complete=true;
            for (int second:pool) if (d.kappa(first,second)>0 && !available.count(second)) { complete=false; break; }
            if (complete) return true;
        }
        return false;
    };
    for (int index : order) {
        int r = index / d.assignment_pool_count();
        if (assigned[r] || gains[index] <= 0 || d.demand[r] <= 0) continue;
        const int pair=index%d.assignment_pool_count(),l=pair/d.assignment_destinations(),
            b=d.origin_assignment() ? 0 : pair%d.n_l;
        auto trial=pools[pair]; trial.push_back(r);
        bool compatible=true;
        for (int first:trial) if (!supported(l,b,first,trial)) { compatible=false; break; }
        if (!compatible) continue;
        // Missing generated branches only reject this heuristic proposal;
        // exact CG and the global master retain every physical possibility.
        pools[pair]=std::move(trial);
        design[layout.assignment + index] = 1; assigned[r] = true;
    }
    if (std::any_of(assigned.begin(), assigned.end(), [](bool x) { return x; })) {
        std::fill(design.begin() + layout.facility, design.begin() + layout.capacity, 1);
        for (int i = 0; i < d.n_h; ++i) {
            int best = 0;
            for (int s = 1; s < d.n_levels; ++s) if (d.slot_counts(i, s) > d.slot_counts(i, best)) best = s;
            design[layout.capacity + i * d.n_levels + best] = 1;
        }
        for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
            design[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v);
    }
    return design;
}

// Asset cleanup preserves every flow, assignment and queue variable. An
// arbitrarily small nonzero flow still protects all its visited assets.
Json cleanup_incumbent(const Input& d, const PathCollection& paths, const Json& original) {
    Json candidate = original;
    auto& sol = candidate["_solution"];
    std::set<int> used_nodes;
    std::set<std::pair<int, int>> used_links;
    std::vector<double> occupancy(d.n_h, 0);
    auto protect = [&](const auto& catalogue, const char* field) {
        for (std::size_t index = 0; index < catalogue.size(); ++index) {
            if (sol[field][index].template get<double>() == 0) continue;
            for (const auto& leg : catalogue[index].legs) {
                for (int node : leg) used_nodes.insert(node);
                for (std::size_t j = 1; j < leg.size(); ++j) used_links.emplace(leg[j - 1], leg[j]);
            }
        }
    };
    protect(paths.service_paths, "flow");
    protect(paths.empty_paths, "empty_flow");
    for (std::size_t index = 0; index < paths.service_paths.size(); ++index)
        for (auto [i, time] : paths.service_paths[index].slot_times)
            occupancy[i] += sol["flow"][index].get<double>() * time;
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        if (!used_links.count({u, v})) sol["airline"][u][v] = 0.0;
    for (int i = 0; i < d.n_h; ++i) {
        if (!used_nodes.count(d.n_l + i) && occupancy[i] == 0) {
            sol["facility"][d.n_l + i] = 0.0;
            for (int s = 0; s < d.n_levels; ++s) sol["capacity_level"][i][s] = 0.0;
        } else if (sol["facility"][d.n_l + i].get<double>() > .5) {
            int level = -1;
            double old_cost = 0;
            for (int s = 0; s < d.n_levels; ++s) {
                old_cost += d.slot_cost(i, s) * sol["capacity_level"][i][s].get<double>();
                if (d.occupancy_limit(i, s) >= occupancy[i] &&
                    (level < 0 || std::make_tuple(d.slot_cost(i, s), d.slot_counts(i, s), s) <
                                  std::make_tuple(d.slot_cost(i, level), d.slot_counts(i, level), level))) level = s;
            }
            if (level >= 0 && d.slot_cost(i, level) <= old_cost) {
                for (int s = 0; s < d.n_levels; ++s) sol["capacity_level"][i][s] = s == level ? 1.0 : 0.0;
            }
        }
    }
    for (int l = 0; l < d.n_l; ++l) {
        bool assigned = false;
        for (int r = 0; r < d.n_od; ++r) for (int m = 0; m < d.assignment_destinations(); ++m)
            assigned = assigned || assignment_value(d,sol["assignment"],r,l,m) != 0 ||
                (!d.origin_assignment() && assignment_value(d,sol["assignment"],r,m,l) != 0);
        if (!used_nodes.count(l) && !assigned && sol["rho"][l].get<double>() == 0 && sol["idle"][l].get<double>() == 0)
            sol["facility"][l] = 0.0;
    }
    double bases = 0, lockers = 0, links = 0;
    for (int l = 0; l < d.n_l; ++l) bases += d.base_cost[l] * sol["facility"][l].get<double>();
    for (int i = 0; i < d.n_h; ++i) for (int s = 0; s < d.n_levels; ++s)
        lockers += d.slot_cost(i, s) * sol["capacity_level"][i][s].get<double>();
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        links += d.link_cost(u, v) * sol["airline"][u][v].get<double>();
    const double opening = bases + lockers + links;
    const double objective = opening + candidate["continuous_cost"].get<double>();
    if (!std::isfinite(objective) || objective > original["objective"].get<double>() + 1e-8) return original;
    // These are exactly the constraints affected by changing x/z/y. All other
    // rows and cones retain their previously checked values and coefficients.
    try { validate_design(d, solution_design(d, candidate)); }
    catch (const std::invalid_argument&) { return original; }
    for (int l = 0; l < d.n_l; ++l)
        if (sol["rho"][l].get<double>() > sol["facility"][l].get<double>() + 1e-8) return original;
    for (int i = 0; i < d.n_h; ++i) {
        double limit = 0;
        for (int s = 0; s < d.n_levels; ++s) limit += d.occupancy_limit(i, s) * sol["capacity_level"][i][s].get<double>();
        if (occupancy[i] > limit + 2e-7 * (1 + std::abs(occupancy[i]) + std::abs(limit))) return original;
    }
    for (auto [u, v] : used_links) if (sol["airline"][u][v].get<double>() == 0) return original;
    candidate["opening_cost"] = opening;
    candidate["objective"] = objective;
    candidate["metrics"]["costs_per_hour"]["bases"] = bases;
    candidate["metrics"]["costs_per_hour"]["lockers"] = lockers;
    candidate["metrics"]["costs_per_hour"]["links"] = links;
    candidate["cleanup"] = {{"fixed_flows", true}, {"affected_constraints_checked", true},
                             {"saving", original["objective"].get<double>() - objective}};
    return candidate;
}

std::vector<double> neighbor_design(const Input& d, const PathCollection& paths, const Json& incumbent) {
    const DesignLayout layout(d);
    auto design = solution_design(d, incumbent);
    std::vector<double> reposition(d.n_l * d.n_l, INFINITY), gains(d.assignment_count(), 0);
    for (const auto& p : paths.empty_paths)
        reposition[p.b[0] * d.n_l + p.b[1]] = std::min(reposition[p.b[0] * d.n_l + p.b[1]], p.cost + d.fleet_cost * p.time);
    for (const auto& p : paths.service_paths) {
        if (p.mode != "single") continue;
        double cost = p.cost + d.fleet_cost * p.time + (p.b[0] == p.b[1] ? 0 : reposition[p.b[1] * d.n_l + p.b[0]]);
        int index = d.assignment_offset(p.first_od,p.b[0],p.b[1]);
        gains[index] = std::max(gains[index], d.ground_cost[p.first_od] - cost);
    }
    std::vector<int> order(gains.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ar = a / d.assignment_pool_count(), br = b / d.assignment_pool_count();
        return std::make_tuple(-d.demand[ar], -gains[a], a) < std::make_tuple(-d.demand[br], -gains[b], b);
    });
    for (int index : order) {
        int r = index / d.assignment_pool_count();
        bool assigned = false;
        for (int a = 0; a < d.n_l; ++a) for (int b = 0; b < d.assignment_destinations(); ++b)
            assigned = assigned || design[layout.assignment_index(r,a,b)] != 0;
        if (assigned || gains[index] <= 0 || d.demand[r] <= 0) continue;
        design[layout.assignment + index] = 1;
        std::fill(design.begin() + layout.facility, design.begin() + layout.capacity, 1);
        std::fill(design.begin() + layout.capacity, design.begin() + layout.assignment, 0);
        for (int i = 0; i < d.n_h; ++i) {
            int best = 0;
            for (int s = 1; s < d.n_levels; ++s) if (d.slot_counts(i, s) > d.slot_counts(i, best)) best = s;
            design[layout.capacity + i * d.n_levels + best] = 1;
        }
        for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
            design[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v);
        return design;
    }
    return {};
}

Json add_master_bounds(const Input& d, const PathCollection& paths, Formulation& master,
                       int theta, const SolverOptions& options) {
    return add_master_strengthening(d, paths, master, theta, options.resource_bounds);
}

struct InitialColumns {
    std::vector<int> service, empty;
    Json metadata;
};
InitialColumns initial_columns(const Input& d, PathCollection& paths,
                               const SolverOptions& options, double deadline) {
    const double started=wall_time();
    InitialColumns seeds;
    Json coverage = {{"enabled",options.branch_covering_seeds}};
    if (options.column_generation && !paths.complete) {
        greedy_seed_paths(d, paths, 1, deadline);
        if (options.branch_covering_seeds) {
            const auto report = branch_covering_seed_paths(d, paths, nullptr,
                std::min(deadline, wall_time()+std::min(5.0,.05*options.seconds)));
            coverage.update({{"service_added",report.service_added},{"empty_added",report.empty_added},
                {"groups_requested",report.groups_requested},{"groups_completed",report.groups_completed},
                {"required_branches",report.required_branches},{"covered_branches",report.covered_branches},
                {"physically_missing_branches",report.physically_missing_branches},
                {"groups_without_complete_prefix",report.groups_without_complete_prefix},{"stopped",report.stopped}});
        }
        seeds.service.resize(paths.service_paths.size());
        seeds.empty.resize(paths.empty_paths.size());
        std::iota(seeds.service.begin(),seeds.service.end(),0);
        std::iota(seeds.empty.begin(),seeds.empty.end(),0);
    }
    seeds.metadata={{"method", !options.column_generation ? "all_columns" :
                              (paths.complete ? "empty_rmp_catalogue_validation" :
                               options.branch_covering_seeds ? "greedy_with_branch_coverage" : "greedy_physical_cost")},
                    {"service",seeds.service.size()},{"empty",seeds.empty.size()},
                    {"seconds",wall_time()-started},
                    {"feasibility", "all_courier_independent_of_seed_quality"}, {"branch_coverage",coverage}};
    return seeds;
}

Json continuous_cg_result(const Input& d, PathCollection& paths, const SolverOptions& options,
                          const std::vector<double>* fixed_design) {
    if (!paths.support_complete) throw std::invalid_argument("Continuous CG requires complete physical support");
    if (d.origin_assignment() && (options.assignment_support_cuts || options.double_support_cuts || options.joint_base_cost_cuts))
        throw std::invalid_argument("K=1 origin assignment does not support base-pair assignment-support, double-support, or joint-base-cost cuts; disable these optional cuts");
    if (!options.column_generation && !paths.complete)
        throw std::invalid_argument("Full continuous recourse requires a complete path catalogue");
    if (!std::isfinite(options.seconds) || options.seconds < 0 || options.cg_max_columns < 1 ||
        !std::isfinite(options.cg_duality_tolerance) || options.cg_duality_tolerance <= 0 ||
        !std::isfinite(options.pricing_tolerance) || options.pricing_tolerance < 0)
        throw std::invalid_argument("Invalid continuous CG budget or certification tolerance");
    if (fixed_design) validate_design(d, *fixed_design, true);
    const double started = wall_time();
    double deadline = started + options.seconds;
    if (options.progress) deadline = std::min(deadline, started + options.progress->remaining());
    Json result = {{"method",fixed_design ? "fractional_recourse_cg" : "full_continuous_cg"},
        {"solver","CPLEX"},{"tight_airline_bounds",options.tight_airline_bounds},
        {"scope",fixed_design ? "fixed_design_recourse" : continuous_scope(options)},
        {"structural_strengthening",structural_metadata(options)},
        {"objective_includes_fixed_costs",!fixed_design},{"is_integer_solution",false},
        {"certified_lower_bound",nullptr},{"bound_valid",false},{"continuous_objective",nullptr},
        {"primal_valid",false},
        {"operating_objective",nullptr},{"design",fixed_design ? Json(*fixed_design) : Json::array()},
        {"certified",false},{"pricing_complete",false},{"intercept",nullptr},
        {"coefficients",Json::array()},{"ordinary_cut",nullptr},{"cg_trace",Json::array()},
        {"timings",Json::object()},{"termination_reason","time_limit"},
        {"model_construction_complete",false},{"support_complete",true},
        {"catalogue_complete",paths.complete},{"budget_seconds",options.seconds},
        {"budget_scope","seed_construction_model_build_optimization_pricing_validation"},
        {"pricing_method",options.column_generation ? "exact_resource_labeling" : "all_columns"},
        {"cg_max_columns",options.cg_max_columns},
        {"continuous_numerical_recovery",true},
        {"cg_duality_tolerance",options.cg_duality_tolerance},{"pricing_tolerance",options.pricing_tolerance}};
    InitialColumns seeds;
    double initialization_seconds = 0;
    try {
        if (remaining(deadline) > 0) {
            const double initialization_started = wall_time();
            seeds = initial_columns(d, paths, options, deadline);
            initialization_seconds = wall_time() - initialization_started;
            result["initial_columns"] = seeds.metadata;
            if (remaining(deadline) > 0) {
                const std::vector<double> no_design;
                auto sub = subproblem(d, paths, fixed_design ? *fixed_design : no_design, options,
                    deadline, seeds.service, seeds.empty, nullptr, true, fixed_design == nullptr, true);
                result["model_construction_complete"] = true;
                result["termination_reason"] = sub.reason;
                result["cg_trace"] = std::move(sub.trace);
                result["cg_iterations"] = sub.iterations;
                result["service_columns"] = sub.service.size();
                result["empty_columns"] = sub.empty.size();
                result["timings"] = sub.timings;
                result["primal_residuals"] = sub.continuous_residuals;
                result["operating_point"] = sub.continuous_operating_point;
                result["model_size"] = sub.model_size;
                if (std::isfinite(sub.continuous_objective)) {
                    result["continuous_objective"] = sub.continuous_objective;
                    result["primal_valid"] = true;
                    if (fixed_design) result["operating_objective"] = sub.continuous_objective;
                    else result["design"] = sub.continuous_design;
                }
                result["pricing_complete"] = sub.ordinary.valid;
                result["certified"] = sub.ordinary.valid;
                result["failure"] = sub.failure;
                if (!fixed_design && sub.full_bound_valid) {
                    result["bound_valid"] = true;
                    result["certified_lower_bound"] = sub.certified_lower_bound;
                    result["lower_bound_certificate"] = sub.lower_bound_certificate;
                }
                if (fixed_design && sub.ordinary.valid) {
                    result["bound_valid"] = true;
                    result["certified_lower_bound"] = value(sub.ordinary, *fixed_design);
                    result["intercept"] = sub.ordinary.intercept;
                    result["coefficients"] = sub.ordinary.coefficients;
                    result["ordinary_cut"] = {{"kind","ordinary"},{"intercept",sub.ordinary.intercept},
                        {"coefficients",sub.ordinary.coefficients},{"design",*fixed_design},
                        {"operating_objective",result.at("operating_objective")},
                        {"correction",sub.ordinary.correction},{"stationarity_error",sub.ordinary.stationarity_error},
                        {"pricing_complete",true},{"certified",true},
                        {"scope","global_recourse_epigraph"}};
                }
            }
        }
    } catch (const RunBudgetExceeded& error) {
        result["termination_reason"] = "construction_time_limit";
        result["failure"] = {{"stage",error.stage},{"sizes",error.sizes},{"message",error.what()}};
    } catch (const EnumerationStopped& error) {
        result["termination_reason"] = error.reason;
        result["failure"] = {{"stage","initial_columns"},{"message",error.what()}};
    } catch (const IloException& error) {
        result["termination_reason"] = "solver_exception";
        result["failure"] = {{"message",error.getMessage()}};
    }
    result["timings"]["greedy_initial_columns"] = initialization_seconds;
    result["algorithm_wall"] = wall_time() - started;
    double accounted = 0;
    for (const auto& seconds : result.at("timings")) accounted += seconds.get<double>();
    result["timings"]["other_and_interrupted"] = std::max(0.0, result.at("algorithm_wall").get<double>() - accounted);
    result["generated_service_paths"] = paths.service_paths.size();
    result["generated_empty_paths"] = paths.empty_paths.size();
    result["catalogue_complete"] = paths.complete;
    if (options.progress) options.progress->record({{"event","continuous_cg_finished"},
        {"bound_scope",result.at("scope")},{"bound_valid",result.at("bound_valid")},
        {"certified_lower_bound",result.at("certified_lower_bound")},
        {"continuous_objective",result.at("continuous_objective")},
        {"pricing_complete",result.at("pricing_complete")},{"termination_reason",result.at("termination_reason")},
        {"integer_incumbent",false}});
    return result;
}
} // namespace

Json solve_continuous_cg(const Input& d, PathCollection& paths, const SolverOptions& options) {
    return continuous_cg_result(d, paths, options, nullptr);
}

Json evaluate_fractional_recourse_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                                    const std::vector<double>& design) {
    return continuous_cg_result(d, paths, options, &design);
}

Json solve_fixed_design_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                          const std::vector<double>& design) {
    if (!paths.support_complete) throw std::runtime_error("Fixed-design CG requires complete initial-segment support");
    if (!options.column_generation && !paths.complete) throw std::runtime_error("Full recourse requires a complete path catalogue");
    const double started = wall_time();
    const auto seeds=initial_columns(d,paths,options,started+options.seconds);
    auto sub = subproblem(d, paths, design, options, started + options.seconds, seeds.service, seeds.empty, nullptr);
    if (sub.solution.is_null()) throw std::runtime_error("Fixed-design CG has no validated feasible solution");
    Json result = sub.solution;
    const double opening = result["opening_cost"].get<double>();
    const double upper = result["objective"].get<double>();
    const double continuous_bound = sub.ordinary.valid ? std::max(0.0, value(sub.ordinary, design)) : 0;
    const double lower = std::min(upper, opening + continuous_bound);
    result["method"] = options.column_generation ? "fixed_design_cg" : "fixed_design_benders_full";
    result["solver"] = "CPLEX";
    result["incumbent_kind"] = "validated_fixed_design";
    result["bound"] = lower;
    result["continuous_bound"] = continuous_bound;
    result["gap"] = gap(upper, lower);
    result["termination_reason"] = sub.reason;
    result["pricing_complete"] = sub.ordinary.valid;
    result["construction_complete"] = true;
    result["catalogue_complete"] = paths.complete;
    result["support_complete"] = paths.support_complete;
    result["initial_columns"] = seeds.metadata;
    result["initial_service_columns"] = seeds.service.size();
    result["initial_empty_columns"] = seeds.empty.size();
    result["cg_iterations"] = sub.iterations;
    result["cg_trace"] = sub.trace;
    result["service_columns"] = sub.service.size();
    result["empty_columns"] = sub.empty.size();
    result["pricing_method"] = options.column_generation ? "exact_resource_labeling" : "all_columns";
    result["cg_max_columns"] = options.cg_max_columns;
    result["pricing_tolerance"] = options.pricing_tolerance;
    result["cg_duality_tolerance"] = options.cg_duality_tolerance;
    result["phase_one"] = "unnecessary_validated_all_courier_complete_recourse";
    result["timings"] = sub.timings;
    result["timings"]["greedy_initial_columns"] = seeds.metadata.at("seconds");
    result["algorithm_wall"] = wall_time() - started;
    result["demand_formulation"] = options.conditional_demand ? "conditional" : "lifted";
    result["structural_strengthening"] = structural_metadata(options);
    result["subproblem_model_size"] = sub.model_size;
    result["subproblem_model_size_peak"] = sub.model_size;
    if (sub.ordinary.valid)
        result["benders_cuts"] = Json::array({{{"kind", "ordinary"}, {"intercept", sub.ordinary.intercept},
            {"coefficients", sub.ordinary.coefficients}, {"correction", sub.ordinary.correction},
            {"stationarity_error", sub.ordinary.stationarity_error}, {"pricing_complete", true},
            {"catalogue_complete", paths.complete}, {"support_complete", paths.support_complete},
            {"condition",sub.ordinary.condition}}});
    else result["benders_cuts"] = Json::array();
    if (options.progress) {
        options.progress->record({{"event","global_bounds"},{"stage","fixed_design_cg_finished"},
            {"objective",upper},{"upper_bound",upper},{"bound",lower},{"lower_bound",lower},
            {"gap",result.at("gap")},{"bound_scope","fixed_design"},{"validated_incumbent",true},
            {"pricing_complete",sub.ordinary.valid}});
        options.progress->checkpoint(result,"fixed_design_cg_finished",true);
    }
    return result;
}

Json solve_benders(const Input& d, PathCollection& paths, const SolverOptions& options,
                   const Json* start_solution) {
    if (options.heuristic_frequency < 1 || options.pareto_frequency < 1 || options.root_cut_rounds < 0)
        throw std::invalid_argument("Invalid Benders strengthening frequency");
    if (!paths.support_complete) throw std::runtime_error("Benders requires complete initial-segment support");
    if (options.topology_cut_limit < 0 || !std::isfinite(options.topology_seconds) || options.topology_seconds <= 0)
        throw std::invalid_argument("Invalid topology separation budget");
    if (options.assignment_support_cut_limit < 0 || options.assignment_support_max_cuts < 1 ||
        !std::isfinite(options.assignment_support_seconds) || options.assignment_support_seconds <= 0 ||
        !std::isfinite(options.assignment_support_floor) || options.assignment_support_floor <= 0)
        throw std::invalid_argument("Invalid assignment-support separation settings");
    if (options.branch_joint_cut_limit < 0 || options.branch_joint_max_cuts < 1 ||
        !std::isfinite(options.branch_joint_seconds) || options.branch_joint_seconds <= 0 ||
        !std::isfinite(options.branch_joint_preprocessing_seconds) || options.branch_joint_preprocessing_seconds <= 0 ||
        !std::isfinite(options.branch_joint_floor) || options.branch_joint_floor <= 0)
        throw std::invalid_argument("Invalid branch/joint separation settings");
    if (!options.column_generation && !paths.complete) throw std::runtime_error("Full recourse requires a complete path catalogue");
    if (start_solution) validate_design(d, solution_design(d, *start_solution));
    const double started = wall_time(), deadline = started + options.seconds;
    const auto seeds=initial_columns(d,paths,options,deadline);
    Json best = all_courier_result(d, paths);
    best["incumbent_kind"] = "all_courier_baseline";
    double upper = best["objective"].get<double>(), lower = 0;
    Formulation master = design_model(d, true, formulation_strengthening(options));
    int theta = master.matrix.variable("operating_cost", 0, courier_cost(d), 1);
    Json bound_metadata = options.courier_savings_bound ? add_master_bounds(d, paths, master, theta, options)
                            : Json{{"enabled", false}, {"reason", "disabled"}};
    int cutoff_row = -1;
    if (options.incumbent_cutoff) {
        Coefficients objective{{theta, 1}};
        for (int j : master.design_vars)
            if (master.matrix.variables[j].cost != 0)
                objective.emplace_back(j, master.matrix.variables[j].cost);
        cutoff_row = master.matrix.row("incumbent_objective_cutoff", 'L',
                                      upper + 1e-8 * (1 + std::abs(upper)), std::move(objective));
    }
    const std::size_t initial_master_rows = master.matrix.rows.size();
    NativeSolver solver(master.matrix, options, "benders_master");
    const double master_build = wall_time() - started - seeds.metadata.at("seconds").get<double>();
    std::vector<double> core = initial_core(d), original_core = core;
    std::set<int> pool_p(seeds.service.begin(),seeds.service.end()), pool_e(seeds.empty.begin(),seeds.empty.end());
    int cuts = 0, ordinary_count = 0, pareto_count = 0, auxiliary_count = 0, pareto_attempts = 0, cg_iterations = 0;
    int conditional_cuts = 0, conditional_subproblems = 0;
    std::set<std::string> assignment_patterns;
    std::set<std::size_t> conditional_row_indices, topology_row_indices, assignment_support_row_indices;
    std::set<std::string> topology_keys;
    Json topology_records = Json::array(), topology_attempts = Json::array();
    int topology_skipped = 0, topology_certificates = 0;
    double topology_seconds = 0;
    AssignmentSupportCache assignment_support_cache;
    std::set<std::string> assignment_support_keys;
    Json assignment_support_records = Json::array(), assignment_support_attempts = Json::array();
    int assignment_support_skipped = 0, assignment_support_certificates = 0;
    std::size_t assignment_support_numerical_skips = 0;
    double assignment_support_seconds = 0;
    const bool branch_joint_enabled = (options.double_support_cuts || options.joint_base_cost_cuts)
        && options.branch_joint_cut_limit > 0;
    BranchJointCache branch_joint_cache;
    std::set<std::size_t> branch_joint_row_indices;
    std::set<std::string> branch_joint_keys;
    Json branch_joint_records = Json::array(), branch_joint_attempts = Json::array();
    int branch_joint_skipped = 0, double_support_count = 0, joint_base_count = 0;
    std::size_t branch_joint_certificates = 0, branch_joint_numerical_skips = 0;
    double branch_joint_seconds = 0;
    Json subproblem_size=nullptr, subproblem_peak=nullptr;
    int cleanup_improvements = 0, primal_search_attempts = 0;
    int ordinary_cg_iterations = 0, heuristic_cg_iterations = 0, master_starts = 0;
    int root_count = 0, restricted_attempts = 0, restricted_improvements = 0;
    int pareto_failures = 0, next_pareto_call = 0;
    int pareto_budget_failures = 0, master_stalled_passes = 0;
    int root_master_solves = 0, root_cg_calls = 0, integer_master_solves = 0, integer_cg_calls = 0;
    double heuristic_seconds = 0, restricted_seconds = 0, pareto_seconds = 0;
    double cleanup_savings = 0;
    std::vector<double> last_neighbor_design;
    std::vector<double> last_master_start;
    std::set<std::vector<double>> integer_designs;
    int repeated_integer_designs = 0, all_courier_candidates = 0, bound_stalled_passes = 0;
    std::vector<double> pending_design;
    double pending_theta = 0;
    std::string active_candidate, pending_candidate;
    Json iterations = Json::array(), heuristic_records = Json::array(), cut_records = Json::array();
    Json master_records = Json::array(), resource_records = Json::array(), resource_attempts = Json::array();
    Json root_records = Json::array();
    Json continuous_root = {{"enabled", options.continuous_root}, {"bound_valid", false},
                            {"termination_reason", "disabled"}};
    int continuous_bound_rows = 0;
    bool continuous_root_closed = false;
    Json neighborhood_candidates = Json::array();
    std::set<std::vector<double>> neighborhood_seen;
    int neighborhood_cg_calls = 0, neighborhood_cut_count = 0;
    double neighborhood_seconds = 0;
    double root_lower = 0, root_upper = upper;
    std::string root_reason = "disabled";
    Json timing = {{"master_build", master_build}, {"master_optimize", 0.0}, {"master_update", 0.0},
                   {"greedy_initial_columns", seeds.metadata.at("seconds")},
                   {"subproblem_wall", 0.0}, {"subproblem", Json::object()}};
    assignment_support_cache.metadata = {{"enabled", false}};
    if (((options.assignment_support_cuts && options.assignment_support_cut_limit > 0) || branch_joint_enabled)
        && remaining(deadline) > 0) {
        const double phase = wall_time();
        assignment_support_cache = build_assignment_support_cache(d, paths,
            std::min(deadline, phase + options.assignment_support_seconds));
        assignment_support_cache.metadata["enabled"] = true;
        add_time(timing, "assignment_support_preprocessing", phase);
    }
    branch_joint_cache.metadata = {{"enabled", false}};
    if (branch_joint_enabled && remaining(deadline) > 0) {
        const double phase = wall_time();
        branch_joint_cache = build_branch_joint_cache(d, paths, assignment_support_cache,
            std::min(deadline, phase + options.branch_joint_preprocessing_seconds), options.branch_joint_floor);
        branch_joint_cache.metadata["enabled"] = true;
        add_time(timing, "branch_joint_preprocessing", phase);
    }
    auto branch_joint_summary = [&]() -> Json {
        return {{"enabled",branch_joint_enabled},{"double_support_enabled",options.double_support_cuts},
            {"joint_base_cost_enabled",options.joint_base_cost_cuts},{"cuts",branch_joint_records.size()},
            {"double_cuts",double_support_count},{"joint_cuts",joint_base_count},
            {"attempts",branch_joint_attempts.size()},{"skipped_integer_subproblems",branch_joint_skipped},
            {"certificates",branch_joint_certificates},{"numerical_skips",branch_joint_numerical_skips},
            {"seconds",branch_joint_seconds},{"preprocessing_seconds",branch_joint_cache.seconds},
            {"master_added_variables",0},{"validity_scope","binary_designs_branch_support_and_distinct_origin_costs"}};
    };
    auto observe_global = [&](const char* stage, bool checkpoint = false) {
        if (!options.progress) return;
        options.progress->record({{"event","global_bounds"},{"stage",stage},{"objective",upper},{"upper_bound",upper},
            {"bound",lower},{"lower_bound",lower},{"gap",gap(upper,lower)},{"bound_scope","global"},
            {"validated_incumbent",true},{"certified_cuts",cuts},
            {"path_sizes",{{"segments",paths.initial_segments.size()},{"service",paths.service_paths.size()},
                {"empty",paths.empty_paths.size()}}}});
        if (checkpoint && options.progress->checkpoint_needed(upper)) {
            Json snapshot=best;
            synchronize_result_paths(d,paths,snapshot);
            snapshot["bound"]=lower; snapshot["gap"]=gap(upper,lower);
            snapshot["tight_airline_bounds"]=options.tight_airline_bounds;
            snapshot["od_airline_bounds"]=options.od_airline_bounds;
            snapshot["topology_summary"]={{"enabled",options.topology_cuts},{"cuts",topology_records.size()},
                {"attempts",topology_attempts.size()},{"skipped_integer_subproblems",topology_skipped},
                {"zero_service_certificates",topology_certificates},{"seconds",topology_seconds}};
            snapshot["assignment_support_summary"]={{"enabled",options.assignment_support_cuts},
                {"cuts",assignment_support_records.size()},{"attempts",assignment_support_attempts.size()},
                {"skipped_integer_subproblems",assignment_support_skipped},
                {"certificates",assignment_support_certificates},{"numerical_skips",assignment_support_numerical_skips},
                {"seconds",assignment_support_seconds},{"preprocessing_seconds",assignment_support_cache.seconds}};
            snapshot["construction_complete"]=true;
            snapshot["branch_joint_summary"]=branch_joint_summary();
            snapshot["support_complete"]=paths.support_complete;
            snapshot["catalogue_complete"]=paths.complete;
            options.progress->checkpoint(snapshot,stage);
        }
    };
    observe_global("benders_initial_baseline",true);
    CapacitySavingsNetwork capacity_network;
    const bool capacity_enabled = options.capacity_cuts && options.capacity_cut_limit > 0;
    Json capacity_records = Json::array();
    std::set<std::string> capacity_partitions;
    if (capacity_enabled) {
        const double phase = wall_time();
        capacity_network = capacity_savings_network(d);
        add_time(timing,"capacity_cut_build",phase);
    }
    const bool resource_enabled = options.resource_lp_cuts && options.resource_lp_cut_limit > 0;
    const bool integer_strengthened_root = resource_enabled && options.od_airline_bounds;
    std::unique_ptr<ResourceSavingsRelaxation> resource_relaxation;
    std::set<std::string> resource_keys;
    if (resource_enabled && remaining(deadline) > 0) {
        const double phase = wall_time();
        resource_relaxation = capacity_enabled
            ? std::make_unique<ResourceSavingsRelaxation>(resource_savings_model(d, capacity_network, options.resource_airline_bounds, options.tight_airline_bounds, options.od_airline_bounds), options)
            : std::make_unique<ResourceSavingsRelaxation>(d, options);
        add_time(timing, "resource_lp_build", phase);
    }
    auto separate_capacity = [&](const std::vector<double>& point,const char* stage) {
        if (!capacity_enabled || static_cast<int>(capacity_records.size())>=options.capacity_cut_limit ||
            remaining(deadline)<=0) return false;
        const double phase = wall_time();
        std::vector<double> design(master.layout.count);
        for (int j=0;j<master.layout.count;++j) design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        auto cut=capacity_network.separate(design,point[theta],theta);
        const bool insert=cut.violated && capacity_partitions.insert(cut.partition_key).second;
        if (insert) {
            cut.row.name += "_"+std::to_string(capacity_records.size());
            master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
            capacity_records.push_back({{"stage",stage},{"candidate_id",active_candidate},{"violation",cut.violation},
                {"cut_capacity",cut.cut_capacity},{"maxflow_diagnostic",cut.maxflow},
                {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},
                {"partition",cut.partition_key},{"certificate","original_crossing_affine_capacities"}});
        }
        add_time(timing,"capacity_cut_separation",phase);
        return insert;
    };
    auto separate_resource = [&](const std::vector<double>& point, const char* stage, double stage_deadline) {
        if (!resource_relaxation || static_cast<int>(resource_records.size()) >= options.resource_lp_cut_limit ||
            remaining(stage_deadline) <= 0) return false;
        const double phase = wall_time();
        std::vector<double> design(master.layout.count);
        for (int j = 0; j < master.layout.count; ++j)
            design[j] = std::clamp(point[master.design_vars[j]], 0.0, 1.0);
        auto cut = resource_relaxation->separate(design, point[theta], theta,
            std::min(options.resource_lp_seconds, remaining(stage_deadline)));
        const bool insert = cut.valid && cut.violated && resource_keys.insert(cut.key).second;
        Json record = {{"stage", stage}, {"candidate_id", active_candidate}, {"accepted", insert},
            {"valid", cut.valid}, {"violated", cut.violated}, {"optimal", cut.optimal},
            {"fallback", cut.fallback}, {"status", cut.status}, {"seconds", cut.seconds},
            {"solver_seconds", cut.solver_seconds}, {"violation", cut.violation},
            {"dual_savings_upper_bound", cut.dual_savings},
            {"max_dual_violation_before", cut.max_dual_violation_before},
            {"max_dual_violation_after", cut.max_dual_violation_after},
            {"validity_scope",options.od_airline_bounds ? "integer_design_convex_hull" : "original_continuous_relaxation"},
            {"certificate", "full_occurrence_domain_dual_feasibility"}};
        const std::size_t first_od = resource_relaxation->metadata().value("legacy_airline_resource_rows", cut.airline_dual.size());
        std::size_t active_od = 0;
        if (options.od_airline_bounds)
            for (std::size_t a = first_od; a < cut.airline_dual.size(); ++a)
                if (cut.airline_dual[a] > 1e-10) ++active_od;
        record["active_od_airline_prices"] = active_od;
        resource_attempts.push_back(record);
        if (insert) {
            cut.row.name += "_" + std::to_string(resource_records.size());
            master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
            record["rhs"] = cut.row.rhs; record["coefficients"] = cut.row.coefficients;
            record["demand_dual"] = cut.demand_dual; record["locker_dual"] = cut.locker_dual;
            record["airline_dual"] = cut.airline_dual;
            resource_records.push_back(record);
        }
        if (options.progress) {
            record["event"] = "resource_lp_separation";
            options.progress->record(std::move(record));
        }
        add_time(timing, "resource_lp_separation", phase);
        return insert;
    };
    std::string reason = "iteration_limit";
    Json budget_interruption = nullptr;
    bool pricing_complete = false;
    Json last_subproblem_stage = nullptr, last_subproblem_termination = nullptr;
    auto keep_cut = [&](const Cut& cut, const char* kind) {
        if (!cut.valid) return;
        RowRecord row;
        row.name = (cut.condition.is_null()?"":"conditional_") + std::string(kind) + "_cut_" + std::to_string(cuts);
        row.sense = 'G'; row.rhs = cut.intercept; row.coefficients.emplace_back(theta, 1);
        for (std::size_t j = 0; j < cut.coefficients.size(); ++j)
            if (cut.coefficients[j] != 0) row.coefficients.emplace_back(master.design_vars[j], -cut.coefficients[j]);
        if (!cut.condition.is_null()) conditional_row_indices.insert(master.matrix.rows.size());
        master.matrix.rows.push_back(row);
        solver.add_row(row);
        cut_records.push_back({{"kind", kind}, {"candidate_id",active_candidate},{"intercept", cut.intercept},
                               {"coefficients", cut.coefficients}, {"correction", cut.correction},
                               {"stationarity_error", cut.stationarity_error}, {"pricing_complete", true},
                               {"catalogue_complete", paths.complete}, {"support_complete", paths.support_complete},
                               {"condition",cut.condition}});
        if (!cut.condition.is_null()) {
            ++conditional_cuts;
            assignment_patterns.insert(cut.condition.at("assignment_fingerprint").get<std::string>());
        }
        ++cuts;
        if (std::string(kind) == "pareto") ++pareto_count;
        else if (std::string(kind) == "auxiliary_valid") ++auxiliary_count;
        else { ++ordinary_count; if (std::string(kind) == "root") ++root_count; }
    };
    auto update_cutoff = [&]() {
        if (cutoff_row < 0) return;
        double rhs = upper + 1e-8 * (1 + std::abs(upper));
        master.matrix.rows[cutoff_row].rhs = rhs;
        solver.set_rhs(cutoff_row, 'L', rhs);
    };
    auto account = [&](Subproblem& sub, bool heuristic = false, const char* stage = "benders") {
        if (std::string(stage)!="root_relaxation" && !sub.model_size.is_null()) {
            subproblem_size=sub.model_size;
            if (subproblem_peak.is_null()) subproblem_peak=sub.model_size;
            else for (const char* key:{"variables","linear_rows","cones","nonzeros"})
                subproblem_peak[key]=std::max(subproblem_peak.at(key).get<std::size_t>(),sub.model_size.at(key).get<std::size_t>());
        }
        if (sub.conditional_demand) ++conditional_subproblems;
        timing["subproblem_wall"] = timing["subproblem_wall"].get<double>() + sub.wall;
        if (std::string(stage) == "root_relaxation")
            timing["root_projection_cg"] = timing.value("root_projection_cg", 0.0) + sub.wall;
        for (auto it = sub.timings.begin(); it != sub.timings.end(); ++it)
            timing["subproblem"][it.key()] = timing["subproblem"].value(it.key(), 0.0) + it.value().get<double>();
        cg_iterations += sub.iterations;
        (heuristic ? heuristic_cg_iterations : ordinary_cg_iterations) += sub.iterations;
        if (options.reuse_columns) {
            pool_p.insert(sub.service.begin(), sub.service.end());
            pool_e.insert(sub.empty.begin(), sub.empty.end());
        }
        if (options.incumbent_cleanup && !sub.solution.is_null()) {
            double phase = wall_time();
            Json cleaned = cleanup_incumbent(d, paths, sub.solution);
            double saving = sub.solution["objective"].get<double>() - cleaned["objective"].get<double>();
            if (saving > 1e-8) { ++cleanup_improvements; cleanup_savings += saving; }
            sub.solution = std::move(cleaned);
            add_time(timing, "incumbent_cleanup", phase);
        }
        if (!sub.solution.is_null() && sub.solution["objective"].get<double>() < upper) {
            double candidate = sub.solution["objective"].get<double>();
            if (candidate < lower - 1e-7 * (1 + std::abs(candidate)))
                throw std::runtime_error("Native primal contradicts certified Benders lower bound");
            best = sub.solution; upper = candidate; best["incumbent_kind"] = "optimized";
        }
        double phase = wall_time();
        keep_cut(sub.ordinary, std::string(stage) == "root_relaxation" ? "root" : "ordinary");
        const bool retain_auxiliary = options.auxiliary_valid_cuts && sub.auxiliary.valid &&
            sub.pareto.value("core_gain", 0.0) > 1e-7 * (1 + courier_cost(d));
        if (sub.pareto.value("attempted", false)) {
            ++pareto_attempts;
            pareto_seconds += sub.timings.at("pareto").get<double>();
            pareto_failures = (sub.pareto.value("accepted", false) || retain_auxiliary)
                ? 0 : pareto_failures + 1;
        }
        if (sub.pareto.value("accepted", false)) keep_cut(sub.selected, "pareto");
        else if (retain_auxiliary) {
            keep_cut(sub.auxiliary, "auxiliary_valid");
            sub.pareto["retained_as_valid_cut"] = true;
        }
        update_cutoff();
        add_time(timing, "master_update", phase);
        pricing_complete = sub.ordinary.valid;
        last_subproblem_stage = stage;
        last_subproblem_termination = sub.reason;
        observe_global(stage,true);
    };
    auto separate_neighborhood = [&]() {
        const double available = options.seconds * options.neighborhood_cut_budget_fraction - neighborhood_seconds;
        if (!options.neighborhood_cuts || neighborhood_candidates.empty() || available <= .01 || remaining(deadline) <= .01)
            return;
        // Choose a design whose known feasible operating cost is most
        // underestimated by the CURRENT global affine cuts. This is only a
        // candidate score, never a recourse certificate or global bound.
        int chosen = 0;
        double largest_score = -INFINITY, predicted_theta = 0;
        for (std::size_t i = 0; i < neighborhood_candidates.size(); ++i) {
            auto x = neighborhood_candidates[i].at("design").get<std::vector<double>>();
            double floor = 0, fixed = 0;
            for (int j = 0; j < master.layout.count; ++j) fixed += master.matrix.variables[j].cost * x[j];
            for (const auto& row : master.matrix.rows) {
                if (row.sense != 'G') continue;
                double coefficient = 0, other = 0;
                for (auto [j, a] : row.coefficients) {
                    if (j == theta) coefficient += a;
                    else other += a * x[j];
                }
                if (coefficient > 0) floor = std::max(floor, (row.rhs - other) / coefficient);
            }
            const double score = neighborhood_candidates[i].at("objective").get<double>() - fixed - floor;
            if (score > largest_score) { chosen = static_cast<int>(i); largest_score = score; predicted_theta = floor; }
        }
        const Json candidate = neighborhood_candidates[chosen];
        neighborhood_candidates.erase(neighborhood_candidates.begin() + chosen);
        const auto design = candidate.at("design").get<std::vector<double>>();
        const std::string previous_candidate = active_candidate;
        active_candidate = "neighborhood:" + std::to_string(++neighborhood_cg_calls);
        const int cuts_before = cuts;
        std::vector<int> selected_p(pool_p.begin(), pool_p.end()), selected_e(pool_e.begin(), pool_e.end());
        const double allowance = std::min({available, options.neighborhood_cut_seconds, .1 * remaining(deadline)});
        auto sub = subproblem(d, paths, design, options, std::min(deadline, wall_time() + allowance),
                              selected_p, selected_e, nullptr);
        neighborhood_seconds += sub.wall;
        account(sub, true, "neighborhood_cut");
        neighborhood_cut_count += cuts - cuts_before;
        heuristic_records.push_back({{"stage", "neighborhood_cut"}, {"candidate_id", active_candidate},
            {"candidate_source", candidate.value("source", "restricted_mip_pool")},
            {"candidate_score", largest_score}, {"master_operating_floor", predicted_theta},
            {"time_budget", allowance}, {"wall", sub.wall}, {"benders_cuts_added", cuts - cuts_before},
            {"pricing_complete", sub.ordinary.valid}, {"termination_reason", sub.reason},
            {"cg_iterations", sub.iterations}, {"cg_trace", sub.trace},
            {"regional_bound_used", false}});
        if (options.progress) {
            Json record = heuristic_records.back(); record["event"] = "neighborhood_cut_evaluation";
            options.progress->record(std::move(record));
        }
        active_candidate = previous_candidate;
    };
    auto restricted_search = [&](const char* stage, int radius) {
        const double available = options.seconds * options.heuristic_budget_fraction - heuristic_seconds;
        if (!options.restricted_mip || available <= .01 || remaining(deadline) <= .01) return;
        const double before = upper, phase = wall_time();
        const double allowance = std::min({available, 15.0, .08 * options.seconds, remaining(deadline)});
        ++restricted_attempts;
        auto attempt = solve_restricted_heuristic(d, paths, options, phase + allowance, &best, radius);
        const double used = wall_time() - phase;
        heuristic_seconds += used; restricted_seconds += used;
        if (options.neighborhood_cuts && attempt.contains("design_pool")) {
            for (const auto& candidate : attempt.at("design_pool")) {
                auto x = candidate.at("design").get<std::vector<double>>();
                if (!candidate.value("validated", false) || neighborhood_seen.count(x)) continue;
                validate_design(d, x);
                if (neighborhood_candidates.size() >= static_cast<std::size_t>(2 * options.heuristic_pool_size)) break;
                neighborhood_seen.insert(x);
                neighborhood_candidates.push_back(candidate);
            }
        }
        if (attempt.value("validated", false) && attempt.contains("solution") && !attempt["solution"].is_null()) {
            Json candidate = attempt["solution"];
            if (options.incumbent_cleanup) candidate = cleanup_incumbent(d, paths, candidate);
            const double objective = candidate.at("objective").get<double>();
            if (objective < lower - 1e-7 * (1 + std::abs(objective)))
                throw std::runtime_error("Restricted primal contradicts certified global lower bound");
            if (objective < upper) {
                best = std::move(candidate); upper = objective; ++restricted_improvements;
                best["incumbent_kind"] = radius < 0 ? "restricted_column_mip" : "local_branching";
            }
        }
        attempt.erase("solution");
        attempt["stage"] = stage;
        attempt["upper_before"] = before; attempt["upper_after"] = upper;
        attempt["cg_trace"] = Json::array();
        heuristic_records.push_back(std::move(attempt));
        update_cutoff();
        observe_global(stage,true);
    };
    try {
    if (options.continuous_root && remaining(deadline) > 0 && options.continuous_root_budget_fraction > 0) {
        active_candidate = "continuous_root";
        SolverOptions continuous_options = options;
        continuous_options.seconds = std::min({options.continuous_root_seconds,
            options.continuous_root_budget_fraction * options.seconds, remaining(deadline)});
        continuous_options.cg_max_columns = options.continuous_root_max_columns;
        continuous_options.pareto = false;
        const double phase = wall_time();
        if (options.progress) options.progress->record({{"event", "phase_start"},
            {"stage", "continuous_root"}, {"allocated_seconds", continuous_options.seconds}});
        try {
            continuous_root = solve_continuous_cg(d, paths, continuous_options);
        } catch (const RunBudgetExceeded& error) {
            continuous_root = {{"bound_valid", false}, {"termination_reason", "time_limit"},
                               {"budget_interruption", {{"stage", error.stage}, {"error", error.what()}}}};
        }
        continuous_root["enabled"] = true;
        continuous_root["allocated_seconds"] = continuous_options.seconds;
        add_time(timing, "continuous_root", phase);
        // These are physical columns shared by the same formulation. Reusing
        // them does not fix any fractional design or confer integer feasibility.
        for (std::size_t j = 0; j < paths.service_paths.size(); ++j) pool_p.insert(static_cast<int>(j));
        for (std::size_t j = 0; j < paths.empty_paths.size(); ++j) pool_e.insert(static_cast<int>(j));
        if (continuous_root.value("bound_valid", false)) {
            const double bound = continuous_root.at("certified_lower_bound").get<double>();
            // Q includes only operational cost; this is a TOTAL-cost lower
            // bound and must retain F(x) on the left hand side.
            const double safe_bound = bound - 1e-8 * (1 + std::abs(bound));
            if (!std::isfinite(bound) || safe_bound > upper)
                throw std::runtime_error("Certified continuous bound contradicts integer incumbent");
            continuous_root["retained_global_lower_bound"] = safe_bound;
            continuous_root["master_floor_added"] = false;
            if (safe_bound > lower) {
                // The complete-domain dual certificate remains valid even if
                // the global budget expires while updating the master row.
                lower = safe_bound;
                root_lower = std::max(root_lower, safe_bound);
                Coefficients lhs{{theta, 1}};
                for (int j : master.design_vars)
                    if (master.matrix.variables[j].cost != 0)
                        lhs.emplace_back(j, master.matrix.variables[j].cost);
                master.matrix.row("continuous_global_objective_floor", 'G', safe_bound, std::move(lhs));
                ++continuous_bound_rows;
                continuous_root["master_floor_recorded"] = true;
                solver.add_row(master.matrix.rows.back());
                continuous_root["master_floor_added"] = true;
                continuous_root["applied_total_cost_lower_bound"] = safe_bound;
            }
        }
        if (options.progress) options.progress->record({{"event", "continuous_root_result"},
            {"stage", "continuous_root"}, {"bound_scope", continuous_scope(options)},
            {"diagnostics", continuous_root}});
        observe_global("continuous_root_finished", true);
    }
    if ((options.initial_design || start_solution) && remaining(deadline) > 0 &&
        options.seconds * options.heuristic_budget_fraction > heuristic_seconds) {
        std::vector<double> design = start_solution ? solution_design(d, *start_solution) :
            (restricted_improvements ? solution_design(d, best) : initial_design(d, paths));
        double allowance = std::min({30.0,.05*options.seconds,
                                    options.seconds*options.heuristic_budget_fraction-heuristic_seconds});
        double heuristic_deadline = std::min(deadline, wall_time() + allowance);
        active_candidate = start_solution ? "supplied_design" : "initial_design";
        auto sub = subproblem(d, paths, design, options, heuristic_deadline, seeds.service, seeds.empty, nullptr);
        heuristic_seconds += sub.wall;
        account(sub, true, start_solution ? "supplied_design" : "initial_design");
        heuristic_records.push_back({{"stage", start_solution ? "supplied_design" : "initial_design"},
                                     {"candidate_id",active_candidate},
                                     {"termination_reason", sub.reason}, {"pricing_complete", sub.ordinary.valid},
                                     {"wall", sub.wall}, {"cg_iterations", sub.iterations}, {"cg_trace", sub.trace}});
    }
    // Give the joint MIP a useful operational warm start, instead of asking
    // it to discover all discrete design and continuation choices from zero.
    restricted_search("initial_restricted_mip", -1);
    separate_neighborhood();
    // The continuous master and fixed-fractional recourse share exactly the
    // integer model's matrix. Fractional recourse is only for lower cuts; its
    // primal point is never rounded or admitted as an integer incumbent.
    const double root_deadline = std::min(deadline, wall_time() + options.seconds * options.root_budget_fraction);
    root_upper = upper;
    if (!integer_strengthened_root && continuous_root.value("primal_valid", false)) {
        const double continuous_value = continuous_root.at("continuous_objective").get<double>();
        if (!std::isfinite(continuous_value) ||
            continuous_value < root_lower - 2 * options.cg_duality_tolerance * (1 + std::abs(continuous_value)))
            throw std::runtime_error("Validated continuous primal contradicts continuous lower bound");
        root_upper = std::min(root_upper, continuous_value);
    }
    continuous_root_closed = continuous_root.value("bound_valid", false) &&
        continuous_root.value("primal_valid", false) &&
        continuous_root.at("continuous_objective").get<double>() -
            continuous_root.at("certified_lower_bound").get<double>() <=
            2 * options.cg_duality_tolerance *
                (1 + std::abs(continuous_root.at("continuous_objective").get<double>()));
    std::vector<double> projection_core = original_core;
    if (options.root_projection && continuous_root.value("primal_valid", false)) {
        const auto x = continuous_root.at("design").get<std::vector<double>>();
        validate_design(d, x, true);
        for (std::size_t j = 0; j < x.size(); ++j)
            projection_core[j] = .99 * x[j] + .01 * original_core[j];
    }
    auto record_root = [&](Json record) {
        record["lower_bound"] = root_lower;
        record["continuous_upper_bound"] = integer_strengthened_root ? Json(nullptr) : Json(root_upper);
        record["continuous_gap"] = integer_strengthened_root ? Json(nullptr) : Json(gap(root_upper, root_lower));
        record["bound_scope"] = integer_strengthened_root ? "integer_valid_strengthened_root" : continuous_scope(options);
        if (integer_strengthened_root) {
            record["integer_feasible_upper_bound"] = upper;
            record["gap_to_integer_incumbent"] = gap(upper, root_lower);
        }
        root_records.push_back(record);
        if (options.progress) {
            record["event"] = "root_separation_record";
            options.progress->record(std::move(record));
        }
    };
    if (options.root_cuts || capacity_enabled || resource_enabled) root_reason = "round_limit";
    if (continuous_root_closed && !integer_strengthened_root) root_reason = "continuous_gap_certified";
    if ((!continuous_root_closed || options.root_projection || integer_strengthened_root) &&
        (options.root_cuts || capacity_enabled || resource_enabled)) for (int round = 0; round < options.root_cut_rounds; ++round) {
        if (remaining(root_deadline) <= 0) { root_reason = "time_limit"; break; }
        const double phase = wall_time();
        LinearModel relaxation = master.matrix;
        // Conditional, topology and assignment-support cuts use binary decisions, not the chosen
        // lifted continuous feasible set. Keep the root's primal and dual
        // diagnostics on that same continuous relaxation by excluding exactly
        // these append-only row indices from its temporary copy. They remain
        // in the integer master and retain their full-domain integer validity.
        // OD resource cuts are deliberately retained when requested: this
        // then defines a DIFFERENT integer-valid root relaxation, whose LB
        // must never be compared with the original continuous primal as UB.
        if (!conditional_row_indices.empty() || !topology_row_indices.empty() || !assignment_support_row_indices.empty()
            || !branch_joint_row_indices.empty()) {
            std::vector<RowRecord> root_rows;
            root_rows.reserve(relaxation.rows.size()-conditional_row_indices.size()-topology_row_indices.size()
                -assignment_support_row_indices.size()-branch_joint_row_indices.size());
            for (std::size_t j=0;j<relaxation.rows.size();++j)
                if (!conditional_row_indices.count(j) && !topology_row_indices.count(j) && !assignment_support_row_indices.count(j)
                    && !branch_joint_row_indices.count(j))
                    root_rows.push_back(std::move(relaxation.rows[j]));
            relaxation.rows=std::move(root_rows);
        }
        for (auto& variable : relaxation.variables) variable.binary = false;
        NativeSolver root_solver(relaxation, options, "root_master");
        ++root_master_solves;
        auto root = root_solver.optimize(remaining(root_deadline));
        add_time(timing, "root_master", phase);
        if (!root.optimal || !root.has_solution || !feasible_point(relaxation, root.values)) {
            root_reason = "master_not_certified_" + root.status; break;
        }
        if (root.objective > upper + 1e-7 * (1 + std::abs(upper)))
            throw std::runtime_error("Root relaxation contradicts incumbent");
        lower = std::max(lower, std::min(upper, root.objective - 1e-8 * (1 + std::abs(root.objective))));
        root_lower = lower;
        observe_global("root_master_finished");
        active_candidate = "root:" + std::to_string(round + 1);
        const bool capacity_added = separate_capacity(root.values,"root_relaxation");
        const bool resource_added = separate_resource(root.values,"root_relaxation",root_deadline);
        Json root_record = {{"round", round + 1}, {"candidate_id", active_candidate},
            {"conditional_rows_excluded",conditional_row_indices.size()},
            {"topology_rows_excluded",topology_row_indices.size()},
            {"assignment_support_rows_excluded",assignment_support_row_indices.size()},
            {"branch_joint_rows_excluded",branch_joint_row_indices.size()},
            {"master_objective", root.objective}, {"lower_bound", root_lower},
            {"theta", root.values[theta]}, {"capacity_cuts_added", capacity_added ? 1 : 0},
            {"resource_lp_cuts_added", resource_added ? 1 : 0}, {"pricing_complete", false}};
        // A capacity cut and a certified recourse cut can strengthen the same
        // fractional candidate. Capacity separation must not consume a CG round.
        if (!options.root_cuts) {
            record_root(root_record);
            if (capacity_added || resource_added) continue;
            root_reason = "resource_separation_complete_cg_disabled";
            break;
        }
        std::vector<double> design(master.layout.count);
        for (int j=0;j<master.layout.count;++j) design[j]=std::clamp(root.values[master.design_vars[j]],0.0,1.0);
        root_record["design"] = design;
        root_record["evaluations"] = Json::array();
        root_record["benders_cuts_added"] = 0;
        root_record["cg_seconds"] = 0.0;
        const bool use_inout = options.root_projection && options.root_inout;
        bool point_closed = false, separated = false, any_certified = false;
        std::string last_reason = "not_evaluated";
        // In-out selects a legal fractional point. If that cut does not
        // separate the CURRENT master point, certify the original point as
        // a fallback before claiming pointwise consistency.
        for (int pass = 0; pass < (use_inout ? 2 : 1); ++pass) {
            if (remaining(root_deadline) <= 0) break;
            const bool mixed = use_inout && pass == 0;
            std::vector<double> anchor = design;
            if (mixed) for (std::size_t j = 0; j < anchor.size(); ++j)
                anchor[j] = options.root_inout_weight * design[j]
                    + (1 - options.root_inout_weight) * projection_core[j];
            std::vector<int> selected_p(pool_p.begin(),pool_p.end()), selected_e(pool_e.begin(),pool_e.end());
            const int cuts_before = cuts;
            ++root_cg_calls;
            auto sub = subproblem(d,paths,anchor,options,root_deadline,selected_p,selected_e,nullptr,
                                  true,false,options.root_projection);
            account(sub, false, "root_relaxation");
            last_reason = sub.reason;
            Json evaluation = {{"anchor_kind",mixed ? "inout" : "master_point"}, {"design",anchor},
                {"pricing_complete",sub.ordinary.valid},{"termination_reason",sub.reason},
                {"benders_cuts_added",cuts-cuts_before},{"cg_seconds",sub.wall}};
            root_record["benders_cuts_added"] = root_record["benders_cuts_added"].get<int>() + cuts-cuts_before;
            root_record["cg_seconds"] = root_record["cg_seconds"].get<double>() + sub.wall;
            if (sub.ordinary.valid) {
                any_certified = true;
                const double q = sub.continuous_objective;
                double total = q;
                for (int j = 0; j < master.layout.count; ++j)
                    total += master.matrix.variables[j].cost * anchor[j];
                if (!std::isfinite(total) || (!integer_strengthened_root &&
                    total < root_lower - 2 * options.cg_duality_tolerance * (1 + std::abs(total))))
                    throw std::runtime_error("Continuous recourse incumbent contradicts root lower bound");
                if (!integer_strengthened_root) root_upper = std::min(root_upper, total);
                const double violation = value(sub.ordinary,design) - root.values[theta];
                const double tolerance = 2 * options.cg_duality_tolerance * (1 + std::abs(q));
                separated = violation > tolerance;
                point_closed = !mixed && q - root.values[theta] <= tolerance;
                evaluation["operating_objective"] = q;
                evaluation["continuous_feasible_objective"] = total;
                evaluation["original_candidate_cut_violation"] = violation;
                evaluation["original_candidate_primal_gap"] = mixed ? Json(nullptr) : Json(q-root.values[theta]);
                evaluation["ordinary_cut"] = {{"intercept",sub.ordinary.intercept},{"coefficients",sub.ordinary.coefficients}};
                if (!mixed) {
                    root_record["ordinary_cut"] = evaluation["ordinary_cut"];
                    root_record["operating_objective"] = q;
                    root_record["recourse_underestimation"] = q-root.values[theta];
                    root_record["continuous_feasible_objective"] = total;
                }
                for (std::size_t j = 0; j < anchor.size(); ++j)
                    projection_core[j] = .50 * projection_core[j] + .49 * anchor[j] + .01 * original_core[j];
            }
            root_record["evaluations"].push_back(evaluation);
            heuristic_records.push_back({{"stage","root_relaxation"},{"round",round+1},{"bound",lower},
                {"candidate_id",active_candidate},{"anchor_kind",mixed ? "inout" : "master_point"},
                {"capacity_cuts_added",pass==0 && capacity_added ? 1 : 0},
                {"resource_lp_cuts_added",pass==0 && resource_added ? 1 : 0},
                {"benders_cuts_added",cuts-cuts_before},{"termination_reason",sub.reason},
                {"pricing_complete",sub.ordinary.valid},{"cg_iterations",sub.iterations},
                {"cg_trace",sub.trace},{"wall",sub.wall}});
            if (separated || point_closed) break;
        }
        root_record["pricing_complete"] = any_certified;
        root_record["subproblem_termination"] = last_reason;
        root_record["candidate_epigraph_certified"] = !integer_strengthened_root && point_closed;
        root_record["original_candidate_recourse_consistent"] = point_closed;
        root_record["candidate_separated"] = separated;
        root_record["continuous_upper_bound"] = root_upper;
        root_record["continuous_gap"] = gap(root_upper,root_lower);
        record_root(std::move(root_record));
        if (!any_certified) { root_reason = "cg_not_certified_" + last_reason; break; }
        if (options.root_projection && point_closed && !capacity_added && !resource_added) {
            root_reason = integer_strengthened_root ? "original_candidate_recourse_consistent" : "candidate_epigraph_certified";
            break;
        }
        if (!integer_strengthened_root && !options.root_projection && root_upper-root_lower <=
            2*options.cg_duality_tolerance*(1+std::abs(root_upper))) {
            root_reason = "continuous_gap_certified"; break;
        }
        root_reason = "round_limit";
        if (gap(upper,lower)<=options.gap) { root_reason = "global_gap_limit"; break; }
    }
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        if (remaining(deadline) <= 0) { reason = "time_limit"; break; }
        const double lower_before_iteration = lower;
        double phase = wall_time();
        if (options.warm_start && pending_design.empty()) {
            auto design_start = solution_design(d, best);
            std::vector<int> start_indices = master.design_vars;
            std::vector<double> start_values = design_start;
            start_indices.push_back(theta); start_values.push_back(best["continuous_cost"].get<double>());
            if (start_values != last_master_start) {
                solver.add_start(start_values, start_indices);
                last_master_start = std::move(start_values);
                ++master_starts;
            }
        }
        add_time(timing, "master_update", phase);
        phase = wall_time();
        SolveOutcome solved;
        if (pending_design.empty()) {
            ++integer_master_solves;
            const auto control = master_solve_budget(options.adaptive_master, remaining(deadline),
                options.master_initial_gap, options.master_slice_seconds, gap(upper, lower),
                options.gap, master_stalled_passes);
            solver.cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, control.relative_gap);
            solved = solver.optimize(control.seconds);
            Json record = {{"solve", integer_master_solves}, {"seconds", solved.seconds},
                {"time_budget", control.seconds}, {"relative_gap_target", control.relative_gap},
                {"exact_phase", control.exact_phase}, {"stalled_passes", master_stalled_passes},
                {"status", solved.status}, {"optimal", solved.optimal}, {"has_solution", solved.has_solution},
                {"candidate_objective", solved.has_solution ? Json(solved.objective) : Json(nullptr)},
                {"certified_bound", std::isfinite(solved.bound) ? Json(solved.bound) : Json(nullptr)},
                {"global_lower_before", lower}, {"global_upper", upper}, {"nodes", solved.nodes}};
            master_records.push_back(record);
            if (options.progress) { record["event"] = "master_control"; options.progress->record(record); }
        }
        add_time(timing, "master_optimize", phase);
        if (pending_design.empty() && std::isfinite(solved.bound)) {
            if (solved.bound > upper + 1e-7 * (1 + std::abs(upper))) { reason = "numerical_bound_inconsistency"; break; }
            lower = std::max(lower, std::min(upper, solved.bound));
        }
        observe_global("integer_master_finished");
        if (gap(upper, lower) <= options.gap) { reason = "gap_limit"; break; }
        if (pending_design.empty() && !solved.has_solution) {
            if (options.adaptive_master && remaining(deadline) > 0 &&
                (solved.status.find("TimeLim") != std::string::npos || solved.status == "time_limit")) {
                ++master_stalled_passes; continue;
            }
            reason = "master_" + solved.status; break;
        }
        if (pending_design.empty() && !feasible_point(master.matrix, solved.values)) {
            reason = "master_candidate_validation_failed"; break;
        }
        const bool resumed_design = !pending_design.empty();
        active_candidate = resumed_design ? pending_candidate : "integer:" + std::to_string(integer_master_solves);
        // Add both cut families before reoptimizing the integer master. If CG
        // needs another slice, retain this candidate and its already valid cut.
        const bool capacity_added = !resumed_design && separate_capacity(solved.values,"integer_master");
        const bool resource_added = !resumed_design && separate_resource(solved.values,"integer_master",deadline);
        const double candidate_theta = resumed_design ? pending_theta : solved.values[theta];
        std::vector<double> design = pending_design;
        if (design.empty()) {
            design.resize(master.layout.count);
            for (int j = 0; j < master.layout.count; ++j) design[j] = std::round(solved.values[master.design_vars[j]]);
        }
        // Combine branch certificates before individual support cuts so a
        // candidate blocked at several bases receives their joint cost bound.
        if (!resumed_design && branch_joint_enabled && branch_joint_cache.support_complete &&
            static_cast<int>(branch_joint_records.size()) < options.branch_joint_cut_limit && remaining(deadline) > 0) {
            const int allowance = std::min(options.branch_joint_max_cuts,
                options.branch_joint_cut_limit - static_cast<int>(branch_joint_records.size()));
            auto separated = separate_branch_joint_cost(d, paths, assignment_support_cache, branch_joint_cache,
                design, theta, candidate_theta, std::min(deadline, wall_time() + options.branch_joint_seconds),
                options.double_support_cuts, options.joint_base_cost_cuts,
                static_cast<std::size_t>(allowance), options.branch_joint_floor);
            branch_joint_seconds += separated.seconds;
            branch_joint_numerical_skips += separated.numerical_skips;
            branch_joint_certificates += separated.certificates.size();
            int inserted = 0;
            for (auto& cut : separated.cuts) {
                if (!cut.valid || !cut.violated || !branch_joint_keys.insert(cut.key).second) continue;
                cut.row.name += "_" + std::to_string(branch_joint_records.size());
                branch_joint_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(cut.row);
                solver.add_row(cut.row);
                Json record = cut.metadata;
                record.update({{"candidate_id",active_candidate},{"accepted",true},{"key",cut.key},{"kind",cut.kind},
                    {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},{"integer_only",true},
                    {"blocking_arcs",cut.blocking_arcs},{"triggers",cut.triggers},{"origin_bases",cut.origin_bases}});
                branch_joint_records.push_back(record);
                if (cut.kind == "double_branch") ++double_support_count;
                else if (cut.kind == "joint_bases") ++joint_base_count;
                ++inserted;
            }
            Json record = separated.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted_cuts",inserted},
                {"seconds",separated.seconds},{"integer_only",true}});
            branch_joint_attempts.push_back(record);
            if (options.progress) {
                record["event"] = "branch_joint_separation";
                options.progress->record(record);
            }
            if (inserted > 0) {
                ++branch_joint_skipped;
                master_stalled_passes = 0;
                continue;
            }
        }
        // Check mandatory single-order support before the more expensive
        // assignment-independent topology oracle. Each accepted row uses only
        // one assignment trigger and a complete, cached single-path domain.
        if (!resumed_design && options.assignment_support_cuts && assignment_support_cache.complete_domain &&
            static_cast<int>(assignment_support_records.size()) < options.assignment_support_cut_limit && remaining(deadline) > 0) {
            const int allowance = std::min(options.assignment_support_max_cuts,
                options.assignment_support_cut_limit - static_cast<int>(assignment_support_records.size()));
            auto separated = separate_assignment_support_cost(d, assignment_support_cache, design,
                theta, candidate_theta, std::min(deadline, wall_time() + options.assignment_support_seconds),
                static_cast<std::size_t>(allowance), options.assignment_support_floor);
            assignment_support_seconds += separated.seconds;
            assignment_support_numerical_skips += separated.numerical_skips;
            int inserted = 0;
            for (auto& cut : separated.cuts) {
                if (cut.valid) ++assignment_support_certificates;
                if (!cut.valid || !cut.violated || !assignment_support_keys.insert(cut.key).second) continue;
                cut.row.name += "_" + std::to_string(assignment_support_records.size());
                assignment_support_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(cut.row);
                solver.add_row(cut.row);
                Json record = cut.metadata;
                record.update({{"candidate_id",active_candidate},{"accepted",true},{"key",cut.key},
                    {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},{"integer_only",true},
                    {"blocking_arcs",cut.blocking_arcs}});
                assignment_support_records.push_back(record);
                ++inserted;
            }
            Json record = separated.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted_cuts",inserted},
                {"seconds",separated.seconds},{"integer_only",true}});
            assignment_support_attempts.push_back(record);
            if (options.progress) {
                record["event"] = "assignment_support_separation";
                options.progress->record(record);
            }
            if (inserted > 0) {
                ++assignment_support_skipped;
                master_stalled_passes = 0;
                continue;
            }
        }
        // A structural cost certificate covers every assignment on this
        // blocked topology. Reoptimize before paying for the SOCP; the cut is
        // integer-valid and cannot enter the original continuous diagnostics.
        if (!resumed_design && options.topology_cuts &&
            static_cast<int>(topology_records.size()) < options.topology_cut_limit && remaining(deadline) > 0) {
            auto topology = separate_topology_cost(d, paths, design, theta, candidate_theta,
                std::min(deadline, wall_time() + options.topology_seconds));
            topology_seconds += topology.seconds;
            if (topology.zero_service_certified) ++topology_certificates;
            const bool inserted = topology.valid && topology.violated && topology_keys.insert(topology.key).second;
            Json record = topology.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted",inserted},{"valid",topology.valid},
                {"violated",topology.violated},{"status",topology.status},{"seconds",topology.seconds},
                {"zero_service_certified",topology.zero_service_certified},{"timed_out",topology.timed_out},
                {"shrink_complete",topology.shrink_complete},{"violation",topology.violation},
                {"certificate_stage",topology.certificate_stage},{"blocking_count",topology.blocking_count},
                {"initial_closed_count",topology.initial_closed_count},{"integer_only",true}});
            topology_attempts.push_back(record);
            if (inserted) {
                topology.row.name += "_" + std::to_string(topology_records.size());
                topology_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(topology.row);
                solver.add_row(topology.row);
                record["rhs"] = topology.row.rhs;
                record["coefficients"] = topology.row.coefficients;
                record["blocking_arcs"] = topology.blocking_arcs;
                topology_records.push_back(record);
                ++topology_skipped;
                master_stalled_passes = 0;
            }
            if (options.progress) {
                record["event"] = "topology_separation";
                options.progress->record(record);
            }
            if (inserted) continue;
        }
        const bool repeated_design = !resumed_design && !integer_designs.insert(design).second;
        if (repeated_design) ++repeated_integer_designs;
        std::string design_key;
        for (double bit : design) design_key.push_back(bit > .5 ? '1' : '0');
        pending_design.clear();
        pending_candidate.clear();
        std::vector<int> selected_p(pool_p.begin(), pool_p.end()), selected_e(pool_e.begin(), pool_e.end());
        SolverOptions sub_options = options;
        const double pareto_left = options.seconds * options.pareto_budget_fraction - pareto_seconds;
        const int cg_call = integer_cg_calls++;
        const bool try_pareto = options.pareto && cg_call >= next_pareto_call && pareto_left > .01;
        // Successful ordinary dual work gives a problem-size dependent scale.
        // Increase effort after a timed-out auxiliary rather than suppressing
        // all future attempts as though no useful Pareto cut existed.
        double previous_dual_seconds = 1.0;
        if (!iterations.empty() && !iterations.back().at("cg_trace").empty())
            previous_dual_seconds = iterations.back().at("cg_trace").back().value("dual_seconds", 1.0);
        const double pareto_allowance = std::min(options.pareto_call_seconds,
            std::max(2.0, 3 * previous_dual_seconds) * std::pow(2.0, std::min(4, pareto_budget_failures)));
        sub_options.pareto_call_seconds = std::min({pareto_allowance, std::max(0.0,pareto_left), .2 * remaining(deadline)});
        const double sub_deadline = options.restricted_mip
            ? std::min(deadline,wall_time()+std::max(1.0,.45*options.seconds)) : deadline;
        const int cuts_before = cuts;
        const double upper_before = upper;
        auto sub = subproblem(d, paths, design, sub_options, sub_deadline, selected_p, selected_e, try_pareto ? &core : nullptr);
        account(sub);
        if (try_pareto && sub.pareto.value("attempted",false)) {
            const bool timed_out = sub.pareto.value("timeout", false);
            pareto_budget_failures = timed_out ? pareto_budget_failures + 1 : 0;
            next_pareto_call = cg_call + options.pareto_frequency *
                (timed_out ? 1 : (1 << std::min(4,pareto_failures)));
        }
        const double cut_violation = sub.ordinary.valid ? value(sub.ordinary, design) - candidate_theta : 0;
        const bool all_courier_candidate = sub.ordinary.valid && !sub.solution.is_null() &&
            sub.solution.value("served", 0.0) <= 1e-7;
        if (all_courier_candidate) ++all_courier_candidates;
        bound_stalled_passes = lower > lower_before_iteration + 1e-7*(1+std::abs(lower))
            ? 0 : bound_stalled_passes + 1;
        if (sub.ordinary.valid) {
            const double tolerance = 2 * options.cg_duality_tolerance * (1 + std::abs(candidate_theta));
            const bool progress = capacity_added || resource_added || cut_violation > tolerance ||
                                  upper < upper_before - tolerance;
            master_stalled_passes = progress ? 0 : master_stalled_passes + 1;
        }
        iterations.push_back({{"iteration", iteration + 1}, {"objective", upper}, {"bound", lower},
                              {"elapsed_seconds",options.progress?options.progress->elapsed():wall_time()-started},
                              {"candidate_id",active_candidate},{"cg_call",cg_call+1},{"resumed_design",resumed_design},
                              {"capacity_cuts_added",capacity_added ? 1 : 0},{"benders_cuts_added",cuts-cuts_before},
                              {"resource_lp_cuts_added",resource_added ? 1 : 0},
                              {"master_theta",candidate_theta},{"ordinary_cut_violation",cut_violation},
                              {"design_key",design_key},{"repeated_design",repeated_design},
                              {"all_courier_candidate",all_courier_candidate},
                              {"operating_primal_upper",sub.solution.is_null()?Json(nullptr):sub.solution.at("continuous_cost")},
                              {"operating_certified_lower",sub.ordinary.valid?Json(value(sub.ordinary,design)):Json(nullptr)},
                              {"bound_stalled_passes",bound_stalled_passes},
                              {"master_stalled_passes",master_stalled_passes},
                              {"gap", gap(upper, lower)}, {"pricing_complete", sub.ordinary.valid},
                              {"cg_iterations", sub.iterations}, {"service_columns", sub.service.size()},
                              {"empty_columns", sub.empty.size()}, {"certified_cuts", cuts},
                              {"pareto", sub.pareto}, {"timings", sub.timings}, {"wall", sub.wall},
                              {"termination_reason", sub.reason}, {"cg_trace", sub.trace}});
        if (options.restricted_mip && (cg_call == 0 || (cg_call + 1) % options.heuristic_frequency == 0 || !sub.ordinary.valid))
            restricted_search("generated_column_mip", -1);
        if (options.restricted_mip && options.local_branching && (cg_call + 1) % (2 * options.heuristic_frequency) == 0)
            restricted_search("local_branching", options.local_branching_radius);
        if (!sub.ordinary.valid) {
            // A budgeted CG call may stop after adding useful columns. Resume
            // the unchanged design on the next visit using the enlarged pool;
            // do not issue a cut or claim that this subproblem converged.
            if (options.reuse_columns && remaining(deadline)>0 &&
                (sub.service.size()>selected_p.size() || sub.empty.size()>selected_e.size())) {
                pending_design=design; pending_candidate=active_candidate; pending_theta=candidate_theta; continue;
            }
            reason = sub.reason; break;
        }
        for (std::size_t j = 0; j < core.size(); ++j) core[j] = .50 * core[j] + .49 * design[j] + .01 * original_core[j];
        if (gap(upper, lower) <= options.gap) { reason = "gap_limit"; break; }
        // A shifted floating-point certificate can leave a small positive
        // gap even after an exact master returns the same fully priced design.
        // Report the unresolved numerical gap instead of duplicating cuts or
        // pretending that a requested zero gap was reached.
        if (options.adaptive_master && !resumed_design && solved.optimal && master_stalled_passes >= 3 &&
            !master_records.empty() && master_records.back().at("relative_gap_target").get<double>() == 0 &&
            upper - lower <= 4 * options.cg_duality_tolerance * (1 + std::abs(upper))) {
            reason = "certificate_tolerance_limit"; break;
        }
        if ((cg_call + 1) % options.heuristic_frequency == 0) separate_neighborhood();
        if (options.primal_search && !options.restricted_mip && (cg_call + 1) % 5 == 0 && remaining(deadline) > 0
            && options.seconds*options.heuristic_budget_fraction>heuristic_seconds) {
            auto incumbent_design = solution_design(d, best);
            if (incumbent_design != last_neighbor_design) {
                last_neighbor_design = incumbent_design;
                auto neighbor = neighbor_design(d, paths, best);
                if (!neighbor.empty()) {
                    ++primal_search_attempts;
                    active_candidate = "primal_search:" + std::to_string(primal_search_attempts);
                    std::vector<int> reused_p(pool_p.begin(), pool_p.end()), reused_e(pool_e.begin(), pool_e.end());
                    double search_deadline = std::min(deadline, wall_time() + std::min({10.0, 0.1 * remaining(deadline),
                        options.seconds*options.heuristic_budget_fraction-heuristic_seconds}));
                    auto proposal = subproblem(d, paths, neighbor, options, search_deadline, reused_p, reused_e, nullptr);
                    heuristic_seconds += proposal.wall;
                    account(proposal, true, "primal_search");
                    heuristic_records.push_back({{"stage", "primal_search"}, {"termination_reason", proposal.reason},
                                                 {"candidate_id",active_candidate},
                                                 {"pricing_complete", proposal.ordinary.valid}, {"wall", proposal.wall},
                                                 {"cg_iterations", proposal.iterations}, {"cg_trace", proposal.trace}});
                }
            }
        }
    }
    } catch (const RunBudgetExceeded& error) {
        // A late incremental model build must not erase the certified bounds,
        // cuts and diagnostics already obtained by the decomposition.
        reason = "time_limit";
        pricing_complete = false;
        budget_interruption = {{"stage", error.stage}, {"sizes", error.sizes},
            {"candidate_id", active_candidate}, {"error", error.what()}};
        last_subproblem_termination = "model_construction_time_limit";
        if (root_master_solves > 0 && integer_master_solves == 0 && root_reason == "round_limit")
            root_reason = "model_construction_time_limit";
        if (options.progress) options.progress->record({{"event", "benders_budget_interruption"},
            {"details", budget_interruption}, {"bound", lower}, {"objective", upper}});
    }
    synchronize_result_paths(d, paths, best);
    best["bound"] = lower;
    best["gap"] = gap(upper, lower);
    best["termination_reason"] = reason;
    best["budget_interruption"] = budget_interruption;
    best["pricing_complete"] = pricing_complete;
    best["pricing_complete_scope"] = "last_processed_subproblem";
    best["last_subproblem_stage"] = last_subproblem_stage;
    best["last_subproblem_termination_reason"] = last_subproblem_termination;
    best["catalogue_complete"] = paths.complete;
    best["support_complete"] = paths.support_complete;
    best["construction_complete"] = paths.support_complete;
    best["initial_columns"] = seeds.metadata;
    best["initial_service_columns"] = seeds.service.size();
    best["initial_empty_columns"] = seeds.empty.size();
    best["pricing_method"] = options.column_generation ? "exact_resource_labeling" : "all_columns";
    best["cg_max_columns"] = options.cg_max_columns;
    best["pricing_tolerance"] = options.pricing_tolerance;
    best["cg_duality_tolerance"] = options.cg_duality_tolerance;
    best["phase_one"] = "unnecessary_validated_all_courier_complete_recourse";
    best["certified_cuts"] = cuts;
    best["benders_cuts"] = cut_records;
    best["iterations"] = iterations;
    best["master_solves"] = master_records;
    best["continuous_root"] = continuous_root;
    best["root_diagnostics"] = {{"termination_reason", root_reason},
        {"scope", integer_strengthened_root ? "integer_valid_strengthened_root" : continuous_scope(options)},
        {"lower_bound", root_lower},
        {"continuous_upper_bound", integer_strengthened_root ? Json(nullptr) : Json(root_upper)},
        {"gap", integer_strengthened_root ? Json(nullptr) : Json(gap(root_upper, root_lower))},
        {"converged", !integer_strengthened_root && root_upper-root_lower <= 2*options.cg_duality_tolerance*(1+std::abs(root_upper))},
        {"original_continuous_value_closed",continuous_root_closed},
        {"integer_feasible_upper_bound",upper},
        {"gap_to_integer_incumbent",gap(upper,root_lower)},
        {"projection_enabled",options.root_projection},{"inout_enabled",options.root_projection && options.root_inout},
        {"candidate_epigraph_certified",root_reason=="candidate_epigraph_certified"},
        {"original_candidate_recourse_consistent",root_reason=="candidate_epigraph_certified" ||
            root_reason=="original_candidate_recourse_consistent"},
        {"rounds", root_records}};
    best["topology_cut_records"] = topology_records;
    best["topology_attempts"] = topology_attempts;
    best["topology_summary"] = {{"enabled",options.topology_cuts},{"cuts",topology_records.size()},
        {"attempts",topology_attempts.size()},{"skipped_integer_subproblems",topology_skipped},
        {"zero_service_certificates",topology_certificates},{"seconds",topology_seconds},
        {"master_added_variables",0},{"validity_scope","binary_designs_all_assignments"}};
    best["assignment_support_preprocessing"] = assignment_support_cache.metadata;
    best["assignment_support_cut_records"] = assignment_support_records;
    best["assignment_support_attempts"] = assignment_support_attempts;
    best["assignment_support_summary"] = {{"enabled",options.assignment_support_cuts},
        {"cuts",assignment_support_records.size()},{"attempts",assignment_support_attempts.size()},
        {"skipped_integer_subproblems",assignment_support_skipped},{"certificates",assignment_support_certificates},
        {"numerical_skips",assignment_support_numerical_skips},{"seconds",assignment_support_seconds},
        {"preprocessing_seconds",assignment_support_cache.seconds},{"master_added_variables",0},
        {"validity_scope","binary_designs_single_assignment_trigger"}};
    best["resource_lp_cuts"] = resource_records;
    best["branch_joint_preprocessing"] = branch_joint_cache.metadata;
    best["branch_joint_cut_records"] = branch_joint_records;
    best["branch_joint_attempts"] = branch_joint_attempts;
    best["branch_joint_summary"] = branch_joint_summary();
    best["branch_joint_summary"]["completed_double_groups"] = branch_joint_cache.completed_groups;
    best["branch_joint_summary"]["double_candidates_checked"] = branch_joint_cache.candidates_checked;
    best["resource_lp_attempts"] = resource_attempts;
    best["resource_lp_metadata"] = resource_relaxation ? resource_relaxation->metadata() : Json{{"enabled", false}};
    best["od_airline_summary"] = {{"enabled",integer_strengthened_root},
        {"resource_cuts",resource_records.size()},
        {"cuts_with_active_od_prices",std::count_if(resource_records.begin(),resource_records.end(),
            [](const Json& record){return record.value("active_od_airline_prices",0)>0;})},
        {"master_added_variables",0},
        {"validity_scope","integer_design_convex_hull"}};
    best["acceleration_records"] = heuristic_records;
    best["master_bound_metadata"] = bound_metadata;
    best["capacity_cut_metadata"] = capacity_enabled
        ? capacity_network.metadata : Json{{"enabled",false},{"reason","disabled"}};
    best["capacity_cuts"] = capacity_records;
    best["cut_schedule"] = {{"policy","same_candidate_capacity_and_cg"},
        {"root_round_unit","root_master_and_cg"},{"frequency_unit","integer_cg_calls"}};
    best["master_structure"] = {{"form", "design_binaries_plus_operating_cost"},
        {"variables", master.matrix.variables.size()}, {"design_binaries", master.design_vars.size()},
        {"operating_cost_variables", 1}, {"operational_auxiliary_variables", 0},
        {"initial_rows", initial_master_rows}, {"final_rows", master.matrix.rows.size()},
        {"cones", master.matrix.cones.size()}};
    best["master_model_size"]={{"variables",master.matrix.variables.size()},
        {"linear_rows",master.matrix.rows.size()},{"cones",master.matrix.cones.size()}};
    best["subproblem_model_size"]=subproblem_size;
    best["subproblem_model_size_peak"]=subproblem_peak;
    best["subproblem_model_size_scope"]="integer_recourse_including_heuristic_designs_excluding_fractional_root";
    best["demand_formulation"]=options.conditional_demand?"conditional":"lifted";
    best["structural_strengthening"]=structural_metadata(options);
    best["tight_airline_bounds"]=options.tight_airline_bounds;
    best["demand_formulation_scope"]={{"integer_recourse",options.conditional_demand?"conditional":"lifted"},
        {"fractional_recourse","lifted"},{"continuous_root","lifted"},{"restricted_mip","lifted"},
        {"pareto_core",options.conditional_demand?"fixed_assignment_face":"global_design_domain"}};
    best["timings"] = timing;
    best["timings"]["topology_separation"] = topology_seconds;
    best["timings"]["assignment_support_separation"] = assignment_support_seconds;
    best["timings"]["branch_joint_separation"] = branch_joint_seconds;
    best["timings"]["restricted_mip"] = restricted_seconds;
    best["timings"]["neighborhood_cg"] = neighborhood_seconds;
    best["heuristic_budget_used"] = heuristic_seconds;
    best["strengthening_parameters"] = {{"root_cut_rounds",options.root_cut_rounds},
        {"root_budget_fraction",options.root_budget_fraction},{"heuristic_budget_fraction",options.heuristic_budget_fraction},
        {"pareto_frequency",options.pareto_frequency},{"pareto_budget_fraction",options.pareto_budget_fraction},
        {"pareto_call_seconds",options.pareto_call_seconds},{"local_branching_radius",options.local_branching_radius},
        {"capacity_cut_limit",options.capacity_cut_limit},{"topology_cut_limit",options.topology_cut_limit},
        {"topology_seconds",options.topology_seconds}};
    best["strengthening_parameters"].update({{"assignment_support_seconds",options.assignment_support_seconds},
        {"assignment_support_cut_limit",options.assignment_support_cut_limit},
        {"assignment_support_max_cuts",options.assignment_support_max_cuts},
        {"assignment_support_floor",options.assignment_support_floor}});
    best["strengthening_parameters"].update({{"branch_joint_seconds",options.branch_joint_seconds},
        {"branch_joint_preprocessing_seconds",options.branch_joint_preprocessing_seconds},
        {"branch_joint_cut_limit",options.branch_joint_cut_limit},{"branch_joint_max_cuts",options.branch_joint_max_cuts},
        {"branch_joint_floor",options.branch_joint_floor}});
    best["strengthening_parameters"].update({{"master_initial_gap",options.master_initial_gap},
        {"master_slice_seconds",options.master_slice_seconds},{"resource_lp_seconds",options.resource_lp_seconds},
        {"resource_lp_cut_limit",options.resource_lp_cut_limit}});
    best["strengthening_parameters"].update({{"continuous_root_seconds",options.continuous_root_seconds},
        {"continuous_root_max_columns",options.continuous_root_max_columns},
        {"continuous_root_budget_fraction",options.continuous_root_budget_fraction},
        {"root_inout_weight",options.root_inout_weight}});
    best["strengthening_parameters"].update({{"heuristic_pool_size",options.heuristic_pool_size},
        {"neighborhood_cut_seconds",options.neighborhood_cut_seconds},
        {"neighborhood_cut_budget_fraction",options.neighborhood_cut_budget_fraction}});
    best["algorithm_wall"] = wall_time() - started;
    double accounted_wall = restricted_seconds + topology_seconds + assignment_support_seconds + branch_joint_seconds;
    for (const char* key : {"master_build", "greedy_initial_columns", "master_optimize", "master_update",
                           "subproblem_wall", "root_master", "capacity_cut_build", "capacity_cut_separation",
                           "resource_lp_build", "resource_lp_separation", "incumbent_cleanup", "continuous_root",
                           "assignment_support_preprocessing", "branch_joint_preprocessing"})
        accounted_wall += timing.value(key, 0.0);
    best["timings"]["bookkeeping_and_interrupted_work"] =
        std::max(0.0, best["algorithm_wall"].get<double>() - accounted_wall);
    best["accelerations"] = {{"pareto_cuts", options.pareto}, {"reuse_columns", options.reuse_columns},
                             {"column_generation", options.column_generation}, {"reuse_dual_lp", options.reuse_dual_lp},
                             {"courier_savings_bound", options.courier_savings_bound},
                             {"initial_design", options.initial_design}, {"warm_start", options.warm_start},
                             {"incumbent_cleanup", options.incumbent_cleanup}, {"primal_search", options.primal_search}};
    best["accelerations"].update({{"branch_covering_seeds",options.branch_covering_seeds},
        {"resource_bounds",options.courier_savings_bound && options.resource_bounds},
        {"capacity_cuts",capacity_enabled},
        {"resource_lp_cuts",resource_enabled},{"adaptive_master",options.adaptive_master},
        {"neighborhood_cuts",options.neighborhood_cuts && options.restricted_mip},
        {"continuous_root",options.continuous_root},
        {"root_projection",options.root_projection},{"root_inout",options.root_projection && options.root_inout},
        {"auxiliary_valid_cuts",options.auxiliary_valid_cuts},
        {"resource_airline_bounds",resource_enabled && options.resource_airline_bounds},
        {"od_airline_bounds",integer_strengthened_root},
        {"tight_airline_bounds",options.tight_airline_bounds},{"topology_cuts",options.topology_cuts},
        {"assignment_incident_clique",options.assignment_incident_clique},
        {"shared_rho_joint",options.shared_rho_joint},{"queue_perspective",options.queue_perspective},
        {"assignment_support_cuts",options.assignment_support_cuts},
        {"double_support_cuts",options.double_support_cuts},{"joint_base_cost_cuts",options.joint_base_cost_cuts},
        {"root_cuts",options.root_cuts},{"restricted_mip",options.restricted_mip},
        {"local_branching",options.restricted_mip && options.local_branching},{"incumbent_cutoff",options.incumbent_cutoff}});
    best["acceleration_statistics"] = {{"ordinary_cuts", ordinary_count}, {"pareto_cuts", pareto_count},
                                       {"pareto_attempts", pareto_attempts}, {"cg_iterations", cg_iterations},
                                       {"ordinary_cg_iterations", ordinary_cg_iterations}, {"heuristic_cg_iterations", heuristic_cg_iterations},
                                       {"master_starts", master_starts},
                                       {"cleanup_improvements", cleanup_improvements}, {"cleanup_savings", cleanup_savings},
                                       {"primal_search_attempts", primal_search_attempts},
                                       {"reused_service_pool", pool_p.size()}, {"reused_empty_pool", pool_e.size()}};
    best["acceleration_statistics"].update({{"root_cuts",root_count},{"restricted_mip_attempts",restricted_attempts},
        {"root_master_solves",root_master_solves},{"root_cg_calls",root_cg_calls},
        {"integer_master_solves",integer_master_solves},{"integer_cg_calls",integer_cg_calls},
        {"restricted_mip_improvements",restricted_improvements},{"pareto_consecutive_failures",pareto_failures},
        {"pareto_budget_used",pareto_seconds},{"capacity_cuts",capacity_records.size()}});
    best["acceleration_statistics"].update({{"resource_lp_cuts",resource_records.size()},
        {"resource_lp_attempts",resource_attempts.size()},{"master_stalled_passes",master_stalled_passes}});
    best["acceleration_statistics"]["continuous_bound_rows"] = continuous_bound_rows;
    best["acceleration_statistics"].update({{"auxiliary_valid_cuts",auxiliary_count},
        {"unique_integer_designs",integer_designs.size()},{"repeated_integer_designs",repeated_integer_designs},
        {"all_courier_candidates",all_courier_candidates},{"bound_stalled_passes",bound_stalled_passes}});
    best["acceleration_statistics"].update({{"neighborhood_cg_calls",neighborhood_cg_calls},
        {"neighborhood_cuts",neighborhood_cut_count},{"neighborhood_budget_used",neighborhood_seconds}});
    best["acceleration_statistics"].update({{"conditional_cuts",conditional_cuts},
        {"conditional_subproblems",conditional_subproblems},{"conditional_assignment_patterns",assignment_patterns.size()}});
    observe_global("benders_finished");
    if (options.progress) options.progress->checkpoint(best,"benders_finished",true);
    return best;
}
} // namespace mlnd
