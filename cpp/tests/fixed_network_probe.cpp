// Diagnostic only: optimize assignments with the physical network fixed.
// All bounds in this program concern that fixed network, not the free design.
#include "mlnd/formulation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
bool finite_number(double x) { return std::isfinite(x) && std::abs(x) < .1 * IloInfinity; }
Json number(double x) { return finite_number(x) ? Json(x) : Json(nullptr); }
bool eligible_mip_status(int code) {
    return code == IloCplex::Optimal || code == IloCplex::OptimalTol ||
        code == IloCplex::AbortTimeLim || code == IloCplex::AbortDetTimeLim ||
        code == IloCplex::NodeLimFeas || code == IloCplex::NodeLimInfeas ||
        code == IloCplex::SolLim || code == IloCplex::AbortUser ||
        code == CPXMIP_OPTIMAL || code == CPXMIP_TIME_LIM_FEAS ||
        code == CPXMIP_TIME_LIM_INFEAS || code == CPXMIP_ABORT_FEAS ||
        code == CPXMIP_ABORT_INFEAS;
}
struct ActiveProgress {
    explicit ActiveProgress(const std::shared_ptr<RunProgress>& p) { set_active_run_progress(p); }
    ~ActiveProgress() { set_active_run_progress(nullptr); }
};
Json integer_residuals(Formulation& f, const std::vector<double>& values) {
    std::vector<bool> previous;
    for (int j = 0; j < f.layout.count; ++j) {
        previous.push_back(f.matrix.variables[j].binary);
        f.matrix.variables[j].binary = true;
    }
    auto result = point_residuals(f.matrix, values);
    for (int j = 0; j < f.layout.count; ++j) f.matrix.variables[j].binary = previous[j];
    return result;
}
}

int main(int argc, char** argv) {
    using namespace mlnd;
    if (argc < 6 || argc > 8) {
        std::cerr << "Usage: fixed_network_probe INPUT_DIR K DESIGN_JSON integer|relaxed OUTPUT_JSON [SECONDS=600 [original|strict|strict_no_presolve]]\n";
        return 2;
    }
    const double started = wall_time();
    std::filesystem::path output;
    Json report;
    std::shared_ptr<RunProgress> progress;
    bool owns_output = false;
    try {
        const int k = std::stoi(argv[2]);
        const std::string mode = argv[4];
        const bool integer = mode == "integer";
        const double seconds = argc >= 7 ? std::stod(argv[6]) : 600;
        const std::string profile = argc == 8 ? argv[7] : "original";
        if ((k != 1 && k != 2) || (mode != "integer" && mode != "relaxed") ||
            !std::isfinite(seconds) || seconds <= 0 ||
            (profile != "original" && profile != "strict" && profile != "strict_no_presolve"))
            throw std::invalid_argument("K must be 1/2, mode integer/relaxed, seconds positive and finite, profile original/strict/strict_no_presolve");
        const auto input = std::filesystem::absolute(argv[1]);
        const auto design_file = std::filesystem::absolute(argv[3]);
        output = std::filesystem::absolute(argv[5]);
        const std::filesystem::path values_file = output.string() + ".values.json";
        const std::filesystem::path solution_file = output.string() + ".solution.json";
        const std::filesystem::path diagnostics = output.string() + ".diagnostics";
        for (const auto& p : {output, values_file, solution_file, diagnostics})
            if (std::filesystem::exists(p)) throw std::runtime_error("Refusing to overwrite: " + p.string());
        report = {{"schema", "mlnd_fixed_network_probe_v1"}, {"scope", "conditional_on_fixed_network"},
            {"mode", mode}, {"K", k}, {"input", input.string()}, {"design_file", design_file.string()},
            {"started_at", utc_timestamp()}, {"status", "started"}, {"termination", nullptr},
            {"lower_bound_certified", false}, {"upper_bound_validated", false},
            {"total_lower_bound", nullptr}, {"operating_lower_bound", nullptr},
            {"total_upper_bound", nullptr}, {"operating_upper_bound", nullptr},
            {"relative_gap", nullptr}, {"original_integer_feasible", false},
            {"optimal_certificate", false}, {"catalogue", {{"complete", false}, {"support_complete", false}}},
            {"settings", {{"seconds", seconds}, {"threads", 1}, {"seed", 43}, {"gap", 1e-6},
                {"numerical_profile", profile},
                {"conditional_demand", false}, {"tight_airline_bounds", true},
                {"assignment_incident_clique", false}, {"shared_rho_joint", false},
                {"queue_perspective", false}, {"extra_strengthening", false}}},
            {"timings", Json::object()}, {"validation", Json::object()}};
        atomic_json_file(output, report); owns_output = true;
        progress = std::make_shared<RunProgress>(diagnostics, started, seconds,
            Json{{"diagnostic", "fixed_network"}, {"mode", mode}, {"scope", "conditional_on_fixed_network"}});
        ActiveProgress active(progress);
        progress->guard("input");
        const Input d = read_input(input, k, true);
        const DesignLayout layout(d);
        Json source;
        std::ifstream design_stream(design_file);
        if (!design_stream || !(design_stream >> source)) throw std::runtime_error("Cannot read design JSON");
        if (!source.is_object() || !source.contains("design") || !source.at("design").is_array())
            throw std::invalid_argument("Design JSON must contain a full strategic vector: {\"design\":[...]}");
        const auto fixed = source.at("design").get<std::vector<double>>();
        if (fixed.size() != static_cast<std::size_t>(layout.count)) throw std::invalid_argument("Design length mismatch");
        std::vector<double> network;
        for (int j = 0; j < layout.count; ++j) {
            if (j >= layout.assignment && j < layout.airline) continue;
            if (fixed[j] != 0 && fixed[j] != 1)
                throw std::invalid_argument("Every non-assignment design entry must be exactly 0 or 1");
            network.push_back(fixed[j]);
        }
        report["instance"] = d.name;
        report["provenance"] = {{"design_sha256", sha256_file(design_file)}, {"input_manifest", d.manifest},
            {"model_version", d.model_version}, {"flow_policy", d.flow_policy}, {"route_mode", d.route_mode},
            {"demand_formulation", "complete_lifted"}, {"tight_airline_bounds", true},
            {"extra_strengthening", false}, {"source_assignment_ignored", true},
            {"fixed_network_vector", network},
            {"fixed_network_variables", "facility, capacity_level, airline"}};
        report["timings"]["input_seconds"] = wall_time() - started;
        progress->guard("path_enumeration");
        const double enum_start = wall_time();
        const auto paths = enumerate_paths(d, std::nullopt, progress->remaining(),
            [&] { return progress->remaining() <= 0; });
        report["timings"]["enumeration_seconds"] = wall_time() - enum_start;
        report["catalogue"] = {{"complete", paths.complete}, {"support_complete", paths.support_complete},
            {"scope", paths.scope}, {"service_paths", paths.service_paths.size()},
            {"empty_paths", paths.empty_paths.size()}, {"initial_segments", paths.initial_segments.size()},
            {"total_paths", paths.service_paths.size() + paths.empty_paths.size()}};
        if (!paths.complete || !paths.support_complete) throw std::runtime_error("Incomplete full path catalogue");
        progress->guard("formulation");
        const double build_start = wall_time();
        auto f = build_formulation(d, paths, nullptr, nullptr, nullptr, false, false, true, {});
        double fixed_cost = f.matrix.constant;
        for (int j = 0; j < layout.count; ++j) {
            auto& v = f.matrix.variables[j];
            if (j >= layout.assignment && j < layout.airline) {
                if (v.cost != 0) throw std::runtime_error("Unexpected assignment objective cost");
                v.binary = integer;
            } else {
                if (fixed[j] < v.lower || fixed[j] > v.upper)
                    throw std::invalid_argument("Fixed network violates original variable bounds: " + v.name);
                fixed_cost += v.cost * fixed[j];
                v.lower = v.upper = fixed[j];
                v.binary = false;
            }
        }
        std::size_t binaries = 0, nonzeros = 0;
        for (const auto& v : f.matrix.variables) binaries += v.binary;
        for (const auto& row : f.matrix.rows) nonzeros += row.coefficients.size();
        if (!integer && binaries) throw std::runtime_error("Relaxed mode still contains integer variables");
        report["fixed_cost"] = fixed_cost;
        report["model"] = {{"variables", f.matrix.variables.size()}, {"binary_variables", binaries},
            {"linear_constraints", f.matrix.rows.size()}, {"quadratic_constraints", f.matrix.cones.size()},
            {"nonzeros", nonzeros}, {"strategic_variables", layout.count},
            {"assignment_variables", layout.airline - layout.assignment}};
        report["timings"]["formulation_seconds"] = wall_time() - build_start;
        std::vector<double> selected(f.matrix.variables.size(), 0);
        for (int j = 0; j < layout.count; ++j)
            if (j < layout.assignment || j >= layout.airline) selected[j] = fixed[j];
        for (int r = 0; r < d.n_od; ++r) selected[f.courier[r]] = d.demand[r];
        for (const auto& cone : f.matrix.cones)
            selected[cone.up] = selected[cone.down] = selected[cone.one] = 1;
        const auto start_validation = point_residuals(f.matrix, selected);
        if (!start_validation.at("passed").get<bool>())
            throw std::invalid_argument("Fixed network has no validated all-courier starting point: " + start_validation.dump());
        double upper = objective_value(f.matrix, selected);
        std::string selected_source = "all_courier";
        report["warm_start"] = {{"same_all_courier_candidate", true}, {"validation", start_validation},
            {"objective", upper}, {"submitted_to_solver", false},
            {"reason", integer ? "validated_MIP_start" : "NativeSolver_does_not_submit_continuous_starts"}};
        report["upper_bound_validated"] = true;
        report["total_upper_bound"] = upper; report["operating_upper_bound"] = upper - fixed_cost;
        atomic_json_file(output, report);
        SolverOptions options;
        options.seconds = seconds; options.gap = 1e-6; options.threads = 1; options.seed = 43;
        options.tight_airline_bounds = true; options.progress = progress;
        progress->guard("solver_construction");
        const double solver_build_start = wall_time();
        NativeSolver solver(f.matrix, options, "fixed_network_" + mode);
        report["timings"]["solver_construction_seconds"] = wall_time() - solver_build_start;
        if (profile == "strict" || profile == "strict_no_presolve") {
            solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
            solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-9);
            solver.cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol, 1e-11);
            solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical, true);
        }
        if (profile == "strict_no_presolve") {
            solver.cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
            solver.cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
            solver.cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        }
        report["settings"]["actual_tolerances"] = {
            {"simplex_feasibility", solver.cplex.getParam(IloCplex::Param::Simplex::Tolerances::Feasibility)},
            {"simplex_optimality", solver.cplex.getParam(IloCplex::Param::Simplex::Tolerances::Optimality)},
            {"barrier_qcp_convergence", solver.cplex.getParam(IloCplex::Param::Barrier::QCPConvergeTol)},
            {"mip_integrality", solver.cplex.getParam(IloCplex::Param::MIP::Tolerances::Integrality)},
            {"numerical_emphasis", static_cast<bool>(solver.cplex.getParam(IloCplex::Param::Emphasis::Numerical))}};
        report["settings"]["actual_preprocessing"] = {
            {"presolve", static_cast<bool>(solver.cplex.getParam(IloCplex::Param::Preprocessing::Presolve))},
            {"aggregator", solver.cplex.getParam(IloCplex::Param::Preprocessing::Aggregator)},
            {"reduce", solver.cplex.getParam(IloCplex::Param::Preprocessing::Reduce)}};
        solver.add_start(selected);
        report["warm_start"]["submitted_to_solver"] = integer;
        progress->guard("solve");
        const auto outcome = solver.optimize(progress->remaining());
        report["solver"] = {{"status", outcome.status}, {"cplex_status", outcome.cplex_status},
            {"has_solution", outcome.has_solution}, {"optimal", outcome.optimal},
            {"candidate_available", outcome.candidate_available}, {"nodes", outcome.nodes},
            {"objective", outcome.candidate_available ? number(outcome.objective) : Json(nullptr)},
            {"raw_bound", (integer || outcome.optimal) ? number(outcome.bound) : Json(nullptr)},
            {"candidate_extraction_error", outcome.candidate_extraction_error}};
        report["timings"]["solver_seconds"] = outcome.seconds;
        bool primal_valid = false;
        if (outcome.candidate_available) {
            const auto residuals = point_residuals(f.matrix, outcome.values);
            const double recomputed = objective_value(f.matrix, outcome.values);
            const bool matching = finite_number(recomputed) && finite_number(outcome.objective) &&
                std::abs(recomputed - outcome.objective) <= 2e-7 * (1 + std::abs(recomputed));
            primal_valid = residuals.at("passed").get<bool>() && matching;
            report["validation"]["solver_candidate"] = {{"residuals", residuals},
                {"recomputed_objective", number(recomputed)}, {"objective_matches", matching}, {"passed", primal_valid}};
            if (primal_valid && recomputed < upper) {
                selected = outcome.values; upper = recomputed; selected_source = "solver";
            }
        }
        report["validation"]["selected_candidate"] = point_residuals(f.matrix, selected);
        report["validation"]["original_integrality"] = integer_residuals(f, selected);
        report["original_integer_feasible"] = integer && report["validation"]["original_integrality"]["passed"].get<bool>();
        report["selected_candidate_source"] = selected_source;
        report["total_upper_bound"] = upper; report["operating_upper_bound"] = upper - fixed_cost;
        const bool status_valid = integer ? eligible_mip_status(outcome.cplex_status) :
            (outcome.optimal && outcome.cplex_status == IloCplex::Optimal && primal_valid);
        bool bound_valid = status_valid && finite_number(outcome.bound) && (!outcome.has_solution || primal_valid);
        std::string rejection = !status_valid ? "solver_status_not_eligible_for_certificate" :
            !finite_number(outcome.bound) ? "no_finite_global_bound" :
            outcome.has_solution && !primal_valid ? "solver_primal_failed_validation" : "";
        const double margin = 1e-7 * (1 + std::abs(outcome.bound));
        const double lower = outcome.bound - margin;
        if (bound_valid && (!finite_number(lower) || lower > upper)) {
            bound_valid = false; rejection = "bound_exceeds_validated_primal_or_nonfinite";
        }
        report["lower_bound_certified"] = bound_valid;
        report["bound_certificate"] = {{"valid", bound_valid}, {"rejection", rejection},
            {"downward_margin", number(margin)}, {"scope", "conditional_on_fixed_network"},
            {"type", "CPLEX_status_and_original_matrix_numerical_validation"}};
        if (bound_valid) {
            report["total_lower_bound"] = lower; report["operating_lower_bound"] = lower - fixed_cost;
            report["relative_gap"] = (upper - lower) / std::max(1.0, std::abs(upper));
        }
        report["optimal_certificate"] = bound_valid && outcome.optimal && primal_valid &&
            lower <= upper && (upper - lower) <= 2e-6 * std::max(1.0, std::abs(upper));
        std::vector<std::string> names;
        for (const auto& variable : f.matrix.variables) names.push_back(variable.name);
        atomic_json_file(values_file, {{"scope", "conditional_on_fixed_network"}, {"mode", mode},
            {"variable_names", names}, {"raw_solver_values", outcome.values}, {"selected_values", selected},
            {"selected_source", selected_source}, {"source_design", fixed}});
        report["values_file"] = values_file.string();
        if (report["original_integer_feasible"].get<bool>()) {
            const auto solution = solution_json(d, paths, f, selected);
            const double reconstructed = solution.at("objective").get<double>();
            if (std::abs(reconstructed - upper) > 2e-6 * (1 + std::abs(upper)))
                throw std::runtime_error("Solution package objective differs from validated full matrix objective");
            atomic_json_file(solution_file, solution);
            report["solution_file"] = solution_file.string();
        }
        report["status"] = "completed"; report["termination"] = outcome.status;
        report["timings"]["total_seconds"] = wall_time() - started;
        report["timings"]["budget_exceeded_during_reporting"] = progress->remaining() <= 0;
        atomic_json_file(output, report);
        std::cout << output.string() << '\n';
        return 0;
    } catch (const EnumerationStopped& error) {
        report["status"] = "construction_stopped"; report["termination"] = error.reason;
        report["catalogue"]["partial_columns"] = error.columns;
        report["catalogue"]["partial_initial_segments"] = error.initial_segments;
    } catch (const RunBudgetExceeded& error) {
        report["status"] = "construction_time_limit"; report["termination"] = error.stage;
        report["stopped_model_sizes"] = error.sizes;
    } catch (const IloException& error) {
        report["status"] = "error"; report["termination"] = error.getMessage();
    } catch (const std::exception& error) {
        report["status"] = "error"; report["termination"] = error.what();
    }
    set_active_run_progress(nullptr);
    report["timings"]["total_seconds"] = wall_time() - started;
    if (owns_output) atomic_json_file(output, report);
    std::cerr << report.dump(2) << '\n';
    return report.value("status", std::string()) == "error" ? 1 : 0;
}
