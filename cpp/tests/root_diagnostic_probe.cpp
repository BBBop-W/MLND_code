#include "mlnd/benders.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

// Compare projections of the SAME continuous formulation. This is a root
// diagnostic, not an integer optimization benchmark or an integer UB report.
// Usage: mlnd_root_diagnostic_probe INPUT K SECONDS_PER_METHOD OUTPUT_JSON
//                                  [--root-rounds N]
namespace {
using namespace mlnd;
Json sizes(const PathCollection& paths) {
    return {{"initial_segments", paths.initial_segments.size()},
        {"service_paths", paths.service_paths.size()}, {"empty_paths", paths.empty_paths.size()},
        {"support_complete", paths.support_complete}, {"catalogue_complete", paths.complete},
        {"route_scope", paths.scope}};
}
struct ActiveObservation {
    explicit ActiveObservation(const std::shared_ptr<RunProgress>& observer) {
        set_active_run_progress(observer);
    }
    ~ActiveObservation() { set_active_run_progress(nullptr); }
};
std::shared_ptr<RunProgress> observer(const std::filesystem::path& directory,
                                    double start, double seconds, const char* method) {
    return std::make_shared<RunProgress>(directory, start, seconds,
        Json{{"method", method}, {"diagnostic_scope", "full_path_continuous_relaxation"},
             {"problem_scope", "global"}, {"integer_optimization_requested", false}});
}
void failure(Json& result, const RunBudgetExceeded& error) {
    result["status"] = "construction_time_limit";
    result["failure_stage"] = error.stage;
    result["failure_sizes"] = error.sizes;
    result["error"] = error.what();
}
void failure(Json& result, const EnumerationStopped& error) {
    result["status"] = error.reason;
    result["failure_stage"] = "path_preprocessing";
    result["partial_path_sizes"] = {{"columns", error.columns}, {"initial_segments", error.initial_segments}};
    result["error"] = error.what();
}
Json recover_root_records(const std::filesystem::path& directory) {
    std::ifstream stream(directory / "progress.jsonl");
    Json records = Json::array();
    std::string line;
    while (std::getline(stream, line)) {
        auto event = Json::parse(line, nullptr, false);
        if (event.is_discarded() || !event.is_object() ||
            event.value("event", std::string()) != "root_separation_record" ||
            event.value("bound_scope", std::string()) != "full_path_continuous_relaxation") continue;
        if (!event.contains("lower_bound") || !event.at("lower_bound").is_number() ||
            !event.contains("continuous_upper_bound") || !event.at("continuous_upper_bound").is_number()) continue;
        records.push_back(std::move(event));
    }
    if (records.empty()) return nullptr;
    const auto& last = records.back();
    const double lower = last.at("lower_bound"), upper = last.at("continuous_upper_bound");
    if (!std::isfinite(lower) || !std::isfinite(upper)) return nullptr;
    return {{"termination_reason", "interrupted_after_recorded_root_round"},
        {"scope", "full_path_continuous_relaxation"}, {"lower_bound", lower},
        {"continuous_upper_bound", upper},
        {"gap", std::max(0.0, upper - lower) / std::max(1.0, std::abs(upper))},
        {"converged", false}, {"recovered_from_progress", true},
        {"source", (directory / "progress.jsonl").string()},
        {"recorded_round_count", records.size()}, {"rounds", std::move(records)}};
}
Json full_relaxation(const Input& data, double seconds, const std::filesystem::path& directory) {
    const double started = wall_time();
    auto progress = observer(directory, started, seconds, "full_path_continuous");
    ActiveObservation active(progress);
    Json result = {{"budget_seconds", seconds}, {"scope", "full_path_continuous_relaxation"},
        {"budget_scope", "path_enumeration_model_construction_optimization_validation"},
        {"optimal_and_validated", false}, {"objective", nullptr}, {"validated_continuous_lower_bound", nullptr},
        {"preprocessing_seconds", 0.0}, {"model_build_seconds", 0.0},
        {"solver_build_seconds", 0.0}, {"optimize_seconds", 0.0}, {"validation_seconds", 0.0}};
    std::string stage = "path_enumeration";
    double phase = wall_time();
    try {
        auto paths = enumerate_paths(data, std::nullopt, progress->remaining());
        result["preprocessing_seconds"] = wall_time() - phase;
        result["paths"] = sizes(paths);
        if (!paths.complete || !paths.support_complete)
            throw std::runtime_error("Full relaxation requires a complete physical path universe");
        stage = "model_construction"; phase = wall_time();
        auto model = build_formulation(data, paths);
        for (auto& variable : model.matrix.variables) variable.binary = false;
        result["model_build_seconds"] = wall_time() - phase;
        result["model_size"] = {{"variables", model.matrix.variables.size()},
            {"linear_rows", model.matrix.rows.size()}, {"cones", model.matrix.cones.size()}};
        SolverOptions options;
        options.seconds = progress->remaining(); options.threads = 1; options.gap = 0;
        options.progress = progress;
        result["numerical_attempts"] = Json::array();
        for (int profile : {0, 1, 2, 3}) {
            if (progress->remaining() <= 0) { result["status"] = "time_limit"; break; }
            const double qcp_tolerance = profile == 0 ? 1e-9 : profile == 1 ? 1e-10 : profile == 2 ? 1e-11 : 1e-12;
            Json attempt = {{"numerical_profile", profile}, {"same_formulation", true},
                {"qcp_convergence_tolerance", qcp_tolerance}, {"optimal_and_validated", false},
                {"retry_reason", profile == 0 ? "initial_solve" : "no_validated_optimal_point"}};
            stage = "solver_construction"; phase = wall_time();
            NativeSolver solver(model.matrix, options, "full_path_continuous");
            if (profile > 0) {
                solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical, true);
                solver.cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol, qcp_tolerance);
                solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
                solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-9);
                solver.cplex.setParam(IloCplex::Param::Read::Scale, profile == 1 ? 1 : profile == 2 ? -1 : 0);
                if (profile == 3) solver.cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
            }
            attempt["solver_build_seconds"] = wall_time() - phase;
            result["solver_build_seconds"] = result.at("solver_build_seconds").get<double>()
                + attempt.at("solver_build_seconds").get<double>();
            stage = "optimization"; phase = wall_time();
            const auto solved = solver.optimize(profile == 0 ? progress->remaining() : std::min(60.0, progress->remaining()));
            attempt["optimize_seconds"] = wall_time() - phase;
            result["optimize_seconds"] = result.at("optimize_seconds").get<double>()
                + attempt.at("optimize_seconds").get<double>();
            result["status"] = attempt["status"] = solved.status;
            result["solver_status_code"] = attempt["solver_status_code"] = solved.cplex_status;
            result["solver_optimal"] = attempt["solver_optimal"] = solved.optimal;
            result["has_solution"] = attempt["has_solution"] = solved.has_solution;
            result["objective"] = attempt["objective"] = solved.has_solution ? Json(solved.objective) : Json(nullptr);
            stage = "validation"; phase = wall_time();
            if (solved.has_solution) {
                const auto residuals = point_residuals(model.matrix, solved.values);
                const double recomputed = objective_value(model.matrix, solved.values);
                const double tolerance = 2e-7 * (1 + std::abs(recomputed));
                const bool matched = std::abs(recomputed - solved.objective) <= tolerance;
                const bool certified = solved.optimal && residuals.at("passed").get<bool>() && matched;
                result["residuals"] = attempt["residuals"] = residuals;
                result["recomputed_objective"] = attempt["recomputed_objective"] = recomputed;
                result["objective_matches"] = attempt["objective_matches"] = matched;
                result["optimal_and_validated"] = attempt["optimal_and_validated"] = certified;
                if (certified) result["validated_continuous_lower_bound"] = recomputed;
            }
            attempt["validation_seconds"] = wall_time() - phase;
            result["validation_seconds"] = result.at("validation_seconds").get<double>()
                + attempt.at("validation_seconds").get<double>();
            result["numerical_attempts"].push_back(attempt);
            attempt["event"] = "full_relaxation_attempt";
            attempt["bound_scope"] = "full_path_continuous_relaxation";
            progress->record(std::move(attempt));
            if (result.at("optimal_and_validated").get<bool>()) break;
        }
    } catch (const RunBudgetExceeded& error) { failure(result, error); }
      catch (const EnumerationStopped& error) { failure(result, error); }
      catch (const IloException& error) { result["status"] = "solver_exception"; result["error"] = error.getMessage(); }
      catch (const std::exception& error) { result["status"] = "exception"; result["error"] = error.what(); }
    if (result.contains("error")) {
        result["interrupted_stage"] = stage;
        result["interrupted_stage_seconds"] = wall_time() - phase;
    }
    result["wall_seconds"] = wall_time() - started;
    result["budget_overrun_seconds"] = std::max(0.0, result.at("wall_seconds").get<double>() - seconds);
    result["monitoring"] = progress->summary();
    return result;
}

Json root_relaxation(const Input& data, double seconds, int rounds, const std::filesystem::path& directory) {
    const double started = wall_time();
    auto progress = observer(directory, started, seconds, "benders_root_continuous");
    ActiveObservation active(progress);
    Json result = {{"budget_seconds", seconds}, {"scope", "full_path_continuous_relaxation"},
        {"budget_scope", "path_support_model_construction_root_master_and_cg"},
        {"preprocessing_seconds", 0.0}, {"root_algorithm_seconds", 0.0},
        {"integer_master_iterations_requested", 0}, {"root_round_limit", rounds},
        {"root_diagnostics", nullptr}};
    const double preprocessing_started = wall_time();
    double algorithm_started = 0;
    try {
        auto paths = prepare_path_support(data, progress->remaining());
        result["preprocessing_seconds"] = wall_time() - preprocessing_started;
        result["initial_paths"] = sizes(paths);
        if (!paths.support_complete) throw std::runtime_error("Root CG needs complete physical support");
        SolverOptions options;
        options.seconds = progress->remaining(); options.threads = 1; options.gap = 0;
        options.max_iterations = 0; options.root_cut_rounds = rounds; options.root_budget_fraction = 1;
        options.initial_design = options.restricted_mip = options.neighborhood_cuts = false;
        options.pareto = options.primal_search = options.local_branching = false;
        options.warm_start = options.incumbent_cleanup = options.incumbent_cutoff = false;
        options.resource_bounds = options.courier_savings_bound = options.resource_lp_cuts = false;
        options.capacity_cuts = false;
        options.continuous_root = false; // Measure the original Benders projection, not the direct CG floor.
        options.root_cuts = options.column_generation = options.reuse_columns = true;
        options.progress = progress;
        result["strengthening"] = {{"resource_bounds", false}, {"static_cost_bounds", false},
            {"resource_lp_cuts", false}, {"maxflow_capacity_cuts", false},
            {"pareto", false}, {"incumbent_cutoff", false}, {"neighborhood_cuts", false},
            {"continuous_root", false},
            {"root_ordinary_cg_cuts", true}, {"reuse_columns", true}};
        result["cg_duality_tolerance"] = options.cg_duality_tolerance;
        result["pricing_tolerance"] = options.pricing_tolerance;
        algorithm_started = wall_time();
        const auto solved = solve_benders(data, paths, options);
        result["root_algorithm_seconds"] = wall_time() - algorithm_started;
        result["root_diagnostics"] = solved.at("root_diagnostics");
        result["status"] = solved.at("root_diagnostics").at("termination_reason");
        result["timings"] = solved.at("timings");
        result["acceleration_statistics"] = solved.at("acceleration_statistics");
        result["final_paths"] = sizes(paths);
        result["certified_cuts"] = solved.at("certified_cuts");
        const int integer_calls = solved.at("acceleration_statistics").at("integer_master_solves");
        result["no_integer_master_solved"] = integer_calls == 0;
        if (integer_calls != 0) throw std::runtime_error("Root diagnostic unexpectedly solved an integer master");
        // Do not copy solved.objective/gap: those concern the retained integer
        // all-courier incumbent, not an upper bound of comparable tightness.
    } catch (const RunBudgetExceeded& error) { failure(result, error); }
      catch (const EnumerationStopped& error) { failure(result, error); }
      catch (const IloException& error) { result["status"] = "solver_exception"; result["error"] = error.getMessage(); }
      catch (const std::exception& error) { result["status"] = "exception"; result["error"] = error.what(); }
    if (algorithm_started > 0 && result.contains("error")) result["root_algorithm_seconds"] = wall_time() - algorithm_started;
    if (algorithm_started == 0 && result.contains("error")) result["preprocessing_seconds"] = wall_time() - preprocessing_started;
    if (result.at("root_diagnostics").is_null()) {
        const double recovery_started = wall_time();
        result["root_diagnostics"] = recover_root_records(directory);
        result["diagnostic_recovery_seconds"] = wall_time() - recovery_started;
        result["partial_root_records_recovered"] = result.at("root_diagnostics").is_object();
    }
    result["wall_seconds"] = wall_time() - started;
    result["budget_overrun_seconds"] = std::max(0.0, result.at("wall_seconds").get<double>() - seconds);
    result["monitoring"] = progress->summary();
    return result;
}
}

int main(int argc, char** argv) {
    using namespace mlnd;
    try {
        if (argc != 5 && argc != 7)
            throw std::invalid_argument("Usage: mlnd_root_diagnostic_probe INPUT K SECONDS_PER_METHOD OUTPUT_JSON [--root-rounds N]");
        const auto input = std::filesystem::absolute(argv[1]);
        const int k = std::stoi(argv[2]);
        const double seconds = std::stod(argv[3]);
        const auto output = std::filesystem::absolute(argv[4]);
        int rounds = 100;
        if (argc == 7) {
            if (std::string(argv[5]) != "--root-rounds") throw std::invalid_argument("Expected --root-rounds N");
            rounds = std::stoi(argv[6]);
        }
        if ((k != 1 && k != 2) || !std::isfinite(seconds) || seconds <= 0 || rounds < 1)
            throw std::invalid_argument("Require K in {1,2}, positive finite seconds, and positive root rounds");
        auto auxiliary = output;
        auxiliary += ".diagnostics";
        if (std::filesystem::exists(output) || std::filesystem::exists(auxiliary))
            throw std::invalid_argument("Refusing to overwrite an existing output or diagnostic directory");
        const double started = wall_time(), read_started = wall_time();
        const auto data = read_input(input, k, true);
        Json result = {{"schema", "mlnd-root-relaxation-diagnostic/1"},
            {"created_utc", utc_timestamp()}, {"input", input.string()}, {"instance", data.name},
            {"K", k}, {"n_l", data.n_l}, {"n_h", data.n_h}, {"n_od", data.n_od},
            {"model_version", data.model_version}, {"flow_policy", data.flow_policy}, {"route_mode", data.route_mode},
            {"input_manifest", data.manifest}, {"input_model_sha256", sha256_file(input / "model.json")},
            {"input_read_seconds", wall_time() - read_started},
            {"scope", "same_full_path_continuous_relaxation"}, {"threads", 1},
            {"seconds_per_method", seconds}, {"total_method_budget_seconds", 2 * seconds},
            {"shared_input_read_outside_method_budgets", true}, {"output_finalization_outside_method_budgets", true},
            {"integer_upper_bounds_are_not_compared", true}};
        result["full_continuous"] = full_relaxation(data, seconds, auxiliary / "full");
        atomic_json_file(output, result); // Preserve the completed reference if root work is interrupted.
        result["benders_root"] = root_relaxation(data, seconds, rounds, auxiliary / "root");
        Json comparison = {{"available", false}, {"equivalence_established", false},
            {"reason", "full_continuous_optimum_or_root_diagnostics_unavailable"}};
        const auto& full = result.at("full_continuous");
        const auto& root = result.at("benders_root").at("root_diagnostics");
        bool valid = !full.value("has_solution", false) ||
            (full.value("objective_matches", false) && full.contains("residuals") &&
             full.at("residuals").value("passed", false));
        if (!valid) comparison["reason"] = "full_continuous_numerical_validation_failed";
        if (full.at("optimal_and_validated").get<bool>() && root.is_object()) {
            const double reference = full.at("validated_continuous_lower_bound");
            const double lower = root.at("lower_bound"), upper = root.at("continuous_upper_bound");
            const double tolerance = 5e-6 * (1 + std::abs(reference));
            const bool bracketed = lower <= reference + tolerance && upper >= reference - tolerance;
            const bool converged = root.at("converged");
            valid = bracketed && (!converged || std::max(std::abs(reference - lower), std::abs(upper - reference)) <= tolerance);
            comparison = {{"available", true}, {"reference_objective", reference},
                {"benders_root_lower_bound", lower}, {"benders_root_continuous_upper_bound", upper},
                {"reference_minus_root_lower", reference - lower}, {"root_upper_minus_reference", upper - reference},
                {"comparison_tolerance", tolerance}, {"reference_bracketed", bracketed},
                {"root_converged", converged}, {"numerical_consistency_passed", valid},
                {"equivalence_established", converged && valid},
                {"reason", valid ? (converged ? "matching_continuous_optimum" : "root_separation_incomplete") : "bound_inconsistency"}};
        }
        result["comparison"] = comparison;
        result["total_wall_seconds"] = wall_time() - started;
        atomic_json_file(output, result);
        std::cout << Json{{"output", output.string()}, {"full_status", full.at("status")},
            {"root_status", result.at("benders_root").at("status")}, {"comparison", comparison}}.dump(2) << '\n';
        return valid ? 0 : 2;
    } catch (const std::exception& error) {
        set_active_run_progress(nullptr);
        std::cerr << error.what() << '\n'; return 1;
    }
}
