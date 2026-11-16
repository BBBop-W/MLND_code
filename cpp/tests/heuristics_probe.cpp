#include "mlnd/heuristics.hpp"
#include <cmath>
#include <iostream>

namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
mlnd::Input fixture(int k) {
    mlnd::Input d;
    d.n_l = 1; d.n_h = 2; d.n_od = 1; d.n_levels = 1; d.max_orders = k;
    d.route_mode = "direct"; d.flow_policy = "aggregate"; d.model_version = "consecutive";
    d.delta = .01; d.operating_limit = .15; d.fleet_cost = 1;
    d.flight_cost = 1; d.waiting_cost = .4;
    d.demand = {20}; d.ground_cost = {10}; d.deadline = {.2}; d.base_cost = {.1};
    d.slot_cost = mlnd::Matrix<double>(2,1); d.occupancy_limit = mlnd::Matrix<double>(2,1);
    d.flight_times = mlnd::Matrix<double>(3,3); d.allowed_links = mlnd::Matrix<int>(3,3);
    d.link_cost = mlnd::Matrix<double>(3,3); d.kappa = mlnd::Matrix<double>(1,1);
    d.rider_mean = mlnd::Matrix<double>(1,2); d.first_cost = mlnd::Matrix<double>(1,2);
    d.terminal_cost = mlnd::Matrix<double>(1,2); d.receipt_mean = mlnd::Matrix<double>(1,2);
    d.removal_mean = mlnd::Matrix<double>(1,2);
    for (int u = 0; u < 3; ++u) for (int v = 0; v < 3; ++v) {
        d.flight_times(u,v) = u == v ? 0 : .02;
        d.allowed_links(u,v) = u != v;
    }
    for (int h = 0; h < 2; ++h) {
        d.slot_cost(h,0) = .2; d.occupancy_limit(h,0) = 20;
        d.first_cost(0,h) = .1; d.terminal_cost(0,h) = .1;
    }
    d.kappa(0,0) = 1;
    return d;
}
void check_wrapper(const mlnd::Json& result, double maximum) {
    require(result.at("validated"), "Unvalidated heuristic result");
    require(result.at("objective").get<double>() <= maximum + 1e-6, "Heuristic degraded retained incumbent");
    require(!result.contains("bound") && !result.contains("gap"), "Restricted bound leaked into wrapper");
    require(!result.at("solution").contains("bound") && !result.at("solution").contains("gap"), "Restricted bound leaked into solution");
    require(!result.at("global_lower_bound_available").get<bool>(), "Heuristic claimed a global lower bound");
}
}
int main() {
    try {
        mlnd::Json cases = mlnd::Json::array();
        for (int k : {1,2}) {
            auto d = fixture(k);
            const auto paths = mlnd::enumerate_paths(d);
            mlnd::SolverOptions options; options.seconds = 10; options.gap = 1e-8;
            const auto baseline = mlnd::all_courier_result(d, paths);
            const double fallback = baseline.at("objective");
            const auto expired = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() - 1);
            check_wrapper(expired, fallback);
            require(expired.at("source") == "all_courier_fallback", "Expired call did not return all-courier fallback");

            const auto full = mlnd::solve_full(d, paths, options);
            const auto result = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 10);
            check_wrapper(result, fallback);
            require(result.at("improved"), "Restricted MIP did not improve synthetic all-courier solution");
            require(std::abs(result.at("objective").get<double>() - full.at("objective").get<double>()) < 1e-4,
                    "Complete-column heuristic differs from independently solved full model");
            const auto& pool_budget = result.at("design_pool_metadata");
            const double reserved = pool_budget.at("postprocess_reserve_seconds");
            const double time_limit = pool_budget.at("optimize_time_limit_seconds");
            const double available = pool_budget.at("remaining_before_optimize");
            require(reserved > 0 && time_limit > 0, "Pool solve did not reserve postprocessing time");
            require(std::abs(time_limit + reserved - available) < 1e-8,
                    "Pool reservation was not deducted from the optimizer budget");
            require(reserved <= .25 * available + 1e-8, "Pool reservation consumed too much search budget");
            require(pool_budget.at("scanned").get<int>() > 0,
                    "Pool reserve fixture did not read a stored solution");
            require(pool_budget.at("remaining_before_pool").get<double>() > 0,
                    "Pool validation started after its deadline");
            require(result.at("design_pool").size() >= 2,
                    "Pool budget fixture lost the distinct courier and drone designs");
            for (const auto& entry : result.at("design_pool")) {
                require(entry.at("validated"), "Unvalidated pool candidate retained");
                require(!entry.contains("bound") && !entry.contains("gap"), "Pool candidate leaked a restricted bound");
                const auto design = entry.at("design").get<std::vector<double>>();
                require(mlnd::point_residuals(mlnd::design_model(d).matrix, design).at("passed"),
                        "Pool candidate violates the independent strategic model");
            }
            auto no_pool_options = options;
            no_pool_options.neighborhood_cuts = false;
            const auto no_pool = mlnd::solve_restricted_heuristic(d, paths, no_pool_options, mlnd::wall_time() + 10);
            check_wrapper(no_pool, fallback);
            require(!no_pool.contains("design_pool_metadata"), "Disabled pool still reserved/extracted candidates");
            require(std::abs(no_pool.at("objective").get<double>() - result.at("objective").get<double>()) < 1e-4,
                    "Pool budgeting changed the complete-column fixture optimum");
            const auto& incumbent = result.at("solution");
            const auto local = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 10, &incumbent, 0);
            check_wrapper(local, incumbent.at("objective"));
            require(mlnd::solution_design(d, local.at("solution")) == mlnd::solution_design(d, incumbent),
                    "Radius-zero local search changed structural design");
            require(local.at("diagnostics").at("incumbent_validation").at("passed"), "Valid incumbent failed lifting validation");

            auto corrupted = incumbent;
            corrupted["_solution"]["courier"][0] = -3;
            const auto rejected = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 10, &corrupted, 0);
            check_wrapper(rejected, fallback);
            require(!rejected.at("diagnostics").at("incumbent_validation").at("passed").get<bool>(),
                    "Invalid incumbent accepted");

            auto invalid_paths = paths;
            for (std::size_t p = 0; p < paths.service_paths.size(); ++p)
                if (incumbent.at("_solution").at("flow").at(p).get<double>() > 1e-5) {
                    invalid_paths.service_paths[p].cost += 1;
                    break;
                }
            const auto invalid = mlnd::solve_restricted_heuristic(d, invalid_paths, options, mlnd::wall_time() + 10, &incumbent, 0);
            check_wrapper(invalid, fallback);
            require(!invalid.at("diagnostics").at("incumbent_validation").at("passed").get<bool>(),
                    "Tampered physical path coefficients accepted");

            auto empty_pool = mlnd::prepare_path_support(d);
            const auto empty = mlnd::solve_restricted_heuristic(d, empty_pool, options, mlnd::wall_time() + 10);
            check_wrapper(empty, fallback);
            require(std::abs(empty.at("objective").get<double>() - fallback) < 1e-7, "Empty-column model invented drone service");

            // Save a nonzero incumbent on a strict prefix, append new columns,
            // and verify that automatic zero padding preserves the warm start.
            auto growing = mlnd::prepare_path_support(d);
            for (const auto& path : paths.service_paths)
                if (path.segment_id == paths.service_paths.front().segment_id)
                    mlnd::register_service_path(growing, path);
            const auto prefix = mlnd::solve_restricted_heuristic(d, growing, options, mlnd::wall_time() + 10);
            require(prefix.at("solution").at("served").get<double>() > 1, "Prefix fixture has no meaningful incumbent");
            const auto earlier = prefix.at("solution");
            const auto previous_count = growing.service_paths.size();
            for (const auto& path : paths.service_paths) mlnd::register_service_path(growing, path);
            require(growing.service_paths.size() > previous_count, "Dynamic-pool fixture did not grow");
            const auto reused = mlnd::solve_restricted_heuristic(d, growing, options, mlnd::wall_time() + 10, &earlier, 0);
            check_wrapper(reused, earlier.at("objective"));
            require(reused.at("diagnostics").at("incumbent_validation").at("passed"),
                    "Append-only pool growth invalidated a feasible incumbent");
            require(reused.at("solution").at("_solution").at("flow").size() == growing.service_paths.size(),
                    "Returned incumbent did not synchronize column dimensions");
            auto wrong_identity = earlier;
            wrong_identity["_missions"][0]["path"]["id"] = 999;
            const auto wrong = mlnd::solve_restricted_heuristic(d, growing, options, mlnd::wall_time() + 10, &wrong_identity, 0);
            require(!wrong.at("diagnostics").at("incumbent_validation").at("passed").get<bool>(),
                    "Mismatched prefix path identity was silently reinterpreted");
            auto oversized = earlier;
            while (oversized["_solution"]["flow"].size() <= growing.service_paths.size())
                oversized["_solution"]["flow"].push_back(0);
            const auto excess = mlnd::solve_restricted_heuristic(d, growing, options, mlnd::wall_time() + 10, &oversized, 0);
            require(!excess.at("diagnostics").at("incumbent_validation").at("passed").get<bool>(),
                    "Extra nonexistent columns in incumbent were accepted");
            cases.push_back({{"max_orders", k}, {"objective", result.at("objective")},
                             {"full_objective", full.at("objective")}, {"served", incumbent.at("served")},
                             {"local_objective", local.at("objective")}, {"dynamic_pool_warm_start", true},
                             {"pool_budget", pool_budget}, {"distinct_pool_designs", result.at("design_pool").size()},
                             {"passed", true}});
        }
        std::cout << mlnd::Json{{"passed", true}, {"cases", cases}}.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
