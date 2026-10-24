#include "mlnd/heuristics.hpp"
#include <cmath>
#include <iostream>
#include <set>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
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
void check(const mlnd::Input& d, const mlnd::Json& result, int capacity) {
    require(!result.at("global_lower_bound_available").get<bool>(), "Pool exposed a global lower bound");
    require(!result.contains("bound") && !result.contains("gap"), "Pool wrapper leaked a restricted bound");
    require(!result.at("solution").contains("bound") && !result.at("solution").contains("gap"),
            "Pool solution leaked a restricted bound");
    const auto& pool = result.at("design_pool");
    require(pool.size() <= static_cast<std::size_t>(capacity), "Pool exceeded configured capacity");
    const auto model = mlnd::design_model(d);
    std::set<std::vector<double>> distinct;
    for (const auto& entry : pool) {
        require(entry.at("validated").get<bool>(), "Pool retained an unvalidated candidate");
        require(!entry.contains("bound") && !entry.contains("gap") && !entry.contains("solution"),
                "Pool candidate included a bound or unbounded-size solution payload");
        const auto design = entry.at("design").get<std::vector<double>>();
        require(distinct.insert(design).second, "Pool retained duplicate strategic designs");
        require(mlnd::point_residuals(model.matrix, design).at("passed").get<bool>(),
                "Pool strategic design violates independent design constraints");
        require(std::isfinite(entry.at("objective").get<double>()), "Pool candidate has no finite primal objective");
    }
    const auto& times = result.at("timings");
    require(times.at("pool_extraction").get<double>() >= 0, "Negative pool extraction time");
    require(times.at("total").get<double>() + 1e-7 >= times.at("pool_extraction").get<double>()
        + times.at("build").get<double>() + times.at("optimize").get<double>()
        + times.at("validation").get<double>(), "Pool phase timing exceeds total wall time");
}
}

int main() {
    try {
        mlnd::Json records = mlnd::Json::array();
        for (int k : {1, 2}) {
            const auto d = fixture(k);
            const auto paths = mlnd::enumerate_paths(d);
            mlnd::SolverOptions options;
            options.seconds = 5; options.gap = 1e-8;
            options.neighborhood_cuts = true; options.heuristic_pool_size = 4;
            const auto baseline = mlnd::all_courier_result(d, paths);

            // All-closed radius-zero search cannot improve its exact courier
            // incumbent. Its solver design must nevertheless be offered for
            // subsequent full-path cut generation.
            const auto unchanged = mlnd::solve_restricted_heuristic(d, paths, options,
                mlnd::wall_time() + 5, &baseline, 0);
            check(d, unchanged, 4);
            require(!unchanged.at("improved").get<bool>(), "All-closed local search unexpectedly improved cost");
            require(unchanged.at("design_pool").size() == 1, "Nonimproving final solver design was discarded");
            require(unchanged.at("design_pool").front().at("design") == mlnd::solution_design(d, baseline),
                    "Radius-zero pool design differs from its structural center");

            const auto expanded = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 5);
            check(d, expanded, 4);
            require(expanded.at("improved").get<bool>(), "Fixture failed to produce an improving drone design");
            require(expanded.at("design_pool").size() >= 2,
                    "Fixture did not retain the distinct courier and improved drone designs");
            require(expanded.at("design_pool_metadata").at("source") == "optimize_solution_pool_no_populate",
                    "Candidate extraction invoked another optimization procedure");

            options.heuristic_pool_size = 1;
            const auto limited = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 5);
            check(d, limited, 1);
            require(limited.at("design_pool").size() == 1, "Capacity-one pool lost the final solver design");

            options.neighborhood_cuts = false;
            const auto disabled = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() + 5);
            require(!disabled.contains("design_pool") && !disabled.contains("design_pool_metadata"),
                    "Disabled cut generation still extracted a candidate pool");
            require(std::abs(disabled.at("objective").get<double>() - expanded.at("objective").get<double>()) < 1e-4,
                    "Pool storage settings changed the fixture optimum");

            options.neighborhood_cuts = true;
            const auto expired = mlnd::solve_restricted_heuristic(d, paths, options, mlnd::wall_time() - 1);
            check(d, expired, 1);
            require(expired.at("design_pool").empty(), "Expired heuristic fabricated pool candidates");
            records.push_back({{"K", k}, {"distinct_designs", expanded.at("design_pool").size()},
                {"available_pool_solutions", expanded.at("design_pool_metadata").at("available")},
                {"nonimproving_design_retained", true}, {"passed", true}});
        }
        std::cout << mlnd::Json{{"passed", true}, {"cases", records}}.dump(2) << '\n';
    } catch (const IloException& error) {
        std::cerr << error.getMessage() << '\n'; return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
