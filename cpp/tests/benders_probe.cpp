#include "mlnd/benders.hpp"
#include "mlnd/benders_control.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace mlnd;
namespace {
template<class T> Matrix<T> constant(int rows, int columns, T value) {
    Matrix<T> matrix(rows, columns);
    std::fill(matrix.values.begin(), matrix.values.end(), value);
    return matrix;
}

// The same explicit physical fixture as tests/unit/test_solver.py. Its Poisson
// thresholds are frozen values of poisson_mean_threshold(10/20, 0.05).
Input fixture(int max_orders, const std::string& policy) {
    Input d;
    d.n_l = 1; d.n_h = 2; d.n_od = 2; d.n_levels = 2; d.max_orders = max_orders;
    d.name = "native_benders_test"; d.model_version = "consecutive";
    d.route_mode = "direct"; d.flow_policy = policy;
    d.demand = {4, 7};
    d.flight_times = constant<double>(3, 3, 0);
    d.flight_times.values = {0, .05, .07, .05, 0, .1, .07, .1, 0};
    d.rider_mean = constant<double>(2, 2, 0);
    d.rider_mean.values = {.08, .12, .1, .06};
    d.first_cost = constant<double>(2, 2, 1);
    d.terminal_cost = constant<double>(2, 2, .5);
    d.receipt_mean = constant<double>(2, 2, .08);
    d.removal_mean = constant<double>(2, 2, .03);
    d.ground_cost = {5, 5}; d.deadline = {1, 1}; d.base_cost = {.5};
    d.slot_counts = constant<int>(2, 2, 0);
    d.slot_counts.values = {10, 20, 10, 20};
    d.slot_cost = constant<double>(2, 2, 0);
    d.slot_cost.values = {.1, .2, .1, .2};
    d.slot_epsilon = {.05, .05};
    d.link_cost = constant<double>(3, 3, 0);
    d.kappa = constant<double>(2, 2, 0);
    d.kappa.values = {.3, .9, .2, .7};
    d.delta = .04; d.operating_limit = 1;
    d.fleet_cost = 1; d.flight_cost = .2; d.waiting_cost = .3;
    d.allowed_links = constant<int>(3, 3, 1);
    for (int i = 0; i < 3; ++i) d.allowed_links(i, i) = 0;
    d.occupancy_limit = constant<double>(2, 2, 0);
    d.occupancy_limit.values = {6.1690072893953225, 14.072024748341311,
                              6.1690072893953225, 14.072024748341311};
    return d;
}

bool common_matrix(const Input& d, const PathCollection& paths) {
    DesignLayout layout(d);
    std::vector<double> one(layout.count, 0), two(layout.count, 0);
    one[layout.assignment] = 1; two[layout.assignment + 1] = 1;
    auto a = build_formulation(d, paths, &one), b = build_formulation(d, paths, &two);
    if (a.matrix.variables.size() != b.matrix.variables.size() || a.matrix.rows.size() != b.matrix.rows.size()) return false;
    for (std::size_t i = 0; i < a.matrix.variables.size(); ++i) {
        const auto& x = a.matrix.variables[i]; const auto& y = b.matrix.variables[i];
        if (x.lower != y.lower || x.upper != y.upper || x.cost != y.cost || x.binary != y.binary) return false;
    }
    for (std::size_t i = 0; i < a.matrix.rows.size(); ++i) {
        const auto& x = a.matrix.rows[i]; const auto& y = b.matrix.rows[i];
        if (x.coefficients != y.coefficients || x.sense != y.sense) return false;
        if (x.rhs != y.rhs && std::find(a.design_fix_rows.begin(), a.design_fix_rows.end(), i) == a.design_fix_rows.end()) return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto loose = master_solve_budget(true, 1000, .05, 2, .8, 1e-7, 0);
        const auto tightened = master_solve_budget(true, 1000, .05, 2, .8, 1e-7, 2);
        const auto exact = master_solve_budget(true, 1000, .05, 2, .8, 1e-7, 6);
        if (loose.relative_gap != .05 || !(tightened.relative_gap < loose.relative_gap) ||
            !(tightened.seconds > loose.seconds) || exact.relative_gap != 0 || exact.seconds != 1000)
            throw std::runtime_error("Inexact master cannot progress to full certification");
        Json records = Json::array();
        for (const std::string policy : {"aggregate", "prefix"}) for (int max_orders : {1, 2}) {
            auto data = fixture(max_orders, policy);
            auto paths = enumerate_paths(data);
            auto implicit = prepare_path_support(data);
            if (!implicit.support_complete || implicit.complete ||
                !implicit.service_paths.empty() || !implicit.empty_paths.empty() ||
                Json(implicit.initial_segments) != Json(paths.initial_segments))
                throw std::runtime_error("Implicit preprocessing enumerated paths or changed model support");
            SolverOptions options;
            options.seconds = 20; options.gap = 1e-7; options.max_iterations = 1000;
            Json full = solve_full(data, paths, options);
            auto relaxation = build_formulation(data, paths);
            for (auto& variable : relaxation.matrix.variables) variable.binary = false;
            NativeSolver continuous(relaxation.matrix, options, "test_full_continuous");
            const auto continuous_solution = continuous.optimize(20);
            if (!continuous_solution.optimal || !continuous_solution.has_solution ||
                !point_residuals(relaxation.matrix, continuous_solution.values).at("passed").get<bool>())
                throw std::runtime_error("Full continuous reference is not certified");
            SolverOptions root_options = options;
            root_options.max_iterations = 0; root_options.root_cut_rounds = 100;
            root_options.root_budget_fraction = 1;
            root_options.initial_design = root_options.restricted_mip = root_options.neighborhood_cuts = false;
            root_options.continuous_root = false; // Audit the original Benders projection independently.
            root_options.root_projection = root_options.root_inout = false;
            root_options.pareto = false;
            auto root_pool = prepare_path_support(data);
            auto root_only = solve_benders(data, root_pool, root_options);
            const auto& root_diagnostic = root_only.at("root_diagnostics");
            const double relaxation_tolerance = 4e-6 * (1 + std::abs(continuous_solution.objective));
            if (!root_diagnostic.at("converged").get<bool>() ||
                std::abs(root_diagnostic.at("lower_bound").get<double>() - continuous_solution.objective) > relaxation_tolerance ||
                root_diagnostic.at("continuous_upper_bound").get<double>() < continuous_solution.objective - relaxation_tolerance)
                throw std::runtime_error("Certified Benders root disagrees with full continuous relaxation");
            SolverOptions zero_gap_options = options;
            zero_gap_options.gap = 0;
            zero_gap_options.max_iterations = 80;
            zero_gap_options.restricted_mip = zero_gap_options.neighborhood_cuts = false;
            auto zero_gap_pool = prepare_path_support(data);
            const auto zero_gap_result = solve_benders(data, zero_gap_pool, zero_gap_options);
            const auto zero_reason = zero_gap_result.at("termination_reason").get<std::string>();
            if (zero_reason != "gap_limit" && zero_reason != "certificate_tolerance_limit")
                throw std::runtime_error("Zero requested gap caused numerical duplicate-cut stagnation");
            if (zero_reason == "gap_limit" && zero_gap_result.at("gap").get<double>() != 0)
                throw std::runtime_error("Nonzero numerical gap mislabeled as zero-gap convergence");
            if (zero_gap_result.at("bound").get<double>() > full.at("objective").get<double>() + 2e-6 ||
                zero_gap_result.at("gap").get<double>() > 4e-6)
                throw std::runtime_error("Numerical termination lost its certified bounds");
            Json fractional = all_courier_result(data, paths);
            fractional["_solution"]["facility"][0] = .5;
            bool fractional_rejected = false;
            try { solve_benders(data, implicit, options, &fractional); }
            catch (const std::invalid_argument&) { fractional_rejected = true; }
            if (!fractional_rejected) throw std::runtime_error("Fractional Benders start was accepted");
            const auto fixed_design = solution_design(data, full);
            SolverOptions one_column = options;
            one_column.cg_max_columns = 1;
            one_column.pareto = false;
            const auto fixed_cg = solve_fixed_design_cg(data, implicit, one_column, fixed_design);
            if (!fixed_cg.at("pricing_complete").get<bool>() ||
                std::abs(fixed_cg.at("objective").get<double>() - full.at("objective").get<double>()) > 2e-5)
                throw std::runtime_error("One-column CG did not recover full fixed-design optimum");
            SolverOptions no_budget = one_column;
            no_budget.seconds = 0;
            auto stopped_pool = prepare_path_support(data);
            const auto incomplete = solve_fixed_design_cg(data, stopped_pool, no_budget, fixed_design);
            if (incomplete.at("pricing_complete").get<bool>() || !incomplete.at("benders_cuts").empty() ||
                solution_design(data, incomplete) != fixed_design)
                throw std::runtime_error("Incomplete fixed-design CG certified a cut or changed the design");
            Json variants = Json::object();
            for (const std::string mode : {"accelerated", "plain", "full_recourse", "no_pareto", "plain_plus_search", "capacity_only",
                                           "capacity_joint_root", "capacity_joint_integer"}) {
                SolverOptions selected = options;
                if (mode == "plain" || mode == "plain_plus_search" || mode == "capacity_only" ||
                    mode == "capacity_joint_root" || mode == "capacity_joint_integer") {
                    selected.pareto = selected.reuse_columns = selected.courier_savings_bound = false;
                    selected.initial_design = selected.warm_start = selected.incumbent_cleanup = selected.primal_search = false;
                    selected.branch_covering_seeds = selected.resource_bounds = false;
                    selected.capacity_cuts = false;
                    selected.resource_lp_cuts = selected.adaptive_master = selected.neighborhood_cuts = false;
                    selected.continuous_root = false;
                    selected.root_projection = selected.root_inout = selected.auxiliary_valid_cuts = false;
                    selected.root_cuts = selected.restricted_mip = selected.local_branching = selected.incumbent_cutoff = false;
                }
                if (mode == "plain_plus_search") selected.primal_search = true;
                if (mode == "capacity_only") selected.capacity_cuts = true;
                // With no initial cost rows or incumbent heuristic, the first
                // all-closed candidate has theta=0 but positive courier cost.
                // Its violated capacity cut and exact recourse cut must both
                // be collected before another master optimization.
                if (mode == "capacity_joint_root") {
                    selected.capacity_cuts = selected.root_cuts = true;
                    selected.root_cut_rounds = 3;
                }
                if (mode == "capacity_joint_integer") {
                    selected.capacity_cuts = true;
                    selected.root_cut_rounds = 0;
                }
                if (mode == "full_recourse") selected.column_generation = false;
                if (mode == "no_pareto") selected.pareto = false;
                auto pool = mode == "full_recourse" ? paths : prepare_path_support(data);
                variants[mode] = solve_benders(data, pool, selected);
                const auto& structure = variants[mode].at("master_structure");
                if (structure.at("variables").get<int>() != DesignLayout(data).count + 1 ||
                    structure.at("operational_auxiliary_variables").get<int>() != 0 ||
                    structure.at("cones").get<int>() != 0)
                    throw std::runtime_error("Benders master contains operational auxiliaries or cones");
                if (selected.continuous_root) {
                    const auto& continuous = variants[mode].at("continuous_root");
                    if (!continuous.at("bound_valid").get<bool>() ||
                        continuous.at("certified_lower_bound").get<double>() > continuous_solution.objective + relaxation_tolerance ||
                        variants[mode].at("bound").get<double>() + relaxation_tolerance < continuous.at("certified_lower_bound").get<double>())
                        throw std::runtime_error("Continuous initialization lost or invalidated the global lower bound");
                    if (variants[mode].at("acceleration_statistics").at("continuous_bound_rows").get<int>() != 1)
                        throw std::runtime_error("Missing total-cost continuous bound row in lean master");
                    if (selected.root_projection && variants[mode].at("acceleration_statistics").at("root_cg_calls").get<int>() == 0)
                        throw std::runtime_error("A value-only floor incorrectly skipped projection separation");
                }
                std::cerr << "K=" << max_orders << " policy=" << policy << " mode=" << mode
                          << " objective=" << variants[mode]["objective"] << " gap=" << variants[mode]["gap"]
                          << " reason=" << variants[mode]["termination_reason"] << '\n';
            }
            records.push_back({{"K", max_orders}, {"policy", policy}, {"full", full},
                               {"root_diagnostic",root_diagnostic},{"full_continuous_objective",continuous_solution.objective},
                               {"zero_gap_result",zero_gap_result},
                               {"fixed_one_column_cg", fixed_cg}, {"zero_budget_cg", incomplete},
                               {"fractional_start_rejected", fractional_rejected},
                               {"implicit_preprocessing_verified", true},
                               {"matrix_invariance", common_matrix(data, paths)}, {"variants", variants}});
        }
        if (argc > 1) {
            std::ofstream output(argv[1]);
            if (!output) throw std::runtime_error("Cannot open probe output");
            output << records.dump() << '\n';
        } else std::cout << records.dump() << '\n';
    } catch (const IloException& exception) {
        std::cerr << exception << '\n'; return 1;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n'; return 1;
    }
    return 0;
}
