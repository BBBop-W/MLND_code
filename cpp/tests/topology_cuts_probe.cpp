// Exhaustive finite-domain checks; no optimizer or benchmark instance is run.
#include "mlnd/topology_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> mlnd::Matrix<T> filled(int rows, int columns, T value) {
    mlnd::Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
mlnd::Input fixture(int k, const std::string& policy, int bases = 1) {
    mlnd::Input d;
    d.n_l = bases; d.n_h = 2; d.n_od = 2; d.n_levels = 1; d.max_orders = k;
    d.name = "topology_cut_probe"; d.model_version = "consecutive";
    d.route_mode = "direct"; d.flow_policy = policy;
    d.demand = {4, 7}; d.ground_cost = {5, 7}; d.deadline = {1, 1};
    d.base_cost.assign(bases, .5); d.slot_epsilon = {.05, .05};
    const int N = d.n_nodes();
    d.flight_times = filled(N, N, .05);
    d.allowed_links = filled(N, N, 1);
    for (int i = 0; i < N; ++i) {
        d.flight_times(i, i) = 0; d.allowed_links(i, i) = 0;
    }
    d.flight_times(bases, bases + 1) = d.flight_times(bases + 1, bases) = .1;
    d.link_cost = filled(N, N, 0.0);
    d.rider_mean = filled(2, 2, 0.0);
    d.first_cost = filled(2, 2, 1.0); d.terminal_cost = filled(2, 2, .5);
    d.receipt_mean = filled(2, 2, .08); d.removal_mean = filled(2, 2, .03);
    d.slot_counts = filled(2, 1, 10); d.slot_cost = filled(2, 1, .1);
    d.occupancy_limit = filled(2, 1, 6.0); d.kappa = filled(2, 2, 1.0);
    d.delta = .04; d.operating_limit = 1; d.fleet_cost = 1;
    d.flight_cost = .2; d.waiting_cost = .3;
    return d;
}
std::vector<std::pair<int, int>> service_arcs(const mlnd::Input& d) {
    std::vector<std::pair<int, int>> result;
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        if (u != v && !(u < d.n_l && v < d.n_l) && d.allowed_links(u, v))
            result.emplace_back(u, v);
    return result;
}
std::vector<double> design(const mlnd::Input& d,
        const std::vector<std::pair<int, int>>& arcs, unsigned mask, unsigned assignment = 3) {
    const mlnd::DesignLayout layout(d);
    std::vector<double> result(layout.count, 0);
    std::fill(result.begin(), result.begin() + d.n_nodes(), 1);
    for (int i = 0; i < d.n_h; ++i) result[layout.capacity + i] = 1;
    for (int r = 0; r < d.n_od; ++r)
        result[layout.assignment + r * d.n_l * d.n_l] = (assignment >> r) & 1;
    for (std::size_t k = 0; k < arcs.size(); ++k)
        result[layout.airline + arcs[k].first * d.n_nodes() + arcs[k].second] = (mask >> k) & 1;
    return result;
}
bool enabled(const mlnd::Input& d, const mlnd::ServicePath& p,
             const std::vector<double>& x) {
    const mlnd::DesignLayout layout(d);
    for (const auto& [u, v, count] : p.arc_counts) {
        (void)count;
        if (x[layout.airline + u * d.n_nodes() + v] < .5) return false;
    }
    return true;
}
double violation(const mlnd::RowRecord& row, const std::vector<double>& x) {
    long double lhs = 0;
    for (const auto& [j, c] : row.coefficients) lhs += static_cast<long double>(c) * x.at(j);
    return static_cast<double>(row.rhs - lhs);
}

mlnd::Json exhaustive(int k, const std::string& policy) {
    const auto d = fixture(k, policy);
    const auto complete = mlnd::enumerate_paths(d);
    auto support = mlnd::prepare_path_support(d);
    require(support.service_paths.empty() && !support.complete,
            "Fixture must exercise a genuinely empty generated path pool");
    const auto arcs = service_arcs(d);
    const mlnd::DesignLayout layout(d);
    const int theta = layout.count;
    const double baseline = std::inner_product(d.demand.begin(), d.demand.end(),
                                               d.ground_cost.begin(), 0.0);
    std::vector<mlnd::TopologyCostCut> cuts;
    std::size_t supported_networks = 0, aa_paths = 0;
    for (const auto& p : complete.service_paths)
        if (p.second_od == p.first_od) {
            require(p.served_counts.size() == 1 && p.served_counts.front().second == 2,
                    "AA must be one task with two order occurrences");
            ++aa_paths;
        }
    const unsigned networks = 1U << arcs.size();
    for (unsigned mask = 0; mask < networks; ++mask) {
        const auto x = design(d, arcs, mask);
        const bool exists = std::any_of(complete.service_paths.begin(), complete.service_paths.end(),
            [&](const auto& p) { return enabled(d, p, x); });
        const auto cut = mlnd::separate_topology_cost(d, support, x, theta, 0,
                                                     std::numeric_limits<double>::infinity());
        require(!cut.timed_out, "Unlimited tiny oracle must finish");
        require(cut.zero_service_certified == !exists,
                "Independent catalogue disagrees with the topology absence oracle");
        require(cut.valid == !exists && cut.violated == !exists,
                "A physical path witness must prevent a zero-service certificate");
        if (exists) { ++supported_networks; continue; }
        require(cut.shrink_complete && cut.blocking_count <= cut.initial_closed_count,
                "Blocking-set shrinking must finish without adding closed arcs");
        for (const auto& p : complete.service_paths) {
            bool hit = false;
            for (const auto& [u, v, count] : p.arc_counts) {
                (void)count;
                hit = hit || std::binary_search(cut.blocking_arcs.begin(), cut.blocking_arcs.end(),
                                                std::make_pair(u, v));
            }
            require(hit, "Certified blocking set misses a physical complete path");
        }
        // Complete shrinking is inclusion-minimal, not a claimed minimum-cardinality cut.
        for (const auto& removed : cut.blocking_arcs) {
            bool witness = false;
            for (const auto& p : complete.service_paths) {
                bool blocked = false;
                for (const auto& [u, v, count] : p.arc_counts) {
                    (void)count;
                    const auto e = std::make_pair(u, v);
                    blocked = blocked || (e != removed && std::binary_search(
                        cut.blocking_arcs.begin(), cut.blocking_arcs.end(), e));
                }
                witness = witness || !blocked;
            }
            require(witness, "Greedy completed set is not inclusion-minimal");
        }
        cuts.push_back(cut);
    }
    std::size_t projected_points = 0;
    double max_violation = 0;
    // Cross-network and cross-assignment checks use a stronger relaxation
    // than real recourse: courier plus one supported task, with queue/empty
    // costs omitted. Validity here therefore implies validity for these real
    // feasible operations after their additional nonnegative costs are added.
    for (unsigned mask = 0; mask < networks; ++mask) for (unsigned a = 0; a < 4; ++a) {
        auto x = design(d, arcs, mask, a);
        x.push_back(baseline);
        for (const auto& cut : cuts) {
            require(violation(cut.row, x) <= 1e-7, "All-courier fallback was excluded");
            ++projected_points;
        }
        for (const auto& p : complete.service_paths) {
            if (!enabled(d, p, x)) continue;
            double rate = std::numeric_limits<double>::infinity(), ground = 0;
            bool assigned = true;
            for (const auto& [r, n] : p.served_counts) {
                assigned = assigned && x[layout.assignment + r] > .5;
                rate = std::min(rate, d.demand[r] / n);
                ground += n * d.ground_cost[r];
            }
            if (!assigned) continue;
            x[theta] = baseline + rate * (p.cost + d.fleet_cost * p.time - ground);
            require(x[theta] >= -1e-8, "Projected physical cost cannot be negative");
            for (const auto& cut : cuts) {
                max_violation = std::max(max_violation, violation(cut.row, x));
                ++projected_points;
            }
        }
    }
    require(max_violation <= 1e-7, "Topology cut excluded a cross-design operational witness");
    require(k == 1 || aa_paths > 0, "K=2 fixture must contain AA tasks");
    return {{"K", k}, {"policy", policy}, {"networks", networks},
        {"supported_networks", supported_networks}, {"certified_cuts", cuts.size()},
        {"physical_paths", complete.service_paths.size()}, {"AA_paths", aa_paths},
        {"cross_projection_checks", projected_points}, {"max_violation", max_violation}};
}

mlnd::Json boundaries() {
    auto d = fixture(2, "aggregate");
    const mlnd::DesignLayout layout(d);
    const int theta = layout.count;
    auto support = mlnd::prepare_path_support(d);
    auto x = design(d, service_arcs(d), 0);
    auto open = [&](int u, int v) { x[layout.airline + u * d.n_nodes() + v] = 1; };
    auto separate = [&] { return mlnd::separate_topology_cost(d, support, x, theta, 0,
        std::numeric_limits<double>::infinity()); };
    auto first = separate();
    require(first.valid && first.certificate_stage == "complete_first_segment_absence",
            "Fully closed network must be certified at the first-segment stage");
    open(0, 1); open(1, 2);
    auto no_return = separate();
    require(no_return.valid && no_return.certificate_stage == "complete_physical_path_absence",
            "Open first segment without a completion needs the physical-path proof");
    open(2, 1); open(1, 0);
    d.kappa.values = {1, 0, 0, 1}; // Only AA/BB, with zero-length transfer at delivery.
    require(!separate().valid, "An AA-only continuation completion was wrongly excluded");
    d.kappa = filled(2, 2, 0.0);
    require(separate().valid, "Forbidden continuation branches created a false witness");
    d.kappa = filled(2, 2, 1.0); d.delta = 0;
    support = mlnd::prepare_path_support(d);
    require(separate().valid, "Zero effective window must not admit a double task");
    d.delta = .04; support = mlnd::prepare_path_support(d);
    const auto timeout = mlnd::separate_topology_cost(d, support, x, theta, 0, mlnd::wall_time() - 1);
    require(timeout.timed_out && !timeout.valid && !timeout.zero_service_certified,
            "An incomplete initial scan produced a certificate");
    auto partial = support; partial.support_complete = false;
    bool incomplete_rejected = false, fractional_rejected = false;
    try { mlnd::separate_topology_cost(d, partial, x, theta, 0, mlnd::wall_time() + 1); }
    catch (const std::invalid_argument&) { incomplete_rejected = true; }
    auto fractional = x; fractional[layout.airline + 1] = .5;
    try { mlnd::separate_topology_cost(d, support, fractional, theta, 0, mlnd::wall_time() + 1); }
    catch (const std::invalid_argument&) { fractional_rejected = true; }
    require(incomplete_rejected && fractional_rejected,
            "Incomplete support or a fractional anchor must be rejected");
    // Available physical topology must be detected even if no OD is assigned.
    std::fill(x.begin() + layout.assignment, x.begin() + layout.airline, 0);
    require(!separate().valid, "Topology oracle silently depended on assignment");

    auto cross = fixture(2, "aggregate", 2);
    const mlnd::DesignLayout cl(cross);
    const auto cp = mlnd::prepare_path_support(cross);
    auto cx = design(cross, service_arcs(cross), 0);
    cx[cl.airline + 0 * 4 + 2] = cx[cl.airline + 2 * 4 + 3] = cx[cl.airline + 3 * 4 + 1] = 1;
    const auto witness = mlnd::separate_topology_cost(cross, cp, cx, cl.count, 0,
        std::numeric_limits<double>::infinity());
    require(!witness.valid, "Cross-base service was incorrectly conditioned on empty rebalancing");
    std::fill(cx.begin() + cl.airline, cx.end(), 0);
    cx[cl.airline + 1] = cx[cl.airline + 4] = 1;
    const auto empty_only = mlnd::separate_topology_cost(cross, cp, cx, cl.count, 0,
        std::numeric_limits<double>::infinity());
    require(empty_only.valid, "Base-to-base empty flights cannot serve an order");
    for (const auto& [u, v] : empty_only.blocking_arcs)
        require(u >= cross.n_l || v >= cross.n_l, "An empty base arc entered the blocking set");

    d.operating_limit = 0;
    support = mlnd::prepare_path_support(d);
    const auto impossible = separate();
    require(impossible.valid && impossible.blocking_arcs.empty(),
            "Globally impossible physical missions should yield theta >= all-courier cost");
    return {{"first_segment_and_completion_stages", true}, {"AA_only_completion", true},
        {"zero_window_and_zero_compatibility", true}, {"assignment_independent", true},
        {"timeout_not_certified", true}, {"fractional_anchor_rejected", true},
        {"incomplete_support_rejected", true}, {"cross_base_service", true},
        {"empty_flights_excluded_from_blocking_set", true}, {"empty_blocking_set", true}};
}

mlnd::Json saved_zero_service_design(const char* input, const char* design_file) {
    const auto d = mlnd::read_input(input, 2);
    std::ifstream stream(design_file);
    if (!stream) throw std::runtime_error("Cannot open saved topology design");
    mlnd::Json supplied; stream >> supplied;
    const auto x = mlnd::solution_design(d, supplied);
    const auto support = mlnd::prepare_path_support(d);
    const mlnd::DesignLayout layout(d);
    const auto cut = mlnd::separate_topology_cost(d, support, x, layout.count, 0,
                                                  mlnd::wall_time() + 2);
    require(cut.valid && cut.violated && cut.zero_service_certified,
            "Saved zero-service design did not obtain a certified topology cost cut");
    // Full enumeration is a bounded test oracle, never the production separator.
    const auto complete = mlnd::enumerate_paths(d, std::nullopt, 10.0);
    std::size_t enabled_paths = 0;
    for (const auto& p : complete.service_paths) {
        enabled_paths += enabled(d, p, x);
        bool hit = false;
        for (const auto& [u, v, count] : p.arc_counts) {
            (void)count;
            hit = hit || std::binary_search(cut.blocking_arcs.begin(), cut.blocking_arcs.end(),
                                            std::make_pair(u, v));
        }
        require(hit, "Saved-design blocking set misses a complete physical path");
    }
    require(enabled_paths == 0, "Saved fixture actually has an enabled service path");
    auto report = cut.metadata;
    report["input"] = input;
    report["design_file"] = design_file;
    report["complete_physical_paths_checked"] = complete.service_paths.size();
    report["enabled_physical_paths"] = enabled_paths;
    return report;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 2 && argc != 4)
            throw std::invalid_argument("Usage: topology_cuts_probe [OUTPUT_JSON [INPUT SAVED_ZERO_SERVICE_DESIGN]]");
        mlnd::Json report{{"cases", mlnd::Json::array()}};
        for (int k : {1, 2}) for (const std::string policy : {"aggregate", "prefix"})
            report["cases"].push_back(exhaustive(k, policy));
        report["boundaries"] = boundaries();
        if (argc == 4) report["saved_design"] = saved_zero_service_design(argv[2], argv[3]);
        report["passed"] = true;
        report["solver_runs"] = 0;
        if (argc > 1) {
            std::ofstream output(argv[1]);
            if (!output) throw std::runtime_error("Cannot open topology probe report");
            output << report.dump(2) << '\n';
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
