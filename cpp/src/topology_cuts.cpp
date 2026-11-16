#include "mlnd/topology_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mlnd {
namespace {

enum class Existence { absent, present, incomplete };
struct ExistenceResult {
    Existence state = Existence::incomplete;
    bool open_initial_segment = false;
};

// This is an existence oracle, not a column generator. It retains at most
// one candidate path, and uses the same authoritative scalar checks as the
// complete catalogue and pricing. Assignments and base balancing are relaxed.
ExistenceResult service_exists(const Input& d, const PathCollection& paths,
        const std::vector<unsigned char>& open, double deadline,
        TopologyCostCut& report) {
    ++report.oracle_calls;
    const int N = d.n_nodes(), L = d.n_l;
    auto stopped = [&] { return wall_time() >= deadline; };
    auto arc = [&](int u, int v) {
        return u == v || open[static_cast<std::size_t>(u) * N + v] != 0;
    };
    ExistenceResult result;
    if (stopped()) return result;
    for (const auto& segment : paths.initial_segments) {
        if (stopped()) return result;
        ++report.initial_segments_checked;
        const int pickup = L + segment.pickup, delivery = L + segment.delivery;
        if (!arc(segment.b[0], pickup) || !arc(pickup, delivery)) continue;
        result.open_initial_segment = true;
        ServicePath path;
        if (arc(delivery, segment.b[1])) {
            ++report.single_paths_checked;
            const bool feasible = build_service_path(d, segment, nullptr, path);
            if (stopped()) return result;
            if (feasible) { result.state = Existence::present; return result; }
        }
        if (d.effective_delta() == 0) continue;
        for (int r = 0; r < d.n_od; ++r) {
            if (stopped()) return result;
            if (d.kappa(segment.first_od, r) <= 0) continue;
            for (int i = 0; i < d.n_h; ++i) {
                if (stopped()) return result;
                const int next_pickup = L + i;
                if (!arc(delivery, next_pickup)) continue;
                const double transfer = d.flight_times(delivery, next_pickup);
                const double wait = second_wait(d.rider_mean(r, i), transfer,
                                                 d.effective_delta());
                for (int j = 0; j < d.n_h; ++j) {
                    if (stopped()) return result;
                    ++report.continuation_candidates_checked;
                    const int next_delivery = L + j;
                    if (i == j || !arc(next_pickup, next_delivery) ||
                        !arc(next_delivery, segment.b[1])) continue;
                    const PathContinuation next{r, i, j, transfer,
                        d.flight_times(next_pickup, next_delivery),
                        d.flight_times(next_delivery, segment.b[1]), wait};
                    ++report.double_paths_checked;
                    const bool feasible = build_service_path(d, segment, &next, path);
                    if (stopped()) return result;
                    if (feasible) { result.state = Existence::present; return result; }
                }
            }
        }
    }
    if (!stopped()) result.state = Existence::absent;
    return result;
}

void validate_nonnegative(const std::vector<double>& values, const char* name) {
    for (double value : values)
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument(std::string("Invalid topology-cut ") + name);
}

} // namespace

TopologyCostCut separate_topology_cost(const Input& d, const PathCollection& paths,
        const std::vector<double>& design, int theta_index, double theta_value,
        double deadline) {
    const double started = wall_time();
    if (d.route_mode != "direct" || d.has_candidate_routes ||
        d.max_orders < 1 || d.max_orders > 2)
        throw std::invalid_argument("Topology cuts require the direct K=1/K=2 path domain");
    if (!paths.support_complete)
        throw std::invalid_argument("Topology cuts require complete initial-segment support");
    const DesignLayout layout(d);
    const int N = d.n_nodes(), L = d.n_l;
    if (design.size() < static_cast<std::size_t>(layout.count) ||
        theta_index < layout.count || !std::isfinite(theta_value) || std::isnan(deadline))
        throw std::invalid_argument("Invalid topology-cut design, theta or deadline");
    for (int j = 0; j < layout.count; ++j)
        if (!std::isfinite(design[j]) || design[j] < -1e-6 || design[j] > 1 + 1e-6 ||
            std::abs(design[j] - std::round(design[j])) > 1e-6)
            throw std::invalid_argument("Topology cuts require an integer design");
    validate_nonnegative(d.demand, "demand");
    validate_nonnegative(d.ground_cost, "courier costs");
    validate_nonnegative(d.first_cost.values, "first-leg costs");
    validate_nonnegative(d.terminal_cost.values, "terminal costs");
    if (!std::isfinite(d.fleet_cost) || !std::isfinite(d.flight_cost) ||
        !std::isfinite(d.waiting_cost) || d.fleet_cost < 0 || d.flight_cost < 0 || d.waiting_cost < 0)
        throw std::invalid_argument("Topology cuts need nonnegative operating costs");

    TopologyCostCut result;
    long double baseline = 0;
    for (int r = 0; r < d.n_od; ++r)
        baseline += static_cast<long double>(d.demand[r]) * d.ground_cost[r];
    result.courier_cost = static_cast<double>(baseline);
    if (!std::isfinite(result.courier_cost))
        throw std::invalid_argument("Nonfinite all-courier topology-cut cost");
    result.numerical_margin = 1e-8 * (1 + result.courier_cost);
    auto finish = [&] {
        result.seconds = wall_time() - started;
        result.blocking_count = result.blocking_arcs.size();
        result.metadata = {{"status", result.status}, {"valid", result.valid},
            {"violated", result.violated}, {"integer_valid_only", true},
            {"assignment_independent", true}, {"requires_complete_catalogue", false},
            {"support_complete", paths.support_complete},
            {"certificate_stage", result.certificate_stage},
            {"zero_service_certified", result.zero_service_certified},
            {"timed_out", result.timed_out}, {"shrink_complete", result.shrink_complete},
            {"initial_closed_count", result.initial_closed_count},
            {"blocking_count", result.blocking_count}, {"oracle_calls", result.oracle_calls},
            {"initial_segments_checked", result.initial_segments_checked},
            {"single_paths_checked", result.single_paths_checked},
            {"double_paths_checked", result.double_paths_checked},
            {"continuation_candidates_checked", result.continuation_candidates_checked},
            {"shrink_checks", result.shrink_checks}, {"seconds", result.seconds},
            {"courier_cost", result.courier_cost}, {"numerical_margin", result.numerical_margin},
            {"violation", result.violation}, {"blocking_arcs", result.blocking_arcs}};
        return result;
    };
    if (wall_time() >= deadline) {
        result.status = "initial_scan_timeout"; result.timed_out = true; return finish();
    }
    if (result.courier_cost <= result.numerical_margin) {
        result.status = "zero_courier_cost"; return finish();
    }
    std::vector<unsigned char> open(static_cast<std::size_t>(N) * N);
    for (int u = 0; u < N; ++u) for (int v = 0; v < N; ++v) {
        if (wall_time() >= deadline) {
            result.status = "initial_scan_timeout"; result.timed_out = true; return finish();
        }
        if (u == v || (u < L && v < L) || !d.allowed_links(u, v)) continue;
        const int index = u * N + v;
        open[index] = design[layout.airline + index] > .5;
        if (!open[index]) result.blocking_arcs.emplace_back(u, v);
    }
    result.initial_closed_count = result.blocking_arcs.size();
    const auto initial = service_exists(d, paths, open, deadline, result);
    if (initial.state == Existence::incomplete) {
        result.status = "initial_scan_timeout"; result.timed_out = true;
        result.blocking_arcs.clear(); return finish();
    }
    if (initial.state == Existence::present) {
        result.status = "physical_service_path_exists";
        result.blocking_arcs.clear(); return finish();
    }
    result.zero_service_certified = true;
    result.certificate_stage = initial.open_initial_segment ?
        "complete_physical_path_absence" : "complete_first_segment_absence";

    // Reopen one formerly closed service arc at a time. An arc is removed
    // from E only after a new COMPLETE absence proof. A witness preserves it;
    // an interrupted scan preserves the previously certified set unchanged.
    const auto candidates = result.blocking_arcs;
    for (const auto& edge : candidates) {
        if (wall_time() >= deadline) { result.timed_out = true; break; }
        const int index = edge.first * N + edge.second;
        open[index] = 1;
        ++result.shrink_checks;
        const auto trial = service_exists(d, paths, open, deadline, result);
        if (trial.state == Existence::absent) {
            result.blocking_arcs.erase(std::lower_bound(result.blocking_arcs.begin(),
                result.blocking_arcs.end(), edge));
        } else {
            open[index] = 0;
            if (trial.state == Existence::incomplete) { result.timed_out = true; break; }
        }
    }
    result.shrink_complete = !result.timed_out;
    result.row = {"topology_zero_service_cost", 'G',
        result.courier_cost - result.numerical_margin, {{theta_index, 1}}};
    std::ostringstream key;
    key << "topology_zero_service";
    long double lhs = theta_value;
    for (const auto& [u, v] : result.blocking_arcs) {
        const int variable = layout.airline + u * N + v;
        result.row.coefficients.emplace_back(variable, result.courier_cost);
        lhs += static_cast<long double>(result.courier_cost) * design[variable];
        key << ':' << u << ',' << v;
    }
    result.key = key.str();
    result.violation = static_cast<double>(result.row.rhs - lhs);
    result.valid = true;
    result.violated = result.violation > 1e-7 *
        (1 + result.courier_cost + std::abs(theta_value));
    result.status = result.timed_out ? "certified_before_shrink_timeout" : "certified";
    return finish();
}

} // namespace mlnd
