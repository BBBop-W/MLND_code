#include "mlnd/assignment_support_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mlnd {
namespace {
void validate_domain(const Input& d) {
    if (d.route_mode != "direct" || d.has_candidate_routes ||
        d.max_orders < 1 || d.max_orders > 2 || d.flow_policy != "aggregate")
        throw std::invalid_argument("Assignment support cuts require direct K=1/K=2 aggregate flow");
    auto nonnegative = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(), [](double x) {
            return std::isfinite(x) && x >= 0;
        });
    };
    if (!nonnegative(d.demand) || !nonnegative(d.ground_cost) ||
        !nonnegative(d.first_cost.values) || !nonnegative(d.terminal_cost.values) ||
        !nonnegative({d.fleet_cost, d.flight_cost, d.waiting_cost, d.effective_delta()}))
        throw std::invalid_argument("Assignment support cuts require nonnegative operating data");
    if (!std::all_of(d.kappa.values.begin(), d.kappa.values.end(), [](double x) {
            return std::isfinite(x) && x >= 0 && x <= 1;
        }))
        throw std::invalid_argument("Assignment support cuts require kappa in [0,1]");
}

void numerical_floors(const Input& d, AssignmentSupportCache& cache) {
    const int R = d.n_od;
    const double eps = std::numeric_limits<double>::epsilon();
    const double delta = d.effective_delta();
    const long double total = std::accumulate(d.demand.begin(), d.demand.end(), 0.0L);
    cache.positive_single_floor.assign(R, 0);
    if (!std::isfinite(static_cast<double>(total)) ||
        !std::isfinite(static_cast<double>(delta * total)) || R * eps >= 1) return;
    // The lifted model stores p_s=1-exp(-delta*d_s). Cover its upward
    // rounding as well as positive summation error in the conditional model.
    const double gamma = R * eps / (1 - R * eps);
    double delta_upper = delta * (1 + gamma);
    for (double demand : d.demand) {
        if (demand == 0) continue;
        const double p = -std::expm1(-delta * demand);
        if (!std::isfinite(p) || p < 0 || p >= 1) return;
        delta_upper = std::max(delta_upper, -std::log1p(-p) / demand);
    }
    if (!std::isfinite(delta_upper)) return;
    cache.effective_delta_upper = delta_upper;
    const double margin = 64 * (R + 1) * eps;
    for (int r = 0; r < R; ++r) {
        long double incoming = 0, outgoing = 0;
        for (int s = 0; s < R; ++s) {
            incoming += static_cast<long double>(d.kappa(s, r)) * d.demand[s];
            outgoing += static_cast<long double>(d.kappa(r, s)) * d.demand[s];
        }
        const long double head_gap = std::max(0.0L,
            std::exp(-static_cast<long double>(delta_upper) * incoming) - margin);
        const long double single_gap = std::max(0.0L,
            std::exp(-static_cast<long double>(delta_upper) * outgoing) - margin);
        cache.positive_single_floor[r] = static_cast<double>(d.demand[r] * head_gap * single_gap);
    }
    cache.numerical_domain_safe = true;
}

// The full union is certified before this optional shrinking starts. Every
// successful removal preserves coverage, so a deadline retains a valid set.
std::vector<int> blocking_set(const std::vector<std::vector<int>>& closed_choices,
        int arc_count, double deadline) {
    std::vector<std::vector<int>> covered_by(arc_count);
    std::vector<int> counts(arc_count, 0), full;
    for (std::size_t p = 0; p < closed_choices.size(); ++p)
        for (int e : closed_choices[p]) {
            if (counts[e]++ == 0) full.push_back(e);
            covered_by[e].push_back(static_cast<int>(p));
        }
    std::sort(full.begin(), full.end());
    if (wall_time() >= deadline) return full;
    std::vector<unsigned char> uncovered(closed_choices.size(), 1), chosen(arc_count, 0);
    std::size_t remaining = closed_choices.size();
    while (remaining > 0) {
        if (wall_time() >= deadline) return full;
        const auto best = std::max_element(counts.begin(), counts.end());
        if (best == counts.end() || *best == 0)
            throw std::logic_error("Uncovered single path in assignment support certificate");
        const int e = static_cast<int>(best - counts.begin());
        chosen[e] = 1;
        for (int p : covered_by[e]) if (uncovered[p]) {
            uncovered[p] = 0; --remaining;
            for (int other : closed_choices[p]) --counts[other];
        }
    }
    std::vector<int> hits(closed_choices.size(), 0);
    for (std::size_t p = 0; p < closed_choices.size(); ++p)
        for (int e : closed_choices[p]) hits[p] += chosen[e];
    for (auto it = full.rbegin(); it != full.rend(); ++it) {
        if (wall_time() >= deadline) break;
        const int e = *it;
        if (!chosen[e]) continue;
        const bool redundant = std::all_of(covered_by[e].begin(), covered_by[e].end(),
            [&](int p) { return hits[p] >= 2; });
        if (redundant) {
            chosen[e] = 0;
            for (int p : covered_by[e]) --hits[p];
        }
    }
    std::vector<int> result;
    for (int e : full) if (chosen[e]) result.push_back(e);
    return result;
}
} // namespace

AssignmentSupportCache build_assignment_support_cache(const Input& d,
        const PathCollection& paths, double deadline) {
    const double started = wall_time();
    validate_domain(d);
    if (d.origin_assignment()) throw std::invalid_argument("Assignment-support certificates require base-pair assignment (K=2)");
    if (std::isnan(deadline)) throw std::invalid_argument("Invalid assignment support deadline");
    AssignmentSupportCache cache;
    cache.n_od = d.n_od; cache.n_l = d.n_l; cache.n_nodes = d.n_nodes();
    numerical_floors(d, cache);
    const int R = d.n_od, L = d.n_l, N = d.n_nodes();
    cache.groups.resize(static_cast<std::size_t>(R) * L * L);
    for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) for (int b = 0; b < L; ++b) {
        auto& g = cache.groups[(r * L + l) * L + b];
        g.od = r; g.origin_base = l; g.return_base = b;
    }
    auto finish = [&] {
        cache.seconds = wall_time() - started;
        std::size_t singles = 0;
        for (const auto& g : cache.groups) singles += g.single_paths.size();
        cache.metadata = {{"complete_domain", cache.complete_domain},
            {"timed_out", cache.timed_out}, {"seconds", cache.seconds},
            {"initial_segments_checked", cache.initial_segments_checked},
            {"single_paths_checked", cache.single_paths_checked}, {"cached_single_paths", singles},
            {"groups", cache.groups.size()}, {"numerical_domain_safe", cache.numerical_domain_safe},
            {"effective_delta_upper", cache.effective_delta_upper},
            {"positive_single_floor", cache.positive_single_floor},
            {"source", "complete_initial_segments_and_authoritative_single_path_checks"}};
        return cache;
    };
    if (!paths.support_complete) return finish();
    for (const auto& segment : paths.initial_segments) {
        if (wall_time() >= deadline) { cache.timed_out = true; return finish(); }
        ++cache.initial_segments_checked;
        if (segment.first_od < 0 || segment.first_od >= R || segment.b[0] < 0 ||
            segment.b[0] >= L || segment.b[1] < 0 || segment.b[1] >= L)
            throw std::invalid_argument("Invalid assignment support segment index");
        ServicePath path;
        ++cache.single_paths_checked;
        if (!build_service_path(d, segment, nullptr, path)) continue;
        std::vector<int> arcs;
        for (const auto& [u, v, count] : path.arc_counts) {
            if (count <= 0 || u < 0 || u >= N || v < 0 || v >= N ||
                (u < L && v < L) || !d.allowed_links(u, v))
                throw std::logic_error("Invalid direct single path in assignment support cache");
            arcs.push_back(u * N + v);
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
        cache.groups[(segment.first_od * L + segment.b[0]) * L + segment.b[1]]
            .single_paths.push_back(std::move(arcs));
    }
    for (auto& g : cache.groups) {
        if (wall_time() >= deadline) { cache.timed_out = true; return finish(); }
        std::sort(g.single_paths.begin(), g.single_paths.end());
        g.single_paths.erase(std::unique(g.single_paths.begin(), g.single_paths.end()), g.single_paths.end());
    }
    cache.complete_domain = true;
    return finish();
}

AssignmentSupportSeparation separate_assignment_support_cost(const Input& d,
        const AssignmentSupportCache& cache, const std::vector<double>& design,
        int theta_index, double theta_value, double deadline,
        std::size_t max_cuts, double floor_threshold) {
    const double started = wall_time();
    validate_domain(d);
    if (d.origin_assignment()) throw std::invalid_argument("Assignment-support certificates require base-pair assignment (K=2)");
    const DesignLayout layout(d);
    const int L = d.n_l, R = d.n_od, N = d.n_nodes();
    if (cache.n_l != L || cache.n_od != R || cache.n_nodes != N ||
        design.size() < static_cast<std::size_t>(layout.count) || theta_index < layout.count ||
        !std::isfinite(theta_value) || std::isnan(deadline) ||
        !std::isfinite(floor_threshold) || floor_threshold < 0)
        throw std::invalid_argument("Invalid assignment support separation arguments");
    for (int j = 0; j < layout.count; ++j)
        if (!std::isfinite(design[j]) || design[j] < -1e-6 || design[j] > 1 + 1e-6 ||
            std::abs(design[j] - std::round(design[j])) > 1e-6)
            throw std::invalid_argument("Assignment support cuts require a binary design");
    const long double baseline = std::inner_product(d.demand.begin(), d.demand.end(),
                                                   d.ground_cost.begin(), 0.0L);
    const double courier = static_cast<double>(baseline);
    if (!std::isfinite(courier)) throw std::invalid_argument("Nonfinite assignment support courier cost");
    const double margin = 1e-8 * (1 + courier);
    AssignmentSupportSeparation report;
    report.complete_domain = cache.complete_domain;
    auto finish = [&] {
        report.timed_out = report.timed_out || wall_time() >= deadline;
        report.seconds = wall_time() - started;
        report.metadata = {{"complete_domain", report.complete_domain}, {"timed_out", report.timed_out},
            {"seconds", report.seconds}, {"checks", report.checks},
            {"assigned_groups_checked", report.assigned_groups_checked},
            {"numerical_skips", report.numerical_skips}, {"supported_groups", report.supported_groups},
            {"blocked_groups", report.blocked_groups}, {"cuts", report.cuts.size()},
            {"floor_threshold", floor_threshold}, {"integer_valid_only", true},
            {"master_added_variables", 0}};
        return report;
    };
    if (!cache.complete_domain || max_cuts == 0 || courier <= margin) return finish();
    for (const auto& group : cache.groups) {
        if (wall_time() >= deadline) { report.timed_out = true; break; }
        const int r = group.od, l = group.origin_base, b = group.return_base;
        const int trigger = layout.assignment + (r * L + l) * L + b;
        if (design[trigger] < .5 || d.demand[r] == 0) continue;
        ++report.assigned_groups_checked;
        if (!cache.numerical_domain_safe || cache.positive_single_floor[r] <= 0 ||
            cache.positive_single_floor[r] < floor_threshold) {
            ++report.numerical_skips; continue;
        }
        std::vector<std::vector<int>> closed_choices;
        bool supported = false, scan_complete = true;
        for (const auto& path : group.single_paths) {
            if (wall_time() >= deadline) { scan_complete = false; report.timed_out = true; break; }
            ++report.checks;
            std::vector<int> closed;
            for (int e : path) if (design[layout.airline + e] < .5) closed.push_back(e);
            if (closed.empty()) { supported = true; break; }
            closed_choices.push_back(std::move(closed));
        }
        if (!scan_complete) break;
        if (supported) { ++report.supported_groups; continue; }
        ++report.blocked_groups;
        const auto block = blocking_set(closed_choices, N * N, deadline);
        AssignmentSupportCostCut cut;
        cut.od = r; cut.origin_base = l; cut.return_base = b;
        cut.positive_single_floor = cache.positive_single_floor[r];
        cut.row = {"assignment_support_cost", 'G', -margin, {{theta_index, 1}, {trigger, -courier}}};
        std::ostringstream key;
        key << "assignment_support:" << r << ':' << l << ':' << b;
        for (int t = 0; t < R; ++t) for (int a = 0; a < L; ++a) if (a != l)
            for (int other_b = 0; other_b < L; ++other_b) {
                const double coefficient = d.demand[t] * d.ground_cost[t];
                if (coefficient > 0) cut.row.coefficients.emplace_back(
                    layout.assignment + (t * L + a) * L + other_b, coefficient);
            }
        for (int e : block) {
            cut.row.coefficients.emplace_back(layout.airline + e, courier);
            cut.blocking_arcs.emplace_back(e / N, e % N);
            key << ':' << e / N << ',' << e % N;
        }
        cut.key = key.str();
        long double lhs = 0;
        for (const auto& [index, coefficient] : cut.row.coefficients)
            lhs += static_cast<long double>(coefficient) *
                (index == theta_index ? theta_value : design[index]);
        cut.violation = static_cast<double>(cut.row.rhs - lhs);
        cut.valid = true;
        cut.violated = cut.violation > 1e-7 * (1 + courier + std::abs(theta_value));
        cut.metadata = {{"key", cut.key}, {"valid", true}, {"violated", cut.violated},
            {"od", r}, {"origin_base", l}, {"return_base", b}, {"pair", {l, b}},
            {"blocking_arcs", cut.blocking_arcs}, {"blocking_count", block.size()},
            {"single_paths_checked", group.single_paths.size()}, {"complete_domain", true},
            {"violation", cut.violation}, {"positive_single_floor", cut.positive_single_floor},
            {"courier_cost", courier}, {"numerical_margin", margin},
            {"integer_valid_only", true}, {"other_assignments_fixed", false}};
        if (cut.violated) report.cuts.push_back(std::move(cut));
        if (report.cuts.size() >= max_cuts) break;
    }
    return finish();
}
} // namespace mlnd
