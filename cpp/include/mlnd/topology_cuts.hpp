#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {

// Integer-valid operating-cost cut. A certified blocking set E intersects
// every physically admissible service path, independently of assignments.
// theta + C_courier sum_{e in E} y_e >= C_courier (with a downward margin).
// These cuts are not certificates for the original fractional formulation.
struct TopologyCostCut {
    RowRecord row;
    std::string key, status, certificate_stage;
    bool valid = false, violated = false, zero_service_certified = false;
    bool timed_out = false, shrink_complete = false;
    double seconds = 0, violation = 0, courier_cost = 0, numerical_margin = 0;
    std::size_t initial_closed_count = 0, blocking_count = 0;
    std::size_t oracle_calls = 0, initial_segments_checked = 0;
    std::size_t single_paths_checked = 0, double_paths_checked = 0;
    std::size_t continuation_candidates_checked = 0, shrink_checks = 0;
    // Global node pairs, sorted deterministically. Base-to-base empty arcs
    // never belong to this set.
    std::vector<std::pair<int, int>> blocking_arcs;
    Json metadata;
};

// deadline is an absolute wall_time() deadline. Only integer designs are
// accepted; extra entries after DesignLayout::count are ignored. Physical
// absence requires complete initial-segment support, never a restricted pool.
// If shrinking times out, the last fully certified blocking set is retained;
// an incomplete initial scan never returns a cut.
TopologyCostCut separate_topology_cost(const Input&, const PathCollection&,
    const std::vector<double>& design, int theta_index, double theta_value,
    double deadline);

} // namespace mlnd
