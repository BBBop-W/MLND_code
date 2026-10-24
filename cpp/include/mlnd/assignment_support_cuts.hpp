#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {

struct AssignmentSupportGroup {
    int od = -1, origin_base = -1, return_base = -1;
    // Each entry is the sorted, distinct set of global u*N+v arc indices
    // of one physically feasible single-order path.
    std::vector<std::vector<int>> single_paths;
};

// Construct once from complete initial-segment support. The Input must remain
// unchanged while this cache is used; no generated path pool is consulted.
struct AssignmentSupportCache {
    int n_od = 0, n_l = 0, n_nodes = 0;
    bool complete_domain = false, timed_out = false, numerical_domain_safe = false;
    double seconds = 0, effective_delta_upper = 0;
    std::size_t initial_segments_checked = 0, single_paths_checked = 0;
    std::vector<double> positive_single_floor;
    std::vector<AssignmentSupportGroup> groups;
    Json metadata;
};

struct AssignmentSupportCostCut {
    RowRecord row;
    std::string key;
    bool valid = false, violated = false;
    int od = -1, origin_base = -1, return_base = -1;
    double violation = 0, positive_single_floor = 0;
    std::vector<std::pair<int, int>> blocking_arcs;
    Json metadata;
};

struct AssignmentSupportSeparation {
    std::vector<AssignmentSupportCostCut> cuts;
    bool complete_domain = false, timed_out = false;
    std::size_t checks = 0, assigned_groups_checked = 0, numerical_skips = 0;
    std::size_t supported_groups = 0, blocked_groups = 0;
    double seconds = 0;
    Json metadata;
};

// Absolute wall_time() deadlines. An incomplete cache cannot issue a cut.
// Supported domain: direct K=1/K=2 and the aggregate flow policy.
AssignmentSupportCache build_assignment_support_cache(const Input&,
    const PathCollection&, double deadline);

// Integer-valid only. If alpha[r,l,b]=1 and B blocks all single paths for
// (r,l,b), shared rho[l]=0. Nonnegative operating cost therefore implies
// theta + sum_{t,a!=l,b'} c[t]d[t]alpha[t,a,b']
//       + C_courier sum_{e in B} y[e] - C_courier alpha[r,l,b] >= 0.
// A downward numerical margin is applied to the returned row. The positive
// single-flow floor is a conservative coefficient screen, not a solver
// feasibility-tolerance guarantee.
AssignmentSupportSeparation separate_assignment_support_cost(const Input&,
    const AssignmentSupportCache&, const std::vector<double>& design,
    int theta_index, double theta_value, double deadline,
    std::size_t max_cuts = 16, double floor_threshold = 1e-4);

} // namespace mlnd
