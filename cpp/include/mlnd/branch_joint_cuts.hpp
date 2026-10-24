#pragma once

#include "mlnd/assignment_support_cuts.hpp"

namespace mlnd {
struct DoubleBranchSupport {
    int first_od = -1, second_od = -1, origin_base = -1, return_base = -1;
    bool complete = false;
    std::size_t next_candidate = 0, candidates_checked = 0;
    std::vector<std::vector<int>> paths;
};

// Only each complete branch domain may certify absence. Partial groups retain
// their enumeration cursor and are completed lazily under later deadlines.
// The Input and initial-segment support must remain unchanged.
struct BranchJointCache {
    int n_od = 0, n_l = 0, n_nodes = 0;
    bool support_complete = false, numerical_domain_safe = false, timed_out = false;
    double seconds = 0, delta_lower = 0, delta_upper = 0;
    std::vector<double> positive_double_floor;
    std::vector<std::vector<int>> initial_indices;
    std::vector<DoubleBranchSupport> double_groups;
    std::size_t completed_groups = 0, candidates_checked = 0;
    Json metadata;
};

struct BranchSupportCertificate {
    int first_od = -1, second_od = -1, origin_base = -1, return_base = -1;
    double positive_flow_floor = 0;
    std::vector<int> triggers;
    std::vector<std::pair<int,int>> blocking_arcs;
    Json metadata;
};

struct BranchJointCostCut {
    RowRecord row;
    std::string key, kind;
    bool valid = false, violated = false;
    double violation = 0;
    std::vector<int> origin_bases, triggers;
    std::vector<std::pair<int,int>> blocking_arcs;
    Json metadata;
};

struct BranchJointSeparation {
    std::vector<BranchJointCostCut> cuts;
    std::vector<BranchSupportCertificate> certificates;
    bool support_complete = false, timed_out = false;
    double seconds = 0;
    std::size_t single_checks = 0, double_checks = 0, numerical_skips = 0;
    std::size_t incomplete_groups = 0, lazy_candidates_checked = 0;
    Json metadata;
};

BranchJointCache build_branch_joint_cache(const Input&, const PathCollection&,
    const AssignmentSupportCache&, double deadline, double floor_threshold = 1e-4);

// Integer-valid direct aggregate K<=2 cuts. Returns violated individual
// double-branch cuts and, when possible, one joint cost cut over distinct
// departure bases, using one certified missing single/double branch per base.
// Joint triggers and blocking arcs are unions; the unassigned courier cost
// is counted once. No master or operational variables are introduced.
BranchJointSeparation separate_branch_joint_cost(const Input&, const PathCollection&,
    const AssignmentSupportCache&, BranchJointCache&, const std::vector<double>& design,
    int theta_index, double theta_value, double deadline, bool double_cuts = true,
    bool joint_cuts = true, std::size_t max_cuts = 16, double floor_threshold = 1e-4);
} // namespace mlnd
