#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {

// These rows preserve the optimum and the projection onto operating decisions,
// not every redundant assignment encoding of the original feasible set. In
// the fixed-demand task model, assignments at rho[l]=0 can all be cleared
// without changing any operating variable or objective contribution.
struct PathSupportCut {
    RowRecord row;
    std::string key;
    int head = -1, origin = -1, branch = -1; // branch=-1 is the singleton
    std::vector<std::pair<int, int>> blocking_arcs;
    bool valid = false, violated = false;
    double violation = 0;
    Json metadata;
};

struct PathSupportGroup {
    int head = -1, origin = -1, branch = -1;
    double rate = 0;
    // Sorted distinct global u*N+v arc sets, across ALL return bases.
    std::vector<std::vector<int>> paths;
};

struct PathSupportCache {
    int n_od = 0, n_l = 0, n_nodes = 0;
    bool complete_domain = false, timed_out = false;
    double seconds = 0;
    std::size_t paths_checked = 0;
    std::vector<PathSupportGroup> groups;
    // One alpha[q,l] <= 0 row per statically unsupported task assignment.
    std::vector<PathSupportCut> static_cuts;
    Json metadata;
};

struct PathSupportSeparation {
    std::vector<PathSupportCut> cuts;
    bool complete_domain = false, timed_out = false;
    std::size_t checks = 0, assigned_groups_checked = 0;
    std::size_t supported_groups = 0, blocked_groups = 0;
    double seconds = 0;
    Json metadata;
};

// Absolute wall_time() deadline. Supported model: fixed preprocessed demand,
// task assignment, direct paths, aggregate flow, no first-leg consistency.
// Both paths.complete and paths.support_complete are required; finite row
// support alone is NOT a complete physical catalogue. Incomplete or timed-out
// construction yields no static cuts and cannot subsequently issue any cuts.
// The Input and its task rates must remain unchanged while using this cache.
PathSupportCache build_path_support_cache(const Input&, const PathCollection&,
                                         double deadline);

// Integer-design separation. For a positive task branch whose every physical
// path uses a currently closed arc in C, returns alpha[q,l] <= sum_{e in C} y[e].
// A completed scan certifies C before optional shrinking, so a shrinking
// timeout may return the already-certified larger set. No RMP columns or
// continuous availability variables are used in these rows.
PathSupportSeparation separate_path_support(const Input&, const PathSupportCache&,
    const std::vector<double>& design, double deadline, std::size_t max_cuts = 16);

struct FractionalPathSupportOptions {
    double lp_seconds = .05, violation_tolerance = 1e-6;
    std::size_t max_lp_solves = 64, max_cached_lps = 8;
};

// arc_prices use global u*N+v indices, not master variable indices. Projects
// prices to nonnegative values, repairs the intercept, and scans EVERY cached
// physical path. No cut is returned as valid on an incomplete/expired scan.
PathSupportCut certify_fractional_path_support(const Input&, const PathSupportCache&,
    std::size_t group, const Coefficients& arc_prices, double intercept,
    const std::vector<double>& design, double deadline);

// The input/cache must outlive this object and remain unchanged. Uses an
// external normalized path-support LP, with bounded LRU model caching. The
// returned rows are valid for canonical integer designs and their convex hull,
// not redundant rows for every original fractional recourse encoding.
class FractionalPathSupportSeparator {
public:
    FractionalPathSupportSeparator(const Input&, const PathSupportCache&,
        const SolverOptions&, FractionalPathSupportOptions = {});
    ~FractionalPathSupportSeparator();
    PathSupportSeparation separate(const std::vector<double>& design,
        double deadline, std::size_t max_cuts = 16);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlnd
