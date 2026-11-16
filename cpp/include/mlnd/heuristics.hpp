#pragma once

#include "mlnd/formulation.hpp"
#include <functional>
#include <limits>

namespace mlnd {
// A temporary primal neighborhood. In preprocessed models, head IDs are task
// IDs q. Every listed task releases its whole
// assignment group; listed lockers release their facility and all levels.
// Listed bases and airlines may change. All other strategic coordinates are
// fixed to the independently validated incumbent. Operations remain free.
struct CoupledPrimalNeighborhood {
    std::vector<int> heads, lockers, bases;
    std::vector<std::pair<int,int>> airlines;
};
// Temporary restrictions for a primal-only exploration. The union of head
// and locker groups must change at least once; an empty union adds no distance
// row. Each head's assignment and each locker's capacity choice count once.
// Closed assets are constraints, not changes to the global model. The objective
// ceiling permits worse feasible neighbors without degrading the retained best.
struct PrimalRepairRestrictions {
    std::vector<int> change_heads, change_lockers, closed_bases, closed_lockers;
    std::vector<std::pair<int,int>> closed_airlines;
    // Local support loss may disable one origin without removing any task row
    // or excluding the other origin / all-courier head state.
    std::vector<std::pair<int,int>> forbidden_origins;
    bool grouped_radius = false;
    double objective_upper = std::numeric_limits<double>::infinity();
    double validation_reserve_seconds = .5;
    // Called only after original physical and temporary-model validation,
    // before optional pool extraction. The caller owns best/checkpoint policy.
    std::function<void(const Json&)> retain_validated;
};
// Add two temporary distance rows. A reassignment or capacity-level change
// counts once, including activation/deactivation; base and airline bits do not
// enter either metric. Empty group lists add vacuous rows. No global cut is made.
Json add_grouped_primal_radius(const Input&, Formulation&, const std::vector<double>& center,
                              const std::vector<int>& heads, const std::vector<int>& lockers,
                              int head_radius, int capacity_radius);
// A primal heuristic only: optimize design and operations jointly on the
// registered columns. Neither the restricted MIP bound nor a neighbourhood
// bound is a lower bound for the complete path problem, so neither is returned.
// deadline is an absolute wall_time() deadline and includes model construction.
// A nonnegative radius activates local branching. With grouped_local_branching
// positive radii select the separate configured head/capacity group limits.
// Radius zero always retains the original zero bit-distance semantics; otherwise
// the legacy mode restricts facility/capacity/assignment Hamming distance.
// The returned wrapper contains a packaged `solution`, objective, validated,
// improved, source, status, diagnostics and timings. A feasible all-courier
// solution is always retained; invalid supplied incumbents are never accepted.
// When neighborhood_cuts is enabled, design_pool also contains at most
// heuristic_pool_size distinct validated strategic designs found by this MIP
// solve (including a final solver design that does not improve the incumbent).
// Pool objectives are feasible primal costs on registered columns, not bounds;
// callers must run independently certified full-path recourse before adding
// any Benders cut. Pool extraction never launches a populate optimization.
// Optional index lists restrict only this primal search. Null keeps every
// registered column; an empty list selects none. Every strictly positive
// incumbent column is added to the lists before validating the warm start.
// fix_assignment pins only the validated incumbent's assignment variables;
// facilities, capacity levels, airlines and all operating variables stay free.
// fix_infrastructure is the complementary neighborhood: pin only facility and
// capacity variables, allowing reassignment and rerouting. The two fixing
// modes are mutually exclusive. Both require a validated incumbent and affect
// only this primal search, never the master or the full recourse domain.
// coupled is mutually exclusive with the other fixing modes and radius.
// It requires a validated incumbent, retains every positive incumbent path,
// and never modifies a caller-owned formulation.
// With restrictions, `candidate_validated` and optional `candidate` describe
// only a point satisfying the temporary neighborhood, including exploration.
// `solution` remains the best global feasible fallback and can lie outside it.
Json solve_restricted_heuristic(const Input&, const PathCollection&,
                                const SolverOptions&, double deadline,
                                const Json* incumbent = nullptr,
                                int local_branching_radius = -1,
                                const std::vector<int>* service_indices = nullptr,
                                const std::vector<int>* empty_indices = nullptr,
                                bool fix_assignment = false,
                                bool fix_infrastructure = false,
                                const CoupledPrimalNeighborhood* coupled = nullptr,
                                const PrimalRepairRestrictions* restrictions = nullptr);
} // namespace mlnd
