#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {
// A primal heuristic only: optimize design and operations jointly on the
// registered columns. Neither the restricted MIP bound nor a neighbourhood
// bound is a lower bound for the complete path problem, so neither is returned.
// deadline is an absolute wall_time() deadline and includes model construction.
// A nonnegative radius restricts facility/capacity/assignment Hamming distance
// from the validated incumbent; airlines remain free to support rerouting.
// The returned wrapper contains a packaged `solution`, objective, validated,
// improved, source, status, diagnostics and timings. A feasible all-courier
// solution is always retained; invalid supplied incumbents are never accepted.
// When neighborhood_cuts is enabled, design_pool also contains at most
// heuristic_pool_size distinct validated strategic designs found by this MIP
// solve (including a final solver design that does not improve the incumbent).
// Pool objectives are feasible primal costs on registered columns, not bounds;
// callers must run independently certified full-path recourse before adding
// any Benders cut. Pool extraction never launches a populate optimization.
Json solve_restricted_heuristic(const Input&, const PathCollection&,
                                const SolverOptions&, double deadline,
                                const Json* incumbent = nullptr,
                                int local_branching_radius = -1);
} // namespace mlnd
