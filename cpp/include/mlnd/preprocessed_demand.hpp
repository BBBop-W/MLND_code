#pragma once

#include "mlnd/paths.hpp"

namespace mlnd {
// Reconstruct fixed single/double task rates from the complete physical path
// domain, scanned without retaining service paths when the catalogue is implicit.
// This deterministic residual-rate heuristic does not optimize
// over facility decisions and does not claim stochastic-policy equivalence.
void preprocess_generalized_demand(Input&, PathCollection&,
    double deadline = std::numeric_limits<double>::infinity());
// Departure and physical route through the first delivery, excluding the
// eventual return base and all continuation-dependent suffix decisions.
std::vector<int> first_leg_key(const InitialSegment&);
Json first_leg_support(const Input&, const PathCollection&);
} // namespace mlnd
