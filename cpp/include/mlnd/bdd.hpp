#pragma once

#include "mlnd/formulation.hpp"
#include <memory>
#include <optional>

namespace mlnd {
// This diagnostic changes the integrality of copied strategic variables only.
// FacilityCapacity is a relaxation of the integer Lagrangian subproblem; it is
// unrelated to scenario retention / partial Benders for stochastic programs.
enum class BddIntegrality { Continuous, FacilityCapacity, AllDesign };
const char* bdd_integrality_name(BddIntegrality);

struct BddAnchor {
    std::string name;
    std::vector<double> design, coefficients;
    double intercept = 0, operating_objective = 0;
    bool pricing_complete = false;
};
BddAnchor bdd_anchor_from_json(const Json&);

// Explicit full-path diagnostic only. The caller enumerates once and charges
// that work to its total budget. Incomplete path pools are rejected. The common
// mathematical matrix and three Concert models are built once, then reused as
// multipliers change; construction costs are exposed in setup_metadata().
// No production algorithm instantiates this class automatically.
class BddFullDomain {
public:
    BddFullDomain(const Input&, const PathCollection&, const SolverOptions&);
    ~BddFullDomain();
    BddFullDomain(const BddFullDomain&) = delete;
    BddFullDomain& operator=(const BddFullDomain&) = delete;
    Json setup_metadata() const;
    // seconds includes objective update, optimize and validation. As with the
    // other native diagnostics, deadlines are cooperative rather than OS-hard.
    Json evaluate(const BddAnchor&, BddIntegrality, double seconds,
                  double improvement_relative_tolerance = 1e-6);
    // Arbitrary multiplier oracle for phi(beta) = min {q(v) - beta^T z}.
    // Unlike evaluate(), no ordinary-cut anchor or pricing claim is supplied.
    // A validated global bound gives a safe cut; a validated feasible inner
    // point gives an upper affine support plane for multiplier optimization,
    // with a conservative upward intercept margin. Its domain is the same
    // integrality mode: a relaxed point does not upper-bound AllDesign phi.
    // The returned plane is an inner-value upper bound, never a Benders cut.
    Json evaluate_multiplier(const std::vector<double>& coefficients,
                             BddIntegrality, double seconds);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compare the same compact continuous master before/after accepted cuts.
// It contains original strategic constraints, fixed costs, theta, the current
// static analytical strengthening, and all supplied ordinary cuts. No full
// operational variables are placed in this comparison master. An optional
// independently certified TOTAL-cost floor adds f*z + theta >= floor to every
// compared model; callers are responsible for matching its input provenance.
Json compare_bdd_root_bounds(const Input&, const PathCollection&,
    const std::vector<BddAnchor>& ordinary_anchors, const Json& evaluation_records,
    const SolverOptions&, double seconds,
    std::optional<double> certified_total_cost_floor = std::nullopt);
} // namespace mlnd
