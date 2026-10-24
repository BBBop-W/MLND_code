#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {

// Input-only lower resource/cost allocations. Pair-indexed vectors use
// (order * n_l + departure) * n_l + return, like DesignLayout::assignment.
// No generated column is used to compute an extremum.
struct AnalyticalMasterBounds {
    std::vector<double> unit_cost, unit_saving, slot_time;
    std::vector<bool> pair_possible;
    double courier_cost = 0, operating_floor = 0, savings_per_slot_time = 0;
    bool finite_capacity_savings = true;
};

AnalyticalMasterBounds analytical_master_bounds(const Input&);

// Add globally valid lower bounds to a design-only master. No operational
// variable is introduced: only the design and theta appear in these cuts.
// This does not assert feasibility/optimality of any generated-column RMP.
Json add_master_strengthening(const Input&, const PathCollection&,
                              Formulation& master, int theta,
                              bool resource_bounds = true);

// An input-only monetary-flow relaxation. Each original edge has an affine
// capacity in design variables; the residual graph only selects a partition.
// Recomputing every crossing original edge certifies the resulting inequality
// independently of the numerical accuracy of the max-flow value.
struct CapacitySavingsArc {
    int from = 0, to = 0;
    double constant = 0;
    Coefficients coefficients;
};
struct CapacitySavingsCut {
    RowRecord row;
    std::string partition_key;
    double maxflow = 0, cut_capacity = 0, violation = 0;
    bool violated = false;
};
struct CapacitySavingsNetwork {
    int nodes = 0, source = 0, sink = 0;
    double courier_cost = 0, numerical_margin = 0;
    std::vector<CapacitySavingsArc> arcs;
    // Indexed by (order * L * L + departure * L + return) * H + delivery.
    std::vector<double> unit_saving, slot_time, unit_cost;
    std::vector<bool> eligible;
    Json metadata;
    CapacitySavingsCut separate(const std::vector<double>& design,
                                double theta_value, int theta) const;
};
CapacitySavingsNetwork capacity_savings_network(const Input&);

// External weighted occurrence relaxation; none of these variables enters the
// design master. Group and locker supplies are nonnegative affine forms in x.
// With u_gj >= 0, maximize sum alpha_gj*u_gj subject to group demand and
// weighted locker load limits. Unlike monetary max-flow, this keeps each
// occurrence's own saving and slot-time coefficient.
struct ResourceSavingsOccurrence {
    int group = 0, locker = 0;
    double saving = 0, slot_time = 0;
    // Unit consumption in optional aggregate and OD-specific airline rows.
    // Empty for the legacy locker-only relaxation and custom test fixtures.
    std::vector<int> airline_resources;
};
struct ResourceSavingsModel {
    double courier_cost = 0, numerical_margin = 0;
    std::vector<Coefficients> group_rhs, locker_rhs, airline_rhs;
    std::vector<ResourceSavingsOccurrence> occurrences;
    Json metadata;
};
ResourceSavingsModel resource_savings_model(const Input&, bool airline_bounds = true,
                                           bool tight_airline_bounds = false,
                                           bool od_airline_bounds = false);
// Reuse the input-only allocations when a monetary-flow network has already
// been built from exactly the same Input (never from a generated path pool).
ResourceSavingsModel resource_savings_model(const Input&, const CapacitySavingsNetwork&,
                                          bool airline_bounds = true,
                                          bool tight_airline_bounds = false,
                                          bool od_airline_bounds = false);

struct ResourceSavingsCut {
    RowRecord row;
    std::string key, status;
    double dual_savings = 0, solver_objective = 0;
    double seconds = 0, solver_seconds = 0, max_dual_violation_before = 0;
    double max_dual_violation_after = 0, violation = 0;
    bool valid = false, violated = false, optimal = false, fallback = false;
    std::vector<double> demand_dual, locker_dual, airline_dual;
};

// Persistent dual LP. Its feasible multipliers, independently repaired and
// checked over every input-domain occurrence, give a global cost cut even if
// the auxiliary LP times out. The auxiliary objective is an upper bound on
// achievable savings, never a Benders recourse optimum certificate.
class ResourceSavingsRelaxation {
public:
    explicit ResourceSavingsRelaxation(ResourceSavingsModel, const SolverOptions&);
    explicit ResourceSavingsRelaxation(const Input&, const SolverOptions&);
    ResourceSavingsCut separate(const std::vector<double>& design,
                               double theta_value, int theta, double seconds);
    const Json& metadata() const { return data_.metadata; }
private:
    ResourceSavingsModel data_;
    std::unique_ptr<NativeSolver> solver_;
};

} // namespace mlnd
