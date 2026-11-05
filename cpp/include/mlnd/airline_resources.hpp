#pragma once

#include "mlnd/queue_polymatroid.hpp"

namespace mlnd {

// Canonical integer designs: discard zero-service assignments and unused
// locker/airline assets. These rows preserve an optimum, not every redundant
// original design. They never force a physical return to the departure base.
struct AirlineConnectivityCuts {
    std::vector<RowRecord> rows;
    Json metadata;
};
AirlineConnectivityCuts build_airline_connectivity(const Input&, const PathCollection&);

// Complete-route external savings LP. q[p] is path flow divided by the fixed
// positive task rate, so each task sums to its own v[task,origin]. The
// master receives only verified dual cuts, never these auxiliary variables.
class AirlineResourceRelaxation {
public:
    AirlineResourceRelaxation(const Input&, const PathCollection&, const SolverOptions&,
                              bool balance_relocation = false);
    TaskSavingsCut separate(const std::vector<double>& design,
                            double theta_value, int theta, double seconds);
    QueueResourcePrices queue_prices(const std::vector<double>& dual_prices) const;
    const Json& metadata() const { return metadata_; }
private:
    struct DualColumn {
        double constant = 0;
        bool equality = false;
        Coefficients design_rhs, primal_coefficients;
    };
    int design_count_ = 0, groups_ = 0, bases_ = 0, primal_count_ = 0, path_offset_ = 0,
        empty_offset_ = 0;
    double courier_cost_ = 0, fleet_cost_ = 0, numerical_margin_ = 0;
    std::vector<DualColumn> columns_;
    std::vector<double> primal_objective_;
    std::vector<int> assignment_prices_, base_prices_, branch_prices_, path_branches_;
    std::vector<int> balance_prices_, empty_gate_prices_;
    int lockers_ = 0, nodes_ = 0;
    std::vector<int> locker_prices_, arc_prices_;
    std::vector<std::pair<int,QueueBranchArcPrice>> normalized_gate_prices_;
    std::vector<std::vector<int>> tangent_prices_;
    std::unique_ptr<NativeSolver> solver_;
    Json metadata_;
};

} // namespace mlnd
