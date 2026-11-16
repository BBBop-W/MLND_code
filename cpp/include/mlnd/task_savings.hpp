#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {

// Fixed generalized-task allocations over the COMPLETE physical path domain.
// Only group minima/counts are retained; the path catalogue may be implicit.
// Each (task, departure) coefficient includes only that independent task,
// minimizing over all possible return bases before its own positive part.
// No current design or restricted column pool determines these constants.
struct TaskSavingsData {
    // `heads` is the number of assignment entities (independent tasks).
    int heads = 0, bases = 0, lockers = 0, levels = 0;
    double courier_cost = 0, fleet_cost = 0, numerical_margin = 0;
    std::vector<double> saving, signed_saving, chord_coefficient, total_load;
    std::vector<std::vector<double>> locker_load;
    std::vector<bool> supported;
    std::vector<std::size_t> path_counts;
    Json metadata;
};

TaskSavingsData task_savings_data(const Input&, const PathCollection&);
Json add_task_savings_bound(const TaskSavingsData&, Formulation&, int theta);

struct TaskSavingsCut {
    RowRecord row;
    std::string key, status;
    bool valid = false, violated = false, optimal = false, fallback = false;
    double violation = 0, dual_savings = 0, seconds = 0, solver_seconds = 0;
    double solver_objective = 0;
    double max_dual_violation_before = 0, max_dual_violation_after = 0;
    std::vector<double> dual_prices;
    Json certificate = Json::object();
};

// External LP retains shared base availability, task assignment, task-load
// minima, and supporting lines of idle = rho/(1-rho). Its explicitly checked
// dual gives global linear cost cuts, with no new variables in the master.
class TaskResourceSavingsRelaxation {
public:
    TaskResourceSavingsRelaxation(const Input&, TaskSavingsData, const SolverOptions&);
    TaskSavingsCut separate(const std::vector<double>& design,
                            double theta_value, int theta, double seconds);
    const Json& metadata() const { return data_.metadata; }
private:
    struct DualColumn {
        double constant = 0;
        Coefficients design_rhs, primal_coefficients;
    };
    TaskSavingsData data_;
    int design_count_ = 0, primal_count_ = 0;
    std::vector<DualColumn> columns_;
    std::vector<double> primal_objective_;
    std::vector<int> assignment_prices_, base_prices_;
    std::vector<std::vector<int>> tangent_prices_;
    std::unique_ptr<NativeSolver> solver_;
};

} // namespace mlnd
