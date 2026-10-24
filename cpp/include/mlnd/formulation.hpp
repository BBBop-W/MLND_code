#pragma once

#include "mlnd/linear_model.hpp"
#include "mlnd/paths.hpp"

namespace mlnd {
// Redundant for integer strategic designs; stronger fractional relaxations.
// Kept opt-in so existing continuous certificates remain reproducible.
struct FormulationStrengthening {
    bool assignment_incident_clique = false;
    bool shared_rho_joint = false;
    bool queue_perspective = false;
    bool enabled() const { return assignment_incident_clique || shared_rho_joint || queue_perspective; }
};
inline FormulationStrengthening formulation_strengthening(const SolverOptions& options) {
    return {options.assignment_incident_clique, options.shared_rho_joint, options.queue_perspective};
}
struct DesignLayout {
    int facility = 0, capacity = 0, assignment = 0, airline = 0, count = 0;
    explicit DesignLayout(const Input&);
};
struct Formulation {
    LinearModel matrix;
    DesignLayout layout;
    std::vector<double> opening_costs;
    std::vector<int> design_vars, design_fix_rows;
    std::vector<int> rho, idle, head, courier, flow, empty_flow, eta;
    int fleet = -1;
    std::vector<std::vector<int>> lift_groups;
    std::vector<int> segment_group, single_rows, double_rows, executed_head_rows;
    std::vector<int> order_rows, balance_rows, capacity_rows, link_rows;
    int fleet_row = -1;
    std::vector<int> service_indices, empty_indices;
    // Numeric continuation coefficients are valid only for the supplied
    // integer assignment. The original lifted formulation remains the default.
    bool conditional_demand = false;
    bool tight_airline_bounds = false;
    FormulationStrengthening strengthening;
    // Safe upper bounds implied by each finite idle bound when perspective
    // strengthening is active; otherwise the original upper bound is one.
    std::vector<double> rho_upper;
    Json strengthening_metadata;
    std::vector<double> segment_eta_factor;
    explicit Formulation(const Input& d) : layout(d) {}
};
// Design variables are always the first layout.count variables in this order.
Formulation design_model(const Input&, bool integer = true,
                         const FormulationStrengthening& = {});
Formulation build_formulation(const Input&, const PathCollection&,
                             const std::vector<double>* fixed_design = nullptr,
                             const std::vector<int>* service_indices = nullptr,
                             const std::vector<int>* empty_indices = nullptr,
                             bool relax_fixed_design = false,
                             bool conditional_demand = false,
                             bool tight_airline_bounds = false,
                             const FormulationStrengthening& = {});
Coefficients path_column(const Formulation&, const ServicePath&);
Coefficients path_column(const Formulation&, const EmptyPath&);
void add_columns(Formulation&, NativeSolver&, const Input&, const PathCollection&,
                 const std::vector<int>& service, const std::vector<int>& empty);
Json solution_json(const Input&, const PathCollection&, const Formulation&,
                   const std::vector<double>& values);
Json all_courier_result(const Input&, const PathCollection&);
// Repair only inventory-cone numerical infeasibility by purchasing additional
// idle/fleet capacity. All other decisions are preserved; failure is atomic.
Json repair_inventory_primal(const Formulation&, std::vector<double>& values);
Json solve_full(const Input&, const PathCollection&, const SolverOptions&,
                const std::vector<double>* fixed_design = nullptr,
                const Json* start_solution = nullptr);
std::vector<double> solution_design(const Input&, const Json& solution);
} // namespace mlnd
