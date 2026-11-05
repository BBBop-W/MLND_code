#pragma once
#include <cmath>
#include <stdexcept>

#include "mlnd/linear_model.hpp"
#include "mlnd/paths.hpp"

namespace mlnd {
// Redundant for integer strategic designs; stronger fractional relaxations.
// Kept opt-in so existing continuous certificates remain reproducible.
struct FormulationStrengthening {
    bool assignment_incident_clique = false;
    bool shared_rho_joint = false;
    bool queue_perspective = false;
    // Prepared once from the complete physical catalogue and fixed task rates.
    // Never compute these maxima from the current RMP or the current design.
    std::vector<double> task_airline_upper;
    // Exact reformulations, independent of the fractional strengthening flags.
    bool eliminate_zero_task_columns = false;
    bool compact_task_heads = false;
    bool enabled() const { return assignment_incident_clique || shared_rho_joint || queue_perspective || !task_airline_upper.empty(); }
};
inline FormulationStrengthening formulation_strengthening(const SolverOptions& options) {
    return {options.assignment_incident_clique, options.shared_rho_joint, options.queue_perspective,
            options.task_airline_upper,options.eliminate_zero_task_columns,options.compact_task_heads};
}
struct TaskAirlineBounds {
    std::vector<double> upper;
    Json metadata;
};
// Independent tasks may choose different departures, including tasks sharing
// their first OD. Bound arc e by sum_q rate_q max_l max_{p in P_ql} count_e(p).
// The direct base-to-base relocation bounds remain unchanged.
TaskAirlineBounds build_task_airline_bounds(const Input&, const PathCollection&);
struct DesignLayout {
    int facility = 0, capacity = 0, assignment = 0, airline = 0, count = 0;
    int n_l = 0, assignment_destinations = 0;
    int assignment_index(int entity, int l, int m = 0) const {
        return assignment + (entity * n_l + l) * assignment_destinations +
            (assignment_destinations == 1 ? 0 : m);
    }
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
    // Dense task ID x departure rows, including tasks with no physical path.
    std::vector<int> task_rows;
    // In compact mode executed_head_rows maps each segment to its (head,origin)
    // total-flow equality. Path coefficients remain -1 in either representation.
    std::vector<int> order_rows, balance_rows, capacity_rows, link_rows;
    int fleet_row = -1;
    std::vector<int> service_indices, empty_indices;
    // Numeric continuation coefficients are valid only for the supplied
    // integer assignment. The original lifted formulation remains the default.
    bool conditional_demand = false;
    bool tight_airline_bounds = false;
    bool zero_task_columns_eliminated = false, compact_task_heads = false;
    double service_flow_upper = 0;
    int task_rate_branches = 0;
    // first*(R+1)+second+1; second=-1 denotes the single branch. Independent
    // of registered path IDs, including not-yet-registered pricing candidates.
    std::vector<double> fixed_task_rates;
    Json reduction_metadata;
    FormulationStrengthening strengthening;
    // Safe upper bounds implied by each finite idle bound when perspective
    // strengthening is active; otherwise the original upper bound is one.
    std::vector<double> rho_upper;
    Json strengthening_metadata;
    std::vector<double> segment_eta_factor;
    explicit Formulation(const Input& d) : layout(d) {}
};
inline double service_path_upper_bound(const Formulation& f,const ServicePath& p) {
    if(f.zero_task_columns_eliminated) {
        if(p.first_od<0 || p.second_od < -1 || p.second_od>=f.task_rate_branches-1)
            throw std::invalid_argument("Invalid task branch in effective path bound");
        const auto index=static_cast<std::size_t>(p.first_od)*f.task_rate_branches+p.second_od+1;
        if(f.fixed_task_rates.at(index)==0)return 0;
    }
    return f.service_flow_upper;
}
inline bool service_path_enabled(const Formulation& f,const ServicePath& p) {
    return service_path_upper_bound(f,p)>0;
}
// Restore the original public segment-indexed head vector after exact variable
// elimination; physical path IDs and output schemas remain unchanged.
std::vector<double> recover_task_heads(const Formulation&,const PathCollection&,
                                       const std::vector<double>& values);
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
// Canonical emission order is shared by sparse model construction and exact
// reduced-cost evaluation. The visitor avoids one temporary vector per path.
template<class Emit>
void for_each_path_coefficient(const Formulation& f,const ServicePath& p,Emit&& emit) {
    const int R=static_cast<int>(f.order_rows.size());
    const int N=static_cast<int>(std::sqrt(f.link_rows.size()));
    const int j=p.segment_id;
    const int branch=p.second_od<0?f.single_rows.at(j):f.double_rows.at(j*R+p.second_od);
    if(branch<0)throw std::invalid_argument("Missing service branch row");
    emit(branch,-1);
    if(f.executed_head_rows.at(j)>=0)emit(f.executed_head_rows[j],-1);
    for(auto [r,count]:p.served_counts)emit(f.order_rows.at(r),count);
    for(auto [i,time]:p.slot_times)emit(f.capacity_rows.at(i),time);
    if(p.b[0]!=p.b[1]) {emit(f.balance_rows.at(p.b[0]),1);emit(f.balance_rows.at(p.b[1]),-1);}
    emit(f.fleet_row,p.time);
    for(auto [u,v,count]:p.arc_counts)emit(f.link_rows.at(u*N+v),count);
}
template<class Emit>
void for_each_path_coefficient(const Formulation& f,const EmptyPath& p,Emit&& emit) {
    const int N=static_cast<int>(std::sqrt(f.link_rows.size()));
    if(p.b[0]!=p.b[1]) {emit(f.balance_rows.at(p.b[0]),1);emit(f.balance_rows.at(p.b[1]),-1);}
    emit(f.fleet_row,p.time);
    for(auto [u,v,count]:p.arc_counts)emit(f.link_rows.at(u*N+v),count);
}
Coefficients path_column(const Formulation&, const ServicePath&);
Coefficients path_column(const Formulation&, const EmptyPath&);
void add_columns(Formulation&, NativeSolver&, const Input&, const PathCollection&,
                 const std::vector<int>& service, const std::vector<int>& empty, bool batch = false);
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
