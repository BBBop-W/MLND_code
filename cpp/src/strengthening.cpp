#include "mlnd/strengthening.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mlnd {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();

// Round towards a weaker inequality, including a small arithmetic guard for
// path accounting performed in a different order. Zero remains exact.
double lower_nonnegative(double value) {
    return value > 0 ? std::max(0.0, std::nextafter(value - 1e-12 * (1 + value), -infinity)) : 0;
}
double upper_nonnegative(double value) {
    return value > 0 ? std::nextafter(value + 1e-12 * (1 + value), infinity) : 0;
}
// Necessary conditions for ANY loaded occurrence. A second order need not
// satisfy the first order's base access or loaded-flight >= delta condition.
// The input-only superset is shared by allocations and OD airline gates.
bool eligible_loaded_occurrence(const Input& d,int r,int i,int j) {
    if (i==j || !d.allowed_links(d.n_l+i,d.n_l+j)) return false;
    const double loaded=d.flight_times(d.n_l+i,d.n_l+j);
    return loaded<=d.operating_limit+1e-12 &&
        loaded+d.receipt_mean(r,j)<=d.deadline[r]+1e-12;
}
}

AnalyticalMasterBounds analytical_master_bounds(const Input& d) {
    if (d.route_mode != "direct" || d.has_candidate_routes)
        throw std::invalid_argument("Analytical master bounds require the direct-flight path domain");
    if (d.max_orders < 1 || d.max_orders > 2 || !(d.fleet_cost > 0))
        throw std::invalid_argument("Invalid master-bound fleet cost or task cardinality");
    const int L = d.n_l, H = d.n_h, R = d.n_od;
    AnalyticalMasterBounds result;
    result.unit_cost.resize(d.assignment_count());
    result.unit_saving.resize(d.assignment_count());
    result.pair_possible.resize(d.assignment_count());
    result.slot_time.resize(R);
    result.courier_cost = std::inner_product(d.demand.begin(), d.demand.end(), d.ground_cost.begin(), 0.0);
    std::vector<double> outbound(L, infinity), inbound(L, infinity);
    for (int l = 0; l < L; ++l) for (int i = 0; i < H; ++i) {
        if (d.allowed_links(l, L + i)) outbound[l] = std::min(outbound[l], d.flight_times(l, L + i));
        if (d.allowed_links(L + i, l)) inbound[l] = std::min(inbound[l], d.flight_times(L + i, l));
    }
    // An origin-only assignment can return to any base. A lower cost bound
    // therefore minimizes the return access cost over all physical bases.
    if (d.origin_assignment()) inbound[0] = *std::min_element(inbound.begin(),inbound.end());
    const double flight_and_busy_cost = d.flight_cost + d.fleet_cost;
    for (int r = 0; r < R; ++r) {
        double loaded_cost = infinity, slot_time = infinity;
        for (int i = 0; i < H; ++i) for (int j = 0; j < H; ++j) {
            if (i == j || !d.allowed_links(L + i, L + j)) continue;
            const double loaded = d.flight_times(L + i, L + j);
            loaded_cost = std::min(loaded_cost, d.first_cost(r, i) + d.terminal_cost(r, j)
                                                       + flight_and_busy_cost * loaded);
            // Every order consumes q at pickup and q+t+removal at delivery;
            // q is nonnegative even for a second order. Dropping both q terms
            // gives a lower allocation valid for repeated orders/lockers.
            slot_time = std::min(slot_time, loaded + d.removal_mean(r, j));
        }
        result.slot_time[r] = std::isfinite(slot_time) ? lower_nonnegative(slot_time) : 0;
        double best_saving = 0;
        for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) {
            const int index = d.assignment_offset(r,l,b);
            const bool possible = std::isfinite(loaded_cost) && std::isfinite(outbound[l]) && std::isfinite(inbound[b]);
            result.pair_possible[index] = possible;
            if (!possible) continue; // Absence of a required arc proves zero service.
            // A task with n<=K orders pays its outbound/return flights once.
            // Allocating 1/K to each occurrence sums to at most those flights.
            // Loaded flight and rider costs are charged once per occurrence.
            // Nonnegative transfer, waiting, empty and idle costs are omitted.
            const double cost = loaded_cost + flight_and_busy_cost * (outbound[l] + inbound[b]) / d.max_orders;
            result.unit_cost[index] = lower_nonnegative(cost);
            result.unit_saving[index] = upper_nonnegative(std::max(0.0, d.ground_cost[r] - result.unit_cost[index]));
            best_saving = std::max(best_saving, result.unit_saving[index]);
            if (result.unit_saving[index] > 0) {
                if (result.slot_time[r] == 0) result.finite_capacity_savings = false;
                else result.savings_per_slot_time = std::max(result.savings_per_slot_time,
                    upper_nonnegative(result.unit_saving[index] / result.slot_time[r]));
            }
        }
        result.operating_floor += d.demand[r] * std::max(0.0, d.ground_cost[r] - best_saving);
    }
    result.operating_floor = lower_nonnegative(result.operating_floor);
    return result;
}

Json add_master_strengthening(const Input& d, const PathCollection& paths, Formulation& master,
                              int theta, bool resource_bounds) {
    auto& m = master.matrix;
    if (theta < master.layout.count || theta >= static_cast<int>(m.variables.size()))
        throw std::invalid_argument("Master strengthening needs an operating-cost variable after the design");
    const auto bounds = analytical_master_bounds(d);
    const std::size_t old_variables = m.variables.size(), old_rows = m.rows.size();
    const double margin = 1e-8 * (1 + bounds.courier_cost);
    Coefficients assignments{{theta, 1}};
    for (std::size_t j = 0; j < bounds.unit_saving.size(); ++j)
        if (bounds.unit_saving[j] > 0)
            assignments.emplace_back(master.layout.assignment + j,
                                     d.demand[j / d.assignment_pool_count()] * bounds.unit_saving[j]);
    m.row("analytical_courier_savings_bound", 'G', bounds.courier_cost - margin, std::move(assignments));

    // For every task p, saving(p)<=sum_r n_rp*alpha_rb and
    // load(p)>=sum_r n_rp*omega_r. beta=max alpha_rb/omega_r therefore
    // bounds savings/load on the ENTIRE path domain, not merely the pool.
    const bool capacity_cut = resource_bounds && bounds.finite_capacity_savings;
    if (capacity_cut) {
        Coefficients capacity{{theta, 1}};
        for (int i = 0; i < d.n_h; ++i) for (int s = 0; s < d.n_levels; ++s)
            capacity.emplace_back(master.layout.capacity + i * d.n_levels + s,
                                  bounds.savings_per_slot_time * d.occupancy_limit(i, s));
        m.row("analytical_locker_savings_bound", 'G', bounds.courier_cost - margin, std::move(capacity));
    }

    return {{"enabled", true}, {"source", "input_only_direct_flight_cost_and_resource_allocations"},
            {"catalogue_complete", paths.complete}, {"requires_path_catalogue", false},
            {"master_form", "design_binaries_plus_operating_cost"}, {"resource_bounds", resource_bounds},
            {"locker_cut_enabled", capacity_cut},
            {"locker_cut_reason", capacity_cut ? "analytically_valid_over_entire_path_domain" :
                 (resource_bounds ? "positive_savings_with_zero_slot_lower_bound" : "disabled")},
            {"locker_value_per_load", bounds.finite_capacity_savings ? Json(bounds.savings_per_slot_time) : Json(nullptr)},
            {"unconstrained_operating_floor", std::max(0.0, bounds.operating_floor - margin)},
            {"numerical_margin", margin},
            {"added_variables", m.variables.size() - old_variables}, {"added_rows", m.rows.size() - old_rows}};
}

CapacitySavingsNetwork capacity_savings_network(const Input& d) {
    // Use the same domain checks as the analytical bounds. No generated path
    // or design-dependent reachability is used to exclude an occurrence.
    analytical_master_bounds(d);
    const int L = d.n_l, H = d.n_h, G = d.assignment_count();
    const DesignLayout layout(d);
    CapacitySavingsNetwork out;
    out.source = 0; out.sink = 1 + G + H; out.nodes = out.sink + 1;
    out.courier_cost = std::inner_product(d.demand.begin(), d.demand.end(), d.ground_cost.begin(), 0.0);
    out.numerical_margin = 1e-8 * (1 + out.courier_cost);
    out.unit_saving.resize(G * H); out.slot_time.resize(G * H);
    out.unit_cost.resize(G * H); out.eligible.resize(G * H);
    std::vector<double> outbound(L, infinity), inbound(L, infinity), alpha(G), beta(H);
    std::vector<bool> zero_load(H);
    for (int l = 0; l < L; ++l) for (int i = 0; i < H; ++i) {
        if (d.allowed_links(l, L+i)) outbound[l] = std::min(outbound[l], d.flight_times(l, L+i));
        if (d.allowed_links(L+i, l)) inbound[l] = std::min(inbound[l], d.flight_times(L+i, l));
    }
    if (d.origin_assignment()) inbound[0] = *std::min_element(inbound.begin(),inbound.end());
    for (int r = 0; r < d.n_od; ++r) for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) {
        const int g = d.assignment_offset(r,l,b);
        if (!std::isfinite(outbound[l]) || !std::isfinite(inbound[b])) continue;
        for (int j = 0; j < H; ++j) {
            const int index = g * H + j;
            double cost = infinity, load = infinity;
            for (int i = 0; i < H; ++i) {
                if (!eligible_loaded_occurrence(d,r,i,j)) continue;
                const double loaded = d.flight_times(L+i, L+j);
                // Necessary for every occurrence, including the second order.
                // In particular, do not apply first-leg delta or base-access
                // conditions to a second occurrence.
                cost = std::min(cost, d.first_cost(r,i) + d.terminal_cost(r,j) +
                    (d.flight_cost+d.fleet_cost) * (loaded+(outbound[l]+inbound[b])/d.max_orders));
                load = std::min(load, loaded+d.removal_mean(r,j));
            }
            if (!std::isfinite(cost)) continue;
            out.eligible[index] = true;
            out.unit_cost[index] = lower_nonnegative(cost);
            out.slot_time[index] = lower_nonnegative(load);
            out.unit_saving[index] = upper_nonnegative(std::max(0.0,d.ground_cost[r]-out.unit_cost[index]));
            alpha[g] = std::max(alpha[g],out.unit_saving[index]);
            if (out.unit_saving[index] > 0) {
                if (out.slot_time[index] == 0) zero_load[j] = true;
                else beta[j] = std::max(beta[j],upper_nonnegative(out.unit_saving[index]/out.slot_time[index]));
            }
        }
    }
    double total_savings = 0;
    for (int r = 0; r < d.n_od; ++r) {
        double best = 0;
        for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b)
            best = std::max(best,alpha[d.assignment_offset(r,l,b)]);
        total_savings += d.demand[r]*best;
    }
    total_savings = upper_nonnegative(total_savings);
    for (int g = 0; g < G; ++g) {
        const int r = g / d.assignment_pool_count(), assignment = layout.assignment+g;
        if (alpha[g] > 0) out.arcs.push_back({out.source,1+g,0,{{assignment,upper_nonnegative(d.demand[r]*alpha[g])}}});
        for (int j = 0; j < H; ++j) if (out.unit_saving[g*H+j] > 0)
            out.arcs.push_back({1+g,1+G+j,0,{{assignment,upper_nonnegative(d.demand[r]*out.unit_saving[g*H+j])}}});
    }
    for (int j = 0; j < H; ++j) {
        CapacitySavingsArc arc{1+G+j,out.sink,zero_load[j] ? total_savings : 0,{}};
        if (!zero_load[j]) for (int s = 0; s < d.n_levels; ++s)
            if (beta[j] > 0 && d.occupancy_limit(j,s) > 0)
                arc.coefficients.emplace_back(layout.capacity+j*d.n_levels+s,
                    upper_nonnegative(beta[j]*d.occupancy_limit(j,s)));
        out.arcs.push_back(std::move(arc));
    }
    out.metadata = {{"enabled",true},{"source","input_only_occurrence_delivery_relaxation"},
        {"requires_path_catalogue",false},{"nodes",out.nodes},{"arcs",out.arcs.size()},
        {"zero_load_lockers",std::count(zero_load.begin(),zero_load.end(),true)},
        {"zero_load_sink_constant",total_savings},{"numerical_margin",out.numerical_margin}};
    return out;
}

CapacitySavingsCut CapacitySavingsNetwork::separate(const std::vector<double>& design,
                                                   double theta_value,int theta) const {
    struct Edge { int to, reverse; double residual; };
    std::vector<std::vector<Edge>> graph(nodes);
    const double epsilon = 1e-12 * (1+std::abs(courier_cost));
    auto value = [&](const CapacitySavingsArc& arc) {
        double cap = arc.constant;
        for (auto [j,c] : arc.coefficients) cap += c*design.at(j);
        if (!std::isfinite(cap) || cap < -epsilon)
            throw std::invalid_argument("Capacity separator needs a finite nonnegative design");
        return std::max(0.0,cap);
    };
    for (const auto& arc : arcs) {
        const int a = graph[arc.from].size(), b = graph[arc.to].size();
        graph[arc.from].push_back({arc.to,b,value(arc)});
        graph[arc.to].push_back({arc.from,a,0});
    }
    std::vector<int> level(nodes), next(nodes);
    auto layers = [&]() {
        std::fill(level.begin(),level.end(),-1);
        std::queue<int> queue; queue.push(source); level[source]=0;
        while (!queue.empty()) {
            const int u=queue.front(); queue.pop();
            for (const auto& e : graph[u]) if (e.residual>epsilon && level[e.to]<0) {
                level[e.to]=level[u]+1; queue.push(e.to);
            }
        }
        return level[sink]>=0;
    };
    auto augment = [&](auto&& self,int u,double available) -> double {
        if (u==sink) return available;
        for (int& k=next[u]; k<static_cast<int>(graph[u].size()); ++k) {
            auto& e=graph[u][k];
            if (e.residual<=epsilon || level[e.to]!=level[u]+1) continue;
            const double pushed=self(self,e.to,std::min(available,e.residual));
            if (pushed>epsilon) {
                e.residual-=pushed; graph[e.to][e.reverse].residual+=pushed; return pushed;
            }
        }
        return 0;
    };
    CapacitySavingsCut cut;
    while (layers()) {
        std::fill(next.begin(),next.end(),0);
        while (const double pushed=augment(augment,source,infinity)) cut.maxflow+=pushed;
    }
    // The final BFS defines a cut; every original crossing edge contributes,
    // including source-to-group edges. Max-flow itself is only diagnostic.
    std::map<int,double> coefficients;
    double constant=0;
    for (int u=0;u<nodes;++u) cut.partition_key.push_back(level[u]>=0?'1':'0');
    for (const auto& arc : arcs) if (level[arc.from]>=0 && level[arc.to]<0) {
        constant+=arc.constant;
        for (auto [j,c] : arc.coefficients) coefficients[j]+=c;
    }
    constant=upper_nonnegative(constant);
    cut.row.name="capacity_savings_cut"; cut.row.sense='G';
    cut.row.rhs=courier_cost-constant-numerical_margin;
    cut.row.coefficients.emplace_back(theta,1);
    cut.cut_capacity=constant;
    for (auto [j,c] : coefficients) {
        c=upper_nonnegative(c);
        cut.row.coefficients.emplace_back(j,c);
        cut.cut_capacity+=c*design.at(j);
    }
    cut.violation=courier_cost-numerical_margin-cut.cut_capacity-theta_value;
    cut.violated=cut.violation>1e-7*(1+std::abs(courier_cost)+std::abs(theta_value));
    return cut;
}

ResourceSavingsModel resource_savings_model(const Input& d,bool airline_bounds,bool tight_airline_bounds,
                                           bool od_airline_bounds) {
    return resource_savings_model(d,capacity_savings_network(d),airline_bounds,tight_airline_bounds,
                                  od_airline_bounds);
}

ResourceSavingsModel resource_savings_model(const Input& d,const CapacitySavingsNetwork& allocations,
                                           bool airline_bounds,bool tight_airline_bounds,
                                           bool od_airline_bounds) {
    const DesignLayout layout(d);
    const int L=d.n_l, G=d.assignment_count(), H=d.n_h, N=d.n_nodes();
    if (allocations.eligible.size()!=static_cast<std::size_t>(G*H) ||
        allocations.unit_saving.size()!=static_cast<std::size_t>(G*H) ||
        allocations.slot_time.size()!=static_cast<std::size_t>(G*H))
        throw std::invalid_argument("Weighted resource allocations do not match input dimensions");
    ResourceSavingsModel out;
    out.courier_cost = allocations.courier_cost;
    out.numerical_margin = allocations.numerical_margin;
    const int aggregate_rows=airline_bounds ? H+2*L : 0;
    const int od_departure=aggregate_rows,od_return=od_departure+d.n_od*L;
    const int od_delivery=od_return+d.n_od*L;
    out.group_rhs.resize(G); out.locker_rhs.resize(H);
    for (int g=0;g<G;++g) {
        out.group_rhs[g].emplace_back(layout.assignment+g,d.demand[g/d.assignment_pool_count()]);
        for (int j=0;j<H;++j) {
            const int k=g*H+j;
            if (allocations.eligible[k] && allocations.unit_saving[k]>0) {
                ResourceSavingsOccurrence occurrence{g,j,allocations.unit_saving[k],allocations.slot_time[k],{}};
                const int r=g/d.assignment_pool_count(),l=(g/d.assignment_destinations())%L,b=g%L;
                if (airline_bounds) {
                    occurrence.airline_resources={j,H+l};
                    if (!d.origin_assignment()) occurrence.airline_resources.push_back(H+L+b);
                }
                if (od_airline_bounds) {
                    occurrence.airline_resources.push_back(od_departure+r*L+l);
                    if (!d.origin_assignment()) occurrence.airline_resources.push_back(od_return+r*L+b);
                    occurrence.airline_resources.push_back(od_delivery+r*H+j);
                }
                out.occurrences.push_back(std::move(occurrence));
            }
        }
    }
    for (int j=0;j<H;++j) for (int s=0;s<d.n_levels;++s)
        out.locker_rhs[j].emplace_back(layout.capacity+j*d.n_levels+s,d.occupancy_limit(j,s));
    if (airline_bounds) {
        // These are sums of ORIGINAL link rows, not design-dependent path
        // exclusions. A loaded arrival contributes once per order. A task
        // has at most K orders and one departure/return leg. Thus the three
        // gate inequalities hold for fractional lifted designs as well.
        // Match formulation.cpp: M_HH is K*D or the opt-in D; M_baseH=D. Base rows multiply
        // the latter by K, because their left sides count order occurrences.
        out.airline_rhs.resize(H+2*L);
        const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
        const double order_bound=upper_nonnegative(total*d.max_orders);
        const double delivery_bound=upper_nonnegative(total*(tight_airline_bounds ? 1 : d.max_orders));
        for (int j=0;j<H;++j) for (int i=0;i<H;++i)
            if (i!=j && d.allowed_links(L+i,L+j))
                out.airline_rhs[j].emplace_back(layout.airline+(L+i)*N+L+j,delivery_bound);
        for (int l=0;l<L;++l) for (int i=0;i<H;++i) {
            if (d.allowed_links(l,L+i))
                out.airline_rhs[H+l].emplace_back(layout.airline+l*N+L+i,order_bound);
            if (d.allowed_links(L+i,l))
                out.airline_rhs[H+L+l].emplace_back(layout.airline+(L+i)*N+l,order_bound);
        }
    }
    if (od_airline_bounds) {
        // For every OD r, total served occurrences (AA counted twice) <= h_r.
        // At a binary design, positive flow in each left-hand group needs at
        // least one open link in its RHS; otherwise every mapped flow is zero.
        // This proves the h_r-scaled gates for integer designs and their convex
        // hull, NOT for the original fractional lifted operational model.
        // Aggregate across the omitted base index instead of duplicating the
        // same RHS for each (r,l,b); I_rj is independent of the base pair.
        out.airline_rhs.resize(aggregate_rows+d.n_od*(2*L+H));
        for (int r=0;r<d.n_od;++r) {
            const double demand=upper_nonnegative(d.demand[r]);
            for (int l=0;l<L;++l) for (int i=0;i<H;++i) {
                if (d.allowed_links(l,L+i))
                    out.airline_rhs[od_departure+r*L+l].emplace_back(
                        layout.airline+l*N+L+i,demand);
                if (d.allowed_links(L+i,l))
                    out.airline_rhs[od_return+r*L+l].emplace_back(
                        layout.airline+(L+i)*N+l,demand);
            }
            for (int j=0;j<H;++j) for (int i=0;i<H;++i)
                if (eligible_loaded_occurrence(d,r,i,j))
                    out.airline_rhs[od_delivery+r*H+j].emplace_back(
                        layout.airline+(L+i)*N+L+j,demand);
        }
    }
    out.metadata={{"enabled",true},{"source","input_only_weighted_occurrence_lp_dual"},
        {"requires_path_catalogue",false},{"master_added_variables",0},
        {"assignment_scope",d.assignment_scope()},{"return_gate_applied",!d.origin_assignment()},
        {"groups",G},{"lockers",H},{"occurrences",out.occurrences.size()},
        {"airline_bounds",airline_bounds},{"tight_airline_bounds",tight_airline_bounds},{"airline_resource_rows",out.airline_rhs.size()},
        {"aggregate_airline_resource_rows",aggregate_rows},
        {"legacy_airline_resource_rows",aggregate_rows},
        {"od_airline_bounds",od_airline_bounds},
        {"od_airline_resource_rows",od_airline_bounds?d.n_od*(2*L+H):0},
        {"airline_gate_groups",airline_bounds?Json{{"delivery",H},{"departure",L},{"return",L}}:Json::object()},
        {"od_airline_gate_groups",od_airline_bounds?
            Json{{"departure",d.n_od*L},{"return",d.n_od*L},{"delivery",d.n_od*H}}:Json::object()},
        {"airline_mapping_scope",od_airline_bounds?"integer_design_convex_hull":
            "complete_physical_domain_including_fractional_lifted_designs"},
        {"validity_scope",od_airline_bounds?"integer_design_convex_hull":"original_continuous_relaxation"},
        {"od_pickup_support","input_allowed_loaded_link_and_necessary_time_conditions_no_first_leg_filter"},
        {"dual_variables",G+H+out.airline_rhs.size()},{"dual_constraints",out.occurrences.size()},
        {"zero_slot_occurrences",std::count_if(out.occurrences.begin(),out.occurrences.end(),
            [](const auto& o){return o.slot_time==0;})},
        {"certificate","nonnegative_dual_multipliers_checked_over_all_input_occurrences"},
        {"numerical_margin",out.numerical_margin}};
    return out;
}

ResourceSavingsRelaxation::ResourceSavingsRelaxation(const Input& d,const SolverOptions& options)
    : ResourceSavingsRelaxation(resource_savings_model(d,options.resource_airline_bounds,
          options.tight_airline_bounds,options.od_airline_bounds),options) {}

ResourceSavingsRelaxation::ResourceSavingsRelaxation(ResourceSavingsModel data,const SolverOptions& options)
    : data_(std::move(data)) {
    const double started=wall_time();
    const int G=data_.group_rhs.size(),H=data_.locker_rhs.size(),A=data_.airline_rhs.size();
    if (!std::isfinite(data_.courier_cost) || data_.courier_cost<0 ||
        !std::isfinite(data_.numerical_margin) || data_.numerical_margin<0)
        throw std::invalid_argument("Invalid weighted resource relaxation cost or margin");
    auto check_rhs=[](const std::vector<Coefficients>& rows) {
        for (const auto& row:rows) for (auto [j,c]:row)
            if (j<0 || !std::isfinite(c) || c<0)
                throw std::invalid_argument("Resource relaxation needs nonnegative affine supplies");
    };
    check_rhs(data_.group_rhs); check_rhs(data_.locker_rhs); check_rhs(data_.airline_rhs);
    LinearModel dual;
    for (int g=0;g<G;++g) dual.variable("demand_price_"+std::to_string(g),0,IloInfinity);
    for (int j=0;j<H;++j) dual.variable("locker_price_"+std::to_string(j),0,IloInfinity);
    for (int a=0;a<A;++a) dual.variable("airline_price_"+std::to_string(a),0,IloInfinity);
    for (std::size_t k=0;k<data_.occurrences.size();++k) {
        const auto& o=data_.occurrences[k];
        if (o.group<0 || o.group>=G || o.locker<0 || o.locker>=H ||
            !std::isfinite(o.saving) || o.saving<0 || !std::isfinite(o.slot_time) || o.slot_time<0)
            throw std::invalid_argument("Invalid weighted resource occurrence");
        Coefficients coefficients{{o.group,1}};
        if (o.slot_time>0) coefficients.emplace_back(G+o.locker,o.slot_time);
        auto indices=o.airline_resources;
        std::sort(indices.begin(),indices.end());
        if (std::adjacent_find(indices.begin(),indices.end())!=indices.end())
            throw std::invalid_argument("Duplicate airline resource in an occurrence");
        for (int a:indices) {
            if (a<0 || a>=A) throw std::invalid_argument("Invalid occurrence airline resource");
            coefficients.emplace_back(G+H+a,1);
        }
        dual.row("occurrence_"+std::to_string(k),'G',o.saving,std::move(coefficients));
    }
    solver_=std::make_unique<NativeSolver>(dual,options,"resource_lp_dual");
    data_.metadata["dual_build_seconds"]=wall_time()-started;
}

ResourceSavingsCut ResourceSavingsRelaxation::separate(const std::vector<double>& design,
        double theta_value,int theta,double seconds) {
    const double started=wall_time();
    const int G=data_.group_rhs.size(),H=data_.locker_rhs.size(),A=data_.airline_rhs.size();
    if (!std::isfinite(theta_value) || theta<0 || !std::isfinite(seconds))
        throw std::invalid_argument("Invalid weighted resource separation point or budget");
    auto supply=[&](const Coefficients& row) {
        long double value=0;
        for (auto [j,c]:row) {
            const double x=design.at(j);
            if (!std::isfinite(x) || x < -1e-8)
                throw std::invalid_argument("Resource separator needs a nonnegative finite design");
            value+=static_cast<long double>(c)*std::max(0.0,x);
        }
        const double result=static_cast<double>(value);
        if (!std::isfinite(result)) throw std::invalid_argument("Nonfinite resource supply");
        return result;
    };
    for (int g=0;g<G;++g)
        solver_->objective.setLinearCoef(solver_->variables[g],supply(data_.group_rhs[g]));
    for (int j=0;j<H;++j)
        solver_->objective.setLinearCoef(solver_->variables[G+j],supply(data_.locker_rhs[j]));
    for (int a=0;a<A;++a)
        solver_->objective.setLinearCoef(solver_->variables[G+H+a],supply(data_.airline_rhs[a]));
    ResourceSavingsCut cut;
    cut.demand_dual.assign(G,0); cut.locker_dual.assign(H,0); cut.airline_dual.assign(A,0);
    SolveOutcome solution;
    if (seconds>0) {
        try {
            solution=solver_->optimize(std::max(0.0,seconds-(wall_time()-started)));
            cut.status=solution.status;
        } catch (const IloException& error) {
            cut.status=std::string("solver_exception: ")+error.getMessage();
        }
    } else cut.status="budget_exhausted";
    cut.optimal=solution.optimal;
    cut.solver_seconds=solution.seconds;
    cut.solver_objective=solution.objective;
    cut.fallback=!solution.has_solution || solution.values.size()!=static_cast<std::size_t>(G+H+A) ||
        !std::all_of(solution.values.begin(),solution.values.end(),[](double v){return std::isfinite(v);});
    if (!cut.fallback) {
        for (int j=0;j<H;++j) cut.locker_dual[j]=std::max(0.0,solution.values[G+j]);
        for (int a=0;a<A;++a) cut.airline_dual[a]=std::max(0.0,solution.values[G+H+a]);
        for (const auto& o:data_.occurrences) {
            double airline_price=0;
            for (int a:o.airline_resources) airline_price+=solution.values[G+H+a];
            cut.max_dual_violation_before=std::max(cut.max_dual_violation_before,
                o.saving-solution.values[o.group]-o.slot_time*solution.values[G+o.locker]-airline_price);
        }
    }
    // For any nonnegative locker/airline prices, choose the smallest demand prices
    // satisfying every occurrence inequality, rounded upwards. Thus even an
    // inaccurate or time-limited LP point yields an independently checked
    // feasible dual. With no solver point, nu=0 gives mu_g=max_j alpha_gj.
    for (const auto& o:data_.occurrences) {
        long double required=static_cast<long double>(o.saving)-
            static_cast<long double>(o.slot_time)*cut.locker_dual[o.locker];
        for (int a:o.airline_resources) required-=cut.airline_dual[a];
        const double price=upper_nonnegative(static_cast<double>(std::max(0.0L,required)));
        cut.demand_dual[o.group]=std::max(cut.demand_dual[o.group],price);
    }
    for (const auto& o:data_.occurrences) {
        long double slack=static_cast<long double>(cut.demand_dual[o.group])+
            static_cast<long double>(o.slot_time)*cut.locker_dual[o.locker]-o.saving;
        for (int a:o.airline_resources) slack+=cut.airline_dual[a];
        cut.max_dual_violation_after=std::max(cut.max_dual_violation_after,static_cast<double>(-slack));
    }
    cut.valid=cut.max_dual_violation_after<=0 &&
        std::all_of(cut.demand_dual.begin(),cut.demand_dual.end(),[](double v){return std::isfinite(v)&&v>=0;}) &&
        std::all_of(cut.locker_dual.begin(),cut.locker_dual.end(),[](double v){return std::isfinite(v)&&v>=0;}) &&
        std::all_of(cut.airline_dual.begin(),cut.airline_dual.end(),[](double v){return std::isfinite(v)&&v>=0;});
    if (!cut.valid) {cut.seconds=wall_time()-started;return cut;}
    std::map<int,long double> sums;
    for (int g=0;g<G;++g) for (auto [j,c]:data_.group_rhs[g])
        sums[j]+=static_cast<long double>(c)*cut.demand_dual[g];
    for (int i=0;i<H;++i) for (auto [j,c]:data_.locker_rhs[i])
        sums[j]+=static_cast<long double>(c)*cut.locker_dual[i];
    for (int a=0;a<A;++a) for (auto [j,c]:data_.airline_rhs[a])
        sums[j]+=static_cast<long double>(c)*cut.airline_dual[a];
    cut.row.name="resource_lp_savings_cut"; cut.row.sense='G';
    cut.row.rhs=data_.courier_cost-data_.numerical_margin;
    cut.row.coefficients.emplace_back(theta,1);
    std::ostringstream key;
    key<<std::scientific<<std::setprecision(10);
    long double saving=0;
    for (auto [j,sum]:sums) if (sum>0) {
        const double coefficient=upper_nonnegative(static_cast<double>(sum));
        if (!std::isfinite(coefficient)) {cut.valid=false;cut.seconds=wall_time()-started;return cut;}
        cut.row.coefficients.emplace_back(j,coefficient);
        saving+=static_cast<long double>(coefficient)*design.at(j);
        // Only the deduplication key is rounded; the mathematical row retains
        // its conservative full-precision coefficients.
        key<<j<<':'<<coefficient<<';';
    }
    cut.key=key.str();
    cut.dual_savings=static_cast<double>(saving);
    cut.violation=cut.row.rhs-cut.dual_savings-theta_value;
    cut.violated=cut.violation>1e-7*(1+std::abs(data_.courier_cost)+std::abs(theta_value));
    cut.seconds=wall_time()-started;
    return cut;
}
} // namespace mlnd
