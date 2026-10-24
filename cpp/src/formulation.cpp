#include "mlnd/formulation.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
std::string label(const std::string& name, int i, int j = -1, int k = -1) {
    std::string out = name + "_" + std::to_string(i);
    if (j >= 0) out += "_" + std::to_string(j);
    if (k >= 0) out += "_" + std::to_string(k);
    return out;
}
std::vector<int> selected(std::size_t count, const std::vector<int>* supplied) {
    std::vector<int> result;
    if (supplied) result = *supplied;
    else { result.resize(count); std::iota(result.begin(), result.end(), 0); }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        throw std::invalid_argument("Duplicate selected column");
    for (int i : result) if (i < 0 || i >= static_cast<int>(count))
        throw std::invalid_argument("Selected column outside generated path pool");
    return result;
}
}
DesignLayout::DesignLayout(const Input& d) {
    capacity = d.n_nodes();
    assignment = capacity + d.n_h * d.n_levels;
    n_l = d.n_l;
    assignment_destinations = d.assignment_destinations();
    airline = assignment + d.assignment_count();
    count = airline + d.n_nodes() * d.n_nodes();
}
Formulation design_model(const Input& d, bool integer, const FormulationStrengthening& strengthening) {
    Formulation f(d);
    f.strengthening = strengthening;
    f.strengthening_metadata = {{"enabled",strengthening.enabled()},
        {"assignment_incident_clique",strengthening.assignment_incident_clique},
        {"shared_rho_joint",strengthening.shared_rho_joint},
        {"queue_perspective",strengthening.queue_perspective},
        {"assignment_incident_rows",0},{"shared_rho_domain_rows",0},{"shared_rho_rows",0},{"perspective_cones",0},
        {"finite_idle_rho_rows",0},{"local_convexification",strengthening.enabled()},
        {"integer_feasible_set_preserved",true},{"global_convex_hull_claimed",false},
        {"off_base_idle_forced_zero",false},
        {"validity_scope",strengthening.enabled()?"all_original_integer_designs":"original_lifted_relaxation"}};
    auto& m = f.matrix;
    const int N = d.n_nodes(), L = d.n_l;
    auto x = [](int i) { return i; };
    auto z = [&](int i, int s) { return f.layout.capacity + i * d.n_levels + s; };
    auto a = [&](int r, int l, int b) { return f.layout.assignment_index(r,l,b); };
    auto y = [&](int i, int j) { return f.layout.airline + i * N + j; };
    for (int i = 0; i < N; ++i) m.variable(label("facility", i), 0, 1, i < L ? d.base_cost[i] : 0, integer);
    for (int i = 0; i < d.n_h; ++i)
        for (int s = 0; s < d.n_levels; ++s)
            m.variable(label("capacity_level", i, s), 0, 1, d.slot_cost(i, s), integer);
    for (int r = 0; r < d.n_od; ++r)
        for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b)
            m.variable(label("assignment", r, l, d.origin_assignment() ? -1 : b), 0, 1, 0, integer);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j)
        m.variable(label("airline", i, j), 0, 1, d.link_cost(i, j), integer);
    for (int i = 0; i < f.layout.count; ++i) {
        f.design_vars.push_back(i);
        f.opening_costs.push_back(m.variables[i].cost);
    }
    for (int i = 0; i < d.n_h; ++i) {
        Coefficients row{{x(L + i), -1}};
        for (int s = 0; s < d.n_levels; ++s) row.emplace_back(z(i, s), 1);
        m.row(label("install", i), 'E', 0, std::move(row));
    }
    for (int r = 0; r < d.n_od; ++r) {
        Coefficients sum;
        for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) sum.emplace_back(a(r,l,b), 1);
        m.row(label("assign", r), 'L', 1, std::move(sum));
        for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) {
            m.row(label("assign_departure",r,l,b), 'L', 0, {{a(r,l,b),1},{x(l),-1}});
            if (!d.origin_assignment()) m.row(label("assign_return",r,l,b), 'L', 0, {{a(r,l,b),1},{x(b),-1}});
        }
        if (strengthening.assignment_incident_clique) for (int j = 0; j < L; ++j) {
            // An OD selects at most one pair. Every selected pair incident to
            // base j requires that base; the (j,j) pair is counted only once.
            Coefficients incident{{x(j),-1}};
            for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b)
                if (l == j || (!d.origin_assignment() && b == j)) incident.emplace_back(a(r,l,b),1);
            m.row(label("assign_incident",r,j),'L',0,std::move(incident));
        }
        if (strengthening.shared_rho_joint && !strengthening.assignment_incident_clique)
            for (int l = 0; l < L; ++l) {
                // Projection of the two shared-rho rows: sum_b a_rlb <= x_l.
                // Keep this domain row in the strategic master even when the
                // stronger incident clique is disabled, so every master point
                // still has a feasible all-courier operating extension.
                Coefficients departure{{x(l),-1}};
                for (int b = 0; b < d.assignment_destinations(); ++b) departure.emplace_back(a(r,l,b),1);
                m.row(label("assign_shared_rho_domain",r,l),'L',0,std::move(departure));
            }
    }
    f.strengthening_metadata["assignment_incident_rows"] = strengthening.assignment_incident_clique ? d.n_od*L : 0;
    f.strengthening_metadata["shared_rho_domain_rows"] =
        strengthening.shared_rho_joint && !strengthening.assignment_incident_clique ? d.n_od*L : 0;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        m.row(label("airline_from",i,j), 'L', 0, {{y(i,j),1},{x(i),-1}});
        m.row(label("airline_to",i,j), 'L', 0, {{y(i,j),1},{x(j),-1}});
        if (!d.allowed_links(i,j)) m.row(label("airline_forbidden",i,j),'E',0,{{y(i,j),1}});
    }
    return f;
}
Formulation build_formulation(const Input& d, const PathCollection& paths,
                             const std::vector<double>* fixed,
                             const std::vector<int>* service,
                             const std::vector<int>* empty, bool relax_fixed_design,
                             bool conditional_demand, bool tight_airline_bounds,
                             const FormulationStrengthening& strengthening) {
    if (!paths.support_complete) throw std::invalid_argument("Complete initial-segment support is required");
    if (tight_airline_bounds && (d.route_mode != "direct" || d.has_candidate_routes || d.max_orders < 1 || d.max_orders > 2))
        throw std::invalid_argument("Tight airline bounds require the direct K<=2 path domain");
    if (conditional_demand && (!fixed || relax_fixed_design))
        throw std::invalid_argument("Conditional demand requires an integer fixed design");
    auto f = design_model(d, fixed == nullptr, strengthening);
    f.conditional_demand = conditional_demand;
    f.tight_airline_bounds = tight_airline_bounds;
    auto& m = f.matrix;
    const int L = d.n_l, H = d.n_h, R = d.n_od, N = d.n_nodes();
    const int S = static_cast<int>(paths.initial_segments.size());
    const bool pooled = d.flow_policy == "aggregate", active = d.effective_delta() > 0;
    const double total = std::accumulate(d.demand.begin(), d.demand.end(), 0.0);
    const double baseline = std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    const double fleet_bound = baseline / d.fleet_cost;
    f.rho_upper.assign(L,1.0);
    if (strengthening.queue_perspective) {
        if (!std::isfinite(fleet_bound) || fleet_bound < 0)
            throw std::invalid_argument("Queue perspective requires a finite nonnegative idle upper bound");
        // rho <= M/(1+M) when z=1, and rho=0 when z=0. Round the
        // coefficient outward so this redundant integer row never tightens
        // the mathematical bound due solely to division rounding.
        const long double exact = static_cast<long double>(fleet_bound)/(1.0L+fleet_bound);
        double upper = static_cast<double>(exact);
        if (static_cast<long double>(upper) < exact) upper = std::nextafter(upper,INFINITY);
        if (upper > 0 && upper < 1) upper = std::nextafter(upper,INFINITY);
        std::fill(f.rho_upper.begin(),f.rho_upper.end(),std::min(1.0,upper));
        f.strengthening_metadata["finite_idle_upper_bound"] = fleet_bound;
        f.strengthening_metadata["rho_upper"] = f.rho_upper;
    }
    auto a = [&](int r, int l, int b) { return f.layout.assignment_index(r,l,b); };
    auto z = [&](int i, int s) { return f.layout.capacity + i * d.n_levels + s; };
    auto y = [&](int i, int j) { return f.layout.airline + i * N + j; };
    if (fixed) {
        if (fixed->size() != static_cast<std::size_t>(f.layout.count)) throw std::invalid_argument("Wrong fixed design size");
        for (int i = 0; i < f.layout.count; ++i) {
            if (!std::isfinite((*fixed)[i]) || (*fixed)[i] < 0 || (*fixed)[i] > 1 ||
                (!relax_fixed_design && (*fixed)[i] != 0 && (*fixed)[i] != 1))
                throw std::invalid_argument("Fixed design must be binary unless the continuous relaxation is explicit");
            f.design_fix_rows.push_back(m.row(label("fix",i),'E',(*fixed)[i],{{i,1}}));
            m.variables[i].cost = 0;
        }
    }
    // Retain every original group and branch. Only the demand lift is
    // evaluated at the fixed assignment; no physical support is screened here.
    std::vector<double> pool_rate, pool_q;
    if (conditional_demand) {
        pool_rate.assign(L * L, 0);
        pool_q.assign(L * L, 0);
        for (int l = 0; l < L; ++l) for (int b = 0; b < L; ++b) {
            const int pool = l * L + b;
            for (int r = 0; r < R; ++r)
                pool_rate[pool] += d.demand[r] * (*fixed)[a(r,l,b)];
            if (!std::isfinite(pool_rate[pool]))
                throw std::invalid_argument("Conditional demand pool rate must be finite");
            pool_q[pool] = -std::expm1(-d.effective_delta() * pool_rate[pool]);
        }
        f.segment_eta_factor.reserve(S);
        for (const auto& segment : paths.initial_segments) {
            const int pool = segment.b[0] * L + segment.b[1];
            f.segment_eta_factor.push_back(pool_rate[pool] > 0 ? pool_q[pool] / pool_rate[pool] : 0);
        }
    }
    auto conditional_probability = [&](int first, int second, int l, int b) {
        const int pool = l * L + b;
        if (pool_rate[pool] == 0) return 0.0;
        const double share = d.demand[second] * (*fixed)[a(second,l,b)] / pool_rate[pool];
        return pool_q[pool] * d.kappa(first,second) * share;
    };
    auto conditional_single = [&](int first, int l, int b) {
        const int pool = l * L + b;
        if (pool_rate[pool] == 0) return 1.0;
        double incompatible_share = 0;
        for (int r = 0; r < R; ++r)
            incompatible_share += (1 - d.kappa(first,r)) *
                (d.demand[r] * (*fixed)[a(r,l,b)] / pool_rate[pool]);
        return std::exp(-d.effective_delta() * pool_rate[pool]) + pool_q[pool] * incompatible_share;
    };
    std::map<std::tuple<int,int,int>,int> groups;
    for (int j = 0; j < S; ++j) {
        const auto& seg = paths.initial_segments[j];
        if (seg.id != j) throw std::invalid_argument("Segment IDs must be consecutive");
        auto key = pooled ? std::make_tuple(seg.b[0],seg.b[1],seg.first_od) : std::make_tuple(j,-1,-1);
        auto [it, added] = groups.emplace(key,static_cast<int>(f.lift_groups.size()));
        if (added) f.lift_groups.emplace_back();
        f.lift_groups[it->second].push_back(j);
        f.segment_group.push_back(it->second);
    }
    const int G = static_cast<int>(f.lift_groups.size());
    for (int l = 0; l < L; ++l) f.rho.push_back(m.variable(label("rho",l),0,1));
    for (int l = 0; l < L; ++l) f.idle.push_back(m.variable(label("idle",l),0,fleet_bound));
    f.fleet = m.variable("fleet",0,fleet_bound,d.fleet_cost);
    for (int j = 0; j < S; ++j) f.head.push_back(m.variable(label("head",j),0,d.demand[paths.initial_segments[j].first_od]));
    std::vector<int> v(d.assignment_count());
    for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b)
        v[d.assignment_offset(r,l,b)] = m.variable(label("v",r,l,d.origin_assignment() ? -1 : b),0,1);
    for (int r = 0; r < R; ++r) f.courier.push_back(m.variable(label("courier",r),0,d.demand[r],d.ground_cost[r]));
    f.flow.assign(paths.service_paths.size(),-1);
    f.empty_flow.assign(paths.empty_paths.size(),-1);
    f.service_indices = selected(paths.service_paths.size(),service);
    f.empty_indices = selected(paths.empty_paths.size(),empty);
    for (int i : f.service_indices) f.flow[i] = m.variable(label("flow",i),0,total,paths.service_paths[i].cost);
    for (int i : f.empty_indices) f.empty_flow[i] = m.variable(label("empty_flow",i),0,total,paths.empty_paths[i].cost);
    f.eta.assign(G,-1);
    const bool lifted = active && !conditional_demand;
    std::vector<int> u(lifted ? G*R : 0,-1), chain(lifted ? G*(R+1) : 0,-1),
        product(lifted ? G*R : 0,-1), heads(G,-1);
    for (int g = 0; g < G; ++g) {
        const int first = paths.initial_segments[f.lift_groups[g][0]].first_od;
        const double h = d.demand[first], upper = std::min(1.0,d.effective_delta()*h);
        if (lifted) {
            f.eta[g] = m.variable(label("eta",g),0,upper);
            for (int r = 0; r < R; ++r) u[g*R+r] = m.variable(label("u",g,r),0,upper);
            for (int r = 0; r <= R; ++r) chain[g*(R+1)+r] = m.variable(label("Z",g,r),0,h);
            for (int r = 0; r < R; ++r) product[g*R+r] = m.variable(label("Y",g,r),0,h);
        }
        heads[g] = pooled ? m.variable(label("group_head",g),0,h) : f.head[f.lift_groups[g][0]];
    }
    f.executed_head_rows.assign(S,-1);
    f.single_rows.assign(S,-1);
    f.double_rows.assign(S*R,-1);
    for (int l = 0; l < L; ++l) {
        m.row(label("rho_open",l),'L',0,{{f.rho[l],1},{l,-1}});
        int up=m.variable(label("cone_up",l),0,fleet_bound+1);
        int down=m.variable(label("cone_down",l),0,1);
        int one=m.variable(label("cone_one",l),strengthening.queue_perspective?0:1,1);
        if (strengthening.queue_perspective) {
            // (I+z)(z-rho)>=z^2 is the closed perspective epigraph.
            // At z=rho=0 it permits the original nonnegative idle slack.
            m.row(label("cone_up_link",l),'E',0,{{up,1},{f.idle[l],-1},{l,-1}});
            m.row(label("cone_down_link",l),'E',0,{{down,1},{f.rho[l],1},{l,-1}});
            m.row(label("cone_one_link",l),'E',0,{{one,1},{l,-1}});
            m.row(label("rho_inventory_bound",l),'L',0,{{f.rho[l],1},{l,-f.rho_upper[l]}});
        } else {
            m.row(label("cone_up_link",l),'E',1,{{up,1},{f.idle[l],-1}});
            m.row(label("cone_down_link",l),'E',1,{{down,1},{f.rho[l],1}});
        }
        m.cones.push_back({up,down,one,1});
    }
    for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) {
        const int vr=v[d.assignment_offset(r,l,b)], ar=a(r,l,b);
        m.row(label("v_rho",r,l,b),'L',0,{{vr,1},{f.rho[l],-1}});
        m.row(label("v_assignment",r,l,b),'L',0,{{vr,1},{ar,-1}});
        m.row(label("v_lower",r,l,b),'G',-1,{{vr,1},{f.rho[l],-1},{ar,-1}});
    }
    if (strengthening.shared_rho_joint) for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) {
        Coefficients executed{{f.rho[l],-1}}, unexecuted{{f.rho[l],1},{l,-1}};
        for (int b = 0; b < d.assignment_destinations(); ++b) {
            const int vr=v[d.assignment_offset(r,l,b)];
            executed.emplace_back(vr,1);
            unexecuted.emplace_back(a(r,l,b),1);
            unexecuted.emplace_back(vr,-1);
        }
        m.row(label("v_joint_executed",r,l),'L',0,std::move(executed));
        m.row(label("v_joint_unexecuted",r,l),'L',0,std::move(unexecuted));
    }
    f.strengthening_metadata["shared_rho_rows"] = strengthening.shared_rho_joint ? 2*R*L : 0;
    f.strengthening_metadata["perspective_cones"] = strengthening.queue_perspective ? L : 0;
    f.strengthening_metadata["finite_idle_rho_rows"] = strengthening.queue_perspective ? L : 0;
    f.strengthening_metadata["fixed_integer_recourse_strengthened"] = false;
    if (pooled) for (int j=0;j<S;++j)
        f.executed_head_rows[j]=m.row(label("executed_head",j),'E',0,{{f.head[j],1}});
    std::vector<std::vector<int>> by_pair(L*L);
    for (int g=0;g<G;++g) {
        const auto& indices=f.lift_groups[g];
        const auto& segment=paths.initial_segments[indices[0]];
        const int l=segment.b[0], b=segment.b[1], first=segment.first_od;
        const double h=d.demand[first], upper=std::min(1.0,d.effective_delta()*h);
        by_pair[l*L+b].push_back(g);
        if (pooled) {
            Coefficients row{{heads[g],1}};
            for (int j:indices) row.emplace_back(f.head[j],-1);
            m.row(label("group_head_link",g),'E',0,std::move(row));
        }
        if (lifted) {
            m.row(label("eta_upper",g),'L',0,{{f.eta[g],1},{heads[g],-d.effective_delta()}});
            m.row(label("chain_first",g),'E',0,{{chain[g*(R+1)],1},{heads[g],-1}});
            for (int r=0;r<R;++r) {
                const int ur=u[g*R+r], ar=a(r,l,b), yr=product[g*R+r], zr=chain[g*(R+1)+r];
                m.row(label("u_eta",g,r),'L',0,{{ur,1},{f.eta[g],-1}});
                m.row(label("u_assignment",g,r),'L',0,{{ur,1},{ar,-upper}});
                m.row(label("u_lower",g,r),'G',-upper,{{ur,1},{f.eta[g],-1},{ar,-upper}});
                m.row(label("Y_chain",g,r),'L',0,{{yr,1},{zr,-1}});
                m.row(label("Y_assignment",g,r),'L',0,{{yr,1},{ar,-h}});
                m.row(label("Y_lower",g,r),'G',-h,{{yr,1},{zr,-1},{ar,-h}});
                m.row(label("chain_next",g,r),'E',0,
                      {{chain[g*(R+1)+r+1],1},{zr,-1},{yr,-std::expm1(-d.effective_delta()*d.demand[r])}});
            }
            Coefficients row{{heads[g],-1},{chain[g*(R+1)+R],1}};
            for (int r=0;r<R;++r) row.emplace_back(u[g*R+r],d.demand[r]);
            m.row(label("eta_link",g),'E',0,std::move(row));
        }
        Coefficients single{{heads[g],conditional_demand ? conditional_single(first,l,b) : 1}};
        if (lifted) for (int r=0;r<R;++r) single.emplace_back(u[g*R+r],-d.kappa(first,r)*d.demand[r]);
        int row=m.row(label("single",g),'E',0,std::move(single));
        for (int j:indices) f.single_rows[j]=row;
        if (active) for (int r=0;r<R;++r) {
            row=m.row(label("double",g,r),'E',0,
                      conditional_demand ? Coefficients{{heads[g],conditional_probability(first,r,l,b)}}
                                         : Coefficients{{u[g*R+r],d.kappa(first,r)*d.demand[r]}});
            for (int j:indices) f.double_rows[j*R+r]=row;
        }
    }
    for (int l=0;l<L;++l) for (int b=0;b<d.assignment_destinations();++b) for (int r=0;r<R;++r) {
        Coefficients row{{v[d.assignment_offset(r,l,b)],-d.demand[r]}};
        // K=1 assigns only the departure base; physical return bases remain
        // in the route catalogue and the closed-network balance equations.
        std::vector<int> order_groups;
        if (d.origin_assignment()) {
            for (int m=0;m<L;++m)
                order_groups.insert(order_groups.end(),by_pair[l*L+m].begin(),by_pair[l*L+m].end());
        } else order_groups=by_pair[l*L+b];
        for (int g:order_groups) {
            const int first=paths.initial_segments[f.lift_groups[g][0]].first_od;
            if (conditional_demand) {
                const double coefficient = (first == r ? 1.0 : 0.0) +
                    (active ? conditional_probability(first,r,l,b) : 0.0);
                if (coefficient != 0) row.emplace_back(heads[g],coefficient);
            } else {
                if (first==r) row.emplace_back(heads[g],1);
                if (active) row.emplace_back(u[g*R+r],d.demand[r]*d.kappa(first,r));
            }
        }
        m.row(label("orders",r,l,d.origin_assignment() ? -1 : b),'E',0,std::move(row));
    }
    for (int r=0;r<R;++r) f.order_rows.push_back(m.row(label("demand",r),'E',d.demand[r],{{f.courier[r],1}}));
    for (int l=0;l<L;++l) f.balance_rows.push_back(m.row(label("balance",l),'E',0));
    Coefficients fleet{{f.fleet,-1}};
    for (int var:f.idle) fleet.emplace_back(var,1);
    f.fleet_row=m.row("fleet_use",'L',0,std::move(fleet));
    for (int i=0;i<H;++i) {
        Coefficients row;
        for (int s=0;s<d.n_levels;++s) row.emplace_back(z(i,s),-d.occupancy_limit(i,s));
        f.capacity_rows.push_back(m.row(label("capacity",i),'L',0,std::move(row)));
    }
    // Input-derived bounds keep every RMP and the full model on the same
    // matrix, even before a single path has been generated. A direct task
    // visits a base only at departure/return; any directed locker arc occurs
    // at most the task order count among its internal legs (adjacent occurrences would
    // require a self-loop, which the input forbids). The opt-in D bound uses
    // total served-order conservation; it strengthens fractional activations
    // while preserving binary-design feasibility. Sum of service task flows
    // is at most total demand. Each direct empty-arc variable is bounded by
    // total demand; (L-1)*total is the existing conservative relocation bound.
    for (int u=0;u<N;++u) for (int v=0;v<N;++v) {
        double bound=0;
        if (u!=v && d.allowed_links(u,v)) {
            if (u<L && v<L) bound=total*std::max(0,L-1);
            else if (u>=L && v>=L) bound=total*(tight_airline_bounds ? 1 : d.max_orders);
            else bound=total;
        }
        f.link_rows.push_back(m.row(label("link",u,v),'L',0,{{y(u,v),-bound}}));
    }
    for (int i:f.service_indices) for (auto [row,value]:path_column(f,paths.service_paths[i])) m.coefficient(row,f.flow[i],value);
    for (int i:f.empty_indices) for (auto [row,value]:path_column(f,paths.empty_paths[i])) m.coefficient(row,f.empty_flow[i],value);
    return f;
}
Coefficients path_column(const Formulation& f,const ServicePath& p) {
    const int R=static_cast<int>(f.order_rows.size()), N=static_cast<int>(std::sqrt(f.link_rows.size()));
    const int j=p.segment_id;
    int branch=p.second_od<0?f.single_rows.at(j):f.double_rows.at(j*R+p.second_od);
    if (branch<0) throw std::invalid_argument("Missing service branch row");
    Coefficients c{{branch,-1}};
    if (f.executed_head_rows.at(j)>=0) c.emplace_back(f.executed_head_rows[j],-1);
    for (auto [r,count]:p.served_counts) c.emplace_back(f.order_rows.at(r),count);
    for (auto [i,time]:p.slot_times) c.emplace_back(f.capacity_rows.at(i),time);
    if (p.b[0]!=p.b[1]) { c.emplace_back(f.balance_rows.at(p.b[0]),1); c.emplace_back(f.balance_rows.at(p.b[1]),-1); }
    c.emplace_back(f.fleet_row,p.time);
    for (auto [u,v,count]:p.arc_counts) c.emplace_back(f.link_rows.at(u*N+v),count);
    return c;
}
Coefficients path_column(const Formulation& f,const EmptyPath& p) {
    const int N=static_cast<int>(std::sqrt(f.link_rows.size()));
    Coefficients c;
    if (p.b[0]!=p.b[1]) { c.emplace_back(f.balance_rows.at(p.b[0]),1); c.emplace_back(f.balance_rows.at(p.b[1]),-1); }
    c.emplace_back(f.fleet_row,p.time);
    for (auto [u,v,count]:p.arc_counts) c.emplace_back(f.link_rows.at(u*N+v),count);
    return c;
}
void add_columns(Formulation& f,NativeSolver& solver,const Input& d,const PathCollection& paths,
                 const std::vector<int>& service,const std::vector<int>& empty) {
    // Path coefficients do not depend on strategic values. The same sparse
    // insertion therefore applies to fixed-design recourse and the free
    // continuous formulation; all common rows are already materialized.
    const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    f.flow.resize(paths.service_paths.size(),-1);
    f.empty_flow.resize(paths.empty_paths.size(),-1);
    for (int i:selected(paths.service_paths.size(),&service)) if (f.flow[i]<0) {
        int index=f.matrix.variable(label("flow",i),0,total,paths.service_paths[i].cost);
        auto column=path_column(f,paths.service_paths[i]);
        for (auto [row,value]:column) f.matrix.coefficient(row,index,value);
        solver.add_variable(f.matrix.variables.back(),column);
        f.flow[i]=index; f.service_indices.push_back(i);
    }
    for (int i:selected(paths.empty_paths.size(),&empty)) if (f.empty_flow[i]<0) {
        int index=f.matrix.variable(label("empty_flow",i),0,total,paths.empty_paths[i].cost);
        auto column=path_column(f,paths.empty_paths[i]);
        for (auto [row,value]:column) f.matrix.coefficient(row,index,value);
        solver.add_variable(f.matrix.variables.back(),column);
        f.empty_flow[i]=index; f.empty_indices.push_back(i);
    }
    std::sort(f.service_indices.begin(),f.service_indices.end());
    std::sort(f.empty_indices.begin(),f.empty_indices.end());
}
Json repair_inventory_primal(const Formulation& f, std::vector<double>& values) {
    const double started = wall_time();
    const auto before = point_residuals(f.matrix, values);
    const double original_cost = objective_value(f.matrix, values);
    Json report = {{"method", "increase_idle_and_fleet_preserving_design_flows_rho"},
                   {"attempted", false}, {"applied", false}, {"passed", before.at("passed")},
                   {"before", before}, {"after", before}, {"objective_before", original_cost},
                   {"objective_after", original_cost}, {"cost_increase", 0.0},
                   {"idle_increases", Json::array()}, {"fleet_increase", 0.0}};
    if (f.strengthening.queue_perspective) report["queue_perspective"] = true;
    auto finish = [&](const char* reason) {
        report["reason"] = reason;
        report["seconds"] = wall_time() - started;
        return report;
    };
    if (before.at("passed").get<bool>()) return finish("already_feasible");
    // These are the unchanged matrix validator's acceptance thresholds. A
    // cone repair must not mask a violation of an unrelated constraint.
    if (before.at("max_bound_violation").get<double>() > 2e-6 ||
        before.at("max_scaled_linear_violation").get<double>() > 2e-7 ||
        before.at("max_integrality_violation").get<double>() > 1e-7)
        return finish("unrelated_matrix_violation");
    report["attempted"] = true;
    if (f.matrix.cones.size() != f.rho.size() || f.idle.size() != f.rho.size() ||
        f.fleet < 0 || f.fleet_row < 0)
        return finish("unsupported_inventory_structure");
    auto repaired = values;
    auto within_bounds = [&](int j) {
        return std::isfinite(repaired[j]) && repaired[j] >= f.matrix.variables[j].lower &&
               repaired[j] <= f.matrix.variables[j].upper;
    };
    constexpr double padding = 16 * std::numeric_limits<double>::epsilon();
    for (std::size_t l = 0; l < f.rho.size(); ++l) {
        const double rho = values[f.rho[l]];
        const auto& cone = f.matrix.cones[l];
        const bool perspective = f.strengthening.queue_perspective;
        const double open = perspective ? values[f.layout.facility+static_cast<int>(l)] : 1;
        if (!std::isfinite(rho) || !std::isfinite(open) || rho < 0 || open < 0 || open > 1 ||
            (open == 0 ? rho != 0 : rho >= open))
            return finish("availability_outside_repair_domain");
        if (cone.factor != 1 || f.matrix.variables[cone.one].lower != (perspective ? 0 : 1) ||
            f.matrix.variables[cone.one].upper != 1)
            return finish("unsupported_inventory_structure");
        const double down = open - rho;
        // Use the actual rounded down-lift in the reciprocal, then round to
        // the feasible side. Slack already present in idle is never removed.
        const long double required = open == 0 ? 0 :
            std::max(0.0L,static_cast<long double>(open)*open/static_cast<long double>(down)-open);
        const double safe_idle = static_cast<double>(required) +
            ((perspective && required == 0) ? 0 : padding * std::max(1.0, static_cast<double>(required)));
        repaired[f.idle[l]] = std::max(values[f.idle[l]], safe_idle);
        repaired[cone.up] = open + repaired[f.idle[l]];
        repaired[cone.down] = down;
        repaired[cone.one] = open;
        if (!within_bounds(f.idle[l]) || !within_bounds(cone.up) ||
            !within_bounds(cone.down) || !within_bounds(cone.one))
            return finish("inventory_bound_insufficient");
        report["idle_increases"].push_back(repaired[f.idle[l]] - values[f.idle[l]]);
    }
    const auto& fleet_row = f.matrix.rows[f.fleet_row];
    long double required_fleet = -fleet_row.rhs, fleet_coefficient = 0;
    for (auto [j, coefficient] : fleet_row.coefficients) {
        if (j == f.fleet) fleet_coefficient += coefficient;
        else required_fleet += static_cast<long double>(coefficient) * repaired[j];
    }
    if (fleet_row.sense != 'L' || fleet_coefficient != -1)
        return finish("unsupported_fleet_structure");
    repaired[f.fleet] = std::max(values[f.fleet], static_cast<double>(required_fleet) +
                               padding * std::max(1.0, std::abs(static_cast<double>(required_fleet))));
    if (!within_bounds(f.fleet)) return finish("fleet_bound_insufficient");
    report["after"] = point_residuals(f.matrix, repaired);
    if (!report.at("after").at("passed").get<bool>()) return finish("repaired_matrix_invalid");
    const double repaired_cost = objective_value(f.matrix, repaired);
    report["objective_after"] = repaired_cost;
    report["cost_increase"] = repaired_cost - original_cost;
    report["fleet_increase"] = repaired[f.fleet] - values[f.fleet];
    report["passed"] = true;
    report["applied"] = true;
    values = std::move(repaired);
    return finish("validated_inventory_repair");
}

Json solve_full(const Input& d,const PathCollection& paths,const SolverOptions& options,
                const std::vector<double>* fixed,const Json* start) {
    if (!paths.complete) throw std::invalid_argument("Full-path solving requires an enumerated complete path catalogue");
    const double begin=wall_time();
    if (options.progress) options.progress->record({{"event","phase_start"},{"stage","full_model_construction"}});
    auto f=build_formulation(d,paths,fixed,nullptr,nullptr,false,false,options.tight_airline_bounds,
                             formulation_strengthening(options));
    if (options.progress) options.progress->record({{"event","model_record_complete"},{"stage","full_model_construction"},
        {"seconds",wall_time()-begin},{"model_size",{{"variables",f.matrix.variables.size()},
            {"linear_rows",f.matrix.rows.size()},{"cones",f.matrix.cones.size()}}}});
    NativeSolver solver(f.matrix,options,"full_model");
    solver.set_incumbent_observer([&](const SolveOutcome& point) {
        auto values=point.values;
        const auto repair=repair_inventory_primal(f,values);
        if (!repair.at("passed").get<bool>()) {
            options.progress->record({{"event","incumbent_checkpoint_rejected"},{"stage","full_model"},
                {"reason","matrix_validation_failed"},{"residuals",repair.at("after")}});
            return;
        }
        auto candidate=solution_json(d,paths,f,values);
        const double upper=candidate.at("objective").get<double>();
        const double lower=std::max(0.0,point.bound+(fixed?candidate.at("opening_cost").get<double>():0));
        if (!std::isfinite(lower)||lower>upper+1e-6*(1+std::abs(upper))) return;
        candidate["bound"]=std::min(upper,lower);
        candidate["gap"]=(upper-std::min(upper,lower))/std::max(1.0,std::abs(upper));
        candidate["bound_scope"]=fixed?"fixed_design":"global";
        candidate["construction_complete"]=true;
        candidate["catalogue_complete"]=paths.complete;
        candidate["support_complete"]=paths.support_complete;
        candidate["pricing_complete"]=true;
        candidate["solved"]=false;
        candidate["termination_reason"]="running_with_validated_incumbent";
        candidate["native_model_validated"]=true;
        candidate["matrix_residuals"]=repair.at("after");
        candidate["primal_repair"]=repair;
        options.progress->checkpoint(candidate,"full_model_incumbent");
    });
    const double built=wall_time()-begin;
    if (start && !fixed) {
        auto design=solution_design(d,*start);
        solver.add_start(design,f.design_vars);
    }
    const auto outcome=solver.optimize(std::max(0.0,options.seconds-(wall_time()-begin)));
    if (fixed && !outcome.has_solution) {
        throw std::runtime_error("Fixed-design solve returned no feasible point; refusing an unrelated fallback design");
    }
    Json result=all_courier_result(d,paths);
    if (outcome.has_solution) {
        auto values = outcome.values;
        const auto repair = repair_inventory_primal(f, values);
        const auto residuals = repair.at("after");
        if (!repair.at("passed").get<bool>()) throw std::runtime_error("CPLEX primal failed matrix residual validation: "+repair.dump());
        auto candidate=solution_json(d,paths,f,values);
        const double candidate_cost=candidate.at("objective").get<double>();
        if (fixed || candidate_cost<=result.at("objective").get<double>()) {
            result=std::move(candidate);
            result["matrix_residuals"]=residuals;
            result["incumbent_kind"]="optimized";
        }
        result["solver_candidate_objective"]=candidate_cost;
        result["solver_candidate_residuals"]=residuals;
        result["primal_repair"]=repair;
        result["solver_raw_objective"]=outcome.objective;
    }
    result["tight_airline_bounds"]=options.tight_airline_bounds;
    if (f.strengthening.enabled()) result["structural_strengthening"] = f.strengthening_metadata;
    result["method"]=fixed?"fixed_design":"full_path";
    result["solver"]="CPLEX";
    result["solver_status"]=outcome.status;
    result["solver_status_code"]=outcome.cplex_status;
    result["has_solver_solution"]=outcome.has_solution;
    const double opening=result["opening_cost"].get<double>(), upper=result["objective"].get<double>();
    double lower=std::max(0.0,outcome.bound+(fixed?opening:0));
    if (lower>upper+1e-6*(1+std::abs(upper))) throw std::runtime_error("Solver bound exceeds validated incumbent");
    lower=std::min(lower,upper);
    result["bound"]=lower;
    result["gap"]=(upper-lower)/std::max(1.0,std::abs(upper));
    result["termination_reason"]=outcome.optimal
        ? (result["gap"].get<double>() <= options.gap ? "gap_limit" :
           result.value("primal_repair",Json::object()).value("applied",false)
              ? "optimal_with_repaired_incumbent" : "solver_optimal_validated_gap_above_target")
        : outcome.status;
    result["catalogue_complete"]=true;
    result["support_complete"]=paths.support_complete;
    result["pricing_complete"]=true;
    result["construction_complete"]=true;
    result["timings"]={{"build",built},{"optimize",outcome.seconds},{"total",wall_time()-begin}};
    result["model_size"]={{"variables",f.matrix.variables.size()},{"linear_rows",f.matrix.rows.size()},{"cones",f.matrix.cones.size()}};
    result["node_count"]=outcome.nodes;
    if (options.progress) {
        options.progress->record({{"event","global_bounds"},{"stage","full_model_finished"},
            {"objective",upper},{"upper_bound",upper},{"bound",lower},{"lower_bound",lower},
            {"gap",result.at("gap")},{"bound_scope",fixed?"fixed_design":"global"},{"validated_incumbent",true},
            {"solver_status",outcome.status},{"model_size",result.at("model_size")}});
        options.progress->checkpoint(result,"full_model_finished",true);
    }
    return result;
}
} // namespace mlnd
