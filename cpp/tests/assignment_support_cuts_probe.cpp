// Finite exhaustive checks without an optimizer. The independent oracle uses
// the complete catalogue and the explicit 2-by-2 reconstruction equations.
#include "mlnd/assignment_support_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> mlnd::Matrix<T> filled(int rows, int columns, T value) {
    mlnd::Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
mlnd::Input fixture(int k, int bases) {
    mlnd::Input d;
    d.n_l = bases; d.n_h = 2; d.n_od = 2; d.n_levels = 1; d.max_orders = k;
    d.name = "assignment_support_probe"; d.model_version = "consecutive";
    d.route_mode = "direct"; d.flow_policy = "aggregate";
    d.demand = {4, 7}; d.ground_cost = {5, 7}; d.deadline = {1, 1};
    d.base_cost.assign(bases, .5); d.slot_epsilon = {.05, .05};
    const int N = d.n_nodes();
    d.flight_times = filled(N, N, .05);
    d.allowed_links = filled(N, N, bases == 1 ? 1 : 0);
    for (int i = 0; i < N; ++i) {
        d.flight_times(i, i) = 0; d.allowed_links(i, i) = 0;
    }
    if (bases == 2) for (auto [u, v] : std::vector<std::pair<int,int>>{
            {0, 2}, {1, 2}, {2, 3}, {3, 2}, {3, 0}, {3, 1}, {2, 0}, {0, 1}, {1, 0}})
        d.allowed_links(u, v) = 1;
    d.flight_times(bases, bases + 1) = d.flight_times(bases + 1, bases) = .1;
    d.link_cost = filled(N, N, 0.0);
    d.rider_mean = filled(2, 2, 0.0);
    d.first_cost = filled(2, 2, 1.0); d.terminal_cost = filled(2, 2, .5);
    d.receipt_mean = filled(2, 2, .08); d.removal_mean = filled(2, 2, .03);
    d.slot_counts = filled(2, 1, 10); d.slot_cost = filled(2, 1, .1);
    d.occupancy_limit = filled(2, 1, 6.0); d.kappa = filled(2, 2, 1.0);
    d.delta = .04; d.operating_limit = 1; d.fleet_cost = 1;
    d.flight_cost = .2; d.waiting_cost = .3;
    return d;
}
std::vector<std::pair<int,int>> arcs(const mlnd::Input& d) {
    std::vector<std::pair<int,int>> result;
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        if (u != v && !(u < d.n_l && v < d.n_l) && d.allowed_links(u,v)) result.emplace_back(u,v);
    return result;
}
std::vector<double> design(const mlnd::Input& d,
        const std::vector<std::pair<int,int>>& edges, unsigned network, unsigned assignment) {
    const mlnd::DesignLayout layout(d);
    std::vector<double> x(layout.count, 0);
    std::fill(x.begin(), x.begin() + d.n_nodes(), 1);
    for (int i = 0; i < d.n_h; ++i) x[layout.capacity + i] = 1;
    const int choices = 1 + d.n_l * d.n_l;
    for (int r = 0; r < d.n_od; ++r) {
        const int pair = assignment % choices - 1; assignment /= choices;
        if (pair >= 0) x[layout.assignment + r * d.n_l * d.n_l + pair] = 1;
    }
    for (std::size_t e = 0; e < edges.size(); ++e)
        x[layout.airline + edges[e].first * d.n_nodes() + edges[e].second] = (network >> e) & 1;
    // Empty rebalancing never changes order-path support.
    for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.n_l; ++b)
        if (l != b && d.allowed_links(l,b)) x[layout.airline + l * d.n_nodes() + b] = 1;
    return x;
}
bool enabled(const mlnd::Input& d, const mlnd::ServicePath& p, const std::vector<double>& x) {
    const mlnd::DesignLayout layout(d);
    return std::all_of(p.arc_counts.begin(), p.arc_counts.end(), [&](const auto& arc) {
        return x[layout.airline + std::get<0>(arc) * d.n_nodes() + std::get<1>(arc)] > .5;
    });
}
bool supported(const mlnd::Input& d, const mlnd::PathCollection& all,
        const std::vector<double>& x, int r, int s, int l, int b) {
    return std::any_of(all.service_paths.begin(), all.service_paths.end(), [&](const auto& p) {
        return p.first_od == r && p.second_od == s && p.b[0] == l && p.b[1] == b && enabled(d,p,x);
    });
}
// At fixed binary assignments solve (I+P^T)h=d exactly as a 2x2
// system. The extremal rho[l]=1 relaxation is allowed only when all positive
// reconstructed task branches have an open physical path. All operating costs
// other than courier cost are dropped. This is a lower bound on true recourse,
// not a claim that rho=1 has finite fleet size or satisfies balancing/capacity.
double lower_cost(const mlnd::Input& d, const mlnd::PathCollection& all,
        const std::vector<double>& x, const mlnd::AssignmentSupportCache& cache,
        std::size_t& floor_checks) {
    const mlnd::DesignLayout layout(d);
    std::vector<bool> available(d.n_l, true);
    for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.n_l; ++b) {
        double rate[2] = {}, P[2][2] = {};
        for (int r = 0; r < 2; ++r)
            rate[r] = d.demand[r] * x[layout.assignment + (r * d.n_l + l) * d.n_l + b];
        const double lambda = rate[0] + rate[1];
        if (lambda == 0) continue;
        const double q = -std::expm1(-d.effective_delta() * lambda);
        for (int r = 0; r < 2; ++r) for (int s = 0; s < 2; ++s) P[r][s] = q*d.kappa(r,s)*rate[s]/lambda;
        const double det = (1+P[0][0])*(1+P[1][1]) - P[0][1]*P[1][0];
        const double h[2] = {(rate[0]*(1+P[1][1])-rate[1]*P[1][0])/det,
                             (rate[1]*(1+P[0][0])-rate[0]*P[0][1])/det};
        for (int r = 0; r < 2; ++r) {
            require(h[r] >= -1e-12, "Independent reconstruction produced negative heads");
            if (rate[r] == 0) continue;
            const double single = h[r]*(1-P[r][0]-P[r][1]);
            require(single + 1e-12 >= cache.positive_single_floor[r], "Cached coefficient screen is not conservative");
            ++floor_checks;
            if (single > 1e-12 && !supported(d,all,x,r,-1,l,b)) available[l] = false;
            for (int s = 0; s < 2; ++s)
                if (h[r]*P[r][s] > 1e-12 && !supported(d,all,x,r,s,l,b)) available[l] = false;
        }
    }
    double courier = 0;
    for (int r = 0; r < 2; ++r) {
        bool served = false;
        for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.n_l; ++b)
            served = served || (available[l] && x[layout.assignment+(r*d.n_l+l)*d.n_l+b] > .5);
        if (!served) courier += d.demand[r]*d.ground_cost[r];
    }
    return courier;
}
double violation(const mlnd::RowRecord& row, const std::vector<double>& x) {
    long double lhs = 0;
    for (const auto& [j,c] : row.coefficients) lhs += static_cast<long double>(c)*x.at(j);
    return static_cast<double>(row.rhs-lhs);
}
mlnd::Json exhaustive(int K, int L, bool sparse) {
    auto d = fixture(K,L);
    if (sparse) d.kappa.values = {1,0,0,.4};
    const auto all = mlnd::enumerate_paths(d);
    const auto support = mlnd::prepare_path_support(d);
    require(support.service_paths.empty() && !support.complete, "Probe must not use generated path columns");
    const double forever = std::numeric_limits<double>::infinity();
    const auto cache = mlnd::build_assignment_support_cache(d,support,forever);
    require(cache.complete_domain && cache.numerical_domain_safe, "Complete safe fixture cache required");
    const auto edges = arcs(d);
    const unsigned networks = 1U << edges.size(), assignments = (1+L*L)*(1+L*L);
    const mlnd::DesignLayout layout(d);
    std::map<std::string,mlnd::AssignmentSupportCostCut> cuts;
    std::size_t blocked = 0, certified = 0, aa_paths = 0, positive_projections = 0;
    for (const auto& p : all.service_paths) aa_paths += p.second_od == p.first_od;
    for (unsigned mask = 0; mask < networks; ++mask) for (unsigned a = 0; a < assignments; ++a) {
        const auto x = design(d,edges,mask,a);
        const auto result = mlnd::separate_assignment_support_cost(d,cache,x,layout.count,0,forever,1000);
        require(!result.timed_out, "Unlimited tiny separation did not complete");
        for (int r = 0; r < 2; ++r) for (int l = 0; l < L; ++l) for (int b = 0; b < L; ++b) {
            const int index = (r*L+l)*L+b;
            if (x[layout.assignment+index] < .5 || supported(d,all,x,r,-1,l,b)) continue;
            ++blocked;
            const auto found = std::find_if(result.cuts.begin(),result.cuts.end(),[&](const auto& c) {
                return c.od==r && c.origin_base==l && c.return_base==b;
            });
            require(found != result.cuts.end(), "Missing independent single-support certificate");
        }
        for (const auto& cut : result.cuts) {
            require(cut.valid && cut.violated, "Only certified violated cuts may be emitted");
            require(!supported(d,all,x,cut.od,-1,cut.origin_base,cut.return_base), "Present single path was blocked");
            for (const auto& p : all.service_paths) if (p.first_od == cut.od && p.second_od<0 &&
                    p.b[0]==cut.origin_base && p.b[1]==cut.return_base) {
                bool hit = false;
                for (const auto& [u,v,count] : p.arc_counts) {
                    (void)count;
                    hit = hit || std::binary_search(cut.blocking_arcs.begin(),cut.blocking_arcs.end(),std::make_pair(u,v));
                }
                require(hit, "Blocking certificate misses a feasible single path");
            }
            for (const auto& [u,v] : cut.blocking_arcs) {
                require(x[layout.airline+u*d.n_nodes()+v]<.5, "Certificate contains an open arc");
                require(u>=L || v>=L, "Empty base-to-base arc entered single-support certificate");
            }
            cuts.emplace(cut.key,cut); ++certified;
        }
    }
    std::size_t checks = 0, floor_checks = 0;
    double max_violation = 0;
    const double baseline = std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    for (unsigned mask = 0; mask < networks; ++mask) for (unsigned a = 0; a < assignments; ++a) {
        auto x = design(d,edges,mask,a);
        const double lower = lower_cost(d,all,x,cache,floor_checks);
        positive_projections += lower < baseline-1e-8;
        x.push_back(lower);
        for (const auto& [key,cut] : cuts) {
            (void)key;
            max_violation = std::max(max_violation,violation(cut.row,x)); ++checks;
        }
    }
    require(max_violation <= 1e-7, "Cut excludes an independent cross-network/assignment recourse lower envelope");
    require(positive_projections > 0, "Cross checks tested only all-courier operation");
    require(K==1 || aa_paths>0, "K=2 domain must include same-OD consecutive tasks");
    return {{"K",K},{"bases",L},{"sparse_kappa",sparse},{"networks",networks},
        {"assignments",assignments},{"physical_paths",all.service_paths.size()},{"AA_paths",aa_paths},
        {"blocked_groups",blocked},{"issued_cuts",certified},{"unique_cuts",cuts.size()},
        {"cross_network_assignment_checks",checks},{"positive_service_relaxations",positive_projections},
        {"positive_flow_floor_checks",floor_checks},{"max_violation",max_violation}};
}
mlnd::Json boundaries() {
    auto d = fixture(2,2);
    const double forever = std::numeric_limits<double>::infinity();
    auto paths = mlnd::prepare_path_support(d);
    auto cache = mlnd::build_assignment_support_cache(d,paths,forever);
    const mlnd::DesignLayout layout(d);
    auto x = design(d,arcs(d),0,1); // OD0 assigned to (0,0), empty arcs open.
    auto separate = [&](double deadline = std::numeric_limits<double>::infinity()) {
        return mlnd::separate_assignment_support_cost(d,cache,x,layout.count,0,deadline);
    };
    require(separate().cuts.size()==1,"Closed network must trigger its single assigned OD");
    auto fractional=x; fractional[layout.assignment]=.5;
    bool fractional_rejected=false, policy_rejected=false, kappa_rejected=false;
    try { mlnd::separate_assignment_support_cost(d,cache,fractional,layout.count,0,forever); }
    catch(const std::invalid_argument&) { fractional_rejected=true; }
    auto invalid=d; invalid.flow_policy="prefix";
    try { mlnd::build_assignment_support_cache(invalid,paths,forever); }
    catch(const std::invalid_argument&) { policy_rejected=true; }
    invalid=d; invalid.kappa(0,1)=1.01;
    try { mlnd::build_assignment_support_cache(invalid,paths,forever); }
    catch(const std::invalid_argument&) { kappa_rejected=true; }
    require(fractional_rejected&&policy_rejected&&kappa_rejected,"Invalid domains were accepted");
    require(separate(mlnd::wall_time()-1).cuts.empty(),"Expired separation issued a certificate");
    auto timed_cache=mlnd::build_assignment_support_cache(d,paths,mlnd::wall_time()-1);
    require(timed_cache.timed_out&&!timed_cache.complete_domain,"Expired cache claimed completeness");
    require(mlnd::separate_assignment_support_cost(d,timed_cache,x,layout.count,0,forever).cuts.empty(),
            "Incomplete timed cache issued a certificate");
    auto incomplete=paths; incomplete.support_complete=false;
    auto partial=mlnd::build_assignment_support_cache(d,incomplete,forever);
    require(!partial.complete_domain && mlnd::separate_assignment_support_cost(d,partial,x,layout.count,0,forever).cuts.empty(),
            "Incomplete initial support issued a certificate");
    const double baseline=69;
    require(mlnd::separate_assignment_support_cost(d,cache,x,layout.count,baseline,forever).cuts.empty(),
            "Nonviolated all-courier cut should not be emitted");
    require(mlnd::separate_assignment_support_cost(d,cache,x,layout.count,0,forever,0).cuts.empty(),"Zero cut cap ignored");
    // A cross-base single path is a genuine support witness, even if its
    // empty balancing flight is absent. Assign OD0 to (0,1).
    x=design(d,arcs(d),0,2);
    for(auto [u,v]:std::vector<std::pair<int,int>>{{0,2},{2,3},{3,1}})
        x[layout.airline+u*d.n_nodes()+v]=1;
    x[layout.airline+1]=x[layout.airline+d.n_nodes()]=0;
    require(separate().cuts.empty(),"Cross-base witness incorrectly required empty balancing");
    // Physical impossibility gives the valid empty blocking set.
    d.operating_limit=0; paths=mlnd::prepare_path_support(d);
    cache=mlnd::build_assignment_support_cache(d,paths,forever);
    auto impossible=separate();
    require(impossible.cuts.size()==1&&impossible.cuts[0].blocking_arcs.empty(),"Global single impossibility lost empty-set certificate");
    d=fixture(2,2); d.kappa=filled(2,2,0.0);
    paths=mlnd::prepare_path_support(d); cache=mlnd::build_assignment_support_cache(d,paths,forever);
    x=design(d,arcs(d),0,1);
    require(separate().cuts.size()==1,"Zero kappa must preserve single-support implication");
    d.delta=1e4;
    paths=mlnd::prepare_path_support(d); cache=mlnd::build_assignment_support_cache(d,paths,forever);
    require(!cache.numerical_domain_safe&&separate().cuts.empty()&&separate().numerical_skips==1,
            "Rounded-to-one continuation coefficient did not disable numerical certificate");
    d=fixture(2,2); d.demand[0]=1e-8;
    paths=mlnd::prepare_path_support(d); cache=mlnd::build_assignment_support_cache(d,paths,forever);
    require(separate().cuts.empty()&&separate().numerical_skips==1,"Tiny mandatory-flow coefficient not screened");
    return {{"fractional_rejected",true},{"unsupported_policy_rejected",true},{"bad_kappa_rejected",true},
        {"expired_deadlines_safe",true},{"incomplete_domain_safe",true},{"empty_blocking_set",true},
        {"cross_base_and_empty_arcs",true},{"zero_kappa",true},{"saturated_probability_screen",true},
        {"small_flow_screen",true},{"nonviolated_and_zero_limit",true}};
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc>2) throw std::invalid_argument("Usage: assignment_support_cuts_probe [OUTPUT_JSON]");
        mlnd::Json report{{"cases",mlnd::Json::array()}};
        for(int K:{2}) for(int L:{1,2}) for(bool sparse:{false,true})
            report["cases"].push_back(exhaustive(K,L,sparse));
        report["boundaries"]=boundaries(); report["passed"]=true; report["solver_runs"]=0;
        if(argc==2) {
            std::ofstream out(argv[1]); if(!out) throw std::runtime_error("Cannot write probe JSON");
            out<<report.dump(2)<<'\n';
        }
        std::cout<<report.dump(2)<<'\n'; return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
