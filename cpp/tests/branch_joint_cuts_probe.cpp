// Finite exhaustive checks without an optimizer. The independent oracle uses
// the complete catalogue and the explicit 2-by-2 reconstruction equations.
#include "mlnd/branch_joint_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
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
        std::size_t& floor_checks, const mlnd::BranchJointCache& branches) {
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
            for(int s=0;s<2;++s) if(rate[s]>0) {
                require(h[r]*P[r][s]+1e-12>=branches.positive_double_floor[r*2+s], "Double coefficient floor is not conservative");
                ++floor_checks;
            }
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
mlnd::Json exhaustive(int K,int L,bool sparse) {
    auto d=fixture(K,L);
    if(sparse) d.kappa.values={1,0,0,.4};
    const double forever=std::numeric_limits<double>::infinity();
    const auto all=mlnd::enumerate_paths(d),support=mlnd::prepare_path_support(d);
    const auto singles=mlnd::build_assignment_support_cache(d,support,forever);
    auto cache=mlnd::build_branch_joint_cache(d,support,singles,forever);
    require(cache.support_complete&&cache.numerical_domain_safe,"Complete branch support required");
    for(const auto& g:cache.double_groups) {
        if(cache.positive_double_floor[g.first_od*d.n_od+g.second_od]<=0) continue;
        require(g.complete,"Eligible tiny cache group is incomplete");
        std::vector<std::vector<int>> independent;
        for(const auto& p:all.service_paths) if(p.first_od==g.first_od&&p.second_od==g.second_od&&
                p.b[0]==g.origin_base&&p.b[1]==g.return_base) {
            std::vector<int> edges;
            for(auto [u,v,n]:p.arc_counts) {(void)n;edges.push_back(u*d.n_nodes()+v);}
            std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
            independent.push_back(edges);
        }
        std::sort(independent.begin(),independent.end());independent.erase(std::unique(independent.begin(),independent.end()),independent.end());
        require(independent==g.paths,"Double cache disagrees with independent complete enumeration");
    }
    const auto edges=arcs(d);
    const unsigned networks=1U<<edges.size(),assignments=(1+L*L)*(1+L*L);
    const mlnd::DesignLayout layout(d);
    std::map<std::string,mlnd::BranchJointCostCut> cuts;
    std::size_t issued=0,double_count=0,joint_count=0,aa_certificates=0,duplicate_union=0;
    for(unsigned mask=0;mask<networks;++mask) for(unsigned a=0;a<assignments;++a) {
        auto x=design(d,edges,mask,a);
        auto result=mlnd::separate_branch_joint_cost(d,support,singles,cache,x,layout.count,0,forever,true,true,1000);
        require(!result.timed_out,"Unlimited tiny branch separation timed out");
        for(const auto& c:result.certificates) {
            require(!supported(d,all,x,c.first_od,c.second_od,c.origin_base,c.return_base),"Certificate excludes an open task branch");
            if(c.first_od==c.second_od) {
                require(c.triggers.size()==1,"AA trigger counted twice");++aa_certificates;
            }
            for(const auto& p:all.service_paths) if(p.first_od==c.first_od&&p.second_od==c.second_od&&
                    p.b[0]==c.origin_base&&p.b[1]==c.return_base) {
                bool hit=false;
                for(auto [u,v,n]:p.arc_counts) {(void)n;hit=hit||std::binary_search(c.blocking_arcs.begin(),c.blocking_arcs.end(),std::make_pair(u,v));}
                require(hit,"Branch blocking set misses an original feasible task path");
            }
        }
        for(const auto& cut:result.cuts) {
            require(cut.valid&&cut.violated,"Only valid violated branch/joint cuts may be returned");
            require(cut.origin_bases.size()==std::set<int>(cut.origin_bases.begin(),cut.origin_bases.end()).size(),"Base counted twice");
            require(cut.triggers.size()==std::set<int>(cut.triggers.begin(),cut.triggers.end()).size(),"Trigger counted twice");
            require(cut.blocking_arcs.size()==std::set<std::pair<int,int>>(cut.blocking_arcs.begin(),cut.blocking_arcs.end()).size(),"Union arc counted twice");
            if(cut.kind=="joint_bases") {
                ++joint_count;std::size_t raw=0;
                for(const auto& c:cut.metadata.at("certificates")) raw+=c.at("blocking_arcs").size();
                duplicate_union+=raw>cut.blocking_arcs.size();
            } else ++double_count;
            cuts.emplace(cut.key,cut);++issued;
        }
    }
    std::size_t checks=0,floor_checks=0,positive=0;
    double worst=0;
    const double baseline=69;
    for(unsigned mask=0;mask<networks;++mask) for(unsigned a=0;a<assignments;++a) {
        auto x=design(d,edges,mask,a);
        const double lower=lower_cost(d,all,x,singles,floor_checks,cache);
        positive+=lower<baseline-1e-8;x.push_back(lower);
        for(const auto& [key,cut]:cuts) {(void)key;worst=std::max(worst,violation(cut.row,x));++checks;}
    }
    require(worst<=1e-7,"Branch/joint cut excludes a cross-network/assignment recourse lower envelope");
    require(positive>0,"Cross check must include positive service");
    require(K==1||double_count>0,"K2 must test individual double cuts");
    require(L==1||joint_count>0,"Two bases must test joint cuts");
    require(K==1||aa_certificates>0,"K2 must test AA certificates");
    return {{"K",K},{"bases",L},{"sparse_kappa",sparse},{"networks",networks},{"assignments",assignments},
        {"physical_paths",all.service_paths.size()},{"issued_cuts",issued},{"unique_cuts",cuts.size()},
        {"double_cuts",double_count},{"joint_cuts",joint_count},{"AA_certificates",aa_certificates},
        {"joint_repeated_arc_unions",duplicate_union},{"cross_checks",checks},{"floor_checks",floor_checks},
        {"positive_service_relaxations",positive},{"max_violation",worst}};
}
mlnd::Json boundaries() {
    auto d=fixture(2,2);
    const double forever=std::numeric_limits<double>::infinity();
    auto paths=mlnd::prepare_path_support(d);
    auto singles=mlnd::build_assignment_support_cache(d,paths,forever);
    auto cache=mlnd::build_branch_joint_cache(d,paths,singles,forever);
    const mlnd::DesignLayout layout(d);
    // OD0 -> (0,0), OD1 -> (1,1), all service arcs closed.
    auto x=design(d,arcs(d),0,21);
    auto separate=[&](double deadline=std::numeric_limits<double>::infinity(),bool doubles=true,bool joint=true,std::size_t cap=16,double theta=0) {
        return mlnd::separate_branch_joint_cost(d,paths,singles,cache,x,layout.count,theta,deadline,doubles,joint,cap);
    };
    auto both=separate();
    require(!both.cuts.empty()&&both.cuts.front().kind=="joint_bases","Joint cut must take priority");
    auto cap=separate(forever,true,true,1);
    require(cap.cuts.size()==1&&cap.cuts[0].kind=="joint_bases","Cut cap displaced the joint cut");
    auto noindividual=separate(forever,false,true,16,50);
    require(noindividual.cuts.size()==1&&noindividual.cuts[0].kind=="joint_bases","Nonviolated individual costs prevented violated joint cost");
    require(separate(forever,false,false).cuts.empty(),"Disabled flags emitted a cut");
    auto onlydouble=separate(forever,true,false);
    require(!onlydouble.cuts.empty()&&std::all_of(onlydouble.cuts.begin(),onlydouble.cuts.end(),[](auto& c){return c.kind=="double_branch";}),"Double-only mode produced joint cuts");
    require(separate(forever,true,true,0).cuts.empty(),"Zero cut cap ignored");
    require(separate(forever,true,true,16,69).cuts.empty(),"All-courier feasible point was cut");
    require(separate(mlnd::wall_time()-1).cuts.empty(),"Expired separation issued cuts");
    auto partial=cache;
    for(auto& g:partial.double_groups) {g.complete=false;g.paths.clear();g.next_candidate=0;g.candidates_checked=0;}
    partial.completed_groups=0;partial.candidates_checked=0;
    auto no_time=mlnd::separate_branch_joint_cost(d,paths,singles,partial,x,layout.count,0,mlnd::wall_time()-1,true,false);
    require(no_time.cuts.empty(),"Incomplete group with no time issued a cut");
    auto resumed=mlnd::separate_branch_joint_cost(d,paths,singles,partial,x,layout.count,0,forever,true,false);
    require(!resumed.cuts.empty()&&partial.completed_groups>0,"Lazy group completion failed");
    auto incomplete=paths;incomplete.support_complete=false;
    auto unsafe=mlnd::build_branch_joint_cache(d,incomplete,singles,forever);
    require(!unsafe.support_complete&&mlnd::separate_branch_joint_cost(d,incomplete,singles,unsafe,x,layout.count,0,forever).cuts.empty(),"Partial input support emitted a cut");
    auto fractional=x;fractional[layout.assignment]=.5;bool rejected=false;
    try {mlnd::separate_branch_joint_cost(d,paths,singles,cache,fractional,layout.count,0,forever);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Fractional design was accepted");
    auto prefix=d;prefix.flow_policy="prefix";rejected=false;
    try {mlnd::build_branch_joint_cache(prefix,paths,singles,forever);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Unsupported policy accepted");
    // A blocked single branch may still have double paths, and vice versa.
    // Complete branch cache is independent of current column pool contents.
    auto absent_pool=paths;absent_pool.service_paths.clear();
    auto same=mlnd::build_branch_joint_cache(d,absent_pool,singles,forever);
    require(same.completed_groups==cache.completed_groups,"Double support used generated columns");
    d.kappa=filled(2,2,0.0);paths=mlnd::prepare_path_support(d);
    singles=mlnd::build_assignment_support_cache(d,paths,forever);cache=mlnd::build_branch_joint_cache(d,paths,singles,forever);
    require(separate(forever,true,false).cuts.empty(),"Zero kappa produced mandatory double cuts");
    require(!separate(forever,false,true).cuts.empty(),"Zero kappa removed valid joint single certificates");
    d=fixture(2,2);d.delta=1e4;paths=mlnd::prepare_path_support(d);
    singles=mlnd::build_assignment_support_cache(d,paths,forever);cache=mlnd::build_branch_joint_cache(d,paths,singles,forever);
    require(!cache.numerical_domain_safe&&separate().cuts.empty(),"Saturated numeric probability was trusted");
    d=fixture(2,2);d.operating_limit=0;paths=mlnd::prepare_path_support(d);
    singles=mlnd::build_assignment_support_cache(d,paths,forever);cache=mlnd::build_branch_joint_cache(d,paths,singles,forever);
    auto impossible=separate(forever,true,false);
    require(!impossible.cuts.empty()&&impossible.cuts[0].blocking_arcs.empty(),"Global double impossibility lost empty-set certificate");
    return {{"joint_priority_and_cap",true},{"joint_without_individual_violation",true},{"flag_isolation",true},
        {"incomplete_and_expired_safe",true},{"lazy_completion",true},{"fractional_and_policy_rejected",true},
        {"generated_pool_independence",true},{"zero_kappa",true},{"saturation_screen",true},{"empty_blocking_set",true}};
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc>2) throw std::invalid_argument("Usage: branch_joint_cuts_probe [OUTPUT_JSON]");
        mlnd::Json out{{"cases",mlnd::Json::array()}};
        for(int k:{2}) for(int l:{1,2}) for(bool sparse:{false,true}) out["cases"].push_back(exhaustive(k,l,sparse));
        out["boundaries"]=boundaries();out["passed"]=true;out["solver_runs"]=0;
        if(argc==2) {std::ofstream file(argv[1]);if(!file)throw std::runtime_error("Cannot write branch/joint probe JSON");file<<out.dump(2)<<'\n';}
        std::cout<<out.dump(2)<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
