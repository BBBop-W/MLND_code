// Independent witnesses and complete tiny-network enumeration for the
// optimality-preserving support restrictions. These are unit fixtures, not
// new production benchmark instances.
#include "mlnd/path_support_cuts.hpp"
#include "mlnd/benders.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows, int cols, T value) {
    Matrix<T> result(rows, cols);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
Input fixture(int bases, int cardinality) {
    Input d; d.n_l=bases; d.n_h=2; d.n_od=2; d.n_levels=1; d.max_orders=cardinality;
    d.name="path_support_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={4,7}; d.ground_cost={2,20}; d.deadline={1,1};
    const int n=bases+2;
    d.flight_times=filled(n,n,.05);
    for (int u=0;u<n;++u) d.flight_times(u,u)=0;
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.allowed_links=filled(n,n,1);
    for (int u=0;u<n;++u) d.allowed_links(u,u)=0;
    d.link_cost=filled(n,n,.01); d.base_cost=std::vector<double>(bases,.5);
    d.slot_epsilon={.05,.05}; d.rider_mean=filled(2,2,.08);
    d.first_cost=filled(2,2,1.0); d.terminal_cost=filled(2,2,.5);
    d.receipt_mean=filled(2,2,.08); d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20); d.slot_cost=filled(2,1,.1);
    d.occupancy_limit=filled(2,1,14.0); d.kappa=filled(2,2,1.0);
    d.delta=.04; d.operating_limit=1.5;
    d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.preprocessed_demand=true; d.pair_rates=filled(2,2,0.0);
    if (cardinality==1) { d.single_rates=d.demand; d.head_rates=d.demand; }
    else {
        d.single_rates={.5,4.5}; d.pair_rates(0,0)=.5;
        d.pair_rates(0,1)=2; d.pair_rates(1,0)=.5; d.head_rates={3,5};
    }
    d.common_origin_support=filled(2,bases,1);
    return d;
}
std::vector<double> design(const Input& d, int first, int second, bool open=true) {
    const DesignLayout layout(d); std::vector<double> x(layout.count);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for (int h=0;h<d.n_h;++h) x[layout.capacity+h]=1;
    for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v) &&
            (open || (u<d.n_l && v<d.n_l));
    if (first>=0) x[layout.assignment_index(0,first)]=1;
    if (second>=0) x[layout.assignment_index(1,second)]=1;
    return x;
}
double lhs(const RowRecord& row, const std::vector<double>& x) {
    double value=0; for (auto [j,c]:row.coefficients) value+=c*x.at(j); return value;
}
double violation(const RowRecord& row, const std::vector<double>& x) {
    const double difference=lhs(row,x)-row.rhs;
    return row.sense=='L'?difference:row.sense=='G'?-difference:std::abs(difference);
}
double objective(const LinearModel& m, const std::vector<double>& x) {
    double out=0; for (std::size_t j=0;j<x.size();++j) out+=m.variables[j].cost*x[j]; return out;
}
bool physical_support(const PathCollection& paths, const Input& d,
                      const std::vector<double>& x, int r, int l, int s) {
    const DesignLayout layout(d); const int n=d.n_nodes();
    for (const auto& p:paths.service_paths) {
        if (p.first_od!=r || p.b[0]!=l || p.second_od!=s) continue;
        bool supported=true;
        for (auto [u,v,count]:p.arc_counts) {
            (void)count;
            if (x[layout.airline+u*n+v]<.5) { supported=false; break; }
        }
        if (supported) return true;
    }
    return false;
}
bool head_support(const PathCollection& paths, const Input& d,
                  const std::vector<double>& x, int r, int l) {
    if (d.single_rates[r]>0 && !physical_support(paths,d,x,r,l,-1)) return false;
    for (int s=0;s<d.n_od;++s)
        if (d.pair_rates(r,s)>0 && !physical_support(paths,d,x,r,l,s)) return false;
    return true;
}
std::vector<double> zero_service_point(const Formulation& f, const Input& d,
                                       const std::vector<double>& design_values) {
    auto x=design_values; x.resize(f.matrix.variables.size());
    for (int r=0;r<d.n_od;++r) x[f.courier[r]]=d.demand[r];
    for (const auto& cone:f.matrix.cones) x[cone.up]=x[cone.down]=x[cone.one]=1;
    return x;
}
Json canonical_witness() {
    const auto d=fixture(2,2); const auto paths=enumerate_paths(d);
    const auto cache=build_path_support_cache(d,paths,INFINITY);
    const auto f=build_formulation(d,paths);
    auto x=zero_service_point(f,d,design(d,0,1,false));
    require(point_residuals(f.matrix,x).at("passed"),"Original zero-service witness is infeasible");
    const auto separated=separate_path_support(d,cache,
        std::vector<double>(x.begin(),x.begin()+f.layout.count),INFINITY,100);
    require(!separated.cuts.empty(),"Zero-service empty network did not yield support cuts");
    double largest=0; for (const auto& cut:separated.cuts) largest=std::max(largest,violation(cut.row,x));
    require(largest>.99,"Witness must violate a dominance restriction in the original feasible set");
    auto canonical=x;
    for (int r=0;r<d.n_od;++r) for (int l=0;l<d.n_l;++l)
        canonical[f.layout.assignment_index(r,l)]=0;
    require(point_residuals(f.matrix,canonical).at("passed"),"Clearing zero-rho assignments changed feasibility");
    require(std::abs(objective(f.matrix,x)-objective(f.matrix,canonical))<1e-12,
            "Clearing zero-rho assignments changed objective");
    for (const auto& cut:separated.cuts)
        require(violation(cut.row,canonical)<1e-12,"Canonical witness violates support restriction");
    return {{"original_feasible",true},{"original_support_violation",largest},
        {"canonical_feasible",true},{"objective_change",0},
        {"operational_variables_unchanged",true}};
}
Json cross_return_witness() {
    auto d=fixture(2,2);
    // No service path can return to base 0; base-to-base empty relocation
    // remains available to close the drone network.
    for (int u=d.n_l;u<d.n_nodes();++u) d.allowed_links(u,0)=0;
    const auto paths=enumerate_paths(d); const auto cache=build_path_support_cache(d,paths,INFINITY);
    auto f=build_formulation(d,paths);
    auto x=design(d,0,-1); x.resize(f.matrix.variables.size());
    std::map<std::string,int> vars;
    for (std::size_t j=0;j<f.matrix.variables.size();++j) vars[f.matrix.variables[j].name]=j;
    const double rho=.3; x[f.rho[0]]=rho; x[f.idle[0]]=rho/(1-rho);
    x[vars.at("v_0_0")]=rho;
    std::vector<double> heads(f.lift_groups.size()),served(d.n_od); double busy=0,total=0;
    for (int s=-1;s<d.n_od;++s) {
        const double rate=s<0?d.single_rates[0]:d.pair_rates(0,s); if (rate==0) continue;
        require(!std::any_of(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p) {
            return p.first_od==0 && p.second_od==s && p.b==BasePair{0,0};
        }),"Cross-return witness unexpectedly has an own-base return path");
        const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p) {
            return p.first_od==0 && p.second_od==s && p.b==BasePair{0,1};
        });
        require(p!=paths.service_paths.end(),"Cross-return witness missing a positive task path");
        const double flow=rate*rho; x[f.flow[p->id]]=flow; x[f.head[p->segment_id]]+=flow;
        heads[f.segment_group[p->segment_id]]+=flow; total+=flow; busy+=flow*p->time;
        for (auto [r,count]:p->served_counts) served[r]+=flow*count;
    }
    const auto empty=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),
        [](const EmptyPath& p) { return p.b==BasePair{1,0}; });
    require(empty!=paths.empty_paths.end(),"Cross-return witness lacks empty repositioning");
    x[f.empty_flow[empty->id]]=total; busy+=total*empty->time;
    for (std::size_t g=0;g<heads.size();++g) x[vars.at("group_head_"+std::to_string(g))]=heads[g];
    for (int r=0;r<d.n_od;++r) x[f.courier[r]]=d.demand[r]-served[r];
    x[f.fleet]=busy+x[f.idle[0]];
    for (std::size_t l=0;l<f.rho.size();++l) {
        const auto cone=f.matrix.cones[l];
        x[cone.up]=x[f.idle[l]]+1; x[cone.down]=1-x[f.rho[l]]; x[cone.one]=1;
    }
    const auto residual=point_residuals(f.matrix,x);
    require(residual.at("passed"),"Cross-return/AA/incoming-second witness failed: "+residual.dump());
    require(served[1]>0 && x[f.layout.assignment_index(1,0)]==0 && x[f.layout.assignment_index(1,1)]==0,
            "Second OD should be served without its own assignment");
    auto redundant=x; redundant[f.layout.assignment_index(1,1)]=1;
    require(point_residuals(f.matrix,redundant).at("passed"),
            "Zero-rho assignment should coexist with incoming-second service");
    require(std::abs(objective(f.matrix,redundant)-objective(f.matrix,x))<1e-12,
            "Clearing zero-rho assignment changed incoming-second cost");
    const auto sep=separate_path_support(d,cache,
        std::vector<double>(x.begin(),x.begin()+f.layout.count),INFINITY,100);
    require(sep.cuts.empty(),"Origin support mistakenly requires same-base return or second-OD assignment");
    return {{"passed",true},{"served",served},{"all_service_returns_differ_from_origin",true},
        {"empty_repositioning_flow",total},{"AA_supported",true},{"second_od_unassigned",true},
        {"incoming_second_unchanged_after_zero_rho_assignment_clear",true},
        {"residuals",residual}};
}
Json guards() {
    auto d=fixture(1,2); auto paths=enumerate_paths(d);
    auto partial=paths; partial.complete=false;
    auto cache=build_path_support_cache(d,partial,INFINITY);
    require(!cache.complete_domain && cache.static_cuts.empty(),"Partial catalogue issued static cuts");
    require(separate_path_support(d,cache,design(d,0,0,false),INFINITY,100).cuts.empty(),
            "Partial catalogue issued dynamic cuts");
    partial=paths; partial.support_complete=false;
    require(!build_path_support_cache(d,partial,INFINITY).complete_domain,
            "Incomplete row support was certified");
    cache=build_path_support_cache(d,paths,wall_time()-1);
    require(!cache.complete_domain && cache.static_cuts.empty(),"Timed-out preprocessing issued static cuts");
    cache=build_path_support_cache(d,paths,INFINITY);
    require(separate_path_support(d,cache,design(d,0,0,false),wall_time()-1,100).cuts.empty(),
            "Expired separation issued an unchecked certificate");
    auto fractional=design(d,0,0,false); fractional[DesignLayout(d).assignment_index(0,0)]=.5;
    bool rejected=false;
    try { (void)separate_path_support(d,cache,fractional,INFINITY,100); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"Integer separator accepted fractional assignment");
    require(separate_path_support(d,cache,design(d,0,0,false),INFINITY,0).cuts.empty(),
            "Zero cut budget was ignored");
    d.single_rates={0,5}; d.pair_rates=filled(2,2,0.0);
    d.pair_rates(0,0)=1; d.pair_rates(0,1)=2; d.head_rates={3,5};
    cache=build_path_support_cache(d,paths,INFINITY);
    require(std::none_of(cache.groups.begin(),cache.groups.end(),[](const PathSupportGroup& g) {
        return g.head==0 && g.branch==-1;
    }),"Zero single demand incorrectly requires a singleton route");
    require(std::any_of(cache.groups.begin(),cache.groups.end(),[](const PathSupportGroup& g) {
        return g.head==0 && g.branch==0;
    }),"Positive AA branch omitted");
    require(cache.groups.size()==3,"Only positive task rates should create support groups");
    return {{"partial_catalogue_guard",true},{"incomplete_support_guard",true},
        {"deadline_guards",true},{"fractional_input_rejected",true},
        {"zero_cut_budget",true},{"zero_single_ignored",true},{"positive_AA_checked",true}};
}
Json enumerate_and_compare(int bases, int cardinality, bool impossible_order) {
    auto d=fixture(bases,cardinality);
    if (impossible_order) d.deadline[1]=1e-5;
    const auto paths=enumerate_paths(d); const auto cache=build_path_support_cache(d,paths,INFINITY);
    require(cache.complete_domain && !cache.timed_out,"Complete tiny catalogue was not certified");
    const DesignLayout layout(d); const int n=d.n_nodes();
    std::vector<std::pair<int,int>> arcs;
    for (int u=0;u<n;++u) for (int v=0;v<n;++v)
        if (d.allowed_links(u,v) && (u>=bases || v>=bases)) arcs.emplace_back(u,v);
    std::map<std::string,PathSupportCut> cuts;
    for (const auto& c:cache.static_cuts) cuts.emplace(c.key,c);
    std::size_t networks=0,blocked=0,coverage_checks=0,canonical_checks=0;
    for (unsigned mask=0;mask<(1U<<arcs.size());++mask) {
        for (int l=0;l<bases;++l) {
            auto x=design(d,l,l);
            for (std::size_t k=0;k<arcs.size();++k)
                x[layout.airline+arcs[k].first*n+arcs[k].second]=(mask>>k)&1U;
            const auto separated=separate_path_support(d,cache,x,INFINITY,100);
            require(separated.complete_domain && !separated.timed_out,"Tiny separation was incomplete");
            bool lacks=false;
            for (int r=0;r<d.n_od;++r) lacks=lacks||!head_support(paths,d,x,r,l);
            require(lacks==!separated.cuts.empty(),"Separator differs from full physical path scan");
            ++networks; blocked+=lacks;
            auto canonical=x;
            if (lacks) for (int r=0;r<d.n_od;++r) canonical[layout.assignment_index(r,l)]=0;
            for (const auto& c:separated.cuts) {
                require(c.valid && c.violated && violation(c.row,x)>.99,"Returned support cut does not cut anchor");
                require(violation(c.row,canonical)<1e-12,"Support cut rejects canonical design");
                ++canonical_checks;
                require(!physical_support(paths,d,x,c.head,c.origin,c.branch),"Cut trigger actually has a path");
                for (const auto& p:paths.service_paths) {
                    if (p.first_od!=c.head || p.second_od!=c.branch || p.b[0]!=c.origin) continue;
                    bool hit=false;
                    for (auto [u,v,count]:p.arc_counts) {
                        (void)count;
                        hit=hit||std::find(c.blocking_arcs.begin(),c.blocking_arcs.end(),std::pair<int,int>{u,v})!=c.blocking_arcs.end();
                    }
                    require(hit,"Blocking certificate does not cover a full-domain path"); ++coverage_checks;
                }
                for (auto [u,v]:c.blocking_arcs)
                    require(x[layout.airline+u*n+v]==0,"Blocking certificate contains an open arc");
                cuts.emplace(c.key,c);
            }
        }
    }
    if (impossible_order) require(!cache.static_cuts.empty(),"Physically impossible task not statically filtered");
    auto ordinary=build_formulation(d,paths); auto supported=ordinary;
    for (const auto& [key,c]:cuts) {
        (void)key; supported.matrix.row(c.row.name,c.row.sense,c.row.rhs,c.row.coefficients);
    }
    SolverOptions options; options.seconds=15; options.threads=1; options.gap=1e-9;
    NativeSolver reference(ordinary.matrix,options,"path_support_original");
    NativeSolver strengthened(supported.matrix,options,"path_support_canonical");
    const auto a=reference.optimize(15),b=strengthened.optimize(15);
    require(a.optimal && b.optimal && a.has_solution && b.has_solution,"Tiny equivalence MISOCP not solved to optimality");
    require(point_residuals(ordinary.matrix,a.values).at("passed"),"Original tiny incumbent invalid");
    require(point_residuals(supported.matrix,b.values).at("passed"),"Supported tiny incumbent invalid");
    const double difference=std::abs(a.objective-b.objective);
    require(difference<1e-5,"Support dominance restrictions changed tiny integer optimum");
    return {{"bases",bases},{"task_cardinality",cardinality},{"physically_impossible_order",impossible_order},
        {"complete_paths",paths.service_paths.size()},{"networks_and_assignments_checked",networks},
        {"blocked_cases",blocked},{"full_path_coverage_checks",coverage_checks},
        {"canonical_cut_checks",canonical_checks},{"static_cuts",cache.static_cuts.size()},
        {"distinct_support_cuts",cuts.size()},{"original_objective",a.objective},
        {"supported_objective",b.objective},{"objective_difference",difference}};
}
Json benders_integration(bool impossible_order) {
    auto d=fixture(impossible_order?2:1,impossible_order?1:2);
    if (impossible_order) d.deadline[1]=1e-5;
    auto paths=enumerate_paths(d); auto f=build_formulation(d,paths);
    SolverOptions options; options.seconds=5; options.gap=1e-7; options.threads=1;
    options.preprocessed_demand=true; options.path_support_cuts=true;
    options.pareto=false; options.max_iterations=1000;
    options.initial_design=options.restricted_mip=options.primal_search=options.neighborhood_cuts=false;
    options.continuous_root=options.root_cuts=options.root_projection=false;
    options.capacity_cuts=options.resource_lp_cuts=options.resource_bounds=false;
    NativeSolver reference(f.matrix,options,"path_support_integration_reference");
    const auto solved=reference.optimize(5);
    require(solved.optimal && solved.has_solution,"Integration reference did not solve");
    const auto result=solve_benders(d,paths,options);
    require(result.contains("objective") && std::isfinite(result.at("objective").get<double>()),
            "Support Benders lost finite incumbent");
    require(result.at("bound").get<double>()<=solved.objective+1e-4,
            "Support Benders lower bound exceeds full integer optimum");
    require(result.at("objective").get<double>()>=solved.objective-1e-4,
            "Support Benders incumbent undercuts full integer optimum");
    const auto summary=result.at("path_support_summary");
    require(summary.at("enabled") && summary.at("complete_domain"),"Support integration was not enabled/certified");
    if (impossible_order) require(summary.at("static_filters").get<int>()>0,"Static filters not added to Benders master");
    else require(summary.at("dynamic_cuts").get<int>()>0,"Dynamic support cuts not added to Benders master");
    require(summary.at("master_added_variables").get<int>()==0,"Support cuts unexpectedly add master variables");
    if (impossible_order) require(result.at("root_diagnostics").at("scope")=="optimality_preserving_support_root",
            "Canonical root restriction incorrectly labelled as original continuous relaxation");
    return {{"physically_impossible_order",impossible_order},{"full_objective",solved.objective},
        {"objective",result.at("objective")},{"bound",result.at("bound")},
        {"gap",result.at("gap")},{"summary",summary},
        {"root_scope",result.at("root_diagnostics").at("scope")}};
}
}
int main() { try {
    const auto started=wall_time(); Json cases=Json::array();
    const auto guard_results=guards(); const auto normalization=canonical_witness();
    const auto returns=cross_return_witness();
    for (int l:{1,2}) for (int k:{1,2}) for (bool impossible:{false,true})
        cases.push_back(enumerate_and_compare(l,k,impossible));
    const auto integration=Json::array({benders_integration(false),benders_integration(true)});
    std::cout<<Json{{"passed",true},{"guards",guard_results},{"canonical_witness",normalization},
        {"cross_return_witness",returns},{"cases",cases},{"benders_integration",integration},
        {"seconds",wall_time()-started}}.dump(2)<<'\n';
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
