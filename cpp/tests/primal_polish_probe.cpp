// Primal-only selected-column and fixed-assignment regression witnesses.
// These small physical fixtures do not change production benchmark instances.
#include "mlnd/heuristics.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> Matrix<T> filled(int rows, int cols, T value) {
    Matrix<T> out(rows,cols); std::fill(out.values.begin(),out.values.end(),value); return out;
}
Input fixture(int bases, int cardinality) {
    Input d; d.n_l=bases; d.n_h=2; d.n_od=2; d.n_levels=1; d.max_orders=cardinality;
    d.name="primal_polish_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={4,7}; d.ground_cost={20,20}; d.deadline={1,1};
    const int n=bases+2; d.flight_times=filled(n,n,.05);
    for(int u=0;u<n;++u)d.flight_times(u,u)=0;
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.allowed_links=filled(n,n,1); for(int u=0;u<n;++u)d.allowed_links(u,u)=0;
    d.link_cost=filled(n,n,.4); d.base_cost=std::vector<double>(bases,.5);
    d.slot_epsilon={.05,.05}; d.rider_mean=filled(2,2,.08);
    d.first_cost=filled(2,2,1.0); d.terminal_cost=filled(2,2,.5);
    d.receipt_mean=filled(2,2,.08); d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20); d.slot_cost=filled(2,1,.1); d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,1.0); d.delta=.04; d.operating_limit=1.5;
    d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.preprocessed_demand=true; d.pair_rates=filled(2,2,0.0);
    if(cardinality==1) { d.single_rates=d.demand; d.head_rates=d.demand; }
    else {
        d.single_rates={.5,4.5}; d.pair_rates(0,0)=.5;
        d.pair_rates(0,1)=2; d.pair_rates(1,0)=.5; d.head_rates={3,5};
    }
    d.common_origin_support=filled(2,bases,1); return d;
}
std::vector<double> generous_design(const Input& d) {
    DesignLayout layout(d); std::vector<double> x(layout.count);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h]=1;
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)x[layout.assignment_index(q,q%d.n_l)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
void check_wrapper(const Json& result, double maximum) {
    require(result.at("validated"),"Unvalidated primal result");
    require(result.at("objective").get<double>()<=maximum+1e-6,"Primal search degraded retained incumbent");
    require(!result.contains("bound")&&!result.contains("gap"),"Restricted bound leaked into wrapper");
    require(!result.at("solution").contains("bound")&&!result.at("solution").contains("gap"),
            "Restricted bound leaked into solution");
    require(!result.at("global_lower_bound_available").get<bool>(),"Primal search claimed global lower bound");
}
Json test_case(int bases,int cardinality) {
    const auto d=fixture(bases,cardinality); const auto paths=enumerate_paths(d);
    SolverOptions options; options.seconds=15; options.gap=1e-8; options.threads=1;
    options.neighborhood_cuts=false;
    options.grouped_local_branching=true;
    const auto design=generous_design(d);
    auto incumbent=solve_full(d,paths,options,&design);
    require(incumbent.at("has_solver_solution"),"Fixture has no fixed-design incumbent");
    require(incumbent.at("served").get<double>()>1,"Fixture has no positive drone flow");
    std::vector<int> empty_selection;
    const auto polished=solve_restricted_heuristic(d,paths,options,wall_time()+15,&incumbent,-1,
        &empty_selection,&empty_selection,true);
    check_wrapper(polished,incumbent.at("objective"));
    require(polished.at("assignment_fixed"),"Assignment polishing did not fix assignments");
    require(polished.at("solution").at("_solution").at("assignment")==incumbent.at("_solution").at("assignment"),
            "Assignment polishing changed assignment");
    require(polished.at("diagnostics").at("incumbent_validation").at("passed"),
            "Selected-column warm start was not fully validated");
    std::size_t positive_service=0,positive_empty=0;
    for(const auto& value:incumbent.at("_solution").at("flow"))positive_service+=value.get<double>()>0;
    for(const auto& value:incumbent.at("_solution").at("empty_flow"))positive_empty+=value.get<double>()>0;
    require(polished.at("column_selection").at("service").at("incumbent_columns_added")==positive_service,
            "Some strictly positive incumbent service columns were dropped");
    require(polished.at("column_selection").at("empty").at("incumbent_columns_added")==positive_empty,
            "Some strictly positive incumbent empty columns were dropped");
    require(polished.at("model_size").at("service_columns")==positive_service,
            "Selected-column model silently used the full catalogue");

    // A requested strict subset cannot silently remove a required positive
    // task branch. With only single-task columns pair tasks must stop independently.
    std::vector<int> singles;
    if(cardinality==2)for(std::size_t p=0;p<paths.service_paths.size();++p)
        if(paths.service_paths[p].second_od<0)singles.push_back(static_cast<int>(p));
    const auto unsupported=solve_restricted_heuristic(d,paths,options,wall_time()+15,nullptr,-1,
        &singles,&empty_selection,false);
    check_wrapper(unsupported,all_courier_result(d,paths).at("objective"));
    const auto& unsupported_solution=unsupported.at("solution").at("_solution");
    for(const auto& p:paths.service_paths)if(p.second_od>=0)
        require(unsupported_solution.at("flow").at(p.id).get<double>()<1e-8,"Missing pair task was invented");
    if(cardinality==2)require(unsupported.at("solution").at("served").get<double>()>0,
            "Unsupported pairs incorrectly disabled feasible single tasks");
    const auto full_matrix=build_formulation(d,paths);
    const auto subset_matrix=build_formulation(d,paths,nullptr,&singles,&empty_selection);
    require(full_matrix.matrix.rows.size()==subset_matrix.matrix.rows.size(),
            "Column restriction removed original task or physical constraints");
    for(std::size_t i=0;i<full_matrix.matrix.rows.size();++i)
        require(full_matrix.matrix.rows[i].name==subset_matrix.matrix.rows[i].name,
                "Column restriction changed original row identities");

    // The group limits (4 heads, 2 lockers) cover this entire tiny design
    // domain; this also exercises the ordinary local-branching entry point.
    const auto free_search=solve_restricted_heuristic(d,paths,options,wall_time()+15,&polished.at("solution"),2,
        &empty_selection,&empty_selection,false);
    check_wrapper(free_search,polished.at("objective"));
    require(!free_search.at("assignment_fixed").get<bool>(),"Free-assignment search retained fixings");
    require(free_search.contains("grouped_local_branching") &&
            free_search.at("grouped_local_branching").at("metric")=="categorical_state_changes",
            "Ordinary local branching did not install configured group distances");
    const auto local=solve_restricted_heuristic(d,paths,options,wall_time()+15,&incumbent,0,
        &empty_selection,&empty_selection,false);
    check_wrapper(local,incumbent.at("objective"));
    require(local.at("local_branching_radius")==0,"Local search lost its requested radius");
    require(!local.contains("grouped_local_branching"),"Radius zero was expanded to configured positive group radii");
    const auto local_design=solution_design(d,local.at("solution"));
    const auto center=solution_design(d,incumbent);
    const DesignLayout layout(d);
    for(int j=0;j<layout.airline;++j)
        require(local_design[j]==center[j],"Zero-radius neighborhood changed a design group");
    require(local.at("diagnostics").at("incumbent_validation").at("passed"),
            "Local branching did not preserve the verified positive-flow center");
    const auto missing=solve_restricted_heuristic(d,paths,options,wall_time()+15,nullptr,-1,
        &empty_selection,&empty_selection,true);
    require(missing.at("status")=="assignment_polish_requires_valid_incumbent",
            "Assignment fix accepted without a valid center");
    auto invalid=incumbent; invalid["_solution"]["courier"][0]=-1;
    const auto rejected=solve_restricted_heuristic(d,paths,options,wall_time()+15,&invalid,-1,
        &empty_selection,&empty_selection,true);
    require(rejected.at("status")=="assignment_polish_requires_valid_incumbent",
            "Invalid incumbent supplied assignment fixings");
    bool bad_index_rejected=false; std::vector<int> bad{-1};
    try { solve_restricted_heuristic(d,paths,options,wall_time()+15,nullptr,-1,&bad); }
    catch(const std::invalid_argument&) { bad_index_rejected=true; }
    require(bad_index_rejected,"Out-of-domain column index accepted");
    return {{"bases",bases},{"max_orders",cardinality},{"catalogue_columns",paths.service_paths.size()},
        {"positive_incumbent_columns",positive_service},{"fixed_assignment_objective",polished.at("objective")},
        {"incumbent_objective",incumbent.at("objective")},{"free_assignment_objective",free_search.at("objective")},
        {"missing_branch_served",unsupported.at("solution").at("served")},{"passed",true}};
}
}
int main() {
    try {
        Json cases=Json::array(); for(int bases:{1,2})for(int k:{1,2})cases.push_back(test_case(bases,k));
        std::cout<<Json({{"passed",true},{"cases",cases}}).dump(2)<<'\n'; return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
