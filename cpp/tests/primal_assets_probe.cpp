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
    for(int r=0;r<d.n_od;++r)x[layout.assignment_index(r,r%d.n_l)]=1;
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

// The existing two-base physical regression fixture, with a deliberately slow
// departure base. This is a correctness witness, not a performance instance.
Json test_assets(int cardinality) {
    auto d=fixture(2,cardinality);
    d.fleet_cost=10;
    for(int h=0;h<d.n_h;++h)d.flight_times(1,d.n_l+h)=.50;
    const auto paths=enumerate_paths(d);
    SolverOptions options;options.seconds=15;options.gap=1e-8;options.threads=1;
    options.neighborhood_cuts=false;
    auto design=generous_design(d);const DesignLayout layout(d);
    for(int r=0;r<d.n_od;++r) {
        design[layout.assignment_index(r,0)]=0;
        design[layout.assignment_index(r,1)]=1;
    }
    const auto incumbent=solve_full(d,paths,options,&design);
    require(incumbent.at("has_solver_solution"),"No incumbent for asset neighborhood");
    require(incumbent.at("served").get<double>()>1,"Asset fixture has no drone service");
    const auto fixed_assignment=solve_restricted_heuristic(d,paths,options,wall_time()+15,
        &incumbent,-1,nullptr,nullptr,true);
    const auto assets=solve_restricted_heuristic(d,paths,options,wall_time()+15,
        &incumbent,-1,nullptr,nullptr,false,true);
    check_wrapper(assets,incumbent.at("objective"));
    require(assets.value("infrastructure_fixed",false),"Infrastructure neighborhood did not fix assets");
    require(!assets.at("assignment_fixed").get<bool>(),"Infrastructure neighborhood fixed assignments");
    const auto moved=solution_design(d,assets.at("solution"));
    require(std::equal(design.begin(),design.begin()+layout.assignment,moved.begin()),
        "Infrastructure neighborhood changed facility or capacity");
    require(!std::equal(design.begin()+layout.assignment,design.begin()+layout.airline,
        moved.begin()+layout.assignment),"Infrastructure neighborhood failed to change assignment");
    require(assets.at("objective").get<double>()+1 < fixed_assignment.at("objective").get<double>(),
        "Complementary neighborhood cannot escape fixed-assignment solution");
    require(assets.at("diagnostics").at("candidate_validation").at("passed"),
        "Returned asset-neighborhood point was not physically validated");
    const auto missing=solve_restricted_heuristic(d,paths,options,wall_time()+15,
        nullptr,-1,nullptr,nullptr,false,true);
    require(missing.at("status")=="infrastructure_polish_requires_valid_incumbent",
        "Infrastructure fix accepted without validated incumbent");
    auto invalid=incumbent;invalid["_solution"]["courier"][0]=-1;
    const auto rejected=solve_restricted_heuristic(d,paths,options,wall_time()+15,
        &invalid,-1,nullptr,nullptr,false,true);
    require(rejected.at("status")=="infrastructure_polish_requires_valid_incumbent",
        "Infrastructure fix accepted invalid incumbent");
    bool combined_rejected=false;
    try {solve_restricted_heuristic(d,paths,options,wall_time()+15,
        &incumbent,-1,nullptr,nullptr,true,true);}
    catch(const std::invalid_argument&){combined_rejected=true;}
    require(combined_rejected,"Conflicting fixed-assignment and fixed-infrastructure modes accepted");
    return {{"max_orders",cardinality},{"incumbent_objective",incumbent.at("objective")},
        {"fixed_assignment_objective",fixed_assignment.at("objective")},
        {"fixed_infrastructure_objective",assets.at("objective")},{"passed",true}};
}
}
int main(){try{
    Json cases=Json::array();for(int k:{1,2})cases.push_back(test_assets(k));
    std::cout<<Json({{"passed",true},{"cases",cases}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
