#include "mlnd/primal_alns.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool test,const char* message) {if(!test)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int r,int c,T value) {
    Matrix<T> m(r,c);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.name="alns_search_fixture";d.model_version="consecutive";
    d.n_l=2;d.n_h=3;d.n_od=2;d.n_levels=2;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand={4,7};d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(5,5,.05);d.allowed_links=filled(5,5,1);
    for(int j=0;j<5;++j){d.flight_times(j,j)=0;d.allowed_links(j,j)=0;}
    d.link_cost=filled(5,5,.4);d.base_cost={.5,.5};d.slot_epsilon={.05,.05,.05};
    d.rider_mean=filled(2,3,.08);d.first_cost=filled(2,3,1.0);d.terminal_cost=filled(2,3,.5);
    d.receipt_mean=filled(2,3,.08);d.removal_mean=filled(2,3,.03);
    d.slot_counts=filled(3,2,20);d.slot_cost=filled(3,2,.1);d.occupancy_limit=filled(3,2,14.0);
    for(int h=0;h<3;++h){d.slot_counts(h,1)=40;d.slot_cost(h,1)=.2;d.occupancy_limit(h,1)=28;}
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;d.preprocessed_demand=true;
    d.single_rates={.5,4.5};d.head_rates={3,5};d.pair_rates=filled(2,2,0.0);
    d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;
    d.common_origin_support=filled(2,2,1);return d;
}
}
int main(){try{
    // The stipulated initial temperature accepts a 1% worsening with p=1/2.
    require(std::abs(primal_alns_acceptance_probability(100,101,100,0,180)-.5)<1e-12,
        "Initial annealing probability is not one half for a one-percent deterioration");
    require(primal_alns_acceptance_probability(100,101,100,180,180)<1e-20,
        "Primal wall-time cooling did not reduce the final temperature");
    require(primal_alns_acceptance_probability(100,99,100,180,180)==1,
        "An improving feasible neighbor can be rejected");
    const auto d=fixture();const auto paths=enumerate_paths(d);auto initial=all_courier_result(d,paths);
    SolverOptions options;options.seconds=30;options.gap=.02;options.threads=1;options.neighborhood_cuts=false;
    PrimalAlnsSearch search(d,paths,options,initial,24);
    search.startup(wall_time());
    require(search.records().empty(),"Expired startup executed a repair");
    require(search.best().at("objective")==220,"Expired startup degraded all-courier fallback");
    double submitted=220;int submissions=0;
    const auto submit=[&](const Json& point){
        require(point.value("native_model_validated",false),"Search submitted unvalidated point");
        require(point.at("objective").get<double>()<submitted,"Submitted best is not monotone");
        submitted=point.at("objective");++submissions;
    };
    search.startup(wall_time()+15,submit);
    require(search.best().at("objective").get<double>()<200,"ALNS did not activate service from all courier");
    require(search.best().at("served").get<double>()>1,"ALNS startup has no positive drone service");
    require(submissions>0,"ALNS failed to submit improved incumbent immediately");
    const double before=search.best().at("objective");
    const auto elite_before=search.summary().at("elite_size");
    search.observe_validated(search.best());
    require(search.summary().at("elite_size")==elite_before,"Same design duplicated elite pool");
    search.step(wall_time()+5,nullptr,submit);
    require(search.best().at("objective").get<double>()<=before+1e-7,"Search replaced global best with a worse center");
    require(!search.best().contains("bound")&&!search.summary().at("global_lower_bound_available").get<bool>(),
        "ALNS leaked a local bound into global lower bound");
    require(search.used_seconds()>0,"Primal wall time was not charged");
    for(int pass=0;pass<5;++pass)search.step(wall_time()+2,nullptr,submit);
    const auto exercised=search.summary();
    for(const auto& op:exercised.at("operators"))
        require(op.at("attempts").get<int>()>0,"An available resource operator was never tried");
    const auto refine=search.take_elite_refinement();
    require(refine.has_value(),"A positive-service elite was not available for refinement");
    const auto other_refine=search.take_elite_refinement();
    require(other_refine.has_value(),"Fixture lacks two distinct eligible elite designs");
    require(!search.take_elite_refinement().has_value(),"More than two elite refinements were scheduled");
    // Extra idle fleet is an original-model feasible slack, not a fabricated
    // objective. Re-run physical/matrix validation without solver optimization
    // before offering these same-design but more expensive points to ALNS.
    const auto validated_slack=[&](const Json& base,double extra) {
        auto point=base;point.erase("native_model_validated");
        point["_solution"]["fleet"]=point.at("_solution").at("fleet").get<double>()+extra;
        synchronize_result_paths(d,paths,point);
        PrimalRepairRestrictions validation_only;validation_only.validation_reserve_seconds=10;
        auto checked=solve_restricted_heuristic(d,paths,options,wall_time()+2,&point,-1,
            nullptr,nullptr,false,false,nullptr,&validation_only);
        require(checked.at("diagnostics").at("incumbent_validation").at("passed"),
            "Fleet slack witness failed original physical/matrix validation");
        require(checked.at("status")=="validation_reserve_exhausted",
            "Validation witness unexpectedly optimized away its fleet slack");
        return checked.at("solution");
    };
    const auto expensive_a=validated_slack(*refine,20),expensive_b=validated_slack(*other_refine,19);
    PrimalAlnsSearch prune(d,paths,options,expensive_a,10);
    prune.observe_validated(expensive_b);
    require(prune.summary().at("elite_size")==2,"Fleet slack witness did not retain both nearby designs");
    prune.observe_validated(*other_refine);
    const auto pruned=prune.summary();
    require(pruned.at("elite_size")==1,"Same-signature best improvement left stale expensive elites");
    require(std::abs(pruned.at("elite_costs").at(0).get<double>()-other_refine->at("objective").get<double>())<1e-6,
        "Elite best pointer was not replaced by the cheaper same-design state");
    PrimalAlnsSearch replay_a(d,paths,options,search.best(),10),replay_b(d,paths,options,search.best(),10);
    replay_a.step(wall_time()+2);replay_b.step(wall_time()+2);
    require(replay_a.records().at(0).at("operator")==replay_b.records().at(0).at("operator") &&
        replay_a.records().at(0).at("selection").at("released_heads")==
            replay_b.records().at(0).at("selection").at("released_heads"),
        "The same input/center/seed did not replay the operator and head block");
    replay_a.synchronize_used_seconds(9);
    require(replay_a.used_seconds()>=9,"External elite CG time was omitted from primal cooling time");
    replay_a.synchronize_used_seconds(0);
    require(replay_a.used_seconds()>=9,"Shared primal accounting moved the cooling clock backward");
    std::cout<<Json({{"passed",true},{"best",search.best().at("objective")},{"submissions",submissions},
        {"summary",search.summary()},{"records",search.records()}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
