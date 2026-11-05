#include "mlnd/benders.hpp"
#include "mlnd/solver_presets.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F call,const char* message) {
    bool rejected=false;try{call();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,message);
}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.name="cg_reuse_physical_k2";d.model_version="consecutive";
    d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={4,7};d.single_rates={.5,4.5};d.head_rates={3,5};
    d.pair_rates=filled(2,2,0.0);d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);
    for(int u=0;u<4;++u){d.allowed_links(u,u)=0;d.flight_times(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.01);d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
SolverOptions settings() {
    SolverOptions options;options.seconds=10;options.threads=1;options.seed=43;
    options.preprocessed_demand=true;options.pareto=false;options.cg_max_columns=8;
    options.cg_seed_reuse=true;return options;
}

Json fixed_seed_tracer(){
    auto d=fixture();auto paths=enumerate_paths(d);auto o=settings();
    auto f=build_formulation(d,paths);NativeSolver native(f.matrix,o,"fixed_seed_reference");
    const auto ref=native.optimize(10);require(ref.has_solution && ref.optimal,"Reference integer optimum unavailable");
    std::vector<double> design;for(int j:f.design_vars)design.push_back(ref.values[j]>.5?1.0:0.0);
    CgColumnSeeds seeds;auto cold=solve_fixed_design_cg(d,paths,o,design,&seeds);
    require(!seeds.service().empty(),"Fixed CG did not retain physical columns");const auto count=seeds.service().size();
    auto warm=solve_fixed_design_cg(d,paths,o,design,&seeds);
    require(warm.at("initial_columns").at("method")=="reused_cg_columns" && warm.at("initial_service_columns")==count,
            "Fixed CG ignored actual retained columns");
    require(warm.at("native_model_validated") && warm.at("bound_valid"),"Reused fixed CG lost physical/certificate validation");
    require(std::abs(warm.at("objective").get<double>()-ref.objective)<1e-4,"Fixed CG changed known optimal design cost");
    require(warm.at("bound").get<double>()<=ref.objective+1e-4,"Fixed CG overstated certified bound");
    return {{"passed",true},{"reference",ref.objective},{"cold",cold.at("objective")},{"warm",warm.at("objective")},{"reused_columns",count}};
}
Json benders_tracer(){
    auto d=fixture();auto paths=enumerate_paths(d);auto o=settings();o.seconds=60;
    o.primal_alns=true;o.reuse_columns=true;o.initial_design=false;o.continuous_root=false;
    o.root_cuts=false;o.root_projection=false;o.restricted_mip=false;o.neighborhood_cuts=false;
    o.primal_search=false;o.capacity_cuts=false;o.resource_lp_cuts=false;o.incumbent_cleanup=true;
    const auto folder=std::filesystem::temp_directory_path()/("mlnd_alns_budget_probe_"+std::to_string(wall_time()));
    o.progress=std::make_shared<RunProgress>(folder,wall_time()-10,70,Json{{"problem_scope","global"}});
    set_active_run_progress(o.progress);
    auto result=solve_benders(d,paths,o);
    set_active_run_progress(nullptr);
    require(result.at("primal_alns_schedule").at("budget")==21.0,
            "Primal allowance used remaining time instead of original end-to-end budget");
    require(result.at("primal_alns_summary").at("enabled"),"Benders did not instantiate ALNS");
    require(!result.at("primal_alns_records").empty(),"Benders never offered all-courier seed repair");
    require(result.at("objective").get<double>()<100 && result.at("served").get<double>()>0,
            "Native ALNS integration failed the explicit feasible service fixture");
    require(result.at("native_model_validated") && result.at("bound").get<double>()<=result.at("objective").get<double>()+1e-5,
            "Integration violated original feasibility or bound ordering");
    require(result.at("primal_alns_schedule").at("seconds").get<double>()<=18.5,"Primal work exceeded unified allowance");
    return {{"passed",true},{"objective",result.at("objective")},{"bound",result.at("bound")},
        {"alns",result.at("primal_alns_summary")},{"schedule",result.at("primal_alns_schedule")}};
}
void preset_tracer(){
    SolverConfiguration c;c.apply_preset("speed-20261030");auto old=c.metadata();
    c.apply_preset("alns-20261030");c.validate();auto candidate=c.metadata();
    require(candidate.at("flags").at("primal_alns") && !old.at("flags").at("primal_alns").get<bool>(),"Preset is not opt in");
    candidate["flags"]["primal_alns"]=false;candidate["preset"]="speed-20261030";
    require(candidate==old,"ALNS preset changed unrelated flags or solver settings");
}
}
int main(){try {preset_tracer();std::cout<<Json{{"passed",true},{"fixed_seed",fixed_seed_tracer()},{"benders",benders_tracer()}}.dump(2)<<'\n';return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
