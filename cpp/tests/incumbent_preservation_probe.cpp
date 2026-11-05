#include "mlnd/benders.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.name="cg_reuse_physical_k2";d.model_version="consecutive";
    d.n_l=2;d.n_h=32;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={4,7};d.single_rates={.5,4.5};d.head_rates={3,5};
    d.pair_rates=filled(2,2,0.0);d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    // Dormant lockers retain the original 80 physical paths but make the
    // design layout cross the production construction-guard batch of 1024.
    d.flight_times=filled(34,34,.05);d.allowed_links=filled(34,34,0);
    for(int u=0;u<34;++u)d.flight_times(u,u)=0;
    for(int u=0;u<4;++u)for(int v=0;v<4;++v)d.allowed_links(u,v)=u!=v;
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(34,34,.01);d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,32,.08);d.first_cost=filled(2,32,1.0);
    d.terminal_cost=filled(2,32,.5);d.receipt_mean=filled(2,32,.08);
    d.removal_mean=filled(2,32,.03);d.slot_counts=filled(32,1,20);
    d.slot_cost=filled(32,1,.1);d.occupancy_limit=filled(32,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}

}
Json scenario(const std::filesystem::path& folder,bool cg){
    require(!std::filesystem::exists(folder),"Use a fresh evidence directory");
    const auto d=fixture();auto paths=enumerate_paths(d);const double courier=220.0; // (4+7) orders at ground cost 20
    SolverOptions options;options.seconds=20;options.seed=43;options.threads=1;options.preprocessed_demand=true;
    options.pareto=false;options.column_generation=cg;options.batch_columns=cg;options.cg_max_columns=1;
    options.continuous_root=false;
    options.root_projection=false;options.root_cuts=false;options.restricted_mip=false;
    options.neighborhood_cuts=false;options.primal_search=false;options.capacity_cuts=false;
    options.resource_lp_cuts=false;options.incumbent_cleanup=true;options.max_iterations=10;
    bool expired=false;
    auto clock=[&]() {
        const auto checkpoint=folder/"checkpoint.json";
        if(!expired && std::filesystem::is_regular_file(checkpoint)) {
            Json point;std::ifstream(checkpoint)>>point;
            if(point.value("checkpoint_stage",std::string())=="integer_rmp_feasible" &&
               point.value("native_model_validated",false) &&
               point.at("objective").get<double>()<courier-1e-5)expired=true;
        }
        return expired?21.0:0.0;
    };
    options.progress=std::make_shared<RunProgress>(folder,0,20,
        Json{{"problem_scope","global"},{"max_orders",2},{"assignment_scope","origin"}},clock);
    set_active_run_progress(options.progress);
    const auto result=solve_benders(d,paths,options);
    set_active_run_progress(nullptr);
    require(expired,"Regression did not expire after an improving validated point");
    Json checkpoint;std::ifstream(folder/"checkpoint.json")>>checkpoint;
    require(checkpoint.at("objective").get<double>()<courier-1e-5,"No nontrivial native feasible checkpoint");
    atomic_json_file(folder/"final_candidate.json",result);
    require(result.at("termination_reason")=="time_limit","Deadline did not reach production interruption path");
    require(result.at("objective").get<double>()<=checkpoint.at("objective").get<double>()+1e-6,
        "Final result regressed above its better validated checkpoint");
    require(result.value("native_model_validated",false),"Final incumbent lost its native feasibility proof");
    require(result.at("bound").get<double>()<=result.at("objective").get<double>()+1e-6,"Final incumbent conflicts with bound");
    validate_result_path_records(paths,result);
    const std::string expected_stage=cg?"cplex_column_batch_insertion":"model_record_construction";
    require(result.at("budget_interruption").at("stage")==expected_stage,
        "Deadline did not reach the intended production cleanup/CG insertion boundary");
    return Json{{"passed",true},{"mode",cg?"during_cg_update":"before_optional_cleanup"},
        {"all_courier",courier},{"checkpoint_objective",checkpoint.at("objective")},
        {"final_objective",result.at("objective")},{"interruption",result.at("budget_interruption")},
        {"clock_expired",expired},{"output_path_records_valid",true}};
}
int main(int argc,char** argv){try {
    require(argc==2,"Usage: mlnd_incumbent_preservation_probe NEW_DIRECTORY");
    const std::filesystem::path root=argv[1];require(!std::filesystem::exists(root),"Use a fresh evidence directory");
    const Json cases=Json::array({scenario(root/"cleanup",false),scenario(root/"cg_update",true)});
    std::cout<<Json{{"passed",true},{"cases",cases}}.dump(2)<<'\n';return 0;
} catch(const std::exception& e){set_active_run_progress(nullptr);std::cerr<<e.what()<<'\n';return 1;}}
