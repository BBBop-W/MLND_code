#include "mlnd/design_face_cuts.hpp"
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
Json reuse_tracer() {
    const auto d=fixture();auto paths=enumerate_paths(d);auto options=settings();CgColumnSeeds seeds;
    const auto cold=solve_continuous_cg(d,paths,options,&seeds);
    require(cold.at("primal_valid")&&cold.at("pricing_complete"),"Cold continuous CG did not close the tiny fixture");
    require(!seeds.service().empty(),"Continuous CG did not retain its actual service columns");
    const auto kept=seeds.service().size();
    const auto warm=solve_continuous_cg(d,paths,options,&seeds);
    require(warm.at("initial_columns").at("service")==kept,"Warm CG did not start with its retained columns");
    require(warm.at("initial_columns").at("method")=="reused_cg_columns", "Reuse was not identified at the public result");
    require(warm.at("primal_valid")&&warm.at("pricing_complete"),"Warm continuous CG lost full-domain closure");
    auto full=build_formulation(d,paths,nullptr,nullptr,nullptr,true,false,options.tight_airline_bounds,
                                formulation_strengthening(options));
    for(auto& v:full.matrix.variables)v.binary=false;
    NativeSolver solver(full.matrix,options,"cg_reuse_full_reference");const auto solved=solver.optimize(10);
    require(solved.optimal&&solved.has_solution&&point_residuals(full.matrix,solved.values).at("passed"),
            "Full reference did not certify an original feasible optimum");
    const double optimum=objective_value(full.matrix,solved.values),tol=1e-5*(1+std::abs(optimum));
    for(const auto* result:{&cold,&warm}) {
        require(result->at("bound_valid"),"CG run lost certified full-domain bound");
        require(std::abs(result->at("continuous_objective").get<double>()-optimum)<tol,
                "Seed reuse changed the continuous optimum");
        require(result->at("certified_lower_bound").get<double>()<=optimum+tol,
                "Warm-started CG reported an invalid bound");
    }
    return {{"passed",true},{"full_optimum",optimum},{"retained_service",kept},
        {"cold_iterations",cold.at("cg_iterations")},{"warm_iterations",warm.at("cg_iterations")}};
}
Json face_tracer() {
    const auto d=fixture();auto paths=enumerate_paths(d);auto options=settings();
    auto full=build_formulation(d,paths);NativeSolver solver(full.matrix,options,"face_integer_floor");
    const auto solved=solver.optimize(10);
    require(solved.optimal&&solved.has_solution&&point_residuals(full.matrix,solved.values).at("passed"),
            "Independent integer optimum did not provide a certified global floor");
    const double floor=solved.bound-1e-5*(1+std::abs(solved.bound));
    std::vector<double> design;
    for(int j:full.design_vars)design.push_back(solved.values[j]);
    CgColumnSeeds seeds;
    const auto separated=separate_design_face(d,paths,options,0,design,0,full.layout.count,floor,20,&seeds);
    require(separated.faces.size()==3&&separated.cut.valid,"Face separation failed to package all three states");
    int stopped=0;
    for(int state=0;state<3;++state) {
        const auto& face=separated.faces.at(state);
        require(face.at("head")==0&&face.at("state")==state&&face.at("face_fix_rows")==2,
                "Column reuse transferred the previous face restriction");
        require(!face.at("direct_global_bound_update").get<bool>(),"A face value escaped its conditional scope");
        if(state>0)require(face.at("initial_columns").at("method")=="reused_cg_columns",
                "Adjacent face did not reuse physical columns");
        if(face.at("termination_reason")=="face_cannot_strengthen_global_floor") {
            ++stopped;
            require(face.at("primal_valid")&&face.at("continuous_objective").get<double>()<=floor,
                    "No-gain face stop lacked a verified feasible continuous upper bound");
            require(!face.at("bound_valid").get<bool>()&&!face.at("pricing_complete").get<bool>()&&
                    face.at("certified_lower_bound").is_null(),"No-gain primal was promoted to a lower-bound certificate");
            require(separated.cut.certificate.at("face_bounds").at(state)==floor,
                    "Stopped face did not fall back to the caller's certified global floor");
        }
    }
    require(stopped>0,"No face stopped after its verified upper bound could not improve the global floor");
    return {{"passed",true},{"global_integer_floor",floor},{"stopped_states",stopped},
        {"retained_service",seeds.service().size()}};
}
Json guard_tracer() {
    auto d=fixture();auto paths=enumerate_paths(d);CgColumnSeeds seeds;
    seeds.include(d,paths,{0},{0});const auto original=seeds.service();
    auto foreign_input=d;auto foreign_paths=paths;
    rejects([&]{seeds.validate(foreign_input,paths);},"Foreign input accepted old column indices");
    rejects([&]{seeds.validate(d,foreign_paths);},"Foreign catalogue accepted old column indices");
    rejects([&]{seeds.include(d,paths,{-1},{});},"Negative seed index was accepted");
    rejects([&]{seeds.include(d,paths,{}, {static_cast<int>(paths.empty_paths.size())});},
            "Out-of-range seed index was accepted");
    require(seeds.service()==original,"Rejected seed insertion changed the cache");
    const double original_rate=d.single_rates[0];d.single_rates[0]+=1;
    rejects([&]{seeds.validate(d,paths);},"Changed task data accepted stale column seeds");
    d.single_rates[0]=original_rate;
    const auto original_path=paths.service_paths[0];paths.service_paths[0].cost+=1;
    rejects([&]{seeds.validate(d,paths);},"Changed registered path coefficients accepted stale seeds");
    paths.service_paths[0]=original_path;
    auto incremental=prepare_path_support(d);CgColumnSeeds growing;
    const int first=register_service_path(incremental,paths.service_paths[0]);
    growing.include(d,incremental,{first},{});
    const int second=register_service_path(incremental,paths.service_paths[1]);
    require(second!=first,"Append-only guard fixture did not add a distinct physical path");
    growing.validate(d,incremental);growing.include(d,incremental,{second},{});
    incremental.service_paths.pop_back();
    rejects([&]{growing.validate(d,incremental);},"Shrinking catalogue accepted stale seeds");
    auto options=settings();options.seconds=0;
    const auto zero=solve_continuous_cg(d,paths,options,&seeds);
    require(!zero.at("bound_valid").get<bool>()&&!zero.at("primal_valid").get<bool>()&&
            zero.at("cg_trace").empty()&&seeds.service()==original,
            "Zero-budget reuse invented a bound or altered retained columns");
    options.cg_seed_reuse=false;
    const auto legacy=solve_continuous_cg(foreign_input,foreign_paths,options,&seeds);
    require(!legacy.at("bound_valid").get<bool>()&&legacy.at("cg_trace").empty(),
            "Disabled reuse changed legacy zero-budget behavior");
    return {{"passed",true},{"append_only",true},{"foreign_and_stale_rejected",true},
        {"invalid_additions_atomic",true},{"zero_budget_no_certificate",true},{"disabled_flag_ignores_cache",true}};
}
}
int main(int argc,char** argv) {try {
    if(argc==2&&std::string(argv[1])=="guards") {
        std::cout<<guard_tracer().dump(2)<<'\n';
    } else {
        std::cout<<Json({{"guards",guard_tracer()},{"reuse",reuse_tracer()},
            {"face",face_tracer()},{"passed",true}}).dump(2)<<'\n';
    }
    return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
