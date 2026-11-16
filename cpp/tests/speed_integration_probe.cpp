#include "mlnd/benders.hpp"
#include "mlnd/solver_presets.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
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

Input large_fixture(int h=3,int r=3){
    Input d;d.name="pricing_speed_exact";d.model_version="consecutive";
    d.n_l=2;d.n_h=h;d.n_od=r;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand.assign(r,2+r*.5);d.single_rates.assign(r,2);d.head_rates.assign(r,2+r*.25);
    d.pair_rates=filled(r,r,.25);d.common_origin_support=filled(r,2,1);
    d.ground_cost.assign(r,20);d.deadline.assign(r,1);d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.flight_times=filled(h+2,h+2,.05);d.allowed_links=filled(h+2,h+2,1);d.link_cost=filled(h+2,h+2,.01);
    for(int u=0;u<h+2;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.rider_mean=filled(r,h,.08);d.first_cost=filled(r,h,1.0);d.terminal_cost=filled(r,h,.5);
    d.receipt_mean=filled(r,h,.08);d.removal_mean=filled(r,h,.03);d.slot_counts=filled(h,1,20);
    d.slot_cost=filled(h,1,.1);d.occupancy_limit=filled(h,1,14.0);d.kappa=filled(r,r,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}

Json matrix_record(const LinearModel& m) {
    Json v=Json::array(),r=Json::array();
    for(const auto& item:m.variables)v.push_back({item.name,item.lower,item.upper,item.cost,item.binary});
    for(const auto& item:m.rows)r.push_back({item.name,std::string(1,item.sense),item.rhs,item.coefficients});
    return {{"variables",v},{"rows",r}};
}
Json formulation_case() {
    const auto d=fixture();const auto paths=enumerate_paths(d);std::vector<int> none;
    auto old=build_formulation(d,paths,nullptr,&none,&none);
    auto fast=old;
    for(auto* f:{&old,&fast})for(auto& v:f->matrix.variables)v.binary=false;
    SolverOptions options;options.seconds=10;
    NativeSolver old_solver(old.matrix,options),fast_solver(fast.matrix,options);
    for(int first=0;first<static_cast<int>(paths.service_paths.size());first+=11) {
        std::vector<int> ids;
        for(int j=first;j<std::min(first+11,static_cast<int>(paths.service_paths.size()));++j)ids.push_back(j);
        std::reverse(ids.begin(),ids.end());
        add_columns(old,old_solver,d,paths,ids,none,false);
        add_columns(fast,fast_solver,d,paths,ids,none,true);
        require(matrix_record(old.matrix)==matrix_record(fast.matrix),"Batched service rows/bounds/cost/order changed");
        require(old.flow==fast.flow&&old.service_indices==fast.service_indices,"Batched path indices diverged");
    }
    std::vector<int> empty(paths.empty_paths.size());std::iota(empty.begin(),empty.end(),0);
    add_columns(old,old_solver,d,paths,none,empty,false);
    add_columns(fast,fast_solver,d,paths,none,empty,true);
    require(matrix_record(old.matrix)==matrix_record(fast.matrix),"Batched empty columns changed model");
    const auto before=matrix_record(fast.matrix);bool rejected=false;
    try{add_columns(fast,fast_solver,d,paths,{0},{999999},true);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&matrix_record(fast.matrix)==before,"Invalid path batch changed mathematical model");
    add_columns(fast,fast_solver,d,paths,{0},empty,true);
    require(matrix_record(fast.matrix)==before,"Existing path columns were inserted twice");
    const auto a=old_solver.optimize(10),b=fast_solver.optimize(10);
    auto full=build_formulation(d,paths);
    for(auto& v:full.matrix.variables)v.binary=false;
    NativeSolver full_solver(full.matrix,options);const auto reference=full_solver.optimize(10);
    require(reference.optimal&&reference.has_solution&&point_residuals(full.matrix,reference.values).at("passed"),
            "Independent full continuous model did not certify a feasible optimum");
    require(a.optimal&&b.optimal&&a.has_solution&&b.has_solution,"Incremental conic models did not solve");
    require(std::abs(reference.objective-b.objective)<1e-6*(1+std::abs(reference.objective)),
            "Incremental batch model differs from independent complete model");
    require(std::abs(a.objective-b.objective)<1e-6*(1+std::abs(a.objective)),"Batch insertion changed conic optimum");
    require(point_residuals(fast.matrix,b.values).at("passed"),"Batch solution fails canonical original matrix");
    require(fast_solver.read_dual(fast.matrix).available,"Batched conic model lost native dual access");
    return {{"passed",true},{"old_objective",a.objective},{"batch_objective",b.objective},{"full_objective",reference.objective}};
}
Json insertion_case() {
    const auto d=large_fixture(10,4);const auto paths=enumerate_paths(d);std::vector<int> none;
    Json runs=Json::array();double old_total=0,new_total=0;
    for(int round=0;round<4;++round)for(bool batch:{round%2==1,round%2==0}) {
        auto f=build_formulation(d,paths,nullptr,&none,&none);
        for(auto& v:f.matrix.variables)v.binary=false;
        SolverOptions options;NativeSolver solver(f.matrix,options);
        const int count=std::min(12288,static_cast<int>(paths.service_paths.size()));
        const double start=wall_time();
        for(int first=0;first<count;first+=512) {
            std::vector<int> ids;for(int j=first;j<std::min(first+512,count);++j)ids.push_back(j);
            add_columns(f,solver,d,paths,ids,none,batch);
        }
        const double elapsed=wall_time()-start;
        require(static_cast<int>(f.service_indices.size())==count,"Insertion benchmark lost column indices");
        if(batch)new_total+=elapsed;else old_total+=elapsed;
        runs.push_back({{"round",round},{"batch",batch},{"columns",count},{"seconds",elapsed}});
    }
    return {{"passed",true},{"old_seconds",old_total},{"new_seconds",new_total},
        {"observed_speedup",old_total/new_total},{"runs",runs}};
}

Json preset_case() {
    SolverConfiguration config;config.apply_preset("lb-20261030");config.validate();auto old=config.metadata();
    config.apply_preset("speed-20261030");config.validate();auto fast=config.metadata();
    for(const char* flag:{"fast_pricing","batch_columns"}) {
        require(!old.at("flags").at(flag).get<bool>()&&fast.at("flags").at(flag).get<bool>(),"Speed flags not isolated");
        fast["flags"][flag]=false;
    }
    fast["preset"]=old["preset"];require(fast==old,"Performance preset changed mathematical or primal policy");
    for(const char* name:{"legacy","baseline-20261029","bounds-20261030","full-cplex-20261029"}) {
        config.apply_preset(name);config.validate();const auto flags=config.metadata().at("flags");
        require(!flags.at("fast_pricing").get<bool>()&&!flags.at("batch_columns").get<bool>(),"Historical preset enabled new path");
    }
    return {{"passed",true}};
}
Json main_case() {
    const auto d=fixture();auto paths=enumerate_paths(d);SolverOptions o;o.seconds=10;o.preprocessed_demand=true;
    o.pareto=false;o.branch_pricing_correction=true;o.cg_seed_reuse=true;o.staged_continuous_root=true;
    o.fast_pricing=true;o.batch_columns=true;
    const auto result=solve_benders(d,paths,o);
    require(result.at("bound").get<double>()<=result.at("objective").get<double>(),"Main accelerated solve lost safe bounds");
    require(std::abs(result.at("objective").get<double>()-47.54252458958)<1e-4,"Main accelerated solve lost known feasible optimum");
    return {{"passed",true},{"objective",result.at("objective")},{"bound",result.at("bound")}};
}
}
int main(){try {
    std::cout<<Json({{"formulation",formulation_case()},{"insertion",insertion_case()},{"preset",preset_case()},{"integration",main_case()},{"passed",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
