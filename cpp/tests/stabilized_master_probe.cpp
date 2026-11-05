// Tiny MILP candidate-selection checks, independent of recourse or pricing.
#include "mlnd/stabilized_master.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace mlnd;
namespace {
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int n,int m,T value) {
    Matrix<T> out(n,m);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture() {
    Input d;d.name="stabilized_master_three_origins";
    d.n_l=3;d.n_h=1;d.n_od=2;d.n_levels=2;d.max_orders=1;
    d.base_cost={0,0,0};d.slot_cost=filled(1,2,0.0);
    d.link_cost=filled(4,4,0.0);d.allowed_links=filled(4,4,1);
    for(int i=0;i<4;++i)d.allowed_links(i,i)=0;
    return d;
}
Formulation master_model(const Input& d,int& theta) {
    auto f=design_model(d);
    theta=f.matrix.variable("operating_epigraph",0,100,1);
    f.matrix.constant=7;
    // A move to base 2 saves ten units; a capacity upgrade saves nine.
    // The current extra cut forbids the otherwise preferable base 1.
    f.matrix.row("current_cost_cut",'G',20,{{theta,1},
        {f.layout.assignment_index(0,1),12},{f.layout.assignment_index(0,2),10},
        {f.layout.capacity+1,9}});
    f.matrix.row("current_design_cut",'E',0,{{f.layout.assignment_index(0,1),1}});
    return f;
}
std::vector<double> center(const Formulation& f) {
    std::vector<double> x(f.layout.count,0);
    x[f.layout.facility]=1;x[f.layout.facility+3]=1;
    x[f.layout.capacity]=1;x[f.layout.assignment_index(0,0)]=1;
    return x;
}
SolverOptions options() {
    SolverOptions result;result.seconds=5;result.threads=1;result.log=false;result.gap=1e-9;
    result.stabilized_master=true;result.stabilized_master_seconds=5;result.stabilized_master_level=.5;
    return result;
}
void private_bound(const StabilizedMasterResult& out) {
    require(!out.candidate.optimal&&std::isinf(out.candidate.bound)&&out.candidate.bound<0,
            "Trial optimality or objective bound escaped into the candidate");
    require(out.metadata.at("bound_scope")=="not_global"&&!out.metadata.at("global_bound_exported").get<bool>(),
            "Trial bound scope was not explicit");
}
Json level_and_distance() {
    const auto d=fixture();int theta=-1;const auto master=master_model(d,theta);const auto o=options();
    const auto rows=master.matrix.rows.size();const auto original_cost=master.matrix.variables[theta].cost;
    const auto out=select_stabilized_design(d,master,theta,center(master),7,27,o,wall_time()+5);
    private_bound(out);
    require(out.candidate.has_solution&&out.candidate.candidate_available&&out.metadata.at("candidate_validated"),
            "Stabilized master found no validated candidate");
    require(out.metadata.at("cost_level").get<double>()==17,"Wrong level interpolation");
    require(out.candidate.objective<=17+1e-7&&
            std::abs(out.candidate.objective-objective_value(master.matrix,out.candidate.values))<1e-9,
            "Original objective constant or original candidate objective was lost");
    require(out.candidate.values.at(master.layout.assignment_index(0,1))<1e-7&&
            out.candidate.values.at(master.layout.assignment_index(0,2))>1-1e-7,
            "Current master cuts were not copied, or the three-origin group distance changed");
    require(out.candidate.values.at(master.layout.capacity)>1-1e-7,
            "A capacity change displaced the only level-feasible one-group head move");
    const double distance=out.metadata.at("distance_objective").get<double>();
    require(distance>=1-1e-7&&distance<1.25,"Origin migration was counted as two group changes");
    require(assess_point(master.matrix,out.candidate.values).feasible,"Candidate violates original master");
    require(master.matrix.rows.size()==rows&&master.matrix.variables[theta].cost==original_cost&&master.matrix.constant==7,
            "Trial selector mutated the unrestricted master");
    NativeSolver global(master.matrix,o,"stabilized_probe_unrestricted");const auto unrestricted=global.optimize(5);
    require(unrestricted.has_solution&&unrestricted.optimal&&std::abs(unrestricted.objective-8)<1e-7,
            "Tiny unrestricted master reference failed");
    const double global_bound=unrestricted.bound;
    const double preserved=std::max(global_bound,out.candidate.bound);
    require(preserved==global_bound&&out.candidate.objective>=global_bound-1e-7,
            "Trial bound could replace the unrestricted lower bound");
    return {{"passed",true},{"candidate_cost",out.candidate.objective},{"group_distance_with_tie",distance},
        {"global_objective",unrestricted.objective},{"global_bound",global_bound},{"trial",out.metadata}};
}
Json local_infeasibility_and_budget() {
    const auto d=fixture();int theta=-1;const auto master=master_model(d,theta);auto o=options();
    // Valid bounds are LB=7 and UB=8, while the true optimum is 8. A 7.5
    // level is locally infeasible, although the unrestricted master is feasible.
    const auto local=select_stabilized_design(d,master,theta,center(master),7,8,o,wall_time()+5);
    private_bound(local);
    require(!local.candidate.has_solution&&local.metadata.at("solver_calls")==1&&
            local.metadata.at("status")=="no_trial_candidate","Locally infeasible level did not fall back");
    const auto expired=select_stabilized_design(d,master,theta,center(master),7,27,o,0);
    private_bound(expired);
    require(!expired.candidate.has_solution&&expired.metadata.at("solver_calls")==0&&
            !expired.metadata.at("model_built").get<bool>(),"Zero remaining budget built or optimized a trial");
    o.stabilized_master_seconds=0;
    const auto disabled=select_stabilized_design(d,master,theta,center(master),7,27,o,INFINITY);
    require(disabled.metadata.at("solver_calls")==0&&disabled.metadata.at("status")=="no_budget",
            "Zero configured slice still optimized");
    return {{"passed",true},{"local_infeasible",local.metadata},{"expired",expired.metadata}};
}
Json invalid_inputs() {
    auto d=fixture();int theta=-1;const auto master=master_model(d,theta);auto o=options();
    auto bad_center=center(master);bad_center[master.layout.assignment_index(0,1)]=1;
    const auto bad=select_stabilized_design(d,master,theta,bad_center,7,27,o,INFINITY);
    require(bad.metadata.at("status")=="invalid_center_head_state"&&bad.metadata.at("solver_calls")==0,
            "Noncategorical center was accepted");
    private_bound(bad);
    o.stabilized_master_level=1;
    const auto invalid_level=select_stabilized_design(d,master,theta,center(master),7,27,o,INFINITY);
    require(invalid_level.metadata.at("status")=="invalid_level_fraction","Invalid level fraction was accepted");
    o=options();d.max_orders=2;d.preprocessed_demand=false;
    const auto pair=select_stabilized_design(d,master,theta,center(master),7,27,o,INFINITY);
    require(pair.metadata.at("status")=="unsupported_scope"&&pair.metadata.at("solver_calls")==0,
            "Pair-assignment model silently used origin group distance");
    return {{"passed",true},{"invalid_center_rejected",true},{"pair_assignment_fallback",true}};
}
}
int main() {
    try {std::cout<<Json{{"passed",true},{"level_and_distance",level_and_distance()},
        {"local_infeasibility_and_budget",local_infeasibility_and_budget()},{"invalid_inputs",invalid_inputs()}}.dump(2)<<'\n';return 0;}
    catch(const IloException& error){std::cerr<<error<<'\n';return 1;}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
