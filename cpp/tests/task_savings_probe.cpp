// Mathematical regression fixtures; these are not production benchmark inputs.
#include "mlnd/task_savings.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& message) { if(!ok)throw std::runtime_error(message); }
template<class T> Matrix<T> filled(int n,int m,T value) {
    Matrix<T> out(n,m);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture(int bases,int k) {
    Input d;d.n_l=bases;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=k;
    d.name="task_savings_validity_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={4,7};d.ground_cost={2,20};d.deadline={1,1};
    const int n=bases+2;d.flight_times=filled(n,n,.05);d.allowed_links=filled(n,n,1);
    for(int u=0;u<n;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.link_cost=filled(n,n,.01);d.base_cost=std::vector<double>(bases,.5);
    d.slot_epsilon={.05,.05};d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;d.pair_rates=filled(2,2,0.0);
    if(k==1){d.single_rates=d.demand;d.head_rates=d.demand;}
    else {d.single_rates={.5,4.5};d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;
          d.pair_rates(1,0)=.5;d.head_rates={3,5};}
    d.common_origin_support=filled(2,bases,1);return d;
}
std::vector<double> design(const Input& d,int a,int b,bool open=true) {
    DesignLayout layout(d);std::vector<double>x(layout.count);
    for(int j=0;j<d.n_nodes();++j)x[j]=1;
    for(int i=0;i<d.n_h;++i)x[layout.capacity+i]=1;
    if(a>=0)x[layout.assignment_index(0,a)]=1;if(b>=0)x[layout.assignment_index(1,b)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v)&&(open||(u<d.n_l&&v<d.n_l));
    return x;
}
double row_violation(const RowRecord& row,const std::vector<double>& x) {
    double a=0;for(auto [j,c]:row.coefficients)a+=c*x.at(j);
    return row.sense=='G'?row.rhs-a:row.sense=='L'?a-row.rhs:std::abs(a-row.rhs);
}
Json evaluate(int bases,int k,const std::string& variant) {
    auto d=fixture(bases,k);
    if(variant=="missing")d.deadline[1]=1e-5;
    if(variant=="negative")d.ground_cost={.01,.02};
    if(variant=="zero_head"){
        d.single_rates={0,7};d.demand={0,7};d.pair_rates=filled(2,2,0.0);d.head_rates={0,7};
    }
    if(variant=="cross_return")for(int u=bases;u<d.n_nodes();++u)d.allowed_links(u,0)=0;
    if(variant=="capacity")d.occupancy_limit=filled(2,1,.15);
    const auto paths=enumerate_paths(d);const auto data=task_savings_data(d,paths);
    auto master=design_model(d);const int theta=master.matrix.variable("theta",0,INFINITY,0);
    const auto before_vars=master.matrix.variables.size();add_task_savings_bound(data,master,theta);
    require(master.matrix.variables.size()==before_vars,"Task bound added a master auxiliary variable");
    const auto static_row=master.matrix.rows.back();
    SolverOptions options;options.seconds=15;options.gap=1e-9;options.threads=1;
    TaskResourceSavingsRelaxation resource(d,data,options);
    std::vector<std::vector<double>> targets;std::vector<RowRecord> cuts;
    double maxerror=0;int solved_points=0;bool incoming_witness=false;
    auto check_point=[&](std::vector<double> fixed){
        auto f=build_formulation(d,paths,&fixed);
        NativeSolver solver(f.matrix,options,"task_bound_fixed_validation");const auto q=solver.optimize(15);
        require(q.optimal&&q.has_solution,"Fixed recourse not solved: "+q.status);
        require(point_residuals(f.matrix,q.values).at("passed"),"Fixed recourse point invalid");
        if(k==2&&(variant=="normal"||variant=="cross_return")){
            bool own_second=false;for(int l=0;l<bases;++l)own_second=own_second||fixed[f.layout.assignment_index(1,l)]>.5;
            if(!own_second&&q.values[f.courier[1]]<d.demand[1]-1e-5)incoming_witness=true;
        }
        auto point=fixed;point.push_back(q.objective);targets.push_back(point);++solved_points;
        require(row_violation(static_row,point)<1e-5,"Static task bound rejects fixed recourse");
        for(double budget:{0.0,5.0}){
            const auto cut=resource.separate(fixed,0,theta,budget);
            require(cut.valid&&cut.max_dual_violation_after<=0,"Task resource dual certificate invalid");
            if(budget==0)require(cut.fallback,"Zero-budget call did not use the safe fallback");
            cuts.push_back(cut.row);
        }
    };
    for(int a=-1;a<bases;++a)for(int b=-1;b<bases;++b)check_point(design(d,a,b));
    check_point(design(d,0,bases-1,false));
    auto allzero=std::vector<double>(DesignLayout(d).count,0);check_point(allzero);
    const auto originals=targets.size();
    for(std::size_t j=1;j<originals;++j){
        auto x=targets[j];for(std::size_t v=0;v<x.size();++v)x[v]=.37*x[v]+.63*targets[0][v];
        targets.push_back(std::move(x));
    }
    for(const auto& cut:cuts)for(const auto& x:targets)maxerror=std::max(maxerror,row_violation(cut,x));
    require(maxerror<1e-5,"Task resource cut excludes recourse at another design or convex mixture");
    auto original=build_formulation(d,paths),strong=original;
    // Link a zero-objective epigraph variable exactly to existing operating
    // cost so rows can be checked inside the complete integer formulation.
    const int epigraph=strong.matrix.variable("operating_epigraph",0,INFINITY,0);
    Coefficients operating{{epigraph,1}};
    for(int j=strong.layout.count;j<epigraph;++j)
        if(strong.matrix.variables[j].cost)operating.emplace_back(j,-strong.matrix.variables[j].cost);
    strong.matrix.row("operating_cost_identity",'E',0,std::move(operating));
    add_task_savings_bound(data,strong,epigraph);
    for(auto cut:cuts){for(auto&[j,c]:cut.coefficients)if(j==theta)j=epigraph;
        strong.matrix.row(cut.name,cut.sense,cut.rhs,std::move(cut.coefficients));}
    NativeSolver a(original.matrix,options,"task_bound_full_reference"),b(strong.matrix,options,"task_bound_full_strengthened");
    const auto qa=a.optimize(15);
    require(qa.optimal&&qa.has_solution,"Full original MISOCP not solved");
    auto extended=qa.values;extended.resize(strong.matrix.variables.size());
    double op=0;for(int j=original.layout.count;j<static_cast<int>(original.matrix.variables.size());++j)op+=original.matrix.variables[j].cost*qa.values[j];extended[epigraph]=op;
    require(point_residuals(strong.matrix,extended).at("passed"),"A new task cut excludes original integer optimum");
    // Supplying the already independently checked original optimum avoids
    // an observed CPLEX symmetric-MISOCP start sensitivity of one link cost.
    b.add_start(extended);const auto qb=b.optimize(15);
    require(qa.optimal&&qb.optimal&&qa.has_solution&&qb.has_solution,"Full MISOCP comparison did not solve");
    require(point_residuals(original.matrix,qa.values).at("passed")&&point_residuals(strong.matrix,qb.values).at("passed"),"Full MISOCP incumbent invalid");
    require(std::abs(qa.objective-qb.objective)<1e-5,"Task cost cuts changed integer optimum: "+std::to_string(bases)+"/"+std::to_string(k)+"/"+variant+" reference="+std::to_string(qa.objective)+" strengthened="+std::to_string(qb.objective));
    if(variant=="negative")require(std::all_of(data.saving.begin(),data.saving.end(),[](double a){return a==0;}),"Negative head savings were not safely discarded");
    if(variant=="missing")require(std::any_of(data.supported.begin(),data.supported.end(),[](bool a){return !a;}),"Missing branch fixture did not trigger unsupported groups");
    if(variant=="zero_head")for(int l=0;l<bases;++l)require(data.saving[l]==0&&data.total_load[l]==0,"Zero head has positive savings/load");
    if(k==2&&(variant=="normal"||variant=="cross_return"))require(incoming_witness,"No positive-service witness for second OD without its own assignment");
    return {{"bases",bases},{"K",k},{"variant",variant},{"incoming_second_without_own_assignment",incoming_witness},{"paths",paths.service_paths.size()},
        {"fixed_recourse_solves",solved_points},{"cuts",cuts.size()},{"cross_checks",cuts.size()*targets.size()},
        {"max_cut_violation",maxerror},{"reference_objective",qa.objective},{"strengthened_objective",qb.objective},
        {"objective_difference",std::abs(qa.objective-qb.objective)},{"resource_metadata",resource.metadata()}};
}
Json guards_and_chord(){
    auto d=fixture(1,2);auto paths=enumerate_paths(d);int guards=0;
    for(int flag=0;flag<3;++flag){auto copy=paths;auto x=d;
        if(flag==0)copy.complete=false;if(flag==1)copy.support_complete=false;if(flag==2)x.first_leg_consistency=true;
        bool rejected=false;try{(void)task_savings_data(x,copy);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Unsupported task-bound domain not rejected");++guards;
    }
    const auto data=task_savings_data(d,paths);
    double maxerror=0;std::size_t samples=0;
    // Check the claimed analytic envelope independently over low/high fleet
    // costs and savings ranges, including Smax <= F and zero.
    for(double f:{.001,1.0,100.0})for(double maxs:{0.0,.01,1.0,10.0,1000.0}){
        const double eta=maxs>f?std::pow(1-std::sqrt(f/maxs),2):0;
        for(int j=0;j<=1000;++j){double s=maxs*j/1000.0;
            double gain=s>f?std::pow(std::sqrt(s)-std::sqrt(f),2):0;
            maxerror=std::max(maxerror,gain-eta*s);++samples;}
    }
    require(maxerror<1e-9,"Queue chord underestimates attainable saving");
    return {{"domain_guards",guards},{"chord_samples",samples},{"maximum_chord_error",maxerror}};
}
}
int main(){try{
    const double started=wall_time();Json cases=Json::array();
    for(int l:{1,2})for(int k:{1,2})cases.push_back(evaluate(l,k,"normal"));
    for(const std::string variant:{"missing","negative","zero_head","capacity"})cases.push_back(evaluate(1,2,variant));
    cases.push_back(evaluate(2,2,"cross_return"));
    std::cout<<Json{{"passed",true},{"guards",guards_and_chord()},{"cases",cases},{"seconds",wall_time()-started}}.dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
