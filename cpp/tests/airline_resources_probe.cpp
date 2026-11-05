// Independent bounded fixtures; no production benchmark is launched here.
#include "mlnd/airline_resources.hpp"
#include "mlnd/preprocessed_demand.hpp"
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

std::vector<double> courier_point(const Input& d,const Formulation& f,const std::vector<double>& design) {
    std::vector<double> x(f.matrix.variables.size(),0);
    std::copy(design.begin(),design.end(),x.begin());
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r];
    for(std::size_t j=0;j<x.size();++j){const auto& n=f.matrix.variables[j].name;
        if(n.rfind("cone_up_",0)==0||n.rfind("cone_down_",0)==0||n.rfind("cone_one_",0)==0)x[j]=1;}
    return x;
}
std::vector<double> exact_witness(const Input& d,const PathCollection& paths,const Formulation& f) {
    const int N=d.n_nodes(),L=d.n_l;const double rho=.1;
    auto fixed=std::vector<double>(f.layout.count,0);
    fixed[f.layout.assignment_index(0,0)]=1;
    // A second base may be used exclusively for return/relocation.
    for(int l=0;l<L;++l)fixed[l]=1;
    auto x=courier_point(d,f,fixed);
    x[f.rho[0]]=rho;x[f.idle[0]]=rho/(1-rho);
    std::map<std::string,int> names;for(std::size_t j=0;j<x.size();++j)names[f.matrix.variables[j].name]=j;
    x[names.at("v_0_0")]=rho;
    std::vector<double> returned(L);double busy=0;
    auto mark=[&](const auto& p){for(auto[u,v,count]:p.arc_counts){(void)count;
        x[f.layout.airline+u*N+v]=1;x[u]=x[v]=1;
        if(u>=L)x[f.layout.capacity+(u-L)*d.n_levels]=1;
        if(v>=L)x[f.layout.capacity+(v-L)*d.n_levels]=1;}};
    for(int s=-1;s<d.n_od;++s){double rate=s<0?d.single_rates[0]:d.pair_rates(0,s);if(rate<=0)continue;
        int chosen=-1;for(std::size_t j=0;j<paths.service_paths.size();++j){const auto&p=paths.service_paths[j];
            if(p.first_od==0&&p.second_od==s&&p.b[0]==0){chosen=j;break;}}
        require(chosen>=0,"Positive witness branch missing");const auto&p=paths.service_paths[chosen];
        const double flow=rate*rho;x[f.flow[chosen]]=flow;x[f.head[p.segment_id]]+=flow;
        for(auto[r,c]:p.served_counts)x[f.courier[r]]-=c*flow;
        busy+=flow*p.time;returned[p.b[1]]+=flow;mark(p);
    }
    for(int l=1;l<L;++l)if(returned[l]>0){int selected=-1;
        for(std::size_t j=0;j<paths.empty_paths.size();++j)if(paths.empty_paths[j].b==BasePair{l,0}){selected=j;break;}
        require(selected>=0,"Missing empty return witness");const auto&p=paths.empty_paths[selected];
        x[f.empty_flow[selected]]=returned[l];busy+=returned[l]*p.time;mark(p);}
    for(std::size_t g=0;g<f.lift_groups.size();++g){double head=0;for(int j:f.lift_groups[g])head+=x[f.head[j]];
        x[names.at("group_head_"+std::to_string(g))]=head;}
    for(int l=0;l<L;++l){x[names.at("cone_up_"+std::to_string(l))]=1+x[f.idle[l]];
        x[names.at("cone_down_"+std::to_string(l))]=1-x[f.rho[l]];}
    x[f.fleet]=busy+x[f.idle[0]];return x;
}
Json evaluate(int bases,int k,const std::string& variant) {
    auto d=fixture(bases,k);
    if(variant=="missing")d.deadline[1]=1e-5;
    if(variant=="negative")d.ground_cost={.01,.02};
    if(variant=="capacity")d.occupancy_limit=filled(2,1,.15);
    if(variant=="zero_head"){d.single_rates={0,7};d.demand={0,7};d.pair_rates=filled(2,2,0.0);d.head_rates={0,7};}
    if(variant=="cross_return")for(int u=bases;u<d.n_nodes();++u)d.allowed_links(u,0)=0;
    const auto paths=enumerate_paths(d);const auto connectivity=build_airline_connectivity(d,paths);
    SolverOptions options;options.seconds=15;options.gap=1e-9;options.threads=1;options.tight_airline_bounds=true;
    options.task_airline_upper=build_task_airline_bounds(d,paths).upper;
    AirlineResourceRelaxation resource(d,paths,options);
    AirlineResourceRelaxation balanced(d,paths,options,true);
    TaskResourceSavingsRelaxation old_resource(d,task_savings_data(d,paths),options);
    const auto layout=DesignLayout(d);const int theta=layout.count;
    std::vector<std::vector<double>> targets;std::vector<RowRecord> cuts;
    int solves=0,lp_compares=0,balance_compares=0;double worst=0,weakness=0,balance_weakness=0;
    double closed_relocation_gain=0;bool incoming=false;
    auto check=[&](const std::vector<double>& fixed){
        auto f=build_formulation(d,paths,&fixed,nullptr,nullptr,false,false,true,formulation_strengthening(options));
        NativeSolver solver(f.matrix,options,"airline_resource_fixed_validation");const auto q=solver.optimize(15);
        require(q.optimal&&q.has_solution,"Fixed original recourse not optimal: "+q.status);
        require(point_residuals(f.matrix,q.values).at("passed"),"Invalid fixed original point");
        double fixedcost=0;for(int j=0;j<layout.count;++j)fixedcost+=f.matrix.variables[j].cost*fixed[j];
        auto target=fixed;target.push_back(q.objective-fixedcost);targets.push_back(target);++solves;
        if(k==2&&fixed[layout.assignment_index(0,0)]>.5){bool second_assigned=false;
            for(int l=0;l<bases;++l)second_assigned=second_assigned||fixed[layout.assignment_index(1,l)]>.5;
            incoming=incoming||(!second_assigned&&q.values[f.courier[1]]<d.demand[1]-1e-5);}
        for(double budget:{0.0,2.0}){const auto c=resource.separate(fixed,0,theta,budget);
            require(c.valid&&c.max_dual_violation_after<=0,"Uncertified route resource cut");
            if(budget==0)require(c.fallback,"Zero budget did not use safe fallback");cuts.push_back(c.row);
            if(budget>0){auto old=old_resource.separate(fixed,0,theta,2);
                if(c.optimal&&old.optimal){weakness=std::max(weakness,c.solver_objective-old.solver_objective);++lp_compares;}}}
        for(double budget:{0.0,2.0}){const auto c=balanced.separate(fixed,0,theta,budget);
            require(c.valid&&c.max_dual_violation_after<=0,"Uncertified balanced route resource cut");
            if(budget==0)require(c.fallback,"Zero-budget balanced LP did not use safe fallback");
            cuts.push_back(c.row);
            if(budget>0){const auto old=resource.separate(fixed,0,theta,2);
                if(c.optimal&&old.optimal){
                    balance_weakness=std::max(balance_weakness,c.solver_objective-old.solver_objective);
                    ++balance_compares;
                }
            }
        }
    };
    for(int a=-1;a<bases;++a)for(int b=-1;b<bases;++b)check(design(d,a,b));
    check(design(d,0,bases-1,false));check(std::vector<double>(layout.count,0));
    if(bases>1){auto no_empty=design(d,0,0);
        for(int u=0;u<bases;++u)for(int v=0;v<bases;++v)
            no_empty[layout.airline+u*d.n_nodes()+v]=0;
        check(no_empty);
        const auto before=resource.separate(no_empty,0,theta,2),after=balanced.separate(no_empty,0,theta,2);
        require(before.optimal&&after.optimal,"Closed-relocation comparison not optimal");
        closed_relocation_gain=before.solver_objective-after.solver_objective;
        if(variant=="cross_return")require(closed_relocation_gain>1,
            "Balance relaxation failed to remove service with no possible relocation");
    }
    const auto n=targets.size();for(std::size_t j=1;j<n;++j){auto x=targets[j];
        for(std::size_t v=0;v<x.size();++v)x[v]=.37*x[v]+.63*targets[0][v];targets.push_back(x);}
    for(const auto&c:cuts)for(const auto&x:targets)worst=std::max(worst,row_violation(c,x));
    require(worst<1e-5,"Route-resource cut excludes original integer recourse or its convex mixture");
    require(weakness<1e-5,"Route resource LP weaker than independent minima unexpectedly");
    require(balance_weakness<1e-5,"Balance/relocation LP unexpectedly weaker than prior route LP");
    auto original=build_formulation(d,paths);auto full=original;
    for(const auto& row:connectivity.rows)full.matrix.row(row.name,row.sense,row.rhs,row.coefficients);
    // All-courier redundant original encoding is valid but may violate dominance rows.
    auto redundant=design(d,0,0,false);auto courier=courier_point(d,original,redundant);
    require(point_residuals(original.matrix,courier).at("passed"),"Redundant all-courier witness not original feasible");
    double dominance_violation=0;for(const auto&r:connectivity.rows)dominance_violation=std::max(dominance_violation,row_violation(r,courier));
    require(dominance_violation>.5,"Static rows failed to distinguish redundant original encoding");
    auto canonical=courier_point(d,full,std::vector<double>(layout.count,0));
    require(point_residuals(full.matrix,canonical).at("passed"),"Canonical all-courier point rejected");
    bool witness=false;
    if(variant=="normal"||variant=="cross_return"){
        auto x=exact_witness(d,paths,original);
        require(point_residuals(original.matrix,x).at("passed"),"Hand-constructed positive original witness invalid");
        require(point_residuals(full.matrix,x).at("passed"),"Static rows reject positive cross-return/second-OD witness");
        auto target=std::vector<double>(x.begin(),x.begin()+layout.count);
        double operating=objective_value(original.matrix,x);
        for(int j=0;j<layout.count;++j)operating-=original.matrix.variables[j].cost*x[j];
        target.push_back(operating);
        for(const auto& cut:cuts)require(row_violation(cut,target)<1e-5,
            "A resource cut rejects positive original service/relocation witness");
        witness=true;
    }
    NativeSolver ref(original.matrix,options,"airline_connectivity_reference"),strong(full.matrix,options,"airline_connectivity_strengthened");
    const auto a=ref.optimize(15);require(a.optimal&&a.has_solution,"Original MISOCP not optimal");
    require(point_residuals(original.matrix,a.values).at("passed"),"Original optimum invalid");
    if(point_residuals(full.matrix,a.values).at("passed"))strong.add_start(a.values);
    const auto b=strong.optimize(15);require(b.optimal&&b.has_solution,"Strengthened MISOCP not optimal");
    require(point_residuals(full.matrix,b.values).at("passed"),"Strengthened optimum invalid");
    require(std::abs(a.objective-b.objective)<1e-5,"Static connectivity changed full optimum");
    return {{"bases",bases},{"K",k},{"variant",variant},{"fixed_recourse_solves",solves},
        {"cross_checks",cuts.size()*targets.size()},{"max_cost_cut_violation",worst},
        {"lp_dominance_comparisons",lp_compares},{"max_route_lp_excess_over_minima",weakness},
        {"balance_dominance_comparisons",balance_compares},{"max_balanced_lp_excess",balance_weakness},
        {"closed_relocation_savings_reduction",closed_relocation_gain},
        {"static_original_redundant_violation",dominance_violation},{"positive_hand_witness",witness},
        {"incoming_second_without_own_assignment",incoming},{"reference_objective",a.objective},
        {"strengthened_objective",b.objective},{"full_objective_difference",std::abs(a.objective-b.objective)},
        {"connectivity",connectivity.metadata},{"resource",resource.metadata()},
        {"balanced_resource",balanced.metadata()}};
}
Json guards(){auto d=fixture(1,2);const auto paths=enumerate_paths(d);int checks=0;SolverOptions o;
    for(int i=0;i<3;++i){auto x=d;auto p=paths;if(i==0)p.complete=false;if(i==1)p.support_complete=false;if(i==2)x.first_leg_consistency=true;
        bool a=false,b=false;try{(void)build_airline_connectivity(x,p);}catch(const std::invalid_argument&){a=true;}
        try{AirlineResourceRelaxation r(x,p,o);}catch(const std::invalid_argument&){b=true;}
        require(a&&b,"Incomplete catalogue/domain not rejected");++checks;}
    auto two=fixture(2,2);auto p=enumerate_paths(two);
    p.empty_paths[0].arc_counts={{0,1,2}};bool rejected=false;
    try{AirlineResourceRelaxation r(two,p,o,true);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Malformed direct relocation path not rejected");
    return {{"domain_checks",checks},{"malformed_empty_rejected",rejected}};
}
}
int main(int argc,char**argv){try{
    const double started=wall_time();
    if(argc>1){auto d=read_input(argv[1],2);d.preprocessed_demand=true;auto paths=enumerate_paths(d);
        preprocess_generalized_demand(d,paths);SolverOptions o;o.tight_airline_bounds=true;
        o.task_airline_upper=build_task_airline_bounds(d,paths).upper;
        const bool balance=argc>2&&std::string(argv[argc-1])=="--balance-relocation";
        auto c=build_airline_connectivity(d,paths);AirlineResourceRelaxation r(d,paths,o,balance);
        Json updates=Json::array();
        const bool repeat=argc>2&&std::string(argv[2])=="--repeat-updates";
        if(repeat){
            const DesignLayout layout(d);auto full=std::vector<double>(layout.count,0);
            const auto support=task_savings_data(d,paths).supported;
            for(int j=0;j<d.n_nodes();++j)full[j]=1;
            for(int i=0;i<d.n_h;++i)full[layout.capacity+i*d.n_levels+d.n_levels-1]=1;
            for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
                full[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
            for(int pass=0;pass<6;++pass){auto point=full;
                for(int od=0;od<d.n_od;++od)if(pass%3!=1&&(pass%3!=2||od%2==0))
                    for(int offset=0;offset<d.n_l;++offset){const int l=(od+(pass%3==2?1:0)+offset)%d.n_l;
                        if(support[od*d.n_l+l]){point[layout.assignment_index(od,l)]=1;break;}}
                const auto cut=r.separate(point,0,layout.count,2.0);
                const double update=r.metadata().at("last_objective_update_seconds");
                const double budget=r.metadata().at("last_solver_budget_seconds");
                require(budget>0&&cut.status!="budget_exhausted","Repeated objective update consumed entire two-second budget");
                require(cut.valid,"Repeated objective update produced invalid dual certificate");
                updates.push_back({{"pass",pass},{"pattern",pass%3},{"allocated_seconds",2},
                    {"objective_update_seconds",update},{"solver_budget_seconds",budget},
                    {"solver_seconds",cut.solver_seconds},{"total_seconds",cut.seconds},{"status",cut.status},
                    {"optimal",cut.optimal},{"valid",cut.valid},{"fallback",cut.fallback},
                    {"solver_objective",cut.solver_objective},{"dual_savings",cut.dual_savings}});
            }
            for(int pass=3;pass<6;++pass)if(updates[pass]["optimal"]==true&&updates[pass-3]["optimal"]==true)
                require(std::abs(updates[pass]["solver_objective"].get<double>()-updates[pass-3]["solver_objective"].get<double>())<1e-5,
                        "Identical repeated design has inconsistent optimal LP objective");
        }
        std::cout<<Json{{"passed",true},{"construction_only",!repeat},{"repeated_updates",updates},
            {"connectivity",c.metadata},{"resource",r.metadata()},{"seconds",wall_time()-started}}.dump(2)<<'\n';return 0;}
    Json cases=Json::array();for(int l:{1,2})for(int k:{1,2})cases.push_back(evaluate(l,k,"normal"));
    for(const std::string variant:{"missing","negative","capacity","zero_head"})cases.push_back(evaluate(1,2,variant));
    cases.push_back(evaluate(2,2,"cross_return"));
    // More than two bases verifies omission of exactly one redundant
    // balance row and the per-empty D gate when the old arc bound is 2D.
    cases.push_back(evaluate(3,2,"normal"));
    std::cout<<Json{{"passed",true},{"guards",guards()},{"cases",cases},{"seconds",wall_time()-started}}.dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
