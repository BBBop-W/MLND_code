// Fixed-task airline bounds on the existing two-OD physical unit fixture.
// These are regression witnesses, not production benchmark instances.
#include "mlnd/formulation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& message) { if(!ok)throw std::runtime_error(message); }
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture(int bases,int cardinality) {
    Input d;d.n_l=bases;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=cardinality;
    d.name="task_airline_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.demand={4,7};d.ground_cost={2,20};d.deadline={1,1};
    const int n=bases+2;d.flight_times=filled(n,n,.05);
    for(int u=0;u<n;++u)d.flight_times(u,u)=0;
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.allowed_links=filled(n,n,1);for(int u=0;u<n;++u)d.allowed_links(u,u)=0;
    d.link_cost=filled(n,n,.01);d.base_cost=std::vector<double>(bases,.5);
    d.slot_epsilon={.05,.05};d.rider_mean=filled(2,2,.08);
    d.first_cost=filled(2,2,1.0);d.terminal_cost=filled(2,2,.5);
    d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.preprocessed_demand=true;d.pair_rates=filled(2,2,0.0);
    if(cardinality==1) {d.single_rates=d.demand;d.head_rates=d.demand;}
    else {
        d.single_rates={.5,4.5};d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;
        d.pair_rates(1,0)=.5;d.head_rates={3,5};
    }
    d.common_origin_support=filled(2,bases,1);return d;
}
std::vector<double> design(const Input& d,int origin0,int origin1) {
    DesignLayout layout(d);std::vector<double> x(layout.count);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    if(origin0>=0)x[layout.assignment_index(0,origin0)]=1;
    if(origin1>=0)x[layout.assignment_index(1,origin1)]=1;
    return x;
}
Json test_case(int bases,int cardinality) {
    auto d=fixture(bases,cardinality);auto paths=enumerate_paths(d);
    auto bounds=build_task_airline_bounds(d,paths);
    FormulationStrengthening strength;strength.task_airline_upper=bounds.upper;
    auto old=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,true);
    auto newer=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,true,strength);
    require(old.matrix.rows.size()==newer.matrix.rows.size()&&old.matrix.variables.size()==newer.matrix.variables.size(),
            "Task bound changed model dimensions");
    std::size_t changed=0;
    for(std::size_t j=0;j<old.matrix.rows.size();++j) {
        const auto& a=old.matrix.rows[j];const auto& b=newer.matrix.rows[j];
        require(a.name==b.name&&a.sense==b.sense&&a.rhs==b.rhs,"Task bound changed row structure");
        if(a.coefficients!=b.coefficients) {
            require(a.name.rfind("link_",0)==0,"Task bound changed a non-link row");++changed;
        }
    }
    // Independently check every origin assignment, choosing each branch's
    // maximum arc use. The second OD need not have its own assignment.
    int checks=0;
    for(int l0=-1;l0<bases;++l0)for(int l1=-1;l1<bases;++l1)
        for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v) {
            if(u<bases&&v<bases)continue;
            double load=0;
            for(int r=0;r<2;++r) {
                const int l=r?l1:l0;if(l<0)continue;
                for(int s=-1;s<2;++s) {
                    const double rate=s<0?d.single_rates[r]:d.pair_rates(r,s);int count=0;
                    for(const auto& p:paths.service_paths)if(p.first_od==r&&p.second_od==s&&p.b[0]==l) {
                        int uses=0;for(auto [a,b,n]:p.arc_counts)if(a==u&&b==v)uses+=n;
                        count=std::max(count,uses);
                    }
                    load+=rate*count;
                }
            }
            require(load<=bounds.upper[u*d.n_nodes()+v]+1e-9,"An origin assignment violates a task arc bound");++checks;
        }
    SolverOptions options;options.seconds=30;options.gap=1e-8;options.threads=1;
    options.tight_airline_bounds=true;
    auto plain=solve_full(d,paths,options);
    options.task_airline_bounds=true;options.task_airline_upper=bounds.upper;
    auto tight=solve_full(d,paths,options);
    require(plain.value("has_solver_solution",false)&&tight.value("has_solver_solution",false)&&
            plain.at("gap").get<double>()<1e-6&&tight.at("gap").get<double>()<1e-6,
            "Full optimum comparison lacks an optimality certificate");
    const double a=plain.at("objective"),b=tight.at("objective");
    require(std::abs(a-b)<=1e-5*(1+std::abs(a)),"Task bound changed the integer optimum");
    // Check a fixed design with OD 1 unassigned: its occurrence in a task
    // headed by OD 0 must remain serviceable. For L2 this also admits returns
    // to the other base and balancing empty relocation.
    const auto x=design(d,0,-1);
    auto partial_tight=solve_full(d,paths,options,&x);
    options.task_airline_upper.clear();options.task_airline_bounds=false;
    auto partial_plain=solve_full(d,paths,options,&x);
    const double pa=partial_plain.at("objective"),pb=partial_tight.at("objective");
    require(std::abs(pa-pb)<=1e-5*(1+std::abs(pa)),"Task bound changed unassigned-second fixed recourse");
    if(cardinality==2)require(partial_tight.at("served").get<double>()>0,"Unassigned-second witness has no service");
    return {{"bases",bases},{"max_orders",cardinality},{"path_count",paths.service_paths.size()},
        {"assignment_arc_checks",checks},{"changed_link_rows",changed},
        {"objective_plain",a},{"objective_task_bound",b},{"objective_difference",std::abs(a-b)},
        {"partial_objective_difference",std::abs(pa-pb)},{"bounds",bounds.metadata},{"passed",true}};
}
}
int main() {
    try {
        Json cases=Json::array();for(int bases:{1,2})for(int k:{1,2})cases.push_back(test_case(bases,k));
        auto d=fixture(1,2);auto paths=enumerate_paths(d);paths.complete=false;bool rejected=false;
        try{build_task_airline_bounds(d,paths);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Incomplete catalogue accepted");
        std::cout<<Json({{"cases",cases},{"incomplete_catalogue_rejected",rejected},{"passed",true}}).dump(2)<<'\n';
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
