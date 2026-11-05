#include "mlnd/formulation.hpp"
#include "mlnd/preprocessed_demand.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool ok,const std::string& why) { if (!ok) throw std::runtime_error(why); }
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> result(rows,cols);
    std::fill(result.values.begin(),result.values.end(),value);
    return result;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.name="independent_task_witness";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand={4,7};d.ground_cost={5,5};d.deadline={1,1};
    d.flight_times=filled(4,4,0.0);
    d.flight_times.values={0,.03,.15,.17,.03,0,.16,.14,.15,.16,0,.1,.17,.14,.1,0};
    d.allowed_links=filled(4,4,1);for(int i=0;i<4;++i)d.allowed_links(i,i)=0;
    d.link_cost=filled(4,4,0.0);d.base_cost={.5,.4};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,0.0);d.kappa(0,1)=1;
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;
    d.flight_cost=10;d.waiting_cost=.3;
    return d;
}
Json independent_witness(bool compact) {
    auto d=fixture();auto paths=enumerate_paths(d);preprocess_generalized_demand(d,paths);
    const double expected=4*(-std::expm1(-7*d.delta));
    require(std::abs(d.pair_rates(0,1)-expected)<1e-12,"Task reconstruction changed");
    require(d.task_count()==6&&d.assignment_count()==12,"Incorrect task assignment dimensions");
    for(int q=0;q<d.task_count();++q)
        require(d.task_index(d.task_first(q),d.task_second(q))==q,"Task ID does not round trip");
    FormulationStrengthening strengthening;
    strengthening.compact_task_heads=compact;
    strengthening.eliminate_zero_task_columns=true;
    strengthening.assignment_incident_clique=true;
    strengthening.shared_rho_joint=true;
    strengthening.queue_perspective=true;
    strengthening.task_airline_upper=build_task_airline_bounds(d,paths).upper;
    auto f=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,true,strengthening);
    std::map<std::string,int> var;
    for(std::size_t i=0;i<f.matrix.variables.size();++i)var[f.matrix.variables[i].name]=i;
    std::vector<double>x(f.matrix.variables.size()),served(d.n_od),heads(f.lift_groups.size());
    for(int n=0;n<d.n_nodes();++n)x[n]=1;
    for(int h=0;h<d.n_h;++h)x[f.layout.capacity+h]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[f.layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    const std::vector<double> rho{.1,.2};
    for(int l=0;l<d.n_l;++l) {
        x[f.rho[l]]=rho[l];x[f.idle[l]]=rho[l]/(1-rho[l]);
        const auto c=f.matrix.cones[l];
        x[c.up]=x[f.idle[l]]+1;x[c.down]=1-rho[l];x[c.one]=1;
    }
    double busy=0;
    for(const auto& assignment:std::vector<std::pair<int,int>>{{d.task_index(0),0},{d.task_index(0,1),1}}) {
        const int q=assignment.first,l=assignment.second;
        x[f.layout.assignment_index(q,l)]=1;
        x[var.at("v_"+std::to_string(q)+"_"+std::to_string(l))]=rho[l];
        for(int destination=0;destination<d.n_l;++destination) {
            const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p) {
                return p.first_od==d.task_first(q)&&p.second_od==d.task_second(q)&&p.b==BasePair{l,destination};
            });
            require(p!=paths.service_paths.end(),"Missing independent-task witness path");
            const double flow=d.task_rate(q)*rho[l]/d.n_l;
            x[f.flow[p->id]]=flow;
            if(!compact)x[f.head[p->segment_id]]+=flow;
            heads[f.segment_group[p->segment_id]]+=flow;busy+=flow*p->time;
            for(auto [r,count]:p->served_counts)served[r]+=count*flow;
            if(destination!=l) {
                const auto empty=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[&](const auto& e) {
                    return e.b==BasePair{destination,l};
                });
                require(empty!=paths.empty_paths.end(),"Missing balancing relocation");
                x[f.empty_flow[empty->id]]+=flow;busy+=flow*empty->time;
            }
        }
    }
    if(!compact)for(std::size_t g=0;g<heads.size();++g)
        x[var.at("group_head_"+std::to_string(g))]=heads[g];
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r]-served[r];
    x[f.fleet]=busy;for(int idle:f.idle)x[f.fleet]+=x[idle];
    const auto residual=point_residuals(f.matrix,x);
    require(residual.at("passed"),"Independent task point failed: "+residual.dump());
    require(x[f.layout.assignment_index(d.task_index(0),0)]==1&&
            x[f.layout.assignment_index(d.task_index(0,1),1)]==1,
            "Tasks sharing their first OD cannot choose distinct departure bases");
    require(x[f.layout.assignment_index(d.task_index(1),0)]==0&&served[1]>0,
            "Pair service incorrectly needs the second single-task assignment");
    auto wrong=x;wrong[var.at("v_"+std::to_string(d.task_index(0,1))+"_1")]=rho[0];
    require(!point_residuals(f.matrix,wrong).at("passed").get<bool>(),
            "A task was allowed to use its head's other departure service fraction");
    double objective=0;
    for(std::size_t i=0;i<x.size();++i)objective+=x[i]*f.matrix.variables[i].cost;
    return {{"compact",compact},{"residuals",residual},{"objective",objective},
        {"assignment_variables",d.assignment_count()},{"task_equations",f.task_rows.size()},
        {"single_departure",0},{"pair_departure",1},{"independent_service_fractions",rho}};
}
Json edge_cases() {
    auto d=fixture();require(DesignLayout(d).airline-DesignLayout(d).assignment==8,"Legacy K2 layout changed");
    d.max_orders=1;require(d.assignment_count()==4,"Legacy K1 layout changed");
    auto p=enumerate_paths(d);preprocess_generalized_demand(d,p);
    require(d.task_count()==2&&d.assignment_count()==4,"K1 task layout is not single-only");
    const auto single=build_formulation(d,p);
    require(single.task_rows.size()==4,"K1 is missing task equations");
    auto missing=fixture();missing.allowed_links=filled(4,4,0);
    const auto missing_paths=enumerate_paths(missing);preprocess_generalized_demand(missing,missing_paths);
    const auto mf=build_formulation(missing,missing_paths);
    require(mf.task_rows.size()==static_cast<std::size_t>(missing.task_count()*missing.n_l),
            "Unsupported task equations were omitted");
    for(int q=0;q<missing.task_count();++q)for(int l=0;l<missing.n_l;++l) {
        const auto& row=mf.matrix.rows[mf.task_rows[q*missing.n_l+l]];
        require(row.sense=='E'&&row.rhs==0&&row.coefficients.size()==1,
                "Unsupported task did not force its executed assignment to zero");
        require(row.coefficients[0].second==missing.task_rate(q),"Incorrect unsupported task rate");
    }
    auto first_leg=fixture();auto fp=enumerate_paths(first_leg);preprocess_generalized_demand(first_leg,fp);
    first_leg.first_leg_consistency=true;bool rejected=false;
    try{build_formulation(first_leg,fp);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Historical first-leg coupling silently changed independent task semantics");
    return {{"legacy_layouts_preserved",true},{"K1_task_equations",single.task_rows.size()},
        {"unsupported_task_equations",mf.task_rows.size()},{"first_leg_coupling_rejected",rejected}};
}
Json independent_airline_bound() {
    auto d=fixture();d.preprocessed_demand=true;d.single_rates={1,0};
    d.pair_rates=filled(2,2,0.0);d.pair_rates(0,1)=1;d.head_rates={2,0};d.demand={2,1};
    PathCollection paths;
    ServicePath single;single.first_od=0;single.b={0,0};single.arc_counts={{2,3,1}};
    ServicePath pair;pair.first_od=0;pair.second_od=1;pair.b={1,1};pair.arc_counts={{2,3,1}};
    paths.service_paths={single,pair};
    const auto bound=build_task_airline_bounds(d,paths);
    require(bound.upper[2*d.n_nodes()+3]>=2&&bound.upper[2*d.n_nodes()+3]<2+1e-9,
            "Airline bound retained unsafe head max-of-sum coupling");
    return {{"independent_task_load",2},{"historical_head_bound",1},
        {"valid_task_bound",bound.upper[2*d.n_nodes()+3]}};
}
}
int main() {try {
    const auto original=independent_witness(false),compact=independent_witness(true);
    require(std::abs(original.at("objective").get<double>()-compact.at("objective").get<double>())<1e-10,
            "Compact projection changed the independent-task objective");
    std::cout<<Json{{"passed",true},{"independent_task_witness",original},{"compact_witness",compact},
        {"edge_cases",edge_cases()},{"airline_bound",independent_airline_bound()}}.dump(2)<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
