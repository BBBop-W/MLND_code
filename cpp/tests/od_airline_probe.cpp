// Regression certificates for integer-valid OD airline resource projections.
// Full paths and fixed-design SOCPs below are validation oracles only.
#include "mlnd/strengthening.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

namespace {
using namespace mlnd;
void require(bool value,const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows,int columns,T value) {
    Matrix<T> out(rows,columns);
    std::fill(out.values.begin(),out.values.end(),value); return out;
}
Input fixture(int k,const std::string& policy,int bases=1,int hubs=2) {
    Input d;
    d.n_l=bases; d.n_h=hubs; d.n_od=2; d.n_levels=1; d.max_orders=k;
    d.name="od_airline_regression"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy=policy;
    d.demand={4,7}; d.ground_cost={5,5}; d.deadline={1,1};
    d.base_cost.assign(bases,.5); d.slot_epsilon.assign(hubs,.05);
    const int N=d.n_nodes();
    d.flight_times=filled(N,N,.1); d.allowed_links=filled(N,N,1);
    for (int i=0;i<N;++i) {d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    for (int l=0;l<bases;++l) for (int h=0;h<hubs;++h)
        d.flight_times(l,bases+h)=d.flight_times(bases+h,l)=.05+.02*h+.01*l;
    d.link_cost=filled(N,N,.01);
    d.rider_mean=filled(2,hubs,.08); d.first_cost=filled(2,hubs,1.0);
    d.terminal_cost=filled(2,hubs,.5); d.receipt_mean=filled(2,hubs,.08);
    d.removal_mean=filled(2,hubs,.03); d.slot_counts=filled(hubs,1,10);
    d.slot_cost=filled(hubs,1,.1); d.occupancy_limit=filled(hubs,1,6.1690072893953225);
    d.kappa=Matrix<double>(2,2); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1;
    d.flight_cost=.2; d.waiting_cost=.3;
    return d;
}
std::vector<double> design(const Input& d,int assignment_mask,double airline_fraction=1) {
    const DesignLayout layout(d);
    std::vector<double> out(layout.count,0);
    for (int i=0;i<d.n_nodes();++i) out[layout.facility+i]=1;
    for (int j=0;j<d.n_h;++j) out[layout.capacity+j*d.n_levels]=1;
    for (int r=0;r<d.n_od;++r) out[layout.assignment+r*d.n_l*d.n_l]=(assignment_mask>>r)&1;
    for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
        out[layout.airline+u*d.n_nodes()+v]=airline_fraction*d.allowed_links(u,v);
    return out;
}
double activity(const Coefficients& row,const std::vector<double>& point) {
    long double sum=0;
    for (auto [j,c]:row) sum+=static_cast<long double>(c)*point.at(j);
    return static_cast<double>(sum);
}
double violation(const RowRecord& row,const std::vector<double>& point) {
    return row.rhs-activity(row.coefficients,point);
}
std::map<int,const ResourceSavingsOccurrence*> occurrence_index(const ResourceSavingsModel& model,int H) {
    std::map<int,const ResourceSavingsOccurrence*> result;
    for (const auto& o:model.occurrences)
        require(result.emplace(o.group*H+o.locker,&o).second,"Duplicate resource occurrence");
    return result;
}
Json certify_mapping(const Input& d,bool require_second_exceptions=false) {
    const auto paths=enumerate_paths(d);
    const auto allocations=capacity_savings_network(d);
    const auto legacy=resource_savings_model(d,allocations,true,false,false);
    const auto model=resource_savings_model(d,allocations,true,false,true);
    const auto standalone=resource_savings_model(d,allocations,false,false,true);
    const DesignLayout layout(d);
    const int old_rows=d.n_h+2*d.n_l,extra=d.n_od*(2*d.n_l+d.n_h);
    require(model.metadata.at("validity_scope")=="integer_design_convex_hull" &&
        legacy.metadata.at("validity_scope")=="original_continuous_relaxation",
        "The strengthened and original continuous domains were not distinguished");
    require(model.airline_rhs.size()==static_cast<std::size_t>(old_rows+extra) &&
        standalone.airline_rhs.size()==static_cast<std::size_t>(extra),"Wrong OD row count");
    for (int a=0;a<old_rows;++a)
        require(model.airline_rhs[a]==legacy.airline_rhs[a],"OD option changed a legacy airline gate");
    for (int a=0;a<extra;++a)
        require(model.airline_rhs[old_rows+a]==standalone.airline_rhs[a],"OD gates depend on legacy gates");
    require(model.group_rhs==legacy.group_rhs && model.locker_rhs==legacy.locker_rhs &&
            model.occurrences.size()==legacy.occurrences.size(),"OD option changed the underlying resource LP");
    const auto occurrences=occurrence_index(model,d.n_h);
    std::size_t occurrence_checks=0,same_od_paths=0,short_second_paths=0,no_base_access_second_paths=0;
    double max_gate_violation=0;
    for (const auto& p:paths.service_paths) {
        std::vector<double> x(layout.count,0),use(model.airline_rhs.size(),0);
        for (auto [u,v,count]:p.arc_counts) x[layout.airline+u*d.n_nodes()+v]=1;
        double rate=INFINITY;
        for (auto [r,count]:p.served_counts) rate=std::min(rate,d.demand[r]/count);
        for (std::size_t k=0;k<p.deliveries.size();++k) {
            const int r=k==0?p.first_od:p.second_od;
            const int group=(r*d.n_l+p.b[0])*d.n_l+p.b[1];
            const auto found=occurrences.find(group*d.n_h+p.deliveries[k]);
            require(allocations.eligible[group*d.n_h+p.deliveries[k]],"OD support excluded a physical occurrence");
            if (found==occurrences.end()) continue; // Zero-saving occurrences need no auxiliary variable.
            const auto& occurrence=*found->second;
            require(occurrence.airline_resources.size()==6,"An occurrence lacks an OD gate");
            for (int a:occurrence.airline_resources) use[a]+=rate;
            const int delivery_row=old_rows+2*d.n_od*d.n_l+r*d.n_h+p.deliveries[k];
            const int loaded_link=layout.airline+(d.n_l+p.pickups[k])*d.n_nodes()+d.n_l+p.deliveries[k];
            require(std::any_of(model.airline_rhs[delivery_row].begin(),model.airline_rhs[delivery_row].end(),
                [&](const auto& coefficient){return coefficient.first==loaded_link;}),
                "Eligible pickup set omitted a physical loaded link");
            ++occurrence_checks;
        }
        for (std::size_t a=0;a<use.size();++a)
            max_gate_violation=std::max(max_gate_violation,use[a]-activity(model.airline_rhs[a],x));
        if (p.second_od>=0) {
            same_od_paths+=p.first_od==p.second_od;
            short_second_paths+=d.flight_times(d.n_l+p.pickups[1],d.n_l+p.deliveries[1])<d.effective_delta();
            no_base_access_second_paths+=!d.allowed_links(p.b[0],d.n_l+p.pickups[1]);
        }
    }
    require(max_gate_violation<=1e-9,"OD gate cuts off a demand-normalized physical task mapping");
    if (d.max_orders==2) require(same_od_paths>0,"K2 regression did not exercise repeated OD occurrences");
    if (require_second_exceptions)
        require(short_second_paths>0 && no_base_access_second_paths>0,"Second-order exception fixture is empty");
    return {{"K",d.max_orders},{"policy",d.flow_policy},{"bases",d.n_l},{"hubs",d.n_h},
        {"service_paths",paths.service_paths.size()},{"occurrence_checks",occurrence_checks},
        {"same_od_paths",same_od_paths},{"second_loaded_below_delta",short_second_paths},
        {"second_without_departure_base_access",no_base_access_second_paths},
        {"max_gate_violation",max_gate_violation},{"metadata",model.metadata}};
}
// Put exactly the new auxiliary OD rows on their actual physical-flow image.
// This is an independent validity oracle; production only projects LP duals.
void add_actual_flow_gates(Formulation& f,const Input& d,const PathCollection& paths,
                           const ResourceSavingsModel& model) {
    const int first=model.metadata.at("legacy_airline_resource_rows");
    const auto occurrences=occurrence_index(model,d.n_h);
    std::vector<Coefficients> rows(model.airline_rhs.size());
    for (int a=first;a<static_cast<int>(rows.size());++a)
        for (auto [j,c]:model.airline_rhs[a]) rows[a].emplace_back(j,-c);
    for (const auto& p:paths.service_paths) {
        std::map<int,int> counts;
        for (std::size_t k=0;k<p.deliveries.size();++k) {
            const int r=k==0?p.first_od:p.second_od;
            const int g=(r*d.n_l+p.b[0])*d.n_l+p.b[1];
            const auto found=occurrences.find(g*d.n_h+p.deliveries[k]);
            if (found==occurrences.end()) continue;
            for (int a:found->second->airline_resources) if (a>=first) ++counts[a];
        }
        for (auto [a,count]:counts) rows[a].emplace_back(f.flow[p.id],count);
    }
    for (int a=first;a<static_cast<int>(rows.size());++a)
        f.matrix.row("od_airline_actual_"+std::to_string(a),'L',0,std::move(rows[a]));
}
Json cross_design_checks(int k,const std::string& policy) {
    const auto d=fixture(k,policy);
    const auto paths=enumerate_paths(d);
    const auto model=resource_savings_model(d,true,false,true);
    SolverOptions options; options.threads=1; options.gap=1e-9;
    ResourceSavingsRelaxation resource(model,options);
    const DesignLayout layout(d);
    std::vector<ResourceSavingsCut> cuts;
    for (int mask=0;mask<4;++mask) for (double fraction:{0.0,.025,.2,1.0}) {
        const auto x=design(d,mask,fraction);
        for (double budget:{0.0,5.0}) {
            auto cut=resource.separate(x,0,layout.count,budget);
            require(cut.valid && cut.max_dual_violation_after==0,"Repaired resource dual is invalid");
            require(budget>0 ? cut.optimal&&!cut.fallback : cut.fallback&&!cut.optimal,
                    "Resource dual certificate misreports optimization/fallback");
            cuts.push_back(std::move(cut));
        }
    }
    // Every assignment combination and each single-link closure, plus the
    // fully connected and disconnected cases. Both demand formulations use
    // full physical paths. The test covers both prefix and aggregate policies.
    std::vector<std::pair<int,int>> links;
    for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
        if (d.allowed_links(u,v)) links.emplace_back(u,v);
    std::size_t solves=0,cross_checks=0;
    double max_cross_violation=0,max_objective_difference=0,max_service=0;
    for (int mask=0;mask<4;++mask) for (int close=-2;close<static_cast<int>(links.size());++close) {
        auto x=design(d,mask,close==-2?0:1);
        if (close>=0) x[layout.airline+links[close].first*d.n_nodes()+links[close].second]=0;
        require(point_residuals(design_model(d).matrix,x).at("passed"),"Invalid binary test design");
        for (bool conditional:{false,true}) {
            auto original=build_formulation(d,paths,&x,nullptr,nullptr,false,conditional);
            auto strengthened=original;
            add_actual_flow_gates(strengthened,d,paths,model);
            NativeSolver a(original.matrix,options,"od_gate_original");
            NativeSolver b(strengthened.matrix,options,"od_gate_strengthened");
            auto old=a.optimize(5),fresh=b.optimize(5);
            require(old.has_solution && old.optimal && fresh.has_solution && fresh.optimal,
                "Fixed-design SOCP comparison lacks optimal solutions: "+old.status+" / "+fresh.status);
            require(repair_inventory_primal(original,old.values).at("passed") &&
                repair_inventory_primal(strengthened,fresh.values).at("passed"),"SOCP primal repair failed");
            require(point_residuals(strengthened.matrix,old.values).at("passed") &&
                point_residuals(original.matrix,fresh.values).at("passed"),
                "New gate excluded an original integer-design feasible solution");
            const double original_cost=objective_value(original.matrix,old.values);
            const double new_cost=objective_value(strengthened.matrix,fresh.values);
            max_objective_difference=std::max(max_objective_difference,std::abs(original_cost-new_cost));
            require(std::abs(original_cost-new_cost)<=2e-7*(1+model.courier_cost),
                "Integer-design operational optimum changed");
            auto point=x; point.push_back(original_cost);
            for (const auto& cut:cuts) {
                max_cross_violation=std::max(max_cross_violation,violation(cut.row,point));
                ++cross_checks;
            }
            double service=0;
            for (int j:original.flow) if (j>=0) service+=old.values[j];
            max_service=std::max(max_service,service);
            ++solves;
        }
    }
    require(max_cross_violation<=2e-7*(1+model.courier_cost),"A projected cut excludes another binary design");
    require(max_service>1,"Integer equivalence tested only zero service");
    return {{"K",k},{"policy",policy},{"fixed_design_pairs",solves},{"projected_cuts",cuts.size()},
        {"cross_design_checks",cross_checks},{"max_cross_violation",max_cross_violation},
        {"max_objective_difference",max_objective_difference},{"max_service_task_rate",max_service}};
}
Json fractional_counterexample() {
    auto d=fixture(1,"aggregate");
    d.demand={1,9}; d.ground_cost={10,10}; d.deadline={1,.05};
    d.allowed_links=filled(3,3,0);
    d.allowed_links(0,1)=d.allowed_links(1,2)=d.allowed_links(2,0)=1;
    d.flight_times=filled(3,3,.1);
    for (int i=0;i<3;++i) d.flight_times(i,i)=0;
    d.rider_mean=filled(2,2,0.0); d.receipt_mean=filled(2,2,0.0);
    d.removal_mean=filled(2,2,0.0); d.first_cost=filled(2,2,0.0);
    d.terminal_cost=filled(2,2,0.0); d.occupancy_limit=filled(2,1,100.0);
    d.flight_cost=d.waiting_cost=0;
    const auto paths=enumerate_paths(d);
    require(paths.service_paths.size()==1,"Fractional fixture must have exactly one feasible path");
    auto x=design(d,1,.05);
    auto f=build_formulation(d,paths,&x,nullptr,nullptr,true);
    f.matrix.row("counterexample_flow",'E',.5,{{f.flow[0],1}});
    SolverOptions options; options.threads=1; options.gap=1e-9;
    NativeSolver solver(f.matrix,options,"od_fractional_counterexample");
    auto solution=solver.optimize(5);
    require(solution.has_solution && solution.optimal,"Original fractional counterexample is infeasible");
    require(repair_inventory_primal(f,solution.values).at("passed") &&
        point_residuals(f.matrix,solution.values).at("passed"),"Counterexample is not physically validated");
    const auto old=resource_savings_model(d,true,false,false);
    const auto fresh=resource_savings_model(d,true,false,true);
    ResourceSavingsRelaxation old_lp(old,options),new_lp(fresh,options);
    const int theta=DesignLayout(d).count;
    const auto a=old_lp.separate(x,0,theta,5),b=new_lp.separate(x,0,theta,5);
    auto projected=x; projected.push_back(objective_value(f.matrix,solution.values));
    require(a.valid && b.valid && a.optimal && b.optimal,"Fractional projection oracle failed");
    require(a.dual_savings>b.dual_savings+4 && violation(a.row,projected)<1e-7 &&
        violation(b.row,projected)>3,"Integer-valid projection failed to strictly cut the original fractional domain");
    return {{"original_fractional_primal_validated",true},{"original_operating_cost",projected.back()},
        {"legacy_savings_upper_bound",a.dual_savings},{"od_savings_upper_bound",b.dual_savings},
        {"legacy_cut_violation",violation(a.row,projected)},
        {"od_cut_violation",violation(b.row,projected)},
        {"scope","Expected exclusion: original continuous feasibility is not strengthened-root feasibility"}};
}
Json incompatible_pickup_fixture() {
    auto d=fixture(1,"aggregate",1,3);
    d.demand={1,9}; d.deadline={.15,.01};
    d.receipt_mean=filled(2,3,0.0); d.occupancy_limit=filled(3,1,100.0);
    d.allowed_links=filled(4,4,0);
    d.allowed_links(0,1)=d.allowed_links(0,2)=d.allowed_links(1,3)=
        d.allowed_links(2,3)=d.allowed_links(3,0)=1;
    d.flight_times(1,3)=.05; d.flight_times(2,3)=.2;
    auto x=design(d,1); const DesignLayout layout(d);
    x[layout.airline+1*4+3]=0; // Only the OD-incompatible incoming airline is open.
    SolverOptions options; options.threads=1;
    ResourceSavingsRelaxation legacy(resource_savings_model(d,true,false,false),options);
    ResourceSavingsRelaxation fresh(resource_savings_model(d,true,false,true),options);
    const auto a=legacy.separate(x,0,layout.count,5),b=fresh.separate(x,0,layout.count,5);
    require(a.valid && b.valid && a.optimal && b.optimal && a.dual_savings>1 && b.dual_savings<1e-7,
        "Eligible pickup gate did not detect an integer-design incompatibility");
    return {{"legacy_savings_upper_bound",a.dual_savings},{"od_savings_upper_bound",b.dual_savings},
        {"integer_design",true},{"incompatible_loaded_flight",.2},{"deadline",.15}};
}
} // namespace

int main(int argc,char** argv) {
    try {
        if (argc<2 || argc>3) throw std::invalid_argument("Usage: mlnd_od_airline_probe OUTPUT_JSON [STANDARD_INPUT]");
        Json mapping=Json::array(),equivalence=Json::array();
        for (const std::string policy:{"aggregate","prefix"}) for (int k:{1,2}) {
            mapping.push_back(certify_mapping(fixture(k,policy)));
            auto rich=fixture(k,policy,2,3);
            rich.flight_times(3,4)=.01;
            rich.allowed_links(0,3)=rich.allowed_links(1,3)=0;
            mapping.push_back(certify_mapping(rich,k==2));
            equivalence.push_back(cross_design_checks(k,policy));
        }
        if (argc==3) for (int k:{1,2}) mapping.push_back(certify_mapping(read_input(argv[2],k)));
        Json result={{"passed",true},{"physical_mapping",mapping},{"integer_design_equivalence",equivalence},
            {"fractional_domain_counterexample",fractional_counterexample()},
            {"eligible_pickup_integer_strengthening",incompatible_pickup_fixture()},
            {"validity_scope","integer_design_convex_hull"},{"master_added_variables",0}};
        std::ofstream stream(argv[1]);
        if (!stream) throw std::runtime_error("Cannot write output JSON");
        stream<<result.dump(2)<<'\n';
        std::cout<<"OD airline probe passed: "<<mapping.size()<<" complete physical catalogues; "
                 <<equivalence.size()<<" K/policy integer-design equivalence suites\n";
    } catch (const IloException& error) {
        std::cerr<<error<<'\n';return 1;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
