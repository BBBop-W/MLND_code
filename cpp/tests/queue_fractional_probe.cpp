// Public current-master LP separation witnesses; all physical cases have two bases.
#include "mlnd/queue_fractional.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=1;
    d.name="current_master_fractional_queue";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.single_rates=d.head_rates=d.demand;d.pair_rates=filled(2,2,0.0);
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);
    for(int u=0;u<4;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.01);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
std::vector<double> opened(const Input& d,int first,int second) {
    const DesignLayout l(d);std::vector<double> x(l.count,0);
    for(int u=0;u<d.n_nodes();++u)x[u]=1;
    for(int h=0;h<d.n_h;++h)x[l.capacity+h]=1;
    for(int u=0;u<4;++u)for(int v=0;v<4;++v)x[l.airline+u*4+v]=d.allowed_links(u,v);
    if(first>=0)x[l.assignment_index(0,first)]=1;
    if(second>=0)x[l.assignment_index(1,second)]=1;
    return x;
}
double row_value(const TaskSavingsCut& cut,const std::vector<double>& x,int theta) {
    require(cut.valid,"Invalid public queue cut");long double rhs=cut.row.rhs;
    for(const auto& [j,c]:cut.row.coefficients)if(j!=theta)rhs-=c*x.at(j);
    return static_cast<double>(rhs);
}
QueueNativePrices zero_donor(const Input& d) {
    QueueNativePrices donor;donor.valid=true;donor.status="zero_resource_prices";
    donor.prices.source="public_zero_donor";donor.prices.locker.assign(d.n_h,0);
    donor.prices.airline.assign(d.n_nodes()*d.n_nodes(),0);donor.prices.balance.assign(d.n_l,0);
    return donor;
}
Json fractional_hand_tracer() {
    auto d=fixture();const auto paths=enumerate_paths(d);const DesignLayout l(d);
    for(int r=0;r<2;++r) {
        double cheapest=std::numeric_limits<double>::infinity();
        for(const auto& path:paths.service_paths)if(path.first_od==r&&path.second_od==-1)
            cheapest=std::min(cheapest,path.cost+d.fleet_cost*path.time);
        d.ground_cost[r]=4+cheapest;
    }
    const double courier=d.ground_cost[0]+d.ground_cost[1];
    SolverOptions options;options.threads=1;options.seed=43;options.pareto=false;
    QueuePolymatroidCuts pool(d,paths,options);const auto anchor=opened(d,0,-1);
    // Only the first head's base-0 assignment is free; this is a controlled
    // master snapshot on physically feasible opened infrastructure. The .4
    // assignment cost is an explicit test objective, not a production metric.
    LinearModel master;master.constant=2.5;
    const int free=l.assignment_index(0,0);
    for(int j=0;j<l.count;++j)master.variable("x_"+std::to_string(j),
        j==free?0:anchor[j],j==free?1:anchor[j],j==free?.4:0,true);
    const int theta=master.variable("theta",0,IloInfinity,1,false);
    Coefficients floor{{theta,1}},zero{{theta,1}};
    for(int r=0;r<2;++r)for(int base=0;base<2;++base) {
        const int j=l.assignment_index(r,base);zero.emplace_back(j,4);
        if(j!=free)floor.emplace_back(j,4);
    }
    // Continuous-valid analytical recourse supports: busy net savings <=4v,
    // v<=alpha, and min_rho rho/(1-rho)-4rho=-1 for the distinguished head.
    master.row("ordinary_anchor_floor",'G',courier-1,floor);
    master.row("ordinary_zero_support",'G',courier,zero);
    const auto old_queue=pool.separate_profile(0,opened(d,-1,0),0,theta);
    master.rows.push_back(old_queue.row);
    const auto at_anchor=pool.separate(anchor,courier-1,theta);
    require(std::abs(at_anchor.violation)<1e-5,"Integer queue value is not closed by the anchor support");
    const auto result=separate_fractional_queue(master,l,theta,pool,zero_donor(d),
        std::set<std::string>{old_queue.key},options,2,1,1,3);
    const double expected=2.5+courier-.6;
    require(result.cuts.size()==1,"Current LP envelope did not produce exactly one needed queue row");
    require(std::isfinite(result.certified_lower)&&std::abs(result.certified_lower-expected)<1e-4,
            "Reoptimized current-master LP missed the hand-derived certified bound");
    const auto& rounds=result.diagnostics.at("rounds");
    require(rounds.size()>=2&&rounds.at(0).at("fractional_coordinates").get<int>()>0&&
            rounds.at(0).at("accepted").get<bool>(),"No actual fractional LP point was cut");
    require(rounds.at(0).at("cut_violation").get<double>()>
            rounds.at(0).at("acceptance_threshold").get<double>(),"Accepted row lacked current-theta marginal value");
    require(master.rows.size()==3&&master.variables[free].binary&&master.constant==2.5,
            "Fractional separation mutated its input master");
    require(result.lp_seconds>=0&&result.price_seconds>=0&&
            result.wall+1e-7>=result.lp_seconds+result.price_seconds,"LP and price times overlap");
    double price_sum=0;
    for(const auto& record:result.price_records) {
        require(record.contains("diagnostics")&&record.contains("time_budget")&&record.contains("accepted"),
                "Price record lost the existing nested audit schema");
        price_sum+=record.at("seconds").get<double>();
    }
    require(std::abs(price_sum-result.price_seconds)<1e-9,"Price ledger is not the sum of its disjoint calls");

    // Independent physical recourse points; no assertion is made about the
    // original fractional McCormick relaxation, only integer epigraph mixtures.
    struct Point {std::vector<double> design;double cost;};
    std::vector<Point> points;double anchor_recourse=0;
    for(int first=-1;first<2;++first)for(int second=-1;second<2;++second) {
        const auto x=opened(d,first,second);
        const auto full=build_formulation(d,paths,&x);
        NativeSolver oracle(full.matrix,options,"fractional_queue_full_recourse_oracle");
        const auto solved=oracle.optimize(20);
        require(solved.optimal&&solved.has_solution&&point_residuals(full.matrix,solved.values).at("passed").get<bool>(),
                "Independent full physical recourse did not certify a feasible optimum");
        const double cost=objective_value(full.matrix,solved.values);
        points.push_back({x,cost});if(first==0&&second==-1)anchor_recourse=cost;
    }
    require(std::abs(anchor_recourse-(courier-1))<1e-4,
            "Analytical integer-anchor support is not tight against physical recourse");
    const std::vector<double> closed(l.count,0);
    const auto all_closed=build_formulation(d,paths,&closed);
    NativeSolver closed_oracle(all_closed.matrix,options,"fractional_queue_closed_recourse_oracle");
    const auto closed_result=closed_oracle.optimize(20);
    require(closed_result.optimal&&closed_result.has_solution&&
            point_residuals(all_closed.matrix,closed_result.values).at("passed").get<bool>(),
            "All-closed physical design was not certified");
    points.push_back({closed,objective_value(all_closed.matrix,closed_result.values)});
    const auto integer_count=points.size();
    for(std::size_t j=1;j<integer_count;++j) {
        auto mixed=points[j];
        for(std::size_t k=0;k<mixed.design.size();++k)
            mixed.design[k]=.37*points[j].design[k]+.63*points[0].design[k];
        mixed.cost=.37*points[j].cost+.63*points[0].cost;points.push_back(std::move(mixed));
    }
    double maximum_violation=0;
    for(const auto& cut:result.cuts)for(const auto& point:points) {
        const double violation=row_value(cut,point.design,theta)-point.cost;
        maximum_violation=std::max(maximum_violation,violation);
        require(violation<=1e-6*(1+std::abs(point.cost)),"Fractional-master row excludes original integer recourse");
    }
    const auto no_price=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,2,1,0,3);
    require(no_price.cuts.empty()&&no_price.price_records.empty()&&std::isfinite(no_price.certified_lower)&&
            no_price.diagnostics.at("reason")=="price_budget_exhausted","Zero price budget lost its certified LP-only bound");
    const auto one_round=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,2,1,1,1);
    require(one_round.cuts.empty()&&one_round.price_records.empty()&&
            one_round.diagnostics.at("reason")=="final_lp_certification","Last LP round incorrectly added an un-reoptimized cut");
    return {{"integer_anchor_queue_minus_support",at_anchor.violation},{"expected_after",expected},
        {"certified_after",result.certified_lower},{"diagnostics",result.diagnostics},
        {"price_records",result.price_records},{"integer_full_recourse_solves",integer_count},
        {"convex_mixtures",points.size()-integer_count},{"cross_checks",result.cuts.size()*points.size()},
        {"maximum_violation",maximum_violation},{"anchor_physical_recourse",anchor_recourse},
        {"zero_price_reason",no_price.diagnostics.at("reason")},{"passed",true}};
}

Json certification_guards() {
    const auto d=fixture();const auto paths=enumerate_paths(d);const DesignLayout l(d);
    SolverOptions options;options.threads=1;options.seed=43;options.pareto=false;
    QueuePolymatroidCuts pool(d,paths,options);const auto count=pool.profile_count();
    LinearModel master;master.constant=2;
    for(int j=0;j<l.count;++j)master.variable("fixed_"+std::to_string(j),0,0,0,true);
    const int theta=master.variable("theta",0,IloInfinity,1,false);
    master.row("current_rhs",'G',5,{{theta,1}});
    const auto zero=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,0,1,1);
    require(!std::isfinite(zero.certified_lower)&&zero.cuts.empty()&&zero.price_records.empty()&&
            zero.diagnostics.at("lp_solves")==0&&pool.profile_count()==count,"Zero total budget solved or mutated state");
    const auto tiny=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,2,
        std::numeric_limits<double>::min(),1);
    require(!std::isfinite(tiny.certified_lower)&&tiny.cuts.empty(),"Exhausted LP budget produced a bound");
    const auto integer=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,2,1,1);
    require(std::abs(integer.certified_lower-(7-8e-7))<1e-8&&integer.price_records.empty()&&
            integer.diagnostics.at("reason")=="integer_lp_point","Integer point did not use LP-only certification");
    master.rows.front().rhs=6;
    const auto updated=separate_fractional_queue(master,l,theta,pool,zero_donor(d),{},options,2,1,1);
    require(std::abs(updated.certified_lower-(8-9e-7))<1e-8,"Current RHS or objective constant was not copied");
    auto impossible=master;impossible.variables[theta].upper=4;
    const auto infeasible=separate_fractional_queue(impossible,l,theta,pool,zero_donor(d),{},options,2,1,1);
    require(!std::isfinite(infeasible.certified_lower)&&infeasible.cuts.empty()&&
            infeasible.diagnostics.at("reason")=="lp_not_certified", "Infeasible LP was promoted to a certified objective bound");
    auto unbounded=master;unbounded.variable("unbounded_direction",0,IloInfinity,-1,false);
    const auto ray=separate_fractional_queue(unbounded,l,theta,pool,zero_donor(d),{},options,2,1,1);
    require(!std::isfinite(ray.certified_lower)&&ray.cuts.empty()&&ray.diagnostics.at("reason")=="lp_not_certified",
            "Non-optimal unbounded LP was promoted to a certified objective bound");
    return {{"zero_budget",zero.diagnostics.at("reason")},{"tiny_lp_budget",tiny.diagnostics.at("reason")},
        {"integer_point",integer.diagnostics.at("reason")},{"updated_rhs_bound",updated.certified_lower},
        {"infeasible_status",infeasible.diagnostics.at("rounds").at(0).at("status")},
        {"unbounded_status",ray.diagnostics.at("rounds").at(0).at("status")},{"passed",true}};
}
}
int main(){try {std::cout<<Json({{"fractional_hand",fractional_hand_tracer()},
    {"certification_guards",certification_guards()},{"passed",true}}).dump(2)<<'\n';return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
