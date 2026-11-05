// Public LP-separation integration at a controlled fractional target. This
// target LP is a mechanism fixture, not a production global-bound certificate.
#include "mlnd/queue_fractional.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.name="joint_fractional_integration";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.single_rates=d.head_rates=d.demand;d.pair_rates=filled(2,2,0.0);
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);
    for(int u=0;u<4;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.01);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.1);
    d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}
Json exercise() {
    auto d=fixture();auto paths=enumerate_paths(d);const DesignLayout layout(d);
    for(int r=0;r<2;++r) {
        double minimum=std::numeric_limits<double>::infinity();
        for(const auto& p:paths.service_paths)if(p.first_od==r&&p.second_od==-1)
            minimum=std::min(minimum,p.cost+d.fleet_cost*p.time);
        d.ground_cost[r]=4+minimum;
    }
    SolverOptions options;options.threads=1;options.seed=43;options.pareto=false;
    QueuePolymatroidCuts pool(d,paths,options);QueueNativePrices donor;donor.valid=true;
    donor.status="complete_signed_fixture";donor.prices.source="joint_fractional_fixture";
    donor.prices.locker.assign(2,0);donor.prices.airline.assign(16,0);donor.prices.balance.assign(2,0);
    donor.prices.branch_gates={{1,1,-1,6,8},{1,1,-1,7,8}};
    require(pool.add_profile(donor.prices,5),"Signed profile preparation failed");
    std::vector<double> target(layout.count,1);
    for(int u=0;u<4;++u)for(int v=0;v<4;++v)target[layout.airline+4*u+v]=d.allowed_links(u,v);
    // Close priced base-1 outgoing arcs so the signed profile is stronger than
    // the unpriced profile at this target; this is not a feasible integer design.
    target[layout.airline+6]=target[layout.airline+7]=0;
    for(int r=0;r<2;++r)for(int l=0;l<2;++l)target[layout.assignment_index(r,l)]=.5;
    LinearModel master;master.constant=2;
    for(int j=0;j<layout.count;++j) {
        master.variable("target_"+std::to_string(j),0,1,0,true);
        master.row("target_value_"+std::to_string(j),'E',target[j],{{j,1}});
    }
    const int theta=master.variable("theta",-IloInfinity,IloInfinity,1,false);
    const auto old=pool.separate_profile(1,target,0,theta);
    require(old.valid,"Old matched-profile queue row is invalid");master.rows.push_back(old.row);
    const std::set<std::string> old_keys{old.key};
    const auto off=separate_fractional_queue(master,layout,theta,pool,donor,old_keys,options,2,1,1,3);
    require(off.cuts.empty()&&std::isfinite(off.certified_lower),"Default-off behavior changed");
    options.queue_joint_cuts=true;
    const auto on=separate_fractional_queue(master,layout,theta,pool,donor,old_keys,options,2,1,1,3);
    require(on.cuts.size()==1&&on.cuts.front().certificate.at("kind")=="queue_joint",
            "Joint row did not enter the actual fractional master LP");
    const auto& rounds=on.diagnostics.at("rounds");
    require(rounds.size()>=2&&rounds.front().at("accepted").get<bool>(),"Joint row was not followed by LP reoptimization");
    const double gain=on.certified_lower-off.certified_lower;
    require(std::abs(gain-.5)<2e-5,"Reoptimized LP did not recover the analytical 0.5 gain");
    require(on.price_records.size()>=1&&on.price_records.front().at("joint").at("valid").get<bool>(),
            "Joint price ledger lost its mechanism record");
    require(on.wall+1e-7>=on.lp_seconds+on.price_seconds,"Fractional LP/price ledgers overlap");
    master.rows.push_back(on.cuts.front().row);auto keys=old_keys;keys.insert(on.cuts.front().key);
    const auto repeated=separate_fractional_queue(master,layout,theta,pool,donor,keys,options,2,1,1,3);
    require(repeated.cuts.empty()&&std::abs(repeated.certified_lower-on.certified_lower)<2e-5,
            "Joint row was duplicated or the reoptimized bound changed on replay");
    return {{"passed",true},{"off_certified_LP",off.certified_lower},{"on_certified_LP",on.certified_lower},
        {"certified_LP_gain",gain},{"accepted_kind",on.cuts.front().certificate.at("kind")},
        {"accepted_rows",on.cuts.size()},{"repeated_rows",repeated.cuts.size()},
        {"rounds",rounds},{"price_records",on.price_records},
        {"scope","controlled_fractional_target_LP_not_production_global_bound"}};
}
}
int main() {try {std::cout<<exercise().dump(2)<<'\n';return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
