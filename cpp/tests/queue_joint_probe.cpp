// Analytical public-separator witnesses, not production performance results.
#include "mlnd/queue_polymatroid.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows,int columns,T value) {
    Matrix<T> out(rows,columns);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture(int heads=2) {
    Input d;d.n_l=2;d.n_h=2;d.n_od=heads;d.n_levels=1;d.max_orders=2;
    d.name="signed_joint_queue_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand.assign(heads,1);d.ground_cost.assign(heads,20);d.deadline.assign(heads,1);
    const int n=d.n_nodes();d.flight_times=filled(n,n,.05);
    d.allowed_links=filled(n,n,1);
    for(int u=0;u<n;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(n,n,.01);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(heads,2,.08);d.first_cost=filled(heads,2,1.0);
    d.terminal_cost=filled(heads,2,.5);d.receipt_mean=filled(heads,2,.08);
    d.removal_mean=filled(heads,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(heads,heads,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;d.preprocessed_demand=true;
    d.pair_rates=filled(heads,heads,0.0);d.single_rates=d.demand;d.head_rates=d.demand;
    d.common_origin_support=filled(heads,2,1);return d;
}
void calibrate(Input& d,const PathCollection& paths) {
    for(int r=0;r<d.n_od;++r) {
        double minimum=std::numeric_limits<double>::infinity();
        for(const auto& p:paths.service_paths)if(p.first_od==r)
            minimum=std::min(minimum,p.cost+d.fleet_cost*p.time);
        if(std::isfinite(minimum))d.ground_cost[r]=4+minimum;
    }
}
QueueResourcePrices prices(const Input& d) {
    QueueResourcePrices out;out.source="analytical_signed_joint_witness";
    out.locker.assign(d.n_h,0);out.airline.assign(d.n_nodes()*d.n_nodes(),0);
    out.balance.assign(d.n_l,0);
    // Every path of head 1 from base 1 uses exactly one of these first arcs.
    for(int h=0;h<d.n_h;++h)out.branch_gates.push_back({1,1,-1,d.n_nodes()+d.n_l+h,8});
    return out;
}
std::vector<double> open_design(const Input& d) {
    DesignLayout layout(d);std::vector<double> x(layout.count,0);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h*d.n_levels]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
double value(const TaskSavingsCut& cut,const std::vector<double>& x,int theta) {
    require(cut.valid&&cut.row.sense=='G',"No valid joint epigraph row");
    long double out=cut.row.rhs;double epigraph=0;
    for(const auto& [j,a]:cut.row.coefficients)if(j==theta)epigraph+=a;else {
        require(j>=0&&j<static_cast<int>(x.size())&&std::isfinite(a),"Invalid design coefficient");out-=a*x[j];
    }
    require(epigraph==1,"Incorrect epigraph coefficient");return static_cast<double>(out);
}
Json analytical_witness() {
    auto d=fixture();auto paths=enumerate_paths(d);calibrate(d,paths);
    SolverOptions options;QueuePolymatroidCuts separator(d,paths,options);
    require(separator.add_profile(prices(d),5),"Certified signed profile was not created");
    DesignLayout layout(d);auto x=open_design(d);
    for(int r=0;r<2;++r)for(int l=0;l<2;++l)x[layout.assignment_index(r,l)]=.5;
    const auto old=separator.separate_profile(1,x,0,layout.count);
    const auto joint=separator.separate_joint(x,0,layout.count,.25);
    const double improvement=value(joint,x,layout.count)-value(old,x,layout.count);
    require(std::abs(improvement-.5)<2e-5,"Signed (4,4),(4,-4) witness must improve by 0.5");
    const auto& p=joint.certificate.at("profile");
    long double affine=p.at("constant").get<double>();
    const auto coefficients=p.at("lower_coefficients").get<std::vector<double>>();
    for(int j=0;j<layout.count;++j)affine+=coefficients[j]*x[j];
    const double expected=static_cast<double>(affine)-(9-4*std::sqrt(2.0))/2;
    require(std::abs(value(joint,x,layout.count)-expected)<2e-5,"Joint value disagrees with worked nine-state envelope");
    return {{"passed",true},{"joint_value",value(joint,x,layout.count)},
        {"separate_base_value",value(old,x,layout.count)},{"strict_gain",improvement}};
}
// Evaluate the original nonlinear certified-profile surrogate on an integer
// design. This oracle has no facets, dual variables, or lifting coefficients.
double integer_profile_value(const Input& d,const Json& profile,const std::vector<double>& x) {
    DesignLayout layout(d);const auto b=profile.at("signed_savings").get<std::vector<double>>();
    const auto unsupported=profile.at("unsupported").get<std::vector<bool>>();
    const auto affine=profile.at("lower_coefficients").get<std::vector<double>>();
    long double out=profile.at("constant").get<double>();
    for(int j=0;j<layout.count;++j)out+=affine[j]*x[j];
    for(int l=0;l<2;++l) {
        long double sum=0;bool blocked=false;
        for(int r=0;r<d.n_od;++r)if(x[layout.assignment_index(r,l)]==1) {
            sum+=b[2*r+l];blocked=blocked||unsupported[2*r+l];
        }
        if(!blocked&&sum>d.fleet_cost) {
            const long double difference=std::sqrt(sum)-std::sqrt(static_cast<long double>(d.fleet_cost));
            out-=difference*difference;
        }
    }
    return static_cast<double>(out);
}
Json exhaustive_lifting(bool missing) {
    auto d=fixture(4);if(missing)d.deadline[2]=.001;
    auto paths=enumerate_paths(d);calibrate(d,paths);SolverOptions options;
    QueuePolymatroidCuts separator(d,paths,options);auto resource=prices(d);
    if(!missing)for(int h=0;h<d.n_h;++h)
        resource.branch_gates.push_back({2,0,-1,d.n_l+h,8});
    require(separator.add_profile(resource,5),"Lifted fixture did not produce a complete profile");
    DesignLayout layout(d);int checks=0;double maximum=-1e100;
    bool positive_external=false,negative_external=false,unsupported_external=false;
    for(int sample=0;sample<4;++sample) {
        auto point=open_design(d);
        for(int r=0;r<d.n_od;++r)for(int l=0;l<2;++l)
            point[layout.assignment_index(r,l)]=sample==0?(r<2?.5:0):
                sample==1?1.0/3:sample==2?(r==2?.5:0):(l==r%2?.8:.1);
        const auto cut=separator.separate_joint(point,0,layout.count,.5);
        require(cut.valid,"Joint lifting did not return a valid row");
        const auto& profile=cut.certificate.at("profile");
        const auto selected=cut.certificate.at("selected_heads").get<std::vector<int>>();
        const auto b=profile.at("signed_savings").get<std::vector<double>>();
        const auto unsupported=profile.at("unsupported").get<std::vector<bool>>();
        for(int r=0;r<d.n_od;++r)if(std::find(selected.begin(),selected.end(),r)==selected.end())
            for(int l=0;l<2;++l) {
                positive_external=positive_external||(!unsupported[2*r+l]&&b[2*r+l]>0);
                negative_external=negative_external||(!unsupported[2*r+l]&&b[2*r+l]<0);
                unsupported_external=unsupported_external||unsupported[2*r+l];
            }
        for(int code=0;code<81;++code) {
            auto integer=open_design(d);int remaining=code;
            for(int r=0;r<d.n_od;++r) {
                const int state=remaining%3;remaining/=3;
                if(state)integer[layout.assignment_index(r,state-1)]=1;
            }
            // The resource affine term is retained exactly even on a design
            // with closed assets; the profile bound may then be conservative.
            for(int asset=0;asset<2;++asset) {
                if(asset)for(int j=layout.airline;j<layout.count;++j)integer[j]=0;
                const double excess=value(cut,integer,layout.count)-integer_profile_value(d,profile,integer);
                maximum=std::max(maximum,excess);++checks;
                require(excess<=1e-8,"Joint cut excludes an integer full-domain surrogate state");
            }
        }
    }
    require(positive_external,"Positive unselected head was not exercised");
    require(missing?unsupported_external:negative_external,"Missing or negative unselected head was not exercised");
    return {{"passed",true},{"integer_design_checks",checks},{"maximum_excess",maximum},
        {"positive_external",positive_external},{"negative_external",negative_external},
        {"unsupported_external",unsupported_external}};
}
Json guards() {
    auto d=fixture();auto paths=enumerate_paths(d);calibrate(d,paths);SolverOptions options;
    QueuePolymatroidCuts separator(d,paths,options);DesignLayout layout(d);auto x=open_design(d);
    const auto count=separator.profile_count();
    require(!separator.separate_joint(x,0,layout.count,0).valid,"Zero budget emitted a row");
    require(separator.profile_count()==count,"Joint separation modified the certified profile pool");
    auto nonnegative=prices(d);nonnegative.branch_gates.clear();nonnegative.source="all_nonnegative";
    require(separator.add_profile(nonnegative,5),"Nonnegative resource fixture failed");
    require(!separator.separate_joint(x,0,layout.count,.1).valid,"Compatible nonnegative profile should be skipped");
    require(separator.add_profile(prices(d),5),"Signed fixture failed");
    const double scan_seconds=separator.metadata().at("profile_scan_seconds");
    require(separator.separate_joint(x,0,layout.count,.1).valid,"Signed retained profile was skipped");
    require(separator.metadata().at("profile_scan_seconds").get<double>()==scan_seconds,
            "Joint separation rescanned the physical catalogue");
    bool rejected=false;x[layout.assignment_index(0,0)]=1;x[layout.assignment_index(0,1)]=1;
    try {separator.separate_joint(x,0,layout.count,.1);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Non-exclusive head point was accepted");
    x[layout.assignment_index(0,1)]=0;paths.complete=false;
    require(!separator.separate_joint(x,0,layout.count,.1).valid,"Incomplete catalogue remained eligible");
    auto k1=fixture();k1.max_orders=1;auto k1paths=enumerate_paths(k1);
    QueuePolymatroidCuts k1separator(k1,k1paths,options);
    require(!k1separator.separate_joint(open_design(k1),0,DesignLayout(k1).count,.1).valid,"K1 scope was silently changed");
    return {{"passed",true},{"zero_budget",true},{"signed_applicability",true},
        {"no_new_path_scan",true},{"invalid_assignment_rejected",true},{"scope_guard",true}};
}
}
int main() {
    try {std::cout<<Json({{"analytical_witness",analytical_witness()},
        {"signed_external_lifting",exhaustive_lifting(false)},
        {"unsupported_external_lifting",exhaustive_lifting(true)},
        {"guards",guards()},{"passed",true}}).dump(2)<<'\n';return 0;}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
