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
Input fixture(int heads=2,int bases=2) {
    Input d;d.n_l=bases;d.n_h=2;d.n_od=heads;d.n_levels=1;d.max_orders=2;
    d.name="signed_joint_queue_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand.assign(heads,1);d.ground_cost.assign(heads,20);d.deadline.assign(heads,1);
    const int n=d.n_nodes();d.flight_times=filled(n,n,.05);
    d.allowed_links=filled(n,n,1);
    for(int u=0;u<n;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.link_cost=filled(n,n,.01);d.base_cost.assign(bases,.5);d.slot_epsilon={.05,.05};
    d.rider_mean=filled(heads,2,.08);d.first_cost=filled(heads,2,1.0);
    d.terminal_cost=filled(heads,2,.5);d.receipt_mean=filled(heads,2,.08);
    d.removal_mean=filled(heads,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(heads,heads,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;d.preprocessed_demand=true;
    d.pair_rates=filled(heads,heads,0.0);d.single_rates=d.demand;d.head_rates=d.demand;
    d.common_origin_support=filled(heads,bases,1);return d;
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
    for(int l=0;l<d.n_l;++l) {
        long double sum=0;bool blocked=false;
        for(int r=0;r<d.task_count();++r)if(x[layout.assignment_index(r,l)]==1) {
            sum+=b[d.n_l*r+l];blocked=blocked||unsupported[d.n_l*r+l];
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
Json general_l_witness(int bases) {
    auto d=fixture(2,bases);auto paths=enumerate_paths(d);calibrate(d,paths);
    SolverOptions options;QueuePolymatroidCuts separator(d,paths,options);
    auto resource=prices(d);
    // Two negative entries have even complement parity. The extra negative
    // entry lies on an unused base, so the old L2 witness remains a face.
    // This specifically catches incorrectly applying L2 parity to general L.
    for(int h=0;h<d.n_h;++h)resource.branch_gates.push_back(
        {1,bases-1,-1,(bases-1)*d.n_nodes()+d.n_l+h,8});
    require(separator.add_profile(resource,5),"General-L signed profile was not created");
    DesignLayout layout(d);auto x=open_design(d);
    for(int r=0;r<2;++r)for(int l=0;l<2;++l)x[layout.assignment_index(r,l)]=.5;
    const auto old=separator.separate_profile(1,x,0,layout.count);
    const auto joint=separator.separate_joint(x,0,layout.count,2);
    const double improvement=value(joint,x,layout.count)-value(old,x,layout.count);
    require(std::abs(improvement-.5)<2e-5,"General-L even-parity witness lost its strict 0.5 gain");
    const auto& cert=joint.certificate;
    require(cert.at("vertices").size()==static_cast<std::size_t>((bases+1)*(bases+1)),"Incomplete state certificate");
    require(cert.at("affine_coefficients").size()==static_cast<std::size_t>(1+2*bases),"Incorrect coefficient dimensions");
    require(cert.at("coefficient_method")=="complete_state_coefficient_lp", "General-L LP method not recorded");
    require(cert.at("auxiliary_optimization_calls")==1&&cert.at("all_states_validated")==true,
        "LP calls or full-state validation not recorded");
    int checks=0;
    for(int first=0;first<=bases;++first)for(int second=0;second<=bases;++second) {
        auto integer=open_design(d);
        if(first)integer[layout.assignment_index(0,first-1)]=1;
        if(second)integer[layout.assignment_index(1,second-1)]=1;
        require(value(joint,integer,layout.count)<=integer_profile_value(d,cert.at("profile"),integer)+1e-8,
            "General-L witness excludes an integer state");++checks;
    }
    return {{"passed",true},{"bases",bases},{"strict_gain",improvement},{"integer_state_checks",checks},
        {"even_complement_parity",true},{"cut_certificate",cert}};
}
Json general_l_lifting(int bases,bool missing) {
    auto d=fixture(4,bases);if(missing)d.deadline[2]=.001;
    auto paths=enumerate_paths(d);calibrate(d,paths);SolverOptions options;
    QueuePolymatroidCuts separator(d,paths,options);auto resource=prices(d);
    if(!missing)for(int h=0;h<d.n_h;++h)resource.branch_gates.push_back({2,0,-1,d.n_l+h,8});
    require(separator.add_profile(resource,5),"General-L lifting profile unavailable");
    DesignLayout layout(d);int checks=0;double maximum=-1e100;
    bool positive_external=false,negative_external=false,unsupported_external=false;
    int combinations=1;for(int r=0;r<d.n_od;++r)combinations*=bases+1;
    for(int sample=0;sample<4;++sample) {
        auto point=open_design(d);
        for(int r=0;r<d.n_od;++r)for(int l=0;l<bases;++l)
            point[layout.assignment_index(r,l)]=sample==0?(r<2&&l<2?.5:0):
                sample==1?1.0/(bases+1):sample==2?(r==2&&l<2?.5:0):
                (l==r%bases?.8:.1/(bases-1));
        const auto cut=separator.separate_joint(point,0,layout.count,2);
        require(cut.valid,"General-L lifting did not return a validated row");
        const auto& profile=cut.certificate.at("profile");
        const auto selected=cut.certificate.at("selected_tasks").get<std::vector<int>>();
        const auto b=profile.at("signed_savings").get<std::vector<double>>();
        const auto unsupported=profile.at("unsupported").get<std::vector<bool>>();
        for(int r=0;r<d.n_od;++r)if(std::find(selected.begin(),selected.end(),r)==selected.end())
            for(int l=0;l<bases;++l) {
                const int group=r*bases+l;
                positive_external|=!unsupported[group]&&b[group]>0;
                negative_external|=!unsupported[group]&&b[group]<0;
                unsupported_external=unsupported_external||unsupported[group];
            }
        for(int code=0;code<combinations;++code) {
            auto integer=open_design(d);int remaining=code;
            for(int r=0;r<d.n_od;++r) {
                const int state=remaining%(bases+1);remaining/=bases+1;
                if(state)integer[layout.assignment_index(r,state-1)]=1;
            }
            for(int asset=0;asset<2;++asset) {
                if(asset)for(int j=layout.airline;j<layout.count;++j)integer[j]=0;
                const double excess=value(cut,integer,layout.count)-integer_profile_value(d,profile,integer);
                maximum=std::max(maximum,excess);++checks;
                require(excess<=1e-8,"General-L lifted cut excludes a nonlinear-surrogate integer design");
            }
        }
    }
    require(positive_external,"Positive external task was not tested");
    require(missing?unsupported_external:negative_external,"Missing/negative external task was not tested");
    return {{"passed",true},{"bases",bases},{"integer_design_checks",checks},{"maximum_excess",maximum},
        {"positive_external",positive_external},{"negative_external",negative_external},
        {"unsupported_external",unsupported_external}};
}
Json general_l_guards() {
    auto d=fixture(2,3);auto paths=enumerate_paths(d);calibrate(d,paths);SolverOptions options;
    QueuePolymatroidCuts separator(d,paths,options);auto resource=prices(d);resource.branch_gates.clear();
    require(separator.add_profile(resource,5),"Nonnegative general-L profile unavailable");
    auto x=open_design(d);DesignLayout layout(d);
    for(int r=0;r<2;++r)for(int l=0;l<3;++l)x[layout.assignment_index(r,l)]=.25;
    require(!separator.separate_joint(x,0,layout.count,.5).valid,"Nonnegative general-L pair should be skipped");
    require(separator.metadata().at("last_joint").at("auxiliary_optimization_calls")==0,
        "Skipped nonnegative profile solved a coefficient LP");
    require(separator.add_profile(prices(d),5),"Signed general-L guard profile unavailable");
    require(!separator.separate_joint(x,0,layout.count,1e-12).valid,"Expired budget emitted a partial row");
    x[layout.assignment_index(0,2)]=.75;bool rejected=false;
    try{separator.separate_joint(x,0,layout.count,.5);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"General-L row-sum exclusivity was not checked across all bases");
    return {{"passed",true},{"nonnegative_skip",true},{"expired_budget_rejected",true},{"all_base_exclusivity",true}};
}

Json bounded_candidate_selection(int bases) {
    auto d=fixture(4,bases);auto paths=enumerate_paths(d);calibrate(d,paths);SolverOptions options;
    QueuePolymatroidCuts separator(d,paths,options);
    auto resource=prices(d);
    require(separator.add_profile(resource,5),"First selection profile unavailable");
    for(int h=0;h<d.n_h;++h)resource.branch_gates.push_back({2,0,-1,d.n_l+h,8});
    require(separator.add_profile(resource,5),"Second selection profile unavailable");
    const DesignLayout layout(d);int checks=0;std::size_t most_candidates=0;
    for(bool tied:{false,true}) {
        auto x=open_design(d);
        for(int q=0;q<d.n_od;++q)for(int l=0;l<bases;++l)
            x[layout.assignment_index(q,l)]=tied?0.0:.125*(q%3)/bases;
        struct Candidate {std::size_t profile;int first,second;double score;};
        std::vector<Candidate> exhaustive;
        for(std::size_t index=1;index<separator.profile_count();++index) {
            const auto individual=separator.separate_profile(index,x,0,layout.count);
            require(individual.valid,"Selection reference profile was invalid");
            const auto& profile=individual.certificate.at("profile");
            const auto savings=profile.at("signed_savings").get<std::vector<double>>();
            const auto missing=profile.at("unsupported").get<std::vector<bool>>();
            for(int first=0;first<d.task_count();++first)for(int second=first+1;second<d.task_count();++second) {
                bool odd=false,unsupported=false,signed_profile=false;double score=0;
                for(int q:{first,second})for(int l=0;l<bases;++l) {
                    const int group=q*bases+l;
                    odd^=missing[group]||savings[group]<0;
                    unsupported|=missing[group];signed_profile|=savings[group]<0;
                    const double alpha=x[layout.assignment_index(q,l)];
                    score+=std::min(alpha,1-alpha);
                }
                if(unsupported||(bases==2?odd:signed_profile))exhaustive.push_back({index,first,second,score});
            }
        }
        // This is the previous selection algorithm, retained only as a small
        // differential oracle. In particular, score ties preserve traversal.
        std::stable_sort(exhaustive.begin(),exhaustive.end(),
            [](const Candidate& a,const Candidate& b){return a.score>b.score;});
        require(exhaustive.size()>8,"Selection fixture did not exercise truncation");
        most_candidates=std::max(most_candidates,exhaustive.size());
        Json expected=Json::array();
        for(std::size_t i=0;i<8;++i) {
            const auto& c=exhaustive[i];
            expected.push_back({{"profile",c.profile},{"first",c.first},{"second",c.second},{"score",c.score}});
        }
        separator.separate_joint(x,0,layout.count,2);
        const auto& diagnostics=separator.metadata().at("last_joint");
        require(diagnostics.at("candidate_selection_complete").get<bool>(),"Candidate scan was incomplete");
        require(diagnostics.at("eligible_pairs").get<std::size_t>()==exhaustive.size(),"Candidate eligibility changed");
        require(diagnostics.at("candidate_pairs_retained")==8,"Candidate storage exceeded the retained limit");
        require(diagnostics.at("selected_pairs")==expected,"Bounded selection changed stable top-eight order");
        ++checks;
    }
    separator.separate_joint(open_design(d),0,layout.count,0);
    const auto& expired=separator.metadata().at("last_joint");
    require(!expired.at("candidate_selection_complete").get<bool>(),"Expired selection claimed a complete ranking");
    require(expired.at("candidate_pairs_retained")==0,"Expired selection retained stale candidates");
    return {{"passed",true},{"bases",bases},{"order_checks",checks},{"eligible_pairs",most_candidates},
        {"retained_limit",8},{"expired_selection_complete",false}};
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
        {"guards",guards()},{"general_l3_witness",general_l_witness(3)},
        {"general_l4_witness",general_l_witness(4)},
        {"general_l3_signed_lifting",general_l_lifting(3,false)},
        {"general_l3_unsupported_lifting",general_l_lifting(3,true)},
        {"general_l4_signed_lifting",general_l_lifting(4,false)},
        {"general_l4_unsupported_lifting",general_l_lifting(4,true)},
        {"bounded_l2_selection",bounded_candidate_selection(2)},
        {"bounded_l3_selection",bounded_candidate_selection(3)},
        {"general_l_guards",general_l_guards()},{"passed",true}}).dump(2)<<'\n';return 0;}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
