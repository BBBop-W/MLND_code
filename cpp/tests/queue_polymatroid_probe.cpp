// Two-base public-interface witnesses for integer-valid queue EPI cuts.
// These small fixtures are validity oracles, not production benchmarks.
#include "mlnd/queue_polymatroid.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool ok,const std::string& message) {
    if(!ok)throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);
    std::fill(out.values.begin(),out.values.end(),value);
    return out;
}
Input physical_fixture(int cardinality=1) {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=cardinality;
    d.name="queue_polymatroid_l2_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand={1,1};d.ground_cost={20,20};d.deadline={1,1};
    const int n=d.n_nodes();d.flight_times=filled(n,n,.05);
    for(int u=0;u<n;++u)d.flight_times(u,u)=0;
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.allowed_links=filled(n,n,1);
    for(int u=0;u<n;++u)d.allowed_links(u,u)=0;
    d.link_cost=filled(n,n,.01);d.base_cost={.5,.5};
    d.slot_epsilon={.05,.05};d.rider_mean=filled(2,2,.08);
    d.first_cost=filled(2,2,1.0);d.terminal_cost=filled(2,2,.5);
    d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.1);
    d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.preprocessed_demand=true;d.pair_rates=filled(2,2,0.0);
    if(cardinality==1) {d.single_rates=d.demand;d.head_rates=d.demand;}
    else {
        // Exact raw-order conservation, including AA's two occurrences and
        // the second OD served under a different head's assignment.
        d.demand={4,7};d.single_rates={.5,4.5};d.pair_rates(0,0)=.5;
        d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;d.head_rates={3,5};
    }
    d.common_origin_support=filled(2,2,1);
    return d;
}
std::vector<double> open_design(const Input& d,int first,int second,bool airlines=true) {
    const DesignLayout layout(d);std::vector<double> x(layout.count,0);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h*d.n_levels]=1;
    if(first>=0)x[layout.assignment_index(0,first)]=1;
    if(second>=0)x[layout.assignment_index(1,second)]=1;
    if(airlines)for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
double cut_value(const TaskSavingsCut& cut,const std::vector<double>& x,int theta) {
    require(cut.valid&&cut.row.sense=='G',"No valid queue lower-bound row returned");
    long double value=cut.row.rhs;double theta_coefficient=0;
    for(const auto& [index,coefficient]:cut.row.coefficients) {
        if(index==theta)theta_coefficient+=coefficient;
        else {
            require(index>=0&&index<static_cast<int>(x.size()),"Queue cut introduced a new design variable");
            value-=static_cast<long double>(coefficient)*x[index];
        }
    }
    require(std::abs(theta_coefficient-1)<1e-12,"Queue cut has an invalid epigraph coefficient");
    return static_cast<double>(value);
}
Json hand_strength_tracer() {
    auto d=physical_fixture();auto paths=enumerate_paths(d);
    // The physical route catalogue is unchanged by a courier price. Calibrate
    // each unit-rate head to a saving of exactly four before queue costs.
    for(int r=0;r<2;++r) {
        double minimum=std::numeric_limits<double>::infinity();
        for(const auto& p:paths.service_paths)if(p.first_od==r&&p.second_od==-1)
            minimum=std::min(minimum,p.cost+d.fleet_cost*p.time);
        require(std::isfinite(minimum),"Hand witness has no physical single-order path");
        d.ground_cost[r]=4+minimum;
    }
    const auto baseline=task_savings_data(d,paths);
    for(double saving:baseline.saving)
        require(std::abs(saving-4)<1e-8,"Physical hand witness lost its four-unit saving");
    SolverOptions options;options.threads=1;
    QueuePolymatroidCuts separator(d,paths,options,&baseline,8);
    const DesignLayout layout(d);const auto design=open_design(d,0,-1);
    const auto cut=separator.separate_profile(0,design,baseline.courier_cost-10,layout.count);
    const double bound=cut_value(cut,design,layout.count);
    const double expected=baseline.courier_cost-1;
    const double old_chord=baseline.courier_cost-(9-4*std::sqrt(2.0))/2;
    require(std::abs(bound-expected)<1e-5,"F=1, S=(4,4) EPI must give courier cost minus one");
    require(bound-old_chord>.6715,"Queue EPI failed the hand-proved strict improvement over the chord");
    return {{"bases",2},{"fleet_cost",1},{"saving_each",4},
        {"epi_bound",bound},{"hand_expected",expected},{"old_chord_bound",old_chord},
        {"strict_improvement",bound-old_chord},{"passed",true}};
}

struct ReferencePoint {
    std::vector<double> design;
    double operating_cost=0;
};
SolverOptions oracle_options() {
    SolverOptions options;options.seconds=20;options.threads=1;options.seed=43;
    options.gap=1e-9;options.tight_airline_bounds=true;options.pareto=false;
    return options;
}
ReferencePoint fixed_recourse(const Input& d,const PathCollection& paths,
                              const SolverOptions& options,const std::vector<double>& design,
                              bool* incoming_second=nullptr) {
    const auto full=build_formulation(d,paths,&design,nullptr,nullptr,false,false,
                                      options.tight_airline_bounds);
    NativeSolver oracle(full.matrix,options,"queue_epi_full_fixed_oracle");
    const auto solved=oracle.optimize(20);
    require(solved.optimal&&solved.has_solution,"Independent complete recourse failed: "+solved.status);
    require(point_residuals(full.matrix,solved.values).at("passed"),
            "Independent complete recourse violates original model constraints");
    if(incoming_second&&d.max_orders==2) {
        const bool unassigned=design[full.layout.assignment_index(1,0)]==0&&
                              design[full.layout.assignment_index(1,1)]==0;
        if(unassigned&&solved.values[full.courier[1]]<d.demand[1]-1e-5)*incoming_second=true;
    }
    return {design,objective_value(full.matrix,solved.values)};
}
QueueResourcePrices zero_prices(const Input& d,const std::string& source) {
    QueueResourcePrices prices;prices.source=source;
    prices.locker.assign(d.n_h,0);prices.airline.assign(d.n_nodes()*d.n_nodes(),0);
    prices.balance.assign(d.n_l,0);return prices;
}
std::size_t add_profile(QueuePolymatroidCuts& separator,const QueueResourcePrices& prices) {
    const auto index=separator.profile_count();
    require(separator.add_profile(prices,5),"Complete resource price profile was not prepared");
    require(separator.profile_count()==index+1,"Resource profile silently reused the static fallback");
    return index;
}
const Json& price_certificate(const TaskSavingsCut& cut) {
    require(cut.valid,"Resource profile returned no valid cut");
    require(cut.certificate.at("kind")=="queue_polymatroid"&&
            cut.certificate.at("profile").at("kind")=="resource_price_epi",
            "Resource test did not exercise the requested priced EPI profile");
    return cut.certificate.at("profile");
}
double check_full_integer_optimum(const Input& d,const PathCollection& paths,
                                 const SolverOptions& options,const std::vector<TaskSavingsCut>& cuts,
                                 int old_theta) {
    const auto original=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,
                                         options.tight_airline_bounds);
    NativeSolver oracle(original.matrix,options,"queue_epi_full_integer_oracle");
    const auto solved=oracle.optimize(20);
    require(solved.optimal&&solved.has_solution,"Independent full MISOCP did not establish its optimum");
    require(point_residuals(original.matrix,solved.values).at("passed"),
            "Independent full MISOCP returned an invalid point");
    // Attach the actual original operating objective to an epigraph. Checking
    // this independently optimal original solution against every new row is
    // a necessary preservation check without an implementation-mirrored oracle.
    auto strengthened=original;
    const int epigraph=strengthened.matrix.variable("queue_test_operating_cost",0,IloInfinity,0);
    Coefficients identity{{epigraph,1}};double operating=0;
    for(int j=original.layout.count;j<epigraph;++j)if(original.matrix.variables[j].cost) {
        identity.emplace_back(j,-original.matrix.variables[j].cost);
        operating+=original.matrix.variables[j].cost*solved.values[j];
    }
    strengthened.matrix.row("queue_test_cost_identity",'E',0,std::move(identity));
    for(auto cut:cuts) {
        for(auto& [index,coefficient]:cut.row.coefficients)if(index==old_theta)index=epigraph;
        strengthened.matrix.row(cut.row.name,cut.row.sense,cut.row.rhs,std::move(cut.row.coefficients));
    }
    auto witness=solved.values;witness.push_back(operating);
    require(point_residuals(strengthened.matrix,witness).at("passed"),
            "Queue EPI excludes the independently optimal original integer solution");
    return solved.objective;
}
Json independent_validity(int cardinality) {
    const auto d=physical_fixture(cardinality);const auto paths=enumerate_paths(d);
    const auto options=oracle_options();const DesignLayout layout(d);
    QueuePolymatroidCuts separator(d,paths,options);
    const auto signed_zero=add_profile(separator,zero_prices(d,"oracle_zero_resource"));
    auto resources=zero_prices(d,"oracle_resource_and_negative_empty");
    resources.locker={.2,.5};resources.balance={-10,10};
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        if(d.allowed_links(u,v))resources.airline[u*d.n_nodes()+v]=.1;
    resources.branch_gates.push_back({0,0,-1,2,.3});
    const auto priced=add_profile(separator,resources);
    const auto witness_cut=separator.separate_profile(priced,open_design(d,0,1),0,layout.count);
    const auto& certificate=price_certificate(witness_cut);
    require(!certificate.at("negative_empty_paths").empty(),"Balance prices did not trigger negative empty-flow correction");
    for(const auto& empty:certificate.at("negative_empty_paths"))
        require(empty.at("adjusted_cost").get<double>()<0&&empty.at("affine_coefficient").get<double>()<0,
                "Negative empty path lacks its finite design-dependent correction");
    // On the direct 0->1 empty route: c=.01, F*t=.05, arc price=.1,
    // potential difference=-20, so d=-19.84. Its finite flow correction
    // plus the separate -.1*D airline term must give -19.94*D in K.
    const double demand=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    require(std::abs(certificate.at("lower_coefficients").at(layout.airline+1).get<double>()+
                     19.94*demand)<1e-6*(1+demand),
            "Negative empty-flow certificate omitted or double-counted its affine correction");
    require(certificate.at("service_paths_scanned")==paths.service_paths.size()&&
            certificate.at("empty_paths_scanned")==paths.empty_paths.size(),
            "Priced certificate did not inspect the complete physical catalogue");

    std::vector<ReferencePoint> points;bool incoming_second=false;
    for(int first=-1;first<2;++first)for(int second=-1;second<2;++second)
        points.push_back(fixed_recourse(d,paths,options,open_design(d,first,second),&incoming_second));
    points.push_back(fixed_recourse(d,paths,options,open_design(d,0,0,false),&incoming_second));
    points.push_back(fixed_recourse(d,paths,options,std::vector<double>(layout.count,0),&incoming_second));
    const auto integer_count=points.size();
    for(std::size_t j=1;j<integer_count;++j) {
        auto mixed=points[j];
        for(std::size_t i=0;i<mixed.design.size();++i)
            mixed.design[i]=.37*points[j].design[i]+.63*points[0].design[i];
        mixed.operating_cost=.37*points[j].operating_cost+.63*points[0].operating_cost;
        points.push_back(std::move(mixed));
    }
    std::vector<TaskSavingsCut> cuts;
    for(std::size_t profile:{std::size_t{0},signed_zero,priced})for(const auto& point:points)
        cuts.push_back(separator.separate_profile(profile,point.design,0,layout.count));
    double maximum_violation=0;
    for(const auto& cut:cuts)for(const auto& point:points) {
        const double violation=cut_value(cut,point.design,layout.count)-point.operating_cost;
        maximum_violation=std::max(maximum_violation,violation);
        require(violation<=1e-6*(1+std::abs(point.operating_cost)),
                "Queue cut excludes an original integer recourse point or its convex mixture");
    }
    if(cardinality==2)require(incoming_second,"K2 oracle never served the second OD without its own assignment");
    const double optimum=check_full_integer_optimum(d,paths,options,cuts,layout.count);
    return {{"bases",2},{"K",cardinality},{"integer_full_recourse_solves",integer_count},
        {"convex_mixture_points",points.size()-integer_count},{"cuts",cuts.size()},
        {"cross_checks",cuts.size()*points.size()},{"maximum_violation",maximum_violation},
        {"full_integer_optimum",optimum},{"incoming_second_without_own_assignment",incoming_second},
        {"negative_empty_paths",certificate.at("negative_empty_paths")},
        {"fractional_scope","convex mixtures of original integer epigraph points; not the original continuous root"},
        {"passed",true}};
}

Json signed_savings_witness() {
    const auto d=physical_fixture(2);const auto paths=enumerate_paths(d);
    const auto options=oracle_options();const DesignLayout layout(d);
    QueuePolymatroidCuts separator(d,paths,options);
    auto prices=zero_prices(d,"signed_head_gates");
    for(int second=-1;second<2;++second) {
        const double rate=second<0?d.single_rates[0]:d.pair_rates(0,second);
        if(rate<=0)continue;
        for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
            if(d.allowed_links(u,v))prices.branch_gates.push_back({0,0,second,u*d.n_nodes()+v,100});
    }
    const auto profile=add_profile(separator,prices);
    const auto x=open_design(d,0,0,false);
    const auto cut=separator.separate_profile(profile,x,0,layout.count);
    const auto& certificate=price_certificate(cut);
    const auto savings=certificate.at("signed_savings").get<std::vector<double>>();
    require(savings.at(0)<0&&savings.at(2)>0&&savings.at(0)+savings.at(2)<0,
            "Signed witness did not create opposing savings at the same base");
    const double courier=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    const auto reference=fixed_recourse(d,paths,options,x);
    const double bound=cut_value(cut,x,layout.count);
    // With every physical link closed, K=C0. The two assigned groups have
    // negative total signed saving, so the exact integer queue minimum is C0.
    // Clamping the negative B to zero would lose a large amount at this point.
    require(std::abs(reference.operating_cost-courier)<1e-4,"Closed-link witness unexpectedly serves a drone task");
    require(std::abs(bound-courier)<1e-3,"Signed EPI lost the negative group's shared-rho coupling");
    require(bound<=reference.operating_cost+1e-5,"Signed EPI excludes original fixed recourse");
    return {{"bases",2},{"B_head0_base0",savings.at(0)},{"B_head1_base0",savings.at(2)},
        {"bound",bound},{"courier_cost",courier},{"reference_recourse",reference.operating_cost},
        {"profile_kind",certificate.at("kind")},{"passed",true}};
}

Json unsupported_and_witness() {
    auto d=physical_fixture(1);d.deadline[0]=1e-5;
    const auto paths=enumerate_paths(d);const auto options=oracle_options();const DesignLayout layout(d);
    QueuePolymatroidCuts separator(d,paths,options);
    const auto profile=add_profile(separator,zero_prices(d,"unsupported_AND_zero_prices"));
    const auto x=open_design(d,0,0);
    const auto coupled=separator.separate_profile(profile,x,0,layout.count);
    const auto& certificate=price_certificate(coupled);
    const auto unsupported=certificate.at("unsupported").get<std::vector<bool>>();
    require(unsupported.at(0)&&!unsupported.at(2),"AND witness needs one missing and one supported positive head");
    require(coupled.certificate.at("bases").at(0).at("necessary_gates").at(0).get<bool>(),
            "Missing positive branch did not enter the shared-rho AND gate");
    const auto reference=fixed_recourse(d,paths,options,x);
    const double courier=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    const double bound=cut_value(coupled,x,layout.count);
    const double static_bound=cut_value(separator.separate_profile(0,x,0,layout.count),x,layout.count);
    require(std::abs(reference.operating_cost-courier)<1e-4,"Missing positive branch failed to force the common availability to zero");
    require(std::abs(bound-courier)<1e-4,"Unsupported AND EPI does not retain the shared-rho consequence");
    require(bound-static_bound>1,"Unsupported AND case did not strictly strengthen the independent S+ profile");
    require(bound<=reference.operating_cost+1e-5,"Unsupported AND EPI excludes original fixed recourse");
    return {{"bases",2},{"bound",bound},{"static_bound",static_bound},
        {"reference_recourse",reference.operating_cost},{"unsupported_groups",unsupported},
        {"passed",true}};
}

Json complete_domain_guards() {
    const auto d=physical_fixture(2);const auto paths=enumerate_paths(d);const auto options=oracle_options();
    int rejected=0;
    for(int flag:{0,1}) {
        auto partial=paths;if(flag==0)partial.complete=false;else partial.support_complete=false;
        bool failed=false;
        try{QueuePolymatroidCuts invalid(d,partial,options);}catch(const std::invalid_argument&){failed=true;}
        require(failed,"Queue certificate accepted an incomplete path or task-support domain");++rejected;
    }
    QueuePolymatroidCuts separator(d,paths,options);
    const auto count=separator.profile_count();
    require(!separator.add_profile(zero_prices(d,"no_budget"),0)&&separator.profile_count()==count,
            "Zero-budget profile scan fabricated a complete-domain certificate");
    return {{"bases",2},{"incomplete_domain_rejections",rejected},
        {"zero_budget_no_profile",true},{"passed",true}};
}
}
int main() {
    try {
        const double started=wall_time();
        Json cases=Json::array();cases.push_back(independent_validity(1));cases.push_back(independent_validity(2));
        std::cout<<Json{{"hand_strength",hand_strength_tracer()},{"independent_cases",cases},
            {"signed_savings",signed_savings_witness()},{"unsupported_AND",unsupported_and_witness()},
            {"guards",complete_domain_guards()},{"seconds",wall_time()-started},{"passed",true}}.dump(2)<<'\n';
        return 0;
    } catch(const IloException& e) {std::cerr<<e<<'\n';return 1;}
      catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
