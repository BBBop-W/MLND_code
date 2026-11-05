// Public seam tests for atomic, complete-domain queue-price search.
#include "mlnd/queue_polymatroid.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=1;
    d.name="queue_price_search_two_bases";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.single_rates=d.head_rates=d.demand;
    d.pair_rates=filled(2,2,0.0);d.common_origin_support=filled(2,2,1);
    d.ground_cost={20,20};d.deadline={1,1};d.flight_times=filled(4,4,.05);
    d.allowed_links=filled(4,4,1);
    for(int i=0;i<4;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.01);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}
std::vector<double> design(const Input& d,int first,int second,double capacity=1) {
    const DesignLayout l(d);std::vector<double> x(l.count,0);
    for(int j=0;j<d.n_nodes();++j)x[j]=1;
    for(int h=0;h<d.n_h;++h)x[l.capacity+h*d.n_levels]=capacity;
    if(first>=0)x[l.assignment_index(0,first)]=1;
    if(second>=0)x[l.assignment_index(1,second)]=1;
    for(int u=0;u<4;++u)for(int v=0;v<4;++v)x[l.airline+u*4+v]=d.allowed_links(u,v);
    return x;
}
QueueResourcePrices prices(double left,double right,const std::string& source) {
    QueueResourcePrices p;p.locker={left,right};p.airline.assign(16,0);p.balance.assign(2,0);
    p.source=source;return p;
}
double value(const TaskSavingsCut& cut,const std::vector<double>& x,int theta) {
    require(cut.valid,"No valid public queue row");long double result=cut.row.rhs;double coefficient=0;
    for(const auto& [j,c]:cut.row.coefficients)if(j==theta)coefficient+=c;else {
        require(j>=0&&j<static_cast<int>(x.size()),"New master variable appeared");result-=c*x[j];
    }
    require(std::abs(coefficient-1)<1e-12,"Wrong theta coefficient");return static_cast<double>(result);
}
Json hand_mix() {
    auto d=fixture();const auto paths=enumerate_paths(d);
    for(int r=0;r<2;++r) {
        double cheapest=std::numeric_limits<double>::infinity();
        for(const auto& p:paths.service_paths)if(p.first_od==r&&p.second_od==-1)
            cheapest=std::min(cheapest,p.cost+d.fleet_cost*p.time);
        d.ground_cost[r]=4+cheapest;
    }
    SolverOptions options;options.threads=1;const DesignLayout l(d);
    const auto x=design(d,0,0,.01);const auto a=prices(10,0,"hand_left"),b=prices(0,10,"hand_right");
    QueuePolymatroidCuts cut_pool(d,paths,options,nullptr,3),endpoint_b(d,paths,options);
    require(cut_pool.add_profile(a,5)&&endpoint_b.add_profile(b,5),"Endpoint profile was not complete");
    const double va=value(cut_pool.separate_profile(1,x,0,l.count),x,l.count);
    const double vb=value(endpoint_b.separate_profile(1,x,0,l.count),x,l.count);
    const double before=value(cut_pool.separate(x,0,l.count),x,l.count);
    const auto result=cut_pool.search_prices(b,x,0,l.count,5,7);
    const double best=value(result.cut,x,l.count);
    const double courier=d.ground_cost[0]+d.ground_cost[1];
    // Each path uses the two lockers. Its pickup occupancy q=.05+.08 exp(-.05/.08);
    // delivery occupancy q+.13. Equal prices of five give two-head saving
    // s=8-10(2q+.13), and K=C0-1.4 at the chosen fractional capacity.
    const double q=.05+.08*std::exp(-.05/.08),s=8-10*(2*q+.13);
    const double expected=courier-1.4-std::pow(std::sqrt(s)-1,2);
    require(std::abs(best-expected)<1e-5,"Interior price mixture missed the hand-derived value");
    require(best>std::max(va,vb)+.5&&best>before+.4,"Mixture did not strictly strengthen endpoints and old pool");
    require(result.diagnostics.at("complete_scans").get<int>()<=7,"Search exceeded seven complete scans");
    require(result.diagnostics.at("best_line_weight").get<double>()>0&&
            result.diagnostics.at("best_line_weight").get<double>()<1,"Winning price was not interior");
    QueuePolymatroidCuts saturated(d,paths,options,nullptr,2);
    require(saturated.add_profile(a,5),"No protected endpoint in saturated pool");
    const auto unretained=saturated.search_prices(b,x,0,l.count,5);
    require(std::abs(value(unretained.cut,x,l.count)-expected)<1e-5&&saturated.profile_count()==2&&
            saturated.separate_profile(1,x,0,l.count).certificate.at("profile").at("source")=="hand_left",
            "Price search evicted its strongest endpoint from a saturated pool");
    return {{"endpoint_a",va},{"endpoint_b",vb},{"before",before},{"best",best},
            {"hand_expected",expected},{"diagnostics",result.diagnostics},{"passed",true}};
}

Json native_mapping() {
    const auto d=fixture();const auto paths=enumerate_paths(d);
    auto f=build_formulation(d,paths);std::vector<double> dual(f.matrix.rows.size(),0);
    dual.at(f.capacity_rows[0])=-2;dual.at(f.capacity_rows[1])=-3;
    dual.at(f.link_rows[2])=-.25;
    dual.at(f.balance_rows[0])=7;dual.at(f.balance_rows[1])=2;
    const auto mapped=native_queue_resource_prices(d,f,dual);
    require(mapped.valid,"Finite native resource rows were rejected");
    require(mapped.prices.locker==std::vector<double>({2,3})&&mapped.prices.airline[2]==.25,
            "Native <= row prices have wrong sign or resource index");
    require(mapped.prices.balance==std::vector<double>({-5,0})&&mapped.prices.branch_gates.empty(),
            "Native equality potential or zero tau mapping is wrong");
    auto invalid=dual;invalid[f.capacity_rows[0]]=1;
    require(!native_queue_resource_prices(d,f,invalid).valid,"Wrong native resource sign was accepted");
    invalid=dual;invalid[f.link_rows[2]]=std::numeric_limits<double>::quiet_NaN();
    require(!native_queue_resource_prices(d,f,invalid).valid,"Nonfinite native price was accepted");
    invalid.pop_back();
    require(!native_queue_resource_prices(d,f,invalid).valid,"Wrong native row dimension was accepted");
    return {{"locker",mapped.prices.locker},{"balance",mapped.prices.balance},{"passed",true}};
}
Json atomic_guards() {
    const auto d=fixture();const auto paths=enumerate_paths(d);SolverOptions options;
    const DesignLayout l(d);const auto x=design(d,0,0,.01);
    QueuePolymatroidCuts pool(d,paths,options,nullptr,3);
    require(pool.add_profile(prices(10,0,"protected_endpoint"),5),"No guard endpoint");
    const auto before=pool.separate(x,0,l.count);const auto count=pool.profile_count();
    const auto zero=pool.search_prices(prices(0,10,"new_endpoint"),x,0,l.count,0);
    require(zero.cut.key==before.key&&pool.profile_count()==count&&
            zero.diagnostics.at("complete_scans")==0,"Zero budget mutated the pool or baseline row");
    auto bad=prices(0,10,"overflowing_balance");
    bad.balance={-std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};
    const auto failed=pool.search_prices(bad,x,0,l.count,5);
    require(failed.diagnostics.at("atomic_rollback")==true&&failed.diagnostics.at("scan_attempts")==1&&
            failed.diagnostics.at("complete_scans")==0,"Incomplete numeric scan was not rejected atomically");
    require(pool.profile_count()==count&&pool.separate(x,0,l.count).key==before.key,
            "Failed complete scan polluted retained profiles");
    const auto endpoint=pool.separate_profile(1,x,0,l.count);
    require(endpoint.certificate.at("profile").at("source")=="protected_endpoint",
            "Rejected search lost the cached endpoint");
    const auto capped=pool.search_prices(prices(0,10,"cap_endpoint"),x,0,l.count,5,99);
    require(capped.diagnostics.at("evaluation_count").get<int>()<=7&&
            capped.diagnostics.at("complete_scans").get<int>()<=7,"Public max-evaluation cap was bypassed");
    auto unsupported=d;unsupported.deadline[0]=1e-5;
    const auto missing_paths=enumerate_paths(unsupported);
    QueuePolymatroidCuts identical(unsupported,missing_paths,options);
    const auto and_result=identical.search_prices(prices(0,0,"same_zero_endpoint"),
        design(unsupported,0,0),0,l.count,5);
    require(and_result.diagnostics.at("status")=="identical_endpoints"&&
            and_result.diagnostics.at("profile_added")==true&&identical.profile_count()==2,
            "Improving identical zero-price AND endpoint was not committed: "+and_result.diagnostics.dump());
    const auto first_id=and_result.cut.certificate.at("profile").at("profile_id").get<std::size_t>();
    require(identical.add_profile(prices(1,0,"later_distinct_prices"),5),"No later distinct price profile");
    require(identical.separate_profile(2,design(unsupported,0,0),0,l.count).
            certificate.at("profile").at("profile_id").get<std::size_t>()>first_id,
            "Identical endpoint reused an emitted immutable profile identity");
    return {{"zero_budget",zero.diagnostics},{"failed_scan",failed.diagnostics},
            {"capped_evaluations",capped.diagnostics.at("evaluation_count")},
            {"identical_AND_endpoint",and_result.diagnostics},{"passed",true}};
}
Json complete_recourse_crosschecks() {
    const auto d=fixture();const auto paths=enumerate_paths(d);const DesignLayout l(d);
    SolverOptions options;options.threads=1;options.seed=43;options.seconds=20;options.gap=1e-9;
    options.tight_airline_bounds=true;options.pareto=false;
    struct Point {std::vector<double> x;double cost;};
    std::vector<Point> points;std::vector<TaskSavingsCut> cuts;int native_exports=0;
    QueuePolymatroidCuts pool(d,paths,options,nullptr,4);
    require(pool.add_profile(prices(10,0,"full_recourse_endpoint"),5),"No oracle endpoint");
    cuts.push_back(pool.search_prices(prices(0,10,"full_recourse_other_endpoint"),
        design(d,0,0,.01),0,l.count,5).cut);
    for(int first=-1;first<2;++first)for(int second=-1;second<2;++second) {
        auto x=design(d,first,second);
        auto f=build_formulation(d,paths,&x,nullptr,nullptr,false,false,true);
        NativeSolver solver(f.matrix,options,"queue_price_search_full_recourse_oracle");
        const auto solved=solver.optimize(20);
        require(solved.has_solution&&solved.optimal,"Independent complete recourse did not solve");
        require(point_residuals(f.matrix,solved.values).at("passed").get<bool>(),
                "Independent complete recourse failed physical matrix residuals");
        points.push_back({x,solved.objective});
        const auto dual=solver.read_dual(f.matrix);
        require(dual.available,"Existing native row multipliers were not available");
        const auto exported=native_queue_resource_prices(d,f,dual.row_prices);
        require(exported.valid,"Native complete-recourse prices failed export: "+exported.status);
        ++native_exports;
        const auto searched=pool.search_prices(exported.prices,x,solved.objective,l.count,5);
        require(searched.cut.valid&&searched.diagnostics.at("complete_scans").get<int>()<=7,
                "Native price search failed its public contract");
        cuts.push_back(searched.cut);
    }
    // Add a physically feasible zero-capacity design; then only convex mixtures
    // of original integer epigraph points are used as fractional oracles.
    std::vector<double> closed(l.count,0);
    auto f=build_formulation(d,paths,&closed,nullptr,nullptr,false,false,true);
    NativeSolver solver(f.matrix,options,"queue_price_search_zero_capacity_oracle");
    const auto solved=solver.optimize(20);
    require(solved.has_solution&&solved.optimal&&point_residuals(f.matrix,solved.values).at("passed").get<bool>(),
            "Zero-capacity original recourse was not certified");
    points.push_back({closed,solved.objective});
    const auto integer_count=points.size();
    for(std::size_t j=1;j<integer_count;++j) {
        auto mixed=points[j];
        for(std::size_t k=0;k<mixed.x.size();++k)mixed.x[k]=.37*points[j].x[k]+.63*points[0].x[k];
        mixed.cost=.37*points[j].cost+.63*points[0].cost;points.push_back(std::move(mixed));
    }
    double worst=0;
    for(const auto& cut:cuts)for(const auto& point:points) {
        const double excess=value(cut,point.x,l.count)-point.cost;worst=std::max(worst,excess);
        require(excess<=1e-6*(1+std::abs(point.cost)),
                "Search row excludes original full recourse or its integer convex mixture");
    }
    return {{"bases",2},{"native_exports",native_exports},{"integer_full_recourse_solves",integer_count},
        {"convex_mixtures",points.size()-integer_count},{"cross_checks",cuts.size()*points.size()},
        {"maximum_violation",worst},{"ordinary_auxiliary_optimization_calls",0},{"passed",true}};
}

}
int main(){try {std::cout<<Json({{"hand_mix",hand_mix()},{"native_mapping",native_mapping()},
    {"atomic_guards",atomic_guards()},{"full_recourse",complete_recourse_crosschecks()},{"passed",true}}).dump(2)<<'\n';return 0;}
catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
