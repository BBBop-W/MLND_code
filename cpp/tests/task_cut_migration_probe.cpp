// Independent-task migration: complete tiny recourse and cross-design cut checks.
#include "mlnd/airline_resources.hpp"
#include "mlnd/design_face_cuts.hpp"
#include "mlnd/path_support_cuts.hpp"
#include "mlnd/strengthening.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
template<class T> Matrix<T> fill(int n,int m,T value) {
    Matrix<T> result(n,m);std::fill(result.values.begin(),result.values.end(),value);return result;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.name="independent_task_cut_migration";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.single_rates={.5,.5};d.pair_rates=fill(2,2,0.0);
    d.pair_rates(0,0)=.25;d.pair_rates(0,1)=1;d.head_rates={1.75,.5};
    d.demand={2,1.5};d.ground_cost={.1,20};d.deadline={1,1};
    d.flight_times=fill(4,4,.05);d.allowed_links=fill(4,4,1);
    for(int u=0;u<4;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=fill(4,4,.01);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=fill(2,2,.08);d.first_cost=fill(2,2,1.0);
    d.terminal_cost=fill(2,2,.5);d.receipt_mean=fill(2,2,.08);
    d.home_receipt_mean=fill(2,2,.09);d.removal_mean=fill(2,2,.03);
    d.slot_counts=fill(2,1,20);d.slot_cost=fill(2,1,.1);d.occupancy_limit=fill(2,1,14.0);
    d.kappa=fill(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.common_origin_support=fill(2,2,1);return d;
}
std::vector<double> design(const Input& d,int single,int pair,int repeated) {
    const DesignLayout layout(d);std::vector<double> x(layout.count);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h]=1;
    for(auto [q,l]:std::vector<std::pair<int,int>>{{0,single},{d.task_index(0,1),pair},{d.task_index(0,0),repeated}})
        if(l>=0)x[layout.assignment_index(q,l)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
double coefficient(const Coefficients& terms,int index) {
    double value=0;for(auto [j,c]:terms)if(j==index)value+=c;return value;
}
double violation(const RowRecord& row,const std::vector<double>& point) {
    long double activity=0;for(auto [j,c]:row.coefficients)activity+=static_cast<long double>(c)*point.at(j);
    return static_cast<double>(row.sense=='G'?row.rhs-activity:activity-row.rhs);
}
}
int main() { try {
    const double started=wall_time();const auto d=fixture();const DesignLayout layout(d);
    const auto paths=enumerate_paths(d);const auto savings=task_savings_data(d,paths);
    const int pair=d.task_index(0,1),repeated=d.task_index(0,0),theta=layout.count;
    require(savings.heads==d.task_count(),"Task savings still use raw-OD dimension");
    require(savings.saving[0]==0 && savings.signed_saving[0]<0 && savings.saving[pair*2]>1,
            "Positive and negative siblings were aggregated before positive part");
    const auto network=capacity_savings_network(d);
    const auto resources=resource_savings_model(d,network,true,true,true);
    require(resources.group_rhs.size()==static_cast<std::size_t>(d.n_od*d.n_l),
            "Occurrence resources lost raw-OD grouping");
    require(std::abs(coefficient(resources.group_rhs[0],layout.assignment_index(0,0))-.5)<1e-12 &&
            std::abs(coefficient(resources.group_rhs[0],layout.assignment_index(repeated,0))-.5)<1e-12 &&
            std::abs(coefficient(resources.group_rhs[0],layout.assignment_index(pair,0))-1)<1e-12 &&
            std::abs(coefficient(resources.group_rhs[2],layout.assignment_index(pair,0))-1)<1e-12,
            "Occurrence supply does not count independent tasks and AA twice");
    auto tight=d;tight.home_receipt_mean=fill(2,2,2.0);
    const auto tight_network=capacity_savings_network(tight);
    require(std::none_of(tight_network.eligible.begin(),tight_network.eligible.end(),[](bool a){return a;}),
            "Occurrence deadline used mixed rather than home receipt");

    auto missing=paths;
    missing.service_paths.erase(std::remove_if(missing.service_paths.begin(),missing.service_paths.end(),
        [&](const auto& p){return p.b[0]==1 && p.first_od==0 && (p.second_od<0 || p.second_od==0);}),missing.service_paths.end());
    const auto missing_savings=task_savings_data(d,missing);
    require(!missing_savings.supported[1] && !missing_savings.supported[repeated*2+1] &&
            missing_savings.supported[pair*2+1],"An unsupported sibling disabled another task");
    const auto support=build_path_support_cache(d,missing,wall_time()+5);
    require(support.complete_domain && support.static_cuts.size()==2 &&
            support.static_cuts[0].key!=support.static_cuts[1].key,
            "Unsupported sibling cuts were merged by raw first OD");
    for(const auto& cut:support.static_cuts)
        require(coefficient(cut.row.coefficients,layout.assignment_index(pair,1))==0,
                "A path-support cut forbids its supported sibling");
    const auto connectivity=build_airline_connectivity(d,missing);
    for(const auto& row:connectivity.rows)if(row.coefficients.size()==1 && row.sense=='L' && row.rhs==0)
        require(coefficient(row.coefficients,layout.assignment_index(pair,1))==0,
                "A connectivity cut forbids its supported sibling");

    SolverOptions options;options.seconds=10;options.gap=1e-9;options.threads=1;options.seed=43;
    options.tight_airline_bounds=true;
    QueuePolymatroidCuts queue(d,paths,options,&savings);
    QueueResourcePrices prices;prices.source="task_migration_signed_profile";
    prices.locker.assign(d.n_h,0);prices.airline.assign(d.n_nodes()*d.n_nodes(),0);prices.balance.assign(d.n_l,0);
    require(queue.add_profile(prices,5),"Signed queue profile missing");
    TaskResourceSavingsRelaxation task_resource(d,savings,options);
    AirlineResourceRelaxation route_resource(d,paths,options,true);
    ResourceSavingsRelaxation occurrence_resource(resources,options);
    auto master=design_model(d);master.matrix.variable("theta",0,std::numeric_limits<double>::infinity());
    add_task_savings_bound(savings,master,theta);
    std::vector<RowRecord> cuts{master.matrix.rows.back()};
    const auto first_row=master.matrix.rows.size();add_master_strengthening(d,paths,master,theta);
    for(std::size_t i=first_row;i<master.matrix.rows.size();++i)cuts.push_back(master.matrix.rows[i]);
    std::vector<std::vector<double>> points;double max_violation=0;int independent_witness=0;
    for(int s=-1;s<2;++s)for(int p=-1;p<2;++p)for(int a=-1;a<2;++a) {
        const auto x=design(d,s,p,a);auto full=build_formulation(d,paths,&x,nullptr,nullptr,false,false,true);
        NativeSolver solver(full.matrix,options,"task_migration_fixed_recourse");const auto solved=solver.optimize(10);
        require(solved.optimal && solved.has_solution,"Tiny complete recourse not optimal");
        require(point_residuals(full.matrix,solved.values).at("passed"),"Tiny recourse is infeasible");
        if(s==0 && p==1 && a<0 && solved.values[full.courier[1]]<d.demand[1]-1e-4)++independent_witness;
        auto point=x;point.push_back(solved.objective);points.push_back(std::move(point));
        for(std::size_t profile=0;profile<queue.profile_count();++profile) {
            const auto cut=queue.separate_profile(profile,x,0,theta);require(cut.valid,"Invalid queue certificate");cuts.push_back(cut.row);
        }
        const auto joint=queue.separate_joint(x,0,theta,.1);if(joint.valid)cuts.push_back(joint.row);
        cuts.push_back(network.separate(x,0,theta).row);
        if(s==0 && p==1 && a<0)for(double budget:{0.0,3.0}) {
            const auto task=task_resource.separate(x,0,theta,budget);
            const auto route=route_resource.separate(x,0,theta,budget);
            const auto occurrence=occurrence_resource.separate(x,0,theta,budget);
            require(task.valid && route.valid && occurrence.valid,"Resource dual certificate invalid");
            cuts.push_back(task.row);cuts.push_back(route.row);cuts.push_back(occurrence.row);
        }
    }
    require(independent_witness==1,"Pair cannot serve at a departure different from its singleton sibling");
    const auto count=points.size();
    for(std::size_t k=1;k<count;++k) {
        auto point=points[k];for(std::size_t j=0;j<point.size();++j)point[j]=.37*point[j]+.63*points[0][j];
        points.push_back(std::move(point));
    }
    for(const auto& cut:cuts)for(const auto& point:points)max_violation=std::max(max_violation,violation(cut,point));
    require(max_violation<1e-5,"A migrated cut excludes complete recourse or a convex mixture");
    std::array<Json,3> no_faces{Json::object(),Json::object(),Json::object()};
    require(assemble_design_face_cut(d,pair,no_faces,0,design(d,0,1,-1),0,theta).valid,
            "Design face rejects a pair-task ID above raw OD count");
    std::cout<<Json{{"passed",true},{"raw_ods",d.n_od},{"tasks",d.task_count()},
        {"complete_paths",paths.service_paths.size()},{"fixed_recourse_solves",count},
        {"independent_departure_witnesses",independent_witness},{"cuts",cuts.size()},
        {"cross_checks",cuts.size()*points.size()},{"maximum_cut_violation",max_violation},
        {"unsupported_sibling_cuts",support.static_cuts.size()},{"seconds",wall_time()-started}}.dump(2)<<'\n';
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }
