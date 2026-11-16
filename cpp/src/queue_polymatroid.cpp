#include "mlnd/queue_polymatroid.hpp"
#include "mlnd/pricing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace mlnd {
namespace {
constexpr double infinity=std::numeric_limits<double>::infinity();

double rounded_lower(long double value,long double magnitude=0) {
    if(value==0 && magnitude==0)return 0;
    const long double slack=1e-12L*(1+std::max(std::abs(value),magnitude));
    return std::nextafter(static_cast<double>(value-slack),-infinity);
}
double rounded_upper(long double value,long double magnitude=0) {
    if(value==0 && magnitude==0)return 0;
    const long double slack=1e-12L*(1+std::max(std::abs(value),magnitude));
    return std::nextafter(static_cast<double>(value+slack),infinity);
}
long double conjugate(long double saving,long double fleet) {
    if(saving<=fleet)return 0;
    // Equivalent to (sqrt(saving)-sqrt(fleet))^2, without near-threshold
    // cancellation or a potentially overflowing square of saving-fleet.
    const long double value=(saving-fleet)/(std::sqrt(saving)+std::sqrt(fleet));
    return value*value;
}
void require_prices(const std::vector<double>& prices,std::size_t count,bool signed_values) {
    if(prices.size()!=count)throw std::invalid_argument("Queue resource price dimensions disagree");
    for(double price:prices)if(!std::isfinite(price) || (!signed_values && price<0))
        throw std::invalid_argument("Queue resource prices must be finite and have valid signs");
}

using JointAffine = std::array<long double,5>;
struct JointVertex {
    JointAffine coordinates{};
    long double value=0;
};
struct JointFacet {
    JointAffine coefficients{};
    std::array<int,5> basis{};
    long double value=-std::numeric_limits<long double>::infinity(),correction=0;
    int examined=0;
    bool valid=false,complete=false;
};

// The 0/1, five-column vertex matrix has exact small integer entries. Partial
// pivoting solves each candidate supporting plane; the separate nine-vertex
// check below, rather than the equality solve, establishes its validity.
bool joint_plane(const std::array<JointVertex,9>& vertices,
                 const std::array<int,5>& basis,JointAffine& coefficients) {
    std::array<std::array<long double,6>,5> matrix{};
    for(int i=0;i<5;++i) {
        for(int j=0;j<5;++j)matrix[i][j]=vertices[basis[i]].coordinates[j];
        matrix[i][5]=vertices[basis[i]].value;
    }
    for(int j=0;j<5;++j) {
        int pivot=j;
        for(int i=j+1;i<5;++i)if(std::abs(matrix[i][j])>std::abs(matrix[pivot][j]))pivot=i;
        if(std::abs(matrix[pivot][j])<1e-12L)return false;
        std::swap(matrix[pivot],matrix[j]);
        const long double divisor=matrix[j][j];
        for(int k=j;k<6;++k)matrix[j][k]/=divisor;
        for(int i=0;i<5;++i)if(i!=j) {
            const long double multiplier=matrix[i][j];
            for(int k=j;k<6;++k)matrix[i][k]-=multiplier*matrix[j][k];
        }
    }
    for(int j=0;j<5;++j) {
        coefficients[j]=matrix[j][5];if(!std::isfinite(coefficients[j]))return false;
    }
    return true;
}
long double joint_value(const JointAffine& coefficients,const JointAffine& coordinates) {
    return std::inner_product(coefficients.begin(),coefficients.end(),coordinates.begin(),0.0L);
}
JointFacet joint_facet(const std::array<JointVertex,9>& vertices,
                       const JointAffine& point,double deadline) {
    JointFacet best;
    for(int a=0;a<5;++a)for(int b=a+1;b<6;++b)for(int c=b+1;c<7;++c)
    for(int d=c+1;d<8;++d)for(int e=d+1;e<9;++e) {
        if(wall_time()>=deadline)return best;
        const std::array<int,5> basis{a,b,c,d,e};JointAffine coefficients{};++best.examined;
        if(!joint_plane(vertices,basis,coefficients))continue;
        long double magnitude=1,maximum=-std::numeric_limits<long double>::infinity();
        for(long double coefficient:coefficients)magnitude+=std::abs(coefficient);
        for(const auto& vertex:vertices) {
            maximum=std::max(maximum,joint_value(coefficients,vertex.coordinates)-vertex.value);
            magnitude=std::max(magnitude,std::abs(vertex.value));
        }
        const long double guard=(1e-12L+512*std::numeric_limits<long double>::epsilon())*magnitude;
        if(!std::isfinite(guard)||maximum>guard)continue;
        const long double correction=std::max(0.0L,maximum)+guard;
        coefficients[0]-=correction;
        bool valid=true;
        for(const auto& vertex:vertices)
            valid=valid&&joint_value(coefficients,vertex.coordinates)<=vertex.value;
        const long double value=joint_value(coefficients,point);
        if(valid&&std::isfinite(value)&&(!best.valid||value>best.value)) {
            best.valid=true;best.value=value;best.coefficients=coefficients;
            best.basis=basis;best.correction=correction;
        }
    }
    best.complete=true;return best;
}

struct GeneralJointFacet {
    std::vector<long double> coefficients;
    long double correction=0;
    int optimization_calls=0,states_checked=0;
    bool valid=false,optimal=false;
    std::string status="not_solved";
};

// The LP supplies coefficients only. A complete scan of the original state
// table, followed by a conservative intercept correction and a second scan,
// certifies the returned row independently of the LP's termination status.
GeneralJointFacet general_joint_facet(const std::vector<double>& values,
        const std::vector<double>& point,int bases,double deadline) {
    GeneralJointFacet out;
    const int states=bases+1,dimension=1+2*bases;
    if(values.size()!=static_cast<std::size_t>(states)*states ||
       point.size()!=static_cast<std::size_t>(dimension))return out;
    LinearModel model;model.minimize=false;
    for(int j=0;j<dimension;++j)
        model.variable("joint_coefficient_"+std::to_string(j),-IloInfinity,IloInfinity,point[j]);
    for(int first=0;first<states;++first)for(int second=0;second<states;++second) {
        if(wall_time()>=deadline){out.status="table_build_budget_exhausted";return out;}
        Coefficients row{{0,1}};
        if(first)row.emplace_back(first,1);
        if(second)row.emplace_back(bases+second,1);
        model.row("joint_state_"+std::to_string(first)+"_"+std::to_string(second),
                  'L',values[static_cast<std::size_t>(first)*states+second],std::move(row));
    }
    try {
        if(wall_time()>=deadline){out.status="lp_build_budget_exhausted";return out;}
        SolverOptions options;options.threads=1;options.log=false;
        NativeSolver solver(model,options,"joint_queue_coefficient_lp");
        const double remaining=deadline-wall_time();
        if(remaining<=0){out.status="lp_budget_exhausted";return out;}
        ++out.optimization_calls;const auto solved=solver.optimize(remaining);
        out.status=solved.status;out.optimal=solved.optimal;
        if((!solved.has_solution&&!solved.candidate_available)||
           solved.values.size()!=static_cast<std::size_t>(dimension))return out;
        long double magnitude=1;
        for(double value:solved.values) {
            if(!std::isfinite(value))return out;
            out.coefficients.push_back(value);magnitude+=std::abs(static_cast<long double>(value));
        }
        auto activity=[&](int first,int second) {
            return out.coefficients[0]+(first?out.coefficients[first]:0)+
                   (second?out.coefficients[bases+second]:0);
        };
        long double maximum=0;
        for(int first=0;first<states;++first)for(int second=0;second<states;++second) {
            if(wall_time()>=deadline){out.status="validation_budget_exhausted";return out;}
            const double value=values[static_cast<std::size_t>(first)*states+second];
            maximum=std::max(maximum,activity(first,second)-value);
            magnitude=std::max(magnitude,std::abs(static_cast<long double>(value)));
        }
        out.correction=maximum+(1e-12L+512*std::numeric_limits<long double>::epsilon())*magnitude;
        if(!std::isfinite(out.correction))return out;
        out.coefficients[0]-=out.correction;
        for(int first=0;first<states;++first)for(int second=0;second<states;++second) {
            if(wall_time()>=deadline){out.status="validation_budget_exhausted";return out;}
            ++out.states_checked;
            if(!(activity(first,second)<=values[static_cast<std::size_t>(first)*states+second]))return out;
        }
        out.valid=true;
    } catch(const IloException&) {out.status="coefficient_lp_exception";}
      catch(const std::exception&) {out.status="coefficient_lp_exception";}
    return out;
}

}

QueueNativePrices native_queue_resource_prices(const Input& d,const Formulation& f,
        const std::vector<double>& row_prices) {
    QueueNativePrices result;
    auto fail=[&](const char* reason){result.status=reason;return result;};
    if(row_prices.size()!=f.matrix.rows.size() ||
       f.capacity_rows.size()!=static_cast<std::size_t>(d.n_h) ||
       f.link_rows.size()!=static_cast<std::size_t>(d.n_nodes()*d.n_nodes()) ||
       f.balance_rows.size()!=static_cast<std::size_t>(d.n_l) || d.n_l<=0)
        return fail("native_resource_dimensions");
    for(double price:row_prices)if(!std::isfinite(price))return fail("nonfinite_native_row_price");
    auto valid_row=[&](int row,char sense) {
        return row>=0 && row<static_cast<int>(row_prices.size()) && f.matrix.rows[row].sense==sense;
    };
    QueueResourcePrices prices;prices.source="native_cplex_dual_resource_rows";
    for(int row:f.capacity_rows) {
        if(!valid_row(row,'L'))return fail("native_capacity_row");
        if(row_prices[row]>0)return fail("native_capacity_sign");
        prices.locker.push_back(-row_prices[row]);
    }
    for(int row:f.link_rows) {
        if(!valid_row(row,'L'))return fail("native_link_row");
        if(row_prices[row]>0)return fail("native_link_sign");
        prices.airline.push_back(-row_prices[row]);
    }
    for(int row:f.balance_rows) {
        if(!valid_row(row,'E'))return fail("native_balance_row");
        prices.balance.push_back(-row_prices[row]);
    }
    const long double reference=prices.balance.back();
    for(double& potential:prices.balance) {
        potential=static_cast<double>(static_cast<long double>(potential)-reference);
        if(!std::isfinite(potential))return fail("nonfinite_native_balance_difference");
    }
    result.valid=true;result.prices=std::move(prices);result.status="native_resource_prices";
    return result;
}

QueuePolymatroidCuts::QueuePolymatroidCuts(const Input& d,const PathCollection& paths,
        const SolverOptions& options,const TaskSavingsData* supplied,std::size_t max_profiles)
    :data_(d),paths_(paths),layout_(d),heads_(d.task_count()),bases_(d.n_l),nodes_(d.n_nodes()),
     fleet_cost_(d.fleet_cost),service_count_(paths.service_paths.size()),
     empty_count_(paths.empty_paths.size()),max_profiles_(max_profiles) {
    const double started=wall_time();
    if(!d.preprocessed_demand || !d.origin_assignment() || d.flow_policy!="aggregate" ||
       d.first_leg_consistency || d.route_mode!="direct" || d.has_candidate_routes ||
       d.max_orders<1 || d.max_orders>2 || (!paths.complete && !paths.domain_complete) || !paths.support_complete ||
       heads_<=0 || bases_<=0 || !(fleet_cost_>0) || !std::isfinite(fleet_cost_) || max_profiles_==0)
        throw std::invalid_argument("Queue cuts require a complete direct physical task domain and aggregate flow");
    TaskSavingsData owned;
    if(!supplied) {owned=task_savings_data(d,paths);supplied=&owned;}
    if(supplied->heads!=heads_ || supplied->bases!=bases_ || supplied->fleet_cost!=fleet_cost_ ||
       supplied->saving.size()!=static_cast<std::size_t>(heads_*bases_) ||
       !std::isfinite(supplied->courier_cost) || supplied->courier_cost<0)
        throw std::invalid_argument("Queue baseline dimensions or costs disagree");
    for(double saving:supplied->saving)if(!std::isfinite(saving) || saving<0)
        throw std::invalid_argument("Queue baseline savings must be finite and nonnegative");
    if(d.single_rates.size()!=static_cast<std::size_t>(d.n_od) ||
       d.pair_rates.rows!=d.n_od || d.pair_rates.cols!=d.n_od ||
       d.demand.size()!=static_cast<std::size_t>(d.n_od) || d.ground_cost.size()!=d.demand.size())
        throw std::invalid_argument("Queue fixed task rates have incorrect dimensions");
    courier_cost_=supplied->courier_cost;margin_=1e-8*(1+courier_cost_);
    total_demand_=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    if(!std::isfinite(total_demand_) || total_demand_<0)
        throw std::invalid_argument("Queue total demand must be finite and nonnegative");
    const int branches_per_group=d.n_od+1;
    branch_lookup_.assign(d.n_od*bases_*branches_per_group,-1);
    unsupported_.assign(heads_*bases_,false);
    for(int r=0;r<d.n_od;++r)for(int l=0;l<bases_;++l)for(int s=-1;s<d.n_od;++s) {
        const double rate=s<0?d.single_rates[r]:d.pair_rates(r,s);
        if(!std::isfinite(rate) || rate<0 || (d.max_orders==1 && s>=0 && rate!=0))
            throw std::invalid_argument("Queue branch rate is invalid");
        if(rate>0) {
            branch_lookup_[(r*bases_+l)*branches_per_group+s+1]=static_cast<int>(branches_.size());
            branches_.push_back({r,l,s,rate,0});
        }
    }
    // Reuse full-domain counts already computed for the savings profile.
    // Their size is task/base, independent of the number of physical paths.
    if(supplied->path_counts.size()==static_cast<std::size_t>(heads_*bases_)) {
        service_count_=std::accumulate(supplied->path_counts.begin(),supplied->path_counts.end(),std::size_t{0});
        for(auto& branch:branches_)
            branch.paths=supplied->path_counts[d.task_index(branch.head,branch.second)*bases_+branch.origin];
    } else {
        service_count_=visit_service_paths(d,paths,[&](const ServicePath& p) {
            if(p.first_od<0 || p.first_od>=d.n_od || p.second_od<-1 || p.second_od>=d.n_od ||
               p.b[0]<0 || p.b[0]>=bases_ || p.b[1]<0 || p.b[1]>=bases_)
                throw std::invalid_argument("Queue physical path has invalid task or base indices");
            const int branch=branch_lookup_[(p.first_od*bases_+p.b[0])*branches_per_group+p.second_od+1];
            if(branch>=0)++branches_[branch].paths;
        });
    }
    for(const auto& branch:branches_)if(branch.paths==0)
        unsupported_[d.task_index(branch.head,branch.second)*bases_+branch.origin]=true;
    if(!options.task_airline_upper.empty() && options.task_airline_upper.size()!=static_cast<std::size_t>(nodes_*nodes_))
        throw std::invalid_argument("Queue physical airline bound dimensions disagree");
    airline_upper_.assign(nodes_*nodes_,0);
    for(int u=0;u<nodes_;++u)for(int v=0;v<nodes_;++v)if(u!=v && d.allowed_links(u,v)) {
        double bound=u<bases_&&v<bases_?total_demand_*std::max(0,bases_-1):
            u>=bases_&&v>=bases_?total_demand_*(options.tight_airline_bounds?1:d.max_orders):total_demand_;
        if(!options.task_airline_upper.empty()) {
            const double prepared=options.task_airline_upper[u*nodes_+v];
            if(!std::isfinite(prepared) || prepared<0)throw std::invalid_argument("Queue prepared arc bound is invalid");
            bound=std::min(bound,prepared);
        }
        airline_upper_[u*nodes_+v]=bound;
    }
    empty_count_=visit_empty_paths(d,paths,[](const EmptyPath&){});
    domain_identity_=pricing_domain_identity(d,paths);
    if(!complete_domain_unchanged())
        throw std::invalid_argument("Queue complete physical domain counts disagree");
    Profile zero;zero.kind="nonnegative_savings_epi";zero.source="complete_task_savings_S_plus";
    zero.savings=supplied->saving;zero.unsupported.assign(heads_*bases_,false);
    zero.lower_coefficients.assign(layout_.count,0);zero.constant=courier_cost_;
    zero.certificate={{"profile_id",0},{"kind",zero.kind},{"source",zero.source},
        {"heads",heads_},{"tasks",heads_},{"assignment_scope","task"},{"bases",bases_},{"fleet_cost",fleet_cost_},{"constant",zero.constant},
        {"signed_savings",zero.savings},{"unsupported",zero.unsupported},
        {"lower_coefficients",zero.lower_coefficients},{"negative_empty_paths",Json::array()},
        {"prices",nullptr},{"complete_domain",true},{"catalogue_complete",paths_.complete},{"support_complete",true},
        {"service_paths_scanned",service_count_},{"empty_paths_scanned",0},
        {"coefficient_source","task_savings_data_nonnegative_outward_bounds"}};
    profiles_.push_back(std::move(zero));
    metadata_={{"enabled",true},{"master_added_variables",0},{"ordinary_dual_optimization_calls",0},
        {"validity_scope","all_original_integer_designs_and_their_convex_hull"},
        {"heads",heads_},{"tasks",heads_},{"assignment_scope","task"},{"bases",bases_},{"maximum_profiles",max_profiles_},{"profiles",1},
        {"price_profile_attempts",0},{"price_profiles_added",0},{"profile_duplicates",0},
        {"profile_budget_rejections",0},{"profile_numerical_rejections",0},{"separation_calls",0},
        {"profile_scan_seconds",0.0},{"separation_seconds",0.0},
        {"complete_service_paths",service_count_},{"complete_empty_paths",empty_count_},
        {"positive_branches",branches_.size()},
        {"unsupported_groups",std::count(unsupported_.begin(),unsupported_.end(),true)},
        {"build_seconds",wall_time()-started}};
}

bool QueuePolymatroidCuts::complete_domain_unchanged() const {
    // The retained profiles certify the physical domain, not the current RMP.
    // Growing an implicit registry changes neither this identity nor its counts.
    if(!paths_.support_complete || (!paths_.complete && !paths_.domain_complete) ||
       (paths_.domain_complete && paths_.physical_service_count!=service_count_) ||
       (paths_.complete && (paths_.service_paths.size()!=service_count_ ||
                           paths_.empty_paths.size()!=empty_count_)))return false;
    return pricing_domain_identity(data_,paths_)==domain_identity_;
}

bool QueuePolymatroidCuts::prepare_profile(const QueueResourcePrices& prices,double seconds,
        Profile& prepared,std::string& status,bool reject_duplicates) const {
    const double started=wall_time();
    auto reject=[&](const char*,const char* reason){status=reason;return false;};
    if(!std::isfinite(seconds) || seconds<=0)
        return reject("profile_budget_rejections","budget_exhausted");
    if(max_profiles_<=1)return reject("profile_budget_rejections","price_pool_disabled");
    if(!complete_domain_unchanged())
        throw std::invalid_argument("Queue complete physical domain changed after construction");
    require_prices(prices.locker,data_.n_h,false);require_prices(prices.airline,nodes_*nodes_,false);
    require_prices(prices.balance,bases_,true);
    const double deadline=started+seconds;
    std::map<std::pair<int,int>,long double> gates;
    Json gate_records=Json::array();
    for(const auto& gate:prices.branch_gates) {
        if(gate.head<0 || gate.head>=data_.n_od || gate.origin<0 || gate.origin>=bases_ ||
           gate.second<-1 || gate.second>=data_.n_od || gate.arc<0 || gate.arc>=nodes_*nodes_ ||
           !std::isfinite(gate.price) || gate.price<0)
            throw std::invalid_argument("Queue normalized branch gate price is invalid");
        const int branch=branch_lookup_[(gate.head*bases_+gate.origin)*(data_.n_od+1)+gate.second+1];
        if(branch<0)throw std::invalid_argument("Queue gate price requires a positive-rate task branch");
        if(gate.price>0)gates[{branch,gate.arc}]+=gate.price;
    }
    std::unordered_map<std::size_t,long double> gate_lookup;
    std::vector<long double> lower_coefficients(layout_.count,0);
    for(const auto& [position,sum]:gates) {
        const double price=static_cast<double>(sum);
        if(!std::isfinite(price))return reject("profile_numerical_rejections","nonfinite_gate_sum");
        const auto& branch=branches_[position.first];
        gate_lookup[static_cast<std::size_t>(position.first)*nodes_*nodes_+position.second]=price;
        lower_coefficients[layout_.airline+position.second]-=price;
        gate_records.push_back({{"head",branch.head},{"origin",branch.origin},{"second",branch.second},
            {"arc",position.second},{"price",price}});
    }
    const Json price_record={{"source",prices.source},{"locker",prices.locker},{"airline",prices.airline},
        {"balance",prices.balance},{"branch_gates",gate_records}};
    Json identity=price_record;identity.erase("source");const std::string key=identity.dump();
    for(const auto& profile:profiles_)if(reject_duplicates && profile.key==key)
        return reject("profile_duplicates","duplicate_prices");
    for(int h=0;h<data_.n_h;++h)for(int level=0;level<data_.n_levels;++level)
        lower_coefficients[layout_.capacity+h*data_.n_levels+level]-=
            static_cast<long double>(prices.locker[h])*data_.occupancy_limit(h,level);
    for(int e=0;e<nodes_*nodes_;++e)
        lower_coefficients[layout_.airline+e]-=static_cast<long double>(prices.airline[e])*airline_upper_[e];
    std::vector<double> minimum(branches_.size(),infinity);
    std::size_t scanned=0,positive_scanned=0;
    bool finite_minima=true;
    try {
      scanned=visit_service_paths(data_,paths_,[&](const ServicePath& path) {
        if(path.first_od<0 || path.first_od>=data_.n_od || path.second_od<-1 || path.second_od>=data_.n_od ||
           path.b[0]<0 || path.b[0]>=bases_ || path.b[1]<0 || path.b[1]>=bases_)
            throw std::invalid_argument("Queue physical path has invalid task or base indices");
        const int branch_id=branch_lookup_[(path.first_od*bases_+path.b[0])*(data_.n_od+1)+path.second_od+1];
        if(branch_id<0)return;
        ++positive_scanned;const auto& branch=branches_[branch_id];
        if(!std::isfinite(path.cost) || path.cost<0 || !std::isfinite(path.time) || path.time<0)
            throw std::invalid_argument("Queue path cost or time is invalid");
        const long double reward=data_.ground_cost[branch.head]+(branch.second<0?0.0L:data_.ground_cost[branch.second]);
        long double cost=path.cost,absolute=std::abs(cost);
        auto add=[&](long double term){cost+=term;absolute+=std::abs(term);};
        add(static_cast<long double>(fleet_cost_)*path.time);add(-reward);
        add(prices.balance[path.b[0]]);add(-static_cast<long double>(prices.balance[path.b[1]]));
        for(const auto& [h,time]:path.slot_times) {
            if(h<0 || h>=data_.n_h || !std::isfinite(time) || time<0)
                throw std::invalid_argument("Queue path occupancy is invalid");
            add(static_cast<long double>(prices.locker[h])*time);
        }
        std::vector<int> present;
        for(const auto& [u,v,count]:path.arc_counts) {
            if(u<0 || u>=nodes_ || v<0 || v>=nodes_ || count<=0 || !data_.allowed_links(u,v))
                throw std::invalid_argument("Queue physical path arc is invalid");
            const int arc=u*nodes_+v;add(static_cast<long double>(prices.airline[arc])*count);
            if(std::find(present.begin(),present.end(),arc)==present.end())present.push_back(arc);
        }
        cost*=branch.rate;absolute*=branch.rate;
        for(int arc:present) {
            const auto found=gate_lookup.find(static_cast<std::size_t>(branch_id)*nodes_*nodes_+arc);
            if(found!=gate_lookup.end()){cost+=found->second;absolute+=std::abs(found->second);}
        }
        const double safe=rounded_lower(cost,absolute);
        if(!std::isfinite(safe)){finite_minima=false;return;}
        minimum[branch_id]=std::min(minimum[branch_id],safe);
      },deadline);
    } catch(const EnumerationStopped&) {
        return reject("profile_budget_rejections","incomplete_path_scan");
    }
    if(!finite_minima)return reject("profile_numerical_rejections","nonfinite_path_minimum");
    if(scanned!=service_count_)throw std::invalid_argument("Queue physical path count changed after construction");
    std::vector<long double> savings(heads_*bases_,0),saving_magnitude(heads_*bases_,0);
    Json missing=Json::array();
    for(std::size_t k=0;k<branches_.size();++k) {
        const auto& branch=branches_[k];const int group=data_.task_index(branch.head,branch.second)*bases_+branch.origin;
        if(branch.paths==0) {missing.push_back({branch.head,branch.origin,branch.second});continue;}
        savings[group]-=minimum[k];saving_magnitude[group]+=std::abs(static_cast<long double>(minimum[k]));
    }
    Json negative_empty=Json::array();
    bool finite_empty=true;std::size_t empty_scanned=0;
    try {
      empty_scanned=visit_empty_paths(data_,paths_,[&](const EmptyPath& path) {
        if(path.b[0]<0 || path.b[0]>=bases_ || path.b[1]<0 || path.b[1]>=bases_ || path.b[0]==path.b[1] ||
           !std::isfinite(path.cost) || path.cost<0 || !std::isfinite(path.time) || path.time<0)
            throw std::invalid_argument("Queue empty path is invalid");
        const int arc=path.b[0]*nodes_+path.b[1];
        if(path.arc_counts.size()!=1 || std::get<0>(path.arc_counts.front())!=path.b[0] ||
           std::get<1>(path.arc_counts.front())!=path.b[1] || std::get<2>(path.arc_counts.front())!=1)
            throw std::invalid_argument("Queue empty correction requires direct base-to-base paths");
        const long double busy=static_cast<long double>(fleet_cost_)*path.time;
        const long double cost=path.cost+busy+prices.airline[arc]+
            static_cast<long double>(prices.balance[path.b[0]])-prices.balance[path.b[1]];
        const long double absolute=path.cost+std::abs(busy)+prices.airline[arc]+
            std::abs(static_cast<long double>(prices.balance[path.b[0]]))+std::abs(static_cast<long double>(prices.balance[path.b[1]]));
        const double safe=rounded_lower(cost,absolute);
        if(!std::isfinite(safe)){finite_empty=false;return;}
        if(safe<0) {
            const long double coefficient=static_cast<long double>(total_demand_)*safe;
            lower_coefficients[layout_.airline+arc]+=coefficient;
            negative_empty.push_back({{"path_id",path.id},{"base_pair",path.b},{"arc",arc},{"adjusted_cost",safe},
                {"affine_coefficient",static_cast<double>(coefficient)}});
        }
      },deadline);
    } catch(const EnumerationStopped&) {
        return reject("profile_budget_rejections","incomplete_empty_scan");
    }
    if(!finite_empty)return reject("profile_numerical_rejections","nonfinite_empty_cost");
    if(empty_scanned!=empty_count_)throw std::invalid_argument("Queue physical empty-path count changed after construction");
    Profile profile;profile.id=next_id_;profile.kind="resource_price_epi";profile.source=prices.source;profile.key=key;
    profile.prices=prices;
    profile.constant=courier_cost_;profile.unsupported=unsupported_;
    profile.savings.resize(heads_*bases_);profile.lower_coefficients.resize(layout_.count);
    for(int group=0;group<heads_*bases_;++group) {
        profile.savings[group]=unsupported_[group]?0:rounded_upper(savings[group],saving_magnitude[group]);
        if(!std::isfinite(profile.savings[group]))return reject("profile_numerical_rejections","nonfinite_task_saving");
    }
    for(int j=0;j<layout_.count;++j) {
        profile.lower_coefficients[j]=rounded_lower(lower_coefficients[j]);
        if(!std::isfinite(profile.lower_coefficients[j]))return reject("profile_numerical_rejections","nonfinite_resource_coefficient");
    }
    profile.certificate={{"profile_id",profile.id},{"kind",profile.kind},{"source",profile.source},
        {"heads",heads_},{"tasks",heads_},{"assignment_scope","task"},{"bases",bases_},{"fleet_cost",fleet_cost_},{"constant",profile.constant},
        {"signed_savings",profile.savings},{"unsupported",profile.unsupported},
        {"lower_coefficients",profile.lower_coefficients},{"negative_empty_paths",negative_empty},
        {"prices",price_record},{"airline_upper",airline_upper_},{"total_demand",total_demand_},
        {"missing_branches",missing},{"complete_domain",true},{"catalogue_complete",paths_.complete},{"support_complete",true},
        {"service_paths_scanned",scanned},{"positive_service_paths_scanned",positive_scanned},{"empty_paths_scanned",empty_count_},
        {"normalized_adjusted_cost","rate*(cost+F*time-reward+mu*slots+eta*arc_counts+balance_difference)+tau_presence"},
        {"rounding_relative_guard",1e-12},{"scan_seconds",wall_time()-started}};
    if(wall_time()>=deadline)return reject("profile_budget_rejections","incomplete_profile_finalization");
    prepared=std::move(profile);status="complete_price_profile";return true;
}

bool QueuePolymatroidCuts::add_profile(const QueueResourcePrices& prices,double seconds) {
    const double started=wall_time();
    auto increment=[&](const char* key){metadata_[key]=metadata_.at(key).get<int>()+1;};
    increment("price_profile_attempts");Profile profile;std::string status;
    const bool complete=prepare_profile(prices,seconds,profile,status,true);
    metadata_["last_profile_status"]=status;
    metadata_["profile_scan_seconds"]=metadata_.at("profile_scan_seconds").get<double>()+wall_time()-started;
    if(!complete) {
        increment(status=="duplicate_prices"?"profile_duplicates":
            status.find("nonfinite")==0?"profile_numerical_rejections":"profile_budget_rejections");
        return false;
    }
    // Commit only after all physical columns and the certificate are complete.
    if(profiles_.size()>=max_profiles_)profiles_.erase(profiles_.begin()+1);
    profiles_.push_back(std::move(profile));++next_id_;increment("price_profiles_added");
    metadata_["profiles"]=profiles_.size();metadata_["last_profile_id"]=next_id_-1;
    metadata_["last_profile_status"]="complete_price_profile";
    return true;
}

TaskSavingsCut QueuePolymatroidCuts::separate_profile(std::size_t index,
        const std::vector<double>& design,double theta_value,int theta) const {
    if(index>=profiles_.size())throw std::invalid_argument("Queue profile index is invalid");
    return separate_prepared(profiles_[index],index,design,theta_value,theta);
}

TaskSavingsCut QueuePolymatroidCuts::separate_prepared(const Profile& profile,std::size_t index,
        const std::vector<double>& design,double theta_value,int theta) const {
    const double started=wall_time();
    if(design.size()<static_cast<std::size_t>(layout_.count) ||
       theta<layout_.count || !std::isfinite(theta_value))
        throw std::invalid_argument("Queue separation point or profile index is invalid");
    for(int j=0;j<layout_.count;++j)if(!std::isfinite(design[j]) || design[j]<-1e-8 || design[j]>1+1e-8)
        throw std::invalid_argument("Queue separation needs a unit-box design");
    std::vector<long double> coefficients(profile.lower_coefficients.begin(),profile.lower_coefficients.end());
    long double constant=profile.constant,absolute=std::abs(constant),increment_guard=0;
    for(long double coefficient:coefficients)absolute+=std::abs(coefficient);
    Json bases=Json::array();
    for(int l=0;l<bases_;++l) {
        std::vector<int> order(heads_);std::iota(order.begin(),order.end(),0);
        std::vector<double> point(heads_),weights(heads_),deltas(heads_);
        std::vector<bool> complemented(heads_),required(heads_);
        long double shift=0,weight_magnitude=0;int missing=0;
        for(int r=0;r<heads_;++r) {
            const int group=r*bases_+l;const double saving=profile.savings[group];
            required[r]=profile.unsupported[group];missing+=required[r];
            complemented[r]=required[r] || saving<0;
            weights[r]=required[r]?0:std::abs(saving);
            if(!required[r] && saving<0)shift+=saving;
            weight_magnitude+=weights[r];
            const double alpha=std::clamp(design[layout_.assignment_index(r,l)],0.0,1.0);
            point[r]=complemented[r]?1-alpha:alpha;
        }
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return point[a]>point[b];});
        long double sum=shift,previous=0;
        Json prefix_values=Json::array({0.0});
        for(int r:order) {
            sum+=weights[r];if(required[r])--missing;
            const long double current=missing==0?conjugate(sum,fleet_cost_):0;
            const long double guard=1e-12L+128*(heads_+10)*std::numeric_limits<long double>::epsilon();
            const long double error=guard*(1+std::abs(shift)+weight_magnitude+std::abs(previous)+std::abs(current));
            // Upper-bound each nonnegative saving increment. This makes the
            // affine saving envelope conservative on every transformed bit.
            const long double increment=std::max(0.0L,current-previous)+error;
            const double safe_increment=std::nextafter(static_cast<double>(increment),infinity);
            if(!std::isfinite(safe_increment)) {
                TaskSavingsCut failed;failed.status="nonfinite_queue_increment";failed.seconds=wall_time()-started;return failed;
            }
            deltas[r]=-safe_increment;increment_guard+=error;
            const int assignment=layout_.assignment_index(r,l);
            if(complemented[r]) {constant-=safe_increment;coefficients[assignment]+=safe_increment;}
            else coefficients[assignment]-=safe_increment;
            absolute+=safe_increment*(complemented[r]?2:1);
            previous=current;prefix_values.push_back(static_cast<double>(current));
        }
        bases.push_back({{"base",l},{"shift",static_cast<double>(shift)},{"weights",weights},
            {"order",order},{"complemented",complemented},{"necessary_gates",required},
            {"transformed_point",point},{"deltas",deltas},{"prefix_savings",prefix_values}});
    }
    TaskSavingsCut cut;cut.status="queue_polymatroid_complete";
    cut.row.name="queue_polymatroid_cut";cut.row.sense='G';cut.row.coefficients.emplace_back(theta,1);
    // Conversion and the complement expansion are protected over [0,1]^n by
    // subtracting an absolute error bound from the affine intercept.
    long double conversion=0;
    for(int j=0;j<layout_.count;++j)if(coefficients[j]!=0) {
        const double coefficient=static_cast<double>(coefficients[j]);
        if(!std::isfinite(coefficient)) {cut.status="nonfinite_queue_coefficient";cut.seconds=wall_time()-started;return cut;}
        conversion+=std::abs(static_cast<long double>(coefficient)-coefficients[j]);
        cut.row.coefficients.emplace_back(j,-coefficient);
    }
    const long double arithmetic=256*std::numeric_limits<long double>::epsilon()*(1+absolute);
    const long double correction=margin_+conversion+arithmetic;
    cut.row.rhs=rounded_lower(constant-correction);
    if(!std::isfinite(cut.row.rhs)) {cut.status="nonfinite_queue_intercept";cut.seconds=wall_time()-started;return cut;}
    long double lower=cut.row.rhs;
    std::ostringstream key;key<<std::scientific<<std::setprecision(12)<<cut.row.rhs<<';';
    for(const auto& [variable,coefficient]:cut.row.coefficients)if(variable!=theta) {
        lower-=static_cast<long double>(coefficient)*design[variable];key<<variable<<':'<<coefficient<<';';
    }
    cut.violation=static_cast<double>(lower-theta_value);
    if(!std::isfinite(static_cast<double>(lower)) || !std::isfinite(cut.violation)) {
        cut.status="nonfinite_queue_evaluation";cut.seconds=wall_time()-started;return cut;
    }
    cut.valid=true;cut.key=key.str();
    cut.violated=cut.valid && cut.violation>1e-7*(1+std::abs(courier_cost_)+std::abs(theta_value));
    cut.dual_savings=courier_cost_-static_cast<double>(lower);
    cut.certificate={{"kind","queue_polymatroid"},{"profile_index",index},{"profile",profile.certificate},
        {"bases",bases},{"global_integer_valid",true},{"complete_domain",true},
        {"ordinary_auxiliary_optimization_calls",0},{"numerical_margin",margin_},
        {"increment_error_allowance",static_cast<double>(increment_guard)},
        {"conversion_error",static_cast<double>(conversion)},{"arithmetic_error",static_cast<double>(arithmetic)},
        {"intercept_correction",static_cast<double>(correction)},
        {"separation_design",std::vector<double>(design.begin(),design.begin()+layout_.count)},
        {"separation_theta",theta_value},
        {"value_at_separation_point",static_cast<double>(lower)},
        {"direct_global_bound_update",false}};
    cut.seconds=wall_time()-started;return cut;
}

TaskSavingsCut QueuePolymatroidCuts::separate(const std::vector<double>& design,double theta_value,int theta) {
    const double started=wall_time();TaskSavingsCut best;best.status="no_valid_queue_profile";
    for(std::size_t index=0;index<profiles_.size();++index) {
        auto cut=separate_profile(index,design,theta_value,theta);
        if(cut.valid && (!best.valid || cut.violation>best.violation))best=std::move(cut);
    }
    best.seconds=wall_time()-started;
    metadata_["separation_calls"]=metadata_.at("separation_calls").get<int>()+1;
    metadata_["separation_seconds"]=metadata_.at("separation_seconds").get<double>()+best.seconds;
    if(best.valid)metadata_["last_selected_profile_id"]=best.certificate.at("profile").at("profile_id");
    return best;
}

TaskSavingsCut QueuePolymatroidCuts::separate_joint(const std::vector<double>& design,
        double theta_value,int theta,double seconds) {
    const double started=wall_time();TaskSavingsCut best;best.status="no_applicable_joint_queue_profile";
    int attempted=0,completed=0,facets=0,lp_calls=0,states_checked=0;bool budget_exhausted=false;
    constexpr std::size_t pair_limit=8;
    std::size_t eligible_pairs=0;
    bool candidate_selection_complete=false;
    Json selected_pairs=Json::array();
    auto finish=[&]() {
        best.seconds=wall_time()-started;
        metadata_["joint_calls"]=metadata_.value("joint_calls",0)+1;
        metadata_["joint_seconds"]=metadata_.value("joint_seconds",0.0)+best.seconds;
        metadata_["joint_pairs_attempted"]=metadata_.value("joint_pairs_attempted",0)+attempted;
        metadata_["joint_pairs_completed"]=metadata_.value("joint_pairs_completed",0)+completed;
        metadata_["last_joint"]={{"status",best.status},{"time_budget",seconds},
            {"seconds",best.seconds},{"pairs_attempted",attempted},{"pairs_completed",completed},
            {"candidate_bases_examined",facets},{"budget_exhausted",budget_exhausted},
            {"path_scans",0},{"auxiliary_optimization_calls",lp_calls},{"pair_limit",pair_limit},
            {"eligible_pairs",eligible_pairs},{"candidate_selection_complete",candidate_selection_complete},
            {"candidate_pairs_retained",selected_pairs.size()},{"selected_pairs",selected_pairs},
            {"states_checked",states_checked},{"bases",bases_},
            {"scope","complete_profile_general_L_K2_task_integer_design"},{"direct_global_bound_update",false}};
        return best;
    };
    if(!std::isfinite(seconds)||seconds<=0) {
        best.status="joint_budget_exhausted";budget_exhausted=true;return finish();
    }
    if(bases_<1||heads_<2||data_.max_orders!=2||!complete_domain_unchanged()) {
        best.status="joint_scope_unavailable";return finish();
    }
    if(design.size()<static_cast<std::size_t>(layout_.count)||theta<layout_.count||!std::isfinite(theta_value))
        throw std::invalid_argument("Joint queue separation point is invalid");
    for(int j=0;j<layout_.count;++j)if(!std::isfinite(design[j])||design[j]<-1e-8||design[j]>1+1e-8)
        throw std::invalid_argument("Joint queue separation needs a unit-box design");
    for(int r=0;r<heads_;++r) {
        double sum=0;for(int l=0;l<bases_;++l)sum+=design[layout_.assignment_index(r,l)];
        if(sum>1+1e-8)
            throw std::invalid_argument("Joint queue separation needs exclusive task assignments");
    }
    const double deadline=started+seconds;
    struct Pair {std::size_t profile=0;int first=0,second=0;double score=0;};
    std::vector<Pair> pairs;
    pairs.reserve(pair_limit);
    // The tie order is the original profile/task traversal order. Keeping the
    // worst retained candidate at the heap root reproduces stable_sort's first
    // eight entries without storing the O(profiles * tasks^2) candidate list.
    const auto better=[](const Pair& a,const Pair& b) {
        if(a.score!=b.score)return a.score>b.score;
        return std::tie(a.profile,a.first,a.second)<std::tie(b.profile,b.first,b.second);
    };
    for(std::size_t index=1;index<profiles_.size();++index) {
        const auto& profile=profiles_[index];
        if(!profile.certificate.value("complete_domain",false)||
           !profile.certificate.value("support_complete",false)||
           profile.certificate.value("service_paths_scanned",std::size_t{0})!=service_count_||
           profile.certificate.value("empty_paths_scanned",std::size_t{0})!=empty_count_||
           profile.savings.size()!=static_cast<std::size_t>(heads_*bases_)||
           profile.unsupported.size()!=profile.savings.size()||
           profile.lower_coefficients.size()!=static_cast<std::size_t>(layout_.count))continue;
        for(int first=0;first<heads_&&!budget_exhausted;++first)for(int second=first+1;second<heads_;++second) {
            if(wall_time()>=deadline){budget_exhausted=true;break;}
            bool odd_complements=false,unsupported=false,signed_profile=false;double fractionality=0;
            for(int r:{first,second})for(int l=0;l<bases_;++l) {
                const int group=r*bases_+l;const bool missing=profile.unsupported[group];
                odd_complements^=missing||profile.savings[group]<0;unsupported|=missing;
                signed_profile|=profile.savings[group]<0;
                const double alpha=std::clamp(design[layout_.assignment_index(r,l)],0.0,1.0);
                fractionality+=std::min(alpha,1-alpha);
            }
            // Two-task nonnegative supported profiles admit a maximal coupling
            // for every L. Complement parity is an L2-only screening rule;
            // for other L retain every signed or unsupported candidate.
            if(unsupported||(bases_==2?odd_complements:signed_profile)) {
                ++eligible_pairs;
                const Pair candidate{index,first,second,fractionality};
                if(pairs.size()<pair_limit) {
                    pairs.push_back(candidate);std::push_heap(pairs.begin(),pairs.end(),better);
                } else if(better(candidate,pairs.front())) {
                    std::pop_heap(pairs.begin(),pairs.end(),better);
                    pairs.back()=candidate;std::push_heap(pairs.begin(),pairs.end(),better);
                }
            }
        }
        if(budget_exhausted)break;
    }
    candidate_selection_complete=!budget_exhausted;
    std::sort(pairs.begin(),pairs.end(),better);
    for(const auto& pair:pairs)selected_pairs.push_back({{"profile",pair.profile},
        {"first",pair.first},{"second",pair.second},{"score",pair.score}});
    for(const auto& pair:pairs) {
        if(wall_time()>=deadline){budget_exhausted=true;break;}
        ++attempted;const auto& profile=profiles_[pair.profile];
        const std::array<int,2> selected{pair.first,pair.second};
        const int state_count=bases_+1;
        std::array<JointVertex,9> vertices{};Json vertex_records=Json::array();
        std::vector<double> state_values;
        std::vector<long double> affine_coefficients;
        Json facet_basis=nullptr;
        long double facet_correction=0;
        bool facet_complete=false,lp_optimal=false;
        int facet_examined=0,pair_lp_calls=0,pair_states_checked=0;
        std::string lp_status="not_used";
        bool finite=true,table_complete=true;
        for(int first=0;first<state_count;++first)for(int second=0;second<state_count;++second) {
            if(wall_time()>=deadline){table_complete=false;break;}
            long double value=0,magnitude=0;
            for(int l=0;l<bases_;++l) {
                long double saving=0,absolute=0;bool blocked=false;
                for(int i=0;i<2;++i)if((i==0?first:second)==l+1) {
                    const int group=bases_*selected[i]+l;blocked|=profile.unsupported[group];
                    saving+=profile.savings[group];absolute+=std::abs(profile.savings[group]);
                }
                if(!blocked){value-=conjugate(saving,fleet_cost_);magnitude+=absolute;}
            }
            const double safe=rounded_lower(value,magnitude);
            finite=finite&&std::isfinite(safe);state_values.push_back(safe);
            if(bases_==2) {
                auto& vertex=vertices[3*first+second];
                vertex.coordinates={1,first==1?1.0L:0.0L,first==2?1.0L:0.0L,
                    second==1?1.0L:0.0L,second==2?1.0L:0.0L};vertex.value=safe;
            }
            vertex_records.push_back({{"states",{first,second}},
                {"queue_value",static_cast<double>(value)},{"safe_queue_value",safe}});
        }
        if(!table_complete){budget_exhausted=true;break;}
        if(!finite)continue;
        std::vector<double> point(1+2*bases_,1);
        for(int i=0;i<2;++i)for(int l=0;l<bases_;++l)
            point[1+bases_*i+l]=design[layout_.assignment_index(selected[i],l)];
        if(bases_==2) {
            JointAffine old_point{};std::copy(point.begin(),point.end(),old_point.begin());
            const auto facet=joint_facet(vertices,old_point,deadline);
            facets+=facet.examined;completed+=facet.complete;
            if(!facet.valid)continue;
            affine_coefficients.assign(facet.coefficients.begin(),facet.coefficients.end());
            facet_basis=facet.basis;facet_correction=facet.correction;
            facet_complete=facet.complete;facet_examined=facet.examined;pair_states_checked=9;
        } else {
            const auto facet=general_joint_facet(state_values,point,bases_,deadline);
            lp_calls+=facet.optimization_calls;pair_lp_calls=facet.optimization_calls;
            states_checked+=facet.states_checked;
            if(!facet.valid)continue;
            ++completed;affine_coefficients=facet.coefficients;facet_correction=facet.correction;
            lp_optimal=facet.optimal;lp_status=facet.status;pair_states_checked=facet.states_checked;
        }
        if(bases_==2)states_checked+=pair_states_checked;
        std::vector<long double> coefficients(profile.lower_coefficients.begin(),profile.lower_coefficients.end());
        long double constant=profile.constant+affine_coefficients[0];
        for(int i=0;i<2;++i)for(int l=0;l<bases_;++l)
            coefficients[layout_.assignment_index(selected[i],l)]+=affine_coefficients[1+bases_*i+l];
        std::vector<long double> positive_sums(bases_,0);
        for(int r=0;r<heads_;++r)for(int l=0;l<bases_;++l)if(!profile.unsupported[r*bases_+l])
            positive_sums[l]+=std::max(0.0,profile.savings[r*bases_+l]);
        std::vector<double> penalties(heads_*bases_,0);
        for(int r=0;r<heads_;++r)if(r!=pair.first&&r!=pair.second)for(int l=0;l<bases_;++l) {
            const int group=r*bases_+l;const double saving=profile.savings[group];
            if(profile.unsupported[group]||saving<=0)continue;
            const long double sum=positive_sums[l];
            const long double penalty=conjugate(sum,fleet_cost_)-conjugate(sum-saving,fleet_cost_);
            penalties[group]=rounded_upper(std::max(0.0L,penalty),sum);
            finite=finite&&std::isfinite(penalties[group]);
            coefficients[layout_.assignment_index(r,l)]-=penalties[group];
        }
        if(!finite)continue;
        TaskSavingsCut cut;cut.row.name="queue_joint_cut";cut.row.sense='G';cut.row.coefficients.emplace_back(theta,1);
        long double conversion=0,absolute=std::abs(constant);
        for(int j=0;j<layout_.count;++j)if(coefficients[j]!=0) {
            const double coefficient=static_cast<double>(coefficients[j]);
            if(!std::isfinite(coefficient)){finite=false;break;}
            conversion+=std::abs(static_cast<long double>(coefficient)-coefficients[j]);
            absolute+=std::abs(coefficients[j]);cut.row.coefficients.emplace_back(j,-coefficient);
        }
        const long double arithmetic=(256+16*heads_)*std::numeric_limits<long double>::epsilon()*(1+absolute);
        const long double correction=margin_+conversion+arithmetic;
        cut.row.rhs=rounded_lower(constant-correction);
        if(!finite||!std::isfinite(cut.row.rhs))continue;
        long double lower=cut.row.rhs;std::ostringstream key;
        key<<"queue_joint:"<<std::scientific<<std::setprecision(12)<<cut.row.rhs<<';';
        for(const auto& [j,a]:cut.row.coefficients)if(j!=theta) {
            lower-=static_cast<long double>(a)*design[j];key<<j<<':'<<a<<';';
        }
        cut.violation=static_cast<double>(lower-theta_value);
        if(!std::isfinite(lower)||!std::isfinite(cut.violation))continue;
        cut.valid=true;cut.key=key.str();cut.status="queue_joint_vertex_certified";
        cut.violated=cut.violation>1e-7*(1+std::abs(courier_cost_)+std::abs(theta_value));
        cut.dual_savings=courier_cost_-static_cast<double>(lower);
        std::vector<double> affine;for(long double a:affine_coefficients)affine.push_back(static_cast<double>(a));
        cut.certificate={{"kind","queue_joint"},{"profile_index",pair.profile},{"profile",profile.certificate},
            {"selected_heads",selected},{"selected_tasks",selected},{"assignment_scope","task"},{"vertices",vertex_records},{"affine_coefficients",affine},
            {"facet_basis",facet_basis},{"facet_validation_correction",static_cast<double>(facet_correction)},
            {"facet_enumeration_complete",facet_complete},{"candidate_bases_examined",facet_examined},
            {"coefficient_method",bases_==2?"L2_facet_enumeration":"complete_state_coefficient_lp"},
            {"coefficient_lp_optimal",lp_optimal},{"coefficient_lp_status",lp_status},
            {"auxiliary_optimization_calls",pair_lp_calls},{"states_checked",pair_states_checked},
            {"all_states_validated",true},{"state_count_per_task",state_count},{"bases",bases_},
            {"external_positive_sums",positive_sums},
            {"external_penalties",penalties},{"lifting_rule","JL_positive_marginal_at_full_positive_sum"},
            {"global_integer_valid",true},{"complete_domain",true},{"scope","general_L_K2_two_task_integer_design"},
            {"path_scans",0},{"ordinary_auxiliary_optimization_calls",pair_lp_calls},{"numerical_margin",margin_},
            {"conversion_error",static_cast<double>(conversion)},{"arithmetic_error",static_cast<double>(arithmetic)},
            {"intercept_correction",static_cast<double>(correction)},
            {"separation_design",std::vector<double>(design.begin(),design.begin()+layout_.count)},
            {"separation_theta",theta_value},{"value_at_separation_point",static_cast<double>(lower)},
            {"direct_global_bound_update",false}};
        if(!best.valid||cut.violation>best.violation)best=std::move(cut);
    }
    if(wall_time()>=deadline)budget_exhausted=true;
    if(!best.valid&&budget_exhausted)best.status="joint_budget_exhausted";
    return finish();
}


QueuePriceSearchResult QueuePolymatroidCuts::search_prices(
        const QueueResourcePrices& candidate,const std::vector<double>& design,
        double theta_value,int theta,double seconds,std::size_t max_evaluations) {
    const double started=wall_time();
    QueuePriceSearchResult result;result.cut=separate(design,theta_value,theta);
    const TaskSavingsCut before=result.cut;
    const std::size_t before_count=profiles_.size();
    int attempts=0,complete_scans=0;
    Json evaluations=Json::array();
    result.diagnostics={{"status","not_started"},{"max_evaluations",std::min<std::size_t>(7,max_evaluations)},
        {"profile_count_before",before_count},{"best_before",before.valid?Json(before.violation+theta_value):Json(nullptr)},
        {"profile_added",false},{"best_line_weight",nullptr},{"ordinary_auxiliary_optimization_calls",0},
        {"master_added_variables",0},{"direct_global_bound_update",false}};
    auto finish=[&](const std::string& status) {
        result.diagnostics["status"]=status;
        result.diagnostics["evaluation_count"]=evaluations.size();
        result.diagnostics["scan_attempts"]=attempts;
        result.diagnostics["complete_scans"]=complete_scans;
        result.diagnostics["evaluations"]=evaluations;
        result.diagnostics["profile_count_after"]=profiles_.size();
        result.diagnostics["best_after"]=result.cut.valid?Json(result.cut.violation+theta_value):Json(nullptr);
        result.diagnostics["seconds"]=wall_time()-started;
        metadata_["price_search_calls"]=metadata_.value("price_search_calls",0)+1;
        metadata_["price_search_complete_scans"]=metadata_.value("price_search_complete_scans",0)+complete_scans;
        metadata_["price_search_seconds"]=metadata_.value("price_search_seconds",0.0)+wall_time()-started;
        if(result.cut.valid)metadata_["last_selected_profile_id"]=result.cut.certificate.at("profile").at("profile_id");
        metadata_["last_price_search"]=result.diagnostics;
        return result;
    };
    if(!std::isfinite(seconds) || seconds<=0)return finish("budget_exhausted");
    if(max_profiles_<=1)return finish("price_pool_disabled");
    max_evaluations=std::min<std::size_t>(7,max_evaluations);
    if(max_evaluations<2)return finish("insufficient_endpoint_budget");
    const double deadline=started+seconds;
    std::size_t endpoint_index=profiles_.size();
    double endpoint_value=-infinity;
    for(std::size_t i=1;i<profiles_.size();++i) {
        const auto cut=separate_profile(i,design,theta_value,theta);
        if(cut.valid && cut.violation>endpoint_value) {endpoint_value=cut.violation;endpoint_index=i;}
    }
    QueueResourcePrices first;
    if(endpoint_index<profiles_.size())first=profiles_[endpoint_index].prices;
    else {
        first.source="zero_resource_price_endpoint";
        first.locker.assign(data_.n_h,0);first.airline.assign(nodes_*nodes_,0);first.balance.assign(bases_,0);
    }
    result.diagnostics["endpoint_a_source"]=first.source;
    result.diagnostics["endpoint_b_source"]=candidate.source;
    struct Observation {double weight=0,value=0;};
    std::vector<Observation> observed;
    Profile winner;bool new_winner=false;
    double winning_weight=0;
    std::string failure;
    std::string first_key,second_key;
    auto evaluate=[&](double weight,const QueueResourcePrices& prices,const Profile* cached) {
        if(wall_time()>=deadline){failure="budget_exhausted";return false;}
        Profile prepared;
        if(cached)prepared=*cached;
        else {
            ++attempts;
            try {
                if(!prepare_profile(prices,deadline-wall_time(),prepared,failure,false))return false;
            } catch(const std::invalid_argument& error) {
                failure=std::string("invalid_prices: ")+error.what();return false;
            }
            ++complete_scans;
        }
        auto cut=separate_prepared(prepared,profiles_.size(),design,theta_value,theta);
        if(!cut.valid){failure="invalid_complete_profile_row";return false;}
        const double lower=cut.violation+theta_value;
        observed.push_back({weight,lower});
        evaluations.push_back({{"line_weight",weight},{"value",lower},{"cached_endpoint",cached!=nullptr},
            {"complete_domain",true},{"source",prices.source}});
        if(weight==0)first_key=prepared.key;
        if(weight==1)second_key=prepared.key;
        if(!result.cut.valid || cut.violation>result.cut.violation) {
            result.cut=std::move(cut);winner=std::move(prepared);new_winner=cached==nullptr;
            winning_weight=weight;result.diagnostics["best_line_weight"]=weight;
        }
        return true;
    };
    auto rollback=[&]() {
        result.cut=before;result.diagnostics["best_line_weight"]=nullptr;
        result.diagnostics["atomic_rollback"]=true;return finish(failure);
    };
    if(!evaluate(0,first,endpoint_index<profiles_.size()?&profiles_[endpoint_index]:nullptr))return rollback();
    if(!evaluate(1,candidate,nullptr))return rollback();
    const bool identical_endpoints=first_key==second_key;
    auto mixture=[&](double weight) {
        QueueResourcePrices mixed;mixed.source="queue_price_line_search";
        auto blend=[&](const std::vector<double>& a,const std::vector<double>& b) {
            std::vector<double> out(a.size());
            for(std::size_t i=0;i<a.size();++i)
                out[i]=static_cast<double>((1-static_cast<long double>(weight))*a[i]+
                    static_cast<long double>(weight)*b[i]);
            return out;
        };
        mixed.locker=blend(first.locker,candidate.locker);
        mixed.airline=blend(first.airline,candidate.airline);
        mixed.balance=blend(first.balance,candidate.balance);
        std::map<std::tuple<int,int,int,int>,long double> gates;
        for(const auto& g:first.branch_gates)
            gates[{g.head,g.origin,g.second,g.arc}]+=(1-static_cast<long double>(weight))*g.price;
        for(const auto& g:candidate.branch_gates)
            gates[{g.head,g.origin,g.second,g.arc}]+=static_cast<long double>(weight)*g.price;
        for(const auto& [position,price]:gates) {
            const auto [head,origin,second,arc]=position;
            mixed.branch_gates.push_back({head,origin,second,arc,static_cast<double>(price)});
        }
        return mixed;
    };
    while(!identical_endpoints && observed.size()<max_evaluations) {
        std::sort(observed.begin(),observed.end(),[](const Observation& a,const Observation& b){return a.weight<b.weight;});
        const auto best=std::max_element(observed.begin(),observed.end(),
            [](const Observation& a,const Observation& b){return a.value<b.value;});
        const auto pos=static_cast<std::size_t>(best-observed.begin());
        const double left=pos>0?observed[pos].weight-observed[pos-1].weight:-1;
        const double right=pos+1<observed.size()?observed[pos+1].weight-observed[pos].weight:-1;
        const std::size_t lo=left>=right?pos-1:pos;
        const double weight=(observed[lo].weight+observed[lo+1].weight)/2;
        if(!evaluate(weight,mixture(weight),nullptr))return rollback();
    }
    if(new_winner) {
        // Keep the old strongest resource endpoint while evaluating and committing
        // the sole winner. A saturated two-profile pool can return a valid row
        // without retaining it, rather than evicting its protected endpoint.
        std::size_t existing=profiles_.size();
        for(std::size_t i=0;i<profiles_.size();++i)if(profiles_[i].key==winner.key){existing=i;break;}
        if(existing<profiles_.size())result.cut=separate_profile(existing,design,theta_value,theta);
        else {
            winner.id=next_id_++;winner.certificate["profile_id"]=winner.id;
            winner.certificate["price_search"]={{"line_weight",winning_weight},
                {"endpoint_a_source",first.source},{"endpoint_b_source",candidate.source}};
            std::size_t evict=profiles_.size();
            if(profiles_.size()>=max_profiles_)for(std::size_t i=1;i<profiles_.size();++i)
                if(i!=endpoint_index){evict=i;break;}
            if(profiles_.size()<max_profiles_ || evict<profiles_.size()) {
                if(evict<profiles_.size())profiles_.erase(profiles_.begin()+evict);
                profiles_.push_back(std::move(winner));
                metadata_["profiles"]=profiles_.size();
                metadata_["price_profiles_added"]=metadata_.at("price_profiles_added").get<int>()+1;
                metadata_["last_profile_id"]=profiles_.back().id;
                result.diagnostics["profile_added"]=true;
                result.cut=separate_profile(profiles_.size()-1,design,theta_value,theta);
            } else {
                winner.certificate["retained"]=false;
                result.cut=separate_prepared(winner,profiles_.size(),design,theta_value,theta);
            }
        }
    }
    result.cut.certificate["price_search"]={{"line_weight",result.diagnostics["best_line_weight"]},
        {"endpoint_a_source",first.source},{"endpoint_b_source",candidate.source},
        {"evaluation_count",evaluations.size()},{"complete_scans",complete_scans}};
    return finish(identical_endpoints?"identical_endpoints":"complete_price_search");
}

} // namespace mlnd
