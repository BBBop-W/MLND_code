#include "mlnd/airline_resources.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
double upper_signed(long double value) {
    if (value == 0) return 0;
    return std::nextafter(static_cast<double>(value+1e-12L*(1+std::abs(value))),infinity);
}
double lower_nonnegative(long double value) {
    if (value <= 0) return 0;
    return std::max(0.0,std::nextafter(static_cast<double>(value-1e-12L*(1+value)),-infinity));
}
struct Branch {
    int head, origin, second;
    double rate;
    std::vector<int> paths;
};
std::vector<Branch> branches(const Input& d,const PathCollection& paths) {
    if (!d.preprocessed_demand || !d.origin_assignment() || d.flow_policy!="aggregate" ||
        d.first_leg_consistency || d.route_mode!="direct" || d.has_candidate_routes ||
        d.max_orders<1 || d.max_orders>2 || !paths.complete || !paths.support_complete)
        throw std::invalid_argument("Airline resources require complete direct fixed origin tasks and aggregate flow");
    if (d.n_l<1 || d.n_od<1 || d.n_h<0 ||
        d.single_rates.size()!=static_cast<std::size_t>(d.n_od) ||
        d.pair_rates.rows!=d.n_od || d.pair_rates.cols!=d.n_od ||
        !(d.fleet_cost>0) || !std::isfinite(d.fleet_cost))
        throw std::invalid_argument("Invalid fixed task dimensions or fleet cost");
    std::vector<Branch> out;
    std::map<std::tuple<int,int,int>,int> index;
    for (int r=0;r<d.n_od;++r) for (int l=0;l<d.n_l;++l) for (int s=-1;s<d.n_od;++s) {
        const double rate=s<0?d.single_rates[r]:d.pair_rates(r,s);
        if (!std::isfinite(rate) || rate<0 || (d.max_orders==1 && s>=0 && rate!=0))
            throw std::invalid_argument("Invalid fixed task rate");
        if (rate>0) {
            index.emplace(std::make_tuple(r,l,s),static_cast<int>(out.size()));
            out.push_back({r,l,s,rate,{}});
        }
    }
    for (std::size_t j=0;j<paths.service_paths.size();++j) {
        const auto& p=paths.service_paths[j];
        if (p.first_od<0 || p.first_od>=d.n_od || p.second_od<-1 || p.second_od>=d.n_od ||
            p.b[0]<0 || p.b[0]>=d.n_l || p.b[1]<0 || p.b[1]>=d.n_l)
            throw std::invalid_argument("Malformed complete service path");
        const auto found=index.find({p.first_od,p.b[0],p.second_od});
        if (found!=index.end()) out[found->second].paths.push_back(static_cast<int>(j));
    }
    return out;
}
template<class Path>
std::map<int,int> arc_counts(const Input& d,const Path& p) {
    std::map<int,int> counts;
    for (auto [u,v,n]:p.arc_counts) {
        if (u<0 || v<0 || u>=d.n_nodes() || v>=d.n_nodes() || n<=0 || !d.allowed_links(u,v))
            throw std::invalid_argument("Malformed physical arc in airline resources");
        counts[u*d.n_nodes()+v]+=n;
    }
    return counts;
}
void nonnegative_costs(const Input& d) {
    for (const auto* values:{&d.base_cost,&d.link_cost.values,&d.slot_cost.values})
        for (double v:*values) if (!std::isfinite(v) || v<0)
            throw std::invalid_argument("Asset canonicalization needs nonnegative opening costs");
}
}

AirlineConnectivityCuts build_airline_connectivity(const Input& d,const PathCollection& paths) {
    const double started=wall_time();
    const auto groups=branches(d,paths);
    nonnegative_costs(d);
    const DesignLayout layout(d);
    const int N=d.n_nodes(),L=d.n_l;
    AirlineConnectivityCuts result;
    std::set<std::vector<std::pair<int,double>>> seen;
    std::map<std::string,int> counts;
    auto add=[&](const std::string& name,Coefficients row) {
        std::sort(row.begin(),row.end());
        if (!seen.insert(row).second) return;
        result.rows.push_back({name+"_"+std::to_string(result.rows.size()),'L',0,std::move(row)});++counts[name];
    };
    for (int h=L;h<N;++h) {
        Coefficients incoming{{layout.facility+h,1}},outgoing=incoming;
        for (int u=0;u<N;++u) {
            if (d.allowed_links(u,h)) incoming.emplace_back(layout.airline+u*N+h,-1);
            if (d.allowed_links(h,u)) outgoing.emplace_back(layout.airline+h*N+u,-1);
        }
        add("airline_locker_in",std::move(incoming));
        add("airline_locker_out",std::move(outgoing));
    }
    std::vector<std::set<int>> supporting_assignments(N*N);
    std::size_t scanned=0;
    for (const auto& b:groups) {
        const int alpha=layout.assignment_index(d.task_index(b.head,b.second),b.origin);
        if (b.paths.empty()) {add("airline_missing_task",{{alpha,1}});continue;}
        // Each role set intersects every complete path in this task branch.
        // The return role ranges over ALL physical return bases.
        std::vector<std::set<int>> roles(b.second<0?3:4);
        for (int j:b.paths) {
            const auto& p=paths.service_paths[j];++scanned;
            if (p.pickups.size()!=(b.second<0?1u:2u) || p.deliveries.size()!=p.pickups.size())
                throw std::invalid_argument("Malformed direct task roles");
            roles[0].insert(b.origin*N+L+p.pickups.front());
            roles[1].insert((L+p.pickups[0])*N+L+p.deliveries[0]);
            roles[2].insert((L+p.deliveries.back())*N+p.b[1]);
            if (b.second>=0) roles[3].insert((L+p.pickups[1])*N+L+p.deliveries[1]);
            const auto arcs=arc_counts(d,p);
            for (auto [e,n]:arcs) { (void)n; supporting_assignments[e].insert(alpha); }
            // Validate role arcs against this path, not only input allowance.
            for (int e:{b.origin*N+L+p.pickups.front(),
                        (L+p.pickups[0])*N+L+p.deliveries[0],
                        (L+p.deliveries.back())*N+p.b[1]})
                if (!arcs.count(e)) throw std::invalid_argument("A role arc is absent from its physical path");
            if (b.second>=0 && !arcs.count((L+p.pickups[1])*N+L+p.deliveries[1]))
                throw std::invalid_argument("Second loaded role absent from path");
        }
        for (std::size_t role=0;role<roles.size();++role) {
            Coefficients row{{alpha,1}};
            for (int e:roles[role]) row.emplace_back(layout.airline+e,-1);
            const char* names[]{"airline_task_departure","airline_task_loaded_first","airline_task_return","airline_task_loaded_second"};
            add(names[role],std::move(row));
        }
    }
    for (int u=0;u<N;++u) for (int v=0;v<N;++v) {
        if (!d.allowed_links(u,v) || (u<L && v<L)) continue; // Empty relocation is not owned by a head.
        const int e=u*N+v;
        Coefficients row{{layout.airline+e,1}};
        for (int alpha:supporting_assignments[e]) row.emplace_back(alpha,-1);
        add("airline_task_owner",std::move(row));
    }
    result.metadata={{"enabled",true},{"source","complete_positive_task_routes_all_returns"},
        {"validity_scope","optimality_preserving_asset_and_assignment_canonicalization"},
        {"assignment_scope","task"},{"tasks",d.task_count()},
        {"preserves_all_original_design_encodings",false},{"master_added_variables",0},
        {"rows",result.rows.size()},{"row_count",result.rows.size()},{"row_types",counts},{"positive_branches",groups.size()},
        {"paths_scanned",scanned},{"all_return_bases",true},{"seconds",wall_time()-started}};
    return result;
}

AirlineResourceRelaxation::AirlineResourceRelaxation(const Input& d,const PathCollection& paths,
        const SolverOptions& options,bool balance_relocation) {
    const double started=wall_time();
    const auto bs=branches(d,paths);
    const DesignLayout layout(d);
    const int L=d.n_l,H=d.n_h,N=d.n_nodes(),G=d.task_count()*L;
    lockers_=H;nodes_=N;
    design_count_=layout.count;groups_=G;bases_=L;path_offset_=G+2*L;
    fleet_cost_=d.fleet_cost;
    courier_cost_=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    if (!std::isfinite(courier_cost_) || courier_cost_<0)
        throw std::invalid_argument("Invalid courier baseline");
    numerical_margin_=1e-8*(1+courier_cost_);
    std::size_t positive_paths=0;for (const auto& b:bs) positive_paths+=b.paths.size();
    const std::size_t empty_paths=balance_relocation?paths.empty_paths.size():0;
    if (positive_paths+empty_paths>static_cast<std::size_t>(std::numeric_limits<int>::max()-path_offset_))
        throw std::length_error("Route resource LP exceeds integer index range");
    empty_offset_=path_offset_+static_cast<int>(positive_paths);
    primal_count_=empty_offset_+static_cast<int>(empty_paths);
    primal_objective_.assign(primal_count_,0);
    for (int l=0;l<L;++l) primal_objective_[G+L+l]=-fleet_cost_;
    auto add=[&](double constant,Coefficients rhs,Coefficients coeff,bool equality=false) {
        columns_.push_back({constant,equality,std::move(rhs),std::move(coeff)});
        return static_cast<int>(columns_.size()-1);
    };
    assignment_prices_.resize(G);base_prices_.resize(L);tangent_prices_.resize(L);
    for (int r=0;r<d.task_count();++r) for (int l=0;l<L;++l) {
        const int g=r*L+l,a=layout.assignment_index(r,l);
        assignment_prices_[g]=add(0,{{a,1}},{{g,1}});
        add(0,{},{{g,1},{G+l,-1}});
        add(1,{{a,-1}},{{G+l,1},{g,-1}});
    }
    for (int l=0;l<L;++l) base_prices_[l]=add(0,{{layout.facility+l,1}},{{G+l,1}});
    std::vector<int> locker_rows(H),arc_rows(N*N,-1);
    for (int i=0;i<H;++i) {
        Coefficients rhs;
        for (int s=0;s<d.n_levels;++s) rhs.emplace_back(layout.capacity+i*d.n_levels+s,d.occupancy_limit(i,s));
        locker_rows[i]=add(0,std::move(rhs),{});
    }
    const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    if (balance_relocation && (!std::isfinite(total) || total<0))
        throw std::invalid_argument("Invalid original demand bound for empty relocation");
    if (!options.task_airline_upper.empty() && options.task_airline_upper.size()!=static_cast<std::size_t>(N*N))
        throw std::invalid_argument("Prepared task arc bounds have incorrect dimensions");
    for (int u=0;u<N;++u) for (int v=0;v<N;++v) if (d.allowed_links(u,v)) {
        double upper=u<L&&v<L ? total*std::max(0,L-1) :
            (u>=L&&v>=L ? total*(options.tight_airline_bounds?1:d.max_orders) : total);
        if (!options.task_airline_upper.empty()) upper=std::min(upper,options.task_airline_upper[u*N+v]);
        if (!std::isfinite(upper) || upper<0) throw std::invalid_argument("Invalid physical arc bound");
        arc_rows[u*N+v]=add(0,{{layout.airline+u*N+v,upper}},{});
    }
    locker_prices_=locker_rows;arc_prices_=arc_rows;
    // Every physical path has one departure and one return, so the last
    // base balance is the negative sum of the first L-1 equalities.
    if (balance_relocation) for (int l=0;l<L-1;++l)
        balance_prices_.push_back(add(0,{}, {},true));
    std::size_t gates=0,missing=0;int q=path_offset_;
    branch_prices_.reserve(bs.size());path_branches_.reserve(positive_paths);
    for (std::size_t k=0;k<bs.size();++k) {
        const auto& b=bs[k];const int g=d.task_index(b.head,b.second)*L+b.origin;
        const int balance=add(0,{},{{g,-1}},true);branch_prices_.push_back(balance);
        if (b.paths.empty()) ++missing;
        std::map<int,int> gate_rows;
        for (int j:b.paths) {
            const auto& p=paths.service_paths[j];
            if (!std::isfinite(p.cost) || p.cost<0 || !std::isfinite(p.time) || p.time<0)
                throw std::invalid_argument("Invalid path cost/time in airline resource LP");
            long double reward=0;
            for (auto [r,count]:p.served_counts) {
                if (r<0 || r>=d.n_od || count<=0) throw std::invalid_argument("Invalid task occurrence count");
                reward+=static_cast<long double>(count)*d.ground_cost[r];
            }
            primal_objective_[q]=upper_signed(static_cast<long double>(b.rate)*
                (reward-p.cost-static_cast<long double>(fleet_cost_)*p.time));
            columns_[balance].primal_coefficients.emplace_back(q,1);
            if (balance_relocation && p.b[0]!=p.b[1]) {
                if (p.b[0]<L-1) columns_[balance_prices_[p.b[0]]].primal_coefficients.emplace_back(q,b.rate);
                if (p.b[1]<L-1) columns_[balance_prices_[p.b[1]]].primal_coefficients.emplace_back(q,-b.rate);
            }
            for (auto [i,time]:p.slot_times) {
                if (i<0 || i>=H || !std::isfinite(time) || time<0)
                    throw std::invalid_argument("Invalid path slot time");
                const double use=lower_nonnegative(static_cast<long double>(b.rate)*time);
                if (use>0) columns_[locker_rows[i]].primal_coefficients.emplace_back(q,use);
            }
            for (auto [e,count]:arc_counts(d,p)) {
                const double use=lower_nonnegative(static_cast<long double>(b.rate)*count);
                if (use>0) columns_[arc_rows[e]].primal_coefficients.emplace_back(q,use);
                auto it=gate_rows.find(e);
                if (it==gate_rows.end()) {
                    const int row=add(0,{{layout.airline+e,1}},{});
                    it=gate_rows.emplace(e,row).first;++gates;
                    normalized_gate_prices_.push_back({row,{b.head,b.origin,b.second,e,0}});
                }
                // Presence, not multiplicity: an AA path can traverse e twice.
                columns_[it->second].primal_coefficients.emplace_back(q,1);
            }
            path_branches_.push_back(static_cast<int>(k));++q;
        }
    }
    std::vector<int> relocation_path_ids;
    for (std::size_t j=0;j<empty_paths;++j) {
        const auto& p=paths.empty_paths[j];
        if (p.b[0]<0 || p.b[0]>=L || p.b[1]<0 || p.b[1]>=L || p.b[0]==p.b[1] ||
            !std::isfinite(p.cost) || p.cost<0 || !std::isfinite(p.time) || p.time<0)
            throw std::invalid_argument("Malformed direct empty relocation path");
        const auto arcs=arc_counts(d,p);
        const int e=p.b[0]*N+p.b[1],v=empty_offset_+static_cast<int>(j);
        if (arcs.size()!=1 || arcs.begin()->first!=e || arcs.begin()->second!=1)
            throw std::invalid_argument("Airline relocation relaxation requires direct base-to-base paths");
        primal_objective_[v]=upper_signed(-static_cast<long double>(p.cost)-
            static_cast<long double>(fleet_cost_)*p.time);
        if (p.b[0]<L-1) columns_[balance_prices_[p.b[0]]].primal_coefficients.emplace_back(v,1);
        if (p.b[1]<L-1) columns_[balance_prices_[p.b[1]]].primal_coefficients.emplace_back(v,-1);
        columns_[arc_rows[e]].primal_coefficients.emplace_back(v,1);
        // The original empty-flow variable is bounded by total demand, and
        // its physical arc must be open. This gate is valid at every binary
        // design, including redundant empty-flow circulations.
        empty_gate_prices_.push_back(add(0,{{layout.airline+e,total}},{{v,1}}));
        relocation_path_ids.push_back(p.id);
    }
    const std::vector<double> anchors{0,.25,.5,.75,.9,.95,.97,.98,.99};
    for (int l=0;l<L;++l) for (double t:anchors) {
        const long double a=1/((1-static_cast<long double>(t))*(1-static_cast<long double>(t)));
        tangent_prices_[l].push_back(add(upper_signed(a*t*t),{},
            {{G+l,lower_nonnegative(a)},{G+L+l,-1}}));
    }
    LinearModel dual;
    std::vector<Coefficients> stationarity(primal_count_);
    std::size_t nonzeros=0;
    for (std::size_t j=0;j<columns_.size();++j) {
        dual.variable("route_resource_price_"+std::to_string(j),columns_[j].equality?-IloInfinity:0,IloInfinity);
        for (auto [v,c]:columns_[j].primal_coefficients) {stationarity[v].emplace_back(static_cast<int>(j),c);++nonzeros;}
    }
    for (int v=0;v<primal_count_;++v)
        dual.row("route_resource_column_"+std::to_string(v),'G',primal_objective_[v],std::move(stationarity[v]));
    SolverOptions quiet=options;quiet.log=false;
    solver_=std::make_unique<NativeSolver>(dual,quiet,"airline_resource_dual");
    metadata_={{"enabled",true},{"source","complete_route_normalized_fixed_task_resource_lp"},
        {"assignment_scope","task"},{"tasks",d.task_count()},
        {"catalogue_complete",true},{"support_complete",true},{"all_return_bases",true},
        {"master_added_variables",0},{"resource_primal_variables",primal_count_},
        {"resource_primal_rows",columns_.size()},{"dual_variables",columns_.size()},
        {"dual_constraints",primal_count_},{"dual_nonzeros",nonzeros},
        {"physical_catalogue_service_paths",paths.service_paths.size()},{"positive_rate_path_variables",positive_paths},
        {"positive_task_branches",bs.size()},{"missing_positive_branches",missing},
        {"normalized_branch_arc_gates",gates},{"idle_tangent_anchors",anchors},
        {"dropped_empty_flow_and_base_balance",!balance_relocation},
        {"balance_relocation",balance_relocation},{"base_balance_rows",balance_prices_.size()},
        {"empty_relocation_variables",empty_paths},{"empty_gate_rows",empty_gate_prices_.size()},
        {"base_balance_price_indices",balance_prices_},{"empty_gate_price_indices",empty_gate_prices_},
        {"service_primal_begin",path_offset_},{"empty_primal_begin",empty_offset_},
        {"relocation_path_ids",relocation_path_ids},{"empty_flow_upper_bound",total},
        {"negative_path_savings_retained",true},
        {"validity_scope","all_original_integer_designs_and_their_convex_hull"},
        {"certificate","explicit_dual_repair_and_all_columns_checked"},
        {"numerical_margin",numerical_margin_},{"build_seconds",wall_time()-started}};
}

QueueResourcePrices AirlineResourceRelaxation::queue_prices(const std::vector<double>& values) const {
    if(values.size()!=columns_.size())
        throw std::invalid_argument("Queue resource price vector has incorrect dimensions");
    for(std::size_t j=0;j<values.size();++j)
        if(!std::isfinite(values[j]) || (!columns_[j].equality && values[j]<0))
            throw std::invalid_argument("Queue resource price vector has an invalid sign or value");
    QueueResourcePrices out;
    out.source="airline_resource_verified_prices";
    out.locker.assign(lockers_,0);out.airline.assign(nodes_*nodes_,0);out.balance.assign(bases_,0);
    for(int i=0;i<lockers_;++i)out.locker[i]=values[locker_prices_[i]];
    for(int e=0;e<nodes_*nodes_;++e)if(arc_prices_[e]>=0)out.airline[e]=values[arc_prices_[e]];
    for(std::size_t l=0;l<balance_prices_.size();++l)out.balance[l]=values[balance_prices_[l]];
    for(const auto& [index,descriptor]:normalized_gate_prices_)if(values[index]>0) {
        auto price=descriptor;price.price=values[index];out.branch_gates.push_back(price);
    }
    return out;
}

TaskSavingsCut AirlineResourceRelaxation::separate(const std::vector<double>& design,
        double theta_value,int theta,double seconds) {
    const double started=wall_time();
    if (design.size()<static_cast<std::size_t>(design_count_) || theta<design_count_ ||
        !std::isfinite(theta_value) || !std::isfinite(seconds))
        throw std::invalid_argument("Invalid airline resource separation point");
    for (int j=0;j<design_count_;++j)
        if (!std::isfinite(design[j]) || design[j]<-1e-8 || design[j]>1+1e-8)
            throw std::invalid_argument("Airline resource separation needs a unit-box design");
    const double update_started=wall_time();
    if (solver_->variables.getSize()!=static_cast<IloInt>(columns_.size()))
        throw std::logic_error("Route-resource objective and native variable dimensions disagree");
    IloNumArray objective_values(solver_->env,solver_->variables.getSize());
    for (std::size_t j=0;j<columns_.size();++j) {
        long double rhs=columns_[j].constant;
        for (auto [v,c]:columns_[j].design_rhs) rhs+=static_cast<long double>(c)*std::clamp(design[v],0.0,1.0);
        objective_values[static_cast<IloInt>(j)]=static_cast<double>(rhs);
    }
    // One Concert update preserves the exact coefficient vector while
    // avoiding repeated extracted-objective rebuilds after the first solve.
    try {solver_->objective.setLinearCoefs(solver_->variables,objective_values);}
    catch (...) {objective_values.end();throw;}
    objective_values.end();
    metadata_["last_objective_update_seconds"]=wall_time()-update_started;
    metadata_["objective_update_method"]="one_setLinearCoefs_batch";
    metadata_["objective_update_coefficients"]=columns_.size();
    TaskSavingsCut cut;SolveOutcome solved;
    const double budget=std::max(0.0,seconds-(wall_time()-started));
    metadata_["last_solver_budget_seconds"]=budget;
    if (budget>0) {
        try {solved=solver_->optimize(budget);cut.status=solved.status;}
        catch (const IloException& e) {cut.status=std::string("solver_exception: ")+e.getMessage();}
    } else cut.status="budget_exhausted";
    cut.optimal=solved.optimal;cut.solver_seconds=solved.seconds;cut.solver_objective=solved.objective;
    cut.fallback=!solved.has_solution || solved.values.size()!=columns_.size() ||
        !std::all_of(solved.values.begin(),solved.values.end(),[](double v){return std::isfinite(v);});
    cut.dual_prices.assign(columns_.size(),0);
    if (!cut.fallback) for (std::size_t j=0;j<columns_.size();++j)
        cut.dual_prices[j]=columns_[j].equality?solved.values[j]:std::max(0.0,solved.values[j]);
    auto activities=[&] {
        std::vector<long double> a(primal_count_,0);
        for (std::size_t j=0;j<columns_.size();++j)
            for (auto [v,c]:columns_[j].primal_coefficients) a[v]+=static_cast<long double>(c)*cut.dual_prices[j];
        return a;
    };
    auto activity=activities();
    for (int v=0;v<primal_count_;++v) cut.max_dual_violation_before=std::max(
        cut.max_dual_violation_before,static_cast<double>(primal_objective_[v]-activity[v]));
    // Repair w first; reducing tangent prices can affect rho, but not paths.
    for (int l=0;l<bases_;++l) {
        long double sum=0;for (int j:tangent_prices_[l]) sum+=cut.dual_prices[j];
        const double target=lower_nonnegative(fleet_cost_);
        if (sum>target) for (int j:tangent_prices_[l]) cut.dual_prices[j]=std::max(0.0,
            std::nextafter(static_cast<double>(cut.dual_prices[j]*target/sum),0.0));
    }
    activity=activities();
    // A free base-balance price may violate an empty-flow column. Its own
    // nonnegative gate price raises that column without weakening any other
    // column, so repair all empty columns before the branch-price repair.
    for (int v=empty_offset_;v<primal_count_;++v) if (activity[v]<primal_objective_[v]) {
        const int j=empty_gate_prices_[v-empty_offset_];
        cut.dual_prices[j]=upper_signed(static_cast<long double>(cut.dual_prices[j])+
            primal_objective_[v]-activity[v]);
    }
    activity=activities();
    std::vector<long double> branch_deficit(branch_prices_.size(),0);
    for (int q=path_offset_;q<empty_offset_;++q) {
        auto& deficit=branch_deficit[path_branches_[q-path_offset_]];
        deficit=std::max(deficit,static_cast<long double>(primal_objective_[q])-activity[q]);
    }
    // An unrestricted equality price raises every q-column in its branch.
    for (std::size_t k=0;k<branch_prices_.size();++k) if (branch_deficit[k]>0) {
        const int j=branch_prices_[k];cut.dual_prices[j]=upper_signed(
            static_cast<long double>(cut.dual_prices[j])+branch_deficit[k]);
    }
    activity=activities();
    for (int g=0;g<groups_;++g) if (activity[g]<primal_objective_[g]) {
        const int j=assignment_prices_[g];cut.dual_prices[j]=upper_signed(
            static_cast<long double>(cut.dual_prices[j])+primal_objective_[g]-activity[g]);
    }
    activity=activities();
    for (int l=0;l<bases_;++l) if (activity[groups_+l]<0) {
        const int j=base_prices_[l];cut.dual_prices[j]=upper_signed(
            static_cast<long double>(cut.dual_prices[j])-activity[groups_+l]);
    }
    activity=activities();cut.valid=true;
    for (int v=0;v<primal_count_;++v) {
        if (!std::isfinite(activity[v])) cut.valid=false;
        cut.max_dual_violation_after=std::max(cut.max_dual_violation_after,
            static_cast<double>(primal_objective_[v]-activity[v]));
    }
    for (std::size_t j=0;j<columns_.size();++j)
        if (!std::isfinite(cut.dual_prices[j]) || (!columns_[j].equality && cut.dual_prices[j]<0)) cut.valid=false;
    cut.valid=cut.valid && cut.max_dual_violation_after<=0;
    if (!cut.valid) {cut.seconds=wall_time()-started;return cut;}
    long double constant=0;std::map<int,long double> coefficients;
    for (std::size_t j=0;j<columns_.size();++j) {
        constant+=static_cast<long double>(columns_[j].constant)*cut.dual_prices[j];
        for (auto [v,c]:columns_[j].design_rhs) coefficients[v]+=static_cast<long double>(c)*cut.dual_prices[j];
    }
    const double constant_upper=upper_signed(constant);
    cut.row={"airline_route_resource_cut",'G',courier_cost_-numerical_margin_-constant_upper,{{theta,1}}};
    long double saving=constant_upper;
    std::ostringstream key;key<<std::scientific<<std::setprecision(10)<<cut.row.rhs<<';';
    for (auto [v,c]:coefficients) if (c!=0) {
        const double rounded=upper_signed(c);
        if (!std::isfinite(rounded)) {cut.valid=false;cut.seconds=wall_time()-started;return cut;}
        cut.row.coefficients.emplace_back(v,rounded);saving+=static_cast<long double>(rounded)*design[v];
        key<<v<<':'<<rounded<<';';
    }
    cut.key=key.str();cut.dual_savings=static_cast<double>(saving);
    cut.violation=courier_cost_-numerical_margin_-cut.dual_savings-theta_value;
    cut.violated=cut.violation>1e-7*(1+std::abs(courier_cost_)+std::abs(theta_value));
    cut.seconds=wall_time()-started;return cut;
}

} // namespace mlnd
