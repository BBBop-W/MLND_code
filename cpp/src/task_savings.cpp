#include "mlnd/task_savings.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mlnd {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
double upper_nonnegative(long double x) {
    if (!(x > 0)) return 0;
    return std::nextafter(static_cast<double>(x + 1e-12L*(1+x)), infinity);
}
double lower_nonnegative(long double x) {
    if (!(x > 0)) return 0;
    return std::max(0.0,std::nextafter(static_cast<double>(x-1e-12L*(1+x)),-infinity));
}
void require_domain(const Input& d, const PathCollection& paths) {
    if (!d.preprocessed_demand || !d.origin_assignment() || d.flow_policy != "aggregate" ||
        d.first_leg_consistency || d.route_mode != "direct" || d.has_candidate_routes ||
        d.max_orders < 1 || d.max_orders > 2 || !paths.complete || !paths.support_complete)
        throw std::invalid_argument("Task savings require complete preprocessed task-assignment aggregate paths");
    if (!(d.fleet_cost > 0) || !std::isfinite(d.fleet_cost) ||
        d.single_rates.size()!=static_cast<std::size_t>(d.n_od) ||
        d.pair_rates.rows!=d.n_od || d.pair_rates.cols!=d.n_od)
        throw std::invalid_argument("Invalid fixed generalized-task savings data");
}
}

TaskSavingsData task_savings_data(const Input& d,const PathCollection& paths) {
    require_domain(d,paths);
    const double started=wall_time();
    const int R=d.n_od,Q=d.task_count(),L=d.n_l,H=d.n_h,G=Q*L;
    TaskSavingsData out;
    out.heads=Q;out.bases=L;out.lockers=H;out.levels=d.n_levels;
    out.fleet_cost=d.fleet_cost;
    out.courier_cost=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    out.numerical_margin=1e-8*(1+out.courier_cost);
    out.saving.assign(G,0);out.signed_saving.assign(G,0);out.chord_coefficient.assign(G,0);
    out.total_load.assign(G,0);out.locker_load.assign(G,std::vector<double>(H,0));
    out.supported.assign(G,true);
    std::vector<double> mincost(G,infinity),mintotal(G,infinity);
    std::vector<std::vector<double>> minload(G,std::vector<double>(H,infinity));
    std::vector<int> pathcount(G,0);
    for (const auto& p:paths.service_paths) {
        if (p.first_od<0 || p.first_od>=R || p.second_od<-1 || p.second_od>=R ||
            (d.max_orders==1 && p.second_od>=0) ||
            p.b[0]<0 || p.b[0]>=L || p.b[1]<0 || p.b[1]>=L ||
            !std::isfinite(p.cost) || p.cost<0 || !std::isfinite(p.time) || p.time<0)
            throw std::invalid_argument("Invalid path in full-task savings catalogue");
        const int k=d.task_index(p.first_od,p.second_od)*L+p.b[0];
        std::vector<double> load(H,0);
        for (auto [i,t]:p.slot_times) {
            if (i<0 || i>=H || !std::isfinite(t) || t<0)
                throw std::invalid_argument("Invalid task slot occupancy");
            load[i]+=t;
        }
        mincost[k]=std::min(mincost[k],lower_nonnegative(
            static_cast<long double>(p.cost)+static_cast<long double>(d.fleet_cost)*p.time));
        mintotal[k]=std::min(mintotal[k],lower_nonnegative(std::accumulate(load.begin(),load.end(),0.0L)));
        for (int i=0;i<H;++i) minload[k][i]=std::min(minload[k][i],lower_nonnegative(load[i]));
        ++pathcount[k];
    }
    int missing_groups=0,positive_branches=0;
    for (int q=0;q<Q;++q) for (int l=0;l<L;++l) {
        const int g=q*L+l,r=d.task_first(q),second=d.task_second(q);
        const double rate=d.task_rate(q);
        if (!std::isfinite(rate) || rate<0) throw std::invalid_argument("Invalid fixed task rate");
        if (rate==0) continue;
        ++positive_branches;
        // Each task owns its assignment and service fraction. An absent path
        // fixes only this task/base service to zero; siblings remain usable.
        if (!pathcount[g]) {out.supported[g]=false;++missing_groups;continue;}
        const double reward=d.ground_cost[r]+(second<0?0:d.ground_cost[second]);
        const long double saving=static_cast<long double>(rate)*(reward-mincost[g]);
        out.signed_saving[g]=static_cast<double>(saving);
        out.saving[g]=upper_nonnegative(std::max(0.0L,saving));
        out.total_load[g]=lower_nonnegative(static_cast<long double>(rate)*mintotal[g]);
        for (int i=0;i<H;++i)
            out.locker_load[g][i]=lower_nonnegative(static_cast<long double>(rate)*minload[g][i]);
    }
    std::vector<double> smax(L,0),eta(L,0);
    for (int l=0;l<L;++l) {
        long double sum=0;
        for (int r=0;r<Q;++r) sum+=out.saving[r*L+l];
        smax[l]=upper_nonnegative(sum);
        if (smax[l]>d.fleet_cost) {
            const long double a=1-std::sqrt(static_cast<long double>(d.fleet_cost)/smax[l]);
            eta[l]=std::min(1.0,upper_nonnegative(a*a));
        }
        for (int r=0;r<Q;++r) out.chord_coefficient[r*L+l]=
            upper_nonnegative(static_cast<long double>(eta[l])*out.saving[r*L+l]);
    }
    out.metadata={{"enabled",true},{"source","complete_physical_task_catalogue"},
        {"catalogue_complete",true},{"service_paths",paths.service_paths.size()},
        {"heads",Q},{"tasks",Q},{"raw_ods",R},{"bases",L},{"groups",G},{"positive_branches",positive_branches},
        {"unsupported_groups",missing_groups},{"return_scope","all_return_bases"},
        {"assignment_scope","task"},{"saving",out.saving},{"signed_saving",out.signed_saving},
        {"chord_coefficient",out.chord_coefficient},{"maximum_base_saving",smax},
        {"queue_chord_factor",eta},{"total_load",out.total_load},{"locker_load",out.locker_load},
        {"supported",out.supported},{"courier_cost",out.courier_cost},
        {"numerical_margin",out.numerical_margin},{"preprocessing_seconds",wall_time()-started},
        {"validity_scope","all_original_integer_designs"},{"master_added_variables",0}};
    return out;
}

Json add_task_savings_bound(const TaskSavingsData& data,Formulation& master,int theta) {
    if (theta<master.layout.count || theta>=static_cast<int>(master.matrix.variables.size()))
        throw std::invalid_argument("Task savings bound needs a master operating-cost variable");
    Coefficients row{{theta,1}};
    for (int r=0;r<data.heads;++r) for (int l=0;l<data.bases;++l) {
        const double c=data.chord_coefficient[r*data.bases+l];
        if (c>0) row.emplace_back(master.layout.assignment_index(r,l),c);
    }
    master.matrix.row("full_task_assignment_savings_bound",'G',
        data.courier_cost-data.numerical_margin,std::move(row));
    Json meta=data.metadata;meta["added_rows"]=1;meta["added_variables"]=0;
    return meta;
}

TaskResourceSavingsRelaxation::TaskResourceSavingsRelaxation(const Input& d,
        TaskSavingsData data,const SolverOptions& options):data_(std::move(data)) {
    const double started=wall_time();
    const DesignLayout layout(d);
    design_count_=layout.count;
    const int R=data_.heads,L=data_.bases,H=data_.lockers,G=R*L;
    if (R!=d.assignment_entities() || L!=d.n_l || H!=d.n_h || data_.levels!=d.n_levels ||
        data_.saving.size()!=static_cast<std::size_t>(G))
        throw std::invalid_argument("Task resource relaxation data dimension mismatch");
    // Primal max saving: v, rho, w >=0; w pays a supporting-line lower bound
    // of required idle fleet. Every primal row is <= affine design RHS.
    primal_count_=G+2*L;
    primal_objective_.assign(primal_count_,0);
    for (int g=0;g<G;++g) primal_objective_[g]=data_.saving[g];
    for (int l=0;l<L;++l) primal_objective_[G+L+l]=-data_.fleet_cost;
    auto add=[&](double constant,Coefficients rhs,Coefficients coeff) {
        columns_.push_back({constant,std::move(rhs),std::move(coeff)});
        return static_cast<int>(columns_.size()-1);
    };
    assignment_prices_.resize(G);base_prices_.resize(L);tangent_prices_.resize(L);
    for (int r=0;r<R;++r) for (int l=0;l<L;++l) {
        const int g=r*L+l,a=layout.assignment_index(r,l),rho=G+l;
        assignment_prices_[g]=add(0,{{a,1}},{{g,1}}); // v <= alpha
        add(0,{},{{g,1},{rho,-1}});                  // v <= rho
        add(1,{{a,-1}},{{rho,1},{g,-1}});            // v >= rho+alpha-1
        if (!data_.supported[g]) add(0,{},{{g,1}});  // missing positive branch
    }
    for (int l=0;l<L;++l) base_prices_[l]=add(0,{{layout.facility+l,1}},{{G+l,1}});
    for (int i=0;i<H;++i) {
        Coefficients rhs,coeff;
        for (int s=0;s<d.n_levels;++s) rhs.emplace_back(layout.capacity+i*d.n_levels+s,d.occupancy_limit(i,s));
        for (int g=0;g<G;++g) if (data_.locker_load[g][i]>0) coeff.emplace_back(g,data_.locker_load[g][i]);
        add(0,std::move(rhs),std::move(coeff));
    }
    Coefficients total_rhs,total_coeff;
    for (int i=0;i<H;++i) for (int s=0;s<d.n_levels;++s)
        total_rhs.emplace_back(layout.capacity+i*d.n_levels+s,d.occupancy_limit(i,s));
    for (int g=0;g<G;++g) if (data_.total_load[g]>0) total_coeff.emplace_back(g,data_.total_load[g]);
    add(0,std::move(total_rhs),std::move(total_coeff));
    const std::vector<double> anchors{0,.25,.5,.75,.9,.95,.97,.98,.99};
    for (int l=0;l<L;++l) for (double t:anchors) {
        const long double a=1/((1-static_cast<long double>(t))*(1-static_cast<long double>(t)));
        tangent_prices_[l].push_back(add(upper_nonnegative(a*t*t),{},
            {{G+l,lower_nonnegative(a)},{G+L+l,-1}}));
    }
    LinearModel dual;
    for (std::size_t j=0;j<columns_.size();++j) dual.variable("task_price_"+std::to_string(j),0,IloInfinity);
    for (int k=0;k<primal_count_;++k) {
        Coefficients row;
        for (std::size_t j=0;j<columns_.size();++j)
            for (auto [v,c]:columns_[j].primal_coefficients) if (v==k) row.emplace_back(j,c);
        dual.row("task_primal_column_"+std::to_string(k),'G',primal_objective_[k],std::move(row));
    }
    SolverOptions quiet=options;quiet.log=false;
    solver_=std::make_unique<NativeSolver>(dual,quiet,"task_resource_dual");
    data_.metadata["resource_primal_variables"]=primal_count_;
    data_.metadata["resource_primal_rows"]=columns_.size();
    data_.metadata["dual_variables"]=columns_.size();
    data_.metadata["dual_constraints"]=primal_count_;
    data_.metadata["idle_tangent_anchors"]=anchors;
    data_.metadata["dual_build_seconds"]=wall_time()-started;
    data_.metadata["certificate"]="repaired_nonnegative_dual_checked_over_every_primal_column";
}

TaskSavingsCut TaskResourceSavingsRelaxation::separate(const std::vector<double>& design,
        double theta_value,int theta,double seconds) {
    const double started=wall_time();
    const int G=data_.heads*data_.bases,L=data_.bases;
    if (design.size()<static_cast<std::size_t>(design_count_) || theta<design_count_ ||
        !std::isfinite(theta_value) || !std::isfinite(seconds))
        throw std::invalid_argument("Invalid task savings separation point");
    for (int j=0;j<design_count_;++j)
        if (!std::isfinite(design[j]) || design[j]<-1e-8 || design[j]>1+1e-8)
            throw std::invalid_argument("Task savings separator needs a unit-box design");
    for (std::size_t j=0;j<columns_.size();++j) {
        long double rhs=columns_[j].constant;
        for (auto [v,c]:columns_[j].design_rhs) rhs+=static_cast<long double>(c)*std::clamp(design[v],0.0,1.0);
        solver_->objective.setLinearCoef(solver_->variables[j],static_cast<double>(rhs));
    }
    TaskSavingsCut cut;
    SolveOutcome solved;
    if (seconds>0) {
        try {solved=solver_->optimize(std::max(0.0,seconds-(wall_time()-started)));cut.status=solved.status;}
        catch (const IloException& e) {cut.status=std::string("solver_exception: ")+e.getMessage();}
    } else cut.status="budget_exhausted";
    cut.optimal=solved.optimal;cut.solver_seconds=solved.seconds;cut.solver_objective=solved.objective;
    cut.fallback=!solved.has_solution || solved.values.size()!=columns_.size() ||
        !std::all_of(solved.values.begin(),solved.values.end(),[](double v){return std::isfinite(v);});
    cut.dual_prices.assign(columns_.size(),0);
    if (!cut.fallback) for (std::size_t j=0;j<columns_.size();++j) cut.dual_prices[j]=std::max(0.0,solved.values[j]);
    auto activities=[&]() {
        std::vector<long double> out(primal_count_,0);
        for (std::size_t j=0;j<columns_.size();++j)
            for (auto [v,c]:columns_[j].primal_coefficients) out[v]+=static_cast<long double>(c)*cut.dual_prices[j];
        return out;
    };
    auto activity=activities();
    for (int k=0;k<primal_count_;++k)
        cut.max_dual_violation_before=std::max(cut.max_dual_violation_before,
            static_cast<double>(primal_objective_[k]-activity[k]));
    // The w-column demands sum tangent prices <= fleet cost. Round scaling
    // inward; increasing the base price below repairs any resulting rho gap.
    for (int l=0;l<L;++l) {
        long double sum=0;for (int j:tangent_prices_[l]) sum+=cut.dual_prices[j];
        const double target=lower_nonnegative(data_.fleet_cost);
        if (sum>target) for (int j:tangent_prices_[l])
            cut.dual_prices[j]=std::max(0.0,std::nextafter(static_cast<double>(cut.dual_prices[j]*target/sum),0.0));
    }
    activity=activities();
    for (int g=0;g<G;++g) {
        const long double deficit=primal_objective_[g]-activity[g];
        if (deficit>0) cut.dual_prices[assignment_prices_[g]]=upper_nonnegative(
            static_cast<long double>(cut.dual_prices[assignment_prices_[g]])+deficit);
    }
    activity=activities();
    for (int l=0;l<L;++l) if (activity[G+l]<0)
        cut.dual_prices[base_prices_[l]]=upper_nonnegative(
            static_cast<long double>(cut.dual_prices[base_prices_[l]])-activity[G+l]);
    activity=activities();
    for (int k=0;k<primal_count_;++k)
        cut.max_dual_violation_after=std::max(cut.max_dual_violation_after,
            static_cast<double>(primal_objective_[k]-activity[k]));
    cut.valid=cut.max_dual_violation_after<=0 &&
        std::all_of(cut.dual_prices.begin(),cut.dual_prices.end(),[](double v){return std::isfinite(v)&&v>=0;});
    if (!cut.valid) {cut.seconds=wall_time()-started;return cut;}
    long double constant=0;
    std::map<int,long double> coefficients;
    for (std::size_t j=0;j<columns_.size();++j) {
        constant+=static_cast<long double>(columns_[j].constant)*cut.dual_prices[j];
        for (auto [v,c]:columns_[j].design_rhs) coefficients[v]+=static_cast<long double>(c)*cut.dual_prices[j];
    }
    // Signed affine coefficients are rounded upward, weakening the row for
    // every nonnegative design. Do not clamp negative assignment prices.
    const double constant_upper=upper_nonnegative(constant);
    cut.row={"task_resource_savings_cut",'G',data_.courier_cost-data_.numerical_margin-constant_upper,{{theta,1}}};
    std::ostringstream key;key<<std::scientific<<std::setprecision(10)<<cut.row.rhs<<';';
    long double saving=constant_upper;
    for (auto [v,c]:coefficients) if (c!=0) {
        const double rounded=std::nextafter(static_cast<double>(c+1e-12L*(1+std::abs(c))),infinity);
        if (!std::isfinite(rounded)) {cut.valid=false;cut.seconds=wall_time()-started;return cut;}
        cut.row.coefficients.emplace_back(v,rounded);saving+=static_cast<long double>(rounded)*design[v];
        key<<v<<':'<<rounded<<';';
    }
    cut.key=key.str();cut.dual_savings=static_cast<double>(saving);
    cut.violation=data_.courier_cost-data_.numerical_margin-cut.dual_savings-theta_value;
    cut.violated=cut.violation>1e-7*(1+std::abs(data_.courier_cost)+std::abs(theta_value));
    cut.seconds=wall_time()-started;
    return cut;
}

} // namespace mlnd
