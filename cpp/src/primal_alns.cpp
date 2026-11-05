#include "mlnd/primal_alns.hpp"
#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace mlnd {
namespace {
constexpr std::array<const char*,6> operator_names{{"activate_cluster","share_assets",
    "close_or_swap","relieve_congestion","balance_bases","elite_recombine"}};
double cost(const Json& point) {return point.at("objective").get<double>();}
bool improves(double a,double b) {return a < b-1e-9*(1+std::abs(b));}
struct Charge {
    double& used;double& active;double begin;bool owner;
    Charge(double& total,double& started):used(total),active(started),begin(wall_time()),owner(started<0) {
        if(owner)active=begin;
    }
    ~Charge(){if(owner){used+=wall_time()-begin;active=-1;}}
};
}

double primal_alns_acceptance_probability(double current,double candidate,double initial_best,
    double used_seconds,double budget_seconds,double reheat_fraction) {
    if(!std::isfinite(current)||!std::isfinite(candidate)||!std::isfinite(initial_best)||
       !std::isfinite(used_seconds)||!std::isfinite(budget_seconds)||!std::isfinite(reheat_fraction))
        return 0;
    if(candidate<=current)return 1;
    if(initial_best<=0 || budget_seconds<=0)return 0;
    const double fraction=std::clamp(used_seconds/budget_seconds,0.0,1.0);
    const double temperature=.01*initial_best/std::log(2.0)*
        std::max(std::pow(.01,fraction),std::clamp(reheat_fraction,0.0,.25));
    return std::exp(-(candidate-current)/temperature);
}

struct PrimalAlnsSearch::Impl {
    struct Entry {
        std::shared_ptr<Json> point;
        PrimalSupportState support;
        std::string signature;
        std::vector<int> groups;
        bool first_seen=false;
    };
    using EntryPtr=std::shared_ptr<Entry>;
    const Input& d;
    const PathCollection& paths;
    SolverOptions options;
    PrimalSupportBank bank;
    std::mt19937 rng;
    double budget=0,used=0,active=-1,initial_best=0;
    double last_validation_seconds=.5,reheat_used_at=-1;
    EntryPtr best,current;
    std::vector<EntryPtr> elite;
    std::vector<int> service_columns,empty_columns;
    std::size_t column_version=0;
    std::set<std::string> seen_designs,refined_designs,seed_domains;
    std::map<std::string,Json> evaluation_history;
    std::array<double,6> weights{{1,1,1,1,1,1}};
    std::array<std::size_t,6> attempts{{0,0,0,0,0,0}},valid{{0,0,0,0,0,0}},wins{{0,0,0,0,0,0}};
    Json records=Json::array();
    std::size_t block_size=4,optional_limit=5000,no_best=0,valid_no_best=0;
    std::size_t slow_builds=0,fast_closed=0,restarts=0,refinements=0;
    std::size_t repairs=0,accepted_count=0,validated_count=0,selection_failures=0;
    bool startup_done=false,supported=false;
    std::string last_status="not_started";

    Impl(const Input& input,const PathCollection& catalogue,const SolverOptions& settings,
         const Json& initial,double allowance):d(input),paths(catalogue),options(settings),
         bank(input,catalogue),rng(static_cast<std::mt19937::result_type>(settings.seed)),budget(allowance) {
        const double begin=wall_time();
        if(!std::isfinite(budget)||budget<0)throw std::invalid_argument("Invalid primal ALNS budget");
        supported=d.n_l==2 && d.max_orders==2 && d.preprocessed_demand && d.origin_assignment() &&
            d.flow_policy=="aggregate" && !d.first_leg_consistency && paths.support_complete;
        best=std::make_shared<Entry>();best->point=std::make_shared<Json>(all_courier_result(d,paths));
        best->point->erase("bound");best->point->erase("gap");
        current=best;
        // The all-courier point is constructed internally, not accepted merely
        // because a caller labels an arbitrary supplied point a fallback.
        if(initial.value("native_model_validated",false)) {
            check_validated(initial);
            if(improves(cost(initial),cost(*best->point))) {
                best=copy_entry(initial);current=best;
            }
        }
        initial_best=cost(*best->point);
        if(supported && budget>0 && ensure_support(best,begin+std::min(1.0,budget)))add_elite(best);
        used=wall_time()-begin;
    }
    double elapsed() const {return used+(active>=0?wall_time()-active:0);}
    double bounded_deadline(double deadline) const {
        return std::min(deadline,wall_time()+std::max(0.0,budget-elapsed()));
    }
    void check_validated(const Json& candidate) const {
        if(!candidate.value("native_model_validated",false) || !candidate.contains("_solution") ||
           !std::isfinite(cost(candidate)) ||
           candidate.value("assignment_scope",std::string())!=d.assignment_scope() ||
           candidate.value("flow_policy",std::string())!=d.flow_policy ||
           (candidate.value("demand_formulation",std::string())=="preprocessed")!=d.preprocessed_demand)
            throw std::invalid_argument("ALNS accepts only independently validated same-model integer points");
        // Encoding is checked here; native/physical validation is the caller's
        // trust boundary and is performed by every repair before its callback.
        (void)solution_design(d,candidate);
    }
    EntryPtr copy_entry(const Json& point) const {
        auto entry=std::make_shared<Entry>();entry->point=std::make_shared<Json>(point);
        entry->point->erase("bound");entry->point->erase("gap");return entry;
    }
    bool ensure_support(const EntryPtr& entry,double deadline) {
        if(entry->support.ready)return true;
        if(wall_time()>=deadline)return false;
        entry->support=bank.inspect(*entry->point,deadline);
        if(!entry->support.ready)return false;
        const DesignLayout layout(d);const auto& design=entry->support.design;
        entry->groups=entry->support.head_state;
        for(int h=0;h<d.n_h;++h) {
            int level=0;
            for(int z=0;z<d.n_levels;++z)if(design.at(layout.capacity+h*d.n_levels+z)>.5)level=z+1;
            entry->groups.push_back(level);
        }
        std::vector<int> base_state;
        for(int l=0;l<d.n_l;++l)base_state.push_back(design.at(layout.facility+l)>.5?1:0);
        std::set<std::pair<int,int>> used_airlines;
        for(const auto& resources:entry->support.head_resources)
            used_airlines.insert(resources.airlines.begin(),resources.airlines.end());
        for(int index:entry->support.positive_empty)
            for(auto [u,v,count]:paths.empty_paths.at(index).arc_counts)
                if(count)used_airlines.insert({u,v});
        entry->signature=Json({entry->groups,base_state,used_airlines}).dump();
        entry->first_seen=seen_designs.insert(entry->signature).second;return true;
    }
    static std::size_t distance(const EntryPtr& a,const EntryPtr& b) {
        if(a->groups.size()!=b->groups.size())return 0;
        std::size_t result=0;
        for(std::size_t i=0;i<a->groups.size();++i)result+=a->groups[i]!=b->groups[i];
        return result;
    }
    void add_elite(const EntryPtr& point) {
        if(point->signature.empty())return;
        bool duplicate=false;
        for(auto& old:elite)if(old->signature==point->signature) {
            // A same-signature best replacement must also refresh pointer
            // identity, even when its cost difference is below ranking noise.
            if(point==best || improves(cost(*point->point),cost(*old->point)))old=point;
            duplicate=true;break;
        }
        const double ceiling=cost(*best->point)*(elite.size()<8?1.05:1.02);
        if(!duplicate && (point==best || cost(*point->point)<=ceiling+1e-7))elite.push_back(point);
        // Always prune: improving an existing signature can make other elites
        // ineligible just as a brand-new best design can.
        const double prune_ceiling=cost(*best->point)*(elite.size()>=8?1.02:1.05);
        elite.erase(std::remove_if(elite.begin(),elite.end(),[&](const auto& e) {
            return e!=best && cost(*e->point)>prune_ceiling+1e-7;
        }),elite.end());
        if(elite.size()<=8)return;
        // Keep the best, then greedily retain head/capacity diversity. Ties
        // choose the less costly feasible state, not another hash of its flow.
        std::vector<EntryPtr> kept{best};
        while(kept.size()<8) {
            EntryPtr chosen;std::size_t chosen_distance=0;
            for(const auto& candidate:elite) {
                if(std::find(kept.begin(),kept.end(),candidate)!=kept.end())continue;
                std::size_t nearest=std::numeric_limits<std::size_t>::max();
                for(const auto& old:kept)nearest=std::min(nearest,distance(candidate,old));
                if(!chosen || nearest>chosen_distance ||
                   (nearest==chosen_distance && cost(*candidate->point)<cost(*chosen->point))) {
                    chosen=candidate;chosen_distance=nearest;
                }
            }
            if(!chosen)break;
            kept.push_back(chosen);
        }
        elite=std::move(kept);
    }
    bool promote(const Json& point,const Submit& submit) {
        check_validated(point);
        if(!improves(cost(point),cost(*best->point)))return false;
        // Saving is the first action: optional support indexing, elite work,
        // acceptance and pool extraction may all run out of their allowance.
        if(submit)submit(point);
        best=copy_entry(point);
        return true;
    }
    EntryPtr alternative_elite() const {
        EntryPtr result;std::size_t farthest=0;
        const double ceiling=cost(*best->point)*(elite.size()<8?1.05:1.02);
        for(const auto& e:elite)if(e->signature!=current->signature &&
            cost(*e->point)<=ceiling+1e-7) {
            const auto separation=distance(e,current);
            if(!result || separation>farthest) {result=e;farthest=separation;}
        }
        return result;
    }
    int select_operator() {
        if(current->point->value("served",0.0)<=1e-8)return 0;
        std::vector<int> available{0,1,4};
        const bool resources=std::any_of(current->support.head_resources.begin(),current->support.head_resources.end(),
            [](const auto& a){return !a.lockers.empty()||!a.airlines.empty();});
        if(resources)available.push_back(2);
        if(std::any_of(current->support.locker_load.begin(),current->support.locker_load.end(),
                       [](double value){return value>0;}))available.push_back(3);
        if(alternative_elite())available.push_back(5);
        std::sort(available.begin(),available.end());
        for(int op:available)if(attempts[op]==0)return op;
        double sum=0;for(int op:available)sum+=weights[op];
        std::uniform_real_distribution<double> draw(0,1);double p=draw(rng);
        for(int op:available) {
            p-=.2/static_cast<double>(available.size())+.8*weights[op]/sum;
            if(p<=0)return op;
        }
        return available.back();
    }
    void restart_if_needed(Json& record) {
        if(valid_no_best<8)return;
        auto different=alternative_elite();
        if(different)current=different;
        block_size=std::min<std::size_t>(16,block_size*2);
        valid_no_best=0;++restarts;
        record["restart"]=different?"diverse_elite":"enlarged_block";
        if(reheat_used_at<0) {reheat_used_at=elapsed();record["reheated"]=true;}
    }
    void execute(PrimalBlockRequest request,double begin,double deadline,const Submit& submit,
                 const char* stage) {
        const int op=request.operator_index;
        if(op<0||op>=6)throw std::invalid_argument("Unknown ALNS operator");
        ++attempts[op];
        const double old_best=cost(*best->point),old_current=cost(*current->point);
        Json record={{"stage",stage},{"operator",operator_names[op]},{"block_size",block_size},
            {"best_before",old_best},{"current_before",old_current},{"validated",false},
            {"accepted",false},{"global_lower_bound_available",false}};
        request.optional_column_limit=optional_limit;
        request.complete_missing_support=options.primal_support_completion;
        request.indexed_support_completion=options.indexed_support_completion;
        // Startup may need simultaneous activation of more than one seed
        // block. The ordinary ALNS phase uses the configured group radii.
        request.restrictions.grouped_radius=options.grouped_local_branching && std::string(stage)!="startup";
        // Alternate feasible intensification and forced-change exploration.
        // A forced resource closure is inherently exploratory on every call.
        const bool forced=!request.restrictions.closed_bases.empty()||!request.restrictions.closed_lockers.empty()||
            !request.restrictions.closed_airlines.empty();
        const bool explore=forced || attempts[op]%2==1 || current->point->value("served",0.0)<=1e-8;
        if(explore) {
            if(!forced && request.restrictions.change_heads.empty() && request.restrictions.change_lockers.empty())
                request.restrictions.change_heads=request.heads;
            request.restrictions.objective_upper=1.03*old_best;
        } else {
            request.restrictions.change_heads.clear();request.restrictions.change_lockers.clear();
            request.restrictions.objective_upper=std::numeric_limits<double>::infinity();
        }
        record["mode"]=explore?"exploration":"intensification";
        auto block=bank.close(current->support,request,std::min(deadline,wall_time()+.2));
        record["selection"]=block.diagnostics;
        record["selection_seconds"]=wall_time()-begin;
        bool accepted=false,novel=false,improved_current=false,worse_neighbor=false,valid_neighbor=false;
        double build_seconds=0,call_seconds=0;
        if(!block.ready || wall_time()>=deadline) {
            ++selection_failures;record["status"]=block.ready?"selection_budget":block.status;
        } else {
            const std::string domain=Json({block.coupled.heads,block.coupled.bases,block.service_indices,
                block.restrictions.closed_lockers,block.restrictions.closed_bases,block.restrictions.closed_airlines}).dump();
            if(std::string(stage)=="startup" && !seed_domains.insert(domain).second) {
                ++selection_failures;record["status"]="duplicate_seed_domain";
            } else {
                const auto merge_columns=[](std::vector<int>& into,const std::vector<int>& added) {
                    const auto before=into.size();into.insert(into.end(),added.begin(),added.end());
                    std::sort(into.begin(),into.end());into.erase(std::unique(into.begin(),into.end()),into.end());
                    return before!=into.size();
                };
                const bool service_grew=merge_columns(service_columns,block.service_indices);
                const bool empty_grew=merge_columns(empty_columns,block.empty_indices);
                if(service_grew || empty_grew)++column_version;
                SolverOptions local=options;local.neighborhood_cuts=false;local.gap=.02;
                local.seconds=std::max(0.0,deadline-wall_time());
                block.restrictions.validation_reserve_seconds=std::max({.5,.1*local.seconds,last_validation_seconds});
                block.restrictions.retain_validated=[&](const Json& candidate){promote(candidate,submit);};
                const std::string evaluation_key=Json({current->support.design,paths.service_paths.size(),
                    block.service_indices,block.empty_indices,block.coupled.heads,block.coupled.lockers,
                    block.coupled.bases,block.coupled.airlines,block.restrictions.change_heads,
                    block.restrictions.change_lockers,block.restrictions.closed_bases,
                    block.restrictions.closed_lockers,block.restrictions.closed_airlines,
                    block.restrictions.forbidden_origins,block.restrictions.grouped_radius,
                    options.local_branching_head_radius,options.local_branching_capacity_radius,
                    std::isfinite(block.restrictions.objective_upper)?Json(block.restrictions.objective_upper):Json(nullptr)}).dump();
                record["previous_same_domain_evaluation"]=evaluation_history.count(evaluation_key)>0;
                record["path_domain_version"]=paths.service_paths.size();
                const double call_start=wall_time();++repairs;
                auto result=solve_restricted_heuristic(d,paths,local,deadline,current->point.get(),-1,
                    &block.service_indices,&block.empty_indices,false,false,&block.coupled,&block.restrictions);
                call_seconds=wall_time()-call_start;
                record["status"]=result.value("status",std::string("unknown"));
                record["restricted_model_optimal"]=result.value("restricted_model_optimal",false);
                record["repair_timings"]=result.value("timings",Json::object());
                record["model_size"]=result.value("model_size",Json::object());
                record["grouped_local_branching"]=result.value("grouped_local_branching",Json::object());
                record["local_domain_columns"]=block.service_indices.size();
                if(result.contains("timings")) {
                    build_seconds=result.at("timings").value("build",0.0);
                    last_validation_seconds=std::max(.5,1.5*result.at("timings").value("validation",0.0));
                }
                if(result.value("candidate_validated",false) && result.contains("candidate")) {
                    const auto& candidate=result.at("candidate");check_validated(candidate);
                    promote(candidate,submit);
                    EntryPtr neighbor;
                    if(cost(candidate)==cost(*best->point) && solution_design(d,candidate)==solution_design(d,*best->point))neighbor=best;
                    else neighbor=copy_entry(candidate);
                    valid_neighbor=true;++validated_count;++valid[op];
                    // Feasibility has already been established. Indexing may
                    // be deferred if the local deadline is exhausted.
                    if(ensure_support(neighbor,deadline)) {
                        novel=neighbor->first_seen &&
                            std::none_of(elite.begin(),elite.end(),[&](const auto& old){return old->signature==neighbor->signature;});
                        add_elite(neighbor);neighbor->first_seen=false;
                    }
                    double reheat=0;
                    if(reheat_used_at>=0)reheat=.25*std::pow(.01,
                        std::clamp((elapsed()-reheat_used_at)/std::max(1e-9,budget-reheat_used_at),0.0,1.0));
                    const double p=primal_alns_acceptance_probability(old_current,cost(candidate),initial_best,elapsed(),budget,reheat);
                    std::uniform_real_distribution<double> draw(0,1);
                    accepted=p>=1||draw(rng)<p;
                    improved_current=improves(cost(candidate),old_current);
                    worse_neighbor=cost(candidate)>old_current;
                    record["candidate_objective"]=cost(candidate);record["acceptance_probability"]=p;
                    record["validated"]=true;record["accepted"]=accepted;record["novel_design"]=novel;
                    if(accepted){current=neighbor;++accepted_count;}
                }
                // A fallback is never interpreted as a neighborhood result.
                record["candidate_available"]=valid_neighbor;
                evaluation_history[evaluation_key]={{"status",record.at("status")},
                    {"validated_feasible",valid_neighbor},{"restricted_optimal",record.at("restricted_model_optimal")},
                    {"path_domain_version",paths.service_paths.size()},{"global_inference",false}};
            }
        }
        if(ensure_support(best,deadline))add_elite(best);
        const bool new_best=improves(cost(*best->point),old_best);
        if(new_best) {++wins[op];no_best=0;valid_no_best=0;block_size=4;}
        else {
            ++no_best;if(valid_neighbor)++valid_no_best;
            if(no_best%3==0)block_size=std::min<std::size_t>(16,block_size*2);
        }
        const double seconds=wall_time()-begin;
        if(call_seconds>0 && build_seconds>.3*call_seconds)++slow_builds;else slow_builds=0;
        if(slow_builds>=3) {
            block_size=std::max<std::size_t>(2,block_size/2);
            optional_limit=std::max<std::size_t>(64,optional_limit/2);slow_builds=0;
            record["adaptation"]="reduce_optional_domain";
        }
        if(valid_neighbor && record.value("restricted_model_optimal",false) &&
           !new_best && seconds<.5*std::max(0.0,deadline-begin))++fast_closed;else fast_closed=0;
        if(fast_closed>=2) {block_size=std::min<std::size_t>(16,block_size*2);fast_closed=0;record["adaptation"]="enlarge_fast_closed_block";}
        const double base_reward=new_best?4:(accepted&&improved_current?2:(accepted&&novel&&worse_neighbor?1:0));
        const double bonus=new_best?std::min(1.0,(old_best-cost(*best->point))/std::max(1.0,std::abs(old_best))):0;
        const double reward=(base_reward+bonus)/(5*(1+seconds/5));
        weights[op]=.8*weights[op]+.2*std::clamp(reward,0.0,1.0);
        record["reward"]=reward;record["weight_after"]=weights[op];
        record["seconds"]=seconds;record["best_after"]=cost(*best->point);record["current_after"]=cost(*current->point);
        record["primal_seconds"]=elapsed();record["new_best"]=new_best;
        restart_if_needed(record);
        last_status=record.at("status").get<std::string>();records.push_back(std::move(record));
    }
};

PrimalAlnsSearch::PrimalAlnsSearch(const Input& d,const PathCollection& p,const SolverOptions& o,
    const Json& initial,double budget):impl_(std::make_unique<Impl>(d,p,o,initial,budget)){}
PrimalAlnsSearch::~PrimalAlnsSearch()=default;
PrimalAlnsSearch::PrimalAlnsSearch(PrimalAlnsSearch&&) noexcept=default;
PrimalAlnsSearch& PrimalAlnsSearch::operator=(PrimalAlnsSearch&&) noexcept=default;
void PrimalAlnsSearch::startup(double deadline,const Submit& submit) {
    auto& s=*impl_;Charge charge(s.used,s.active);
    deadline=s.bounded_deadline(deadline);
    if(s.startup_done||!s.supported||wall_time()>=deadline)return;
    s.bank.extend(std::min(deadline,wall_time()+8),s.options.indexed_support_completion);
    for(int direction=0;direction<3 && wall_time()<deadline;++direction) {
        if(!s.ensure_support(s.current,deadline)) {s.last_status="incumbent_support_budget";break;}
        const double begin=wall_time();
        auto request=s.bank.seed(direction,s.current->support,s.rng,std::min(deadline,begin+2));
        s.execute(std::move(request),begin,std::min(deadline,begin+6),submit,"startup");
    }
    s.startup_done=true;
    s.current=s.best;s.initial_best=cost(*s.best->point);
}
void PrimalAlnsSearch::step(double deadline,const std::vector<double>* guide,const Submit& submit) {
    auto& s=*impl_;Charge charge(s.used,s.active);deadline=s.bounded_deadline(deadline);
    if(!s.supported||wall_time()>=deadline)return;
    const double begin=wall_time();
    s.bank.extend(std::min(deadline,begin+.2),s.options.indexed_support_completion);
    if(!s.ensure_support(s.current,deadline)) {s.last_status="incumbent_support_budget";return;}
    const int op=s.select_operator();auto other=s.alternative_elite();
    auto request=s.bank.request(op,s.current->support,s.block_size,s.rng,std::min(deadline,wall_time()+.2),
        guide,other?&other->support:nullptr);
    s.execute(std::move(request),begin,deadline,submit,"regular");
}
void PrimalAlnsSearch::observe_validated(const Json& point) {
    auto& s=*impl_;Charge charge(s.used,s.active);s.check_validated(point);
    if(!improves(cost(point),cost(*s.best->point)))return;
    auto entry=s.copy_entry(point);s.best=entry;s.current=entry;
    if(s.ensure_support(entry,wall_time()+std::min(1.0,std::max(0.0,s.budget-s.elapsed()))))s.add_elite(entry);
}
Json PrimalAlnsSearch::observe_cg_columns(const PrimalCgFeedback& feedback,double deadline) {
    auto& s=*impl_;
    if(!s.options.cg_primal_feedback)return {{"status","disabled"},{"accepted",0},{"certificate_eligible",false}};
    Charge charge(s.used,s.active);deadline=s.bounded_deadline(deadline);
    return s.bank.observe_cg_columns(feedback,deadline);
}
const Json& PrimalAlnsSearch::best() const {return *impl_->best->point;}
const Json& PrimalAlnsSearch::current() const {return *impl_->current->point;}
const Json& PrimalAlnsSearch::records() const {return impl_->records;}
double PrimalAlnsSearch::used_seconds() const {return impl_->elapsed();}
void PrimalAlnsSearch::synchronize_used_seconds(double shared_used) {
    auto& s=*impl_;
    if(!std::isfinite(shared_used)||shared_used<0)
        throw std::invalid_argument("Invalid shared primal elapsed time");
    if(s.active>=0)throw std::logic_error("Synchronize primal time only between search calls");
    s.used=std::max(s.used,shared_used);
}
std::optional<Json> PrimalAlnsSearch::take_elite_refinement() {
    auto& s=*impl_;Charge charge(s.used,s.active);
    if(s.refinements>=2)return std::nullopt;
    if(s.best->signature.empty())return std::nullopt;
    std::vector<Impl::EntryPtr> candidates{s.best};
    for(const auto& entry:s.elite)if(entry!=s.best)candidates.push_back(entry);
    for(const auto& entry:candidates)if(entry->point->value("served",0.0)>1e-8 &&
        cost(*entry->point)<=1.02*cost(*s.best->point)+1e-7 &&
        s.refined_designs.insert(entry->signature).second) {
        ++s.refinements;return *entry->point;
    }
    return std::nullopt;
}
const std::vector<int>& PrimalAlnsSearch::service_columns() const {return impl_->service_columns;}
const std::vector<int>& PrimalAlnsSearch::empty_columns() const {return impl_->empty_columns;}
Json PrimalAlnsSearch::summary() const {
    const auto& s=*impl_;Json operators=Json::array();
    for(int op=0;op<6;++op)operators.push_back({{"name",operator_names[op]},{"attempts",s.attempts[op]},
        {"validated_neighbors",s.valid[op]},{"best_improvements",s.wins[op]},{"weight",s.weights[op]}});
    Json costs=Json::array();for(const auto& entry:s.elite)costs.push_back(cost(*entry->point));
    return {{"enabled",true},{"supported",s.supported},{"status",s.last_status},{"startup_done",s.startup_done},
        {"best",cost(*s.best->point)},{"current",cost(*s.current->point)},
        {"used_seconds",s.elapsed()},{"budget_seconds",s.budget},{"repair_calls",s.repairs},
        {"validated_neighbors",s.validated_count},{"accepted_neighbors",s.accepted_count},
        {"selection_failures",s.selection_failures},{"elite_size",s.elite.size()},{"elite_costs",costs},
        {"evaluation_history_entries",s.evaluation_history.size()},{"different_designs",s.seen_designs.size()},{"restarts",s.restarts},{"reheated",s.reheat_used_at>=0},
        {"elite_refinements_requested",s.refinements},{"next_block_size",s.block_size},
        {"column_version",s.column_version},{"service_columns",s.service_columns.size()},{"empty_columns",s.empty_columns.size()},
        {"optional_column_limit",s.optional_limit},{"operators",operators},{"support",s.bank.metadata()},
        {"global_lower_bound_available",false}};
}
} // namespace mlnd
