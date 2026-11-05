#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"
#include "mlnd/pricing.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
bool expired(double deadline) {return wall_time()>=deadline;}
std::string input_key(const Input& d) {
    return Json({d.n_l,d.n_h,d.n_od,d.n_levels,d.max_orders,d.preprocessed_demand,
        d.first_leg_consistency,d.model_version,d.assignment_scope(),d.flow_policy,d.route_mode,d.delta,d.operating_limit,
        d.demand,d.single_rates,d.pair_rates.values,d.head_rates,d.common_origin_support.values,
        d.ground_cost,d.base_cost,d.slot_cost.values,d.link_cost.values,d.fleet_cost,
        d.flight_cost,d.waiting_cost,d.flight_times.values,d.allowed_links.values,
        d.rider_mean.values,d.first_cost.values,d.terminal_cost.values,d.receipt_mean.values,d.home_receipt_mean.values,
        d.removal_mean.values,d.slot_epsilon,d.slot_counts.values,d.occupancy_limit.values,
        d.kappa.values,d.deadline}).dump();
}
std::string path_key(const ServicePath& p) {
    return Json({p.id,p.segment_id,p.first_od,p.second_od,p.b,p.mode,p.pickups,p.deliveries,
                 p.legs,p.arc_counts,p.slot_times,p.served_counts,p.cost,p.time,p.flight_time,p.wait,
                 p.pickup_times,p.delivery_times}).dump();
}
std::string empty_key(const EmptyPath& p) {
    return Json({p.id,p.b,p.legs,p.arc_counts,p.cost,p.time,p.flight_time}).dump();
}
void add(PrimalResources& to,const PrimalResources& from) {
    to.bases.insert(from.bases.begin(),from.bases.end());
    to.lockers.insert(from.lockers.begin(),from.lockers.end());
    to.airlines.insert(from.airlines.begin(),from.airlines.end());
}
PrimalResources resources(const Input& d,const ServicePath& p) {
    PrimalResources a;a.bases.insert(p.b[0]);a.bases.insert(p.b[1]);
    for(auto [h,load]:p.slot_times) { (void)load;a.lockers.insert(h); }
    for(const auto& leg:p.legs)for(int node:leg)if(node>=d.n_l)a.lockers.insert(node-d.n_l);
    for(auto [u,v,count]:p.arc_counts)if(count)a.airlines.insert({u,v});
    return a;
}
void add_empty(const EmptyPath& p,PrimalResources& a) {
    a.bases.insert(p.b[0]);a.bases.insert(p.b[1]);
    for(auto [u,v,count]:p.arc_counts)if(count)a.airlines.insert({u,v});
}
double fixed_cost(const Input& d,const PrimalResources& a) {
    double c=0;for(int l:a.bases)c+=d.base_cost.at(l);
    for(int h:a.lockers) {double lowest=INFINITY;for(int k=0;k<d.n_levels;++k)lowest=std::min(lowest,d.slot_cost(h,k));c+=lowest;}
    for(auto [u,v]:a.airlines)c+=d.link_cost(u,v);
    return c;
}
double overlap(const PrimalResources& a,const PrimalResources& b) {
    double score=0;for(int h:a.lockers)if(b.lockers.count(h))score+=1;
    for(auto arc:a.airlines)if(b.airlines.count(arc))score+=.25;
    return score;
}
bool uses_closed(const PrimalResources& a,const PrimalRepairRestrictions& r) {
    for(int h:r.closed_lockers)if(a.lockers.count(h))return true;
    for(int l:r.closed_bases)if(a.bases.count(l))return true;
    for(auto arc:r.closed_airlines)if(a.airlines.count(arc))return true;
    return false;
}
template<class T> void unique(std::vector<T>& x) {std::sort(x.begin(),x.end());x.erase(std::unique(x.begin(),x.end()),x.end());}
void state_owner(const PrimalSupportState& state,const void* owner) {
    if(!state.ready || state.owner!=owner)throw std::invalid_argument("Support state is incomplete or belongs to another bank");
}
}

PrimalCgFeedback make_primal_cg_feedback(const Input& d,const PathCollection& paths,
        const std::vector<double>& design,const std::vector<std::pair<int,double>>& negative,
        const std::vector<std::pair<int,double>>& positive,std::size_t max_columns) {
    const DesignLayout layout(d);
    if(design.size()!=static_cast<std::size_t>(layout.count) ||
       !std::all_of(design.begin(),design.end(),[](double x){return std::isfinite(x)&&x>=-1e-7&&x<=1+1e-7;}))
        throw std::invalid_argument("Invalid CG feedback source design");
    PrimalCgFeedback out;out.source_design=design;out.domain_identity=pricing_domain_identity(d,paths);
    std::map<int,PrimalCgColumn> columns;
    const auto add=[&](int id,double value,bool flow) {
        if(id<0 || static_cast<std::size_t>(id)>=paths.service_paths.size() || !std::isfinite(value) ||
           (flow?value<=0:value>=0))throw std::invalid_argument("Invalid CG feedback observation");
        auto& column=columns[id];column.path=id;
        if(flow)column.validated_flow=std::max(column.validated_flow,value);
        else if(std::isnan(column.reduced_cost) || value<column.reduced_cost)column.reduced_cost=value;
    };
    for(auto [id,value]:negative)add(id,value,false);
    for(auto [id,value]:positive)add(id,value,true);
    auto ranked_negative=negative,ranked_positive=positive;
    std::sort(ranked_negative.begin(),ranked_negative.end(),[](const auto& a,const auto& b){
        return std::tie(a.second,a.first)<std::tie(b.second,b.first);});
    std::sort(ranked_positive.begin(),ranked_positive.end(),[](const auto& a,const auto& b){
        return a.second!=b.second?a.second>b.second:a.first<b.first;});
    const std::size_t limit=std::min<std::size_t>(64,max_columns);std::set<int> selected;
    const auto keep=[&](int id) {
        if(out.service.size()>=limit || !selected.insert(id).second)return;
        auto column=columns.at(id);column.path_identity=path_key(paths.service_paths.at(id));
        out.service.push_back(std::move(column));
    };
    for(std::size_t i=0;i<std::max(ranked_negative.size(),ranked_positive.size()) && out.service.size()<limit;++i) {
        if(i<ranked_negative.size())keep(ranked_negative[i].first);
        if(i<ranked_positive.size())keep(ranked_positive[i].first);
    }
    return out;
}

PrimalSupportBank::PrimalSupportBank(const Input& d,const PathCollection& paths)
    :d_(d),paths_(paths),input_key_(input_key(d)),scope_(paths.scope),empty_count_(paths.empty_paths.size()) {
    supported_=d.n_l==2 && d.max_orders==2 && d.preprocessed_demand && d.origin_assignment() &&
        d.flow_policy=="aggregate" && !d.first_leg_consistency && paths.support_complete;
    for(const auto& p:paths.empty_paths)empty_keys_.push_back(empty_key(p));
}
void PrimalSupportBank::validate(const std::string* feedback_identity) const {
    checked_in_operation_.clear();
    if(input_key_!=input_key(d_) || scope_!=paths_.scope || paths_.service_paths.size()<scanned_ || paths_.service_paths.size()<indexed_ ||
       paths_.empty_paths.size()<empty_count_ || !paths_.support_complete)
        throw std::invalid_argument("Primal support bank model/catalogue identity changed");
    if(!feedback_.empty() && (feedback_identity?feedback_domain_identity_!=*feedback_identity:
        feedback_domain_identity_!=pricing_domain_identity(d_,paths_)))
        throw std::invalid_argument("CG feedback input/task/support identity changed");
    for(const auto& batch:feedback_)for(const auto& column:batch.service)
        if(column.path<0 || static_cast<std::size_t>(column.path)>=paths_.service_paths.size() ||
           column.path_identity!=path_key(paths_.service_paths[column.path]))
            throw std::invalid_argument("CG feedback canonical path changed on use");
    for(std::size_t i=0;i<empty_keys_.size();++i)if(empty_key(paths_.empty_paths.at(i))!=empty_keys_[i])
        throw std::invalid_argument("Primal support bank empty path changed");
}
std::vector<int> PrimalSupportBank::required(int r) const {
    std::vector<int> out;
    if(d_.task_rate(r)>0)out.push_back(d_.task_second(r));
    return out;
}
void PrimalSupportBank::extend_index(double deadline) const {
    if(expired(deadline))return;
    for(;indexed_<paths_.service_paths.size();++indexed_) {
        if(indexed_%64==0 && expired(deadline))break;
        const auto& p=paths_.service_paths[indexed_];
        complete_branches_[{d_.task_index(p.first_od,p.second_od),p.b[0],p.second_od}].push_back(static_cast<int>(indexed_));
    }
}
void PrimalSupportBank::extend(double deadline,bool index_paths) {
    validate();if(!supported_)return;
    if(index_paths)extend_index(deadline);
    for(;scanned_<paths_.service_paths.size();++scanned_) {
        if(scanned_%128==0 && expired(deadline))break;
        const auto& p=paths_.service_paths[scanned_];
        if(p.first_od<0 || p.first_od>=d_.n_od || p.b[0]<0 || p.b[0]>=d_.n_l)continue;
        const double rate=p.second_od<0?d_.single_rates[p.first_od]:d_.pair_rates(p.first_od,p.second_od);
        if(rate<=0)continue;
        double load=0;for(auto [h,t]:p.slot_times) { (void)h;load+=t; }
        RankedPath candidate{static_cast<int>(scanned_),p.cost+d_.fleet_cost*p.time,
            fixed_cost(d_,resources(d_,p)),load,p.b[1]};
        auto& list=branches_[{d_.task_index(p.first_od,p.second_od),p.b[0],p.second_od}];
        list.push_back(candidate);
        // Keep two cheap operating paths, two cheap asset bundles, two low
        // load paths and the cheapest witness for each return base. A missing
        // empty return never discards a cross-base service witness.
        std::vector<RankedPath> selected;std::set<int> ids;
        auto keep=[&](const RankedPath& x){if(ids.insert(x.id).second)selected.push_back(x);};
        auto cheapest=[&](auto score) {
            auto ranked=list;std::sort(ranked.begin(),ranked.end(),[&](const auto& a,const auto& b){
                return std::make_pair(score(a),a.id)<std::make_pair(score(b),b.id);});
            for(std::size_t i=0;i<std::min<std::size_t>(2,ranked.size());++i)keep(ranked[i]);
        };
        cheapest([](const auto& p){return p.operating;});
        cheapest([](const auto& p){return p.fixed;});
        cheapest([](const auto& p){return p.load;});
        for(int back=0;back<d_.n_l;++back) {
            const RankedPath* best=nullptr;for(const auto& x:list)if(x.return_base==back &&
                (!best || std::tie(x.operating,x.id)<std::tie(best->operating,best->id)))best=&x;
            if(best)keep(*best);
        }
        for(const auto& old:list)if(!ids.count(old.id))retained_keys_.erase(old.id);
        if(ids.count(candidate.id))retained_keys_[candidate.id]=path_key(p);
        list=std::move(selected);
    }
    empty_count_=paths_.empty_paths.size();
    empty_keys_.clear();for(const auto& p:paths_.empty_paths)empty_keys_.push_back(empty_key(p));
}
Json PrimalSupportBank::observe_cg_columns(const PrimalCgFeedback& feedback,double deadline) {
    if(expired(deadline))return {{"status","feedback_budget"},{"accepted",0},{"certificate_eligible",false}};
    const auto identity=pricing_domain_identity(d_,paths_);validate(&identity);
    if(!supported_)return {{"status","unsupported_scope"},{"accepted",0},{"certificate_eligible",false}};
    const DesignLayout layout(d_);
    if(feedback.domain_identity!=identity || feedback.source_design.size()!=static_cast<std::size_t>(layout.count) ||
       !std::all_of(feedback.source_design.begin(),feedback.source_design.end(),[](double x){return std::isfinite(x)&&x>=-1e-7&&x<=1+1e-7;}))
        throw std::invalid_argument("CG feedback belongs to another domain or has an invalid source design");
    PrimalCgFeedback accepted;accepted.domain_identity=identity;accepted.source_design=feedback.source_design;
    std::set<int> seen;
    for(const auto& column:feedback.service) {
        if(accepted.service.size()>=64 || expired(deadline))break;
        if(column.path<0 || static_cast<std::size_t>(column.path)>=paths_.service_paths.size() ||
           (!std::isnan(column.reduced_cost) && (!std::isfinite(column.reduced_cost) || column.reduced_cost>=0)) ||
           !std::isfinite(column.validated_flow) || column.validated_flow<0 ||
           (std::isnan(column.reduced_cost) && column.validated_flow==0) ||
           column.path_identity!=path_key(paths_.service_paths[column.path]))
            throw std::invalid_argument("Invalid or changed CG feedback path payload");
        if(seen.insert(column.path).second)accepted.service.push_back(column);
    }
    const auto count=accepted.service.size();std::size_t negative_count=0,positive_count=0;
    for(const auto& column:accepted.service) {negative_count+=std::isfinite(column.reduced_cost);positive_count+=column.validated_flow>0;}
    if(count) {
        if(feedback_.size()==2)feedback_.erase(feedback_.begin());
        feedback_.push_back(std::move(accepted));feedback_domain_identity_=identity;
        ++feedback_observations_;feedback_columns_received_+=count;
    }
    return {{"status",count?"accepted":"empty_or_budget"},{"accepted",count},
        {"original_negative_columns",negative_count},{"validated_positive_flow_columns",positive_count},
        {"source_design",feedback.source_design},{"retained_batches",feedback_.size()},
        {"different_design_use","ranking_only"},{"certificate_eligible",false}};
}
Json PrimalSupportBank::metadata() const {
    std::size_t feedback_columns=0;for(const auto& batch:feedback_)feedback_columns+=batch.service.size();
    return {{"supported",supported_},{"assignment_scope",d_.assignment_scope()},{"task_count",d_.task_count()},{"scanned_service_paths",scanned_},
        {"registered_service_paths",paths_.service_paths.size()},
        {"catalogue_scan_complete",scanned_==paths_.service_paths.size()},
        {"catalogue_complete",paths_.complete},{"partial",scanned_<paths_.service_paths.size() || !paths_.complete},
        {"branch_groups",branches_.size()},{"retained_paths",retained_keys_.size()},
        {"complete_branch_index",{{"indexed_paths",indexed_},{"groups",complete_branches_.size()},
            {"registered_pool_complete",indexed_==paths_.service_paths.size()},{"physical_domain_complete",paths_.complete}}},
        {"cg_feedback",{{"observations",feedback_observations_},{"accepted_columns",feedback_columns_received_},
            {"retained_batches",feedback_.size()},{"retained_columns",feedback_columns},{"batch_limit",64},
            {"per_branch_optional_limit",2},{"certificate_eligible",false}}},
        {"path_domain_version",paths_.service_paths.size()},
        {"cache_kind",feedback_.empty()?"physical_support_only":"physical_support_and_primal_ranking_only"}};
}
std::vector<int> PrimalSupportBank::shortlist(int r,int l,std::size_t limit,const PrimalSupportState* state) const {
    const auto branches=required(r);if(branches.empty())return {};
    std::vector<int> out;
    for(int s:branches) {
        std::vector<int> selected;auto keep=[&](int id){if(std::find(selected.begin(),selected.end(),id)==selected.end() && selected.size()<limit)selected.push_back(id);};
        const auto found=branches_.find({r,l,s});
        if(found!=branches_.end() && !found->second.empty()) {
        const auto& list=found->second;
        // Verify only the physical candidates about to be consumed. Rehashing
        // every retained branch on every neighbor would exhaust its budget.
        for(const auto& p:list)if(checked_in_operation_.insert(p.id).second &&
            path_key(paths_.service_paths.at(p.id))!=retained_keys_.at(p.id))
            throw std::invalid_argument("Primal support bank canonical path changed");
        auto cheapest=[&](auto score){auto it=std::min_element(list.begin(),list.end(),[&](const auto& a,const auto& b){return std::make_pair(score(a),a.id)<std::make_pair(score(b),b.id);});keep(it->id);};
        cheapest([](const auto& p){return p.operating;});
        cheapest([&](const auto& p){
            if(!state)return p.fixed;
            const auto a=resources(d_,paths_.service_paths[p.id]);const DesignLayout layout(d_);double added=0;
            for(int b:a.bases)if(state->design[layout.facility+b]<.5)added+=d_.base_cost[b];
            for(int h:a.lockers)if(state->design[layout.facility+d_.n_l+h]<.5) {double c=INFINITY;for(int k=0;k<d_.n_levels;++k)c=std::min(c,d_.slot_cost(h,k));added+=c;}
            for(auto [u,v]:a.airlines)if(state->design[layout.airline+u*d_.n_nodes()+v]<.5)added+=d_.link_cost(u,v);
            return added;
        });
        cheapest([&](const auto& p){double c=p.load;if(state)for(auto [h,t]:paths_.service_paths[p.id].slot_times)c+=t*state->locker_load.at(h);return c;});
        for(int back=0;back<d_.n_l;++back) {
            const RankedPath* best=nullptr;for(const auto& p:list)if(p.return_base==back &&
                (!best || std::tie(p.operating,p.id)<std::tie(best->operating,best->id)))best=&p;
            if(best)keep(best->id);
        }
        for(const auto& p:list)keep(p.id);
        }
        // Keep the existing low-cost witnesses intact, then add at most two
        // CG observations. Cross-design reduced costs affect ranking only.
        std::vector<const PrimalCgColumn*> observed;
        const PrimalCgColumn* negative=nullptr;const PrimalCgColumn* positive=nullptr;
        for(auto batch=feedback_.rbegin();batch!=feedback_.rend();++batch)for(const auto& column:batch->service) {
            const auto& p=paths_.service_paths.at(column.path);
            if(d_.task_index(p.first_od,p.second_od)!=r || p.b[0]!=l || p.second_od!=s)continue;
            observed.push_back(&column);
            if(std::isfinite(column.reduced_cost) && (!negative || column.reduced_cost<negative->reduced_cost))negative=&column;
            if(column.validated_flow>0 && (!positive || column.validated_flow>positive->validated_flow))positive=&column;
        }
        std::size_t added=0;
        const auto keep_feedback=[&](const PrimalCgColumn* column) {
            if(column && added<2 && std::find(selected.begin(),selected.end(),column->path)==selected.end()) {
                selected.push_back(column->path);++added;
            }
        };
        keep_feedback(negative);keep_feedback(positive);
        for(const auto* column:observed)keep_feedback(column);
        out.insert(out.end(),selected.begin(),selected.end());
    }
    return out;
}
PrimalSupportState PrimalSupportBank::inspect(const Json& result,double deadline) const {
    validate();PrimalSupportState state;state.owner=this;state.status="selection_budget";
    if(expired(deadline))return state;
    try {
        state.design=solution_design(d_,result);const DesignLayout layout(d_);
        if(state.design.size()!=static_cast<std::size_t>(layout.count))throw std::invalid_argument("Wrong design shape");
        for(double x:state.design)if(!std::isfinite(x) || std::min(std::abs(x),std::abs(x-1))>1e-6)throw std::invalid_argument("Nonbinary support design");
        state.head_state.assign(d_.assignment_entities(),0);state.head_resources.resize(d_.assignment_entities());state.locker_load.assign(d_.n_h,0);
        for(int r=0;r<d_.assignment_entities();++r)for(int l=0;l<d_.n_l;++l)if(state.design[layout.assignment_index(r,l)]>.5) {
            if(state.head_state[r])throw std::invalid_argument("Multiple task assignments");state.head_state[r]=l+1;
        }
        const auto& saved=result.contains("_solution")?result.at("_solution"):result;
        const auto& flows=saved.at("flow");const auto& empty=saved.at("empty_flow");
        if(!flows.is_array() || flows.size()>paths_.service_paths.size() || !empty.is_array() || empty.size()>paths_.empty_paths.size())throw std::invalid_argument("Support flow shape changed");
        for(std::size_t i=0;i<flows.size();++i) {
            if(i%1024==0 && expired(deadline))return state;
            const double value=flows.at(i).get<double>();if(!std::isfinite(value)||value<0)throw std::invalid_argument("Invalid support flow");
            if(value==0)continue;
            const auto& p=paths_.service_paths[i];state.positive_service.push_back(static_cast<int>(i));add(state.head_resources.at(d_.task_index(p.first_od,p.second_od)),resources(d_,p));
            for(auto [h,t]:p.slot_times)state.locker_load.at(h)+=t*value;
        }
        for(std::size_t i=0;i<empty.size();++i) {
            const double value=empty.at(i).get<double>();if(!std::isfinite(value)||value<0)throw std::invalid_argument("Invalid empty support flow");
            if(value==0)continue;state.positive_empty.push_back(static_cast<int>(i));const auto& p=paths_.empty_paths[i];
            for(int r=0;r<d_.assignment_entities();++r)if(state.head_resources[r].bases.count(p.b[0]) || state.head_resources[r].bases.count(p.b[1]) ||
                state.head_state[r]==p.b[0]+1 || state.head_state[r]==p.b[1]+1)add_empty(p,state.head_resources[r]);
        }
        state.ready=true;state.status="ready";
    } catch(const std::exception& e) {state.status=std::string("invalid_support: ")+e.what();}
    return state;
}

PrimalBlockRequest PrimalSupportBank::seed(int direction,const PrimalSupportState& state,std::mt19937& rng,double deadline) const {
    validate();state_owner(state,this);PrimalBlockRequest out;out.operator_index=0;
    direction=((direction%3)+3)%3;out.allowed_bases=direction==0?std::vector<int>{0}:direction==1?std::vector<int>{1}:std::vector<int>{0,1};
    for(int l=0;l<d_.n_l;++l)if(std::find(out.allowed_bases.begin(),out.allowed_bases.end(),l)==out.allowed_bases.end())out.restrictions.closed_bases.push_back(l);
    struct Beam {std::vector<int> heads;PrimalResources assets;double gain=0,balance=0,rank=0;};
    std::vector<Beam> beam(1);Beam best;double best_rank=-INFINITY;
    std::vector<int> order(d_.assignment_entities());std::iota(order.begin(),order.end(),0);
    order.erase(std::remove_if(order.begin(),order.end(),[&](int q){return d_.task_rate(q)<=0;}),order.end());
    std::shuffle(order.begin(),order.end(),rng);
    const std::size_t maximum=std::min<std::size_t>(8,d_.assignment_entities());
    for(std::size_t depth=1;depth<=maximum && !expired(deadline);++depth) {
        std::vector<Beam> next;
        for(const auto& item:beam)for(int r:order)if(std::find(item.heads.begin(),item.heads.end(),r)==item.heads.end())for(int l:out.allowed_bases) {
            if(expired(deadline))break;
            const auto candidates=shortlist(r,l,4,&state);if(candidates.empty())continue;
            Beam trial=item;trial.heads.push_back(r);double ground=0,operating=0,balance=0;
            for(int s:required(r)) {
                int chosen=-1;double value=INFINITY;
                for(int id:candidates) {const auto& p=paths_.service_paths[id];if(p.second_od!=s || uses_closed(resources(d_,p),out.restrictions))continue;
                    auto assets=trial.assets;add(assets,resources(d_,p));const double rate=d_.task_rate(r);
                    const double score=rate*(p.cost+d_.fleet_cost*p.time)+fixed_cost(d_,assets)-fixed_cost(d_,trial.assets);
                    if(score<value) {value=score;chosen=id;}}
                if(chosen<0) {trial.heads.clear();break;}
                const auto& p=paths_.service_paths[chosen];const double rate=d_.task_rate(r);
                ground+=rate*(d_.ground_cost[d_.task_first(r)]+(s<0?0:d_.ground_cost[s]));operating+=rate*(p.cost+d_.fleet_cost*p.time);
                balance+=rate*((p.b[0]==0?1:-1)-(p.b[1]==0?1:-1));add(trial.assets,resources(d_,p));
            }
            if(trial.heads.empty())continue;
            trial.gain+=ground-operating-fixed_cost(d_,trial.assets)+fixed_cost(d_,item.assets);trial.balance+=balance;
            trial.rank=trial.gain-(direction==2?.1*std::abs(trial.balance):0);
            next.push_back(std::move(trial));
        }
        std::stable_sort(next.begin(),next.end(),[](const auto& a,const auto& b){return a.rank>b.rank;});
        std::set<std::vector<int>> seen;beam.clear();
        // Saving direction samples from the best 20% before retaining its
        // small beam. This changes actual groups, not just a base filter.
        if(direction==1 && next.size()>1) {
            const auto rcl=std::max<std::size_t>(1,(next.size()+4)/5);std::shuffle(next.begin(),next.begin()+rcl,rng);
        }
        for(auto& item:next) {auto key=item.heads;unique(key);if(seen.insert(key).second)beam.push_back(std::move(item));if(beam.size()==8)break;}
        if(beam.empty())break;
        if(depth==2 || depth==4 || depth==8 || depth==maximum)for(const auto& item:beam)if(item.rank>best_rank) {best=item;best_rank=item.rank;}
    }
    out.heads=best.heads;out.restrictions.change_heads=out.heads;
    out.diagnostics={{"seed_direction",direction},{"beam_width",8},{"joint_head_count",out.heads.size()},
        {"proxy_gain",std::isfinite(best_rank)?Json(best.gain):Json(nullptr)},
        {"positive_single_head_gate",false},{"selection_status",expired(deadline)?"selection_budget":out.heads.empty()?"missing_branch":"ready"}};
    return out;
}

PrimalBlockRequest PrimalSupportBank::request(int op,const PrimalSupportState& state,std::size_t block_size,
    std::mt19937& rng,double deadline,const std::vector<double>* guide,const PrimalSupportState* elite) const {
    validate();state_owner(state,this);PrimalBlockRequest out;out.operator_index=op;out.allowed_bases={0,1};
    if(expired(deadline) || block_size==0 || op<0 || op>5) {out.diagnostics={{"selection_status","selection_budget"}};return out;}
    if(elite)state_owner(*elite,this);
    std::vector<double> score(d_.assignment_entities(),0);std::vector<int> order(d_.assignment_entities());std::iota(order.begin(),order.end(),0);
    order.erase(std::remove_if(order.begin(),order.end(),[&](int q){return d_.task_rate(q)<=0;}),order.end());
    std::shuffle(order.begin(),order.end(),rng);
    const int anchor=order.empty()?-1:order.front();const DesignLayout layout(d_);
    PrimalResources anchor_assets;if(anchor>=0)anchor_assets=state.head_resources[anchor];
    if(anchor>=0 && anchor_assets.lockers.empty())for(int id:shortlist(anchor,0,1,&state))add(anchor_assets,resources(d_,paths_.service_paths[id]));
    int target=-1;
    if(op==2 || op==3) {
        double top=-INFINITY;for(int h=0;h<d_.n_h;++h)if(state.design[layout.facility+d_.n_l+h]>.5) {
            double capacity=0,cost=0;for(int k=0;k<d_.n_levels;++k)if(state.design[layout.capacity+h*d_.n_levels+k]>.5) {capacity=d_.occupancy_limit(h,k);cost=d_.slot_cost(h,k);}
            const double value=op==2?cost/(1+state.locker_load[h]):state.locker_load[h]/std::max(1e-8,capacity);
            if(value>top) {top=value;target=h;}
        }
        if(target>=0 && op==2)out.restrictions.closed_lockers.push_back(target);
    }
    const bool guide_valid=guide && guide->size()>=static_cast<std::size_t>(layout.count) &&
        std::all_of(guide->begin(),guide->begin()+layout.count,[](double x){return std::isfinite(x) && x>=-1e-7 && x<=1+1e-7;});
    for(int r=0;r<d_.assignment_entities();++r) {
        if(expired(deadline))break;
        auto assets=state.head_resources[r];if(assets.lockers.empty())for(int id:shortlist(r,0,1,&state))add(assets,resources(d_,paths_.service_paths[id]));
        if(op==0)score[r]=state.head_state[r]==0?100:0;
        else if(op==1)score[r]=overlap(anchor_assets,assets);
        else if(op==2 || op==3)score[r]=target>=0 && state.head_resources[r].lockers.count(target)?100:overlap(anchor_assets,assets);
        else if(op==4)score[r]=(state.head_state[r]>0?10:0)+overlap(anchor_assets,assets);
        else if(op==5 && elite) {
            score[r]=state.head_state[r]!=elite->head_state[r]?100:0;
            for(int h:assets.lockers)for(int k=0;k<d_.n_levels;++k)score[r]+=std::abs(state.design[layout.capacity+h*d_.n_levels+k]-elite->design[layout.capacity+h*d_.n_levels+k]);
        }
        if(guide_valid)for(int l=0;l<d_.n_l;++l)score[r]+=5*std::abs((*guide)[layout.assignment_index(r,l)]-state.design[layout.assignment_index(r,l)]);
    }
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return score[a]>score[b];});
    if(op==5 && !elite)order.clear();
    for(int r:order) {out.heads.push_back(r);if(out.heads.size()>=block_size)break;}
    if(op==0 || op==2 || op==4 || op==5)out.restrictions.change_heads=out.heads;
    if(op==5 && elite) {out.extra_service=elite->positive_service;out.extra_empty=elite->positive_empty;}
    out.diagnostics={{"operator_index",op},{"master_guided",guide_valid},{"target_locker",target},
        {"selection_status",out.heads.empty()?"no_eligible_group":"ready"}};
    return out;
}

PrimalSupportBlock PrimalSupportBank::close(const PrimalSupportState& state,const PrimalBlockRequest& request,double deadline) const {
    validate();state_owner(state,this);PrimalSupportBlock out;out.restrictions=request.restrictions;out.status="selection_budget";
    out.diagnostics=request.diagnostics;out.diagnostics["bank"]=metadata();
    if(!supported_) {out.status="unsupported_scope";return out;}
    if(expired(deadline))return out;
    std::set<int> heads(request.heads.begin(),request.heads.end());
    for(int r:heads)if(r<0 || r>=d_.assignment_entities())throw std::invalid_argument("Task outside support domain");
    // Forced resource closure releases every current dependent head. Empty
    // relocation dependence was propagated to both endpoint head groups by
    // inspect(), so closing a return base cannot strand a fixed head.
    for(int r=0;r<d_.assignment_entities();++r)if(uses_closed(state.head_resources[r],out.restrictions) ||
        std::find(out.restrictions.closed_bases.begin(),out.restrictions.closed_bases.end(),state.head_state[r]-1)!=out.restrictions.closed_bases.end())heads.insert(r);
    if(heads.empty()) {out.status="no_eligible_group";return out;}
    std::set<int> columns(state.positive_service.begin(),state.positive_service.end());
    std::set<int> empty(state.positive_empty.begin(),state.positive_empty.end());
    columns.insert(request.extra_service.begin(),request.extra_service.end());empty.insert(request.extra_empty.begin(),request.extra_empty.end());
    const std::size_t mandatory=columns.size();PrimalResources release;Json coverage=Json::array();
    for(int r:heads)add(release,state.head_resources[r]);
    for(int id:request.extra_service) {if(id<0 || static_cast<std::size_t>(id)>=paths_.service_paths.size())throw std::invalid_argument("Elite column outside bank");add(release,resources(d_,paths_.service_paths[id]));}
    auto allowed=request.allowed_bases;if(allowed.empty())allowed={0,1};
    std::size_t optional=0,shortlist_expansions=0;
    std::map<Branch,std::vector<int>> witnesses_by_branch;
    std::set<Branch> missing;
    for(int r:heads)for(int l:allowed) {
        if(expired(deadline))return out;
        if(l<0 || l>=d_.n_l)throw std::invalid_argument("Base outside support domain");
        if(std::find(out.restrictions.closed_bases.begin(),out.restrictions.closed_bases.end(),l)!=out.restrictions.closed_bases.end())continue;
        const auto candidates=shortlist(r,l,std::min<std::size_t>(8,std::max<std::size_t>(1,request.paths_per_branch)),&state);
        for(int branch:required(r)) {
            std::vector<int> witnesses;for(int id:candidates)if(paths_.service_paths[id].second_od==branch && !uses_closed(resources(d_,paths_.service_paths[id]),out.restrictions))witnesses.push_back(id);
            if(witnesses.empty() && request.paths_per_branch<8) {
                ++shortlist_expansions;
                for(int id:shortlist(r,l,8,&state))if(paths_.service_paths[id].second_od==branch &&
                    !uses_closed(resources(d_,paths_.service_paths[id]),out.restrictions))witnesses.push_back(id);
            }
            if(witnesses.empty()) {
                if(!request.complete_missing_support) {out.status="missing_branch";out.diagnostics["missing_branch"]={r,l,branch};out.diagnostics["all_positive_branches_preserved"]=false;return out;}
                missing.insert({r,l,branch});
            }
            witnesses_by_branch[{r,l,branch}]=std::move(witnesses);
        }
    }
    // A ranking bank is only a shortlist. Scan the registered physical pool once for
    // all missing branches, rather than treating a missing ranked witness as
    // a reason to discard an otherwise useful joint neighborhood. Leave time
    // for materializing the block even if this optional scan is interrupted.
    const std::size_t missing_before=missing.size(),index_before=indexed_;std::size_t completion_scanned=0;
    const double completion_deadline=wall_time()+.6*std::max(0.0,deadline-wall_time());
    bool catalogue_exhausted=false,missing_branch_search_complete=false;
    if(!missing.empty() && request.indexed_support_completion) {
        extend_index(completion_deadline);
        // Merge only missing-branch ID lists in the same global ID order as
        // the legacy full scan. Each branch therefore keeps its first usable
        // witness; unrelated branch IDs need no catalogue visit.
        using Cursor=std::tuple<int,Branch,std::size_t>;
        std::priority_queue<Cursor,std::vector<Cursor>,std::greater<Cursor>> next;
        for(const auto& key:missing) {
            const auto found=complete_branches_.find(key);
            if(found!=complete_branches_.end() && !found->second.empty())next.emplace(found->second.front(),key,0);
        }
        bool interrupted=false;
        while(!next.empty()) {
            if(completion_scanned%64==0 && expired(completion_deadline)) {interrupted=true;break;}
            const auto [id,key,offset]=next.top();next.pop();++completion_scanned;
            const auto& path=paths_.service_paths.at(id);
            if(Branch{d_.task_index(path.first_od,path.second_od),path.b[0],path.second_od}!=key)
                throw std::invalid_argument("Complete branch index canonical path changed");
            if(!uses_closed(resources(d_,path),out.restrictions)) {
                witnesses_by_branch[key].push_back(id);missing.erase(key);
            } else {
                const auto& ids=complete_branches_.at(key);
                if(offset+1<ids.size())next.emplace(ids[offset+1],key,offset+1);
            }
        }
        missing_branch_search_complete=!interrupted && indexed_==paths_.service_paths.size();
    } else if(!missing.empty()) {
        for(std::size_t id=0;id<paths_.service_paths.size() && !missing.empty();++id) {
            if(id%64==0 && expired(completion_deadline))break;
            ++completion_scanned;const auto& path=paths_.service_paths[id];
            const Branch key={d_.task_index(path.first_od,path.second_od),path.b[0],path.second_od};
            if(missing.count(key) && !uses_closed(resources(d_,path),out.restrictions)) {
                witnesses_by_branch[key].push_back(static_cast<int>(id));missing.erase(key);
            }
        }
        catalogue_exhausted=completion_scanned==paths_.service_paths.size();
        missing_branch_search_complete=catalogue_exhausted;
    }
    const bool forced_closure=!out.restrictions.closed_bases.empty() || !out.restrictions.closed_lockers.empty() ||
        !out.restrictions.closed_airlines.empty();
    std::set<std::pair<int,int>> forbidden(out.restrictions.forbidden_origins.begin(),out.restrictions.forbidden_origins.end());
    std::set<std::pair<int,int>> retained_incomplete;
    for(const auto& key:missing) {
        const int r=key[0],l=key[1];
        // A feasible zero-throughput incumbent can retain a redundant origin
        // assignment without any path witness. Preserve it in intensification;
        // the unchanged original task rows still forbid positive throughput.
        if(!forced_closure && state.head_state[r]==l+1)retained_incomplete.emplace(r,l);
        else forbidden.emplace(r,l);
    }
    out.restrictions.forbidden_origins.assign(forbidden.begin(),forbidden.end());
    for(const auto& [key,witnesses]:witnesses_by_branch) {
        if(forbidden.count({key[0],key[1]}))continue;
        // One witness is mandatory for each enabled branch. Optional breadth
        // may shrink, but its soft cap never removes that witness.
        for(std::size_t i=0;i<witnesses.size();++i)if(i==0 || (optional<request.optional_column_limit && columns.size()<30000)) {
            const int id=witnesses[i];if(columns.insert(id).second && i>0)++optional;
            add(release,resources(d_,paths_.service_paths[id]));
        }
        coverage.push_back({{"head",key[0]},{"departure",key[1]},{"branch",key[2]},{"witnesses",witnesses.size()}});
    }
    // Every tiny L2 relocation column is allowed; service circuits can also
    // balance one another without any individually paired relocation.
    for(std::size_t i=0;i<paths_.empty_paths.size();++i) {empty.insert(static_cast<int>(i));add_empty(paths_.empty_paths[i],release);}
    release.bases.insert(out.restrictions.closed_bases.begin(),out.restrictions.closed_bases.end());
    release.lockers.insert(out.restrictions.closed_lockers.begin(),out.restrictions.closed_lockers.end());
    release.airlines.insert(out.restrictions.closed_airlines.begin(),out.restrictions.closed_airlines.end());
    for(int r:heads)if(state.head_state[r]>0)release.bases.insert(state.head_state[r]-1);
    out.coupled.heads.assign(heads.begin(),heads.end());out.coupled.bases.assign(release.bases.begin(),release.bases.end());
    out.coupled.lockers.assign(release.lockers.begin(),release.lockers.end());out.coupled.airlines.assign(release.airlines.begin(),release.airlines.end());
    out.service_indices.assign(columns.begin(),columns.end());out.empty_indices.assign(empty.begin(),empty.end());
    out.ready=!expired(deadline);out.status=out.ready?"ready":"selection_budget";
    out.diagnostics["branch_coverage"]=std::move(coverage);out.diagnostics["all_positive_branches_preserved"]=retained_incomplete.empty();
    out.diagnostics["support_completion"]={{"enabled",request.complete_missing_support},{"missing_shortlist_branches",missing_before},
        {"completed_branches",missing_before-missing.size()},{"catalogue_paths_scanned",completion_scanned},
        {"catalogue_exhausted",catalogue_exhausted},{"scan_domain_complete",paths_.complete},
        {"indexed",request.indexed_support_completion},{"index_entries_added",indexed_-index_before},
        {"indexed_paths",indexed_},{"index_complete",indexed_==paths_.service_paths.size()},
        {"missing_branch_search_complete",missing_branch_search_complete},
        {"unknown_support",!missing.empty() && (!paths_.complete || !missing_branch_search_complete)},
        {"scan_domain","registered_physical_paths"},{"forbidden_origins",forbidden},
        {"retained_incumbent_origins_without_full_support",retained_incomplete},
        {"original_task_rows_preserved",true},{"global_inference",false},
        {"coverage_scope","locally_enabled_origin_states"}};
    out.diagnostics["shortlist_expansions"]=shortlist_expansions;
    out.diagnostics["mandatory_positive_columns"]=mandatory;out.diagnostics["optional_columns"]=optional;
    out.diagnostics["released_heads"]=out.coupled.heads;out.diagnostics["released_lockers"]=out.coupled.lockers;
    out.diagnostics["service_columns"]=columns.size();out.diagnostics["empty_columns"]=empty.size();
    out.diagnostics["soft_column_limit_exceeded"]=columns.size()>30000;
    out.diagnostics["closure_complete"]=out.ready;out.diagnostics["local_bound_is_global"]=false;
    return out;
}
} // namespace mlnd
