#include "mlnd/primal_portfolio.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace mlnd {
namespace {
using Branch = std::array<int,3>; // task ID, departure base, second order (-1 for single)
struct RankedPath { int index; double score; };
struct Assets {
    std::set<int> bases, lockers;
    std::set<std::pair<int,int>> airlines;
};
struct PathBank {
    std::map<Branch,std::vector<RankedPath>> branches;
    std::map<BasePair,int> returns;
    std::size_t scanned=0;
    bool complete=false;
};
struct Selection {
    std::set<int> service, empty, heads;
    Assets assets;
    Json groups=Json::array();
};
bool supported(const Input& d,const PathCollection& paths) {
    return d.n_l==2 && d.max_orders==2 && d.preprocessed_demand && d.origin_assignment() &&
        d.flow_policy=="aggregate" && !d.first_leg_consistency && paths.complete && paths.support_complete;
}
double time_left(double deadline) {return std::max(0.0,deadline-wall_time());}
double rate(const Input& d,int q) {return d.task_rate(q);}
std::vector<int> required_branches(const Input& d,int r) {
    std::vector<int> branches;
    if(d.task_rate(r)>0)branches.push_back(d.task_second(r));
    return branches;
}
void add_path_assets(const Input& d,const ServicePath& p,Assets& assets) {
    assets.bases.insert(p.b[0]);assets.bases.insert(p.b[1]);
    for(const auto& leg:p.legs)for(int node:leg)
        if(node>=d.n_l)assets.lockers.insert(node-d.n_l);
    for(auto [u,v,count]:p.arc_counts)if(count)assets.airlines.insert({u,v});
}
void merge_assets(Assets& into,const Assets& from) {
    into.bases.insert(from.bases.begin(),from.bases.end());
    into.lockers.insert(from.lockers.begin(),from.lockers.end());
    into.airlines.insert(from.airlines.begin(),from.airlines.end());
}
double asset_cost(const Input& d,const Assets& assets) {
    double result=0;
    for(int l:assets.bases)result+=d.base_cost[l];
    for(int h:assets.lockers) {
        double cheapest=INFINITY;
        for(int z=0;z<d.n_levels;++z)cheapest=std::min(cheapest,d.slot_cost(h,z));
        result+=cheapest;
    }
    for(auto [u,v]:assets.airlines)result+=d.link_cost(u,v);
    return result;
}
int return_path(const PathBank& bank,const ServicePath& p) {
    if(p.b[0]==p.b[1])return -1;
    const auto found=bank.returns.find({p.b[1],p.b[0]});
    return found==bank.returns.end()?-2:found->second;
}
double path_operating(const Input& d,const PathCollection& paths,const ServicePath& p,int empty) {
    double cost=p.cost+d.fleet_cost*p.time;
    if(empty>=0)cost+=paths.empty_paths[empty].cost+d.fleet_cost*paths.empty_paths[empty].time;
    return cost;
}
PathBank make_bank(const Input& d,const PathCollection& paths,std::size_t per_branch,double deadline) {
    PathBank bank;
    for(std::size_t i=0;i<paths.empty_paths.size();++i) {
        const auto& p=paths.empty_paths[i];const auto old=bank.returns.find(p.b);
        if(old==bank.returns.end() || p.cost+d.fleet_cost*p.time<
            paths.empty_paths[old->second].cost+d.fleet_cost*paths.empty_paths[old->second].time)
            bank.returns[p.b]=static_cast<int>(i);
    }
    const std::size_t limit=std::max<std::size_t>(1,std::min<std::size_t>(16,per_branch*4));
    for(std::size_t i=0;i<paths.service_paths.size();++i) {
        if(i%256==0 && time_left(deadline)<=0)return bank;
        const auto& p=paths.service_paths[i];++bank.scanned;
        const double demand=rate(d,d.task_index(p.first_od,p.second_od));
        if(demand<=0)continue;
        const int empty=return_path(bank,p);if(empty==-2)continue;
        Assets assets;add_path_assets(d,p,assets);
        if(empty>=0)assets.airlines.insert({p.b[1],p.b[0]});
        const double score=demand*path_operating(d,paths,p,empty)+asset_cost(d,assets);
        if(!std::isfinite(score))continue;
        auto& candidates=bank.branches[{d.task_index(p.first_od,p.second_od),p.b[0],p.second_od}];
        if(candidates.size()==limit && std::tie(score,i)>=
            std::tie(candidates.back().score,candidates.back().index))continue;
        candidates.push_back({static_cast<int>(i),score});
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){
            return std::tie(a.score,a.index)<std::tie(b.score,b.index);
        });
        if(candidates.size()>limit)candidates.pop_back();
    }
    bank.complete=true;return bank;
}
bool add_group(const Input& d,const PathCollection& paths,const PathBank& bank,
               int r,int l,std::size_t max_lockers,Selection& selected,double& benefit,
               const Assets* mandatory_release=nullptr) {
    const auto branches=required_branches(d,r);
    if(branches.empty())return false;
    Selection trial=selected;
    double operating=0,ground=0;
    for(int s:branches) {
        const auto found=bank.branches.find({r,l,s});if(found==bank.branches.end())return false;
        double best=INFINITY;int chosen=-1,chosen_empty=-1;Assets chosen_assets;
        for(const auto& candidate:found->second) {
            const auto& p=paths.service_paths[candidate.index];const int empty=return_path(bank,p);
            Assets assets=trial.assets;add_path_assets(d,p,assets);
            if(empty>=0)assets.airlines.insert({p.b[1],p.b[0]});
            Assets release=assets;
            if(mandatory_release)merge_assets(release,*mandatory_release);
            if(release.lockers.size()>max_lockers)continue;
            const double value=rate(d,r)*path_operating(d,paths,p,empty)+
                asset_cost(d,assets)-asset_cost(d,trial.assets);
            if(value<best) {best=value;chosen=candidate.index;chosen_empty=empty;chosen_assets=std::move(assets);}
        }
        if(chosen<0)return false;
        const auto& p=paths.service_paths[chosen];
        operating+=rate(d,r)*path_operating(d,paths,p,chosen_empty);
        ground+=rate(d,r)*(d.ground_cost[d.task_first(r)]+(s<0?0:d.ground_cost[s]));
        trial.assets=std::move(chosen_assets);trial.service.insert(chosen);
        if(chosen_empty>=0)trial.empty.insert(chosen_empty);
    }
    benefit=ground-operating-asset_cost(d,trial.assets)+asset_cost(d,selected.assets);
    trial.heads.insert(r);
    trial.groups.push_back({{"head",r},{"departure",l},{"required_branches",branches},
        {"complete",true},{"proxy_benefit",benefit}});
    selected=std::move(trial);return true;
}
Selection select_seed(const Input& d,const PathCollection& paths,const PathBank& bank,
                     const PrimalPortfolioOptions& settings,std::size_t candidate,double deadline) {
    Selection selected;
    const int base_filter=candidate%3==0?-1:static_cast<int>((candidate-1)%2);
    while(selected.heads.size()<settings.max_heads && time_left(deadline)>0) {
        double best=-INFINITY;Selection winner;bool found=false;
        for(int r=0;r<d.assignment_entities();++r)if(!selected.heads.count(r))for(int l=0;l<d.n_l;++l) {
            if(base_filter>=0 && l!=base_filter)continue;
            Selection trial=selected;double benefit=0;
            if(add_group(d,paths,bank,r,l,settings.max_lockers,trial,benefit) && benefit>best) {
                best=benefit;winner=std::move(trial);found=true;
            }
        }
        if(!found || best<=0)break;
        selected=std::move(winner);
    }
    return selected;
}
Json selection_record(const Selection& selected,const PathBank& bank) {
    return {{"service_indices",selected.service},{"empty_indices",selected.empty},
        {"head_groups",selected.groups},{"lockers",selected.assets.lockers},
        {"bases",selected.assets.bases},{"airlines",selected.assets.airlines},
        {"catalogue_paths_scanned",bank.scanned},{"catalogue_scan_complete",bank.complete},
        {"all_positive_branches_preserved",true},{"task_rates_changed",false}};
}
Json fallback(const Input& d,const PathCollection& paths) {
    Json solution=all_courier_result(d,paths);
    solution.erase("bound");solution.erase("gap");
    return {{"validated",true},{"improved",false},{"objective",solution.at("objective")},
        {"solution",std::move(solution)},{"status","time_limit"},{"source","all_courier_fallback"},
        {"attempts",Json::array()},{"global_lower_bound_available",false},{"timings",{{"total",0.0}}}};
}
bool valid_settings(const PrimalPortfolioOptions& settings) {
    return settings.max_heads>0 && settings.max_lockers>0 && settings.max_candidates>0 &&
        settings.paths_per_branch>0 && std::isfinite(settings.candidate_seconds) && settings.candidate_seconds>0;
}
}
Json solve_primal_seed_portfolio(const Input& d, const PathCollection& paths,
    const SolverOptions& options,double deadline,const Json* incumbent,const PrimalPortfolioOptions& settings) {
    const double begin=wall_time();Json result=fallback(d,paths);
    auto finish=[&](){result["timings"]["total"]=wall_time()-begin;return result;};
    if(!std::isfinite(deadline) || !std::isfinite(options.seconds) || options.seconds<=0)return finish();
    deadline=std::min(deadline,begin+options.seconds);
    if(time_left(deadline)<=0)return finish();
    if(!supported(d,paths)) {result["status"]="unsupported_scope";return finish();}
    if(!valid_settings(settings)) {result["status"]="empty_or_invalid_portfolio";return finish();}
    const auto bank=make_bank(d,paths,settings.paths_per_branch,
        std::min(deadline,wall_time()+.25*time_left(deadline)));
    result["timings"]["selection"]=wall_time()-begin;
    double reference=result.at("objective");
    std::set<std::vector<int>> seen;
    for(std::size_t i=0;i<settings.max_candidates && time_left(deadline)>.01;++i) {
        const double selection_begin=wall_time();
        const auto selected=select_seed(d,paths,bank,settings,i,deadline);
        result["timings"]["selection"]=result["timings"]["selection"].get<double>()+wall_time()-selection_begin;
        const std::vector<int> service(selected.service.begin(),selected.service.end());
        const std::vector<int> empty(selected.empty.begin(),selected.empty.end());
        if(service.empty() || !seen.insert(service).second)continue;
        const double allowance=std::min(settings.candidate_seconds,
            time_left(deadline)/static_cast<double>(settings.max_candidates-i));
        if(allowance<=.01)break;
        const Json* start=result.at("attempts").empty()?incumbent:&result.at("solution");
        auto attempt=solve_restricted_heuristic(d,paths,options,std::min(deadline,wall_time()+allowance),
            start,-1,&service,&empty);
        if(result.at("attempts").empty() && incumbent && attempt.at("diagnostics").contains("incumbent_validation") &&
            attempt.at("diagnostics").at("incumbent_validation").value("passed",false))
            reference=incumbent->value("objective",reference);
        attempt["selection"]=selection_record(selected,bank);attempt["allocated_seconds"]=allowance;
        if(attempt.value("validated",false) && attempt.at("objective").get<double>()<result.at("objective").get<double>()) {
            result["solution"]=attempt.at("solution");result["objective"]=attempt.at("objective");
            result["source"]="primal_seed_portfolio";
        }
        attempt.erase("solution");result["attempts"].push_back(std::move(attempt));
    }
    result["improved"]=result.at("objective").get<double>()<reference-1e-7;
    result["status"]=time_left(deadline)<=.01?"time_limit":"portfolio_complete";
    return finish();
}

Json solve_primal_coupled_neighborhood(const Input& d,const PathCollection& paths,
    const SolverOptions& options,double deadline,const Json& incumbent,
    const std::vector<double>* guide,std::size_t rotation,const PrimalPortfolioOptions& settings) {
    const double begin=wall_time();Json result=fallback(d,paths);
    auto finish=[&](){result["timings"]["total"]=wall_time()-begin;return result;};
    if(!std::isfinite(deadline)||!std::isfinite(options.seconds)||options.seconds<=0)return finish();
    deadline=std::min(deadline,begin+options.seconds);
    if(time_left(deadline)<=0)return finish();
    if(!supported(d,paths)) {result["status"]="unsupported_scope";return finish();}
    if(!valid_settings(settings)) {result["status"]="empty_or_invalid_portfolio";return finish();}
    std::vector<double> center;
    try {center=solution_design(d,incumbent);}
    catch(const std::exception&) {result["status"]="invalid_incumbent_design";return finish();}
    if(std::any_of(center.begin(),center.end(),[](double value) {
        return !std::isfinite(value) || value<0 || value>1;
    })) {result["status"]="invalid_incumbent_design";return finish();}
    // A released head must be able to retire its previous resources, not
    // merely use new ones. Positive incumbent columns are retained below by
    // the restricted solver, but retaining a column does not release assets.
    std::vector<Assets> incumbent_assets(d.assignment_entities());
    try {
        const auto& saved=incumbent.contains("_solution")?incumbent.at("_solution"):incumbent;
        const auto& flows=saved.at("flow");const auto& empty=saved.at("empty_flow");
        if(!flows.is_array() || flows.size()>paths.service_paths.size() ||
           !empty.is_array() || empty.size()>paths.empty_paths.size())
            throw std::invalid_argument("Incumbent flow shape differs from catalogue");
        for(std::size_t i=0;i<flows.size();++i) {
            if(i%256==0 && time_left(deadline)<=0)return finish();
            const double flow=flows.at(i).get<double>();
            if(!std::isfinite(flow)||flow<0)throw std::invalid_argument("Invalid incumbent flow");
            if(flow>0)add_path_assets(d,paths.service_paths[i],incumbent_assets.at(d.task_index(paths.service_paths[i].first_od,paths.service_paths[i].second_od)));
        }
        for(std::size_t i=0;i<empty.size();++i) {
            const double flow=empty.at(i).get<double>();
            if(!std::isfinite(flow)||flow<0)throw std::invalid_argument("Invalid incumbent empty flow");
            if(flow<=0)continue;
            const auto& path=paths.empty_paths[i];
            for(auto& assets:incumbent_assets)
                if(assets.bases.count(path.b[0])||assets.bases.count(path.b[1])) {
                    assets.bases.insert(path.b[0]);assets.bases.insert(path.b[1]);
                    for(auto [u,v,count]:path.arc_counts)if(count)assets.airlines.insert({u,v});
                }
        }
    } catch(const std::exception&) {result["status"]="invalid_incumbent_support";return finish();}
    const DesignLayout layout(d);
    bool use_guide=guide && guide->size()>=static_cast<std::size_t>(layout.count);
    if(use_guide)for(int j=0;j<layout.count;++j)
        use_guide=use_guide&&std::isfinite((*guide)[j])&&(*guide)[j]>=-1e-7&&(*guide)[j]<=1+1e-7;
    std::vector<std::pair<double,int>> ranked;
    for(int r=0;r<d.assignment_entities();++r) {
        if(d.task_rate(r)<=0)continue;
        double disagreement=0;
        if(use_guide)for(int l=0;l<d.n_l;++l)
            disagreement+=std::abs((*guide)[layout.assignment_index(r,l)]-center[layout.assignment_index(r,l)]);
        ranked.emplace_back(disagreement,r);
    }
    const std::size_t offset=d.assignment_entities()?rotation%static_cast<std::size_t>(d.assignment_entities()):0;
    std::sort(ranked.begin(),ranked.end(),[&](const auto& a,const auto& b) {
        if(a.first!=b.first)return a.first>b.first;
        return (a.second+d.assignment_entities()-offset)%d.assignment_entities()<(b.second+d.assignment_entities()-offset)%d.assignment_entities();
    });
    const auto bank=make_bank(d,paths,settings.paths_per_branch,
        std::min(deadline,wall_time()+.25*time_left(deadline)));
    Selection selected;CoupledPrimalNeighborhood neighborhood;Assets mandatory;
    std::size_t resource_limit_skips=0;
    for(const auto& [score,r]:ranked) {
        (void)score;
        if(neighborhood.heads.size()>=settings.max_heads || time_left(deadline)<=0)break;
        Assets next_mandatory=mandatory;merge_assets(next_mandatory,incumbent_assets[r]);
        if(next_mandatory.lockers.size()>settings.max_lockers) {++resource_limit_skips;continue;}
        Selection accepted=selected;bool found=false;
        for(int l=0;l<d.n_l;++l) {
            Selection trial=accepted;double benefit=0;
            if(add_group(d,paths,bank,r,l,settings.max_lockers,trial,benefit,&next_mandatory)) {
                accepted=std::move(trial);found=true;
            }
        }
        if(found) {
            selected=std::move(accepted);mandatory=std::move(next_mandatory);
            neighborhood.heads.push_back(r);
        }
    }
    result["resource_limit_heads_skipped"]=resource_limit_skips;
    if(neighborhood.heads.empty()) {result["status"]="no_complete_resource_block";return finish();}
    Assets release=selected.assets;merge_assets(release,mandatory);
    neighborhood.lockers.assign(release.lockers.begin(),release.lockers.end());
    neighborhood.bases.assign(release.bases.begin(),release.bases.end());
    neighborhood.airlines.assign(release.airlines.begin(),release.airlines.end());
    const std::vector<int> service(selected.service.begin(),selected.service.end());
    const std::vector<int> empty(selected.empty.begin(),selected.empty.end());
    const double selection_seconds=wall_time()-begin;
    if(time_left(deadline)<=.01)return finish();
    auto attempt=solve_restricted_heuristic(d,paths,options,deadline,&incumbent,-1,
        &service,&empty,false,false,&neighborhood);
    result=attempt;
    attempt["selection"]=selection_record(selected,bank);attempt["guide_used"]=use_guide;
    attempt["incumbent_resource_closure"]={{"lockers",mandatory.lockers},{"bases",mandatory.bases},
        {"airlines",mandatory.airlines},{"strictly_positive_paths",true}};
    attempt["resource_limit_heads_skipped"]=resource_limit_skips;
    attempt["rotation"]=rotation;attempt.erase("solution");
    result["attempts"]=Json::array({std::move(attempt)});
    result["timings"]["selection"]=selection_seconds;
    if(result.value("improved",false))result["source"]="primal_coupled_neighborhood";
    return finish();
}
} // namespace mlnd
