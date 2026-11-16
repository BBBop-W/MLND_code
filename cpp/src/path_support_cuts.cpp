#include "mlnd/path_support_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
void validate_domain(const Input& d) {
    if (!d.preprocessed_demand || !d.origin_assignment() ||
        d.flow_policy != "aggregate" || d.first_leg_consistency ||
        d.route_mode != "direct" || d.has_candidate_routes ||
        d.max_orders < 1 || d.max_orders > 2)
        throw std::invalid_argument("Path support dominance requires fixed preprocessed origin demand, direct aggregate paths and no first-leg consistency");
    if (d.n_od < 1 || d.n_l < 1 || d.n_h < 0 ||
        d.single_rates.size() != static_cast<std::size_t>(d.n_od) ||
        d.pair_rates.rows != d.n_od || d.pair_rates.cols != d.n_od ||
        d.pair_rates.values.size() != static_cast<std::size_t>(d.n_od) * d.n_od ||
        d.allowed_links.rows != d.n_nodes() || d.allowed_links.cols != d.n_nodes())
        throw std::invalid_argument("Invalid fixed task arrays for path support dominance");
    for (double value : d.single_rates)
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("Path support requires finite nonnegative singleton rates");
    for (double value : d.pair_rates.values)
        if (!std::isfinite(value) || value < 0 || (d.max_orders == 1 && value != 0))
            throw std::invalid_argument("Path support requires finite nonnegative pair rates, zero for single mode");
}

// Adapted from the existing assignment-support blocker: first certify the
// full union, then greedily cover paths and delete redundant chosen arcs.
// Shrinking is optional; expiration returns a set whose coverage is known.
std::vector<int> blocking_set(const std::vector<std::vector<int>>& closed_choices,
                             int arc_count, double deadline) {
    std::vector<std::vector<int>> covered_by(arc_count);
    std::vector<int> counts(arc_count, 0), full;
    for (std::size_t p = 0; p < closed_choices.size(); ++p) {
        if (closed_choices[p].empty())
            throw std::logic_error("An open path cannot define a path support blocker");
        for (int e : closed_choices[p]) {
            if (counts[e]++ == 0) full.push_back(e);
            covered_by[e].push_back(static_cast<int>(p));
        }
    }
    std::sort(full.begin(), full.end());
    if (wall_time() >= deadline) return full;
    std::vector<unsigned char> uncovered(closed_choices.size(), 1), chosen(arc_count, 0);
    std::size_t remaining = closed_choices.size();
    while (remaining > 0) {
        if (wall_time() >= deadline) return full;
        const auto best = std::max_element(counts.begin(), counts.end());
        if (best == counts.end() || *best == 0)
            throw std::logic_error("Incomplete path support blocker coverage");
        const int e = static_cast<int>(best - counts.begin());
        chosen[e] = 1;
        for (int p : covered_by[e]) if (uncovered[p]) {
            uncovered[p] = 0; --remaining;
            for (int other : closed_choices[p]) --counts[other];
        }
    }
    std::vector<int> hits(closed_choices.size(), 0);
    for (std::size_t p = 0; p < closed_choices.size(); ++p)
        for (int e : closed_choices[p]) hits[p] += chosen[e];
    for (auto it = full.rbegin(); it != full.rend(); ++it) {
        if (wall_time() >= deadline) break;
        const int e = *it;
        if (!chosen[e]) continue;
        const bool redundant = std::all_of(covered_by[e].begin(), covered_by[e].end(),
            [&](int p) { return hits[p] >= 2; });
        if (redundant) {
            chosen[e] = 0;
            for (int p : covered_by[e]) --hits[p];
        }
    }
    std::vector<int> result;
    for (int e : full) if (chosen[e]) result.push_back(e);
    return result;
}

PathSupportCut make_cut(const Input& d, const PathSupportGroup& group,
                        const std::vector<int>& block, bool is_static) {
    const DesignLayout layout(d);
    PathSupportCut cut;
    cut.head = group.head; cut.origin = group.origin; cut.branch = group.branch;
    cut.row = {is_static ? "path_support_static" : "path_support_dominance", 'L', 0,
               {{layout.assignment_index(d.task_index(group.head,group.branch), group.origin), 1}}};
    std::ostringstream key;
    // Independently assigned siblings need distinct cut keys.
    key << "path_support:" << d.task_index(group.head,group.branch) << ':' << group.origin;
    for (int e : block) {
        cut.row.coefficients.emplace_back(layout.airline + e, -1);
        cut.blocking_arcs.emplace_back(e / d.n_nodes(), e % d.n_nodes());
        key << ':' << e;
    }
    cut.key = key.str(); cut.valid = true;
    cut.metadata = {{"key",cut.key},{"head",cut.head},{"origin",cut.origin},
        {"branch",cut.branch},{"task",d.task_index(group.head,group.branch)},{"assignment_scope","task"},{"rate",group.rate},{"static",is_static},
        {"blocking_arcs",cut.blocking_arcs},{"blocking_count",block.size()},
        {"paths_checked",group.paths.size()},{"complete_domain",true},
        {"validity_scope","optimality_preserving_assignment_canonicalization"},
        {"preserves_all_original_assignment_encodings",false},
        {"master_added_variables",0},{"all_return_bases",true}};
    return cut;
}
} // namespace

PathSupportCache build_path_support_cache(const Input& d,
        const PathCollection& paths, double deadline) {
    const double started = wall_time();
    validate_domain(d);
    if (std::isnan(deadline)) throw std::invalid_argument("Invalid path support deadline");
    PathSupportCache cache;
    cache.n_od=d.n_od; cache.n_l=d.n_l; cache.n_nodes=d.n_nodes();
    auto finish = [&] {
        if (!cache.complete_domain) cache.static_cuts.clear();
        std::size_t cached_paths=0;
        for (const auto& group : cache.groups) cached_paths+=group.paths.size();
        cache.seconds=wall_time()-started;
        cache.metadata={{"complete_domain",cache.complete_domain},{"timed_out",cache.timed_out},
            {"seconds",cache.seconds},{"paths_checked",cache.paths_checked},
            {"groups",cache.groups.size()},{"cached_paths",cached_paths},
            {"static_cuts",cache.static_cuts.size()},
            {"source","complete_physical_catalogue_all_return_bases"},
            {"catalogue_complete",paths.complete},{"support_complete",paths.support_complete},
            {"validity_scope","optimality_preserving_assignment_canonicalization"}};
        return cache;
    };
    if (!paths.complete || !paths.support_complete) return finish();
    if (wall_time() >= deadline) {cache.timed_out=true;return finish();}
    using GroupKey=std::tuple<int,int,int>;
    std::map<GroupKey,std::size_t> index;
    for (int r=0;r<d.n_od;++r) for (int l=0;l<d.n_l;++l) {
        for (int s=-1;s<d.n_od;++s) {
            const double rate=s<0?d.single_rates[r]:d.pair_rates(r,s);
            if (rate<=0) continue;
            index.emplace(GroupKey{r,l,s},cache.groups.size());
            cache.groups.push_back({r,l,s,rate,{}});
        }
    }
    for (const auto& path : paths.service_paths) {
        if (wall_time() >= deadline) {cache.timed_out=true;return finish();}
        ++cache.paths_checked;
        if (path.first_od<0 || path.first_od>=d.n_od || path.second_od < -1 ||
            path.second_od>=d.n_od || path.b[0]<0 || path.b[0]>=d.n_l ||
            path.b[1]<0 || path.b[1]>=d.n_l)
            throw std::invalid_argument("Invalid path indices in complete support catalogue");
        const auto found=index.find(GroupKey{path.first_od,path.b[0],path.second_od});
        if (found==index.end()) continue; // A zero-rate branch imposes no support requirement.
        std::vector<int> arcs;
        for (const auto& [u,v,count] : path.arc_counts) {
            if (u<0 || v<0 || u>=d.n_nodes() || v>=d.n_nodes() || count<=0 || !d.allowed_links(u,v))
                throw std::invalid_argument("Invalid arc in complete support catalogue");
            arcs.push_back(u*d.n_nodes()+v);
        }
        std::sort(arcs.begin(),arcs.end());
        arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
        cache.groups[found->second].paths.push_back(std::move(arcs));
    }
    std::set<std::pair<int,int>> fixed;
    for (auto& group : cache.groups) {
        if (wall_time() >= deadline) {cache.timed_out=true;return finish();}
        std::sort(group.paths.begin(),group.paths.end());
        group.paths.erase(std::unique(group.paths.begin(),group.paths.end()),group.paths.end());
        if (group.paths.empty() && fixed.emplace(d.task_index(group.head,group.branch),group.origin).second)
            cache.static_cuts.push_back(make_cut(d,group,{},true));
    }
    cache.complete_domain=true;
    return finish();
}

PathSupportSeparation separate_path_support(const Input& d,
        const PathSupportCache& cache, const std::vector<double>& design,
        double deadline, std::size_t max_cuts) {
    const double started=wall_time();
    validate_domain(d);
    const DesignLayout layout(d);
    if (cache.n_od!=d.n_od || cache.n_l!=d.n_l || cache.n_nodes!=d.n_nodes() ||
        design.size()<static_cast<std::size_t>(layout.count) || std::isnan(deadline))
        throw std::invalid_argument("Invalid path support separation arguments");
    for (int j=0;j<layout.count;++j)
        if (!std::isfinite(design[j]) || design[j]<-1e-6 || design[j]>1+1e-6 ||
            std::abs(design[j]-std::round(design[j]))>1e-6)
            throw std::invalid_argument("Path support separation requires an integer design");
    PathSupportSeparation report;
    report.complete_domain=cache.complete_domain;
    auto finish=[&] {
        report.seconds=wall_time()-started;
        report.metadata={{"complete_domain",report.complete_domain},{"timed_out",report.timed_out},
            {"seconds",report.seconds},{"checks",report.checks},
            {"assigned_groups_checked",report.assigned_groups_checked},
            {"supported_groups",report.supported_groups},{"blocked_groups",report.blocked_groups},
            {"cuts",report.cuts.size()},{"master_added_variables",0},
            {"validity_scope","optimality_preserving_assignment_canonicalization"}};
        return report;
    };
    if (!cache.complete_domain || max_cuts==0) return finish();
    std::set<std::string> emitted;
    for (const auto& group : cache.groups) {
        if (wall_time()>=deadline) {report.timed_out=true;break;}
        if (design[layout.assignment_index(d.task_index(group.head,group.branch),group.origin)]<.5) continue;
        ++report.assigned_groups_checked;
        std::vector<std::vector<int>> closed_choices;
        bool supported=false,scan_complete=true;
        for (const auto& path : group.paths) {
            if (wall_time()>=deadline) {report.timed_out=true;scan_complete=false;break;}
            ++report.checks;
            std::vector<int> closed;
            for (int e : path) if (design[layout.airline+e]<.5) closed.push_back(e);
            if (closed.empty()) {supported=true;break;}
            closed_choices.push_back(std::move(closed));
        }
        if (!scan_complete) break;
        if (supported) {++report.supported_groups;continue;}
        ++report.blocked_groups;
        const auto block=blocking_set(closed_choices,d.n_nodes()*d.n_nodes(),deadline);
        auto cut=make_cut(d,group,block,group.paths.empty());
        long double lhs=0;
        for (const auto& [j,c] : cut.row.coefficients) lhs+=static_cast<long double>(c)*design[j];
        cut.violation=static_cast<double>(lhs-cut.row.rhs);
        cut.violated=cut.violation>1e-6;
        cut.metadata["violation"]=cut.violation;
        cut.metadata["violated"]=cut.violated;
        if (cut.violated && emitted.insert(cut.key).second) report.cuts.push_back(std::move(cut));
        if (wall_time()>=deadline) {report.timed_out=true;break;}
        if (report.cuts.size()>=max_cuts) break;
    }
    return finish();
}

PathSupportCut certify_fractional_path_support(const Input& d, const PathSupportCache& cache,
        std::size_t group_index, const Coefficients& arc_prices, double intercept,
        const std::vector<double>& design, double deadline) {
    validate_domain(d);
    const DesignLayout layout(d);
    if (group_index>=cache.groups.size() || design.size()<static_cast<std::size_t>(layout.count) ||
        cache.n_od!=d.n_od || cache.n_l!=d.n_l || cache.n_nodes!=d.n_nodes() || std::isnan(deadline))
        throw std::invalid_argument("Invalid fractional support certificate arguments");
    PathSupportCut cut;
    cut.metadata={{"certificate","complete_path_support_dual"},{"valid",false}};
    if (!cache.complete_domain || wall_time()>=deadline) return cut;
    const auto& group=cache.groups[group_index];
    std::vector<double> prices(d.n_nodes()*d.n_nodes(),0);
    if (!std::isfinite(intercept)) return cut;
    for (auto [e,value]:arc_prices) {
        if (e<0 || e>=static_cast<int>(prices.size()) || !std::isfinite(value)) return cut;
        prices[e]+=value;
        if (!std::isfinite(prices[e])) return cut;
    }
    for (auto& value:prices) value=std::max(0.0,value);
    long double needed=std::max(0.0,intercept), maximum_deficit=0;
    for (const auto& path:group.paths) {
        if (wall_time()>=deadline) return cut;
        long double activity=0;
        for (int e:path) activity+=prices.at(e);
        maximum_deficit=std::max(maximum_deficit,1-activity-std::max(0.0,intercept));
        needed=std::max(needed,1-activity);
    }
    // Outward rounding covers both the repair and the exported double RHS.
    const double rhs=std::nextafter(static_cast<double>(needed+1e-12L*(1+needed)),
                                    std::numeric_limits<double>::infinity());
    if (!std::isfinite(rhs)) return cut;
    long double min_activity=std::numeric_limits<long double>::infinity();
    for (const auto& path:group.paths) {
        if (wall_time()>=deadline) return cut;
        long double activity=rhs;
        for (int e:path) activity+=prices.at(e);
        min_activity=std::min(min_activity,activity);
        if (activity<1) return cut;
    }
    cut.head=group.head;cut.origin=group.origin;cut.branch=group.branch;
    cut.row={"fractional_path_support",'L',rhs,{{layout.assignment_index(d.task_index(group.head,group.branch),group.origin),1}}};
    std::ostringstream key; key<<std::setprecision(17)<<"fractional_path_support:"<<d.task_index(group.head,group.branch)<<':'<<group.origin<<':'<<rhs;
    Coefficients nonzero;
    for (std::size_t e=0;e<prices.size();++e) if (prices[e]>0) {
        cut.row.coefficients.emplace_back(layout.airline+e,-prices[e]);
        nonzero.emplace_back(e,prices[e]); key<<':'<<e<<':'<<prices[e];
    }
    long double violation=-rhs;
    for (auto [j,c]:cut.row.coefficients) {
        if (!std::isfinite(design[j])) return PathSupportCut{};
        violation+=static_cast<long double>(c)*design[j];
    }
    cut.key=key.str();cut.valid=true;cut.violation=static_cast<double>(violation);
    cut.violated=cut.violation>1e-6;
    cut.metadata={{"key",cut.key},{"valid",true},{"head",cut.head},{"origin",cut.origin},
        {"branch",cut.branch},{"task",d.task_index(group.head,group.branch)},{"assignment_scope","task"},{"rate",group.rate},{"arc_prices",nonzero},{"intercept",rhs},
        {"intercept_before_repair",intercept},{"maximum_deficit_before_repair",static_cast<double>(maximum_deficit)},
        {"minimum_path_dual_activity",group.paths.empty()?Json(nullptr):Json(static_cast<double>(min_activity))},
        {"paths_checked",group.paths.size()},{"complete_domain",true},{"all_return_bases",true},
        {"certificate","complete_path_support_dual"},{"violation",cut.violation},
        {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},{"sense","L"},
        {"validity_scope","optimality_preserving_assignment_canonicalization"},
        {"preserves_all_original_assignment_encodings",false},{"master_added_variables",0}};
    return cut;
}

struct FractionalPathSupportSeparator::Impl {
    struct CachedLP {
        std::vector<int> arcs;
        std::unique_ptr<NativeSolver> solver;
        std::size_t stamp=0;
    };
    const Input& d;
    const PathSupportCache& cache;
    SolverOptions solver_options;
    FractionalPathSupportOptions options;
    DesignLayout layout;
    std::map<std::size_t,CachedLP> models;
    std::size_t cursor=0, stamp=0;
    Impl(const Input& input,const PathSupportCache& support,const SolverOptions& solver,
         FractionalPathSupportOptions control):d(input),cache(support),solver_options(solver),options(control),layout(input) {
        validate_domain(d);
        if (cache.n_od!=d.n_od || cache.n_l!=d.n_l || cache.n_nodes!=d.n_nodes() ||
            !std::isfinite(options.lp_seconds) || options.lp_seconds<0 ||
            !std::isfinite(options.violation_tolerance) || options.violation_tolerance<0 || !options.max_cached_lps)
            throw std::invalid_argument("Invalid fractional support separator configuration");
        solver_options.log=false;
        // Aggregate diagnostics below avoid thousands of small model events.
        solver_options.progress.reset();
    }
    CachedLP* model(std::size_t index,double deadline,bool& hit) {
        auto found=models.find(index);
        if (found!=models.end()) {hit=true;found->second.stamp=++stamp;return &found->second;}
        hit=false;
        if (wall_time()>=deadline) return nullptr;
        CachedLP entry; std::set<int> arcs;
        for (const auto& path:cache.groups[index].paths) {
            if (wall_time()>=deadline) return nullptr;
            arcs.insert(path.begin(),path.end());
        }
        entry.arcs.assign(arcs.begin(),arcs.end());
        std::map<int,int> variables;
        LinearModel dual;
        for (int e:entry.arcs) variables[e]=dual.variable("support_arc_"+std::to_string(e),0,1);
        const int xi=dual.variable("support_intercept",0,1,1);
        std::size_t row=0;
        for (const auto& path:cache.groups[index].paths) {
            if (wall_time()>=deadline) return nullptr;
            Coefficients coefficients={{xi,1}};
            for (int e:path) coefficients.emplace_back(variables.at(e),1);
            dual.row("support_path_"+std::to_string(row++),'G',1,std::move(coefficients));
        }
        if (wall_time()>=deadline) return nullptr;
        if (models.size()>=options.max_cached_lps) {
            const auto oldest=std::min_element(models.begin(),models.end(),[](const auto& a,const auto& b) {
                return a.second.stamp<b.second.stamp;
            });
            models.erase(oldest);
        }
        entry.solver=std::make_unique<NativeSolver>(dual,solver_options,"fractional_path_support_dual");
        entry.solver->cplex.setParam(IloCplex::Param::RootAlgorithm,IloCplex::Dual);
        entry.stamp=++stamp;
        return &models.emplace(index,std::move(entry)).first->second;
    }
};

FractionalPathSupportSeparator::FractionalPathSupportSeparator(const Input& d,
        const PathSupportCache& cache,const SolverOptions& solver,FractionalPathSupportOptions options)
    :impl_(std::make_unique<Impl>(d,cache,solver,options)) {}
FractionalPathSupportSeparator::~FractionalPathSupportSeparator()=default;

PathSupportSeparation FractionalPathSupportSeparator::separate(const std::vector<double>& design,
        double deadline,std::size_t max_cuts) {
    const double started=wall_time(); auto& w=*impl_; const auto& cache=w.cache;
    if (design.size()<static_cast<std::size_t>(w.layout.count) || std::isnan(deadline))
        throw std::invalid_argument("Invalid fractional support point/deadline");
    for (int j=0;j<w.layout.count;++j)
        if (!std::isfinite(design[j]) || design[j]<-1e-6 || design[j]>1+1e-6)
            throw std::invalid_argument("Fractional support separation needs a finite unit-box design");
    PathSupportSeparation report;report.complete_domain=cache.complete_domain;
    std::size_t lp_calls=0,cache_hits=0,built=0,cheap_supported=0,certificate_failures=0;
    double lp_seconds=0,build_seconds=0; Json statuses=Json::object();
    auto finish=[&] {
        report.seconds=wall_time()-started;
        report.metadata={{"complete_domain",report.complete_domain},{"timed_out",report.timed_out},
            {"seconds",report.seconds},{"checks",report.checks},{"assigned_groups_checked",report.assigned_groups_checked},
            {"supported_groups",report.supported_groups},{"blocked_groups",report.blocked_groups},
            {"cuts",report.cuts.size()},{"lp_solves",lp_calls},{"lp_seconds",lp_seconds},
            {"lp_build_seconds",build_seconds},{"lp_models_built",built},{"lp_cache_hits",cache_hits},
            {"lp_models_cached",w.models.size()},{"lp_cache_limit",w.options.max_cached_lps},
            {"cheap_packing_supported",cheap_supported},{"certificate_failures",certificate_failures},
            {"lp_statuses",statuses},{"master_added_variables",0},{"complete_separation",false},
            {"validity_scope","optimality_preserving_assignment_canonicalization"}};
        return report;
    };
    if (!cache.complete_domain || !max_cuts || cache.groups.empty()) return finish();
    std::vector<double> y(w.d.n_nodes()*w.d.n_nodes());
    for (std::size_t e=0;e<y.size();++e) y[e]=std::clamp(design[w.layout.airline+e],0.0,1.0);
    std::set<std::string> emitted;
    const std::size_t start=w.cursor;
    for (std::size_t checked=0;checked<cache.groups.size();++checked) {
        if (wall_time()>=deadline) {report.timed_out=true;break;}
        const std::size_t index=(start+checked)%cache.groups.size(); w.cursor=(index+1)%cache.groups.size();
        const auto& group=cache.groups[index];
        const double alpha=std::clamp(design[w.layout.assignment_index(w.d.task_index(group.head,group.branch),group.origin)],0.0,1.0);
        if (alpha<=w.options.violation_tolerance) continue;
        ++report.assigned_groups_checked;
        // A feasible fractional path packing is a lower bound on z*(y).
        // If it already reaches alpha there cannot be a violated support row.
        auto residual=y; double packed=0;bool interrupted=false;
        for (const auto& path:group.paths) {
            if (wall_time()>=deadline) {interrupted=true;break;}
            ++report.checks;double amount=1-packed, single=1;
            for (int e:path) {amount=std::min(amount,residual[e]);single=std::min(single,y[e]);}
            if (single>=alpha) {packed=alpha;break;}
            if (amount>0) {packed+=amount;for (int e:path) residual[e]=std::max(0.0,residual[e]-amount);}
            if (packed>=alpha) break;
        }
        if (interrupted) {w.cursor=index;report.timed_out=true;break;}
        if (packed>=alpha) {++cheap_supported;++report.supported_groups;continue;}
        Coefficients prices;double intercept=0;std::string status="empty_physical_group";
        if (!group.paths.empty()) {
            if (lp_calls>=w.options.max_lp_solves || w.options.lp_seconds==0) {w.cursor=index;break;}
            const double before_build=wall_time();bool hit=false;
            Impl::CachedLP* entry=nullptr;
            try {entry=w.model(index,deadline,hit);} catch (const IloException&) {++certificate_failures;continue;}
            build_seconds+=wall_time()-before_build;
            if (!entry) {w.cursor=index;report.timed_out=true;break;}
            if (hit) ++cache_hits;else ++built;
            for (std::size_t j=0;j<entry->arcs.size();++j)
                entry->solver->objective.setLinearCoef(entry->solver->variables[j],y[entry->arcs[j]]);
            if (wall_time()>=deadline) {w.cursor=index;report.timed_out=true;break;}
            SolveOutcome solved;++lp_calls;
            try {solved=entry->solver->optimize(std::max(0.0,std::min(w.options.lp_seconds,deadline-wall_time())));}
            catch(const IloException&) {++certificate_failures;continue;}
            lp_seconds+=solved.seconds;status=solved.status;
            statuses[status]=statuses.value(status,0)+1;
            if (!(solved.has_solution || solved.candidate_available) || solved.values.size()!=entry->arcs.size()+1) {
                ++certificate_failures;continue;
            }
            for (std::size_t j=0;j<entry->arcs.size();++j) prices.emplace_back(entry->arcs[j],solved.values[j]);
            intercept=solved.values.back();
        }
        auto cut=certify_fractional_path_support(w.d,cache,index,prices,intercept,design,deadline);
        if (!cut.valid) {
            ++certificate_failures;
            if (wall_time()>=deadline) {w.cursor=index;report.timed_out=true;break;}
            continue;
        }
        cut.violated=cut.violation>w.options.violation_tolerance;
        cut.metadata["lp_status"]=status;cut.metadata["violated"]=cut.violated;
        if (cut.violated && emitted.insert(cut.key).second) {++report.blocked_groups;report.cuts.push_back(std::move(cut));}
        if (report.cuts.size()>=max_cuts) break;
    }
    return finish();
}
} // namespace mlnd
