#include "mlnd/pricing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace mlnd {
namespace {
constexpr double resource_tolerance = 1e-12;
// Each operand is nonnegative. Round both the product and the sum outward,
// so this envelope does not depend on extended long-double precision or on
// a fixed bound on the number of task contributions. Exact zero terms add no
// padding; nonfinite results are rejected by the caller's certificate guard.
long double add_nonnegative_product_up(long double total,long double rate,long double value) {
    if(rate==0 || value==0)return total;
    const auto infinity=std::numeric_limits<long double>::infinity();
    const long double product=std::nextafter(rate*value,infinity);
    return std::nextafter(total+product,infinity);
}
using SegmentKey = std::array<int,5>;
template<std::size_t N> struct KeyHash {
    std::size_t operator()(const std::array<int,N>& key) const {
        std::size_t h = 0;
        for (int v : key) h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
std::vector<int> leg(int from,int to) { return from==to ? std::vector<int>{from} : std::vector<int>{from,to}; }
struct DeliveryExtension { int delivery; double loaded, back; };
struct PickupExtension {
    int od, pickup;
    double transfer, wait, q, minimum_remaining;
    std::vector<DeliveryExtension> deliveries;
};
struct Terminal { int id, delivery; double reduced; };
// A label is a partial TASK, not an elementarity state of a vehicle-day route.
// First-order identity and its segment are retained: these determine branch and
// executed-head rows, and AA and repeated lockers are allowed.
struct Label {
    BasePair b{};
    int first=-1, pickup=-1;
    InitialSegment segment;
    const PickupExtension* next=nullptr;
    double t0=0, q1=0, wait1=0, busy=0, reduced=0;
};
template<class T> void identity_scalar(std::string& key,const T& value) {
    static_assert(std::is_arithmetic_v<T>);
    key.append(reinterpret_cast<const char*>(&value),sizeof(value));
}
template<class T> void identity_vector(std::string& key,const std::vector<T>& values) {
    identity_scalar(key,values.size());
    for(const auto& value:values)identity_scalar(key,value);
}
template<class T> void identity_matrix(std::string& key,const Matrix<T>& matrix) {
    identity_scalar(key,matrix.rows);identity_scalar(key,matrix.cols);
    identity_vector(key,matrix.values);
}
void identity_string(std::string& key,const std::string& value) {
    identity_scalar(key,value.size());key.append(value);
}
}

std::string pricing_domain_identity(const Input& d,const PathCollection& paths) {
    // Compare exact snapshots, not a probabilistic hash or object address.
    // Include task rates and all coefficients used by the recourse model so
    // this key also protects a persistent fixed-design RMP from input edits.
    std::string key="independent-tasks-courier-home/1";
    for(int value:{d.n_l,d.n_h,d.n_od,d.n_levels,d.max_orders})identity_scalar(key,value);
    for(double value:{d.delta,d.operating_limit,d.fleet_cost,d.flight_cost,d.waiting_cost})identity_scalar(key,value);
    for(bool value:{d.has_candidate_routes,d.preprocessed_demand,d.first_leg_consistency,paths.support_complete})
        identity_scalar(key,value);
    for(const auto* value:{&d.route_mode,&d.flow_policy,&d.model_version,&paths.scope})identity_string(key,*value);
    for(const auto* value:{&d.demand,&d.ground_cost,&d.deadline,&d.base_cost,&d.slot_epsilon,&d.single_rates,&d.head_rates})
        identity_vector(key,*value);
    for(const auto* value:{&d.flight_times,&d.rider_mean,&d.first_cost,&d.terminal_cost,
                          &d.receipt_mean,&d.home_receipt_mean,&d.removal_mean,&d.slot_cost,&d.link_cost,&d.kappa,
                          &d.occupancy_limit,&d.pickup_share,&d.last_rider_cost,&d.last_rider_mean,&d.pair_rates})
        identity_matrix(key,*value);
    for(const auto* value:{&d.slot_counts,&d.allowed_links,&d.common_origin_support})identity_matrix(key,*value);
    identity_vector(key,d.od_origin);identity_vector(key,d.od_destination);
    identity_scalar(key,d.candidate_routes.size());
    for(const auto& route:d.candidate_routes)identity_vector(key,route);
    identity_scalar(key,paths.initial_segments.size());
    for(const auto& segment:paths.initial_segments) {
        for(int value:{segment.id,segment.b[0],segment.b[1],segment.first_od,segment.pickup,segment.delivery})
            identity_scalar(key,value);
        identity_scalar(key,segment.t0);identity_scalar(key,segment.t1);
        identity_scalar(key,segment.legs.size());
        for(const auto& leg:segment.legs)identity_vector(key,leg);
    }
    return key;
}

struct LabelPricingOracle::Graph {
    std::unordered_map<SegmentKey,int,KeyHash<5>> segments;
    // Input-derived event graph: first-delivery -> second-pickup ->
    // second-delivery -> return. No completed catalogue paths are traversed.
    std::vector<std::vector<PickupExtension>> pickups;
    bool complete=false;
};

LabelPricingOracle::LabelPricingOracle(const Input& d,PathCollection& paths,double deadline,
                                     LabelPricingGraphCache* cache,bool prefer_catalogue,
                                     bool capture_identity) : data_(&d), paths_(&paths) {
    if (d.route_mode!="direct" || d.has_candidate_routes)
        throw std::invalid_argument("Label pricing supports the direct-flight K=1/K=2 universe only");
    if (!paths.support_complete) throw std::invalid_argument("Label pricing requires complete model-support metadata");
    if(cache || capture_identity)domain_identity_=pricing_domain_identity(d,paths);
    // No event graph, registry rebuild, or completed-path copy is needed by
    // the complete-catalogue backend. A later label call can still build it.
    if(prefer_catalogue && paths.complete)return;
    if(cache) {
        if(cache->graph_ && cache->identity_==domain_identity_) {
            ++cache->hits_;
            graph_=cache->graph_;graph_reused_=true;
            if(paths.service_registry.size()!=paths.service_paths.size() ||
               paths.empty_registry.size()!=paths.empty_paths.size())rebuild_path_registry(paths);
            return;
        }
        if(cache->graph_)++cache->misses_;
        ++cache->attempts_;
    }
    auto g=std::make_shared<Graph>();
    g->complete=false;
    graph_=g;
    auto stopped=[&] { return wall_time()>=deadline; };
    if (stopped()) return;
    if (paths.service_registry.size()!=paths.service_paths.size() ||
        paths.empty_registry.size()!=paths.empty_paths.size()) rebuild_path_registry(paths);
    for (const auto& s:paths.initial_segments) {
        if (stopped()) return;
        if (!g->segments.emplace(SegmentKey{s.b[0],s.b[1],s.first_od,s.pickup,s.delivery},s.id).second)
            throw std::invalid_argument("Duplicate initial-segment identity");
    }
    g->pickups.resize(d.n_h*d.n_l);
    if (d.effective_delta()>0) for (int j=0;j<d.n_h;++j) for (int b=0;b<d.n_l;++b) {
        auto& extensions=g->pickups[j*d.n_l+b];
        for (int s=0;s<d.n_od;++s) for (int i=0;i<d.n_h;++i) {
            if (stopped()) return;
            if (i!=j && !d.allowed_links(d.n_l+j,d.n_l+i)) continue;
            PickupExtension e;
            e.od=s; e.pickup=i; e.transfer=d.flight_times(d.n_l+j,d.n_l+i);
            e.wait=second_wait(d.rider_mean(s,i),e.transfer,d.effective_delta());
            e.q=e.transfer+d.effective_delta()/2+e.wait;
            e.minimum_remaining=std::numeric_limits<double>::infinity();
            for (int k=0;k<d.n_h;++k) {
                if (stopped()) return;
                if (i==k || !d.allowed_links(d.n_l+i,d.n_l+k) || !d.allowed_links(d.n_l+k,b)) continue;
                const double loaded=d.flight_times(d.n_l+i,d.n_l+k), back=d.flight_times(d.n_l+k,b);
                if (e.q+loaded+d.delivery_receipt(s,k)>d.deadline[s]+resource_tolerance) continue;
                e.deliveries.push_back({k,loaded,back});
                e.minimum_remaining=std::min(e.minimum_remaining,loaded+back);
            }
            if (!e.deliveries.empty()) extensions.push_back(std::move(e));
        }
    }
    g->complete=!stopped();
    if(cache && g->complete) {
        cache->identity_=domain_identity_;
        cache->graph_=g;
        ++cache->builds_;
    }
}

const std::string& LabelPricingOracle::domain_identity() const {
    if(domain_identity_.empty())throw std::logic_error("Pricing identity capture was not requested");
    return domain_identity_;
}

namespace {
// Complete-catalogue candidate storage contains IDs and scalars only. Physical
// path payloads and sparse matrix columns are never copied by this backend.
struct CatalogueCandidate { bool empty=false; int id=-1; double reduced=0; };

LabelPricingResult price_catalogue(const Input& d,const PathCollection& paths,const Formulation& f,
        const std::vector<double>& prices,const std::vector<double>& missing_service,
        const std::vector<double>& missing_empty,const LabelPricingOptions& options) {
    if(!paths.complete)throw std::logic_error("Catalogue pricing requires a complete physical domain");
    const auto service_count=paths.service_paths.size(),empty_count=paths.empty_paths.size();
    if(prices.size()!=f.matrix.rows.size() || missing_service.size()>service_count ||
       missing_empty.size()>empty_count || f.flow.size()>service_count || f.empty_flow.size()>empty_count)
        throw std::invalid_argument("Catalogue pricing dual/catalogue dimensions disagree");
    if(!std::isfinite(options.tolerance) || options.tolerance<0 || options.max_columns==0 ||
       std::isnan(options.deadline) || (options.branch_correction && std::isnan(options.branch_deadline)))
        throw std::invalid_argument("Invalid catalogue pricing options");
    for(double value:prices)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite pricing dual");
    for(const auto* values:{&missing_service,&missing_empty})for(double value:*values)
        if(!std::isfinite(value)||value<0)throw std::invalid_argument("Invalid omitted upper-bound dual");
    const std::set<int> extra_service(options.excluded_service.begin(),options.excluded_service.end());
    const std::set<int> extra_empty(options.excluded_empty.begin(),options.excluded_empty.end());
    for(int id:extra_service)if(id<0||static_cast<std::size_t>(id)>=service_count)
        throw std::invalid_argument("Invalid extra service exclusion");
    for(int id:extra_empty)if(id<0||static_cast<std::size_t>(id)>=empty_count)
        throw std::invalid_argument("Invalid extra empty exclusion");
    auto excluded=[&](bool empty,std::size_t id) {
        const auto& membership=empty?f.empty_flow:f.flow;
        return (id<membership.size()&&membership[id]>=0) ||
            (empty?extra_empty:extra_service).count(static_cast<int>(id));
    };
    LabelPricingResult result;
    result.catalogue_pricing_used=true;
    result.fast_pricing_used=true;
    result.diverse_pricing_used=options.diverse_pricing;
    result.dominance="complete-catalogue-canonical-scan";
    if(options.diverse_pricing)result.diversity_policy="task-return-base/resource-support-strata;q=4";
    result.branch_correction_requested=options.branch_correction;
    if(options.branch_correction)result.branch_correction_status="catalogue_incomplete";
    auto stopped=[&] {return wall_time()>=options.deadline;};
    if(stopped())return result;
    const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    if(!std::isfinite(total)||total<0)throw std::invalid_argument("Invalid catalogue total demand");

    const double branch_started=wall_time();
    const double branch_deadline=std::min(options.deadline,options.branch_deadline);
    auto branch_stopped=[&] {return wall_time()>=branch_deadline;};
    bool branch_active=options.branch_correction;
    if(branch_active && (!d.preprocessed_demand || !d.origin_assignment() ||
       d.flow_policy!="aggregate" || d.first_leg_consistency)) {
        branch_active=false;result.branch_correction_status="unsupported_scope";
    } else if(branch_active) {
        result.branch_correction_scope="preprocessed_origin_aggregate";
        if(branch_stopped()) {branch_active=false;result.branch_correction_status="budget_exhausted";}
    }
    if(branch_active) {
        if(d.single_rates.size()!=static_cast<std::size_t>(d.n_od) ||
           d.pair_rates.rows!=d.n_od || d.pair_rates.cols!=d.n_od)
            throw std::invalid_argument("Branch pricing task-rate dimensions disagree");
        for(double rate:d.single_rates)if(!std::isfinite(rate)||rate<0)
            throw std::invalid_argument("Invalid branch pricing single rate");
        for(double rate:d.pair_rates.values)if(!std::isfinite(rate)||rate<0)
            throw std::invalid_argument("Invalid branch pricing pair rate");
    }
    const int branches=d.n_od+1;
    std::vector<long double> branch_minima(branch_active?static_cast<std::size_t>(d.n_od)*d.n_l*branches:0,0);
    auto branch_index=[&](int r,int l,int s) {return (static_cast<std::size_t>(r)*d.n_l+l)*branches+s+1;};
    const auto candidate_limit=options.diverse_pricing && options.max_columns<=std::numeric_limits<std::size_t>::max()/4
        ?4*options.max_columns:options.max_columns;
    std::vector<CatalogueCandidate> candidates;
    auto signature=[&](const CatalogueCandidate& c) {
        if(!c.empty)return service_path_signature(paths.service_paths[c.id]);
        const auto& p=paths.empty_paths[c.id];
        return ServiceSignature{p.b[0],p.b[1],-1,-1,-1,-1,-1,-1};
    };
    auto less=[&](const CatalogueCandidate& a,const CatalogueCandidate& b) {
        if(a.reduced!=b.reduced)return a.reduced<b.reduced;
        if(a.empty!=b.empty)return a.empty<b.empty;
        return signature(a)<signature(b);
    };
    auto record=[&](bool empty,int id,double rc) {
        if(!std::isfinite(rc))throw std::invalid_argument("Nonfinite catalogue reduced cost");
        ++result.omitted_paths;
        auto& minimum=empty?result.min_empty:result.min_service;
        minimum=std::min(minimum,rc);
        if(rc>=-options.tolerance)return;
        ++result.negative_paths;
        const CatalogueCandidate c{empty,id,rc};
        if(candidates.size()==candidate_limit && !less(c,candidates.front()))return;
        if(candidates.size()==candidate_limit) {
            std::pop_heap(candidates.begin(),candidates.end(),less);candidates.pop_back();
        }
        candidates.push_back(c);std::push_heap(candidates.begin(),candidates.end(),less);
    };
    for(std::size_t i=0;i<service_count;++i) {
        if(i%128==0) {
            if(stopped())return result;
            if(branch_active&&branch_stopped()) {
                branch_active=false;result.branch_correction_status="budget_exhausted";
            }
        }
        ++result.catalogue_paths_scanned;
        if(branch_active)++result.branch_paths_scanned;
        const auto& p=paths.service_paths[i];
        if(p.id!=static_cast<int>(i))throw std::logic_error("Catalogue service IDs are not canonical");
        if(!service_path_enabled(f,p)) {++result.resource_pruned;continue;}
        ++result.terminal_paths;
        if(excluded(false,i)) {++result.excluded_paths;continue;}
        if(branch_active && (p.first_od<0||p.first_od>=d.n_od||p.b[0]<0||p.b[0]>=d.n_l||
           p.second_od < -1||p.second_od>=d.n_od))
            throw std::logic_error("Complete pricing catalogue has an invalid branch identity");
        const double upper=i<missing_service.size()?missing_service[i]:0;
        double rc=p.cost+upper;
        long double safe_rc=static_cast<long double>(p.cost)+upper;
        long double magnitude=1+std::abs(safe_rc);
        // The double candidate arithmetic and conservative long-double branch
        // arithmetic share exactly one canonical coefficient traversal.
        for_each_path_coefficient(f,p,[&](int row,double coefficient) {
            const double price=prices.at(row);
            rc-=coefficient*price;
            if(branch_active) {
                const long double term=static_cast<long double>(coefficient)*price;
                safe_rc-=term;magnitude+=std::abs(term);
            }
        });
        ++result.catalogue_path_reuses;
        record(false,static_cast<int>(i),rc);
        if(branch_active) {
            if(!std::isfinite(safe_rc)||!std::isfinite(magnitude)) {
                branch_active=false;result.branch_correction_status="invalid_numerics";
            } else {
                safe_rc-=64*std::numeric_limits<double>::epsilon()*magnitude;
                auto& minimum=branch_minima[branch_index(p.first_od,p.b[0],p.second_od)];
                minimum=std::min(minimum,safe_rc);
            }
        }
    }
    // A separate branch budget can expire without invalidating the ordinary
    // scan. Finalize the service-only strengthening before scanning empties.
    double branch_rounded=0;
    bool branch_finished=false;
    if(branch_active&&branch_stopped()) {
        branch_active=false;result.branch_correction_status="budget_exhausted";
    }
    if(branch_active) {
        long double correction=0;
        for(int q=0;q<d.task_count();++q) {
            const int r=d.task_first(q),s=d.task_second(q);
            const double rate=d.task_rate(q);
            if(rate<=0)continue;
            long double largest=0;
            for(int l=0;l<d.n_l;++l) {
                ++result.branch_groups;
                largest=std::max(largest,-branch_minima[branch_index(r,l,s)]);
            }
            correction=add_nonnegative_product_up(correction,rate,largest);
        }
        const long double safe=correction+64*std::numeric_limits<double>::epsilon()*(1+correction);
        branch_rounded=std::nextafter(static_cast<double>(safe),std::numeric_limits<double>::infinity());
        if(!std::isfinite(correction)||!std::isfinite(branch_rounded))result.branch_correction_status="invalid_numerics";
        else if(branch_stopped())result.branch_correction_status="budget_exhausted";
        else branch_finished=true;
    }
    if(options.branch_correction)result.branch_seconds=wall_time()-branch_started;
    std::size_t omitted_empty=0;
    for(std::size_t i=0;i<empty_count;++i) {
        if(stopped())return result;
        ++result.catalogue_paths_scanned;++result.terminal_paths;
        const auto& p=paths.empty_paths[i];
        if(p.id!=static_cast<int>(i))throw std::logic_error("Catalogue empty IDs are not canonical");
        if(excluded(true,i)) {++result.excluded_paths;continue;}
        // Deliberately retain the original empty-column semantics at D=0.
        ++omitted_empty;
        double rc=p.cost+(i<missing_empty.size()?missing_empty[i]:0);
        for_each_path_coefficient(f,p,[&](int row,double coefficient){rc-=coefficient*prices.at(row);});
        ++result.catalogue_path_reuses;record(true,static_cast<int>(i),rc);
    }
    result.minimum=std::min(result.min_service,result.min_empty);
    result.old_service_correction=total*std::max(0.0,-result.min_service);
    result.chosen_service_correction=result.old_service_correction;
    result.empty_correction=total*omitted_empty*std::max(0.0,-result.min_empty);
    result.correction=result.old_service_correction+result.empty_correction;
    if(!std::isfinite(result.correction))throw std::invalid_argument("Nonfinite catalogue missing-column correction");
    std::sort(candidates.begin(),candidates.end(),less);
    std::vector<std::size_t> chosen;
    if(options.diverse_pricing && !candidates.empty()) {
        // Deterministic strata on the bounded candidate pool. Preserve the
        // global minimum first, then prefer underrepresented task/return-base
        // groups and resource-support patterns; cost breaks all remaining ties.
        using Group=std::array<int,4>;
        std::vector<std::size_t> groups,resources;
        std::map<Group,std::size_t> group_ids;
        std::map<std::pair<Group,std::vector<int>>,std::size_t> resource_ids;
        for(const auto& c:candidates) {
            Group group;std::vector<int> support;
            if(c.empty) {
                const auto& p=paths.empty_paths[c.id];group={1,-1,-1,p.b[1]};
                for(auto [u,v,count]:p.arc_counts)if(count) support.push_back(d.n_h+u*d.n_nodes()+v);
            } else {
                const auto& p=paths.service_paths[c.id];group={0,p.first_od,p.second_od,p.b[1]};
                for(auto [i,time]:p.slot_times)if(time!=0)support.push_back(i);
                for(auto [u,v,count]:p.arc_counts)if(count)support.push_back(d.n_h+u*d.n_nodes()+v);
            }
            std::sort(support.begin(),support.end());support.erase(std::unique(support.begin(),support.end()),support.end());
            const auto gid=group_ids.emplace(group,group_ids.size()).first->second;
            const auto rid=resource_ids.emplace(std::make_pair(group,std::move(support)),resource_ids.size()).first->second;
            groups.push_back(gid);resources.push_back(rid);
        }
        result.diversity_candidates=candidates.size();result.diversity_groups=group_ids.size();
        std::vector<std::size_t> group_use(group_ids.size(),0),resource_use(resource_ids.size(),0);
        std::vector<bool> selected(candidates.size(),false);
        while(chosen.size()<options.max_columns && chosen.size()<candidates.size()) {
            if(stopped())return result;
            std::size_t best=candidates.size();
            for(std::size_t i=0;i<candidates.size();++i)if(!selected[i]) {
                if(best==candidates.size() ||
                   std::make_pair(group_use[groups[i]],resource_use[resources[i]]) <
                   std::make_pair(group_use[groups[best]],resource_use[resources[best]]))best=i;
            }
            selected[best]=true;chosen.push_back(best);
            ++group_use[groups[best]];++resource_use[resources[best]];
        }
        std::sort(chosen.begin(),chosen.end());
    } else for(std::size_t i=0;i<std::min(options.max_columns,candidates.size());++i)chosen.push_back(i);
    for(auto i:chosen) {
        const auto& c=candidates[i];
        (c.empty?result.empty:result.service).push_back(c.id);
    }
    result.complete=!stopped();
    if(result.complete&&branch_finished) {
        result.branch_correction_complete=true;result.branch_correction_status="complete";
        result.branch_service_correction=branch_rounded;
        result.chosen_service_correction=std::min(result.old_service_correction,branch_rounded);
        result.branch_correction_used=branch_rounded<result.old_service_correction;
        result.correction=result.chosen_service_correction+result.empty_correction;
    }
    return result;
}
}

LabelPricingResult LabelPricingOracle::price(const Formulation& f,const std::vector<double>& prices,
        const std::vector<double>& missing_service,const std::vector<double>& missing_empty,
        const LabelPricingOptions& options) const {
    if(options.catalogue_pricing && paths_->complete)
        return price_catalogue(*data_,*paths_,f,prices,missing_service,missing_empty,options);
    if(!graph_) {
        // A caller may change backend options, or invalidate catalogue
        // completeness after constructing a catalogue-preferred oracle.
        LabelPricingOracle fallback(*data_,*paths_,options.deadline);
        return fallback.price(f,prices,missing_service,missing_empty,options);
    }
    const auto& g=*graph_;
    const auto& d=*data_;
    auto& paths=*paths_;
    const auto service_count=paths.service_paths.size(), empty_count=paths.empty_paths.size();
    LabelPricingResult result;
    result.dominance="same-second-pickup-exact-suffix-extension";
    result.fast_pricing_used=options.fast_pricing;
    result.branch_correction_requested=options.branch_correction;
    if(options.branch_correction)result.branch_correction_status="label_incomplete";
    // A deadline-interrupted graph is never a smaller substitute path domain.
    // It cannot be searched or used to certify absence of negative columns.
    if (!g.complete) return result;
    if (prices.size()!=f.matrix.rows.size() ||
        missing_service.size()>service_count ||
        missing_empty.size()>empty_count ||
        f.flow.size()>service_count || f.empty_flow.size()>empty_count)
        throw std::invalid_argument("Label pricing dual/catalogue dimensions disagree");
    if (!std::isfinite(options.tolerance) || options.tolerance<0 || options.max_columns==0)
        throw std::invalid_argument("Invalid label pricing tolerance/batch size");
    for (double pi:prices) if (!std::isfinite(pi)) throw std::invalid_argument("Nonfinite pricing dual");
    for (const auto* values:{&missing_service,&missing_empty})
        for (double u:*values) if (!std::isfinite(u) || u<0) throw std::invalid_argument("Invalid omitted upper-bound dual");
    std::set<int> extra_service(options.excluded_service.begin(),options.excluded_service.end());
    std::set<int> extra_empty(options.excluded_empty.begin(),options.excluded_empty.end());
    auto excluded=[&](bool empty,int id) {
        const auto& indices=empty?f.empty_flow:f.flow;
        const auto& extra=empty?extra_empty:extra_service;
        return id>=0 && ((static_cast<std::size_t>(id)<indices.size() && indices[id]>=0) || extra.count(id)!=0);
    };
    for (int id:extra_service) if (id<0 || static_cast<std::size_t>(id)>=service_count)
        throw std::invalid_argument("Invalid extra service exclusion");
    for (int id:extra_empty) if (id<0 || static_cast<std::size_t>(id)>=empty_count)
        throw std::invalid_argument("Invalid extra empty exclusion");
    auto stopped=[&] { return wall_time()>=options.deadline; };
    if (stopped()) return result;
    const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    struct Candidate {
        bool empty=false; double reduced=0; ServiceSignature signature{};
        ServicePath service; EmptyPath relocation;
    };
    std::vector<Candidate> candidates;
    auto less=[](const Candidate& a,const Candidate& b) {
        return std::tie(a.reduced,a.empty,a.signature)<std::tie(b.reduced,b.empty,b.signature);
    };
    auto retain=[&](Candidate c) {
        if (candidates.size()==options.max_columns && !less(c,candidates.front())) return;
        if (candidates.size()==options.max_columns) {
            std::pop_heap(candidates.begin(),candidates.end(),less); candidates.pop_back();
        }
        candidates.push_back(std::move(c)); std::push_heap(candidates.begin(),candidates.end(),less);
    };
    auto record=[&](const auto& path,double reduced) {
        constexpr bool empty=std::is_same_v<std::decay_t<decltype(path)>,EmptyPath>;
        ++result.omitted_paths;
        auto& minimum=empty?result.min_empty:result.min_service;
        minimum=std::min(minimum,reduced);
        if (reduced < -options.tolerance) {
            ++result.negative_paths;
            Candidate c; c.empty=empty; c.reduced=reduced;
            if constexpr (empty) c.signature={path.b[0],path.b[1],-1,-1,-1,-1,-1,-1};
            else c.signature=service_path_signature(path);
            // The bounded heap compares only cost/type/signature. Avoid copying
            // the physical payload of a negative path that cannot enter it.
            if(options.fast_pricing && candidates.size()==options.max_columns && !less(c,candidates.front()))return;
            if constexpr (empty) c.relocation=path;
            else c.service=path;
            retain(std::move(c));
        }
    };
    auto reduced=[&](const auto& path,bool empty,int id) {
        const auto& upper=empty?missing_empty:missing_service;
        double value=path.cost+(id>=0 && static_cast<std::size_t>(id)<upper.size()?upper[id]:0);
        if(options.fast_pricing)
            for_each_path_coefficient(f,path,[&](int row,double coefficient){value-=coefficient*prices.at(row);});
        else for (auto [row,coefficient]:path_column(f,path)) value-=coefficient*prices.at(row);
        return value;
    };
    auto service_id=[&](const ServicePath& path) {
        if(options.fast_pricing && path.id>=0)return path.id;
        const auto it=paths.service_registry.find(service_path_signature(path));
        return it==paths.service_registry.end()?-1:it->second;
    };
    auto physical_service=[&](const InitialSegment& segment,const PathContinuation* next,
                              ServicePath& temporary)->const ServicePath* {
        ServicePath branch;
        branch.first_od=segment.first_od;branch.second_od=next?next->od:-1;
        if(!service_path_enabled(f,branch))return nullptr;
        if(options.fast_pricing && paths.complete) {
            const ServiceSignature signature={segment.b[0],segment.b[1],segment.first_od,
                segment.pickup,segment.delivery,next?next->od:-1,next?next->pickup:-1,next?next->delivery:-1};
            const auto found=paths.service_registry.find(signature);
            if(found!=paths.service_registry.end()) {
                ++result.catalogue_path_reuses;
                return &paths.service_paths.at(found->second);
            }
            // A missing signature can be physically infeasible. Keep the
            // authoritative scalar check, never infer feasibility from lookup.
        }
        ++result.physical_path_builds;
        return build_service_path(d,segment,next,temporary)?&temporary:nullptr;
    };
    auto flight_cost=[&](int from,int to,double time) {
        return (d.flight_cost-prices[f.fleet_row])*time
             -(from==to?0:prices[f.link_rows[from*d.n_nodes()+to]]);
    };
    auto pickup_cost=[&](int od,int i,double q,double wait) {
        return d.first_cost(od,i)+(d.waiting_cost-prices[f.fleet_row])*wait
             -prices[f.capacity_rows[i]]*q;
    };
    auto delivery_cost=[&](int od,int j,double q,double loaded) {
        return d.terminal_cost(od,j)-prices[f.order_rows[od]]
             -prices[f.capacity_rows[j]]*(q+loaded+d.removal_mean(od,j));
    };
    std::size_t omitted_empty=0;
    // Empty repositioning is the two-layer origin/return-base graph.
    for (int from=0;from<d.n_l;++from) for (int to=0;to<d.n_l;++to) {
        if (stopped()) return result;
        ++result.labels_created; ++result.labels_extended;
        if (from==to || !d.allowed_links(from,to)) continue;
        const double t=d.flight_times(from,to);
        if (t>d.operating_limit+resource_tolerance) { ++result.resource_pruned; continue; }
        EmptyPath p;
        p.b={from,to}; p.legs={leg(from,to)}; p.arc_counts={{from,to,1}};
        p.flight_time=p.time=t; p.cost=d.flight_cost*t;
        const auto it=paths.empty_registry.find(p.b);
        const int id=it==paths.empty_registry.end()?-1:it->second;
        ++result.terminal_paths;
        if (excluded(true,id)) { ++result.excluded_paths; continue; }
        ++omitted_empty;
        record(p,reduced(p,true,id));
    }
    // Label-setting on the acyclic task-stage graph. Negative reduced costs are
    // permitted: topological stages, rather than Dijkstra's nonnegative-arc
    // assumption, ensure each retained label is extended only once.
    std::vector<Label> departure,first_pickup,first_delivery;
    for (int from=0;from<d.n_l;++from) for (int to=0;to<d.n_l;++to)
        for (int r=0;r<d.n_od;++r) {
            Label l; l.b={from,to}; l.first=r;
            if (from!=to) l.reduced=-prices[f.balance_rows[from]]+prices[f.balance_rows[to]];
            departure.push_back(l); ++result.labels_created;
        }
    for (const auto& source:departure) {
        if (stopped()) return result;
        ++result.labels_extended;
        for (int i=0;i<d.n_h;++i) {
            if (!d.allowed_links(source.b[0],d.n_l+i)) continue;
            Label l=source; l.pickup=i;
            l.t0=d.flight_times(source.b[0],d.n_l+i);
            l.wait1=first_wait(d.rider_mean(l.first,i),l.t0); l.q1=l.t0+l.wait1;
            if (l.q1>d.deadline[l.first]+resource_tolerance || l.q1>d.operating_limit+resource_tolerance) {
                ++result.resource_pruned; continue;
            }
            l.busy=l.q1;
            l.reduced+=flight_cost(l.b[0],d.n_l+i,l.t0)+pickup_cost(l.first,i,l.q1,l.wait1);
            first_pickup.push_back(std::move(l)); ++result.labels_created;
        }
    }
    for (const auto& source:first_pickup) {
        if (stopped()) return result;
        ++result.labels_extended;
        for (int j=0;j<d.n_h;++j) {
            Label l=source;
            if (!build_initial_segment(d,l.b,l.first,l.pickup,j,-1,l.segment)) {
                ++result.resource_pruned; continue;
            }
            auto it=g.segments.find({l.b[0],l.b[1],l.first,l.pickup,j});
            if (it==g.segments.end()) throw std::runtime_error("Label-generated segment absent from model support");
            l.segment.id=it->second;
            l.busy=l.t0+l.segment.t1+l.wait1;
            l.reduced+=flight_cost(d.n_l+l.pickup,d.n_l+j,l.segment.t1)
                +delivery_cost(l.first,j,l.q1,l.segment.t1);
            if (f.executed_head_rows[l.segment.id]>=0) l.reduced+=prices[f.executed_head_rows[l.segment.id]];
            first_delivery.push_back(std::move(l)); ++result.labels_created;
        }
    }
    // Second-pickup buckets have the same remaining event graph. Historical
    // differences are kept in their labels and checked by exact suffix
    // dominance below; they are never discarded merely for being at one node.
    using Bucket=std::array<int,4>; // departure base, return base, second OD, pickup
    std::map<Bucket,std::vector<Label>> buckets;
    for (const auto& source:first_delivery) {
        if (stopped()) return result;
        ++result.labels_extended;
        ServicePath temporary;
        if (const auto* path=physical_service(source.segment,nullptr,temporary)) {
            const auto& p=*path;
            const int id=service_id(p);
            ++result.terminal_paths; ++result.labels_created;
            if (!excluded(false,id)) record(p,reduced(p,false,id));
            else ++result.excluded_paths;
        } else ++result.resource_pruned;
        for (const auto& next:g.pickups[source.segment.delivery*d.n_l+source.b[1]]) {
            if (stopped()) return result;
            if (d.kappa(source.first,next.od)<=0) continue;
            // Conservative lower resource bound. Boundary survivors always use
            // the authoritative accurately-summed complete mission check.
            const double busy=source.busy+next.transfer+next.wait;
            const double slack=resource_tolerance+64*std::numeric_limits<double>::epsilon()
                                *std::max(1.0,std::abs(d.operating_limit));
            if (busy+next.minimum_remaining>d.operating_limit+slack) { ++result.resource_pruned; continue; }
            Label l=source; l.next=&next; l.busy=busy;
            l.reduced+=flight_cost(d.n_l+source.segment.delivery,d.n_l+next.pickup,next.transfer)
                 +pickup_cost(next.od,next.pickup,next.q,next.wait)
                 +prices[f.double_rows.at(source.segment.id*d.n_od+next.od)];
            buckets[{l.b[0],l.b[1],next.od,next.pickup}].push_back(std::move(l));
            ++result.labels_created;
        }
    }
    bool interrupted=false;
    // A complete suffix at this stage has only one decision: delivery locker.
    // The exact extension profile contains every feasible OMITTED completion,
    // with its canonical sparse-column reduced cost and upper-bound multiplier.
    // It makes dominance valid under selected-column exclusions, arbitrary
    // omitted upper prices, AA, shared locker occupancy and numerical summation.
    // Bounded transient cache: at most 128 suffix profiles, each with at most
    // H scalar completions. Never retain the complete physical task universe.
    std::map<const Label*,std::vector<Terminal>> profiles;
    auto profile=[&](const Label& l)->std::vector<Terminal> {
        const auto found=profiles.find(&l);
        if (found!=profiles.end()) return found->second;
        std::vector<Terminal> terminals;
        for (const auto& e:l.next->deliveries) {
            if (stopped()) { interrupted=true; break; }
            PathContinuation c{l.next->od,l.next->pickup,e.delivery,l.next->transfer,e.loaded,e.back,l.next->wait};
            ServicePath temporary;
            ++result.labels_created;
            const auto* path=physical_service(l.segment,&c,temporary);
            if (!path) { ++result.resource_pruned; continue; }
            const auto& p=*path;
            const int id=service_id(p);
            ++result.terminal_paths;
            if (excluded(false,id)) { ++result.excluded_paths; continue; }
            terminals.push_back({id,e.delivery,reduced(p,false,id)});
        }
        if (profiles.size()>=128) profiles.clear();
        profiles.emplace(&l,terminals);
        return terminals;
    };
    auto dominates=[&](const Label& a,const Label& b) {
        // Cheap resource/cost screen is not itself a dominance proof. Exact
        // extension inclusion below is authoritative; no dual-sign assumption.
        if (a.busy>b.busy || a.next->q>b.next->q || a.reduced>b.reduced) return false;
        const auto& aa=profile(a); const auto& bb=profile(b);
        if (interrupted) return false;
        std::size_t pos=0;
        for (const auto& t:bb) {
            while (pos<aa.size() && aa[pos].delivery<t.delivery) ++pos;
            if (pos==aa.size() || aa[pos].delivery!=t.delivery || aa[pos].reduced>t.reduced) return false;
        }
        return true;
    };
    for (auto& entry:buckets) {
        profiles.clear();
        auto& labels=entry.second;
        std::stable_sort(labels.begin(),labels.end(),[](const Label& a,const Label& b) {
            return std::tie(a.reduced,a.busy)<std::tie(b.reduced,b.busy);
        });
        std::vector<std::size_t> active;
        for (std::size_t i=0;i<labels.size();++i) {
            if (stopped()) return result;
            bool discard=false;
            for (std::size_t j:active) if (dominates(labels[j],labels[i])) { discard=true; break; }
            if (interrupted) return result;
            if (discard) { ++result.labels_dominated; continue; }
            // Equal-cost labels can dominate earlier ones through better
            // resources or a larger eligible completion set.
            for (auto it=active.begin();it!=active.end();) {
                if (dominates(labels[i],labels[*it])) { it=active.erase(it); ++result.labels_dominated; }
                else ++it;
                if (interrupted) return result;
            }
            active.push_back(i);
        }
        for (std::size_t i:active) {
            ++result.labels_extended;
            const auto& l=labels[i];
            for (const auto& terminal:profile(l)) {
                const auto e=std::find_if(l.next->deliveries.begin(),l.next->deliveries.end(),
                    [&](const DeliveryExtension& x){return x.delivery==terminal.delivery;});
                PathContinuation c{l.next->od,l.next->pickup,e->delivery,l.next->transfer,e->loaded,e->back,l.next->wait};
                ServicePath temporary;
                const auto* path=physical_service(l.segment,&c,temporary);
                if (!path) throw std::logic_error("Stable suffix ceased to be feasible");
                record(*path,terminal.reduced);
            }
            if (interrupted) return result;
        }
    }
    std::sort(candidates.begin(),candidates.end(),less);
    // IDs are assigned only to returned columns, after the complete search;
    // generated columns do not alter RMP membership or exclusions mid-search.
    for (auto& c:candidates) {
        if (c.empty) result.empty.push_back(register_empty_path(paths,std::move(c.relocation)));
        else result.service.push_back(register_service_path(paths,std::move(c.service)));
    }
    result.minimum=std::min(result.min_service,result.min_empty);
    // Demand conservation gives sum(service task flows)<=total demand, since
    // each task serves at least one order. Empty variables only have their
    // individual total-demand bounds. This corrects ALL omitted columns even
    // when dominated negative completions were not retained in the batch.
    result.old_service_correction=total*std::max(0.0,-result.min_service);
    result.chosen_service_correction=result.old_service_correction;
    result.empty_correction=total*omitted_empty*std::max(0.0,-result.min_empty);
    result.correction=result.old_service_correction+result.empty_correction;
    result.complete=g.complete && !interrupted && !stopped();
    if(!options.branch_correction || !result.complete)return result;

    const double branch_started=wall_time();
    auto finish_branch=[&](const char* status) {
        result.branch_correction_status=status;
        result.branch_seconds=wall_time()-branch_started;
        return result;
    };
    if(!d.preprocessed_demand || !d.origin_assignment() || d.flow_policy!="aggregate" ||
       d.first_leg_consistency)return finish_branch("unsupported_scope");
    result.branch_correction_scope="preprocessed_origin_aggregate";
    if(!paths.complete)return finish_branch("incomplete_catalogue");
    if(std::isnan(options.branch_deadline))
        throw std::invalid_argument("Invalid branch pricing deadline");
    const double branch_deadline=std::min(options.deadline,options.branch_deadline);
    auto branch_stopped=[&] {return wall_time()>=branch_deadline;};
    if(branch_stopped())return finish_branch("budget_exhausted");
    if(d.single_rates.size()!=static_cast<std::size_t>(d.n_od) ||
       d.pair_rates.rows!=d.n_od || d.pair_rates.cols!=d.n_od)
        throw std::invalid_argument("Branch pricing task-rate dimensions disagree");
    for(double rate:d.single_rates)if(!std::isfinite(rate)||rate<0)
        throw std::invalid_argument("Invalid branch pricing single rate");
    for(double rate:d.pair_rates.values)if(!std::isfinite(rate)||rate<0)
        throw std::invalid_argument("Invalid branch pricing pair rate");
    const int branches=d.n_od+1;
    std::vector<long double> minima(static_cast<std::size_t>(d.n_od)*d.n_l*branches,0);
    auto index=[&](int r,int l,int s) {return (static_cast<std::size_t>(r)*d.n_l+l)*branches+s+1;};
    // The global oracle can dominate across first-head identities. Only a
    // complete, independent scan below certifies every (head, base, branch)
    // minimum; neither its retained labels nor its capped column batch do.
    for(std::size_t i=0;i<service_count;++i) {
        if(i%128==0&&branch_stopped())return finish_branch("budget_exhausted");
        ++result.branch_paths_scanned;
        if(excluded(false,static_cast<int>(i)))continue;
        const auto& p=paths.service_paths[i];
        if(!service_path_enabled(f,p))continue;
        if(p.first_od<0||p.first_od>=d.n_od||p.b[0]<0||p.b[0]>=d.n_l||
           p.second_od < -1||p.second_od>=d.n_od)
            throw std::logic_error("Complete pricing catalogue has an invalid branch identity");
        long double rc=p.cost+(i<missing_service.size()?static_cast<long double>(missing_service[i]):0);
        long double magnitude=1+std::abs(rc);
        auto subtract=[&](int row,double coefficient) {
            const long double term=static_cast<long double>(coefficient)*prices.at(row);
            rc-=term;magnitude+=std::abs(term);
        };
        if(options.fast_pricing)for_each_path_coefficient(f,p,subtract);
        else for(const auto& [row,coefficient]:path_column(f,p))subtract(row,coefficient);
        if(!std::isfinite(rc)||!std::isfinite(magnitude))return finish_branch("invalid_numerics");
        // Downward reduced-cost margin and upward final conversion protect
        // the new correction independently of the legacy double arithmetic.
        rc-=64*std::numeric_limits<double>::epsilon()*magnitude;
        auto& minimum=minima[index(p.first_od,p.b[0],p.second_od)];
        minimum=std::min(minimum,rc);
    }
    if(branch_stopped())return finish_branch("budget_exhausted");
    long double correction=0;
    for(int q=0;q<d.task_count();++q) {
        const int r=d.task_first(q),s=d.task_second(q);
        const double rate=d.task_rate(q);
        if(rate<=0)continue;
        long double largest=0;
        for(int l=0;l<d.n_l;++l) {
            ++result.branch_groups;
            largest=std::max(largest,-minima[index(r,l,s)]);
        }
        correction=add_nonnegative_product_up(correction,rate,largest);
    }
    if(!std::isfinite(correction))return finish_branch("invalid_numerics");
    const long double safe=correction+64*std::numeric_limits<double>::epsilon()*(1+correction);
    const double rounded=std::nextafter(static_cast<double>(safe),std::numeric_limits<double>::infinity());
    if(!std::isfinite(rounded))return finish_branch("invalid_numerics");
    result.branch_correction_complete=true;
    result.branch_service_correction=rounded;
    result.chosen_service_correction=std::min(result.old_service_correction,rounded);
    result.branch_correction_used=rounded<result.old_service_correction;
    result.correction=result.chosen_service_correction+result.empty_correction;
    return finish_branch("complete");
}

LabelPricingResult reprice_discovered_columns(const Formulation& f,const PathCollection& paths,
        const std::vector<double>& prices,const LabelPricingResult& discovery,double tolerance) {
    if(prices.size()!=f.matrix.rows.size() || !std::isfinite(tolerance) || tolerance<0)
        throw std::invalid_argument("Invalid original-dual discovery filter");
    for(double value:prices)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite original pricing dual");
    LabelPricingResult result;
    result.discovery_only=true;
    auto reduced=[&](const auto& p) {
        double value=p.cost;
        for_each_path_coefficient(f,p,[&](int row,double coefficient){value-=coefficient*prices.at(row);});
        if(!std::isfinite(value))throw std::invalid_argument("Nonfinite discovery reduced cost");
        return value;
    };
    for(int id:discovery.service) {
        const auto& p=paths.service_paths.at(id);
        if(!service_path_enabled(f,p) || (static_cast<std::size_t>(id)<f.flow.size() && f.flow[id]>=0))continue;
        const double rc=reduced(p);
        if(rc < -tolerance) {result.service.push_back(id);result.min_service=std::min(result.min_service,rc);}
    }
    for(int id:discovery.empty) {
        const auto& p=paths.empty_paths.at(id);
        if(static_cast<std::size_t>(id)<f.empty_flow.size() && f.empty_flow[id]>=0)continue;
        const double rc=reduced(p);
        if(rc < -tolerance) {result.empty.push_back(id);result.min_empty=std::min(result.min_empty,rc);}
    }
    result.minimum=std::min(result.min_service,result.min_empty);
    result.negative_paths=result.service.size()+result.empty.size();
    return result;
}
} // namespace mlnd
