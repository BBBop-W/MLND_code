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
}

struct LabelPricingOracle::Graph {
    const Input* data;
    std::unordered_map<SegmentKey,int,KeyHash<5>> segments;
    PathCollection* paths;
    // Input-derived event graph: first-delivery -> second-pickup ->
    // second-delivery -> return. No completed catalogue paths are traversed.
    std::vector<std::vector<PickupExtension>> pickups;
    bool complete=false;
};

LabelPricingOracle::LabelPricingOracle(const Input& d,PathCollection& paths,double deadline) {
    if (d.route_mode!="direct" || d.has_candidate_routes)
        throw std::invalid_argument("Label pricing supports the direct-flight K=1/K=2 universe only");
    if (!paths.support_complete) throw std::invalid_argument("Label pricing requires complete model-support metadata");
    auto g=std::make_shared<Graph>();
    g->data=&d; g->paths=&paths; g->complete=false;
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
                if (e.q+loaded+d.receipt_mean(s,k)>d.deadline[s]+resource_tolerance) continue;
                e.deliveries.push_back({k,loaded,back});
                e.minimum_remaining=std::min(e.minimum_remaining,loaded+back);
            }
            if (!e.deliveries.empty()) extensions.push_back(std::move(e));
        }
    }
    g->complete=!stopped();
}

LabelPricingResult LabelPricingOracle::price(const Formulation& f,const std::vector<double>& prices,
        const std::vector<double>& missing_service,const std::vector<double>& missing_empty,
        const LabelPricingOptions& options) const {
    const auto& g=*graph_;
    const auto& d=*g.data;
    auto& paths=*g.paths;
    const auto service_count=paths.service_paths.size(), empty_count=paths.empty_paths.size();
    LabelPricingResult result;
    result.dominance="same-second-pickup-exact-suffix-extension";
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
            if constexpr (empty) { c.relocation=path; c.signature={path.b[0],path.b[1],-1,-1,-1,-1,-1,-1}; }
            else { c.service=path; c.signature=service_path_signature(path); }
            retain(std::move(c));
        }
    };
    auto reduced=[&](const auto& path,bool empty,int id) {
        const auto& upper=empty?missing_empty:missing_service;
        double value=path.cost+(id>=0 && static_cast<std::size_t>(id)<upper.size()?upper[id]:0);
        for (auto [row,coefficient]:path_column(f,path)) value-=coefficient*prices.at(row);
        return value;
    };
    auto service_id=[&](const ServicePath& path) {
        const auto it=paths.service_registry.find(service_path_signature(path));
        return it==paths.service_registry.end()?-1:it->second;
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
        ServicePath p;
        if (build_service_path(d,source.segment,nullptr,p)) {
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
            ServicePath p;
            ++result.labels_created;
            if (!build_service_path(d,l.segment,&c,p)) { ++result.resource_pruned; continue; }
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
                ServicePath p;
                if (!build_service_path(d,l.segment,&c,p)) throw std::logic_error("Stable suffix ceased to be feasible");
                record(p,terminal.reduced);
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
    result.correction=total*std::max(0.0,-result.min_service)
                     +total*omitted_empty*std::max(0.0,-result.min_empty);
    result.complete=g.complete && !interrupted && !stopped();
    return result;
}
} // namespace mlnd
