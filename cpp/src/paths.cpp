#include "mlnd/paths.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace mlnd {
namespace {
constexpr double tolerance = 1e-12;

// Error-free partial sums, followed by the same final half-even correction as
// Python math.fsum. Path resource totals contain only a few nonnegative terms.
double accurate_sum(const std::vector<double>& values) {
    std::vector<double> partials;
    for (double x : values) {
        std::size_t kept = 0;
        for (double y : partials) {
            if (std::abs(x) < std::abs(y)) std::swap(x, y);
            const double high = x + y;
            const double low = y - (high - x);
            if (low != 0) partials[kept++] = low;
            x = high;
        }
        partials.resize(kept);
        if (x != 0) partials.push_back(x);
    }
    if (partials.empty()) return 0;
    double high = partials.back(), low = 0;
    partials.pop_back();
    while (!partials.empty()) {
        const double x = high, y = partials.back();
        partials.pop_back();
        high = x + y;
        low = y - (high - x);
        if (low != 0) break;
    }
    if (!partials.empty() && ((low < 0 && partials.back() < 0) ||
                              (low > 0 && partials.back() > 0))) {
        const double twice_low = low * 2, corrected = high + twice_low;
        if (corrected - high == twice_low) high = corrected;
    }
    return high;
}

ArcCounts count_arcs(const Legs& legs) {
    std::map<std::pair<int, int>, int> counts;
    for (const auto& leg : legs)
        for (std::size_t p = 1; p < leg.size(); ++p)
            ++counts[{leg[p - 1], leg[p]}];
    ArcCounts out;
    for (const auto& entry : counts)
        out.emplace_back(entry.first.first, entry.first.second, entry.second);
    return out;
}

std::vector<int> direct_leg(int origin, int destination) {
    return origin == destination ? std::vector<int>{origin}
                                 : std::vector<int>{origin, destination};
}

bool first_segment(const Input& data, BasePair b, int r, int i, int j,
                   int id, InitialSegment& segment) {
    const int pickup = data.n_l + i, delivery = data.n_l + j;
    if (i == j || !data.allowed_links(b[0], pickup) ||
        !data.allowed_links(pickup, delivery)) return false;
    const double t0 = data.flight_times(b[0], pickup);
    const double t1 = data.flight_times(pickup, delivery);
    if (t1 + tolerance < data.effective_delta()) return false;
    const double q1 = t0 + first_wait(data.rider_mean(r, i), t0);
    const double final = q1 + t1 + data.receipt_mean(r, j);
    if (final > data.deadline[r] + tolerance) return false;
    segment = {id, b, r, i, j,
               {direct_leg(b[0], pickup), direct_leg(pickup, delivery)}, t0, t1};
    return true;
}

using Continuation = PathContinuation;

bool service_path(const Input& data, const InitialSegment& segment,
                  const Continuation* continuation, ServicePath& path) {
    const int r = segment.first_od, i = segment.pickup, j = segment.delivery;
    const double q1 = segment.t0 + first_wait(data.rider_mean(r, i), segment.t0);
    path = {};
    path.segment_id = segment.id;
    path.b = segment.b;
    path.first_od = r;
    path.mode = continuation ? "double" : "single";
    path.pickups = {i};
    path.deliveries = {j};
    path.legs = segment.legs;
    path.pickup_times = {q1};
    std::vector<int> ods{r};
    std::vector<double> loaded_times{segment.t1};
    std::vector<double> durations{segment.t0, segment.t1};
    path.wait = q1 - segment.t0;
    if (continuation) {
        const auto& c = *continuation;
        path.second_od = c.od;
        path.pickups.push_back(c.pickup);
        path.deliveries.push_back(c.delivery);
        path.legs.push_back(direct_leg(data.n_l + j, data.n_l + c.pickup));
        path.legs.push_back(direct_leg(data.n_l + c.pickup, data.n_l + c.delivery));
        path.legs.push_back(direct_leg(data.n_l + c.delivery, segment.b[1]));
        path.pickup_times.push_back(c.transfer + data.effective_delta() / 2 + c.wait);
        ods.push_back(c.od);
        loaded_times.push_back(c.loaded);
        durations.insert(durations.end(), {c.transfer, c.loaded, c.back});
        path.wait += c.wait;
    } else {
        const int from = data.n_l + j, to = segment.b[1];
        if (!data.allowed_links(from, to)) return false;
        path.legs.push_back(direct_leg(from, to));
        durations.push_back(data.flight_times(from, to));
    }
    path.flight_time = accurate_sum(durations);
    path.time = path.flight_time + path.wait;
    if (path.time > data.operating_limit + tolerance) return false;
    std::map<int, int> served;
    std::map<int, double> slots;
    std::vector<double> courier_cost;
    for (std::size_t k = 0; k < ods.size(); ++k) {
        const int od = ods[k], pickup = path.pickups[k], delivery = path.deliveries[k];
        const double q = path.pickup_times[k], t = loaded_times[k];
        const double receipt = q + t + data.receipt_mean(od, delivery);
        if (receipt > data.deadline[od] + tolerance) return false;
        path.delivery_times.push_back(receipt);
        ++served[od];
        slots[pickup] += q;
        slots[delivery] += q + t + data.removal_mean(od, delivery);
        courier_cost.push_back(data.first_cost(od, pickup) + data.terminal_cost(od, delivery));
    }
    path.served_counts.assign(served.begin(), served.end());
    path.slot_times.assign(slots.begin(), slots.end());
    path.arc_counts = count_arcs(path.legs);
    path.cost = accurate_sum(courier_cost) + data.flight_cost * path.flight_time
                + data.waiting_cost * path.wait;
    return true;
}
} // namespace

bool build_initial_segment(const Input& data, BasePair b, int first_od,
                           int pickup, int delivery, int id, InitialSegment& result) {
    return first_segment(data, b, first_od, pickup, delivery, id, result);
}
bool build_service_path(const Input& data, const InitialSegment& segment,
                        const PathContinuation* next, ServicePath& result) {
    return service_path(data, segment, next, result);
}

ServiceSignature service_path_signature(const ServicePath& p) {
    return {p.b[0],p.b[1],p.first_od,p.pickups.at(0),p.deliveries.at(0),p.second_od,
            p.second_od<0?-1:p.pickups.at(1),p.second_od<0?-1:p.deliveries.at(1)};
}
void rebuild_path_registry(PathCollection& paths, std::function<void()> checkpoint) {
    paths.service_registry.clear(); paths.empty_registry.clear();
    for (std::size_t i=0;i<paths.service_paths.size();++i) {
        if (checkpoint && i%1024==0) checkpoint();
        const auto& p=paths.service_paths[i];
        if (p.id!=static_cast<int>(i) || !paths.service_registry.emplace(service_path_signature(p),p.id).second)
            throw std::invalid_argument("Invalid or duplicate generated service ID");
    }
    for (std::size_t i=0;i<paths.empty_paths.size();++i) {
        if (checkpoint && i%1024==0) checkpoint();
        const auto& p=paths.empty_paths[i];
        if (p.id!=static_cast<int>(i) || !paths.empty_registry.emplace(p.b,p.id).second)
            throw std::invalid_argument("Invalid or duplicate generated empty ID");
    }
}
int register_service_path(PathCollection& paths, ServicePath p) {
    if (paths.service_registry.size()!=paths.service_paths.size() || paths.empty_registry.size()!=paths.empty_paths.size())
        rebuild_path_registry(paths);
    const auto signature=service_path_signature(p);
    const auto found=paths.service_registry.find(signature);
    if (found!=paths.service_registry.end()) return found->second;
    p.id=static_cast<int>(paths.service_paths.size());
    paths.service_registry.emplace(signature,p.id);
    paths.service_paths.push_back(std::move(p));
    return static_cast<int>(paths.service_paths.size())-1;
}
int register_empty_path(PathCollection& paths, EmptyPath p) {
    if (paths.service_registry.size()!=paths.service_paths.size() || paths.empty_registry.size()!=paths.empty_paths.size())
        rebuild_path_registry(paths);
    const auto found=paths.empty_registry.find(p.b);
    if (found!=paths.empty_registry.end()) return found->second;
    p.id=static_cast<int>(paths.empty_paths.size());
    paths.empty_registry.emplace(p.b,p.id); paths.empty_paths.push_back(std::move(p));
    return static_cast<int>(paths.empty_paths.size())-1;
}

PathCollection prepare_path_support(const Input& data,
        std::optional<double> seconds,std::function<bool()> stop_requested,
        std::function<void(std::size_t,std::size_t,std::size_t)> progress) {
    if (data.route_mode!="direct" || data.has_candidate_routes)
        throw std::invalid_argument("Implicit path support requires direct routes and null candidate_routes");
    if (seconds && (!std::isfinite(*seconds) || *seconds<0))
        throw std::invalid_argument("Path support time limit must be finite and nonnegative");
    const auto started=std::chrono::steady_clock::now();
    PathCollection paths; paths.complete=false; paths.support_complete=false;
    std::size_t checked=0;
    for (int from=0;from<data.n_l;++from) for (int to=0;to<data.n_l;++to)
        for (int r=0;r<data.n_od;++r) for (int i=0;i<data.n_h;++i) for (int j=0;j<data.n_h;++j) {
            if (progress && (++checked%4096==0)) progress(0,0,paths.initial_segments.size());
            if (stop_requested && stop_requested()) throw EnumerationStopped("user_stop",0,paths.initial_segments.size());
            if (seconds && std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=*seconds)
                throw EnumerationStopped("time_limit",0,paths.initial_segments.size());
            InitialSegment segment;
            if (first_segment(data,{from,to},r,i,j,static_cast<int>(paths.initial_segments.size()),segment))
                paths.initial_segments.push_back(std::move(segment));
        }
    paths.support_complete=true;
    if (progress) progress(0,0,paths.initial_segments.size());
    return paths;
}

std::pair<std::size_t,std::size_t> greedy_seed_paths(const Input& d,PathCollection& paths,
        std::size_t per_group,double deadline) {
    if (!paths.support_complete) throw std::invalid_argument("Greedy seed needs complete row support");
    if (!per_group) return {0,0};
    const auto old_service=paths.service_paths.size(),old_empty=paths.empty_paths.size();
    auto stopped=[&] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline; };
    using Group=std::array<int,3>;
    // Rank only first segments; no full double-task domain is constructed.
    std::map<Group,std::vector<std::pair<double,int>>> prefixes;
    for (const auto& s:paths.initial_segments) {
        if (stopped()) break;
        ServicePath p;
        const bool single=build_service_path(d,s,nullptr,p);
        const double busy=s.t0+s.t1+first_wait(d.rider_mean(s.first_od,s.pickup),s.t0);
        const double score=single?p.cost+d.fleet_cost*p.time:
            d.first_cost(s.first_od,s.pickup)+d.terminal_cost(s.first_od,s.delivery)+
            (d.flight_cost+d.fleet_cost)*(s.t0+s.t1)+(d.waiting_cost+d.fleet_cost)*(busy-s.t0-s.t1);
        auto& best=prefixes[{s.b[0],s.b[1],s.first_od}];
        best.emplace_back(score,s.id); std::sort(best.begin(),best.end());
        if (best.size()>per_group) best.resize(per_group);
    }
    for (const auto& group:prefixes) for (const auto& candidate:group.second) {
        if (stopped()) break;
        const auto& s=paths.initial_segments.at(candidate.second);
        ServicePath single;
        if (build_service_path(d,s,nullptr,single)) register_service_path(paths,std::move(single));
        if (d.effective_delta()==0) continue;
        ServicePath best; double score=std::numeric_limits<double>::infinity(); bool found=false;
        for (int r=0;r<d.n_od && !stopped();++r) {
            if (d.kappa(s.first_od,r)<=0) continue;
            for (int i=0;i<d.n_h && !stopped();++i) {
                if (i!=s.delivery && !d.allowed_links(d.n_l+s.delivery,d.n_l+i)) continue;
                const double transfer=d.flight_times(d.n_l+s.delivery,d.n_l+i);
                const double wait=second_wait(d.rider_mean(r,i),transfer,d.effective_delta());
                for (int j=0;j<d.n_h;++j) {
                    if (i==j || !d.allowed_links(d.n_l+i,d.n_l+j) || !d.allowed_links(d.n_l+j,s.b[1])) continue;
                    PathContinuation c{r,i,j,transfer,d.flight_times(d.n_l+i,d.n_l+j),d.flight_times(d.n_l+j,s.b[1]),wait};
                    ServicePath p;
                    if (!build_service_path(d,s,&c,p)) continue;
                    // Prefer total two-order saving, not individual K=1 gain.
                    const double value=p.cost+d.fleet_cost*p.time-d.ground_cost[s.first_od]-d.ground_cost[r];
                    if (value<score) { score=value; best=std::move(p); found=true; }
                }
            }
        }
        if (found) register_service_path(paths,std::move(best));
    }
    for (int from=0;from<d.n_l;++from) for (int to=0;to<d.n_l;++to) {
        if (stopped()) break;
        if (from==to || !d.allowed_links(from,to)) continue;
        const double time=d.flight_times(from,to); if (time>d.operating_limit+tolerance) continue;
        EmptyPath p; p.b={from,to}; p.legs={direct_leg(from,to)}; p.arc_counts=count_arcs(p.legs);
        p.flight_time=p.time=time; p.cost=d.flight_cost*time; register_empty_path(paths,std::move(p));
    }
    return {paths.service_paths.size()-old_service,paths.empty_paths.size()-old_empty};
}

BranchSeedReport branch_covering_seed_paths(const Input& d, PathCollection& paths,
        const std::vector<double>* assignment_mask, double deadline) {
    if (!paths.support_complete) throw std::invalid_argument("Branch seed needs complete row support");
    if (assignment_mask) {
        if (assignment_mask->size()!=static_cast<std::size_t>(d.assignment_count()))
            throw std::invalid_argument("Branch seed assignment mask has wrong size");
        for (double value:*assignment_mask)
            if (value!=0 && value!=1) throw std::invalid_argument("Branch seed assignment mask must be binary");
    }
    if (std::isnan(deadline)) throw std::invalid_argument("Branch seed deadline must not be NaN");
    BranchSeedReport report;
    const auto old_service=paths.service_paths.size(),old_empty=paths.empty_paths.size();
    auto stopped=[&] {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline)
            report.stopped=true;
        return report.stopped;
    };
    auto assigned=[&](int r, BasePair b) {
        return d.demand[r]>0 && (!assignment_mask || (*assignment_mask)[d.assignment_offset(r,b[0],b[1])]!=0);
    };
    using Group=std::array<int,3>;
    using Branch=std::array<int,4>;
    std::map<Group,std::vector<std::pair<double,int>>> groups;
    // This stores first-segment support, not combinations of task paths.
    for (const auto& s:paths.initial_segments) {
        if (stopped()) break;
        if (!assigned(s.first_od,s.b)) continue;
        ServicePath p;
        const bool single=build_service_path(d,s,nullptr,p);
        const double wait=first_wait(d.rider_mean(s.first_od,s.pickup),s.t0);
        const double score=single?p.cost+d.fleet_cost*p.time:
            d.first_cost(s.first_od,s.pickup)+d.terminal_cost(s.first_od,s.delivery)+
            (d.flight_cost+d.fleet_cost)*(s.t0+s.t1)+(d.waiting_cost+d.fleet_cost)*wait;
        groups[{s.b[0],s.b[1],s.first_od}].emplace_back(score,s.id);
    }
    std::set<Branch> existing;
    for (const auto& p:paths.service_paths)
        existing.insert({p.b[0],p.b[1],p.first_od,p.second_od});
    // Search all second-leg choices only for the current cheap prefix. The
    // first prefix admitting a completion wins; exact CG later improves cost.
    auto completion=[&](const InitialSegment& s,int r,ServicePath& best) {
        if (r<0) return build_service_path(d,s,nullptr,best);
        bool found=false;
        double best_score=std::numeric_limits<double>::infinity();
        for (int i=0;i<d.n_h && !stopped();++i) {
            if (i!=s.delivery && !d.allowed_links(d.n_l+s.delivery,d.n_l+i)) continue;
            const double transfer=d.flight_times(d.n_l+s.delivery,d.n_l+i);
            const double wait=second_wait(d.rider_mean(r,i),transfer,d.effective_delta());
            for (int j=0;j<d.n_h && !stopped();++j) {
                if (i==j || !d.allowed_links(d.n_l+i,d.n_l+j) || !d.allowed_links(d.n_l+j,s.b[1])) continue;
                PathContinuation c{r,i,j,transfer,d.flight_times(d.n_l+i,d.n_l+j),
                                   d.flight_times(d.n_l+j,s.b[1]),wait};
                ServicePath p;
                if (!build_service_path(d,s,&c,p)) continue;
                const double score=p.cost+d.fleet_cost*p.time;
                if (score<best_score) { best_score=score; best=std::move(p); found=true; }
            }
        }
        return found;
    };
    for (auto& entry:groups) {
        const auto& key=entry.first;
        std::sort(entry.second.begin(),entry.second.end());
        std::vector<int> branches{-1};
        if (d.effective_delta()>0)
            for (int r=0;r<d.n_od;++r)
                if (assigned(r,{key[0],key[1]}) && d.kappa(key[2],r)>0) branches.push_back(r);
        ++report.groups_requested;
        report.required_branches+=branches.size();
        if (stopped()) continue;
        if (d.flow_policy=="aggregate") {
            bool group_complete=true;
            for (int r:branches) {
                const Branch branch{key[0],key[1],key[2],r};
                if (existing.count(branch)) { ++report.covered_branches; continue; }
                bool found=false;
                for (const auto& ranked:entry.second) {
                    if (stopped()) break;
                    ServicePath p;
                    if (!completion(paths.initial_segments.at(ranked.second),r,p)) continue;
                    register_service_path(paths,std::move(p)); existing.insert(branch);
                    ++report.covered_branches; found=true; break;
                }
                if (!found) {
                    group_complete=false;
                    if (!report.stopped) ++report.physically_missing_branches;
                }
            }
            report.groups_completed+=group_complete;
        } else {
            // Prefix policy has separate branch rows for every initial segment.
            // Covering different branches with different prefixes cannot release
            // those rows. Register a complete bundle on one common prefix.
            bool found=false;
            for (const auto& ranked:entry.second) {
                if (stopped()) break;
                std::vector<ServicePath> bundle;
                for (int r:branches) {
                    ServicePath p;
                    if (stopped() || !completion(paths.initial_segments.at(ranked.second),r,p)) break;
                    bundle.push_back(std::move(p));
                }
                if (bundle.size()!=branches.size()) continue;
                for (auto& p:bundle) register_service_path(paths,std::move(p));
                report.covered_branches+=branches.size();
                ++report.groups_completed; found=true; break;
            }
            if (!found && !report.stopped) ++report.groups_without_complete_prefix;
        }
    }
    for (int from=0;from<d.n_l;++from) for (int to=0;to<d.n_l;++to) {
        if (stopped()) break;
        if (from==to || !d.allowed_links(from,to)) continue;
        const double time=d.flight_times(from,to);
        if (time>d.operating_limit+tolerance) continue;
        EmptyPath p; p.b={from,to}; p.legs={direct_leg(from,to)}; p.arc_counts=count_arcs(p.legs);
        p.flight_time=p.time=time; p.cost=d.flight_cost*time;
        register_empty_path(paths,std::move(p));
    }
    report.service_added=paths.service_paths.size()-old_service;
    report.empty_added=paths.empty_paths.size()-old_empty;
    return report;
}

EnumerationStopped::EnumerationStopped(std::string why, std::size_t count,
                                       std::size_t segments)
    : std::runtime_error("Path enumeration stopped: " + why),
      reason(std::move(why)), columns(count), initial_segments(segments) {}

double first_wait(double mean, double travel) {
    if (mean < 0 || travel < 0)
        throw std::invalid_argument("Courier mean and flight time must be nonnegative");
    return mean > 0 ? mean * std::exp(-travel / mean) : 0;
}

double second_wait(double mean, double travel, double delta) {
    if (mean < 0 || travel < 0 || delta < 0)
        throw std::invalid_argument("Courier mean, flight time and window must be nonnegative");
    if (mean == 0) return 0;
    if (delta == 0) return first_wait(mean, travel);
    return first_wait(mean, travel) * (-std::expm1(-delta / mean)) / (delta / mean);
}

PathCollection enumerate_paths(const Input& data,
                               std::optional<std::size_t> column_limit,
                               std::optional<double> time_limit_seconds,
                               std::function<bool()> stop_requested,
                               std::function<void(std::size_t,std::size_t,std::size_t)> progress) {
    if (data.route_mode != "direct" || data.has_candidate_routes)
        throw std::invalid_argument("Native path enumeration currently requires direct routes and null candidate_routes");
    if (time_limit_seconds && (!std::isfinite(*time_limit_seconds) || *time_limit_seconds < 0))
        throw std::invalid_argument("Path time limit must be finite and nonnegative");
    const auto started = std::chrono::steady_clock::now();
    PathCollection result;
    std::size_t checked=0;
    auto stop = [&](const std::string& reason) {
        throw EnumerationStopped(reason, result.service_paths.size() + result.empty_paths.size(),
                                 result.initial_segments.size());
    };
    auto check = [&]() {
        if (progress && (++checked%4096==0))
            progress(result.service_paths.size(),result.empty_paths.size(),result.initial_segments.size());
        if (stop_requested && stop_requested()) stop("user_stop");
        if (time_limit_seconds && std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count() >= *time_limit_seconds)
            stop("time_limit");
    };
    auto check_append = [&]() {
        check();
        if (column_limit && result.service_paths.size() + result.empty_paths.size() >= *column_limit)
            stop("column_limit");
    };
    // Every cached continuation still passes all authoritative scalar receipt
    // and mission checks; this cache excludes only unavailable arcs.
    std::map<std::pair<int, int>, std::vector<Continuation>> continuations;
    auto candidates = [&](int first_delivery, int return_base) -> const std::vector<Continuation>& {
        const auto key = std::make_pair(first_delivery, return_base);
        auto found = continuations.find(key);
        if (found != continuations.end()) return found->second;
        std::vector<Continuation> list;
        const int start = data.n_l + first_delivery;
        for (int s = 0; s < data.n_od; ++s) {
            check();
            for (int i = 0; i < data.n_h; ++i) {
                const int pickup = data.n_l + i;
                if (i != first_delivery && !data.allowed_links(start, pickup)) continue;
                const double transfer = data.flight_times(start, pickup);
                const double wait = second_wait(data.rider_mean(s, i), transfer, data.effective_delta());
                for (int j = 0; j < data.n_h; ++j) {
                    check();
                    const int delivery = data.n_l + j;
                    if (i == j || !data.allowed_links(pickup, delivery) ||
                        !data.allowed_links(delivery, return_base)) continue;
                    const double loaded = data.flight_times(pickup, delivery);
                    const double back = data.flight_times(delivery, return_base);
                    // Exact q2/receipt expression from build_service_path.
                    const double q2 = transfer + data.effective_delta() / 2 + wait;
                    const double receipt = q2 + loaded + data.receipt_mean(s, j);
                    if (receipt > data.deadline[s] + tolerance) continue;
                    list.push_back({s, i, j, transfer, loaded, back, wait});
                }
            }
        }
        return continuations.emplace(key, std::move(list)).first->second;
    };

    for (int origin = 0; origin < data.n_l; ++origin) {
        for (int destination = 0; destination < data.n_l; ++destination) {
            check();
            const BasePair b{origin, destination};
            if (origin != destination && data.allowed_links(origin, destination)) {
                const double flight = data.flight_times(origin, destination);
                if (flight <= data.operating_limit + tolerance) {
                    check_append();
                    EmptyPath path;
                    path.id = static_cast<int>(result.empty_paths.size());
                    path.b = b;
                    path.legs = {direct_leg(origin, destination)};
                    path.arc_counts = count_arcs(path.legs);
                    path.flight_time = path.time = flight;
                    path.cost = data.flight_cost * flight;
                    result.empty_paths.push_back(std::move(path));
                }
            }
            for (int r = 0; r < data.n_od; ++r) {
                for (int i = 0; i < data.n_h; ++i) {
                    for (int j = 0; j < data.n_h; ++j) {
                        check();
                        InitialSegment segment;
                        if (!first_segment(data, b, r, i, j,
                                           static_cast<int>(result.initial_segments.size()), segment)) continue;
                        result.initial_segments.push_back(segment);
                        ServicePath path;
                        if (service_path(data, segment, nullptr, path)) {
                            check_append();
                            path.id = static_cast<int>(result.service_paths.size());
                            result.service_paths.push_back(std::move(path));
                        }
                        if (data.effective_delta() == 0) continue;
                        const double initial = segment.t0 + segment.t1 +
                            first_wait(data.rider_mean(r, i), segment.t0);
                        for (const auto& c : candidates(j, destination)) {
                            check();
                            if (data.kappa(r, c.od) <= 0) continue;
                            // Conservative cheap check. All boundary survivors
                            // are rebuilt with accurate_sum before acceptance.
                            const double approximate = initial + c.transfer + c.loaded + c.back + c.wait;
                            const double slack = tolerance + 64 * std::numeric_limits<double>::epsilon() *
                                std::max(1.0, std::abs(data.operating_limit));
                            if (approximate > data.operating_limit + slack) continue;
                            if (!service_path(data, segment, &c, path)) continue;
                            check_append();
                            path.id = static_cast<int>(result.service_paths.size());
                            result.service_paths.push_back(std::move(path));
                        }
                    }
                }
            }
        }
    }
    rebuild_path_registry(result,check);
    if (progress) progress(result.service_paths.size(),result.empty_paths.size(),result.initial_segments.size());
    return result;
}

void to_json(Json& out, const InitialSegment& p) {
    out = {{"id", p.id}, {"b", p.b}, {"first_od", p.first_od}, {"pickup", p.pickup},
           {"delivery", p.delivery}, {"legs", p.legs}, {"t0", p.t0}, {"t1", p.t1}};
}
void to_json(Json& out, const ServicePath& p) {
    out = {{"id", p.id}, {"segment_id", p.segment_id}, {"b", p.b},
           {"first_od", p.first_od}, {"second_od", p.second_od < 0 ? Json(nullptr) : Json(p.second_od)},
           {"mode", p.mode}, {"pickups", p.pickups}, {"deliveries", p.deliveries},
           {"legs", p.legs}, {"served_counts", p.served_counts}, {"arc_counts", p.arc_counts},
           {"flight_time", p.flight_time}, {"time", p.time}, {"wait", p.wait},
           {"slot_times", p.slot_times}, {"pickup_times", p.pickup_times},
           {"delivery_times", p.delivery_times}, {"cost", p.cost}};
}
void to_json(Json& out, const EmptyPath& p) {
    out = {{"id", p.id}, {"b", p.b}, {"legs", p.legs}, {"arc_counts", p.arc_counts},
           {"flight_time", p.flight_time}, {"time", p.time}, {"cost", p.cost}};
}
void to_json(Json& out, const PathCollection& paths) {
    out = {{"initial_segments", paths.initial_segments}, {"service_paths", paths.service_paths},
           {"empty_paths", paths.empty_paths}, {"complete", paths.complete},
           {"support_complete",paths.support_complete}, {"scope", paths.scope}};
}
} // namespace mlnd
