#pragma once

#include "mlnd/input.hpp"
#include <array>
#include <functional>
#include <map>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mlnd {
using BasePair = std::array<int, 2>;
using Legs = std::vector<std::vector<int>>;
using ArcCounts = std::vector<std::tuple<int, int, int>>;

struct InitialSegment {
    int id = -1;
    BasePair b{};
    int first_od = -1, pickup = -1, delivery = -1;
    Legs legs;
    double t0 = 0, t1 = 0;
};

struct ServicePath {
    int id = -1, segment_id = -1;
    BasePair b{};
    int first_od = -1, second_od = -1;
    std::string mode;
    std::vector<int> pickups, deliveries;
    Legs legs;
    std::vector<std::pair<int, int>> served_counts;
    ArcCounts arc_counts;
    double flight_time = 0, time = 0, wait = 0;
    std::vector<std::pair<int, double>> slot_times;
    std::vector<double> pickup_times, delivery_times;
    double cost = 0;
};

struct EmptyPath {
    int id = -1;
    BasePair b{};
    Legs legs;
    ArcCounts arc_counts;
    double flight_time = 0, time = 0, cost = 0;
};

// Resource extension for a second order, independent of any path catalogue.
struct PathContinuation {
    int od = -1, pickup = -1, delivery = -1;
    double transfer = 0, loaded = 0, back = 0, wait = 0;
};

// Authoritative scalar checks/resource accounting shared by enumeration and
// implicit pricing. These build a single candidate; neither enumerates paths.
bool build_initial_segment(const Input&, BasePair, int first_od, int pickup,
                           int delivery, int id, InitialSegment&);
bool build_service_path(const Input&, const InitialSegment&,
                        const PathContinuation*, ServicePath&);

struct PathCollection {
    std::vector<InitialSegment> initial_segments;
    std::vector<ServicePath> service_paths;
    std::vector<EmptyPath> empty_paths;
    bool complete = true;
    // Complete finite first-segment/task support is independent of whether
    // every physical task path has been materialized.
    bool support_complete = true;
    std::string scope = "direct";
    std::map<std::array<int,8>,int> service_registry;
    std::map<BasePair,int> empty_registry;
};

using ServiceSignature = std::array<int,8>;
ServiceSignature service_path_signature(const ServicePath&);
void rebuild_path_registry(PathCollection&, std::function<void()> checkpoint = {});
int register_service_path(PathCollection&, ServicePath);
int register_empty_path(PathCollection&, EmptyPath);

// Build the finite row-index support only. No service or empty task paths are
// enumerated; complete is false even when support_complete is true.
PathCollection prepare_path_support(const Input&,
    std::optional<double> time_limit_seconds = std::nullopt,
    std::function<bool()> stop_requested = {},
    std::function<void(std::size_t,std::size_t,std::size_t)> progress = {});

// Cheap primal seed, not a certificate or feasibility requirement. Retain a
// few low cost single prefixes per (base pair, OD), then greedily extend those
// prefixes to one best two-order path each. Returns newly registered counts.
std::pair<std::size_t,std::size_t> greedy_seed_paths(const Input&, PathCollection&,
    std::size_t max_per_group = 1,
    double deadline = std::numeric_limits<double>::infinity());

struct BranchSeedReport {
    std::size_t service_added = 0, empty_added = 0;
    std::size_t groups_requested = 0, groups_completed = 0;
    std::size_t required_branches = 0, covered_branches = 0;
    // Aggregate-policy branches exhaustively searched with no physical path.
    // Prefix-policy failures instead count groups lacking a complete prefix.
    std::size_t physically_missing_branches = 0;
    std::size_t groups_without_complete_prefix = 0;
    bool stopped = false;
};

// Primal-only seed: cover the single branch and each positive-demand, positive-
// kappa continuation required by the given binary assignment mask. Its shape
// is Input::assignment_count(), indexed by Input::assignment_offset; nullptr
// considers every departure (K=1) or base pair (K=2). Aggregate groups may use different first segments for their branches.
// Segment-policy groups receive one common prefix with a complete branch
// bundle, when one exists. Prefixes are ranked cheaply, then a low-cost second
// leg is chosen; this is not a shortest-path or pricing certificate. Storage is
// at most one new path per requested branch, with no full catalogue built.
// A complete search still need not give positive feasible service: capacity,
// open links and inventory are handled by the subsequent operational solve.
BranchSeedReport branch_covering_seed_paths(const Input&, PathCollection&,
    const std::vector<double>* assignment_mask = nullptr,
    double deadline = std::numeric_limits<double>::infinity());

class EnumerationStopped : public std::runtime_error {
public:
    std::string reason;
    std::size_t columns, initial_segments;
    EnumerationStopped(std::string why, std::size_t count, std::size_t segments);
};

double first_wait(double mean, double travel);
double second_wait(double mean, double travel, double delta);

// Complete direct-flight universe, with exactly the Python catalogue's order.
// Simple multihop/supplied route universes are explicitly rejected in this port.
PathCollection enumerate_paths(
    const Input& data,
    std::optional<std::size_t> column_limit = std::nullopt,
    std::optional<double> time_limit_seconds = std::nullopt,
    std::function<bool()> stop_requested = {},
    std::function<void(std::size_t,std::size_t,std::size_t)> progress = {});

void to_json(Json& out, const InitialSegment& value);
void to_json(Json& out, const ServicePath& value);
void to_json(Json& out, const EmptyPath& value);
void to_json(Json& out, const PathCollection& value);
} // namespace mlnd
