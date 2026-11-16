#pragma once

#include "mlnd/formulation.hpp"
#include <limits>
#include <memory>

namespace mlnd {
struct LabelPricingGraphCache;
// Exact byte snapshot of model inputs and immutable initial-segment support.
// Appending physical paths does not change this identity. Registered paths
// must retain their documented immutable, append-only IDs and payloads.
std::string pricing_domain_identity(const Input&, const PathCollection&);
struct LabelPricingOptions {
    double deadline = std::numeric_limits<double>::infinity();
    double tolerance = 1e-7;
    std::size_t max_columns = 64;
    // Supplement complete label pricing with a full catalogue branch scan.
    // The optional extra deadline limits only this strengthening; expiration
    // never invalidates an already complete ordinary pricing certificate.
    bool branch_correction = false;
    // Reuse immutable catalogue paths and stream coefficients without temporary
    // sparse vectors. The same complete label domain and exact ordering apply.
    bool fast_pricing = false;
    // Scan the canonical coefficients once when the physical catalogue is
    // complete. Certified implicit fixed-task domains use bounded-memory streaming;
    // other incomplete catalogues fall back to exact labels.
    bool catalogue_pricing = false;
    // Catalogue and streaming backends: diversify a bounded 4*max_columns pool while
    // retaining the most negative witness. Certificates do not use the batch.
    bool diverse_pricing = false;
    double branch_deadline = std::numeric_limits<double>::infinity();
    // Additional exclusions for separated stationarity constraints in a
    // restricted auxiliary dual. Always unioned with the primal RMP columns.
    std::vector<int> excluded_service, excluded_empty;
};

struct LabelPricingResult {
    std::vector<int> service, empty;
    // Clipped at zero. On complete=true these are minima over ALL omitted
    // paths, not just over the retained negative-column batch.
    double minimum = 0, min_service = 0, min_empty = 0, correction = 0;
    bool complete = false;
    // Discovery prices never define a full-domain dual certificate, even if
    // their underlying label traversal happened to finish.
    bool discovery_only = false;
    bool fast_pricing_used = false;
    bool catalogue_pricing_used = false, implicit_streaming_pricing_used = false, diverse_pricing_used = false;
    std::size_t catalogue_paths_scanned = 0, diversity_candidates = 0, diversity_groups = 0;
    std::string diversity_policy = "disabled";
    std::size_t catalogue_path_reuses = 0, physical_path_builds = 0;
    double old_service_correction = 0, branch_service_correction = 0;
    double chosen_service_correction = 0, empty_correction = 0, branch_seconds = 0;
    bool branch_correction_requested = false, branch_correction_complete = false;
    bool branch_correction_used = false;
    std::size_t branch_paths_scanned = 0, branch_groups = 0;
    std::string branch_correction_status = "disabled", branch_correction_scope = "none";
    std::size_t labels_created = 0, labels_extended = 0, labels_dominated = 0;
    // In label mode terminal_paths counts completion evaluations (including repeated bounded
    // suffix-cache evaluation), not domain cardinality. omitted/negative count retained eligible terminal
    // labels; dominated completions need not be counted. negative_paths==0 is
    // nevertheless equivalent to no omitted negative column when complete.
    // Catalogue mode counts every enabled, visited physical completion and
    // every eligible omitted/negative path, without dominance reduction.
    std::size_t terminal_paths = 0, omitted_paths = 0, negative_paths = 0;
    std::size_t resource_pruned = 0, excluded_paths = 0;
    std::string dominance = "exact-prefix-equivalence-and-terminal-cost";
};

// Label calls extend Input-derived paths and register only their bounded batch.
// Optional catalogue calls scan an already complete immutable physical domain.
// The collection must outlive the oracle and may grow between calls.
class LabelPricingOracle {
public:
    LabelPricingOracle(const Input&, PathCollection&,
                       double deadline = std::numeric_limits<double>::infinity(),
                       LabelPricingGraphCache* cache = nullptr,
                       bool prefer_catalogue = false, bool capture_identity = false);
    bool graph_reused() const { return graph_reused_; }
    // A constructor-owned exact snapshot can also protect the same synchronous
    // slice's RMP. No externally supplied token can bypass input verification.
    // The caller must request capture_identity (or provide a graph cache), and
    // must not mutate Input/support during the oracle's lifetime.
    const std::string& domain_identity() const;
    LabelPricingResult price(const Formulation&, const std::vector<double>& row_prices,
                             const std::vector<double>& missing_service,
                             const std::vector<double>& missing_empty,
                             const LabelPricingOptions& = {}) const;
private:
    friend struct LabelPricingGraphCache;
    struct Graph;
    std::shared_ptr<const Graph> graph_;
    const Input* data_ = nullptr;
    PathCollection* paths_ = nullptr;
    bool graph_reused_ = false;
    std::string domain_identity_;
};
// Explicitly owned by one optimization invocation; no process-global cache.
// Incomplete constructions are never published. Each oracle keeps its own
// collection binding while the immutable event graph can be shared.
struct LabelPricingGraphCache {
    Json metadata() const {
        return {{"hits",hits_},{"construction_attempts",attempts_},{"completed_builds",builds_},
                {"identity_misses",misses_},{"live_graphs",graph_?1:0}};
    }
private:
    friend class LabelPricingOracle;
    std::string identity_;
    std::shared_ptr<const LabelPricingOracle::Graph> graph_;
    std::size_t hits_=0,attempts_=0,builds_=0,misses_=0;
};

// Keep only columns negative under the current ORIGINAL row multipliers.
// The returned object is discovery-only and cannot certify optimality or a
// missing-column correction, irrespective of discovery.complete.
LabelPricingResult reprice_discovered_columns(const Formulation&, const PathCollection&,
    const std::vector<double>& original_prices, const LabelPricingResult& discovery,
    double tolerance);
} // namespace mlnd
