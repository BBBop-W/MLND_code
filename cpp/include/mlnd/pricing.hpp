#pragma once

#include "mlnd/formulation.hpp"
#include <limits>
#include <memory>

namespace mlnd {
struct LabelPricingOptions {
    double deadline = std::numeric_limits<double>::infinity();
    double tolerance = 1e-7;
    std::size_t max_columns = 64;
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
    std::size_t labels_created = 0, labels_extended = 0, labels_dominated = 0;
    // terminal_paths counts completion evaluations (including repeated bounded
    // suffix-cache evaluation), not domain cardinality. omitted/negative count retained eligible terminal
    // labels; dominated completions need not be counted. negative_paths==0 is
    // nevertheless equivalent to no omitted negative column when complete.
    std::size_t terminal_paths = 0, omitted_paths = 0, negative_paths = 0;
    std::size_t resource_pruned = 0, excluded_paths = 0;
    std::string dominance = "exact-prefix-equivalence-and-terminal-cost";
};

// Every pricing call extends labels from Input and registers only its bounded
// returned batch. IDs identify generated columns, not a pre-enumerated domain.
// The collection must outlive the oracle and may grow between calls.
class LabelPricingOracle {
public:
    LabelPricingOracle(const Input&, PathCollection&,
                       double deadline = std::numeric_limits<double>::infinity());
    LabelPricingResult price(const Formulation&, const std::vector<double>& row_prices,
                             const std::vector<double>& missing_service,
                             const std::vector<double>& missing_empty,
                             const LabelPricingOptions& = {}) const;
private:
    struct Graph;
    std::shared_ptr<const Graph> graph_;
};
} // namespace mlnd
