#pragma once

#include "mlnd/queue_polymatroid.hpp"
#include <limits>
#include <set>

namespace mlnd {
struct QueueFractionalResult {
    std::vector<TaskSavingsCut> cuts;
    double certified_lower = -std::numeric_limits<double>::infinity();
    double lp_seconds = 0, price_seconds = 0, wall = 0;
    Json diagnostics = Json::object();
    Json price_records = Json::array();
};

// Snapshot all current rows/RHS/bounds/objective, relax binary flags only,
// and separate against the resulting current theta. All returned rows are
// complete-domain queue cuts; only a validated optimal LP yields a bound.
// The three budgets are total, cumulative LP, and cumulative price wall time.
QueueFractionalResult separate_fractional_queue(
    const LinearModel& current_master, const DesignLayout&, int theta,
    QueuePolymatroidCuts&, const QueueNativePrices& donor,
    const std::set<std::string>& existing_keys, const SolverOptions&,
    double total_seconds, double lp_budget_seconds, double price_budget_seconds,
    std::size_t max_rounds = 3);
} // namespace mlnd
