#pragma once

#include "mlnd/input.hpp"
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace mlnd {
// Observation only: this object never changes an optimization decision or a
// numerical tolerance. Its deadline is the inclusive end-to-end solve budget.
class RunBudgetExceeded : public std::runtime_error {
public:
    std::string stage;
    Json sizes;
    RunBudgetExceeded(std::string phase, Json counts);
};

void atomic_json_file(const std::filesystem::path&, const Json&);
std::string utc_timestamp();

class RunProgress {
public:
    RunProgress(std::filesystem::path directory, double started, double budget, Json metadata);
    double elapsed() const;
    double remaining() const;
    void record(Json event);
    void heartbeat(const std::string& stage, Json details = Json::object());
    void guard(const std::string& stage, Json sizes = Json::object());
    // The caller must have verified global feasibility. Solver-model/Master
    // candidates must never be passed here as global incumbents.
    void checkpoint(const Json& result, const std::string& stage, bool force = false);
    bool checkpoint_needed(double objective, bool force = false) const;
    void status(const std::string& state, Json details = Json::object());
    void update_metadata(Json details);
    std::size_t next_solver_id();
    Json summary() const;
    const std::filesystem::path& directory() const { return directory_; }
private:
    std::filesystem::path directory_;
    double started_, budget_, last_heartbeat_ = -1e100, best_checkpoint_ = INFINITY, global_lower_ = 0;
    Json metadata_;
    Json last_model_size_=nullptr,last_path_sizes_=nullptr;
    std::ofstream events_;
    std::size_t sequence_ = 0, solver_sequence_ = 0, checkpoints_ = 0;
    std::size_t models_built_=0,solver_calls_=0;
    double observation_seconds_=0;
    mutable std::mutex mutex_;
    Json stamp(Json event) const;
    void record_unlocked(Json event);
};

// Model record construction has no SolverOptions argument. A scoped process
// observer allows its potentially large loops to honor the same deadline.
// CLI installs it; library/test callers have no observer by default.
void set_active_run_progress(const std::shared_ptr<RunProgress>&);
void model_construction_guard(const std::string& stage, std::size_t variables,
                              std::size_t rows, std::size_t cones, bool force = false);
} // namespace mlnd
