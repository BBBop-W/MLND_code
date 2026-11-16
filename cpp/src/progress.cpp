#include "mlnd/progress.hpp"
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace mlnd {
namespace {
double steady_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::weak_ptr<RunProgress> active_progress;
std::size_t construction_operations = 0;
}
RunBudgetExceeded::RunBudgetExceeded(std::string phase, Json counts)
    : std::runtime_error("End-to-end solve budget exhausted during " + phase),
      stage(std::move(phase)), sizes(std::move(counts)) {}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    if (gmtime_s(&utc, &seconds) != 0) throw std::runtime_error("Cannot convert UTC timestamp");
#else
    if (!gmtime_r(&seconds, &utc)) throw std::runtime_error("Cannot convert UTC timestamp");
#endif
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << millis << 'Z';
    return out.str();
}

void atomic_json_file(const std::filesystem::path& path, const Json& value) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        if (!stream) throw std::runtime_error("Cannot open atomic JSON output " + temporary.string());
        stream << std::setw(2) << value << '\n';
        stream.flush();
        if (!stream) throw std::runtime_error("Cannot finish atomic JSON output " + temporary.string());
    }
#ifdef _WIN32
    // Both names are in the same directory/volume. Use an explicit Win32
    // replacement without first deleting the old checkpoint.
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::filesystem::filesystem_error("Cannot replace atomic JSON output", temporary, path,
            std::error_code(static_cast<int>(GetLastError()), std::system_category()));
#else
    std::filesystem::rename(temporary, path);
#endif
}

RunProgress::RunProgress(std::filesystem::path directory, double started, double budget, Json metadata,
                         std::function<double()> clock)
    : directory_(std::move(directory)), started_(started), budget_(budget), metadata_(std::move(metadata)),
      clock_(std::move(clock)) {
    std::filesystem::create_directories(directory_);
    events_.open(directory_ / "progress.jsonl", std::ios::app);
    if (!events_) throw std::runtime_error("Cannot open progress.jsonl");
}
double RunProgress::elapsed() const { return std::max(0.0, (clock_ ? clock_() : steady_seconds()) - started_); }
double RunProgress::remaining() const { return std::max(0.0, budget_ - elapsed()); }
Json RunProgress::stamp(Json event) const {
    event["timestamp_utc"] = utc_timestamp();
    event["elapsed_seconds"] = elapsed();
    event["budget_seconds"] = budget_;
    return event;
}
void RunProgress::record_unlocked(Json event) {
    const double begin=steady_seconds();
    if(event.contains("model_size")) last_model_size_=event.at("model_size");
    if(event.contains("path_sizes")) last_path_sizes_=event.at("path_sizes");
    if(event.value("event",std::string())=="solver_model_build_end") ++models_built_;
    if(event.value("event",std::string())=="solver_solve_start") ++solver_calls_;
    if (event.value("bound_scope", std::string()) == metadata_.value("problem_scope",std::string("global")) &&
        event.contains("bound") && event.at("bound").is_number()) {
        const double candidate=event.at("bound").get<double>();
        if (std::isfinite(candidate)) global_lower_=std::max(global_lower_,candidate);
    }
    event = stamp(std::move(event));
    event["sequence"] = ++sequence_;
    events_ << event.dump() << '\n';
    events_.flush();
    if (!events_) throw std::runtime_error("Cannot append progress.jsonl");
    atomic_json_file(directory_ / "progress.json", event);
    observation_seconds_+=steady_seconds()-begin;
}
void RunProgress::record(Json event) {
    std::lock_guard<std::mutex> lock(mutex_);
    record_unlocked(std::move(event));
}
void RunProgress::heartbeat(const std::string& stage, Json details) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double now = elapsed();
    if (now - last_heartbeat_ < 10.0) return;
    last_heartbeat_ = now;
    details["event"] = "heartbeat";
    details["stage"] = stage;
    record_unlocked(std::move(details));
}
void RunProgress::guard(const std::string& stage, Json sizes) {
    if (remaining() <= 0) throw RunBudgetExceeded(stage, std::move(sizes));
    heartbeat(stage, std::move(sizes));
}
void RunProgress::checkpoint(const Json& result, const std::string& stage, bool force) {
    if (!result.contains("_solution") || !result.contains("objective") || !result.at("objective").is_number())
        throw std::invalid_argument("A checkpoint requires a complete feasible native solution");
    const double objective = result.at("objective").get<double>();
    if (!std::isfinite(objective)) throw std::invalid_argument("Checkpoint objective must be finite");
    std::lock_guard<std::mutex> lock(mutex_);
    if (objective > best_checkpoint_ || (!force && objective >= best_checkpoint_)) return;
    const double begin=steady_seconds();
    Json stored = metadata_;
    stored.update(result);
    stored["checkpoint_stage"] = stage;
    stored["checkpoint"] = true;
    stored["validated_incumbent"] = true;
    stored["has_feasible_solution"] = true;
    stored["feasible"] = true;
    double lower=global_lower_;
    if (result.contains("bound") && result.at("bound").is_number()) lower=std::max(lower,result.at("bound").get<double>());
    if (!std::isfinite(lower) || lower>objective+1e-6*(1+std::abs(objective)))
        throw std::invalid_argument("Checkpoint lower bound contradicts validated incumbent");
    stored["bound"]=std::min(objective,lower);
    stored["gap"]=(objective-std::min(objective,lower))/std::max(1.0,std::abs(objective));
    stored["bound_scope"]=metadata_.value("problem_scope",std::string("global"));
    stored = stamp(std::move(stored));
    atomic_json_file(directory_ / "checkpoint.json", stored);
    best_checkpoint_ = std::min(best_checkpoint_, objective);
    ++checkpoints_;
    observation_seconds_+=steady_seconds()-begin;
    record_unlocked({{"event", "incumbent_checkpoint"}, {"stage", stage},
        {"objective", objective}, {"upper_bound", objective}, {"validated_incumbent", true},
        {"served",result.value("served",0.0)},
        {"bound_scope", stored.at("bound_scope")}, {"bound", stored.at("bound")},
        {"lower_bound", stored.at("bound")}, {"gap", stored.at("gap")}});
}
bool RunProgress::checkpoint_needed(double objective, bool force) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::isfinite(objective) && (objective < best_checkpoint_ || (force && objective == best_checkpoint_));
}
void RunProgress::status(const std::string& state, Json details) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double begin=steady_seconds();
    Json value = metadata_;
    value.update(details);
    value["state"] = state;
    atomic_json_file(directory_ / "run_status.json", stamp(std::move(value)));
    observation_seconds_+=steady_seconds()-begin;
}
void RunProgress::update_metadata(Json details) {
    std::lock_guard<std::mutex> lock(mutex_);
    metadata_.update(details);
}
std::size_t RunProgress::next_solver_id() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ++solver_sequence_;
}
Json RunProgress::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"events", sequence_}, {"solver_models", solver_sequence_}, {"checkpoints", checkpoints_},
        {"solver_models_built",models_built_},{"solver_calls",solver_calls_},
        {"last_model_size",last_model_size_},{"last_path_sizes",last_path_sizes_},
        {"global_lower_bound",global_lower_},{"observation_io_seconds",observation_seconds_},
        {"observation_timing_scope","overlaps_enclosing_phase_wall_do_not_add"},
        {"progress_file", "progress.jsonl"}, {"latest_progress_file", "progress.json"},
        {"checkpoint_file", "checkpoint.json"}, {"status_file", "run_status.json"}};
}
void set_active_run_progress(const std::shared_ptr<RunProgress>& observer) {
    active_progress = observer;
    construction_operations = 0;
}
void model_construction_guard(const std::string& stage, std::size_t variables,
                              std::size_t rows, std::size_t cones, bool force) {
    if (!force && (++construction_operations % 1024 != 0)) return;
    if (auto observer = active_progress.lock())
        observer->guard(stage, {{"model_size", {{"variables", variables}, {"linear_rows", rows}, {"cones", cones}}}});
}
} // namespace mlnd
