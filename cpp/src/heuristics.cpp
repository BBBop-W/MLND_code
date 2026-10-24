#include "mlnd/heuristics.hpp"
#include "mlnd/output.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
double remaining(double deadline) { return std::max(0.0, deadline - wall_time()); }
std::string name(const std::string& stem, int a, int b = -1, int c = -1) {
    auto out = stem + "_" + std::to_string(a);
    if (b >= 0) out += "_" + std::to_string(b);
    if (c >= 0) out += "_" + std::to_string(c);
    return out;
}

// Reconstruct the deterministic lifting variables, rather than trusting saved
// internal values. This provides a full feasible MIP start and validates the
// exported, rounded design against the original matrix before UB acceptance.
std::vector<double> lifted_point(const Input& d, const PathCollection& paths,
                                 const Formulation& f, const Json& result) {
    const Json& s = result.contains("_solution") ? result.at("_solution") : result;
    auto design = solution_design(d, s);
    std::vector<double> x(f.matrix.variables.size(), 0);
    std::copy(design.begin(), design.end(), x.begin());
    std::map<std::string, int> indices;
    for (std::size_t j = 0; j < f.matrix.variables.size(); ++j)
        indices.emplace(f.matrix.variables[j].name, static_cast<int>(j));
    auto put = [&](const std::string& key, double value) { x.at(indices.at(key)) = value; };
    auto copy = [&](const char* key, const std::vector<int>& vars) {
        const auto values = s.at(key).get<std::vector<double>>();
        if (values.size() != vars.size()) throw std::invalid_argument(std::string("Wrong shape: ") + key);
        for (std::size_t j = 0; j < vars.size(); ++j) {
            if (vars[j] >= 0) x.at(vars[j]) = values[j];
            else if (values[j] != 0) throw std::invalid_argument("Positive flow on absent column");
        }
    };
    copy("rho", f.rho); copy("idle", f.idle); copy("head", f.head);
    copy("courier", f.courier); copy("flow", f.flow); copy("empty_flow", f.empty_flow);
    x.at(f.fleet) = s.at("fleet");
    auto assignment = [&](int r, int l, int b) {
        return design.at(f.layout.assignment + (r * d.n_l + l) * d.n_l + b);
    };
    for (int l = 0; l < d.n_l; ++l) {
        const double indicator = f.strengthening.queue_perspective ? design[f.layout.facility+l] : 1;
        put(name("cone_up", l), indicator + x[f.idle[l]]);
        put(name("cone_down", l), indicator - x[f.rho[l]]);
        put(name("cone_one", l), indicator);
    }
    for (int r = 0; r < d.n_od; ++r)
        for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.n_l; ++b)
            put(name("v", r, l, b), assignment(r, l, b) * x[f.rho[l]]);
    for (std::size_t g = 0; g < f.lift_groups.size(); ++g) {
        const auto& group = f.lift_groups[g];
        const auto& segment = paths.initial_segments.at(group.front());
        double head = 0, lambda = 0;
        for (int j : group) head += x[f.head[j]];
        if (d.flow_policy == "aggregate") put(name("group_head", g), head);
        if (d.effective_delta() <= 0) continue;
        for (int r = 0; r < d.n_od; ++r)
            lambda += d.demand[r] * assignment(r, segment.b[0], segment.b[1]);
        const double eta = lambda > 0 ? head * -std::expm1(-d.effective_delta() * lambda) / lambda : 0;
        x.at(f.eta[g]) = eta;
        double chain = head;
        put(name("Z", g, 0), chain);
        for (int r = 0; r < d.n_od; ++r) {
            const double assigned = assignment(r, segment.b[0], segment.b[1]);
            put(name("u", g, r), eta * assigned);
            put(name("Y", g, r), chain * assigned);
            chain += chain * assigned * std::expm1(-d.effective_delta() * d.demand[r]);
            put(name("Z", g, r + 1), chain);
        }
    }
    return x;
}

// Reconstruct physical loads and cost from the returned point, independently
// of the matrix rows. Positive service paths are rebuilt from input primitives
// before their cached resource coefficients may contribute to an incumbent.
Json physical_residuals(const Input& d, const PathCollection& paths, const Json& result) {
    const auto& s = result.at("_solution");
    double maximum = 0;
    std::string worst;
    auto check = [&](const std::string& label, double violation, double scale = 1.0) {
        if (!std::isfinite(violation) || !std::isfinite(scale)) throw std::runtime_error("Nonfinite physical value");
        const double relative = std::max(0.0, violation) / std::max(1.0, scale);
        if (relative > maximum) { maximum = relative; worst = label; }
    };
    auto equal = [&](const std::string& label, double a, double b) {
        check(label, std::abs(a - b), 1 + std::abs(a) + std::abs(b));
    };
    std::vector<double> served(d.n_od), load(d.n_h), balance(d.n_l), by_pair(d.n_od * d.n_l * d.n_l);
    Matrix<double> arcs(d.n_nodes(), d.n_nodes());
    double busy = 0, operating = 0, opening = 0;
    const double total = std::accumulate(d.demand.begin(), d.demand.end(), 0.0);
    for (std::size_t k = 0; k < paths.service_paths.size(); ++k) {
        const double flow = s.at("flow").at(k);
        check("nonnegative_service_flow", -flow);
        if (flow == 0) continue;
        const auto& p = paths.service_paths[k];
        const auto& seg = paths.initial_segments.at(p.segment_id);
        if (p.b != seg.b || p.first_od != seg.first_od || p.pickups.empty() || p.deliveries.empty() ||
            p.pickups.front() != seg.pickup || p.deliveries.front() != seg.delivery)
            throw std::runtime_error("Service path and initial segment disagree");
        InitialSegment rebuilt_segment;
        if (!build_initial_segment(d, seg.b, seg.first_od, seg.pickup, seg.delivery,
                                   seg.id, rebuilt_segment)) throw std::runtime_error("Invalid initial segment");
        ServicePath physical;
        PathContinuation continuation;
        if (p.second_od >= 0) {
            if (d.max_orders < 2 || d.effective_delta() <= 0 || p.second_od >= d.n_od ||
                d.kappa(p.first_od, p.second_od) <= 0)
                throw std::runtime_error("Ineligible consecutive order pair");
            continuation.od = p.second_od;
            continuation.pickup = p.pickups.at(1); continuation.delivery = p.deliveries.at(1);
            if (continuation.pickup < 0 || continuation.pickup >= d.n_h || continuation.delivery < 0 ||
                continuation.delivery >= d.n_h || continuation.pickup == continuation.delivery ||
                (seg.delivery != continuation.pickup &&
                 !d.allowed_links(d.n_l + seg.delivery, d.n_l + continuation.pickup)) ||
                !d.allowed_links(d.n_l + continuation.pickup, d.n_l + continuation.delivery) ||
                !d.allowed_links(d.n_l + continuation.delivery, p.b[1]))
                throw std::runtime_error("Forbidden continuation flight");
            continuation.transfer = d.flight_times(d.n_l + seg.delivery, d.n_l + continuation.pickup);
            continuation.loaded = d.flight_times(d.n_l + continuation.pickup, d.n_l + continuation.delivery);
            continuation.back = d.flight_times(d.n_l + continuation.delivery, p.b[1]);
            continuation.wait = second_wait(d.rider_mean(continuation.od, continuation.pickup),
                                            continuation.transfer, d.effective_delta());
        }
        if (!build_service_path(d, rebuilt_segment, p.second_od < 0 ? nullptr : &continuation, physical))
            throw std::runtime_error("Invalid physical service path");
        if (physical.b != p.b || physical.mode != p.mode || physical.pickups != p.pickups ||
            physical.deliveries != p.deliveries || physical.legs != p.legs || physical.served_counts != p.served_counts ||
            physical.arc_counts != p.arc_counts || physical.slot_times.size() != p.slot_times.size())
            throw std::runtime_error("Cached service path structure differs from input reconstruction");
        equal("cached_service_time", physical.time, p.time);
        equal("cached_service_cost", physical.cost, p.cost);
        for (std::size_t j = 0; j < physical.slot_times.size(); ++j) {
            if (physical.slot_times[j].first != p.slot_times[j].first) throw std::runtime_error("Cached slot index differs");
            equal("cached_slot_time", physical.slot_times[j].second, p.slot_times[j].second);
        }
        busy += flow * physical.time; operating += flow * physical.cost;
        balance[p.b[0]] += flow; balance[p.b[1]] -= flow;
        for (auto [r, count] : physical.served_counts) {
            served[r] += flow * count;
            by_pair[(r * d.n_l + p.b[0]) * d.n_l + p.b[1]] += flow * count;
        }
        for (auto [h, value] : physical.slot_times) load[h] += flow * value;
        for (auto [u, v, count] : physical.arc_counts) arcs(u, v) += flow * count;
    }
    for (std::size_t k = 0; k < paths.empty_paths.size(); ++k) {
        const double flow = s.at("empty_flow").at(k);
        check("nonnegative_empty_flow", -flow);
        if (flow == 0) continue;
        const auto& p = paths.empty_paths[k];
        const int l = p.b[0], b = p.b[1];
        if (l == b || !d.allowed_links(l, b) || p.legs != Legs{{l, b}} ||
            p.arc_counts != ArcCounts{{l, b, 1}}) throw std::runtime_error("Invalid direct empty path");
        const double time = d.flight_times(l, b);
        check("empty_operating_limit", time - d.operating_limit);
        equal("cached_empty_time", time, p.time);
        equal("cached_empty_cost", d.flight_cost * time, p.cost);
        busy += flow * time; operating += flow * d.flight_cost * time;
        balance[l] += flow; balance[b] -= flow; arcs(l, b) += flow;
    }
    double idle = 0;
    for (int l = 0; l < d.n_l; ++l) {
        const double rho = s.at("rho").at(l), inventory = s.at("idle").at(l);
        idle += inventory;
        check("rho_lower", -rho); check("rho_upper", rho - 1); check("idle_lower", -inventory);
        check("inventory_availability", 1 - (1 + inventory) * (1 - rho), 2 + inventory);
        equal("base_balance", balance[l], 0);
        opening += d.base_cost[l] * s.at("facility").at(l).get<double>();
    }
    const double fleet = s.at("fleet");
    check("fleet_budget", idle + busy - fleet, 1 + idle + busy + std::abs(fleet));
    operating += d.fleet_cost * fleet;
    for (int h = 0; h < d.n_h; ++h) {
        double capacity = 0;
        for (int z = 0; z < d.n_levels; ++z) {
            const double chosen = s.at("capacity_level").at(h).at(z);
            capacity += d.occupancy_limit(h, z) * chosen; opening += d.slot_cost(h, z) * chosen;
        }
        check("locker_capacity", load[h] - capacity, 1 + load[h] + std::abs(capacity));
    }
    for (int r = 0; r < d.n_od; ++r) {
        const double courier = s.at("courier").at(r);
        check("courier_lower", -courier); equal("demand_balance", served[r] + courier, d.demand[r]);
        operating += d.ground_cost[r] * courier;
        for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.n_l; ++b)
            equal("served_assignment_rho", by_pair[(r * d.n_l + l) * d.n_l + b],
                  d.demand[r] * s.at("assignment").at(r).at(l).at(b).get<double>() * s.at("rho").at(l).get<double>());
    }
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v) {
        const double y = s.at("airline").at(u).at(v);
        double bound = 0;
        if (u != v && d.allowed_links(u, v))
            bound = total * (u < d.n_l && v < d.n_l ? std::max(0, d.n_l - 1) :
                             u >= d.n_l && v >= d.n_l ? d.max_orders : 1);
        check("airline_activation", arcs(u, v) - bound * y, 1 + arcs(u, v) + std::abs(bound * y));
        opening += d.link_cost(u, v) * y;
    }
    equal("physical_objective", opening + operating, result.at("objective"));
    return {{"passed", maximum <= 2e-7}, {"max_scaled_violation", maximum}, {"worst_constraint", worst}};
}

Json validate_result(const Input& d, const PathCollection& paths, const Formulation& f,
                     Json& result, std::vector<double>& point) {
    try {
        // Synchronization pads only an append-only prefix. Check any supplied
        // physical records before it overwrites them with the current pool;
        // otherwise a snapshot from a differently ordered pool could silently
        // reinterpret its indexed flows. Raw solution-only inputs explicitly
        // use the current pool's IDs and still undergo full point validation.
        const auto& saved = result.at("_solution");
        auto prefix = [&](const char* records, const char* flow, const auto& pool) {
            if (!result.contains(records)) return;
            const auto& rows = result.at(records);
            if (!rows.is_array() || rows.size() != saved.at(flow).size() || rows.size() > pool.size())
                throw std::invalid_argument("Incumbent physical records do not match its column prefix");
            for (std::size_t p = 0; p < rows.size(); ++p) {
                const Json expected = pool[p];
                if (rows.at(p).at("path") != expected)
                    throw std::invalid_argument("Incumbent path identity differs from the current column prefix");
            }
        };
        prefix("_missions", "flow", paths.service_paths);
        prefix("_relocations", "empty_flow", paths.empty_paths);
        synchronize_result_paths(d, paths, result);
        point = lifted_point(d, paths, f, result);
        auto matrix = point_residuals(f.matrix, point);
        auto physical = physical_residuals(d, paths, result);
        const double objective = objective_value(f.matrix, point);
        const bool costs_match = std::abs(objective - result.at("objective").get<double>()) <=
                                 2e-7 * (1 + std::abs(objective));
        return {{"passed", matrix.at("passed").get<bool>() && physical.at("passed").get<bool>() && costs_match},
                {"matrix", matrix}, {"physical", physical}, {"costs_match", costs_match}};
    } catch (const std::exception& error) {
        point.clear();
        return {{"passed", false}, {"reason", error.what()}};
    }
}
} // namespace

Json solve_restricted_heuristic(const Input& d, const PathCollection& paths,
                                const SolverOptions& options, double deadline,
                                const Json* incumbent, int radius) {
    if (radius < -1) throw std::invalid_argument("Local branching radius must be -1 or nonnegative");
    if (options.heuristic_pool_size < 0) throw std::invalid_argument("Heuristic pool size must be nonnegative");
    if (!paths.support_complete) throw std::invalid_argument("Restricted heuristic needs complete initial support");
    const double begin = wall_time();
    deadline = std::min(deadline, begin + std::max(0.0, options.seconds));
    Json best = all_courier_result(d, paths);
    // These fields belong to the outer problem and must not leak from the
    // baseline or a supplied incumbent into a restricted-model result.
    best.erase("bound"); best.erase("gap");
    Json report = {{"validated", true}, {"improved", false}, {"source", "all_courier_fallback"},
                   {"status", "time_limit"}, {"local_branching_radius", radius},
                   {"local_branching_blocks", {"facility", "capacity", "assignment"}},
                   {"global_lower_bound_available", false}, {"diagnostics", Json::object()}};
    const bool collect_pool = options.neighborhood_cuts && options.heuristic_pool_size > 0;
    if (collect_pool) {
        report["design_pool"] = Json::array();
        report["design_pool_metadata"] = {{"source", "optimize_solution_pool_no_populate"},
            {"capacity", options.heuristic_pool_size}, {"replacement", "diversity"},
            {"available", 0}, {"scanned", 0}, {"duplicates", 0}, {"invalid", 0},
            {"postprocess_reserve_seconds", 0.0}, {"optimize_time_limit_seconds", 0.0},
            {"remaining_before_optimize", 0.0}, {"remaining_after_optimize", 0.0},
            {"remaining_before_pool", 0.0}, {"pool_validation_seconds", 0.0},
            {"stop_reason", "not_solved"}};
    }
    std::set<std::vector<double>> pool_designs;
    auto keep_design = [&](const Json& candidate, const char* source) {
        if (!collect_pool || static_cast<int>(report.at("design_pool").size()) >= options.heuristic_pool_size) return;
        const auto design = solution_design(d, candidate);
        if (!pool_designs.insert(design).second) {
            auto& duplicates = report["design_pool_metadata"]["duplicates"];
            duplicates = duplicates.get<int>() + 1;
            return;
        }
        // This is a validated primal candidate, never a neighborhood bound.
        // A full-column CG certificate must still be obtained by the caller
        // before this design can contribute a global Benders cut.
        report["design_pool"].push_back({{"design", design}, {"objective", candidate.at("objective")},
            {"source", source}, {"validated", true}});
    };
    double build = 0, optimize = 0, validation = 0, pool_extraction = 0;
    std::unique_ptr<NativeSolver> solver;
    auto finish = [&]() {
        solver.reset();
        best.erase("bound"); best.erase("gap");
        best["native_model_validated"] = true;
        best["pricing_complete"] = false;
        best["incumbent_kind"] = report.at("source");
        report["objective"] = best.at("objective"); report["solution"] = std::move(best);
        report["timings"] = {{"build", build}, {"optimize", optimize}, {"validation", validation}, {"total", wall_time() - begin}};
        if (collect_pool) report["timings"]["pool_extraction"] = pool_extraction;
        return report;
    };
    if (remaining(deadline) <= 0) return finish();
    const double construction = wall_time();
    const std::vector<int>* all = nullptr;
    auto f = build_formulation(d, paths, nullptr, all, all, false, false, options.tight_airline_bounds,
                              formulation_strengthening(options));
    build += wall_time() - construction;
    report["model_size"] = {{"variables", f.matrix.variables.size()}, {"linear_rows", f.matrix.rows.size()},
                            {"cones", f.matrix.cones.size()}, {"service_columns", paths.service_paths.size()},
                            {"empty_columns", paths.empty_paths.size()}};
    std::vector<double> start;
    double phase = wall_time();
    auto baseline_validation = validate_result(d, paths, f, best, start);
    if (!baseline_validation.at("passed").get<bool>()) throw std::runtime_error("Invalid all-courier heuristic fallback");
    report["diagnostics"]["fallback_validation"] = baseline_validation;
    const double baseline_objective = best.at("objective");
    double reference_objective = baseline_objective;
    bool valid_incumbent = false;
    std::vector<double> center;
    if (incumbent) {
        Json candidate = *incumbent;
        if (!candidate.contains("_solution")) candidate = {{"_solution", candidate}};
        std::vector<double> point;
        auto checked = validate_result(d, paths, f, candidate, point);
        report["diagnostics"]["incumbent_validation"] = checked;
        if (checked.at("passed").get<bool>()) {
            valid_incumbent = true;
            center = solution_design(d, candidate);
            reference_objective = std::min(reference_objective, candidate.at("objective").get<double>());
            if (candidate.at("objective").get<double>() <= baseline_objective) {
                best = std::move(candidate); start = std::move(point); report["source"] = "retained_incumbent";
            }
        }
    }
    validation += wall_time() - phase;
    if (radius >= 0 && !valid_incumbent) {
        report["status"] = "local_branching_requires_valid_incumbent";
        return finish();
    }
    if (radius >= 0) {
        Coefficients row;
        int ones = 0;
        for (int j = 0; j < f.layout.airline; ++j) {
            const bool one = center[j] > 0.5;
            row.emplace_back(j, one ? -1 : 1); ones += one;
        }
        f.matrix.row("heuristic_local_branching", 'L', radius - ones, std::move(row));
        // The global fallback need not belong to the neighbourhood. Only a
        // neighbourhood-feasible start is supplied to this restricted solve.
        if (!point_residuals(f.matrix, start).at("passed").get<bool>()) start.clear();
    }
    if (remaining(deadline) <= 0) return finish();
    phase = wall_time();
    SolveOutcome outcome;
    try {
        solver = std::make_unique<NativeSolver>(f.matrix, options);
        if (collect_pool) {
            // These are storage/replacement controls for solutions found by
            // the existing solve. Do not launch a separate populate search.
            solver->cplex.setParam(IloCplex::Param::MIP::Pool::Capacity, options.heuristic_pool_size);
            solver->cplex.setParam(IloCplex::Param::MIP::Pool::Replace, CPX_SOLNPOOL_DIV);
        }
        if (!start.empty()) solver->add_start(start);
        build += wall_time() - phase;
        double optimize_budget = remaining(deadline);
        if (collect_pool) {
            if (options.progress) optimize_budget = std::min(optimize_budget, options.progress->remaining());
            optimize_budget = std::max(0.0, optimize_budget);
            // Reserve time before handing the remaining budget to CPLEX:
            // otherwise a normal time-limited solve consumes every second
            // and none of the already found pool designs can be validated.
            // Observed fallback/incumbent validation estimates the work per
            // candidate. This is a scheduling estimate, not a time guarantee.
            const double per_candidate = validation / (incumbent ? 2.0 : 1.0);
            const double reserve = std::min(.25 * optimize_budget,
                std::max(std::min(2.0, .10 * optimize_budget),
                         1.5 * (options.heuristic_pool_size + 1) * per_candidate));
            auto& metadata = report["design_pool_metadata"];
            metadata["remaining_before_optimize"] = optimize_budget;
            metadata["postprocess_reserve_seconds"] = reserve;
            optimize_budget -= reserve;
            metadata["optimize_time_limit_seconds"] = optimize_budget;
        }
        outcome = solver->optimize(optimize_budget);
        if (collect_pool) report["design_pool_metadata"]["remaining_after_optimize"] =
            std::max(0.0, std::min(remaining(deadline), options.progress
                ? options.progress->remaining() : remaining(deadline)));
    } catch (const IloException& error) {
        report["status"] = "heuristic_solver_exception";
        report["diagnostics"]["solver_exception"] = error.getMessage();
        return finish();
    } catch (const std::exception& error) {
        report["status"] = "heuristic_solver_exception";
        report["diagnostics"]["solver_exception"] = error.what();
        return finish();
    }
    optimize += outcome.seconds;
    report["status"] = outcome.status;
    report["restricted_model_optimal"] = outcome.optimal;
    report["has_solver_solution"] = outcome.has_solution;
    report["node_count"] = outcome.nodes;
    if (outcome.has_solution) {
        phase = wall_time();
        const auto original = point_residuals(f.matrix, outcome.values);
        report["diagnostics"]["solver_matrix"] = original;
        if (original.at("passed").get<bool>()) {
            auto candidate = solution_json(d, paths, f, outcome.values);
            std::vector<double> reconstructed;
            auto checked = validate_result(d, paths, f, candidate, reconstructed);
            report["diagnostics"]["candidate_validation"] = checked;
            if (checked.at("passed").get<bool>()) {
                // A design can generate a useful cut even when its primal
                // objective does not improve the retained global incumbent.
                keep_design(candidate, radius >= 0 ? "restricted_columns_local_branching" : "restricted_columns_misocp");
                if (candidate.at("objective").get<double>() < best.at("objective").get<double>()) {
                    best = std::move(candidate);
                    report["source"] = radius >= 0 ? "restricted_columns_local_branching" : "restricted_columns_misocp";
                }
            }
        }
        validation += wall_time() - phase;
    }
    if (collect_pool && outcome.has_solution && solver) {
        auto& metadata = report["design_pool_metadata"];
        const auto pool_remaining = [&]() {
            return std::max(0.0, std::min(remaining(deadline), options.progress
                ? options.progress->remaining() : remaining(deadline)));
        };
        metadata["remaining_before_pool"] = pool_remaining();
        try {
            const double extraction_started = wall_time();
            const int available = solver->cplex.getSolnPoolNsolns();
            metadata["available"] = available;
            pool_extraction += wall_time() - extraction_started;
            metadata["stop_reason"] = "pool_exhausted";
            for (int index = 0; index < available; ++index) {
                if (static_cast<int>(report.at("design_pool").size()) >= options.heuristic_pool_size) {
                    metadata["stop_reason"] = "capacity"; break;
                }
                if (pool_remaining() <= 0) { metadata["stop_reason"] = "deadline"; break; }
                const double read_started = wall_time();
                IloNumArray values(solver->env);
                solver->cplex.getValues(values, solver->variables, index);
                std::vector<double> point(values.getSize());
                for (int j = 0; j < values.getSize(); ++j) point[j] = values[j];
                values.end();
                pool_extraction += wall_time() - read_started;
                metadata["scanned"] = metadata.at("scanned").get<int>() + 1;
                const double validation_started = wall_time();
                bool valid = point_residuals(f.matrix, point).at("passed").get<bool>();
                if (valid) {
                    auto candidate = solution_json(d, paths, f, point);
                    std::vector<double> reconstructed;
                    const auto checked = validate_result(d, paths, f, candidate, reconstructed);
                    valid = checked.at("passed").get<bool>();
                    if (valid) keep_design(candidate, radius >= 0
                        ? "restricted_columns_local_branching_pool" : "restricted_columns_misocp_pool");
                }
                if (!valid) metadata["invalid"] = metadata.at("invalid").get<int>() + 1;
                const double validation_seconds = wall_time() - validation_started;
                validation += validation_seconds;
                metadata["pool_validation_seconds"] = metadata.at("pool_validation_seconds").get<double>()
                    + validation_seconds;
            }
        } catch (const IloException& error) {
            metadata["stop_reason"] = "pool_extraction_error";
            metadata["error"] = error.getMessage();
        } catch (const std::exception& error) {
            metadata["stop_reason"] = "pool_extraction_error";
            metadata["error"] = error.what();
        }
    }
    report["improved"] = best.at("objective").get<double>() < reference_objective - 1e-7;
    return finish();
}
} // namespace mlnd
