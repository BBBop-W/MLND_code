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
bool valid_indices(const std::vector<int>& values,int count) {
    return std::all_of(values.begin(),values.end(),[&](int i){return i>=0 && i<count;});
}
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
    copy("rho", f.rho); copy("idle", f.idle);
    if(!f.compact_task_heads)copy("head", f.head);
    copy("courier", f.courier); copy("flow", f.flow); copy("empty_flow", f.empty_flow);
    if(f.compact_task_heads) {
        const auto saved=s.at("head").get<std::vector<double>>();
        const auto restored=recover_task_heads(f,paths,x);
        if(saved.size()!=restored.size())throw std::invalid_argument("Wrong shape: head");
        for(std::size_t j=0;j<saved.size();++j)
            if(!std::isfinite(saved[j]) || !std::isfinite(restored[j]) ||
               std::abs(saved[j]-restored[j])>2e-7*(1+std::abs(saved[j])+std::abs(restored[j])))
                throw std::invalid_argument("Saved head disagrees with path-flow recovery");
    }
    x.at(f.fleet) = s.at("fleet");
    auto assignment = [&](int r, int l, int b) {
        return design.at(f.layout.assignment_index(r,l,b));
    };
    for (int l = 0; l < d.n_l; ++l) {
        const double indicator = f.strengthening.queue_perspective ? design[f.layout.facility+l] : 1;
        put(name("cone_up", l), indicator + x[f.idle[l]]);
        put(name("cone_down", l), indicator - x[f.rho[l]]);
        put(name("cone_one", l), indicator);
    }
    for (int r = 0; r < d.assignment_entities(); ++r)
        for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.assignment_destinations(); ++b)
            put(name("v", r, l, d.origin_assignment() ? -1 : b), assignment(r, l, b) * x[f.rho[l]]);
    for (std::size_t g = 0; g < f.lift_groups.size(); ++g) {
        if(f.compact_task_heads)continue;
        const auto& group = f.lift_groups[g];
        const auto& segment = paths.initial_segments.at(group.front());
        double head = 0, lambda = 0;
        for (int j : group) head += x[f.head[j]];
        if (d.flow_policy == "aggregate") put(name("group_head", g), head);
        if (d.effective_delta() <= 0 || d.preprocessed_demand) continue;
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
    std::vector<double> served(d.n_od), load(d.n_h), balance(d.n_l),
        by_pair(d.n_od*d.assignment_pool_count()), task_flow(d.preprocessed_demand?d.assignment_count():0);
    const auto occurrence_index=[&](int r,int l,int b) {
        return r*d.assignment_pool_count()+l*d.assignment_destinations()+(d.origin_assignment()?0:b);
    };
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
            by_pair[occurrence_index(r,p.b[0],p.b[1])] += flow * count;
        }
        if(d.preprocessed_demand)
            task_flow[d.assignment_offset(d.task_index(p.first_od,p.second_od),p.b[0])] += flow;
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
        for (int l = 0; l < d.n_l; ++l) for (int b = 0; b < d.assignment_destinations(); ++b) {
            double rate=d.demand[r]*assignment_value(d,s.at("assignment"),r,l,b);
            if(d.preprocessed_demand) {
                rate=0;
                for(int q=0;q<d.task_count();++q) {
                    const int multiplicity=(d.task_first(q)==r)+(d.task_second(q)==r);
                    if(multiplicity)rate+=multiplicity*d.task_rate(q)*assignment_value(d,s.at("assignment"),q,l);
                }
            }
            equal("served_assignment_rho", by_pair[occurrence_index(r,l,b)],
                  rate*s.at("rho").at(l).get<double>());
        }
    }
    if(d.preprocessed_demand)for(int q=0;q<d.task_count();++q)for(int l=0;l<d.n_l;++l)
        equal("task_assignment_rho",task_flow[d.assignment_offset(q,l)],
            d.task_rate(q)*assignment_value(d,s.at("assignment"),q,l)*s.at("rho").at(l).get<double>());
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
        validate_result_path_records(paths, result);
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

Json add_grouped_primal_radius(const Input& d, Formulation& f, const std::vector<double>& center,
                              const std::vector<int>& requested_heads, const std::vector<int>& requested_lockers,
                              int head_radius, int capacity_radius) {
    if(center.size()!=static_cast<std::size_t>(f.layout.count) || head_radius<0 || capacity_radius<0 ||
       !valid_indices(requested_heads,d.assignment_entities()) || !valid_indices(requested_lockers,d.n_h))
        throw std::invalid_argument("Invalid grouped primal radius domain");
    const std::set<int> heads(requested_heads.begin(),requested_heads.end());
    const std::set<int> lockers(requested_lockers.begin(),requested_lockers.end());
    Coefficients head_row,capacity_row;int head_constant=0,capacity_constant=0;
    const auto add_group=[&](const std::vector<int>& indices,Coefficients& row,int& constant) {
        int active=-1;
        for(int j:indices) {
            if(!std::isfinite(center[j]) || (std::abs(center[j])>1e-6 && std::abs(center[j]-1)>1e-6))
                throw std::invalid_argument("Grouped primal center is not binary");
            if(center[j]>.5) {
                if(active>=0)throw std::invalid_argument("Grouped primal center has multiple active states");
                active=j;
            }
        }
        if(active>=0) {row.emplace_back(active,-1);++constant;}
        else for(int j:indices)row.emplace_back(j,1);
    };
    for(int r:heads) {
        std::vector<int> indices;
        for(int l=0;l<d.n_l;++l)for(int b=0;b<d.assignment_destinations();++b)
            indices.push_back(f.layout.assignment_index(r,l,b));
        add_group(indices,head_row,head_constant);
    }
    for(int h:lockers) {
        std::vector<int> indices;
        for(int z=0;z<d.n_levels;++z)indices.push_back(f.layout.capacity+h*d.n_levels+z);
        add_group(indices,capacity_row,capacity_constant);
    }
    f.matrix.row("primal_head_group_radius",'L',head_radius-head_constant,std::move(head_row));
    f.matrix.row("primal_capacity_group_radius",'L',capacity_radius-capacity_constant,std::move(capacity_row));
    return {{"metric","categorical_state_changes"},{"head_radius",head_radius},{"capacity_radius",capacity_radius},
        {"heads",heads},{"lockers",lockers},{"base_facilities_counted",false},{"airlines_counted",false},
        {"scope","temporary_primal_only"}};
}

Json solve_restricted_heuristic(const Input& d, const PathCollection& paths,
                                const SolverOptions& options, double deadline,
                                const Json* incumbent, int radius,
                                const std::vector<int>* service_indices,
                                const std::vector<int>* empty_indices,
                                bool fix_assignment, bool fix_infrastructure,
                                const CoupledPrimalNeighborhood* coupled,
                                const PrimalRepairRestrictions* restrictions) {
    if (fix_assignment && fix_infrastructure)
        throw std::invalid_argument("Primal assignment and infrastructure fixing modes are mutually exclusive");
    if (coupled && (fix_assignment || fix_infrastructure || radius >= 0))
        throw std::invalid_argument("Coupled primal block cannot be combined with another neighborhood");
    if (coupled) {
        if(!valid_indices(coupled->heads,d.assignment_entities()) || !valid_indices(coupled->lockers,d.n_h) ||
           !valid_indices(coupled->bases,d.n_l))
            throw std::invalid_argument("Coupled primal resource index is out of range");
        for(auto [u,v]:coupled->airlines)if(u<0||v<0||u>=d.n_nodes()||v>=d.n_nodes())
            throw std::invalid_argument("Coupled primal airline index is out of range");
    }
    if (restrictions) {
        if(!valid_indices(restrictions->change_heads,d.assignment_entities()) ||
           !valid_indices(restrictions->change_lockers,d.n_h) ||
           !valid_indices(restrictions->closed_lockers,d.n_h) ||
           !valid_indices(restrictions->closed_bases,d.n_l))
            throw std::invalid_argument("Primal exploration resource index is out of range");
        for(auto [u,v]:restrictions->closed_airlines)
            if(u<0 || v<0 || u>=d.n_nodes() || v>=d.n_nodes())
                throw std::invalid_argument("Primal exploration airline index is out of range");
        if(!restrictions->change_heads.empty() && !d.origin_assignment())
            throw std::invalid_argument("Head-state exploration requires origin assignment");
        if(!restrictions->forbidden_origins.empty() && !d.origin_assignment())
            throw std::invalid_argument("Origin restrictions require origin assignment");
        for(auto [r,l]:restrictions->forbidden_origins) {
            if(r<0 || r>=d.assignment_entities() || l<0 || l>=d.n_l)
                throw std::invalid_argument("Forbidden primal origin is out of range");
            if(coupled && std::find(coupled->heads.begin(),coupled->heads.end(),r)==coupled->heads.end())
                throw std::invalid_argument("Forbidden primal origin lies outside the released head groups");
        }
        if(restrictions->grouped_radius && !coupled)
            throw std::invalid_argument("Grouped repair radius requires a coupled neighborhood");
        if(std::isnan(restrictions->objective_upper) || restrictions->objective_upper==-INFINITY ||
           !std::isfinite(restrictions->validation_reserve_seconds) ||
           restrictions->validation_reserve_seconds<0)
            throw std::invalid_argument("Invalid primal exploration budget or objective ceiling");
    }
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
                   {"assignment_fix_requested", fix_assignment}, {"assignment_fixed", false},
                   {"infrastructure_fix_requested", fix_infrastructure}, {"infrastructure_fixed", false},
                   {"global_lower_bound_available", false}, {"diagnostics", Json::object()}};
    if(restrictions)report["candidate_validated"]=false;
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
    std::vector<int> selected_service, selected_empty;
    // Selection changes only a heuristic's feasible subset. In particular,
    // adding every strictly positive incumbent column preserves a possible
    // warm start without truncating tiny flows or dropping task requirements.
    auto select = [&](const std::vector<int>* requested, std::size_t count,
                      const char* flow_field, const char* metadata_field,
                      std::vector<int>& selected) -> const std::vector<int>* {
        if (!requested) {
            report["column_selection"][metadata_field] = {{"scope", "all_registered"},
                {"catalogue_columns", count}, {"selected_columns", count},
                {"incumbent_columns_added", 0}};
            return nullptr;
        }
        std::set<int> unique;
        for (int index : *requested) {
            if (index < 0 || static_cast<std::size_t>(index) >= count)
                throw std::invalid_argument("Primal heuristic column index is out of range");
            unique.insert(index);
        }
        const auto requested_count = unique.size();
        if (incumbent) {
            const auto& saved = incumbent->contains("_solution") ? incumbent->at("_solution") : *incumbent;
            // Malformed incumbents still reach the ordinary validation path
            // and cannot supply assignment fixings or an accepted upper bound.
            if (saved.contains(flow_field) && saved.at(flow_field).is_array() &&
                saved.at(flow_field).size() <= count) {
                const auto& flow = saved.at(flow_field);
                for (std::size_t j = 0; j < flow.size(); ++j)
                    if (flow[j].is_number() && flow[j].get<double>() > 0)
                        unique.insert(static_cast<int>(j));
            }
        }
        selected.assign(unique.begin(), unique.end());
        report["column_selection"][metadata_field] = {{"scope", "selected_plus_incumbent_support"},
            {"catalogue_columns", count}, {"requested_columns", requested_count},
            {"selected_columns", selected.size()}, {"incumbent_columns_added", selected.size() - requested_count}};
        return &selected;
    };
    const auto* service = select(service_indices, paths.service_paths.size(), "flow", "service", selected_service);
    const auto* empty = select(empty_indices, paths.empty_paths.size(), "empty_flow", "empty", selected_empty);
    auto f = build_formulation(d, paths, nullptr, service, empty, false, false, options.tight_airline_bounds,
                              formulation_strengthening(options));
    build += wall_time() - construction;
    report["model_size"] = {{"variables", f.matrix.variables.size()}, {"linear_rows", f.matrix.rows.size()},
                            {"cones", f.matrix.cones.size()}, {"service_columns", f.service_indices.size()},
                            {"empty_columns", f.empty_indices.size()}};
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
    if (fix_assignment && !valid_incumbent) {
        report["status"] = "assignment_polish_requires_valid_incumbent";
        return finish();
    }
    if (fix_infrastructure && !valid_incumbent) {
        report["status"] = "infrastructure_polish_requires_valid_incumbent";
        return finish();
    }
    if (coupled && !valid_incumbent) {
        report["status"] = "coupled_polish_requires_valid_incumbent";
        return finish();
    }
    if (restrictions && (!restrictions->change_heads.empty() || !restrictions->change_lockers.empty()) &&
        !valid_incumbent) {
        report["status"]="exploration_requires_valid_incumbent";
        return finish();
    }
    if (coupled) {
        std::vector<bool> released(f.layout.count,false);
        for(int r:coupled->heads)for(int l=0;l<d.n_l;++l)
            for(int b=0;b<d.assignment_destinations();++b)
                released[f.layout.assignment_index(r,l,b)]=true;
        for(int h:coupled->lockers) {
            released[f.layout.facility+d.n_l+h]=true;
            for(int z=0;z<d.n_levels;++z)released[f.layout.capacity+h*d.n_levels+z]=true;
        }
        for(int l:coupled->bases)released[f.layout.facility+l]=true;
        for(auto [u,v]:coupled->airlines)released[f.layout.airline+u*d.n_nodes()+v]=true;
        std::vector<int> indices;
        for(int j=0;j<f.layout.count;++j) {
            if(released[j])indices.push_back(j);
            else f.matrix.variables[j].lower=f.matrix.variables[j].upper=center[j];
        }
        report["coupled_neighborhood"]={{"heads",coupled->heads},{"lockers",coupled->lockers},
            {"bases",coupled->bases},{"airlines",coupled->airlines},{"released_design_indices",indices},
            {"fixed_design_variables",f.layout.count-indices.size()},
            {"whole_assignment_groups",true},{"scope","temporary_primal_only"}};
        if (!point_residuals(f.matrix, start).at("passed").get<bool>()) start.clear();
    }
    if (fix_assignment) {
        for (int j = f.layout.assignment; j < f.layout.airline; ++j) {
            f.matrix.variables[j].lower = center[j];
            f.matrix.variables[j].upper = center[j];
        }
        report["assignment_fixed"] = true;
        report["fixed_assignment_variables"] = f.layout.airline - f.layout.assignment;
        // A retained global fallback may lie outside this primal neighborhood.
        // It remains a valid global incumbent but is not a valid MIP start here.
        if (!point_residuals(f.matrix, start).at("passed").get<bool>()) start.clear();
    }
    if (fix_infrastructure) {
        for (int j = 0; j < f.layout.assignment; ++j) {
            f.matrix.variables[j].lower = center[j];
            f.matrix.variables[j].upper = center[j];
        }
        report["infrastructure_fixed"] = true;
        report["fixed_infrastructure_variables"] = f.layout.assignment;
        // A fallback outside this neighborhood is still a global feasible
        // incumbent, but must not be passed to CPLEX as a neighborhood start.
        if (!point_residuals(f.matrix, start).at("passed").get<bool>()) start.clear();
    }
    if (radius >= 0 && !valid_incumbent) {
        report["status"] = "local_branching_requires_valid_incumbent";
        return finish();
    }
    const bool grouped_radius=options.grouped_local_branching &&
        (radius>0 || (coupled && restrictions && restrictions->grouped_radius));
    if(grouped_radius) {
        std::vector<int> heads,lockers;
        if(coupled) {heads=coupled->heads;lockers=coupled->lockers;}
        else {
            for(int r=0;r<d.assignment_entities();++r)heads.push_back(r);
            for(int h=0;h<d.n_h;++h)lockers.push_back(h);
        }
        report["grouped_local_branching"]=add_grouped_primal_radius(d,f,center,heads,lockers,
            options.local_branching_head_radius,options.local_branching_capacity_radius);
        report["local_branching_blocks"]={"head_state","capacity_state"};
        if(!point_residuals(f.matrix,start).at("passed").get<bool>())start.clear();
    } else if (radius >= 0) {
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
    if (restrictions) {
        Coefficients change;
        int constant=0;
        // A migration between two bases is one change, not two binary flips.
        const std::set<int> heads(restrictions->change_heads.begin(),restrictions->change_heads.end());
        const std::set<int> lockers(restrictions->change_lockers.begin(),restrictions->change_lockers.end());
        for(int r:heads) {
            int active=-1;
            for(int l=0;l<d.n_l;++l)if(center[f.layout.assignment_index(r,l)]>.5)active=l;
            if(active>=0) {change.emplace_back(f.layout.assignment_index(r,active),-1);++constant;}
            else for(int l=0;l<d.n_l;++l)change.emplace_back(f.layout.assignment_index(r,l),1);
        }
        for(int h:lockers) {
            int active=-1;
            for(int z=0;z<d.n_levels;++z)if(center[f.layout.capacity+h*d.n_levels+z]>.5)active=z;
            if(active>=0) {change.emplace_back(f.layout.capacity+h*d.n_levels+active,-1);++constant;}
            else for(int z=0;z<d.n_levels;++z)change.emplace_back(f.layout.capacity+h*d.n_levels+z,1);
        }
        if(!change.empty())f.matrix.row("primal_exploration_group_change",'G',1-constant,std::move(change));
        for(int l:restrictions->closed_bases)
            f.matrix.row(name("primal_closed_base",l),'E',0,{{f.layout.facility+l,1}});
        for(int h:restrictions->closed_lockers)
            f.matrix.row(name("primal_closed_locker",h),'E',0,{{f.layout.facility+d.n_l+h,1}});
        for(auto [u,v]:restrictions->closed_airlines)
            f.matrix.row(name("primal_closed_airline",u,v),'E',0,{{f.layout.airline+u*d.n_nodes()+v,1}});
        for(auto [r,l]:restrictions->forbidden_origins)
            f.matrix.row(name("primal_forbidden_origin",r,l),'E',0,{{f.layout.assignment_index(r,l),1}});
        if(std::isfinite(restrictions->objective_upper)) {
            Coefficients costs;
            for(std::size_t j=0;j<f.matrix.variables.size();++j)
                if(f.matrix.variables[j].cost!=0)costs.emplace_back(static_cast<int>(j),f.matrix.variables[j].cost);
            f.matrix.row("primal_exploration_cost_ceiling",'L',restrictions->objective_upper-f.matrix.constant,
                         std::move(costs));
        }
        report["repair_restrictions"]={{"change_heads",heads},{"change_lockers",lockers},
            {"closed_bases",restrictions->closed_bases},{"closed_lockers",restrictions->closed_lockers},
            {"closed_airlines",restrictions->closed_airlines},{"objective_upper",restrictions->objective_upper},
            {"forbidden_origins",restrictions->forbidden_origins},
            {"scope","temporary_primal_only"},{"validation_reserve_seconds",0.0}};
        if(!point_residuals(f.matrix,start).at("passed").get<bool>())start.clear();
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
        if (restrictions) {
            if(options.progress)optimize_budget=std::min(optimize_budget,options.progress->remaining());
            optimize_budget=std::max(0.0,optimize_budget);
            const double per_candidate=validation/(incumbent?2.0:1.0);
            const double reserve=std::min(optimize_budget,std::max({restrictions->validation_reserve_seconds,
                .10*(deadline-begin),1.5*per_candidate}));
            report["repair_restrictions"]["validation_reserve_seconds"]=reserve;
            optimize_budget-=reserve;
        }
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
        if(restrictions && optimize_budget<=0) {
            report["status"]="validation_reserve_exhausted";
            return finish();
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
                // A valid worse neighbor belongs to the search trajectory,
                // while the independently retained global best remains safe.
                if(restrictions) {
                    candidate.erase("bound");candidate.erase("gap");
                    candidate["native_model_validated"]=true;
                    if(restrictions->retain_validated)restrictions->retain_validated(candidate);
                    report["candidate_validated"]=true;
                    report["candidate"]=candidate;
                }
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
