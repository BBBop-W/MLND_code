#include "mlnd/output.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace mlnd {
namespace {
double variable_value(const std::vector<double>& values, int index) {
    return index < 0 ? 0 : values.at(static_cast<std::size_t>(index));
}

std::vector<double> extract(const std::vector<double>& values, const std::vector<int>& indices) {
    std::vector<double> result;
    result.reserve(indices.size());
    for (int index : indices) result.push_back(variable_value(values, index));
    return result;
}

Json zeros(int size) { return std::vector<double>(size, 0); }

Json empty_solution(const Input& data, const PathCollection& paths) {
    Json capacity = Json::array(), assignment = Json::array(), airline = Json::array();
    for (int h = 0; h < data.n_h; ++h) capacity.push_back(zeros(data.n_levels));
    for (int r = 0; r < data.n_od; ++r) {
        Json row = Json::array();
        for (int l = 0; l < data.n_l; ++l)
            row.push_back(data.origin_assignment() ? Json(0.0) : zeros(data.n_l));
        assignment.push_back(std::move(row));
    }
    for (int u = 0; u < data.n_nodes(); ++u) airline.push_back(zeros(data.n_nodes()));
    return {{"facility", zeros(data.n_nodes())}, {"capacity_level", capacity},
            {"assignment", assignment}, {"assignment_scope", data.assignment_scope()}, {"airline", airline},
            {"rho", zeros(data.n_l)}, {"idle", zeros(data.n_l)},
            {"head", zeros(static_cast<int>(paths.initial_segments.size()))},
            {"eta", zeros(static_cast<int>(paths.initial_segments.size()))},
            {"courier", zeros(data.n_od)}, {"fleet", 0.0},
            {"flow", zeros(static_cast<int>(paths.service_paths.size()))},
            {"empty_flow", zeros(static_cast<int>(paths.empty_paths.size()))}};
}

struct Metrics {
    std::vector<double> served, first, second, receipt_weight, pickup_flow, delivery_flow;
    std::vector<double> occupancy, outflow, inflow, busy, arcs_out, arcs_in;
    Matrix<double> arc_flow;
    double single_tasks = 0, double_tasks = 0, empty_tasks = 0;
    double service_busy = 0, empty_busy = 0, flight_busy = 0, wait_busy = 0;
    double base_cost = 0, locker_cost = 0, link_cost = 0, fleet_cost = 0;
    double service_cost = 0, empty_cost = 0, courier_cost = 0;
    Metrics(const Input& d)
        : served(d.n_od), first(d.n_od), second(d.n_od), receipt_weight(d.n_od),
          pickup_flow(d.n_h), delivery_flow(d.n_h), occupancy(d.n_h),
          outflow(d.n_l), inflow(d.n_l), busy(d.n_l), arcs_out(d.n_nodes()),
          arcs_in(d.n_nodes()), arc_flow(d.n_nodes(), d.n_nodes()) {}
    double opening() const { return base_cost + locker_cost + link_cost; }
    double continuous() const { return fleet_cost + service_cost + empty_cost + courier_cost; }
};

Metrics metrics(const Input& data, const PathCollection& paths, const Json& solution) {
    Metrics out(data);
    for (int l = 0; l < data.n_l; ++l)
        out.base_cost += data.base_cost[l] * solution.at("facility").at(l).get<double>();
    for (int h = 0; h < data.n_h; ++h)
        for (int level = 0; level < data.n_levels; ++level)
            out.locker_cost += data.slot_cost(h, level) * solution.at("capacity_level").at(h).at(level).get<double>();
    for (int u = 0; u < data.n_nodes(); ++u)
        for (int v = 0; v < data.n_nodes(); ++v)
            out.link_cost += data.link_cost(u, v) * solution.at("airline").at(u).at(v).get<double>();
    out.fleet_cost = data.fleet_cost * solution.at("fleet").get<double>();
    for (int r = 0; r < data.n_od; ++r)
        out.courier_cost += data.ground_cost[r] * solution.at("courier").at(r).get<double>();
    auto arc_activity = [&](const ArcCounts& arcs, double flow) {
        for (const auto& [u, v, count] : arcs) {
            out.arc_flow(u, v) += flow * count;
            out.arcs_out[u] += flow * count;
            out.arcs_in[v] += flow * count;
        }
    };
    for (std::size_t index = 0; index < paths.service_paths.size(); ++index) {
        const auto& path = paths.service_paths[index];
        const double flow = solution.at("flow").at(index).get<double>();
        out.outflow[path.b[0]] += flow;
        out.inflow[path.b[1]] += flow;
        out.busy[path.b[0]] += flow * path.time;
        out.service_busy += flow * path.time;
        out.flight_busy += flow * path.flight_time;
        out.wait_busy += flow * path.wait;
        out.service_cost += flow * path.cost;
        (path.second_od < 0 ? out.single_tasks : out.double_tasks) += flow;
        for (const auto& [h, time] : path.slot_times) out.occupancy[h] += flow * time;
        for (std::size_t p = 0; p < path.pickups.size(); ++p) {
            const int od = p == 0 ? path.first_od : path.second_od;
            out.served[od] += flow;
            (p == 0 ? out.first : out.second)[od] += flow;
            out.receipt_weight[od] += flow * path.delivery_times[p];
            out.pickup_flow[path.pickups[p]] += flow;
            out.delivery_flow[path.deliveries[p]] += flow;
        }
        arc_activity(path.arc_counts, flow);
    }
    for (std::size_t index = 0; index < paths.empty_paths.size(); ++index) {
        const auto& path = paths.empty_paths[index];
        const double flow = solution.at("empty_flow").at(index).get<double>();
        out.outflow[path.b[0]] += flow;
        out.inflow[path.b[1]] += flow;
        out.busy[path.b[0]] += flow * path.time;
        out.empty_busy += flow * path.time;
        out.flight_busy += flow * path.flight_time;
        out.empty_cost += flow * path.cost;
        out.empty_tasks += flow;
        arc_activity(path.arc_counts, flow);
    }
    return out;
}

Json metric_json(const Input& data, const Json& solution, const Metrics& m) {
    const double demand = std::accumulate(data.demand.begin(), data.demand.end(), 0.0);
    const double served = std::accumulate(m.served.begin(), m.served.end(), 0.0);
    const auto idle = solution.at("idle").get<std::vector<double>>();
    const double idle_total = std::accumulate(idle.begin(), idle.end(), 0.0);
    return {{"demand_orders_per_hour", demand}, {"served_orders_per_hour", served},
            {"courier_orders_per_hour", demand - served}, {"service_fraction", demand ? served / demand : 0},
            {"single_tasks_per_hour", m.single_tasks}, {"double_tasks_per_hour", m.double_tasks},
            {"empty_tasks_per_hour", m.empty_tasks}, {"idle_drones", idle_total},
            {"fleet", solution.at("fleet")}, {"service_busy_drones", m.service_busy},
            {"empty_busy_drones", m.empty_busy}, {"flight_busy_drones", m.flight_busy},
            {"waiting_busy_drones", m.wait_busy},
            {"fleet_slack", solution.at("fleet").get<double>() - idle_total - m.service_busy - m.empty_busy},
            {"costs_per_hour", {{"bases", m.base_cost}, {"lockers", m.locker_cost},
                {"links", m.link_cost}, {"fleet", m.fleet_cost}, {"service", m.service_cost},
                {"repositioning", m.empty_cost}, {"direct_courier", m.courier_cost}}}};
}

Json package(const Input& data, const PathCollection& paths, Json solution) {
    const Metrics m = metrics(data, paths, solution);
    Json missions = Json::array(), relocations = Json::array();
    for (std::size_t index = 0; index < paths.service_paths.size(); ++index)
        missions.push_back({{"path", paths.service_paths[index]}, {"flow", solution.at("flow").at(index)}});
    for (std::size_t index = 0; index < paths.empty_paths.size(); ++index)
        relocations.push_back({{"path", paths.empty_paths[index]}, {"flow", solution.at("empty_flow").at(index)}});
    const double demand = std::accumulate(data.demand.begin(), data.demand.end(), 0.0);
    const double served = std::accumulate(m.served.begin(), m.served.end(), 0.0);
    const Json totals = metric_json(data, solution, m);
    return {{"assignment_scope", data.assignment_scope()}, {"model_version", data.model_version}, {"flow_policy", data.flow_policy},
            {"objective", m.opening() + m.continuous()}, {"continuous_cost", m.continuous()},
            {"opening_cost", m.opening()}, {"demand", demand}, {"served", served},
            {"fleet", solution.at("fleet")}, {"metrics", totals},
            {"_solution", std::move(solution)}, {"_segments", paths.initial_segments},
            {"_missions", std::move(missions)}, {"_relocations", std::move(relocations)}};
}

void json_file(const std::filesystem::path& path, const Json& data, int indent = 2) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream stream(temporary);
    if (!stream) throw std::runtime_error("Cannot open output: " + path.string());
    stream << data.dump(indent) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write output: " + path.string());
    std::filesystem::rename(temporary, path);
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result = "\"";
    for (char ch : value) { if (ch == '"') result += '"'; result += ch; }
    return result + '"';
}

class Csv {
    std::ofstream stream_;
public:
    Csv(const std::filesystem::path& path, const std::vector<std::string>& header) : stream_(path) {
        if (!stream_) throw std::runtime_error("Cannot open CSV: " + path.string());
        row(header);
    }
    void row(const std::vector<std::string>& cells) {
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (i) stream_ << ',';
            stream_ << csv_escape(cells[i]);
        }
        stream_ << '\n';
        if (!stream_) throw std::runtime_error("Cannot write CSV output");
    }
};

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}
} // namespace

Json solution_json(const Input& data, const PathCollection& paths,
                   const Formulation& formulation, const std::vector<double>& values) {
    Json solution = empty_solution(data, paths);
    const auto& layout = formulation.layout;
    for (int u = 0; u < data.n_nodes(); ++u)
        solution["facility"][u] = std::round(values.at(layout.facility + u));
    for (int h = 0; h < data.n_h; ++h)
        for (int level = 0; level < data.n_levels; ++level)
            solution["capacity_level"][h][level] = std::round(values.at(layout.capacity + h * data.n_levels + level));
    for (int r = 0; r < data.n_od; ++r)
        for (int l = 0; l < data.n_l; ++l)
            for (int m = 0; m < data.assignment_destinations(); ++m) {
                const double value = std::round(values.at(layout.assignment_index(r,l,m)));
                if (data.origin_assignment()) solution["assignment"][r][l] = value;
                else solution["assignment"][r][l][m] = value;
            }
    for (int u = 0; u < data.n_nodes(); ++u)
        for (int v = 0; v < data.n_nodes(); ++v)
            solution["airline"][u][v] = std::round(values.at(layout.airline + u * data.n_nodes() + v));
    solution["rho"] = extract(values, formulation.rho);
    solution["idle"] = extract(values, formulation.idle);
    solution["head"] = extract(values, formulation.head);
    solution["courier"] = extract(values, formulation.courier);
    solution["fleet"] = variable_value(values, formulation.fleet);
    solution["flow"] = extract(values, formulation.flow);
    solution["empty_flow"] = extract(values, formulation.empty_flow);
    while (solution["flow"].size() < paths.service_paths.size()) solution["flow"].push_back(0.0);
    while (solution["empty_flow"].size() < paths.empty_paths.size()) solution["empty_flow"].push_back(0.0);
    const auto head = solution.at("head").get<std::vector<double>>();
    std::vector<double> eta(paths.initial_segments.size(), 0);
    if (formulation.conditional_demand) {
        if (formulation.segment_eta_factor.size() != eta.size())
            throw std::invalid_argument("Conditional demand eta factors do not match initial segments");
        for (std::size_t j = 0; j < eta.size(); ++j)
            eta[j] = formulation.segment_eta_factor[j] * head.at(j);
    } else for (std::size_t g = 0; g < formulation.lift_groups.size(); ++g) {
        if (g >= formulation.eta.size() || formulation.eta[g] < 0) continue;
        const auto& group = formulation.lift_groups[g];
        const double pooled_eta = variable_value(values, formulation.eta[g]);
        if (data.flow_policy == "aggregate") {
            double total = 0;
            for (int segment : group) total += head.at(segment);
            if (total > 0)
                for (int segment : group) eta.at(segment) = head.at(segment) * pooled_eta / total;
        } else if (!group.empty()) {
            eta.at(group.front()) = pooled_eta;
        }
    }
    solution["eta"] = std::move(eta);
    return package(data, paths, std::move(solution));
}

void synchronize_result_paths(const Input& data, const PathCollection& paths, Json& result) {
    if (!result.contains("_solution") || result.at("_solution").is_null()) return;
    Json solution = result.at("_solution");
    if (solution.at("flow").size() > paths.service_paths.size() ||
        solution.at("empty_flow").size() > paths.empty_paths.size())
        throw std::runtime_error("Incumbent columns do not belong to the current generated pool");
    while (solution["flow"].size() < paths.service_paths.size()) solution["flow"].push_back(0.0);
    while (solution["empty_flow"].size() < paths.empty_paths.size()) solution["empty_flow"].push_back(0.0);
    result.update(package(data, paths, std::move(solution)));
}

Json all_courier_result(const Input& data, const PathCollection& paths) {
    Json solution = empty_solution(data, paths);
    solution["courier"] = data.demand;
    Json result = package(data, paths, std::move(solution));
    result["bound"] = 0;
    result["gap"] = result.at("objective").get<double>() / std::max(1.0, std::abs(result.at("objective").get<double>()));
    result["termination_reason"] = "all_courier_baseline";
    result["pricing_complete"] = false;
    result["incumbent_kind"] = "all_courier_baseline";
    return result;
}

std::vector<double> solution_design(const Input& data, const Json& supplied) {
    const Json& solution = supplied.contains("_solution") ? supplied.at("_solution") : supplied;
    const DesignLayout layout(data);
    if (supplied.contains("assignment_scope") && supplied.at("assignment_scope") != data.assignment_scope())
        throw std::invalid_argument("Assignment scope does not match this model");
    if (solution.contains("assignment_scope") && solution.at("assignment_scope") != data.assignment_scope())
        throw std::invalid_argument("Solution assignment scope does not match this model");
    const auto& assignments = solution.at("assignment");
    if (!assignments.is_array() || assignments.size() != static_cast<std::size_t>(data.n_od))
        throw std::invalid_argument("Wrong assignment OD dimension");
    for (const auto& row : assignments) {
        if (!row.is_array() || row.size() != static_cast<std::size_t>(data.n_l))
            throw std::invalid_argument("Wrong assignment departure dimension");
        for (const auto& cell : row) {
            if (data.origin_assignment() ? !cell.is_number() :
                (!cell.is_array() || cell.size() != static_cast<std::size_t>(data.n_l)))
                throw std::invalid_argument("Assignment shape mismatch: K=1 requires [OD, departure]; legacy base-pair solutions are not a fixed design for this model");
        }
    }
    std::vector<double> design(layout.count, 0);
    for (int u = 0; u < data.n_nodes(); ++u) design[layout.facility + u] = solution.at("facility").at(u);
    for (int h = 0; h < data.n_h; ++h)
        for (int level = 0; level < data.n_levels; ++level)
            design[layout.capacity + h * data.n_levels + level] = solution.at("capacity_level").at(h).at(level);
    for (int r = 0; r < data.n_od; ++r)
        for (int l = 0; l < data.n_l; ++l)
            for (int m = 0; m < data.assignment_destinations(); ++m)
                design[layout.assignment_index(r,l,m)] = assignment_value(data,assignments,r,l,m);
    for (int u = 0; u < data.n_nodes(); ++u)
        for (int v = 0; v < data.n_nodes(); ++v)
            design[layout.airline + u * data.n_nodes() + v] = solution.at("airline").at(u).at(v);
    return design;
}

void write_result(const std::filesystem::path& directory, const Input& data,
                  const PathCollection& paths, const Json& result) {
    std::filesystem::create_directories(directory);
    Json overview = Json::object();
    for (auto it = result.begin(); it != result.end(); ++it)
        if (it.key().empty() || it.key().front() != '_') overview[it.key()] = it.value();
    json_file(directory / "result.json", overview);
    if (!result.contains("_solution") || result.at("_solution").is_null()) return;
    const Json& solution = result.at("_solution");
    json_file(directory / "solution.json", solution);
    json_file(directory / "path_flows.json", {{"segments", result.at("_segments")},
              {"missions", result.at("_missions")}, {"relocations", result.at("_relocations")}}, -1);
    json_file(directory / "input_reference.json", {{"instance_path", data.instance_path.string()},
              {"manifest", data.manifest}, {"model", data.model_json}, {"max_orders", data.max_orders}});
    const Metrics m = metrics(data, paths, solution);
    const Json totals = metric_json(data, solution, m);
    json_file(directory / "metrics.json", totals);
    Csv costs(directory / "costs.csv", {"component", "cny_per_hour"});
    for (auto it = totals.at("costs_per_hour").begin(); it != totals.at("costs_per_hour").end(); ++it)
        costs.row({it.key(), number(it.value().get<double>())});

    Csv flows(directory / "path_flows.csv", {"kind", "path_id", "segment_id", "mode", "departure_base", "return_base",
        "first_od", "second_od", "flow_per_hour", "flight_minutes", "wait_minutes", "operating_minutes", "unit_cost_cny", "cost_cny_per_hour", "legs"});
    for (std::size_t index = 0; index < paths.service_paths.size(); ++index) {
        const auto& p = paths.service_paths[index];
        const double flow = solution.at("flow").at(index);
        if (flow == 0) continue;
        flows.row({"service", std::to_string(index), std::to_string(p.segment_id), p.mode,
            data.base_ids.at(p.b[0]), data.base_ids.at(p.b[1]), data.od_ids.at(p.first_od),
            p.second_od < 0 ? "" : data.od_ids.at(p.second_od), number(flow),
            number(60 * p.flight_time), number(60 * p.wait), number(60 * p.time),
            number(p.cost), number(flow * p.cost), Json(p.legs).dump()});
    }
    for (std::size_t index = 0; index < paths.empty_paths.size(); ++index) {
        const auto& p = paths.empty_paths[index];
        const double flow = solution.at("empty_flow").at(index);
        if (flow == 0) continue;
        flows.row({"empty", std::to_string(index), "", "empty", data.base_ids.at(p.b[0]),
            data.base_ids.at(p.b[1]), "", "", number(flow), number(60 * p.flight_time), "0",
            number(60 * p.time), number(p.cost), number(flow * p.cost), Json(p.legs).dump()});
    }

    Csv sites(directory / "sites.csv", {"site_id", "type", "x_km", "y_km", "open", "capacity_level", "compartments",
        "mean_occupancy", "occupancy_limit", "rho", "idle_drones", "task_departures_per_hour", "task_returns_per_hour",
        "busy_drones_by_departure", "accounted_drone_equivalents", "order_pickups_per_hour", "order_deliveries_per_hour",
        "arc_departures_per_hour", "arc_arrivals_per_hour"});
    for (int l = 0; l < data.n_l; ++l) {
        const double idle = solution.at("idle").at(l);
        sites.row({data.base_ids[l], "L", number(data.bases(l, 0)), number(data.bases(l, 1)),
            number(solution.at("facility").at(l)), "", "", "", "", number(solution.at("rho").at(l)),
            number(idle), number(m.outflow[l]), number(m.inflow[l]), number(m.busy[l]), number(idle + m.busy[l]), "", "",
            number(m.arcs_out[l]), number(m.arcs_in[l])});
    }
    for (int h = 0; h < data.n_h; ++h) {
        int selected = -1;
        double count = 0, limit = 0;
        for (int level = 0; level < data.n_levels; ++level) {
            const double z = solution.at("capacity_level").at(h).at(level);
            if (z > 0.5) selected = level;
            count += data.slot_counts(h, level) * z;
            limit += data.occupancy_limit(h, level) * z;
        }
        const int u = data.n_l + h;
        sites.row({data.hub_ids[h], "H", number(data.hubs(h, 0)), number(data.hubs(h, 1)),
            number(solution.at("facility").at(u)), selected < 0 ? "" : data.level_ids.at(selected), number(count),
            number(m.occupancy[h]), number(limit), "", "", "", "", "", "",
            number(m.pickup_flow[h]), number(m.delivery_flow[h]), number(m.arcs_out[u]), number(m.arcs_in[u])});
    }

    Csv ods(directory / "od_flows.csv", {"od_id", "demand_per_hour", "served_per_hour", "first_order_per_hour", "second_order_per_hour",
        "courier_per_hour", "served_fraction", "assignment_departure", "assignment_return", "rho_departure", "mean_receipt_minutes"});
    for (int r = 0; r < data.n_od; ++r) {
        int departure = -1, destination = -1;
        for (int l = 0; l < data.n_l; ++l)
            for (int b = 0; b < data.assignment_destinations(); ++b)
                if (assignment_value(data,solution.at("assignment"),r,l,b) > 0.5) {
                    departure = l; destination = data.origin_assignment() ? -1 : b;
                }
        ods.row({data.od_ids[r], number(data.demand[r]), number(m.served[r]), number(m.first[r]), number(m.second[r]),
            number(solution.at("courier").at(r)), data.demand[r] > 0 ? number(m.served[r] / data.demand[r]) : "",
            departure < 0 ? "" : data.base_ids[departure], destination < 0 ? "" : data.base_ids[destination],
            departure < 0 ? "" : number(solution.at("rho").at(departure)),
            m.served[r] > 0 ? number(60 * m.receipt_weight[r] / m.served[r]) : ""});
    }
    Csv links(directory / "links.csv", {"from_site", "to_site", "open", "traversals_per_hour", "flight_minutes", "fixed_cost_cny_per_hour"});
    for (int u = 0; u < data.n_nodes(); ++u)
        for (int v = 0; v < data.n_nodes(); ++v) {
            const double open = solution.at("airline").at(u).at(v);
            if (open == 0 && m.arc_flow(u, v) == 0) continue;
            links.row({data.flight_node_ids[u], data.flight_node_ids[v], number(open), number(m.arc_flow(u, v)),
                       number(60 * data.flight_times(u, v)), number(open * data.link_cost(u, v))});
        }
    std::ofstream notes(directory / "OUTPUT_README.txt");
    notes << "Native C++ outputs; no Python runtime is required.\n"
          << "result.json is the solve overview; solution.json stores all decision arrays.\n"
          << "path_flows.json preserves the registered columns including zero flows for independent verification.\n"
          << "For implicit CG this is a generated subset, not the complete feasible path universe.\n"
          << "path_flows.csv includes every nonzero flow; tiny numerical flows are not silently removed.\n"
          << "All rates are per hour; all monetary rates are CNY/hour. Path unit costs are CNY/task.\n"
          << "Busy drones = task rate times mean mission duration in hours (drone-hours/hour).\n"
          << "sites.csv attributes busy drones to their departure base for accounting only.\n"
          << "accounted_drone_equivalents = idle at base + busy attributed to departure; fleet is a global continuous variable.\n"
          << "rho is base availability; hubs have no rho decision and those cells are blank.\n"
          << "mean_occupancy counts reserved and occupied compartments; occupancy_limit is the selected Poisson threshold.\n"
          << "Unassigned OD rows and closed hub levels are blank rather than fabricated IDs.\n";
    if (!notes) throw std::runtime_error("Cannot write output field documentation");
}
} // namespace mlnd
