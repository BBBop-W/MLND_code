// Complete physical-path certificate and two-way fixed-design equivalence.
// This is a deterministic regression test, not a performance benchmark.
#include "mlnd/formulation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows, int columns, T value) {
    Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
// Frozen physical fixture shared with conditional_formulation_probe, extended
// to two capacity levels to cover both sparse networks and relocation.
Input fixture(int k, const std::string& policy, bool two_bases = false) {
    Input d;
    d.n_l = two_bases ? 2 : 1; d.n_h = d.n_od = d.n_levels = 2;
    d.max_orders = k; d.name = "tight_airline_fixture";
    d.model_version = "consecutive"; d.route_mode = "direct"; d.flow_policy = policy;
    d.demand = {4, 7}; d.ground_cost = {5, 5}; d.deadline = {1, 1};
    d.base_cost.assign(d.n_l, .5); d.slot_epsilon = {.05, .05};
    d.flight_times = Matrix<double>(d.n_nodes(), d.n_nodes());
    if (two_bases)
        d.flight_times.values = {0,.03,.05,.07, .03,0,.06,.04, .05,.06,0,.1, .07,.04,.1,0};
    else d.flight_times.values = {0,.05,.07, .05,0,.1, .07,.1,0};
    d.allowed_links = filled(d.n_nodes(), d.n_nodes(), 1);
    for (int i = 0; i < d.n_nodes(); ++i) d.allowed_links(i,i) = 0;
    d.link_cost = filled(d.n_nodes(), d.n_nodes(), 0.0);
    d.rider_mean = Matrix<double>(2,2); d.rider_mean.values = {.08,.12,.1,.06};
    d.first_cost = filled(2,2,1.0); d.terminal_cost = filled(2,2,.5);
    d.receipt_mean = filled(2,2,.08); d.removal_mean = filled(2,2,.03);
    d.slot_counts = Matrix<int>(2,2); d.slot_counts.values = {10,20,10,20};
    d.slot_cost = Matrix<double>(2,2); d.slot_cost.values = {.1,.2,.1,.2};
    d.occupancy_limit = Matrix<double>(2,2);
    d.occupancy_limit.values = {6.1690072893953225,14.072024748341311,
                              6.1690072893953225,14.072024748341311};
    d.kappa = Matrix<double>(2,2); d.kappa.values = {.3,.9,.2,.7};
    d.delta = .04; d.operating_limit = 1; d.fleet_cost = 1;
    d.flight_cost = .2; d.waiting_cost = .3;
    return d;
}
std::vector<double> design(const Input& d, const std::string& mode) {
    const DesignLayout layout(d);
    std::vector<double> result(layout.count, 0);
    if (mode == "closed") return result;
    for (int i = 0; i < d.n_nodes(); ++i) result[layout.facility+i] = 1;
    for (int i = 0; i < d.n_h; ++i) result[layout.capacity+i*d.n_levels] = 1;
    for (int r = 0; r < d.n_od; ++r) {
        if (mode == "partial_assignment" && r % 2) continue;
        const int destination = mode == "cross_base" ? 1 : 0;
        result[layout.assignment_index(r,0,destination)] = 1;
    }
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        result[layout.airline+u*d.n_nodes()+v] = mode == "assigned_without_links" ? 0 : d.allowed_links(u,v);
    if (mode == "one_closed_hh") result[layout.airline+d.n_l*d.n_nodes()+d.n_l+1] = 0;
    require(point_residuals(design_model(d).matrix,result).at("passed"), "Test design is invalid");
    return result;
}
Json certify_paths(const Input& d, const PathCollection& paths) {
    require(d.route_mode == "direct" && d.max_orders >= 1 && d.max_orders <= 2,
            "Certificate is scoped to current direct K<=2 paths");
    require(paths.complete && paths.support_complete, "Physical certificate requires complete catalogue");
    std::size_t service_arc_checks = 0, empty_arc_checks = 0, repeated_hh = 0, same_od_pairs = 0;
    int maximum_hh_count = 0;
    for (const auto& p : paths.service_paths) {
        int orders = 0;
        for (auto [r,count] : p.served_counts) {
            require(r >= 0 && r < d.n_od && count > 0, "Invalid served-count certificate input");
            orders += count;
        }
        require(orders >= 1 && orders <= d.max_orders, "Service path exceeds declared K");
        if (p.first_od == p.second_od) ++same_od_pairs;
        for (auto [u,v,count] : p.arc_counts) {
            require(count > 0 && u != v, "Self-loop or nonpositive arc count in direct path");
            if (u >= d.n_l && v >= d.n_l) {
                ++service_arc_checks;
                require(count <= orders, "HH arc traversals exceed number of served orders");
                maximum_hh_count = std::max(maximum_hh_count,count);
                if (count > 1) ++repeated_hh;
            }
        }
    }
    for (const auto& p : paths.empty_paths) for (auto [u,v,count] : p.arc_counts) {
        ++empty_arc_checks;
        require(u < d.n_l && v < d.n_l && count > 0, "Empty path consumes an HH arc");
    }
    return {{"service_paths",paths.service_paths.size()},{"empty_paths",paths.empty_paths.size()},
        {"hh_arc_checks",service_arc_checks},{"empty_arc_checks",empty_arc_checks},
        {"repeated_hh_occurrences",repeated_hh},{"maximum_hh_count",maximum_hh_count},
        {"same_od_pairs",same_od_pairs},{"passed",true}};
}
std::size_t check_only_bound_changes(const Input& d, const Formulation& old, const Formulation& tight) {
    require(!old.tight_airline_bounds && tight.tight_airline_bounds, "Formulation profile metadata is incorrect");
    require(old.matrix.variables.size() == tight.matrix.variables.size() &&
            old.matrix.rows.size() == tight.matrix.rows.size() &&
            old.matrix.cones.size() == tight.matrix.cones.size(), "Tight bound changed model dimensions");
    const double total = std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    std::size_t changed = 0;
    require(old.matrix.constant == tight.matrix.constant && old.matrix.minimize == tight.matrix.minimize,
            "Tight bound changed the objective constant or direction");
    for (std::size_t i = 0; i < old.matrix.cones.size(); ++i) {
        const auto& a = old.matrix.cones[i]; const auto& b = tight.matrix.cones[i];
        require(a.up == b.up && a.down == b.down && a.one == b.one && a.factor == b.factor,
                "Tight bound changed an inventory cone");
    }
    for (std::size_t i = 0; i < old.matrix.variables.size(); ++i) {
        const auto& a = old.matrix.variables[i]; const auto& b = tight.matrix.variables[i];
        require(a.name == b.name && a.lower == b.lower && a.upper == b.upper &&
                a.cost == b.cost && a.binary == b.binary, "Tight bound changed a variable or objective");
    }
    for (std::size_t i = 0; i < old.matrix.rows.size(); ++i) {
        const auto& a = old.matrix.rows[i]; const auto& b = tight.matrix.rows[i];
        require(a.name == b.name && a.sense == b.sense && a.rhs == b.rhs &&
                a.coefficients.size() == b.coefficients.size(), "Tight bound changed row structure");
        for (std::size_t j = 0; j < a.coefficients.size(); ++j) {
            const auto [index,value] = a.coefficients[j];
            const auto [new_index,new_value] = b.coefficients[j];
            require(index == new_index, "Tight bound changed coefficient ordering");
            if (value == new_value) continue;
            require(index >= old.layout.airline, "Tight bound changed a non-airline coefficient");
            const int edge = index-old.layout.airline, u = edge/d.n_nodes(), v = edge%d.n_nodes();
            require(u >= d.n_l && v >= d.n_l && u != v && d.allowed_links(u,v) &&
                    static_cast<int>(i) == old.link_rows[u*d.n_nodes()+v],
                    "Tight bound changed a non-HH link row");
            require(value == -total*d.max_orders && new_value == -total,
                    "Tight HH coefficient is not D versus old K*D");
            ++changed;
        }
    }
    std::size_t expected = 0;
    if (d.max_orders == 2) for (int u=d.n_l;u<d.n_nodes();++u)
        for (int v=d.n_l;v<d.n_nodes();++v) expected += u!=v && d.allowed_links(u,v);
    require(changed == expected, "Tight profile failed to change every expected HH gate");
    return changed;
}
Json compare(const Input& d, const PathCollection& paths, const std::vector<double>& fixed,
             const std::string& label, bool conditional) {
    auto old = build_formulation(d,paths,&fixed,nullptr,nullptr,false,conditional,false);
    auto tight = build_formulation(d,paths,&fixed,nullptr,nullptr,false,conditional,true);
    const auto changed = check_only_bound_changes(d,old,tight);
    SolverOptions options; options.seconds = 15; options.threads = 1; options.gap = 1e-9;
    NativeSolver old_solver(old.matrix,options,"tight_probe_old");
    NativeSolver tight_solver(tight.matrix,options,"tight_probe_new");
    auto a = old_solver.optimize(15), b = tight_solver.optimize(15);
    require(a.has_solution && b.has_solution && a.optimal && b.optimal,
            label+": fixed-design optimum not certified ("+a.status+" / "+b.status+")");
    require(repair_inventory_primal(old,a.values).at("passed"),label+": old primal repair failed");
    require(repair_inventory_primal(tight,b.values).at("passed"),label+": tight primal repair failed");
    const auto old_in_tight = point_residuals(tight.matrix,a.values);
    const auto tight_in_old = point_residuals(old.matrix,b.values);
    require(old_in_tight.at("passed"),label+": old solution violates new gate: "+old_in_tight.dump());
    require(tight_in_old.at("passed"),label+": tight solution violates old model: "+tight_in_old.dump());
    const double ca = objective_value(old.matrix,a.values), cb = objective_value(tight.matrix,b.values);
    const double tolerance = 2e-7*(1+std::max(std::abs(ca),std::abs(cb)));
    require(std::abs(ca-cb) <= tolerance,label+": fixed-design optimal objective changed");
    require(objective_value(tight.matrix,a.values) == ca && objective_value(old.matrix,b.values) == cb,
            label+": cross-profile objective calculation changed");
    double service = 0;
    for (int j : tight.flow) if (j >= 0) service += b.values[j];
    return {{"case",label},{"K",d.max_orders},{"policy",d.flow_policy},{"conditional",conditional},
        {"changed_hh_coefficients",changed},{"old_objective",ca},{"tight_objective",cb},
        {"absolute_objective_difference",std::abs(ca-cb)},{"tolerance",tolerance},
        {"old_solution_in_tight",old_in_tight},{"tight_solution_in_old",tight_in_old},
        {"service_task_flow",service},{"old_status",a.status},{"tight_status",b.status}};
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: mlnd_tight_airline_probe L01_INPUT OUTPUT_JSON");
        Json path_checks = Json::array(), solves = Json::array();
        for (const std::string policy : {"aggregate","prefix"}) for (int k : {1,2}) {
            const auto d = fixture(k,policy); const auto paths = enumerate_paths(d);
            auto proof = certify_paths(d,paths);
            if (k == 2) require(proof.at("repeated_hh_occurrences").get<int>() > 0 &&
                                proof.at("same_od_pairs").get<int>() > 0,
                                "K2 fixture did not exercise repeated arcs and repeated OD orders");
            proof["case"] = "tiny_"+policy+"_K"+std::to_string(k); path_checks.push_back(proof);
            for (const std::string mode : {"all_open","partial_assignment","one_closed_hh","assigned_without_links","closed"})
                for (bool conditional : {false,true})
                    solves.push_back(compare(d,paths,design(d,mode),"tiny_"+mode,conditional));
        }
        for (const std::string policy : {"aggregate","prefix"}) {
            const auto d = fixture(2,policy,true); const auto paths = enumerate_paths(d);
            auto proof = certify_paths(d,paths);
            require(proof.at("empty_paths").get<int>() > 0,"Relocation fixture did not contain empty paths");
            proof["case"] = "two_bases_"+policy; path_checks.push_back(proof);
            for (bool conditional : {false,true})
                solves.push_back(compare(d,paths,design(d,"cross_base"),"cross_base_relocation",conditional));
        }
        auto standard = read_input(argv[1],2);
        require(standard.n_l == 1 && standard.n_h == 10 && standard.n_od == 25,
                "Expected the standard 1L/10H/25OD L01 fixture");
        const auto paths = enumerate_paths(standard);
        auto proof = certify_paths(standard,paths); proof["case"] = "standard_L01_K2";
        path_checks.push_back(proof);
        for (const std::string mode : {"all_open","one_closed_hh","assigned_without_links","closed"})
            for (bool conditional : {false,true})
                solves.push_back(compare(standard,paths,design(standard,mode),"L01_"+mode,conditional));
        require(std::any_of(solves.begin(),solves.end(),[](const Json& s) {
            return s.at("service_task_flow").get<double>() > 1;
        }),"All comparisons were trivial zero-service cases");
        Json output = {{"passed",true},{"physical_path_checks",path_checks},
            {"fixed_design_comparisons",solves},{"comparison_count",solves.size()},
            {"scope","binary strategic designs; both demand formulations; current direct K<=2 path domain"},
            {"fractional_relaxation_note","D*y tightens the old fractional K*D*y relaxation; no fractional-equivalence claim"},
            {"standard_input",std::filesystem::absolute(argv[1]).string()}};
        std::ofstream stream(argv[2]);
        if (!stream) throw std::runtime_error("Cannot write probe output");
        stream << output.dump(2) << '\n';
        std::cout << "Tight airline probe passed: " << path_checks.size() << " complete catalogues, "
                  << solves.size() << " two-way fixed-design comparisons\n";
    } catch (const IloException& error) {
        std::cerr << error << '\n'; return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
