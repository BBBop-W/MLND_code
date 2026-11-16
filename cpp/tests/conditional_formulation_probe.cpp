// Fixed-design equivalence checks for the numeric demand-coefficient SOCP.
// This is a small deterministic regression probe, not a performance benchmark.
#include "mlnd/formulation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> mlnd::Matrix<T> filled(int rows, int columns, T value) {
    mlnd::Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
mlnd::Input fixture(int k, const std::string& policy) {
    mlnd::Input d;
    d.n_l = 1; d.n_h = d.n_od = 2; d.n_levels = 1; d.max_orders = k;
    d.name = "conditional_formulation"; d.model_version = "consecutive";
    d.route_mode = "direct"; d.flow_policy = policy;
    d.demand = {4, 7}; d.ground_cost = {5, 5}; d.deadline = {1, 1};
    d.base_cost = {.5}; d.slot_epsilon = {.05, .05};
    d.flight_times = mlnd::Matrix<double>(3,3);
    d.flight_times.values = {0,.05,.07, .05,0,.1, .07,.1,0};
    d.allowed_links = filled(3,3,1);
    for (int i = 0; i < 3; ++i) d.allowed_links(i,i) = 0;
    d.link_cost = filled(3,3,0.0);
    d.rider_mean = mlnd::Matrix<double>(2,2); d.rider_mean.values = {.08,.12,.1,.06};
    d.first_cost = filled(2,2,1.0); d.terminal_cost = filled(2,2,.5);
    d.receipt_mean = filled(2,2,.08); d.removal_mean = filled(2,2,.03);
    d.slot_counts = filled(2,1,10); d.slot_cost = filled(2,1,.1);
    d.occupancy_limit = filled(2,1,6.1690072893953225);
    d.kappa = mlnd::Matrix<double>(2,2); d.kappa.values = {.3,.9,.2,.7};
    d.delta = .04; d.operating_limit = 1; d.fleet_cost = 1;
    d.flight_cost = .2; d.waiting_cost = .3;
    return d;
}
std::vector<double> design(const mlnd::Input& d, int assignment_mask, bool closed = false) {
    const mlnd::DesignLayout layout(d);
    std::vector<double> result(layout.count,0);
    if (closed) return result;
    for (int i = 0; i < d.n_nodes(); ++i) result[layout.facility+i] = 1;
    for (int i = 0; i < d.n_h; ++i) result[layout.capacity+i] = 1;
    for (int r = 0; r < d.n_od; ++r)
        result[layout.assignment_index(r,0,0)] = (assignment_mask >> r) & 1;
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        result[layout.airline+u*d.n_nodes()+v] = d.allowed_links(u,v);
    return result;
}
std::map<std::string,int> indices(const mlnd::Formulation& f) {
    std::map<std::string,int> result;
    for (std::size_t i = 0; i < f.matrix.variables.size(); ++i)
        require(result.emplace(f.matrix.variables[i].name,static_cast<int>(i)).second,
                "Variable names must be unique");
    return result;
}
std::vector<double> project(const mlnd::Formulation& source, const std::vector<double>& values,
                            const mlnd::Formulation& target) {
    const auto names = indices(source);
    std::vector<double> out(target.matrix.variables.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = values.at(names.at(target.matrix.variables[i].name));
    return out;
}
std::vector<double> restore(const mlnd::Input& d, const mlnd::PathCollection& paths,
                           const std::vector<double>& fixed, const mlnd::Formulation& compact,
                           const std::vector<double>& values, const mlnd::Formulation& old) {
    const auto source = indices(compact), target = indices(old);
    std::vector<double> out(old.matrix.variables.size());
    std::vector<bool> written(out.size(),false);
    auto set = [&](const std::string& name, double value) {
        const int i = target.at(name); out[i] = value; written[i] = true;
    };
    for (std::size_t i = 0; i < old.matrix.variables.size(); ++i) {
        const auto found = source.find(old.matrix.variables[i].name);
        if (found != source.end()) { out[i] = values[found->second]; written[i] = true; }
    }
    if (d.effective_delta() > 0) for (std::size_t g = 0; g < old.lift_groups.size(); ++g) {
        const auto& group = old.lift_groups[g];
        const auto& segment = paths.initial_segments[group.front()];
        const std::string suffix = std::to_string(g);
        double head = 0;
        if (d.flow_policy == "aggregate") head = out[target.at("group_head_"+suffix)];
        else head = out[old.head[group.front()]];
        double rate = 0;
        for (int r = 0; r < d.n_od; ++r)
            rate += d.demand[r] * fixed[old.layout.assignment_index(r,segment.b[0],segment.b[1])];
        const double eta = rate > 0 ? -std::expm1(-d.effective_delta()*rate)/rate*head : 0;
        set("eta_"+suffix,eta);
        double cumulative = 0;
        for (int r = 0; r < d.n_od; ++r) {
            const double alpha = fixed[old.layout.assignment_index(r,segment.b[0],segment.b[1])];
            const double chain = head * std::exp(-d.effective_delta()*cumulative);
            set("u_"+suffix+"_"+std::to_string(r),alpha*eta);
            set("Z_"+suffix+"_"+std::to_string(r),chain);
            set("Y_"+suffix+"_"+std::to_string(r),alpha*chain);
            cumulative += d.demand[r]*alpha;
        }
        set("Z_"+suffix+"_"+std::to_string(d.n_od),head*std::exp(-d.effective_delta()*cumulative));
    }
    require(std::all_of(written.begin(),written.end(),[](bool x){return x;}),
            "Restoration left an original variable undefined");
    return out;
}
mlnd::Json compare(const mlnd::Input& d, mlnd::PathCollection paths,
                   const std::vector<double>& fixed, const std::string& name,
                   double fixed_rho = -1, bool expected_feasible = true) {
    auto old = mlnd::build_formulation(d,paths,&fixed);
    auto compact = mlnd::build_formulation(d,paths,&fixed,nullptr,nullptr,false,true);
    const std::size_t groups = old.lift_groups.size();
    const std::size_t removed_variables = d.effective_delta() > 0 ? groups*(3*d.n_od+2) : 0;
    const std::size_t removed_rows = d.effective_delta() > 0 ? groups*(7*d.n_od+3) : 0;
    require(old.matrix.variables.size()-compact.matrix.variables.size() == removed_variables,"Wrong eliminated variable count");
    require(old.matrix.rows.size()-compact.matrix.rows.size() == removed_rows,"Wrong eliminated row count");
    require(old.matrix.cones.size() == compact.matrix.cones.size(),"Cones changed");
    require(old.lift_groups == compact.lift_groups,"Task support changed");
    for (int index : compact.eta) require(index == -1,"Compact eta variable remains");
    if (fixed_rho >= 0) {
        old.matrix.row("test_fixed_rho",'E',fixed_rho,{{old.rho[0],1}});
        compact.matrix.row("test_fixed_rho",'E',fixed_rho,{{compact.rho[0],1}});
    }
    mlnd::SolverOptions options; options.seconds = 10; options.gap = 1e-9;
    mlnd::NativeSolver old_solver(old.matrix,options,"conditional_probe_old");
    mlnd::NativeSolver compact_solver(compact.matrix,options,"conditional_probe_compact");
    auto first = old_solver.optimize(10), second = compact_solver.optimize(10);
    require(first.has_solution == expected_feasible && second.has_solution == expected_feasible,
            name+": feasibility mismatch: "+first.status+" / "+second.status);
    mlnd::Json row = {{"case",name},{"policy",d.flow_policy},{"K",d.max_orders},
        {"feasible",expected_feasible},{"old_variables",old.matrix.variables.size()},
        {"compact_variables",compact.matrix.variables.size()},{"old_rows",old.matrix.rows.size()},
        {"compact_rows",compact.matrix.rows.size()},
        {"old_status",first.status},{"compact_status",second.status}};
    if (!expected_feasible) {
        require(old_solver.cplex.getStatus() == IloAlgorithm::Infeasible &&
                compact_solver.cplex.getStatus() == IloAlgorithm::Infeasible,
                name+": no-point outcomes are not certified infeasibility");
        return row;
    }
    require(first.optimal && second.optimal,name+": small fixture did not solve optimally");
    require(mlnd::repair_inventory_primal(old,first.values).at("passed"),name+": old primal invalid");
    require(mlnd::repair_inventory_primal(compact,second.values).at("passed"),name+": compact primal invalid");
    const auto projected = project(old,first.values,compact);
    const auto restored = restore(d,paths,fixed,compact,second.values,old);
    const auto projected_check = mlnd::point_residuals(compact.matrix,projected);
    const auto restored_check = mlnd::point_residuals(old.matrix,restored);
    require(projected_check.at("passed"),name+": old-to-compact projection invalid: "+projected_check.dump());
    require(restored_check.at("passed"),name+": compact-to-old restoration invalid: "+restored_check.dump());
    const double old_cost = mlnd::objective_value(old.matrix,first.values);
    const double compact_cost = mlnd::objective_value(compact.matrix,second.values);
    require(std::abs(old_cost-compact_cost) <= 2e-5,name+": optimal costs differ");
    require(std::abs(mlnd::objective_value(old.matrix,restored)-compact_cost) <= 1e-10,name+": restoration changed cost");
    require(std::abs(mlnd::objective_value(compact.matrix,projected)-old_cost) <= 1e-10,name+": projection changed cost");
    const auto output = mlnd::solution_json(d,paths,compact,second.values);
    const auto heads = output.at("_solution").at("head").get<std::vector<double>>();
    const auto eta = output.at("_solution").at("eta").get<std::vector<double>>();
    for (std::size_t j = 0; j < heads.size(); ++j) {
        const auto b = paths.initial_segments[j].b;
        double rate = 0;
        for (int r = 0; r < d.n_od; ++r)
            rate += d.demand[r]*fixed[compact.layout.assignment_index(r,b[0],b[1])];
        const double factor = rate > 0 ? -std::expm1(-d.effective_delta()*rate)/rate : 0;
        require(compact.segment_eta_factor[j] == factor,name+": stored eta factor is incorrect");
        require(eta[j] == heads[j]*factor,name+": public eta is not per segment");
    }
    auto tiny = second.values;
    for (std::size_t j = 0; j < compact.head.size(); ++j) tiny[compact.head[j]] = (j+1)*1e-18;
    const auto tiny_output = mlnd::solution_json(d,paths,compact,tiny).at("_solution").at("eta");
    for (std::size_t j = 0; j < heads.size(); ++j)
        require(tiny_output[j].get<double>() == tiny[compact.head[j]]*compact.segment_eta_factor[j],"Tiny eta was clipped");
    row["old_objective"] = old_cost; row["compact_objective"] = compact_cost;
    row["objective_difference"] = std::abs(old_cost-compact_cost);
    row["projection"] = projected_check; row["restoration"] = restored_check;
    row["served"] = output.at("served");
    return row;
}
mlnd::Input route_tradeoff(const std::string& policy) {
    auto d = fixture(2,policy); d.n_od = 3;
    d.demand = {4,7,5}; d.ground_cost = {5,5,5}; d.deadline = {.3,.3,.3};
    d.flight_times.values = {0,.05,.05,.05,0,.1,.05,.1,0};
    d.rider_mean = mlnd::Matrix<double>(3,2); d.rider_mean.values = {0,0,2,0,0,2};
    d.first_cost = filled(3,2,1.0); d.terminal_cost = filled(3,2,.5);
    d.receipt_mean = filled(3,2,0.0); d.removal_mean = filled(3,2,0.0);
    d.slot_counts = filled(2,1,100); d.occupancy_limit = filled(2,1,84.13927721831419);
    d.kappa = mlnd::Matrix<double>(3,3); d.kappa.values = {0,1,1,0,0,0,0,0,0};
    d.operating_limit = .31;
    return d;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: mlnd_conditional_formulation_probe OUTPUT_JSON");
        mlnd::Json cases = mlnd::Json::array();
        for (const std::string policy : {"aggregate","prefix"}) {
            for (int k : {1,2}) {
                const auto d = fixture(k,policy); const auto paths = mlnd::enumerate_paths(d);
                require(!paths.initial_segments.empty() && !paths.service_paths.empty(),"Empty fixture path catalogue");
                for (int mask : {0,1,2,3})
                    cases.push_back(compare(d,paths,design(d,mask),"assignment_"+std::to_string(mask)));
                cases.push_back(compare(d,paths,design(d,0,true),"closed_empty_pool"));
            }
            auto d = fixture(2,policy); d.kappa.values = {1,0,0,0};
            cases.push_back(compare(d,mlnd::enumerate_paths(d),design(d,3),"zero_kappa_still_in_lambda"));
            d.kappa.values = {1,1,1,1};
            auto paths = mlnd::enumerate_paths(d);
            require(std::any_of(paths.service_paths.begin(),paths.service_paths.end(),[](const auto& p){return p.first_od==p.second_od;}),"Self-pair fixture absent");
            paths.service_paths.erase(std::remove_if(paths.service_paths.begin(),paths.service_paths.end(),
                [](const auto& p){return p.second_od >= 0;}),paths.service_paths.end());
            const auto missing = compare(d,paths,design(d,3),"positive_pair_without_path");
            require(missing.at("served").get<double>() <= 1e-7,"Missing required branch did not suppress drone service");
            cases.push_back(missing);
            d = fixture(2,policy); d.demand[0] = 0;
            cases.push_back(compare(d,mlnd::enumerate_paths(d),design(d,3),"zero_original_rate"));
            d = route_tradeoff(policy);
            cases.push_back(compare(d,mlnd::enumerate_paths(d),design(d,7),"route_tradeoff",.5,policy=="aggregate"));
            d = fixture(2,policy); d.n_l = 2; d.base_cost = {.5,.5};
            d.flight_times = mlnd::Matrix<double>(4,4);
            d.flight_times.values = {0,.03,.05,.07, .03,0,.06,.04, .05,.06,0,.1, .07,.04,.1,0};
            d.allowed_links = filled(4,4,1); d.link_cost = filled(4,4,0.0);
            for (int i = 0; i < 4; ++i) d.allowed_links(i,i) = 0;
            auto cross_base = design(d,3);
            const mlnd::DesignLayout layout(d);
            for (int r = 0; r < d.n_od; ++r) {
                cross_base[layout.assignment+r*4] = 0;
                cross_base[layout.assignment+r*4+1] = 1;
            }
            const auto cross_paths = mlnd::enumerate_paths(d);
            require(!cross_paths.empty_paths.empty(),"Cross-base fixture has no relocation paths");
            cases.push_back(compare(d,cross_paths,cross_base,"cross_base_relocation"));
        }
        const auto d = fixture(2,"aggregate"); const auto paths = mlnd::enumerate_paths(d);
        auto fixed = design(d,3);
        for (int kind : {0,1,2}) {
            bool rejected = false;
            auto candidate = fixed;
            if (kind == 2) candidate[mlnd::DesignLayout(d).assignment] = .5;
            try { mlnd::build_formulation(d,paths,kind==0 ? nullptr : &candidate,nullptr,nullptr,kind==1,true); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected,"Noninteger or free conditional mode was accepted");
        }
        std::ofstream stream(argv[1]);
        if (!stream) throw std::runtime_error("Cannot write conditional formulation probe output");
        stream << mlnd::Json{{"passed",true},{"cases",cases},{"case_count",cases.size()},
            {"invalid_modes_rejected",3},{"objective_absolute_tolerance",2e-5}}.dump(2) << '\n';
        std::cout << "Conditional formulation cases passed: " << cases.size() << '\n';
    } catch (const IloException& error) { std::cerr << error << '\n'; return 1; }
      catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
