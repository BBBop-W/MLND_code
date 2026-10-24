#include "mlnd/strengthening.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

namespace {
void require(bool test, const std::string& message) {
    if (!test) throw std::runtime_error(message);
}
std::vector<double> lift_solution(const mlnd::Input& d, const mlnd::Formulation& master,
                                  int theta, const mlnd::Json& result) {
    auto design = mlnd::solution_design(d, result);
    design.resize(master.matrix.variables.size(), 0);
    design[theta] = result.at("continuous_cost");
    return design;
}
double row_violation(const mlnd::RowRecord& row, const std::vector<double>& point) {
    double lhs=0;
    for (auto [j,c] : row.coefficients) lhs+=c*point.at(j);
    return row.rhs-lhs;
}
void mincut_fixture() {
    // Total supply and total capacity are both 20, but the first order can
    // reach only the empty locker. The mincut proves savings <=10.
    mlnd::CapacitySavingsNetwork n;
    n.nodes=6; n.source=0; n.sink=5; n.courier_cost=20; n.numerical_margin=1e-9;
    n.arcs={{0,1,0,{{0,10}}},{0,2,0,{{1,10}}},{1,3,0,{{0,10}}},
            {2,4,0,{{1,10}}},{3,5,0,{{2,20}}},{4,5,0,{{3,20}}}};
    const auto bottleneck=n.separate({1,1,0,1},0,4);
    require(bottleneck.violated && std::abs(bottleneck.maxflow-10)<1e-8 &&
            std::abs(bottleneck.cut_capacity-10)<1e-8,"Mincut missed a local locker bottleneck");
    std::vector<std::vector<double>> points;
    for (double a : {0.0,.5,1.0}) for (double b : {0.0,.5,1.0})
        for (double c : {0.0,.5,1.0}) for (double d : {0.0,.5,1.0})
            points.push_back({a,b,c,d,20-std::min(10*a,20*c)-std::min(10*b,20*d)});
    for (const auto& p : points) {
        const auto cut=n.separate(p,0,4);
        require(std::abs(cut.maxflow-(20-p[4]))<1e-8,"Maxflow disagrees with explicit tiny network");
        require(std::abs(cut.cut_capacity-cut.maxflow)<1e-8,"Cut did not recover source crossing edges");
        for (const auto& q : points)
            require(row_violation(cut.row,q)<1e-8,"A capacity cut excludes another fractional feasible point");
    }
}
mlnd::Json resource_lp_fixture() {
    // The first service saves 10 per unit load; the second saves only 1.
    // Monetary max-flow can spend its global ratio on both orders and allows
    // savings 19. The weighted LP retains the distinction and permits 14.
    mlnd::ResourceSavingsModel model;
    model.courier_cost=20; model.numerical_margin=1e-9;
    model.group_rhs={{{0,1}},{{1,1}}}; model.locker_rhs={{{2,5}}};
    model.occurrences={{0,0,10,1,{}},{1,0,9,9,{}}};
    mlnd::SolverOptions options; options.threads=1; options.seconds=5;
    mlnd::ResourceSavingsRelaxation relaxation(model,options);
    mlnd::CapacitySavingsNetwork network;
    network.nodes=5; network.source=0; network.sink=4; network.courier_cost=20;
    network.arcs={{0,1,0,{{0,10}}},{0,2,0,{{1,9}}},
        {1,3,0,{{0,10}}},{2,3,0,{{1,9}}},{3,4,0,{{2,50}}}};
    const auto oracle=[](double a,double b,double c) {
        const double first=std::min(a,5*c);
        return 10*first+9*std::min(b,std::max(0.0,5*c-first)/9);
    };
    std::vector<std::vector<double>> points;
    for (double a:{0.0,.5,1.0}) for (double b:{0.0,.5,1.0}) for (double c:{0.0,.5,1.0})
        points.push_back({a,b,c,20-oracle(a,b,c)});
    double max_projection_violation=0,max_oracle_difference=0;
    for (const auto& p:points) {
        const auto cut=relaxation.separate(p,0,3,5);
        require(cut.valid && cut.optimal && !cut.fallback,"Weighted resource LP lacks a certified optimal fixture solve");
        require(cut.max_dual_violation_after==0,"Weighted resource repair left a dual infeasibility");
        max_oracle_difference=std::max(max_oracle_difference,std::abs(cut.dual_savings-(20-p[3])));
        require(cut.dual_savings<=network.separate(p,0,3).maxflow+1e-7,
            "Weighted resource relaxation is weaker than monetary max-flow on fixture");
        for (const auto& q:points)
            max_projection_violation=std::max(max_projection_violation,row_violation(cut.row,q));
        const auto fallback=relaxation.separate(p,0,3,0);
        require(fallback.valid && fallback.fallback && !fallback.optimal,
            "No-budget resource separator must use a certified fallback, not claim optimization");
        for (const auto& q:points)
            require(row_violation(fallback.row,q)<=1e-7,"Fallback resource dual cut excludes a feasible point");
    }
    require(max_projection_violation<=1e-7 && max_oracle_difference<=1e-7,
        "Weighted resource cut failed the analytic fractional LP oracle");
    const auto strict=relaxation.separate({1,1,1},0,3,5);
    require(std::abs(strict.dual_savings-14)<1e-7 &&
        std::abs(network.separate({1,1,1},0,3).maxflow-19)<1e-7,
        "Weighted resource LP failed to improve the max-flow savings upper bound");
    model.occurrences[0].slot_time=0;
    mlnd::ResourceSavingsRelaxation zero_load(model,options);
    const auto free_service=zero_load.separate({1,1,0},0,3,5);
    require(free_service.valid && free_service.optimal && std::abs(free_service.dual_savings-10)<1e-7,
        "Zero slot-time service was discarded or divided by zero");
    auto invalid=model; invalid.occurrences[0].slot_time=-1;
    bool rejected=false;
    try {mlnd::ResourceSavingsRelaxation bad(invalid,options);}
    catch (const std::invalid_argument&) {rejected=true;}
    require(rejected,"Negative resource coefficients must be rejected");
    return {{"fractional_cross_projection_checks",points.size()*points.size()},
        {"max_projection_violation",max_projection_violation},
        {"max_analytic_oracle_difference",max_oracle_difference},
        {"weighted_savings_upper_bound",strict.dual_savings},{"maxflow_savings_upper_bound",19},
        {"zero_slot_time_savings",free_service.dual_savings},{"fallback_certified",true}};
}
mlnd::Json airline_lp_fixture() {
    // Three independently binding gates (loaded arrival, departure, return).
    // Capacity is deliberately ample: locker-only cuts miss a closed route.
    mlnd::ResourceSavingsModel model;
    model.courier_cost=20; model.numerical_margin=1e-9;
    model.group_rhs={{{0,1}}}; model.locker_rhs={{{1,10}}};
    model.airline_rhs={{{2,1}},{{3,1}},{{4,1}}};
    model.occurrences={{0,0,10,1,{0,1,2}}};
    mlnd::SolverOptions options; options.threads=1;
    mlnd::ResourceSavingsRelaxation gated(model,options);
    auto ungated=model; ungated.airline_rhs.clear(); ungated.occurrences[0].airline_resources.clear();
    mlnd::ResourceSavingsRelaxation legacy(ungated,options);
    std::vector<std::vector<double>> points;
    for (double a:{0.0,.5,1.0}) for (double load:{0.0,.5,1.0})
        for (double dep:{0.0,.5,1.0}) for (double ret:{0.0,.5,1.0})
            points.push_back({a,1,load,dep,ret,20-10*std::min({a,load,dep,ret})});
    double max_oracle_difference=0,max_cross_violation=0;
    for (const auto& p:points) {
        const auto cut=gated.separate(p,0,5,5);
        require(cut.valid && cut.optimal && !cut.fallback && cut.airline_dual.size()==3,
            "Airline LP fixture did not return a certified optimal dual");
        max_oracle_difference=std::max(max_oracle_difference,std::abs(cut.dual_savings-(20-p[5])));
        for (const auto& q:points)
            max_cross_violation=std::max(max_cross_violation,row_violation(cut.row,q));
    }
    require(max_oracle_difference<1e-7 && max_cross_violation<1e-7,
        "Airline LP failed its analytic oracle or fractional cross-design projection");
    for (int gate=2;gate<=4;++gate) {
        std::vector<double> point{1,1,1,1,1}; point[gate]=0;
        const auto strengthened=gated.separate(point,0,5,5);
        const auto old=legacy.separate(point,0,5,5);
        require(strengthened.valid && strengthened.dual_savings<1e-7 &&
                old.valid && std::abs(old.dual_savings-10)<1e-7,
            "A closed airline gate failed to strengthen the locker-only relaxation");
        const auto fallback=gated.separate(point,0,5,0);
        require(fallback.valid && fallback.fallback && std::abs(fallback.dual_savings-10)<1e-7,
            "Airline zero-budget fallback was not conservatively certified");
    }
    auto invalid=model; invalid.occurrences[0].airline_resources={3};
    bool invalid_rejected=false;
    try {mlnd::ResourceSavingsRelaxation bad(invalid,options);}
    catch(const std::invalid_argument&) {invalid_rejected=true;}
    require(invalid_rejected,"Out-of-range airline resource was accepted");
    invalid=model; invalid.occurrences[0].airline_resources={0,0};
    invalid_rejected=false;
    try {mlnd::ResourceSavingsRelaxation bad(invalid,options);}
    catch(const std::invalid_argument&) {invalid_rejected=true;}
    require(invalid_rejected,"Duplicate airline resource was accepted");
    return {{"fractional_designs",points.size()},
        {"fractional_cross_projection_checks",points.size()*points.size()},
        {"max_analytic_oracle_difference",max_oracle_difference},
        {"max_cross_projection_violation",max_cross_violation},
        {"closed_gate_savings_upper_bound",0},{"ungated_savings_upper_bound",10},
        {"fallback_certified",true},{"invalid_resource_rejected",true}};
}
template<class T> mlnd::Matrix<T> constant(int rows,int cols,T value) {
    mlnd::Matrix<T> matrix(rows,cols);
    std::fill(matrix.values.begin(),matrix.values.end(),value);
    return matrix;
}
// Same physical fixture as tests/unit/test_solver.py; input construction is
// local to this test and never enters production instance generation.
mlnd::Input physical_fixture(int k,const std::string& policy,int bases=1) {
    require((bases==1 || bases==2) && (policy=="aggregate" || policy=="prefix"),
        "Fixture needs one/two bases and aggregate/prefix policy");
    mlnd::Input d;
    d.n_l=bases; d.n_h=2; d.n_od=2; d.n_levels=2; d.max_orders=k;
    d.name="airline_resource_physical_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy=policy; d.demand={4,7};
    const int N=bases+2;
    d.flight_times=constant<double>(N,N,0);
    for(int l=0;l<bases;++l) for(int i=0;i<2;++i)
        d.flight_times(l,bases+i)=d.flight_times(bases+i,l)=.05+.02*i+.03*l;
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    if(bases==2) d.flight_times(0,1)=d.flight_times(1,0)=.03;
    d.rider_mean=constant<double>(2,2,0); d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant<double>(2,2,1); d.terminal_cost=constant<double>(2,2,.5);
    d.receipt_mean=constant<double>(2,2,.08); d.removal_mean=constant<double>(2,2,.03);
    d.ground_cost={5,5}; d.deadline={1,1}; d.base_cost.assign(bases,.5);
    d.slot_counts=constant<int>(2,2,0); d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant<double>(2,2,0); d.slot_cost.values={.1,.2,.1,.2};
    d.slot_epsilon={.05,.05}; d.link_cost=constant<double>(N,N,0);
    d.kappa=constant<double>(2,2,0); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.allowed_links=constant<int>(N,N,1);
    for(int i=0;i<N;++i) d.allowed_links(i,i)=0;
    d.occupancy_limit=constant<double>(2,2,0);
    d.occupancy_limit.values={6.1690072893953225,14.072024748341311,6.1690072893953225,14.072024748341311};
    return d;
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6)
            throw std::invalid_argument("Usage: mlnd_strengthening_probe INSTANCE K OUTPUT_JSON [REFERENCE_RESULT_JSON], or --self-test K OUTPUT_JSON [aggregate|prefix] [BASES]");
        const bool self_test=std::string(argv[1])=="--self-test";
        require(self_test || argc<=5,"Unexpected extra argument for standard input");
        const auto d = self_test ? physical_fixture(std::stoi(argv[2]),argc>=5?argv[4]:"aggregate",argc==6?std::stoi(argv[5]):1) :
            mlnd::read_input(argv[1], std::stoi(argv[2]));
        const auto bounds = mlnd::analytical_master_bounds(d);
        const auto network = mlnd::capacity_savings_network(d);
        mincut_fixture();
        const auto resource_fixture=resource_lp_fixture();
        const auto airline_fixture=airline_lp_fixture();
        mlnd::SolverOptions resource_options; resource_options.threads=1;
        const auto resource_model=mlnd::resource_savings_model(d,network,true);
        const auto legacy_resource_model=mlnd::resource_savings_model(d,network,false);
        mlnd::ResourceSavingsRelaxation resource(resource_model,resource_options);
        require(resource_model.airline_rhs.size()==static_cast<std::size_t>(d.n_h+2*d.n_l) &&
            legacy_resource_model.airline_rhs.empty() &&
            std::all_of(legacy_resource_model.occurrences.begin(),legacy_resource_model.occurrences.end(),
                [](const auto& o){return o.airline_resources.empty();}),
            "Airline relaxation switch does not preserve the locker-only model");
        for (const auto& o:resource_model.occurrences)
            require(o.airline_resources==std::vector<int>{o.locker,d.n_h+(o.group/d.n_l)%d.n_l,
                d.n_h+d.n_l+o.group%d.n_l},"Occurrence is attached to the wrong airline gates");
        const auto paths = mlnd::enumerate_paths(d); // Validation oracle only.
        double max_cost_violation = -INFINITY, max_slot_violation = -INFINITY, max_ratio_violation = -INFINITY;
        double max_delivery_cost_violation=-INFINITY,max_delivery_slot_violation=-INFINITY;
        double max_airline_path_violation=0;
        for (const auto& p : paths.service_paths) {
            double allocated = 0, minimum_slots = 0, courier = 0, actual_slots = 0;
            for (auto [r, count] : p.served_counts) {
                const int index = (r * d.n_l + p.b[0]) * d.n_l + p.b[1];
                require(bounds.pair_possible[index], "A real task was incorrectly marked impossible");
                allocated += count * bounds.unit_cost[index];
                minimum_slots += count * bounds.slot_time[r];
                courier += count * d.ground_cost[r];
            }
            for (auto [i, duration] : p.slot_times) actual_slots += duration;
            const double augmented_cost = p.cost + d.fleet_cost * p.time;
            max_cost_violation = std::max(max_cost_violation, allocated - augmented_cost);
            max_slot_violation = std::max(max_slot_violation, minimum_slots - actual_slots);
            if (bounds.finite_capacity_savings)
                max_ratio_violation = std::max(max_ratio_violation, courier - augmented_cost - bounds.savings_per_slot_time * actual_slots);
            double delivery_allocated=0;
            std::map<int,double> delivery_load,actual_load;
            for (auto [j,load] : p.slot_times) actual_load[j]=load;
            for (std::size_t k=0;k<p.deliveries.size();++k) {
                const int r=k==0?p.first_od:p.second_od,j=p.deliveries[k];
                const int index=((r*d.n_l+p.b[0])*d.n_l+p.b[1])*d.n_h+j;
                require(network.eligible[index],"Capacity network excluded a real order occurrence");
                delivery_allocated+=network.unit_cost[index];
                delivery_load[j]+=network.slot_time[index];
            }
            max_delivery_cost_violation=std::max(max_delivery_cost_violation,delivery_allocated-augmented_cost);
            for (auto [j,load] : delivery_load)
                max_delivery_slot_violation=std::max(max_delivery_slot_violation,load-actual_load[j]);
            std::vector<double> occurrence_load(d.n_h+2*d.n_l),link_load(occurrence_load.size());
            for (int j:p.deliveries) ++occurrence_load[j];
            occurrence_load[d.n_h+p.b[0]]=p.deliveries.size();
            occurrence_load[d.n_h+d.n_l+p.b[1]]=p.deliveries.size();
            for (auto [u,v,count]:p.arc_counts) {
                if (u>=d.n_l && v>=d.n_l) link_load[v-d.n_l]+=count;
                else if (u<d.n_l && v>=d.n_l) link_load[d.n_h+u]+=d.max_orders*count;
                else if (u>=d.n_l && v<d.n_l) link_load[d.n_h+d.n_l+v]+=d.max_orders*count;
            }
            for (std::size_t a=0;a<occurrence_load.size();++a)
                max_airline_path_violation=std::max(max_airline_path_violation,occurrence_load[a]-link_load[a]);
        }
        require(max_cost_violation <= 1e-9, "Per-order cost allocation overestimates a real task cost");
        require(max_slot_violation <= 1e-9, "Per-order slot allocation overestimates a real task load");
        require(max_ratio_violation <= 1e-9, "Analytical savings/load upper bound fails on a real task");
        require(max_delivery_cost_violation<=1e-9 && max_delivery_slot_violation<=1e-9,
                "Delivery-network allocations overestimate a real path resource or cost");
        require(max_airline_path_violation==0,"Airline gate order counts exceed original path arc counts");
        // Verify RHS against the authoritative common full-matrix link rows.
        // This catches any later change of M in formulation.cpp.
        const std::vector<int> no_columns;
        const auto full=mlnd::build_formulation(d,paths,nullptr,&no_columns,&no_columns);
        for (std::size_t a=0;a<resource_model.airline_rhs.size();++a) {
            std::map<int,double> expected;
            for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v) {
                bool belongs=false;
                if (a<static_cast<std::size_t>(d.n_h)) belongs=u>=d.n_l && v==d.n_l+static_cast<int>(a);
                else if (a<static_cast<std::size_t>(d.n_h+d.n_l)) belongs=u==static_cast<int>(a)-d.n_h && v>=d.n_l;
                else belongs=u>=d.n_l && v==static_cast<int>(a)-d.n_h-d.n_l;
                if (!belongs) continue;
                const int index=full.layout.airline+u*d.n_nodes()+v;
                for (auto [j,c]:full.matrix.rows.at(full.link_rows.at(u*d.n_nodes()+v)).coefficients)
                    if (j==index && c<0) expected[index]=-c*(a<static_cast<std::size_t>(d.n_h)?1:d.max_orders);
            }
            require(expected.size()==resource_model.airline_rhs[a].size(),"Airline gate includes incorrect physical links");
            for (auto [j,c]:resource_model.airline_rhs[a])
                require(expected.count(j) && c>=expected[j] && c-expected[j]<=1e-9*(1+expected[j]),
                    "Airline gate M differs from the conservative common link row bound");
        }

        // Empty generated pool must yield exactly the same master coefficients
        // as a complete catalogue, including the resource bound.
        auto master = mlnd::design_model(d, false);
        const int theta = master.matrix.variable("operating_cost", 0, bounds.courier_cost, 1);
        const auto base_rows = master.matrix.rows.size();
        mlnd::PathCollection implicit; implicit.complete = false;
        const auto stats = mlnd::add_master_strengthening(d, implicit, master, theta);
        require(master.matrix.variables.size() == static_cast<std::size_t>(master.layout.count + 1),
                "Master strengthening introduced an operational auxiliary variable");
        require(stats.at("added_variables").get<int>() == 0 && master.matrix.cones.empty(),
                "Design master must not contain operational auxiliaries or cones");
        require(master.matrix.rows.size() == base_rows + 1 + bounds.finite_capacity_savings,
                "Design master contains unexpected strengthening constraints");
        require(master.matrix.rows.at(base_rows).name == "analytical_courier_savings_bound",
                "Design master lost the assignment savings cut");
        if (bounds.finite_capacity_savings)
            require(master.matrix.rows.back().name == "analytical_locker_savings_bound",
                    "Design master lost the locker capacity savings cut");
        auto assignment_only = mlnd::design_model(d, false);
        assignment_only.matrix.variable("operating_cost", 0, bounds.courier_cost, 1);
        const auto assignment_stats = mlnd::add_master_strengthening(d, implicit, assignment_only, theta, false);
        require(assignment_stats.at("added_variables").get<int>() == 0 &&
                assignment_only.matrix.rows.size() == base_rows + 1,
                "Disabling resource bounds changed design variables or assignment savings cut");
        auto complete_master = mlnd::design_model(d, false);
        complete_master.matrix.variable("operating_cost", 0, bounds.courier_cost, 1);
        mlnd::add_master_strengthening(d, paths, complete_master, theta);
        require(master.matrix.rows.size() == complete_master.matrix.rows.size(), "Bounds depend on generated pool size");
        for (std::size_t i = 0; i < master.matrix.rows.size(); ++i) {
            const auto& a = master.matrix.rows[i]; const auto& b = complete_master.matrix.rows[i];
            require(a.name == b.name && a.sense == b.sense && a.rhs == b.rhs && a.coefficients == b.coefficients,
                    "Bound coefficients depend on catalogue completeness");
        }
        const auto courier = mlnd::all_courier_result(d, implicit);
        const auto zero_point = lift_solution(d, master, theta, courier);
        const auto zero_residual = mlnd::point_residuals(master.matrix, zero_point);
        require(zero_residual.at("passed").get<bool>(), "Strengthening removes all-courier solution");

        // Required-arc absence is conclusive. Zero-duration load is not:
        // profitable zero-load orders disable the finite resource ratio.
        auto disconnected = d;
        std::fill(disconnected.allowed_links.values.begin(), disconnected.allowed_links.values.end(), 0);
        const auto disconnected_bounds = mlnd::analytical_master_bounds(disconnected);
        require(std::none_of(disconnected_bounds.pair_possible.begin(), disconnected_bounds.pair_possible.end(), [](bool v) { return v; }),
                "Disconnected physical graph admitted a potentially supported order");
        require(std::abs(disconnected_bounds.operating_floor - disconnected_bounds.courier_cost) <= 1e-7,
                "Disconnected graph lost its all-courier lower bound");
        auto zero_load = d;
        std::fill(zero_load.flight_times.values.begin(), zero_load.flight_times.values.end(), 0);
        std::fill(zero_load.removal_mean.values.begin(), zero_load.removal_mean.values.end(), 0);
        std::fill(zero_load.first_cost.values.begin(), zero_load.first_cost.values.end(), 0);
        std::fill(zero_load.terminal_cost.values.begin(), zero_load.terminal_cost.values.end(), 0);
        if (std::any_of(d.ground_cost.begin(), d.ground_cost.end(), [](double c) { return c > 0; }) && !paths.service_paths.empty())
            require(!mlnd::analytical_master_bounds(zero_load).finite_capacity_savings, "Profitable zero-load orders need an unbounded savings/load ratio");
        const auto zero_network=mlnd::capacity_savings_network(zero_load);
        for (const auto& arc : zero_network.arcs) if (arc.constant>0)
            require(arc.to==zero_network.sink && arc.coefficients.empty(),
                    "Zero-duration sink capacity must be a constant, not scaled by a facility");

        mlnd::Json report{{"K", d.max_orders}, {"service_paths_checked", paths.service_paths.size()},
            {"cost_allocation_max_violation", max_cost_violation}, {"slot_allocation_max_violation", max_slot_violation},
            {"savings_ratio_max_violation", max_ratio_violation}, {"all_courier_projection", zero_residual},
            {"bounds", stats}, {"pool_independent", true}, {"edge_cases_passed", true}};
        report["input"]={{"path",self_test?"in_memory_physical_fixture":std::filesystem::absolute(d.instance_path).string()},
            {"bases",d.n_l},{"lockers",d.n_h},{"orders",d.n_od},{"levels",d.n_levels},
            {"manifest_sha256",self_test?mlnd::Json(nullptr):mlnd::Json(mlnd::sha256_file(d.instance_path/"manifest.json"))},
            {"manifest_files_verified",!self_test},{"flow_policy",d.flow_policy}};
        report["master_structure"] = {{"variables", master.matrix.variables.size()},
            {"design_variables", master.layout.count}, {"operational_auxiliary_variables", 0},
            {"base_rows", base_rows}, {"rows_before_incumbent_cutoff", master.matrix.rows.size()},
            {"cones", master.matrix.cones.size()}};
        report["capacity_network"]={{"metadata",network.metadata},{"local_bottleneck_fixture_passed",true},
            {"fractional_cross_projection_checks",81*81},
            {"path_cost_max_violation",max_delivery_cost_violation},
            {"path_locker_load_max_violation",max_delivery_slot_violation}};
        report["resource_lp"]={{"metadata",resource.metadata()},{"fixture",resource_fixture},
            {"airline_fixture",airline_fixture},{"airline_path_max_violation",max_airline_path_violation},
            {"airline_rhs_matches_common_formulation",true},{"airline_switch_restores_legacy",true}};
        if (argc == 5 && !self_test) {
            std::ifstream stream(argv[4]); mlnd::Json reference; stream >> reference;
            const auto source_path=std::filesystem::path(argv[4]).parent_path()/"input_reference.json";
            if (std::filesystem::exists(source_path)) {
                std::ifstream source_stream(source_path); mlnd::Json source; source_stream>>source;
                require(source.at("manifest")==d.manifest && source.at("max_orders")==d.max_orders,
                        "Saved feasible reference belongs to another input or K");
                report["reference_input_manifest_verified"]=true;
            }
            if (!reference.contains("_solution")) {
                std::ifstream solution_stream(std::filesystem::path(argv[4]).parent_path() / "solution.json");
                solution_stream >> reference["_solution"];
            }
            const auto point = lift_solution(d, master, theta, reference);
            const auto residual = mlnd::point_residuals(master.matrix, point);
            require(residual.at("passed").get<bool>(), "Strengthening excludes a saved feasible solution");
            report["reference_projection"] = residual;
            // Every convex mixture of valid projections remains feasible in
            // the continuous master; exercise a nonintegral design explicitly.
            auto fractional = point;
            for (std::size_t i = 0; i < fractional.size(); ++i) fractional[i] = .37 * point[i] + .63 * zero_point[i];
            const auto fractional_residual = mlnd::point_residuals(master.matrix, fractional);
            require(fractional_residual.at("passed").get<bool>(), "Strengthening excludes a convex projection");
            report["fractional_projection"] = fractional_residual;
            double max_cut_violation=-INFINITY;
            double max_resource_cut_violation=-INFINITY;
            for (int sample=0;sample<32;++sample) {
                std::vector<double> candidate(master.layout.count);
                for (std::size_t j=0;j<candidate.size();++j)
                    candidate[j]=((j*37+sample*19+sample*j)%11)/10.0;
                const auto cut=network.separate(candidate,0,theta);
                const auto resource_cut=resource.separate(candidate,0,theta,5);
                require(resource_cut.valid,"Resource separator failed to certify a finite input design");
                const std::vector<double>* projections[]={&point,&fractional,&zero_point};
                for (const auto* projection : projections) {
                    max_cut_violation=std::max(max_cut_violation,row_violation(cut.row,*projection));
                    max_resource_cut_violation=std::max(max_resource_cut_violation,row_violation(resource_cut.row,*projection));
                }
            }
            require(max_cut_violation<=1e-6,"A capacity cut excludes a saved feasible solution or its convex mixture");
            require(max_resource_cut_violation<=1e-6,"A weighted resource cut excludes a saved feasible solution or its convex mixture");
            report["capacity_network"]["reference_cross_projection_max_violation"]=max_cut_violation;
            report["resource_lp"]["reference_cross_projection_max_violation"]=max_resource_cut_violation;
        }
        mlnd::SolverOptions options; options.seconds = 10; options.gap = 1e-9;
        mlnd::NativeSolver solver(master.matrix, options);
        const auto solved = solver.optimize(options.seconds);
        require(solved.optimal, "Strengthened root LP did not solve to optimality");
        require(solved.objective + 1e-6 >= stats.at("unconstrained_operating_floor").get<double>(), "Master bound is weaker than its input-only floor");
        require(solved.objective <= bounds.courier_cost + 1e-6, "Master lower bound exceeds all-courier optimum candidate");
        report["root_lp_objective"] = solved.objective;
        const auto root_resource_cut=resource.separate(solved.values,solved.values[theta],theta,5);
        require(root_resource_cut.valid,"Weighted resource root cut was not certified");
        require(row_violation(root_resource_cut.row,zero_point)<=1e-6,
                "Weighted resource root cut excludes all-courier solution");
        require(root_resource_cut.dual_savings<=network.separate(solved.values,0,theta).cut_capacity+1e-5,
                "Weighted resource savings relaxation is weaker than monetary max-flow at root");
        report["resource_lp"]["root_separation"]={{"dual_savings_upper_bound",root_resource_cut.dual_savings},
            {"violation",root_resource_cut.violation},{"optimal",root_resource_cut.optimal},
            {"seconds",root_resource_cut.seconds},{"dual_max_violation",root_resource_cut.max_dual_violation_after}};
        report["passed"] = true;
        std::ofstream output(argv[3]); output << report.dump(2) << '\n';
        std::cout << "K=" << d.max_orders << " paths=" << paths.service_paths.size()
                  << " root_LP=" << solved.objective << " path_cost_max_violation=" << max_cost_violation << '\n';
    } catch (const IloException& e) { std::cerr << e << '\n'; return 1; }
      catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
