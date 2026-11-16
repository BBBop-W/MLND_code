// Frozen-reference differential checks for the unified point assessment and
// direct solution-field construction. Reference functions were copied verbatim
// from /tmp/mlnd-performance-20261105-before/cpp/src/{linear_model,benders,output}.cpp
// on 2026-11-05 (only qualification and the output packaging boundary changed).
// This probe never constructs NativeSolver, optimizes, or reads production data.
#include "mlnd/formulation.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace mlnd;
namespace frozen_reference {
Json point_residuals(const LinearModel& model, const std::vector<double>& x) {
    if (x.size() != model.variables.size()) throw std::invalid_argument("Solution dimension mismatch");
    double bound = 0, linear = 0, scaled = 0, cone = 0, integer = 0;
    Json worst_linear=nullptr,worst_scaled=nullptr;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error("Nonfinite solution value");
        bound = std::max({bound, model.variables[i].lower - x[i], x[i] - model.variables[i].upper});
        if (model.variables[i].binary) integer = std::max(integer, std::abs(x[i] - std::round(x[i])));
    }
    for (const auto& row : model.rows) {
        long double activity = 0, magnitude = std::abs(row.rhs);
        for (auto [i, value] : row.coefficients) {
            activity += static_cast<long double>(value) * x[i];
            magnitude += std::abs(static_cast<long double>(value) * x[i]);
        }
        const double diff = static_cast<double>(activity - row.rhs);
        const double violation = row.sense == 'E' ? std::abs(diff) :
                                 std::max(0.0, row.sense == 'L' ? diff : -diff);
        const double scaled_violation=violation/static_cast<double>(1+magnitude);
        if(violation>linear || scaled_violation>scaled) {
            double minimum=INFINITY,maximum=0;
            for(auto [j,a]:row.coefficients) if(a!=0) {
                minimum=std::min(minimum,std::abs(a));maximum=std::max(maximum,std::abs(a));
            }
            Json detail={{"name",row.name},{"sense",std::string(1,row.sense)},
                {"rhs",row.rhs},{"activity",static_cast<double>(activity)},
                {"magnitude",static_cast<double>(magnitude)},{"absolute_violation",violation},
                {"scaled_violation",scaled_violation},{"nonzeros",row.coefficients.size()},
                {"minimum_absolute_coefficient",std::isfinite(minimum)?Json(minimum):Json(nullptr)},
                {"maximum_absolute_coefficient",maximum}};
            if(violation>linear)worst_linear=detail;
            if(scaled_violation>scaled)worst_scaled=detail;
        }
        linear = std::max(linear, violation);
        scaled = std::max(scaled, scaled_violation);
    }
    for (const auto& c : model.cones) {
        const double lhs = x[c.one] * x[c.one], rhs = c.factor * x[c.up] * x[c.down];
        cone = std::max(cone, (lhs - rhs) / (1 + std::abs(lhs) + std::abs(rhs)));
    }
    return {{"max_bound_violation", bound}, {"max_linear_violation", linear},
            {"max_scaled_linear_violation", scaled}, {"max_scaled_cone_violation", cone},
            {"max_integrality_violation", integer},
            {"worst_linear_constraint",worst_linear},{"worst_scaled_linear_constraint",worst_scaled},
            {"passed", bound <= 2e-6 && scaled <= 2e-7 && cone <= 2e-7 && integer <= 1e-7}};
}
bool feasible_point(const LinearModel& model, const std::vector<double>& values) {
    if (values.size() != model.variables.size()) return false;
    for (std::size_t j = 0; j < values.size(); ++j) {
        const auto& v = model.variables[j];
        if (!std::isfinite(values[j])) return false;
        double tol = 2e-7 * (1 + std::abs(values[j]));
        if (values[j] < v.lower - tol || values[j] > v.upper + tol) return false;
        if (v.binary && std::abs(values[j] - std::round(values[j])) > 1e-7) return false;
    }
    for (const auto& row : model.rows) {
        double lhs = 0, scale = 1 + std::abs(row.rhs);
        for (auto [j, coefficient] : row.coefficients) {
            lhs += coefficient * values[j];
            scale += std::abs(coefficient * values[j]);
        }
        double residual = lhs - row.rhs;
        if (row.sense == 'L') residual = std::max(0.0, residual);
        else if (row.sense == 'G') residual = std::max(0.0, -residual);
        else residual = std::abs(residual);
        if (residual > 2e-7 * scale) return false;
    }
    for (const auto& cone : model.cones) {
        double lhs = values[cone.one] * values[cone.one];
        double rhs = cone.factor * values[cone.up] * values[cone.down];
        if (lhs > rhs + 2e-7 * (1 + std::abs(lhs) + std::abs(rhs))) return false;
    }
    return true;
}

Json feasibility_diagnostics(const LinearModel& model,const std::vector<double>& values) {
    if (values.size()!=model.variables.size()) return {{"passed",false},{"reason","dimension"}};
    if (!std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);}))
        return {{"passed",false},{"reason","nonfinite"}};
    auto report=frozen_reference::point_residuals(model,values);
    report["passed"]=feasible_point(model,values);
    double worst=0; Json detail=nullptr;
    auto check=[&](double residual,double tolerance,const std::string& kind,const std::string& name) {
        if (residual/tolerance>worst) {
            worst=residual/tolerance;
            detail={{"kind",kind},{"name",name},{"violation",residual},{"tolerance",tolerance}};
        }
    };
    for (std::size_t j=0;j<values.size();++j) {
        const auto& v=model.variables[j];
        check(std::max({0.0,v.lower-values[j],values[j]-v.upper}),2e-7*(1+std::abs(values[j])),"bound",v.name);
        if (v.binary) check(std::abs(values[j]-std::round(values[j])),1e-7,"integrality",v.name);
    }
    for (const auto& row:model.rows) {
        double lhs=0,scale=1+std::abs(row.rhs);
        for (auto [j,c]:row.coefficients) { lhs+=c*values[j]; scale+=std::abs(c*values[j]); }
        double residual=lhs-row.rhs;
        residual=row.sense=='L'?std::max(0.0,residual):row.sense=='G'?std::max(0.0,-residual):std::abs(residual);
        check(residual,2e-7*scale,"linear",row.name);
    }
    for (const auto& cone:model.cones) {
        const double lhs=values[cone.one]*values[cone.one],rhs=cone.factor*values[cone.up]*values[cone.down];
        check(std::max(0.0,lhs-rhs),2e-7*(1+std::abs(lhs)+std::abs(rhs)),"cone",model.variables[cone.one].name);
    }
    report["worst_tolerance_ratio"]=worst;
    report["worst_constraint"]=detail;
    return report;
}

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
    for (int r = 0; r < data.assignment_entities(); ++r) {
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

Json solution_fields(const Input& data, const PathCollection& paths,
                   const Formulation& formulation, const std::vector<double>& values) {
    Json solution = empty_solution(data, paths);
    const auto& layout = formulation.layout;
    for (int u = 0; u < data.n_nodes(); ++u)
        solution["facility"][u] = std::round(values.at(layout.facility + u));
    for (int h = 0; h < data.n_h; ++h)
        for (int level = 0; level < data.n_levels; ++level)
            solution["capacity_level"][h][level] = std::round(values.at(layout.capacity + h * data.n_levels + level));
    for (int r = 0; r < data.assignment_entities(); ++r)
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
    solution["head"] = recover_task_heads(formulation,paths,values);
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
    solution["delivery_time_scope"]="courier_home";
    if(data.preprocessed_demand) {
        solution["task_indexing"]="singles_then_row_major_pairs";
        Json tasks=Json::array();
        for(int q=0;q<data.task_count();++q)tasks.push_back({{"task",q},
            {"first_od",data.task_first(q)},{"second_od",data.task_second(q)},{"rate",data.task_rate(q)}});
        solution["task_table"]=tasks;
    }
    return solution;
}


} // namespace frozen_reference
namespace {
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
template<class Function> Json observed(Function function) {
    try {return {{"value",function()}};}
    catch(const std::invalid_argument& e) {return {{"exception","invalid_argument"},{"message",e.what()}};}
    catch(const std::runtime_error& e) {return {{"exception","runtime_error"},{"message",e.what()}};}
}
void same(const Json& actual,const Json& expected,const std::string& label) {
    // dump also provides a stable comparison for diagnostic nonfinite numbers
    // (the old report serializes them as null), and preserves signed zero.
    require(actual.dump()==expected.dump(),label+" differs from the frozen reference");
}
std::size_t cases=0;
void check(const LinearModel& model,const std::vector<double>& point,const std::string& label) {
    const auto assessment=assess_point(model,point);
    require(assessment.feasible==frozen_reference::feasible_point(model,point),label+" gate changed");
    const auto expected=observed([&]{return frozen_reference::point_residuals(model,point);});
    same(observed([&]{return assessment.residuals();}),expected,label+" residuals");
    same(observed([&]{return point_residuals(model,point);}),expected,label+" public residuals");
    same(assessment.diagnostics(),frozen_reference::feasibility_diagnostics(model,point),label+" diagnostics");
    same(assessment.diagnostics(),assessment.diagnostics(),label+" repeated report");
    ++cases;
}
void boundary_cases() {
    LinearModel bounds;bounds.variable("zero",0,0);
    for(double value:{0.0,-0.0,1e-8,2e-7,std::nextafter(2e-7,INFINITY),
        -2e-7,std::nextafter(-2e-7,-INFINITY),1e-6,2e-6,std::nextafter(2e-6,INFINITY)})
        check(bounds,{value},"small bound");
    const auto separated=assess_point(bounds,{1e-6});
    require(!separated.feasible && separated.residuals().at("passed").get<bool>(),
            "The two original bound policies were accidentally unified");
    bounds.variables[0].upper=1e6;
    check(bounds,{1e6+.1},"large relative bound");
    const auto large=assess_point(bounds,{1e6+.1});
    require(large.feasible && !large.residuals().at("passed").get<bool>(),
            "The absolute residual threshold was replaced by a relative threshold");
    bounds.variables[0].binary=true;bounds.variables[0].upper=2;
    for(double delta:{0.0,1e-7,std::nextafter(1e-7,INFINITY),-1e-7})
        check(bounds,{1+delta},"integrality boundary");
    LinearModel rows;rows.variable("x",-1e20,1e20);
    rows.row("equal",'E',1,{{0,1}});rows.row("less",'L',1,{{0,1}});
    rows.row("greater",'G',1,{{0,1}});rows.row("empty",'E',1,{});
    rows.row("zero_coeff",'E',1,{{0,0}});
    for(double value:{1.0,1+2e-7,1+6e-7,std::nextafter(1+6e-7,INFINITY),-1.0})
        check(rows,{value},"linear threshold");
    LinearModel cancellation;
    for(int i=0;i<3;++i)cancellation.variable("c"+std::to_string(i),-10,10);
    cancellation.row("cancellation",'E',1,{{0,1e16},{1,1},{2,-1e16}});
    cancellation.row("opposite",'G',-1,{{0,-1e16},{1,-1},{2,1e16}});
    check(cancellation,{1,1,1},"coefficient cancellation");
    check(cancellation,{1,std::nextafter(1.0,INFINITY),1},"cancellation nextafter");
    LinearModel cone;
    cone.variable("up",0,10);cone.variable("down",0,10);cone.variable("one",0,10);
    cone.cones.push_back({0,1,2,1});
    for(double one:{0.0,1.0,std::sqrt(1+6e-7),std::nextafter(std::sqrt(1+6e-7),INFINITY),1.01})
        check(cone,{1,1,one},"cone threshold");
    check(cone,{0,0,0},"closed zero cone");
    check(cone,{1,0,0},"zero cone coordinate");
    for(double value:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()})
        check(cone,{1,value,1},"nonfinite point");
    check(cone,{},"wrong dimension");check(cone,{1,1,1,1},"extra dimension");
    LinearModel empty;check(empty,{},"empty model");
    LinearModel revision;revision.variable("x",0,2);revision.row("revision",'E',0,{{0,1}});
    const auto old=assess_point(revision,{0});revision.rows[0].rhs=1;
    check(revision,{0},"changed RHS");
    require(old.feasible && !assess_point(revision,{0}).feasible,"Point assessment did not preserve its original value");
}
void randomized_cases() {
    std::mt19937_64 random(20261105);
    std::uniform_real_distribution<double> value(-2,2);
    const std::array<double,7> scales={0,1e-12,1e-4,1,1e4,1e12,1e16};
    for(int sample=0;sample<160;++sample) {
        LinearModel model;std::vector<double> point(6);
        for(int j=0;j<6;++j) {
            model.variable("v"+std::to_string(j),-1,1,0,j%3==0);
            point[j]=value(random);
        }
        for(int r=0;r<12;++r) {
            Coefficients coefficients;
            for(int j=0;j<6;++j)coefficients.emplace_back(j,value(random)*scales[random()%scales.size()]);
            model.row("r"+std::to_string(r),"ELG"[r%3],value(random)*scales[random()%scales.size()],coefficients);
        }
        model.cones.push_back({0,1,2,1});model.cones.push_back({3,4,5,.5});
        check(model,point,"random mixed-scale "+std::to_string(sample));
    }
}
void repair_case() {
    Input data;data.n_l=1;Formulation f(data);
    const int rho=f.matrix.variable("rho",0,1),idle=f.matrix.variable("idle",0,10);
    f.rho={rho};f.idle={idle};f.fleet=f.matrix.variable("fleet",0,10,1);
    const int up=f.matrix.variable("up",0,11),down=f.matrix.variable("down",0,1),one=f.matrix.variable("one",1,1);
    f.matrix.row("up_link",'E',1,{{up,1},{idle,-1}});
    f.matrix.row("down_link",'E',1,{{down,1},{rho,1}});
    f.fleet_row=f.matrix.row("fleet_use",'L',0,{{idle,1},{f.fleet,-1}});
    f.matrix.cones.push_back({up,down,one,1});
    std::vector<double> point={.5,.999,.999,1.999,.5,1};
    check(f.matrix,point,"before inventory repair");
    require(!assess_point(f.matrix,point).feasible,"Repair fixture starts feasible");
    const auto report=repair_inventory_primal(f,point);
    require(report.at("applied").get<bool>() && report.at("passed").get<bool>(),"Inventory repair failed");
    check(f.matrix,point,"after inventory repair");
    require(assess_point(f.matrix,point).feasible,"Repaired point failed old gate");
}
template<class T> Matrix<T> filled(int rows,int columns,T value) {
    Matrix<T> out(rows,columns);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input output_fixture(int k,bool preprocessed) {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=k;
    d.name="point_assessment_output";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.demand={4,7};d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);d.link_cost=filled(4,4,.4);
    for(int i=0;i<4;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);d.terminal_cost=filled(2,2,.5);
    d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.preprocessed_demand=preprocessed;d.single_rates={4,7};d.head_rates=d.single_rates;
    d.pair_rates=filled(2,2,0.0);d.common_origin_support=filled(2,2,1);return d;
}
std::size_t output_cases() {
    std::size_t count=0;
    for(int k:{1,2})for(bool preprocessed:{false,true})for(bool compact:{false,true}) {
        auto d=output_fixture(k,preprocessed);auto paths=enumerate_paths(d);
        FormulationStrengthening options;options.compact_task_heads=compact;
        const auto f=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,options);
        std::vector<double> point(f.matrix.variables.size(),0);
        for(std::size_t j=0;j<point.size();++j)point[j]=f.matrix.variables[j].lower;
        for(int j=0;j<f.layout.count;++j)point[j]=(j%3==0)?.499999999:.500000001;
        for(int id:f.service_indices)point[f.flow[id]]=id%5==0?-.000000000001:id%5==1?.125:0;
        for(int id:f.empty_indices)point[f.empty_flow[id]]=id%2==0?-.000000000001:.25;
        for(std::size_t j=0;j<f.head.size();++j)if(f.head[j]>=0)point[f.head[j]]=.25;
        for(int id:f.eta)if(id>=0)point[id]=.02;
        const auto actual=solution_json(d,paths,f,point);
        const auto expected=frozen_reference::solution_fields(d,paths,f,point);
        same(actual.at("_solution"),expected,"direct output fields");
        ++count;
        // An older RMP can refer to a strict registered prefix. Its public
        // output must still pad newly registered columns with exact zeros.
        auto prefix=f;
        if(!prefix.flow.empty())prefix.flow.pop_back();
        if(!prefix.empty_flow.empty())prefix.empty_flow.pop_back();
        prefix.service_indices.erase(std::remove_if(prefix.service_indices.begin(),prefix.service_indices.end(),
            [&](int id){return static_cast<std::size_t>(id)>=prefix.flow.size();}),prefix.service_indices.end());
        same(solution_json(d,paths,prefix,point).at("_solution"),
             frozen_reference::solution_fields(d,paths,prefix,point),"prefix padding");
        ++count;
    }
    return count;
}
}
int main() {
    try {
        boundary_cases();randomized_cases();repair_case();
        const auto outputs=output_cases();
        std::cout<<Json{{"passed",true},{"assessment_cases",cases},{"output_cases",outputs},
            {"optimizer_calls",0},{"frozen_reference","mlnd-performance-20261105-before"}}.dump(2)<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
