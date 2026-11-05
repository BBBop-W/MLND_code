// Public complete-domain face oracle and total-cost cut witnesses.
#include "mlnd/design_face_cuts.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fallback_fixture() {
    Input d;d.name="design_face_all_courier";d.model_version="consecutive";
    d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.single_rates=d.head_rates=d.demand;d.pair_rates=filled(2,2,0.0);
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,0);
    d.link_cost=filled(4,4,.01);d.base_cost={7,11};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
Json fallback_hand_tracer(std::array<Json,3>& faces) {
    const auto d=fallback_fixture();auto paths=enumerate_paths(d);const DesignLayout layout(d);
    SolverOptions options;options.seconds=10;options.threads=1;options.seed=43;
    options.pareto=false;options.shared_rho_joint=true;options.queue_perspective=true;
    for(int state=0;state<3;++state) {
        faces[state]=solve_design_face_cg(d,paths,options,0,state);
        const double expected=state==0?40:state==1?47:51;
        require(faces[state].at("bound_valid").get<bool>(),"All-courier face lacks a certified lower bound");
        require(std::abs(faces[state].at("certified_lower_bound").get<double>()-expected)<1e-4,
                "Face free-design total cost disagrees with the all-courier hand calculation");
        require(faces[state].at("scope")=="complete_domain_head_design_face",
                "A partial-design face was mislabeled as a global bound");
    }
    std::vector<double> point(layout.count,0);point[0]=point[1]=.5;
    point[layout.assignment_index(0,0)]=point[layout.assignment_index(0,1)]=.5;
    const auto cut=assemble_design_face_cut(d,0,faces,40,point,39,layout.count);
    require(cut.valid&&cut.violated&&std::abs(cut.violation-1)<1e-4,
            "Three-state total-cost cut lost its fixed cost or face correction");
    require(cut.certificate.at("direct_global_bound_update")==false,
            "A face bound was exposed as a direct global update");
    return {{"face_bounds",{faces[0].at("certified_lower_bound"),faces[1].at("certified_lower_bound"),
                             faces[2].at("certified_lower_bound")}},
        {"violation",cut.violation},{"passed",true}};
}

Json certificate_guards(const std::array<Json,3>& proved) {
    const auto d=fallback_fixture();auto paths=enumerate_paths(d);const DesignLayout layout(d);
    const std::vector<double> closed(layout.count,0);SolverOptions options;
    // Invalid oracle configuration is a programming error, not missing face
    // evidence. It must survive the separator's normal budget fallback.
    options.cg_max_columns=0;bool invalid_propagated=false;
    try {(void)separate_design_face(d,paths,options,0,closed,0,layout.count,40,1);}
    catch(const std::invalid_argument&) {invalid_propagated=true;}
    require(invalid_propagated,"Invalid face oracle configuration was swallowed as a normal fallback");
    options.cg_max_columns=64;
    auto faces=proved;
    faces[1]["certified_lower_bound"]=1e9;
    faces[1]["lower_bound_certificate"]["pricing"]["complete"]=false;
    auto missing=assemble_design_face_cut(d,0,faces,40,closed,40,layout.count);
    require(missing.valid&&missing.fallback&&missing.certificate.at("certified_faces")==2&&
            missing.certificate.at("face_bounds").at(1)==40,
            "An incomplete pricing pass supplied a face bound");
    faces=proved;faces[0]["pricing_complete"]=false;
    const auto retained=assemble_design_face_cut(d,0,faces,0,closed,0,layout.count);
    require(retained.valid&&retained.certificate.at("certified_faces")==3,
            "A retained complete-pass certificate was discarded because the last CG pass was incomplete");
    faces=proved;faces[0].erase("lower_bound_certificate");
    faces[1]["scope"]="fixed_design_recourse";
    faces[2]["certified_lower_bound"]=std::numeric_limits<double>::infinity();
    const auto malformed=assemble_design_face_cut(d,0,faces,40,closed,40,layout.count);
    require(malformed.valid&&malformed.certificate.at("certified_faces")==0&&
            malformed.status=="global_bound_fallback_only"&&!malformed.violated,
            "Malformed or wrong-domain evidence escaped the certified global fallback");
    for(auto& face:faces)face={{"bound_valid",false},{"continuous_objective",1e9},
        {"termination_reason","Infeasible"}};
    const auto unavailable=assemble_design_face_cut(d,0,faces,40,closed,40,layout.count);
    require(unavailable.valid&&unavailable.row.sense=='G'&&
            unavailable.certificate.at("face_bounds")==Json::array({40,40,40})&&!unavailable.violated,
            "An unavailable face became an infeasibility cut or used its restricted primal value");
    const auto no_time=separate_design_face(d,paths,options,0,closed,0,layout.count,40,0);
    require(!no_time.cut.valid&&no_time.faces.empty()&&no_time.diagnostics.at("calls")==0&&
            no_time.diagnostics.at("status")=="budget_exhausted","Zero face budget started an oracle");
    options.seconds=0;
    const auto no_oracle_time=solve_design_face_cg(d,paths,options,0,0);
    require(!no_oracle_time.at("bound_valid").get<bool>()&&no_oracle_time.at("cg_trace").empty(),
            "A zero-budget oracle published an unproved face bound");
    paths.support_complete=false;
    const auto missing_support=separate_design_face(d,paths,options,0,closed,0,layout.count,40,1);
    require(!missing_support.cut.valid&&missing_support.diagnostics.at("calls")==0,
            "Incomplete physical support was sent to a face oracle");
    paths.support_complete=true;paths.complete=false;
    const auto missing_catalogue=separate_design_face(d,paths,options,0,closed,0,layout.count,40,1);
    require(!missing_catalogue.cut.valid&&missing_catalogue.diagnostics.at("calls")==0,
            "An incomplete preprocessed catalogue was accepted");
    paths.complete=true;auto k1=d;k1.max_orders=1;
    const auto unsupported=separate_design_face(k1,paths,options,0,closed,0,layout.count,40,1);
    require(!unsupported.cut.valid&&unsupported.diagnostics.at("calls")==0,
            "Unsupported K1 scope silently changed the face model");
    return {{"invalid_configuration_propagated",invalid_propagated},{"missing_pricing_fallback",true},
        {"retained_complete_pass",true},{"malformed_evidence_fallback",true},
        {"no_infeasibility_fallback",true},{"zero_budget",true},{"unsupported_scope",true},{"passed",true}};
}

std::vector<double> opened(const Input& d,int first,int second) {
    const DesignLayout layout(d);std::vector<double> x(layout.count,0);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h*d.n_levels]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    if(first>=0)x[layout.assignment_index(0,first)]=1;
    if(second>=0)x[layout.assignment_index(1,second)]=1;
    return x;
}
double operating_floor(const TaskSavingsCut& cut,const std::vector<double>& x,int theta) {
    long double rhs=cut.row.rhs;double theta_coefficient=0;
    for(const auto& [j,c]:cut.row.coefficients) {
        if(j==theta)theta_coefficient+=c;
        else rhs-=static_cast<long double>(c)*x.at(j);
    }
    require(theta_coefficient==1,"Face row changed the operating epigraph coefficient");
    return static_cast<double>(rhs);
}
double full_optimum(const LinearModel& model,const SolverOptions& options,const char* role) {
    NativeSolver solver(model,options,role);const auto solved=solver.optimize(20);
    require(solved.optimal&&solved.has_solution&&point_residuals(model,solved.values).at("passed").get<bool>(),
            std::string("Independent complete physical SOCP did not certify a feasible optimum: ")+role);
    return objective_value(model,solved.values);
}
Json physical_crossface_tracer() {
    auto d=fallback_fixture();d.name="design_face_physical_k2";d.base_cost={.5,.7};
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v) {
        d.allowed_links(u,v)=u!=v;if(u==v)d.flight_times(u,v)=0;
    }
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    // Preserve raw-order conservation, including twice-counted AA tasks.
    d.demand={4,7};d.single_rates={.5,4.5};d.pair_rates(0,0)=.5;
    d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;d.head_rates={3,5};
    auto paths=enumerate_paths(d);const DesignLayout layout(d);
    require(!paths.service_paths.empty()&&!paths.empty_paths.empty(),"Physical witness has no complete service/empty domain");
    SolverOptions options;options.seconds=15;options.threads=1;options.seed=43;options.pareto=false;
    options.shared_rho_joint=true;options.queue_perspective=true;
    std::array<Json,3> faces;Json references=Json::array();
    for(int state=0;state<3;++state) {
        faces[state]=solve_design_face_cg(d,paths,options,0,state);
        require(faces[state].at("bound_valid").get<bool>(),"Physical face has no complete-domain certificate");
        auto full=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,
                                    formulation_strengthening(options));
        for(auto& variable:full.matrix.variables)variable.binary=false;
        // Independent reference uses variable bounds on the complete model,
        // while the CG oracle uses rows in a restricted path formulation.
        for(int l=0;l<2;++l) {
            auto& assignment=full.matrix.variables[layout.assignment_index(0,l)];
            assignment.lower=assignment.upper=state==l+1?1:0;
        }
        const double optimum=full_optimum(full.matrix,options,"design_face_full_continuous_reference");
        const double certified=faces[state].at("certified_lower_bound").get<double>();
        require(certified<=optimum+1e-6*(1+std::abs(optimum)),
                "Complete-pricing face bound exceeds independent full-face optimum");
        references.push_back({{"state",state},{"certified_lower",certified},{"full_face_optimum",optimum},
            {"certificate_gap",optimum-certified}});
    }
    auto target=opened(d,-1,-1);target[layout.assignment_index(0,0)]=.5;
    target[layout.assignment_index(0,1)]=.5;
    const auto cut=assemble_design_face_cut(d,0,faces,0,target,0,layout.count);
    require(cut.valid&&cut.violated&&cut.certificate.at("certified_faces")==3,
            "Physical faces did not create a genuine total-cost row");
    struct Point {std::vector<double> design;double operating;};std::vector<Point> points;
    for(int first=-1;first<2;++first)for(int second=-1;second<2;++second) {
        const auto x=opened(d,first,second);
        // Unstrengthened full fixed-design recourse is the independent
        // original-model validity oracle; fixed costs stay in the cut row.
        const auto full=build_formulation(d,paths,&x);
        points.push_back({x,full_optimum(full.matrix,options,"design_face_full_integer_recourse")});
    }
    const std::vector<double> closed(layout.count,0);const auto full=build_formulation(d,paths,&closed);
    points.push_back({closed,full_optimum(full.matrix,options,"design_face_all_closed_recourse")});
    const auto integer_count=points.size();
    for(std::size_t j=1;j<integer_count;++j) {
        auto mixed=points[j];
        for(int k=0;k<layout.count;++k)mixed.design[k]=.37*points[j].design[k]+.63*points[0].design[k];
        mixed.operating=.37*points[j].operating+.63*points[0].operating;points.push_back(std::move(mixed));
    }
    double maximum_violation=0;
    for(const auto& point:points) {
        const double violation=operating_floor(cut,point.design,layout.count)-point.operating;
        maximum_violation=std::max(maximum_violation,violation);
        require(violation<=1e-6*(1+std::abs(point.operating)),
                "Face cut excludes a complete-model integer recourse point or its convex mixture");
    }
    return {{"face_references",references},{"service_paths",paths.service_paths.size()},
        {"empty_paths",paths.empty_paths.size()},{"integer_points",integer_count},
        {"convex_mixtures",points.size()-integer_count},{"cross_checks",points.size()},
        {"maximum_violation",maximum_violation},{"separation_violation",cut.violation},{"passed",true}};
}
}
int main() {try {
    std::array<Json,3> proved;const auto hand=fallback_hand_tracer(proved);
    const auto guards=certificate_guards(proved);const auto physical=physical_crossface_tracer();
    std::cout<<Json({{"fallback_hand",hand},{"guards",guards},{"physical_crossface",physical},{"passed",true}}).dump(2)<<'\n';return 0;}
catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
