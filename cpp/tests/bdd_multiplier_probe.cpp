#include "mlnd/bdd.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition, const std::string& text) {
    if (!condition) throw std::runtime_error(text);
}
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
template<class T> Matrix<T> constant(int rows, int cols, T value) {
    Matrix<T> matrix(rows, cols);
    std::fill(matrix.values.begin(), matrix.values.end(), value);
    return matrix;
}
// Shared physical specification with bdd_probe and the Python solver fixture.
Input fixture(int k, const std::string& policy) {
    Input d;
    d.n_l=1; d.n_h=2; d.n_od=2; d.n_levels=2; d.max_orders=k;
    d.name="bdd_multiplier_physical_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy=policy; d.demand={4,7};
    d.flight_times=constant<double>(3,3,0);
    d.flight_times.values={0,.05,.07,.05,0,.1,.07,.1,0};
    d.rider_mean=constant<double>(2,2,0); d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant<double>(2,2,1); d.terminal_cost=constant<double>(2,2,.5);
    d.receipt_mean=constant<double>(2,2,.08); d.removal_mean=constant<double>(2,2,.03);
    d.ground_cost={5,5}; d.deadline={1,1}; d.base_cost={.5};
    d.slot_counts=constant<int>(2,2,0); d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant<double>(2,2,0); d.slot_cost.values={.1,.2,.1,.2};
    d.slot_epsilon={.05,.05}; d.link_cost=constant<double>(3,3,0);
    d.kappa=constant<double>(2,2,0); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.allowed_links=constant<int>(3,3,1);
    for (int i=0;i<3;++i) d.allowed_links(i,i)=0;
    d.occupancy_limit=constant<double>(2,2,0);
    d.occupancy_limit.values={6.1690072893953225,14.072024748341311,6.1690072893953225,14.072024748341311};
    return d;
}
std::vector<double> design(const Input& d, int assignments, int level) {
    DesignLayout layout(d);
    std::vector<double> x(layout.count,0);
    for (int i=0;i<d.n_nodes();++i) x[i]=1;
    for (int i=0;i<d.n_h;++i) x[layout.capacity+i*d.n_levels+level]=1;
    for (int r=0;r<d.n_od;++r) x[layout.assignment+r]=(assignments&(1<<r))?1:0;
    for (int i=0;i<d.n_nodes();++i) for (int j=0;j<d.n_nodes();++j)
        x[layout.airline+i*d.n_nodes()+j]=d.allowed_links(i,j)?1:0;
    return x;
}
Json run() {
    Json report={{"schema","mlnd-bdd-arbitrary-multiplier-regression/1"},{"cases",Json::array()}};
    int cross_checks=0, plane_checks=0;
    double worst=0, worst_plane=0;
    for (const auto& policy : {std::string("aggregate"),std::string("prefix")}) for (int k : {1,2}) {
        const auto data=fixture(k,policy);
        const auto paths=enumerate_paths(data);
        SolverOptions options; options.threads=1; options.gap=0;
        const auto original=build_formulation(data,paths);
        NativeSolver full(original.matrix,options,"bdd_multiplier_identity_reference");
        const auto optimum=full.optimize(10);
        require(optimum.optimal && optimum.has_solution &&
            point_residuals(original.matrix,optimum.values).at("passed").get<bool>(),"Full reference invalid");
        BddFullDomain oracle(data,paths,options);
        std::vector<std::vector<double>> multipliers(4,std::vector<double>(original.layout.count,0));
        for (int j=0;j<original.layout.count;++j) {
            multipliers[0][j]=-original.opening_costs[j];
            multipliers[1][j]=(j%3-1)*.137;
            multipliers[2][j]=(j%2?1:-1)*(1+.071*j);
        }
        Json results=Json::array();
        for (const auto mode : {BddIntegrality::Continuous,BddIntegrality::FacilityCapacity,BddIntegrality::AllDesign})
            for (std::size_t b=0;b<multipliers.size();++b) {
                auto value=oracle.evaluate_multiplier(multipliers[b],mode,10);
                value["multiplier_index"]=b;
                require(value.at("valid_global_lower_bound").get<bool>() && value.at("solver_optimal").get<bool>() &&
                    value.at("primal_validated").get<bool>(),"Arbitrary multiplier oracle failed");
                require(value.at("support_affine_plane").is_object(),"Validated point lacks support plane");
                const auto z=value.at("inner_design").get<std::vector<double>>();
                const double q=value.at("inner_operating_cost").get<double>();
                require(value.at("safe_support_intercept").get<double>()>q &&
                    value.at("support_affine_plane").at("inner_integrality")==value.at("mode") &&
                    value.at("support_affine_plane").at("valid_for_all_design_inner_value").get<bool>()==
                        (mode==BddIntegrality::AllDesign),"Support margin/integrality domain incorrect");
                require(std::abs(q-dot(multipliers[b],z)-value.at("inner_upper_bound").get<double>())<2e-5,
                    "Operating/Lagrangian objective identity failed");
                require(value.at("inner_lower_bound").get<double>()<=value.at("inner_upper_bound").get<double>()+2e-5,
                    "Safe lower bound exceeds validated primal");
                if(mode==BddIntegrality::AllDesign) {
                    require(!value.at("validated_original_integer_upper_bound").is_null(),"Integer primal lacks original UB");
                    const double original_ub=value.at("validated_original_integer_upper_bound").get<double>();
                    require(std::abs(original_ub-q-dot(original.opening_costs,z))<2e-5 &&
                        original_ub>=optimum.objective-2e-5,"Invalid original integer UB");
                    if(b==0) require(std::abs(value.at("inner_upper_bound").get<double>()-optimum.objective)<2e-5,
                        "beta=-fixed_cost did not reproduce the original full model");
                } else require(value.at("validated_original_integer_upper_bound").is_null(),"Relaxed primal leaked integer UB");
                results.push_back(std::move(value));
            }
        // A feasible point defines an upper plane for the inner value at any
        // other multiplier in the SAME integrality domain, even if not optimal.
        for (const auto& source:results) for (const auto& target:results) {
            if(source.at("mode")!=target.at("mode")) continue;
            const auto& plane=source.at("support_affine_plane");
            const double upper=plane.at("intercept").get<double>()+dot(
                plane.at("coefficients").get<std::vector<double>>(),
                target.at("coefficients").get<std::vector<double>>());
            const double violation=target.at("inner_lower_bound").get<double>()-upper;
            worst_plane=std::max(worst_plane,violation); ++plane_checks;
            require(violation<=2e-5,"Support plane has incorrect sign or scope");
        }
        std::vector<std::vector<double>> designs;
        for(int assignments=0;assignments<4;++assignments) for(int level=0;level<2;++level)
            designs.push_back(design(data,assignments,level));
        auto closed=design(data,3,0); closed[original.layout.airline+1]=0; designs.push_back(closed);
        designs.emplace_back(original.layout.count,0);
        for(const auto& x:designs) {
            const auto fixed=build_formulation(data,paths,&x);
            NativeSolver reference(fixed.matrix,options,"bdd_multiplier_cross_design");
            const auto solved=reference.optimize(10);
            require(solved.optimal && solved.has_solution &&
                point_residuals(fixed.matrix,solved.values).at("passed").get<bool>(),"Fixed recourse invalid");
            for(const auto& value:results) {
                const double cut=value.at("safe_intercept").get<double>()+
                    dot(value.at("coefficients").get<std::vector<double>>(),x);
                const double violation=cut-solved.objective;
                worst=std::max(worst,violation); ++cross_checks;
                require(violation<=3e-6*(1+std::abs(solved.objective)),"Arbitrary multiplier cut excludes a feasible design");
            }
        }
        const auto zero=oracle.evaluate_multiplier(multipliers[0],BddIntegrality::AllDesign,0);
        require(!zero.at("valid_global_lower_bound").get<bool>() && !zero.at("primal_validated").get<bool>() &&
            zero.at("inner_lower_bound").is_null() && zero.at("safe_cut").is_null(),"Zero-budget call leaked stale bound");
        bool invalid_size=false,invalid_coefficient=false,invalid_budget=false;
        try { oracle.evaluate_multiplier({},BddIntegrality::AllDesign,1); }
        catch(const std::invalid_argument&) { invalid_size=true; }
        auto invalid=multipliers[0]; invalid[0]=std::numeric_limits<double>::infinity();
        try { oracle.evaluate_multiplier(invalid,BddIntegrality::AllDesign,1); }
        catch(const std::invalid_argument&) { invalid_coefficient=true; }
        try { oracle.evaluate_multiplier(multipliers[0],BddIntegrality::AllDesign,-1); }
        catch(const std::invalid_argument&) { invalid_budget=true; }
        require(invalid_size && invalid_coefficient && invalid_budget,"Invalid multiplier inputs were accepted");
        report["cases"].push_back({{"K",k},{"policy",policy},{"full_objective",optimum.objective},
            {"evaluations",results},{"zero_budget_bound_rejected",true},{"invalid_inputs_rejected",true}});
    }
    report["cross_design_cut_checks"]=cross_checks; report["support_affine_plane_checks"]=plane_checks;
    report["maximum_cross_design_violation"]=worst; report["maximum_support_plane_violation"]=worst_plane;
    report["passed"]=true;
    return report;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Usage: mlnd_bdd_multiplier_probe OUTPUT_JSON");
        const auto output=std::filesystem::absolute(argv[1]);
        if(std::filesystem::exists(output)) throw std::invalid_argument("Refusing to overwrite existing regression report");
        const auto result=run(); mlnd::atomic_json_file(output,result);
        std::cout<<"BDD arbitrary multiplier checks="<<result.at("cross_design_cut_checks")
                 <<" support planes="<<result.at("support_affine_plane_checks")<<" passed\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n'; return 1;}
}
