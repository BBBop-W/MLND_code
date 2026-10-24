#include "mlnd/bdd_search.hpp"
#include "mlnd/strengthening.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mlnd {
namespace {
double dot(const std::vector<double>& a,const std::vector<double>& b) {
    if(a.size()!=b.size()) throw std::invalid_argument("BDD search dimension mismatch");
    long double result=0;
    for(std::size_t j=0;j<a.size();++j) result+=static_cast<long double>(a[j])*b[j];
    return static_cast<double>(result);
}
double margin(double value) {return 1e-7*(1+std::abs(value));}
bool finite_vector(const std::vector<double>& x) {
    return std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);});
}
struct Plane {std::vector<double> design;double cost;};
RowRecord make_cut(int theta,double intercept,const std::vector<double>& beta,const std::string& name) {
    Coefficients terms{{theta,1}};
    for(std::size_t j=0;j<beta.size();++j) if(beta[j]) terms.emplace_back(static_cast<int>(j),-beta[j]);
    return {name,'G',intercept,std::move(terms)};
}
}

Json search_bdd_root_bounds(const Input& data,const PathCollection& paths,BddFullDomain& oracle,
    const std::vector<BddAnchor>& anchors,double floor,const SolverOptions& supplied,const BddSearchOptions& config) {
    if(config.method!="search"&&config.method!="fixed") throw std::invalid_argument("Unknown BDD search method");
    if(anchors.empty()||!std::isfinite(floor)||!std::isfinite(config.box_radius)||config.box_radius<=0||
       !std::isfinite(config.inner_seconds)||config.inner_seconds<=0||!std::isfinite(config.master_seconds)||
       config.master_seconds<=0||!std::isfinite(config.started)||!std::isfinite(config.deadline)||
       config.multipliers_per_root<1||config.max_evaluations<1)
        throw std::invalid_argument("Invalid BDD root search configuration");
    const double entry=wall_time();
    auto remaining=[&](){return std::max(0.0,config.deadline-wall_time());};
    Json report={{"method",config.method},{"algorithm","boxed_Kelley_integer_Lagrangian_root_diagnostic"},
        {"fixed_control_scope","fixed_input_multiplier_pool_not_dynamic_BDD1"},
        {"published_quadratic_proximal_BDD_reproduced",false},{"unrestricted_dual_convergence_claimed",false},
        {"inner_seconds",config.inner_seconds},{"master_seconds",config.master_seconds},
        {"box_radius",config.box_radius},{"multipliers_per_root",config.multipliers_per_root},
        {"max_evaluations",config.max_evaluations},{"certified_continuous_total_cost_floor",floor},
        {"bound_scope","original_integer_problem"},{"continuous_primal_is_integer_upper_bound",false},
        {"tight_airline_bounds",supplied.tight_airline_bounds},{"conditional_demand",false},
        {"baseline",nullptr},{"latest_root",nullptr},{"root_curve",Json::array()},
        {"evaluations",Json::array()},{"cuts",Json::array()},{"primal_planes",Json::array()},
        {"multiplier_steps",Json::array()},{"best_lower_bound",floor},
        {"validated_original_integer_upper_bound",nullptr},{"status","running"},
        {"times",{{"master_record_build",0.0},{"master_solver_build",0.0},{"master_optimize",0.0},
            {"master_validation",0.0},{"oracle_wall",0.0},{"multiplier_lp_build",0.0},
            {"multiplier_lp_optimize",0.0},{"multiplier_lp_validation",0.0},{"cut_insert",0.0},{"checkpoint_io",0.0}}}};
    auto add_time=[&](const std::string& key,double value){report["times"][key]=report["times"][key].get<double>()+value;};
    auto checkpoint=[&](){report["search_wall_seconds"]=wall_time()-entry;
        report["elapsed_seconds"]=wall_time()-config.started;
        if(config.checkpoint){const double save_started=wall_time();config.checkpoint(report);add_time("checkpoint_io",wall_time()-save_started);}};
    try {
        double phase=wall_time();
        auto master=design_model(data,false,formulation_strengthening(supplied));
        const int n=master.layout.count,theta=master.matrix.variable("theta",0,IloInfinity,1);
        report["static_bound_metadata"]=add_master_strengthening(data,paths,master,theta,true);
        Coefficients floor_terms{{theta,1}};
        for(int j:master.design_vars) if(master.matrix.variables[j].cost)
            floor_terms.emplace_back(j,master.matrix.variables[j].cost);
        master.matrix.row("certified_continuous_total_cost_floor",'G',floor,std::move(floor_terms));
        for(std::size_t i=0;i<anchors.size();++i) {
            const auto& anchor=anchors[i];
            if(!anchor.pricing_complete||anchor.coefficients.size()!=static_cast<std::size_t>(n)||
               !finite_vector(anchor.coefficients)||!std::isfinite(anchor.intercept))
                throw std::invalid_argument("Invalid ordinary anchor in BDD root search");
            master.matrix.rows.push_back(make_cut(theta,anchor.intercept,anchor.coefficients,"ordinary_"+std::to_string(i)));
        }
        add_time("master_record_build",wall_time()-phase);
        report["master_size"]={{"strategic_variables",n},{"operating_variables",1},
            {"variables",master.matrix.variables.size()},{"initial_rows",master.matrix.rows.size()},
            {"cones",master.matrix.cones.size()}};
        SolverOptions options=supplied;options.gap=0;options.threads=1;
        phase=wall_time();NativeSolver root(master.matrix,options,"bdd_search_root");
        add_time("master_solver_build",wall_time()-phase);
        auto solve_root=[&](int batch) {
            Json record={{"batch",batch},{"certified",false},{"lower_bound",nullptr},{"objective",nullptr},
                {"design",nullptr},{"theta",nullptr},{"cuts",report["cuts"].size()}};
            const double started=wall_time();
            const auto value=root.optimize(std::min(config.master_seconds,remaining()));
            add_time("master_optimize",wall_time()-started);
            const double validation=wall_time();
            record["status"]=value.status;record["solver_status_code"]=value.cplex_status;
            if(value.has_solution) {
                const auto residuals=point_residuals(master.matrix,value.values);
                const double objective=objective_value(master.matrix,value.values);
                const bool certified=value.optimal&&value.cplex_status==IloCplex::Optimal&&
                    residuals.at("passed").get<bool>()&&std::isfinite(value.bound)&&
                    std::abs(objective-value.objective)<=2e-7*(1+std::abs(objective));
                record["residuals"]=residuals;record["certified"]=certified;
                if(certified) {
                    const double safe=value.bound-1e-8*(1+std::abs(value.bound));
                    record["lower_bound"]=safe;record["objective"]=objective;
                    record["design"]=std::vector<double>(value.values.begin(),value.values.begin()+n);
                    record["theta"]=value.values.at(theta);
                    report["best_lower_bound"]=std::max(report["best_lower_bound"].get<double>(),safe);
                    if(report["baseline"].is_object()&&report["baseline"].value("certified",false)) {
                        record["root_objective_change"]=objective-report["baseline"]["objective"].get<double>();
                        record["root_gain_lower_bound"]=std::max(0.0,safe-report["baseline"]["objective"].get<double>());
                    }
                }
            }
            add_time("master_validation",wall_time()-validation);
            record["elapsed_seconds"]=wall_time()-config.started;
            report["latest_root"]=record;report["root_curve"].push_back(record);
            return record;
        };
        report["baseline"]=solve_root(0);checkpoint();
        if(!report["baseline"].value("certified",false)) {
            report["status"]="initial_root_not_certified";checkpoint();return report;
        }
        std::vector<Plane> planes;
        const double courier=std::inner_product(data.demand.begin(),data.demand.end(),data.ground_cost.begin(),0.0);
        planes.push_back({std::vector<double>(n,0),std::nextafter(courier+margin(courier),std::numeric_limits<double>::infinity())});
        report["primal_planes"].push_back({{"source","analytic_all_closed_all_courier"},
            {"design",planes.back().design},{"operating_cost",courier},{"safe_upper_operating_cost",planes.back().cost}});
        report["validated_original_integer_upper_bound"]=courier+margin(courier);
        std::vector<bool> fixed_done(anchors.size(),false);
        std::vector<std::vector<double>> beta_pool;
        std::vector<double> beta_intercepts;
        for(const auto& a:anchors){beta_pool.push_back(a.coefficients);beta_intercepts.push_back(a.intercept);}
        int evaluation=0,batch=0;
        std::size_t fixed_cursor=0;
        bool stop=false;
        while(!stop&&remaining()>config.master_seconds+.05&&evaluation<config.max_evaluations) {
            ++batch;
            const auto target=report["latest_root"]["design"].get<std::vector<double>>();
            const double target_theta=report["latest_root"]["theta"].get<double>();
            std::size_t seed=0;
            for(std::size_t i=1;i<beta_pool.size();++i)
                if(beta_intercepts[i]+dot(beta_pool[i],target)>beta_intercepts[seed]+dot(beta_pool[seed],target)) seed=i;
            const auto center=beta_pool[seed];
            auto beta=center;
            double target_best=-std::numeric_limits<double>::infinity();
            bool batch_added=false;
            for(int inner=0;inner<config.multipliers_per_root&&remaining()>config.master_seconds+.05&&evaluation<config.max_evaluations;++inner) {
                if(config.method=="fixed") {
                    if(std::all_of(fixed_done.begin(),fixed_done.end(),[](bool v){return v;})){stop=true;break;}
                    for(std::size_t j=0;j<anchors.size();++j) {
                        const std::size_t candidate=(fixed_cursor+j)%anchors.size();
                        if(!fixed_done[candidate]){seed=candidate;fixed_cursor=(candidate+1)%anchors.size();break;}
                    }
                    beta=anchors[seed].coefficients;
                }
                const double allowed=std::min(config.inner_seconds,std::max(0.0,remaining()-config.master_seconds));
                phase=wall_time();
                Json value=oracle.evaluate_multiplier(beta,BddIntegrality::AllDesign,allowed);
                add_time("oracle_wall",wall_time()-phase);
                value["evaluation"]=evaluation;value["batch"]=batch;value["inner_iteration"]=inner;
                value["target_design"]=target;value["target_theta"]=target_theta;
                value["elapsed_seconds"]=wall_time()-config.started;
                if(config.method=="fixed"&&value.value("solver_optimal",false)&&value.value("valid_global_lower_bound",false))
                    fixed_done[seed]=true;
                if(value.value("valid_global_lower_bound",false)) {
                    const double intercept=value.at("inner_lower_bound").get<double>();
                    const double at_target=intercept+dot(beta,target);
                    target_best=std::max(target_best,at_target);
                    value["certified_target_value_lower_bound"]=at_target;
                    bool duplicate=false;
                    for(std::size_t i=0;i<beta_pool.size();++i)
                        if(beta_pool[i]==beta&&beta_intercepts[i]>=intercept-1e-9){duplicate=true;break;}
                    if(!duplicate) {
                        const auto row=make_cut(theta,intercept,beta,"bdd_search_"+std::to_string(evaluation));
                        phase=wall_time();master.matrix.rows.push_back(row);root.add_row(row);
                        add_time("cut_insert",wall_time()-phase);batch_added=true;
                        beta_pool.push_back(beta);beta_intercepts.push_back(intercept);
                        report["cuts"].push_back({{"evaluation",evaluation},{"intercept",intercept},{"coefficients",beta},
                            {"target_violation",at_target-target_theta},{"full_path_domain",true},
                            {"bound_source","certified_integer_lagrangian_lower_bound"}});
                    }
                    value["cut_inserted"]=!duplicate;
                }
                const bool valid_plane=value.value("primal_validated",false)&&value.contains("inner_design")&&
                    value["inner_design"].is_array()&&value.contains("inner_operating_cost")&&value["inner_operating_cost"].is_number()&&
                    value.contains("safe_support_intercept")&&value["safe_support_intercept"].is_number()&&
                    value.contains("support_affine_plane")&&value["support_affine_plane"].is_object()&&
                    value["support_affine_plane"].value("valid_for_all_design_inner_value",false);
                value["primal_support_plane_usable"]=valid_plane;
                if(valid_plane) {
                    const auto z=value["inner_design"].get<std::vector<double>>();
                    const double q=value["inner_operating_cost"].get<double>();
                    if(z.size()!=static_cast<std::size_t>(n)||!finite_vector(z)||!std::isfinite(q))
                        throw std::logic_error("Validated oracle primal has invalid coordinates");
                    const double safe_q=value.at("safe_support_intercept").get<double>();
                    if(!std::isfinite(safe_q)||safe_q<q)throw std::logic_error("Invalid primal support upper intercept");
                    bool duplicate=false;
                    for(const auto& p:planes)if(p.design==z&&p.cost<=safe_q){duplicate=true;break;}
                    if(!duplicate){planes.push_back({z,safe_q});
                        report["primal_planes"].push_back({{"source","validated_oracle_integer_primal"},
                            {"evaluation",evaluation},{"design",z},{"operating_cost",q},{"safe_upper_operating_cost",safe_q}});}
                }
                if(value.contains("validated_original_integer_upper_bound")&&value["validated_original_integer_upper_bound"].is_number()) {
                    const double ub=value["validated_original_integer_upper_bound"].get<double>();
                    report["validated_original_integer_upper_bound"]=std::min(report["validated_original_integer_upper_bound"].get<double>(),ub+margin(ub));
                }
                report["evaluations"].push_back(value);++evaluation;checkpoint();
                if(config.method=="fixed")continue;
                if(remaining()<=config.master_seconds+.05)break;
                phase=wall_time();
                LinearModel model;model.minimize=false;
                for(int j=0;j<n;++j)model.variable("beta_"+std::to_string(j),center[j]-config.box_radius,center[j]+config.box_radius);
                const int t=model.variable("dual_value",-IloInfinity,IloInfinity,1);
                for(std::size_t i=0;i<planes.size();++i){Coefficients terms{{t,1}};
                    for(int j=0;j<n;++j)if(target[j]!=planes[i].design[j])terms.emplace_back(j,planes[i].design[j]-target[j]);
                    model.row("primal_plane_"+std::to_string(i),'L',planes[i].cost,std::move(terms));}
                NativeSolver kelley(model,options,"bdd_multiplier_box_lp");
                add_time("multiplier_lp_build",wall_time()-phase);
                phase=wall_time();const auto choice=kelley.optimize(std::min(1.0,std::max(0.0,remaining()-config.master_seconds)));
                add_time("multiplier_lp_optimize",wall_time()-phase);
                phase=wall_time();
                Json step={{"evaluation",evaluation-1},{"batch",batch},{"center",center},{"radius",config.box_radius},
                    {"target_design",target},{"plane_count",planes.size()},{"status",choice.status},
                    {"solver_status_code",choice.cplex_status},{"certified_box_upper_bound",nullptr},
                    {"certified_target_lower_bound",std::isfinite(target_best)?Json(target_best):Json(nullptr)},
                    {"used_feasible_primal_upper_planes",true},{"exact_supergradient_claimed",false}};
                bool box_closed=false;
                if(choice.has_solution){const auto residuals=point_residuals(model,choice.values);
                    step["residuals"]=residuals;
                    const double objective=objective_value(model,choice.values);
                    const bool valid=choice.optimal&&choice.cplex_status==IloCplex::Optimal&&residuals.at("passed").get<bool>()&&
                        std::isfinite(choice.bound)&&std::abs(choice.objective-objective)<=2e-7*(1+std::abs(objective));
                    step["certified"]=valid;
                    if(valid){beta.assign(choice.values.begin(),choice.values.begin()+n);
                        const double upper=std::max(objective,choice.bound)+margin(std::max(objective,choice.bound));
                        step["coefficients"]=beta;step["certified_box_upper_bound"]=upper;
                        box_closed=std::isfinite(target_best)&&upper-target_best<=2e-6*(1+std::abs(upper));
                        step["box_gap_closed"]=box_closed;
                    }else{step["failure"]="multiplier_lp_not_certified";}
                    if(!valid){report["multiplier_steps"].push_back(step);add_time("multiplier_lp_validation",wall_time()-phase);break;}
                }else{step["certified"]=false;report["multiplier_steps"].push_back(step);
                    add_time("multiplier_lp_validation",wall_time()-phase);break;}
                report["multiplier_steps"].push_back(step);add_time("multiplier_lp_validation",wall_time()-phase);
                if(box_closed)break;
            }
            if(remaining()>.01){const auto updated=solve_root(batch);checkpoint();
                if(!updated.value("certified",false)){report["status"]="root_reoptimization_not_certified";stop=true;}}
            if(config.method=="fixed"&&std::all_of(fixed_done.begin(),fixed_done.end(),[](bool v){return v;})){
                report["status"]="all_fixed_multipliers_solved";stop=true;}
            // A flat root value or one solved box is not a certificate for the
            // full integer convex hull; search continues at the next root.
            (void)batch_added;
        }
        if(report["status"]=="running")report["status"]=evaluation>=config.max_evaluations?"evaluation_limit":"time_limit";
        report["evaluations_count"]=evaluation;report["root_batches"]=batch;
        report["accepted_bdd_cut_count"]=report["cuts"].size();
        report["primal_plane_count"]=planes.size();
        report["root_gain_lower_bound"]=std::max(0.0,report["best_lower_bound"].get<double>()-report["baseline"]["objective"].get<double>());
        const double ub=report["validated_original_integer_upper_bound"].get<double>();
        const double lb=report["best_lower_bound"].get<double>();
        report["relative_gap_to_validated_integer_upper_bound"]=std::max(0.0,(ub-lb)/std::max(1e-10,std::abs(ub)));
        report["bound_consistency_passed"]=lb<=ub+1e-6*(1+std::abs(ub));
        if(!report["bound_consistency_passed"].get<bool>())throw std::logic_error("BDD root bound exceeds validated integer UB");
    }catch(const RunBudgetExceeded& error){report["status"]="budget_interruption";report["error"]=error.what();}
     catch(const IloException& error){report["status"]="solver_exception";report["error"]=error.getMessage();}
     catch(const std::exception& error){report["status"]="exception";report["error"]=error.what();}
    report["evaluations_count"]=report["evaluations"].size();
    report["accepted_bdd_cut_count"]=report["cuts"].size();
    report["primal_plane_count"]=report["primal_planes"].size();
    checkpoint();
    report["search_wall_seconds"]=wall_time()-entry;
    double accounted=0;for(auto it=report["times"].begin();it!=report["times"].end();++it)accounted+=it.value().get<double>();
    report["times"]["other_seconds"]=std::max(0.0,report["search_wall_seconds"].get<double>()-accounted);
    return report;
}
} // namespace mlnd
