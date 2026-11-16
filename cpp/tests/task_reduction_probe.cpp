// Exact fixed-task projection, inverse lifting and column-insertion witnesses.
// No production instance or experiment output is read by this probe.
#define main task_reduction_fixture_original_main
#include "primal_polish_probe.cpp"
#undef main
#include "mlnd/output.hpp"
#include <map>

namespace {
std::map<std::string,int> variable_names(const Formulation& f) {
    std::map<std::string,int> out;
    for(std::size_t j=0;j<f.matrix.variables.size();++j)out[f.matrix.variables[j].name]=static_cast<int>(j);
    return out;
}
std::vector<double> project_point(const Formulation& from,const Formulation& to,
                                  const std::vector<double>& values) {
    const auto names=variable_names(from);
    std::vector<double> out(to.matrix.variables.size(),0);
    for(std::size_t j=0;j<out.size();++j) {
        const auto found=names.find(to.matrix.variables[j].name);
        if(found!=names.end())out[j]=values.at(found->second);
    }
    return out;
}
std::vector<double> restore_point(const Formulation& from,const Formulation& to,
                                 const PathCollection& paths,const std::vector<double>& values) {
    auto out=project_point(from,to,values);
    const auto head=recover_task_heads(from,paths,values);
    const auto names=variable_names(to);
    for(std::size_t j=0;j<to.head.size();++j)out.at(to.head[j])=head[j];
    for(std::size_t g=0;g<to.lift_groups.size();++g) {
        double sum=0;for(int j:to.lift_groups[g])sum+=head[j];
        out.at(names.at("group_head_"+std::to_string(g)))=sum;
    }
    return out;
}
Input reduction_fixture(int k) {
    auto d=fixture(2,k);
    if(k==2) {
        d.single_rates[0]=0;d.head_rates[0]-=.5;d.demand[0]-=.5;
    }
    return d;
}
FormulationStrengthening reductions(bool compact=true) {
    FormulationStrengthening s;
    s.eliminate_zero_task_columns=true;s.compact_task_heads=compact;
    s.shared_rho_joint=s.queue_perspective=true;
    return s;
}
std::vector<double> analytic_point(const Input& d,const PathCollection& paths,
                                   const Formulation& f,const std::vector<double>& design,bool fractional) {
    auto names=variable_names(f);std::vector<double> x(f.matrix.variables.size(),0);
    std::copy(design.begin(),design.end(),x.begin());
    const double rho=.25,activity=fractional?.15:rho;
    double busy=0;std::vector<double> served(d.n_od),heads(paths.initial_segments.size(),0);
    for(int l=0;l<d.n_l;++l) {
        const double open=f.strengthening.queue_perspective?design[l]:1;
        x[f.rho[l]]=rho;x[f.idle[l]]=open*rho/(open-rho);
        const auto& cone=f.matrix.cones[l];
        x[cone.up]=x[f.idle[l]]+open;x[cone.down]=open-rho;x[cone.one]=open;
        busy+=x[f.idle[l]];
    }
    for(int q=0;q<d.task_count();++q) {
        const double rate=d.task_rate(q);
        if(rate==0)continue;
        const int l=q%d.n_l,r=d.task_first(q),s=d.task_second(q);
        x[names.at("v_"+std::to_string(q)+"_"+std::to_string(l))]=activity;
            const auto found=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p) {
                return p.first_od==r && p.second_od==s && p.b==BasePair{l,l};
            });
            require(found!=paths.service_paths.end(),"Positive branch witness missing");
            const auto& p=*found;const double flow=rate*activity;
            x[f.flow.at(p.id)]=flow;heads.at(p.segment_id)+=flow;busy+=p.time*flow;
            for(auto [od,count]:p.served_counts)served[od]+=flow*count;
    }
    for(std::size_t j=0;j<heads.size();++j)x.at(f.head[j])=heads[j];
    for(std::size_t g=0;g<f.lift_groups.size();++g) {
        double sum=0;for(int j:f.lift_groups[g])sum+=heads[j];
        x[names.at("group_head_"+std::to_string(g))]=sum;
    }
    x[f.fleet]=busy;
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r]-served[r];
    return x;
}
Json projection_case(int k,bool fractional) {
    const auto d=reduction_fixture(k);const auto paths=enumerate_paths(d);
    auto design=generous_design(d);
    const DesignLayout layout(d);
    std::fill(design.begin()+layout.assignment,design.begin()+layout.airline,0);
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)
        design[layout.assignment_index(q,q%d.n_l)]=1;
    if(fractional)for(int j=0;j<static_cast<int>(design.size());++j)if(design[j]>0)
        design[j]=(j>=DesignLayout(d).assignment && j<DesignLayout(d).airline)?.5:.75;
    auto strong=reductions();auto plain=strong;
    plain.eliminate_zero_task_columns=plain.compact_task_heads=false;
    auto original=build_formulation(d,paths,&design,nullptr,nullptr,fractional,false,false,plain);
    auto reduced=build_formulation(d,paths,&design,nullptr,nullptr,fractional,false,false,strong);
    const auto x=analytic_point(d,paths,original,design,fractional);
    require(point_residuals(original.matrix,x).at("passed"),"Analytic original witness is infeasible");
    const auto projected=project_point(original,reduced,x);
    require(point_residuals(reduced.matrix,projected).at("passed"),"Projection lost a feasible point");
    const auto restored=restore_point(reduced,original,paths,projected);
    require(point_residuals(original.matrix,restored).at("passed"),"Inverse lifting violates original constraints");
    require(std::abs(objective_value(original.matrix,x)-objective_value(reduced.matrix,projected))<1e-10,
            "Exact reduction changed the witness objective");
    require(reduced.matrix.rows.size()<original.matrix.rows.size() &&
            reduced.matrix.variables.size()<original.matrix.variables.size(),"Head compression did not reduce matrix size");
    std::size_t zeros=0;
    for(const auto& p:paths.service_paths) {
        const double rate=p.second_od<0?d.single_rates[p.first_od]:d.pair_rates(p.first_od,p.second_od);
        if(rate==0) {++zeros;require(reduced.flow[p.id]<0 && service_path_upper_bound(reduced,p)==0,"Zero branch column survived");}
        else require(reduced.flow[p.id]>=0,"Positive branch column was deleted");
    }
    if(k==2)require(zeros>0,"Zero-rate regression case contains no zero columns");
    // Fixed binary and fixed fractional designs are continuous SOCPs. Compare
    // their actual optimized values, not only one manufactured primal point.
    SolverOptions options;options.seconds=10;options.threads=1;options.gap=1e-8;
    NativeSolver full_solver(original.matrix,options),small_solver(reduced.matrix,options);
    const auto full=full_solver.optimize(10),small=small_solver.optimize(10);
    require(full.has_solution && small.has_solution && full.optimal && small.optimal,
            "Projection comparison lacks optimal continuous solutions");
    require(std::abs(full.objective-small.objective)<=2e-6*(1+std::abs(full.objective)),
            "Reduced and original recourse values disagree");
    auto repaired=small.values;
    require(repair_inventory_primal(reduced,repaired).at("passed"),"Reduced optimizer point failed validation");
    require(point_residuals(original.matrix,restore_point(reduced,original,paths,repaired)).at("passed"),
            "Optimized reduced point cannot be lifted into original model");
    if(!fractional) {
        const auto saved=solution_json(d,paths,original,x);
        options.eliminate_zero_task_columns=options.compact_task_heads=true;
        options.shared_rho_joint=options.queue_perspective=true;
        options.neighborhood_cuts=false;
        const std::vector<int> none;
        const auto warm=solve_restricted_heuristic(d,paths,options,wall_time()+10,&saved,0,&none,&none,false);
        require(warm.at("validated") && warm.at("diagnostics").at("incumbent_validation").at("passed"),
                "Original-format incumbent cannot warm-start the compact model");
        const auto recovered=solution_json(d,paths,reduced,projected);
        require(recovered.at("_solution").at("head")==saved.at("_solution").at("head"),
                "Compact output changed the public segment head vector");
    }
    return {{"k",k},{"fractional",fractional},{"zero_columns",zeros},
        {"original_variables",original.matrix.variables.size()},{"reduced_variables",reduced.matrix.variables.size()},
        {"original_rows",original.matrix.rows.size()},{"reduced_rows",reduced.matrix.rows.size()},
        {"original_objective",full.objective},{"reduced_objective",small.objective}};
}
Json structural_edges() {
    auto d=reduction_fixture(2);
    d.pair_rates(1,1)=1e-12;d.head_rates[1]+=1e-12;d.demand[1]+=2e-12;
    auto paths=enumerate_paths(d);const auto strong=reductions();
    const auto small=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,strong);
    int tiny=0;
    for(const auto& p:paths.service_paths)if(p.first_od==1 && p.second_od==1) {
        ++tiny;require(small.flow[p.id]>=0,"A tiny positive task rate was thresholded to zero");
    }
    require(tiny>0,"Tiny-rate witness absent");
    d.first_leg_consistency=true;
    bool first_leg_rejected=false;
    try{build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,strong);}
    catch(const std::invalid_argument&){first_leg_rejected=true;}
    require(first_leg_rejected,"Historical first-leg coupling was accepted for independent tasks");
    d.first_leg_consistency=false;d.preprocessed_demand=false;
    const auto legacy=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,strong);
    require(!legacy.compact_task_heads && !legacy.zero_task_columns_eliminated,"Reduction changed historical lifted model");
    d=reduction_fixture(2);
    for(int h=0;h<d.n_h;++h)d.allowed_links(1,d.n_l+h)=0;
    const auto unsupported_paths=enumerate_paths(d);
    const auto unsupported=build_formulation(d,unsupported_paths,nullptr,nullptr,nullptr,false,false,false,strong);
    for(int q=0;q<d.task_count();++q) {
        const auto& row=unsupported.matrix.rows.at(unsupported.task_rows.at(q*d.n_l+1));
        require(row.coefficients.size()==1 && row.coefficients[0].second==d.task_rate(q),
                "Empty-origin support lost an independent task equality");
    }
    d=reduction_fixture(2);
    d.pair_rates(0,1)+=d.demand[0]+1-d.head_rates[0];d.head_rates[0]=d.demand[0]+1;
    const auto bounded=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,strong);
    require(bounded.reduction_metadata.at("retained_head_bound_rows")==d.n_l,
            "Nonredundant literal head upper bounds were removed");
    return {{"tiny_positive_paths",tiny},{"empty_support_preserved",true},{"legacy_unchanged",true},
        {"first_leg_coupling_rejected",true},{"literal_head_bounds_preserved",true}};
}
Json insertion_case(bool batch) {
    const auto d=reduction_fixture(2);const auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none,false,false,false,reductions());
    const auto complete=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,reductions());
    const auto rows=f.matrix.rows.size();SolverOptions options;options.threads=1;
    NativeSolver solver(f.matrix,options);
    std::vector<int> all(paths.service_paths.size());std::iota(all.begin(),all.end(),0);
    std::vector<int> empty(paths.empty_paths.size());std::iota(empty.begin(),empty.end(),0);
    add_columns(f,solver,d,paths,all,empty,batch);
    const auto size=f.matrix.variables.size();add_columns(f,solver,d,paths,all,empty,batch);
    require(size==f.matrix.variables.size() && rows==f.matrix.rows.size(),"Insertion was not idempotent or changed row domain");
    require(f.service_indices==complete.service_indices && f.empty_indices==complete.empty_indices,
            "Incremental insertion disagrees with reduced full model");
    auto named_matrix=[](const Formulation& model) {
        std::map<std::string,std::map<std::string,double>> out;
        for(const auto& row:model.matrix.rows)for(auto [j,a]:row.coefficients)if(a)
            out[row.name][model.matrix.variables[j].name]+=a;
        return out;
    };
    require(named_matrix(f)==named_matrix(complete),"Incremental compact columns differ from full construction");
    return {{"batch",batch},{"active_service_columns",f.service_indices.size()},{"idempotent",true}};
}
}
int main() {
    try {
        Json cases=Json::array();
        for(int k:{1,2})for(bool fractional:{false,true})cases.push_back(projection_case(k,fractional));
        std::cout<<Json{{"passed",true},{"projection_cases",cases},{"edges",structural_edges()},
            {"insertion",Json::array({insertion_case(false),insertion_case(true)})}}.dump(2)<<'\n';
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
