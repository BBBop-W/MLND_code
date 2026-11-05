// Private memo/RHS-workspace regression seam. Link core objects except benders.o.
// Default mode performs no optimize calls; --solve uses only this tiny fixture.
#include "../src/benders.cpp"
#include <iostream>

namespace performance_probe {
using namespace mlnd;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int columns,T value) {
    Matrix<T> out(rows,columns);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture() {
    Input d;d.name="performance_state";d.model_version="consecutive";
    d.n_l=2;d.n_h=3;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.single_rates={0,.5};d.head_rates={.25,.75};d.demand={.5,1};
    d.pair_rates=filled(2,2,.125);d.common_origin_support=filled(2,2,1);
    d.ground_cost={20,20};d.deadline={1,1};d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.flight_times=filled(5,5,.05);d.allowed_links=filled(5,5,1);d.link_cost=filled(5,5,.01);
    for(int i=0;i<5;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.rider_mean=filled(2,3,.08);d.first_cost=filled(2,3,1.0);d.terminal_cost=filled(2,3,.5);
    d.receipt_mean=filled(2,3,.08);d.removal_mean=filled(2,3,.03);d.slot_counts=filled(3,1,20);
    d.slot_cost=filled(3,1,.1);d.occupancy_limit=filled(3,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
SolverOptions options() {
    SolverOptions o;o.threads=1;o.log=false;o.pareto=false;o.weak_dual_cuts=true;
    o.persist_recourse_model=false;o.reuse_recourse_rhs=true;o.cache_pricing_graph=true;
    o.eliminate_zero_task_columns=o.compact_task_heads=true;
    o.catalogue_pricing=o.fast_pricing=o.batch_columns=o.streamlined_recourse=true;
    o.stabilized_pricing=o.adaptive_pricing=o.diverse_pricing=false;o.cg_max_columns=16;
    return o;
}
std::vector<double> generous_design(const Input& d,bool swapped=false) {
    const DesignLayout layout(d);std::vector<double> x(layout.count,0);
    std::fill(x.begin()+layout.facility,x.begin()+layout.assignment,1);
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)x[layout.assignment_index(q,(q+(swapped?1:0))%d.n_l)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
Json baseline(const Input& d,const PathCollection& paths,const std::vector<double>& design) {
    const std::vector<int> none;
    const auto f=build_formulation(d,paths,&design,&none,&none,false,false,false,formulation_strengthening(options()));
    std::vector<double> x(f.matrix.variables.size(),0);
    for(std::size_t j=0;j<x.size();++j)x[j]=f.matrix.variables[j].lower;
    for(std::size_t j=0;j<design.size();++j)x[f.design_vars[j]]=design[j];
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r];
    for(const auto& c:f.matrix.cones)x[c.up]=x[c.down]=x[c.one]=1;
    require(feasible_point(f.matrix,x),"Analytic all-courier point is infeasible");
    auto result=solution_json(d,paths,f,x);result["native_model_validated"]=true;return result;
}
// A proven global lower cut, independent of a solver: courier savings cannot
// exceed sum_r,l [nu_r*g_r + sum_s nu_rs*(g_r+g_s)]*alpha_rl. All other costs
// are nonnegative. Scaling this lower bound by 0<=strength<=1 is also valid.
Subproblem analytic_subproblem(const Input& d,const PathCollection& paths,
        const std::vector<double>& design,double strength=.5,bool certified_optimal=false) {
    Subproblem sub;sub.solution=baseline(d,paths,design);
    sub.ordinary.valid=true;sub.ordinary.optimal=certified_optimal;
    sub.ordinary.intercept=strength*courier_cost(d);
    const DesignLayout layout(d);sub.ordinary.coefficients.assign(layout.count,0);
    for(int r=0;r<d.n_od;++r) {
        double saving=d.single_rates[r]*d.ground_cost[r];
        for(int s=0;s<d.n_od;++s)saving+=d.pair_rates(r,s)*(d.ground_cost[r]+d.ground_cost[s]);
        for(int l=0;l<d.n_l;++l)sub.ordinary.coefficients[layout.assignment_index(r,l)]=-strength*saving;
    }
    sub.selected=sub.ordinary;sub.recourse_optimal=certified_optimal;
    sub.ordinary.diagnostics["full_domain_certificate"]={{"source","analytic_courier_savings"}};
    return sub;
}
Json memo_checks() {
    auto d=fixture();const auto paths=enumerate_paths(d);const DesignLayout layout(d);
    std::vector<double> a(layout.count,0),b=a,c=a;b[layout.facility]=1;c[layout.facility+1]=1;
    auto sa=analytic_subproblem(d,paths,a),sb=analytic_subproblem(d,paths,b),sc=analytic_subproblem(d,paths,c);
    sa.ordinary.prices.assign(1000,1);sa.ordinary.missing_service.assign(2000,2);
    sa.ordinary.missing_empty.assign(1000,3);
    const double upper=sa.solution.at("continuous_cost"),lower=value(sa.ordinary,a),tolerance=2e-6;
    DesignCertificateMemo memo(2);
    require(memo.remember(d,paths,a,sa)&&memo.remember(d,paths,b,sb),"Valid analytic intervals were not stored");
    const auto& cached=memo.entries.at(a).lower;
    require(cached.prices.capacity()==0 && cached.missing_service.capacity()==0 &&
            cached.missing_empty.capacity()==0,"Certificate memo retained catalogue multiplier buffers");
    const auto separation=memo.reuse(d,paths,a,lower-1,tolerance);
    require(separation && separation->reason=="cached_design_separation" && !separation->recourse_optimal,
            "A-B-A failed to restore a violated valid cut");
    require(separation->ordinary.coefficients==sa.ordinary.coefficients &&
            separation->solution.at("_solution")==sa.solution.at("_solution"),"Memo changed the original cut or point");
    require(!memo.reuse(d,paths,a,(lower+upper)/2,tolerance),"Unclosed interval silently skipped recourse");
    const auto epigraph=memo.reuse(d,paths,a,upper,tolerance);
    require(epigraph && epigraph->reason=="cached_design_epigraph" && !epigraph->recourse_optimal,
            "Known feasible epigraph did not reuse the point or falsely claimed optimality");
    const auto enormous=memo.reuse(d,paths,a,1e10,tolerance);
    require(enormous && !enormous->recourse_optimal,"Large queried theta manufactured interval closure");
    // B is now the least recently used design; a capacity eviction must keep A.
    require(memo.remember(d,paths,c,sc),"Third design was not stored");
    require(!memo.reuse(d,paths,b,upper,tolerance) && memo.reuse(d,paths,a,upper,tolerance),
            "Memo capacity did not evict the least recently used design");
    require(memo.metadata().at("entries")==2 && memo.metadata().at("evictions")==1,"Memo capacity accounting changed");
    DesignCertificateMemo rejected(4);
    auto invalid=sa;invalid.ordinary.valid=false;
    require(!rejected.remember(d,paths,a,invalid),"Invalid lower cut was cached");
    invalid=sa;invalid.conditional_demand=true;
    require(!rejected.remember(d,paths,a,invalid),"Conditional recourse interval was cached");
    invalid=sa;invalid.ordinary.condition={{"assignment_fingerprint","test"}};
    require(!rejected.remember(d,paths,a,invalid),"Conditional lower cut was cached");
    invalid=sa;invalid.solution["native_model_validated"]=false;
    require(!rejected.remember(d,paths,a,invalid),"Unvalidated primal point was cached");
    require(!rejected.remember(d,paths,b,sa),"Point from another design was cached");
    require(rejected.metadata().at("entries")==0,"Rejected certificate left a cache entry");
    // Q=all-courier cost when every assignment is zero; the strength-one cut
    // and the feasible point form an analytic exact certificate for this case.
    DesignCertificateMemo exact(2);auto proved=analytic_subproblem(d,paths,a,1,true);
    require(exact.remember(d,paths,a,proved),"Analytically closed interval was not stored");
    const auto closed=exact.reuse(d,paths,a,upper-1e-7,tolerance);
    require(closed&&closed->recourse_optimal,"Stored exact recourse certificate lost closure");
    DesignCertificateMemo not_proved(2);proved.recourse_optimal=false;
    require(not_proved.remember(d,paths,a,proved)&&!not_proved.reuse(d,paths,a,upper-1e-7,tolerance),
            "Numerically equal bounds without recourse certification were promoted");
    // Production stores before account()/cleanup. Check the reason: an unused
    // open base may be closed by cleanup, changing B into A.
    DesignCertificateMemo cleanup_memo(2);require(cleanup_memo.remember(d,paths,b,sb),"Original B point not stored");
    const auto cleaned=cleanup_incumbent(d,paths,sb.solution);
    require(solution_design(d,cleaned)!=b,"Cleanup witness did not change the strategic design");
    auto cleaned_sub=sb;cleaned_sub.solution=cleaned;
    require(!cleanup_memo.remember(d,paths,b,cleaned_sub),"Post-cleanup point was misattributed to original B");
    const auto original_b=cleanup_memo.reuse(d,paths,b,upper,tolerance);
    require(original_b&&solution_design(d,original_b->solution)==b,"Cleanup contaminated the stored original point");
    return {{"passed",true},{"memo",memo.metadata()},{"cleanup_design_preserved",true},{"analytic_lower",lower},{"upper",upper}};
}
Json sparse_checks() {
    auto d=fixture();auto paths=enumerate_paths(d);const auto design=generous_design(d);
    auto f=build_formulation(d,paths,&design);std::map<std::string,int> names;
    for(std::size_t j=0;j<f.matrix.variables.size();++j)names[f.matrix.variables[j].name]=static_cast<int>(j);
    std::vector<double> x(f.matrix.variables.size(),0),heads(paths.initial_segments.size(),0),served(d.n_od,0);
    std::copy(design.begin(),design.end(),x.begin());const double rho=1e-20;double busy=0;
    for(int l=0;l<d.n_l;++l) {
        x[f.rho[l]]=rho;x[f.idle[l]]=rho/(1-rho);busy+=x[f.idle[l]];
        const auto& cone=f.matrix.cones[l];x[cone.up]=1+x[f.idle[l]];x[cone.down]=1-rho;x[cone.one]=1;
    }
    for(int q=0;q<d.task_count();++q) {
        const int r=d.task_first(q),s=d.task_second(q),l=q%d.n_l;
        const double rate=d.task_rate(q);if(rate==0)continue;
        x[names.at("v_"+std::to_string(q)+"_"+std::to_string(l))]=rho;
        const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& path){
            return path.first_od==r&&path.second_od==s&&path.b==BasePair{l,l};});
        require(p!=paths.service_paths.end(),"Tiny-flow branch witness missing");
        const double flow=rate*rho;x[f.flow[p->id]]=flow;heads[p->segment_id]+=flow;busy+=p->time*flow;
        for(auto [od,count]:p->served_counts)served[od]+=flow*count;
    }
    for(std::size_t j=0;j<heads.size();++j)x[f.head[j]]=heads[j];
    for(std::size_t g=0;g<f.lift_groups.size();++g) {
        double head=0;for(int j:f.lift_groups[g])head+=heads[j];x[names.at("group_head_"+std::to_string(g))]=head;
    }
    x[f.fleet]=busy;for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r]-served[r];
    require(feasible_point(f.matrix,x),"Tiny-flow original point failed native residual validation");
    auto saved=solution_json(d,paths,f,x);saved["native_model_validated"]=true;
    // The final physical columns are unused by the first witness of each
    // branch. Remove only those zero-flow tails to represent an older prefix.
    require(saved.at("_solution").at("flow").back()==0 && saved.at("_solution").at("empty_flow").back()==0,
            "Sparse append witness unexpectedly uses a tail column");
    const auto last_service=paths.service_paths.back();const auto last_empty=paths.empty_paths.back();
    paths.service_paths.pop_back();paths.empty_paths.pop_back();paths.complete=false;rebuild_path_registry(paths);
    saved["_solution"]["flow"].erase(saved["_solution"]["flow"].end()-1);
    saved["_solution"]["empty_flow"].erase(saved["_solution"]["empty_flow"].end()-1);
    synchronize_result_paths(d,paths,saved);
    SparseRecoursePoint sparse(saved);const auto before=sparse.restore(d,paths);
    require(before.at("_solution")==saved.at("_solution"),"Sparse restore changed a tiny nonzero value");
    int tiny=0;for(const auto& item:before.at("_solution").at("flow"))
        if(item.get<double>()>0&&item.get<double>()<1e-18)++tiny;
    require(tiny>0,"Tiny positive sparse witness vanished");
    register_service_path(paths,last_service);register_empty_path(paths,last_empty);
    const auto after=sparse.restore(d,paths);
    require(after.at("_solution").at("flow").size()==paths.service_paths.size() &&
            after.at("_solution").at("empty_flow").size()==paths.empty_paths.size(),"Restored arrays did not extend with the catalogue");
    require(after.at("_solution").at("flow").back()==0 && after.at("_solution").at("empty_flow").back()==0,
            "Appended physical columns acquired nonzero cached flow");
    for(std::size_t i=0;i<before.at("_solution").at("flow").size();++i)
        require(after.at("_solution").at("flow").at(i)==before.at("_solution").at("flow").at(i),"Append changed a retained sparse value");
    require(after.at("_solution").at("head")==before.at("_solution").at("head") &&
            after.at("objective")==before.at("objective"),"Append changed cached head or cost");
    return {{"passed",true},{"tiny_positive_columns",tiny},{"append_zero_padding",true}};
}
void fixed_rhs(const SubproblemStateCache& cache,const std::vector<double>& design) {
    const auto& state=*cache.state;const auto& f=*state.formulation;
    require(state.design==design,"Workspace design identity is stale");
    for(std::size_t j=0;j<design.size();++j) {
        const int row=f.design_fix_rows[j];
        require(f.matrix.rows[row].rhs==design[j] && state.solver->rows[row].getLB()==design[j] &&
                state.solver->rows[row].getUB()==design[j],"Record/native fixed RHS disagree after design update");
    }
}
Json rhs_checks() {
    auto d=fixture();auto paths=enumerate_paths(d);auto o=options();const DesignLayout layout(d);
    std::vector<double> a(layout.count,0),b=a;b[layout.facility]=1;
    const std::vector<int> none;SubproblemStateCache cache;LabelPricingGraphCache graph;
    auto slice=[&](const std::vector<double>& design,const std::vector<int>& selected=std::vector<int>{}) {
        return subproblem(d,paths,design,o,0,selected,none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&cache);
    };
    const auto sa=slice(a);auto* solver=cache.state->solver.get();auto* formulation=cache.state->formulation.get();fixed_rhs(cache,a);
    const auto sb=slice(b);fixed_rhs(cache,b);
    require(cache.state->solver.get()==solver&&cache.state->formulation.get()==formulation,"A-B rebuilt an ordinary RHS-only workspace");
    const auto again=slice(a);fixed_rhs(cache,a);
    require(cache.state->solver.get()==solver&&cache.state->formulation.get()==formulation&&cache.rhs_updates==2,
            "A-B-A did not retain the ordinary native model");
    require(sa.iterations==0&&sb.iterations==0&&again.iterations==0&&!again.ordinary.valid,
            "Zero-deadline workspace test optimized or transferred a certificate");
    cache.state->price_center={1};cache.state->pricing_round=9;cache.state->recent_service={0};
    cache.state->recent_empty={0};cache.state->previous_pricing_seconds=2;
    cache.state->pricing_control.certificate(1);cache.state->primal_retries_for_columns=2;
    cache.state->primal_polish_attempted_for_columns=true;cache.state->primal_retry_reason="test";
    cache.state->active_primal_profile=1;
    cache.state->solver->cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol,1e-10);
    slice(b);fixed_rhs(cache,b);
    require(cache.state->formulation.get()==formulation && cache.state->solver.get()!=solver &&
            cache.state->active_primal_profile==0 && cache.state->primal_retries_for_columns==0 &&
            !cache.state->primal_polish_attempted_for_columns && cache.state->primal_retry_reason.empty(),
            "Cross-design numerical recovery profile was not reset by fresh extraction");
    require(cache.state->solver->cplex.getParam(IloCplex::Param::Barrier::QCPConvergeTol)==1e-9 &&
            cache.state->price_center.empty() && cache.state->pricing_round==0 &&
            cache.state->recent_service.empty()&&cache.state->recent_empty.empty()&&
            cache.state->previous_pricing_seconds==0 && !std::isfinite(cache.state->pricing_control.certified_lower),
            "New design inherited old pricing or numerical state");
    auto builds=cache.builds;o.compact_task_heads=false;slice(b);
    require(cache.builds==++builds&&cache.last_rebuild_reason=="options_changed","Changed options reused old RMP");
    d.ground_cost[0]+=.1;slice(b);
    require(cache.builds==++builds&&cache.last_rebuild_reason=="input_or_support_changed","Changed domain reused old RMP");
    int positive=-1;for(const auto& p:paths.service_paths)if(service_path_enabled(*cache.state->formulation,p)){positive=p.id;break;}
    require(positive>=0,"Positive canonical column is absent");slice(b,{positive});
    paths.service_paths[positive].cost+=.01;slice(b);
    require(cache.builds==++builds&&cache.last_rebuild_reason=="registered_columns_changed","Edited selected payload reused old native coefficients");
    return {{"passed",true},{"optimizer_calls",0},{"state",cache.metadata()},{"native_rhs_checked",true}};
}
Json solve_checks() {
    auto d=fixture();auto paths=enumerate_paths(d);auto o=options();const std::vector<int> none;
    const auto a=generous_design(d),b=generous_design(d,true);
    auto cold=o;cold.persist_recourse_model=cold.reuse_recourse_rhs=false;
    const auto reference=subproblem(d,paths,a,cold,wall_time()+15,none,none,nullptr);
    SubproblemStateCache cache;LabelPricingGraphCache graph;
    auto hot=[&](const std::vector<double>& design) {
        return subproblem(d,paths,design,o,wall_time()+15,none,none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&cache);
    };
    const auto first=hot(a),middle=hot(b),last=hot(a);
    auto full=build_formulation(d,paths,&a);NativeSolver solver(full.matrix,cold,"performance_state_full_reference");
    const auto solved=solver.optimize(15);
    require(solved.has_solution&&solved.optimal&&feasible_point(full.matrix,solved.values),"Full-domain reference SOCP did not validate");
    const double expected=solved.objective,tolerance=2e-5*(1+std::abs(expected));
    for(const auto* sub:{&reference,&first,&middle,&last}) {
        const auto& design=sub==&middle?b:a;
        require(sub->ordinary.valid&&sub->recourse_optimal,"Cold/hot CG did not complete its original-domain certificate");
        require(std::abs(sub->solution.at("continuous_cost").get<double>()-expected)<=tolerance,
                "Cold/hot recourse objective differs from symmetric full reference");
        require(std::abs(value(sub->ordinary,design)-expected)<=tolerance,"Cold/hot certified lower value differs from full reference");
    }
    require(cache.rhs_updates==2&&cache.hits>=2,"Solved A-B-A did not exercise RHS reuse");
    DesignCertificateMemo memo(2);require(memo.remember(d,paths,a,last),"Solved original certificate was not cacheable");
    const auto reused=memo.reuse(d,paths,a,last.solution.at("continuous_cost"),2*o.cg_duality_tolerance);
    require(reused&&reused->recourse_optimal,"Solved exact interval did not survive memo restoration");
    return {{"passed",true},{"full_reference",expected},{"cold_iterations",reference.iterations},
        {"hot_first_iterations",first.iterations},{"hot_return_iterations",last.iterations},{"cache",cache.metadata()}};
}
}
int main(int argc,char** argv) {
    try {
        mlnd::Json result={{"passed",true},{"memo",performance_probe::memo_checks()},
            {"sparse_point",performance_probe::sparse_checks()},{"rhs_workspace",performance_probe::rhs_checks()}};
        if(argc==2&&std::string(argv[1])=="--solve")result["solve_comparison"]=performance_probe::solve_checks();
        else if(argc!=1)throw std::invalid_argument("Usage: performance_state_probe [--solve]");
        std::cout<<result.dump(2)<<'\n';return 0;
    }catch(const IloException& e){std::cerr<<e<<'\n';return 1;}
     catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
