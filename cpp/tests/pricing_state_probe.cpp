// Private continuation/certificate checks. Link core objects except benders.o.
// Default mode constructs tiny matrices only; --solve also runs two tiny CGs.
#include "../src/benders.cpp"
#include <iostream>
#include <random>

namespace probe {
using namespace mlnd;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int n,int m,T value) {
    Matrix<T> x(n,m);std::fill(x.values.begin(),x.values.end(),value);return x;
}
Input fixture() {
    Input d;d.name="pricing_state";d.model_version="consecutive";
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
    o.cache_pricing_graph=o.persist_recourse_model=o.stabilized_pricing=true;
    o.eliminate_zero_task_columns=o.compact_task_heads=true;
    o.fast_pricing=o.batch_columns=true;o.cg_max_columns=3;
    return o;
}
Json graph_and_pricing() {
    auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    const auto o=options();auto f=build_formulation(d,paths,nullptr,&none,&none,false,false,false,formulation_strengthening(o));
    LabelPricingGraphCache graph;
    LabelPricingOracle first(d,paths,INFINITY,&graph),hit(d,paths,INFINITY,&graph);
    require(!first.graph_reused() && hit.graph_reused(),"Complete input graph was not reused");
    std::mt19937 rng(501);std::uniform_real_distribution<double> random(-10,10);
    int checks=0;
    for(int k=0;k<8;++k) {
        std::vector<double> prices(f.matrix.rows.size());for(auto& price:prices)price=random(rng);
        LabelPricingOptions po;po.fast_pricing=true;po.branch_correction=true;po.max_columns=5;
        const auto a=first.price(f,prices,{}, {},po),b=hit.price(f,prices,{}, {},po);
        require(a.complete&&b.complete && a.service==b.service && a.empty==b.empty &&
                a.minimum==b.minimum && a.correction==b.correction,"Cached graph changed complete pricing");
        double service_min=0;
        for(const auto& p:paths.service_paths)if(service_path_enabled(f,p)) {
            double rc=p.cost;for(auto [row,c]:path_column(f,p))rc-=c*prices[row];
            service_min=std::min(service_min,rc);
        }
        require(a.min_service==service_min,"Zero-task elimination changed positive-domain pricing minimum");
        for(int id:a.service)require(service_path_enabled(f,paths.service_paths.at(id)),"Zero task returned as a new column");
        ++checks;
    }
    const auto old_identity=pricing_domain_identity(d,paths);
    d.single_rates[0]=.1;
    require(pricing_domain_identity(d,paths)!=old_identity,"Task-rate mutation did not alter graph identity");
    LabelPricingOracle changed(d,paths,INFINITY,&graph);
    require(!changed.graph_reused(),"Task-rate mutation reused an old graph identity");
    LabelPricingGraphCache interrupted;
    LabelPricingOracle stopped(d,paths,0,&interrupted);
    require(interrupted.metadata().at("completed_builds")==0,"Interrupted graph was published");
    require(!stopped.price(f,std::vector<double>(f.matrix.rows.size(),0),{}, {}).complete,
            "Interrupted graph certified a smaller domain");
    LabelPricingOracle completed(d,paths,INFINITY,&interrupted);
    require(!completed.graph_reused() && interrupted.metadata().at("completed_builds")==1,
            "Incomplete graph prevented a complete rebuild");
    d.single_rates[0]=0;
    // An arbitrary discovery price can produce attractive columns that are
    // nonnegative at the current original multiplier. They must be rejected.
    std::vector<double> discovery_prices(f.matrix.rows.size(),0),original(f.matrix.rows.size(),0);
    for(int row:f.order_rows)discovery_prices[row]=20;
    const auto discovered=first.price(f,discovery_prices,{}, {});
    require(!discovered.service.empty(),"Discovery fixture did not expose a negative path");
    auto filtered=reprice_discovered_columns(f,paths,original,discovered,1e-7);
    require(filtered.discovery_only && !filtered.complete && filtered.service.empty() && filtered.empty.empty(),
            "Smoothed-only column survived original-dual filtering");
    filtered=reprice_discovered_columns(f,paths,discovery_prices,discovered,1e-7);
    require(!filtered.service.empty() && !filtered.complete,"Valid discovery columns were lost or certified");
    Cut cut;cut.valid=cut.optimal=true;
    require(!correct_missing_columns(cut,filtered) && !certify(cut,filtered,1e-7),
            "Discovery prices entered a lower-bound or termination certificate");
    filtered.complete=true;filtered.service.clear();filtered.empty.clear();filtered.negative_paths=0;filtered.minimum=0;
    require(!correct_missing_columns(cut,filtered) && !certify(cut,filtered,1e-7),
            "A misleading discovery completion flag bypassed the certificate guard");
    return {{"passed",true},{"exact_differential_checks",checks},{"graph",graph.metadata()},
            {"interrupted_graph",interrupted.metadata()}};
}
Json continuation() {
    auto d=fixture();auto paths=enumerate_paths(d);auto o=options();
    const DesignLayout layout(d);std::vector<double> a(layout.count,0),b=a;b[layout.facility]=1;
    const std::vector<int> none;LabelPricingGraphCache graph;SubproblemStateCache state;
    auto slice=[&](const std::vector<double>& design,const std::vector<int>& seed=std::vector<int>{}) {
        return subproblem(d,paths,design,o,0,seed,none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&state);
    };
    auto first=slice(a);
    require(state.state && state.builds==1 && state.hits==0,"First continuation state missing");
    auto* solver=state.state->solver.get();auto* formulation=state.state->formulation.get();
    const auto second=slice(a);
    require(state.state->solver.get()==solver && state.state->formulation.get()==formulation && state.hits==1,
            "Same-design resume rebuilt native RMP");
    require(!first.ordinary.valid && !second.ordinary.valid && !second.recourse_optimal,
            "A time-sliced model cache transferred a certificate");
    int positive=-1;
    for(const auto& p:paths.service_paths)if(service_path_enabled(*formulation,p)){positive=p.id;break;}
    require(positive>=0,"Positive seed missing");
    slice(a,{positive});
    require(state.state->solver.get()==solver && state.state->formulation->flow.at(positive)>=0,
            "Same-design external seed was not appended to persistent native RMP");
    const auto before=state.builds;slice(b);
    require(state.builds==before+1 && state.last_rebuild_reason=="design_changed","Changed design reused recourse state");
    o.compact_task_heads=false;slice(b);
    require(state.last_rebuild_reason=="options_changed","Changed formulation options reused recourse state");
    d.ground_cost[0]+=.1;slice(b);
    require(state.last_rebuild_reason=="input_or_support_changed","Edited input reused recourse state");
    slice(b,{positive});paths.service_paths.at(positive).cost+=.01;slice(b);
    require(state.last_rebuild_reason=="registered_columns_changed","Changed registered column reused recourse state");
    bool failed=false;
    try {slice(b,{static_cast<int>(paths.service_paths.size())+1});}catch(const std::exception&){failed=true;}
    require(failed && !state.state,"Failed continuation left a possibly mismatched native cache");
    return {{"passed",true},{"optimizer_calls",0},{"state",state.metadata()}};
}
Json solve_comparison() {
    auto d=fixture();auto paths=enumerate_paths(d);auto o=options();
    const DesignLayout layout(d);std::vector<double> design(layout.count,0);
    std::fill(design.begin()+layout.facility,design.begin()+layout.assignment,1);
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)design[layout.assignment_index(q,q%d.n_l)]=1;
    for(int i=0;i<d.n_nodes();++i)for(int j=0;j<d.n_nodes();++j)
        design[layout.airline+i*d.n_nodes()+j]=d.allowed_links(i,j);
    const std::vector<int> none;
    auto plain=o;plain.stabilized_pricing=plain.persist_recourse_model=plain.cache_pricing_graph=false;
    const auto a=subproblem(d,paths,design,plain,wall_time()+15,none,none,nullptr);
    LabelPricingGraphCache graph;SubproblemStateCache state;
    const auto b=subproblem(d,paths,design,o,wall_time()+15,none,none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&state);
    require(a.recourse_optimal && b.recourse_optimal,"Small exact/stabilized CG did not finish certification");
    const double reference=a.solution.at("objective").get<double>(),accelerated=b.solution.at("objective").get<double>();
    require(std::abs(reference-accelerated)<=2e-5*(1+std::abs(reference)),"Stabilized CG changed the certified recourse value");
    int discovery_rounds=0;
    for(const auto& row:b.trace)if(row.contains("stabilized_discovery")) {
        ++discovery_rounds;
        if(!row.at("stabilized_discovery").at("exact_fallback").get<bool>())
            require(!row.at("certified").get<bool>() && !row.at("pricing").at("complete").get<bool>(),
                    "Discovery-only round certified recourse");
    }
    require(discovery_rounds>0,"Stabilized CG comparison did not exercise discovery");
    auto* saved_model=state.state->formulation.get();
    const auto hits=state.hits;
    const auto resumed=subproblem(d,paths,design,o,wall_time()+15,none,none,nullptr,{},false,false,false,
        -1,-1,NAN,&b.solution,&graph,&state);
    require(state.hits==hits+1 && state.state->formulation.get()==saved_model && resumed.recourse_optimal,
            "Same-design second optimization failed to reuse and recertify the persistent RMP");
    require(std::abs(resumed.solution.at("objective").get<double>()-accelerated)<=2e-5*(1+std::abs(accelerated)),
            "Persistent second optimization changed the certified recourse value");
    // Emulate a retained numerical recovery profile, then append one new
    // column without optimizing. The unchanged RMP keeps its profile across
    // slices; a changed column set must restore fresh native defaults.
    state.state->active_primal_profile=1;
    state.state->solver->cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol,1e-10);
    subproblem(d,paths,design,o,0,none,none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&state);
    require(state.state->active_primal_profile==1 &&
            state.state->solver->cplex.getParam(IloCplex::Param::Barrier::QCPConvergeTol)==1e-10,
            "Unchanged continuation lost its numerical recovery profile");
    int missing=-1;
    for(const auto& p:paths.service_paths)if(service_path_enabled(*state.state->formulation,p) &&
        state.state->formulation->flow.at(p.id)<0){missing=p.id;break;}
    require(missing>=0,"Numerical-reset fixture has no omitted positive-bound column");
    auto* profiled_solver=state.state->solver.get();
    subproblem(d,paths,design,o,0,{missing},none,nullptr,{},false,false,false,-1,-1,NAN,nullptr,&graph,&state);
    require(state.state->formulation.get()==saved_model && state.state->active_primal_profile==0 &&
            state.state->solver.get()!=profiled_solver &&
            state.state->solver->cplex.getParam(IloCplex::Param::Barrier::QCPConvergeTol)==1e-9,
            "Appending a resumed column did not reset the native numerical profile");
    return {{"passed",true},{"reference_objective",reference},{"accelerated_objective",accelerated},
            {"discovery_rounds",discovery_rounds},{"reference_iterations",a.iterations},{"accelerated_iterations",b.iterations},
            {"resumed_iterations",resumed.iterations},{"numerical_profile_reset",true}};
}
}
int main(int argc,char** argv) {
    try {
        mlnd::Json result={{"passed",true},{"graph_and_pricing",probe::graph_and_pricing()},
                          {"continuation",probe::continuation()}};
        if(argc==2 && std::string(argv[1])=="--solve")result["solve_comparison"]=probe::solve_comparison();
        else if(argc!=1)throw std::invalid_argument("Usage: pricing_state_probe [--solve]");
        std::cout<<result.dump(2)<<'\n';return 0;
    }catch(const IloException& error){std::cerr<<error<<'\n';return 1;}
     catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
