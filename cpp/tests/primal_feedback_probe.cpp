#include "mlnd/primal_alns.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F action,const char* message) {
    bool rejected=false;try {action();}catch(const std::invalid_argument&) {rejected=true;}require(rejected,message);
}
template<class T> Matrix<T> fill(int rows,int cols,T value) {
    Matrix<T> result(rows,cols);std::fill(result.values.begin(),result.values.end(),value);return result;
}
Input fixture() {
    Input d;d.name="primal_feedback_domain";d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=2;
    d.max_orders=2;d.preprocessed_demand=true;d.flow_policy="aggregate";d.route_mode="direct";
    d.demand={1,1};d.ground_cost={6,6};d.single_rates={1,1};d.head_rates={1,1};
    d.pair_rates=fill(2,2,0.);d.common_origin_support=fill(2,2,1);d.base_cost={1,1};
    d.slot_cost=fill(2,2,0.);d.link_cost=fill(4,4,0.);d.allowed_links=fill(4,4,1);
    d.slot_counts=fill(2,2,10);d.occupancy_limit=fill(2,2,8.);return d;
}
void append(PathCollection& paths,int head,int base,int locker,double cost=0,int branch=-1) {
    ServicePath p;p.id=static_cast<int>(paths.service_paths.size());p.first_od=head;p.second_od=branch;
    p.b={base,base};p.pickups={locker};p.deliveries={locker};p.legs={{base,2+locker,base}};
    p.served_counts={{head,1}};p.slot_times={{locker,.1+cost}};p.cost=cost;
    p.arc_counts={{base,2+locker,1},{2+locker,base,1}};paths.service_paths.push_back(std::move(p));
}
PrimalBlockRequest request() {
    PrimalBlockRequest q;q.heads={0};q.allowed_bases={0};q.paths_per_branch=1;
    q.complete_missing_support=true;return q;
}
Json bounded_feedback() {
    auto d=fixture();PathCollection paths;
    for(int i=0;i<150;++i)append(paths,0,0,i<2?0:1,i==0?0:i==1?.1:10+i);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+2);
    auto ground=all_courier_result(d,paths);auto state=bank.inspect(ground,wall_time()+1);
    auto q=request();auto before=bank.close(state,q,wall_time()+1);
    require(before.service_indices==std::vector<int>{0},"Baseline shortlist is not the expected cheap witness");
    auto source=state.design;DesignLayout layout(d);source[layout.assignment_index(0,1)]=1;
    // These are producer-trust-boundary witnesses, not fabricated primal UB
    // certificates. No solver or physical-feasibility claim is made here.
    auto feedback=make_primal_cg_feedback(d,paths,source,{{2,-3}},{{3,.2}});
    const auto accepted=bank.observe_cg_columns(feedback,wall_time()+1);
    require(accepted.at("accepted")==2 && !accepted.at("certificate_eligible").get<bool>(),
            "Feedback was not accepted as ranking-only evidence");
    auto after=bank.close(state,q,wall_time()+1);
    require(after.service_indices==std::vector<int>({0,2,3}),"Feedback replaced the cheap witness or failed to add both evidence types");
    require(after.coupled.heads==std::vector<int>{0},"Different-design feedback changed a fixed head group");
    require(after.restrictions.forbidden_origins.empty(),"Feedback invented a local origin restriction");
    // Strictly positive incumbent support is mandatory, even when optional
    // breadth is zero and the old flow is arbitrarily small.
    ground["_solution"]["flow"][4]=1e-12;ground["_solution"]["assignment"][0][0]=1;
    state=bank.inspect(ground,wall_time()+1);q.optional_column_limit=0;
    const auto mandatory=bank.close(state,q,wall_time()+1);
    require(mandatory.service_indices==std::vector<int>({0,4}),"Feedback cap removed tiny incumbent support or ignored the optional cap");
    std::vector<std::pair<int,double>> many;
    for(int id=10;id<150;++id)many.emplace_back(id,-id);
    auto capped=make_primal_cg_feedback(d,paths,source,many,{},1000);
    require(capped.service.size()==64,"Producer batch exceeded the hard 64-column cap");
    bank.observe_cg_columns(capped,wall_time()+1);bank.observe_cg_columns(capped,wall_time()+1);
    const auto meta=bank.metadata().at("cg_feedback");
    require(meta.at("retained_batches")==2 && meta.at("retained_columns").get<int>()<=128,
            "Feedback history was not bounded to two capped batches");
    q.optional_column_limit=5000;auto capped_block=bank.close(state,q,wall_time()+1);
    require(capped_block.service_indices.size()<=4,"Per-branch feedback exceeded two optional columns plus cheap and incumbent witnesses");
    require(std::find(capped_block.service_indices.begin(),capped_block.service_indices.end(),2)==capped_block.service_indices.end(),
            "Expired feedback batch remained in the shortlist");
    append(paths,0,1,1,25);
    bank.observe_cg_columns(make_primal_cg_feedback(d,paths,source,{},{{150,.3}}),wall_time()+1);
    const auto fresh=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    q.allowed_bases={1};q.complete_missing_support=false;
    const auto appended_feedback=bank.close(fresh,q,wall_time()+1);
    require(appended_feedback.ready && appended_feedback.service_indices==std::vector<int>{150},
            "Feedback for an appended path required an unrelated ranked-bank rescan");
    return {{"case","bounded_union_different_design_and_mandatory_support"},{"passed",true}};
}
Json stale_evidence() {
    auto d=fixture();PathCollection paths;
    append(paths,0,0,0);append(paths,0,0,0,.1);append(paths,0,0,1,10);
    const auto center=solution_design(d,all_courier_result(d,paths));
    auto feedback=make_primal_cg_feedback(d,paths,center,{{2,-1}},{});
    auto changed=d;changed.single_rates[0]=.5;PrimalSupportBank other(changed,paths);
    rejects([&]{other.observe_cg_columns(feedback,wall_time()+1);},"Changed task rates accepted a stale feedback domain");
    auto changed_support=paths;InitialSegment segment;segment.id=0;changed_support.initial_segments.push_back(segment);
    PrimalSupportBank other_support(d,changed_support);
    rejects([&]{other_support.observe_cg_columns(feedback,wall_time()+1);},"Changed initial support accepted a stale feedback domain");
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    paths.service_paths[2].pickup_times={.4};
    rejects([&]{bank.observe_cg_columns(feedback,wall_time()+1);},"Payload identity omitted pickup timing");
    paths.service_paths[2].pickup_times.clear();bank.observe_cg_columns(feedback,wall_time()+1);
    paths.service_paths[2].cost+=1;
    rejects([&]{bank.close(state,request(),wall_time()+1);},"A changed feedback path remained usable after observation");
    paths.service_paths[2].cost-=1;d.single_rates[0]=.7;
    rejects([&]{bank.close(state,request(),wall_time()+1);},"In-place task-rate mutation failed to invalidate feedback");
    rejects([&]{make_primal_cg_feedback(d,paths,center,{{2,.1}},{});},"Nonnegative original reduced cost was labeled negative feedback");
    return {{"case","exact_domain_and_full_payload_identity"},{"passed",true}};
}
Json indexed_completion() {
    auto d=fixture();PathCollection paths;
    for(int i=0;i<100;++i)append(paths,1,0,0,i);
    append(paths,0,0,0);append(paths,0,0,0,.1);append(paths,0,0,1,10);append(paths,0,0,1,11);
    // This zero-rate branch still belongs to the full registered-ID index.
    append(paths,1,1,1,0,1);
    PrimalSupportBank legacy(d,paths),indexed(d,paths);
    legacy.extend(wall_time()+2);indexed.extend(wall_time()+2,true);
    auto ground=all_courier_result(d,paths);
    const auto old_state=legacy.inspect(ground,wall_time()+1),new_state=indexed.inspect(ground,wall_time()+1);
    auto q=request();q.restrictions.closed_lockers={0};
    auto old_block=legacy.close(old_state,q,wall_time()+1);q.indexed_support_completion=true;
    auto new_block=indexed.close(new_state,q,wall_time()+1);
    require(old_block.ready && new_block.ready && old_block.service_indices==new_block.service_indices &&
            new_block.service_indices==std::vector<int>{102},"Index changed the first admissible witness or completed domain");
    require(old_block.restrictions.forbidden_origins==new_block.restrictions.forbidden_origins,
            "Index changed local origin exclusions");
    const auto old_scan=old_block.diagnostics.at("support_completion").at("catalogue_paths_scanned").get<int>();
    const auto new_scan=new_block.diagnostics.at("support_completion").at("catalogue_paths_scanned").get<int>();
    require(new_scan==3 && old_scan==103,"Indexed completion scanned unrelated branches");
    require(indexed.metadata().at("complete_branch_index").at("indexed_paths")==paths.service_paths.size(),
            "Full branch index omitted registered zero-rate paths");
    require(legacy.metadata().at("complete_branch_index").at("indexed_paths")==0,
            "Legacy mode incurred full-index construction");
    append(paths,0,1,1,20);q.allowed_bases={1};
    const auto appended=indexed.close(new_state,q,wall_time()+1);
    require(appended.ready && appended.service_indices==std::vector<int>{105} &&
            appended.diagnostics.at("support_completion").at("index_entries_added")==1,
            "Append-only index failed to expose a newly registered branch");
    q.allowed_bases={0};paths.service_paths[102].first_od=1;
    rejects([&]{indexed.close(new_state,q,wall_time()+1);},"Changed indexed branch identity was silently reused");
    return {{"case","complete_index_first_witness_and_append"},{"legacy_path_visits",old_scan},
        {"indexed_path_visits",new_scan},{"passed",true}};
}
Json incomplete_and_deadline() {
    auto d=fixture();d.pair_rates(0,1)=.2;PathCollection paths;append(paths,0,0,0);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()-1,true);
    require(bank.metadata().at("complete_branch_index").at("indexed_paths")==0,
            "Expired index construction still traversed paths");
    auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    auto q=request();q.heads={d.task_index(0,1)};q.indexed_support_completion=true;
    auto expired=bank.close(state,q,wall_time()-1);
    require(!expired.ready && bank.metadata().at("complete_branch_index").at("indexed_paths")==0,
            "Expired completion fabricated a completed index");
    auto missing=bank.close(state,q,wall_time()+1);
    require(missing.ready && missing.restrictions.forbidden_origins==std::vector<std::pair<int,int>>{{d.task_index(0,1),0}} &&
            !missing.diagnostics.at("support_completion").at("unknown_support").get<bool>(),
            "Exhaustive registered-domain absence was not distinguished from interruption");
    paths.complete=false;auto partial=bank.close(state,q,wall_time()+1);
    require(partial.diagnostics.at("support_completion").at("unknown_support").get<bool>() &&
            !partial.diagnostics.at("support_completion").at("global_inference").get<bool>(),
            "Indexed registered support was confused with a global path certificate");
    return {{"case","deadline_and_incomplete_physical_domain"},{"passed",true}};
}
Json alns_entry_point() {
    auto d=fixture();PathCollection paths;append(paths,0,0,0);append(paths,0,0,1,10);
    auto ground=all_courier_result(d,paths);auto center=solution_design(d,ground);
    auto feedback=make_primal_cg_feedback(d,paths,center,{{1,-1}},{});
    SolverOptions options;PrimalAlnsSearch disabled(d,paths,options,ground,60);
    require(disabled.observe_cg_columns(feedback,wall_time()+1).at("status")=="disabled", "Disabled ALNS feedback was consumed");
    options.cg_primal_feedback=true;PrimalAlnsSearch enabled(d,paths,options,ground,60);
    const auto objective=enabled.best().at("objective");
    require(enabled.observe_cg_columns(feedback,wall_time()+1).at("accepted")==1,"ALNS did not forward bounded feedback");
    require(enabled.best().at("objective")==objective && enabled.current().at("objective")==objective &&
            !enabled.summary().at("global_lower_bound_available").get<bool>(),"Feedback changed an incumbent or invented a bound");
    return {{"case","alns_optional_ranking_interface"},{"passed",true}};
}
}
int main() {try {
    Json cases=Json::array({bounded_feedback(),stale_evidence(),indexed_completion(),incomplete_and_deadline(),alns_entry_point()});
    std::cout<<Json({{"passed",true},{"cases",cases},{"solver_called",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
