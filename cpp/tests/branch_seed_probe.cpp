// Branch support, deadline and prefix-policy checks against a test-only full
// physical catalogue. No optimizer or production path catalogue is required.
#include "mlnd/paths.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <set>

using namespace mlnd;
namespace {
using Group=std::array<int,3>;
using Branch=std::array<int,4>;
void require(bool ok,const std::string& why) { if (!ok) throw std::runtime_error(why); }
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
Input fixture(int seed,int k,const std::string& policy) {
    Input d;
    d.n_l=2; d.n_h=3; d.n_od=3; d.max_orders=k;
    d.route_mode="direct"; d.flow_policy=policy;
    d.delta=.01; d.operating_limit=.15; d.fleet_cost=1;
    d.flight_cost=.3; d.waiting_cost=.4;
    d.demand={2,3,4}; d.ground_cost={5,6,7}; d.deadline={.12,.14,.15};
    d.flight_times=Matrix<double>(5,5); d.allowed_links=Matrix<int>(5,5);
    d.kappa=Matrix<double>(3,3); d.rider_mean=Matrix<double>(3,3);
    d.first_cost=Matrix<double>(3,3); d.terminal_cost=Matrix<double>(3,3);
    d.receipt_mean=Matrix<double>(3,3); d.removal_mean=Matrix<double>(3,3);
    std::mt19937 rng(seed); std::uniform_real_distribution<double> random(0,1);
    for (int i=0;i<5;++i) for (int j=0;j<5;++j) {
        d.flight_times(i,j)=i==j?0:(seed==0?.02:.005+.05*random(rng));
        d.allowed_links(i,j)=i!=j && (seed==0 || random(rng)>.15);
    }
    for (int r=0;r<3;++r) for (int i=0;i<3;++i) {
        d.kappa(r,i)=seed==0?1:(random(rng)>.2?1:0);
        d.rider_mean(r,i)=seed==0?0:.03*random(rng);
        d.first_cost(r,i)=.1*random(rng); d.terminal_cost(r,i)=.1*random(rng);
        d.receipt_mean(r,i)=.02*random(rng); d.removal_mean(r,i)=.02*random(rng);
    }
    return d;
}
Json check(const Input& d,const std::vector<double>* mask=nullptr) {
    auto reference=enumerate_paths(d);
    rebuild_path_registry(reference);
    auto paths=prepare_path_support(d);
    const double start=now();
    const auto report=branch_covering_seed_paths(d,paths,mask);
    const double seconds=now()-start;
    require(!report.stopped,"unlimited seed stopped");
    require(report.service_added==paths.service_paths.size(),"wrong service count");
    require(report.empty_added==paths.empty_paths.size(),"wrong empty count");
    require(!paths.complete && paths.support_complete,"seed changed catalogue semantics");
    auto assigned=[&](int r,BasePair b) {
        return d.demand[r]>0 && (!mask || (*mask)[(r*d.n_l+b[0])*d.n_l+b[1]]!=0);
    };
    std::set<Group> groups;
    std::map<Group,std::vector<int>> segments;
    for (const auto& s:paths.initial_segments) if (assigned(s.first_od,s.b)) {
        const Group g{s.b[0],s.b[1],s.first_od}; groups.insert(g); segments[g].push_back(s.id);
    }
    std::set<Branch> expected,actual;
    std::map<int,std::set<int>> expected_prefix,actual_prefix;
    for (const auto& p:reference.service_paths) {
        expected.insert({p.b[0],p.b[1],p.first_od,p.second_od});
        expected_prefix[p.segment_id].insert(p.second_od);
    }
    for (const auto& p:paths.service_paths) {
        require(reference.service_registry.count(service_path_signature(p))==1,"seed outside physical domain");
        require(assigned(p.first_od,p.b),"seed first order violates assignment mask");
        require(p.second_od<0 || assigned(p.second_od,p.b),"seed continuation violates assignment mask");
        actual.insert({p.b[0],p.b[1],p.first_od,p.second_od});
        actual_prefix[p.segment_id].insert(p.second_od);
    }
    std::size_t required=0,covered=0,complete_groups=0,missing=0,prefix_failures=0;
    for (const auto& g:groups) {
        std::set<int> branches{-1};
        if (d.effective_delta()>0) for (int r=0;r<d.n_od;++r)
            if (assigned(r,{g[0],g[1]}) && d.kappa(g[2],r)>0) branches.insert(r);
        required+=branches.size();
        bool complete=true;
        if (d.flow_policy=="aggregate") {
            for (int r:branches) {
                const Branch b{g[0],g[1],g[2],r};
                require(expected.count(b)==actual.count(b),"aggregate branch missed despite physical support");
                covered+=actual.count(b);
                if (!expected.count(b)) { complete=false; ++missing; }
            }
        } else {
            bool possible=false,found=false;
            for (int id:segments[g]) {
                bool ref_ok=true,actual_ok=true;
                for (int r:branches) { ref_ok&=expected_prefix[id].count(r)!=0; actual_ok&=actual_prefix[id].count(r)!=0; }
                possible|=ref_ok; found|=actual_ok;
            }
            require(possible==found,"prefix bundle mismatch");
            complete=found; covered+=found?branches.size():0; prefix_failures+=!possible;
        }
        complete_groups+=complete;
    }
    require(report.groups_requested==groups.size() && report.required_branches==required,"target count mismatch");
    require(report.covered_branches==covered && report.groups_completed==complete_groups,"coverage count mismatch");
    require(report.physically_missing_branches==missing,"physical impossibility count mismatch");
    require(report.groups_without_complete_prefix==prefix_failures,"prefix failure count mismatch");
    require(paths.service_paths.size()<=required,"more than one path per branch registered");
    const auto repeat=branch_covering_seed_paths(d,paths,mask);
    require(repeat.service_added==0 && repeat.empty_added==0,"seed is not idempotent");
    auto expired=prepare_path_support(d);
    const auto stopped=branch_covering_seed_paths(d,expired,mask,now()-1);
    require(stopped.stopped && expired.service_paths.empty() && expired.empty_paths.empty(),"expired seed generated paths");
    return {{"policy",d.flow_policy},{"K",d.max_orders},{"seconds",seconds},
            {"service",report.service_added},{"empty",report.empty_added},
            {"required_branches",required},{"covered_branches",covered},
            {"physically_missing_branches",missing},{"groups_without_complete_prefix",prefix_failures}};
}
}
int main(int argc,char** argv) {
    try {
        Json cases=Json::array();
        for (int seed=0;seed<12;++seed) for (int k:{1,2}) for (const auto& policy:{"aggregate","segment"}) {
            auto d=fixture(seed,k,policy); cases.push_back(check(d));
            std::vector<double> mask(d.n_od*d.n_l*d.n_l,0);
            mask[1]=1; mask[5]=1; // Two assigned ODs, cross-base AA/AB/BA/BB.
            cases.push_back(check(d,&mask));
        }
        if (argc>1) for (int k:{1,2}) {
            auto d=read_input(argv[1],k); cases.push_back(check(d));
        }
        std::cout<<Json{{"passed",true},{"cases",cases.size()},{"results",cases}}.dump(2)<<'\n';
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
