// Reuse the existing physical fixtures and independent support checker.
#define main legacy_path_support_probe_main
#include "path_support_probe.cpp"
#undef main

namespace {
Json fractional_triangle() {
    auto d=fixture(1,1);d.single_rates={1,0};d.head_rates={1,0};
    const int n=d.n_nodes(),a=1,b=n+2,c=2*n;
    const std::vector<std::vector<int>> edges={{a,b},{a,c},{b,c}};
    PathCollection paths;paths.complete=true;paths.support_complete=true;
    for (const auto& set:edges) {
        ServicePath p;p.id=paths.service_paths.size();p.first_od=0;p.second_od=-1;p.b={0,0};
        for (int e:set) p.arc_counts.emplace_back(e/n,e%n,2); // presence, not multiplicity
        paths.service_paths.push_back(p);
    }
    const auto cache=build_path_support_cache(d,paths,INFINITY);
    require(cache.groups.size()==1,"Zero-rate branches were not ignored");
    auto x=design(d,0,-1,false);const DesignLayout layout(d);
    for (int e:{a,b,c}) x[layout.airline+e]=.5;
    // Every old 0/1 hitting-set row is satisfied at this fractional point.
    int old_checked=0;
    for (int mask=0;mask<8;++mask) {
        bool hits=true;double rhs=0;
        for (const auto& set:edges) {
            bool hit=false;
            for (int k=0;k<3;++k) if ((mask>>k)&1) {
                const int e=std::vector<int>{a,b,c}[k];
                hit=hit||std::find(set.begin(),set.end(),e)!=set.end();
            }
            hits=hits&&hit;
        }
        if (!hits) continue;
        for (int k=0;k<3;++k) if ((mask>>k)&1) rhs+=.5;
        require(rhs>=1,"An old blocker unexpectedly separates the triangle");++old_checked;
    }
    SolverOptions solver;solver.log=false;solver.threads=1;
    FractionalPathSupportOptions control;control.lp_seconds=1;control.max_cached_lps=1;
    FractionalPathSupportSeparator sep(d,cache,solver,control);
    auto found=sep.separate(x,wall_time()+5,4);
    require(found.cuts.size()==1,"Triangle fractional row was not found");
    const auto& cut=found.cuts.front();
    require(cut.valid && cut.violated && std::abs(cut.violation-.25)<1e-8,
            "Expected fractional support violation 0.25");
    int canonical=0;
    auto validate=[&](const PathSupportCut& row) {
        for (int mask=0;mask<8;++mask) for (int alpha=0;alpha<2;++alpha) {
            auto point=x;point[layout.assignment_index(0,0)]=alpha;
            for (int k=0;k<3;++k) point[layout.airline+std::vector<int>{a,b,c}[k]]=(mask>>k)&1;
            bool supported=false;
            for (const auto& set:edges) {
                bool open=true;for (int e:set) open=open&&point[layout.airline+e]>.5;
                supported=supported||open;
            }
            if (alpha && !supported) continue;
            require(violation(row.row,point)<=1e-10,"Fractional row excludes a canonical integer design");++canonical;
        }
    };
    validate(cut);
    // A certificate considering only paths containing a misses the third.
    // The full scan must repair xi to one rather than export alpha <= y_a.
    auto incomplete=certify_fractional_path_support(d,cache,0,{{a,1}},0,x,INFINITY);
    require(incomplete.valid && !incomplete.violated && incomplete.row.rhs>=1,
            "An incomplete-domain dual escaped the full scan");validate(incomplete);
    auto negative=certify_fractional_path_support(d,cache,0,{{a,-.2},{b,.5},{c,.5}},0,x,INFINITY);
    require(negative.valid && negative.row.rhs>=.5,"Projected negative dual was not repaired");validate(negative);
    require(!certify_fractional_path_support(d,cache,0,{{a,NAN}},0,x,INFINITY).valid,"NaN dual certified");
    require(!certify_fractional_path_support(d,cache,0,{},0,x,wall_time()-1).valid,"Expired scan certified");
    auto repeat=sep.separate(x,wall_time()+5,4);
    require(repeat.metadata.at("lp_cache_hits").get<int>()==1,"LP cache was not reused");
    auto open=x;for (int e:{a,b,c}) open[layout.airline+e]=1;
    auto cheap=sep.separate(open,wall_time()+5,4);
    require(cheap.cuts.empty() && cheap.metadata.at("lp_solves")==0,"Cheap support filter failed");
    require(sep.separate(x,wall_time()+5,0).cuts.empty(),"Zero row budget ignored");
    auto expired=sep.separate(x,wall_time()-1,4);
    require(expired.cuts.empty() && expired.timed_out,"Expired separation ignored");
    auto partial=cache;partial.complete_domain=false;
    FractionalPathSupportSeparator disabled(d,partial,solver,control);
    require(disabled.separate(x,INFINITY,4).cuts.empty(),"Incomplete cache issued a fractional row");
    control.max_lp_solves=0;
    FractionalPathSupportSeparator no_lp(d,cache,solver,control);
    require(no_lp.separate(x,wall_time()+5,4).cuts.empty(),"Zero LP budget ignored");
    auto both_d=d;both_d.single_rates[1]=1;both_d.head_rates[1]=1;
    auto both_paths=paths;
    for (auto p:paths.service_paths) {
        p.first_od=1;p.id=both_paths.service_paths.size();both_paths.service_paths.push_back(p);
    }
    const auto both_cache=build_path_support_cache(both_d,both_paths,INFINITY);
    auto both_x=x;both_x[layout.assignment_index(1,0)]=1;
    control.max_lp_solves=1;
    FractionalPathSupportSeparator rotating(both_d,both_cache,solver,control);
    const auto first=rotating.separate(both_x,wall_time()+5,4);
    const auto second=rotating.separate(both_x,wall_time()+5,4);
    require(first.cuts.size()==1 && second.cuts.size()==1 && first.cuts[0].head!=second.cuts[0].head,
            "LP call budget repeatedly skipped the next group");
    require(second.metadata.at("lp_models_cached")==1,"Bounded model cache exceeded its limit");
    return {{"old_blockers_checked",old_checked},{"canonical_integer_checks",canonical},
        {"fractional_violation",cut.violation},{"cut",cut.metadata},{"separation",found.metadata},
        {"dual_repair_checks",true},{"cache_reused",true},{"budget_rotation_and_eviction",true},
        {"deadline_and_budget_guards",true}};
}

Json physical_fractional_checks() {
    auto d=fixture(2,2);
    // Complete physical catalogue with positive AA/AB and only cross returns
    // to base 1 from the origin base 0.
    for (int u=d.n_l;u<d.n_nodes();++u) d.allowed_links(u,0)=0;
    const auto paths=enumerate_paths(d);
    const auto cache=build_path_support_cache(d,paths,INFINITY);
    SolverOptions options;options.log=false;
    FractionalPathSupportOptions control;control.lp_seconds=1;control.max_lp_solves=64;
    FractionalPathSupportSeparator sep(d,cache,options,control);
    auto x=design(d,0,-1);const DesignLayout layout(d);
    require(sep.separate(x,wall_time()+5,32).cuts.empty(),"Cross returns incorrectly rejected");
    for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
        if (d.allowed_links(u,v)) x[layout.airline+u*d.n_nodes()+v]=.1;
    auto found=sep.separate(x,wall_time()+10,32);
    require(!found.cuts.empty(),"Physical fractional assignment was not separated");
    std::size_t checked=0;
    for (const auto& cut:found.cuts) {
        for (const auto& group:cache.groups) if (group.head==cut.head && group.origin==cut.origin && group.branch==cut.branch) {
            for (const auto& path:group.paths) {
                auto witness=design(d,-1,-1,false);witness[layout.assignment_index(cut.head,cut.origin)]=1;
                for (int e:path) witness[layout.airline+e]=1;
                require(violation(cut.row,witness)<=1e-9,"Row rejects a full physical support witness");++checked;
            }
        }
    }
    return {{"physical_witness_checks",checked},{"cuts",found.cuts.size()},
        {"cross_return_preserved",true},{"second_order_assignment_not_required",true}};
}
}

int main() {
    try {std::cout<<Json{{"passed",true},{"triangle",fractional_triangle()},
        {"physical",physical_fractional_checks()}}.dump(2)<<'\n';return 0;}
    catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
