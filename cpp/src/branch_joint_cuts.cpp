#include "mlnd/branch_joint_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mlnd {
namespace {
void validate(const Input& d) {
    if(d.route_mode!="direct" || d.has_candidate_routes || d.flow_policy!="aggregate" ||
       d.max_orders<1 || d.max_orders>2)
        throw std::invalid_argument("Branch/joint cuts require direct aggregate K<=2");
    auto positive=[](const std::vector<double>& v) {
        return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x)&&x>=0;});
    };
    if(!positive(d.demand)||!positive(d.ground_cost)||!positive(d.first_cost.values)||
       !positive(d.terminal_cost.values)||!positive({d.fleet_cost,d.flight_cost,d.waiting_cost,d.effective_delta()}))
        throw std::invalid_argument("Branch/joint cuts require nonnegative operating data");
    if(!std::all_of(d.kappa.values.begin(),d.kappa.values.end(),[](double x){return std::isfinite(x)&&x>=0&&x<=1;}))
        throw std::invalid_argument("Branch/joint cuts require kappa in [0,1]");
}
int index(const Input& d,int r,int s,int l,int b) {return ((r*d.n_od+s)*d.n_l+l)*d.n_l+b;}
void refresh(BranchJointCache& c) {
    std::size_t paths=0;
    for(const auto& g:c.double_groups) paths+=g.paths.size();
    c.metadata={{"support_complete",c.support_complete},{"numerical_domain_safe",c.numerical_domain_safe},
        {"timed_out",c.timed_out},{"seconds",c.seconds},{"completed_groups",c.completed_groups},
        {"total_groups",c.double_groups.size()},{"candidates_checked",c.candidates_checked},
        {"cached_double_paths",paths},{"delta_lower",c.delta_lower},{"delta_upper",c.delta_upper},
        {"positive_double_floor",c.positive_double_floor},
        {"domain_source","original_initial_segments_and_authoritative_double_path_builder"}};
}
void floors(const Input& d,const AssignmentSupportCache& single,BranchJointCache& c) {
    const int R=d.n_od;
    c.positive_double_floor.assign(R*R,0);
    if(!single.numerical_domain_safe) return;
    const double eps=std::numeric_limits<double>::epsilon(), e=64*(R+1)*eps;
    const double gamma=R*eps/(1-R*eps), delta=d.effective_delta();
    const long double D=std::accumulate(d.demand.begin(),d.demand.end(),0.0L);
    const long double upper=D*(1+gamma);
    c.delta_lower=delta; c.delta_upper=single.effective_delta_upper;
    for(double rate:d.demand) if(rate>0) {
        const double p=-std::expm1(-delta*rate);
        if(!std::isfinite(p)||p<0||p>=1) return;
        c.delta_lower=std::min(c.delta_lower,-std::log1p(-p)/rate);
    }
    if(!std::isfinite(static_cast<double>(upper))) return;
    const long double q=std::max(0.0L,-std::expm1(-static_cast<long double>(c.delta_lower)*upper)-e);
    const long double factor=upper>0?q/upper:0;
    for(int r=0;r<R;++r) {
        long double incoming=0;
        for(int t=0;t<R;++t) incoming+=static_cast<long double>(d.kappa(t,r))*d.demand[t];
        const long double head=std::max(0.0L,std::exp(-static_cast<long double>(c.delta_upper)*incoming)-e);
        for(int s=0;s<R;++s) c.positive_double_floor[r*R+s]=
            static_cast<double>(d.demand[r]*head*d.kappa(r,s)*d.demand[s]*factor*(1-e));
    }
    c.numerical_domain_safe=true;
}
bool complete_group(const Input& d,const PathCollection& paths,BranchJointCache& cache,
        DoubleBranchSupport& g,double deadline) {
    if(g.complete) return true;
    if(!cache.support_complete) return false;
    const auto& segments=cache.initial_indices[(g.first_od*d.n_l+g.origin_base)*d.n_l+g.return_base];
    const std::size_t per=static_cast<std::size_t>(d.n_h)*d.n_h,total=segments.size()*per;
    while(g.next_candidate<total) {
        if(wall_time()>=deadline) return false;
        const auto position=g.next_candidate++;
        ++g.candidates_checked; ++cache.candidates_checked;
        const auto& a=paths.initial_segments[segments[position/per]];
        const int i=static_cast<int>((position%per)/d.n_h),j=static_cast<int>(position%d.n_h);
        const int from=d.n_l+a.delivery,pick=d.n_l+i,drop=d.n_l+j;
        if(i==j || (from!=pick&&!d.allowed_links(from,pick)) ||
           !d.allowed_links(pick,drop) || !d.allowed_links(drop,g.return_base)) continue;
        const double transfer=d.flight_times(from,pick);
        const PathContinuation next{g.second_od,i,j,transfer,d.flight_times(pick,drop),
            d.flight_times(drop,g.return_base),second_wait(d.rider_mean(g.second_od,i),transfer,d.effective_delta())};
        ServicePath p;
        if(!build_service_path(d,a,&next,p)) continue;
        std::vector<int> arcs;
        for(const auto& [u,v,count]:p.arc_counts) {
            if(count<=0 || (u<d.n_l&&v<d.n_l)||!d.allowed_links(u,v))
                throw std::logic_error("Invalid direct double branch in support cache");
            arcs.push_back(u*d.n_nodes()+v);
        }
        std::sort(arcs.begin(),arcs.end());
        arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
        g.paths.push_back(std::move(arcs));
    }
    std::sort(g.paths.begin(),g.paths.end());
    g.paths.erase(std::unique(g.paths.begin(),g.paths.end()),g.paths.end());
    g.complete=true; ++cache.completed_groups;
    return true;
}

// Completed absence is certified by the union of currently closed arcs.
// Greedy deletion only removes an arc when every path remains hit.
std::vector<int> block(const std::vector<std::vector<int>>& choices,int count,double deadline) {
    std::vector<std::vector<int>> paths_by_arc(count);
    std::vector<int> full,hits(choices.size(),0);
    std::vector<unsigned char> chosen(count,0);
    for(std::size_t p=0;p<choices.size();++p) for(int e:choices[p]) {
        if(!chosen[e]) {chosen[e]=1;full.push_back(e);}
        paths_by_arc[e].push_back(static_cast<int>(p)); ++hits[p];
    }
    std::sort(full.begin(),full.end());
    // Retain heavily shared arcs by trying to remove rare arcs first.
    auto candidates=full;
    std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return paths_by_arc[a].size()<paths_by_arc[b].size();});
    for(int e:candidates) {
        if(wall_time()>=deadline) break;
        if(std::all_of(paths_by_arc[e].begin(),paths_by_arc[e].end(),[&](int p){return hits[p]>=2;})) {
            chosen[e]=0;for(int p:paths_by_arc[e]) --hits[p];
        }
    }
    std::vector<int> result;
    for(int e:full) if(chosen[e]) result.push_back(e);
    return result;
}
bool certify(const Input& d,const std::vector<std::vector<int>>& paths,
        const std::vector<double>& x,int r,int s,int l,int b,double floor,double deadline,
        BranchSupportCertificate& certificate,std::size_t& checks,bool& interrupted) {
    const DesignLayout layout(d);
    std::vector<std::vector<int>> choices;
    for(const auto& path:paths) {
        if(wall_time()>=deadline) {interrupted=true;return false;}
        ++checks;
        std::vector<int> closed;
        for(int e:path) if(x[layout.airline+e]<.5) closed.push_back(e);
        if(closed.empty()) return false;
        choices.push_back(std::move(closed));
    }
    certificate.first_od=r;certificate.second_od=s;certificate.origin_base=l;certificate.return_base=b;
    certificate.positive_flow_floor=floor;
    certificate.triggers={layout.assignment+(r*d.n_l+l)*d.n_l+b};
    if(s>=0&&s!=r) certificate.triggers.push_back(layout.assignment+(s*d.n_l+l)*d.n_l+b);
    std::sort(certificate.triggers.begin(),certificate.triggers.end());
    for(int e:block(choices,d.n_nodes()*d.n_nodes(),deadline)) certificate.blocking_arcs.emplace_back(e/d.n_nodes(),e%d.n_nodes());
    certificate.metadata={{"first_od",r},{"second_od",s},{"origin_base",l},{"return_base",b},
        {"kind",s<0?"single":"double"},{"triggers",certificate.triggers},
        {"blocking_arcs",certificate.blocking_arcs},{"domain_complete",true},
        {"physical_paths",paths.size()},{"positive_flow_floor",floor}};
    return true;
}
BranchJointCostCut make_cut(const Input& d,const std::vector<BranchSupportCertificate>& certificates,
        const std::vector<double>& x,int theta,double value,const std::string& kind) {
    const DesignLayout layout(d);
    std::set<int> origins,triggers;
    std::set<std::pair<int,int>> edges;
    Json evidence=Json::array();
    for(const auto& c:certificates) {
        if(!origins.insert(c.origin_base).second) throw std::logic_error("Duplicate base in joint support cost cut");
        triggers.insert(c.triggers.begin(),c.triggers.end());edges.insert(c.blocking_arcs.begin(),c.blocking_arcs.end());
        evidence.push_back(c.metadata);
    }
    const double CR=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    if(!std::isfinite(CR)) throw std::invalid_argument("Nonfinite branch/joint courier cost");
    const double margin=1e-8*(1+CR);
    BranchJointCostCut cut;
    cut.kind=kind;cut.valid=true;
    cut.origin_bases.assign(origins.begin(),origins.end());cut.triggers.assign(triggers.begin(),triggers.end());
    cut.blocking_arcs.assign(edges.begin(),edges.end());
    cut.row={kind+"_cost",'G',CR*(1-static_cast<double>(triggers.size()))-margin,{{theta,1}}};
    std::ostringstream key;
    key<<kind;
    for(int l:origins) key<<":L"<<l;
    for(int t:triggers) {cut.row.coefficients.emplace_back(t,-CR);key<<":A"<<t;}
    for(int r=0;r<d.n_od;++r) for(int l=0;l<d.n_l;++l) if(!origins.count(l)) for(int b=0;b<d.n_l;++b) {
        const double cost=d.demand[r]*d.ground_cost[r];
        if(cost>0) cut.row.coefficients.emplace_back(layout.assignment+(r*d.n_l+l)*d.n_l+b,cost);
    }
    for(auto [u,v]:edges) {cut.row.coefficients.emplace_back(layout.airline+u*d.n_nodes()+v,CR);key<<":E"<<u<<','<<v;}
    cut.key=key.str();
    long double lhs=0;
    for(auto [j,c]:cut.row.coefficients) lhs+=static_cast<long double>(c)*(j==theta?value:x[j]);
    cut.violation=static_cast<double>(cut.row.rhs-lhs);
    cut.violated=cut.violation>1e-7*(1+CR+std::abs(value));
    cut.metadata={{"kind",kind},{"valid",true},{"violated",cut.violated},{"integer_valid_only",true},
        {"origin_bases",cut.origin_bases},{"triggers",cut.triggers},{"blocking_arcs",cut.blocking_arcs},
        {"trigger_count",triggers.size()},{"blocking_count",edges.size()},{"base_count",origins.size()},
        {"courier_cost",CR},{"numerical_margin",margin},{"violation",cut.violation},{"certificates",evidence},
        {"joint_union",true},{"unassigned_cost_counted_once",true}};
    return cut;
}
} // namespace

BranchJointCache build_branch_joint_cache(const Input& d,const PathCollection& paths,
        const AssignmentSupportCache& singles,double deadline,double threshold) {
    validate(d);
    if(std::isnan(deadline)||!std::isfinite(threshold)||threshold<0)
        throw std::invalid_argument("Invalid branch/joint cache budget or floor");
    const double started=wall_time();
    BranchJointCache cache;
    cache.n_od=d.n_od;cache.n_l=d.n_l;cache.n_nodes=d.n_nodes();
    floors(d,singles,cache);
    cache.initial_indices.resize(static_cast<std::size_t>(d.n_od)*d.n_l*d.n_l);
    cache.double_groups.resize(static_cast<std::size_t>(d.n_od)*d.n_od*d.n_l*d.n_l);
    auto finish=[&] {cache.seconds=wall_time()-started;cache.timed_out=wall_time()>=deadline;refresh(cache);return cache;};
    if(!paths.support_complete||!singles.complete_domain||singles.n_od!=d.n_od||singles.n_l!=d.n_l) return finish();
    for(std::size_t j=0;j<paths.initial_segments.size();++j) {
        if(wall_time()>=deadline) return finish();
        const auto& a=paths.initial_segments[j];
        cache.initial_indices[(a.first_od*d.n_l+a.b[0])*d.n_l+a.b[1]].push_back(static_cast<int>(j));
    }
    for(int r=0;r<d.n_od;++r) for(int s=0;s<d.n_od;++s) for(int l=0;l<d.n_l;++l) for(int b=0;b<d.n_l;++b) {
        auto& g=cache.double_groups[index(d,r,s,l,b)];g.first_od=r;g.second_od=s;g.origin_base=l;g.return_base=b;
    }
    cache.support_complete=true;
    if(cache.numerical_domain_safe) for(auto& g:cache.double_groups) {
        if(wall_time()>=deadline) return finish();
        const double beta=cache.positive_double_floor[g.first_od*d.n_od+g.second_od];
        if(beta>0&&beta>=threshold) complete_group(d,paths,cache,g,deadline);
    }
    return finish();
}

BranchJointSeparation separate_branch_joint_cost(const Input& d,const PathCollection& paths,
        const AssignmentSupportCache& singles,BranchJointCache& cache,const std::vector<double>& x,
        int theta,double value,double deadline,bool double_cuts,bool joint_cuts,
        std::size_t max_cuts,double threshold) {
    validate(d);
    const double started=wall_time();
    const DesignLayout layout(d);
    if(x.size()<static_cast<std::size_t>(layout.count)||theta<layout.count||!std::isfinite(value)||
       std::isnan(deadline)||!std::isfinite(threshold)||threshold<0||cache.n_od!=d.n_od||cache.n_l!=d.n_l)
        throw std::invalid_argument("Invalid branch/joint separation arguments");
    for(int j=0;j<layout.count;++j) if(!std::isfinite(x[j])||x[j]<-1e-6||x[j]>1+1e-6||std::abs(x[j]-std::round(x[j]))>1e-6)
        throw std::invalid_argument("Branch/joint separation requires binary designs");
    BranchJointSeparation report;
    report.support_complete=cache.support_complete&&paths.support_complete&&singles.complete_domain;
    const auto old_candidates=cache.candidates_checked;
    auto finish=[&] {
        report.seconds=wall_time()-started;report.timed_out=report.timed_out||wall_time()>=deadline;
        report.lazy_candidates_checked=cache.candidates_checked-old_candidates;
        report.metadata={{"support_complete",report.support_complete},{"timed_out",report.timed_out},
            {"seconds",report.seconds},{"single_checks",report.single_checks},{"double_checks",report.double_checks},
            {"numerical_skips",report.numerical_skips},{"incomplete_groups",report.incomplete_groups},
            {"lazy_candidates_checked",report.lazy_candidates_checked},{"cache_completed_groups",cache.completed_groups},
            {"certificates",report.certificates.size()},{"cuts",report.cuts.size()},{"floor_threshold",threshold},
            {"master_added_variables",0}};
        return report;
    };
    if(!report.support_complete||max_cuts==0||(!double_cuts&&!joint_cuts)) return finish();
    std::vector<std::vector<int>> assigned(d.n_l*d.n_l);
    for(int r=0;r<d.n_od;++r) for(int l=0;l<d.n_l;++l) for(int b=0;b<d.n_l;++b)
        if(d.demand[r]>0&&x[layout.assignment+(r*d.n_l+l)*d.n_l+b]>.5) assigned[l*d.n_l+b].push_back(r);
    std::vector<int> best(d.n_l,-1);
    std::set<int> attempted;
    auto record=[&](BranchSupportCertificate c) {
        const int l=c.origin_base;
        const auto score=[](const BranchSupportCertificate& a){return a.triggers.size()+a.blocking_arcs.size();};
        if(best[l]<0||score(c)<score(report.certificates[best[l]])) best[l]=static_cast<int>(report.certificates.size());
        report.certificates.push_back(std::move(c));
    };
    auto try_double=[&](int r,int s,int l,int b) {
        const int id=index(d,r,s,l,b);
        if(!attempted.insert(id).second) return false;
        const double beta=cache.positive_double_floor[r*d.n_od+s];
        if(!cache.numerical_domain_safe||beta<=0||beta<threshold) {++report.numerical_skips;return false;}
        auto& g=cache.double_groups[id];
        if(!complete_group(d,paths,cache,g,deadline)) {++report.incomplete_groups;return false;}
        BranchSupportCertificate certificate;
        if(!certify(d,g.paths,x,r,s,l,b,beta,deadline,certificate,report.double_checks,report.timed_out)) return false;
        record(certificate);
        if(double_cuts) {
            auto cut=make_cut(d,{certificate},x,theta,value,"double_branch");
            if(cut.violated) report.cuts.push_back(std::move(cut));
        }
        return true;
    };
    // First collect at least one certificate per origin before spending the
    // remaining budget on additional individual double-branch cuts.
    for(int l=0;l<d.n_l;++l) {
        if(wall_time()>=deadline) break;
        if(joint_cuts) for(int b=0;b<d.n_l&&best[l]<0;++b) for(int r:assigned[l*d.n_l+b]) {
            const double beta=singles.positive_single_floor[r];
            if(!singles.numerical_domain_safe||beta<=0||beta<threshold) {++report.numerical_skips;continue;}
            const auto& group=singles.groups[(r*d.n_l+l)*d.n_l+b];
            BranchSupportCertificate certificate;
            if(certify(d,group.single_paths,x,r,-1,l,b,beta,deadline,certificate,report.single_checks,report.timed_out)) {
                record(std::move(certificate));break;
            }
        }
        if(d.effective_delta()<=0) continue;
        bool found=false;
        for(int b=0;b<d.n_l&&!found;++b) for(int r:assigned[l*d.n_l+b]) {
            for(int s:assigned[l*d.n_l+b]) {
                if(wall_time()>=deadline) break;
                if(try_double(r,s,l,b)) {found=true;break;}
            }
            if(found||wall_time()>=deadline) break;
        }
    }
    if(joint_cuts) {
        std::vector<BranchSupportCertificate> joint;
        for(int id:best) if(id>=0) joint.push_back(report.certificates[id]);
        if(joint.size()>=2) {
            auto cut=make_cut(d,joint,x,theta,value,"joint_bases");
            if(cut.violated) report.cuts.insert(report.cuts.begin(),std::move(cut));
        }
    }
    if(double_cuts&&d.effective_delta()>0&&report.cuts.size()<max_cuts)
        for(int l=0;l<d.n_l;++l) for(int b=0;b<d.n_l;++b) for(int r:assigned[l*d.n_l+b]) {
            if(report.cuts.size()>=max_cuts||wall_time()>=deadline) break;
            for(int s:assigned[l*d.n_l+b]) {
                if(report.cuts.size()>=max_cuts||wall_time()>=deadline) break;
                try_double(r,s,l,b);
            }
        }
    if(report.cuts.size()>max_cuts) report.cuts.resize(max_cuts);
    return finish();
}
} // namespace mlnd
