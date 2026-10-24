// Independent exhaustive-catalogue oracle checks. No CPLEX optimization is run.
#include "mlnd/pricing.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <set>

using namespace mlnd;
namespace {
void require(bool good,const std::string& message) { if (!good) throw std::runtime_error(message); }
Input fixture(int seed,int k) {
    Input d;
    d.n_l=2; d.n_h=3; d.n_od=3; d.n_levels=2; d.max_orders=k;
    d.route_mode="direct"; d.flow_policy=seed%2?"segment":"aggregate";
    d.delta=.01; d.operating_limit=.18; d.fleet_cost=1;
    d.flight_cost=.3; d.waiting_cost=.4;
    d.demand={2,3,4}; d.ground_cost={5,6,7}; d.deadline={.15,.16,.17};
    d.base_cost={1,1};
    d.slot_cost=Matrix<double>(3,2); d.occupancy_limit=Matrix<double>(3,2);
    d.flight_times=Matrix<double>(5,5); d.allowed_links=Matrix<int>(5,5);
    d.link_cost=Matrix<double>(5,5); d.kappa=Matrix<double>(3,3);
    d.rider_mean=Matrix<double>(3,3); d.first_cost=Matrix<double>(3,3);
    d.terminal_cost=Matrix<double>(3,3); d.receipt_mean=Matrix<double>(3,3); d.removal_mean=Matrix<double>(3,3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> random(0,1);
    for(int u=0;u<5;++u) for(int v=0;v<5;++v) {
        d.flight_times(u,v)=u==v?0:.005+.04*random(rng);
        d.allowed_links(u,v)=u!=v && (seed==0 || random(rng)>.2);
        if(seed%3==2 && u!=v && !d.allowed_links(u,v)) d.flight_times(u,v)=INFINITY;
    }
    for(int r=0;r<3;++r) for(int h=0;h<3;++h) {
        d.rider_mean(r,h)=seed==0?0:.04*random(rng);
        d.first_cost(r,h)=.1*random(rng); d.terminal_cost(r,h)=.1*random(rng);
        d.receipt_mean(r,h)=.02*random(rng); d.removal_mean(r,h)=.03*random(rng);
        d.kappa(r,h)=seed==0?1:(random(rng)>.2?1:0);
    }
    if(seed==0) {
        for(int u=0;u<5;++u) for(int v=0;v<5;++v) d.flight_times(u,v)=u==v?0:.02;
        std::fill(d.first_cost.values.begin(),d.first_cost.values.end(),.1);
        std::fill(d.terminal_cost.values.begin(),d.terminal_cost.values.end(),.1);
        std::fill(d.receipt_mean.values.begin(),d.receipt_mean.values.end(),0);
        std::fill(d.removal_mean.values.begin(),d.removal_mean.values.end(),0);
    }
    return d;
}
struct Expected { double service=0,empty=0; int best_service=-1; std::size_t negatives=0; };
Expected scan(const PathCollection& paths,const Formulation& f,const std::vector<double>& pi,
              const std::vector<double>& us,const std::vector<double>& ue,const LabelPricingOptions& opt) {
    Expected out;
    std::set<int> es(opt.excluded_service.begin(),opt.excluded_service.end()), ee(opt.excluded_empty.begin(),opt.excluded_empty.end());
    auto visit=[&](bool empty,const auto& catalogue,const auto& upper) {
        for(std::size_t i=0;i<catalogue.size();++i) {
            if((empty?f.empty_flow:f.flow)[i]>=0 || (empty?ee:es).count(static_cast<int>(i))) continue;
            double rc=catalogue[i].cost+(upper.empty()?0:upper[i]);
            for(auto [row,a]:path_column(f,catalogue[i])) rc-=a*pi[row];
            auto& minimum=empty?out.empty:out.service;
            if(rc<minimum) { minimum=rc; if(!empty) out.best_service=static_cast<int>(i); }
            out.negatives+=rc < -opt.tolerance;
        }
    };
    visit(false,paths.service_paths,us); visit(true,paths.empty_paths,ue);
    return out;
}
Json exercise(const Input& d,const std::string& name,int rounds,std::size_t& total_checks,std::size_t& total_dominated) {
    auto paths=enumerate_paths(d);
    std::vector<double> fixed(DesignLayout(d).count,0);
    const std::vector<int> none;
    auto f=build_formulation(d,paths,&fixed,&none,&none);
    LabelPricingOracle oracle(d,paths);
    std::mt19937 rng(21);
    std::uniform_real_distribution<double> random(-10,10);
    std::size_t dominated=0;
    for(int round=0;round<rounds;++round) {
        std::vector<double> pi(f.matrix.rows.size(),0), us(paths.service_paths.size(),0),ue(paths.empty_paths.size(),0);
        if(round==0) for(int row:f.order_rows) pi[row]=5;
        else for(double& v:pi) v=random(rng);
        std::fill(f.flow.begin(),f.flow.end(),-1); std::fill(f.empty_flow.begin(),f.empty_flow.end(),-1);
        LabelPricingOptions opt; opt.max_columns=round%3+1;
        if(round>=2) {
            for(std::size_t i=0;i<f.flow.size();++i) {
                if(rng()%5==0) f.flow[i]=0;
                if(rng()%9==0) opt.excluded_service.push_back(static_cast<int>(i));
                if(round%2==0) us[i]=std::abs(random(rng));
            }
            for(std::size_t i=0;i<f.empty_flow.size();++i) {
                if(rng()%3==0) f.empty_flow[i]=0;
                if(round%2==0) ue[i]=std::abs(random(rng));
                if(round%7==0) opt.excluded_empty.push_back(static_cast<int>(i));
            }
        }
        if(round==10) {
            // A solved full-column RMP has no omitted variables, regardless
            // of how negative its row-only reduced costs would have been.
            std::fill(f.flow.begin(),f.flow.end(),0);
            std::fill(f.empty_flow.begin(),f.empty_flow.end(),0);
        }
        if(round==11 && !paths.service_paths.empty()) {
            // The last (usually symmetric, seemingly dominated) prefix owns
            // the ONLY cheap completion. Route-specific upper multipliers
            // must stop earlier prefixes from discarding it.
            std::fill(pi.begin(),pi.end(),0);
            for(int row:f.order_rows) pi[row]=5;
            std::fill(f.flow.begin(),f.flow.end(),-1);
            std::fill(f.empty_flow.begin(),f.empty_flow.end(),-1);
            opt.excluded_service.clear(); opt.excluded_empty.clear();
            std::fill(us.begin(),us.end(),100);
            std::fill(ue.begin(),ue.end(),0);
            us.back()=0;
        }
        auto expected=scan(paths,f,pi,us,ue,opt);
        // Adversarial exclusion: selected best path must not dominate an
        // omitted second-best negative path with the same suffix.
        if(round==1 && expected.best_service>=0) {
            f.flow[expected.best_service]=0;
            expected=scan(paths,f,pi,us,ue,opt);
        }
        auto got=oracle.price(f,pi,us,ue,opt);
        require(got.complete,name+": pricing incomplete");
        require(got.min_service==expected.service,name+": service minimum mismatch "+std::to_string(got.min_service)+" vs "+std::to_string(expected.service));
        require(got.min_empty==expected.empty,name+": empty minimum mismatch");
        require(got.minimum==std::min(expected.service,expected.empty),name+": global minimum mismatch");
        require(got.service.size()+got.empty.size()<=opt.max_columns,name+": batch cap exceeded");
        require((expected.negatives==0)==(got.service.empty()&&got.empty.empty()),name+": wrong negative-column certificate");
        require((expected.negatives==0)==(got.negative_paths==0),name+": wrong negative count sign");
        double best_returned=0;
        std::set<int> extras(opt.excluded_service.begin(),opt.excluded_service.end());
        for(int i:got.service) {
            require(f.flow.at(i)<0 && !extras.count(i),name+": selected column returned");
            double rc=paths.service_paths[i].cost+us[i];
            for(auto [row,a]:path_column(f,paths.service_paths[i])) rc-=a*pi[row];
            require(rc < -opt.tolerance,name+": nonnegative service column returned");
            best_returned=std::min(best_returned,rc);
        }
        for(int i:got.empty) {
            require(f.empty_flow.at(i)<0,name+": selected empty returned");
            double rc=paths.empty_paths[i].cost+ue[i];
            for(auto [row,a]:path_column(f,paths.empty_paths[i])) rc-=a*pi[row];
            require(rc < -opt.tolerance,name+": nonnegative empty column returned");
            best_returned=std::min(best_returned,rc);
        }
        if(expected.negatives) require(best_returned==got.minimum,name+": global best not retained in capped batch");
        require(got.correction>=0 && std::isfinite(got.correction),name+": bad dual correction");
        dominated+=got.labels_dominated; total_checks+=9;
    }
    std::vector<double> pi(f.matrix.rows.size(),0);
    LabelPricingOptions expired; expired.deadline=wall_time()-1;
    require(!oracle.price(f,pi,{}, {},expired).complete,name+": expired search certified"); ++total_checks;
    total_dominated+=dominated;
    return Json{{"name",name},{"service_paths",paths.service_paths.size()},{"empty_paths",paths.empty_paths.size()},
                {"rounds",rounds},{"labels_dominated",dominated},{"passed",true}};
}

Json exercise_dynamic(const Input& d,const std::string& name,int rounds,std::size_t& checks) {
    // Full enumeration exists only in this independent test oracle. Production
    // setup below starts with zero materialized service and empty columns.
    const auto reference=enumerate_paths(d);
    auto paths=prepare_path_support(d);
    require(paths.support_complete && !paths.complete,name+": incorrect support/domain flags");
    require(paths.service_paths.empty() && paths.empty_paths.empty(),name+": support enumerated complete paths");
    require(paths.initial_segments.size()==reference.initial_segments.size(),name+": support size mismatch");
    for (std::size_t i=0;i<paths.initial_segments.size();++i)
        require(Json(paths.initial_segments[i])==Json(reference.initial_segments[i]),name+": support identity mismatch");
    const std::vector<int> none; std::vector<double> fixed(DesignLayout(d).count,0);
    auto f=build_formulation(d,paths,&fixed,&none,&none);
    auto full=build_formulation(d,reference,&fixed,&none,&none);
    require(f.matrix.rows.size()==full.matrix.rows.size(),name+": rows depend on generated pool");
    LabelPricingOracle oracle(d,paths);
    std::mt19937 rng(51); std::uniform_real_distribution<double> random(-10,10);
    LabelPricingOptions expired; expired.deadline=wall_time()-1;
    require(!oracle.price(f,std::vector<double>(f.matrix.rows.size(),0),{}, {},expired).complete,
            name+": expired implicit pricing certified");
    require(paths.service_paths.empty() && paths.empty_paths.empty(),name+": expired search registered paths");
    LabelPricingOracle incomplete(d,paths,wall_time()-1);
    require(!incomplete.price(f,std::vector<double>(f.matrix.rows.size(),0),{}, {}).complete,
            name+": constructor-interrupted graph certified");
    for (int round=0;round<rounds;++round) {
        const auto old=paths.service_paths.size()+paths.empty_paths.size();
        f.flow.assign(paths.service_paths.size(),-1); f.empty_flow.assign(paths.empty_paths.size(),-1);
        full.flow.assign(reference.service_paths.size(),-1); full.empty_flow.assign(reference.empty_paths.size(),-1);
        std::vector<double> pi(f.matrix.rows.size(),0),us(f.flow.size(),0),ue(f.empty_flow.size(),0);
        std::vector<double> full_us(full.flow.size(),0),full_ue(full.empty_flow.size(),0);
        if (round==0) for (int row:f.order_rows) pi[row]=5;
        else for (double& value:pi) value=random(rng);
        LabelPricingOptions opt; opt.max_columns=3+round%3; LabelPricingOptions full_opt=opt;
        for (std::size_t i=0;i<f.flow.size();++i) {
            const int target=reference.service_registry.at(service_path_signature(paths.service_paths[i]));
            if (rng()%3==0) f.flow[i]=full.flow[target]=0;
            if (rng()%5==0) { opt.excluded_service.push_back(static_cast<int>(i)); full_opt.excluded_service.push_back(target); }
            us[i]=full_us[target]=std::abs(random(rng));
        }
        for (std::size_t i=0;i<f.empty_flow.size();++i) {
            const int target=reference.empty_registry.at(paths.empty_paths[i].b);
            if (rng()%3==0) f.empty_flow[i]=full.empty_flow[target]=0;
            ue[i]=full_ue[target]=std::abs(random(rng));
        }
        const auto expected=scan(reference,full,pi,full_us,full_ue,full_opt);
        const auto got=oracle.price(f,pi,us,ue,opt);
        require(got.complete,name+": dynamic pricing incomplete");
        require(got.min_service==expected.service && got.min_empty==expected.empty,name+": implicit-domain minimum mismatch");
        require((expected.negatives==0)==(got.service.empty()&&got.empty.empty()),name+": false implicit certificate");
        require(paths.service_paths.size()+paths.empty_paths.size()<=old+opt.max_columns,name+": nonreturned terminal paths registered");
        require(got.service.size()+got.empty.size()<=opt.max_columns,name+": implicit batch exceeded");
        for (int id:got.service) {
            const auto& path=paths.service_paths.at(id);
            require(reference.service_registry.count(service_path_signature(path))!=0,name+": generated path outside independent universe");
            require(id>=static_cast<int>(f.flow.size()) || f.flow[id]<0,name+": selected dynamic ID returned");
            require(register_service_path(paths,path)==id,name+": repeated path not deduplicated");
        }
        for (int id:got.empty) require(reference.empty_registry.count(paths.empty_paths.at(id).b)!=0,name+": invalid empty path");
        // A second call may reuse old generated IDs, but never renumbers them.
        for (std::size_t i=0;i<paths.service_paths.size();++i)
            require(paths.service_paths[i].id==static_cast<int>(i),name+": dynamic IDs not append stable");
        checks+=10;
    }
    auto seeded=prepare_path_support(d);
    const auto counts=greedy_seed_paths(d,seeded);
    require(counts.first==seeded.service_paths.size() && counts.second==seeded.empty_paths.size(),name+": seed count mismatch");
    require(counts.first<=static_cast<std::size_t>(2*d.n_l*d.n_l*d.n_od),name+": greedy seed unbounded");
    for (const auto& path:seeded.service_paths)
        require(reference.service_registry.count(service_path_signature(path))!=0,name+": greedy seed outside physical universe");
    const auto duplicate=greedy_seed_paths(d,seeded);
    require(duplicate.first==0 && duplicate.second==0,name+": greedy seed not idempotent");
    checks+=10;
    return Json{{"name",name},{"implicit",true},{"generated_service",paths.service_paths.size()},
        {"full_service",reference.service_paths.size()},{"seed_service",counts.first},{"rounds",rounds},{"passed",true}};
}
}
int main(int argc,char** argv) {
    try {
        std::size_t checks=0, dominated=0;
        Json results=Json::array();
        for(int seed=0;seed<10;++seed) for(int k:{1,2}) {
            auto d=fixture(seed,k);
            results.push_back(exercise(d,"synthetic_"+std::to_string(seed)+"_K"+std::to_string(k),12,checks,dominated));
        }
        auto d=fixture(0,2); auto paths=enumerate_paths(d);
        auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[](const ServicePath& p){return p.second_od>=0;});
        require(p!=paths.service_paths.end(),"fixture missing double path");
        require(std::any_of(paths.service_paths.begin(),paths.service_paths.end(),[](const ServicePath& p){return p.first_od==p.second_od;}),"fixture missing AA");
        require(std::any_of(paths.service_paths.begin(),paths.service_paths.end(),[](const ServicePath& p){return p.second_od>=0&&p.pickups[0]==p.pickups[1];}),"fixture missing repeated locker");
        for(double shift:{-2e-12,-1e-12,-1e-15,0.,1e-15,1e-12,2e-12}) {
            auto boundary=d; boundary.operating_limit=p->time+shift;
            results.push_back(exercise(boundary,"mission_boundary_"+std::to_string(shift),4,checks,dominated));
            boundary=d; boundary.deadline[p->second_od]=p->delivery_times[1]+shift;
            results.push_back(exercise(boundary,"receipt_boundary_"+std::to_string(shift),4,checks,dominated));
        }
        auto blocked=d; std::fill(blocked.allowed_links.values.begin(),blocked.allowed_links.values.end(),0);
        results.push_back(exercise(blocked,"all_links_blocked",4,checks,dominated));
        require(dominated>0,"No non-identical-prefix resource label was dominated");
        for (int seed=0;seed<4;++seed) for (int k:{1,2})
            results.push_back(exercise_dynamic(fixture(seed,k),"implicit_synthetic_"+std::to_string(seed)+"_K"+std::to_string(k),8,checks));
        if(argc>=2) for(int k:{1,2}) results.push_back(exercise(read_input(argv[1],k),"TSL20H_K"+std::to_string(k),6,checks,dominated));
        if(argc>=2) for(int k:{1,2}) results.push_back(exercise_dynamic(read_input(argv[1],k),"implicit_TSL20H_K"+std::to_string(k),4,checks));
        Json report{{"passed",true},{"checks",checks},{"cases",results.size()},{"labels_dominated",dominated},{"results",results}};
        if(argc>=3) { std::ofstream stream(argv[2]); if(!stream)throw std::runtime_error("Cannot write report"); stream<<report.dump(2)<<'\n'; }
        std::cout<<report.dump(2)<<'\n';
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
