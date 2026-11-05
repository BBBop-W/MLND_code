// Public exact-pricing differential checks; no optimization is performed.
#include "mlnd/pricing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int n,int m,T value){Matrix<T> x(n,m);std::fill(x.values.begin(),x.values.end(),value);return x;}
Input fixture(int h=3,int r=3){
    Input d;d.name="pricing_speed_exact";d.model_version="consecutive";
    d.n_l=2;d.n_h=h;d.n_od=r;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand.assign(r,2+r*.5);d.single_rates.assign(r,2);d.head_rates.assign(r,2+r*.25);
    d.pair_rates=filled(r,r,.25);d.common_origin_support=filled(r,2,1);
    d.ground_cost.assign(r,20);d.deadline.assign(r,1);d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.flight_times=filled(h+2,h+2,.05);d.allowed_links=filled(h+2,h+2,1);d.link_cost=filled(h+2,h+2,.01);
    for(int u=0;u<h+2;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.rider_mean=filled(r,h,.08);d.first_cost=filled(r,h,1.0);d.terminal_cost=filled(r,h,.5);
    d.receipt_mean=filled(r,h,.08);d.removal_mean=filled(r,h,.03);d.slot_counts=filled(h,1,20);
    d.slot_cost=filled(h,1,.1);d.occupancy_limit=filled(h,1,14.0);d.kappa=filled(r,r,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}
void equivalent(const LabelPricingResult& a,const LabelPricingResult& b){
    require(a.complete&&b.complete,"Pricing certificate did not complete");
    require(a.service==b.service&&a.empty==b.empty,"Fast pricing changed selected columns/order");
    require(a.minimum==b.minimum&&a.min_service==b.min_service&&a.min_empty==b.min_empty,
            "Fast pricing changed complete-domain reduced-cost minimum");
    require(a.correction==b.correction&&a.branch_service_correction==b.branch_service_correction,
            "Fast pricing changed the safe omitted-column correction");
    require(a.negative_paths==b.negative_paths&&a.omitted_paths==b.omitted_paths&&a.labels_dominated==b.labels_dominated,
            "Fast pricing changed exact dominance semantics");
}
Json first_slice(){
    auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingOracle oracle(d,paths);
    std::vector<double> pi(f.matrix.rows.size(),0);for(int row:f.order_rows)pi[row]=10;
    LabelPricingOptions opt;opt.branch_correction=true;opt.max_columns=7;
    const auto old=oracle.price(f,pi,{}, {},opt);opt.fast_pricing=true;
    const auto fast=oracle.price(f,pi,{}, {},opt);equivalent(old,fast);
    require(fast.fast_pricing_used,"Requested exact fast pricing was not applied");
    require(fast.catalogue_path_reuses>0,"Fast pricing did not reuse authoritative physical paths");
    require(fast.physical_path_builds<old.physical_path_builds,"Fast pricing still rebuilt all physical paths");
    return {{"passed",true},{"service_paths",paths.service_paths.size()},{"old_path_builds",old.physical_path_builds},
        {"new_path_builds",fast.physical_path_builds},{"catalogue_path_reuses",fast.catalogue_path_reuses}};
}
Json differential_slice(){
    std::mt19937 rng(7841);std::uniform_real_distribution<double> random(-9,9);
    int checks=0;std::size_t reuses=0,builds=0;
    for(int variant=0;variant<6;++variant){
        auto d=fixture();
        if(variant==1||variant==2||variant==4)d.preprocessed_demand=false;
        if(variant==2)d.flow_policy="segment";
        if(variant==3)d.first_leg_consistency=true;
        if(variant==4)d.max_orders=1;
        if(variant==5){
            d.operating_limit=.25;
            for(int i=0;i<d.n_nodes();++i)for(int j=0;j<d.n_nodes();++j)if(i!=j){
                d.flight_times(i,j)=.01+.035*std::abs(random(rng))/9;
                if(rng()%9==0)d.allowed_links(i,j)=0;
            }
        }
        auto paths=enumerate_paths(d);const std::vector<int> none;
        if(variant==3) {
            bool rejected=false;try{build_formulation(d,paths,nullptr,&none,&none);}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"Unsupported first-leg coupling was accepted");continue;
        }
        auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingOracle oracle(d,paths);
        for(int round=0;round<8;++round){
            std::vector<double> pi(f.matrix.rows.size()),upper(paths.service_paths.size()),empty_upper(paths.empty_paths.size());
            for(auto& v:pi)v=random(rng);for(auto& v:upper)v=std::abs(random(rng));for(auto& v:empty_upper)v=std::abs(random(rng));
            LabelPricingOptions opt;opt.branch_correction=true;opt.max_columns=1+round*3;
            for(std::size_t i=0;i<f.flow.size();++i){f.flow[i]=rng()%5==0?0:-1;if(rng()%11==0)opt.excluded_service.push_back(static_cast<int>(i));}
            for(std::size_t i=0;i<f.empty_flow.size();++i){f.empty_flow[i]=rng()%3==0?0:-1;if(rng()%7==0)opt.excluded_empty.push_back(static_cast<int>(i));}
            if(round==6){std::fill(f.flow.begin(),f.flow.end(),0);std::fill(f.empty_flow.begin(),f.empty_flow.end(),0);}
            if(round==7)opt.branch_deadline=0;
            const auto old=oracle.price(f,pi,upper,empty_upper,opt);opt.fast_pricing=true;
            const auto fast=oracle.price(f,pi,upper,empty_upper,opt);equivalent(old,fast);
            require(old.branch_correction_complete==fast.branch_correction_complete&&old.branch_correction_status==fast.branch_correction_status,
                "Fast pricing changed fallback/certificate completeness");
            double min_service=0,min_empty=0;
            for(std::size_t i=0;i<paths.service_paths.size();++i){
                if(f.flow[i]>=0||std::find(opt.excluded_service.begin(),opt.excluded_service.end(),static_cast<int>(i))!=opt.excluded_service.end())continue;
                double rc=paths.service_paths[i].cost+upper[i];
                for(auto [row,a]:path_column(f,paths.service_paths[i]))rc-=a*pi[row];
                min_service=std::min(min_service,rc);
            }
            for(std::size_t i=0;i<paths.empty_paths.size();++i){
                if(f.empty_flow[i]>=0||std::find(opt.excluded_empty.begin(),opt.excluded_empty.end(),static_cast<int>(i))!=opt.excluded_empty.end())continue;
                double rc=paths.empty_paths[i].cost+empty_upper[i];
                for(auto [row,a]:path_column(f,paths.empty_paths[i]))rc-=a*pi[row];
                min_empty=std::min(min_empty,rc);
            }
            require(fast.min_service==min_service&&fast.min_empty==min_empty,"Exact fast pricing differs from exhaustive physical-path reference");
            ++checks;reuses+=fast.catalogue_path_reuses;builds+=fast.physical_path_builds;
        }
        LabelPricingOptions stopped;stopped.fast_pricing=true;stopped.deadline=0;
        require(!oracle.price(f,std::vector<double>(f.matrix.rows.size(),0),{}, {},stopped).complete,"Expired fast pricing certified a domain");
    }
    // Same public oracle with an incomplete materialized pool keeps generating
    // its complete physical domain and uses authoritative scalar construction.
    auto d=fixture();d.preprocessed_demand=false;auto paths=prepare_path_support(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingOracle oracle(d,paths);
    for(int round=0;round<4;++round){
        f.flow.assign(paths.service_paths.size(),-1);f.empty_flow.assign(paths.empty_paths.size(),-1);
        std::vector<double> pi(f.matrix.rows.size());for(auto& v:pi)v=random(rng);
        LabelPricingOptions opt;opt.branch_correction=true;opt.max_columns=4;
        const auto old=oracle.price(f,pi,{}, {},opt);opt.fast_pricing=true;
        const auto fast=oracle.price(f,pi,{}, {},opt);equivalent(old,fast);
        require(fast.catalogue_path_reuses==0&&fast.branch_correction_status=="unsupported_scope",
            "Incomplete catalogue was used as an exact physical-domain cache");++checks;
    }
    return {{"passed",true},{"differential_rounds",checks},{"complete_catalogue_reuses",reuses},{"fallback_builds",builds}};
}
Json microbenchmark(){
    const auto d=fixture(6,6);auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingOracle oracle(d,paths);
    std::vector<double> pi(f.matrix.rows.size(),0);for(int row:f.order_rows)pi[row]=10;
    for(std::size_t i=0;i<f.flow.size();++i)if(i%7==0)f.flow[i]=0;
    LabelPricingOptions opt;opt.max_columns=512;opt.branch_correction=true;
    Json rounds=Json::array();double old_total=0,fast_total=0;
    for(int round=0;round<4;++round){
        LabelPricingResult old,fast;double old_seconds=0,fast_seconds=0;
        for(int side=0;side<2;++side){
            opt.fast_pricing=(side+round)%2==1;const double start=wall_time();const auto got=oracle.price(f,pi,{}, {},opt);const double seconds=wall_time()-start;
            if(opt.fast_pricing){fast=got;fast_seconds=seconds;}else{old=got;old_seconds=seconds;}
        }
        equivalent(old,fast);old_total+=old_seconds;fast_total+=fast_seconds;
        rounds.push_back({{"reference_seconds",old_seconds},{"fast_seconds",fast_seconds},
            {"reference_physical_builds",old.physical_path_builds},{"fast_physical_builds",fast.physical_path_builds},
            {"catalogue_path_reuses",fast.catalogue_path_reuses}});
    }
    return {{"passed",true},{"catalogue_paths",paths.service_paths.size()},{"rounds",rounds},
        {"reference_seconds",old_total},{"fast_seconds",fast_total},{"ratio",old_total/fast_total},
        {"timing_is_diagnostic_not_assertion",true},{"optimizer_calls",0}};
}

}
int main(){try{std::cout<<Json({{"passed",true},{"first_slice",first_slice()},{"differential",differential_slice()},{"microbenchmark",microbenchmark()}}).dump(2)<<'\n';return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
