// Differential checks for the complete-catalogue pricing backend. No native
// optimization or experimental output is used by this probe.
#include "mlnd/pricing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>

using namespace mlnd;
namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
template<class F> void rejects(F&& action,const char* message) {
    bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
    require(rejected,message);
}
template<class T> Matrix<T> filled(int n,int m,T value) {
    Matrix<T> result(n,m);std::fill(result.values.begin(),result.values.end(),value);return result;
}
Input fixture() {
    Input d;d.name="catalogue_pricing";d.model_version="consecutive";
    d.n_l=2;d.n_h=3;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.single_rates={1,2};d.head_rates={1.5,2.5};d.demand={2,3};
    d.pair_rates=filled(2,2,.25);d.common_origin_support=filled(2,2,1);
    d.ground_cost={20,20};d.deadline={1,1};d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.flight_times=filled(5,5,.05);d.allowed_links=filled(5,5,1);d.link_cost=filled(5,5,.01);
    for(int i=0;i<5;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.rider_mean=filled(2,3,.08);d.first_cost=filled(2,3,1.0);d.terminal_cost=filled(2,3,.5);
    d.receipt_mean=filled(2,3,.08);d.removal_mean=filled(2,3,.03);d.slot_counts=filled(3,1,20);
    d.slot_cost=filled(3,1,.1);d.occupancy_limit=filled(3,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
double reduced(const Formulation& f,const ServicePath& p,const std::vector<double>& pi,double upper) {
    double rc=p.cost+upper;for(auto [row,a]:path_column(f,p))rc-=a*pi.at(row);return rc;
}
double reduced(const Formulation& f,const EmptyPath& p,const std::vector<double>& pi,double upper) {
    double rc=p.cost+upper;for(auto [row,a]:path_column(f,p))rc-=a*pi.at(row);return rc;
}
void verify(const PathCollection& paths,const Formulation& f,const std::vector<double>& pi,
        const std::vector<double>& us,const std::vector<double>& ue,const LabelPricingOptions& options,
        const LabelPricingResult& result) {
    require(result.complete&&result.catalogue_pricing_used,"Catalogue scan did not complete");
    double min_service=0,min_empty=0,best_returned=0;
    std::size_t negatives=0,omitted=0;
    const std::set<int> es(options.excluded_service.begin(),options.excluded_service.end());
    const std::set<int> ee(options.excluded_empty.begin(),options.excluded_empty.end());
    for(const auto& p:paths.service_paths) {
        if(!service_path_enabled(f,p)||(static_cast<std::size_t>(p.id)<f.flow.size()&&f.flow[p.id]>=0)||es.count(p.id))continue;
        const double rc=reduced(f,p,pi,static_cast<std::size_t>(p.id)<us.size()?us[p.id]:0);
        min_service=std::min(min_service,rc);negatives+=rc < -options.tolerance;++omitted;
    }
    for(const auto& p:paths.empty_paths) {
        if((static_cast<std::size_t>(p.id)<f.empty_flow.size()&&f.empty_flow[p.id]>=0)||ee.count(p.id))continue;
        const double rc=reduced(f,p,pi,static_cast<std::size_t>(p.id)<ue.size()?ue[p.id]:0);
        min_empty=std::min(min_empty,rc);negatives+=rc < -options.tolerance;++omitted;
    }
    require(result.min_service==min_service&&result.min_empty==min_empty&&
            result.minimum==std::min(min_service,min_empty),"Canonical scan minimum mismatch");
    require(result.negative_paths==negatives&&result.omitted_paths==omitted,"Full scan counts mismatch");
    require(result.catalogue_paths_scanned==paths.service_paths.size()+paths.empty_paths.size(),"Scan missed catalogue entries");
    require(result.service.size()+result.empty.size()<=options.max_columns,"Returned batch exceeds cap");
    std::set<std::pair<bool,int>> selected;
    for(int id:result.service) {
        require(selected.emplace(false,id).second&&service_path_enabled(f,paths.service_paths.at(id))&&
                (static_cast<std::size_t>(id)>=f.flow.size()||f.flow[id]<0)&&!es.count(id),"Invalid returned service ID");
        const double rc=reduced(f,paths.service_paths.at(id),pi,static_cast<std::size_t>(id)<us.size()?us[id]:0);
        require(rc < -options.tolerance,"Returned service column is not negative");best_returned=std::min(best_returned,rc);
    }
    for(int id:result.empty) {
        require(selected.emplace(true,id).second&&(static_cast<std::size_t>(id)>=f.empty_flow.size()||f.empty_flow[id]<0)&&
                !ee.count(id),"Invalid returned empty ID");
        const double rc=reduced(f,paths.empty_paths.at(id),pi,static_cast<std::size_t>(id)<ue.size()?ue[id]:0);
        require(rc < -options.tolerance,"Returned empty column is not negative");best_returned=std::min(best_returned,rc);
    }
    if(negatives)require(best_returned==result.minimum,"Batch lost the most negative witness");
    else require(selected.empty(),"Zero-negative scan returned columns");
}
Json differential() {
    std::mt19937 random(20261105);std::uniform_real_distribution<double> value(-12,12);
    int checks=0;
    for(int variant=0;variant<9;++variant) {
        auto d=fixture();FormulationStrengthening strengthening;
        if(variant==1) {
            d.max_orders=1;std::fill(d.pair_rates.values.begin(),d.pair_rates.values.end(),0);
            d.head_rates=d.single_rates;d.demand=d.single_rates;
        }
        if(variant==2||variant==3) {
            d.single_rates[0]=0;d.pair_rates(0,0)=0;d.pair_rates(1,1)=1e-20;
            d.head_rates={.25,2.25};d.demand={.5,2.5};
            strengthening.eliminate_zero_task_columns=true;
            strengthening.compact_task_heads=variant==3;
        }
        if(variant==4||variant==5)d.preprocessed_demand=false;
        if(variant==5)d.flow_policy="segment";
        if(variant==6)d.first_leg_consistency=true;
        if(variant==7) {
            d.demand=d.single_rates=d.head_rates={0,0};
            std::fill(d.pair_rates.values.begin(),d.pair_rates.values.end(),0);
        }
        auto paths=enumerate_paths(d);const std::vector<int> none;
        if(variant==6) {
            bool rejected=false;try{build_formulation(d,paths,nullptr,&none,&none);}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"Unsupported first-leg coupling was accepted");continue;
        }
        auto f=build_formulation(d,paths,nullptr,&none,&none,false,false,false,strengthening);
        LabelPricingOracle label(d,paths),catalogue(d,paths,INFINITY,nullptr,true,true);
        require(catalogue.domain_identity()==pricing_domain_identity(d,paths),"Captured domain identity differs");
        for(int round=0;round<7;++round) {
            std::vector<double> pi(f.matrix.rows.size()),us(paths.service_paths.size()),ue(paths.empty_paths.size());
            for(auto& x:pi)x=value(random)*(variant==8?1e6:1);
            for(auto& x:us)x=std::abs(value(random));for(auto& x:ue)x=std::abs(value(random));
            LabelPricingOptions options;options.branch_correction=true;options.fast_pricing=true;
            options.max_columns=1+round*2;
            for(std::size_t i=0;i<f.flow.size();++i) {
                f.flow[i]=random()%5==0?0:-1;if(random()%13==0)options.excluded_service.push_back(static_cast<int>(i));
            }
            for(std::size_t i=0;i<f.empty_flow.size();++i) {
                f.empty_flow[i]=random()%3==0?0:-1;if(random()%7==0)options.excluded_empty.push_back(static_cast<int>(i));
            }
            if(round==4)options.branch_deadline=0;
            if(round==5) {std::fill(f.flow.begin(),f.flow.end(),0);std::fill(f.empty_flow.begin(),f.empty_flow.end(),0);}
            if(round==6) {us.resize(us.size()/2);ue.clear();}
            const auto original=label.price(f,pi,us,ue,options);
            options.catalogue_pricing=true;
            const auto scan=catalogue.price(f,pi,us,ue,options);
            verify(paths,f,pi,us,ue,options,scan);
            require(original.complete&&original.minimum==scan.minimum&&original.min_service==scan.min_service&&
                    original.min_empty==scan.min_empty,"Catalogue and label minima differ");
            require(original.correction==scan.correction&&original.branch_service_correction==scan.branch_service_correction&&
                    original.branch_correction_complete==scan.branch_correction_complete&&
                    original.branch_correction_status==scan.branch_correction_status,"Catalogue and label corrections differ");
            options.diverse_pricing=true;
            const auto diverse=catalogue.price(f,pi,us,ue,options);
            verify(paths,f,pi,us,ue,options,diverse);
            require(diverse.diverse_pricing_used&&diverse.correction==scan.correction&&
                    diverse.minimum==scan.minimum&&diverse.negative_paths==scan.negative_paths,
                    "Diverse selection changed full-domain certification");
            require(diverse.diversity_candidates<=4*options.max_columns,"Diversity pool exceeds 4B");
            const auto repeat=catalogue.price(f,pi,us,ue,options);
            require(repeat.service==diverse.service&&repeat.empty==diverse.empty,"Diverse selection is not deterministic");
            ++checks;
        }
    }
    return {{"passed",true},{"differential_rounds",checks},{"optimizer_calls",0}};
}
Json boundaries() {
    auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingGraphCache cache;
    LabelPricingOracle oracle(d,paths,INFINITY,&cache,true,true);
    require(cache.metadata().at("construction_attempts")==0,"Catalogue-preferred oracle built a useless graph");
    const auto snapshot=oracle.domain_identity();d.single_rates[0]+=.01;
    LabelPricingOracle changed(d,paths,INFINITY,nullptr,true,true);
    require(changed.domain_identity()!=snapshot,"Fresh same-address input mutation escaped identity capture");
    d.single_rates[0]-=.01;
    std::vector<double> pi(f.matrix.rows.size(),0);for(int row:f.order_rows)pi[row]=10;
    LabelPricingOptions options;options.catalogue_pricing=true;options.branch_correction=true;
    options.deadline=0;
    require(!oracle.price(f,pi,{}, {},options).complete,"Expired catalogue scan certified the domain");
    options.deadline=INFINITY;options.branch_deadline=0;
    const auto exhausted=oracle.price(f,pi,{}, {},options);
    require(exhausted.complete&&!exhausted.branch_correction_complete&&exhausted.branch_correction_status=="budget_exhausted"&&
            exhausted.correction==exhausted.old_service_correction+exhausted.empty_correction,
            "Independent branch deadline discarded ordinary certification");
    options.branch_deadline=INFINITY;
    auto bad_pi=pi;bad_pi[0]=INFINITY;
    rejects([&]{oracle.price(f,bad_pi,{}, {},options);},"Nonfinite row multiplier accepted");
    rejects([&]{oracle.price(f,pi,{-1}, {},options);},"Negative upper multiplier accepted");
    auto bad_options=options;bad_options.excluded_service={static_cast<int>(paths.service_paths.size())};
    rejects([&]{oracle.price(f,pi,{}, {},bad_options);},"Out-of-range exclusion accepted");
    bad_options=options;bad_options.deadline=NAN;
    rejects([&]{oracle.price(f,pi,{}, {},bad_options);},"NaN deadline accepted");
    const double cost=paths.service_paths[0].cost;
    paths.service_paths[0].cost=std::numeric_limits<double>::max();
    auto overflow_pi=pi;overflow_pi[f.fleet_row]=-std::numeric_limits<double>::max();
    rejects([&]{oracle.price(f,overflow_pi,{}, {},options);},"Finite-input reduced-cost overflow certified");
    paths.service_paths[0].cost=cost;
    // Switching back to labels on a graph-free oracle must remain available.
    options.catalogue_pricing=false;
    require(oracle.price(f,pi,{}, {},options).complete,"Deferred label graph fallback failed");
    // A materialized subset never substitutes for the physical domain.
    d.preprocessed_demand=false;auto implicit=prepare_path_support(d);
    auto implicit_f=build_formulation(d,implicit,nullptr,&none,&none);
    LabelPricingOracle fallback(d,implicit,INFINITY,nullptr,true);
    std::vector<double> implicit_pi(implicit_f.matrix.rows.size(),0);
    for(int row:implicit_f.order_rows)implicit_pi[row]=10;
    options.catalogue_pricing=options.diverse_pricing=true;
    const auto got=fallback.price(implicit_f,implicit_pi,{}, {},options);
    require(!implicit.complete&&got.complete&&!got.catalogue_pricing_used&&!got.diverse_pricing_used&&
            got.labels_created>0&&!got.service.empty(),"Incomplete catalogue did not use the exact label fallback");
    return {{"passed",true},{"deferred_graph",cache.metadata()},{"fallback_labels",got.labels_created}};
}
}
int main() {
    try {std::cout<<Json{{"passed",true},{"differential",differential()},{"boundaries",boundaries()}}.dump(2)<<'\n';return 0;}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
