// Public pricing certificates, with independent worked branch-cost witnesses.
// This probe performs no solver optimization.
#include "mlnd/pricing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool good,const char* message) {if(!good)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int columns,T value) {
    Matrix<T> m(rows,columns);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.name="branch_correction_worked_witness";d.model_version="consecutive";
    d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1.75,9.75};d.single_rates={1,9};d.head_rates={1.5,9.25};
    d.pair_rates=filled(2,2,0.0);d.pair_rates(0,1)=.5;d.pair_rates(1,0)=.25;
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);d.link_cost=filled(4,4,.01);
    for(int u=0;u<4;++u){d.flight_times(u,u)=0;d.allowed_links(u,u)=0;}
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,0.0);d.terminal_cost=filled(2,2,0.0);
    d.receipt_mean=filled(2,2,.08);d.removal_mean=filled(2,2,.03);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=0;d.waiting_cost=0;return d;
}
std::vector<double> worked_prices(const PathCollection& paths,const Formulation& f) {
    std::vector<double> prices(f.matrix.rows.size(),0);
    const double single[2][2]={{10,3},{1,2}};
    const double pair[2][2]={{2,4},{5,.5}};
    for(const auto& p:paths.service_paths) {
        const int row=p.second_od<0?f.single_rows[p.segment_id]:f.double_rows[p.segment_id*2+p.second_od];
        prices[row]=p.second_od<0?-single[p.first_od][p.b[0]]:
            p.second_od!=p.first_od?-pair[p.first_od][p.b[0]]:0;
    }
    return prices;
}
Json worked_witness() {
    const auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);
    LabelPricingOracle oracle(d,paths);const auto prices=worked_prices(paths,f);
    LabelPricingOptions options;options.max_columns=1;
    const auto old=oracle.price(f,prices,{}, {},options);
    options.branch_correction=true;
    const auto improved=oracle.price(f,prices,{}, {},options);
    require(old.complete&&improved.complete,"Worked pricing domain did not complete");
    require(std::abs(old.correction-115)<1e-10,"Old worked correction must be 11.5 times 10");
    // Singles choose base0/base1; pairs choose base1/base0 independently.
    // The envelope is 1*10 + 9*2 + .5*4 + .25*5 = 31.25.
    require(improved.branch_correction_complete&&improved.branch_correction_used,
            "Complete branch correction was not selected");
    require(improved.correction>=31.25&&improved.correction<31.25+1e-8,
            "Branch correction disagrees with the hand-derived 31.25 envelope");
    require(improved.correction<old.correction-80,"Expected strict certificate improvement was lost");
    require(improved.service==old.service&&improved.empty==old.empty,
            "Certificate strengthening changed the bounded negative-column batch");
    return {{"passed",true},{"old_correction",old.correction},{"new_correction",improved.correction},
        {"strict_bound_gain",old.correction-improved.correction},
        {"catalogue_paths",paths.service_paths.size()},{"scanned_paths",improved.branch_paths_scanned},
        {"labels_dominated",improved.labels_dominated}};
}
double assignment_reference(const Input& d,const PathCollection& paths,const Formulation& f,
        const std::vector<double>& prices,const std::vector<double>& upper,const LabelPricingOptions& options) {
    // Independently enumerate all ground/base0/base1 assignments of positive-rate tasks.
    // For each assignment let every positive-rate branch use its cheapest
    // omitted column, ignoring network restrictions to obtain an upper envelope.
    const std::set<int> extra(options.excluded_service.begin(),options.excluded_service.end());
    std::vector<int> tasks;for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)tasks.push_back(q);
    int assignments=1;for(int q:tasks){(void)q;assignments*=d.n_l+1;}
    long double worst=0;
    for(int assignment=0;assignment<assignments;++assignment) {
        int state=assignment;long double violation=0;
        for(int q:tasks) {
            const int r=d.task_first(q),second=d.task_second(q);
            const int base=state%(d.n_l+1)-1;state/=d.n_l+1;
            if(base<0)continue;
            {
                const double rate=d.task_rate(q);
                double best=0;
                for(std::size_t i=0;i<paths.service_paths.size();++i) {
                    const auto& p=paths.service_paths[i];
                    if(p.first_od!=r||p.b[0]!=base||p.second_od!=second||f.flow[i]>=0||extra.count(static_cast<int>(i)))continue;
                    double rc=p.cost+(upper.empty()?0:upper[i]);
                    for(const auto& [row,a]:path_column(f,p))rc-=a*prices[row];
                    best=std::min(best,rc);
                }
                violation-=static_cast<long double>(rate)*best;
            }
        }
        worst=std::max(worst,violation);
    }
    return static_cast<double>(worst);
}
Json exclusions_and_reference() {
    const auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);
    const auto prices=worked_prices(paths,f);LabelPricingOracle oracle(d,paths);
    LabelPricingOptions options;options.branch_correction=true;options.max_columns=2;
    std::vector<double> upper(paths.service_paths.size(),0);
    for(std::size_t i=0;i<paths.service_paths.size();++i) {
        const auto& p=paths.service_paths[i];
        if(p.first_od==0&&p.b[0]==0)f.flow[i]=0;
        if(p.first_od==1&&p.b[0]==1&&p.second_od==0)options.excluded_service.push_back(static_cast<int>(i));
        if(p.first_od==1&&p.b[0]==1&&p.second_od<0)upper[i]=100;
    }
    const auto got=oracle.price(f,prices,upper,{},options);
    // Exclude head0/base0 and head1/base1/pair; upper multipliers remove the
    // remaining head1/base1/single advantage: 5 + (9*1+.25*5) = 15.25.
    require(got.complete&&got.branch_correction_complete,"Exclusion certificate incomplete");
    require(got.branch_service_correction>=15.25&&got.branch_service_correction<15.25+1e-8,
            "RMP/external exclusions or omitted upper multipliers were ignored");
    for(int id:got.service)require(f.flow[id]<0&&std::find(options.excluded_service.begin(),options.excluded_service.end(),id)==options.excluded_service.end(),
                                "Excluded column returned");
    std::mt19937 rng(923);std::uniform_real_distribution<double> random(-8,8);
    std::size_t dominated=got.labels_dominated;int strict=0;
    constexpr int rounds=12;
    for(int round=0;round<rounds;++round) {
        std::vector<double> pi(f.matrix.rows.size());for(auto& value:pi)value=random(rng);
        options.excluded_service.clear();options.excluded_empty.clear();options.max_columns=round%3+1;
        for(std::size_t i=0;i<f.flow.size();++i) {
            f.flow[i]=rng()%5==0?0:-1;upper[i]=std::abs(random(rng));
            if(rng()%7==0)options.excluded_service.push_back(static_cast<int>(i));
        }
        std::vector<double> empty_upper(paths.empty_paths.size());
        for(std::size_t i=0;i<f.empty_flow.size();++i) {
            f.empty_flow[i]=rng()%3==0?0:-1;empty_upper[i]=std::abs(random(rng));
            if(rng()%5==0)options.excluded_empty.push_back(static_cast<int>(i));
        }
        const double reference=assignment_reference(d,paths,f,pi,upper,options);
        const auto branch=oracle.price(f,pi,upper,empty_upper,options);
        options.branch_correction=false;
        const auto old=oracle.price(f,pi,upper,empty_upper,options);
        options.branch_correction=true;
        require(branch.complete&&branch.branch_correction_complete,"Random complete catalogue scan incomplete");
        require(branch.branch_service_correction+1e-11>=reference&&branch.branch_service_correction<reference+1e-8,
                "Branch envelope disagrees with independent assignment enumeration");
        require(branch.correction<=old.correction&&branch.correction+1e-10>=std::min(reference,old.old_service_correction)+old.empty_correction,
                "Chosen correction lost safety or dominance over legacy bound");
        require(branch.empty_correction==old.empty_correction,"Branch strengthening changed empty correction");
        require(branch.service==old.service&&branch.empty==old.empty&&branch.min_service==old.min_service&&branch.min_empty==old.min_empty,
                "Supplemental scan changed ordinary pricing semantics");
        require(!old.branch_correction_requested&&old.branch_paths_scanned==0&&old.branch_correction_status=="disabled",
                "Disabled option still scanned or changed default status");
        dominated+=branch.labels_dominated;strict+=branch.correction<old.correction;
    }
    require(dominated>0,"Reference tests did not exercise label dominance");
    return {{"passed",true},{"worked_exclusion_correction",got.branch_service_correction},
        {"reference_rounds",rounds},{"assignment_states_per_round",81},{"strict_gain_rounds",strict},{"labels_dominated",dominated}};
}
Json positive_rate_cross_head_witness() {
    auto d=fixture();d.pair_rates(0,0)=1;d.pair_rates(1,1)=1;
    for(int r=0;r<2;++r){d.demand[r]+=2;d.head_rates[r]+=1;}
    auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);auto prices=worked_prices(paths,f);
    for(const auto& p:paths.service_paths)if(p.second_od==p.first_od)
        prices[f.double_rows[p.segment_id*2+p.second_od]]=p.first_od==0?-2:-1;
    LabelPricingOracle oracle(d,paths);LabelPricingOptions options;options.branch_correction=true;options.max_columns=1;
    const auto got=oracle.price(f,prices,{}, {},options);
    // At second-OD0, otherwise physically identical head1 pair labels cost -5
    // (base0), dominating head0 AA labels at -2. At second-OD1, head0 pair
    // labels cost -2/-4, dominating head1 AA labels at -1. All these AA branches
    // now have positive rates, so surviving global minima cannot certify them.
    // Independent tasks add AA contributions 2+1 to the 31.25 envelope.
    const double independent=assignment_reference(d,paths,f,prices,{},options);
    require(got.complete&&got.branch_correction_complete&&got.labels_dominated>0,
            "Positive-rate cross-head dominance witness did not complete");
    require(independent==34.25&&got.branch_service_correction>=34.25&&got.branch_service_correction<34.25+1e-8,
            "A dominated positive-rate AA branch was omitted from its correction");
    return {{"passed",true},{"reference_correction",independent},{"branch_correction",got.branch_service_correction},
        {"labels_dominated",got.labels_dominated},{"old_correction",got.old_service_correction}};
}
Json fallbacks_and_empty() {
    auto d=fixture();auto paths=enumerate_paths(d);const std::vector<int> none;
    auto f=build_formulation(d,paths,nullptr,&none,&none);LabelPricingOracle oracle(d,paths);
    auto prices=worked_prices(paths,f);LabelPricingOptions options;
    const auto old=oracle.price(f,prices,{}, {},options);
    options.branch_correction=true;options.branch_deadline=wall_time()-1;
    const auto exhausted=oracle.price(f,prices,{}, {},options);
    require(exhausted.complete&&!exhausted.branch_correction_complete&&!exhausted.branch_correction_used&&exhausted.correction==old.correction&&
            exhausted.branch_correction_status=="budget_exhausted"&&exhausted.branch_paths_scanned==0,
            "Supplemental timeout discarded or replaced the complete ordinary certificate");
    options.deadline=wall_time()-1;
    const auto incomplete=oracle.price(f,prices,{}, {},options);
    require(!incomplete.complete&&!incomplete.branch_correction_complete&&incomplete.branch_correction_status=="label_incomplete",
            "Incomplete ordinary label search certified branch minima");
    options.deadline=options.branch_deadline=std::numeric_limits<double>::infinity();
    paths.complete=false;
    const auto partial_catalogue=oracle.price(f,prices,{}, {},options);
    require(partial_catalogue.complete&&!partial_catalogue.branch_correction_complete&&partial_catalogue.correction==old.correction&&
            partial_catalogue.branch_correction_status=="incomplete_catalogue",
            "Incomplete path catalogue was certified or destroyed old label bound");
    paths.complete=true;
    // Unsupported modes remain fully usable through the ordinary oracle.
    int scopes=0;
    for(int mode=0;mode<3;++mode) {
        auto other=fixture();
        if(mode==0)other.preprocessed_demand=false;
        if(mode==1){other.preprocessed_demand=false;other.flow_policy="segment";}
        if(mode==2)other.first_leg_consistency=true;
        auto catalogue=enumerate_paths(other);
        if(mode==2) {
            bool rejected=false;try{build_formulation(other,catalogue,nullptr,&none,&none);}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"Unsupported first-leg coupling was accepted");++scopes;continue;
        }
        auto formulation=build_formulation(other,catalogue,nullptr,&none,&none);
        LabelPricingOracle unsupported(other,catalogue);
        std::vector<double> pi(formulation.matrix.rows.size(),0);for(int row:formulation.order_rows)pi[row]=3;
        options.branch_correction=false;const auto legacy=unsupported.price(formulation,pi,{}, {},options);
        options.branch_correction=true;const auto result=unsupported.price(formulation,pi,{}, {},options);
        require(result.complete&&!result.branch_correction_complete&&result.branch_correction_status=="unsupported_scope"&&
                result.correction==legacy.correction&&result.branch_paths_scanned==0,
                "Unsupported formulation received a new correction or changed legacy certificate");++scopes;
    }
    prices[f.balance_rows[0]]=3;
    options.branch_correction=false;const auto with_empty_old=oracle.price(f,prices,{}, {},options);
    options.branch_correction=true;const auto with_empty=oracle.price(f,prices,{}, {},options);
    require(with_empty.complete&&with_empty.branch_correction_complete&&with_empty.empty_correction==69&&
            with_empty.empty_correction==with_empty_old.empty_correction,
            "Positive empty relocation correction was not preserved");
    std::fill(f.flow.begin(),f.flow.end(),0);std::fill(f.empty_flow.begin(),f.empty_flow.end(),0);
    const auto full=oracle.price(f,prices,{}, {},options);
    require(full.complete&&full.branch_correction_complete&&full.correction==0&&!full.branch_correction_used&&full.service.empty()&&full.empty.empty(),
            "Fully represented RMP received a nonzero correction");
    return {{"passed",true},{"unsupported_scopes",scopes},{"timeout_fallback_retained_correction",exhausted.correction},
        {"preserved_positive_empty_correction",with_empty.empty_correction},{"full_rmp_correction",full.correction}};
}
}
int main() {
    try {std::cout<<Json{{"worked_witness",worked_witness()},{"exclusions_and_reference",exclusions_and_reference()},
                       {"positive_rate_cross_head_witness",positive_rate_cross_head_witness()},
                       {"fallbacks_and_empty",fallbacks_and_empty()}}.dump(2)<<'\n';return 0;}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
