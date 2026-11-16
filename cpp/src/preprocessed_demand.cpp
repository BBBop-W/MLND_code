#include "mlnd/preprocessed_demand.hpp"
#include "mlnd/progress.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace mlnd {
std::vector<int> first_leg_key(const InitialSegment& segment) {
    if(segment.legs.size()<2)
        throw std::invalid_argument("First-leg grouping requires both first-delivery route legs");
    std::vector<int> key{segment.b[0],segment.first_od,segment.pickup,segment.delivery};
    for(int leg=0;leg<2;++leg) {
        key.push_back(static_cast<int>(segment.legs[leg].size()));
        key.insert(key.end(),segment.legs[leg].begin(),segment.legs[leg].end());
    }
    return key;
}

Json first_leg_support(const Input& d,const PathCollection& paths) {
    if(!d.preprocessed_demand || !paths.complete)
        throw std::invalid_argument("First-leg support requires fixed demand and the complete catalogue");
    std::map<std::vector<int>,std::vector<int>> branches;
    for(const auto& segment:paths.initial_segments)
        branches.emplace(first_leg_key(segment),std::vector<int>(d.n_od+1));
    for(const auto& p:paths.service_paths)
        branches.at(first_leg_key(paths.initial_segments.at(p.segment_id)))[p.second_od+1]=1;
    Matrix<int> available(d.n_od,d.n_l),complete(d.n_od,d.n_l);
    int complete_count=0,positive_heads=0,supported_heads=0;
    for(const auto& [key,support]:branches) {
        const int l=key[0],r=key[1];++available(r,l);
        bool valid=d.single_rates[r]<=0 || support[0];
        for(int s=0;s<d.n_od;++s)
            if(d.pair_rates(r,s)>0 && !support[s+1])valid=false;
        if(valid) { ++complete(r,l);++complete_count; }
    }
    Json counts=Json::array(),total=Json::array();
    for(int r=0;r<d.n_od;++r) {
        Json row=Json::array(),all=Json::array();bool supported=false;
        for(int l=0;l<d.n_l;++l) {
            row.push_back(complete(r,l));all.push_back(available(r,l));
            supported=supported||complete(r,l)>0;
        }
        counts.push_back(std::move(row));total.push_back(std::move(all));
        if(d.head_rates[r]>0) { ++positive_heads;if(supported)++supported_heads; }
    }
    return {{"grouping","departure_first_od_pickup_delivery_first_two_route_legs"},
        {"return_base_shared",false},{"first_leg_groups",branches.size()},
        {"complete_first_leg_groups",complete_count},
        {"groups_by_od_origin",total},{"complete_groups_by_od_origin",counts},
        {"positive_head_ods",positive_heads},{"supported_positive_head_ods",supported_heads},
        {"unsupported_positive_head_ods",positive_heads-supported_heads},
        {"demand_rates_recomputed",false}};
}

void preprocess_generalized_demand(Input& d, PathCollection& paths, double deadline) {
    if ((d.max_orders != 1 && d.max_orders != 2) || d.flow_policy != "aggregate")
        throw std::invalid_argument("Preprocessed demand requires K=1 or K=2 and aggregate flow policy");
    if (!paths.support_complete)
        throw std::invalid_argument("Preprocessed demand requires complete initial-segment support");
    if(std::isnan(deadline))throw std::invalid_argument("Preprocessing deadline must not be NaN");
    paths.domain_complete=false;
    paths.physical_service_count=0;
    d.preprocessed_demand_report=Json::object();
    std::size_t physical_count=0,operations=0;
    auto guard=[&](bool force=false) {
        if(force || ++operations%256==0) {
            const double now=std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if(now>=deadline)throw EnumerationStopped("time_limit",physical_count,paths.initial_segments.size());
        }
        model_construction_guard("demand_preprocessing",0,0,0,force);
    };
    guard(true);
    if (d.max_orders == 2) for (double value : d.kappa.values) {
        guard();
        if (value != 0 && value != 1)
            throw std::invalid_argument("Preprocessed demand prototype requires binary kappa");
    }
    const int R=d.n_od, L=d.n_l;
    const double infinity=std::numeric_limits<double>::infinity();
    std::vector<double> single_cost(R,infinity), pair_cost(R*R,infinity);
    Matrix<int> single_support(R,L), pair_support(R*R,L);
    physical_count=visit_service_paths(d,paths,[&](const ServicePath& p) {
        guard();
        if (p.second_od<0) {
            single_cost[p.first_od]=std::min(single_cost[p.first_od],p.cost);
            single_support(p.first_od,p.b[0])=1;
        } else {
            const int pair=p.first_od*R+p.second_od;
            pair_cost[pair]=std::min(pair_cost[pair],p.cost);
            pair_support(pair,p.b[0])=1;
        }
    },deadline);
    struct Candidate { int r,s; double score; };
    std::vector<Candidate> candidates;
    if(d.max_orders==2) for(int r=0;r<R;++r) for(int s=0;s<R;++s) {
        guard();
        if(d.kappa(r,s)==0 || !std::isfinite(single_cost[r]) ||
           !std::isfinite(single_cost[s]) || !std::isfinite(pair_cost[r*R+s])) continue;
        const double saving=single_cost[r]+single_cost[s]-pair_cost[r*R+s];
        if(saving>0) candidates.push_back({r,s,saving});
    }
    std::sort(candidates.begin(),candidates.end(),[&](const Candidate& a,const Candidate& b) {
        guard();
        if(a.score!=b.score) return a.score>b.score;
        return std::tie(a.r,a.s)<std::tie(b.r,b.s);
    });
    d.single_rates=d.demand;
    d.pair_rates=Matrix<double>(R,R);
    d.common_origin_support=single_support;
    Json order=Json::array(),trace=Json::array();
    std::size_t accepted=0, support_rejected=0;
    for(const auto& c:candidates) {
        guard();
        order.push_back({c.r,c.s});
        const double ar=d.single_rates[c.r],as=d.single_rates[c.s];
        const double q=-std::expm1(-as*d.effective_delta());
        const double rate=(c.r==c.s ? ar/2 : std::min(ar,as))*q;
        Json step={{"r",c.r},{"s",c.s},{"score",c.score},
                   {"first_residual",ar},{"second_residual",as},
                   {"probability",q},{"q",q},{"proposed_rate",rate},{"rate",0.0},{"gamma",0.0}};
        if(rate==0) { step["status"]="zero_residual"; trace.push_back(std::move(step)); continue; }
        std::vector<int> support(L);
        for(int l=0;l<L;++l)
            support[l]=d.common_origin_support(c.r,l) && pair_support(c.r*R+c.s,l);
        if(std::none_of(support.begin(),support.end(),[](int v){return v!=0;})) {
            ++support_rejected; step["status"]="no_common_origin";
            trace.push_back(std::move(step)); continue;
        }
        d.pair_rates(c.r,c.s)=rate;
        d.single_rates[c.r]-=rate;
        d.single_rates[c.s]-=rate;
        for(int l=0;l<L;++l)d.common_origin_support(c.r,l)=support[l];
        ++accepted;step["status"]="accepted";step["rate"]=rate;step["gamma"]=rate;
        step["common_origin_support"]=support;
        trace.push_back(std::move(step));
    }
    d.head_rates=d.single_rates;
    double max_residual=0,pair_total=0;
    for(int r=0;r<R;++r) {
        guard();
        double reconstructed=d.single_rates[r];
        if(d.single_rates[r]<0 || !std::isfinite(d.single_rates[r]))
            throw std::runtime_error("Preprocessing produced an invalid residual demand");
        for(int s=0;s<R;++s) {
            guard();
            reconstructed+=d.pair_rates(r,s)+d.pair_rates(s,r);
            d.head_rates[r]+=d.pair_rates(r,s);
            pair_total+=d.pair_rates(r,s);
        }
        max_residual=std::max(max_residual,std::abs(reconstructed-d.demand[r]));
    }
    const double total=std::accumulate(d.demand.begin(),d.demand.end(),0.0);
    if(max_residual>1e-10*(1+total))
        throw std::runtime_error("Preprocessed demand failed raw-order conservation");
    Json pairs=Json::array(),support=Json::array();
    for(int r=0;r<R;++r) {
        guard();
        Json row=Json::array(), origins=Json::array();
        for(int s=0;s<R;++s) { guard();row.push_back(d.pair_rates(r,s)); }
        for(int l=0;l<L;++l)origins.push_back(d.common_origin_support(r,l));
        pairs.push_back(std::move(row));support.push_back(std::move(origins));
    }
    d.preprocessed_demand=true;
    Json report={
        {"method","residual_pair_opportunity_v1"},
        {"ranking","descending_min_single_cost_sum_minus_min_double_cost_then_lexicographic"},
        {"positive_proxy_saving_only",true},{"proxy_scope","global_minima_over_all_candidate_facilities"},
        {"optimal_matching_claimed",false},{"binary_kappa_required",d.max_orders==2},
        {"max_task_orders",d.max_orders},{"assignment_scope",d.assignment_scope()},
        {"common_departure_support_scope","task_reconstruction_only"},
        {"common_departure_support_required",true},{"delta_hours",d.effective_delta()},
        {"single_rates",d.single_rates},{"pair_rates",pairs},{"head_rates",d.head_rates},
        {"common_origin_support",support},{"candidate_order",order},{"processing_trace",trace},
        {"eligible_positive_saving_pairs",candidates.size()},{"accepted_pairs",accepted},
        {"support_rejected_pairs",support_rejected},{"max_conservation_error",max_residual},
        {"raw_order_rate",total},{"double_task_rate",pair_total},
        {"single_task_rate",std::accumulate(d.single_rates.begin(),d.single_rates.end(),0.0)},
        {"task_rate",total-pair_total},
        {"approximation","Poisson residual opportunities and effective min-rate matching; rates conserve mean OD demand"}};
    guard(true);
    d.preprocessed_demand_report=std::move(report);
    paths.domain_complete=true;
    paths.physical_service_count=physical_count;
}
} // namespace mlnd
