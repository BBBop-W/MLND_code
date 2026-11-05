#pragma once
#include "mlnd/heuristics.hpp"
#include <map>
#include <random>
#include <set>

namespace mlnd {
// Trusted CG observations are ranking hints only. The producer must have
// checked negative reduced costs at the CURRENT ORIGINAL dual, and positive
// flows in a validated original-model primal point. Neither is a certificate
// for a later ALNS neighborhood or another design.
struct PrimalCgColumn {
    int path=-1;
    double reduced_cost=std::numeric_limits<double>::quiet_NaN(), validated_flow=0;
    std::string path_identity;
};
struct PrimalCgFeedback {
    std::string domain_identity;
    std::vector<double> source_design;
    std::vector<PrimalCgColumn> service;
};
// Capture exact input/support and path payload identities. The returned batch
// has at most min(max_columns,64) entries, balancing the two evidence types.
PrimalCgFeedback make_primal_cg_feedback(const Input&,const PathCollection&,
    const std::vector<double>& source_design,
    const std::vector<std::pair<int,double>>& original_negative_service,
    const std::vector<std::pair<int,double>>& validated_positive_service,
    std::size_t max_columns=64);
struct PrimalResources {
    std::set<int> bases, lockers;
    std::set<std::pair<int,int>> airlines;
};
// A sparse support view of an independently validated incumbent. inspect()
// checks encoding and identity, but is not a substitute for physical validation.
// Cache this view with the incumbent; no catalogue scan is needed per neighbor.
struct PrimalSupportState {
    bool ready=false;
    std::string status;
    std::vector<double> design, locker_load;
    // The historical head fields now index independent stable task IDs q.
    std::vector<int> head_state, positive_service, positive_empty;
    std::vector<PrimalResources> head_resources;
    const void* owner=nullptr;
};
struct PrimalBlockRequest {
    int operator_index=0;
    // Independent task IDs; no same-first-order bundle is implied.
    std::vector<int> heads, allowed_bases;
    PrimalRepairRestrictions restrictions;
    std::vector<int> extra_service, extra_empty;
    std::size_t paths_per_branch=4, optional_column_limit=5000;
    bool complete_missing_support=false;
    bool indexed_support_completion=false;
    Json diagnostics;
};
struct PrimalSupportBlock {
    bool ready=false;
    std::string status;
    CoupledPrimalNeighborhood coupled;
    PrimalRepairRestrictions restrictions;
    std::vector<int> service_indices, empty_indices;
    Json diagnostics;
};

// Bank lifetime is bound to immutable model data and append-only physical
// paths. A changed input, scope or selected canonical path is rejected. Only
// physical support/rankings are cached: no feasibility or optimality verdicts.
class PrimalSupportBank {
public:
    PrimalSupportBank(const Input&, const PathCollection&);
    void extend(double deadline,bool index_paths=false);
    Json observe_cg_columns(const PrimalCgFeedback&,double deadline);
    Json metadata() const;
    PrimalSupportState inspect(const Json& independently_validated, double deadline) const;
    PrimalBlockRequest seed(int direction, const PrimalSupportState&, std::mt19937&,
                           double deadline) const;
    // Operators: activation, sharing, close/swap, congestion, base balance,
    // elite differences. guide is a heuristic strategic point only.
    PrimalBlockRequest request(int operator_index, const PrimalSupportState&,
                              std::size_t block_size, std::mt19937&, double deadline,
                              const std::vector<double>* guide=nullptr,
                              const PrimalSupportState* elite=nullptr) const;
    PrimalSupportBlock close(const PrimalSupportState&, const PrimalBlockRequest&,
                            double deadline) const;
private:
    using Branch=std::array<int,3>;
    struct RankedPath {int id;double operating,fixed,load;int return_base;};
    const Input& d_;
    const PathCollection& paths_;
    std::string input_key_,scope_;
    std::size_t scanned_=0,empty_count_=0;
    bool supported_=false;
    std::map<Branch,std::vector<RankedPath>> branches_;
    // Full registered-ID index, distinct from the small ranked bank. Each
    // vector is in append/ID order; zero-rate branches are indexed as well.
    mutable std::map<Branch,std::vector<int>> complete_branches_;
    mutable std::size_t indexed_=0;
    std::vector<PrimalCgFeedback> feedback_;
    std::string feedback_domain_identity_;
    std::size_t feedback_observations_=0,feedback_columns_received_=0;
    std::map<int,std::string> retained_keys_;
    std::vector<std::string> empty_keys_;
    mutable std::set<int> checked_in_operation_;
    void validate(const std::string* feedback_identity=nullptr) const;
    void extend_index(double deadline) const;
    std::vector<int> required(int head) const;
    std::vector<int> shortlist(int head,int base,std::size_t limit,
                               const PrimalSupportState* state=nullptr) const;
};
} // namespace mlnd
