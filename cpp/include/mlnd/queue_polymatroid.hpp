#pragma once

#include "mlnd/task_savings.hpp"

namespace mlnd {

struct QueueBranchArcPrice {
    // Physical first/second OD indices; convert with Input::task_index for alpha.
    int head = -1, origin = -1, second = -1, arc = -1;
    double price = 0;
};

// Prices multiply original <= resource rows in a minimization Lagrangian.
// Balance potentials are free; every other price must be nonnegative.
struct QueueResourcePrices {
    std::vector<double> locker, airline, balance;
    std::vector<QueueBranchArcPrice> branch_gates;
    std::string source;
};

struct QueueNativePrices {
    bool valid = false;
    QueueResourcePrices prices;
    std::string status;
};
// Read already available native minimization row multipliers; never optimize.
QueueNativePrices native_queue_resource_prices(const Input&, const Formulation&,
                                               const std::vector<double>& row_prices);

struct QueuePriceSearchResult {
    TaskSavingsCut cut;
    Json diagnostics;
};

// Complete-domain, integer-valid queue cuts. Profile zero is the nonnegative
// S+ EPI-0 bound. Other profiles retain signed savings, missing-branch gates,
// and finite negative-cost empty flows. Ordinary per-base separation uses no
// auxiliary optimization; general-L joint separation solves a coefficient LP.
class QueuePolymatroidCuts {
public:
    QueuePolymatroidCuts(const Input&, const PathCollection&, const SolverOptions&,
                        const TaskSavingsData* baseline = nullptr,
                        std::size_t max_profiles = 8);
    // Atomic: false on a duplicate, exhausted budget, or unusable numerics;
    // a partial scan never changes the retained profile pool.
    bool add_profile(const QueueResourcePrices&, double seconds);
    TaskSavingsCut separate(const std::vector<double>& design,
                            double theta_value, int theta);
    TaskSavingsCut separate_profile(std::size_t profile,
                                   const std::vector<double>& design,
                                   double theta_value, int theta) const;
    // Joint two-task, (L+1)-state convexification for fixed K2 tasks.
    // L2 preserves facet enumeration; other L use a single-thread coefficient
    // LP. Uses retained complete profiles, without rescanning physical paths.
    // Only a fully state-validated, conservatively lifted row can be returned.
    TaskSavingsCut separate_joint(const std::vector<double>& design,
                                 double theta_value, int theta, double seconds);
    QueuePriceSearchResult search_prices(const QueueResourcePrices& candidate,
        const std::vector<double>& design, double theta_value, int theta,
        double seconds, std::size_t max_evaluations = 7);
    std::size_t profile_count() const { return profiles_.size(); }
    const Json& metadata() const { return metadata_; }
private:
    struct Branch {
        int head = 0, origin = 0, second = -1;
        double rate = 0;
        std::size_t paths = 0;
    };
    struct Profile {
        std::size_t id = 0;
        std::string kind, source, key;
        std::vector<double> savings, lower_coefficients;
        std::vector<bool> unsupported;
        double constant = 0;
        QueueResourcePrices prices;
        Json certificate;
    };
    bool complete_domain_unchanged() const;
    bool prepare_profile(const QueueResourcePrices&, double seconds, Profile&,
                         std::string& status, bool reject_duplicates) const;
    TaskSavingsCut separate_prepared(const Profile&, std::size_t index,
        const std::vector<double>& design, double theta_value, int theta) const;
    const Input& data_;
    const PathCollection& paths_;
    DesignLayout layout_;
    int heads_ = 0, bases_ = 0, nodes_ = 0;
    double fleet_cost_ = 0, courier_cost_ = 0, margin_ = 0, total_demand_ = 0;
    std::size_t service_count_ = 0, empty_count_ = 0, max_profiles_ = 8, next_id_ = 1;
    // Exact input/support identity; independent of the generated-column pool.
    std::string domain_identity_;
    std::vector<double> airline_upper_;
    std::vector<Branch> branches_;
    std::vector<int> branch_lookup_;
    std::vector<bool> unsupported_;
    std::vector<Profile> profiles_;
    Json metadata_;
};

} // namespace mlnd
