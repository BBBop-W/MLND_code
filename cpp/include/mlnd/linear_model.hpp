#pragma once

#include "mlnd/input.hpp"
#include "mlnd/progress.hpp"
#include <ilcplex/ilocplex.h>
#include <array>
#include <memory>
#include <functional>
#include <optional>
#include <utility>

namespace mlnd {
using Coefficients = std::vector<std::pair<int, double>>;
struct VariableRecord {
    std::string name;
    double lower = 0, upper = IloInfinity, cost = 0;
    bool binary = false;
};
struct RowRecord {
    std::string name;
    char sense = 'E'; // E equality, L <=, G >=
    double rhs = 0;
    Coefficients coefficients;
};
struct ConeRecord {
    int up, down, one;
    double factor = 1; // one^2 <= factor * up * down
};
struct LinearModel {
    std::vector<VariableRecord> variables;
    std::vector<RowRecord> rows;
    std::vector<ConeRecord> cones;
    bool minimize = true;
    double constant = 0;
    int variable(std::string name, double lower, double upper,
                 double cost = 0, bool binary = false);
    int row(std::string name, char sense, double rhs, Coefficients coefficients = {});
    void coefficient(int row, int variable, double value);
};
struct SolverOptions {
    // Low-level legacy defaults stay stable. Versioned production settings
    // are selected through SolverConfiguration, not by changing these values.
    double seconds = 600, gap = 0.005;
    int threads = 1, seed = 43, max_iterations = 1000;
    // A batch cap limits columns inserted after a complete pricing pass; it
    // never truncates the label search or certifies an incomplete pass.
    int cg_max_columns = 64;
    double cg_duality_tolerance = 1e-6, pricing_tolerance = 1e-7;
    bool log = false, pareto = true, column_generation = true, reuse_columns = true;
    // Integer recourse only: exact fixed-assignment elimination and gated
    // dual cuts. Free/fractional strengthening retains the lifted model.
    bool conditional_demand = false;
    // Fixed generalized-task reconstruction, shared by full-path and Benders.
    bool preprocessed_demand = false;
    bool first_leg_consistency = false;
    // Joint convexification of the integer design/operating model. Disabled
    // by default so frozen baseline experiments remain reproducible.
    bool assignment_incident_clique = false, shared_rho_joint = false;
    bool queue_perspective = false;
    // Opt-in redundant integer strengthening; baseline remains reproducible.
    bool tight_airline_bounds = false, topology_cuts = false;
    double topology_seconds = 2;
    int topology_cut_limit = 1000;
    bool assignment_support_cuts = false;
    double assignment_support_seconds = .25, assignment_support_floor = 1e-4;
    int assignment_support_cut_limit = 2000, assignment_support_max_cuts = 16;
    // Optimality-preserving canonicalization for fixed generalized tasks.
    // Does not preserve redundant zero-service assignment encodings.
    bool path_support_cuts = false;
    bool fractional_path_support = false;
    double path_support_seconds = .25, path_support_preprocessing_seconds = 2;
    int path_support_cut_limit = 50000, path_support_max_cuts = 32;
    // Complete fixed-task cost/resource bounds. External separation leaves
    // the master as design binaries plus its operating-cost epigraph.
    bool task_savings_bound = false, task_resource_cuts = false;
    double task_resource_seconds = 2;
    int task_resource_cut_limit = 1000;
    bool task_airline_bounds = false;
    // Complete-path resource LP and canonical facility/assignment connectivity.
    // Both are optional; separation adds no variables to the Benders master.
    bool airline_connectivity_cuts = false, airline_resource_cuts = false;
    bool airline_resource_balance = false;
    // Integer-valid queue epigraph cuts, separated from cached full-path
    // resource prices; no operational variables enter the master.
    bool queue_polymatroid_cuts = false;
    bool queue_price_search = false, target_master_refinement = false;
    bool queue_fractional_separation = false;
    // Candidate bound/search extensions; frozen production presets stay off.
    bool design_face_cuts = false, queue_joint_cuts = false;
    bool primal_coupled_neighborhood = false, primal_seed_portfolio = false;
    double airline_resource_seconds = 2;
    int airline_resource_cut_limit = 1000;
    std::vector<double> task_airline_upper;
    Json task_airline_metadata = {{"enabled",false}};
    bool double_support_cuts = false, joint_base_cost_cuts = false;
    double branch_joint_seconds = .25, branch_joint_preprocessing_seconds = 2;
    double branch_joint_floor = 1e-4;
    int branch_joint_cut_limit = 2000, branch_joint_max_cuts = 16;
    bool reuse_dual_lp = true;
    bool dual_lp_first = false;
    bool courier_savings_bound = true, initial_design = true, warm_start = true;
    bool incumbent_cleanup = true, primal_search = true;
    bool branch_covering_seeds = true, resource_bounds = true;
    bool capacity_cuts = true;
    int capacity_cut_limit = 100;
    bool resource_lp_cuts = true, adaptive_master = true;
    // Search policy only: bounded master refinement, periodic complete
    // recourse, and wall-clock primal searches. Certification is unchanged.
    bool search_scheduling = false;
    // Full-domain weak cuts remain distinct from tight recourse certificates.
    bool weak_dual_cuts = false, interior_cuts = false;
    // Complement fixed-assignment polishing with fixed-infrastructure search.
    bool primal_asset_neighborhood = false;
    bool neighborhood_cuts = true;
    int heuristic_pool_size = 4;
    double neighborhood_cut_seconds = 15, neighborhood_cut_budget_fraction = .05;
    int resource_lp_cut_limit = 100;
    double resource_lp_seconds = 5, master_initial_gap = .05, master_slice_seconds = 30;
    // External free-design continuous relaxation; only its complete-domain
    // certified lower bound enters the compact master, never its primal point.
    bool continuous_root = true;
    // Complete-domain certificate improvement and reusable continuous stages.
    // Opt-in to preserve historical preset and diagnostic behavior.
    bool branch_pricing_correction = false, cg_seed_reuse = false;
    bool staged_continuous_root = false;
    bool fast_pricing = false, batch_columns = false;
    // Resource-block ALNS replaces the old primal schedule and shares 30%
    // of the end-to-end budget; all historical presets keep it disabled.
    bool primal_alns = false;
    // 2026-11-05 extensions have separate switches for controlled comparisons.
    bool eliminate_zero_task_columns = false, compact_task_heads = false;
    bool cache_pricing_graph = false, persist_recourse_model = false;
    bool stabilized_pricing = false;
    double pricing_smoothing_weight = .5;
    int pricing_exact_interval = 4;
    bool primal_support_completion = false, grouped_local_branching = false;
    int local_branching_head_radius = 4, local_branching_capacity_radius = 2;
    bool cut_pool_deduplication = false, cut_pool_management = false;
    int cut_pool_active_limit = 256, cut_pool_age = 8;
    // Follow-up policies are opt-in; frozen presets retain their old settings.
    bool streamlined_recourse = false, catalogue_pricing = false, diverse_pricing = false;
    bool reuse_recourse_rhs = false, design_certificate_cache = false;
    bool adaptive_pricing = false, cg_primal_feedback = false;
    bool indexed_support_completion = false, stabilized_master = false;
    int design_cache_limit = 16;
    double stabilized_master_seconds = 2, stabilized_master_level = .5;
    int continuous_root_max_columns = 512;
    double continuous_root_seconds = 120, continuous_root_budget_fraction = .65;
    // A continuous value certificate does not describe Q(x) at other points.
    bool root_projection = true, root_inout = true, auxiliary_valid_cuts = true;
    bool resource_airline_bounds = true;
    // OD-demand strong linking in the external resource LP. Valid for the
    // integer-design convex hull, not the original fractional lifted model.
    bool od_airline_bounds = false;
    double root_inout_weight = .90;
    bool root_cuts = true, restricted_mip = true, local_branching = true, incumbent_cutoff = true;
    // Primal-only search on generated columns plus incumbent support. A fixed
    // assignment pass can reroute flows and resize assets; no local bound is
    // promoted to the global Benders lower bound.
    bool primal_polish = false;
    int root_cut_rounds = 20, pareto_frequency = 10, heuristic_frequency = 5;
    int local_branching_radius = 8;
    double pareto_budget_fraction = .05, pareto_call_seconds = 30;
    double root_budget_fraction = .10, heuristic_budget_fraction = .20;
    // Output-only: write the full catalogue after optimization when requested.
    bool export_catalogue = false;
    std::filesystem::path output;
    std::shared_ptr<RunProgress> progress;
};
struct SolverObservation;
struct SolveOutcome {
    bool has_solution = false, optimal = false;
    // A numerical termination may retain a point although solve() returned
    // false. Such a candidate is not a feasibility or optimality certificate.
    bool candidate_available = false;
    double objective = 0, bound = 0, seconds = 0, nodes = 0;
    std::string status;
    std::string candidate_extraction_error;
    int cplex_status = 0;
    std::vector<double> values;
};

// Native multipliers from the solver's current continuous minimization
// solution. Availability does not certify feasibility or optimality.
struct NativeDualPoint {
    bool available = false;
    std::vector<double> row_prices, bound_prices;
    // Coordinates correspond to each ConeRecord's (up, down, one).
    std::vector<std::array<double, 3>> cone_vectors;
    std::string error;
    Json diagnostics;
};

// The sparse record is the source of truth for both CPLEX and explicit duals.
class NativeSolver {
public:
    IloEnv env;
    IloModel model;
    IloCplex cplex;
    IloNumVarArray variables;
    IloRangeArray rows;
    IloRangeArray cone_rows;
    IloObjective objective;
    explicit NativeSolver(const LinearModel&, const SolverOptions&, std::string role = "auxiliary");
    ~NativeSolver();
    NativeSolver(const NativeSolver&) = delete;
    SolveOutcome optimize(double seconds);
    NativeDualPoint read_dual(const LinearModel&);
    void set_rhs(int row, char sense, double rhs);
    void add_row(const RowRecord&);
    void add_variable(const VariableRecord&, const Coefficients& column);
    // Continuous columns only. All argument checks precede model mutation.
    // On a solver exception the caller must discard this solver.
    void add_variables(const std::vector<VariableRecord>&, const std::vector<Coefficients>&);
    void add_start(const std::vector<double>& values, const std::vector<int>& indices = {});
    void export_model(const std::filesystem::path&);
    void set_incumbent_observer(std::function<void(const SolveOutcome&)>);
private:
    bool mixed_ = false;
    bool callback_installed_ = false;
    std::shared_ptr<SolverObservation> observation_;
};
// One immutable assessment of one point against one matrix revision. The
// boolean gate intentionally retains the original relative-bound/double-sum
// policy, while residuals() retains the absolute-bound/long-double report.
// Callers must discard it after changing the point or any matrix data.
struct PointAssessment {
    bool dimension_valid = false, finite = false, feasible = false;
    // Legacy point_residuals: throws on wrong dimensions or nonfinite values.
    Json residuals() const;
    // Legacy Benders diagnostics: reports those failures without throwing.
    Json diagnostics() const;
private:
    struct LinearDetail {
        std::string name;
        char sense = 'E';
        double rhs = 0, activity = 0, magnitude = 0, violation = 0, scaled = 0;
        std::size_t nonzeros = 0;
        double minimum = 0, maximum = 0;
    };
    struct ToleranceDetail {
        std::string kind, name;
        double violation = 0, tolerance = 0;
    };
    double bound_ = 0, linear_ = 0, scaled_ = 0, cone_ = 0, integer_ = 0, worst_ = 0;
    std::optional<LinearDetail> worst_linear_, worst_scaled_;
    std::optional<ToleranceDetail> worst_constraint_;
    friend PointAssessment assess_point(const LinearModel&, const std::vector<double>&);
};
PointAssessment assess_point(const LinearModel&, const std::vector<double>&);
Json point_residuals(const LinearModel&, const std::vector<double>&);
enum class PrimalPolishRestriction { FixedConeCoordinates, MonotoneConeBox, AdaptiveFixedThenBox };
// Primal-only recovery: fix every cone's squared coordinate, and either fix
// the other coordinates or bound them below by a certified nonnegative box
// corner. Both are LP INNER restrictions of the original cones. Validate the
// ORIGINAL matrix and replace values atomically only on success. The LP's
// optimum/dual bound is never an original-conic optimality/LB certificate.
// The default preserves fixed-coordinate success, then tries the box only on
// restricted-LP infeasibility/residual failure within the same total budget.
Json polish_conic_primal(const LinearModel&, const SolverOptions&, double seconds,
                        std::vector<double>& values,
                        PrimalPolishRestriction restriction = PrimalPolishRestriction::AdaptiveFixedThenBox);
// Try extending an old vector with zero for appended columns. Every new
// bound must contain zero, and the complete current matrix is revalidated.
// This does not promise feasibility merely because columns were appended.
Json extend_primal_candidate(const LinearModel&, const std::vector<double>& previous,
                             std::vector<double>& values);
double objective_value(const LinearModel&, const std::vector<double>&);
double wall_time();
} // namespace mlnd
