#pragma once

#include "mlnd/input.hpp"
#include "mlnd/progress.hpp"
#include <ilcplex/ilocplex.h>
#include <array>
#include <memory>
#include <functional>
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
    bool double_support_cuts = false, joint_base_cost_cuts = false;
    double branch_joint_seconds = .25, branch_joint_preprocessing_seconds = 2;
    double branch_joint_floor = 1e-4;
    int branch_joint_cut_limit = 2000, branch_joint_max_cuts = 16;
    bool reuse_dual_lp = true;
    bool courier_savings_bound = true, initial_design = true, warm_start = true;
    bool incumbent_cleanup = true, primal_search = true;
    bool branch_covering_seeds = true, resource_bounds = true;
    bool capacity_cuts = true;
    int capacity_cut_limit = 100;
    bool resource_lp_cuts = true, adaptive_master = true;
    bool neighborhood_cuts = true;
    int heuristic_pool_size = 4;
    double neighborhood_cut_seconds = 15, neighborhood_cut_budget_fraction = .05;
    int resource_lp_cut_limit = 100;
    double resource_lp_seconds = 5, master_initial_gap = .05, master_slice_seconds = 30;
    // External free-design continuous relaxation; only its complete-domain
    // certified lower bound enters the compact master, never its primal point.
    bool continuous_root = true;
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
    int root_cut_rounds = 20, pareto_frequency = 10, heuristic_frequency = 5;
    int local_branching_radius = 8;
    double pareto_budget_fraction = .05, pareto_call_seconds = 30;
    double root_budget_fraction = .10, heuristic_budget_fraction = .20;
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

// The sparse record is the source of truth for both CPLEX and explicit duals.
class NativeSolver {
public:
    IloEnv env;
    IloModel model;
    IloCplex cplex;
    IloNumVarArray variables;
    IloRangeArray rows;
    IloObjective objective;
    explicit NativeSolver(const LinearModel&, const SolverOptions&, std::string role = "auxiliary");
    ~NativeSolver();
    NativeSolver(const NativeSolver&) = delete;
    SolveOutcome optimize(double seconds);
    void set_rhs(int row, char sense, double rhs);
    void add_row(const RowRecord&);
    void add_variable(const VariableRecord&, const Coefficients& column);
    void add_start(const std::vector<double>& values, const std::vector<int>& indices = {});
    void export_model(const std::filesystem::path&);
    void set_incumbent_observer(std::function<void(const SolveOutcome&)>);
private:
    bool mixed_ = false;
    bool callback_installed_ = false;
    std::shared_ptr<SolverObservation> observation_;
};
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
