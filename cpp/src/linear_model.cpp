#include "mlnd/linear_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mlnd {
struct SolverObservation {
    std::shared_ptr<RunProgress> progress;
    std::function<void(const SolveOutcome&)> incumbent;
    std::size_t id = 0;
    std::string role;
    Json model_size;
    double last_log = -1e100, last_checkpoint = -1e100, last_incumbent = INFINITY;
};
namespace {
class ObservationalMipCallback : public IloCplex::MIPInfoCallbackI {
    std::shared_ptr<SolverObservation> observation_;
    IloNumVarArray variables_;
public:
    ObservationalMipCallback(IloEnv env, std::shared_ptr<SolverObservation> observation, IloNumVarArray variables)
        : IloCplex::MIPInfoCallbackI(env), observation_(std::move(observation)), variables_(variables) {}
    IloCplex::CallbackI* duplicateCallback() const override {
        return new (getEnv()) ObservationalMipCallback(*this);
    }
    void main() override {
        // Informational callback only: no branching, cuts, aborts or solver
        // parameter changes. Master-model bounds stay explicitly scoped.
        auto& observation = *observation_;
        const double now = wall_time();
        const bool incumbent = hasIncumbent();
        const double objective = incumbent ? getIncumbentObjValue() : INFINITY;
        const bool improvement = incumbent && objective < observation.last_incumbent;
        try {
            if (now - observation.last_log >= 10.0) {
                observation.progress->record({{"event", "solver_progress"}, {"stage", observation.role},
                    {"solver_id", observation.id}, {"bound_scope", "solver_model"}, {"solver_reported", true},
                    {"solver_objective", incumbent ? Json(objective) : Json(nullptr)},
                    {"solver_bound", getBestObjValue()},
                    {"solver_gap", incumbent ? Json(getMIPRelativeGap()) : Json(nullptr)},
                    {"has_incumbent", incumbent}, {"nodes", getNnodes64()}});
                observation.last_log = now;
            }
            if (observation.incumbent && improvement && now - observation.last_checkpoint >= 300.0) {
                SolveOutcome point;
                point.has_solution = true;
                point.candidate_available = true;
                point.objective = objective;
                point.bound = getBestObjValue();
                point.nodes = static_cast<double>(getNnodes64());
                point.status = "MIPInfoIncumbent";
                IloNumArray values(getEnv());
                getIncumbentValues(values, variables_);
                point.values.resize(values.getSize());
                for (int j = 0; j < values.getSize(); ++j) point.values[j] = values[j];
                values.end();
                observation.incumbent(point);
                observation.last_checkpoint = now;
                observation.last_incumbent = objective;
            }
        } catch (const std::exception& error) {
            getEnv().warning() << "Progress/checkpoint observation failed: " << error.what() << '\n';
        } catch (const IloException& error) {
            getEnv().warning() << "Progress/checkpoint observation failed: " << error.getMessage() << '\n';
        }
    }
};
}
double wall_time() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
int LinearModel::variable(std::string name, double lower, double upper,
                          double cost, bool binary) {
    model_construction_guard("model_record_construction", variables.size(), rows.size(), cones.size());
    const int index = static_cast<int>(variables.size());
    variables.push_back({std::move(name), lower, upper, cost, binary});
    return index;
}
int LinearModel::row(std::string name, char sense, double rhs, Coefficients coefficients) {
    model_construction_guard("model_record_construction", variables.size(), rows.size(), cones.size());
    if (sense != 'E' && sense != 'L' && sense != 'G') throw std::invalid_argument("Invalid row sense");
    const int index = static_cast<int>(rows.size());
    rows.push_back({std::move(name), sense, rhs, std::move(coefficients)});
    return index;
}
void LinearModel::coefficient(int row, int variable, double value) {
    model_construction_guard("model_record_construction", variables.size(), rows.size(), cones.size());
    if (value != 0) rows.at(row).coefficients.emplace_back(variable, value);
}
NativeSolver::NativeSolver(const LinearModel& source, const SolverOptions& options, std::string role)
    : env(), model(env), cplex(env), variables(env), rows(env), cone_rows(env), objective(env) {
    const double build_started = wall_time();
    model_construction_guard("cplex_model_construction", source.variables.size(), source.rows.size(), source.cones.size(), true);
    if (options.progress) {
        observation_ = std::make_shared<SolverObservation>();
        observation_->progress = options.progress;
        observation_->id = options.progress->next_solver_id();
        observation_->role = std::move(role);
        observation_->model_size = {{"variables", source.variables.size()}, {"linear_rows", source.rows.size()}, {"cones", source.cones.size()}};
        options.progress->record({{"event", "solver_model_build_start"}, {"stage", observation_->role},
            {"solver_id", observation_->id}, {"model_size", observation_->model_size}, {"minimize", source.minimize}});
    }
    for (const auto& var : source.variables) {
        model_construction_guard("cplex_model_construction", variables.getSize(), rows.getSize(), source.cones.size());
        variables.add(IloNumVar(env, var.lower, var.upper,
                      var.binary ? ILOBOOL : ILOFLOAT, var.name.c_str()));
        mixed_ = mixed_ || var.binary;
    }
    model.add(variables);
    IloExpr cost(env);
    cost += source.constant;
    for (int i = 0; i < variables.getSize(); ++i)
        if (source.variables[i].cost) cost += source.variables[i].cost * variables[i];
    objective = source.minimize ? IloMinimize(env, cost) : IloMaximize(env, cost);
    model.add(objective);
    cost.end();
    for (const auto& row : source.rows) add_row(row);
    for (const auto& cone : source.cones) {
        IloExpr expression(env);
        expression=variables[cone.one] * variables[cone.one]
            - cone.factor * variables[cone.up] * variables[cone.down];
        IloRange row(env,-IloInfinity,expression,0);
        cone_rows.add(row);
        model.add(row);
        expression.end();
    }
    model_construction_guard("cplex_model_construction", source.variables.size(), source.rows.size(), source.cones.size(), true);
    const double extraction_started=wall_time();
    if (observation_) observation_->progress->record({{"event","solver_extraction_start"},{"stage",observation_->role},
        {"solver_id",observation_->id},{"model_size",observation_->model_size}});
    cplex.extract(model);
    if (observation_) observation_->progress->record({{"event","solver_extraction_end"},{"stage",observation_->role},
        {"solver_id",observation_->id},{"seconds",wall_time()-extraction_started}});
    model_construction_guard("cplex_model_construction", source.variables.size(), source.rows.size(), source.cones.size(), true);
    if (!options.log) {
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }
    cplex.setParam(IloCplex::Param::Threads, options.threads);
    cplex.setParam(IloCplex::Param::RandomSeed, options.seed);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, options.gap);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::Integrality, 1e-9);
    cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-8);
    cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-8);
    cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol, 1e-9);
    // Fixed cone coordinates (for example cone_one == 1) may otherwise be
    // presolved in a way that prevents native dual recovery after an optimal
    // solve. Ask CPLEX to preserve QCP dual information during that solve.
    if(!mixed_ && !source.cones.empty())
        cplex.setParam(IloCplex::Param::Preprocessing::QCPDuals,IloCplex::QCPDualsForce);
    if (observation_) observation_->progress->record({{"event", "solver_model_build_end"},
        {"stage", observation_->role}, {"solver_id", observation_->id},
        {"seconds", wall_time() - build_started}, {"model_size", observation_->model_size}, {"mixed_integer", mixed_}});
}
NativeSolver::~NativeSolver() { env.end(); }
void NativeSolver::add_row(const RowRecord& source) {
    IloExpr expression(env);
    for (auto [index, value] : source.coefficients)
        if (value) {
            model_construction_guard("cplex_model_construction", variables.getSize(), rows.getSize(), 0);
            expression += value * variables[index];
        }
    const double lower = source.sense == 'L' ? -IloInfinity : source.rhs;
    const double upper = source.sense == 'G' ? IloInfinity : source.rhs;
    IloRange row(env, lower, expression, upper, source.name.c_str());
    rows.add(row);
    model.add(row);
    expression.end();
}
void NativeSolver::set_rhs(int index, char sense, double rhs) {
    rows[index].setBounds(sense == 'L' ? -IloInfinity : rhs,
                          sense == 'G' ? IloInfinity : rhs);
}
void NativeSolver::add_variable(const VariableRecord& source, const Coefficients& column) {
    model_construction_guard("cplex_column_insertion", variables.getSize(), rows.getSize(), 0);
    IloNumVar variable(env, source.lower, source.upper,
                       source.binary ? ILOBOOL : ILOFLOAT, source.name.c_str());
    variables.add(variable);
    model.add(variable);
    objective.setLinearCoef(variable, source.cost);
    for (auto [index, value] : column) rows[index].setLinearCoef(variable, value);
    mixed_ = mixed_ || source.binary;
}
void NativeSolver::add_variables(const std::vector<VariableRecord>& sources,
                                 const std::vector<Coefficients>& coefficients) {
    if(sources.size()!=coefficients.size())throw std::invalid_argument("Column batch size mismatch");
    if(sources.empty())return;
    // Validate the whole batch, including duplicate rows (whose interpretation
    // could differ between column construction and setLinearCoef), first.
    std::vector<int> seen(rows.getSize(),-1);
    for(std::size_t j=0;j<sources.size();++j) {
        const auto& v=sources[j];
        if(v.binary || !std::isfinite(v.lower) || !std::isfinite(v.upper) ||
           !std::isfinite(v.cost) || v.lower>v.upper)
            throw std::invalid_argument("Invalid continuous column bounds, type or cost");
        for(auto [row,value]:coefficients[j]) {
            if(row<0 || row>=rows.getSize() || !std::isfinite(value) || seen[row]==static_cast<int>(j))
                throw std::invalid_argument("Invalid or duplicate sparse column row");
            seen[row]=static_cast<int>(j);
        }
    }
    IloNumColumnArray columns(env);
    IloNumArray lower(env),upper(env);
    try {
        for(std::size_t j=0;j<sources.size();++j) {
            model_construction_guard("cplex_column_batch_preparation",variables.getSize(),rows.getSize(),0);
            IloNumColumn column=objective(sources[j].cost);
            for(auto [row,value]:coefficients[j])if(value)column+=rows[row](value);
            columns.add(column);lower.add(sources[j].lower);upper.add(sources[j].upper);
        }
        model_construction_guard("cplex_column_batch_insertion",variables.getSize(),rows.getSize(),0,true);
        IloNumVarArray added(env,columns,lower,upper,ILOFLOAT);
        for(IloInt j=0;j<added.getSize();++j)added[j].setName(sources[j].name.c_str());
        variables.add(added);model.add(added);added.end();
    } catch(...) {
        for(IloInt j=0;j<columns.getSize();++j)columns[j].end();
        columns.end();lower.end();upper.end();throw;
    }
    for(IloInt j=0;j<columns.getSize();++j)columns[j].end();
    columns.end();lower.end();upper.end();
}
void NativeSolver::add_start(const std::vector<double>& values, const std::vector<int>& indices) {
    if (!mixed_) return;
    IloNumVarArray chosen(env);
    IloNumArray point(env);
    if (indices.empty()) {
        if (values.size() != static_cast<std::size_t>(variables.getSize()))
            throw std::invalid_argument("MIP start dimension mismatch");
        for (int i = 0; i < variables.getSize(); ++i) {
            chosen.add(variables[i]); point.add(values[i]);
        }
    } else {
        if (values.size() != indices.size()) throw std::invalid_argument("Partial MIP start mismatch");
        for (std::size_t i = 0; i < indices.size(); ++i) {
            chosen.add(variables[indices[i]]); point.add(values[i]);
        }
    }
    cplex.addMIPStart(chosen, point, IloCplex::MIPStartAuto);
    point.end(); chosen.end();
}
SolveOutcome NativeSolver::optimize(double seconds) {
    SolveOutcome out;
    if (observation_) seconds = std::min(seconds, observation_->progress->remaining());
    if (seconds <= 0) { out.status = "time_limit"; return out; }
    if (observation_) {
        observation_->progress->record({{"event", "solver_solve_start"}, {"stage", observation_->role},
            {"solver_id", observation_->id}, {"time_limit_seconds", seconds},
            {"model_size", {{"variables", variables.getSize()}, {"linear_rows", rows.getSize()},
                {"cones", observation_->model_size.at("cones")}}}});
        if (mixed_ && !callback_installed_) {
            cplex.use(new (env) ObservationalMipCallback(env, observation_, variables));
            callback_installed_ = true;
        }
    }
    cplex.setParam(IloCplex::Param::TimeLimit, std::max(1e-6, seconds));
    const double start = wall_time();
    out.has_solution = cplex.solve();
    out.seconds = wall_time() - start;
    out.cplex_status = static_cast<int>(cplex.getCplexStatus());
    std::ostringstream state; state << cplex.getCplexStatus();
    out.status = state.str();
    out.optimal = cplex.getStatus() == IloAlgorithm::Optimal;
    if (out.has_solution) {
        IloNumArray values(env);
        cplex.getValues(values, variables);
        out.values.resize(values.getSize());
        for (int i = 0; i < values.getSize(); ++i) out.values[i] = values[i];
        values.end();
        out.objective = cplex.getObjValue();
        out.candidate_available = true;
        if (!mixed_ && out.optimal) out.bound = out.objective;
    } else if (!mixed_ && cplex.getCplexStatus() == IloCplex::NumBest) {
        // CPX_STAT_NUM_BEST can retain the last numerical iterate even though
        // solve() is false. Expose it only as an unvalidated candidate; callers
        // must check the original matrix, and must independently certify dual
        // feasibility before using any lower bound. Preserve solver semantics.
        IloNumArray values(env);
        try {
            cplex.getValues(values, variables);
            const double candidate_objective = cplex.getObjValue();
            bool finite = values.getSize() == variables.getSize() && std::isfinite(candidate_objective);
            for (int i = 0; i < values.getSize(); ++i) finite = finite && std::isfinite(values[i]);
            if (finite) {
                out.values.resize(values.getSize());
                for (int i = 0; i < values.getSize(); ++i) out.values[i] = values[i];
                out.objective = candidate_objective;
                out.candidate_available = true;
            } else out.candidate_extraction_error = "nonfinite_or_dimension_mismatch";
        } catch (const IloException& error) {
            out.candidate_extraction_error = error.getMessage();
        }
        values.end();
    }
    if (mixed_) {
        out.bound = cplex.getBestObjValue();
        out.nodes = static_cast<double>(cplex.getNnodes64());
    }
    if (observation_) observation_->progress->record({{"event", "solver_solve_end"}, {"stage", observation_->role},
        {"solver_id", observation_->id}, {"seconds", out.seconds}, {"solver_status", out.status},
        {"solver_status_code", out.cplex_status}, {"has_solution", out.has_solution}, {"solver_optimal", out.optimal},
        {"candidate_available", out.candidate_available},
        {"candidate_objective", out.candidate_available ? Json(out.objective) : Json(nullptr)},
        {"candidate_extraction_error", out.candidate_extraction_error.empty() ? Json(nullptr) : Json(out.candidate_extraction_error)},
        {"solver_objective", out.has_solution ? Json(out.objective) : Json(nullptr)},
        {"solver_bound", mixed_ || out.optimal ? Json(out.bound) : Json(nullptr)},
        {"bound_scope", "solver_model"}, {"nodes", out.nodes}});
    return out;
}
NativeDualPoint NativeSolver::read_dual(const LinearModel& source) {
    const double started=wall_time();
    NativeDualPoint result;
    result.diagnostics={{"method","cplex_native_dual"},{"additional_optimization_calls",0}};
    if(mixed_ || !source.minimize || objective.getSense()!=IloObjective::Minimize) {
        result.error="Native dual extraction requires a continuous minimization model";
        return result;
    }
    if(source.variables.size()!=static_cast<std::size_t>(variables.getSize()) ||
       source.rows.size()!=static_cast<std::size_t>(rows.getSize()) ||
       source.cones.size()!=static_cast<std::size_t>(cone_rows.getSize())) {
        result.error="Native dual source dimensions do not match the extracted model";
        return result;
    }
    IloNumArray row_values(env),bound_values(env),cone_values(env);
    IloNumVarArray cone_variables(env);
    try {
        // This getter also rejects a model without a current solution. No
        // solve or parameter change is performed by this extraction method.
        result.diagnostics["solver_objective"]=cplex.getObjValue();
        result.diagnostics["solver_status_code"]=static_cast<int>(cplex.getCplexStatus());
        if(rows.getSize())cplex.getDuals(row_values,rows);
        if(variables.getSize())cplex.getReducedCosts(bound_values,variables);
        if(row_values.getSize()!=rows.getSize() || bound_values.getSize()!=variables.getSize())
            throw std::runtime_error("Native linear dual dimensions are inconsistent");
        result.row_prices.resize(row_values.getSize());
        result.bound_prices.resize(bound_values.getSize());
        for(IloInt i=0;i<row_values.getSize();++i)result.row_prices[i]=row_values[i];
        for(IloInt j=0;j<bound_values.getSize();++j)result.bound_prices[j]=bound_values[j];
        std::unordered_map<IloInt,int> variable_indices;
        for(IloInt j=0;j<variables.getSize();++j)variable_indices.emplace(variables[j].getId(),j);
        result.cone_vectors.resize(source.cones.size(),{0,0,0});
        for(std::size_t k=0;k<source.cones.size();++k) {
            const auto& cone=source.cones[k];
            const std::array<int,3> coordinates{cone.up,cone.down,cone.one};
            if(cone.up<0 || cone.down<0 || cone.one<0 ||
               cone.up>=variables.getSize() || cone.down>=variables.getSize() || cone.one>=variables.getSize() ||
               cone.up==cone.down || cone.up==cone.one || cone.down==cone.one)
                throw std::runtime_error("Invalid native cone coordinate mapping");
            cplex.getQCDSlack(cone_rows[static_cast<IloInt>(k)],cone_values,cone_variables);
            if(cone_values.getSize()!=cone_variables.getSize())
                throw std::runtime_error("Native cone dual dimensions are inconsistent");
            for(IloInt h=0;h<cone_variables.getSize();++h) {
                const auto found=variable_indices.find(cone_variables[h].getId());
                if(found==variable_indices.end())throw std::runtime_error("Unknown native cone dual variable");
                const auto position=std::find(coordinates.begin(),coordinates.end(),found->second);
                if(position==coordinates.end())throw std::runtime_error("Native cone dual has an unexpected coordinate");
                result.cone_vectors[k][position-coordinates.begin()]+=cone_values[h];
            }
        }
        const auto finite=[](double value){return std::isfinite(value);};
        bool all_finite=std::all_of(result.row_prices.begin(),result.row_prices.end(),finite) &&
            std::all_of(result.bound_prices.begin(),result.bound_prices.end(),finite);
        for(const auto& cone:result.cone_vectors)
            all_finite=all_finite && std::all_of(cone.begin(),cone.end(),finite);
        if(!all_finite)throw std::runtime_error("Native dual information contains a nonfinite value");
        result.available=true;
    } catch(const IloException& error) {
        result.error=error.getMessage();
    } catch(const std::exception& error) {
        result.error=error.what();
    }
    row_values.end();bound_values.end();cone_values.end();cone_variables.end();
    result.diagnostics["available"]=result.available;
    result.diagnostics["seconds"]=wall_time()-started;
    if(!result.available)result.diagnostics["error"]=result.error;
    return result;
}
void NativeSolver::set_incumbent_observer(std::function<void(const SolveOutcome&)> observer) {
    if (observation_) observation_->incumbent = std::move(observer);
}
void NativeSolver::export_model(const std::filesystem::path& path) {
    cplex.exportModel(path.string().c_str());
}
double objective_value(const LinearModel& model, const std::vector<double>& x) {
    long double value = model.constant;
    for (std::size_t i = 0; i < x.size(); ++i) value += static_cast<long double>(model.variables[i].cost) * x[i];
    return static_cast<double>(value);
}
Json extend_primal_candidate(const LinearModel& original,const std::vector<double>& previous,
                             std::vector<double>& values) {
    Json report={{"method","append_zero_and_validate_original_matrix"},{"passed",false},
        {"applied",false},{"lower_bound",nullptr},{"original_model_optimality_certified",false}};
    if(previous.size()>original.variables.size() ||
       !std::all_of(previous.begin(),previous.end(),[](double x){return std::isfinite(x);})) {
        report["reason"]="invalid_previous_candidate";return report;
    }
    auto candidate=previous;
    for(std::size_t j=previous.size();j<original.variables.size();++j) {
        if(original.variables[j].lower>0 || original.variables[j].upper<0) {
            report["reason"]="new_variable_excludes_zero";report["variable"]=original.variables[j].name;
            return report;
        }
        candidate.push_back(0);
    }
    report["appended_variables"]=candidate.size()-previous.size();
    report["residuals"]=point_residuals(original,candidate);
    if(!report["residuals"]["passed"].get<bool>()) {
        report["reason"]="original_matrix_residual_failed";return report;
    }
    const double objective=objective_value(original,candidate);
    if(!std::isfinite(objective)) {report["reason"]="nonfinite_objective";return report;}
    values=std::move(candidate);
    report["applied"]=true;report["passed"]=true;report["objective"]=objective;
    report["reason"]="original_matrix_primal_validated";
    return report;
}
Json polish_conic_primal(const LinearModel& original,const SolverOptions& options,
                        double seconds,std::vector<double>& values,PrimalPolishRestriction restriction) {
    const double started=wall_time(),deadline=started+std::max(0.0,seconds);
    if(restriction==PrimalPolishRestriction::AdaptiveFixedThenBox) {
        // Preserve a successful established fixed-coordinate trajectory.
        // Try the larger inner restriction only after a proved restricted-LP
        // failure, with the SAME call's remaining budget; no profile sweep.
        auto fixed=polish_conic_primal(original,options,seconds,values,
                                      PrimalPolishRestriction::FixedConeCoordinates);
        const auto brief=[](const Json& result) {
            return Json{{"method",result.at("method")},{"applied",result.at("applied")},
                {"reason",result.at("reason")},{"status",result.value("status",std::string())},
                {"seconds",result.at("seconds")},{"solver_seconds",result.at("solver_seconds")},
                {"allocated_seconds",result.at("budget_seconds")}};
        };
        Json attempts=Json::array({brief(fixed)});
        const bool eligible=!fixed.at("applied").get<bool>() &&
            (fixed.value("status",std::string())=="Infeasible" ||
             fixed.at("reason")=="original_matrix_residual_failed");
        Json selected=fixed;
        const double left=std::max(0.0,deadline-wall_time());
        if(eligible && left>0) {
            selected=polish_conic_primal(original,options,left,values,
                                        PrimalPolishRestriction::MonotoneConeBox);
            attempts.push_back(brief(selected));
            for(const char* key:{"build_seconds","solver_seconds","validation_seconds"})
                selected[key]=selected.at(key).get<double>()+fixed.at(key).get<double>();
        }
        selected["requested_restriction"]="adaptive_fixed_then_box";
        selected["adaptive_attempts"]=std::move(attempts);
        selected["adaptive_fallback_eligible"]=eligible;
        selected["budget_seconds"]=seconds;
        selected["seconds"]=wall_time()-started;
        return selected;
    }
    const bool box=restriction==PrimalPolishRestriction::MonotoneConeBox;
    Json report={{"method",box?"monotone_cone_box_lp_restriction":"fixed_cone_coordinates_lp_restriction"},
        {"applied",false},{"passed",false},{"budget_seconds",seconds},
        {"original_model_optimality_certified",false},{"lower_bound",nullptr},
        {"build_seconds",0.0},{"solver_seconds",0.0},{"validation_seconds",0.0}};
    auto finish=[&](const char* reason) {
        report["reason"]=reason;report["seconds"]=wall_time()-started;
        return report;
    };
    if(!std::isfinite(seconds)||seconds<=0)return finish("no_budget");
    if(!original.minimize || std::any_of(original.variables.begin(),original.variables.end(),
            [](const auto& variable){return variable.binary;}))
        return finish("requires_continuous_minimization");
    if(values.size()!=original.variables.size() ||
       !std::all_of(values.begin(),values.end(),[](double x){return std::isfinite(x);}))
        return finish("invalid_anchor_point");
    report["original_residuals"]=point_residuals(original,values);
    LinearModel linear=original;
    linear.cones.clear();
    auto coordinates=values;
    if(box) {
        // All squared coordinates are frozen FIRST. A coordinate may be a
        // resource in another cone; changing it there would invalidate that
        // other cone's constant RHS. Every remaining update is nonnegative
        // and monotone, so it cannot invalidate an earlier product bound.
        std::vector<bool> squared(values.size(),false),resource(values.size(),false);
        for(const auto& c:original.cones) {
            if(!std::isfinite(c.factor) || !(c.factor>0))return finish("invalid_cone_factor");
            squared.at(c.one)=true;resource.at(c.up)=resource.at(c.down)=true;
            coordinates[c.one]=std::clamp(values[c.one],original.variables[c.one].lower,original.variables[c.one].upper);
        }
        for(std::size_t j=0;j<coordinates.size();++j) if(resource[j]) {
            if(squared[j]) {
                if(coordinates[j]<0)return finish("shared_squared_coordinate_outside_nonnegative_branch");
            } else {
                const double lower=std::max(0.0,original.variables[j].lower);
                if(original.variables[j].upper<lower)return finish("nonnegative_cone_branch_outside_bounds");
                coordinates[j]=std::clamp(values[j],lower,original.variables[j].upper);
            }
        }
        const auto ceiling=[](long double x) {
            double value=static_cast<double>(x);
            if(static_cast<long double>(value)<x)value=std::nextafter(value,INFINITY);
            return value;
        };
        const auto cap=[&](int j) {return squared[j]?coordinates[j]:original.variables[j].upper;};
        for(const auto& c:original.cones) {
            const long double square=static_cast<long double>(coordinates[c.one])*coordinates[c.one];
            const long double required=square/c.factor;
            if(!std::isfinite(required))return finish("nonfinite_cone_corner_requirement");
            auto valid=[&]() {return static_cast<long double>(c.factor)*coordinates[c.up]*coordinates[c.down]>=square;};
            if(!valid()) {
                if(c.up==c.down) {
                    const double target=ceiling(std::sqrt(required));
                    if(target>cap(c.up))return finish("cone_corner_not_repairable_within_bounds");
                    coordinates[c.up]=std::max(coordinates[c.up],target);
                } else {
                    const double up=coordinates[c.down]>0?ceiling(required/coordinates[c.down]):INFINITY;
                    const double down=coordinates[c.up]>0?ceiling(required/coordinates[c.up]):INFINITY;
                    if(up<=cap(c.up))coordinates[c.up]=std::max(coordinates[c.up],up);
                    else if(down<=cap(c.down))coordinates[c.down]=std::max(coordinates[c.down],down);
                    else {
                        // Neither coordinate alone suffices (including a
                        // zero/zero corner); raise both inside their caps.
                        const double up_target=std::max(coordinates[c.up],std::min(cap(c.up),ceiling(std::sqrt(required))));
                        double down_target=up_target>0?std::max(coordinates[c.down],ceiling(required/up_target)):INFINITY;
                        double adjusted_up=up_target;
                        if(down_target>cap(c.down)) {
                            down_target=cap(c.down);
                            adjusted_up=down_target>0?std::max(coordinates[c.up],ceiling(required/down_target)):INFINITY;
                        }
                        if(adjusted_up>cap(c.up) || down_target>cap(c.down) ||
                           !std::isfinite(adjusted_up) || !std::isfinite(down_target))
                            return finish("cone_corner_not_repairable_within_bounds");
                        coordinates[c.up]=adjusted_up;coordinates[c.down]=down_target;
                    }
                }
                // Outward one-ulp repair when division/product rounding
                // disagree. A squared coordinate shared by another cone is
                // never moved. Reject instead of weakening the cone check.
                for(int j:{c.up,c.down}) if(!valid() && !squared[j]) {
                    const double raised=std::nextafter(coordinates[j],INFINITY);
                    if(raised<=cap(j))coordinates[j]=raised;
                }
                if(!valid())return finish("cone_corner_not_certified");
            }
        }
        for(std::size_t j=0;j<coordinates.size();++j) {
            if(squared[j])linear.variables[j].lower=linear.variables[j].upper=coordinates[j];
            else if(resource[j])linear.variables[j].lower=std::max(linear.variables[j].lower,coordinates[j]);
        }
        report["restriction_certificate"]={{"kind","nonnegative_product_monotonicity"},
            {"all_squared_coordinates_fixed",true},{"shared_coordinates_checked",true},
            {"contains_same_corner_fixed_coordinate_restriction",true}};
        report["cone_box_corners"]=Json::array();
        for(const auto& c:original.cones)
            report["cone_box_corners"].push_back({{"up",c.up},{"down",c.down},{"one",c.one},
                {"factor",c.factor},{"up_lower",coordinates[c.up]},{"down_lower",coordinates[c.down]},
                {"one_fixed",coordinates[c.one]}});
    } else for(const auto& c:original.cones) {
        if(!(c.factor>0))return finish("invalid_cone_factor");
        for(int j:{c.up,c.down,c.one})
            coordinates[j]=std::clamp(coordinates[j],original.variables[j].lower,original.variables[j].upper);
        const long double required=static_cast<long double>(coordinates[c.one])*coordinates[c.one]/c.factor;
        if(static_cast<long double>(coordinates[c.up])*coordinates[c.down]<required) {
            const double up=coordinates[c.down]>0
                ?std::nextafter(static_cast<double>(required/coordinates[c.down]),INFINITY):INFINITY;
            const double down=coordinates[c.up]>0
                ?std::nextafter(static_cast<double>(required/coordinates[c.up]),INFINITY):INFINITY;
            if(up<=original.variables[c.up].upper)coordinates[c.up]=std::max(coordinates[c.up],up);
            else if(down<=original.variables[c.down].upper)coordinates[c.down]=std::max(coordinates[c.down],down);
            else return finish("cone_anchor_not_repairable_within_bounds");
        }
    }
    for(const auto& c:original.cones) {
        if(static_cast<long double>(c.factor)*coordinates[c.up]*coordinates[c.down] <
           static_cast<long double>(coordinates[c.one])*coordinates[c.one])
            return finish("cone_anchor_not_certified");
        if(!box)for(int j:{c.up,c.down,c.one})
            linear.variables[j].lower=linear.variables[j].upper=coordinates[j];
    }
    if(wall_time()>=deadline)return finish("deadline_before_solver_build");
    try {
        NativeSolver solver(linear,options,"fixed_cone_primal_recovery");
        solver.cplex.setParam(IloCplex::Param::RootAlgorithm,IloCplex::Dual);
        solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility,1e-9);
        report["build_seconds"]=wall_time()-started;
        if(wall_time()>=deadline)return finish("deadline_during_solver_build");
        auto solved=solver.optimize(std::max(0.0,deadline-wall_time()));
        report["status"]=solved.status;report["restricted_lp_optimal"]=solved.optimal;
        report["solver_seconds"]=solved.seconds;report["has_solution"]=solved.has_solution;
        if(!solved.has_solution)return finish("restricted_lp_no_point");
        const double validation_started=wall_time();
        report["residuals"]=point_residuals(original,solved.values);
        report["objective"]=objective_value(original,solved.values);
        report["validation_seconds"]=wall_time()-validation_started;
        if(!report["residuals"]["passed"].get<bool>())return finish("original_matrix_residual_failed");
        if(!std::isfinite(report["objective"].get<double>()))return finish("nonfinite_objective");
        values=std::move(solved.values);
        report["applied"]=true;report["passed"]=true;
        report["validated_primal_upper_bound"]=report["objective"];
        return finish("original_matrix_primal_validated");
    }catch(const IloException& error) {
        report["error"]=error.getMessage();return finish("restricted_lp_solver_error");
    }
}
PointAssessment assess_point(const LinearModel& model, const std::vector<double>& x) {
    PointAssessment out;
    if (x.size() != model.variables.size()) return out;
    out.dimension_valid = true;
    out.feasible = true;
    auto check = [&](double residual, double tolerance, const char* kind, const std::string& name) {
        if (residual / tolerance > out.worst_) {
            out.worst_ = residual / tolerance;
            out.worst_constraint_ = PointAssessment::ToleranceDetail{kind,name,residual,tolerance};
        }
    };
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) { out.feasible = false; return out; }
        const auto& v = model.variables[i];
        out.bound_ = std::max({out.bound_,v.lower-x[i],x[i]-v.upper});
        const double tolerance = 2e-7 * (1 + std::abs(x[i]));
        // Do not rewrite these comparisons as residual > tolerance: their
        // floating-point rounding is part of the existing acceptance gate.
        if (x[i] < v.lower-tolerance || x[i] > v.upper+tolerance) out.feasible = false;
        check(std::max({0.0,v.lower-x[i],x[i]-v.upper}),tolerance,"bound",v.name);
        if (v.binary) {
            const double violation = std::abs(x[i]-std::round(x[i]));
            out.integer_ = std::max(out.integer_,violation);
            if (violation > 1e-7) out.feasible = false;
            check(violation,1e-7,"integrality",v.name);
        }
    }
    out.finite = true;
    for (const auto& row : model.rows) {
        // Keep the two old accumulation policies, in the same coefficient
        // order, within one traversal of the sparse row.
        long double activity = 0, magnitude = std::abs(row.rhs);
        double lhs = 0, scale = 1 + std::abs(row.rhs);
        double minimum = INFINITY, maximum = 0;
        for (auto [i, value] : row.coefficients) {
            activity += static_cast<long double>(value) * x[i];
            magnitude += std::abs(static_cast<long double>(value) * x[i]);
            lhs += value * x[i];
            scale += std::abs(value * x[i]);
            if (value != 0) {
                minimum = std::min(minimum,std::abs(value));
                maximum = std::max(maximum,std::abs(value));
            }
        }
        const double diff = static_cast<double>(activity - row.rhs);
        const double violation = row.sense == 'E' ? std::abs(diff) :
                                 std::max(0.0, row.sense == 'L' ? diff : -diff);
        const double scaled_violation=violation/static_cast<double>(1+magnitude);
        if (violation>out.linear_ || scaled_violation>out.scaled_) {
            PointAssessment::LinearDetail detail{row.name,row.sense,row.rhs,
                static_cast<double>(activity),static_cast<double>(magnitude),violation,
                scaled_violation,row.coefficients.size(),minimum,maximum};
            if (violation>out.linear_) out.worst_linear_=detail;
            if (scaled_violation>out.scaled_) out.worst_scaled_=std::move(detail);
        }
        out.linear_ = std::max(out.linear_,violation);
        out.scaled_ = std::max(out.scaled_,scaled_violation);
        double residual = lhs-row.rhs;
        if (row.sense=='L') residual=std::max(0.0,residual);
        else if (row.sense=='G') residual=std::max(0.0,-residual);
        else residual=std::abs(residual);
        const double tolerance = 2e-7*scale;
        if (residual>tolerance) out.feasible=false;
        check(residual,tolerance,"linear",row.name);
    }
    for (const auto& c : model.cones) {
        const double lhs = x[c.one] * x[c.one], rhs = c.factor * x[c.up] * x[c.down];
        out.cone_ = std::max(out.cone_,(lhs-rhs)/(1+std::abs(lhs)+std::abs(rhs)));
        const double tolerance=2e-7*(1+std::abs(lhs)+std::abs(rhs));
        if (lhs>rhs+tolerance) out.feasible=false;
        check(std::max(0.0,lhs-rhs),tolerance,"cone",model.variables[c.one].name);
    }
    return out;
}
Json PointAssessment::residuals() const {
    if (!dimension_valid) throw std::invalid_argument("Solution dimension mismatch");
    if (!finite) throw std::runtime_error("Nonfinite solution value");
    auto detail=[](const std::optional<LinearDetail>& entry)->Json {
        if (!entry) return nullptr;
        const auto& d=*entry;
        return {{"name",d.name},{"sense",std::string(1,d.sense)},
            {"rhs",d.rhs},{"activity",d.activity},{"magnitude",d.magnitude},
            {"absolute_violation",d.violation},{"scaled_violation",d.scaled},
            {"nonzeros",d.nonzeros},
            {"minimum_absolute_coefficient",std::isfinite(d.minimum)?Json(d.minimum):Json(nullptr)},
            {"maximum_absolute_coefficient",d.maximum}};
    };
    return {{"max_bound_violation",bound_},{"max_linear_violation",linear_},
        {"max_scaled_linear_violation",scaled_},{"max_scaled_cone_violation",cone_},
        {"max_integrality_violation",integer_},
        {"worst_linear_constraint",detail(worst_linear_)},{"worst_scaled_linear_constraint",detail(worst_scaled_)},
        {"passed",bound_<=2e-6 && scaled_<=2e-7 && cone_<=2e-7 && integer_<=1e-7}};
}
Json PointAssessment::diagnostics() const {
    if (!dimension_valid) return {{"passed",false},{"reason","dimension"}};
    if (!finite) return {{"passed",false},{"reason","nonfinite"}};
    auto report=residuals();
    report["passed"]=feasible;
    report["worst_tolerance_ratio"]=worst_;
    report["worst_constraint"]=worst_constraint_
        ? Json{{"kind",worst_constraint_->kind},{"name",worst_constraint_->name},
               {"violation",worst_constraint_->violation},{"tolerance",worst_constraint_->tolerance}}
        : Json(nullptr);
    return report;
}
Json point_residuals(const LinearModel& model, const std::vector<double>& x) {
    return assess_point(model,x).residuals();
}
} // namespace mlnd
