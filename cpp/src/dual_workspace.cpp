#include "mlnd/dual_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace mlnd {
namespace {
Coefficients canonical(const Coefficients& source, const std::vector<int>& map) {
    Coefficients terms;
    terms.reserve(source.size());
    for (auto [j,c] : source) {
        if (j < 0 || j >= static_cast<int>(map.size()) || !std::isfinite(c))
            throw std::invalid_argument("Invalid persistent dual LP coefficient");
        if (c != 0) terms.emplace_back(map[j],c);
    }
    std::sort(terms.begin(),terms.end());
    Coefficients combined;
    for (auto [j,c] : terms) {
        if (!combined.empty() && combined.back().first == j) combined.back().second += c;
        else combined.emplace_back(j,c);
    }
    combined.erase(std::remove_if(combined.begin(),combined.end(),[](const auto& term){return term.second == 0;}),combined.end());
    return combined;
}
bool basis(NativeSolver& solver, std::vector<IloCplex::BasisStatus>& columns,
           std::vector<IloCplex::BasisStatus>& rows) {
    IloCplex::BasisStatusArray cs(solver.env),rs(solver.env);
    bool available = false;
    try {
        solver.cplex.getBasisStatuses(cs,solver.variables,rs,solver.rows);
        available = cs.getSize() == solver.variables.getSize() && rs.getSize() == solver.rows.getSize();
        if (available) {
            for (int j=0;j<cs.getSize();++j) columns.push_back(cs[j]);
            for (int i=0;i<rs.getSize();++i) rows.push_back(rs[i]);
        }
    } catch (const IloException&) { /* No solved simplex basis yet. */ }
    cs.end(); rs.end();
    return available;
}
}
void DualRayWorkspace::clear() {
    solver_.reset(); stored_=LinearModel{};
    variable_index_.clear(); row_index_.clear(); source_to_native_.clear();
}
NativeSolver& DualRayWorkspace::synchronize(const LinearModel& source,
                                           const SolverOptions& options, Json& report) {
    const double started=wall_time();
    if (!source.cones.empty()) throw std::invalid_argument("Persistent dual workspace accepts LPs only");
    std::set<std::string> variables,rows;
    for (const auto& variable : source.variables)
        if (variable.binary || variable.name.empty() || !variables.insert(variable.name).second)
            throw std::invalid_argument("Persistent dual LP requires continuous, uniquely named variables");
    for (const auto& row : source.rows)
        if (row.name.empty() || !rows.insert(row.name).second)
            throw std::invalid_argument("Persistent dual LP requires uniquely named rows");
    std::string reset_reason;
    if (solver_ && stored_.minimize != source.minimize) reset_reason="objective_sense_changed";
    if (solver_) for (const auto& entry : variable_index_)
        if (!variables.count(entry.first)) { reset_reason="variable_removed"; break; }
    if (solver_) for (const auto& entry : row_index_)
        if (!rows.count(entry.first)) { reset_reason="row_removed"; break; }
    if (!reset_reason.empty()) clear();
    report={{"enabled",true},{"reused_solver",static_cast<bool>(solver_)},
        {"reset_reason",reset_reason.empty()?Json(nullptr):Json(reset_reason)},
        {"new_variables",0},{"new_rows",0},{"changed_coefficients",0},
        {"changed_bounds",0},{"changed_row_bounds",0},{"changed_objective_coefficients",0},
        {"basis_available_before_update",false},{"basis_submitted",false},
        {"advance",1},{"algorithm","dual_simplex"}};
    try {
        if (!solver_) {
            stored_=source;
            source_to_native_.resize(source.variables.size());
            for (int j=0;j<static_cast<int>(source.variables.size());++j) {
                variable_index_[source.variables[j].name]=j; source_to_native_[j]=j;
            }
            for (int i=0;i<static_cast<int>(source.rows.size());++i) {
                row_index_[source.rows[i].name]=i;
                stored_.rows[i].coefficients=canonical(source.rows[i].coefficients,source_to_native_);
            }
            solver_=std::make_unique<NativeSolver>(stored_,options,"benders_dual_ray_lp");
            report["new_variables"]=source.variables.size(); report["new_rows"]=source.rows.size();
        } else {
            std::vector<IloCplex::BasisStatus> basis_columns,basis_rows;
            const bool have_basis=basis(*solver_,basis_columns,basis_rows);
            report["basis_available_before_update"]=have_basis;
            source_to_native_.resize(source.variables.size());
            for (int j=0;j<static_cast<int>(source.variables.size());++j) {
                model_construction_guard("dual_workspace_update",source.variables.size(),source.rows.size(),0);
                const auto& variable=source.variables[j];
                auto found=variable_index_.find(variable.name);
                if (found==variable_index_.end()) {
                    const int native=static_cast<int>(stored_.variables.size());
                    variable_index_[variable.name]=native; source_to_native_[j]=native;
                    solver_->add_variable(variable,{}); stored_.variables.push_back(variable);
                    report["new_variables"]=report.at("new_variables").get<int>()+1;
                    if (have_basis) basis_columns.push_back(variable.lower > -IloInfinity
                        ? IloCplex::AtLower : variable.upper < IloInfinity ? IloCplex::AtUpper : IloCplex::FreeOrSuperbasic);
                } else {
                    const int native=found->second; source_to_native_[j]=native;
                    auto& old=stored_.variables[native];
                    if (variable.lower!=old.lower || variable.upper!=old.upper) {
                        solver_->variables[native].setBounds(variable.lower,variable.upper);
                        report["changed_bounds"]=report.at("changed_bounds").get<int>()+1;
                    }
                    if (variable.cost!=old.cost) {
                        solver_->objective.setLinearCoef(solver_->variables[native],variable.cost);
                        report["changed_objective_coefficients"]=report.at("changed_objective_coefficients").get<int>()+1;
                    }
                    old=variable;
                }
            }
            if (source.constant!=stored_.constant) {
                solver_->objective.setConstant(source.constant); stored_.constant=source.constant;
            }
            for (const auto& row : source.rows) {
                model_construction_guard("dual_workspace_update",source.variables.size(),source.rows.size(),0);
                auto terms=canonical(row.coefficients,source_to_native_);
                auto found=row_index_.find(row.name);
                if (found==row_index_.end()) {
                    const int native=static_cast<int>(stored_.rows.size()); row_index_[row.name]=native;
                    RowRecord added=row; added.coefficients=std::move(terms);
                    solver_->add_row(added); stored_.rows.push_back(std::move(added));
                    report["new_rows"]=report.at("new_rows").get<int>()+1;
                    if (have_basis) basis_rows.push_back(IloCplex::Basic);
                } else {
                    const int native=found->second; auto& old=stored_.rows[native];
                    if (row.sense!=old.sense || row.rhs!=old.rhs) {
                        solver_->set_rhs(native,row.sense,row.rhs);
                        report["changed_row_bounds"]=report.at("changed_row_bounds").get<int>()+1;
                    }
                    std::size_t a=0,b=0;
                    while (a<old.coefficients.size() || b<terms.size()) {
                        const int old_j=a<old.coefficients.size()?old.coefficients[a].first:std::numeric_limits<int>::max();
                        const int new_j=b<terms.size()?terms[b].first:std::numeric_limits<int>::max();
                        const int j=std::min(old_j,new_j);
                        const double old_c=old_j==j?old.coefficients[a++].second:0;
                        const double new_c=new_j==j?terms[b++].second:0;
                        if (old_c!=new_c) {
                            solver_->rows[native].setLinearCoef(solver_->variables[j],new_c);
                            report["changed_coefficients"]=report.at("changed_coefficients").get<int>()+1;
                        }
                    }
                    old.sense=row.sense; old.rhs=row.rhs; old.coefficients=std::move(terms);
                }
            }
            if (have_basis) {
                // Extend the old basis with nonbasic new columns and basic
                // slacks for new rows. CPLEX may repair/reject a singular or
                // infeasible start after coefficient changes; it still must
                // optimize and every returned point is validated downstream.
                IloCplex::BasisStatusArray cs(solver_->env),rs(solver_->env);
                try {
                    for (auto status : basis_columns) cs.add(status);
                    for (auto status : basis_rows) rs.add(status);
                    solver_->cplex.extract(solver_->model);
                    solver_->cplex.setBasisStatuses(cs,solver_->variables,rs,solver_->rows);
                    report["basis_submitted"]=true;
                } catch (const IloException& error) { report["basis_start_error"]=error.getMessage(); }
                cs.end(); rs.end();
            }
        }
        const int advance=report.contains("basis_start_error") ? 0 : 1;
        solver_->cplex.setParam(IloCplex::Param::Advance,advance);
        report["advance"]=advance;
        solver_->cplex.setParam(IloCplex::Param::RootAlgorithm,IloCplex::Dual);
        report["synchronize_seconds"]=wall_time()-started;
        return *solver_;
    } catch (...) { clear(); throw; }
}
SolveOutcome DualRayWorkspace::optimize(double seconds,Json& report) {
    if (!solver_) throw std::logic_error("Persistent dual LP was not synchronized");
    auto solved=solver_->optimize(seconds);
    report["optimize_seconds"]=solved.seconds;
    try { report["simplex_iterations"]=solver_->cplex.getNiterations64(); }
    catch (const IloException&) { report["simplex_iterations"]=nullptr; }
    if (solved.has_solution || solved.candidate_available) {
        std::vector<double> source(source_to_native_.size());
        for (std::size_t j=0;j<source.size();++j) source[j]=solved.values.at(source_to_native_[j]);
        solved.values=std::move(source);
    }
    return solved;
}
} // namespace mlnd
