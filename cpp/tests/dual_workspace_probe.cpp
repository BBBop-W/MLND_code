#include "mlnd/dual_workspace.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool test,const char* reason) { if (!test) throw std::runtime_error(reason); }
mlnd::LinearModel fixture(int phase,bool reordered) {
    constexpr int blocks=120;
    mlnd::LinearModel model; model.minimize=false; model.constant=.125*phase;
    std::vector<std::string> names;
    for (int b=0;b<blocks;++b) {
        names.push_back("x_"+std::to_string(b)); names.push_back("y_"+std::to_string(b));
        if (phase>=1) names.push_back("z_"+std::to_string(b));
        if (phase>=3) names.push_back("w_"+std::to_string(b));
    }
    if (reordered) { std::reverse(names.begin(),names.end()); std::rotate(names.begin(),names.begin()+17,names.end()); }
    std::unordered_map<std::string,int> index;
    for (const auto& name:names) {
        const char type=name.front();
        const double cost=type=='x'?3+.03*phase:type=='y'?4:type=='z'?4.5:1.2;
        index[name]=model.variable(name,0,type=='x'?20-phase:20,cost);
    }
    auto j=[&](char type,int b) { return index.at(std::string(1,type)+"_"+std::to_string(b)); };
    mlnd::Coefficients common;
    for (int b=0;b<blocks;++b) {
        mlnd::Coefficients capacity={{j('x',b),1},{j('y',b),2+.03*phase}};
        mlnd::Coefficients limit={{j('x',b),1}};
        if (phase>=1) {
            capacity.emplace_back(j('z',b),1);
            if (phase%2) limit.emplace_back(j('z',b),.4); // Subsequent updates must remove this coefficient.
            model.row("z_bound_"+std::to_string(b),'L',2+.1*phase,{{j('z',b),1}});
        }
        if (phase>=3) capacity.emplace_back(j('w',b),.2);
        model.row("capacity_"+std::to_string(b),'L',10+.05*phase,std::move(capacity));
        model.row("limit_"+std::to_string(b),phase==4?'E':'L',8,std::move(limit));
        common.emplace_back(j('y',b),1+(b%7)*.05);
    }
    model.row("common",'L',blocks*.85+.1*phase,std::move(common));
    if (reordered) std::reverse(model.rows.begin(),model.rows.end());
    return model;
}
void configure(mlnd::NativeSolver& solver) {
    solver.cplex.setParam(IloCplex::Param::RootAlgorithm,IloCplex::Dual);
    solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical,true);
    solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility,1e-9);
    solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality,1e-9);
    // Keep nontrivial simplex work in this controlled fixture; both arms use
    // this setting. Production retains its normal presolve setting.
    solver.cplex.setParam(IloCplex::Param::Preprocessing::Presolve,false);
}
}
int main() {
    using namespace mlnd;
    try {
        SolverOptions options; options.threads=1; options.seconds=10;
        DualRayWorkspace workspace;
        Json records=Json::array(); double warm_time=0,cold_time=0;
        int basis_submissions=0;
        for (int phase=0;phase<6;++phase) for (bool reordered : {false,true}) {
            const auto model=fixture(phase,reordered);
            Json diagnostics;
            auto& persistent=workspace.synchronize(model,options,diagnostics);
            configure(persistent);
            auto warm=workspace.optimize(10,diagnostics);
            NativeSolver fresh(model,options); configure(fresh);
            fresh.cplex.setParam(IloCplex::Param::Advance,0);
            auto cold=fresh.optimize(10);
            require(warm.has_solution && warm.optimal && cold.has_solution && cold.optimal,"LP arm did not solve optimally");
            require(point_residuals(model,warm.values).at("passed"),"Mapped persistent point violates the new source matrix");
            require(point_residuals(model,cold.values).at("passed"),"Cold point violates the source matrix");
            const double actual=objective_value(model,warm.values);
            require(std::abs(actual-warm.objective)<1e-7,"Persistent objective constant/coefficients were stale");
            require(std::abs(warm.objective-cold.objective)<1e-7,"Persistent and fresh LP objectives differ");
            if (phase>0 || reordered) {
                require(diagnostics.at("reused_solver"),"Append/reorder unexpectedly rebuilt the LP");
                require(diagnostics.at("basis_available_before_update"),"No basis retained between LP optimizations");
                require(diagnostics.at("basis_submitted"),"Retained simplex basis was not submitted");
                ++basis_submissions;
            }
            if (reordered) {
                require(diagnostics.at("new_variables")==0 && diagnostics.at("new_rows")==0,
                    "Source-order permutation created duplicate LP objects");
                require(diagnostics.at("changed_coefficients")==0,"Source-order permutation changed mathematical coefficients");
                require(diagnostics.at("simplex_iterations").get<long long>()==0,
                    "Unchanged reordered LP did not reuse its optimal simplex basis");
            }
            warm_time+=warm.seconds; cold_time+=cold.seconds;
            records.push_back({{"phase",phase},{"reordered",reordered},{"objective",warm.objective},
                {"cold_seconds",cold.seconds},{"cold_simplex_iterations",fresh.cplex.getNiterations64()},
                {"workspace",diagnostics}});
        }
        // Structural deletion must rebuild, never silently leave stale rows.
        auto smaller=fixture(0,true); Json reset;
        auto& rebuilt=workspace.synchronize(smaller,options,reset); configure(rebuilt);
        const auto solved=workspace.optimize(10,reset);
        require(!reset.at("reused_solver").get<bool>() && !reset.at("reset_reason").is_null(),"Deleted structure was silently retained");
        require(solved.optimal && point_residuals(smaller,solved.values).at("passed"),"Rebuilt deletion fixture failed");
        auto duplicate=smaller; duplicate.variables[1].name=duplicate.variables[0].name;
        bool rejected=false;
        try { Json report; workspace.synchronize(duplicate,options,report); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"Ambiguous variable identity accepted");
        std::cout<<Json{{"passed",true},{"rounds",records},{"basis_submissions",basis_submissions},
            {"warm_optimize_seconds",warm_time},{"cold_optimize_seconds",cold_time},
            {"reset_on_deletion",true},{"ambiguous_names_rejected",true}}.dump(2)<<'\n';
    } catch (const IloException& error) { std::cerr<<error.getMessage()<<'\n'; return 1; }
      catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
