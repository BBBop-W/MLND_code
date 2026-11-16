#include "mlnd/linear_model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void close(double actual,double expected,const char* message) {
    require(std::abs(actual-expected)<=2e-6*(1+std::abs(expected)),message);
}
void check_duality(const mlnd::LinearModel& model,const mlnd::NativeDualPoint& dual,double optimum) {
    require(dual.available,dual.error.c_str());
    std::vector<double> stationary(model.variables.size(),0);
    double lower=model.constant;
    for(std::size_t i=0;i<model.rows.size();++i) {
        const auto& row=model.rows[i];const double price=dual.row_prices.at(i);
        require(row.sense!='G' || price>=-2e-6,"Wrong sign for >= row price");
        require(row.sense!='L' || price<=2e-6,"Wrong sign for <= row price");
        lower+=price*row.rhs;
        for(auto [j,a]:row.coefficients)stationary.at(j)+=price*a;
    }
    for(std::size_t j=0;j<model.variables.size();++j) {
        const double price=dual.bound_prices.at(j);
        stationary[j]+=price;
        lower+=price*(price>=0?model.variables[j].lower:model.variables[j].upper);
    }
    for(std::size_t k=0;k<model.cones.size();++k) {
        const auto& c=model.cones[k];const auto& z=dual.cone_vectors.at(k);
        require(z[0]>=-2e-6 && z[1]>=-2e-6,"Negative rotated dual cone coordinate");
        require(z[2]*z[2]<=4*z[0]*z[1]/c.factor+2e-6,"Extracted vector is outside the dual cone");
        stationary[c.up]+=z[0];stationary[c.down]+=z[1];stationary[c.one]+=z[2];
    }
    for(std::size_t j=0;j<stationary.size();++j)
        close(stationary[j],model.variables[j].cost,"Native dual fails stationarity");
    close(lower,optimum,"Native dual disagrees with the analytic optimum");
}

mlnd::Json cone_case(double factor,double one,bool fix_down,bool fixed_one_bounds=false) {
    using namespace mlnd;
    LinearModel model;
    const int u=model.variable("up",0,10,1);
    const int v=model.variable("down",0,10,fix_down?0:1);
    const int w=model.variable("one",fixed_one_bounds?one:-10,fixed_one_bounds?one:10);
    if(!fixed_one_bounds)model.row("fix_one",'E',one,{{w,1}});
    if(fix_down)model.row("fix_down",'E',1,{{v,1}});
    model.cones.push_back({u,v,w,factor});
    SolverOptions options;
    NativeSolver solver(model,options);
    const auto solved=solver.optimize(10);
    require(solved.optimal,"Rotated cone reference did not solve");
    const auto dual=solver.read_dual(model);
    const double optimum=fix_down?one*one/factor:2*std::abs(one)/std::sqrt(factor);
    close(solved.objective,optimum,"Rotated cone objective disagrees with analytic optimum");
    check_duality(model,dual,optimum);
    if(one!=0 && !fix_down) {
        close(dual.cone_vectors.at(0)[0],1,"Rotated cone up mapping is wrong");
        close(dual.cone_vectors.at(0)[1],1,"Rotated cone down mapping is wrong");
        close(dual.cone_vectors.at(0)[2],-2*std::copysign(1,one)/std::sqrt(factor),
              "Rotated cone signed coordinate mapping is wrong");
    }
    return {{"factor",factor},{"one",one},{"fixed_down",fix_down},{"fixed_one_bounds",fixed_one_bounds},
            {"objective",solved.objective},{"row_prices",dual.row_prices},
            {"bound_prices",dual.bound_prices},{"cone_vectors",dual.cone_vectors}};
}
}

int main() {
    using namespace mlnd;
    try {
        LinearModel model;
        model.constant=3.25;
        const int x=model.variable("x",0,10,2);
        model.variable("at_upper",1,4,-5);
        model.variable("at_lower",2,8,7);
        model.row("demand",'G',3,{{x,1}});
        SolverOptions options;
        NativeSolver solver(model,options);
        require(!solver.read_dual(model).available,"An unsolved model exposed a dual solution");
        const auto solved=solver.optimize(10);
        require(solved.optimal,"LP reference did not solve");
        const auto dual=solver.read_dual(model);
        require(dual.available,dual.error.c_str());
        close(solved.objective,3.25,"LP objective disagrees with analytic optimum");
        close(dual.row_prices.at(0),2,"LP demand multiplier is incorrect");
        close(dual.bound_prices.at(1),-5,"Upper-bound multiplier is incorrect");
        close(dual.bound_prices.at(2),7,"Lower-bound multiplier is incorrect");
        require(dual.cone_vectors.empty(),"LP unexpectedly has cone multipliers");
        check_duality(model,dual,3.25);
        const int z=model.variable("new_column",0,10,1);
        model.coefficient(0,z,1);
        solver.add_variable(model.variables[z],{{0,1}});
        const auto updated=solver.optimize(10);
        require(updated.optimal,"Incremental LP did not solve");
        const auto updated_dual=solver.read_dual(model);
        close(updated.objective,.25,"Incremental LP optimum is wrong");
        check_duality(model,updated_dual,.25);
        close(updated_dual.row_prices.at(0),1,"Incremental LP retained its old row price");
        close(updated_dual.bound_prices.at(0),1,"Incremental LP new reduced cost is wrong");
        auto wrong=model;wrong.rows.clear();
        require(!solver.read_dual(wrong).available,"Mismatched source model was accepted");
        LinearModel maximization;
        maximization.minimize=false;
        maximization.variable("x",0,1,1);
        NativeSolver max_solver(maximization,options);
        require(!max_solver.read_dual(maximization).available,"Unsupported maximization dual was accepted");
        LinearModel integer_model;
        integer_model.variable("binary",0,1,1,true);
        NativeSolver integer_solver(integer_model,options);
        require(!integer_solver.read_dual(integer_model).available,"A MIP exposed continuous dual multipliers");
        Json cones=Json::array();
        cones.push_back(cone_case(1,1,false));
        cones.push_back(cone_case(4,-2,false));
        cones.push_back(cone_case(1,0,true));
        cones.push_back(cone_case(1,0,false));
        // The non-perspective queue formulation fixes the squared coordinate
        // through its bounds. CPLEX's default preprocessing can discard the
        // dual even though the original conic problem solves optimally.
        cones.push_back(cone_case(1,1,true,true));
        std::cout<<Json{{"passed",true},{"linear_objective",solved.objective},
            {"incremental_objective",updated.objective},{"cones",cones}}.dump(2)<<'\n';
    } catch(const IloException& e) {std::cerr<<e<<'\n';return 1;}
      catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
