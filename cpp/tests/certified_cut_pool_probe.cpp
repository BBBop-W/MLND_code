#include "mlnd/certified_cut_pool.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mlnd;
    int checks=0;
    auto require=[&](bool condition,const char* message) {
        if(!condition)throw std::runtime_error(message);++checks;
    };
    LinearModel model;
    const int x=model.variable("x",0,1,.1,true),theta=model.variable("theta",0,IloInfinity,1);
    model.row("a",'G',0,{{theta,1},{x,-3}});
    model.row("b",'G',4,{{theta,1},{x,4}});
    CertifiedCutPool pool;
    pool.remember(model.rows[0],0,model);pool.remember(model.rows[1],1,model);
    auto same=model.rows[0];same.name="different_provenance";
    std::reverse(same.coefficients.begin(),same.coefficients.end());
    require(pool.duplicate(same),"Exact duplicate with reordered coefficients was not recognized");
    auto distinct=same;distinct.rhs=std::nextafter(0.0,1.0);
    require(!pool.duplicate(distinct),"Near-equal cut was incorrectly discarded");
    SolverOptions options;options.seconds=10;options.threads=1;options.gap=0;
    NativeSolver solver(model,options,"cut_pool_probe");
    auto rhs=[&](int row,char sense,double value){solver.set_rhs(row,sense,value);};
    const auto reference=solver.optimize(10);
    require(reference.has_solution && std::abs(reference.objective-3.1)<1e-7,"Wrong full-cut reference");
    require(pool.suspend({0,10},1,1,model,rhs)==1,"Inactive cut was not suspended");
    require(model.rows.size()==2 && pool.metadata().at("dormant")==1,"Suspension changed stable row identities");
    const auto relaxed=solver.optimize(10);
    require(relaxed.has_solution && relaxed.bound<=reference.objective+1e-7,"Sleeping cuts invalidated the lower bound");
    require(relaxed.objective<reference.objective-1,"Fixture did not expose the missing cut");
    require(pool.reactivate(relaxed.values,model,rhs)==1,"Violated dormant cut was not restored");
    const auto recovered=solver.optimize(10);
    require(recovered.has_solution && std::abs(recovered.objective-reference.objective)<1e-7,
            "Reactivation failed to recover the full-cut optimum");
    pool.suspend({0,10},1,1,model,rhs);pool.restore_all(model,rhs);
    require(pool.metadata().at("dormant")==0 && model.rows[0].rhs==0 && model.rows[1].rhs==4,
            "Final restoration did not restore original certified rows");
    require(pool.reactivate(recovered.values,model,rhs)==0,"An active cut was restored twice");
    std::cout<<Json{{"passed",true},{"checks",checks},{"reference",reference.objective},
        {"relaxed",relaxed.objective},{"recovered",recovered.objective},{"pool",pool.metadata()}}.dump()<<'\n';
}
