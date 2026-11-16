#include "mlnd/formulation.hpp"
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
template<class T> mlnd::Matrix<T> singleton(T value) {
    mlnd::Matrix<T> matrix(1,1); matrix(0,0)=value; return matrix;
}
mlnd::Input fixture(int k) {
    mlnd::Input d;
    d.n_l=d.n_h=d.n_od=d.n_levels=1; d.max_orders=k;
    d.name="fixed_design_boundary"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={2}; d.ground_cost={5}; d.deadline={1}; d.base_cost={.5}; d.slot_epsilon={.05};
    d.flight_times=mlnd::Matrix<double>(2,2); d.flight_times.values={0,.01,.01,0};
    d.allowed_links=mlnd::Matrix<int>(2,2); d.allowed_links.values={0,1,1,0};
    d.link_cost=mlnd::Matrix<double>(2,2);
    d.rider_mean=singleton(.02); d.first_cost=singleton(1.0); d.terminal_cost=singleton(.5);
    d.receipt_mean=singleton(.03); d.removal_mean=singleton(.01);
    d.slot_counts=singleton(10); d.slot_cost=singleton(.1);
    d.occupancy_limit=singleton(6.1690072893953225); d.kappa=singleton(1.0);
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    return d;
}
}

int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Usage: mlnd_fixed_design_probe OUTPUT_JSON");
        mlnd::Json records=mlnd::Json::array();
        for(int k:{1,2}) {
            const auto d=fixture(k);
            const auto paths=mlnd::enumerate_paths(d);
            mlnd::SolverOptions options; options.seconds=5; options.gap=1e-8;
            const mlnd::DesignLayout layout(d);
            std::vector<double> zero(layout.count,0);
            const auto result=mlnd::solve_full(d,paths,options,&zero);
            records.push_back({{"K",k},{"case","legal_zero"},{"rejected",false},
                {"objective",result.at("objective")},{"served",result.at("served")},
                {"has_solver_solution",result.at("has_solver_solution")},
                {"design",mlnd::solution_design(d,result)},{"expected_design",zero}});

            // Assignment alone does not force service: rho=0 remains feasible.
            auto no_links=zero; no_links[layout.facility]=1; no_links[layout.assignment]=1;
            const auto assigned=mlnd::solve_full(d,paths,options,&no_links);
            records.push_back({{"K",k},{"case","assigned_without_airlines"},{"rejected",false},
                {"objective",assigned.at("objective")},{"served",assigned.at("served")},
                {"has_solver_solution",assigned.at("has_solver_solution")},
                {"design",mlnd::solution_design(d,assigned)},{"expected_design",no_links}});

            for(const std::string scenario:{"closed_assigned_base","no_solve_budget"}) {
                auto fixed=zero;
                auto budget=options;
                if(scenario=="closed_assigned_base") fixed[layout.assignment]=1;
                else budget.seconds=0;
                bool rejected=false;
                std::string error;
                try { mlnd::solve_full(d,paths,budget,&fixed); }
                catch(const std::runtime_error& e) { rejected=true; error=e.what(); }
                if(!rejected || error.find("Fixed-design solve returned no feasible point")==std::string::npos)
                    throw std::runtime_error("Fixed-design no-point boundary was not rejected: "+scenario+" "+error);
                records.push_back({{"K",k},{"case",scenario},{"rejected",true},{"error",error}});
            }
        }
        std::ofstream stream(argv[1]);
        if(!stream) throw std::runtime_error("Cannot write probe results");
        stream<<records.dump(2)<<'\n';
        std::cout<<"Fixed-design boundary cases: "<<records.size()<<'\n';
    } catch(const IloException& error) {
        std::cerr<<error<<'\n'; return 1;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
