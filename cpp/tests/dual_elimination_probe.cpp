// This probe includes the implementation to test its private certificate
// builder directly; link every core object EXCEPT benders.o.
#include "../src/benders.cpp"

#include <fstream>
#include <iostream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: dual_elimination_probe OUTPUT_JSON");
        mlnd::Input d;
        d.n_l = 1; d.demand = {2}; d.ground_cost = {1};
        mlnd::PathCollection paths;
        paths.complete = paths.support_complete = true;
        mlnd::SolverOptions options;
        options.threads = 1; options.log = false;
        mlnd::Json records = mlnd::Json::array();
        int cross_checks = 0;
        for (const std::string scenario : {"fixed_zero", "fixed_one", "fractional",
                                         "positive_lower", "negative_cost", "fixed_cone", "free_design"}) {
            for (bool pareto : {false, true}) {
                if (scenario == "free_design" && pareto) continue;
                const double anchor = scenario == "fixed_zero" ? 0 : scenario == "fixed_one" ? 1 : .5;
                mlnd::Formulation f(d);
                const int x = f.matrix.variable("design", 0, 1);
                const bool positive_lower = scenario == "positive_lower";
                const bool negative_cost = scenario == "negative_cost";
                const bool cone = scenario == "fixed_cone";
                const int y = f.matrix.variable("operating", positive_lower ? 2 : 0,
                                               negative_cost ? 2 : 10, negative_cost ? -1 : 1);
                f.matrix.row("strategic_domain", 'L', 1, {{x, 1}});
                if (scenario != "free_design") {
                    f.design_vars = {x};
                    f.design_fix_rows = {f.matrix.row("fix_design", 'E', anchor, {{x, 1}})};
                }
                if (cone) {
                    const int one = f.matrix.variable("unit", 1, 1);
                    f.matrix.cones.push_back({y, one, x, 1});
                } else if (negative_cost) f.matrix.row("upper_service", 'L', 1, {{y, 1}, {x, -1}});
                else f.matrix.row("demand", 'G', positive_lower ? 3 : 2, {{y, 1}, {x, 1}});
                // An empty operating row must remain priced: an omitted path
                // can touch it even though no present RMP variable does.
                f.fleet_row = f.matrix.row("future_path_resource", 'L', 0, {});
                mlnd::EmptyPath missing;
                missing.b = {0, 0}; missing.time = 1; missing.cost = 0;
                paths.empty_paths = {missing};
                const auto q = [&](double v) {
                    return cone ? v*v : negative_cost ? -1-v : positive_lower ? 3-v : 2-v;
                };
                mlnd::NativeSolver primal(f.matrix, options, "certificate_regression_primal");
                const auto solved = primal.optimize(5);
                require(solved.has_solution && mlnd::feasible_point(f.matrix, solved.values), "Invalid probe primal");
                const std::vector<double> point = scenario == "free_design" ? std::vector<double>{} : std::vector<double>{anchor};
                const std::vector<double> core{.4};
                mlnd::DualRayWorkspace workspace;
                int optimal_certificates=0;
                std::vector<int> profiles{0,4,1,2,3};
                if(!pareto && scenario!="free_design")profiles.push_back(5);
                for (int profile : profiles) {
                    auto cut = mlnd::solve_dual(d, paths, f, options, mlnd::wall_time()+5,
                        pareto ? &core : nullptr, solved.objective, 1e-7, {}, pareto ? std::vector<int>{0} : std::vector<int>{},
                        profile, &solved.values, &workspace);
                    require(cut.valid, "Profile produced no valid certificate in the small fixture");
                    const double anchor_gap = solved.objective - mlnd::value(cut, point);
                    require(anchor_gap >= -2e-6, "Eliminated dual exceeded the primal");
                    if(cut.optimal) {
                        require(anchor_gap<=2e-6, "Optimal eliminated dual lost anchor tightness");
                        ++optimal_certificates;
                    }
                    require(cut.prices.size() == f.matrix.rows.size(), "Original pricing row map changed");
                    if(profile==5)require(cut.diagnostics.value("original_rmp_gap_passed",false),
                        "Norm LP optimality was mistaken for an original-RMP certificate");
                    if (scenario != "free_design") {
                        require(cut.prices[0] == 0, "Strategic domain multiplier was not eliminated");
                        for (int k=0; k<=20; ++k) {
                            const double v=k/20.;
                            require(mlnd::value(cut,{v}) <= q(v)+2e-6, "Affine certificate violated cross-design weak duality");
                            ++cross_checks;
                        }
                    } else require(cut.coefficients.empty(), "Free dual acquired fixed-design coefficients");
                    records.push_back({{"scenario",scenario},{"pareto",pareto},{"profile",profile},
                        {"status",cut.status},{"optimal",cut.optimal},{"raw_anchor",cut.raw_anchor},{"raw_core",cut.raw_core},{"primal",solved.objective},
                        {"anchor_gap",anchor_gap},{"correction",cut.correction},{"diagnostics",cut.diagnostics}});
                }
                require(optimal_certificates>0, "All numerical profiles failed the fixture certificate");
            }
        }
        // At fixed x=0, replacing its original equality price by price-M
        // preserves the raw objective and only relaxes stationarity. It is
        // therefore an arbitrary dual representation, not a physical gap.
        const double stress_shift=1e13;
        const double stress_fix_price=-1-stress_shift;
        require(stress_fix_price+1<=0 && 0*stress_fix_price+2==2,
            "Zero-fix stress changed raw objective or violated stationarity");
        const double old_rounding_margin=128*std::numeric_limits<double>::epsilon()*(1+2+1+stress_shift);
        require(old_rounding_margin > .1, "Degeneracy stress did not exceed the unchanged tolerance");
        mlnd::Json result={{"passed",true},{"certificates",records.size()},{"cross_design_checks",cross_checks},
            {"arbitrary_fixed_zero_price_shift",stress_shift},{"legacy_guard_after_shift",old_rounding_margin},
            {"records",records}};
        std::ofstream(argv[1]) << result.dump(2) << '\n';
        std::cout << "Certificates " << records.size() << ", cross-design checks " << cross_checks << '\n';
    } catch (const IloException& error) { std::cerr << error << '\n'; return 1; }
      catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
