#include "mlnd/formulation.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Fixture {
    mlnd::Input input;
    mlnd::Formulation f;
    std::vector<double> values;
    int design, flow;
    explicit Fixture(double rho, double deficit = 2e-6) : f(input) {
        auto& m = f.matrix;
        design = m.variable("facility", 0, 1, 2, true);
        f.design_vars = {design};
        flow = m.variable("flow", 0, 10, 3);
        f.flow = {flow};
        f.rho = {m.variable("rho", 0, 1)};
        f.idle = {m.variable("idle", 0, 1e9)};
        f.fleet = m.variable("fleet", 0, 1e9, 1);
        const int up = m.variable("cone_up", 0, 1e9 + 1);
        const int down = m.variable("cone_down", 0, 1);
        const int one = m.variable("cone_one", 1, 1);
        m.row("rho_open", 'L', 0, {{f.rho[0], 1}, {design, -1}});
        m.row("flow_fixed", 'E', 2, {{flow, 1}});
        m.row("cone_up_link", 'E', 1, {{up, 1}, {f.idle[0], -1}});
        m.row("cone_down_link", 'E', 1, {{down, 1}, {f.rho[0], 1}});
        f.fleet_row = m.row("fleet_use", 'L', 0, {{f.idle[0], 1}, {flow, 3}, {f.fleet, -1}});
        m.cones.push_back({up, down, one, 1});
        const double idle = rho < 1 ? std::max(0.0, rho / (1-rho) - deficit) : 10;
        values = {1, 2, rho, idle, idle + 6, idle + 1, 1-rho, 1};
    }
};
void check_preserved(const Fixture& fixture, const std::vector<double>& before) {
    for (int j : {fixture.design, fixture.flow, fixture.f.rho[0]})
        require(fixture.values[j] == before[j], "Repair changed a protected decision");
    require(fixture.values[fixture.f.idle[0]] >= before[fixture.f.idle[0]], "Idle slack decreased");
    require(fixture.values[fixture.f.fleet] >= before[fixture.f.fleet], "Fleet slack decreased");
}
}

int main() {
    try {
        mlnd::Json cases = mlnd::Json::array();
        for (double rho : {.5, .999999}) {
            Fixture fixture(rho, rho > .9 ? 1.0 : 2e-6);
            const auto before = fixture.values;
            require(!mlnd::point_residuals(fixture.f.matrix, before).at("passed").get<bool>(),
                    "Regression fixture did not violate its cone");
            auto result = mlnd::repair_inventory_primal(fixture.f, fixture.values);
            require(result.at("applied") && result.at("passed"), "Repair did not pass");
            require(mlnd::point_residuals(fixture.f.matrix, fixture.values).at("passed"), "Full matrix invalid after repair");
            require(result.at("cost_increase").get<double>() > 0, "Repaired cost was not recomputed");
            require(result.at("after").at("max_scaled_cone_violation").get<double>() <= 0, "Cone not repaired to feasible side");
            check_preserved(fixture, before);
            cases.push_back({{"rho",rho},{"result",result}});
        }
        {
            Fixture fixture(0);
            const auto before = fixture.values;
            const auto result = mlnd::repair_inventory_primal(fixture.f, fixture.values);
            require(result.at("passed") && !result.at("applied").get<bool>(), "Feasible rho=0 point should be retained");
            require(fixture.values == before, "Feasible point changed");
            cases.push_back({{"case","rho_zero_unchanged"},{"passed",true}});
        }
        for (const std::string reason : {"availability_outside_repair_domain", "fleet_bound_insufficient",
                                         "inventory_bound_insufficient", "unrelated_matrix_violation"}) {
            Fixture fixture(reason == "availability_outside_repair_domain" ? 1 : .5);
            if (reason == "fleet_bound_insufficient") fixture.f.matrix.variables[fixture.f.fleet].upper = 6.999999;
            if (reason == "inventory_bound_insufficient") fixture.f.matrix.variables[fixture.f.idle[0]].upper = .999999;
            if (reason == "unrelated_matrix_violation") fixture.values[fixture.flow] = 1.9;
            const auto before = fixture.values;
            const auto result = mlnd::repair_inventory_primal(fixture.f, fixture.values);
            require(!result.at("passed").get<bool>() && result.at("reason") == reason, "Invalid point not explicitly rejected");
            require(fixture.values == before, "Rejected repair changed caller values");
            cases.push_back({{"case",reason},{"passed",true}});
        }
        {
            Fixture fixture(.5);
            fixture.values[fixture.f.fleet] = 50;
            const auto before = fixture.values;
            auto result = mlnd::repair_inventory_primal(fixture.f, fixture.values);
            require(result.at("passed"), "Slack-fleet repair failed");
            require(fixture.values[fixture.f.fleet] == 50, "Existing fleet slack decreased");
            check_preserved(fixture, before);
            cases.push_back({{"case","existing_fleet_slack_preserved"},{"passed",true}});
        }
        std::cout << mlnd::Json{{"passed",true},{"optimization_calls",0},{"cases",cases}}.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
