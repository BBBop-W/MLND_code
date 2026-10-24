#include "mlnd/conditional_cut.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
long double value(double intercept, const std::vector<double>& coefficients,
                  const std::vector<double>& point) {
    long double answer = intercept;
    for (std::size_t j = 0; j < point.size(); ++j)
        answer += static_cast<long double>(coefficients[j]) * point[j];
    return answer;
}
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Invalid conditional certificate was accepted");
}
}

int main() {
    try {
        int checked = 0;
        // Layout: two facility coordinates, two assignment bits, one capacity.
        const std::vector<double> anchor{0, 0, 1, 0, 0};
        const auto example = mlnd::condition_assignment_cut(95, {-20,-30,5,17,0}, anchor, 2, 4);
        require(example.assignment_key == "10", "Wrong assignment fingerprint");
        require(std::abs(example.anchor_value_before - 100) < 1e-12, "Wrong absorbed anchor");
        require(example.anchor_value_after <= example.anchor_value_before + 1e-12,
                "Rounding increased the anchor certificate");
        require(example.coefficients[2] > 0 && example.coefficients[3] < 0,
                "Wrong conditional gate signs");
        std::mt19937_64 rng(43);
        std::uniform_real_distribution<double> coefficients(-200, 200);
        for (int trial = 0; trial < 128; ++trial) {
            std::vector<double> raw(5), x(5), target(5);
            for (double& c : raw) c = coefficients(rng);
            for (int j = 0; j < 5; ++j) x[j] = (trial >> j) & 1;
            const double intercept = coefficients(rng);
            const double floor = trial % 3 == 0 ? -20 : 0;
            const auto cut = mlnd::condition_assignment_cut(intercept, raw, x, 2, 4, floor);
            for (int mask = 0; mask < 32; ++mask) {
                for (int j = 0; j < 5; ++j) target[j] = (mask >> j) & 1;
                const bool same = target[2] == x[2] && target[3] == x[3];
                const long double gated = value(cut.intercept, cut.coefficients, target);
                if (same) {
                    const long double original = value(intercept, raw, target);
                    require(gated <= original + 1e-11L, "Conditional cut exceeded its raw face bound");
                    require(original - gated <= cut.additional_correction + 1e-11L,
                            "Additional rounding correction did not cover face loss");
                } else require(gated <= floor + 1e-11L, "Conditional cut remained active on another assignment");
                ++checked;
            }
            require(cut.gate_m >= 0 && std::isfinite(cut.gate_m), "Gate M must be finite and nonnegative");
        }
        // Fractional non-assignment coordinates are allowed for face checks;
        // assignment coordinates are always exactly binary.
        const auto fractional_y = mlnd::condition_assignment_cut(1, {2,3}, {.4,1}, 1, 2);
        require(std::isfinite(fractional_y.anchor_value_after), "Fractional y was rejected");
        rejects([&] { mlnd::condition_assignment_cut(0, {1,2}, {0,0}, 1, 3); });
        rejects([&] { mlnd::condition_assignment_cut(0, {1}, {0,0}, 0, 1); });
        rejects([&] { mlnd::condition_assignment_cut(0, {1}, {.5}, 0, 1); });
        rejects([&] { mlnd::condition_assignment_cut(0, {1}, {1.1}, 0, 0); });
        rejects([&] { mlnd::condition_assignment_cut(std::numeric_limits<double>::infinity(), {1}, {1}, 0, 1); });
        rejects([&] { mlnd::condition_assignment_cut(0, {std::numeric_limits<double>::quiet_NaN()}, {1}, 0, 1); });
        rejects([&] { mlnd::condition_assignment_cut(std::numeric_limits<double>::max(),
                    {std::numeric_limits<double>::max()}, {1}, 0, 1); });
        std::cout << "{\"passed\":true,\"binary_design_checks\":" << checked
                  << ",\"invalid_inputs_rejected\":7}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
