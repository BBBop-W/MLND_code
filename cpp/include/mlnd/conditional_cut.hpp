#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlnd {

struct ConditionalCutResult {
    double intercept = 0;
    std::vector<double> coefficients;
    double affine_intercept = 0;
    double gate_m = 0, affine_upper_bound = 0, gate_rounding_margin = 0;
    double additional_correction = 0;
    double anchor_value_before = 0, anchor_value_after = 0;
    std::string assignment_key;
};

namespace conditional_cut_detail {
inline double finite(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("Conditional cut arithmetic must remain finite");
    return value;
}
// One outward ulp bounds a correctly rounded binary64 addition. Do not enable
// fast-math: the certificate relies on normal IEEE arithmetic and finite data.
inline double add_down(double a, double b) {
    return finite(std::nextafter(finite(a + b), -std::numeric_limits<double>::infinity()));
}
inline double add_up(double a, double b) {
    return finite(std::nextafter(finite(a + b), std::numeric_limits<double>::infinity()));
}
inline double multiply_down(double a, double b) {
    return finite(std::nextafter(finite(a * b), -std::numeric_limits<double>::infinity()));
}
inline double multiply_up(double a, double b) {
    return finite(std::nextafter(finite(a * b), std::numeric_limits<double>::infinity()));
}
inline double evaluate(double intercept, const std::vector<double>& coefficients,
                       const std::vector<double>& point) {
    long double value = intercept;
    for (std::size_t i = 0; i < coefficients.size(); ++i)
        value += static_cast<long double>(coefficients[i]) * point[i];
    return finite(static_cast<double>(value));
}
} // namespace conditional_cut_detail

// The raw affine inequality must already be certified for the complete path
// domain at the anchor assignment. Other design coordinates lie in [0,1].
// This helper only gates that certificate; it cannot certify a raw dual or an
// incomplete pricing pass. The returned inequality is valid for every binary
// assignment, and specializes to the safely rounded raw inequality on its face.
inline ConditionalCutResult condition_assignment_cut(
    double raw_intercept, const std::vector<double>& raw_coefficients,
    const std::vector<double>& anchor, int assignment_begin, int assignment_end,
    double lower_bound = 0) {
    using namespace conditional_cut_detail;
    if (raw_coefficients.size() != anchor.size() || assignment_begin < 0 ||
        assignment_end < assignment_begin ||
        static_cast<std::size_t>(assignment_end) > anchor.size())
        throw std::invalid_argument("Invalid conditional cut assignment layout");
    finite(raw_intercept);
    finite(lower_bound);
    ConditionalCutResult result;
    result.coefficients = raw_coefficients;
    double affine_low = raw_intercept, affine_high = raw_intercept;
    std::size_t ones = 0;
    for (std::size_t i = 0; i < anchor.size(); ++i) {
        finite(raw_coefficients[i]);
        finite(anchor[i]);
        if (anchor[i] < 0 || anchor[i] > 1)
            throw std::invalid_argument("Conditional cut design coordinates must lie in [0,1]");
        if (static_cast<int>(i) < assignment_begin || static_cast<int>(i) >= assignment_end)
            continue;
        if (anchor[i] != 0 && anchor[i] != 1)
            throw std::invalid_argument("Conditional cut assignment must be exactly binary");
        result.assignment_key.push_back(anchor[i] == 1 ? '1' : '0');
        if (anchor[i] == 1) {
            ++ones;
            affine_low = add_down(affine_low, raw_coefficients[i]);
            affine_high = add_up(affine_high, raw_coefficients[i]);
        }
    }
    result.affine_intercept = affine_low;
    double upper = affine_low;
    double scale = add_up(1, std::abs(affine_low));
    for (std::size_t i = 0; i < raw_coefficients.size(); ++i) {
        if (static_cast<int>(i) >= assignment_begin && static_cast<int>(i) < assignment_end)
            continue;
        scale = add_up(scale, std::abs(raw_coefficients[i]));
        if (raw_coefficients[i] > 0) upper = add_up(upper, raw_coefficients[i]);
    }
    result.affine_upper_bound = upper;
    const double minimum_m = std::max(0.0, add_up(upper, -lower_bound));
    const double margin = multiply_up(128 * std::numeric_limits<double>::epsilon(), scale);
    result.gate_m = add_up(minimum_m, margin);
    result.gate_rounding_margin = add_up(result.gate_m, -minimum_m);
    const double count = static_cast<double>(ones);
    const double product_up = ones ? multiply_up(result.gate_m, count) : 0;
    const double product_down = ones ? multiply_down(result.gate_m, count) : 0;
    result.intercept = add_down(affine_low, -product_up);
    for (int i = assignment_begin; i < assignment_end; ++i)
        result.coefficients[static_cast<std::size_t>(i)] = anchor[static_cast<std::size_t>(i)] == 1
            ? result.gate_m : -result.gate_m;
    // Upper bound on the extra constant loss from absorption/gating rounding.
    // The gate M itself cancels on the anchor face and is not a dual correction.
    result.additional_correction = std::max(0.0,
        add_up(add_up(affine_high, -result.intercept), -product_down));
    result.anchor_value_before = evaluate(raw_intercept, raw_coefficients, anchor);
    result.anchor_value_after = evaluate(result.intercept, result.coefficients, anchor);
    return result;
}

} // namespace mlnd
