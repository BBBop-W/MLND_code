#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace mlnd {
using Json = nlohmann::json;

template<class T> struct Matrix {
    int rows = 0, cols = 0;
    std::vector<T> values;
    Matrix() = default;
    Matrix(int r, int c) : rows(r), cols(c), values(static_cast<std::size_t>(r) * c) {}
    T& operator()(int r, int c) { return values[static_cast<std::size_t>(r) * cols + c]; }
    const T& operator()(int r, int c) const { return values[static_cast<std::size_t>(r) * cols + c]; }
};

// All matrices use row-major storage. Time is hours, demand is orders/hour.
struct Input {
    int n_l = 0, n_h = 0, n_od = 0, n_levels = 0, max_orders = 2;
    double delta = 0, operating_limit = 0, fleet_cost = 0;
    double flight_cost = 0, waiting_cost = 0;
    std::string name, route_mode, flow_policy, model_version;
    std::filesystem::path instance_path;
    std::vector<std::string> base_ids, hub_ids, od_ids, origin_ids;
    std::vector<std::string> destination_ids, level_ids, flight_node_ids;
    std::vector<double> demand, ground_cost, deadline, base_cost, slot_epsilon;
    Matrix<double> flight_times, rider_mean, first_cost, terminal_cost;
    Matrix<double> receipt_mean, removal_mean, slot_cost, link_cost, kappa, occupancy_limit;
    Matrix<int> slot_counts, allowed_links;
    // Coordinates and diagnostic coefficients are retained for result reporting.
    Matrix<double> bases, hubs, origins, destinations;
    Matrix<double> pickup_share, last_rider_cost, last_rider_mean;
    std::vector<int> od_origin, od_destination;
    std::vector<std::vector<int>> candidate_routes;
    bool has_candidate_routes = false;
    Json metadata, model_json, manifest;

    int n_nodes() const { return n_l + n_h; }
    bool origin_assignment() const { return max_orders == 1; }
    int assignment_destinations() const { return origin_assignment() ? 1 : n_l; }
    int assignment_pool_count() const { return n_l * assignment_destinations(); }
    int assignment_count() const { return n_od * assignment_pool_count(); }
    int assignment_offset(int r, int l, int m = 0) const {
        return (r * n_l + l) * assignment_destinations() + (origin_assignment() ? 0 : m);
    }
    const char* assignment_scope() const { return origin_assignment() ? "origin" : "base_pair"; }
    double effective_delta() const { return max_orders == 1 ? 0.0 : delta; }
};

inline double assignment_value(const Input& d, const Json& values, int r, int l, int m = 0) {
    const auto& origin = values.at(r).at(l);
    return d.origin_assignment() ? origin.get<double>() : origin.at(m).get<double>();
}

// max_orders=0 uses the default in model.json; 1/2 choose the declared case.
Input read_input(const std::filesystem::path& path, int max_orders = 0,
                 bool verify_hashes = true);
std::string sha256_file(const std::filesystem::path& path);
} // namespace mlnd
