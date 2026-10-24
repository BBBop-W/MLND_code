// Standalone catalogue exporter used to cross-check C++ against Python. This
// diagnostic does not initialize CPLEX or perform mathematical optimization.
#include "mlnd/input.hpp"
#include "mlnd/paths.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
template<class T> mlnd::Matrix<T> matrix(const mlnd::Json& source) {
    const int rows = static_cast<int>(source.size());
    const int cols = rows ? static_cast<int>(source.at(0).size()) : 0;
    mlnd::Matrix<T> out(rows, cols);
    for (int r = 0; r < rows; ++r) {
        if (source.at(r).size() != static_cast<std::size_t>(cols))
            throw std::invalid_argument("Ragged test fixture matrix");
        for (int c = 0; c < cols; ++c) {
            // The Python test encoder represents absent-flight infinity as null.
            out(r, c) = source.at(r).at(c).is_null()
                ? static_cast<T>(std::numeric_limits<double>::infinity())
                : source.at(r).at(c).get<T>();
        }
    }
    return out;
}

// Explicitly test-only fixture reader. Production instances always pass through
// read_input and the consecutive-csv/1 validation contract.
mlnd::Input read_fixture(const char* filename) {
    std::ifstream stream(filename);
    if (!stream) throw std::runtime_error("Cannot read test fixture");
    const auto p = mlnd::Json::parse(stream);
    mlnd::Input out;
    out.n_l = p.at("n_l"); out.n_h = p.at("n_h");
    out.n_od = static_cast<int>(p.at("demand").size());
    out.max_orders = p.at("max_orders");
    out.delta = p.at("delta"); out.operating_limit = p.at("operating_limit");
    out.flight_cost = p.at("flight_cost"); out.waiting_cost = p.at("waiting_cost");
    out.route_mode = p.at("route_mode");
    out.has_candidate_routes = !p.at("candidate_routes").is_null();
    out.demand = p.at("demand").get<std::vector<double>>();
    out.deadline = p.at("deadline").get<std::vector<double>>();
    out.flight_times = matrix<double>(p.at("flight_times"));
    out.rider_mean = matrix<double>(p.at("rider_mean"));
    out.first_cost = matrix<double>(p.at("first_cost"));
    out.terminal_cost = matrix<double>(p.at("terminal_cost"));
    out.receipt_mean = matrix<double>(p.at("receipt_mean"));
    out.removal_mean = matrix<double>(p.at("removal_mean"));
    out.kappa = matrix<double>(p.at("kappa"));
    out.allowed_links = matrix<int>(p.at("allowed_links"));
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("usage: mlnd_path_catalogue INSTANCE K OUTPUT_JSON or --fixture INPUT_JSON OUTPUT_JSON");
        const auto begin = std::chrono::steady_clock::now();
        const auto input = std::string(argv[1]) == "--fixture"
            ? read_fixture(argv[2]) : mlnd::read_input(argv[1], std::stoi(argv[2]));
        const auto loaded = std::chrono::steady_clock::now();
        const auto paths = mlnd::enumerate_paths(input);
        const auto enumerated = std::chrono::steady_clock::now();
        const mlnd::Json summary = {
            {"max_orders", input.max_orders},
            {"initial_segments", paths.initial_segments.size()},
            {"service_paths", paths.service_paths.size()},
            {"empty_paths", paths.empty_paths.size()},
            {"read_seconds", std::chrono::duration<double>(loaded - begin).count()},
            {"enumeration_seconds", std::chrono::duration<double>(enumerated - loaded).count()}
        };
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open catalogue output");
        output << mlnd::Json{{"summary", summary}, {"paths", paths}}.dump() << '\n';
        if (!output) throw std::runtime_error("Cannot write catalogue output");
        std::cout << summary.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
