// Test native decision extraction and output using the feasible all-courier
// solution, plus append-only pool growth after an incumbent snapshot. No
// optimization is performed; full solves are checked separately.
#include "mlnd/output.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: mlnd_output_probe INSTANCE K OUTPUT_DIR");
        const auto input = mlnd::read_input(argv[1], std::stoi(argv[2]));
        const auto paths = mlnd::enumerate_paths(input);
        const auto formulation = mlnd::build_formulation(input, paths);
        std::vector<double> values(formulation.matrix.variables.size(), 0);
        for (int r = 0; r < input.n_od; ++r) values[formulation.courier[r]] = input.demand[r];
        const auto extracted = mlnd::solution_json(input, paths, formulation, values);
        const auto baseline = mlnd::all_courier_result(input, paths);
        if (extracted.at("_solution") != baseline.at("_solution") ||
            extracted.at("objective") != baseline.at("objective"))
            throw std::runtime_error("All-courier extraction differs from baseline");
        const auto design = mlnd::solution_design(input, extracted);
        if (design.size() != static_cast<std::size_t>(formulation.layout.count))
            throw std::runtime_error("Extracted design shape differs");
        for (double value : design)
            if (value != 0) throw std::runtime_error("All-courier design must be zero");

        auto generated = mlnd::prepare_path_support(input);
        auto earlier = mlnd::all_courier_result(input, generated);
        earlier["audit_metadata"] = "preserve_after_pool_growth";
        if (generated.complete || !generated.support_complete ||
            !earlier["_solution"]["flow"].empty() || !earlier["_solution"]["empty_flow"].empty())
            throw std::runtime_error("Implicit baseline unexpectedly materialized task paths");
        if (paths.service_paths.empty()) throw std::runtime_error("Output fixture needs a service path");
        const int first = mlnd::register_service_path(generated, paths.service_paths.front());
        if (first != 0 || mlnd::register_service_path(generated, paths.service_paths.front()) != first ||
            generated.service_paths.size() != 1)
            throw std::runtime_error("Service registration is not stable and idempotent");
        if (!paths.empty_paths.empty()) {
            const int empty = mlnd::register_empty_path(generated, paths.empty_paths.front());
            if (empty != 0 || mlnd::register_empty_path(generated, paths.empty_paths.front()) != empty ||
                generated.empty_paths.size() != 1)
                throw std::runtime_error("Empty registration is not stable and idempotent");
        }
        mlnd::synchronize_result_paths(input, generated, earlier);
        if (earlier.at("objective") != baseline.at("objective") ||
            earlier.at("audit_metadata") != "preserve_after_pool_growth" ||
            earlier["_solution"]["flow"].size() != generated.service_paths.size() ||
            earlier["_solution"]["empty_flow"].size() != generated.empty_paths.size() ||
            earlier["_missions"].size() != generated.service_paths.size() ||
            earlier["_relocations"].size() != generated.empty_paths.size())
            throw std::runtime_error("Earlier incumbent did not synchronize after pool growth");
        for (const char* field : {"flow", "empty_flow"})
            for (const auto& value : earlier["_solution"][field])
                if (value.get<double>() != 0) throw std::runtime_error("New column received nonzero old-incumbent flow");

        // A synthetic nonzero payload tests index/value preservation, not
        // operational feasibility; only the actual all-courier solution is
        // written below and used by independent physical validation.
        auto marked = earlier;
        marked["_solution"]["flow"][first] = 0.25;
        mlnd::synchronize_result_paths(input, generated, marked);
        const auto old_objective = marked.at("objective");
        const auto old_metrics = marked.at("metrics");
        if (paths.service_paths.size() > 1)
            mlnd::register_service_path(generated, paths.service_paths[1]);
        mlnd::synchronize_result_paths(input, generated, marked);
        if (marked["_solution"]["flow"][first] != 0.25 ||
            marked["_missions"][first]["flow"] != 0.25 ||
            marked.at("objective") != old_objective || marked.at("metrics") != old_metrics ||
            mlnd::service_path_signature(generated.service_paths[first]) !=
                mlnd::service_path_signature(paths.service_paths.front()))
            throw std::runtime_error("Appending columns changed earlier nonzero flows or metrics");
        for (std::size_t i = 1; i < generated.service_paths.size(); ++i)
            if (marked["_solution"]["flow"][i].get<double>() != 0)
                throw std::runtime_error("Appended service column did not receive zero flow");
        mlnd::synchronize_result_paths(input, generated, earlier);
        if (earlier.at("objective") != baseline.at("objective"))
            throw std::runtime_error("Repeated synchronization changed the baseline objective");
        const auto synchronized = earlier;
        mlnd::synchronize_result_paths(input, generated, earlier);
        if (synchronized != earlier) throw std::runtime_error("Output synchronization is not idempotent");
        mlnd::write_result(argv[3], input, paths, baseline);
        mlnd::write_result(std::filesystem::path(argv[3]) / "generated_pool", input, generated, earlier);
        std::cout << mlnd::Json{{"passed", true}, {"max_orders", input.max_orders},
            {"objective", baseline.at("objective")}, {"files_written", true},
            {"earlier_incumbent_pool_growth", true}, {"nonzero_flow_preserved", true},
            {"registration_and_synchronization_idempotent", true}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
