// Test native decision extraction and output using the feasible all-courier
// solution, plus append-only pool growth after an incumbent snapshot. No
// optimization is performed; full solves are checked separately.
#include "mlnd/output.hpp"
#include "mlnd/preprocessed_demand.hpp"
#include <chrono>
#include <iostream>
#include <map>

namespace {
auto statistics(const mlnd::Json& result) {
    std::map<std::tuple<int,int,int>,double> costs;
    std::map<std::pair<int,int>,int> traversals;
    for (const auto& record : result.at("_missions")) {
        const auto& p = record.at("path");
        const auto key=std::make_tuple(p.at("first_od").get<int>(),
            p.at("second_od").is_null() ? -1 : p.at("second_od").get<int>(),p.at("b").at(0).get<int>());
        const double cost=p.at("cost");
        auto [entry, inserted]=costs.emplace(key,cost);
        if (!inserted) entry->second=std::min(entry->second,cost);
    }
    for (const char* name : {"_missions","_relocations"})
        for (const auto& record : result.at(name))
            for (const auto& arc : record.at("path").at("arc_counts")) {
                const auto key=std::make_pair(arc.at(0).get<int>(),arc.at(1).get<int>());
                traversals[key]=std::max(traversals[key],arc.at(2).get<int>());
            }
    return std::make_pair(costs,traversals);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 4 && (argc != 5 || std::string(argv[4]) != "--compact-only"))
            throw std::invalid_argument("usage: mlnd_output_probe INSTANCE K OUTPUT_DIR [--compact-only]");
        const auto begun=std::chrono::steady_clock::now();
        const auto input = mlnd::read_input(argv[1], std::stoi(argv[2]));
        const auto paths = mlnd::enumerate_paths(input);
        if (argc == 5) {
            auto fixed=input;
            mlnd::preprocess_generalized_demand(fixed,paths);
            const auto start=std::chrono::steady_clock::now();
            auto compact=mlnd::all_courier_result(fixed,paths);
            mlnd::validate_result_path_records(paths,compact);
            const double package_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            compact["max_orders"]=fixed.max_orders;
            compact["catalogue_complete"]=paths.complete;
            compact["catalogue"]={{"service",paths.service_paths.size()},{"empty",paths.empty_paths.size()},
                                  {"segments",paths.initial_segments.size()}};
            const auto write_start=std::chrono::steady_clock::now();
            mlnd::write_result(argv[3],fixed,paths,compact);
            std::cout << mlnd::Json{{"passed",true},{"solver_invoked",false},
                {"service_paths",paths.service_paths.size()},{"saved_service_records",compact.at("_missions").size()},
                {"objective",compact.at("objective")},{"package_and_record_check_seconds",package_seconds},
                {"write_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-write_start).count()},
                {"total_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count()}}.dump()<<'\n';
            return 0;
        }
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
        mlnd::Json compact_checks=nullptr;
        if (input.flow_policy == "aggregate") {
            auto fixed=input;
            mlnd::preprocess_generalized_demand(fixed,paths);
            auto compact=mlnd::all_courier_result(fixed,paths);
            auto dense_input=fixed;
            dense_input.first_leg_consistency=true; // exercise the legacy record fallback only
            const auto dense=mlnd::all_courier_result(dense_input,paths);
            if (compact.at("_solution")!=dense.at("_solution") ||
                compact.at("metrics")!=dense.at("metrics") || statistics(compact)!=statistics(dense))
                throw std::runtime_error("Compact records changed decisions, resources or reconstruction statistics");
            mlnd::validate_result_path_records(paths,compact);
            mlnd::validate_result_path_records(paths,dense);
            auto marked_compact=compact;
            std::vector<bool> recorded(paths.service_paths.size(),false);
            for (const auto& row:compact.at("_missions"))
                recorded[row.at("path").at("id").get<int>()]=true;
            int marked_id=static_cast<int>(paths.service_paths.size()-1);
            for (std::size_t index=0;index<recorded.size();++index)
                if (!recorded[index]) { marked_id=static_cast<int>(index);break; }
            marked_compact["_solution"]["flow"][marked_id]=1e-250;
            mlnd::synchronize_result_paths(fixed,paths,marked_compact);
            auto find_marked=[&]() -> const mlnd::Json& {
                for (const auto& row:marked_compact.at("_missions"))
                    if (row.at("path").at("id")==marked_id) return row;
                throw std::runtime_error("Tiny nonzero flow lost its physical record");
            };
            if (find_marked().at("flow")!=1e-250)
                throw std::runtime_error("Tiny positive flow was rounded away");
            marked_compact["_solution"]["flow"][marked_id]=-1e-250;
            mlnd::synchronize_result_paths(fixed,paths,marked_compact);
            if (find_marked().at("flow")!=-1e-250)
                throw std::runtime_error("Tiny negative flow was rounded away");
            mlnd::validate_result_path_records(paths,marked_compact);
            const auto stable=marked_compact;
            mlnd::synchronize_result_paths(fixed,paths,marked_compact);
            if (stable!=marked_compact) throw std::runtime_error("Compact synchronization is not idempotent");
            int rejected=0;
            auto reject=[&](mlnd::Json malformed) {
                try { mlnd::validate_result_path_records(paths,malformed); }
                catch (const std::exception&) { ++rejected; return; }
                throw std::runtime_error("Corrupt compact record was accepted");
            };
            auto corrupt=marked_compact;
            corrupt["_missions"].push_back(corrupt["_missions"].front()); reject(corrupt);
            corrupt=marked_compact;corrupt["_missions"][0]["path"]["id"]=-1;reject(corrupt);
            corrupt=marked_compact;corrupt["_missions"][0]["path"]["id"]=paths.service_paths.size();reject(corrupt);
            corrupt=marked_compact;corrupt["_missions"][0]["flow"]=99;reject(corrupt);
            corrupt=marked_compact;corrupt["_missions"][0]["path"]["cost"]=99;reject(corrupt);
            corrupt=marked_compact;
            auto& records=corrupt["_missions"];
            for (auto it=records.begin();it!=records.end();++it)
                if (it->at("path").at("id")==marked_id) { records.erase(it);break; }
            reject(corrupt);
            auto compact_pool=mlnd::prepare_path_support(fixed);
            mlnd::register_service_path(compact_pool,paths.service_paths.front());
            auto growing=mlnd::all_courier_result(fixed,compact_pool);
            if (paths.service_paths.size()>1) mlnd::register_service_path(compact_pool,paths.service_paths[1]);
            mlnd::synchronize_result_paths(fixed,compact_pool,growing);
            if (compact_pool.output_witness_service_size!=compact_pool.service_paths.size() ||
                growing.at("objective")!=compact.at("objective"))
                throw std::runtime_error("Compact witness cache failed after pool growth");
            mlnd::validate_result_path_records(compact_pool,growing);
            compact["max_orders"]=fixed.max_orders;
            mlnd::write_result(std::filesystem::path(argv[3])/"compact",fixed,paths,compact);
            compact_checks={{"saved_service_records",compact.at("_missions").size()},
                {"registered_service_paths",paths.service_paths.size()},
                {"task_minima_origin_support_and_arc_maxima_preserved",true},
                {"tiny_positive_and_negative_flows_preserved",true},{"corrupt_records_rejected",rejected},
                {"tiny_flow_exercises_omitted_zero_record",!recorded[marked_id]},
                {"legacy_dense_records_accepted",true},{"cache_pool_growth",true}};
        }
        std::cout << mlnd::Json{{"passed", true}, {"max_orders", input.max_orders},
            {"objective", baseline.at("objective")}, {"files_written", true},
            {"earlier_incumbent_pool_growth", true}, {"nonzero_flow_preserved", true},
            {"registration_and_synchronization_idempotent", true},{"compact_checks",compact_checks}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
