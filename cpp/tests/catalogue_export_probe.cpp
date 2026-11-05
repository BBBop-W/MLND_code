#include "mlnd/output.hpp"
#include <fstream>
#include <iostream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: catalogue_export_probe OUTPUT_DIR");
        const std::filesystem::path directory(argv[1]);
        const auto target = directory / "full_catalogue.json";
        mlnd::Input input;
        input.n_l = 1; input.n_h = 2; input.n_od = 1; input.max_orders = 2;
        input.model_version = "consecutive";
        input.flow_policy = "aggregate";
        input.preprocessed_demand = true;
        mlnd::PathCollection paths;
        mlnd::InitialSegment segment;
        segment.id = 0; segment.first_od = 0; segment.pickup = 0; segment.delivery = 1;
        paths.initial_segments.push_back(segment);
        for (int index = 0; index < 2; ++index) {
            mlnd::ServicePath path;
            path.id = index; path.segment_id = 0; path.first_od = 0;
            path.mode = "single"; path.pickups = {0}; path.deliveries = {1};
            path.cost = index ? 8 : 3;
            path.arc_counts = {{0,1,index + 1}};
            paths.service_paths.push_back(path);
        }
        mlnd::EmptyPath empty;
        empty.id = 0; empty.cost = 2;
        paths.empty_paths.push_back(empty);

        const auto metadata = mlnd::write_complete_catalogue(target, input, paths);
        std::ifstream stream(target);
        mlnd::Json saved; stream >> saved;
        require(saved.at("schema") == "mlnd-complete-catalogue/1" && saved.at("complete"),
                "Missing complete catalogue identity");
        require(saved.at("counts") == mlnd::Json{{"segments",1},{"service",2},{"empty",1}},
                "Full path counts changed");
        require(saved.at("service").size() == 2 && saved.at("service").at(1).at("cost") == 8,
                "A nonminimum path was lost from the complete domain");
        require(saved.at("service").at(1).at("arc_counts").at(0).at(2) == 2,
                "Arc multiplicity was not preserved");
        require(!saved.at("service").at(0).contains("flow"), "Catalogue confused with incumbent flows");
        require(saved.at("assignment_scope") == "origin" && saved.at("max_orders") == 2,
                "Catalogue model identity changed");
        require(metadata.at("file") == "full_catalogue.json" &&
                metadata.at("sha256") == mlnd::sha256_file(target), "Export hash differs");
        require(!std::filesystem::exists(target.string() + ".tmp"), "Atomic output was not finalized");
        const auto original_hash = mlnd::sha256_file(target);
        for (bool support_only : {false, true}) {
            auto incomplete = paths;
            if (support_only) incomplete.support_complete = false;
            else incomplete.complete = false;
            bool rejected = false;
            try { mlnd::write_complete_catalogue(target, input, incomplete); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && mlnd::sha256_file(target) == original_hash,
                    "Incomplete export overwrote a complete catalogue");
        }
        std::cout << mlnd::Json{{"passed",true},{"solver_invoked",false},
            {"complete_paths",3},{"incomplete_exports_rejected",2},{"metadata",metadata}}.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
