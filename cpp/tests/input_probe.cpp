#include "mlnd/input.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::runtime_error("Usage: mlnd_input_probe INSTANCE [1|2]");
        const auto d = mlnd::read_input(argv[1],argc == 3 ? std::stoi(argv[2]) : 0);
        mlnd::Json arrays;
        arrays["demand"] = d.demand;
        arrays["ground_cost"] = d.ground_cost;
        arrays["deadline"] = d.deadline;
        arrays["base_cost"] = d.base_cost;
        arrays["slot_epsilon"] = d.slot_epsilon;
        arrays["od_origin"] = d.od_origin;
        arrays["od_destination"] = d.od_destination;
#define MLND_MATRIX(name) arrays[#name] = d.name.values
        MLND_MATRIX(flight_times); MLND_MATRIX(rider_mean); MLND_MATRIX(first_cost);
        MLND_MATRIX(terminal_cost); MLND_MATRIX(receipt_mean); MLND_MATRIX(removal_mean);
        MLND_MATRIX(slot_cost); MLND_MATRIX(slot_counts); MLND_MATRIX(link_cost);
        MLND_MATRIX(kappa); MLND_MATRIX(allowed_links); MLND_MATRIX(bases);
        MLND_MATRIX(hubs); MLND_MATRIX(origins); MLND_MATRIX(destinations);
        MLND_MATRIX(pickup_share); MLND_MATRIX(last_rider_cost); MLND_MATRIX(last_rider_mean);
#undef MLND_MATRIX
        std::cout << mlnd::Json{{"arrays",arrays},{"model",d.model_json},{"max_orders",d.max_orders},
            {"effective_delta",d.effective_delta()},{"occupancy_limit",d.occupancy_limit.values}}.dump() << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
