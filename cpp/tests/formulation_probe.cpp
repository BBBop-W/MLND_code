#include "mlnd/formulation.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("Usage: mlnd_formulation_probe INSTANCE K full|zero|pattern OUTPUT_JSON");
        const auto d = mlnd::read_input(argv[1],std::stoi(argv[2]));
        const auto paths = mlnd::enumerate_paths(d);
        const std::string mode = argv[3];
        std::vector<double> fixed;
        if (mode != "full") {
            if (mode != "zero" && mode != "pattern") throw std::runtime_error("Unknown design mode");
            fixed.resize(mlnd::DesignLayout(d).count);
            if (mode == "pattern")
                for (std::size_t i = 0; i < fixed.size(); ++i) fixed[i] = i % 3 == 0;
        }
        const auto f = mlnd::build_formulation(d,paths,mode == "full" ? nullptr : &fixed);
        mlnd::Json vars = mlnd::Json::array(), rows = vars, cones = vars;
        for (const auto& v : f.matrix.variables)
            vars.push_back({v.name,v.lower,v.upper,v.cost,v.binary});
        for (const auto& r : f.matrix.rows)
            rows.push_back({r.name,std::string(1,r.sense),r.rhs,r.coefficients});
        for (const auto& c : f.matrix.cones)
            cones.push_back({c.up,c.down,c.one,c.factor});
        std::ofstream stream(argv[4]);
        if (!stream) throw std::runtime_error("Cannot open output file");
        stream << mlnd::Json{{"variables",vars},{"rows",rows},{"cones",cones},
            {"minimize",f.matrix.minimize},{"constant",f.matrix.constant},
            {"fixed",fixed}}.dump() << '\n';
        std::cout << "variables=" << vars.size() << " rows=" << rows.size() << " cones=" << cones.size() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
