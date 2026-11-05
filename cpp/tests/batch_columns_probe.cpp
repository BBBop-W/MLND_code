#include "mlnd/formulation.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Json coefficient_case() {
    Input d;d.n_l=2;d.n_h=1;d.n_od=2;d.n_levels=1;Formulation f(d);
    f.single_rows={10};f.double_rows={11,12};f.executed_head_rows={13};
    f.order_rows={20,21};f.capacity_rows={30};f.balance_rows={40,41};f.fleet_row=50;
    f.link_rows={60,61,62,63,64,65,66,67,68};
    ServicePath p;p.segment_id=0;p.second_od=1;p.b={0,1};p.time=.75;
    p.served_counts={{0,1},{1,1}};p.slot_times={{0,.25}};p.arc_counts={{0,2,1},{2,1,1}};
    const Coefficients expected={{12,-1},{13,-1},{20,1},{21,1},{30,.25},{40,1},{41,-1},{50,.75},{62,1},{67,1}};
    require(path_column(f,p)==expected,"Canonical service coefficients differ from hand calculation");
    Coefficients observed;for_each_path_coefficient(f,p,[&](int r,double v){observed.emplace_back(r,v);});
    require(observed==expected,"Streaming coefficients differ from hand calculation");
    EmptyPath e;e.b={0,1};e.time=.5;e.arc_counts={{0,1,1}};
    require(path_column(f,e)==Coefficients{{40,1},{41,-1},{50,.5},{61,1}},"Empty flow hand coefficients differ");
    return {{"passed",true},{"service_terms",expected.size()}};
}
Json linear_case() {
    LinearModel matrix;matrix.variable("dummy",0,0);matrix.row("demand",'G',5);matrix.row("x_limit",'L',4);
    SolverOptions options;options.seconds=10;NativeSolver solver(matrix,options);
    std::vector<VariableRecord> vars={{"x",0,10,2,false},{"y",0,10,1,false}};
    std::vector<Coefficients> cols={{{0,1},{1,1}},{{0,1}}};
    solver.add_variables(vars,cols);
    for(std::size_t j=0;j<vars.size();++j) {
        matrix.variables.push_back(vars[j]);
        for(auto [row,value]:cols[j])matrix.coefficient(row,static_cast<int>(j)+1,value);
    }
    const auto solved=solver.optimize(10);
    require(solved.optimal&&solved.has_solution,"Batch LP did not solve");
    require(std::abs(solved.objective-5)<1e-8&&point_residuals(matrix,solved.values).at("passed"),"Batch LP changed objective or rows");
    const auto dual=solver.read_dual(matrix);require(dual.available,"Batch LP lost native dual access");
    const auto old_size=solver.variables.getSize();bool rejected=false;
    try{solver.add_variables(vars,{{{0,1}},{{99,1}}});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&solver.variables.getSize()==old_size,"Invalid batch partially changed native model");
    std::cout.flush();
    return {{"passed",true},{"objective",solved.objective},{"native_dual",dual.available}};
}
Json insertion_benchmark(bool batch) {
    LinearModel matrix;
    for(int r=0;r<600;++r)matrix.row("row_"+std::to_string(r),'L',100000);
    for(int j=0;j<10000;++j) {
        const int index=matrix.variable("existing_"+std::to_string(j),0,1,1);
        for(int k=0;k<8;++k)matrix.coefficient((j*13+k*37)%600,index,1+.1*k);
    }
    SolverOptions options;NativeSolver solver(matrix,options);
    const double start=wall_time();
    for(int group=0;group<10;++group) {
        std::vector<VariableRecord> vars;std::vector<Coefficients> cols;
        for(int j=0;j<512;++j) {
            const int id=group*512+j;vars.push_back({"appended_"+std::to_string(id),0,1,1,false});
            Coefficients col;for(int k=0;k<8;++k)col.emplace_back((id*13+k*37)%600,1+.1*k);
            cols.push_back(std::move(col));
        }
        if(batch)solver.add_variables(vars,cols);
        else for(std::size_t j=0;j<vars.size();++j)solver.add_variable(vars[j],cols[j]);
    }
    const double elapsed=wall_time()-start;
    require(solver.variables.getSize()==15120,"Benchmark insertion count differs");
    return {{"batch",batch},{"columns_added",5120},{"existing_columns",10000},{"seconds",elapsed}};
}
}
int main(){try {
    const auto coefficients=coefficient_case();
    const auto check=linear_case();
    const auto legacy=insertion_benchmark(false),batch=insertion_benchmark(true);
    std::cout<<Json({{"passed",true},{"coefficients",coefficients},{"linear",check},{"legacy",legacy},{"batch",batch},
        {"observed_speedup",legacy.at("seconds").get<double>()/batch.at("seconds").get<double>()}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
