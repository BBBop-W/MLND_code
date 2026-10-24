#include "mlnd/progress.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

// Pure observation/serialization test: no CPLEX objects or solver calls.
int main(int argc,char** argv) {
    using namespace mlnd;
    try {
        if(argc!=2) throw std::invalid_argument("Usage: mlnd_progress_probe NEW_DIRECTORY");
        const std::filesystem::path root=argv[1];
        if(std::filesystem::exists(root)) throw std::invalid_argument("Use a new test directory");
        const double now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto observer=std::make_shared<RunProgress>(root,now,10,Json{{"max_orders",2},
            {"input_manifest_sha256","mock-only"},{"problem_scope","global"}});
        int checks=0;
        auto require=[&](bool condition,const char* message) { ++checks; if(!condition) throw std::runtime_error(message); };
        auto read=[&](const char* filename) { Json value;std::ifstream(root/filename)>>value;return value; };
        observer->record({{"event","global_bounds"},{"stage","mock"},{"bound_scope","global"},{"bound",2}});
        Json point={{"objective",10},{"bound",1},{"_solution",Json::object()},{"_segments",Json::array()},
            {"_missions",Json::array()},{"_relocations",Json::array()}};
        observer->checkpoint(point,"mock_validated_point");
        auto stored=read("checkpoint.json");
        require(stored.at("bound")==2,"Global bound was lost");
        require(stored.at("max_orders")==2,"Checkpoint K metadata was lost");
        require(stored.at("validated_incumbent")==true,"Missing feasibility marker");
        require(stored.contains("timestamp_utc")&&stored.contains("elapsed_seconds"),"Missing time provenance");
        point["objective"]=11;observer->checkpoint(point,"worse",true);
        require(read("checkpoint.json").at("objective")==10,"Worse point overwrote checkpoint");
        point["objective"]=9;observer->checkpoint(point,"better");
        require(read("checkpoint.json").at("objective")==9,"Improvement not checkpointed");
        observer->record({{"event","global_bounds"},{"bound_scope","global"},{"bound",4}});
        observer->checkpoint(point,"bound_update",true);
        require(read("checkpoint.json").at("bound")==4,"Bound update not checkpointed");
        observer->record({{"event","solver_progress"},{"bound_scope","solver_model"},{"bound",100}});
        observer->checkpoint(point,"local_bound_ignored",true);
        require(read("checkpoint.json").at("bound")==4,"A local model bound became a global bound");
        observer->status("finished",{{"solved",false}});
        require(read("run_status.json").at("state")=="finished","Missing terminal status");
        require(!std::filesystem::exists(root/"checkpoint.json.tmp"),"Atomic checkpoint left temporary file");
        auto expired=std::make_shared<RunProgress>(root/"expired",now-10,1,Json::object());
        set_active_run_progress(expired);
        bool rejected=false;
        try { model_construction_guard("mock_build",7,11,2,true); }
        catch(const RunBudgetExceeded& error) {
            rejected=error.stage=="mock_build"&&error.sizes.at("model_size").at("variables")==7;
        }
        set_active_run_progress(nullptr);
        require(rejected,"Expired construction was not stopped with counts");
        std::ifstream stream(root/"progress.jsonl");std::string line;std::size_t expected=0;
        while(std::getline(stream,line)) {
            auto event=Json::parse(line);
            require(event.at("sequence")==++expected,"Progress sequence is not monotone");
        }
        std::cout<<Json{{"checks",checks},{"solver_calls",0},{"passed",true},{"directory",root.string()}}.dump()<<'\n';
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
    return 0;
}
