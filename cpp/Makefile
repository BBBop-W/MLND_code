CXX ?= clang++
CPLEX_ROOT ?= $(HOME)/Applications/CPLEX_Studio222
NLOHMANN_INCLUDE ?= /opt/anaconda3/include
BUILD_DIR ?= build
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    CPLEX_PLATFORM ?= arm64_osx
  else
    CPLEX_PLATFORM ?= x86-64_osx
  endif
  PLATFORM_LIBS = -framework CoreFoundation -framework IOKit -framework Accelerate
else
  CPLEX_PLATFORM ?= x86-64_linux
  PLATFORM_LIBS = -ldl -lcrypto
endif
CPPFLAGS += -Iinclude -isystem $(NLOHMANN_INCLUDE) -isystem $(CPLEX_ROOT)/cplex/include -isystem $(CPLEX_ROOT)/concert/include -DIL_STD
CXXFLAGS ?= -O3 -DNDEBUG
CXXFLAGS += -std=c++17 -fPIC -fexceptions -ffp-contract=off -Wall -Wextra -Wpedantic
LDFLAGS += -L$(CPLEX_ROOT)/cplex/lib/$(CPLEX_PLATFORM)/static_pic -L$(CPLEX_ROOT)/concert/lib/$(CPLEX_PLATFORM)/static_pic
LDLIBS += -lilocplex -lconcert -lcplex -lm -lpthread $(PLATFORM_LIBS)
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)
CORE_OBJECTS := $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))

.PHONY: all clean input-probe path-catalogue formulation-probe output-probe benders-probe fixed-design-probe pricing-probe strengthening-probe branch-seed-probe heuristic-probe progress-probe cut-pool-probe root-diagnostic-probe continuous-probe continuous-matrix-probe bdd-probe
all: $(BUILD_DIR)/mlnd_cpp
input-probe: $(BUILD_DIR)/mlnd_input_probe
path-catalogue: $(BUILD_DIR)/mlnd_path_catalogue
formulation-probe: $(BUILD_DIR)/mlnd_formulation_probe
output-probe: $(BUILD_DIR)/mlnd_output_probe
benders-probe: $(BUILD_DIR)/mlnd_benders_probe
fixed-design-probe: $(BUILD_DIR)/mlnd_fixed_design_probe
pricing-probe: $(BUILD_DIR)/mlnd_pricing_probe
strengthening-probe: $(BUILD_DIR)/mlnd_strengthening_probe
branch-seed-probe: $(BUILD_DIR)/mlnd_branch_seed_probe
heuristic-probe: $(BUILD_DIR)/mlnd_heuristics_probe
progress-probe: $(BUILD_DIR)/mlnd_progress_probe
cut-pool-probe: $(BUILD_DIR)/mlnd_cut_pool_probe
root-diagnostic-probe: $(BUILD_DIR)/mlnd_root_diagnostic_probe
continuous-probe: $(BUILD_DIR)/mlnd_continuous_probe
continuous-matrix-probe: $(BUILD_DIR)/mlnd_continuous_matrix_probe
bdd-probe: $(BUILD_DIR)/mlnd_bdd_probe
bdd-search-probe: $(BUILD_DIR)/mlnd_bdd_search_probe
.PHONY: bdd-search-probe
dual-workspace-probe: $(BUILD_DIR)/mlnd_dual_workspace_probe
conditional-cut-probe: $(BUILD_DIR)/mlnd_conditional_cut_probe
conditional-formulation-probe: $(BUILD_DIR)/mlnd_conditional_formulation_probe
conditional-benders-probe: $(BUILD_DIR)/mlnd_conditional_benders_probe
.PHONY: conditional-cut-probe conditional-formulation-probe conditional-benders-probe tight-airline-probe topology-cuts-probe
tight-airline-probe: $(BUILD_DIR)/mlnd_tight_airline_probe
topology-cuts-probe: $(BUILD_DIR)/mlnd_topology_cuts_probe
.PHONY: assignment-support-cuts-probe assignment-support-validity-probe
assignment-support-cuts-probe: $(BUILD_DIR)/mlnd_assignment_support_cuts_probe
assignment-support-validity-probe: $(BUILD_DIR)/mlnd_assignment_support_validity_probe
.PHONY: branch-joint-cuts-probe branch-joint-validity-probe
branch-joint-cuts-probe: $(BUILD_DIR)/mlnd_branch_joint_cuts_probe
branch-joint-validity-probe: $(BUILD_DIR)/mlnd_branch_joint_validity_probe
.PHONY: od-airline-probe od-airline-root-probe
od-airline-probe: $(BUILD_DIR)/mlnd_od_airline_probe
od-airline-root-probe: $(BUILD_DIR)/mlnd_od_airline_root_probe
.PHONY: structural-strengthening-probe structural-cg-cut-probe
structural-strengthening-probe: $(BUILD_DIR)/mlnd_structural_strengthening_probe
structural-cg-cut-probe: $(BUILD_DIR)/mlnd_structural_cg_cut_probe

$(BUILD_DIR)/mlnd_%_probe: tests/%_probe.cpp $(CORE_OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/mlnd_input_probe: tests/input_probe.cpp $(BUILD_DIR)/input.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(if $(filter Darwin,$(UNAME_S)),,-lcrypto) -o $@

$(BUILD_DIR)/mlnd_progress_probe: tests/progress_probe.cpp $(BUILD_DIR)/progress.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/mlnd_path_catalogue: tests/path_catalogue.cpp $(BUILD_DIR)/input.o $(BUILD_DIR)/paths.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(if $(filter Darwin,$(UNAME_S)),,-lcrypto) -o $@

$(BUILD_DIR)/mlnd_cpp: $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
