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
# BDD is a retained research diagnostic, not part of the production executable.
DIAGNOSTIC_SOURCES := src/bdd.cpp src/bdd_search.cpp
SOURCES := $(filter-out $(DIAGNOSTIC_SOURCES),$(wildcard src/*.cpp))
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DIAGNOSTIC_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(DIAGNOSTIC_SOURCES))
DEPS := $(OBJECTS:.o=.d) $(DIAGNOSTIC_OBJECTS:.o=.d)
CORE_OBJECTS := $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))

# Every public probe gets a hyphenated target from its source filename.
# Keep the historical singular alias and the standalone catalogue target.
PROBE_STEMS := $(patsubst tests/%_probe.cpp,%,$(wildcard tests/*_probe.cpp))
PROBE_TARGETS := $(foreach probe,$(PROBE_STEMS),$(subst _,-,$(probe))-probe)
.PHONY: all clean path-catalogue heuristic-probe $(PROBE_TARGETS)
all: $(BUILD_DIR)/mlnd_cpp
path-catalogue: $(BUILD_DIR)/mlnd_path_catalogue
heuristic-probe: heuristics-probe

define PROBE_ALIAS
$(subst _,-,$(1))-probe: $(BUILD_DIR)/mlnd_$(1)_probe
endef
$(foreach probe,$(PROBE_STEMS),$(eval $(call PROBE_ALIAS,$(probe))))

BDD_PROBES := $(addprefix $(BUILD_DIR)/mlnd_,$(addsuffix _probe,$(filter bdd bdd_%,$(PROBE_STEMS))))
$(BDD_PROBES): $(DIAGNOSTIC_OBJECTS)

# The private certificate probe includes benders.cpp in its translation unit.
$(BUILD_DIR)/mlnd_dual_elimination_probe: tests/dual_elimination_probe.cpp src/benders.cpp $(filter-out $(BUILD_DIR)/benders.o,$(CORE_OBJECTS))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(filter %.o,$^) $(LDLIBS) -o $@

$(BUILD_DIR)/mlnd_pricing_state_probe: tests/pricing_state_probe.cpp src/benders.cpp $(filter-out $(BUILD_DIR)/benders.o,$(CORE_OBJECTS))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(filter %.o,$^) $(LDLIBS) -o $@

$(BUILD_DIR)/mlnd_performance_state_probe: tests/performance_state_probe.cpp src/benders.cpp $(filter-out $(BUILD_DIR)/benders.o,$(CORE_OBJECTS))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(filter %.o,$^) $(LDLIBS) -o $@

$(BUILD_DIR)/mlnd_primal_recovery_probe: tests/primal_recovery_probe.cpp tests/performance_state_probe.cpp src/benders.cpp $(filter-out $(BUILD_DIR)/benders.o,$(CORE_OBJECTS))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(filter %.o,$^) $(LDLIBS) -o $@

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
