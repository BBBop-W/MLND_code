# MLND

The current model has an integer strategic design and continuous CQN operations.
Facilities, cabinet levels, OD/base-pair assignments and directed route openings
are binary. Availability, fleet size and service/repositioning flows are continuous.
All five solution methods use the same input data, costs and physical route rules.

## Code structure

```text
src/
  config.py                 Parameters and validation
  cli.py                    Command-line entry
  data/                     Orders, instances, calibration and utility
  model/                    Shared network, design and route-state graph
  solvers/
    api.py                  Common solve interface and result format
    methods.py              Five method profiles and acceleration switches
    settings.py             Shared solver precision
    result.py               Common solution packing and validation
    benders/
      driver.py             Benders and optional incumbent improvement stages
      engine.py             Integer master / continuous subproblem loop
      master.py             Integer master and initial lower bound
      subproblem.py         Continuous conic recourse and column generation
      cuts.py               Benders cut construction and certification
      primal.py             Primal solution recovery
      initialization.py     Initial designs and incumbent cleanup
      state.py              Bounds, iteration records and checkpoints
    pricing/
      backend.py            Select label or solver pricing
      label.py              Label pricing
      solver.py             Gurobi shortest-path pricing on the state graph
      enumeration.py        Complete feasible-path enumeration
      inputs.py             Pricing data validation
      legs.py               Feasible flight legs and return envelopes
      kernels.py            Label search kernel
      routes.py             Physical route reconstruction and coefficients
    direct/
      arc.py                State-expanded arc-flow MISOCP
      path.py               Full-path direct solver
      path_model.py         Full-path MISOCP construction
    heuristic/
      search.py             Restricted-column integer incumbent search
      pool.py               Candidate columns and seed repricing
      model.py              Restricted-column MISOCP construction
      refinement.py         Fixed-design full-graph CG refinement
  experiments/              Plans, execution, progress and persistence
  operations/               Independent operating metrics and validation
  reporting/                Reports, figures and geographic maps
```

`solvers` contains algorithms; `experiments` only organizes their execution and
saved artifacts. Algorithm selection occurs once, in `solvers/api.py`. The
Benders variants share one engine and one conic subproblem. Pricing and
acceleration choices do not require separate copies of these models.

## Five comparison methods

| Method ID | Formulation | Pricing | Custom accelerations |
|---|---|---|---|
| `benders_cg` | Integer-master Benders-CG | Label | Enabled |
| `arc` | Direct state-expanded arc-flow MISOCP | Not used | Not used |
| `full_path` | Direct full-path MISOCP | Complete enumeration | Not used |
| `benders_cg_solver` | Integer-master Benders-CG | Gurobi | Same as `benders_cg` |
| `benders_cg_plain` | Integer-master Benders-CG | Label | Disabled |

Both CG variants solve their continuous conic subproblem with Gurobi. The
label/solver option selects the **pricing problem** implementation. Solver
pricing uses a shortest-path LP on an acyclic state graph. The arc formulation
uses flows on that graph rather than first enumerating complete paths; states
retain stage, timing and residual energy. Numerical equivalence and performance
of the newly connected methods remain to be checked in the next experiment phase.

Acceleration switches are `initial_bound`, `initial_design`, `reuse_columns`,
`warm_start`, `incumbent_cleanup`, and `primal_search`. The plain profile turns
all six off. Necessary Benders optimality cuts, CG initialization and complete
candidate-graph pricing for cut certification remain active. There are currently
no separately implemented optional master valid inequalities; method metadata
records this as an empty list. Gurobi's own cuts and presolve keep their defaults.

For a finite time budget, `primal_search` allocates 65% to Benders and uses the
remaining time for restricted-column integer search and fixed-design CG
refinement, when needed. These stages update the incumbent and preserve the
complete-model Benders lower bound. An unlimited run stays in the Benders loop.
The plain method gives the entire budget to its Benders loop.

## Inspect and plan without solving

```sh
python main.py methods
python main.py plan --hubs 11 120 --methods all --seconds 600 --threads 4
```

`methods` and `plan` do not create a Gurobi model, generate instances, or create
experiment directories. `inspect --hubs 120` reads/builds an instance to display
candidate counts, demand and model parameters without optimization.

## Interfaces for later experiments

```python
from src.solvers import solve

result = solve(data, method="benders_cg", time_limit=600, threads=4)
# Future ablation using the same engine:
result = solve(data, method="benders_cg", time_limit=600,
               accelerations={"initial_bound": False, "primal_search": False})
```

The five methods accept the same time/thread/seed/gap and stop/progress options.
Results record the selected method and its acceleration settings. Complete-path
and state-graph construction have explicit size guards (`column_limit` and
`graph_limit`); a guard stops construction instead of truncating the model.
Set a guard to `null` in the configuration to remove it.

Commands below **start optimization** and are provided for the later experiment phase:

```sh
python main.py run --hubs 11 120 --methods all --seconds 600 --threads 4 \
  --out outputs/experiments/comparison
python main.py run --hubs 120 --methods benders_cg --seconds none --threads 4 \
  --out outputs/experiments/long_run
```

Each case is stored as `H120_benders_cg`, `H120_arc`, etc. Every method is an
independent run with the same chosen input configuration. Existing outputs are
never silently overwritten. `--input CASE_DIRECTORY` replays the saved input
arrays for a single selected method. Reports and validation do not optimize:

```sh
python main.py status outputs/experiments/comparison/H120_benders_cg
python main.py stop outputs/experiments/comparison/H120_benders_cg
python main.py validate outputs/experiments/comparison/H120_benders_cg
python main.py report outputs/experiments/comparison/H120_benders_cg
```

Each completed case saves input arrays, resolved configuration, source and
calibration snapshots, solver results, path flows and independently reconstructed
operating metrics. Incomplete runs keep available checkpoints and termination
reasons. Console summaries and `OPERATIONS.md` support mobile reading.

## Model scope and data

The main design includes every OD and ordered base pair, including identical
start/end bases. Its continuous recourse splits flow across feasible H-H routes.
A service route is L-H-H-L; each of its three legs is direct or visits one swap
station. An empty repositioning route is direct or visits one swap station.
The state-graph and enumerated formulations use these same route rules.

Only the optional restricted-column incumbent search adds up to two promising
base pairs per OD. This limits the heuristic search; the main Benders model and
its final full-graph pricing retain the complete candidate set.

Canonical instances have H=11/24/43/68/87/120, respectively L=2/2/3/5/6/8 and
E=2/3/5/7/9/13. Versioned calibration is in `data/model_inputs/`; generated
instances, map caches, historical `mlnd_path/` experiments and new `outputs/`
are separate from production code. No production module imports historical
experiment scripts. Manuscript and literature directories are independent.

Use Python 3.11+ and a Gurobi license. Dependencies are in `pyproject.toml`.
Regression definitions are in `tests/current/`; optimization tests are deferred
until experiments are requested. The present restructuring is checked through
syntax, imports, method configuration and no-solve planning only.

## Fixed demand for candidate-count experiments

Default `data.demand_hubs=120` decouples the OD market from `--hubs`: all six
candidate sizes retain the same 11,427 OD pairs, 8,196.7143 hourly orders, rider
baseline costs, and delivery benchmarks. `data/demand.py` fingerprints these
inputs and checks fixed-demand batches before optimization. `audit-demand`
performs this check without solving. Candidate-to-OD travel and utility arrays
are recomputed for the chosen facilities.

Use another `demand_hubs` value to select a different fixed market, or explicitly
set it to `null` (`configs/geographic.json`) for the original varying-area
experiments. See `configs/README.md` for examples and candidate-location scope.
