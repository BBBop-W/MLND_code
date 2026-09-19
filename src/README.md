# MLND code

The production implementation is the path-based model with an integer design
master and continuous closed-queueing-network operations. Facilities, cabinet
levels, OD/base-pair assignments and directed route openings are binary.
Availability, fleet size and service/repositioning flows are continuous. Each
service route connects two bases through a pickup and delivery locker, with an
optional battery swap on each of its three stages. Rider pickup triggers drone
dispatch; first-mile rider travel is exponential by default. Destination-locker
choice probabilities and analytic expected synchronization times are computed
before optimization.

## Layout

| Location | Responsibility |
|---|---|
| `main.py`, `src/cli.py` | Command-line interface |
| `configs/default.json`, `src/config.py` | Baseline parameters and validated overrides |
| `src/data/` | Raw orders, canonical instances, travel-time calibration, utility |
| `src/model/` | Network, capacity levels and strategic design representation |
| `src/pricing/` | Flight legs, battery feasibility, route pricing and enumeration |
| `src/optimization/` | Master, recourse, cuts, primal recovery and solution methods |
| `src/operations/` | Independent reconstruction of flows, time, energy, fleet and cost |
| `src/experiments/` | Run budgets, progress, checkpoints, inputs and result persistence |
| `src/reporting/` | Chinese/English reports, operating figures and geographic maps |
| `tests/current/` | Production regression and integration tests |
| `data/model_inputs/` | Versioned processed calibration inputs |
| `data/instances/` | Generated geographic instances; excluded from Git |
| `data/map_cache/` | Optional offline map tiles and district boundary; excluded from Git |
| `outputs/experiments/` | New run outputs; excluded from Git |

Dependencies flow from data and model into pricing and optimization. Operations
reconstructs physical routes independently of the pricing implementation.
Experiments orchestrates these components; reporting reads the saved outputs.
No production module imports an archived experiment.

## Run from the repository root

Use Python 3.11+ and an available Gurobi license. Install project dependencies
in the selected environment with `python -m pip install -e '.[test]'`.

```sh
# Inspect actual candidate counts, demand and parameters.
python main.py inspect --hubs 120

# Solve multiple geographic instances, keeping the integer design master.
python main.py run --hubs 11 24 43 68 87 120 --seconds 600 --threads 4 \
  --out outputs/experiments/baseline

# Add joint integer incumbent improvement and fixed-design CG refinement.
python main.py run --hubs 68 120 --method hybrid --seconds 1800 --threads 4 \
  --out outputs/experiments/design_search

# Unlimited Benders run, with Gurobi's default relative gap.
python main.py run --hubs 120 --seconds none --threads 4 \
  --out outputs/experiments/long_run

python main.py status outputs/experiments/long_run/H120
python main.py stop outputs/experiments/long_run/H120
python main.py validate outputs/experiments/baseline/H120
python main.py report outputs/experiments/baseline/H120
python main.py report outputs/experiments/baseline
python -m pytest
```

Use a new output directory for each run. The seed defaults to 43 and the
relative MIP gap to `1e-4`; `--seed` and `--gap` override them. The Benders master
contains the strategic binaries and one recourse-value variable. `hybrid`
allocates its solve budget sequentially to Benders, restricted-column integer
improvement, and full-graph fixed-design refinement. Only the complete-model
bound determines the reported gap. `direct` explicitly enumerates paths for
small validation instances; it has an enumeration-size guard.

A graceful stop saves the current incumbent. A complete checkpoint is also
saved during Benders improvements. Reports and validation require no new
optimization. Individual reports include both a facilities map and a route map,
plus the operating composition figures. `--no-maps` generates a report without
the map cache. Console summaries and `OPERATIONS.md` are available on mobile.

## Parameters and saved runs

Create a partial JSON file to change only the desired baseline parameters:

```json
{
  "name": "capacity_sensitivity",
  "operations": {"cabinet_capacity": 27.36},
  "solver": {"time_limit": 1800, "threads": 4}
}
```

Pass it with `--config configs/capacity_sensitivity.json`. Fixed investments
are converted using the common depreciation settings; units and baseline
values are documented in `configs/README.md`.

Every case saves `config.json`, `input_metadata.json`, `input_arrays.npz`,
source/calibration snapshots, `result.json`, `solution.npz`, compressed path
flows, the independent operating tables, `summary.json` and `manifest.json`.
`progress.json` and `checkpoint.json` track a running case. Existing results
are never silently overwritten. `--input CASE_DIRECTORY` reruns its exact
saved input arrays as one new case; it does not warm-start the old solution or
change its model parameters. Solver options can still be changed.

The six canonical instance sizes are H=11/24/43/68/87/120, with respectively
L=2/2/3/5/6/8 and E=2/3/5/7/9/13. Regeneration from the local raw order data is
available through `python -m src.data.scales --help`. Processed calibration
inputs are separate from generated instance data and historical results.
