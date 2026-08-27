# Unified OD-aware drone network design with aggregate energy sufficiency

This repository implements the formulation in `MLND/main.tex` under the
machine-readable contract

```text
formulation_id = mlnd-unified-chi-epsilon-energy-v3
summary schema_version = 5
```

The Overleaf-synchronized `MLND/` directory is an input specification. The
Python pipeline does not modify it.

## Model contract

Customer demand remains external to the facility candidates. For every
nonzero market `(o,d)`, `h_od` is the exogenous order rate. The optimization
then uses one unified offered-flow family:

- `chi[o,d,i,j,r]` is market `(o,d)` flow offered to loaded locker corridor
  `(i,j)` before service realization at availability level `r`;
- `w[ell,i,j,r]` allocates the same offered pool to a physical or virtual idle
  state `ell`;
- for every `(i,j,r)`, `sum_od chi = sum_ell w` preserves OD and reliability
  composition before realization;
- `psi` is realized physical or logical drone flow, generated forward by
  multiplying the appropriate `w` or `chi` marginal by the fixed level
  `rho_r`.

There is no endogenous `h_ij` variable and there are no separate
locker/home `a` or `q` flow families.

The customer sees only the destination locker `j`. The backend origin locker
`i` and availability level `r` do not enter customer utility. The conditional
MNL share `eta_odj` is therefore precomputed and applies uniformly across all
backend origins `i` and reliability levels `r` that end at `j`.

In the mixed-delivery formulation, `alpha[d,j]` selects the displayed nearest
open locker and `epsilon[o,d,j]` is the successful market flow that ends at
that displayed locker. Actual locker pickup is `eta_odj * epsilon_odj`.
`epsilon` is a product-linearization auxiliary; it is not offered demand and
has not already been multiplied by `eta`.

The three service scopes share the same physical network:

| Mode | Realized drone service | Locker pickup | Assisted home | Direct courier fallback |
|---|---|---|---|---|
| `MX` | `sum rho*chi` | `sum eta*epsilon` | realized minus locker | demand minus realized |
| `HD` | `sum rho*chi` | `0` | realized | demand minus realized |
| `LO` | `sum rho*chi` | realized | `0` | demand minus realized |

For every feasible reported solution,

```text
locker_pickup + assisted_home + direct_courier_fallback = total_demand.
```

The objective is modeled platform fulfillment cost in CNY/hour. It contains
facility, smart-locker capacity, physical-connection, route-operation, fleet,
and courier-fulfillment costs. It has no revenue or profit term.

Every physical movement is decomposed into a direct option and all feasible
one-swap options. Direct and swap options are retained simultaneously when both
are feasible. The system-wide energy balance counts one fully charged battery
for every physical-base departure and one additional fully charged battery for
every one-swap route. Travel, smart-locker handling, and modeled swap-service
time consume energy; logical idle-state waiting does not. Per-leg range
screening remains exact and separate from this aggregate balance.

## Data and scenario inputs

The formal Nanshan study uses a 20 km by 20 km full-parent window with
`H/L/E = 120/8/13` candidate facilities. Demand is taken from 12:00--13:00
and averaged over the same seven source days; OD cells are 0.5 km and
candidate-site cells are 0.25 km. The number of facilities actually opened is
an optimization result. Opening roughly 50--60 smart lockers is a calibration
target to be checked, not a completed result or an imposed constraint.

Computational scale is evaluated with six concentric square crops. Their side
lengths and candidate targets are:

| Side length | H/L/E candidates |
|---:|---:|
| 6 km | 11 / 2 / 2 |
| 9 km | 24 / 2 / 3 |
| 12 km | 43 / 3 / 5 |
| 15 km | 68 / 5 / 7 |
| 17 km | 87 / 6 / 9 |
| 20 km | 120 / 8 / 13 |

Every level keeps a raw order only when both pickup and dropoff lie inside the
same centered square. Orders are not sampled, repeated, or multiplied, and all
solves use `demand_scale=1`. Each crop regenerates OD aggregation, fits
deterministic weighted K-means clusters from all retained raw orders, and maps
each centroid to the nearest observed active-grid point. Consequently, OD
hashes and total demand legitimately change with geographic coverage.
The solve entry point enforces this metadata-declared unit demand scale for the
nested family while ordinary instances remain eligible for explicit scaling.

TSL preparation retains only rows with `is_courier_grabbed=1`. Within one fixed
map, changing candidate-site counts does not change `od_demand.csv` or total
external demand. Across the formal Nanshan crop family, demand changes only
because the geographic window changes.

The MNL coefficients, delivery-time attributes, price discount, walking
penalty, costs, availability grid, service rates, flight speed, and queueing
limits are scenario inputs rather than estimates from the order records. They
must be varied transparently in sensitivity analysis.

The default operating vector is:

| Parameter | Default | Unit |
|---|---:|---|
| availability levels | 0.50, 0.65, 0.80, 0.90, 0.95 | fraction |
| base / locker / swap fixed cost | 14 / 1 / 3 | CNY per open facility-hour |
| locker capacity levels | 0.5 / 1 / 2 | CNY/hour for 30 / 40 / 50 drone handlings/hour |
| corridor / swap-leg activation | 0.02 / 0.05 | CNY per km-hour |
| flight / fleet | 0.20 / 5 | CNY per flight-km / drone-hour |
| drone cruise speed | 20 (72) | m/s (km/h) |
| maximum distance per battery | 10 | km per direct movement or per swap leg |
| usable battery operating time | 8.333 | minutes |
| library display-origin / general-corridor top-k | all / all | complete paper index set |
| CLI display-origin / general-corridor top-k | 8 / 10 | explicit large-instance screened preset |
| direct courier tariff | 6.40 through 3 road km; 1.80 thereafter | CNY per order / extra road km |
| feeder rider cost factor | 0.30 | Nanshan planning multiplier on first/last-mile rider legs only |
| locker handling | 0.30 | CNY per completed locker pickup |

The 10 km value is the operationally usable distance already applied by route
screening. Together with 20 m/s, it also identifies the 8.333-minute usable
battery duration used by the aggregate energy constraints. Road distance for a
courier leg is estimated as 1.25 times local straight-line distance before the
piecewise tariff is applied. The paper defines direct courier delivery
separately from first- and last-mile feeder movements. The
`--feeder-rider-cost-factor` option preserves the direct tariff while allowing
an explicitly labelled contracted/batched feeder-cost scenario; it must not be
interpreted as an empirical estimate without supporting data. The default
0.30 is an explicitly labelled working calibration input; replace it with
enterprise first-/last-mile contract data when those data become available.
The full-parent calibration target is approximately 50--60 open and active H
sites; it is checked from the optimized design and is not imposed as a
cardinality constraint.

## Environment

Python 3.11+ with NumPy, scikit-learn, Matplotlib, and a licensed Gurobi
installation is required. In this workspace the complete interpreter is:

```bash
/opt/anaconda3/bin/python
```

## Build the formal Nanshan scale family

Build all six centered geographic crops, including the 20 km by 20 km H120
full parent:

```bash
python -m src.utils.nanshan_scale_builder \
  --output-root data/instances/nanshan_nested_h120
```

The builder refuses to overwrite a nonempty output root. It writes
`scale_manifest.json` plus six canonical instance directories, verifies common
raw-data provenance and center coordinates, audits nested raw-order multisets,
and records that demand was neither sampled nor rescaled.

## Prepare another canonical instance

TSL example:

```bash
python main.py prepare \
  --source tsl \
  --candidate-count 30 \
  --instance-dir data/instances/tsl_od_h30
```

Both adapters write:

- `sites.csv`: typed `L/H/E` candidates;
- `od_nodes.csv`: customer origin and destination grid points;
- `od_demand.csv`: raw counts, exposure hours, and `lambda_per_hour`;
- `periods.csv` and `metadata.json`: provenance and time-window definitions.

When `--instance-dir` is omitted, the generated directory name contains a
deterministic preprocessing-configuration hash.

## Solve

```bash
python main.py solve \
  --instance-dir data/instances/nanshan_nested_h120/H120_L08_E13_R10p00 \
  --results-dir results/nanshan_h120_schema5 \
  --modes HD,MX,LO \
  --demand-scale 1 \
  --rho-levels 0.50,0.65,0.80,0.90,0.95 \
  --drone-speed-mps 20 \
  --battery-range-km 10 \
  --locker-service-rates 30,40,50 \
  --locker-capacity-costs 0.5,1,2 \
  --drone-cost-per-hour 5 \
  --flight-cost-per-km 0.20 \
  --feeder-rider-cost-factor 0.30 \
  --display-origin-choices 8 \
  --general-corridor-choices 10 \
  --network-start-seconds 60 \
  --no-rel-heur-seconds 10 \
  --threads 4 \
  --solver-seed 43 \
  --max-root-cut-rounds 3 \
  --root-cut-passes 2 \
  --time-limit 900 \
  --mip-gap 0.01
```

Each successfully solved mode writes `incumbent.sol` in its mode directory. To
reuse one in a later calibration run, pass
`--warm-start-from results/previous_run/HD/incumbent.sol`; the file is applied
only to the first requested mode, while a later MX mode still uses the HD
incumbent produced in the current run.

```bash
python main.py solve \
  --instance-dir data/instances/example \
  --modes HD \
  --warm-start-from results/previous_run/HD/incumbent.sol
```

The OD-to-corridor interface has two optional top-k screening controls, plus the
paper's walking-coverage parameter:

- `--display-origin-choices` limits backend origin lockers used for a
  displayed destination locker;
- `--general-corridor-choices` limits general OD corridor candidates;
- `--max-walk-km` defines the exact customer-visible set `H_d`; a finite value
  is part of the paper model and is not an approximation.

`ModelConfig` defaults both controls to `None`, so direct library use retains
the complete manuscript index set. The command-line interface uses an 8/10
large-instance safety preset because complete OD-to-corridor enumeration grows
very quickly. A finite preset reports `screening_is_exact=false`, and its
incumbent and bound apply only to that screened formulation. On a
computationally manageable instance, restore the complete index set with

```bash
--display-origin-choices all \
--general-corridor-choices all
```

Changing `--max-walk-km` is a model sensitivity experiment, not a step required
to restore exactness. The code keeps `h_od` independent of all these choices.

The default solution method uses exact flight-leg range preprocessing, tight
valid flow bounds, a feasible unified `chi -> w -> psi` network start, exact-conic local
branching, conic barrier at the root, and a limited pool of globally valid user
cuts. See [BRANCH_AND_CUT.md](BRANCH_AND_CUT.md) for the current cut contracts.

## Input and solution figures

Render the canonical input layout without solving:

```bash
python -m src.utils.instance_plotter \
  data/instances/nanshan_nested_h120/H120_L08_E13_R10p00
```

The input figure shows local anonymized coordinates, OD endpoint activity, and
candidate H/L/E sites. Solution figures use only schema-5 artifacts from the
current formulation.

## Outputs

Every mode directory writes a schema-5 `summary.json` containing:

- `formulation_id=mlnd-unified-chi-epsilon-energy-v3`;
- solver status, incumbent objective, best bound, MIP gap, and runtime;
- total demand, total drone offered flow, total drone realized flow, locker
  pickup, assisted home delivery, and direct courier fallback;
- cost components and objective-reconciliation error;
- open L/H/E facilities, active-workload H sites, capacity levels, physical
  connections, and selected availability levels;
- swap share, chaining share, and the idle/locker/transit fleet components;
- battery range, speed, derived usable duration, and maximum aggregate-energy
  violation;
- formulation screening controls and `screening_is_exact`.

The mode directory also contains:

- `open_facilities.csv`;
- `corridor_flows.csv`;
- `od_service.csv`;
- `od_corridor_reliability.csv`, exposing `chi` and `rho*chi` by
  OD/corridor/level;
- `displayed_locker_service.csv`, exposing `alpha`, `eta`, `epsilon`, and
  actual locker pickup;
- `locker_metrics.csv`;
- `energy_balance.csv`, exposing energy use, base-departure and swap supply
  events, slack, and violation of the system-wide aggregate constraint;
- network, OD service, cost, service-funnel, and utilization figures in PNG
  and SVG.

`run_config.json` records the input hashes, demand scale, modes, screening
controls, raw and derived model parameters, formulation ID, manuscript hash,
implementation-file hashes, and a signature also embedded in every summary.
A requested result directory must be absent or empty; the solver refuses to
mix a new run with stale artifacts.

Run the six-level scale experiment sequentially so every level uses the same
model and solver settings. Each run solves `HD` first and then installs that
same-instance incumbent in `MX` by stable variable name. The reported network
design and gap are the MX results; HD is a reproducible algorithmic starting
phase, not an externally supplied warm start. The result directories use simple
`H<number>` names because that is the scale reporter's discovery contract:

```bash
while read -r h instance; do
  python main.py solve \
    --instance-dir "data/instances/nanshan_nested_h120/${instance}" \
    --results-dir "results/nanshan_nested_scale_final_schema5/H${h}" \
    --modes HD,MX \
    --demand-scale 1 \
    --rho-levels 0.50,0.65,0.80,0.90,0.95 \
    --drone-speed-mps 20 \
    --battery-range-km 10 \
    --locker-service-rates 30,40,50 \
    --locker-capacity-costs 0.5,1,2 \
    --drone-cost-per-hour 5 \
    --flight-cost-per-km 0.20 \
    --feeder-rider-cost-factor 0.30 \
    --display-origin-choices 8 \
    --general-corridor-choices 10 \
    --network-start-seconds 60 \
    --no-rel-heur-seconds 10 \
    --threads 4 \
    --solver-seed 43 \
    --max-root-cut-rounds 3 \
    --root-cut-passes 2 \
    --time-limit 300 \
    --mip-gap 0.01
done <<'LEVELS'
11  H011_L02_E02_R03p00
24  H024_L02_E03_R04p50
43  H043_L03_E05_R06p00
68  H068_L05_E07_R07p50
87  H087_L06_E09_R08p50
120 H120_L08_E13_R10p00
LEVELS

python -m src.utils.scale_report results/nanshan_nested_scale_final_schema5
```

Do not pass `--warm-start-from` in this benchmark. The solver records
`cross_mode_start_policy=HD_to_later_MX_by_variable_name`, and the scale
reporter rejects a missing HD summary, a foreign or unmatched MX external
start, or any file-based warm start. The configured time limit applies to each
mode separately. The scale report therefore retains MX runtime and also reports
HD runtime plus the end-to-end `HD -> MX` runtime; the nominal end-to-end budget
is twice the per-mode setting, although barrier timeout checks can delay
wall-clock return.

The dynamic benchmark reporter obtains the H count from the instance and
schema-5 result artifacts; it does not assume H34. After the three H120 modes
finish, create the full physical-route network panel, logarithmic optimality-gap
certificate, and machine-readable summary with:

```bash
python -m src.utils.nanshan_benchmark_report \
  --instance-dir data/instances/nanshan_nested_h120/H120_L08_E13_R10p00 \
  --results-root results/nanshan_h120_schema5
```

The comparison map reads each auditable `incumbent.sol`, reconciles its
physical route flows with the summary, and distinguishes loaded H--H movement
from base dispatch/recovery; one-swap routes bend through the selected swap
station. The certificate prints each incumbent and valid bound while plotting
the MIP gaps on a log scale, so a weak large-instance bound cannot visually
compress the well-solved modes.

The crop-scale report allows OD-node, OD-demand, and whole-file `periods.csv`
hashes to change with geographic coverage. It still requires the same raw-file
hash, selected dates, period definition and exposure, center, demand and
candidate time windows, `demand_scale=1`, model configuration, solver
configuration, per-mode time limit, and target gap. It reports map side length,
area, demand, demand density, HD runtime, MX runtime, end-to-end runtime, gap,
cost saving, service, and network design.
It also rejects a scale family unless every run records exactly 20 m/s drone
speed and 10 km maximum battery-leg range (apart from floating-point noise).
An older fixed-map candidate-count series is labelled separately and cannot be
mixed into the nested-crop report.

Parameter-sweep reports remain available for explicitly declared scenario
inputs:

```bash
python -m src.utils.parameter_report results/nanshan_parameter_schema5
```

## Legacy and debugging results

H34 Nanshan instances and their fixed-map candidate-count series are retained
only for historical provenance, debugging, and regression checks. They are not
the current empirical scale definition and must not be presented as the formal
Nanshan benchmark. Results using an earlier schema or formulation remain
incomparable with schema-5 unified `chi-epsilon` plus aggregate energy. The
reporters enforce these distinctions rather than inferring comparability from
directory names.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/mlnd-mpl \
XDG_CACHE_HOME=/tmp/mlnd-xdg \
python -m pytest -q -p no:cacheprovider
```

The test suite checks the unified variable interface, forward realization,
mode-specific service accounting, objective reconstruction, route-range
boundaries, aggregate energy sufficiency, valid inequalities, output schemas,
plots, data adapters, and licensed Gurobi smoke instances. Large time-limited
runs must always report incumbent, bound, and gap separately.
