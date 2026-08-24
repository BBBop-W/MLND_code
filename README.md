# OD-aware multi-type drone network design

This repository implements the new formulation in `MLND/main.tex`. The
Overleaf-synchronized `MLND/` directory is an input reference and is not
modified by the code.

The new pipeline keeps customer markets `h_od` separate from facility
candidates. During optimization, the model selects the customer-visible
destination locker `j`, assigns the backend origin locker `i`, constructs the
endogenous corridor demand `h_ij`, and realizes it through a selected discrete
availability level `rho_r`. Orders not realized by drones fall back to direct
courier home delivery.

## Main model choices

- `MX`: drones may serve locker-pickup and home-delivery orders.
- `HD`: drones assist home delivery only.
- `LO`: drones serve potential locker-pickup orders only.
- Objective: modeled platform fulfillment cost in CNY/hour; there is no
  revenue or profit term.
- Default availability grid: `0.50, 0.65, 0.80, 0.90, 0.95`.
- Default candidate lockers: 35 (configurable; 30--40 is the intended range).
- Default study region: 16 km by 16 km; demand is the multi-day average for
  12:00--13:00.
- TSL input retains only `is_courier_grabbed=1`, so rejected waybills are not
  double-counted.
- TSL exposure uses the eight complete source partitions in `dt`; event hours
  use the Asia/Shanghai timestamp. Any timestamp-date/`dt` mismatch is counted
  in `metadata.json` instead of silently moving records into incomplete fringe
  dates.

The MNL coefficients are explicit scenario assumptions because no stated-
preference data are available. At the defaults, a locker 0.5 km from the
customer has about a 34% conditional adoption share. These values should be
varied in sensitivity analysis and must not be described as empirical
estimates.

The default operating costs are likewise a provisional mature-operations
scenario, calibrated to produce an economically active network for model and
figure validation rather than estimated from an external cost survey:

| Parameter | Default | Unit |
|---|---:|---|
| base / locker / swap fixed cost | 14 / 1 / 3 | CNY per open facility-hour |
| locker capacity levels | 1.5 / 3 / 5 | CNY/hour for 120 / 240 / 360 orders/hour |
| corridor / swap-connection activation | 0.02 / 0.05 | CNY per km-hour |
| flight / drone fleet | 0.20 / 4 | CNY per flight-km / drone-hour |
| rider leg | 3 + 1.80 × 1.25 × distance | CNY per rider leg |
| locker handling | 0.30 | CNY per locker order |

The earlier conservative vector remains part of the parameter-sweep evidence;
neither vector should be reported as empirically estimated. Every cost and
operating value can be overridden from the solve CLI and is embedded in
`run_config.json`.

## Environment

Python 3.11+ with NumPy, scikit-learn, Matplotlib, and a licensed Gurobi
installation is required. In the current workspace the complete interpreter is:

```bash
/opt/anaconda3/bin/python
```

## Prepare a canonical instance

TSL:

```bash
python main.py prepare \
  --source tsl \
  --candidate-count 35 \
  --instance-dir data/instances/tsl_od_h35
```

Nanshan:

```bash
python main.py prepare \
  --source nanshan \
  --candidate-count 35 \
  --instance-dir data/instances/nanshan_od_h35
```

Both adapters produce the same schema:

- `sites.csv`: typed `L/H/E` candidates;
- `od_nodes.csv`: customer origin and destination grid points;
- `od_demand.csv`: raw counts, exposure hours, and `lambda_per_hour`;
- `periods.csv` and `metadata.json`: provenance and time-window definitions.

`source_orders_all_days` reports all rows emitted by the selected source
adapter; for TSL, `accepted_waybills_all_days` separately makes the accepted-
only filtering explicit.

Changing `--candidate-count` does not change `od_demand.csv` or total demand.
Repeated values in `--dates` are deduplicated, and dates absent from the raw
source are rejected. When `--instance-dir` is omitted, the generated directory
name includes a deterministic preprocessing-configuration hash.

## Solve

```bash
python main.py solve \
  --instance-dir data/instances/tsl_od_h35 \
  --modes MX,HD,LO \
  --rho-levels 0.50,0.65,0.80,0.90,0.95 \
  --time-limit 300 \
  --mip-gap 0.01
```

Cost parameters are exposed directly, for example:

```bash
python main.py solve \
  --instance-dir data/instances/nanshan_od_h34 \
  --base-cost-per-hour 14 \
  --locker-capacity-costs 1.5,3,5 \
  --flight-cost-per-km 0.20 \
  --rider-cost-per-km 1.80
```

Prepare and solve in one command:

```bash
python main.py run --source tsl --candidate-count 35 --modes MX
```

The sparse OD-to-corridor interface is controlled by
`--origin-locker-choices` and `--home-corridor-choices`. The defaults are a
computational screening heuristic for the 12,000-market H35 instances, not an
exact reduction theorem. Use `--origin-locker-choices all
--home-corridor-choices all --max-walk-km all` on small instances to recover
the paper's complete OD--corridor index sets. In either setting, the code never
replaces OD demand with a precomputed `H x H` demand matrix.

The default `max_virtual_wait_hours=5/60` is also a scenario assumption. With
the default availability grid it implies minimum positive offered-flow levels
of roughly 24, 34.3, 60, 120, and 240 orders/hour at an active virtual state.
The summary reports active virtual states so this mechanism can be checked in
sensitivity analysis.

Large OD-aware instances have a linked design/flow feasibility threshold that
is difficult to cross from the all-courier start. By default, at most 20
seconds within the stated total time budget are used for primal construction:
up to 10 seconds for a high-volume display-bundle network seed and the
remainder for exact-conic local branching. The auxiliary display/home bounds
and the Hamming-neighborhood row are all removed before the reported solve;
the original objective and every SOC remain unchanged. Both the resulting
network start and an all-courier start are supplied to the global solve. Set
`--network-start-seconds 0` to disable this search aid.

The default global algorithm uses conic barrier for the root relaxation and a
limited root user-cut pool: OD-disaggregated VUB cuts, discrete-capacity queue
perspective cuts, lifted choice-capacity covers, and conditional directed
return-connectivity cuts. Exact preprocessing first removes dominated
via-swap route options and tightens physical/offered-flow bounds. The full
static strengthening pool is available with `--valid-inequalities`, but is
disabled by default because the H30 ablation improved the bound only slightly
while degrading short-time primal search. See [BRANCH_AND_CUT.md](BRANCH_AND_CUT.md)
for formulas, validity arguments, separation policy, and the reproducible H34
benchmark. Use `--root-method -1` for Gurobi automatic method selection or
`--no-user-cuts` for the solver-native ablation.

Render a canonical study-window layout without solving:

```bash
python -m src.utils.instance_plotter data/instances/nanshan_od_h40
```

The figure follows the repository's historical Nanshan candidate-point style:
gray demand-activity bubbles, red H circles, blue L triangles, and ochre E
squares, with no per-site label clutter. Coordinates are local anonymized
kilometres.

## Outputs

Each mode writes a self-contained folder with:

- solver status, bound, gap, runtime, cost baseline, savings, service funnel,
  open L/H/E counts, exact/screened-index flag, and active virtual-state count
  in `summary.json`;
- every open facility with its selected capacity and applicable discrete rho;
- OD service, corridor flow, and locker queue CSV files;
- network, OD service, cost, demand-funnel, and utilization figures in PNG/SVG.

The console report uses cost and service terminology rather than the old
revenue/profit labels.
It explicitly warns when the incumbent is courier-only and reports both the
requested and achieved MIP gaps. Gurobi `OPTIMAL` means optimal within the
requested MIP tolerance; it does not imply a zero gap unless the reported gap
is zero.
`run_config.json` embeds canonical input hashes, instance metadata, demand
scale, modes, model parameters, and solver parameters. Default results
directories include a deterministic run hash, preventing different scenarios
from being mixed under the same instance basename.

Candidate-scale runs stored in `Hxx` subdirectories can be consolidated with:

```bash
python -m src.utils.scale_report results/nanshan_scale_5min
```

This writes `benchmark_summary.csv` and an annotated runtime chart containing
the actual gap, open L/H/E counts, selected rho values, and whether the network
serves any demand.

Parameter scenarios stored one directory per run can be consolidated with:

```bash
python -m src.utils.parameter_report results/nanshan_parameter_tuning_20260821
```

This writes a CSV plus a two-panel savings/service chart annotated with open
L/H/E counts and the selected discrete rho by facility.

## Tests

```bash
pytest -q
```
