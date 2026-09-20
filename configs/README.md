# Model parameters

`default.json` is the production baseline. Every experiment loads this file,
then deep-merges a partial JSON configuration or explicit overrides. Unknown
keys and invalid units/ranges are rejected. No previous result directory is
used as a parameter source.

- `data`: canonical instance family, processed calibration manifest, demand multiplier.
- `depreciation`: 5 years, 365 operating days per year, 12 operating hours per day.
- `costs`: fixed investment in CNY, plus operating and courier costs.
- `operations`: capacities, travel/handling times, reliability and rider-time family.
- `choice`: destination-locker pickup utility and maximum pickup distance.
- `solver`: time limit, threads, random seed, optional MIP tolerances and iteration limit.

Fixed investments are the **equivalent capital corresponding to the latest
validated hourly coefficients**, including the rounding used in that experiment.
They are not new purchase-price estimates. Each is divided by the depreciation
operating hours (21,900 by default):

| Asset | Equivalent investment | Hourly coefficient |
|---|---:|---:|
| Base equipment | 459,900 CNY/base | 21.00 |
| Cabinet equipment | 275,940 CNY/cabinet | 12.60 |
| Swap equipment | 98,550 CNY/station | 4.50 |
| Drone | 197,100 CNY/drone | 9.00 |
| Directed route | 11,333.25 CNY/km | 0.5175 |
| Base / locker site setup, each | 20,000 CNY/site | 0.913242 |
| Swap site setup | 10,000 CNY/site | 0.456621 |

Site setup is charged once per location; cabinet equipment is charged per unit.
Changing depreciation updates all fixed hourly coefficients consistently.
`flight_per_hour` and `hover_per_hour` are operating CNY/drone-hour. A null
hover cost uses the flight operating rate. Courier fares are CNY/order and
CNY/road-kilometer; the road-distance factor is applied before the included
kilometers are deducted. First/last-mile fares use `feeder_rider_multiplier`.

Model durations are in hours. Configuration action times are in minutes.
Null `rider_minutes_per_km` and `terminal_dispatch_minutes` use the stored fits.
The rider slope controls first/last-mile travel; the historical OD delivery
benchmark remains separately calibrated. `deadline_multiplier` changes the
service deadline without changing the matched courier benchmark.

Example partial sensitivity configuration:

```json
{
  "name": "capacity_90_percent",
  "operations": {"cabinet_capacity": 27.36},
  "solver": {"time_limit": 1800, "threads": 4}
}
```

```python
from src.config import load_config
from src.data import load_instance

config = load_config("configs/default.json", overrides={"solver": {"seed": 43}})
data = load_instance(120, config)
```

`solver.column_limit` (default 500,000) stops full-path enumeration before an
oversized model is built. `solver.graph_limit` (default 1,000,000) similarly
limits state-graph arcs for the direct arc formulation. A null value disables
that guard. These guards do not silently remove paths or arcs. Pricing backend
and acceleration defaults belong to the five method profiles in
`src/solvers/methods.py`; all methods use the same model parameters.

## Keep demand independent of candidate counts

The default is `data.demand_hubs: 120`. All candidate sizes use the complete
H120 demand market, its OD coordinates/order, rider costs and OD delivery-time
calibration: 11,427 OD pairs and 8,196.7143 orders/hour. `--hubs` selects the
candidate facilities only. An OD outside the candidate-site extent remains in
the objective, with direct-rider service for any demand not served by drones.

```json
{"data": {"demand_hubs": 120, "demand_scale": 1.0}}
```

Set `demand_hubs` to another available size (e.g. 24) to fix a different market
for every candidate size. Set it explicitly to `null` to use each candidate
instance's own geographic crop and demand; `configs/geographic.json` provides
that historical mode. `demand_scale` changes volumes without changing OD pairs.

```sh
python main.py audit-demand --hubs 11 24 43 68 87 120
python main.py audit-demand --config configs/geographic.json
```

These commands load and compare data without optimization. Batch runs audit
fixed demand before solving, then check each case against the same fingerprint.
The check includes OD identities, coordinates, volumes, rider baseline costs
and delivery benchmarks; matching only the total number of orders is not enough.
Demand source, counts, baseline cost and fingerprint are recorded in saved runs.

The existing candidate sets were reclustered separately within different
geographic crops; their L/E counts and locations also vary. Fixing demand does
not make these facility sets nested or turn them into an experiment that changes
only H while holding all other candidates fixed. Historical input files and
saved-run replay retain their original demand definitions.
