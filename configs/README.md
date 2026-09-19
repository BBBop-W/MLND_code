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
