# Consecutive-delivery scenario inputs

`default.json` is the default full configuration, schema 2, model `consecutive`.
Partial JSON configurations merge into it; unknown keys fail immediately.
The current default loads the frozen real TSL market with 2 base candidates,
10 locker-site candidates and 50 aggregate ODs, without resampling or demand scaling.
Future parameter comparisons use only its K=1 and K=2 cases, with `full_path`;
the capacity-alignment change itself does not run optimization.
Algorithm experiments instead replay the fixed `data/benchmarks/tsl_2l20h50od`
input using `--input` without a physical `--config`. `tsl_20h_benchmark.json` is
its parameter/provenance snapshot; passing that full snapshot as an override
would rebuild physical inputs and re-screen compatibility. The benchmark README
provides exact replay and independent input-hash verification commands.
An explicit `hubs` argument instead loads the canonical CSV candidate family;
`data.demand_hubs` controls the fixed OD market independently of facilities.

The default is an explicit planning scenario, not an empirical calibration of
consecutive acceptance: distance-compatible OD pairs are proposed before physical screening,
the acceptance window is 0.5 minutes, and each physical cabinet contains 8 compartments.
At each locker site, install 1–5 cabinets (8/16/24/32/40 compartments) or do not open
the site (zero cabinets). The five choices are alternative total capacities, not
five independent increments that can be combined. Site investment is charged once.
`compatibility.mode` accepts `all`, `none`, `distance`, or `matrix`. Distance mode
uses the first customer's location to the next merchant; it is a stated spatial
rule, not an estimated probability. Matrix mode requires an OD-by-OD matrix in
[0, 1]. `max_orders: 1` sets the model's effective acceptance window to zero.

`compatibility.require_feasible_path` defaults to true: retain a proposed pair
only when a complete feasible double-order route exists in the fixed candidate
network. Retained fractional coefficients are unchanged. Screening uses K=2
physics for both K comparisons and records support per base pair. It does not
promise feasibility under every subsequent facility/base-pair choice. Saved
snapshots replay without screening unless explicitly requested.

`flow_policy` defaults to `aggregate`, the requested relaxation: reconstructed
single and ordered-pair task totals still equal their fixed rates times common
base availability, but their first routes may differ by full task type. `prefix`
retains the attachment's per-first-route branch proportions. This is a route
information relaxation, not proof of an online policy. A policy-only override
preserves all physical arrays and the saved compatibility matrix. Old snapshots
without this field use `prefix`.

All configuration times are minutes; model arrays use hours. Ground-cost and
feeder-tariff arrays from saved inputs are retained when available, including
cluster-mean nonlinear courier tariffs. Infrastructure, flight and waiting costs
come from the explicit scenario. Investment is divided by the product of the
three depreciation factors. Locker site cost is charged once through the selected
capacity level; compartment investment is charged per physical slot.

The current depreciation factors already equal 5 years × 365 days/year × 12
operating hours/day = 21,900 operating hours. Do not depreciate an hourly cost again.
The user-specified equipment prices are 40,000 CNY per 8-slot cabinet and
50,000 CNY per drone. The per-compartment input is therefore 5,000 CNY; cabinet
equipment depreciates to 1.826484 CNY/hour and a drone to 2.283105 CNY/hour.
The retained 20,000 CNY site investment is charged once per open locker site.
Other infrastructure investments remain inherited model-equivalent values.
The current full parameter inventory and five-year cost conversion are in
`outputs/tsl_parameter_review_20261015/PARAMETERS.md`.

When a future solve is requested, use `python main.py compare --method full_path
--out outputs/NEW_TSL_COMPARISON` to apply the current defaults to both K cases.
An explicit old `--input` without `--config configs/default.json` deliberately
replays its old saved capacities; old experiment artifacts remain unchanged.

Terminal removal and final receipt are independent inputs:

- Self pickup defaults to 5 minutes to removal and 5 minutes to receipt.
- Courier removal defaults to 2 minutes. A null courier receipt value means this
  removal delay plus calibrated locker-to-customer travel, an explicit assumption.
- Setting `courier_receipt_minutes` provides a separate total receipt mean.
- Receipt before removal is rejected. No historical removal time is relabeled as
  receipt time. Both terminal modes are mixed with the fixed pickup probability.

Choice uses `beta0 - beta_distance * distance_difference_km - beta_time *
time_difference_minutes - beta_price * price_difference_cny - beta_walk *
walking_distance`. Attribute differences mean self pickup minus home delivery.
The default time difference (-10 minutes), price difference (-1 CNY), zero distance
coefficient and walking factor 1 are explicit scenario assumptions. Attributes
may be scalar or broadcastable OD-by-locker arrays. Probability is zero outside
`max_walk_km`; the boundary itself is included.

A null `rider_minutes_per_km` reads the saved or verified calibration slope.
Its exponential distribution is a working assumption; only the mean is fitted.
A null `deadline_minutes` reads the saved/verified per-OD post-fetch benchmark;
`deadline_multiplier` applies explicitly. Missing evidence raises an error.

`route_mode: direct` fixes each leg to a direct flight. `simple` permits simple
L/H-network routes; it can be much larger. Saved `candidate_routes`, when present,
define the supplied route set exactly. The default operating-time limit is
10 minutes of mean complete flight-plus-waiting time, from departure from a base
through the final return to a base, including all pickup waits. A double-order
task shares this one limit across both orders. Customer pickup or final courier
delivery after drone deposit, pre-dispatch idle time, and recharge time are not
part of this duration. There is no explicit battery capacity, state of charge,
or power/energy chance constraint; the mean-duration cap is only a time-based
endurance proxy, not a guarantee that every realized mission finishes within
10 minutes. There are no E facilities or swapping operations.

Every converted input saves the complete resolved configuration, source hashes,
and field-specific source/assumption notes. A new-model snapshot replays exactly
with `load_saved(path)` or a solver-only override. Physical overrides merge over
the saved resolved scenario, preserving other choices; changing only `max_orders`
keeps every physical coefficient unchanged. Without saved scenario metadata,
physical updates require a complete explicit configuration. Demand and deadline
multipliers are reapplied to their original values exactly once.

`prepare_orders` writes only to a new output directory (normally
`outputs/prepared/...`), with independent `n_l`, `n_h` and `n_od`. It preserves raw
order multiplicities, aggregates joint 4D OD tuples, reports clustering distortion,
and rejects writes into original data or manuscript directories.
