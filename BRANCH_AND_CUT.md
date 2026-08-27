# Branch-and-cut for the unified chi-epsilon-energy MLND

## Scope and formulation identity

This document applies only to

```text
formulation_id = mlnd-unified-chi-epsilon-energy-v3
summary schema_version = 5
```

The algorithm solves the OD-aware MISOCP directly. It does not form a Benders
master or decompose by OD market. Every static inequality and every callback
cut must be globally valid for the mode being solved. A useful inequality for
the locker-only restriction is not automatically valid for mixed delivery.

The `ModelConfig` library defaults retain the complete OD-to-corridor index set.
The CLI uses finite 8/10 top-k values as a large-instance computational preset;
these are explicit approximations and cause the affected mode to report
`screening_is_exact=false`. By contrast, `max_walk_km` defines the paper's
customer-visible set `H_d`; a finite walking radius remains exact. CLI value
`all` restores a top-k control to the complete index set on manageable
instances.

## 1. Unified flow notation

For market `(o,d)`, loaded corridor `(i,j)`, and availability level `r`, let

```text
chi_odijr >= 0
```

be offered flow before service realization. For eligible idle state `ell`,

```text
w_ellijr >= 0
```

allocates the same offered pool to an idle source. The linking equality is

```text
sum_od chi_odijr = sum_ell w_ellijr       for every (i,j,r).
```

The realized loaded flow is generated forward:

```text
psi_ij = sum_odr rho_r chi_odijr
       = sum_ellr rho_r w_ellijr.
```

Define the total offered flow on corridor `(i,j)` as the expression

```text
b_ij = sum_odr chi_odijr = sum_ellr w_ellijr.
```

`b_ij` is notation used in validity arguments; it need not be a separate model
variable. Since every `rho_r` is strictly positive, `b_ij > 0` implies
`psi_ij > 0`.

In mixed delivery, `epsilon_odj` is successful flow ending at the displayed
locker `j`, before multiplication by the fixed conditional share `eta_odj`.
The share is uniform across backend origins `i` and levels `r`.

## 2. Exact preprocessing and valid bounds

### 2.1 Route-feasibility screening

For a physical movement `(u,v)`, preprocessing retains the direct option when
its distance does not exceed the configured one-battery range. It independently
retains every one-swap option whose access and egress legs each satisfy that
same range. Thus, a feasible direct option does not remove a feasible swap
option. This distinction is required by the aggregate energy constraints:
every route consumes energy, every physical-base departure supplies an initial
fully charged battery, and a swap-assisted route supplies one additional fully
charged battery.

Flight-leg range infeasibility is exact preprocessing. The system-wide energy
constraint is an aggregate relaxation rather than individual route or drone
state-of-charge tracking. Top-k OD-to-corridor screening is not exact.

### 2.2 Aggregate energy sufficiency

The confirmed operating inputs are a 20 m/s cruise speed (72 km/h) and a
10 km usable one-battery range. Their ratio gives the paper's equivalent usable
battery duration:

```text
T_battery = (1-epsilon) Omega / omega = 10/72 hours.
```

The system-wide constraint is

```text
sum_i xi_i + sum_(u,v,k) t_uvk varphi_uvk
  <= T_battery * (base_departures + swap_events),
```

where `base_departures` counts every route flow leaving a physical base and
`swap_events` counts every one-swap route flow. A base-departing swap route is
intentionally counted in both supply terms because it uses one initial battery
and one replacement battery. Logical virtual waiting states are excluded.

### 2.3 Physical and offered-flow bounds

Let

```text
D         = sum_od h_od
rho_min   = min_r rho_r
rho_max   = max_r rho_r
kappa_max = max_c { mu_c - 1/tau_H }.
```

Every physical movement enters or leaves a locker, while total realized
service cannot exceed `rho_max D`. One certified common choice for the
physical-flow activation constant is

```text
M2_uv = min(rho_max D, kappa_max),
```

possibly tightened further with reachability and mode-specific demand.

Because `psi_ij >= rho_min b_ij`, one certified offered-corridor activation
constant is

```text
M1_ij = min(eligible OD demand for (i,j), kappa_max/rho_min).
```

The implementation must compute the eligible-demand term from the actual
mode-specific corridor keys. It must not reuse a bound derived from deleted
locker/home flow families.

The displayed-flow linearization uses the exact derived bound

```text
M4_odj = rho_max h_od.
```

`M3_i` bounds only the locker population variable `xi_i`. Idle populations at
physical and virtual states are computed directly from selected availability:

```text
sum_r rho_r/(1-rho_r) delta_ellr.
```

No additional global route-flow multiplier is assumed. In particular, the
former unproved bound based on a fixed multiple of loaded flow is not part of
this formulation.

## 3. Exact-conic primal starts

The analytic all-courier solution remains feasible in every mode and is always
installed as a MIP start.

A nontrivial network start must be constructed with the unified forward chain:

1. choose a candidate physical design, capacities, and availability levels;
2. assign a feasible offered pool through `chi` and `w`;
3. generate `psi` with the selected `rho` levels;
4. repair circulation, route decomposition, queueing, holding-state, and fleet
   feasibility in the original conic model;
5. in MX, set `alpha` and `epsilon` consistently with successful flow; in LO,
   enforce the choice-generated offered-flow equalities.

Fixing a displayed locker alone does not force positive drone flow in MX and
is therefore not a valid replacement for this construction.

Exact-conic local branching may search a Hamming neighborhood of the binary
design. It retains every continuous variable and SOC, removes its temporary
neighborhood and cutoff rows before the global solve, and installs a candidate
only when the complete solution is feasible.

Whenever this network seed exists, the global solve receives three ordered MIP
starts: the complete auxiliary/local-branching incumbent; a partial copy that
defines only the structural binaries `(x, z, delta, alpha)` and leaves every
other variable at `GRB.UNDEFINED`; and the always-feasible all-courier start.
The partial copy lets Gurobi reconstruct all continuous and conic variables if
a tiny numerical residual prevents acceptance of the complete copy. This is a
primal-initialization safeguard only and changes no constraint, bound, or
objective. The result summary records the auxiliary seed's open base, locker,
and swap-station counts together with the partial-start pool metadata.

For MX and HD, the auxiliary start search requires only one topologically
supported base--locker--locker--base cycle. The auxiliary MISOCP itself checks
system-wide energy feasibility. It may use all base and swap candidates and
an activity-ranked locker support of
`min(|H|, 60, max(16, ceil(0.5 |H|)))`. These restrictions are temporary: all
bounds are restored before the reported solve, so this support affects only
incumbent construction and never removes a feasible design from the main
model.

An HD incumbent can also be supplied as an exact feasible start to a later MX
solve of the same instance and parameterization. HD corridors form a subset of
the MX corridor set; shared variables retain their values and MX-only choice
variables start at zero, preserving feasibility and objective value. This
cross-mode start is optional and is reported explicitly when installed.

## 4. Root relaxation

The default root method is conic barrier (`Gurobi Method=2`). User cuts are
added only at an optimal root `MIPNODE` relaxation with `PreCrush=1`. They are
strengthening cuts, not lazy feasibility constraints. The default uses three
callback separation rounds and two solver-native root cut passes; this keeps
the H120 screened benchmark from spending its whole budget in automatic root
separation. These settings affect search performance, not the feasible set.

## 5. Dynamically separated cuts

### 5.1 Unified OD-corridor VUB cuts

For every modeled market and loaded corridor, the following disaggregated VUB
is valid in MX, HD, and LO:

```text
sum_r chi_odijr <= h_od y_ij.
```

Its per-level form

```text
chi_odijr <= h_od y_ij
```

is also valid, but adding both families indiscriminately may harm the root
solve. Separation ranks violated rows and respects the configured cut budget.

These cuts strengthen the aggregate corridor activation row without changing
the offered-demand or reliability semantics.

### 5.2 Locker-only choice-corridor cuts

Only LO forces the complete choice-generated locker demand to the displayed
destination locker:

```text
sum_ir chi_odijr = h_od eta_odj alpha_dj.
```

Therefore, if `I_odj` is the set of eligible backend origins for that displayed
choice, the following support cut is valid in LO:

```text
alpha_dj <= sum_(i in union_o I_odj) y_ij.
```

Tighter market-specific variants may be used when their indexing is preserved.
This family is invalid in MX because displaying `j` does not force the platform
to offer any order to drones.

### 5.3 Locker-only choice-capacity covers

For LO, define

```text
B_dj = sum_o h_od eta_odj.
```

Displaying destination `d` at locker `j` creates at least
`rho_min B_dj` realized arrival flow at `j`. For a destination subset `P`, let
`C(P)` be the capacity levels that can cover this flow. The lifted cover

```text
sum_(d in P) alpha_dj
    <= |P| - 1 + sum_(c in C(P)) z_jc
```

is valid for LO. Candidate subsets may be generated heuristically, but every
returned inequality must be checked against its exact capacity condition.

This cover is not valid in MX or HD. In MX, `alpha` qualifies successful flow
for a cost adjustment through `epsilon`; it does not impose minimum arrival.

### 5.4 Discrete-capacity queue perspective cuts

For locker `i`, use the affine arrival expression

```text
arrival_i(psi) = sum_(u:(u,i) in A) psi_ui.
```

For capacity level `c`, define

```text
f_c(t) = t/(mu_c-t),
0 <= t <= kappa_c,
kappa_c = mu_c - 1/tau_H.
```

For slope `pi >= 0`, let

```text
b_c(pi) = min_(0<=t<=kappa_c) { f_c(t) - pi t }.
```

The common-slope inequality

```text
xi_i >= pi arrival_i(psi) + sum_c b_c(pi) z_ic
```

is valid for the union of discrete M/M/1 capacity states. The separator finds
the most violated support over `pi` and adds only rows exceeding the numerical
tolerance. There is no separate arrival-rate decision variable in the unified
model.

### 5.5 Conditional directed return-connectivity cuts

Contract each virtual state into its co-located locker. Positive `b_ij` implies
positive loaded flow from `i` to `j`. A positive arc in a nonnegative balanced
circulation must lie on a directed cycle, so a return path from `j` to `i` must
exist.

For a directed cut set `U` containing `j` but not `i`, a valid conditional cut
is

```text
b_ij <= M1_ij sum_(a in delta+(U)) y_a.
```

In implementation, `b_ij` is expanded as `sum_odr chi_odijr` or equivalently
`sum_ellr w_ellijr`. The separator computes a directed minimum cut at the
fractional `y` point. It never imposes connectivity for an external OD market
that uses courier fallback only.

## 6. Separation policy

The callback uses a global per-round cut cap and deduplicates algebraically
identical signatures. The cut allocation is based on observed violation, not
on historical locker/home quotas. A family producing no cut means the current
root point satisfies that family; it is not a correctness failure.

The full static strengthening pool remains an optional ablation. Any claim
that one cut family improves the bound or incumbent requires a matched run in
which barrier, starts, local branching, time limit, seed, and all other cut
families are held fixed.

## 7. Algorithm outline

```text
build mode-specific unified OD-corridor keys
preprocess direct flights and one-swap flight legs by one-battery range
compute M1, M2, M3, and M4 bounds
build the chi -> w -> psi MISOCP
add the system-wide base-departure-plus-swap energy constraint
install the all-courier start
optionally construct and repair a feasible unified network start
optionally run exact-conic local branching
solve the root relaxation by conic barrier
for each permitted root round:
    separate unified chi VUB cuts
    if mode == LO:
        separate choice-corridor and choice-capacity cuts
    separate queue-perspective cuts
    separate offered-corridor return-connectivity cuts
    add at most the configured number of globally valid cuts
continue solver-native branch-and-bound
```

## 8. Correctness and reporting checks

Small complete-index instances must compare the strengthened algorithm against
the solver-native model with zero target gap. The objectives must agree within
numerical tolerance in MX, HD, and LO.

Every reported incumbent must reconcile:

```text
offered  = sum chi
realized = sum rho*chi = sum_(i,j) psi_ij

MX: locker = sum eta*epsilon; home = realized-locker
HD: locker = 0;               home = realized
LO: locker = realized;        home = 0

fallback = total demand-realized
locker + home + fallback = total demand.
```

Reporting reconstructs the system-wide aggregate energy inequality, including
base-departure battery issuances and one-swap battery exchanges. The maximum
positive violation must be within the numerical feasibility tolerance.

For a minimization problem terminated by a time limit, the incumbent objective
is an upper bound and `ObjBound` is the proven lower bound. Status, incumbent,
bound, gap, and runtime must be reported separately.

## 9. Formal Nanshan benchmark and geographic scale contract

The formal full-parent Nanshan instance covers a centered 20 km by 20 km
window and contains `H/L/E = 120/8/13` candidate facilities. The desired
opening of approximately 50--60 smart lockers is a calibration target that
must be checked by optimization; it is neither a constraint nor a result that
may be stated before the solve finishes.

Computational scaling uses six same-center square crops with side lengths
6, 9, 12, 15, 17, and 20 km and H/L/E candidate targets
11/2/2, 24/2/3, 43/3/5, 68/5/7, 87/6/9, and 120/8/13. A raw order enters one
crop only when both pickup and dropoff coordinates lie inside that crop.
Every retained order is used exactly once, `demand_scale=1`, and facilities
are deterministically reclustered from all retained candidate-period orders
inside each crop. Thus OD-node, OD-demand, and whole-file period hashes may
change with crop area even though the raw file, selected dates, period
definition, exposure, projection center, model, and solver contract remain
fixed.

The scale report distinguishes this nested geographic design from an older
fixed-map candidate-count series. It must reject a result root that mixes the
two contracts. A geographic scale run reports map side length, area, demand,
demand density, incumbent, bound, gap, runtime, service, and network design;
it must not interpret increasing raw demand as an order multiplier.

## 10. Legacy H34 record

The earlier H34 run recorded objective 50,650.20 CNY/hour, bound 45,709.97,
gap 9.75%, 575.42 orders/hour of drone service, and 309 root cuts. It used
schema 1 and the superseded split locker/home flow formulation.

Those numbers are retained only as historical provenance. H34 is a debugging
and regression scale, not the current formal Nanshan benchmark or a member of
the six-level geographic scale curve. The recorded objective, facilities, and
cut counts are not a benchmark, warm start, expected bound, or calibration
target for `mlnd-unified-chi-epsilon-energy-v3`. Any new H34 diagnostic must
still emit schema 5 plus the exact formulation ID, but it must remain labelled
as a historical/debugging experiment.
