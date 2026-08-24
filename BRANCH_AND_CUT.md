# Demand-aware perspective branch-and-cut

## Scope

The algorithm solves the OD-aware MISOCP directly. It does not form a Benders
master or decompose by OD market. All generated inequalities are globally
valid for the formulation already built by `MLNDModel`; they remove fractional
relaxation points, not integer-feasible solutions.

With the default `origin_locker_choices=3`, `home_corridor_choices=4`, and
`max_walk_km=1.5`, the result is exact for that screened formulation, not for
the complete OD--corridor index set. Setting all three screening controls to
`None` restores the complete index sets but produces a substantially larger
model. Extending the method to that case should use branch-cut-and-price rather
than pretending the top-k screen is an exact reduction.

## 1. Exact preprocessing and bounds

For a physical movement `(u,v)`, a one-swap route through `k` is deleted when
direct flight is feasible. By the triangle inequality its distance is no
shorter than the direct distance, it adds positive swap time, and it requires
the parent connection plus two additional swap-leg connections. Replacing its
flow by direct flow therefore preserves feasibility and cannot increase cost.

Let `D=sum_od h_od`, `rho_max=max_r rho_r`, and

```text
kappa_max = max_c { mu_c - 1/tau_H }.
```

Every physical flow satisfies

```text
psi_uv <= min(rho_max D, kappa_max) y_uv.
```

Since `psi_ij >= rho_min h_ij`, an endogenous offered-flow corridor also has

```text
h_ij <= min(M_ij^OD, kappa_max/rho_min) y_ij.
```

On the Nanshan H34 instance, the physical-flow bound falls from 8,006.43 to
348 orders/hour and the largest offered-flow bound is 696 orders/hour.

## 2. Exact-conic primal start

The initial solution has two phases within one stated time budget.

1. Rank `(destination, locker)` display bundles by induced OD volume, with a
   first-order fulfillment-saving score as a tie-breaker. Fix the leading
   display bundle, disable optional home assistance, and solve the original
   MISOCP for at most 10 seconds. This phase finds a linked facility/flow/queue
   feasible network rather than choosing facilities independently.
2. Restore every bound and search a Hamming ball in `(x,z,delta,alpha)`. The
   local problem retains every original continuous variable and SOC. Its center
   seed remains feasible; only a strictly better complete solution is installed.

The global solve receives both the network start and the analytic all-courier
start. The auxiliary restrictions and the local-branching row are removed
before the reported solve.

## 3. Root relaxation

The default root method is conic barrier (`Gurobi Method=2`). This is important
because the H34 formulation contains roughly 300,000 continuous variables and
the root relaxation, rather than the number of integer variables, is the main
bottleneck. The MIP nodes remain solver-native.

## 4. Dynamically separated user cuts

Cuts are added only at an optimal root `MIPNODE` relaxation with `PreCrush=1`.
They are user cuts, not lazy feasibility constraints.

### 4.1 OD-disaggregated VUB cuts

For locker service key `p=(o,i,j,d)` with conditional choice share
`eta_odj`,

```text
a^lk_p <= h_od eta_odj y_ij.
```

For assisted-home key `p=(o,i,j,d)`,

```text
a^hm_p <= h_od y_ij.
```

These inequalities disaggregate the weak corridor fixed-charge row. A single
OD contribution can no longer exploit a corridor `M` that aggregates thousands
of other orders. Only the most violated rows are added; adding all 133,329 H34
VUB rows statically was tested and harmed short-time primal search.

### 4.2 Discrete-capacity queue perspective cuts

For capacity level `c`, define

```text
f_c(t) = t/(mu_c-t),       0 <= t <= kappa_c,
kappa_c = mu_c - 1/tau_H.
```

For any slope `pi>=0`, let

```text
b_c(pi) = min_{0<=t<=kappa_c} { f_c(t) - pi t }.
```

Then the common-slope inequality

```text
xi_i >= pi lambda_i + sum_c b_c(pi) z_ic
```

is valid for the union of all discrete M/M/1 capacity states. The separator
maximizes its violation over `pi` by one-dimensional bisection, using the exact
piecewise support function. This gives the convex-hull geometry missing from
the raw fractional-capacity SOC.

### 4.3 Choice-capacity covers

Let `B_dj=sum_o h_od eta_odj`. For a destination subset `P` displayed at
locker `j`, every selected display creates at least
`rho_min sum_(d in P) B_dj` arrival flow. If only capacity levels in set `C(P)`
can cover this amount, the following lifted cover is valid:

```text
sum_(d in P) alpha_dj <= |P|-1 + sum_(c in C(P)) z_jc.
```

Candidate minimal covers are separated by demand/deficit orderings. The
separation is heuristic, but each returned inequality is globally valid.

### 4.4 Conditional directed return-connectivity cuts

Contract each virtual state `S_i` into its co-located locker `i`. If
`h_ij>0`, then `psi_ij>0` because every `rho_r` is strictly positive. A positive
arc in a nonnegative balanced flow must lie on a directed cycle, so there must
be a return path from `j` to `i`. For any `j`--`i` directed cut `U`,

```text
h_ij <= M^w_ij sum_(a in delta^+(U)) y_a.
```

The separator computes a directed minimum cut at the fractional `y` point.
This conditioning on endogenous `h_ij` is essential: imposing connectivity for
every external OD market would be invalid because all-courier fallback is
always allowed.

## 5. Separation policy

The default uses at most three root rounds and 250 cuts per round. Half of a
round is reserved for OD-VUB cuts, roughly one sixth for queue cuts, one fifth
for choice covers, and the remainder for directed return cuts. Signatures
prevent duplicate rows. A cut family producing zero cuts simply means the
current root point satisfies that family; it is not a correctness failure.

The full static inequality pool is available with `--valid-inequalities`, but
is disabled by default. On H30 it marginally improved the lower bound while
degrading the incumbent search. The default therefore follows the actual
ablation evidence: exact preprocessing, barrier, exact-conic starts, and a
limited dynamic cut pool.

## 6. Algorithm outline

```text
preprocess dominated routes and tighten flow bounds
build the screened OD-aware MISOCP
construct a feasible network seed with the exact conic model
run exact-conic local branching around its binary design
install network and all-courier MIP starts
solve root by conic barrier
for at most R root rounds:
    separate violated OD-VUB cuts
    separate queue perspective cuts
    separate choice-capacity covers
    separate conditional directed return min-cuts
    add at most K user cuts
continue solver-native branch-and-bound
```

## 7. Reproducible H34 benchmark

```bash
python main.py solve \
  --instance-dir data/instances/nanshan_od_h34 \
  --results-dir results/nanshan_branch_cut_20260821/H34_final_barrier_bnc_300s \
  --modes MX \
  --time-limit 300 \
  --mip-gap 0.01 \
  --threads 1 \
  --network-start-seconds 20 \
  --max-root-cut-rounds 3 \
  --max-cuts-per-round 250 \
  --quiet
```

With seed 42, this run terminated at the 300-second limit with objective
50,650.20 CNY/hour, bound 45,709.97, and gap 9.75%. It opened 3 bases, 32
lockers, and 4 swap stations, served 575.42 orders/hour, and saved 317.66
CNY/hour relative to all-courier delivery. Two root rounds added 309 user cuts:
187 locker OD-VUB, 63 home OD-VUB, and 59 queue-perspective cuts. This is a
feasible time-limit incumbent, not a 1%-optimality certificate.

