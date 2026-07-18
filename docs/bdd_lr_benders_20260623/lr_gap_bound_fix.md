# LR gap bound correction, 2026-06-25

This note records the LR gap diagnosis and the first corrected tests.

## Problem

The LR-only method previously reported the raw OD-activation Lagrangian dual
bound as its final bound.  On small instances this produced extremely large
gaps:

- H = 5 raw LR bound: 20681.0;
- H = 10 raw LR bound: 44399.875.

This was not caused by subgradient step size.  Step-size sweeps kept the best
dual near the total service revenue.  The issue is that the OD-activation LR
relaxation deliberately removes operating flow, fleet, circulation, battery,
and queueing restrictions.  Therefore the raw LR dual is a valid but very weak
diagnostic bound, not a useful reported optimality bound.

The OD service symbol should not be written as `q_ij`, because the original
model already uses queue-cone helper variables `q_v1`, `q_v2`, and `q_t`.
The LR implementation uses `selected[(i,j)]` as an internal OD service
selection quantity.  The original operational service-flow variable is
`psi[i,j]`, and `psi` is still forbidden in the pure MILP Master.

## Correction

LR-only now reports

```text
min(raw_lr_bound, global_linearized_bound)
```

when the global bound is available.  The raw LR dual remains in
`raw_lr_bound`.

The global bound is obtained from the full-network model after:

1. materializing and removing all quadratic queue constraints;
2. verifying `NumQConstrs == 0`;
3. adding globally valid queue supporting-line lower bounds;
4. using the maximization `ObjBound` only.

The relaxation incumbent is not used as a primal solution.  This is a valid
upper bound because the cone-free supporting-line model contains the original
feasible set.

## Smoke results

All objectives below were independently fixed-design SOCP validated.

| H | Method | Validated obj | Reported bound | Gap | Raw LR bound | Linearized bound | Wall time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | LR | 539.129628 | 568.219019 | 5.3956% | 20681.000000 | 568.219019 | 0.23 s |
| 10 | LR | 2629.589936 | 2754.151289 | 4.7369% | 44399.875000 | 2754.151289 | 23.69 s |
| 50 | LR | 15009.581715 | 16243.222332 | 8.2190% | 66006.125000 | 16243.222332 | 284.34 s |
| 50 | LR+BD | 15024.531043 | 16228.942274 | 8.0163% | 66006.125000 | 16228.942274 | 300.18 s |

For comparison, the same H = 50 Gurobi 300-second run had validated objective
14991.216596, bound 16808.680690, and gap 12.3759%.  Therefore the corrected
LR and LR+BD reported gaps are now below Gurobi on this H = 50 diagnostic,
although the older no-global-bound LR+BD run still had the better incumbent
15036.474877.

## Interpretation

The abnormal LR gap was primarily a bound-reporting issue: the raw LR dual is
too weak to be the final reported upper bound.  It should remain available for
algorithm diagnosis, but gap comparisons should use the best valid upper bound
available under the method budget.

## Literature-to-implementation mapping for BD upper bounds

Rahmaniani et al. (2020) motivate Benders dual decomposition by introducing
local copies of master variables in the subproblem and pricing those copies in
the objective.  This is implemented in `bdd.py` as strengthened and
regularized Lagrangian BDD cuts.  Those cuts remain valid for the maximization
recourse because the cut intercept uses the local-copy MISOCP maximization
`ObjBound`, not a time-limited incumbent.

Magnanti-Wong style cut selection addresses subproblem degeneracy.  This is
implemented in `pareto.py` by selecting a reference-tight perturbed conic dual
that improves the support value at a strict Master core point.

For the current H = 50 gap issue, the most effective upper-bound improvement
was a relaxation cap rather than another local-copy cut: solve a cone-free
full-network relaxation and inject its `ObjBound` as the method upper bound.
This directly attacks the weak root bound in the same spirit as root
strengthening, while preserving the pure-MILP Master boundary and avoiding any
use of a relaxation incumbent.

The next open issue is not the small-scale LR gap.  It is balancing the time
spent on global upper-bound computation versus incumbent search on larger
instances and multi-seed experiments.
