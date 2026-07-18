# Recovery v6 progress notes

This is the current post-qfix progress note for the independent 20260623
BDD/LR/Benders framework.  It is not a final goal-completion report because
the formal four-method 900-second matrix and multi-instance/period tests are
still incomplete.

## What changed after v5

- LR recovery now builds a diversified fixed-hub seed ensemble:
  - LR weighted hub ranking;
  - greedy OD coverage ranking;
  - raw demand-degree ranking.
- H = 40--59 no longer uses only one sparse-to-dense schedule.  The recovery
  seed pool keeps medium/dense/sparse hub coverage variants so H = 50 does
  not lose the useful boundary candidates.
- Candidate selection now reserves validation slots by source and ranks:
  1. fixed-hub cone-free network MILP candidates,
  2. restricted-network cone-free MILP candidates,
  3. direct LR recovery candidates.
- Gamma variants are penalized by distance from the recovery center.  This
  avoids spending many exact SOCP validations on artificially low-Gamma
  candidates.
- Initial-candidate logs now include candidate source, seed policy, seed
  index, hub fraction, open x/y/z counts, exact-SOCP status, failure reason
  and candidate evaluation wall time.
- Exact fixed-design SOCP validation is now counted as solver-dominated time
  in `timing.csv`; otherwise the C++ decision gate incorrectly treats SOCP
  validation as Python/non-solver overhead.

The Master boundary is unchanged: the decomposition Master remains a pure
MILP.  In the H = 50 LR+BD v6 run, `master_qconstrs = 0`, with 50 aggregate
origin-service proxies and 100 top-pair proxies.

## New H = 50 boundary evidence

All objective values below use independent fixed-design SOCP validation when
available.

| H | Method | Variant | Objective | Bound | Gap | Wall time | Time-to-best |
|---:|---|---|---:|---:|---:|---:|---:|
| 50 | Gurobi | qfix_gurobi_scale300 | 14948.5442 | 16835.5011 | 12.69% | 300.04 | 242.55 |
| 50 | LR | qfix_recovery_v6_lr300 | 14924.0251 | 66006.1250 | 342.28% | 172.11 | 162.97 |
| 50 | LR+BD | qfix_recovery_v6_lrbd300 | 14978.6180 | 16226.2300 | 8.33% | 300.13 | 265.70 |
| 50 | Gurobi | qfix_gurobi_scale600 | 15084.2475 | 16762.4058 | 11.13% | 600.00 | 588.34 |

Interpretation:

- H = 50 is no longer a 300-second boundary failure for LR+BD.  The v6 LR+BD
  incumbent beats the 300-second Gurobi incumbent by about 30.07 objective
  units after independent SOCP validation.
- The gain is not from LR alone.  LR v6 gives 14924.0251; local branching
  inside LR+BD improves it to 14978.6180.
- H = 50 is still not proven stronger than a 600-second Gurobi run.  The old
  600-second Gurobi baseline remains higher at 15084.2475, so the next formal
  comparison must use a shared budget.

## Local branching evidence for H = 50 v6

The H = 50 LR+BD v6 run accepted two local-branching improvements:

| Call | Center objective | Candidate objective | Improvement |
|---:|---:|---:|---:|
| 3 | 14924.0250 | 14933.7824 | 9.7574 |
| 6 | 14933.7824 | 14978.6195 | 44.8372 |

This confirms that local branching is doing incumbent improvement, not merely
adding overhead.

## Timing and C++ decision

For H = 50 v6 LR+BD:

- restricted/fixed-hub network MILP: 156.06 s;
- exact SOCP recourse/validation: 36.96 s over 147 calls;
- all Python OD/graph/ranking phases combined: below 0.2 s.

The C++ trigger is not met.  The bottleneck is solver time, not Python
recovery code.

## Updated evidence status

- H = 50: LR+BD v6 beats Gurobi at 300 seconds, but not the existing Gurobi
  600-second baseline.
- H = 60, 70, 80, 100: existing v5/v4 300-second results already show LR or
  LR+BD beating 300-second Gurobi.
- BD-only remains poor for incumbents at large scale, although its Master
  bounds are useful.

## Remaining work before completion

- Run a clean shared-budget four-method matrix for the selected final
  variants over H = 50, 60, 70, 80, 100.  The formal matrix should preferably
  use 900 seconds, or explicitly state a 300-second exploratory budget.
- Add at least one more period or demand slice if available.
- Re-run BD-only for all large H values under the same selected budget.
- Complete ablations separating:
  - Master proxy bound effect;
  - Pareto/Magnanti-Wong conic cuts;
  - strengthened BDD cuts;
  - Lagrangian BDD cuts;
  - local branching;
  - LR warm start and v6 recovery.
- Do not mark the goal complete until the final matrix proves or refutes the
  large-scale claim under the same budget and solver settings.
