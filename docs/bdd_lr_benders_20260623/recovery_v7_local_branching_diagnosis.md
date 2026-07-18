# Recovery v7 local-branching diagnosis

This note records the H = 50 clean-matrix follow-up after recovery v6.  It is
not a final report.

## Clean H = 50 matrix result

Variant: `qfix_recovery_v6_matrix300`.

| Method | Validation objective | Bound | Gap | Wall time | Time-to-best |
|---|---:|---:|---:|---:|---:|
| Gurobi | 15017.9288 | 16798.2884 | 12.00% | 300.01 | 287.06 |
| LR | 14924.0251 | 66006.1250 | 342.28% | 172.38 | 163.51 |
| BD-only | 804.2173 | 16228.9423 | 1917.98% | 300.61 | 83.36 |
| LR+BD | 14978.6180 | 16226.2300 | 8.33% | 300.08 | 266.48 |

Audit for this variant passes:

- four methods present;
- decomposition Masters are MILP;
- candidate/local/global linearized MILPs are cone-free where executed;
- source audit confirms isolated package, aggregate Master proxy, Pareto,
  BDD, local branching, exact no-good and LR profiler hooks.

This invalidates the weaker statement that H = 50 is solved by comparing
against the older `qfix_gurobi_scale300` run.  The clean 300-second Gurobi
rerun found a stronger incumbent, so H = 50 remains a boundary failure under
the clean matrix.

## Distance diagnosis

Comparing the clean Gurobi design against the LR+BD v6 incumbent:

| Component | Hamming/Gamma distance |
|---|---:|
| x | 4 |
| y | 186 |
| z | 2 |
| total binary distance | 192 |
| Gamma difference | 27 |

The all-variable local branching neighborhood reached radius 96 before the
budget ended; the clean Gurobi design is outside that radius.  The main
obstacle is therefore not Python overhead, but the neighborhood radius/budget
interaction and the instability of LR recovery candidates.

## v7 experiments

| Variant | Change | Validation objective | Diagnosis |
|---|---|---:|---|
| `qfix_recovery_v7_facility_lb300` | local-branching distance over x/z only; y free | 14873.5455 | Too broad.  The cone-free MILP returns poor facility combinations within 5s. |
| `qfix_recovery_v7_lbfocus300` | all-variable distance; less global-bound time; fewer LR candidates; more local-branching calls | 14959.9949 | Local branching steadily improves a weak start but still remains below Gurobi. |
| `qfix_recovery_v8_multicenter300` | top-4 initial centers; all-variable local branching; reduced bound time | 14954.7409 | Multi-center does not solve H = 50; it reaches radius 192 too late to return a useful candidate. |
| `qfix_recovery_v9_layered_lb300` | separate facility and arc distance budgets | 14922.0764 | Layered distance is too loose for the cone-free MILP proxy; exact SOCP rejects the proxy-favored designs. |
| `qfix_recovery_v10_targeted_layered300` | larger arc budget, no global bound, targeted layered search | 14928.9405 | More incumbent time does not help when the proxy neighborhood is misranked; bound falls back to the LR bound. |
| `qfix_recovery_v11_diverse_multicenter300` | source-diverse multi-center local branching | 14848.7747 | Source-diverse center selection can skip the strongest exact LR center; not a final improvement. |
| `qfix_recovery_v13_h50_wide_lb300` | all-variable wide radius 192 with 25s neighborhood calls | 14955.1031 | Wide radius improves 14924 to 14955, but the relaxed MILP is too broad and still misses the Gurobi incumbent. |
| `qfix_recovery_v14_h50_pool_lb300` | local branching solution pool, 8 candidates per neighborhood | 14914.4042 | Pool validation increases exact SOCP work but does not recover a better exact design. |
| `qfix_recovery_v15_h50_restricted_pool300` | restricted-network solution pool, 4 candidates per refinement | 14883.3783 | Restricted pool destabilizes the LR center in this setting; not retained as a positive result. |
| `qfix_recovery_v16_h50_benders_envelope300` | local branching objective replaced by recent Benders-cut envelope with improvement cutoff | 14822.0594 | The Benders envelope is too weak as a stand-alone neighborhood objective: proxy values reached 34k--48k, but exact SOCP objectives were large negative values. |
| `qfix_recovery_v18_h50_benders_trust300` | 25% Benders-envelope blend, solution pool 4, and at most one deletion from the LR center | 14924.0251 | Deletion trust region prevents catastrophic candidates, but becomes too conservative; all five local-branching calls were exact negative improvements. |
| `qfix_recovery_v19_h50_service_preserve300` | protect top-50 active high-demand OD routes inside local branching | 14953.5857 | Service preservation restores useful local branching: 6 accepted moves, but still below Gurobi and below v6 LR+BD. |
| `qfix_recovery_v20_h50_service_preserve_focus300` | no global bound/BDD root, top-4 centers, top-50 route protection, incumbent-focused local branching | 14973.9510 | Best service-preserving run so far; improves from 14922.0765 through 8 accepted local moves, but still misses Gurobi by 43.98. |
| `qfix_recovery_v21_h50_service_preserve_deep300` | single-center deeper service-preserving local branching | 14933.6593 | Failed because the LR restricted-network start was weak at 14801.5776; local branching improved it but could not recover enough. |
| `qfix_recovery_v22_h50_service_preserve_multicenter_deep300` | multi-center service-preserving search with larger per-center budget | 14902.7056 | Also dominated by weak LR centers; extra local-search budget is not useful without a stable high-quality center. |

For `qfix_recovery_v7_lbfocus300`, local branching improved from an initial
14782.0321 to 14959.9989 through 25 calls, with accepted improvements at
multiple radii.  This confirms local branching is useful, but the initial LR
candidate quality and radius path remain insufficient.

For `qfix_recovery_v8_multicenter300`, the initial LR incumbent was stronger
again at 14924.0250, but multi-center local branching peaked at 14954.7412.
The last call reached all-variable radius 192 and Gamma radius 240, matching
the observed distance scale to the clean Gurobi design, but only 2.9 seconds
remained and no valid candidate was returned.  The issue is therefore search
budget placement, not merely the maximum radius parameter.

## Current conclusion

- H = 50 is not yet solved under a clean 300-second matrix.
- H = 60/70/80/100 now have clean 300-second matrix evidence under
  `qfix_recovery_v12_matrix300`: LR or LR+BD beats Gurobi on all four scales.
- C++ remains unjustified: Python OD/graph/ranking time is negligible relative
  to restricted-network MILP and exact SOCP calls.
- The H = 50 failure is not a missing-radius issue alone.  The broader
  experiments show a proxy-misranking problem in cone-free local branching and
  some instability in the LR restricted-network center.
- Pure Benders-envelope ranking is not enough.  The current cut pool is valid
  for bounding, but too weak to rank local incumbent candidates without stronger
  design-retention or service-coverage structure.
- Protecting high-demand OD routes is a positive direction.  It turns local
  branching from mostly negative exact moves into accepted improvement chains.
  However, H = 50 remains sensitive to the LR restricted-network center; weak
  centers lose more than local branching can recover within 300 seconds.

## Next algorithm work

The next changes should focus on recovery and local branching quality:

1. Stabilize restricted-network recovery incumbents.  H = 50 runs show that
   the same seed family can produce either a useful Gamma ≈ 378 candidate or a
   poor Gamma ≈ 154 candidate under the short MILP time limit.
2. Add multi-center local branching from the top LR candidates, not only the
   single best initial design.  The first implementation was not enough by
   itself, but the logs show it can reach the required radius scale.
3. Add a two-level local branching distance:
   - a small x/z facility radius;
   - a separate, larger y arc radius.
   Facility-only is too loose; all-variable is too restrictive.
4. Consider reducing bound-only work only after preserving a competitive LR
   start; otherwise the saved time is spent recovering from a weaker center.
5. For the next H = 50 attempt, do not maximize the Benders envelope by itself.
   Use it only as a secondary score after enforcing service-preserving
   neighborhood structure: retain high-volume OD coverage, protect critical
   return/swap arcs, and repair capacity/Gamma around the LR center.
6. Stabilize the restricted-network LR center before spending more time on
   local branching.  The same H = 50 configuration produced starts between
   roughly 14801 and 14924, and this variance dominates the final incumbent.
