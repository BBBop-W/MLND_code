# Recovery v12 clean 300-second matrix

This note records the current clean 300-second evidence for the isolated
2026-06-23 BDD/LR/Benders implementation.  It is not the final 900-second
matrix.

All decomposition runs keep a MILP Master.  The audit for
`qfix_recovery_v12_matrix300` passes with `master_qconstrs = 0` for BD and
LR+BD, and cone-free checks for the executed candidate/global/local MILPs.

## Incumbent comparison

Validation objectives are used for comparison when available.

| H | Method | Variant | Validation obj. | Bound | Gap | Wall time | Time-to-best |
|---:|---|---|---:|---:|---:|---:|---:|
| 50 | Gurobi | `qfix_recovery_v6_matrix300` | 15017.9288 | 16798.2884 | 12.00% | 300.01 | 287.06 |
| 50 | LR | `qfix_recovery_v6_matrix300` | 14924.0251 | 66006.1250 | 342.28% | 172.38 | 163.51 |
| 50 | BD | `qfix_recovery_v6_matrix300` | 804.2173 | 16228.9423 | 1917.98% | 300.61 | 83.36 |
| 50 | LR+BD | `qfix_recovery_v6_matrix300` | 14978.6180 | 16226.2300 | 8.33% | 300.08 | 266.48 |
| 60 | Gurobi | `qfix_recovery_v12_matrix300` | 15091.4870 | 17910.3090 | 18.69% | 300.01 | 299.99 |
| 60 | LR | `qfix_recovery_v12_matrix300` | 15210.2482 | 66528.6250 | 337.39% | 257.60 | 247.78 |
| 60 | BD | `qfix_recovery_v12_matrix300` | 1271.2026 | 16900.0639 | 1229.45% | 300.54 | 106.83 |
| 60 | LR+BD | `qfix_recovery_v12_matrix300` | 15167.3049 | 16900.0639 | 11.42% | 300.38 | 217.86 |
| 70 | Gurobi | `qfix_recovery_v12_matrix300` | 14816.0466 | 19807.0233 | 33.69% | 300.03 | 141.61 |
| 70 | LR | `qfix_recovery_v12_matrix300` | 15056.3603 | 65875.7500 | 337.53% | 260.14 | 248.47 |
| 70 | BD | `qfix_recovery_v12_matrix300` | 1078.7952 | 17360.5448 | 1509.25% | 301.15 | 144.78 |
| 70 | LR+BD | `qfix_recovery_v12_matrix300` | 15058.1588 | 17360.5448 | 15.29% | 300.30 | 219.36 |
| 80 | Gurobi | `qfix_recovery_v12_matrix300` | 13070.8996 | 19408.7769 | 48.49% | 300.03 | 205.56 |
| 80 | LR | `qfix_recovery_v12_matrix300` | 13543.2568 | 65050.5000 | 380.32% | 267.55 | 251.32 |
| 80 | BD | `qfix_recovery_v12_matrix300` | 1084.9403 | 16027.9958 | 1377.32% | 300.19 | 126.34 |
| 80 | LR+BD | `qfix_recovery_v12_matrix300` | 13543.2568 | 16027.9958 | 18.35% | 302.23 | 220.89 |
| 100 | Gurobi | `qfix_recovery_v12_matrix300` | 9010.0023 | 17448.3247 | 94.25% | 300.03 | 251.00 |
| 100 | LR | `qfix_recovery_v12_matrix300` | 9149.0046 | 62808.7500 | 586.51% | 285.85 | 258.30 |
| 100 | BD | `qfix_recovery_v12_matrix300` | 861.8546 | 13040.4971 | 1413.07% | 300.72 | 202.12 |
| 100 | LR+BD | `qfix_recovery_v12_matrix300` | 9149.0046 | 13040.4971 | 42.53% | 301.25 | 228.19 |

## Current interpretation

- H = 60, 70, 80 and 100 support the intended large-scale claim under a shared
  300-second budget: LR or LR+BD produces a better incumbent than Gurobi.
- H = 50 remains a boundary failure: clean Gurobi is 15017.9288 while LR+BD is
  14978.6180.
- BD alone is not a competitive incumbent method at these scales.  Its current
  role is bound/cut strengthening; without LR recovery it usually returns very
  poor feasible designs.
- LR+BD gives much tighter bounds than LR-only because it uses the global
  linearized/Benders bound, but it often inherits LR's incumbent.

## LR timing and C++ decision

The current bottleneck is solver time, not Python overhead.

| H | Method | LR wall | Non-solver fraction | C++ flag | LR candidates |
|---:|---|---:|---:|---|---:|
| 50 | LR | 161.94 | 0.11% | false | 220 |
| 50 | LR+BD | 161.91 | 0.11% | false | 220 |
| 60 | LR | 240.47 | 0.08% | false | 160 |
| 60 | LR+BD | 210.48 | 0.10% | false | 150 |
| 70 | LR | 240.71 | 0.13% | false | 160 |
| 70 | LR+BD | 210.69 | 0.14% | false | 150 |
| 80 | LR | 240.87 | 0.15% | false | 159 |
| 80 | LR+BD | 210.84 | 0.17% | false | 149 |
| 100 | LR | 241.42 | 0.22% | false | 142 |
| 100 | LR+BD | 211.42 | 0.25% | false | 139 |

Current evidence does not justify a C++ rewrite.  Restricted-network MILPs and
exact SOCP validation dominate; Python graph/ranking/update phases are below
1% of LR time.

## H = 50 follow-ups

The following H = 50 variants diagnose the incumbent-recovery bottleneck after
the clean matrix.  The clean Gurobi validation benchmark is 15017.9288.

| Variant | Main change | Validation obj. |
|---|---|---:|
| `qfix_recovery_v7_facility_lb300` | facility-only local branching distance | 14873.5455 |
| `qfix_recovery_v7_lbfocus300` | more local-branching time, less bound time | 14959.9949 |
| `qfix_recovery_v8_multicenter300` | top-4 local-branching centers | 14954.7409 |
| `qfix_recovery_v9_layered_lb300` | layered facility/arc distance | 14922.0764 |
| `qfix_recovery_v10_targeted_layered300` | targeted layered search | 14928.9405 |
| `qfix_recovery_v11_diverse_multicenter300` | source-diverse multi-center search | 14848.7747 |
| `qfix_recovery_v13_h50_wide_lb300` | wide all-variable radius | 14955.1031 |
| `qfix_recovery_v14_h50_pool_lb300` | local-branching solution pool | 14914.4042 |
| `qfix_recovery_v15_h50_restricted_pool300` | restricted-network solution pool | 14883.3783 |
| `qfix_recovery_v16_h50_benders_envelope300` | Benders-envelope local-branching objective with cutoff | 14822.0594 |
| `qfix_recovery_v18_h50_benders_trust300` | Benders-envelope blend plus deletion trust region | 14924.0251 |
| `qfix_recovery_v19_h50_service_preserve300` | top-50 service-preserving local branching | 14953.5857 |
| `qfix_recovery_v20_h50_service_preserve_focus300` | incumbent-focused service-preserving multi-center local branching | 14973.9510 |
| `qfix_recovery_v21_h50_service_preserve_deep300` | deeper single-center service-preserving local branching | 14933.6593 |
| `qfix_recovery_v22_h50_service_preserve_multicenter_deep300` | deeper multi-center service-preserving local branching | 14902.7056 |
| `qfix_recovery_v23_h50_anchor_service300` | top-demand anchor seed plus service-preserving local branching | 14950.0680 |
| `qfix_recovery_v24_h50_gamma_sweep300` | exact fixed-network Gamma sweep with excessive reserve | 14920.7882 |
| `qfix_recovery_v25_h50_gamma_sweep_late300` | late exact Gamma sweep after deeper service-preserving local branching | 15005.9046 |
| `qfix_recovery_v26_h50_gamma_sweep_upfirst300` | upward-first Gamma sweep; weak LR center in this run | 14946.4515 |
| `qfix_recovery_v27_h50_gamma_sweep_pool300` | restricted-network solution pool plus upward-first Gamma sweep | 14937.8257 |
| `qfix_recovery_v28_h50_restricted24_gamma300` | 24-second restricted-network refinement plus upward-first exact Gamma sweep | 15036.4749 |

The useful direction was not simply larger neighborhoods or more pool
solutions.  The evidence points to two separable recovery issues:

- the cone-free/local-branching network can be good enough, but its fleet
  \(\Gamma\) is often too low; exact fixed-design Gamma sweep corrected this
  without calling the original MISOCP;
- the LR restricted-network center is unstable under short refinement limits.
  Raising the H = 50 restricted-network refinement limit to 24 seconds gave a
  stronger local-branching center, and upward-first Gamma sweep then produced
  a validated incumbent above the clean Gurobi benchmark.

Benders-envelope ranking alone remains insufficient for incumbent ranking.  It
is valid for bounding, but should be used only as secondary guidance unless
paired with service-preserving neighborhood constraints and exact SOCP
validation.
