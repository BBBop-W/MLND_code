# Recovery v5 progress notes

Superseded for the H = 50 boundary case by
`docs/bdd_lr_benders_20260623/recovery_v6_progress.md`.  This file is kept as
the v5 evidence snapshot.

This note records the current post-qfix large-scale recovery work.  It is not
the final goal completion report; it is the current evidence base for the next
round of formal experiments.

## Implementation changes

- Replaced the complete OD-level `service_possible` / `route_ready` Master
  proxy family with aggregate pure-MILP proxies:
  - one continuous service proxy per origin hub;
  - a limited set of top-demand logical pair proxies, capped by
    `min(200, 2 |H|)`;
  - linear origin, destination and aggregate-arc upper bounds.
- The Master remains a MILP and runtime summaries report:
  - `master_qconstrs`;
  - `master_proxy_service_variables`;
  - `master_proxy_top_pair_variables`;
  - `master_proxy_total_variables`.
- Added LR recovery v5:
  - hub coverage seeds from LR demand/multiplier scores;
  - fixed-hub full-network cone-free MILP recovery;
  - adaptive hub coverage order:
    - H < 60: sparse-to-dense;
    - 60 <= H < 80: medium/dense first;
    - H >= 80: 95%, 90%, 85%, 75%, 100% coverage first;
  - validation reserve in LR-only so generation cannot consume all wall time.

All candidate MILPs still remove every quadratic constraint and verify
`NumQConstrs == 0`.  Every incumbent is validated in a fresh fixed-design
SOCP.

## Current 300-second large-scale comparison

The objective column uses the independent fixed-design SOCP validation value
when available.

| H | Method | Variant | Objective | Bound | Gap | Wall time |
|---:|---|---|---:|---:|---:|---:|
| 50 | Gurobi | qfix_gurobi_scale300 | 14948.5442 | 16835.5011 | 12.69% | 300.04 |
| 50 | LR | qfix_lr_scale_recovery_v5_adaptive_hubs300 | 14922.0929 | 66006.1250 | 342.34% | 159.57 |
| 50 | LR+BD | qfix_lrbd_recovery_v5_proxy300 | 14947.3946 | 16223.9820 | 8.54% | 300.08 |
| 60 | Gurobi | qfix_gurobi_scale300 | 15054.5522 | 18047.8408 | 19.88% | 300.02 |
| 60 | LR | qfix_lr_scale_recovery_v4_dense_hubs300 | 15169.9184 | 66528.6250 | 338.56% | 248.43 |
| 60 | LR+BD | qfix_lrbd_recovery_v5_proxy300 | 15206.1907 | 16900.0639 | 11.14% | 300.40 |
| 70 | Gurobi | qfix_gurobi_scale300 | 14816.0466 | 19836.0355 | 33.88% | 300.02 |
| 70 | LR | qfix_lr_scale_recovery_v4_dense_hubs300 | 15102.6008 | 65875.7500 | 336.19% | 250.72 |
| 70 | LR+BD | qfix_lrbd_recovery_v5_proxy300 | 15102.6008 | 17360.5448 | 14.95% | 300.58 |
| 80 | Gurobi | qfix_gurobi_scale300 | 13070.8996 | 19408.7769 | 48.49% | 300.03 |
| 80 | LR | qfix_lr_scale_recovery_v4_dense_hubs300 | 13537.3649 | 65050.5000 | 380.53% | 201.73 |
| 80 | LR+BD | qfix_lrbd_recovery_v5_proxy300 | 13540.5937 | 16027.9958 | 18.37% | 300.02 |
| 100 | Gurobi | qfix_gurobi_scale300 | 9006.2580 | 17474.0770 | 94.54% | 300.04 |
| 100 | LR | qfix_lr_scale_recovery_v4_dense_hubs300 | 9147.9044 | 62808.7500 | 586.59% | 214.01 |
| 100 | LR+BD | qfix_lrbd_recovery_v5_proxy300 | 9127.5465 | 13040.4971 | 42.87% | 303.35 |

Current evidence:

- H = 60, 70, 80, 100: LR or LR+BD beats the 300-second Gurobi incumbent.
- H = 60 and H = 80: LR+BD improves the LR incumbent and greatly tightens the
  bound.
- H = 70 and H = 100: LR-only gives the best incumbent, while LR+BD mainly
  tightens the bound.
- H = 50 remains a boundary failure under equal budget:
  - LR+BD 300 seconds is slightly below Gurobi 300-second validation value;
  - LR+BD 600 seconds reaches 14976.9588, but Gurobi 600 seconds reaches
    15084.2475.

## BD-only representative evidence

| H | Method | Objective | Bound | Gap | Wall time |
|---:|---|---:|---:|---:|---:|
| 60 | BD-only | 1266.2901 | 16900.0639 | 1234.61% | 300.22 |
| 80 | BD-only | 1084.9403 | 16027.9958 | 1377.32% | 300.67 |

BD-only provides useful upper bounds but does not find competitive large-scale
incumbents without LR recovery.

## LR timing and C++ decision

For H = 50--100, non-solver LR time remains below 1% of LR wall time.  The
dominant phase is the Gurobi restricted/fixed-hub network MILP plus exact
SOCP validation.  The C++ trigger is not met.  The next performance work
should improve recovery MILP structure and candidate selection, not Python
rewrites.

## Remaining work before goal completion

- Run a formal shared-budget matrix for the final chosen variants, preferably
  900 seconds, over H = 50, 60, 70, 80, 100.
- Add more instances or demand periods if available, rather than relying on a
  single h12 series.
- Complete component ablations for Master proxy and recovery variants on at
  least H = 60 and H = 80.
- Decide how to report H = 50: current evidence marks it as a boundary
  failure, not a success case.
