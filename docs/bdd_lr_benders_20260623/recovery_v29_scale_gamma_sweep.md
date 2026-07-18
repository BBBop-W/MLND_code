# v28/v29 restricted-network + exact Gamma sweep scale diagnostic

This note records the first scale transfer after the H = 50 recovery fix.  It
is still a diagnostic matrix, not the final multi-seed 900-second experiment.
All reported incumbents are independently fixed-design SOCP validated.

## Configuration

H = 50 uses:

```text
variant = qfix_recovery_v28_h50_restricted24_gamma300
--time-limit 300
--lr-time-limit 165
--lr-iterations 50
--lr-max-designs 50
--lr-restricted-time-limit 24
--bdd-root-rounds 0
--no-global-linearized-bound
--local-branching
--local-branching-start-pool-size 4
--local-branching-max-calls-per-center 8
--local-branching-protect-top-pairs 50
--local-branching-solution-pool-size 4
--local-branching-time-limit 8
--local-branching-radius 24
--local-branching-max-radius 192
--local-branching-gamma-radius 32
--local-branching-max-gamma-radius 240
--gamma-sweep-max-calls 12
--gamma-sweep-radius 48
--gamma-sweep-step 8
--gamma-sweep-time-reserve 10
```

H = 60, 70, 80 and 100 use the same configuration with:

```text
variant = qfix_recovery_v29_scale_restricted30_gamma300
--lr-restricted-time-limit 30
```

The reproducible runner is:

```text
python -m src.experiments.bdd_lr_benders_20260623.run_v29_scale_recovery
```

It accepts `--scales`, `--methods`, `--time-limit`, `--threads` and standard
benchmark passthrough arguments.  The default method is `lr_bd`; use
`--methods gurobi,lr,bd,lr_bd` only when running the full formal comparison,
because that multiplies the total runtime.

The restricted-network and local-branching MILPs remain cone-free, and the
Benders Master remains a pure MILP.

## Validated 300-second comparison

The Gurobi baseline is the existing qfix 300-second recovery/matrix baseline
for each scale.  For H = 50 that is `qfix_recovery_v6_matrix300`; for H = 60,
70, 80 and 100 that is `qfix_recovery_v12_matrix300`.

| H | Gurobi 300s validated obj. | Previous LR+BD validated obj. | LR-only validated obj. | v28/v29 LR+BD validated obj. | v28/v29 - Gurobi |
|---:|---:|---:|---:|---:|---:|
| 50 | 15017.9288 | 14978.6180 | 14924.0251 | 15036.4749 | +18.5461 |
| 60 | 15091.4870 | 15167.3049 | 15210.2482 | 15268.3873 | +176.9003 |
| 70 | 14816.0466 | 15058.1588 | 15056.3603 | 15195.6340 | +379.5874 |
| 80 | 13070.8996 | 13543.2568 | 13543.2568 | 13555.6340 | +484.7344 |
| 100 | 9010.0023 | 9149.0046 | 9149.0046 | 9559.6584 | +549.6561 |

The H = 50 result crosses the previous failure boundary.  The larger scales
were already favorable for LR/LR+BD under v12, and v29 improves every tested
scale further under the same 300-second budget.

## Same-variant four-method checks

Same-variant four-method runs have been completed for H = 50, 60 and 100.
H = 70 and H = 80 still need the Gurobi/LR/BD columns under the v29 variant;
their LR+BD rows are already available.

| H | Method | Validated obj. | Bound | Relative gap | Wall time | Time-to-best |
|---:|---|---:|---:|---:|---:|---:|
| 50 | Gurobi | 14991.2166 | 16808.6807 | 0.1238 | 300.0 | 208.8 |
| 50 | LR-only | 15009.5817 | 66006.1250 | 3.3976 | 254.7 | 252.9 |
| 50 | BD-only | 1432.8710 | 61290.0884 | 41.7743 | 300.2 | 42.7 |
| 50 | LR+BD | 15036.4749 | 66006.1250 | 3.3897 | 300.2 | 296.6 |
| 60 | Gurobi | 15091.4870 | 17909.3972 | 0.1868 | 300.0 | 297.8 |
| 60 | LR-only | 15180.7367 | 66528.6250 | 3.3824 | 260.9 | 257.5 |
| 60 | BD-only | 1142.9154 | 63476.7550 | 54.5393 | 300.1 | 40.9 |
| 60 | LR+BD | 15268.3873 | 66528.6250 | 3.3573 | 300.2 | 294.0 |
| 100 | Gurobi | 9010.0023 | 17448.3247 | 0.9425 | 300.0 | 255.4 |
| 100 | LR-only | 9149.0046 | 62808.7500 | 5.8651 | 295.8 | 258.4 |
| 100 | BD-only | 650.2937 | 61287.3513 | 93.2456 | 300.0 | 71.2 |
| 100 | LR+BD | 9559.6584 | 62808.7500 | 5.5702 | 300.2 | 298.3 |

The same-variant LR+BD improvements over Gurobi are +45.2583 at H = 50,
+176.9003 at H = 60 and +549.6561 at H = 100.  LR+BD also beats LR-only by
+26.8932, +87.6506 and +410.6537 on the same scales.  BD-only remains a poor
incumbent generator without LR warm start.

## Recovery diagnostics

| H | wall time | time-to-best | Gamma sweep evaluations | Gamma-improving evaluations | Master qconstrs |
|---:|---:|---:|---:|---:|---:|
| 50 | 300.2 | 296.6 | 9 | 3 | 0 |
| 60 | 300.2 | 294.0 | 8 | 2 | 0 |
| 70 | 300.5 | 299.5 | 8 | 2 | 0 |
| 80 | 300.9 | 295.1 | 7 | 1 | 0 |
| 100 | 300.2 | 298.3 | 7 | 1 | 0 |

The improvement mechanism is consistent:

- longer restricted-network refinement stabilizes the LR-derived network
  center;
- service-preserving local branching improves the network without putting
  operational variables into the Master;
- exact fixed-network Gamma sweep repairs fleet sizing and contributes the
  final incumbent on every tested scale.

## Timing and C++ decision

The LR profiler does not justify C++ migration in these runs.  The non-solver
fraction of LR time is below 0.3% on H = 60, 70, 80 and 100.  The dominant
phases are Gurobi restricted-network MILPs and exact SOCP validation, not
Python OD/graph processing.

## Remaining work before final claims

This matrix is single-seed and 300 seconds.  It supports continuing with the
v29 recovery design, but it does not finish the goal.  The next formal step is
to run the full four-method comparison under a common budget and then repeat
the best configuration over additional seeds or period/demand slices.
