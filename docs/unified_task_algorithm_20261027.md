# Unified fixed-task model and Benders implementation

## Model shared by the two operational modes

Use `--demand-formulation preprocessed` in both modes. The CLI option `--k 1`
allows single-order physical paths; `--k 2` also allows double-order physical
paths. It does not select the legacy assignment-dependent K2 reconstruction.
Both modes have the same departure assignment `alpha[r,l]`; the physical return
base remains a path attribute and participates in the drone-flow balance.

Preprocessing supplies the fixed rates `nu[r]` and `nu[r,s]` and defines

```text
H[r] = nu[r] + sum_s nu[r,s]
lambda[r] = nu[r] + sum_s nu[r,s] + sum_h nu[h,r]
v[r,l] = alpha[r,l] * rho[l]
single task flow from l, headed by r = nu[r] * v[r,l]
double task flow from l, headed by r followed by s = nu[r,s] * v[r,l]
```

For singles, `nu[r]=lambda[r]` and all pair rates are zero. For doubles, the
archived residual-rate reconstruction is retained: rank eligible ordered pairs
by positive route-cost saving, apply the residual Poisson opportunity formula,
and subtract the accepted rate from both raw OD residuals. An `AA` task consumes
two `A` orders. An admitted head retains common departure support for all its
positive task types. This remains a deterministic approximation with exact
mean-demand conservation; no equivalence to an online matching policy is claimed.

An order `r` can be delivered as the second order of a task headed by `h` without
its own assignment. Consequently, its service flow departing from `l` is

```text
rho[l] * (H[r] * alpha[r,l] + sum_h nu[h,r] * alpha[h,l]).
```

This expression is used by the independent native incumbent validator. The
analytical and resource cuts must use the corresponding occurrence-supply bound
before multiplying by `rho`; the former `lambda[r]*alpha[r,l]` bound is incorrect
for this model. Raw-demand conservation, physical path times, locker occupancy,
base balance, fleet constraints, and operational costs retain their existing
definitions.

The default remains aggregate task-flow allocation. `--first-leg-consistency`
is an optional restriction that shares a physical first-delivery route across
the single and double branches. It does not recompute their demand rates and is
not enabled in the standard comparison.

## Decomposition and path-domain boundary

The master contains the integer strategic variables and the operating-cost
epigraph variable. Fixing its design gives the SOCP recourse model with fixed
task-rate coefficients. Ordinary conic-dual Benders cuts apply globally to this
model: no assignment-pattern condition or conditional-demand cut is needed.
The same fixed-task formulation is used for integer recourse, continuous-root
calculation, fractional recourse, and the restricted primal heuristic.

Three methods are supported:

| Method | Operational columns |
| --- | --- |
| `full_path` | Complete physical catalogue in the joint MISOCP |
| `benders_full` | Complete physical catalogue in each SOCP recourse model |
| `benders_cg` | A subset in each recourse model, extended by exact label pricing |

The current reconstruction uses minimum costs and common departure support from
the **complete physical catalogue**. Therefore `preprocessed + benders_cg` first
enumerates and stores that catalogue, then reuses the existing exact label
pricing oracle to insert columns into the RMP. This reduces recourse-model
columns; it does not remove catalogue storage. An incomplete enumeration is a
construction failure, never a certified reduced path domain.

Results distinguish `catalogue_complete`, `path_storage`, `pricing_domain`,
catalogue path counts, and the columns actually used in the recourse model.
For preprocessed Benders, `generated_columns` counts the union reported by
completed recourse calls, including the continuous root; `generated_columns_scope`
states this convention. `reusable_columns` counts the current reuse pool.
The continuous root passes only its RMP column IDs into later recourse calls,
even though the physical catalogue is stored in full. An interrupted call that
does not return its column list is not included in the completed-call union.
Full physical records are retained in `path_flows.json`, including zero flows,
so reconstruction and operations can be checked independently after the run.
Input reading, enumeration, reconstruction, model construction, and optimization
share the requested wall-time budget; output finalization is reported separately.

## Reused strengthening and heuristic changes

The shared formulation accepts assignment incident rows, the joint `alpha*rho`
strengthening, queue perspective cones, and the tight physical airline bounds.
The existing continuous-root calculation, full-domain dual certification,
Pareto selection, column reuse, resource/capacity feedback, root projection,
incumbent cutoff, cleanup, and primal/neighborhood search remain available.
Ordinary recourse duals now come directly from CPLEX native getters; the legacy
dual-LP options are compatibility-only. Resource separation LPs and Pareto
auxiliary optimization remain distinct strengthening steps. See
[native dual extraction](native_dual_20261028.md).

With optional weak cuts, full-domain validity after complete pricing and
conservative correction is distinct from a tight recourse optimality
certificate. Only the latter sets `recourse_optimal`; a valid weak cut does not
certify an optimal recourse solution or exclude an entire design. See
[bounds improvements](bounds_improvement_20261028.md).

The initial design heuristic now evaluates the entire fixed collection of tasks
headed by `r`, allows different physical return bases, and checks positive fixed
branches. Warm-start reconstruction skips the removed legacy demand-lifting
variables. Native physical validation counts both head and incoming-second
occurrences. These changes affect candidate construction or validation, not the
operational feasible set.

Legacy `--assignment-support-cuts`, `--double-support-cuts`, and
`--joint-base-cost-cuts` are rejected for origin assignment: their existing
fixed-return/base-pair derivations are not transferred to the unified model.
The occurrence-aware cost, capacity, and resource adaptations have a separate
strengthening probe; their derivation is in
[unified_strengthening_validity.md](unified_strengthening_validity.md).

## Recommended comparison commands

Use identical input, seed, threads, task mode, physical bounds, and formulation
strengthening for solver and algorithm. Replace `INSTANCE`, `RUN`, and `K` with
the input directory, a new output directory, and `1` or `2`.

```sh
cpp/build-unified/mlnd_cpp --input INSTANCE --out RUN --k K \
  --demand-formulation preprocessed --method full_path \
  --seconds 3600 --gap 0.01 --threads 1 --seed 43 \
  --path-column-limit 0 --structural-strengthening --tight-airline-bounds
```

```sh
cpp/build-unified/mlnd_cpp --input INSTANCE --out RUN --k K \
  --demand-formulation preprocessed --method benders_cg \
  --seconds 3600 --gap 0.01 --threads 1 --seed 43 \
  --path-column-limit 0 --structural-strengthening --tight-airline-bounds \
  --topology-cuts --od-airline-bounds
```

`--path-column-limit 0` removes the catalogue-count cap, not the external memory
watchdog. Algorithm-only flags strengthen the Benders search; they do not change
the physical model. `--od-airline-bounds` is an integer-design strengthening of
the resource relaxation, so its root certificate must retain the implementation's
strengthened-root scope label. Do not substitute the old `conditional` or
`lifted` demand mode in this comparison.

## Validation completed for the core integration

`cpp/tests/preprocessed_demand_probe.cpp` passed its conservation, repeated-OD,
zero-window, missing-support, multi-return, and optional first-leg checks.
`cpp/tests/unified_benders_probe.cpp` additionally checks:

- Unified singles agree with the legacy origin-assignment K1 recourse.
- Exact CG and complete-path recourse agree for singles/doubles, with and without
  structural strengthening (largest observed absolute difference below `3e-9`).
- Four ordinary global cuts pass 36 comparisons against complete-path recourse
  at different assignments; the maximum observed excess is zero.
- The double-task fixture serves an incoming second order while its own OD has
  no departure assignment; the native heuristic accepts this incumbent.
- The single-order mode does not require binary compatibility data because it
  contains no double tasks.
- With a complete 32-path catalogue, the continuous root uses 12 service
  columns and the first integer recourse starts with those same 12 columns;
  it does not accidentally import all 32 paths. The completed-call union is
  18 service columns in this test.

The small fixture uses 16 single-mode physical paths and 32 double-mode physical
paths. Certified CG used 9 and 16 service columns respectively. These are
implementation checks, not performance evidence for the planned real-instance
batch. Reproduce with:

```sh
make -C cpp -j2 BUILD_DIR=build-unified all preprocessed-demand-probe build-unified/mlnd_unified_benders_probe
cpp/build-unified/mlnd_preprocessed_demand_probe
cpp/build-unified/mlnd_unified_benders_probe
```

## Native reference precision

The independent Python check recomputes physical route costs. At exactly zero
proxy saving, C++ and Python summation can differ in the final floating-point
bit. The reference retains the recorded eligibility decision only within
`1e-12 * (1 + single_cost[r] + single_cost[s] + double_cost[r,s])` of zero.
Every clearly positive compatible pair remains mandatory; clearly negative,
incompatible, unsupported, duplicate, and out-of-range pairs are rejected.
The reconstruction rates and conservation checks are still recomputed, and the
optimizer's reconstruction rule is unchanged. This boundary was exercised by
the Nanshan L2 real-instance smoke test.

## Integrated acceptance before the long batch

The final binary passed eight 60-second end-to-end runs: TSL L1/10H/25OD and
Nanshan L2/20H/50OD, each with single/double tasks and full-path/Benders-CG.
All eight saved results passed independent Python physical, demand, cost and
operation-table validation. These are integration checks, not the requested
one-hour performance comparison. Evidence is in
`outputs/unified_smoke_final_20261027` and
`outputs/validation/unified_task_algorithm_20261027/real_smoke_final.json`.
The related Python regression selection passed 97 tests.

The first smoke batch is retained as a diagnostic record. Its two Nanshan
double-task audit failures were due to the zero-saving floating-point boundary;
their unchanged outputs passed the corrected independent reference. The final
batch also verifies the repaired root-column reuse and output counting.
