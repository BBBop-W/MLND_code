# Regime-II BDD/LR/Benders implementation dated 2026-06-23

This directory documents the isolated implementation in
`src/model/bdd_lr_benders_20260623/`.  The implementation does not import or
wrap the legacy `StrictLagrangianBendersSolver`.  It reuses only:

- the instance representation;
- the fixed-design continuous SOCP recourse oracle;
- the original model as a constructor for a BDD local-copy cut subproblem;
- the independent fixed-design validation routine.

The original MISOCP is never used to provide an incumbent to LR, BD, or LR+BD.
A local-copy BDD solution is also not an incumbent until a fresh fixed-design
SOCP validates it.

## 1. Strict decomposition boundary

The Master contains only \(x,y,z,\Gamma,\eta\) and binary logical variables.
It contains no operational flow, route allocation, queue variable, or conic
constraint.  `assert_pure_master` checks:

```text
NumQConstrs == 0
no variable name has an operational prefix
```

For a first-stage design \(d=(x,y,z,\Gamma)\), let \(Q(d)\) be the value of the
continuous operational SOCP.  The planning model is

\[
\max_{d\in\mathcal D} Q(d)-C(d),
\]

and the Master maximizes \(\eta-C(d)\).

## 2. Classical conic dual cut

The fixed-design SOCP is solved with `QCPDual=1`.  If its parameterized
constraints can be written as

\[
A u\le b(d),
\]

an optimal conic dual solution gives the global upper support

\[
\eta\le Q(\bar d)+\pi^\top[b(d)-b(\bar d)].
\]

No cut is generated when reliable parameter duals are unavailable.

## 3. Reference-tight Pareto/Magnanti-Wong selection

The old practice of solving an ordinary SOCP at a core-reference design is not
treated as a Pareto-optimal cut.

The new selector first computes a strict fractional Master core point \(d^c\)
by maximizing the minimum distance of the strategic binary variables from
their bounds in the Master LP relaxation.  For a sequence
\(\epsilon\downarrow0\), it solves the SOCP at

\[
d^\epsilon=(1-\epsilon)\bar d+\epsilon d^c.
\]

Let the resulting dual support be \(U_\epsilon(d)\).  It is globally valid
because it comes from a dual-feasible conic solution.  It is accepted only if

\[
|U_\epsilon(\bar d)-Q(\bar d)|
\le \tau\max\{1,|Q(\bar d)|\}.
\]

This numerical equality certifies that the selected dual remains on the
optimal-dual face at \(\bar d\).  Since it minimizes the perturbed dual
objective and the reference contribution is constant on that face, it
minimizes \(U_\epsilon(d^c)\) over the reference face.  For this maximization
recourse problem, a smaller upper support at the core is the
Magnanti-Wong criterion.

Every attempt records reference tightness, core value, improvement over the
classical cut, epsilon, and whether the certificate passed.

If degeneracy produces no reliable dual cut at a repeated integer Master
point, the algorithm adds an exact integer no-good cut.  Two auxiliary binary
variables represent \(\Gamma\le\bar\Gamma-1\) or
\(\Gamma\ge\bar\Gamma+1\), so only the exact
\((x,y,z,\Gamma)\) point is removed.  Other fleet sizes for the same network
remain feasible.

## 4. Benders dual decomposition for maximization recourse

Introduce an integer or partially integer local copy \(\tilde d\) of the
Master design in the recourse feasible set \(\mathcal X\).  For a multiplier
vector \(\lambda\), define

\[
\Phi(\lambda)
=
\max_{(u,\tilde d)\in\mathcal X}
\left\{
p^\top u-\lambda^\top\tilde d
\right\}.
\]

For every feasible integer Master design \(d\), choosing
\(\tilde d=d\) proves

\[
Q(d)\le \Phi(\lambda)+\lambda^\top d.
\]

Therefore

\[
\eta\le \Phi(\lambda)+\lambda^\top d
\]

is a globally valid BDD cut.  This is the maximization counterpart of the
lower cuts in Rahmaniani et al. (2020).

### Strengthened BDD cut

The slope \(\lambda\) is initialized from the classical conic dual cut.  The
local-copy MISOCP restores integrality for variables selected by their
fractionality.  Restricting the local-copy feasible set can only reduce
\(\Phi(\lambda)\), hence it tightens the upper cut while preserving validity
for integer Master designs.

Only Gurobi's maximization `ObjBound` is used as the cut intercept.  A
time-limited incumbent objective is not a valid intercept.

### Lagrangian BDD cut

Given previously generated local points
\((u^v,\tilde d^v)\), a regularized multiplier master solves

\[
\min_{\lambda,\eta}
\quad
\eta+\frac{\delta}{2}\|\lambda-\lambda^{t-1}\|_2^2
\]

subject to

\[
\eta\ge p^\top u^v
+\lambda^\top(\bar d-\tilde d^v)
\qquad\forall v.
\]

The updated multiplier is evaluated by another local-copy MISOCP.  The final
cut again uses the rigorous MISOCP `ObjBound`.

### Multiphase scheduling

The root strengthening loop uses:

1. classical and certified Pareto cuts;
2. strengthened BDD cuts after an initial round;
3. regularized Lagrangian BDD cuts in the final configured root round.

BDD is generated at fractional Master LP points.  At integer points,
strengthened and classical cuts generally coincide.

### Epsilon-optimal correction

If the local-copy MISOCP certifies

\[
\operatorname{ObjBound}-\operatorname{ObjVal}\le\varepsilon,
\]

the intercept may use
\(\min\{\operatorname{ObjBound},\operatorname{ObjVal}+\varepsilon\}\).
This remains a valid upper bound.

## 5. LR-only method

The LR method dualizes the service links

\[
q_{ij}\le x_i,\qquad q_{ij}\le x_j,\qquad q_{ij}\le y_{ij}.
\]

The OD part separates and the design part is a pure MILP.  Multipliers use
projected subgradient updates.  The LR value is a valid upper bound because
the relaxation removes operating cost, fleet cost, circulation, battery,
route, and queue restrictions.

Recovered networks are generated without calling the exact SOCP.  A
restricted full-network MILP is built by materializing and removing every
queueing quadratic constraint, verifying `NumQConstrs == 0`, and adding
globally valid queue supporting lines.  The exact SOCP is used only to
evaluate the bounded candidate list and establish the LR-only incumbent.
An optional restricted-network solution pool can keep multiple cone-free MILP
solutions as LR recovery candidates.  This is a recovery diversification
mechanism only; it does not change the Master or introduce operational
variables into the Master.

The LR profiler separates:

- design MILP wall and solver runtime;
- recovery/consolidation MILP wall and solver runtime;
- OD selection;
- multiplier updates;
- candidate construction/ranking;
- exact SOCP evaluation.

A C++ migration is considered only after repeated H=40/80/100 evidence shows
aggregate non-solver time above 20%, or a mechanically portable hotspot above
10% of LR time.

The raw OD-activation LR dual is intentionally retained as `raw_lr_bound`, but
it is too weak to use as the final reported optimality bound.  LR-only now
reports the minimum of the raw LR dual and the full-network cone-free
linearized relaxation bound when that bound is available.  The correction and
H = 5/10/50 evidence are recorded in
`docs/bdd_lr_benders_20260623/lr_gap_bound_fix.md`.

## 6. Adaptive local branching

For an incumbent \(\bar d\), the local branching model imposes a Hamming
neighborhood on binary \(x,y,z\):

\[
1\le \Delta(x,y,z;\bar d)\le k,
\]

and a separate interval

\[
\bar\Gamma-r_\Gamma\le\Gamma\le\bar\Gamma+r_\Gamma.
\]

The neighborhood problem is a full-network cone-free MILP with the same queue
supporting lines used by LR.  Its quadratic constraints are explicitly
removed and checked.  An improving neighborhood shrinks \(k\); a failed
neighborhood expands it.  Every candidate is evaluated by the exact SOCP
before it can update the lower bound.

The local branching module can also evaluate a small solution pool from the
cone-free neighborhood MILP.  This addresses proxy-objective misranking by
validating multiple neighborhood designs with the exact SOCP, while keeping
the local model and the Master cone-free.

Three optional recovery-only stabilizers are implemented for local branching:

- a Benders-cut envelope objective over recent exact SOCP cuts, controlled by
  `--local-branching-benders-envelope-cuts` and blended with the cone-free
  network objective by `--local-branching-benders-envelope-weight`;
- a deletion trust region, controlled by `--local-branching-max-removals`, that
  limits how many currently open \(x,y,z\) decisions may be turned off.
- a service-preserving route lock, controlled by
  `--local-branching-protect-top-pairs`, that keeps the center design's route
  arcs for the highest-demand OD pairs that are already reachable.

Both are candidate-generation devices only.  They do not add variables to the
Master and they do not certify incumbents.  H = 50 tests show that the Benders
envelope is valid for bounding but too optimistic as a stand-alone ranking
objective; it should be used with service-preserving local structure.

## 7. Exact fixed-network Gamma sweep

After initial LR validation and local branching, the solver can keep the
validated incumbent network \((x,y,z)\) fixed and enumerate nearby fleet sizes
\(\Gamma\).  Each enumerated design is solved by the exact fixed-design SOCP
before it can become the incumbent.  Successful sweep points may also generate
classical conic cuts, and improved sweep points may request a Pareto cut.

The sweep is controlled by:

- `--gamma-sweep-max-calls`;
- `--gamma-sweep-radius`;
- `--gamma-sweep-step`;
- `--gamma-sweep-time-reserve`.

The enumeration is upward-first because H = 50 diagnostics showed that the
cone-free recovery model systematically underestimates the best exact fleet
size.  This is a recovery postprocess only: it does not add any variable to the
Master, and it does not call the original monolithic MISOCP.

The H = 50 `qfix_recovery_v28_h50_restricted24_gamma300` run used a
24-second LR restricted-network refinement limit plus upward-first exact
Gamma sweep.  Its independently validated incumbent was 15036.4749, above the
clean Gurobi validation benchmark 15017.9288.  The remaining issue is to test
whether this improvement is stable across H = 60/70/80/100 and multiple
instances.

The first 300-second transfer test is recorded in
`docs/bdd_lr_benders_20260623/recovery_v29_scale_gamma_sweep.md`.  It covers
H = 50, 60, 70, 80 and 100 and improves the existing 300-second Gurobi
baseline on all five scales, but it is still a single-run diagnostic rather
than the final multi-seed matrix.

## 8. Global linearized network upper bound

The algorithm may solve a separate full-network MILP after removing every
queueing quadratic constraint and adding supporting-line lower bounds for the
convex queue functions.  All remaining first-stage, flow, route, energy,
capacity and fleet constraints are retained.  Every original feasible point
satisfies the supporting lines, so the linearized feasible region remains a
relaxation.  The maximization `ObjBound` is therefore a rigorous upper bound
on the original objective.

Only `ObjBound` is added as a Master objective cap.  The relaxation incumbent
is never passed to LR, BD, or LR+BD.

## 9. Reproducible commands

Four-method small smoke test:

```bash
python -m src.experiments.bdd_lr_benders_20260623.benchmark \
  --hub-count 5 \
  --methods gurobi,lr,bd,lr_bd \
  --variant qfix_smoke \
  --time-limit 30
```

Main grid:

```bash
python -m src.experiments.bdd_lr_benders_20260623.run_grid \
  --scales 5,10,20,40,60,80,100 \
  --methods gurobi,lr,bd,lr_bd \
  --time-limit 900
```

Cut and local-branching ablations:

```bash
python -m src.experiments.bdd_lr_benders_20260623.run_ablations \
  --scales 5,10,20 \
  --methods bd,lr_bd \
  --time-limit 900
```

Structural/result audit:

```bash
python -m src.experiments.bdd_lr_benders_20260623.audit \
  --variant qfix_formal900_final
```

## 9. Valid result boundary

An implementation defect was found during development: `MLNDModel.build()`
had not been updated before `getQConstrs()` was called, so early candidate
models still contained their queueing cones.  Every result produced before
the explicit update/remove/assert fix is invalidated.  Formal aggregation
accepts only `qfix_*` variants.  The detailed record is in
`results/bdd_lr_benders_20260623/INVALIDATED_RESULTS.md`.

## 10. Formal four-method matrix

All objectives below have an independent fixed-design SOCP validation.  The
budget is 900 seconds per method; early termination is allowed.

| H | Method | Validated objective | Bound | Gap | Wall time |
|---:|---|---:|---:|---:|---:|
| 5 | Gurobi | 539.129627 | 539.534324 | 0.0751% | 0.04 s |
| 5 | LR | 537.225822 | 20681.000000 | 3749.59% | 0.15 s |
| 5 | BD | 539.129628 | 539.130865 | 0.0002% | 5.87 s |
| 5 | LR+BD | 539.129627 | 539.135962 | 0.0012% | 4.99 s |
| 10 | Gurobi | 2685.816389 | 2863.532523 | 6.6168% | 899.93 s |
| 10 | LR | 2579.501876 | 44399.875000 | 1621.26% | 4.36 s |
| 10 | BD | 2650.313528 | 2754.151289 | 3.9179% | 900.01 s |
| 10 | LR+BD | 2650.313528 | 2754.151289 | 3.9179% | 900.00 s |

H=5 is the correctness closure: Gurobi, BD, and LR+BD validate to the same
objective within SOCP numerical tolerance.  LR-only is feasible but its
current dual bound is too weak to certify convergence.

At H=10, BD and LR+BD provide a tighter upper bound than Gurobi, but Gurobi
finds the better incumbent.  LR warm-starting does not improve BD on this
instance.

## 11. Ablation results

BD component ablations use 120 seconds.

| H | Variant | Objective | Bound | Gap | Time |
|---:|---|---:|---:|---:|---:|
| 5 | Classical conic dual only | 445.675572 | 568.219019 | 27.4961% | 120.00 s |
| 5 | + certified Pareto | 539.129684 | 539.129684 | 0 | 2.56 s |
| 5 | + strengthened BDD | 539.129684 | 539.129684 | 0 | 2.89 s |
| 5 | + Lagrangian BDD | 539.129684 | 539.129684 | 0 | 2.78 s |
| 5 | + local branching | 539.129684 | 539.129684 | 0 | 6.10 s |
| 10 | Classical/Pareto/BDD, no local branching | 0 | 2754.151289 | large | 120.00 s |
| 10 | Full BDD + local branching | 2650.313353 | 2754.151289 | 3.9179% | 120.00 s |

Pareto selection is decisive at H=5.  BDD preserves validity but does not
improve this small instance further.  At H=10, local branching is the
decisive incumbent mechanism; the cut variants without it do not find a
positive design within 120 seconds.

The H=20 LR queue-line ablation uses the same approximately 33-second runtime:

| LR recovery | Validated objective |
|---|---:|
| Cones removed, no queue supporting lines | 5154.120834 |
| Cones removed, queue supporting lines enabled | 6090.827581 |

The supporting lines improve the incumbent by 18.18%, so they are enabled by
default.

## 12. Large-scale Gurobi versus LR

| H | Method | Validated objective | Bound | Gap | Wall time |
|---:|---|---:|---:|---:|---:|
| 20 | Gurobi | 6540.443320 | 6793.722573 | 3.8725% | 900.07 s |
| 20 | LR | 6090.827581 | 55619.000000 | 813.16% | 33.08 s |
| 40 | Gurobi | 13616.167784 | 14720.636060 | 8.1115% | 900.09 s |
| 40 | LR | 7823.185953 | 64469.375000 | 724.08% | 43.68 s |
| 80 | Gurobi | 13211.458561 | 18991.193758 | 43.7479% | 900.05 s |
| 80 | LR | 944.325896 | 65050.500000 | 6788.56% | 52.23 s |

These experiments reject the working hypothesis that the current LR
implementation overtakes Gurobi at large scale.  LR is much faster but its
OD-subset and network recovery quality deteriorates sharply.  The next
algorithmic priority is recovery quality, not lower-level implementation
speed.

## 13. LR timing and C++ decision

At H=20, non-solver LR work is 0.066% of LR time; at H=40 it is 0.139%.
The dominant phase is the Gurobi cone-free restricted-network MILP:
approximately 32.00 of 32.66 seconds at H=20 and 40.01 of 42.12 seconds at
H=40.  No Python hotspot reaches the 10% threshold and aggregate non-solver
time is far below 20%.  A C++ rewrite is therefore not justified by the
measured profile.

## Reference

Rahmaniani, R., Crainic, T. G., Gendreau, M., and Rei, W. (2020).
The Benders Dual Decomposition Method.
