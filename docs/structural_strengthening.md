# Joint structural strengthening of both Benders implementations

The production comparison separates two demand formulations and two versions.
`lifted` retains the auxiliary demand-reconstruction variables in fixed-design recourse.
`conditional` computes demand coefficients after fixing an integer assignment and issues assignment-conditioned cuts.
Both formulations use the same lifted continuous model for free-design initialization and fractional recourse.
The new strengthening must therefore reach that shared model, the strategic master, and restricted-MIP heuristics as well as both integer subproblem implementations.
It does not turn a conditional cut into an unconditional cut.

## Mathematical changes

Let `x_i` be a binary base-opening variable, `a_rlb` the binary choice of an ordered departure/return base pair for OD `r`, `rho_l` the common availability, `v_rlb = a_rlb rho_l`, and `I_l` idle fleet.
Every OD chooses at most one base pair.

1. Assignment incidence: `sum_{(l,b): i in {l,b}} a_rlb <= x_i`, counting `(i,i)` once.
   This adds `R*L` strategic rows and no variables.
2. Shared availability: with `A_rl=sum_b a_rlb` and `V_rl=sum_b v_rlb`, add `V_rl<=rho_l` and `A_rl-V_rl<=x_l-rho_l`.
   These are sums of existing variables, not new master variables.
   When this feature is enabled without the incidence feature, the master also receives its implied domain constraint `A_rl<=x_l` so all-courier fractional recourse remains feasible.
3. Queue perspective: replace `(I_l+1)(1-rho_l)>=1` by `(I_l+x_l)(x_l-rho_l)>=x_l^2`, retaining `I_l>=0` and `0<=rho_l<=x_l`.
   The implementation uses a rotated second-order cone, including the closed boundary at `x_l=rho_l=0`.
   The existing finite idle bound `I_l<=M` also yields the integer-valid row `rho_l<=M/(1+M)*x_l`; its coefficient is rounded toward the weaker side.

Each change preserves the original integer feasible set, including all-courier service and a closed base with nonnegative idle slack.
The strengthened continuous feasible set is generally smaller.
Only the isolated assignment/common-availability block has the stated local convex-hull description; no claim is made about the convex hull of the full model.
For fixed integer designs these constraints are redundant, so different fixed-design optimal values beyond solver tolerance would be a correctness failure, not an algorithm improvement.

The queue cone retains three existing lifted coordinates; its new links are `up=I+x`, `down=x-rho`, `one=x`.
All-courier initialization, primal repair and heuristic point reconstruction use these same links.
The strategic master still contains only the original design variables and one operating-cost epigraph variable, with no operational variables or cones added.

## Controls and certificates

All three features are opt-in to preserve the old implementation for comparison.
Use `--structural-strengthening` for the complete package, or `--assignment-incident-clique`, `--shared-rho-joint`, and `--queue-perspective` separately.
The existing `--disable` mechanism accepts the corresponding underscore names.
`--demand-formulation lifted|conditional` remains independent.

The continuous diagnostic probe accepts the same feature names followed by `true` or `false`.
It exports the best validated fractional design together with `rho`, `idle`, fleet and the `v` array, and separately reports the certified complete-domain lower bound.
A feasible restricted-column objective is not a lower bound.
Only the existing full-domain pricing/dual certification supplies a lower bound or Benders cut.
Conditional cut validity remains restricted by its recorded assignment condition and its global gating construction.

## Reproducible comparison

`src.experiments.structural_benchmark` freezes the current input, new source, executable, and the previously frozen old executable.
The standard comparison is the fixed TSL 2L/20H/50OD case, K=1 and K=2, four implementations per K, 600 seconds each, one solver thread, seed 43 and no more than three concurrent jobs.
Root preparation, model construction and CG are charged to each production job.
Independent continuous-root ablations use separate budgets and are never supplied as free starts to production runs.

The four root variants are baseline, joint assignment strengthening, queue perspective alone, and all features combined.
The four production variants are old lifted, new lifted, old conditional, and new conditional.
Each method's gap uses its own validated upper bound and certified lower bound.
Reports retain solver status, process exit, incomplete pricing, validation, source/input/binary hashes, phase times and cut records separately.

## Research basis

- [Adams and Sherali (1990), Linearization Strategies for a Class of Zero-One Mixed Integer Programming Problems](https://pubsonline.informs.org/doi/10.1287/opre.38.2.217): joint reformulation and linearization of binary/continuous products.
- [Günlük and Linderoth (2010), Perspective reformulations of mixed integer nonlinear programs with indicator variables](https://link.springer.com/article/10.1007/s10107-010-0360-z): indicator-based convex strengthening, including congested network design.

The inequalities and integer-equivalence arguments above are derived for this repository's model.
Those papers support the techniques, not a claim of measured speedup on the TSL instance.
