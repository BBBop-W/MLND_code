# Strict Lagrangian/Benders algorithm for single-period Regime-II MLND

## 1. Scope and non-negotiable formulation boundary

The algorithm solves the original single-period Regime-II planning model by
decomposition.  Its master is a pure MILP.  It contains:

- facility decisions \(x\);
- physical and aggregate arc decisions \(y\);
- hub capacity-level decisions \(z\);
- integer fleet size \(\Gamma\);
- operational-value estimator \(\eta\);
- binary OD service-possibility indicators \(w_{ij}\);
- binary direct/swap route-readiness indicators.

The \(w\) and route-readiness variables are logical upper-bound devices.  They
do not carry flow and do not replace any original operational variable.

The master contains none of the following:

\[
\psi,\ \varphi,\ \rho,\ \alpha,\ \beta,\ \lambda,\ \xi,
\]

and it contains no quadratic or second-order-cone constraint.  The
implementation checks this invariant after constructing the master.

All original continuous network-flow, route-allocation, battery, queueing and
fleet-time constraints remain in the exact continuous SOCP subproblem.

## 2. Exact Benders decomposition

Let

\[
d=(x,y,z,\Gamma)
\]

denote a first-stage design.  Let \(Q(d)\) be the optimal value of the original
continuous operational SOCP after fixing \(d\).  The original model is

\[
\max_{d\in\mathcal D}
\left\{
Q(d)-C(d)
\right\},
\]

where \(\mathcal D\) is the pure-integer first-stage feasible set and \(C(d)\)
is facility, capacity, arc and fleet cost.

The master objective is

\[
\max\ \eta-C(d).
\]

For a fixed reference design \(\bar d\), the convex conic dual of the
operational SOCP gives

\[
Q(d)
\le
Q(\bar d)
+
\pi(\bar d)^\top
\left[
b(d)-b(\bar d)
\right].
\]

Therefore the following classical conic Benders cut is globally valid:

\[
\eta
\le
Q(\bar d)
+
\pi(\bar d)^\top
\left[
b(d)-b(\bar d)
\right].
\]

The subproblem is solved with `QCPDual=1`.  The algorithm only uses a conic
optimality cut when the continuous SOCP is solved and its dual multipliers are
available.

Zero operational flow is feasible for every first-stage design that satisfies
the master capacity-domain constraints.  Consequently the principal cuts are
optimality cuts rather than classical infeasibility rays.

## 3. Maximum-profit estimation subproblems

Before solving the exact SOCP at an integer master design, the algorithm solves
two cheaper LP relaxations.

### 3.1 Aggregate-topology estimator

The topology estimator keeps the original demand-generation equations and
aggregate circulation variables, but relaxes:

- explicit direct/swap route decomposition;
- battery constraints;
- nonlinear queueing constraints.

It keeps first-stage arc activation, linear hub service capacity and a
minimum-flight-time fleet relaxation.  Its objective is served revenue minus a
valid lower bound on flight cost.

Its feasible region contains the projection of every exact operational
solution.  Hence

\[
Q(d)\le U^{\mathrm{top}}(d).
\]

The LP dual produces the globally valid estimator cut

\[
\eta
\le
U^{\mathrm{top}}(\bar d)
+
\pi^{\mathrm{top}}(\bar d)^\top
\left[
b^{\mathrm{top}}(d)-b^{\mathrm{top}}(\bar d)
\right].
\]

At every fixed integer design the algorithm also solves the corresponding
maximum-service LP

\[
F(d)=\max\sum_{(i,j)}\psi_{ij}.
\]

Since operating cost is nonnegative,

\[
Q(d)\le rF(d).
\]

Multiplying the maximum-service LP dual by \(r\) gives another globally valid
cut for \(\eta\).  At a zero-flow design this dual acts as a capacity/min-cut
certificate and is substantially less degenerate than the profit-objective
dual alone.

### 3.2 Direct/swap-aware estimator

The second estimator additionally restores route-decomposition variables,
the linear flight-energy restrictions, and requires each aggregate flow to use
either:

- an activated direct route; or
- an activated route through an open swapping station.

It replaces every nonlinear queue length by supporting-line lower bounds.  For
bases and pools, tangents of

\[
q(\rho)=\frac{\rho}{1-\rho}
\]

are imposed at a fixed utilization grid.  For hubs, tangents of

\[
q_i(\lambda)=\frac{\lambda}{\mu-\lambda}
\]

use \(\mu_{\max}\) in globally valid estimator cuts; fixed-design ranking may
use the selected capacity.  These lower bounds are included in both the fleet
constraint and the hub/pool energy inequalities.  Replacing the exact queue
length by a lower bound enlarges the feasible region.

It therefore remains an LP relaxation and

\[
Q(d)
\le
U^{\mathrm{swap}}(d)
\le
U^{\mathrm{top}}(d).
\]

Its LP dual generates a second globally valid upper-estimator cut.

If the current master value \(\eta\) exceeds either estimator value, the
integer design can initially be screened without solving the exact SOCP.
After a bounded number of consecutive screening iterations, the exact SOCP is
forced to prevent estimator-cut degeneracy from reducing the algorithm to
design enumeration.

## 4. Core-point stabilization

Maximum-profit estimators at an integer design may have the all-zero-flow
solution as a degenerate optimum.  The corresponding dual cut can be weak.

The implementation maintains a fractional core design

\[
d^c\leftarrow
\omega d^c+(1-\omega)\bar d
\]

and solves both estimator LPs at a convex combination of the current integer
design and \(d^c\).  A dual solution at this fractional reference still gives
a globally valid cut because the estimator is a parametric LP.  These
core-point cuts are the estimator analogue of Magnanti-Wong stabilization.

They were necessary for the H=5 strict-master algorithm to close the gap
without reintroducing flow variables into the master.

## 5. Topology maximum-flow cuts

The binary variable \(w_{ij}\) indicates only that OD pair \((i,j)\) may
contribute operational revenue.  It satisfies

\[
w_{ij}\le x_i,\qquad
w_{ij}\le x_j,\qquad
w_{ij}\le y_{ij}.
\]

Suppose \(w_{ij}=1\) at an integer master design.  Positive loaded flow from
\(i\) to \(j\), together with aggregate flow balance, requires a directed
return path from \(j\) to \(i\).  Let \(R\) be the set reachable from \(j\) in
the activated aggregate graph.  If \(i\notin R\), every feasible repair must
activate at least one arc in \(\delta^+(R)\).  Therefore

\[
w_{ij}
\le
\sum_{(u,v)\in\delta^+(R)} y_{uv}
\]

is valid for every integer first-stage solution.

This is the first flow-cut family.  It ignores battery and swapping details and
tests only aggregate directed connectivity.

## 6. Swap-reachability cuts

The second graph contains an aggregate arc \((u,v)\) only if at least one
physical route is currently available:

- \(y_{uv}=1\) for a direct route; or
- \(y_{uv}=y_{uk}=y_{kv}=1\) for a route through swapping station \(k\).

If an OD has aggregate connectivity but no route-feasible loaded/return cycle,
let \(M\) be the union of currently missing physical arcs across:

- its loaded direct/swap alternatives; or
- the direct/swap alternatives crossing a route-infeasible return cut.

If every arc in \(M\) remains closed, the OD still cannot carry positive
flow.  Thus the following logic-based flow cut is valid:

\[
w_{ij}
\le
\sum_{a\in M} y_a.
\]

This cut family is recorded separately from aggregate-topology cuts and can be
disabled independently in the ablation experiments.

## 7. Static master upper bounds

Because operating costs are nonnegative,

\[
\eta
\le
\sum_{(i,j)}
r h_{ij} w_{ij},
\]

where \(r\) is unit order revenue.

Let \(\underline t_{ij}^{cycle}\) be a design-independent lower bound on the
loaded and empty-return time required by one unit of OD service.  The fleet
epigraph implies

\[
\eta
\le
\left[
\max_{(i,j)}
\frac{r}{\underline t_{ij}^{cycle}}
\right]\Gamma.
\]

The hub service-stability constraints also imply

\[
\eta
\le
r\sum_i
\left(
\sum_c\mu_c z_{ic}
-\frac{x_i}{\tau_H}
\right).
\]

Let \(c_f\) and \(c_\Gamma\) be the flight-time and fleet costs.  For every OD
pair define

\[
a_{ij}
=
r-(c_f+c_\Gamma)\underline t_{ij}^{cycle}.
\]

Because \(\Gamma\) dominates total flight and queue time, every feasible
recourse solution satisfies

\[
\eta-c_\Gamma\Gamma
\le
\sum_{ij} h_{ij}[a_{ij}]_+ w_{ij}.
\]

A stronger destination-capacity dual family is obtained with arbitrary
\(\nu_j\ge0\):

\[
\eta-c_\Gamma\Gamma
\le
\sum_j \nu_j
\left(
\sum_c\mu_c z_{jc}-\frac{x_j}{\tau_H}
\right)
+
\sum_{ij}h_{ij}[a_{ij}-\nu_j]_+w_{ij}.
\]

The implementation installs several fixed multiplier vectors, one optimized
for each hub-capacity reference level.  These inequalities use only
\(x,z,\Gamma,\eta,w\); they are pure MILP constraints and contain no
continuous flow proxy.

## 8. Lagrangian-relaxation warm start

The algorithm does not call the original MISOCP to obtain an incumbent.

For each OD pair define a surrogate service fraction \(q_{ij}\in[0,1]\).  A
valid relaxation of the original model contains

\[
q_{ij}\le x_i,\qquad
q_{ij}\le x_j,\qquad
q_{ij}\le y_{ij}.
\]

Dualizing these three linking constraints with nonnegative multipliers
\(\lambda^o,\lambda^d,\lambda^a\) gives

\[
\begin{aligned}
L(\lambda)=
\max\quad&
\sum_{ij} r h_{ij}q_{ij}
-C_0(x,y,z)\\
&+\sum_{ij}\lambda^o_{ij}(x_i-q_{ij})
+\sum_{ij}\lambda^d_{ij}(x_j-q_{ij})
+\sum_{ij}\lambda^a_{ij}(y_{ij}-q_{ij}).
\end{aligned}
\]

The \(q\)-part separates by OD pair.  The design part is a small pure MILP
containing only first-stage assignment, activation and capacity-domain
constraints.  Since this surrogate drops operating cost, fleet cost, route
feasibility, circulation, battery and queueing restrictions, every
\(L(\lambda)\) is an upper bound on the original objective.

Multipliers are updated by projected subgradient steps:

\[
\lambda^{k+1}
=
\left[
\lambda^k
+s_k(q^k-x^k/y^k)
\right]_+,
\qquad
s_k=\frac{s_0}{\sqrt{k}}.
\]

The relaxed design is repaired using feasible loaded and empty-return cycles.
A small pure MILP then jointly consolidates base facilities, swapping stations
and shared physical arcs for the LR-selected OD cycles.  An LR-guided backward
deletion and Tabu search modifies the OD subset using only the swap-aware
estimator.

The strongest recovery phase fixes an LR-selected facility set and solves a
restricted pure MILP that jointly optimizes physical arcs, hub capacity levels,
fleet size and the continuous swap-aware flow estimator.  This neighborhood
contains no queueing cones and no variables from the original MISOCP.  A
second LR-centered facility-neighborhood MILP may add or remove only a bounded
number of hubs, bases and swapping stations relative to that seed.  It then
re-optimizes arcs, capacities and continuous fleet size.  This bounded Hamming
neighborhood is the large-scale primal-recovery mechanism; it is not seeded by
the global upper-bound MILP.

A subsequent network Tabu phase can reassign bases, change capacity levels and
alter route bundles.  Neighboring integer fleet sizes are retained because
the linear queue estimator can under- or over-estimate the best exact fleet
size.  The exact SOCP is not used during any candidate-generation phase.

The best LR designs enter Benders as MIP starts.  Their first exact SOCP
evaluation is part of the formal Benders algorithm, not a separate
Tabu-plus-SOCP heuristic.

## 9. Global MILP-relaxation upper bound

Before the Benders loop, the algorithm solves a separate swap-aware MILP
relaxation.  It contains binary facility, capacity and physical-arc decisions,
continuous \(\Gamma\), and the continuous flow estimator.  Exact queueing cones
are replaced by supporting-line lower bounds and the remaining omitted
constraints only enlarge the feasible region.  Therefore its maximization
objective is an upper bound on the original MISOCP.

If the relaxation reaches its own time limit, the algorithm uses Gurobi's
maximization `ObjBound`, never its incumbent objective, as the rigorous global
upper bound.  The relaxation incumbent is not passed to Benders and cannot
become an algorithm incumbent.  By default this phase receives

\[
\min\{60\text{ seconds},\,0.10\,T\}
\]

inside the shared wall-clock budget \(T\).  The switches
`--global-ub-time-limit` and `--global-ub-time-fraction` expose this allocation,
and `strict_no_global_ub` provides the corresponding ablation.

## 10. Bounds and stopping rule

The algorithm maintains:

- a valid LR upper bound;
- the global swap-aware MILP-relaxation bound;
- the current MILP master bound;
- an incumbent lower bound obtained only from exact SOCP recourse;
- the best validated first-stage design.

The global upper bound is the minimum of the available LR, global-relaxation
and master bounds.
The algorithm stops when

\[
UB-LB
\le
\epsilon\max\{1,|LB|\}
\]

or when the shared wall-clock budget reaches 900 seconds.

## 11. Current verified results

The strict implementation has passed the H=5 regression:

| Item | Value |
|---|---:|
| Master quadratic constraints | 0 |
| Master operational \(\psi\) variables | 0 |
| Final strict Benders objective | 539.129684 |
| Fresh original-model validation | 539.129684 |
| Strict Benders wall time | 6.9 seconds under a 15-second budget |

The current unified 900-second main comparison is:

| H | Strict incumbent | Strict UB | Strict gap | Original Gurobi incumbent | Incumbent difference |
|---:|---:|---:|---:|---:|---:|
| 10 | 2558.2018 | 2777.6383 | 8.58% | 2616.6625 | -2.23% |
| 20 | 6375.7803 | 6770.1768 | 6.19% | 6436.4617 | -0.94% |
| 40 | 12942.8194 | 14251.2738 | 10.11% | 13622.4156 | -4.99% |

Every strict incumbent was independently validated by fixing
\(x,y,z,\Gamma\) in a fresh original model.  The Master has zero quadratic
constraints at every scale.

The stored H=20 900-second ablations currently show:

| Variant | Incumbent | Bound | Reported gap |
|---|---:|---:|---:|
| Full | 6375.7803 | 6770.1768 | 6.19% |
| No LR | 0.0000 | 6767.1636 | no useful incumbent |
| No restricted network MILP | 5784.7812 | 6769.9090 | 17.03% |
| No global MILP upper bound | 6353.0025 | 21023.2415 | 230.92% |
| No topology/swap flow cuts | 6375.7803 | 6771.1999 | 6.20% |
| Topology only | 6353.0025 | 6767.4964 | 6.52% |

All ablations were rerun against the final code with 900-second limits.  The
full algorithm used 310 Master iterations.  Removing both flow-cut/estimator
families required 613 iterations, while retaining only topology cuts required
472.  Thus, once a strong LR incumbent and the global relaxation bound are
available, the flow families primarily improve convergence efficiency.  The
LR and restricted-network ablations quantify incumbent quality; the
global-bound ablation quantifies bound quality.

## 12. H=80 scalability and allowed simplifications

The original-model H=80 pilot retains a valid global upper bound but LR primal
recovery remains weak.  Two explicitly permitted simplifications were tested:

- forcing every swapping station closed gives objective zero;
- relaxing \(\Gamma\) to a continuous variable gives 7026.1033 in 180 seconds.

Neither simplification resolves the H=80 recovery gap.  Consequently H=80 is
reported as a scalability limitation and is not included in the main
performance claim.

## 13. Requirement-to-evidence audit

The implementation and experiment artifacts cover the requested boundaries as
follows.

| Requirement | Evidence |
|---|---|
| Pure MILP Master | Runtime invariant reports `NumQConstrs=0`; no Master variable name begins with `psi` or contains `queue`. |
| Continuous flow, queueing and SOC recourse | `solve_subproblem` in `src/model/Benders.py` owns \(\psi,\rho,\alpha,\beta,\varphi,\lambda,\mu,\xi\) and every quadratic queue constraint. |
| Fixed-design maximum-profit estimators | `solve_topology_profit_estimator` and `solve_swap_profit_estimator` solve LP relaxations at fixed \(x,y,z,\Gamma\). |
| Topology and swapping reachability cuts | `add_topology_connectivity_cuts` and `add_swap_reachability_cuts` maintain separate signatures, counters and ablation switches. |
| LR-only initial incumbents | Every row of `initial_design_history.csv` is marked `source=lagrangian`; exact SOCP is called only after candidate generation. |
| No original MISOCP incumbent injection | `src/model/lagrangian_benders.py` does not import `MLNDModel`; the global MILP relaxation records its incumbent for diagnostics but only its `ObjBound` is used. |
| Valid bound ordering | On the stored H=10/20/40 designs, both LP estimators were numerically verified to be no smaller than exact SOCP recourse. |
| Unified budget and validation | Strict search phases share `--time-limit 900`; fresh fixed-design original-model validation is outside the search budget and cannot improve the design. |
| Multi-scale comparison | `paper_comparison.csv` contains H=10,20,40 strict and original-Gurobi results. |
| Contribution and convergence evidence | `ablation_comparison.csv` and `convergence_checkpoints.csv` record final-code ablations and time checkpoints. |

The paper-facing artifacts are under
`results/strict_lagrangian_benders_benchmark_900s/`.
