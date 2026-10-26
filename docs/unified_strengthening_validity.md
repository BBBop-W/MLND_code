# Cut validity for fixed generalized tasks

This note records the adaptations in `cpp/src/strengthening.cpp` for the fixed generalized-task model.
It concerns mathematical validity and regression coverage; it does not claim that every cut improves the root bound or the running time.
The operating model, fixed task reconstruction, and physical path catalogue remain unchanged.

## Occurrence supply

Let `nu[r]` denote a single-task rate, `gamma[r,s]` an ordered double-task rate, and `H[r] = nu[r] + sum_s gamma[r,s]` the rate of tasks headed by OD `r`.
The preprocessor conserves raw demand:

\[
\lambda_r=\nu_r+\sum_s\gamma_{rs}+\sum_h\gamma_{hr}.
\]

A self-pair `gamma[r,r]` occurs twice in this identity because it serves two orders.
Only the head of a task owns its origin assignment `alpha[r,l]`.
Let `v[r,l]` be its executed fraction, with `0 <= v[r,l] <= alpha[r,l]`.
The existing task-flow equalities imply that the raw-OD occurrence flow from physical origin `l` is exactly

\[
F_{rl}=H_r v_{rl}+\sum_h\gamma_{hr}v_{hl}.
\]

Consequently the following affine upper supply is valid for integer and fractional designs:

\[
F_{rl}\le Q_{rl}(\alpha)
=H_r\alpha_{rl}+\sum_h\gamma_{hr}\alpha_{hl}.
\]

This is the only change required in the occurrence-demand side of the previous analytical, monetary max-flow, and weighted resource-LP cuts.
The implementation coalesces the two contributions on `alpha[r,l]` into `H[r] + gamma[r,r]`.
For single tasks, `gamma = 0` and `H = lambda`, so this reduces to the previous single-order bound.

Using `lambda[r] * alpha[r,l]` in the double-task model is invalid.
An OD can be delivered as the second order of a task headed by another OD, even when its own assignment at that origin is zero.
Its own assignment may also point to a different origin.
The regression probe includes feasible witnesses of both situations and a witness that violates the old analytical bound by a positive margin.

## Physical cost and locker allocations

For each raw-OD occurrence and physical origin, the existing code computes an input-only lower allocation of its first/last rider costs and its loaded flight cost.
It allocates `1 / K` of a minimum outbound base-access flight and a minimum return flight to that occurrence, where a task contains at most `K` orders.
Summing over the orders on a task cannot charge more than one outbound and one return flight.
The return minimum ranges over all physical bases for origin assignment.
Transfer flights, waiting, relocation, and idle inventory costs are nonnegative and can be omitted from a lower bound.
Including fleet cost times flight duration remains valid because fleet use dominates total busy-drone time.

The locker allocation is a lower bound consisting of the loaded flight duration plus the destination removal time.
Other reservation and waiting components are nonnegative.
The more detailed resource model retains each occurrence's destination locker and its corresponding cost/load lower bounds.
These allocations apply to an order in either task position and count self-pairs twice.
No condition specific to the initial base-access leg is imposed on a second order.

Write `c0` for the all-courier cost and `s[r,l]` for the nonnegative upper saving per occurrence derived from the cost allocation.
The analytical assignment cut is therefore

\[
\theta+\sum_{r,l}s_{rl}Q_{rl}(\alpha)\ge c_0.
\]

The global locker savings/load cut is unchanged.
Conservative numerical guards weaken the inequalities; they do not change this argument.

## Capacity and weighted resource cuts

The monetary max-flow relaxation now uses `Q[r,l]` on each source-to-occurrence-group and occurrence-group-to-locker capacity, scaled by the corresponding saving upper bound.
The locker-to-sink capacities retain the original affine capacity-level expressions.
Every feasible operating solution maps to a feasible savings flow in this relaxation.
Any source/sink partition thus gives an upper bound on achievable savings and a lower bound on operating cost.
The separator recomputes the capacity of the original crossing arcs instead of treating a numerical max-flow value as a certificate.

The weighted resource LP instead uses nonnegative occurrence quantities `u[r,l,j]`, with

\[
\sum_j u_{rlj}\le Q_{rl}(\alpha),\qquad
\sum_{r,l}\omega_{rlj}u_{rlj}\le\sum_s\bar m_{js}z_{js}.
\]

Its objective maximizes the sum of occurrence-saving upper bounds times `u`.
It also retains the existing airline resource rows when enabled.
Every feasible operating solution maps into this LP, so a feasible dual vector provides an upper bound on savings.
The implementation independently repairs and verifies dual feasibility before generating its global cost cut.
No LP operational variable is added to the strategic master.

Aggregate airline gates follow from original link-flow rows and task cardinality and remain valid for the corresponding continuous operational relaxation.
OD-specific airline gates use the conserved raw demand `lambda[r]`, including incoming and self-pair occurrences.
At a binary design, each counted occurrence requires an open physical departure/loaded link and total served occurrences cannot exceed `lambda[r]`.
These OD gates are valid for integer designs and their convex hull, but need not be valid for every point in the original fractional operational relaxation.
Accordingly the probe validates OD-gated cuts against integer recourse and convex combinations of those projections; it validates the other cuts against fractional recourse as well.

## Other retained or excluded strengthening

- Queue perspective and shared-availability constraints retain their previous integer-valid arguments. They concern availability and facility decisions, independently of how task demands were reconstructed.
- The assignment-incident constraint is harmless but duplicates an existing departure implication under origin-only assignment.
- The existing topology cut is safe: when its physical-path oracle proves that no service path is supported by a design, every generalized task must use courier service. Searching a superset of the positive-rate tasks can weaken this test but cannot invalidate the absence certificate.
- The tight airline bounds continue to use raw-order conservation and maximum task cardinality.
- Legacy assignment-support, double-support, and cross-base joint-cost cuts are not enabled. Their implementation assumes fixed return assignments and old assignment-dependent rate floors; these assumptions do not apply to the fixed generalized-task model.
- No additional task-support cut is claimed here. The resource relaxation already carries head-assignment contributions into incoming raw-OD occurrences, but it is deliberately a relaxation and does not certify complete route support.

The intended safe options are the existing analytical/capacity/resource-LP defaults together with `--tight-airline-bounds`, `--od-airline-bounds`, `--topology-cuts`, `--assignment-incident-clique`, `--shared-rho-joint`, and `--queue-perspective`.
The integer-valid OD cuts must retain their existing separation from certificates for the original continuous relaxation.

## Reproducible regression

The new probe uses the project's existing two-OD physical-fixture structure with one and two bases, single and double tasks, ample and binding capacities, and integer and fractional assignment/infrastructure designs.
Fixed double rates deliberately include both cross-OD directions and a self-pair.
It checks every catalogue path's cost/load allocation, solves full operational recourse at each test design, and cross-checks each separated cut at all compatible target projections.
It also constructs two explicit flow witnesses in which a second order has no assignment at its task's origin; one has no own assignment at all, and the other has its own assignment at another base.
All physical service-link networks in these fixtures are independently enumerated and compared with the topology absence oracle.

```sh
make -C cpp -j2 build/mlnd_unified_strengthening_probe
cpp/build/mlnd_unified_strengthening_probe
```

These are bounded regression fixtures, not newly generated production benchmark instances.

Validation on 2026-10-27 passed 42 optimal full-recourse solves (including 12 fractional designs), 3,564 capacity/resource cut cross-projection checks, and 120 physical-path cost/load checks.
The topology regression checked all 2,176 fixture networks and 20,624 cut/projection combinations.
Every recorded maximum cut or allocation violation was zero.
The explicit feasible witness without a second-OD assignment violated the old own-assignment analytical bound by approximately 9.425742; the corrected bound admitted it.
The machine-readable report is `outputs/validation/unified_task_algorithm_20261027/unified_strengthening_probe.json`.
