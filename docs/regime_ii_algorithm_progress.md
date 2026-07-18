# Regime-II model-based algorithm progress

## Current algorithm

The leading implementation is a model-based warm-started partial Benders
method:

1. A short primal MIP phase solves the original MIQCP.  The default `auto`
   policy uses root-relaxation max-flow cuts below H=60 and the numerically
   robust Gurobi configuration at H>=60.
2. Its integer design is re-evaluated by the exact continuous SOCP subproblem,
   giving a certified lower bound.
3. The partial-Benders master keeps facility, arc, capacity, fleet, aggregate
   flow, and queue variables.
4. The master includes the valid deterministic lower bound
   `route_cost >= sum(min_route_cost[u,v] * psi[u,v])`.
5. A route recourse LP verifies route allocation, energy, and fleet feasibility
   and generates feasibility or operating-cost cuts.
6. Every reported design is fixed in a fresh copy of the original, uncut model
   and solved again.  This guards against invalid cuts and MIQCP numerical
   artifacts.

The method is model-based.  The primal warm start does not replace Benders and
all of its time is included in the method's 900-second search limit.

## Max-flow cut

For loaded OD flow `psi[i,j]`, flow conservation requires a return path from
`j` to `i`.  For a directed cut `R` containing `j` but not `i`,

```text
psi[i,j] <= demand[i,j] * sum(y[u,v] for (u,v) in delta+(R)).
```

The cut is separated on a continuous root relaxation and then added as an
ordinary constraint.  Python callbacks are intentionally avoided: with Gurobi
13.0.2 on the current macOS environment, enabling even an empty callback on
this MIQCP changed the reported optimum.

Co-located hub/holding-pool arcs are treated as always available during
separation.  A cut crossed by one of these virtual arcs is skipped.

## Development benchmark

These are short development runs, not the final 900-second experiment.  All
objectives below were revalidated by fixing the returned integer design in a
fresh original model.

| H | Budget | Threads | Gurobi | Gurobi + max-flow | Seeded partial Benders |
|---:|---:|---:|---:|---:|---:|
| 5 | 10 s | 1 | 539.1296 | 539.1296 | 539.1297 |
| 10 | 20 s | 1 | 2617.8348 | 2623.1101 | **2636.9544** |
| 20 | 30 s | 4 | 6353.5851 | 6444.0764 | **6464.0932** |
| 40 | 60 s | 4 | 13344.0044 | -211.1833 | **13372.5381** |

At H=40, standalone max-flow cuts harmed Gurobi's primal search despite
strengthening the relaxation.  The partial-Benders variant remained robust
because it used a certified incumbent floor and repeated master/recourse
iterations.  Its H=40 schedule used a 3-second warm start and 10-second master
segments; the formal 900-second defaults use a 60-second warm-start cap and
30-second master segments.

## Formal 900-second results

The comparison uses the fresh-original-model validation objective when it
differs from the solver's reported incumbent.  `Best Gurobi` is the better
validated incumbent from the default and numerically robust Gurobi
configurations.  Validation is a post-solve certification step and is not used
to improve any incumbent.

| H | Default Gurobi | Best Gurobi | Partial Benders | Difference from best Gurobi |
|---:|---:|---:|---:|---:|
| 20 | 6436.4617 | 6540.4433 | **6572.7110** | **+32.2677 (+0.493%)** |
| 40 | **13622.4156** | **13622.4156** | 13520.2886 | -102.1270 (-0.750%) |
| 80 | 5298.0108 | 13211.4586 | **13262.7287** | **+51.2701 (+0.388%)** |

At H=20, partial Benders converged to a 0.0375% internal gap in 562.7 seconds.
At H=80, the automatic policy spent 240 seconds on a numerically robust primal
MIP warm start, then used the remaining budget for partial Benders and max-flow
master cuts.  The total measured wall time was 899.98 seconds.  H=40 remains a
negative crossover point; the method is not claimed to dominate Gurobi on
every scale.

The full summaries and validated designs are under
`results/regime_ii_algorithm_benchmark_900s/`.  The combined machine-readable
table is `comparison_all_scales.csv`.

## Reproducible commands

Single scale:

```bash
python -m src.experiments.benchmark_regime_ii_algorithms \
  --hub-count 80 \
  --methods gurobi,gurobi_numeric,gurobi_maxflow,partial_benders \
  --time-limit 900 \
  --threads 4
```

Scale grid:

```bash
python -m src.experiments.run_regime_ii_algorithm_grid \
  --hub-counts 5,10,20,40,60,80,100 \
  --methods gurobi,gurobi_numeric,gurobi_maxflow,partial_benders \
  --time-limit 900 \
  --threads 4 \
  --parallel-runs 1
```

The grid is sequential by default so each solve receives the requested CPU
threads without oversubscription.

## Simplified-model ablations

Two explicitly non-equivalent variants are available for article-oriented
ablation.  Their results are stored in separate scale folders.

```bash
# Remove only battery-endurance restrictions; detour/swap routing remains.
python -m src.experiments.benchmark_regime_ii_algorithms \
  --hub-count 40 \
  --model-variant no_battery_endurance \
  --methods gurobi,partial_benders \
  --time-limit 900

# Remove swapping stations and allow every physical arc to fly directly.
python -m src.experiments.benchmark_regime_ii_algorithms \
  --hub-count 40 \
  --model-variant no_swapping_unlimited_direct \
  --methods gurobi,partial_benders \
  --time-limit 900
```

The second variant is intentionally aggressive and should be presented as a
different planning model, not as an exact algorithm for the full model.

## Current conclusions

- Classical Benders with only `x/y/z/Gamma` in the master is too degenerate:
  its upper bound remains orders of magnitude above its incumbent.
- Pure route-recourse partial Benders spends too long generating feasibility
  cuts.  Keeping route feasibility in the master is currently the effective
  formulation.
- Max-flow cuts are useful as a strengthening and warm-start component, but a
  standalone max-flow-cut Gurobi run is not uniformly robust at larger scales.
- Under the formal 900-second full-model comparison, partial Benders beats the
  best tested Gurobi incumbent at H=20 and H=80, while losing at H=40.
- The result supports a scale-dependent hybrid algorithm claim.  It does not
  support a blanket claim that decomposition is superior on every instance.
