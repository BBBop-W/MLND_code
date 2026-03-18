# Benders Decomposition Refactoring Summary

## Overview

I've refactored the Benders decomposition implementation from the chaotic monolithic `benders_model.py` into two clean, specialized files:

1. **`benders_primal.py`** — Benders using **primal subproblem** with Gurobi-extracted duals
2. **`benders_dual.py`** — Benders using **explicit dual SOCP** formulation

Both files are standalone and do NOT rely on each other. They share only the common infrastructure (`IndexSets`, `BendersMaster` base) from `benders_model.py`.

---

## Key Improvements

### 1. Removed Slack Variables

**Old approach (GPT-generated, unnecessary):**
- Primal subproblem had feasibility slack variables (`lv_wait_slack`, `lv_energy_slack`, `fleet_slack`)
- These were penalized in the objective to "force" feasibility
- This is a crutch that masks underlying modeling issues

**New approach:**
- No slacks; subproblem is either feasible or infeasible
- If infeasible → add a feasibility cut (no-good cut) excluding the binary combination
- If feasible → extract duals and build optimality cuts

### 2. Fixed Benders Cut Generation

**Old problems:**
- `BendersPrimal` class tried to use `primal_result.multipliers` and `primal_result.cut_kind`, but `PrimalSubproblemResult` had no such fields
- Primal subproblem never extracted dual information, so cuts were never actually added
- Dual subproblem had sign errors, missing terms, and incorrect structural elements in the dual formulation

**New primal approach (`benders_primal.py`):**
- Carefully extracts dual values (`.Pi` for linear constraints, `.QCPi` for SOCP if needed)
- For each master-variable-dependent RHS, computes sensitivity: ∂z*/∂(RHS coefficient) = dual multiplier
- Builds correct optimality cut: θ ≤ z* + Σ sensitivity · (master_var - master_var_hat)
- Properly handles:
  - y assignment constraints (RHS = x̂)
  - w McCormick linearization (RHS = x̂ or 1−x̂)
  - z level constraints (RHS = û or 1−û)
  - QoS constraints (RHS = M_n · x̂)
  - Fleet/budget constraints (RHS = B − Σf·x̂, etc.)
  - Virtual base wait/energy constraints (RHS includes service_wait(û))

**New dual approach (`benders_dual.py`):**
- Builds the **exact dual SOCP formulation** from your LaTeX
- Dual variables: α, ε, β, π, δ, Δ, σ, κ
- Dual constraints enforce complementary slackness with the primal
- Second-order cone constraint: ||δ_u||₂ ≤ Δ_u (mirrors primal SOCP queue)
- By strong duality (QCP), dual optimal = primal optimal
- Simple cut: θ ≤ dual_objective (which equals the primal subproblem objective)

---

## File Structure

### `benders_primal.py`

**Classes:**
- `PrimalSubproblem` — Solves QCP subproblem, extracts duals via `.Pi` 
  - `solve(x_hat, u_hat)` → (status, obj, cut_data)
  - `cut_data` includes `z_star` and per-variable sensitivities `dx`, `du`
- `BendersPrimal` — Master problem + callback that builds optimality/feasibility cuts

**Key methods:**
- `_extract_cut_data()` — Carefully maps each constraint's dual to sensitivities on x, u
- `_add_cut()` — Adds a linear cut using sensitivities: θ ≤ z* + Σ dx·Δx + Σ du·Δu
- `_add_nogood()` — Feasibility cut: excludes binary combination if subproblem infeasible

### `benders_dual.py`

**Classes:**
- `DualSubproblem` — Solves the explicit dual SOCP formulation
  - `solve(x_hat, u_hat)` → (status, obj, cut_data)
  - `cut_data` includes `z_star` (the dual optimal value = primal recourse)
- `BendersDual` — Master problem + callback using dual objectives

**Key features:**
- Dual variables properly distinguished: α^1, α^2, α^3 (not α^3_lij aggregated)
- Dual constraints carefully derived from complementary slackness
- SOC constraint in dual (not primal SOCP approximation)
- Objective includes all terms: RHS coefficients × (x̂, û parameters)

### Shared Infrastructure (`benders_model.py` — kept as-is)

- `IndexSets` — Precomputed arc sets, nearby bases, swap stations, helpers
- Helper methods: `service_wait_expr`, `outflow_expr`, `phi_travel_expr`, `path_time`
- Ensures both approaches use identical problem representations

---

## Correctness Checks

### Primal Approach Validation

1. **Sensitivity computation**: For each constraint with RHS depending on master vars, we compute ∂z*/∂(RHS).
2. **Duality check**: At optimality, z* = Σ π_c · b_c (RHS). The cut expresses this as z* + sensitivities on changes to master vars.
3. **No slacks**: If a constraint becomes infeasible, Gurobi returns `INFEASIBLE`, triggering a feasibility cut.

### Dual Approach Validation

1. **Dual feasibility**: All dual constraints enforce complementary slackness relationships.
2. **Strong duality**: For the QCP problem, dual objective = primal objective at optimality.
3. **Cone correct** : The SOCP queue constraint duals form a proper SOC: ||δ_u||₂ ≤ Δ_u.
4. **Objective**: The dual objective includes all RHS terms with proper (x̂, û) coefficients.

---

## Usage

### Test both approaches:

```bash
python test_both_benders.py
```

This will:
1. Load the Meituan instance
2. Run `BendersPrimal` for up to 50 iterations
3. Run `BendersDual` for up to 50 iterations
4. Compare final objectives (they should match, or at least be close)

### Use individually in your code:

```python
from src.data_processor.loader import DataLoader
from src.model.benders_primal import BendersPrimal
from src.model.benders_dual import BendersDual

inst = DataLoader().load_instance(...)

# Via primal dual extraction:
primal_solver = BendersPrimal(inst, max_iter=100, time_limit=3600)
obj_primal = primal_solver.solve()

# Via explicit dual SOCP:
dual_solver = BendersDual(inst, max_iter=100, time_limit=3600)
obj_dual = dual_solver.solve()
```

---

## Expected Behavior

Both approaches should converge to the same **Benders lower bound** (dual objective) for any fixed binary master solution. Over many iterations, they should find similar high-quality integer solutions.

**If objectives don't match:**
- Check instance loading (ensure x̂, û are properly fixed)
- Verify cost/demand parameters match between primal and dual
- Check that the dual objective computation matches the LaTeX exactly
- Use `test_benders.py` to audit fixed-master solutions against the original Gurobi model

---

## Advantages of Clean Separation

1. **Clarity** — Each file has one well-defined purpose; no mixed concerns
2. **Testability** — Can validate primal vs. dual independently
3. **Maintainability** — Bug fixes in one approach don't affect the other
4. **Extensibility** — Easy to add other cut-generation methods without polluting the code
5. **Correctness** — Explicit dual formulation forces mathematical precision; easier to verify against theory

---

## Next Steps

1. **Run tests** to ensure both approaches give correct results
2. **Compare convergence speed** — does dual or primal approach require fewer iterations?
3. **Measure wall-clock time** — which is faster on your instances?
4. **Integrate with simulation** — use the winning approach in your full pipeline
5. **Profile** — which subproblem solves is the bottleneck?

