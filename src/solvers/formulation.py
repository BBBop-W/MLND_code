"""One finite-path MISOCP and its fixed-design SOCP, with a common matrix.

Every first segment and task-demand row is retained even in a restricted path
model. Aggregate policy pools branch proportions over first segments; prefix
policy imposes them separately. Only complete-path columns may be omitted.
"""

# ruff: noqa: E741 -- l and m denote departure and return bases.
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model.types import MODEL_VERSION
from src.model.assignment import assignment_shape, assignment_scope, assignment_pools
from src.model.locker_capacity import poisson_mean_threshold

DESIGN_NAMES = ("facility", "capacity_level", "assignment", "airline")
MAX_LIFTED_VARIABLES = 2_000_000


class BuildStopped(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def shapes(data):
    return dict(
        facility=(data.n_nodes,),
        capacity_level=data.slot_counts.shape,
        assignment=assignment_shape(data),
        airline=(data.n_nodes, data.n_nodes),
    )


def validate_design_arrays(data, design, *, source="fixed_design"):
    """Require the current binary design shape before fixing or warm-starting.

    In particular, a historical K=1 base-pair assignment is not silently
    squeezed or aggregated into the new departure-assignment model.
    """
    if design.get("assignment_scope", assignment_scope(data)) != assignment_scope(data):
        raise ValueError(f"{source}.assignment_scope disagrees with the current model")
    values = {}
    for name, shape in shapes(data).items():
        if name not in design:
            raise ValueError(f"{source}.{name} is missing")
        array = np.asarray(design[name], dtype=float)
        if (
            array.shape != shape
            or not np.isfinite(array).all()
            or np.any((array != 0) & (array != 1))
        ):
            raise ValueError(f"{source}.{name} must be binary with shape {shape}")
        values[name] = array
    return values


def setup(model, threads=1, seed=43, mip_gap=1e-4):
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Seed = seed
    model.Params.NonConvex = 0
    model.Params.FeasibilityTol = 1e-8
    model.Params.IntFeasTol = 1e-9
    model.Params.OptimalityTol = 1e-8
    model.Params.BarQCPConvTol = 1e-9
    model.Params.MIPGap = mip_gap


def add_design(model, data, binary=True, check=None):
    check = check or (lambda: None)
    check()
    variables = {
        name: model.addVars(
            *shape,
            lb=0,
            ub=1,
            vtype=GRB.BINARY if binary else GRB.CONTINUOUS,
            name=name,
        )
        for name, shape in shapes(data).items()
    }
    x, z, a, y = (variables[name] for name in DESIGN_NAMES)
    for i in range(data.n_h):
        model.addConstr(z.sum(i, "*") == x[data.n_l + i], name=f"install_{i}")
    for r in range(data.n_od):
        check()
        model.addConstr(
            a.sum(r, *(["*"] * (len(assignment_shape(data)) - 1))) <= 1,
            name=f"assign_{r}",
        )
        for pool in assignment_pools(data):
            for base in pool:
                model.addConstr(a[r, *pool] <= x[base])
    for u in range(data.n_nodes):
        check()
        for v in range(data.n_nodes):
            model.addConstr(y[u, v] <= x[u])
            model.addConstr(y[u, v] <= x[v])
            if not data.allowed_links[u, v]:
                model.addConstr(y[u, v] == 0)
    strategy = (
        gp.quicksum(data.base_cost[l] * x[l] for l in range(data.n_l))
        + gp.quicksum(
            data.slot_cost[i, s] * z[i, s]
            for i, s in np.ndindex(data.slot_counts.shape)
        )
        + gp.quicksum(
            data.link_cost[u, v] * y[u, v] for u, v in np.ndindex(data.link_cost.shape)
        )
    )
    return variables, strategy


def activation_bounds(data, paths):
    """Complete-catalogue traversal bounds; never recomputed on an RMP subset."""
    service = np.zeros((data.n_nodes, data.n_nodes))
    empty = service.copy()
    for p in paths.service_paths:
        for u, v, count in p.arc_counts:
            service[u, v] = max(service[u, v], count)
    for p in paths.empty_paths:
        for u, v, count in p.arc_counts:
            empty[u, v] = max(empty[u, v], count)
    return float(data.demand.sum()) * (service + (data.n_l - 1) * empty)


@dataclass
class BuiltModel:
    model: object
    data: object
    paths: object
    variables: dict
    constraints: dict
    cones: list
    strategy_cost: object
    continuous_cost: object
    service_indices: tuple
    empty_indices: tuple
    segment_indices: dict


def build_model(
    data,
    paths,
    *,
    fixed_design=None,
    service_indices=None,
    empty_indices=None,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    time_limit=None,
    stop_requested=None,
    max_lifted_variables=MAX_LIFTED_VARIABLES,
):
    """Build without solving. All coefficients are independent of fixed_design."""
    started = time.perf_counter()

    def check():
        if stop_requested and stop_requested():
            raise BuildStopped("user_stop")
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            raise BuildStopped("time_limit")

    check()
    if not paths.complete:
        raise ValueError("The solver requires a complete finite path catalogue")
    _validate_catalogue(data, paths, check)
    P = (
        tuple(range(len(paths.service_paths)))
        if service_indices is None
        else tuple(sorted(set(service_indices)))
    )
    K = (
        tuple(range(len(paths.empty_paths)))
        if empty_indices is None
        else tuple(sorted(set(empty_indices)))
    )
    if any(i < 0 or i >= len(paths.service_paths) for i in P) or any(
        i < 0 or i >= len(paths.empty_paths) for i in K
    ):
        raise ValueError("Restricted columns must belong to the complete catalogue")
    groups = {}
    for j, segment in enumerate(paths.initial_segments):
        key = (
            (*segment.b, segment.first_od) if data.flow_policy == "aggregate" else (j,)
        )
        groups.setdefault(key, []).append(j)
    lift_groups = tuple(groups.values())
    active_lifts = data.effective_delta > 0
    pooled = data.flow_policy == "aggregate"
    estimate = (
        sum(int(np.prod(shape)) for shape in shapes(data).values())
        + len(paths.initial_segments)
        + (len(lift_groups) if pooled else 0)
        + (len(lift_groups) * (3 * data.n_od + 2) if active_lifts else 0)
        + int(np.prod(assignment_shape(data)))
        + len(P)
        + len(K)
        + 5 * data.n_l
        + data.n_od
        + 1
    )
    if max_lifted_variables is not None and (
        isinstance(max_lifted_variables, bool) or max_lifted_variables < 1
    ):
        raise ValueError("max_lifted_variables must be positive or None")
    if max_lifted_variables is not None and estimate > max_lifted_variables:
        raise BuildStopped("model_size_limit")
    model = gp.Model("consecutive")
    setup(model, threads, seed, mip_gap)
    try:
        variables, strategy = add_design(model, data, fixed_design is None, check)
        x, z, a, y = (variables[name] for name in DESIGN_NAMES)
        fix = {}
        if fixed_design is not None:
            for name, values in validate_design_arrays(data, fixed_design).items():
                for key, var in variables[name].items():
                    fix[name, key] = model.addConstr(
                        var == float(values[key]), name=f"fix_{name}_{key}"
                    )
        L, H, R = data.n_l, data.n_h, data.n_od
        total = float(data.demand.sum())
        baseline = float(data.demand @ data.ground_cost)
        fleet_bound = baseline / data.fleet_cost
        segments = paths.initial_segments
        rho = variables["rho"] = model.addVars(L, ub=1, name="rho")
        idle = variables["idle"] = model.addVars(L, ub=fleet_bound, name="idle")
        fleet = variables["fleet"] = model.addVar(ub=fleet_bound, name="fleet")
        head = variables["head"] = model.addVars(len(segments), name="head")
        v = variables["v"] = model.addVars(*assignment_shape(data), ub=1, name="v")
        courier = variables["courier"] = model.addVars(R, name="courier")
        flow = variables["flow"] = model.addVars(P, ub=total, name="flow")
        empty = variables["empty_flow"] = model.addVars(K, ub=total, name="empty_flow")
        G = len(lift_groups)
        eta = variables["eta"] = (
            model.addVars(G, name="eta") if active_lifts else gp.tupledict()
        )
        u = variables["u"] = (
            model.addVars(G, R, name="u") if active_lifts else gp.tupledict()
        )
        chain = variables["chain"] = (
            model.addVars(G, R + 1, name="Z") if active_lifts else gp.tupledict()
        )
        product = variables["chain_product"] = (
            model.addVars(G, R, name="Y") if active_lifts else gp.tupledict()
        )
        group_head = variables["group_head"] = (
            model.addVars(G, name="group_head") if pooled else gp.tupledict()
        )
        variables["_lift_groups"] = lift_groups
        variables["_pooled"] = pooled
        variables["_active_lifts"] = active_lifts
        by_pool = {b: [] for b in assignment_pools(data)}
        branch_single, branch_double, executed_head = {}, {}, {}
        cones = []
        for l in range(L):
            check()
            model.addConstr(rho[l] <= x[l])
            up = model.addVar(ub=fleet_bound + 1, name=f"cone_up_{l}")
            down = model.addVar(ub=1, name=f"cone_down_{l}")
            one = model.addVar(lb=1, ub=1, name=f"cone_one_{l}")
            model.addConstr(up == idle[l] + 1)
            model.addConstr(down == 1 - rho[l])
            model.addQConstr(one * one <= up * down, name=f"idle_cone_{l}")
            cones.append((up, down, one))
        for r in range(R):
            check()
            courier[r].UB = float(data.demand[r])
            for pool in assignment_pools(data):
                model.addConstr(v[r, *pool] <= rho[pool[0]])
                model.addConstr(v[r, *pool] <= a[r, *pool])
                model.addConstr(v[r, *pool] >= rho[pool[0]] + a[r, *pool] - 1)
        for j, segment in enumerate(segments):
            check()
            head[j].UB = float(data.demand[segment.first_od])
            if pooled:
                executed_head[j] = model.addConstr(
                    head[j] == 0, name=f"executed_head_{j}"
                )
        heads = {}
        for g, indices in enumerate(lift_groups):
            check()
            segment = segments[indices[0]]
            l, m = segment.b
            by_pool[(l,) if data.max_orders == 1 else segment.b].append(g)
            h = float(data.demand[segment.first_od])
            if pooled:
                group_head[g].UB = h
                model.addConstr(
                    group_head[g] == gp.quicksum(head[j] for j in indices),
                    name=f"group_head_{g}",
                )
                heads[g] = group_head[g]
            else:
                heads[g] = head[indices[0]]
            if active_lifts:
                upper = min(1.0, data.effective_delta * h)
                eta[g].UB = upper
                model.addConstr(eta[g] <= data.effective_delta * heads[g])
                model.addConstr(chain[g, 0] == heads[g])
                chain[g, 0].UB = h
                for r in range(R):
                    u[g, r].UB = upper
                    model.addConstr(u[g, r] <= eta[g])
                    model.addConstr(u[g, r] <= upper * a[r, l, m])
                    model.addConstr(u[g, r] >= eta[g] - upper * (1 - a[r, l, m]))
                    chain[g, r + 1].UB = h
                    product[g, r].UB = h
                    model.addConstr(product[g, r] <= chain[g, r])
                    model.addConstr(product[g, r] <= h * a[r, l, m])
                    model.addConstr(product[g, r] >= chain[g, r] - h * (1 - a[r, l, m]))
                    model.addConstr(
                        chain[g, r + 1]
                        == chain[g, r]
                        + float(np.expm1(-data.effective_delta * data.demand[r]))
                        * product[g, r]
                    )
                model.addConstr(
                    gp.quicksum(data.demand[r] * u[g, r] for r in range(R))
                    == heads[g] - chain[g, R],
                    name=f"eta_link_{g}",
                )
            first = segment.first_od
            suffix = "_".join(
                map(str, (*segment.b, first) if pooled else (indices[0],))
            )
            singles = heads[g]
            if active_lifts:
                singles = singles - gp.quicksum(
                    data.kappa[first, r] * data.demand[r] * u[g, r] for r in range(R)
                )
            single_row = model.addConstr(singles == 0, name=f"single_{suffix}")
            for j in indices:
                branch_single[j] = single_row
            if active_lifts:
                for r in range(R):
                    double_row = model.addConstr(
                        data.kappa[first, r] * data.demand[r] * u[g, r] == 0,
                        name=f"double_{suffix}_{r}",
                    )
                    for j in indices:
                        branch_double[j, r] = double_row
        for b, group_indices in by_pool.items():
            for r in range(R):
                check()
                expression = gp.quicksum(
                    heads[g]
                    for g in group_indices
                    if segments[lift_groups[g][0]].first_od == r
                )
                if active_lifts:
                    expression += data.demand[r] * gp.quicksum(
                        data.kappa[segments[lift_groups[g][0]].first_od, r] * u[g, r]
                        for g in group_indices
                    )
                model.addConstr(
                    expression == data.demand[r] * v[r, *b],
                    name="orders_" + "_".join(map(str, (r, *b))),
                )
        order = {
            r: model.addConstr(courier[r] == data.demand[r], name=f"demand_{r}")
            for r in range(R)
        }
        balance = {
            l: model.addConstr(gp.LinExpr() == 0, name=f"balance_{l}") for l in range(L)
        }
        fleet_row = model.addConstr(
            gp.quicksum(idle.values()) - fleet <= 0, name="fleet_use"
        )
        capacity = {
            i: model.addConstr(
                -gp.quicksum(
                    poisson_mean_threshold(
                        int(data.slot_counts[i, s]), float(data.slot_epsilon[i])
                    )
                    * z[i, s]
                    for s in range(data.slot_counts.shape[1])
                )
                <= 0,
                name=f"capacity_{i}",
            )
            for i in range(H)
        }
        bound = activation_bounds(data, paths)
        links = {
            (s, t): model.addConstr(-bound[s, t] * y[s, t] <= 0, name=f"link_{s}_{t}")
            for s, t in np.ndindex(bound.shape)
        }
        rows = dict(
            design_fix=fix,
            single=branch_single,
            double=branch_double,
            executed_head=executed_head,
            order=order,
            balance=balance,
            fleet=fleet_row,
            capacity=capacity,
            link=links,
        )
        built = BuiltModel(
            model,
            data,
            paths,
            variables,
            rows,
            cones,
            strategy,
            None,
            P,
            K,
            {segment.id: j for j, segment in enumerate(segments)},
        )
        model.update()
        for index in P:
            check()
            for row, value in path_column(built, paths.service_paths[index]):
                model.chgCoeff(row, flow[index], value)
        for index in K:
            check()
            for row, value in path_column(built, paths.empty_paths[index], empty=True):
                model.chgCoeff(row, empty[index], value)
        cost = (
            data.fleet_cost * fleet
            + gp.quicksum(paths.service_paths[i].cost * flow[i] for i in P)
            + gp.quicksum(paths.empty_paths[i].cost * empty[i] for i in K)
            + gp.quicksum(data.ground_cost[r] * courier[r] for r in range(R))
        )
        built.continuous_cost = cost
        model.setObjective(cost + (strategy if fixed_design is None else 0))
        model.update()
        return built
    except BaseException:
        model.dispose()
        raise


def _validate_catalogue(data, paths, check):
    """Reject forged resource coefficients before any optimization starts."""
    from src.model.resources import build_initial_segment, rebuild_path

    segments = {s.id: s for s in paths.initial_segments}
    if len(segments) != len(paths.initial_segments):
        raise ValueError("Initial segment IDs must be unique")
    for segment in paths.initial_segments:
        check()
        rebuilt = build_initial_segment(
            data,
            segment.b,
            segment.first_od,
            segment.pickup,
            segment.delivery,
            segment.legs,
            id=segment.id,
        )
        if rebuilt != segment:
            raise ValueError(
                "Initial segment coefficients differ from its physical routes"
            )
    for collection in (paths.service_paths, paths.empty_paths):
        if len({p.id for p in collection}) != len(collection):
            raise ValueError("Path IDs must be unique within each path class")
        for path in collection:
            check()
            if rebuild_path(data, path) != path:
                raise ValueError(
                    "Path coefficients differ from physical resource reconstruction"
                )
    for path in paths.service_paths:
        check()
        segment = segments.get(path.segment_id)
        if (
            segment is None
            or path.b != segment.b
            or path.first_od != segment.first_od
            or path.legs[:2] != segment.legs
        ):
            raise ValueError("A service path must match its retained initial segment")


def path_column(built, path, empty=False):
    """Complete scalar column; used identically by construction and pricing."""
    rows, values = built.constraints, []
    if not empty:
        index = built.segment_indices[path.segment_id]
        branch = (
            rows["single"][index]
            if path.mode == "single"
            else rows["double"][index, path.second_od]
        )
        values.append((branch, -1.0))
        if index in rows["executed_head"]:
            values.append((rows["executed_head"][index], -1.0))
        values.extend((rows["order"][r], float(n)) for r, n in path.served_counts)
        values.extend((rows["capacity"][i], float(t)) for i, t in path.slot_times)
    l, m = path.b
    if l != m:
        values.extend(((rows["balance"][l], 1.0), (rows["balance"][m], -1.0)))
    values.append((rows["fleet"], path.time))
    values.extend((rows["link"][u, v], float(n)) for u, v, n in path.arc_counts)
    return values


def add_path_columns(built, service_indices=(), empty_indices=(), check=None):
    """Extend one validated fixed-matrix RMP without rebuilding its lifted rows.

    Only members of the already physically checked, complete catalogue can be
    added. The same path_column function supplies construction and pricing.
    """
    if not built.constraints["design_fix"]:
        raise ValueError("Incremental columns require a fixed-design subproblem")
    check = check or (lambda: None)
    total = float(built.data.demand.sum())
    service_indices, empty_indices = tuple(service_indices), tuple(empty_indices)
    pending = sum(
        len(set(indices) - set(built.variables[name]))
        for indices, name in ((service_indices, "flow"), (empty_indices, "empty_flow"))
    )
    if built.model.NumVars + pending > MAX_LIFTED_VARIABLES:
        raise BuildStopped("model_size_limit")
    added = 0
    for empty, indices, catalogue, name in (
        (False, service_indices, built.paths.service_paths, "flow"),
        (True, empty_indices, built.paths.empty_paths, "empty_flow"),
    ):
        for index in sorted(set(indices)):
            check()
            if not 0 <= index < len(catalogue):
                raise ValueError("Added columns must belong to the complete catalogue")
            if index in built.variables[name]:
                continue
            path = catalogue[index]
            entries = path_column(built, path, empty=empty)
            variable = built.model.addVar(
                ub=total,
                name=f"{name}[{index}]",
                column=gp.Column(
                    [value for _, value in entries], [row for row, _ in entries]
                ),
            )
            built.variables[name][index] = variable
            built.continuous_cost.addTerms(path.cost, variable)
            added += 1
    built.service_indices = tuple(sorted(built.variables["flow"]))
    built.empty_indices = tuple(sorted(built.variables["empty_flow"]))
    built.model.setObjective(built.continuous_cost)
    built.model.update()
    return added


def read_solution(built, *, value=None):
    """Reconstruct a solution from model values or a callback value accessor."""
    accessor = value or (lambda variable: variable.X)

    def expression_value(expression):
        if value is None:
            return float(expression.getValue())
        return float(expression.getConstant()) + sum(
            expression.getCoeff(i) * accessor(expression.getVar(i))
            for i in range(expression.size())
        )

    data, paths, variables = built.data, built.paths, built.variables
    solution = {}
    for name, shape in shapes(data).items():
        array = np.zeros(shape)
        for key, var in variables[name].items():
            array[key] = round(accessor(var))
        solution[name] = array
    for name in ("rho", "idle", "head", "courier"):
        solution[name] = np.array(
            [accessor(variables[name][i]) for i in range(len(variables[name]))]
        )
    solution["eta"] = np.zeros(len(paths.initial_segments))
    if variables["_active_lifts"]:
        for g, indices in enumerate(variables["_lift_groups"]):
            if variables["_pooled"]:
                total_head = sum(solution["head"][j] for j in indices)
                if total_head > 0:
                    for j in indices:
                        solution["eta"][j] = (
                            solution["head"][j]
                            * accessor(variables["eta"][g])
                            / total_head
                        )
            else:
                solution["eta"][indices[0]] = accessor(variables["eta"][g])
    solution["fleet"] = np.array(accessor(variables["fleet"]))
    solution["flow"] = np.array(
        [
            accessor(variables["flow"][i]) if i in variables["flow"] else 0.0
            for i in range(len(paths.service_paths))
        ]
    )
    solution["empty_flow"] = np.array(
        [
            accessor(variables["empty_flow"][i])
            if i in variables["empty_flow"]
            else 0.0
            for i in range(len(paths.empty_paths))
        ]
    )
    return dict(
        model_version=MODEL_VERSION,
        flow_policy=data.flow_policy,
        assignment_scope=assignment_scope(data),
        objective=float(
            expression_value(built.strategy_cost)
            + expression_value(built.continuous_cost)
        ),
        continuous_cost=float(expression_value(built.continuous_cost)),
        opening_cost=float(expression_value(built.strategy_cost)),
        _solution=solution,
        _segments=[asdict(s) for s in paths.initial_segments],
        _missions=[
            dict(path=dict(asdict(p), id=i), flow=float(solution["flow"][i]))
            for i, p in enumerate(paths.service_paths)
        ],
        _relocations=[
            dict(path=dict(asdict(p), id=i), flow=float(solution["empty_flow"][i]))
            for i, p in enumerate(paths.empty_paths)
        ],
    )
