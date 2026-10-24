"""Methods share one formulation, path catalogue and result schema."""

ACCELERATIONS = (
    "pareto_cuts",
    "reuse_columns",
    "courier_savings_bound",
    "initial_design",
    "warm_start",
    "incumbent_cleanup",
    "primal_search",
)

METHODS = {
    "full_path": {"formulation": "finite_path_misocp", "pricing": None},
    "benders_cg": {
        "formulation": "benders_socp",
        "pricing": "finite_catalogue",
        **{name: True for name in ACCELERATIONS},
    },
    "benders_cg_plain": {
        "formulation": "benders_socp",
        "pricing": "finite_catalogue",
        **{name: False for name in ACCELERATIONS},
    },
}


def get_method(name):
    try:
        return dict(name=name, **METHODS[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown method {name!r}; choose {', '.join(METHODS)}"
        ) from exc


def resolve_accelerations(method, overrides=None):
    """Resolve independent Benders switches without changing instance physics."""
    get_method(method)
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("accelerations must be a dict of boolean switches or None")
    unknown = set(overrides) - set(ACCELERATIONS)
    if unknown:
        raise ValueError(
            f"Unknown acceleration: {', '.join(sorted(map(str, unknown)))}"
        )
    for name, value in overrides.items():
        if not isinstance(value, bool):
            raise ValueError(f"Acceleration {name!r} must be boolean")
    if method == "full_path":
        if overrides:
            raise ValueError("full_path does not accept Benders acceleration overrides")
        return {}
    return {**{name: method == "benders_cg" for name in ACCELERATIONS}, **overrides}
