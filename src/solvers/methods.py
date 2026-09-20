"""The five experimental methods and their explicit acceleration settings."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Accelerations:
    pareto_cuts: bool = True
    initial_bound: bool = True
    initial_design: bool = True
    reuse_columns: bool = True
    warm_start: bool = True
    incumbent_cleanup: bool = True
    primal_search: bool = True

    @classmethod
    def disabled(cls):
        return cls(**{key: False for key in cls.__dataclass_fields__})

    def benders_options(self):
        return {
            key: value for key, value in asdict(self).items() if key != "primal_search"
        }


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    formulation: str
    pricing: str | None = None
    accelerations: Accelerations | None = None
    verification: str = "numerical checks pending after restructuring"

    def describe(self):
        return {
            **asdict(self),
            "benders_cuts": self.formulation == "benders",
            "extra_valid_inequalities": [],
            "gurobi_internal_cuts": "solver default, common across methods",
        }


METHODS = {
    "benders_cg": Method(
        "benders_cg", "Full proposed Benders-CG", "benders", "label", Accelerations()
    ),
    "arc": Method("arc", "Arc-based MISOCP", "arc"),
    "full_path": Method("full_path", "Full-path direct MISOCP", "path"),
    "benders_cg_solver": Method(
        "benders_cg_solver",
        "Benders-CG with solver pricing",
        "benders",
        "solver",
        Accelerations(),
    ),
    "benders_cg_plain": Method(
        "benders_cg_plain",
        "Benders-CG with label pricing, no accelerations",
        "benders",
        "label",
        Accelerations.disabled(),
    ),
}


def get_method(name, accelerations=None):
    """Resolve a method; optional Boolean overrides support later ablations."""
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {', '.join(METHODS)}")
    method = METHODS[name]
    if accelerations is None:
        return method
    if method.accelerations is None:
        raise ValueError(f"{name} does not use Benders acceleration switches")
    changes = (
        asdict(accelerations)
        if isinstance(accelerations, Accelerations)
        else accelerations
    )
    if not isinstance(changes, dict):
        raise ValueError("Acceleration overrides must be a dictionary or Accelerations")
    for key, value in changes.items():
        if key not in Accelerations.__dataclass_fields__ or not isinstance(value, bool):
            raise ValueError(f"Unknown or non-Boolean acceleration: {key}")
    return replace(method, accelerations=replace(method.accelerations, **changes))
