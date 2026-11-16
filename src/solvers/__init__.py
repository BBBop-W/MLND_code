"""Historical Python solver; load Gurobi only when optimization is requested."""


def solve(*args, **kwargs):
    from .api import solve as implementation

    return implementation(*args, **kwargs)

__all__ = ["solve"]
