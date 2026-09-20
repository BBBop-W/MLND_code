"""Five comparison methods behind one solve interface."""

from .methods import METHODS, Accelerations, get_method


def solve(data, **options):
    """Import optimization code only when an actual solve is requested."""
    from .api import solve as run

    return run(data, **options)


__all__ = ["solve", "METHODS", "Accelerations", "get_method"]
