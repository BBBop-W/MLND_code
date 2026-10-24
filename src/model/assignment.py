"""Strategic assignment indexing, separate from physical path endpoints.

New K=1 models assign an OD to its departure base. K=2 retains base-pair
assignment. Historical K=1 base-pair solutions may be read and audited, but
are not silently converted into designs for the new optimization model.
"""

from __future__ import annotations

import numpy as np


def assignment_scope(data):
    return "origin" if data.max_orders == 1 else "base_pair"


def assignment_shape(data):
    return (
        (data.n_od, data.n_l)
        if data.max_orders == 1
        else (data.n_od, data.n_l, data.n_l)
    )


def assignment_pools(data):
    return (
        tuple((base,) for base in range(data.n_l))
        if data.max_orders == 1
        else data.base_pairs
    )


def assignment_key(data, od, base_pair):
    return (od, base_pair[0]) if data.max_orders == 1 else (od, *base_pair)


def saved_assignment_scope(data, result):
    """Validate saved semantics, including historical three-dimensional K=1."""
    shape = np.asarray(result["_solution"]["assignment"]).shape
    if shape == (data.n_od, data.n_l) and data.max_orders == 1:
        inferred = "origin"
    elif shape == (data.n_od, data.n_l, data.n_l):
        inferred = "base_pair"
    else:
        raise ValueError("Invalid saved assignment shape for max_orders")
    explicit = result.get("assignment_scope")
    if explicit is not None and explicit != inferred:
        raise ValueError("Saved assignment_scope disagrees with assignment shape")
    return inferred
