"""Optional valid inequalities for the integer base--path design model.

The additional rows preserve every feasible integer-design operating point.
They strengthen fractional facility/assignment relaxations; they do not impose
physical locker/swap capacities or change any path, time, or cost coefficient.
The baseline matrix is retained and the generic primal/dual builders price the
additional rows through the same fixed all-candidate column matrix.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import Instance
from .solver import _Matrix, _matrix


def add_disaggregated_gates(inst: Instance) -> Instance:
    """Append four valid-inequality families in place and return ``inst``.

    Use a separate/copied instance when comparing with the baseline. Repeated
    application is idempotent. M_od=h_od*rho_max; all swap gate incidences below
    indicate whether a visit occurs, whereas the original gates retain their
    visit counts. Both representations forbid every visit to a closed station.
    """
    original = _matrix(inst)
    # The benchmark clears the matrix cache between methods while retaining
    # descriptive metadata. Idempotence must inspect the actual current matrix.
    marker = "disaggregated_activation_and_od_availability"
    if any(name.startswith("vi_od_origin_") for name in original.names):
        return inst
    names, senses, constants, maps, bases = [], [], [], [], []
    ids = {}
    families = dict(od_origin_availability=0, od_origin_availability_lower=0,
                    od_hub_activation=0,
                    od_swap_activation=0, origin_empty_swap_activation=0)
    nstations = inst.n_l+inst.n_k

    def row(name, family, mapping=None, rho=None, constant=0.0):
        ids[name] = len(names)
        names.append(name)
        senses.append("<")
        constants.append(constant)
        maps.append(mapping or {})
        base = np.zeros(2*inst.n_l)
        if rho is not None:
            l, coefficient = rho
            base[l] = coefficient
        bases.append(base)
        families[family] += 1

    for od, demand in enumerate(inst.demand):
        maximum = demand*inst.rho_max
        for l in range(inst.n_l):
            row(f"vi_od_origin_{od}_{l}", "od_origin_availability", rho=(l, -demand))
            row(f"vi_od_origin_lower_{od}_{l}", "od_origin_availability_lower",
                {f"B{od}:{l}:{m}": -maximum for m in range(inst.n_l)},
                rho=(l, demand), constant=maximum)
        for h in range(inst.n_h):
            row(f"vi_od_hub_{od}_{h}", "od_hub_activation", {f"Y{h}": maximum})
        for k in range(nstations):
            row(f"vi_od_swap_{od}_{k}", "od_swap_activation",
                {inst.station_key(k): maximum})
    for l in range(inst.n_l):
        for k in range(nstations):
            row(f"vi_empty_swap_{l}_{k}", "origin_empty_swap_activation",
                {inst.station_key(k): inst.kappa*inst.rho_max})

    sr, sc, sv, er, ec = [], [], [], [], []
    for p, col in enumerate(inst.service):
        od, l = col["od"], col["l"]
        sr.append(ids[f"vi_od_origin_{od}_{l}"]); sc.append(p); sv.append(1.0)
        sr.append(ids[f"vi_od_origin_lower_{od}_{l}"]); sc.append(p); sv.append(-1.0)
        for h in {col["i"], col["j"]}:
            sr.append(ids[f"vi_od_hub_{od}_{h}"]); sc.append(p); sv.append(1.0)
        for k, count in enumerate(col["swaps"]):
            if count:
                sr.append(ids[f"vi_od_swap_{od}_{k}"]); sc.append(p); sv.append(1.0)
    for r, col in enumerate(inst.empty):
        for k, count in enumerate(col["swaps"]):
            if count:
                er.append(ids[f"vi_empty_swap_{col['l']}_{k}"]); ec.append(r)
    service = sparse.coo_matrix((np.asarray(sv), (sr, sc)),
                                shape=(len(names), len(inst.service))).tocsr()
    empty = sparse.coo_matrix((np.ones(len(er)), (er, ec)),
                              shape=(len(names), len(inst.empty))).tocsr()
    inst._solver_matrix = _Matrix(
        names=original.names+names, senses=original.senses+senses,
        constants=np.r_[original.constants, np.asarray(constants)],
        maps=original.maps+maps,
        base=np.vstack((original.base, np.asarray(bases))),
        service=sparse.vstack((original.service, service), format="csr"),
        empty=sparse.vstack((original.empty, empty), format="csr"),
        cost_service=original.cost_service, cost_empty=original.cost_empty,
    )
    inst.metadata["bound_strengthening"] = marker
    inst.metadata["bound_strengthening_rows"] = families
    inst.metadata["bound_strengthening_added_rows"] = len(names)
    inst.metadata["bound_strengthening_scope"] = (
        "Preserves all integer-design feasible operating points; tightens fractional designs; "
        "source-flow rows complete the binary-origin availability product hull; "
        "all original gates retained; no physical capacity introduced")
    return inst
