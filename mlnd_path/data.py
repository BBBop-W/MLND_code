from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instance:
    name: str
    n_l: int
    n_h: int
    n_k: int  # Remote swap stations; global swap ids are 0 .. n_l+n_k-1.
    demand: list[float]
    ground_cost: list[float]
    base_cost: list[float]
    hub_cost: list[float]
    swap_cost: list[float]  # Remote construction costs only.
    rho_max: float = 0.90
    kappa: float = 30.0
    fleet_cost: float = 3.0
    service: list[dict[str, Any]] = field(default_factory=list)
    empty: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_od(self):
        return len(self.demand)

    @property
    def c0(self):
        return sum(h * c for h, c in zip(self.demand, self.ground_cost))

    @property
    def M(self):
        return self.rho_max * sum(self.demand)

    @property
    def ME(self):
        return self.rho_max * self.n_l * self.kappa

    @property
    def MK(self):
        return 4 * self.M + 2 * self.ME

    def design_keys(self):
        return ([f'X{l}' for l in range(self.n_l)]
                + [f'Y{i}' for i in range(self.n_h)]
                + [f'S{k}' for k in range(self.n_l, self.n_l + self.n_k)]
                + [f'B{o}:{l}:{m}' for o in range(self.n_od)
                   for l in range(self.n_l) for m in range(self.n_l)])

    def zero_design(self):
        return dict.fromkeys(self.design_keys(), 0.0)

    def fixed_cost(self, design):
        return (sum(c * design[f'X{l}'] for l, c in enumerate(self.base_cost))
                + sum(c * design[f'Y{i}'] for i, c in enumerate(self.hub_cost))
                + sum(c * design[f'S{k+self.n_l}'] for k, c in enumerate(self.swap_cost)))

    def station_key(self, k):
        return f'X{k}' if k < self.n_l else f'S{k}'

    def service_cost(self, p):
        c = self.service[p]
        return c['cost'] - self.ground_cost[c['od']] + self.fleet_cost * c['time']

    def empty_cost(self, r):
        c = self.empty[r]
        return c['cost'] + self.fleet_cost * c['time']


# Service column: od,l,m,i,j (integer ids), time (hours), cost (currency/task),
# swaps (tuple of visit counts indexed by all base+remote swap facilities),
# optional route/energy trace fields. Empty column: l,m,time,cost,swaps.
# All candidate columns are fixed before facility decisions; no open-only pricing.
