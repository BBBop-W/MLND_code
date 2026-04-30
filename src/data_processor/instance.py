from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from .. import config

@dataclass
class Instance:
    L: List[str] = field(default_factory=list)  # Physical bases
    H: List[str] = field(default_factory=list)  # Customer-facing hubs
    E: List[str] = field(default_factory=list)  # Swapping stations
    S: List[str] = field(default_factory=list)  # Hub-side holding pools

    N: List[str] = field(default_factory=list)

    # Mapping hubs to holding pools.
    h_to_s: Dict[str, str] = field(default_factory=dict)
    s_to_h: Dict[str, str] = field(default_factory=dict)

    # Arcs (u, v)
    A: Set[Tuple[str, str]] = field(default_factory=set)
    A_HH: Set[Tuple[str, str]] = field(default_factory=set)
    A_LH: Set[Tuple[str, str]] = field(default_factory=set)
    A_HS: Set[Tuple[str, str]] = field(default_factory=set)
    A_SH: Set[Tuple[str, str]] = field(default_factory=set)
    A_SL: Set[Tuple[str, str]] = field(default_factory=set)
    A_SS: Set[Tuple[str, str]] = field(default_factory=set)
    A_phi: Set[Tuple[str, str]] = field(default_factory=set)
    A_F: Set[Tuple[str, str]] = field(default_factory=set)
    
    # Swapping stations covering arc (u, v)
    E_uv: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    # Coordinates and distances
    coords: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    dist_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    time_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict) # travel time
    
    # Demands
    h: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Node properties
    mu: Dict[str, float] = field(default_factory=dict) # service rates
    f: Dict[str, float] = field(default_factory=dict)  # fixed activation costs

    def _dist(self, u: str, v: str) -> float:
        u_idx = self.s_to_h.get(u, u)
        v_idx = self.s_to_h.get(v, v)
        return self.dist_matrix.get((u_idx, v_idx), float('inf'))

    def build_holding_pools(self):
        """Create one hub-side holding pool co-located with each hub."""
        if self.S:
            return

        for h in self.H:
            s = f"S_{h}"
            self.S.append(s)
            self.coords[s] = self.coords[h]

    def build_network(self):
        """
        Build feasible transition and physical-flight arc sets.

        This is instance preprocessing, not Gurobi model construction. It keeps
        candidate locations/demands separate from derived feasibility data such
        as routing options and physical-flight arcs.
        """
        # Reset derived structures so repeated parsing stays consistent.
        self.A.clear()
        self.A_LH.clear()
        self.A_HS.clear()
        self.A_SH.clear()
        self.A_SL.clear()
        self.A_SS.clear()
        self.A_phi.clear()
        self.A_F.clear()
        self.E_uv.clear()
        self.h_to_s.clear()
        self.s_to_h.clear()

        self.build_holding_pools()

        self.N = self.L + self.H + self.S + self.E
        
        # Build mappings
        for h, s in zip(self.H, self.S):
            self.h_to_s[h] = s
            self.s_to_h[s] = h
        
        # Determine valid swapping stations for every potential pair. A contains
        # all arcs the current code may need; A_F records only physical flights.
        all_potential_arcs = set(self.A_HH)
        
        # (L -> H)
        for l in self.L:
            for h in self.H:
                all_potential_arcs.add((l, h))
                
        # Co-located hub/pool handoff arcs.
        for h in self.H:
            s = self.h_to_s[h]
            all_potential_arcs.add((h, s))
            all_potential_arcs.add((s, h))

        for h in self.H:
            s = self.h_to_s[h]
            for h2 in self.H:
                if h != h2:
                    s2 = self.h_to_s[h2]
                    all_potential_arcs.add((s, s2))
            for l in self.L:
                all_potential_arcs.add((s, l))

        # Filter arcs by coverage limits
        for (u, v) in all_potential_arcs:
            dist_uv = self._dist(u, v)
            
            # Find which swapping stations k can cover this arc
            valid_k = []
            for k in self.E:
                # To prevent trivial zero-distance detours like D -> E -> E if D=E, etc.
                # Just check standard detour distance
                dist_uk = self._dist(u, k)
                dist_kv = self._dist(k, v)
                if dist_uk + dist_kv <= config.MAX_DETOUR_DISTANCE:
                    valid_k.append(k)
                    
            can_fly_direct = dist_uv <= config.MAX_DIRECT_DISTANCE
            can_fly_detour = len(valid_k) > 0
            
            # Holding pools are co-located with hubs, so H<->S handoffs are feasible.
            if (u in self.H and v == self.h_to_s.get(u, "")) or (
                u in self.S and self.s_to_h.get(u, "") == v
            ):
                can_fly_direct = True
            
            # If it's feasible to traverse this arc, add it to our valid arc set
            if can_fly_direct or can_fly_detour:
                self.A.add((u, v))
                self.E_uv[(u, v)] = valid_k
                
                # Re-populate specific subsets that survived filtering
                if u in self.L and v in self.H:
                    self.A_LH.add((u, v))
                if u in self.H and v in self.S:
                    self.A_HS.add((u, v))
                if u in self.S and v in self.H:
                    self.A_SH.add((u, v))
                if u in self.S and v in self.L:
                    self.A_SL.add((u, v))
                if u in self.S and v in self.S:
                    self.A_SS.add((u, v))

        # Ensure all A_HH pairs requested exist in A (even if distant, model constraints will force flow=0)
        # But for modeling safety, if it's strictly impossible, we add it to A with E_uv=[]
        for (h1, h2) in self.A_HH:
            if (h1, h2) not in self.A:
                self.A.add((h1, h2))
                self.E_uv[(h1, h2)] = []

        self.A_phi = self.A_LH | self.A_HS | self.A_SH | self.A_SL | self.A_SS
        self.A_F = self.A_HH | self.A_LH | self.A_SL | self.A_SS

    def parse_for_gurobi(self):
        """Backward-compatible wrapper for existing callers."""
        self.build_network()

    def get_dist(self, u: str, v: str) -> float:
        return self._dist(u, v)

    def get_time(self, u: str, v: str) -> float:
        u_idx = self.s_to_h.get(u, u)
        v_idx = self.s_to_h.get(v, v)
        return self.time_matrix.get((u_idx, v_idx), float('inf'))
