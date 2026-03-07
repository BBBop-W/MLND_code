from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from .. import config

@dataclass
class Instance:
    # Physical sets (using string identifiers like 'L1', 'P1', etc.)
    L_hat: List[str] = field(default_factory=list) # Physical bases
    P: List[str] = field(default_factory=list)     # Pickup stations
    D: List[str] = field(default_factory=list)     # Delivery stations
    E: List[str] = field(default_factory=list)     # Swapping stations
    
    # Virtual bases (one per delivery node)
    L_v: List[str] = field(default_factory=list)
    
    # Combined Sets
    L: List[str] = field(default_factory=list)
    N: List[str] = field(default_factory=list)

    # Mapping delivery nodes to virtual bases
    d_to_lv: Dict[str, str] = field(default_factory=dict)
    lv_to_d: Dict[str, str] = field(default_factory=dict)

    # Arcs (u, v)
    A: Set[Tuple[str, str]] = field(default_factory=set)
    A_PD: Set[Tuple[str, str]] = field(default_factory=set)
    A_LP: Set[Tuple[str, str]] = field(default_factory=set)
    A_DL: Set[Tuple[str, str]] = field(default_factory=set)
    
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

    def __post_init__(self):
        pass

    def _dist(self, u: str, v: str) -> float:
        # Internal helper to safely get distance between physical node coordinates
        u_idx = self.lv_to_d.get(u, u)
        v_idx = self.lv_to_d.get(v, v)
        return self.dist_matrix.get((u_idx, v_idx), float('inf'))

    def parse_for_gurobi(self):
        """
        Organize data into structures easily consumable by Gurobi.
        Filters arcs based on maximum coverage distance.
        """
        self.L = self.L_hat + self.L_v
        self.N = self.L_hat + self.P + self.D + self.E
        
        # Build mappings
        for d, lv in zip(self.D, self.L_v):
            self.d_to_lv[d] = lv
            self.lv_to_d[lv] = d
        
        # Determine valid swapping stations for every potential pair
        # We need A to contain arcs where direct flight is possible OR detour flight is possible
        all_potential_arcs = set(self.A_PD)
        
        # (L -> P)
        for l in self.L:
            for p in self.P:
                all_potential_arcs.add((l, p))
                
        # (D -> L_hat)
        for d in self.D:
            for l_hat in self.L_hat:
                all_potential_arcs.add((d, l_hat))
                
        # (D -> E)
        for d in self.D:
            for e in self.E:
                all_potential_arcs.add((d, e))
                
        # (E -> L_hat)
        for e in self.E:
            for l_hat in self.L_hat:
                all_potential_arcs.add((e, l_hat))
                
        # (D -> L_v) chaining link
        for d in self.D:
            lv = self.d_to_lv[d]
            all_potential_arcs.add((d, lv))

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
            
            # Virtual bases are co-located with delivery nodes, so D -> L_v is dist 0
            if u in self.D and v == self.d_to_lv.get(u, ""):
                can_fly_direct = True
            
            # If it's feasible to traverse this arc, add it to our valid arc set
            if can_fly_direct or can_fly_detour:
                self.A.add((u, v))
                self.E_uv[(u, v)] = valid_k
                
                # Re-populate specific subsets that survived filtering
                if (u, v) in self.A_PD:
                    # It's already in A_PD from generation, but we should ensure only feasible ones stay?
                    # Wait, A_PD was generated based on demands. If an OD pair is unreachable, 
                    # we should probably just keep it in A_PD but it won't have flow.
                    pass 
                if u in self.L and v in self.P:
                    self.A_LP.add((u, v))
                if u in self.D and v in self.L:
                    self.A_DL.add((u, v))

        # Ensure all A_PD pairs requested exist in A (even if distant, model constraints will force flow=0)
        # But for modeling safety, if it's strictly impossible, we add it to A with E_uv=[]
        for (p, d) in self.A_PD:
            if (p, d) not in self.A:
                self.A.add((p, d))
                self.E_uv[(p, d)] = []

    def get_dist(self, u: str, v: str) -> float:
        return self._dist(u, v)

    def get_time(self, u: str, v: str) -> float:
        u_idx = self.lv_to_d.get(u, u)
        v_idx = self.lv_to_d.get(v, v)
        return self.time_matrix.get((u_idx, v_idx), float('inf'))
