from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from .. import config

@dataclass
class Instance:
    # Physical sets (using string identifiers like 'L1', 'P1', etc.)
    L: List[str] = field(default_factory=list) # Physical bases
    H: List[str] = field(default_factory=list)     # Hub stations
    E: List[str] = field(default_factory=list)     # Swapping stations
    
    # Virtual bases (one per delivery node)
    L_v: List[str] = field(default_factory=list)
    
    # Combined Sets
    L_tilde: List[str] = field(default_factory=list)
    N: List[str] = field(default_factory=list)

    # Mapping delivery nodes to virtual bases
    h_to_lv: Dict[str, str] = field(default_factory=dict)
    lv_to_h: Dict[str, str] = field(default_factory=dict)

    # Arcs (u, v)
    A: Set[Tuple[str, str]] = field(default_factory=set)
    A_HH: Set[Tuple[str, str]] = field(default_factory=set)
    A_LH: Set[Tuple[str, str]] = field(default_factory=set)
    A_HL: Set[Tuple[str, str]] = field(default_factory=set)
    A_LL: Set[Tuple[str, str]] = field(default_factory=set)
    
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
        u_idx = self.lv_to_h.get(u, u)
        v_idx = self.lv_to_h.get(v, v)
        return self.dist_matrix.get((u_idx, v_idx), float('inf'))

    def parse_for_gurobi(self):
        """
        Organize data into structures easily consumable by Gurobi.
        Filters arcs based on maximum coverage distance.
        """
        # Reset derived structures so repeated parsing stays consistent.
        self.A.clear()
        self.A_LH.clear()
        self.A_HL.clear()
        self.A_LL.clear()
        self.E_uv.clear()
        self.h_to_lv.clear()
        self.lv_to_h.clear()

        self.L_tilde = self.L + self.L_v
        self.N = self.L_tilde + self.H + self.E
        
        # Build mappings
        for h, lv in zip(self.H, self.L_v):
            self.h_to_lv[h] = lv
            self.lv_to_h[lv] = h
        
        # Determine valid swapping stations for every potential pair
        # We need A to contain arcs where direct flight is possible OR detour flight is possible
        all_potential_arcs = set(self.A_HH)
        
        # (L -> H)
        for l in self.L:
            for h in self.H:
                all_potential_arcs.add((l, h))
                
        # (H -> L_v -> H)
        for h in self.H:
            lv = self.h_to_lv[h]
            all_potential_arcs.add((h, lv))
            all_potential_arcs.add((lv, h))

        for h in self.H:
            lv = self.h_to_lv[h]
            for h2 in self.H:
                if h != h2:
                    lv2 = self.h_to_lv[h2]
                    all_potential_arcs.add((lv, lv2))
            for l in self.L:
                all_potential_arcs.add((lv, l))
                # all_potential_arcs.add((l, lv))
                

            

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
            if u in self.H and v == self.h_to_lv.get(u, ""):
                can_fly_direct = True
            
            # If it's feasible to traverse this arc, add it to our valid arc set
            if can_fly_direct or can_fly_detour:
                self.A.add((u, v))
                self.E_uv[(u, v)] = valid_k
                
                # Re-populate specific subsets that survived filtering
                if (u, v) in self.A_HH:
                    # It's already in A_HH from generation, but we should ensure only feasible ones stay?
                    # Wait, A_HH was generated based on demands. If an OD pair is unreachable, 
                    # we should probably just keep it in A_HH but it won't have flow.
                    pass 
                if u in self.L and v in self.H:
                    self.A_LH.add((u, v))
                if u in self.H and v in self.L:
                    self.A_HL.add((u, v))
                if u in self.L_tilde and v in self.L_tilde:
                    self.A_LL.add((u, v))

        # Ensure all A_HH pairs requested exist in A (even if distant, model constraints will force flow=0)
        # But for modeling safety, if it's strictly impossible, we add it to A with E_uv=[]
        for (h1, h2) in self.A_HH:
            if (h1, h2) not in self.A:
                self.A.add((h1, h2))
                self.E_uv[(h1, h2)] = []

    def get_dist(self, u: str, v: str) -> float:
        return self._dist(u, v)

    def get_time(self, u: str, v: str) -> float:
        u_idx = self.lv_to_h.get(u, u)
        v_idx = self.lv_to_h.get(v, v)
        return self.time_matrix.get((u_idx, v_idx), float('inf'))
