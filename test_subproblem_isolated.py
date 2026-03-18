"""Minimal test of primal subproblem to debug Q matrix issue."""

import sys
sys.path.insert(0, '.')

from src.data_processor.loader import DataLoader
from src.model.benders_model import IndexSets
from src.model.benders_primal import PrimalSubproblem

def test_subproblem():
    print("Loading instance...")
    inst = DataLoader().load_instance(
        'data/instances/Meituan/locations.csv',
        'data/instances/Meituan/orders.csv',
    )
    print(f"Instance loaded: P={len(inst.P)}, D={len(inst.D)}, L={len(inst.L_hat)}")
    
    data = IndexSets(inst)
    print(f"Index sets built: A={len(data.A)}, A_LP={len(data.A_LP)}, A_PD={len(data.A_PD)}")
    
    # Create a simple feasible master solution: open all bases and one P/D node each
    x_hat = {n: 0 for n in data.N + data.L_v}
    for l in data.L_hat:
        x_hat[l] = 1
    if data.P:
        x_hat[data.P[0]] = 1
    if data.D:
        x_hat[data.D[0]] = 1
    
    u_hat = {}
    for l in data.L:
        for s in data.S:
            u_hat[l, s] = 1 if s == 0 else 0  # Select first service level
    
    print(f"Created master solution: {sum(x_hat.values())} bases/nodes open")
    print("Solving subproblem...")
    
    sub = PrimalSubproblem(inst, data)
    try:
        status, obj, cut_data = sub.solve(x_hat, u_hat)
        print(f"Status: {status}, Obj: {obj}")
        if cut_data:
            print(f"Cut data kind: {cut_data.get('kind')}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_subproblem()
