"""Compare Benders primal approach vs monolithic Gurobi model."""

import sys
sys.path.insert(0, '.')

from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.model.benders_model import IndexSets
from src.model.benders_primal import BendersPrimal

def main():
    print("=" * 70)
    print("LOADING INSTANCE")
    print("=" * 70)
    
    inst = DataLoader().load_instance(
        'data/instances/Meituan/locations.csv',
        'data/instances/Meituan/orders.csv',
    )
    print(f"Instance: P={len(inst.P)}, D={len(inst.D)}, L={len(inst.L_hat)}")
    print(f"Total demand: {sum(inst.h.values()):.2f}")
    
    print("\n" + "=" * 70)
    print("MONOLITHIC GUROBI MODEL (BASELINE)")
    print("=" * 70)
    
    mlnd = MLNDModel(inst)
    success = mlnd.solve()
    if not success:
        print("ERROR: Monolithic model failed to solve")
        return
    mono_obj = mlnd.model.objVal
    
    print("\n" + "=" * 70)
    print("BENDERS PRIMAL DECOMPOSITION")
    print("=" * 70)
    
    data = IndexSets(inst)
    benders = BendersPrimal(inst, data)
    benders_obj = benders.solve()
    print(f"Benders objective: {benders_obj}")
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    gap = abs(mono_obj - benders_obj) / max(mono_obj, 1e-6)
    print(f"Monolithic:   {mono_obj:.6f}")
    print(f"Benders:      {benders_obj:.6f}")
    print(f"Absolute gap: {abs(mono_obj - benders_obj):.6f}")
    print(f"Relative gap: {gap * 100:.4f}%")
    
    if gap < 1e-3:
        print("✓ PASS: Results match within tolerance")
    else:
        print("✗ FAIL: Results differ significantly")

if __name__ == '__main__':
    main()
