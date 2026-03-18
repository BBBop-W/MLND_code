"""Test script for Benders decomposition methods."""

import sys

from src.utils.output_processor import ResultProcessor
sys.path.insert(0, '.')

from src.data_processor.loader import DataLoader
from src.model.benders_primal import BendersPrimal
from src.model.benders_dual import BendersDual


def main():
    print("=" * 70)
    print("LOADING INSTANCE")
    print("=" * 70)
    inst = DataLoader().load_instance(
        'data/instances/Meituan/locations.csv',
        'data/instances/Meituan/orders.csv',
    )
    print(f"Nodes: P={len(inst.P)}, D={len(inst.D)}, L={len(inst.L_hat)}")
    print(f"Total Demand: {sum(inst.h.values()):.2f}")

    print("\n" + "=" * 70)
    print("BENDERS PRIMAL (Gurobi Dual Extraction)")
    print("=" * 70)
    try:
        primal_solver = BendersPrimal(inst, max_iter=50, verbose=True, time_limit=300)
        primal_obj = primal_solver.solve()
        print(f"\nPrimal Benders objective: {primal_obj}")
        if primal_solver.best_sol:
            print(f"Best solution from primal approach: {primal_solver.best_obj}")
    except Exception as e:
        print(f"ERROR in primal approach: {e}")
        import traceback
        traceback.print_exc()

    processor_mei = ResultProcessor(primal_solver, inst)
    processor_mei.print_summary()

    # print("\n" + "=" * 70)
    # print("BENDERS DUAL (Explicit SOCP Dual)")
    # print("=" * 70)
    # try:
    #     dual_solver = BendersDual(inst, max_iter=50, verbose=True, time_limit=300)
    #     dual_obj = dual_solver.solve()
    #     print(f"\nDual Benders objective: {dual_obj}")
    #     if dual_solver.best_sol:
    #         print(f"Best solution from dual approach: {dual_solver.best_obj}")
    # except Exception as e:
    #     print(f"ERROR in dual approach: {e}")
    #     import traceback
    #     traceback.print_exc()

    # print("\n" + "=" * 70)
    # print("COMPARISON")
    # print("=" * 70)
    # if 'primal_obj' in locals() and 'dual_obj' in locals():
    #     if primal_obj is not None and dual_obj is not None:
    #         gap = abs(primal_obj - dual_obj) / (max(abs(primal_obj), abs(dual_obj)) + 1e-6)
    #         print(f"Primal objective: {primal_obj:.4f}")
    #         print(f"Dual objective:   {dual_obj:.4f}")
    #         print(f"Gap: {gap * 100:.2f}%")
    #         if gap < 0.01:
    #             print("✓ Objectives match closely!")
    #         else:
    #             print("⚠ Objectives have a noticeable gap")


if __name__ == '__main__':
    main()
