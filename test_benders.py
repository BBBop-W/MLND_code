"""Validation harness for the refactored Benders decomposition models."""

import sys

sys.path.insert(0, '.')

from src.data_processor.loader import DataLoader
from src.model.benders_model import (
    BendersDual,
    BendersPrimal,
    FixedMasterAudit,
)


def print_basic_instance_stats(instance):
    print(f"Nodes: P={len(instance.P)}, D={len(instance.D)}, L={len(instance.L_hat)}")
    print(f"Total Demand: {sum(instance.h.values()):.2f}")
    print(f"A_LP: {len(instance.A_LP)}, A_PD: {len(instance.A_PD)}, A_DL: {len(instance.A_DL)}")


def main():
    inst = DataLoader().load_instance(
        'data/instances/Meituan/locations.csv',
        'data/instances/Meituan/orders.csv',
    )
    # print_basic_instance_stats(inst)

    # checker = FixedMasterAudit(inst)
    # benchmark = checker.solve_benchmark(time_limit=300, mip_gap=0.05)
    # print(f"Benchmark status: {benchmark['status']}")
    # if 'objective' not in benchmark:
    #     print('Benchmark returned no incumbent solution.')
    #     return

    # print(f"Benchmark objective: {benchmark['objective']:.4f}")
    # report = checker.audit_fixed_master(
    #     benchmark['x'],
    #     benchmark['u'],
    #     compare_to_reference=True,
    #     reference_time_limit=300,
    # )

    # print(f"Primal recourse objective: {report.primal.objective}")
    # print(f"Dual recourse objective: {report.dual.objective}")
    # print(f"Primal-dual gap: {report.primal_dual_gap}")
    # print(f"Fixed-design benchmark objective: {report.reference_objective}")
    # print(f"Primal-reference gap: {report.primal_reference_gap}")
    # if report.notes:
    #     print('Consistency notes:')
    #     for note in report.notes:
    #         print(f"- {note}")

    dual_solver = BendersDual(inst, max_iter=1000, verbose=True, time_limit=1200)
    dual_objective = dual_solver.solve()
    print(f"Dual Benders objective: {dual_objective}")

    primal_solver = BendersPrimal(inst, max_iter=1000, verbose=True, time_limit=1200)
    primal_objective = primal_solver.solve()
    print(f"Primal Benders objective: {primal_objective}")
    if primal_solver.get_validation_issues():
        print('Incumbent validation issues:')
        for issue in primal_solver.get_validation_issues():
            print(f"- {issue}")


if __name__ == '__main__':
    main()
