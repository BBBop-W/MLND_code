"""Decimal replay of the L3/L4 analytical joint certificates, without a solver.

The probe exports coefficients and profiles rather than its physical catalogue;
this checks the state table and lifting independently, not physical-path minima.
"""
from __future__ import annotations

import argparse
from decimal import localcontext
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments import baseline_cut_audit_20261029 as audit  # noqa: E402
from src.experiments.storage import sha256, write_json  # noqa: E402


def replay(witness):
    cert = witness["cut_certificate"]
    profile = cert["profile"]
    bases, tasks = profile["bases"], profile["tasks"]
    nodes = len(profile["prices"]["balance"]) + len(profile["prices"]["locker"])
    count = len(profile["lower_coefficients"])
    start = count - tasks * bases - nodes * nodes
    constants = dict(Q=tasks, L=bases, design_count=count, assignment=start,
                     assignment_scope="task", F=audit.D(profile["fleet_cost"]))
    affine = [audit.D(v) for v in cert["affine_coefficients"]]
    lower = [audit.D(v) for v in profile["lower_coefficients"]]
    for i, task in enumerate(cert["selected_tasks"]):
        for base in range(bases):
            lower[start + task * bases + base] += affine[1 + i * bases + base]
    for index, penalty in enumerate(cert["external_penalties"]):
        lower[start + index] -= audit.D(penalty)
    rhs = audit.D(profile["constant"]) + affine[0] - audit.D(cert["intercept_correction"])
    row = dict(certificate=cert, rhs=float(rhs),
               coefficients=[[j, float(-v)] for j, v in enumerate(lower) if v] + [[count, 1.]],
               violation=cert["value_at_separation_point"] - cert["separation_theta"])
    checked = audit.replay_joint_queue_row(row, constants)
    # Full categorical enumeration includes outside tasks as well. The common
    # resource affine term cancels, leaving a direct independent lifting test.
    signed = [audit.D(v) for v in profile["signed_savings"]]
    missing = profile["unsupported"]
    selected = cert["selected_tasks"]
    penalties = [audit.D(v) for v in cert["external_penalties"]]
    maximum, total = audit.D("-Infinity"), 0
    for states in product(range(bases + 1), repeat=tasks):
        true_queue = audit.ZERO
        for base in range(bases):
            groups = [q * bases + base for q, state in enumerate(states) if state == base + 1]
            if not any(missing[g] for g in groups):
                true_queue -= audit.conjugate(sum((signed[g] for g in groups), audit.ZERO), constants["F"])
        candidate = affine[0]
        for i, task in enumerate(selected):
            state = states[task]
            if state:
                candidate += affine[1 + i * bases + state - 1]
        for task, state in enumerate(states):
            if state:
                candidate -= penalties[task * bases + state - 1]
        excess = candidate - true_queue
        audit.require(excess <= audit.D("2e-12") * (1 + abs(true_queue)), "Lift excludes categorical assignment")
        maximum = max(maximum, excess)
        total += 1
    return dict(passed=True, bases=bases, state_table_replay=checked,
                full_categorical_assignments=total, maximum_excess=float(maximum),
                physical_profile_rebuilt=False, emitted_sparse_row_available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.probe.read_text())
    results = [replay(source[f"general_l{bases}_witness"]) for bases in (3, 4)]
    report = dict(passed=True, solver_invoked=False, probe_sha256=sha256(args.probe),
                  source_sha256=sha256(Path(__file__)), results=results)
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, categorical_assignments=sum(r["full_categorical_assignments"] for r in results),
                          report=str(args.output))))


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 60
        main()
