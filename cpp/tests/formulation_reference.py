"""Audit every model coefficient, including the explicit structural-M change.

The Python baseline still derives arc activation bounds from its full path
catalogue. Native CG uses input-only bounds so its matrix does not change when
columns are generated. Only this proved and independently checked difference
is normalized before comparing the remaining complete sparse matrices.
"""

# ruff: noqa: E402 -- standalone script adds the repository root before local imports.
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.model.paths import enumerate_paths
from src.solvers.formulation import build_model, shapes


def variable_name(name):
    return re.sub(r"[\[\],]+", "_", name).rstrip("_")


def canonical_rows(rows, names):
    """Ignore row names/order; preserve every coefficient, RHS and multiplicity."""
    groups = defaultdict(list)
    for row_name, sense, rhs, terms in rows:
        terms_by_name = defaultdict(float)
        for index, coefficient in terms:
            terms_by_name[names[index]] += coefficient
        entries = sorted(
            (name, value) for name, value in terms_by_name.items() if value != 0
        )
        if sense in ("G", ">"):
            sign, sense = -1, "L"
        elif sense in ("E", "="):
            sign = -1 if entries and entries[0][1] < 0 else 1
            sense = "E"
        else:
            sign, sense = 1, "L"
        key = (sense, tuple(name for name, _ in entries))
        values = np.asarray([rhs * sign] + [value * sign for _, value in entries])
        groups[key].append((row_name, values))
    return groups


def python_matrix(built):
    model = built.model
    variables = model.getVars()
    names = [variable_name(v.VarName) for v in variables]
    assert len(names) == len(set(names))
    records = {
        name: (v.LB, v.UB, v.Obj, v.VType == "B") for name, v in zip(names, variables)
    }
    rows = []
    sparse = model.getA().tocsr()
    for i, row in enumerate(model.getConstrs()):
        start, end = sparse.indptr[i], sparse.indptr[i + 1]
        rows.append(
            (
                row.ConstrName,
                row.Sense,
                row.RHS,
                list(zip(sparse.indices[start:end], sparse.data[start:end])),
            )
        )
    cones = sorted(
        (
            variable_name(up.VarName),
            variable_name(down.VarName),
            variable_name(one.VarName),
            1.0,
        )
        for up, down, one in built.cones
    )
    assert model.NumQConstrs == len(cones)
    expected_quadratics = sorted(
        tuple(sorted(((tuple(sorted((up, down))), -1.0), ((one, one), 1.0))))
        for up, down, one, _ in cones
    )
    actual_quadratics = []
    for row in model.getQConstrs():
        assert row.QCSense == "<" and row.QCRHS == 0
        expression = model.getQCRow(row)
        assert expression.getLinExpr().size() == 0
        actual_quadratics.append(
            tuple(
                sorted(
                    (
                        tuple(
                            sorted(
                                (
                                    variable_name(expression.getVar1(i).VarName),
                                    variable_name(expression.getVar2(i).VarName),
                                )
                            )
                        ),
                        expression.getCoeff(i),
                    )
                    for i in range(expression.size())
                )
            )
        )
    assert sorted(actual_quadratics) == expected_quadratics
    return records, canonical_rows(rows, names), cones


def audit_structural_link_bounds(native, built, data, paths):
    """Validate both M formulas, then normalize only each airline coefficient.

    A direct service task has one departure and one return arc. Its internal
    arc sequence has at most 2K-1 arcs; repetitions of one directed arc cannot
    be adjacent because locker self-loops are excluded, so it appears at most
    K times. Demand conservation bounds total task flow by total OD demand D.
    Each direct relocation arc has one variable bounded by D; the old
    conservative (L-1)D coefficient is retained. These bounds are independent
    of the chosen facilities, current column pool, and dual prices.
    """
    assert data.route_mode == "direct" and data.max_orders in (1, 2)
    n, bases, k = data.n_nodes, data.n_l, data.max_orders
    demand = float(data.demand.sum())
    service = np.zeros((n, n), dtype=int)
    empty = np.zeros((n, n), dtype=int)
    for u in range(n):
        for v in range(n):
            if u == v or not data.allowed_links[u, v]:
                continue
            if u < bases and v < bases:
                empty[u, v] = 1
            elif u < bases or v < bases:
                service[u, v] = 1
            else:
                service[u, v] = k
    enumerated_service = np.zeros((n, n), dtype=int)
    enumerated_empty = np.zeros((n, n), dtype=int)
    for catalogue, observed in ((paths.service_paths, enumerated_service),
                                (paths.empty_paths, enumerated_empty)):
        for path in catalogue:
            for u, v, count in path.arc_counts:
                observed[u, v] = max(observed[u, v], count)
    assert np.all(service >= enumerated_service), "Unsafe service arc multiplicity"
    assert np.all(empty >= enumerated_empty), "Unsafe relocation arc multiplicity"
    expected = demand * (service + (bases - 1) * empty)
    baseline = demand * (enumerated_service + (bases - 1) * enumerated_empty)
    assert np.all(expected >= baseline)

    names = [row[0] for row in native["variables"]]
    indices = {name: i for i, name in enumerate(names)}
    normalized = []
    observed_arcs = set()
    changed = 0
    max_increase = 0.0
    for row_name, sense, rhs, terms in native["rows"]:
        match = re.fullmatch(r"link_(\d+)_(\d+)", row_name)
        if match is None:
            normalized.append((row_name, sense, rhs, terms))
            continue
        u, v = map(int, match.groups())
        assert (u, v) not in observed_arcs and 0 <= u < n and 0 <= v < n
        observed_arcs.add((u, v))
        assert sense == "L" and rhs == 0
        airline_index = indices[f"airline_{u}_{v}"]
        actual = sum(value for j, value in terms if j == airline_index)
        np.testing.assert_allclose(actual, -expected[u, v], atol=2e-12, rtol=5e-13)
        python_row = built.model.getConstrByName(row_name)
        assert python_row is not None and python_row.Sense == "<" and python_row.RHS == 0
        old = built.model.getCoeff(python_row, built.variables["airline"][u, v])
        np.testing.assert_allclose(old, -baseline[u, v], atol=2e-12, rtol=5e-13)
        replacement = [(j, value) for j, value in terms if j != airline_index]
        if old != 0:
            replacement.append((airline_index, old))
        normalized.append((row_name, sense, rhs, replacement))
        changed += actual != old
        max_increase = max(max_increase, expected[u, v] - baseline[u, v])
    assert len(observed_arcs) == n * n, "Missing or duplicate link activation row"
    return normalized, {
        "rows_checked": n * n,
        "changed_airline_coefficients": changed,
        "maximum_M_increase": max_increase,
        "formula": "D for base-locker; K*D for locker-locker; (L-1)*D for base-base; 0 for forbidden/self arcs",
        "enumerated_service_and_empty_multiplicities_bounded": True,
        "all_other_coefficients_compared_without_normalization": True,
    }


def compare(native, built, data, paths):
    names = [row[0] for row in native["variables"]]
    assert len(names) == len(set(names))
    native_vars = {row[0]: tuple(row[1:]) for row in native["variables"]}
    expected_vars, expected_rows, expected_cones = python_matrix(built)
    assert native_vars.keys() == expected_vars.keys(), {
        "native_only": list(native_vars.keys() - expected_vars.keys())[:20],
        "python_only": list(expected_vars.keys() - native_vars.keys())[:20],
    }
    max_variable_difference = 0.0
    for name, expected in expected_vars.items():
        actual = native_vars[name]
        assert actual[3] == expected[3], ("vtype", name, actual, expected)
        np.testing.assert_allclose(
            actual[:3], expected[:3], atol=2e-12, rtol=5e-13, err_msg=name
        )
        max_variable_difference = max(
            max_variable_difference,
            float(np.max(np.abs(np.asarray(actual[:3]) - np.asarray(expected[:3])))),
        )
    normalized_rows, activation_audit = audit_structural_link_bounds(native, built, data, paths)
    actual_rows = canonical_rows(normalized_rows, names)
    assert actual_rows.keys() == expected_rows.keys(), {
        "native_only": [
            (key[0], key[1][:10], [r[0] for r in actual_rows[key]][:3])
            for key in actual_rows.keys() - expected_rows.keys()
        ][:10],
        "python_only": [
            (key[0], key[1][:10], [r[0] for r in expected_rows[key]][:3])
            for key in expected_rows.keys() - actual_rows.keys()
        ][:10],
    }
    maximum_coefficient_difference = 0.0
    rows_compared = nonzeros_compared = 0
    for key, actual in actual_rows.items():
        expected = expected_rows[key]
        assert len(actual) == len(expected), (
            "multiplicity",
            key,
            len(actual),
            len(expected),
        )
        actual.sort(key=lambda row: tuple(row[1]))
        expected.sort(key=lambda row: tuple(row[1]))
        for (actual_name, actual_values), (expected_name, expected_values) in zip(
            actual, expected
        ):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                atol=2e-12,
                rtol=5e-13,
                err_msg=f"row {actual_name} versus {expected_name}",
            )
            if actual_values.size:
                maximum_coefficient_difference = max(
                    maximum_coefficient_difference,
                    float(np.max(np.abs(actual_values - expected_values))),
                )
            rows_compared += 1
            nonzeros_compared += len(actual_values) - 1
    actual_cones = sorted(
        (names[c[0]], names[c[1]], names[c[2]], c[3]) for c in native["cones"]
    )
    assert actual_cones == expected_cones
    assert native["minimize"] and built.model.ModelSense == 1
    assert native["constant"] == built.model.ObjCon == 0
    return {
        "variables_compared": len(native_vars),
        "rows_compared": rows_compared,
        "nonzero_coefficients_compared": nonzeros_compared,
        "cones_compared": len(actual_cones),
        "max_variable_numeric_difference": max_variable_difference,
        "max_row_coefficient_difference": maximum_coefficient_difference,
        "structural_link_bound_audit": activation_audit,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument(
        "--instance", type=Path, default=ROOT / "examples/tsl_2l20h50od"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    reports = []
    with tempfile.TemporaryDirectory(prefix="mlnd_formulation_") as temp:
        for k in args.cases:
            data = to_planning_data(load_standard(args.instance, case=f"K{k}"))
            paths = enumerate_paths(data)
            for mode in ("full", "zero", "pattern"):
                native_file = Path(temp) / f"K{k}_{mode}.json"
                command = [
                    str(args.binary.resolve()),
                    str(args.instance.resolve()),
                    str(k),
                    mode,
                    str(native_file),
                ]
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode:
                    raise AssertionError(result.stderr)
                native = json.loads(native_file.read_text())
                fixed = None
                if mode != "full":
                    fixed, offset = {}, 0
                    for name, shape in shapes(data).items():
                        size = int(np.prod(shape))
                        fixed[name] = np.asarray(
                            native["fixed"][offset : offset + size]
                        ).reshape(shape)
                        offset += size
                    assert offset == len(native["fixed"])
                built = build_model(data, paths, fixed_design=fixed)
                try:
                    report = compare(native, built, data, paths)
                finally:
                    built.model.dispose()
                report.update(case=f"K{k}", mode=mode)
                reports.append(report)
                print(json.dumps(report), flush=True)
                native_file.unlink()
    provenance = [
        "cpp/src/formulation.cpp",
        "cpp/src/linear_model.cpp",
        "cpp/src/input.cpp",
        "cpp/src/paths.cpp",
        "src/solvers/formulation.py",
        "src/model/paths.py",
        "src/model/resources.py",
        "cpp/tests/formulation_reference.py",
    ]
    final = {
        "passed": True,
        "comparison": "complete sparse matrix, objective, variable bounds/types, cones",
        "permitted_difference": "Only link-row airline coefficients: native input-only structural M versus Python full-catalogue M; both formulas individually verified and every enumerated multiplicity bounded before comparison",
        "atol": 2e-12,
        "rtol": 5e-13,
        "reports": reports,
        "input_manifest_sha256": hashlib.sha256(
            (args.instance / "manifest.json").read_bytes()
        ).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in provenance
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
