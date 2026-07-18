"""Structural and result audit for the isolated 2026-06-23 framework."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "model" / "bdd_lr_benders_20260623"


def source_audit() -> dict:
    texts = {
        path.name: path.read_text(encoding="utf-8")
        for path in PACKAGE_ROOT.glob("*.py")
    }
    combined = "\n".join(texts.values())
    return {
        "isolated_package_exists": PACKAGE_ROOT.is_dir(),
        "does_not_import_legacy_strict_solver": (
            "from src.model.lagrangian_benders import" not in combined
            and "import src.model.lagrangian_benders" not in combined
        ),
        "has_pure_master_invariant": "assert_pure_master" in combined,
        "has_qcpdual_oracle": "BendersRegimeIISolver" in combined,
        "has_pareto_reference_tightness_test": (
            "reference_error" in texts.get("pareto.py", "")
            and "reference_tolerance" in texts.get("pareto.py", "")
        ),
        "has_bdd_local_copy": "BDDLocalCopyOracle20260623" in combined,
        "bdd_uses_objbound": "ObjBound" in texts.get("bdd.py", ""),
        "has_local_branching": "AdaptiveLocalBranching20260623" in combined,
        "has_lr_profiler": "cpp_acceleration_decision" in combined,
        "has_exact_integer_nogood": "_add_exact_design_nogood" in combined,
        "has_explicit_cone_removal_assert": (
            "remove_all_quadratic_constraints" in combined
            and "Expected a cone-free MILP" in combined
        ),
        "has_aggregate_master_proxy": (
            "proxy_service_by_origin" in texts.get("master.py", "")
            and "proxy_top_pair" in texts.get("master.py", "")
        ),
        "removed_full_od_service_proxy": (
            "service_possible" not in texts.get("master.py", "")
            and "route_ready" not in texts.get("master.py", "")
        ),
    }


def result_audit(
    results_root: Path,
    *,
    include_invalidated: bool = False,
    variant_filter: str | None = None,
) -> dict:
    results_root = results_root.resolve()
    summaries = list(results_root.glob("H*/**/summary.json"))
    rows = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = str(data.get("variant") or "")
        if not include_invalidated and not variant.startswith("qfix_"):
            continue
        if variant_filter is not None and variant != variant_filter:
            continue
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "method": data.get("method"),
                "variant": data.get("variant"),
                "has_obj": data.get("obj") is not None,
                "has_bound": data.get("bound") is not None,
                "has_gap": data.get("gap") is not None,
                "has_wall_time": data.get("wall_time") is not None,
                "has_time_to_best": "time_to_best" in data,
                "master_qconstrs": data.get("master_qconstrs"),
                "local_branching_enabled": data.get(
                    "local_branching_enabled"
                ),
                "global_linearized_bound": data.get(
                    "global_linearized_bound"
                ),
                "lr_queue_tangents_enabled": data.get(
                    "lr_queue_tangents_enabled"
                ),
                "restricted_network_qconstrs_after": data.get(
                    "restricted_network_qconstrs_after"
                ),
                "local_branching_qconstrs_after": data.get(
                    "local_branching_qconstrs_after"
                ),
                "global_linearized_qconstrs_after": data.get(
                    "global_linearized_qconstrs_after"
                ),
            }
        )
    required_methods = {"gurobi", "lr", "bd", "lr_bd"}
    observed_methods = {row["method"] for row in rows}
    cone_free_evidence = []
    for row in rows:
        method = row["method"]
        if method == "lr":
            cone_free_evidence.append(
                row["restricted_network_qconstrs_after"] == 0
            )
        if method == "bd":
            if (
                row["local_branching_enabled"]
                and row["local_branching_qconstrs_after"] is not None
            ):
                cone_free_evidence.append(
                    row["local_branching_qconstrs_after"] == 0
                )
            if row["global_linearized_bound"] is not None:
                cone_free_evidence.append(
                    row["global_linearized_qconstrs_after"] == 0
                )
        if method == "lr_bd":
            cone_free_evidence.append(
                row["restricted_network_qconstrs_after"] == 0
            )
            if (
                row["local_branching_enabled"]
                and row["local_branching_qconstrs_after"] is not None
            ):
                cone_free_evidence.append(
                    row["local_branching_qconstrs_after"] == 0
                )
            if row["global_linearized_bound"] is not None:
                cone_free_evidence.append(
                    row["global_linearized_qconstrs_after"] == 0
                )
    return {
        "summary_count": len(rows),
        "observed_methods": sorted(
            method for method in observed_methods if method is not None
        ),
        "four_method_matrix_present": required_methods <= observed_methods,
        "all_decomposition_masters_are_milp": all(
            row["master_qconstrs"] in (None, 0)
            for row in rows
            if row["method"] in {"lr", "bd", "lr_bd"}
        ),
        "all_executed_candidate_milps_are_cone_free": bool(
            cone_free_evidence
        )
        and all(cone_free_evidence),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(
            PROJECT_ROOT / "results" / "bdd_lr_benders_20260623"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "bdd_lr_benders_20260623"
            / "audit.json"
        ),
    )
    parser.add_argument("--include-invalidated", action="store_true")
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    report = {
        "source": source_audit(),
        "results": result_audit(
            Path(args.results_root),
            include_invalidated=args.include_invalidated,
            variant_filter=args.variant,
        ),
    }
    source_passed = all(report["source"].values())
    report["passed"] = bool(
        source_passed
        and report["results"]["four_method_matrix_present"]
        and report["results"]["all_decomposition_masters_are_milp"]
        and report["results"]["all_executed_candidate_milps_are_cone_free"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
