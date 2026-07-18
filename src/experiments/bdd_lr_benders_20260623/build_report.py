"""Build post-qfix formal, scale, ablation, and LR timing tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_summaries(results_root: Path) -> list[dict]:
    rows = []
    for path in sorted(results_root.glob("H*/**/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = str(data.get("variant") or "")
        if not variant.startswith("qfix_"):
            continue
        data["_result_dir"] = str(path.parent.relative_to(PROJECT_ROOT))
        rows.append(data)
    return rows


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(
            PROJECT_ROOT / "results" / "bdd_lr_benders_20260623"
        ),
    )
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    summaries = read_summaries(root)

    common_fields = (
        "H",
        "method",
        "variant",
        "obj",
        "validation_obj",
        "bound",
        "gap",
        "wall_time",
        "time_to_best",
        "initial_design_count",
        "validated_initial_design_count",
        "failed_initial_design_count",
        "best_initial_design_obj",
        "lr_candidate_count",
        "lr_anchor_top_pairs",
        "lr_anchor_refinement_seeds",
        "lr_restricted_time_limit",
        "master_qconstrs",
        "master_proxy_service_variables",
        "master_proxy_top_pair_variables",
        "local_branching_start_pool_size",
        "local_branching_max_calls_per_center",
        "local_branching_solution_pool_size",
        "local_branching_benders_envelope_cuts",
        "local_branching_benders_envelope_weight",
        "local_branching_benders_envelope_cutoff",
        "local_branching_max_removals",
        "local_branching_protect_top_pairs",
        "gamma_sweep_enabled",
        "gamma_sweep_max_calls",
        "gamma_sweep_radius",
        "gamma_sweep_step",
        "gamma_sweep_time_reserve",
        "gamma_sweep_evaluated",
        "gamma_sweep_best_obj",
        "gamma_sweep_improved_count",
        "benders_cut_pool_size",
        "restricted_network_qconstrs_after",
        "local_branching_qconstrs_after",
        "global_linearized_qconstrs_after",
        "_result_dir",
    )
    formal = sorted(
        (
            row
            for row in summaries
            if row.get("variant") == "qfix_formal900_final"
        ),
        key=lambda row: (int(row["H"]), str(row["method"])),
    )
    scale = sorted(
        (
            row
            for row in summaries
            if row.get("variant") == "qfix_scale900_final"
        ),
        key=lambda row: (int(row["H"]), str(row["method"])),
    )
    recovery_scale = sorted(
        (
            row
            for row in summaries
            if row.get("variant")
            in {
                "qfix_gurobi_scale300",
                "qfix_lr_scale_recovery_v3_fixed_hubs300",
                "qfix_lr_scale_recovery_v4_dense_hubs300",
                "qfix_lr_scale_recovery_v5_adaptive_hubs300",
                "qfix_lr_scale_recovery_v5_adaptive_hubs900",
                "qfix_lrbd_recovery_v3_proxy300",
                "qfix_lrbd_recovery_v5_proxy300",
                "qfix_gurobi_scale600",
                "qfix_lrbd_recovery_v5_proxy600_lbdeep",
                "qfix_recovery_v6_lr300",
                "qfix_recovery_v6_lrbd300",
                "qfix_recovery_v6_matrix300",
                "qfix_recovery_v7_facility_lb300",
                "qfix_recovery_v7_lbfocus300",
                "qfix_recovery_v8_multicenter300",
                "qfix_recovery_v9_layered_lb300",
                "qfix_recovery_v10_targeted_layered300",
                "qfix_recovery_v11_diverse_multicenter300",
                "qfix_recovery_v12_matrix300",
                "qfix_recovery_v13_h50_wide_lb300",
                "qfix_recovery_v14_h50_pool_lb300",
                "qfix_recovery_v15_h50_restricted_pool300",
                "qfix_recovery_v16_h50_benders_envelope300",
                "qfix_recovery_v17_h50_benders_envelope_pool300",
                "qfix_recovery_v18_h50_benders_trust300",
                "qfix_recovery_v19_h50_service_preserve300",
                "qfix_recovery_v20_h50_service_preserve_focus300",
                "qfix_recovery_v21_h50_service_preserve_deep300",
                "qfix_recovery_v22_h50_service_preserve_multicenter_deep300",
                "qfix_recovery_v23_h50_anchor_service300",
                "qfix_recovery_v24_h50_gamma_sweep300",
                "qfix_recovery_v25_h50_gamma_sweep_late300",
                "qfix_recovery_v26_h50_gamma_sweep_upfirst300",
                "qfix_recovery_v27_h50_gamma_sweep_pool300",
                "qfix_recovery_v28_h50_restricted24_gamma300",
                "qfix_recovery_v29_scale_restricted30_gamma300",
                "qfix_bd_proxy300",
            }
        ),
        key=lambda row: (
            int(row["H"]),
            str(row["method"]),
            str(row["variant"]),
        ),
    )
    ablation = sorted(
        (
            row
            for row in summaries
            if str(row.get("variant") or "").startswith("qfix_ablation_")
        ),
        key=lambda row: (
            int(row["H"]),
            str(row["variant"]),
            str(row["method"]),
        ),
    )

    write_rows(root / "formal_matrix.csv", formal, common_fields)
    write_rows(root / "scale_comparison.csv", scale, common_fields)
    write_rows(
        root / "recovery_scale_comparison.csv",
        recovery_scale,
        common_fields,
    )
    write_rows(root / "ablation.csv", ablation, common_fields)

    timing_rows = []
    for summary in [*scale, *recovery_scale]:
        if summary.get("method") not in {"lr", "lr_bd"}:
            continue
        lr_wall = float(summary.get("lr_wall_time") or 0.0)
        for timing in summary.get("timing_rows", []):
            timing_rows.append(
                {
                    "H": summary["H"],
                    "method": summary["method"],
                    "variant": summary["variant"],
                    "phase": timing.get("phase"),
                    "wall_time": timing.get("wall_time"),
                    "solver_time": timing.get("solver_time"),
                    "non_solver_time": timing.get("non_solver_time"),
                    "calls": timing.get("calls"),
                    "fraction_of_lr": (
                        None
                        if lr_wall <= 0.0
                        else float(timing.get("wall_time") or 0.0) / lr_wall
                    ),
                }
            )
    write_rows(
        root / "lr_timing.csv",
        timing_rows,
        (
            "H",
            "method",
            "variant",
            "phase",
            "wall_time",
            "solver_time",
            "non_solver_time",
            "calls",
            "fraction_of_lr",
        ),
    )

    manifest = {
        "formal_rows": len(formal),
        "scale_rows": len(scale),
        "recovery_scale_rows": len(recovery_scale),
        "ablation_rows": len(ablation),
        "lr_timing_rows": len(timing_rows),
        "accepted_variant_prefix": "qfix_",
    }
    (root / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
