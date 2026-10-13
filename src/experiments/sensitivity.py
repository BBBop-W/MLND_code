"""Freeze and launch numbered rounds of one-factor arc-based experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from src.config import load_config
from src.data.demand import describe_demand
from . import performance
from .sensitivity_inputs import build_variants
from .storage import ROOT, load_inputs, save_inputs, sha256, write_json
from .progress import utc_now


DEFAULT_BASE = ROOT / "data/instances/algorithm_performance/nanshan/region_2/H020_L04_E08_OD0400"
ROUND_ONE = ROOT / "outputs/experiments/h20_sensitivity_20260920"
ROUND_TWO_FACTORS = dict(speed=(0.5, 1.5), battery=(0.5, 1.5),
                         fleet_cost=(0.5, 0.75, 1.25, 1.5))
ROUND_THREE_FACTORS = dict(
    demand=(0.2, 0.3, 0.4), speed=(0.6, 0.7, 0.8, 0.9),
    max_cabinets=(3 / 5, 4 / 5, 6 / 5, 7 / 5),
    walk_sensitivity=(0.5, 0.75, 1.25, 1.5),
)


def prepare(output, *, base_path=DEFAULT_BASE, workers=None, seconds=3600,
            mip_gap=0.005, round_number=1):
    """Save all parameter variants and the immutable execution plan; do not solve."""
    output, base_path = Path(output).resolve(), Path(base_path).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing sensitivity batch: {output}")
    if round_number not in (1, 2, 3):
        raise ValueError("Choose sensitivity round 1, 2 or 3")
    workers = (3 if round_number == 1 else 2) if workers is None else workers
    base = load_inputs(base_path)
    if tuple(base[k] for k in ("n_h", "n_l", "n_k", "n_od")) != (20, 4, 8, 400):
        raise ValueError("Expected the frozen H20 L4 E8 OD400 instance")
    parent_hashes = {name: sha256(base_path / name) for name in
                     ("input_arrays.npz", "input_metadata.json")}
    if round_number == 1:
        variants = build_variants(base)
        factors = dict(demand=[0.5, 0.75, 1, 1.25, 1.5],
                       rider_cost=[0.5, 0.75, 1, 1.25, 1.5],
                       speed=[0.75, 1, 1.25], battery=[0.75, 1, 1.25])
    elif round_number == 2:
        from src.data.sensitivity_density import build_density_variants

        variants = build_variants(base, factors=ROUND_TWO_FACTORS, include_baseline=False)
        variants += build_density_variants(base, base_path)
        factors = dict(ROUND_TWO_FACTORS, density_H=[10, 15, 25, 30])
    else:
        variants = build_variants(base, factors=ROUND_THREE_FACTORS, include_baseline=False)
        factors = dict(ROUND_THREE_FACTORS, max_cabinets=[3, 4, 6, 7])
    cases = []
    for variant in variants:
        data = variant["data"]
        directory = output / "instances" / variant["id"]
        data["provenance"]["sensitivity"].update(
            parent_input_path=str(base_path), parent_input_sha256=parent_hashes,
            round=round_number,
        )
        hashes = save_inputs(data, directory)
        configuration = load_config(data["provenance"]["resolved_config"])
        write_json(directory / "config.json", configuration)
        cases.append(dict(
            id=f"H{data['n_h']}_{variant['id']}", factor=variant["factor"],
            multiplier=variant["multiplier"], source="nanshan", region="region_2",
            H=data["n_h"], L=data["n_l"], E=data["n_k"], OD=data["n_od"],
            round=round_number, input_path=str(directory),
            input_hashes=hashes, config=str(directory / "config.json"),
            demand=describe_demand(data),
        ))
    manifest = dict(
        schema=1, family=f"Sensitivity round {round_number}", round=round_number,
        created_utc=utc_now(),
        baseline_input=str(base_path), baseline_input_sha256=parent_hashes,
        baseline_shared_across_factors=True, cartesian_product=False,
        total_cases=len(variants), baseline_candidates=dict(H=20, L=4, E=8), od_pairs=400,
        study_area_and_od_identities_fixed=True, factors=factors,
        baseline_run=str((output if round_number == 1 else ROUND_ONE)
                         / "batch/runs/H20_baseline__arc"),
        saving_reference="All-courier cost recalculated with each case's demand and rider tariffs",
        map_layers="Facilities and positive operational flight flows; no OD overlay",
        solution_storage=dict(
            inputs="input_arrays.npz and input_metadata.json",
            primal_solution="solution.npz", paths="path_flows.json.gz",
            solver_result="result.json", operations="operations.json",
            python_loader="src.experiments.storage.load_seed",
            per_od_times="sensitivity/od_times.csv",
            per_path_times="sensitivity/paths.csv",
        ),
        cases=cases,
    )
    write_json(output / "instances.json", manifest)
    plan = performance.prepare(
        output / "instances.json", output / "batch",
        hubs=tuple(sorted({case["H"] for case in cases})), workers=workers,
        seconds=seconds, methods=("arc",), solver_options={"mip_gap": mip_gap},
        postprocess=dict(kind="sensitivity", map_cache=str(ROOT / "data/map_cache")),
    )
    assert len(plan["jobs"]) == {1: 13, 2: 12, 3: 15}[round_number]
    return plan


def finalize_job(plan, job, summary):
    """Retain numerical outputs and render maps after each independently checked solve."""
    specification = plan["postprocess"]
    if specification.get("kind") != "sensitivity":
        raise ValueError("Unknown experiment postprocessing")
    output = Path(job["output"])
    summary = dict(summary, factor=job["factor"], multiplier=job["multiplier"])
    if "round" in job:
        summary["round"] = job["round"]
    if summary.get("passed"):
        try:
            from src.reporting.sensitivity import export_case

            exported = export_case(output, map_cache=specification["map_cache"])
            # Keep run status and solver diagnostics authoritative; report fields
            # extend them with independently reconstructed operating measures.
            summary.update({key: value for key, value in exported["summary"].items()
                            if key not in summary})
            summary["postprocess_status"] = "complete"
        except Exception as error:
            summary["postprocess_status"] = "failed"
            write_json(output / "postprocess_error.json", dict(
                error=str(error), traceback=traceback.format_exc(),
            ))
    else:
        summary["postprocess_status"] = "no_validated_solution"
    write_json(output / "summary.json", summary)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["postprocess_status"] = summary["postprocess_status"]
        manifest["artifacts"] = {
            str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=sha256(path))
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        write_json(manifest_path, manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default=DEFAULT_BASE, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--round", dest="round_number", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--gap", type=float, default=0.005)
    args = parser.parse_args(argv)
    plan = prepare(args.out, base_path=args.base, workers=args.workers,
                   seconds=args.seconds, mip_gap=args.gap, round_number=args.round_number)
    if args.command == "launch":
        answer = performance.launch(args.out / "batch")
    else:
        answer = dict(status="prepared", jobs=len(plan["jobs"]), output=str(args.out.resolve()))
    print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
