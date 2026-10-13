"""Demand sensitivity with service missions returning to their departure base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.data.demand import describe_demand
from . import performance
from .progress import utc_now
from .sensitivity import DEFAULT_BASE
from .sensitivity_inputs import build_variants
from .storage import ROOT, load_inputs, save_inputs, sha256, write_json


MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)


def prepare(output, *, base_path=DEFAULT_BASE, workers=3, seconds=3600, mip_gap=.005):
    output, base_path = Path(output).resolve(), Path(base_path).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing batch: {output}")
    base = load_inputs(base_path)
    if tuple(base[k] for k in ("n_h", "n_l", "n_k", "n_od")) != (20, 4, 8, 400):
        raise ValueError("Expected the frozen H20 L4 E8 OD400 instance")
    hashes = {name: sha256(base_path / name) for name in
              ("input_arrays.npz", "input_metadata.json")}
    cases = []
    for variant in build_variants(base, factors={"demand": MULTIPLIERS}, include_baseline=False):
        data = variant["data"]
        data["same_base_service"] = True
        data["name"] += "_same_base"
        data["provenance"]["sensitivity"].update(
            experiment="same_base_demand", same_base_service=True,
            parent_input_path=str(base_path), parent_input_sha256=hashes,
            service_base_rule="l == m; empty repositioning domain unchanged",
        )
        data["provenance"]["sensitivity"]["changed_model_fields"].append("same_base_service")
        directory = output / "instances" / variant["id"]
        input_hashes = save_inputs(data, directory)
        write_json(directory / "config.json", load_config(data["provenance"]["resolved_config"]))
        cases.append(dict(
            id=f"H20_same_base_{variant['id']}", factor="demand",
            multiplier=variant["multiplier"], source="nanshan", region="region_2",
            H=20, L=4, E=8, OD=400, input_path=str(directory),
            input_hashes=input_hashes, config=str(directory / "config.json"),
            demand=describe_demand(data), same_base_service=True,
        ))
    write_json(output / "instances.json", dict(
        schema=1, family="Same-base demand sensitivity", created_utc=utc_now(),
        baseline_input=str(base_path), baseline_input_sha256=hashes,
        total_cases=5, factors={"demand": MULTIPLIERS},
        same_base_service=True, study_area_and_od_identities_fixed=True,
        service_base_rule="l == m; empty repositioning domain unchanged",
        comparison_batch=str(ROOT / "outputs/experiments/h20_sensitivity_20260920/batch"),
        cases=cases,
    ))
    return performance.prepare(
        output / "instances.json", output / "batch", hubs=(20,), workers=workers,
        seconds=seconds, methods=("arc",), solver_options={"mip_gap": mip_gap},
        postprocess=dict(kind="sensitivity", map_cache=str(ROOT / "data/map_cache")),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    plan = prepare(args.out, workers=args.workers)
    answer = performance.launch(args.out / "batch") if args.command == "launch" else {
        "status": "prepared", "jobs": len(plan["jobs"]), "output": str(args.out.resolve()),
    }
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
