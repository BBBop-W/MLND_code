"""Build exact-count Nanshan crops at constant locker-candidate density.

Run ``python -m src.data.area_regular`` to generate six main scales and the
H10 supplementary scale. Every crop regenerates OD from its retained raw
orders; ``configs/area_regular.json`` selects this per-crop demand at solve time.
"""

from __future__ import annotations

import json
from pathlib import Path

from .instance_io import ROOT
from .scale_config import (
    AREA_REGULAR_LEVELS,
    AREA_REGULAR_MAIN_COUNTS,
    AREA_REGULAR_SUPPLEMENTARY_COUNTS,
    DEFAULT_NANSHAN_PATH,
    NanshanScaleBuildConfig,
)
from .scales import build_nanshan_nested_scale, build_parser as scale_parser


DEFAULT_OUTPUT_ROOT = ROOT / "data/instances/nanshan_area_regular"


def build(raw_path=DEFAULT_NANSHAN_PATH, output_root=DEFAULT_OUTPUT_ROOT, **options):
    """Generate all seven crops and audit counts, demand, and raw provenance."""
    config = NanshanScaleBuildConfig(
        raw_path=Path(raw_path),
        output_root=Path(output_root),
        levels=AREA_REGULAR_LEVELS,
        **options,
    )
    manifest_path = build_nanshan_nested_scale(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    main_hubs = [counts[0] for counts in AREA_REGULAR_MAIN_COUNTS]
    supplementary_hubs = [counts[0] for counts in AREA_REGULAR_SUPPLEMENTARY_COUNTS]
    manifest.update(
        family="area_regular",
        main_hubs=main_hubs,
        supplementary_hubs=supplementary_hubs,
        map_size_rule="half_size_km = 10 * sqrt(H / 120)",
        locker_candidates_per_km2=120.0 / 400.0,
    )
    for level in manifest["levels"]:
        level["scale_role"] = (
            "main" if level["candidate_counts"]["H"] in main_hubs else "supplementary"
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = scale_parser()
    parser.description = (
        "Build six exact-count Nanshan crops H20/40/60/80/100/120 plus H10, "
        "with half_size_km=10*sqrt(H/120) and per-crop raw-order OD."
    )
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = build(
        raw_path=args.raw_data,
        output_root=args.output_root,
        candidate_hour_start=args.candidate_hours[0],
        candidate_hour_end=args.candidate_hours[1],
        demand_hour_start=args.demand_hours[0],
        demand_hour_end=args.demand_hours[1],
        od_grid_km=args.od_grid_km,
        site_grid_km=args.site_grid_km,
        dates=tuple(args.dates),
        seed=args.seed,
    )
    print(f"Regular-area Nanshan scale manifest: {manifest}")


if __name__ == "__main__":
    main()
