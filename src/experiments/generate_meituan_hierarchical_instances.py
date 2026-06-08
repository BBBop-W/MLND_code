import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.generator import DataGenerator


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    if min(values) <= 0:
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_schedule(raw: str | None, hub_counts: list[int], default_value: int) -> int | list[int]:
    if raw is None:
        return int(default_value)
    values = parse_int_list(raw)
    if len(values) == 1:
        return values[0]
    if len(values) != len(hub_counts):
        raise argparse.ArgumentTypeError(
            f"schedule length {len(values)} must be 1 or match hub-counts length {len(hub_counts)}"
        )
    return values


def parse_optional_dates(raw: str | None):
    if raw is None or raw.strip() == "":
        return None
    return parse_int_list(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate nested Meituan H-node instance series for MLND numerical experiments."
    )
    parser.add_argument(
        "--raw-data",
        default=os.path.join(PROJECT_ROOT, "Meituan-TSL", "all_waybill_info_meituan_0322.csv"),
        help="Path to all_waybill_info_meituan_0322.csv.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
        help="Directory that will contain one subdirectory per generated instance.",
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=getattr(config, "MEITUAN_SERIES_HUB_COUNTS", [100, 90, 80, 70, 60, 50, 40, 30, 20]),
        help="Comma-separated H counts. Default: 100,90,...,20.",
    )
    parser.add_argument(
        "--num-bases",
        type=int,
        default=getattr(config, "MEITUAN_SERIES_NUM_BASES", 20),
        help="Fixed L count when --base-counts is omitted.",
    )
    parser.add_argument(
        "--num-swapping",
        type=int,
        default=getattr(config, "MEITUAN_SERIES_NUM_SWAPPING", 20),
        help="Fixed E count when --swapping-counts is omitted.",
    )
    parser.add_argument(
        "--base-counts",
        default=None,
        help="Optional comma-separated L schedule aligned with --hub-counts.",
    )
    parser.add_argument(
        "--swapping-counts",
        default=None,
        help="Optional comma-separated E schedule aligned with --hub-counts.",
    )
    parser.add_argument("--date-filter", type=int, default=20221017)
    parser.add_argument("--candidate-dates", default=None, help="Optional comma-separated candidate dates.")
    parser.add_argument("--demand-dates", default=None, help="Optional comma-separated demand dates.")
    parser.add_argument(
        "--candidate-hour-start",
        type=int,
        default=getattr(config, "MEITUAN_CANDIDATE_HOUR_START", config.MEITUAN_WORK_HOUR_START),
    )
    parser.add_argument(
        "--candidate-hour-end",
        type=int,
        default=getattr(config, "MEITUAN_CANDIDATE_HOUR_END", config.MEITUAN_WORK_HOUR_END + 1),
    )
    parser.add_argument(
        "--demand-hour-start",
        type=int,
        default=getattr(config, "MEITUAN_DEMAND_HOUR", 12),
    )
    parser.add_argument("--demand-hour-end", type=int, default=None)
    parser.add_argument(
        "--map-half-size-km",
        type=float,
        default=getattr(config, "MEITUAN_MAP_HALF_SIZE_KM", 8.0),
    )
    parser.add_argument(
        "--no-colocate-base-swaps",
        action="store_true",
        help="Do not force the first E candidates to be colocated with L candidates.",
    )
    parser.add_argument(
        "--activity-weight",
        type=float,
        default=getattr(config, "MEITUAN_HIERARCHY_ACTIVITY_WEIGHT", 0.35),
        help="Weight on order-volume similarity in hierarchical merging.",
    )
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=getattr(config, "MEITUAN_HIERARCHY_BALANCE_WEIGHT", 0.75),
        help="Penalty weight for overloaded merged clusters.",
    )
    parser.add_argument(
        "--overload-tolerance",
        type=float,
        default=getattr(config, "MEITUAN_HIERARCHY_OVERLOAD_TOLERANCE", 1.35),
        help="Merged load above this multiple of current target receives balance penalty.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    hub_counts = sorted(set(args.hub_counts), reverse=True)
    base_schedule = parse_schedule(args.base_counts, hub_counts, args.num_bases)
    swap_schedule = parse_schedule(args.swapping_counts, hub_counts, args.num_swapping)
    demand_hour_end = args.demand_hour_end
    if demand_hour_end is None:
        demand_hour_end = args.demand_hour_start + 1

    generator = DataGenerator(seed=args.seed)
    summary = generator.generate_meituan_hierarchical_series(
        raw_data_path=args.raw_data,
        save_root=args.output_root,
        hub_counts=hub_counts,
        num_bases=base_schedule,
        num_swapping=swap_schedule,
        date_filter=args.date_filter,
        map_half_size_km=args.map_half_size_km,
        candidate_date_filter=parse_optional_dates(args.candidate_dates),
        demand_date_filter=parse_optional_dates(args.demand_dates),
        candidate_hour_start=args.candidate_hour_start,
        candidate_hour_end=args.candidate_hour_end,
        demand_hour_start=args.demand_hour_start,
        demand_hour_end=demand_hour_end,
        colocate_base_swaps=not args.no_colocate_base_swaps,
        hierarchy_activity_weight=args.activity_weight,
        hierarchy_balance_weight=args.balance_weight,
        hierarchy_overload_tolerance=args.overload_tolerance,
    )

    print(f"Wrote {len(summary)} instances under {args.output_root}")
    print(f"Summary: {os.path.join(args.output_root, 'series_summary.csv')}")


if __name__ == "__main__":
    main()
