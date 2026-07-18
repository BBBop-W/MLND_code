import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOLVE_SCRIPT = PROJECT_ROOT / "src" / "experiments" / "solve_meituan_scale_gurobi.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "src" / "experiments" / "generate_shenzhen_nanshan_candidates.py"
DEFAULT_RAW_DATA = PROJECT_ROOT / "Meituan-Shenzhen" / "深圳市南山区运单抽样_已脱敏.csv"
DEFAULT_INSTANCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "instances"
    / "Meituan_Shenzhen_Nanshan_H020_L05_E05_h12"
)
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "Meituan_Shenzhen_Nanshan_H020_L05_E05_demand_pchain"
)

SUMMARY_FIELDS = [
    "status_name",
    "status",
    "sol_count",
    "obj",
    "bound",
    "gap",
    "runtime",
    "wall_time",
    "H",
    "L",
    "E",
    "OD",
    "original_total_demand",
    "demand_scale",
    "scaled_total_demand",
    "arcs",
    "p_chain",
    "transition_para",
    "order_revenue_multiplier",
    "order_revenue_coef",
    "cost_flight_time",
    "time_limit",
    "mip_gap",
    "threads",
]


@dataclass(frozen=True)
class RunSpec:
    demand_scale: Decimal
    p_chain: Decimal

    @property
    def result_key(self) -> str:
        return f"demand_{value_tag(self.demand_scale)}__p_chain_{value_tag(self.p_chain)}"


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def value_tag(value: Decimal) -> str:
    return decimal_text(value).replace("-", "m").replace(".", "p")


def parse_decimal_values(raw: str) -> list[Decimal]:
    raw = raw.strip()
    if not raw:
        raise argparse.ArgumentTypeError("value list cannot be empty")

    if ":" in raw:
        parts = [Decimal(part.strip()) for part in raw.split(":") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("range syntax must be start:stop:step")
        start, stop, step = parts
        if step == 0:
            raise argparse.ArgumentTypeError("range step cannot be zero")
        values = []
        current = start
        guard = 0
        if step > 0:
            keep_going = lambda v: v <= stop
        else:
            keep_going = lambda v: v >= stop
        while keep_going(current):
            values.append(current)
            current += step
            guard += 1
            if guard > 10000:
                raise argparse.ArgumentTypeError("range generated too many values")
        if not values:
            raise argparse.ArgumentTypeError("range produced no values")
        return values

    values = [Decimal(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return values


def command_for_log(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(shlex.quote(part) for part in command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run demand-scale x p_chain grid on the Shenzhen Nanshan H20/L5/E5 instance."
    )
    parser.add_argument("--instance-dir", default=str(DEFAULT_INSTANCE_DIR))
    parser.add_argument("--raw-data", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--period-id", default="h12")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--time-limit", type=float, default=7200.0)
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0001,
        help="Gurobi default relative MIP gap is 0.0001, i.e. 0.01%%.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-runs", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-hubs", type=int, default=20)
    parser.add_argument("--num-bases", type=int, default=5)
    parser.add_argument("--num-swapping", type=int, default=5)
    parser.add_argument("--transition-para", type=float, default=5.75)
    parser.add_argument("--demand-scale-values", default="1:0.1:-0.1")
    parser.add_argument("--p-chain-values", default="0:1:0.1")
    parser.add_argument("--generate-if-missing", dest="generate_if_missing", action="store_true")
    parser.add_argument("--no-generate-if-missing", dest="generate_if_missing", action="store_false")
    parser.set_defaults(generate_if_missing=True)
    return parser


def count_location_types(instance_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    locations_path = instance_dir / "locations.csv"
    with open(locations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row["type"]
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def validate_instance(args: argparse.Namespace) -> None:
    instance_dir = Path(args.instance_dir)
    missing = [
        path
        for path in (
            instance_dir / "locations.csv",
            instance_dir / "orders" / f"{args.period_id}.csv",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required instance files: "
            + ", ".join(str(path) for path in missing)
        )

    counts = count_location_types(instance_dir)
    expected = {"H": args.num_hubs, "L": args.num_bases, "E": args.num_swapping}
    mismatches = {
        kind: (counts.get(kind, 0), expected_count)
        for kind, expected_count in expected.items()
        if counts.get(kind, 0) != expected_count
    }
    if mismatches:
        detail = ", ".join(
            f"{kind} actual={actual} expected={expected_count}"
            for kind, (actual, expected_count) in mismatches.items()
        )
        raise ValueError(f"Instance scale mismatch under {instance_dir}: {detail}")


def generate_instance_if_needed(args: argparse.Namespace) -> None:
    instance_dir = Path(args.instance_dir)
    if (instance_dir / "locations.csv").exists() and (instance_dir / "orders" / f"{args.period_id}.csv").exists():
        validate_instance(args)
        return

    if not args.generate_if_missing:
        validate_instance(args)
        return

    raw_data = Path(args.raw_data)
    if not raw_data.exists():
        raise FileNotFoundError(f"Cannot generate instance because raw data is missing: {raw_data}")

    os.makedirs(args.results_root, exist_ok=True)
    log_path = Path(args.results_root) / "generate_instance.log"
    command = [
        args.python,
        str(GENERATOR_SCRIPT),
        "--raw-data",
        str(raw_data),
        "--output-dir",
        str(instance_dir),
        "--num-hubs",
        str(args.num_hubs),
        "--num-bases",
        str(args.num_bases),
        "--num-swapping",
        str(args.num_swapping),
        "--demand-hour-start",
        "12",
        "--demand-hour-end",
        "13",
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(command_for_log(command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"Instance generation failed with returncode={completed.returncode}. See {log_path}")
    validate_instance(args)


def build_solve_command(args: argparse.Namespace, spec: RunSpec, results_dir: str) -> list[str]:
    return [
        args.python,
        str(SOLVE_SCRIPT),
        "--instance-dir",
        args.instance_dir,
        "--period-id",
        args.period_id,
        "--results-dir",
        results_dir,
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--demand-scale",
        decimal_text(spec.demand_scale),
        "--p-chain",
        decimal_text(spec.p_chain),
        "--transition-para",
        str(args.transition_para),
    ]


def read_summary(results_dir: str) -> dict:
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def run_one(args: argparse.Namespace, spec: RunSpec) -> dict:
    results_dir = os.path.join(args.results_root, "runs", spec.result_key)
    log_path = os.path.join(args.results_root, "logs", f"{spec.result_key}.log")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    command = build_solve_command(args, spec, results_dir)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(command_for_log(command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    summary = read_summary(results_dir)
    row = {
        "demand_scale_requested": decimal_text(spec.demand_scale),
        "p_chain_requested": decimal_text(spec.p_chain),
        "returncode": int(completed.returncode),
        "results_dir": results_dir,
        "log_path": log_path,
    }
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field)
    return row


def write_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "demand_scale_requested",
        "p_chain_requested",
        "returncode",
        "results_dir",
        "log_path",
    ] + SUMMARY_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_manifest(args: argparse.Namespace, demand_values: list[Decimal], p_values: list[Decimal]) -> None:
    payload = {
        "instance_dir": args.instance_dir,
        "period_id": args.period_id,
        "results_root": args.results_root,
        "raw_data": args.raw_data,
        "num_hubs": args.num_hubs,
        "num_bases": args.num_bases,
        "num_swapping": args.num_swapping,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "parallel_runs": args.parallel_runs,
        "transition_para": args.transition_para,
        "demand_scale_values": [decimal_text(value) for value in demand_values],
        "p_chain_values": [decimal_text(value) for value in p_values],
    }
    with open(os.path.join(args.results_root, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    os.makedirs(args.results_root, exist_ok=True)
    generate_instance_if_needed(args)

    demand_values = parse_decimal_values(args.demand_scale_values)
    p_values = parse_decimal_values(args.p_chain_values)
    write_manifest(args, demand_values, p_values)

    specs = [
        RunSpec(demand_scale=demand_scale, p_chain=p_chain)
        for demand_scale in demand_values
        for p_chain in p_values
    ]

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = [executor.submit(run_one, args, spec) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"demand_scale={row['demand_scale_requested']} "
                f"p_chain={row['p_chain_requested']} "
                f"status={row.get('status_name')} obj={row.get('obj')} "
                f"returncode={row['returncode']}",
                flush=True,
            )

    rows.sort(key=lambda row: (-float(row["demand_scale_requested"]), float(row["p_chain_requested"])))
    write_rows(os.path.join(args.results_root, "all_runs.csv"), rows)

    failed = [row for row in rows if int(row["returncode"]) != 0]
    if failed:
        detail = ", ".join(
            f"demand={row['demand_scale_requested']},p={row['p_chain_requested']}=>{row['returncode']}"
            for row in failed
        )
        raise SystemExit(f"Some demand-pchain runs failed: {detail}")

    print(f"Wrote aggregate table: {os.path.join(args.results_root, 'all_runs.csv')}", flush=True)


if __name__ == "__main__":
    main()
