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
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOLVE_SCRIPT = PROJECT_ROOT / "src" / "experiments" / "solve_meituan_scale_gurobi.py"
DEFAULT_INSTANCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "instances"
    / "Meituan_Shenzhen_Nanshan_H020_L05_E05_h12"
)
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "Meituan_Shenzhen_Nanshan_H020_L05_E05_sensitivity_demand050_tp575"
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
    "arcs",
    "original_total_demand",
    "demand_scale",
    "scaled_total_demand",
    "p_chain",
    "transition_para",
    "cost_arc",
    "energy_rate_omega",
    "order_revenue_multiplier",
    "order_revenue_coef",
    "time_limit",
    "mip_gap",
    "threads",
]


@dataclass(frozen=True)
class RunSpec:
    stage: str
    param_name: str
    param_value: Decimal
    p_chain: Decimal
    result_key: str
    cost_arc: Optional[Decimal] = None
    energy_rate_omega: Optional[Decimal] = None
    order_revenue_multiplier: Optional[Decimal] = None


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def value_tag(value: Decimal) -> str:
    text = decimal_text(value)
    return text.replace("-", "m").replace(".", "p")


def parse_decimal_values(raw: str) -> list[Decimal]:
    raw = raw.strip()
    if not raw:
        raise argparse.ArgumentTypeError("value list cannot be empty")

    if ":" in raw:
        parts = [Decimal(part.strip()) for part in raw.split(":") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("range syntax must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("range step must be positive")
        if start > stop:
            raise argparse.ArgumentTypeError("range start must be <= stop")
        values = []
        current = start
        guard = 0
        while current <= stop:
            values.append(current)
            current += step
            guard += 1
            if guard > 10000:
                raise argparse.ArgumentTypeError("range generated too many values")
        return values

    values = [Decimal(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage sensitivity runner for the Shenzhen Nanshan H20/L5/E5 instance."
    )
    parser.add_argument("--instance-dir", default=str(DEFAULT_INSTANCE_DIR))
    parser.add_argument("--period-id", default="h12")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--demand-scale", type=float, default=0.5)
    parser.add_argument("--transition-para", type=float, default=5.75)
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
    parser.add_argument("--p-chain-values", default="0:1:0.1")
    parser.add_argument("--cost-arc-values", default="0:20:2")
    parser.add_argument("--energy-rate-values", default="10:50:10")
    parser.add_argument("--order-revenue-multipliers", default="2:3:0.1")
    return parser


def build_command(args: argparse.Namespace, spec: RunSpec, results_dir: str) -> list[str]:
    command = [
        args.python,
        str(SOLVE_SCRIPT),
        "--instance-dir",
        args.instance_dir,
        "--period-id",
        args.period_id,
        "--results-dir",
        results_dir,
        "--demand-scale",
        str(args.demand_scale),
        "--transition-para",
        str(args.transition_para),
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--p-chain",
        decimal_text(spec.p_chain),
    ]
    if spec.cost_arc is not None:
        command.extend(["--cost-arc", decimal_text(spec.cost_arc)])
    if spec.energy_rate_omega is not None:
        command.extend(["--energy-rate-omega", decimal_text(spec.energy_rate_omega)])
    if spec.order_revenue_multiplier is not None:
        command.extend(["--order-revenue-multiplier", decimal_text(spec.order_revenue_multiplier)])
    return command


def read_summary(results_dir: str) -> dict:
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def run_one(args: argparse.Namespace, spec: RunSpec) -> dict:
    results_dir = os.path.join(args.results_root, spec.stage, spec.result_key)
    log_path = os.path.join(args.results_root, "logs", f"{spec.stage}_{spec.result_key}.log")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    command = build_command(args, spec, results_dir)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(shlex.quote(part) for part in command) + "\n")
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
        "stage": spec.stage,
        "param_name": spec.param_name,
        "param_value": decimal_text(spec.param_value),
        "param_value_float": float(spec.param_value),
        "fixed_p_chain": decimal_text(spec.p_chain),
        "returncode": int(completed.returncode),
        "results_dir": results_dir,
        "log_path": log_path,
    }
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field)
    return row


def write_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "stage",
        "param_name",
        "param_value",
        "param_value_float",
        "fixed_p_chain",
        "returncode",
        "results_dir",
        "log_path",
    ] + SUMMARY_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def run_phase(args: argparse.Namespace, specs: list[RunSpec], label: str) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = [executor.submit(run_one, args, spec) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            obj = row.get("obj")
            status = row.get("status_name")
            print(
                f"{label}: {row['stage']} {row['param_name']}={row['param_value']} "
                f"status={status} obj={obj} returncode={row['returncode']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["stage"], float(row["param_value_float"])))
    return rows


def write_manifest(args: argparse.Namespace, grids: dict, best_p_chain: Optional[dict] = None) -> None:
    payload = {
        "instance_dir": args.instance_dir,
        "period_id": args.period_id,
        "results_root": args.results_root,
        "demand_scale": args.demand_scale,
        "transition_para": args.transition_para,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "parallel_runs": args.parallel_runs,
        "grids": grids,
        "best_p_chain": best_p_chain,
    }
    with open(os.path.join(args.results_root, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    instance_dir = Path(args.instance_dir)
    if not instance_dir.exists():
        raise FileNotFoundError(f"Missing instance directory: {instance_dir}")
    if not (instance_dir / "locations.csv").exists():
        raise FileNotFoundError(f"Missing locations.csv under {instance_dir}")
    if not (instance_dir / "orders" / f"{args.period_id}.csv").exists():
        raise FileNotFoundError(f"Missing orders/{args.period_id}.csv under {instance_dir}")

    os.makedirs(args.results_root, exist_ok=True)
    p_chain_values = parse_decimal_values(args.p_chain_values)
    cost_arc_values = parse_decimal_values(args.cost_arc_values)
    energy_values = parse_decimal_values(args.energy_rate_values)
    revenue_values = parse_decimal_values(args.order_revenue_multipliers)

    grids = {
        "p_chain_values": [decimal_text(value) for value in p_chain_values],
        "cost_arc_values": [decimal_text(value) for value in cost_arc_values],
        "energy_rate_values": [decimal_text(value) for value in energy_values],
        "order_revenue_multipliers": [decimal_text(value) for value in revenue_values],
    }
    write_manifest(args, grids)

    p_specs = [
        RunSpec(
            stage="p_chain",
            param_name="p_chain",
            param_value=value,
            p_chain=value,
            result_key=f"p_chain_{value_tag(value)}",
        )
        for value in p_chain_values
    ]

    all_rows = run_phase(args, p_specs, "phase1")
    write_rows(os.path.join(args.results_root, "all_runs.csv"), all_rows)

    failed = [row for row in all_rows if int(row["returncode"]) != 0]
    if failed:
        detail = ", ".join(f"{row['param_value']}=>{row['returncode']}" for row in failed)
        raise SystemExit(f"P_CHAIN phase failed: {detail}")

    feasible = [row for row in all_rows if row.get("obj") is not None]
    if not feasible:
        raise SystemExit("P_CHAIN phase produced no incumbent solution; cannot select best P_CHAIN.")

    best = max(feasible, key=lambda row: float(row["obj"]))
    best_p = Decimal(str(best["param_value"]))
    best_payload = {
        "p_chain": decimal_text(best_p),
        "obj": best.get("obj"),
        "status_name": best.get("status_name"),
        "gap": best.get("gap"),
        "results_dir": best.get("results_dir"),
    }
    with open(os.path.join(args.results_root, "best_p_chain.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True)
    write_manifest(args, grids, best_payload)
    print(f"Selected best P_CHAIN={decimal_text(best_p)} by max obj={best.get('obj')}", flush=True)

    other_specs: list[RunSpec] = []
    other_specs.extend(
        RunSpec(
            stage="cost_arc",
            param_name="cost_arc",
            param_value=value,
            p_chain=best_p,
            cost_arc=value,
            result_key=f"cost_arc_{value_tag(value)}",
        )
        for value in cost_arc_values
    )
    other_specs.extend(
        RunSpec(
            stage="energy_rate_omega",
            param_name="energy_rate_omega",
            param_value=value,
            p_chain=best_p,
            energy_rate_omega=value,
            result_key=f"energy_{value_tag(value)}",
        )
        for value in energy_values
    )
    other_specs.extend(
        RunSpec(
            stage="order_revenue_multiplier",
            param_name="order_revenue_multiplier",
            param_value=value,
            p_chain=best_p,
            order_revenue_multiplier=value,
            result_key=f"revenue_{value_tag(value)}",
        )
        for value in revenue_values
    )

    all_rows.extend(run_phase(args, other_specs, "phase2"))
    write_rows(os.path.join(args.results_root, "all_runs.csv"), all_rows)

    failed = [row for row in all_rows if int(row["returncode"]) != 0]
    if failed:
        detail = ", ".join(f"{row['stage']}:{row['param_value']}=>{row['returncode']}" for row in failed)
        raise SystemExit(f"Some sensitivity runs failed: {detail}")

    print(f"Wrote aggregate table: {os.path.join(args.results_root, 'all_runs.csv')}", flush=True)


if __name__ == "__main__":
    main()
