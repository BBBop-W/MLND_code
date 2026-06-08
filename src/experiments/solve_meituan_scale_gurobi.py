import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from gurobipy import GRB

from src import config
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.utils.output_processor import ResultProcessor


@contextmanager
def override_config(**kwargs):
    old_values = {name: getattr(config, name) for name in kwargs}
    try:
        for name, value in kwargs.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(config, name, value)


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _series_rows(series_root: str) -> list[dict[str, str]]:
    summary_path = os.path.join(series_root, "series_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing series summary: {summary_path}")
    rows = _read_csv_rows(summary_path)
    if not rows:
        raise ValueError(f"No rows in {summary_path}")
    return rows


def available_hub_counts(series_root: str) -> list[int]:
    return sorted({int(row["H"]) for row in _series_rows(series_root)})


def infer_instance_dir(series_root: str, hub_count: int | None = None) -> str:
    rows = _series_rows(series_root)
    if hub_count is None:
        best = min(rows, key=lambda row: int(row["H"]))
        return best["instance_dir"]

    matches = [row for row in rows if int(row["H"]) == int(hub_count)]
    if not matches:
        available = ", ".join(str(v) for v in sorted({int(row["H"]) for row in rows}))
        raise ValueError(f"H={hub_count} is not available under {series_root}. Available H values: {available}")
    return matches[0]["instance_dir"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a selected Meituan H-series instance with Gurobi.")
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
        help="Root containing series_summary.csv. Used when --instance-dir is omitted.",
    )
    parser.add_argument(
        "--hub-count",
        "--scale",
        dest="hub_count",
        type=int,
        default=30,
        help="H scale to solve, for example 20, 40, or 100. Defaults to the minimum H in the series.",
    )
    parser.add_argument(
        "--list-scales",
        action="store_true",
        help="Print available H scales in --series-root and exit.",
    )
    parser.add_argument(
        "--instance-dir",
        default=None,
        help="Specific instance directory to solve. Overrides --hub-count.",
    )
    parser.add_argument("--period-id", default="h12")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory. Defaults to results/Meituan_H_series/Hxxx_gurobi.",
    )
    parser.add_argument("--time-limit", type=float, default=7200.0)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--write-lp", action="store_true")
    parser.add_argument("--plot-network", action="store_true")
    parser.add_argument(
        "--print-solution-summary",
        "--verbose-solution",
        action="store_true",
        help="Print ResultProcessor's detailed solution parsing summary to the run log.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_scales:
        print(",".join(str(v) for v in available_hub_counts(args.series_root)))
        return

    instance_dir = args.instance_dir or infer_instance_dir(args.series_root, args.hub_count)
    inst = DataLoader().load_instance_dir(instance_dir, period_id=args.period_id)

    if args.results_dir is None:
        args.results_dir = os.path.join(
            PROJECT_ROOT,
            "results",
            "Meituan_H_series",
            f"H{len(inst.H):03d}_gurobi",
        )
    os.makedirs(args.results_dir, exist_ok=True)

    model = MLNDModel(inst)

    with override_config(MODEL_TIME_LIMIT=args.time_limit, MODEL_MIP_GAP=args.mip_gap):
        model.build()
        model.model.setParam("Threads", int(args.threads))
        model.apply_solver_config()
        if args.write_lp:
            model.write_model(os.path.join(args.results_dir, "model.lp"))
        start = time.perf_counter()
        model.model.optimize()
        wall_time = time.perf_counter() - start

    sol_count = int(model.model.SolCount)
    status = int(model.model.Status)
    status_name = _status_name(status)
    obj = float(model.model.ObjVal) if sol_count > 0 else None
    bound = _safe_model_attr(model.model, "ObjBound")
    gap = _safe_model_attr(model.model, "MIPGap")

    summary = {
        "instance_dir": instance_dir,
        "period_id": args.period_id,
        "requested_hub_count": args.hub_count,
        "H": len(inst.H),
        "L": len(inst.L),
        "S": len(inst.S),
        "E": len(inst.E),
        "OD": len(inst.h),
        "arcs": len(inst.A),
        "status": status,
        "status_name": status_name,
        "sol_count": sol_count,
        "obj": obj,
        "bound": bound,
        "gap": gap,
        "runtime": float(model.model.Runtime),
        "wall_time": wall_time,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "mu_hub": list(config.MU_HUB),
        "cost_hub": list(config.COST_HUB),
    }

    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_summary_csv(os.path.join(args.results_dir, "summary.csv"), summary)

    if sol_count > 0:
        processor = ResultProcessor(model, inst)
        if args.print_solution_summary:
            processor.print_summary()
        processor.save_node_flow_summary(os.path.join(args.results_dir, "node_flow_summary.csv"))
        processor.save_selected_stations(os.path.join(args.results_dir, "selected_stations.csv"))
        processor.save_open_facilities_config(os.path.join(args.results_dir, "open_facilities.csv"))
        processor.save_python_object(os.path.join(args.results_dir, "result_data.pkl"))
        if args.plot_network:
            processor.plot_network(os.path.join(args.results_dir, "network_plot.png"))

    print(
        f"Solved {os.path.basename(instance_dir)}: status={status_name} "
        f"obj={obj} bound={bound} gap={gap} sol_count={sol_count} "
        f"runtime={summary['runtime']:.2f}s"
    )
    print(f"Results: {args.results_dir}")


def _safe_model_attr(model, name: str):
    try:
        value = getattr(model, name)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return value


def _status_name(status: int) -> str:
    names = {}
    for name in (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
    ):
        value = getattr(GRB, name, None)
        if value is not None:
            names[int(value)] = name
    return names.get(status, str(status))


def _write_summary_csv(path: str, summary: dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
