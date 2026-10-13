"""Incumbent, bounds and progress records for a Benders run."""

import time
import numpy as np
from gurobipy import GRB
from src.model.network import network
from src.model.provenance import source_hashes


class BendersState:
    def __init__(
        self,
        data,
        best,
        started,
        budget,
        params,
        features,
        progress=None,
        checkpoint=None,
    ):
        self.data, self.best = data, best
        self.started, self.budget = started, budget
        self.params, self.features = params, features
        self.progress, self.checkpoint = progress, checkpoint
        self.net = network(data)
        self.source_sha256 = source_hashes()
        self.lower = 0.0
        self.status = GRB.LOADED
        self.master_runtime = self.nodes = self.last_report = 0.0
        self.trace, self.errors = [], []
        self.stats = dict(
            master_binaries=0,
            master_integers=0,
            master_continuous=0,
            recourse_calls=0,
            benders_cuts=0,
            cg_iterations=0,
            pricing_calls=0,
            pricing_wall=0.0,
            recourse_wall=0.0,
            generated_columns=0,
            complete_subproblems=0,
            interrupted_subproblems=0,
            pareto_attempts=0,
            pareto_cuts=0,
            pareto_rejected=0,
            pareto_interrupted=0,
            pareto_wall=0.0,
            pareto_cg_iterations=0,
            pareto_pricing_calls=0,
            pareto_pricing_wall=0.0,
        )

    def gap(self):
        return max(0.0, self.best["objective"] - self.lower) / max(
            1.0, abs(self.best["objective"])
        )

    def snapshot(self, reason="running", stage=None):
        data, best, net = self.data, self.best, self.net
        sol = best["solution"]
        L, H = data["n_l"], data["n_h"]
        result = dict(
            name=data.get("name", "path_instance"),
            model_version="path_cqn_v1",
            method="binary_master_benders_CG",
            pricing_backend=self.features["pricing"],
            accelerations=self.features.copy(),
            objective=float(best["objective"]),
            bound=float(self.lower),
            gap=self.gap(),
            wall=time.perf_counter() - self.started,
            budget=self.budget,
            status=self.status,
            termination_reason=reason,
            master_runtime=self.master_runtime,
            nodes=self.nodes,
            stats=self.stats.copy(),
            trace=list(self.trace),
            errors=list(self.errors),
            params=self.params.copy(),
            fleet=float(sol["fleet"]),
            served=float(best["served"]),
            demand=float(np.sum(data["demand"])),
            n_od=data["n_od"],
            rho=sol["rho"].tolist(),
            cabinet_total=float(sol["hub_count"].sum()),
            cabinet_counts=sol["hub_count"].tolist(),
            hub_load=best["hub_load"].tolist(),
            airline_load=best["airline_load"].tolist(),
            primal_residual=float(best["primal_residual"]),
            opened=dict(
                L=int(sol["facility"][:L].sum()),
                H=int(sol["facility"][L : L + H].sum()),
                E=int(sol["facility"][L + H :].sum()),
            ),
            open_airlines=int(sol["airline"][net["physical"]].sum()),
            open_connectors=int(sol["airline"][~net["physical"]].sum()),
            airline_fixed_cost=float(net["fees"] @ sol["airline"]),
            all_ground_cost=float(data["demand"] @ data["ground"]),
            provenance=data.get("provenance", {}),
            source_sha256=self.source_sha256,
            _solution=sol,
            _missions=best["missions"],
            _relocations=best["relocations"],
        )
        if stage is not None:
            result["checkpoint_stage"] = stage
        return result

    def save(self, stage, reason="running"):
        if self.checkpoint is not None:
            self.checkpoint(self.snapshot(reason, stage))

    def emit(self, stage, **extra):
        self.last_report = time.perf_counter()
        if self.progress is not None:
            result = self.snapshot()
            event = {
                k: result[k]
                for k in (
                    "objective",
                    "bound",
                    "gap",
                    "fleet",
                    "served",
                    "rho",
                    "opened",
                    "cabinet_total",
                    "stats",
                )
            }
            self.progress(
                dict(
                    stage=stage,
                    elapsed=self.last_report - self.started,
                    **event,
                    **extra,
                )
            )

    def record_subproblem(self, answer):
        stats = self.stats
        for stat, field in (
            ("recourse_wall", "wall"),
            ("pricing_calls", "pricing_calls"),
            ("pricing_wall", "pricing_wall"),
            ("cg_iterations", "cg_iterations"),
            ("generated_columns", "pool_size"),
        ):
            stats[stat] += answer[field]
        stats["complete_subproblems"] += int(answer["cg_complete"])

    def record_pareto(self, answer, elapsed):
        """Record the auxiliary solve separately from ordinary recourse."""
        self.stats["pareto_wall"] += elapsed
        for name in ("cg_iterations", "pricing_calls", "pricing_wall"):
            self.stats[f"pareto_{name}"] += answer.get(name, 0)
        accepted = (
            answer.get("accepted", False)
            and answer.get("cut_valid", False)
            and answer.get("certified", False)
        )
        interrupted = answer.get("reason") in {
            "time_limit",
            "user_stop",
            "incomplete_pricing",
        }
        self.stats["pareto_interrupted"] += int(interrupted)
        self.stats["pareto_rejected"] += int(not accepted and not interrupted)
