"""Progress, complete checkpoints, and graceful stop requests."""

from datetime import datetime, timezone
from pathlib import Path
import json
import signal
import shutil
import time

from .storage import write_json, save_solution, json_payload


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunMonitor:
    def __init__(self, output):
        self.output = Path(output)
        self.requested = False
        self.count = 0
        self.handlers = {}
        self.stage = "benders"
        self._next_stop_poll = float("-inf")
        self.started = time.perf_counter()

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        return self

    def __exit__(self, *_):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)

    def _signal(self, *_):
        self.requested = True

    def stop_requested(self):
        # Solver/model-building callbacks can call this millions of times.
        # Signals remain immediate; file polling has at most 50 ms latency.
        # Once observed, a stop remains active even if its file is removed.
        if self.requested:
            return True
        now = time.monotonic()
        if now < self._next_stop_poll:
            return False
        self._next_stop_poll = now + 0.05
        file_requested = (self.output / "STOP").exists() or (
            self.output.parent / "STOP"
        ).exists()
        if file_requested:
            self.requested = True
        return self.requested

    def progress(self, record):
        event = {
            "utc": utc_now(),
            "elapsed": time.perf_counter() - self.started,
            "stage": self.stage,
            **record,
        }
        write_json(self.output / "progress.json", event)
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_payload(event), allow_nan=False) + "\n")

    def checkpoint(self, result):
        if result.get("_solution") is None:
            return
        self.count += 1
        directory = self.output / "checkpoints" / f"{self.count:06d}"
        save_solution(result, directory)
        write_json(
            self.output / "checkpoint.json",
            {
                "utc": utc_now(),
                "directory": str(directory.relative_to(self.output)),
                "objective": result.get("objective"),
                "bound": result.get("bound"),
            },
        )
        # Three complete checkpoints are sufficient for recovering a stopped run.
        for previous in sorted(directory.parent.iterdir())[:-3]:
            if previous.is_dir():
                shutil.rmtree(previous)
