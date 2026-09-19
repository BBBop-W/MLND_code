"""Progress, complete checkpoints, and graceful stop requests."""

from datetime import datetime, timezone
from pathlib import Path
import signal
import shutil

from .storage import write_json, save_solution


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunMonitor:
    def __init__(self, output):
        self.output = Path(output)
        self.requested = False
        self.count = 0
        self.handlers = {}
        self.stage = "benders"

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
        return (
            self.requested
            or (self.output / "STOP").exists()
            or (self.output.parent / "STOP").exists()
        )

    def progress(self, record):
        write_json(
            self.output / "progress.json",
            {"utc": utc_now(), "stage": self.stage, **record},
        )

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
