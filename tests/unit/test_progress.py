"""Saved convergence evidence must retain all emitted events and their clocks."""

import json

from src.experiments.progress import RunMonitor


def test_progress_retains_ordered_events_and_latest_snapshot(tmp_path):
    monitor = RunMonitor(tmp_path)
    monitor.progress({"stage": "enumeration", "columns": 0})
    monitor.progress({"stage": "benders", "objective": 12.0, "bound": 10.0})
    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [event["stage"] for event in events] == ["enumeration", "benders"]
    assert 0 <= events[0]["elapsed"] <= events[1]["elapsed"]
    assert json.loads((tmp_path / "progress.json").read_text()) == events[-1]
