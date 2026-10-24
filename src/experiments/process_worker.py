"""POSIX process wrapper: preserve the child exit code and persist wait4 usage.

This module does not import a solver. Unlike /usr/bin/time on macOS, it needs
no global sysctl process permissions merely to collect its own child's usage.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(command, output):
    started = time.monotonic()
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": command}
    received = []
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        # The child receives the same group signal. Wait for its final writes.
        handlers[sig] = signal.signal(
            sig, lambda number, _frame: received.append(number)
        )
    try:
        process = subprocess.Popen(command)
        record["child_pid"] = process.pid
        _, status, usage = os.wait4(process.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        process.returncode = code
        record.update(
            returncode=code,
            cpu_user_seconds=usage.ru_utime,
            cpu_system_seconds=usage.ru_stime,
            peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            minor_faults=usage.ru_minflt,
            major_faults=usage.ru_majflt,
            voluntary_context_switches=usage.ru_nvcsw,
            involuntary_context_switches=usage.ru_nivcsw,
            collection="os.wait4 child rusage",
            status="collected",
        )
    except BaseException as exc:
        record.update(status="wrapper_failed", error=f"{type(exc).__name__}: {exc}")
        code = 125
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    record["received_signals"] = received
    record.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - started,
    )
    output = Path(output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A child command is required")
    raise SystemExit(run(command, args.usage_file))


if __name__ == "__main__":
    main()
