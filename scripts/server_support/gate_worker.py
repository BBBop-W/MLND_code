"""Apply inherited RLIMIT_AS before releasing a Linux job payload."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from job_control import atomic_json


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    directory = spec_path.parent
    guardian_pid = os.getppid()
    try:
        limit = int(spec["memory_bytes"])
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if (soft, hard) != (limit, limit):
            raise RuntimeError("RLIMIT_AS readback differs from request")
        atomic_json(directory / "gate_ready.json", {"pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0),
                    "rlimit_as_soft": soft, "rlimit_as_hard": hard, "guardian_pid": guardian_pid})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if os.getppid() != guardian_pid:
                raise RuntimeError("Guardian disappeared before payload launch")
            try:
                ready = json.loads((directory / "launch_gate.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                time.sleep(.02)
                continue
            if ready.get("token") != spec["token"]:
                raise ValueError("Launch gate token mismatch")
            return subprocess.call(spec["command"], cwd=spec["cwd"])
        raise TimeoutError("Guardian did not release launch gate")
    except BaseException as error:
        atomic_json(directory / "gate_error.json", {"error": f"{type(error).__name__}: {error}"})
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
