#!/usr/bin/env python3
"""Compile/run the generalized ALNS regression against a current CMake build.

Example (Unix Makefiles generator, macOS/Linux):
  python cpp/tests/run_general_alns_probe.py --build-dir outputs/check/build
First build the production mlnd target in that directory. The probe reuses its
compiler flags, dependency libraries and native objects, replacing only main.
All generated files remain in the supplied build directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    metadata = build / "CMakeFiles/mlnd.dir"
    if not (metadata / "flags.make").is_file() or not (metadata / "link.txt").is_file():
        parser.error("--build-dir must contain a completed Unix Makefiles mlnd build")
    flags = {}
    for line in (metadata / "flags.make").read_text().splitlines():
        if " = " in line:
            name, value = line.split(" = ", 1)
            flags[name] = shlex.split(value)
    link = shlex.split((metadata / "link.txt").read_text())
    mains = [i for i, item in enumerate(link) if item.endswith("/main.cpp.o")]
    if len(mains) != 1 or "-o" not in link:
        parser.error("unsupported mlnd link command; expected one main.cpp.o and -o")
    source = Path(__file__).with_name("general_alns_probe.cpp").resolve()
    obj = build / "general_alns_probe.cpp.o"
    executable = build / "general_alns_probe"
    subprocess.run([link[0], *flags.get("CXX_DEFINES", []),
                    *flags.get("CXX_INCLUDES", []), *flags.get("CXX_FLAGS", []),
                    "-c", str(source), "-o", str(obj)], cwd=build, check=True)
    link[mains[0]] = str(obj)
    link[link.index("-o") + 1] = str(executable)
    subprocess.run(link, cwd=build, check=True)
    return subprocess.run([str(executable)], cwd=build, timeout=180).returncode


if __name__ == "__main__":
    raise SystemExit(main())
