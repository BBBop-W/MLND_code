# MLND_sever Linux experiments

This branch is the server source checkout. `START_SERVER.sh` builds the native
target `mlnd` from the root CMake project and uses `mlnd_runtime` for independent
physical validation. The historical `mlnd_cpp`, `run_server_experiment.py`, and
the old archive's `source/` are not used by this entrypoint.

## Start on the existing server

In the Git checkout at `/home/autumn/workspace/wangsongtao/MLND_code`:

```bash
cd /home/autumn/workspace/wangsongtao/MLND_code
git fetch origin
git switch MLND_sever
git pull --ff-only origin MLND_sever
bash START_SERVER.sh
```

The first command sequence assumes this directory is an existing Git checkout
with `origin` pointing to `BBBop-W/MLND_code`. If it is only the old archive
directory, create a separate checkout and point at the existing input suite:

```bash
cd /home/autumn/workspace/wangsongtao
git clone --single-branch --branch MLND_sever --depth 1 https://github.com/BBBop-W/MLND_code.git MLND_code_git
cd MLND_code_git
bash START_SERVER.sh --suite-root /home/autumn/workspace/wangsongtao/MLND_code/MLND_LINUX_ALL_20261115/suite
```

The launcher finds the existing Conda installation and creates an isolated
environment under `outputs/server_environment`. No sudo is needed. Initial
environment installation needs access to conda-forge and the Python package
index. CPLEX is not downloaded or included: the default installation is
`/home/autumn/workspace/wangsongtao/cplex`, including C++/Concert headers,
static libraries, and a working license. Override it with `--cplex-root PATH`.

The existing input bundle must still be available. By default the launcher
looks for `MLND_LINUX_ALL_20261115/suite` under this checkout and at the established
server path. It uses the 78 input directories, the 156-job matrix, and the offline
map backgrounds. It never executes the old bundle's solver or scripts. Input data,
maps, native binaries, environments and results stay outside Git.

## Startup

1. Acquire the same global batch lock used by the old launcher. A running old or
   new capacity batch prevents a second launch and prevents environment updates.
2. Require branch `MLND_sever` and no tracked local changes. Freeze files from the
   exact Git commit, verify all 69 `SOURCE_VERSION.json` entries, and copy inputs
   plus map provenance into a new `outputs/server_runs/run-...` directory.
3. Configure and fully link target `mlnd` with CMake, one compilation job by default.
   Record source, Git commit, build command, platform and actual binary SHA-256.
4. Launch the suite immediately in the background with its own session and log.
   There are no startup resource self-tests, small acceptance solves or preflight
   qualification gates. Keep the terminal connected through preparation; after
   launch succeeds, closing the terminal does not stop the batch.

`bash START_SERVER.sh --prepare-only` optionally builds and freezes the batch
without launching it. `bash START_SERVER.sh --resume` then starts that prepared run.
A normal `bash START_SERVER.sh` builds and starts the matrix directly.

## Matrix and resource policy

All 156 recovered tasks are retained: 100 core jobs and 56 stress jobs, using
78 inputs across approximately 9, 33.33, 64 and 100 km² study windows. The two
historical preflight rows are ordinary core jobs; their original labels are kept
as `source_stage` for provenance. They do not run exclusively or gate other tasks.
The matrix describes capacity screening with one seed, not a replicated paper
experiment.

Every matrix task uses the current algorithm (`--method benders_cg`), native budget
7200 seconds, target gap 0.01, seed 43 and one solver thread. Historical preset
labels in the input matrix are provenance only: new runner flags and the native
`resolved_configuration` determine the actual current implementation.

ALNS (`primal_alns`), both primal support completions
(`primal_support_completion`, `indexed_support_completion`) and CG primal feedback
(`cg_primal_feedback`) are enabled for every base count and both K1 and K2.
The underlying support and repair logic accepts all bases; the former L2/K2-only
configuration restriction is removed. Other algorithm settings are unchanged.

Default concurrency is **8**. The existing 24 GiB reservation budget and 8 GiB
available-memory floor still apply. Each job retains its 3/6/10/16/24 GiB class;
24 GiB jobs run exclusively. Memory admission can therefore reduce the number of
simultaneously active jobs below eight. A new batch may explicitly choose
`--max-parallel 1` through 8.

The server previously reported cgroup v1 without sudo. Controls therefore use
inherited per-process `RLIMIT_AS` and sampled process-group RSS. They are not an
aggregate RSS hard cap, and they do not disable swap. The launcher rechecks live
host resources. Native solver wall allowance is 7500 seconds, independent
validation 1800, map rendering 90, and the outer job allowance 9450 seconds.

## Status, recovery and results

```bash
bash START_SERVER.sh --status
bash START_SERVER.sh --resume
# Explicitly create new attempts for prior failed/interrupted jobs:
bash START_SERVER.sh --resume --retry-failed
```

`--resume` uses the latest prepared run's frozen scripts, binary, inputs, stage and
resource settings. It checks their hashes and the recorded Python environment.
Verified successful attempts are skipped; failed attempts are retained and skipped
unless `--retry-failed` is supplied. An incomplete preparation is preserved; start
without `--resume` to prepare a new run. Resuming does not replace old results with
new-code results. A later `git pull` does not change a running batch's frozen files.

Each run contains:

- `setup.log`, `batch.log` and `prepared.json`.
- `source/SNAPSHOT.json`, `suite/SERVER_INPUT_SNAPSHOT.json`, `build/build_manifest.json`.
- `results/suite_status.json` and `results/summary.csv` / `summary.json`.
- `results/jobs/JOB/attempts/NNN/`: resource records and `native/` solver output,
  physical validation certificate, task status, logs and offline network maps.

Summary rows retain original native objective, global-bound provenance, gap,
termination reason, exit status, physical validation, memory observations and map
status. `completed` does not mean 1%; `qualified_baseline` requires normal completion,
physical feasibility and a consistent native global gap within 1%. A saved feasible
solution from an abnormal termination remains explicitly abnormal. Independent
physical validation does not independently certify the solver's global lower bound.

Maps use the frozen real basemap and display L, H and connections, with no OD nodes.
Map failure is reported separately and does not change the native solve or primal
feasibility result. All original files remain available for inspection.

## Development validation

The 2026-11-17 change passed 60 portable server tests and a fresh macOS arm64
native build. Eight short synthetic cases (L=2/3/5/8, K=1/2, H=3, OD=3) all ran
ALNS, retained all four requested flags, passed independent physical validation,
and reached 1% gap. Direct native support/repair tests additionally cover L=1
and witnesses available only at later bases. These are correctness checks, not
large-instance performance evidence.

These development tests are not part of server startup. The launcher builds and
starts the requested matrix directly; it does not run the short cases or require
a preflight result. Regression coverage includes this startup behavior, the
eight-job default, and compatibility with all 156 historical matrix rows.

Linux execution and the full 156-job matrix have not been run by this delivery.
Each real task still records termination, gap and independent post-solve physical
validation; these result checks do not gate admission of later tasks.
