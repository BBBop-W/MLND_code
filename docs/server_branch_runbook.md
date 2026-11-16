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

## What happens before the long run

1. Acquire the same global batch lock used by the old launcher. A running old or
   new capacity batch prevents a second launch and prevents environment updates.
2. Require branch `MLND_sever` and no tracked local changes. Freeze files from the
   exact Git commit, verify all 69 `SOURCE_VERSION.json` entries, and copy inputs
   plus map provenance into a new `outputs/server_runs/run-...` directory.
3. Configure and fully link target `mlnd` with CMake, one compilation job by default.
   Record source, Git commit, build command, platform and actual binary SHA-256.
4. On Linux, verify the resource controls with short synthetic process tests:
   address-space limits, sampled process-group RSS, wall-time cleanup, and cleanup
   after dispatcher failure. These tests neither solve MLND nor touch other jobs.
5. Solve the included small L2/H3/OD3 input for K1 and K2, each with a 60-second
   native budget. Both must complete normally, pass independent physical validation,
   and meet 1% gap before the batch is marked prepared.
6. Launch the suite in the background with its own session and log. Keep the
   terminal connected through preparation; after the launcher reports success,
   closing the terminal does not stop the batch. No tmux command is required.

`bash START_SERVER.sh --prepare-only` performs steps 1–5 without starting the
156-job matrix. `bash START_SERVER.sh --resume` then launches that prepared run.

## Matrix and resource policy

The recovered matrix is unchanged: 2 serial preflight jobs, 98 core jobs and 56
stress jobs, using 78 inputs across approximately 9, 33.33, 64 and 100 km² study
windows. The two preflight jobs must complete normally, pass independent physical
validation and reach 1% before later stages are admitted. The matrix describes
capacity screening with one seed, not a replicated paper experiment.

Every matrix task uses the current algorithm (`--method benders_cg`), native budget
7200 seconds, target gap 0.01, seed 43 and one solver thread. Historical preset
labels in the input matrix are provenance only: new runner flags and the native
`resolved_configuration` determine the actual current implementation.

Default concurrency is **1**, pending measured Linux memory behavior of this
implementation. A new batch may specify `--max-parallel 2` through 8. The 24 GiB
reservation budget and 8 GiB available-memory floor still apply. Each job retains
its existing 3/6/10/16/24 GiB class; 24 GiB jobs and preflight jobs run exclusively.
These classes are starting limits, not measured guarantees for the new solver.

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

- `setup.log`, `batch.log`, `prepared.json`, `acceptance.json` and Linux resource checks.
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

## Delivery checks

The implementation preserves the 69 synchronized source/build dependency files.
On macOS arm64, a fresh root-CMake build linked successfully, and fresh K1/K2 small
cases passed independent physical validation and the 1% criterion. Portable tests
cover runner failures/timeouts, provenance rejection, resume acceptance, resource
admission, historical status-merge and pidfd compatibility cases, and map evidence.
The actual 156-job input matrix passes dry-run validation.

Linux compilation, CPLEX licensing and kernel resource controls remain target-host
checks and are required by the launcher before the long matrix begins. No new
156-job experiment or Linux performance result is claimed by this delivery.
