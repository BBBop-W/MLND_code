#!/usr/bin/env bash
# Independent full-model CPLEX batch over the same capacity inputs.
set -Eeuo pipefail
REPO=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$REPO/START_SERVER.sh" "$@" --method full_path
