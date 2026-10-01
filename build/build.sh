#!/usr/bin/env bash
# Builds the Mojo kernels into python/_mojo_sparse.so.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Bare `mojo build` does not set MODULAR_HOME; without it every std import fails
# with "unable to locate module 'std'".  pixi run sets it, derive it otherwise.
if [ -z "${MODULAR_HOME:-}" ]; then
    MODULAR_HOME="$(dirname "$(dirname "$(command -v mojo)")")/share/max"
fi
export MODULAR_HOME

out="$root/python/_mojo_sparse.so"
mkdir -p "$(dirname "$out")"

# Everything lives in one compilation unit: build cost is fixed, so splitting
# the kernels across files only multiplies link time.

# Compiler thread count.  0 (the default) uses every available CPU; override it
# on hosts with a tight cgroup pids budget, where LLVM's thread pool fails to
# spawn with "pthread_create failed: Resource temporarily unavailable".
threads="${MOJO_BUILD_THREADS:-0}"
args=()
if [ "$threads" != "0" ]; then
    args+=(-j "$threads")
fi
mojo build "${args[@]}" --emit shared-lib -o "$out" "$root/src/mojo_sparse.mojo"

test -f "$out" || { echo "build.sh: $out was not produced" >&2; exit 1; }
echo "built $out"