#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/gpu-step-db" >&2
  exit 2
fi

gpu_step_db=$1
if [[ ! -x "$gpu_step_db" ]]; then
  echo "gpu-step-db is not executable: $gpu_step_db" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for destructive GPU qualification" >&2
  exit 2
fi

mode_timeout_seconds=${MEEP_GPU_GRAPH_FAILURE_TIMEOUT_SECONDS:-180}
if [[ ! $mode_timeout_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "MEEP_GPU_GRAPH_FAILURE_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

# Direct invocation intentionally creates a one-rank MPI process.  Do not
# wrap this driver in RUNCODE/mpiexec: each destructive mode owns a fresh CUDA
# context and exits before the next mode begins.
failure_modes=(
  api-return
  post-enqueue
  destroy-sync
  owner-destroy-sync
  async-device
)

for failure_mode in "${failure_modes[@]}"; do
  echo "QUALIFY: structured-curl Graph failure mode ${failure_mode}"
  if timeout --signal=TERM --kill-after=10s "${mode_timeout_seconds}s" \
      env MEEP_GPU_TEST_STRUCTURED_CURL_GRAPH_FAILURE_MODE=$failure_mode \
      "$gpu_step_db"; then
    mode_status=0
  else
    mode_status=$?
  fi
  if [[ $mode_status -eq 124 || $mode_status -eq 137 ]]; then
    echo "FAIL: mode ${failure_mode} exceeded ${mode_timeout_seconds}s "\
         "and was terminated (status ${mode_status})" >&2
    exit "$mode_status"
  fi
  if [[ $mode_status -ne 0 ]]; then
    echo "FAIL: mode ${failure_mode} exited with status ${mode_status}" >&2
    exit "$mode_status"
  fi
  echo "PASS: mode ${failure_mode} exited with status 0"
done

echo "PASS: isolated structured-curl Graph failure matrix"
