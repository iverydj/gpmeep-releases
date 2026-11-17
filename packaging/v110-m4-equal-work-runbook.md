# v1.1.0 M4 equal-work qualification runbook

M4 executes the byte-bound plan in `packaging/v110-m4-equal-work-plan.json`.
The controller refuses any other census or ordering: 60 warm-up samples followed
by 60 forward and 60 reverse-order measured samples. It does not silently select
a backend. The six explicit lanes remain separate:

- upstream Meep FP32 CPU (one or eight ranks, as planned);
- gpmeep v1.1.0 FP32 CPU (one or eight ranks);
- frozen gpmeep v1.0.3 FP32 CUDA on GPU 1 and, where planned, GPUs 1+2;
- exact installed gpmeep v1.1.0 FP32 CPU/CUDA package on the planned topology.

## Before execution

Use a result directory outside the source repository. Both selected GPUs must be
idle. Keep the same Linux boot, controller CPU affinity, GPU UUID ordering,
measured v1.1.0 package, upstream receipt, and frozen-v1.0.3 package/provenance files for
the entire run, including resumes. The controller binds all of these identities
into `CHECKPOINT.json` and fails closed if one changes.

NVIDIA 595 open kernel modules must run with UVM HMM disabled. A September 2026
M4 campaign encountered a CUDA-worker process-exit kernel Oops with the same
signature as NVIDIA open-gpu-kernel-modules issue
[#1082](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1082); the
proposed fix [#1110](https://github.com/NVIDIA/open-gpu-kernel-modules/pull/1110)
was not present in either inspected Ubuntu 595.71.05 or 595.91.07 source. The
controller reads the live driver identity and
`/sys/module/nvidia_uvm/parameters/uvm_disable_hmm` before creating output and
at every host-binding replay. It refuses this known-risk combination until the
administrator installs `options nvidia_uvm uvm_disable_hmm=1`, updates the
initramfs, reboots, and verifies that the sysfs value is `Y`. gpmeep uses
explicit `cudaMalloc`/`cudaMemcpy`, not CUDA managed-memory allocation; the
workaround therefore does not remove a gpmeep execution path, though it can
affect unrelated programs that intentionally depend on HMM.

The frozen v1.0.3 prefix must be the fresh dedicated prefix validated by its
provenance report. The v1.1.0 package may be the nonpublishable development
package used to break the qualification/build circularity. Its dedicated
installed prefix and saved provenance report must belong to those exact package
bytes. Candidate promotion later requires byte-for-byte equality of every
runtime payload member; only the embedded candidate receipt may differ.

## Start

```bash
python scripts/python-validation/run_m4_equal_work_matrix.py \
  --repo /absolute/path/to/gpmeep \
  --output /absolute/path/outside/repo/m4-equal-work \
  --python /absolute/path/to/v110/python \
  --build-python /absolute/path/to/v110/build/python \
  --install-prefix /absolute/path/to/v110/install \
  --build-receipt /absolute/path/to/v110/build/build-provenance.json \
  --mpiexec /absolute/path/to/v110/mpirun \
  --upstream-receipt /absolute/path/to/upstream-cpu-baseline-receipt.json \
  --v103-install-prefix /absolute/path/to/fresh/v103-prefix \
  --v103-provenance-report /absolute/path/to/v103-provenance.json \
  --v103-package /absolute/path/to/frozen-v103-package.tar.bz2 \
  --v110-provenance-report /absolute/path/to/v110-installed-provenance.json \
  --v110-package /absolute/path/to/gpmeep-1.1.0-measured_development.conda \
  --gpu-devices GPU-UUID-ONE,GPU-UUID-TWO
```

The development package remains an installable `.conda` file. Its basename
must end in `_development.conda`, and the builder creates a read-only companion
`<package>.NON_PUBLISHABLE` JSON marker binding the package name, byte size,
SHA-256, and source commit. M4 rejects a development package if that marker is
missing or differs.

`python/examples/phase_in_material.py` retains its ordinary validation contract
of two runs ending at timestep 40. M4 performance samples alone set the sealed
internal `GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK=1` contract, extending each run to
timestep 4000. This makes the CUDA interval long enough for the one-second
external process observer and avoids sub-second timing noise without changing
the physics, precision, run count, or numerical oracle across the compared
runtimes. The sample runners remove this internal variable from every other
case, and the plan, development contract, and final registry validator reject a
different fixed-work value.

`--task-timeout-seconds`, `--stdout-limit-mib`, and
`--telemetry-interval-seconds` are sealed into the invocation contract. Changing
them during resume is rejected. The telemetry default remains one second:
several short M4 cases expose a CUDA compute context for only three to four
seconds, so a five-second interval can miss legitimate activity and fail the
independent GPU-execution proof. Replacing the per-sample query process with a
persistent NVML collector is a separate future optimization and must preserve
the same one-second evidence semantics before adoption.

The controller-level `--python`, `--build-python`, `--install-prefix`,
`--build-receipt`, and `--mpiexec` retain the clean source-build identity used
by the release-source contract. They are not the v1.1.0 workload runtime. For
every v1.1.0 child the controller re-derives installed Python, site-packages,
MPI, `_meep`, and `libmeep` paths from `--v110-provenance-report` and launches
those installed binaries.

## Resume and replay

After an interruption, run the same command with `--resume`. Every completed
prefix task is deeply replayed before the next task. Partial artifacts belonging
to the next task are archived by the existing M3 interruption helper; they are
never mistaken for completed evidence.

Each child reconstructs its narrow launch ambient from the canonical
`samples/<task>` and sibling `runtime/<task>/tmp` paths. The derivation contains
exactly `PATH`, `TMPDIR`, `LC_ALL=C`, and `LANG=C`; it never samples the later
verifier process environment. This makes in-process publication and subsequent
controller replay identical while rejecting missing, relocated, or symlinked
task/runtime layouts.

After success, read-only semantic replay needs only:

```bash
python scripts/python-validation/run_m4_equal_work_matrix.py \
  --output /absolute/path/outside/repo/m4-equal-work \
  --verify-complete
```

Replay is deliberately live-bound: source, receipts, both exact package files,
installed provenance, boot, affinity, GPU UUIDs, and `nvidia-smi` must still
match. The retained measured-package SHA-256 may not be substituted with another
build from the same commit. Final authorization independently compares its full
runtime payload with the candidate package.

## Evidence and interpretation

- `samples/` contains each sealed child result.
- `telemetry/` contains host telemetry for every task and selected-GPU telemetry
  for CUDA tasks. CPU tasks do not generate misleading GPU-use evidence.
- `performance-replay.json` uses schema
  `gpmeep-v110-m4-equal-work-aggregation-v2`. It retains all 180 primary seconds,
  two measured seconds and their average for every lane, per-lane phase seconds,
  and explicit same-cycle GPU1/GPU2 ratios.
- `numerical-replay.json` contains 141 same-case, same-cycle comparisons against
  upstream CPU. Manifest JSON tolerances are reused. The embedded-oracle test
  must both pass its unittest oracle and match the generic physical metrics.
- `inversion-records/` contains 13 directly consumable schema-v2 registry
  records. Each lane points to the exact two measured child reports and each
  phase retains its per-cycle basis and native-timer sink inventory.
- `M4_COMPLETE` exists only after the checkpoint, journal, telemetry, all child
  reports, numerical comparisons, and aggregate have replayed successfully.

All ratios are `reference_seconds / candidate_seconds`; above 1 means the
candidate is faster. Upstream CPU and candidate v1.1 CPU remain distinct. The
controller records observations and numerical pass/fail, but it does not invent
a universal performance threshold or hide workloads where CUDA is slower.

Hardware-crossover closure also requires a workload-specific sweep verifier.
It must establish distinct actual grids and step counts, equal CPU/CUDA work at
each coordinate, and the complete original numerical comparisons. Nominal
size/length flags, run counts, and timing crossover alone cannot establish that
proof. The registered phase-in-material family verifies its source-bound work
contract and complete field artifacts. Other families remain unproved until
their corresponding verifier is implemented; their raw diagnostic samples can
still be collected and replayed. This applies to evidence creation, evidence
replay, and schema-only registry inspection, and does not exclude any of the
13 observed inversions or close a record automatically. The original full
sweep, causal evidence, practical M4 anchor, and numerical requirements remain
in force.
