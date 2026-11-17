# gpmeep v1.1.0 Development Plan

Status: normative development contract, not release notes. The accepted v1.0.3
release at commit `b48c9b678f2316cefa95e48a661f7a877547a000` remains frozen.

## Product rules

v1.1.0 is the structural-performance line. It must retain the familiar Meep
Python interface, FP32 CPU and CUDA lanes, broad FDTD/DFT/material support,
adjoint and representative inverse-design paths, MPI, and multi-GPU. Scheme is
not part of the gpmeep package or validation scope.

### No automatic backend selection

The user selects `cpu` or `cuda`. A recommendation engine may advise and guard,
but it must never silently replace that choice, change backend during a run, or
fall back to CPU. When the selected backend disagrees with the recommendation,
the first attempt fails before simulation state changes. The same choice may
proceed only with an explicit acknowledgement and reason; both are recorded in
telemetry. This fail-before-run handshake applies to explicit selections and to
legacy programs whose lack of a gpmeep selection is interpreted as CPU.

The recommendation is not a speed guarantee. It reports a confidence and the
evidence behind its decision, including rank-local cells, expected run length,
monitor structure, transfers, launches, and MPI/GPU topology. A stale or
malformed acknowledgement fails closed.

The first mismatch acknowledgement uses an exact one-shot capability, not a
persistent backend preference. Its digest binds the selected and recommended backends, a canonical
workload fingerprint, recommendation-policy version, MPI-topology digest,
the ordered per-rank native-state digest, process nonce, and run nonce. The
UTF-8 reason must occupy 8--512 bytes. Every MPI rank must agree on the complete
digest, and the capability is consumed before the first simulation-state
mutation. It cannot be replayed by another simulation, process, topology,
native state, policy revision, or changed workload. Missing, stale, replayed,
malformed, out-of-range, or rank-disagreeing acknowledgements fail before the
run. Negative tests cover each of those states as well as a changed
recommendation and explicit attempts at silent fallback or runtime backend
switching.

After that exact retry succeeds, the default `scope="simulation"` cache may
reuse the reason only for the same `Simulation` object's identical policy
scope: selected and recommended backend, reason class, confidence, policy
version, MPI topology, and threshold-category facts. A separate, explicit
`scope="process"` acknowledgement permits later `Simulation` objects in the
same process to reuse only that same policy class for an intentionally accepted
batch. Each cache has at most 32 entries, rejects fork-inherited state, and is
invalidated by any policy-scope change. Environment/device failures remain
non-acknowledgeable and are checked before reuse. Qualification can set
`MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE=1` to force the legacy per-run exact
handshake for same-binary overhead isolation. The process scope is never
implicit and neither scope changes the user-selected backend. On MPI, every
reuse still performs an exact collective workload fingerprint to reject
rank-divergent simulation inputs and callbacks.

The final M4 optimization may deduplicate immutable-package graph validation
inside one exact canonical traversal, provided it performs complete graph
checks before and after token construction plus node-local checks around every
trusted token that reads package state. The semantic workload/native
projections, exact within-version collective rank comparison, and all mutation
and rank-divergence negative tests must remain intact. Opaque digest values are
policy/runtime-identity bound and are not required to match an earlier policy
version. Original trusted function definitions are serialized from the
authenticated import-time references, so a temporary live ``__code__`` or
defaults replacement cannot contaminate a token even if it is restored before
the post-check. This is policy-overhead removal, not a weaker acknowledgement
scope.

## C1-C4 research carried forward

- **C1, guarded zero-state Lorentzian/Drude elision:** omit an unnecessary
  state store only where positive-zero state and the next legacy value are
  bitwise safe. Signed zero, non-finite, anisotropic, nonzero, or uncertain
  states retain the legacy path.
- **C2, AuNP canary for C1:** this was a measurement, not a separate
  optimization. It measured a 4.75% geometric-mean FDTD improvement and 1.62%
  end-to-end improvement across GPU1/GPU2, with accepted numerical identity,
  but did not pass the old 10% promotion threshold.
- **C3, resident sparse Lorentzian plan:** retain active dispersive point
  indices on device and avoid dense-domain ADE work. It measured 6.66%
  geometric-mean FDTD and 2.50% end-to-end improvement and was numerically
  accepted, but was not promoted under the former threshold. Its v1.1.0
  fail-closed boundary and current measurement gates are specified in
  [GPMEEP_V1_1_0_C3_SPARSE_LORENTZIAN_PLAN.md](GPMEEP_V1_1_0_C3_SPARSE_LORENTZIAN_PLAN.md).
- **C4-A2, safe static material initialization:** an earlier profile attributed
  roughly 55.7% of AuNP GPU1 end-to-end time to `init_sim/_set_materials`.
  C4-A2 conservatively identifies immutable built-in media and enables safe
  host parallelism only behind geometry-state serialization, epoch checks,
  prism pre-initialization, affinity control, and a bitwise material oracle.
  The detailed v1.1.0 boundary is specified in
  [GPMEEP_V1_1_0_C4_STATIC_MATERIAL_PLAN.md](GPMEEP_V1_1_0_C4_STATIC_MATERIAL_PLAN.md).
  Implementation and performance acceptance are still required.

C1 and C3 are reapplied independently on v1.0.3 and measured as OFF, C1, C3,
and C1+C3 in one binary. C4-A2 is added only after its safety tests pass and is
then included in the same-binary ablation. No old dirty worktree is treated as
release source.

## Mandatory GPU-slower-than-CPU investigation

Every representative case in which gpmeep CUDA is slower than a suitable
upstream Meep CPU run is compared through these versioned lanes:

1. upstream Meep CPU;
2. gpmeep v1.1.0 CPU, reported separately from upstream CPU;
3. the frozen gpmeep v1.0.3 CUDA package on GPU1;
4. the exact gpmeep v1.1.0 CUDA candidate on GPU1;
5. frozen v1.0.3 and exact-candidate v1.1.0 CUDA on GPU2 when the workload and
   topology make multi-GPU meaningful.

The lanes use the same physics, precision, stopping condition, output contract,
and an appropriate physical-core CPU topology. End-to-end time is decomposed
into initialization, FDTD timestep, sources, boundaries, dispersive ADE,
DFT/reductions, transfers/synchronization, MPI, Python/adjoint/MPB, and I/O.
Each inversion receives exactly one evidence-backed primary class. A phase
that cannot be measured comparably in both frozen binaries is retained as
`available=false` with two explicit null sample values, a null mean, and a
reason code; it is never reported as a fabricated zero-second measurement.

The repository-bound equal-work matrix performs one warm-up followed by two
order-reversed measured cycles. The final table retains both raw measured
times and their mean for CPU, frozen v1.0.3 CUDA, and exact-candidate v1.1.0
CUDA. It also reports upstream-CPU/v1.0.3-CUDA and
upstream-CPU/v1.1.0-CUDA speedups plus the direct
v1.0.3-CUDA/v1.1.0-CUDA improvement. Upstream Meep CPU and gpmeep v1.1.0 CPU
remain separate columns. GPU2 columns are mandatory for the four declared
3D/adjoint cases rather than being inferred from GPU1.

The waveguide-crossing optimizer smoke separates two numerical authorities.
Fixed-point objectives, gradients, beta/callback/evaluation counts, NLopt
termination, checkpoint round trips, bounded design vectors, and actual design
progress remain verdict-bearing. Continuous two-evaluation NLopt trajectory
values are diagnostic only between separately built upstream Meep and gpmeep
CPU runtimes because the upstream CPU trajectory is not repeatable at the
manifest tolerance. CUDA trajectory values remain verdict-bearing against the
same-cycle gpmeep v1.1.0 CPU lane. This preserves a strict CPU/CUDA oracle
without treating a non-repeatable upstream optimizer path as ground truth. A
canonical 950-key inventory and digest prevent paired omissions or same-count
metric substitutions. It contains 893 continuous optimizer metrics, 15
structural optimizer metrics, 22 fixed-physics metrics, and 20 exact-work
timestep scalars (12 fixed-work runs and 8 optimizer runs). Only the 893
continuous optimizer metrics are diagnostic in the upstream/gpmeep CPU
comparison, leaving 57 verdict-bearing metrics. The same-cycle gpmeep
CPU/CUDA comparison keeps all 950 metrics verdict-bearing. Epoch-0 and epoch-1
design-step norms are recomputed from the emitted vectors rather than trusted
as reported scalars.

### A. Hardware/workload crossover

Examples include tiny rank-local domains, very short runs, 1D jobs with little
parallel work, host- or I/O-dominated programs, and over-decomposed multi-GPU
runs. This class requires a size/run-length sweep plus profiler or ablation
evidence showing that fixed launch or communication cost is irreducible at the
relevant scale. A CPU or smaller-GPU-count recommendation may be the terminal
solution only for this class.

### B. gpmeep engine or design overhead

Examples include fragmented kernels, avoidable synchronization or transfers,
host round-trips for resident data, dense work for sparse state, poor layout,
unnecessarily narrow batching/Graph eligibility, CPU initialization dominating
GPU end-to-end time, and failed MPI compute/communication overlap. These are
implementation defects or debt, not intrinsic GPU limits.

A warning is not an optimization. A representative high-impact case in this
class must receive an independently switchable change, same-binary A/B
measurement, and numerical regression validation. If it remains unresolved it
blocks v1.1.0 unless the user explicitly removes that workload from the release
scope. CPU recommendation may protect an immediate run, but cannot close the
development item.

### C. Environment or configuration defect

Examples include GPU oversubscription, incorrect rank-device binding, a wrong
CUDA-aware MPI decision, first-run PTX JIT, thermal or power throttling,
competing processes, CPU affinity/NUMA mistakes, and unsupported driver/runtime
combinations. gpmeep either prevents the configuration before running or emits
a precise actionable diagnostic. This class is not used to hide engine debt.

### D. Unknown

An unproved cause remains Unknown. It is excluded from portable performance
claims, carries low recommendation confidence, and blocks release when it is a
representative high-impact inversion. It is never relabeled as a hardware
crossover merely because optimization is difficult.

Initial design-debt candidates include disc extraction, Cartesian/cylindrical/
JAX adjoint representatives, third-harmonic generation, and parallel waveguide
force. Small bounded `phase_in_material` runs are initial crossover candidates,
but none of these labels become final without new profiling.

### Immutable inversion registry

The initial inversion registry inventory is the checksum-bound set of all 13
CUDA-slower observations in the v1.0.1 release campaign: nine general cases and
four MPI cases. Its source report, each workload identity, the representative
flag, impact, and selection rationale are immutable M0 inputs. Every observed
inversion receives a registry record even when it is not representative, so
deduplication cannot silently erase an inconvenient result. New campaigns append
new immutable inventories; they never rewrite this one.

The immutable identity is not a hash of the display path alone. It binds the
historical source commit and exact source blob, canonical entry point and
arguments, FP32 physics/stopping source, controller route and task receipt,
result row, and measured topology. The two original campaign reports are kept
outside the repository as mode-0600 evidence and their exact SHA-256 values are
required by both full contract validation and the M0 attestation.

Each record conforms to
`packaging/v110-performance-inversion-record.schema.json`, names exactly one
primary class, contains all versioned comparison lanes when relevant, requires at
least two accepted timing repetitions, decomposes the mandated phases, and
binds every evidence file by SHA-256. Hardware/workload crossover may close only
with a size/run-length sweep and profiler or ablation evidence. Representative
high-impact engine-overhead or Unknown records remain release-blocking; an
unregistered observation also blocks release. The release gate validates the
immutable inventory, record completeness, evidence hashes, classification/
resolution consistency, and the absence of open blockers.

## Optimization sequence

1. Reapply C1/C3 on the frozen v1.0.3 base and run the same-binary OFF, C1,
   C3, C1+C3, and C4 ablation with two order-reversed repeats on GPU1/GPU2.
   Record MPI-synchronized initialization, FDTD, and end-to-end seconds as
   absolute values and ratios; no arbitrary minimum-improvement threshold is
   used.
2. Implement C4-A2 and prove its per-array bitwise material oracle before its
   ablation lane is eligible.
3. Add workload characterization and the explicit recommendation guard.
4. Investigate adaptive field-phase batching only for small domains; previous
   large global batching regressions prohibit enabling it globally.
5. Coalesce source/DFT work and use CUDA Graph replay only over a stable,
   fail-closed timestep suffix.
6. Profile GPU1-GPU4 halo/compute overlap and rank-local work. Do not repeat
   previously ineffective Waitall or boundary-Graph ideas without new evidence.
7. Reproduce the externally observed gpmeep CPU regression using equal build,
   precision, ranks, affinity, and physics before changing the CPU engine.

## Validation and evidence

- Deduplicate similar examples by feature coverage, but cover broad Python
  FDTD/DFT, dispersive and nonlinear media, PML/Bloch/symmetry, Near2Far,
  force, LDOS, CW, material mutation, MPI, and real multi-GPU.
- Include adjoint and representative inverse-design programs.
- Retain byte identity for both user-supplied inputs under
  `workspace/meep_gpu/test_example`. At the user's later direction, the very
  long `Hor_TM_60deg_saw_v6.py` case (SHA-256
  `4a14fb7a6770802aeefa116768dd28ea3d92900f36fb06e1f01b9de5706f8bef`)
  is explicitly `NOT_EVALUATED` and is neither a release gate nor eligible for
  a performance claim. This exclusion must be visible in the final report.
  The required AuNP input is `aunp_r4000_repro_20260807code.tar.gz`, SHA-256
  `448e01dcf433435fc5e00ef40ef0cb260870b002a113f3340eacee84f205ff96`;
  its checksum-bound entry points are `fdtd/meep/run_production.py` and
  `fdtd/meep/aunp_periodic_fdtd.py` under the archive root. M0 validates both
  outer bytes and the required archive-member bytes before later execution.
- Run the AuNP validation as TM only with minimum run 20 and DFT field-decay
  `5e-8`; keep its correctness/stopping run separate from a fixed-work
  performance anchor.
- Use two accepted repetitions for reported performance means.
- Select CPU topology by physical-core and affinity evidence, not by the
  fastest accidental single-core comparison.
- Retain numerical outputs, wall-clock phase timings, profiler/ablation IDs,
  topology, thermals, software/runtime identity, failures, and exclusion
  reasons.
- Run an independent adversarial audit after every milestone, repair every
  release-blocking finding, and re-audit that milestone.

## M0 source and environment attestation

Development occurs only in the independent `gpmeep-v1.1.0-dev` repository on
`dev/v1.1.0`. Its Git object store must be a real local `.git` directory with
no alternates; the M0 commit's first parent is the frozen v1.0.3 commit. The
origin URL, branch, tag dereference, merge base, clean worktree, and object-store
identity are machine-attested rather than inferred from directory names.

The builder and test prefixes live under that repository and are compared as
exact URL-plus-SHA package sets against their committed lock files. The local
Micromamba executable, lock files, development contract, inversion schema and
inventory, actual user inputs, and selected external M0 evidence are all hashed.
The attestation is created atomically outside the repository with mode 0600;
verification recomputes live state and rejects stale or modified evidence.

The static contract validator is called from the package builder, the installed
package verifier's v1.1 preflight, and a manual-only GitHub Actions workflow.
That workflow has no `push`, `pull_request`, or schedule trigger, so the M0 gate
does not reintroduce unsolicited Actions-failure email. Source-only automation
may skip external user bytes, but the external attestation and final release
gate may not. A canonical contract digest makes every unreviewed normative leaf
fail closed, while recursive mutation tests prove that property.

The package builder has two explicit, mutually exclusive modes. A development
build may retain an open inversion registry and keeps its installable
`*_development.conda` name. The builder adds a read-only `.NON_PUBLISHABLE`
companion marker bound to the package bytes and source commit. A candidate
build requires full external-input
validation, a closed evidence-backed inversion registry, the exact external M0
attestation, and the M4 option-A closure manifest. Its package-owned receipt is
limited to `CANDIDATE_BUILD_AUTHORIZED`: it always sets package publication and
Git-tag authority to false.

Release authority is created only after the candidate bytes exist. The
external `RELEASE_AUTHORIZED` receipt replays the package-owned candidate
receipt and binds the exact package SHA-256 to installed correctness, real
multi-GPU/no-fallback evidence, AuNP, native Ubuntu 22.04 and newer-Ubuntu
installs, the GLIBC floor, two-repeat six-lane performance evidence, embedded-
image HTML and Markdown reports, and the final adversarial audit. Publication
replays that receipt immediately before and after tag creation and push. It
refuses a dirty or wrong source tree, an existing tag/release, changed package
bytes, or any attempt to rebuild/mutate the qualified package. The complete
operator procedure and manifest shapes are in
[`GPMEEP_V1_1_0_RELEASE_TRUST.md`](GPMEEP_V1_1_0_RELEASE_TRUST.md).

## Portability target

The initial binary target is Linux x86-64 with a required GLIBC floor no newer
than 2.35. Installation, CPU smoke, GPU1, and where available GPU2 must run on
Ubuntu 22.04 and at least one newer Ubuntu environment before the documentation
claims that range. CUDA architectures, driver minimums, embedded SASS/PTX,
MPI/HDF5 ABI, and every gpmeep-owned ELF remain provenance-checked.

Wider Unix and accelerator portability is tracked separately in
[`GPMEEP_FUTURE_TODO.md`](GPMEEP_FUTURE_TODO.md); untested platforms are not
marketed as supported.

## Release decision

v1.1.0 requires at least one repeatable structural end-to-end or core-FDTD
benefit, no material regression on established large positive workloads, a
validated recommendation guard, broad correctness, real multi-GPU validation,
portable packaging/provenance, and passing adversarial audits. There is no
arbitrary 10% threshold for every candidate, but small gains do not justify
unsafe or unmaintainable complexity.

No unresolved representative engine-overhead inversion and no unclassified
representative GPU/CPU inversion may remain at release. The machine-readable
normative form of these rules is
[`packaging/v110-development-contract.json`](../../packaging/v110-development-contract.json)
and is checked by `scripts/validate-v110-development-contract.py`.
