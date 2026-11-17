# gpmeep

**gpmeep is an open-source, GPU-accelerated finite-difference time-domain
(FDTD) package built on Meep.** It uses NVIDIA CUDA, persistent device-resident
field data, and MPI multi-GPU execution to accelerate Meep FDTD simulations
while preserving the familiar Python API:

```python
import meep as mp
```

**Project keywords:** `GPU-accelerated FDTD`, `GPU FDTD`, `CUDA FDTD`,
`Meep GPU`.

## Project creator and gpmeep contribution

**gpmeep was conceived, created, and is led by
[Dong-Joon Yi](https://iverydj.github.io/) ([GitHub](https://github.com/iverydj)).**
The gpmeep-specific contribution is the architecture and implementation that
extends upstream Meep with a persistent FP32 CUDA execution layer, explicit
CPU/CUDA control, MPI-based multi-GPU execution, device-resident DFT and
scientific reductions, release provenance, and correctness/performance gates.
This work includes the production-driven feature requirements, numerical
acceptance criteria, failure policy, portability design, benchmark program,
and release engineering needed to make the accelerator usable as a Meep
distribution rather than as an isolated demonstration kernel.

gpmeep is a derivative of the
[NanoComp Meep project](https://github.com/NanoComp/meep). Meep's FDTD
formulation, public APIs, and upstream history remain the work of the upstream
authors and contributors. The credit above describes the original gpmeep GPU,
integration, validation, and distribution work; it does not replace upstream
attribution. See [`AUTHORS`](AUTHORS), [`CITATION.cff`](CITATION.cff), and the
[license and provenance notes](doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md).

## Compatibility with upstream Meep

For Python users, gpmeep is a distribution of Meep rather than a new Python
package namespace. Scripts continue to import `meep`, and the usual
`mp.Simulation`, geometry, sources, materials, DFT monitors, run callbacks,
`meep.adjoint`, and `meep.mpb` interfaces keep their upstream spelling. This is
source/API compatibility within the validated v1 feature contract, not a claim
that every upstream build option or execution path is present. Do not change a
script to `import gpmeep`: there is no `gpmeep` Python module.

The v1.1.0 release-candidate line retains v1.0.3 compatibility fixes and adds
the explicit CPU/CUDA backend contract, evidence-bound performance
classification, expanded CUDA paths, and the default-off structured-curl CUDA
Graph optimization. It is not a published release until the exact package
passes the M4, portability, and final receipt gates below.

v1.0.3 retains the accepted v1.0.2 CUDA and numerical implementation while
repairing production-reported Linux/Conda portability, build-path, error
propagation, and custom CUDA-architecture provenance defects. A worktree or
version string alone is not a release: use the annotated tag, package checksum,
installed GLIBC attestation, build-produced SASS/PTX inventory, and provenance
checks to identify an accepted artifact.

The user-visible differences are:

| Area | Upstream Meep script | gpmeep |
|---|---|---|
| Conda distribution | usually installed as `pymeep` | installed as `gpmeep` in a separate prefix |
| Python import | `import meep as mp` | exactly the same |
| Backend | CPU | released v1.0.x: CPU default with opt-in `auto` or NVIDIA CUDA; v1.1.0 target: explicit CPU/CUDA choice with a guarded mismatch acknowledgement |
| Extra runtime API | no CUDA controller | `mp.gpu` discovery, selection, diagnostics, and counters |
| Precision | depends on the upstream build; commonly FP64 | the v1 CPU and CUDA lanes are both FP32 |
| Adjoint / MPB execution | build dependent | upstream imports and spelling; FDTD portions may use the selected backend, while packaged JAX and MPB remain CPU-side and MPB is non-MPI |
| Multi-GPU | MPI CPU decomposition | one MPI rank per selected physical/MIG GPU; rank-local selection from the visible-device pool |
| Scheme | available in some upstream builds | intentionally not packaged |
| Native GDSII helpers | build dependent | native support is disabled; wrapper names may remain but fail when called, so use the included Python `gdstk` |
| Platform in the binary release | upstream supports a wider platform set | v1 package is Linux x86-64, Python 3.11, NVIDIA CUDA 12.4, Open MPI |

A Python program within the validated v1 feature contract normally runs
unchanged on gpmeep's default CPU backend. Scripts requiring Scheme, native
GDSII, FP64, or an unvalidated feature combination require migration or
separate qualification.

```python
import meep as mp

sim = mp.Simulation(...)  # existing Meep code
sim.run(...)
```

### Released v1.0.x backend behavior

The following API describes the released v1.0.x packages, including v1.0.3.
To request a backend, add one gpmeep-specific line before constructing or
running simulations:

```python
import meep as mp

mp.gpu.set_backend("cuda")  # "cpu", "auto", or "cuda"
print(mp.gpu.runtime_available, mp.gpu.backend_diagnostic)

sim = mp.Simulation(...)
sim.run(...)
```

The equivalent process-level selection must be set before Python starts:

```sh
MEEP_GPU_BACKEND=cuda CUDA_VISIBLE_DEVICES=0 python simulation.py
```

`cuda` is an explicit, all-or-nothing request for gpmeep's FDTD device backend:
configuration or rank/device-selection failure raises an error instead of
silently selecting the CPU backend. This does not imply that Python control
flow, MPB, JAX/NLopt, or other host-side work executes on the GPU. `auto` is
opt-in and may select CPU for a small or unsuitable workload; it is not a
promise that GPU is faster for every simulation. Inspect
`mp.gpu.active_backend`, `mp.gpu.backend_diagnostic`, and
`mp.gpu.statistics()` when validating which path actually ran.

This is the v1.0.x-only automatic policy. It remains documented so existing
released packages are not misrepresented, but `auto` is not part of the v1.1.0
target contract.

### v1.1.0 development: explicit CPU/CUDA choice

The development target permits only user-selected `cpu` or `cuda`; it never
chooses, switches, or silently falls back between them. Omitting a selection is
interpreted as the legacy CPU choice, but still passes through the recommendation
guard. If the chosen backend disagrees with the recommendation, the first attempt
fails before simulation state changes. The user may then repeat that exact run
with an explicit, reason-bearing acknowledgement bound to its workload and
MPI-topology fingerprints. After that exact retry is accepted, later runs of
that same `Simulation` object reuse the acknowledgement only while the selected/recommended
direction, reason and confidence, policy version, MPI topology, and policy
threshold facts remain identical. This avoids making repeated short runs slower
just to repeat the same warning. By default, a different simulation receives
its own first warning. An explicitly reviewed homogeneous batch can instead
opt into the separately bounded process scope described below; gpmeep still
does not choose or change its backend. The `dev/v1.1.0` branch implements this API;
it becomes a released contract only after the v1.1.0 release gates pass.

`auto` and `automatic` are deliberately rejected in v1.1.0. Select the backend
yourself, then handle a recommendation mismatch explicitly:

```python
import meep as mp
from meep import backend_policy

mp.gpu.set_backend("cuda")  # or "cpu"; gpmeep never changes this choice
sim = mp.Simulation(...)

try:
    sim.run(until_after_sources=20)
except backend_policy.BackendRecommendationMismatch as mismatch:
    print(mismatch.telemetry)
    mp.gpu.acknowledge_backend_mismatch(
        mismatch.challenge,
        reason="production comparison intentionally uses CUDA",
    )
    sim.run(until_after_sources=20)
```

The default acknowledgement scope is `"simulation"`. For a batch that
deliberately creates many equivalent `Simulation` objects, the user may accept
one exact challenge with `scope="process"`:

```python
mp.gpu.acknowledge_backend_mismatch(
    mismatch.challenge,
    reason="reviewed CUDA benchmark batch on this fixed MPI topology",
    scope="process",
)
```

This is an explicit reduction in warning frequency, not automatic backend
selection and not permission to ignore a broken CUDA environment. Reuse is
limited to the same process and the same selected/recommended direction,
reason/confidence class, policy thresholds and version, and MPI/GPU topology.
Changing any of those facts warns again; a forked child inherits no acceptance.
Omit `scope` when each newly constructed simulation should require its own
warning. Every MPI run, including either kind of reuse, still computes the
exact collective workload fingerprint so rank-divergent geometry, sources, or
callbacks fail before solver mutation. Only a one-rank reuse path may skip that
deep rank-consistency check.

The first mismatched call performs no `init_sim`, DFT creation, or timestep.
The acknowledgement permits only the immediately repeated exact workload and
is consumed before that run mutates state. Changed source/geometry/DFT, file
input, initialization or output hook, stopping condition or step callback,
backend, device mapping, MPI topology, process, or policy revision invalidates
it. Initialized simulations are also supported: the challenge binds ordered
per-rank native `fields` and material-`structure` state manifests, including
monotonic mutation epochs, current timestep, field mode, boundary/Bloch state,
and core solver scalars. Native `step`, field/structure load or reset,
initialization, material/source/component change, DFT add/update/load/scale or
removal, boundary change, and synchronization advance an epoch; replacing a
local native owner also invalidates the pending retry. The reason must occupy
8--512 UTF-8 bytes,
and every MPI rank must submit the same challenge and reason.
`mp.gpu.pending_recommendation` exposes the current pre-run telemetry, while
`mp.gpu.recommendation` exposes the last decision.

The same guard runs before the high-level `Simulation.run`, `run_k_point`,
`solve_cw`, and `solve_eigfreq` entry points mutate solver state. A broken CUDA
environment or rank/device mapping raises `BackendEnvironmentError`; it is a
configuration error and cannot be acknowledged. Direct low-level `mp.fields`
calls do not carry a `Simulation` workload description, so the C++ engine
enforces the explicit all-or-nothing backend but cannot make the high-level
CPU/CUDA suitability recommendation. Low-level callers must make and document
that performance choice themselves.

Normal Meep Python code should mutate solver state through the documented
methods (`Simulation`, `add_flux`/other monitor constructors, `scale_dfts`,
`load_*`, monitor `remove`, material/source setters, and field methods), just as
it does upstream. In v1.1.0, engine-owned raw SWIG descriptor members remain
readable for inspection but are no longer assignable from Python. This includes
the linked monitor/DFT descriptors and the exposed storage/configuration members
of native `fields` and `structure` owners: direct assignment could bypass the
mutation epoch, corrupt owner lists, or desynchronize MPI and CUDA mirrors.
This includes `structure.effort`: it is an engine-owned allocation, not a
replaceable Python data buffer, and assigning it could create conflicting
owners during teardown. It remains readable for inspection.
`fields.t` is the deliberate exception because Meep's existing high-level
restart path resets it; its exact value is bound by the native state manifest.
The raw `flux_vol(fields, ...)` SWIG constructor is not exported because it can
attach a monitor to an initialized native owner without going through that
owner's mutation epoch. Use `Simulation.add_flux` (normally) or
`fields.add_flux_vol` (for supported low-level code), both of which preserve
ownership and invalidate any pending acknowledgement.
The owning native `src_time.next` list link is also readable but not assignable;
source-list construction remains an engine operation. Native cycle detection
fails the manifest instead of issuing a retry challenge.
The internal `fields.process_incoming_chunk_data` MPI halo-transport routine is
also no longer exported to Python. These restrictions do not change supported
high-level Meep usage; custom code that assigned raw native members must migrate
to the corresponding public method before using gpmeep v1.1.0.

For Python stopping conditions and step callbacks, the workload fingerprint
binds bytecode, defaults, closures, transitively referenced functions and
mutable globals, statically accessed module attributes, and iterator/generator
position. It also binds function attributes and annotations, transitive
descriptor functions, bound built-in owners, instance `__dict__` and all MRO
`__slots__`, mutable class/base-class state, built-in subclass state, and
deterministic object-reference numbers for aliases, shared subgraphs, and
cycles. Simulation configuration is taken from its semantic instance state;
native derived caches alone are excluded. Epsilon, structure, fields, and
chunk-layout input files are content-hashed when present. Raw Python addresses
and SWIG pointers are never used. Adjoint-generated `sourcedata` combines a
native SHA-256 manifest of its component, chunk, index, and amplitude contents
with canonicalized Python proxy attributes. A module reached through a
container/object has no statically provable attribute path and therefore fails
closed; direct statically accessed module attributes remain supported. Opaque
extension state or any other object graph that cannot be represented without a
type/name-only fallback is rejected before initialization instead of issuing a
reusable challenge. Runtime-only exclusions are restricted to exact Meep-owned
objects; a user object's similarly named field, including a `_gpmeep_` prefix,
remains part of the fingerprint. Meep-owned Python classes are trusted by the
package class and original-member identities captured at import, not by a
spoofable `__module__` string; an added or replaced method is therefore
canonicalized transitively like user code.

For a direct Python `CustomSource`, the initialized native source manifest
records that external state is present and retrieves the retained callable
through a private trusted extension helper. The same transitive canonicalizer
therefore binds its bytecode, closures, globals, descriptors, aliases, and
mutable values. Changing callback state after a mismatch changes the challenge;
an acknowledgement for the earlier callback state cannot authorize the run.

The recommendation uses rank-local grid size, estimated timesteps, source and
monitor structure, estimated launches/transfers, available device memory, and
the actual MPI/GPU mapping. It is advice, not an automatic scheduler or a speed
guarantee. Hardware/workload crossover may be resolved by the warning; an
environment defect fails closed, while confirmed gpmeep engine overhead remains
an optimization/release-gate issue and is not excused by recommending CPU.

For two GPUs, gpmeep uses MPI rank decomposition rather than making one Python
process transparently drive both devices:

```sh
CUDA_VISIBLE_DEVICES=0,1 \
MEEP_GPU_BACKEND=cuda \
MEEP_GPU_ALLOW_OVERSUBSCRIBE=0 \
MEEP_GPU_MPI_TRANSPORT=pinned \
MEEP_GPU_MPI_COMPLETION=waitsome \
mpiexec -n 2 python simulation.py
```

This example exposes the same two-device pool to both ranks. On a two-rank
single-node run, gpmeep selects compatible ordinals by node-local rank and
validates that the ranks selected distinct physical/MIG device identifiers. A
scheduler may instead expose one distinct GPU to each rank, in which case each
rank sees its own rank-local ordinal. Oversubscription is an explicit exception,
not the recommended performance configuration. More GPUs do not guarantee a
faster run: rank-local work must be large enough to repay MPI and CUDA
synchronization costs.

The following implementation section describes the released v1.0.x line;
v1.1.0 removes its `auto` choice according to the explicit-choice contract above.

## How GPU acceleration works

gpmeep keeps Meep's Python control plane, object model, Yee update order, and
MPI domain decomposition. The acceleration layer sits below the existing
`fields::step()` pipeline: a fields owner is admitted to CUDA before a time
step starts, each owned `fields_chunk` acquires a persistent device cache, and
the repeated numerical phases operate on those device-resident arrays. This is
different from copying the complete field state to and from a GPU around every
kernel.

### Design rationale

The central performance observation is that an FDTD run applies the same local
stencil and material updates to large field arrays for many time steps. A GPU
is useful only if those arrays remain on the device long enough to amortize
allocation, kernel-launch, PCIe, and MPI costs. gpmeep therefore accelerates
the repeated data path rather than merely wrapping each upstream operation in
an independent host-to-device/device-to-host copy:

1. **Preserve the numerical contract.** The existing Yee leapfrog order,
   material equations, boundary phases, and MPI domain decomposition define
   the result. CUDA kernels implement those phases in the same dependency
   order instead of introducing a different solver.
2. **Keep evolving state resident.** Each `fields_chunk` has a persistent,
   generation-tracked device mirror. Ordinary time steps reuse allocations
   and leave the newest field state on the GPU.
3. **Move consumers to the data.** Sources, dispersive updates, DFT monitors,
   Near2Far/LDOS/eigenmode work, and compact scientific reductions operate on
   resident arrays where qualified. Only requested output or genuinely
   host-side work crosses the coherence boundary.
4. **Scale through Meep's decomposition.** One MPI rank owns one GPU or MIG
   instance. Rank-local subdomains advance on their devices while halo data
   uses qualified CUDA-aware MPI or retained pinned-host staging.
5. **Fail closed at boundaries.** Collective preflight rejects unsupported
   physics, inconsistent rank/device mappings, or insufficient memory before
   solver mutation. Generation and transaction checks prevent stale or
   partially published host/device state.
6. **Treat speed as a measured crossover.** Small, short, callback-heavy, or
   over-decomposed jobs may not amortize GPU overhead. v1.1.0 keeps the backend
   choice with the user and issues an evidence-based, explicitly
   acknowledgeable warning when that choice disagrees with the recommendation;
   it never silently changes the backend.

The resulting speedup comes primarily from eliminating repeated full-state
transfers and executing the dominant update/monitor pipeline at GPU memory
bandwidth. Kernel batching, cached descriptors, asynchronous staging, overlap,
and optional replay then reduce launch and communication overhead without
changing the physical model.

```text
Python mp.Simulation / the existing Meep API
                     |
     collective backend and feature preflight
              /                      \
        CPU owner                 CUDA owner
                                      |
             persistent mirror cache per fields_chunk
                                      |
       B curl -> H/material -> boundary/MPI halo
       D curl -> E/material -> boundary/MPI halo
                                      |
          resident DFT updates and scientific reductions
                                      |
       selective host publication only when it is required

multi-GPU: MPI rank 0 -> GPU 0, rank 1 -> GPU 1, ...
halo data: validated CUDA-aware MPI or pinned-host staging
```

### Backend selection is owner-wide and state-atomic

CPU is the default. In v1.1.0, `mp.gpu.set_backend("cpu")` and
`mp.gpu.set_backend("cuda")` are explicit user choices; `auto` is rejected.
A `cuda` choice is also an all-or-nothing request: runtime, device assignment,
memory-capacity, or supported-time-step preflight failure raises before the
step modifies any field. All MPI ranks first agree on the requested mode,
guard decision, feature set, transport, and device assignment. A validated
time step is therefore entirely CPU or entirely CUDA for that fields owner;
gpmeep does not silently execute an unsupported chunk on the CPU after
partially advancing other chunks on CUDA.

Before the high-level solver initializes, the recommendation guard estimates
rank-local grid work, known run length, source/monitor work, launch and transfer
cost, device memory, and MPI/GPU topology. It advises CPU or CUDA but never
changes the explicit choice. The first mismatch requires the exact,
reason-bearing acknowledgement described above. The exact challenge is
one-shot and cannot be replayed. The default bounded scope belongs only to the
same `Simulation` object. A separate `scope="process"` opt-in can cover later
simulation objects only when their complete policy class and MPI/GPU topology
match the explicitly accepted class. The current reference floor is
524,288 local cells for one serial FDTD worker per MPI rank, but this is a
versioned calibration input rather than an application API or a speed
guarantee. Unknown stopping duration, Python/optimizer work, output cost, and
machine-specific contention can still require an end-to-end measurement.

Each reusable scope cache holds at most 32 entries and is never inherited as
an authorization by a forked child. Under the default scope, a different
`Simulation` requires a new exact warning. Under explicit process scope,
backend direction, policy classification, MPI topology, policy version, or
threshold-category change still requires a new exact warning and
acknowledgement. CUDA environment/device failures remain
non-acknowledgeable. For diagnostic same-binary
A/B measurements only, set
`MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE=1` to restore an exact warning for
every mismatched run; this is not recommended for normal production use.

The first policy attempt for a live `Simulation` obtains the complete CUDA
device inventory from one coherent native snapshot. Repeated `run` calls on
that same object reuse a mutation-isolated positive snapshot; a different
`Simulation`, a fork, relevant CUDA/MPI environment or topology change, or an
explicit backend/device transition forces fresh discovery. Public
`mp.gpu.devices()` remains uncached. Supported controller transitions are
serialized with the guard, and a native generation/state check is repeated on
cache hits. A device failure during an already-running simulation is still
fail-closed by the native CUDA preflight/step path; the scoped cache never
causes CPU fallback or automatic backend selection. One inventory call also
does not repeat device enumeration separately for every reported property.
The qualification-only
`MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT=1` control restores the legacy
per-property query path solely for same-binary A/B evidence; it is not a
recommended production setting. Qualification excludes its inventory-attestation
probe from the workload-only primary time, requires identical OFF/ON inventory
digests and mismatch-acknowledgement counts, and still records total subprocess
time as a diagnostic.

### Persistent field residency and coherence

Meep's host arrays remain the API-visible storage identity. gpmeep associates
each local fields chunk with a CUDA cache keyed by those host pointers. A cache
records the device allocation, byte size, upload epoch, allocation/content
generations, and whether the device mirror is the authoritative dirty copy. It
also retains reusable allocations and per-chunk descriptors for curl,
constitutive updates, sources, DFT monitors, and LDOS. The resident backend
keeps dependent boundary, eigenmode-overlap, and Near2Far plans in separate
owner-level registries and invalidates them when their cache dependencies
change.

The first device use allocates and uploads a mirror. Later phases and time
steps reuse it. CUDA writes mark the device copy authoritative, and the outer
step session ends with `finish(false)` when the persistent path is eligible,
so a normal step does not publish all fields back to the host. A device-side
finite scan preserves Meep's per-step NaN/Inf failure check while returning
only its compact verdict.

A selective point read can copy one requested scalar without publishing its
complete host mirror. Generic host operations or callbacks that directly read
or mutate host field arrays first synchronize the affected owner; selected
output/checkpoint work, material or structure changes, backend/device changes,
and object destruction establish related coherence boundaries. When a complete
synchronization is required, every dirty D2H copy is first placed in staging;
host fields are updated only after all copies succeed. This transactional rule
prevents a failed transfer from leaving a mixture of old and new host arrays.
Generation checks invalidate cached pointer plans after allocation, topology,
material, or device changes.

### CUDA FDTD pipeline and supported physics

The CUDA path preserves Meep's leapfrog/Yee sequence rather than replacing the
solver algorithm. One step performs B curl, B sources, and B boundaries; H
constitutive and polarization updates plus their boundary phases; D curl, D
sources, and D boundaries; E constitutive and polarization updates plus their
boundary phases; and then DFT accumulation. The main device work is:

| Phase | CUDA implementation |
|---|---|
| B/D curl | FP32 Cartesian 1D/2D/3D and cylindrical stencil kernels, including supported PML curl auxiliaries, electric/magnetic conductivity, 2D beta, and BFAST forms |
| H/E constitutive update | diagonal/off-diagonal inverse susceptibility, PML H/E auxiliaries, integrated-source terms, and supported `chi2`/`chi3` nonlinear terms |
| Dispersive polarization | exact standard Lorentzian/Drude, eligible gyrotropic, and eligible multilevel/gain state updates and polarization subtraction |
| Sources | scalar time amplitudes are prepared by host control code; supported indexed spatial profiles are applied to resident fields on CUDA |
| Boundaries | resident metal zeroing, local copy/negation/Bloch phase, retained gather/scatter plans, and eligible CUDA Graph replay |
| Validation | device finite checks retain the NaN/Inf contract without a full field readback |

Cylindrical coordinates and BFAST each have supported CUDA paths, but their
combination is not a supported curl configuration.

Tile coalescing, separate topology-aware H/E and curl phase batching/replay,
cached boundary descriptors, and eligible
communication/compute overlap reduce launch and scheduling overhead. Curl
automatic mode requires at least four operations and admits the fused kernel
only when the entire phase fits within 24 logical blocks per GPU SM and no
single operation exceeds 12 blocks per SM. In the sealed RTX 3090 Ti one/two-
GPU matrix this bound rejects the 128³/256³ stress grids where the historical
forced path regressed while retaining launch-bound production phases;
`MEEP_GPU_ENABLE_PHASE_BATCHED_CURL=1` still bypasses the bound as an expert
diagnostic opt-in. Unset H/E controls retain their distinct measured adaptive
policy, which uses a per-operation block budget. These are conditional fast
paths, not extra physics restrictions: an ordinary supported CUDA kernel still
runs when a particular batching, overlap, or replay optimization is ineligible.
Presence of `MEEP_GPU_DISABLE_PHASE_BATCHED_CURL` disables curl collection
entirely. Indexed-source batching and multi-monitor DFT fusion remain
experimental/forced-only; neither is presented as an unconditional speedup.
An arbitrary custom susceptibility subclass is not implicitly CUDA-capable:
required CUDA rejects it during preflight, and the user must explicitly choose
CPU for that workload. gpmeep never switches the requested backend silently.

M4 also provides a distinct, default-off structured-curl CUDA Graph ablation:

```sh
MEEP_GPU_ENABLE_STRUCTURED_CURL_GRAPH=1 python your_simulation.py
```

It does not use the regressive fused curl-phase kernel. Instead, it constructs
one graph node for each existing structured curl kernel and chains those nodes
in the original submission order. The arithmetic kernel, material/PML inputs,
and Yee update order are unchanged; only repeated host launch submission is
collapsed. The executable graph is reused only after its device ordinal,
complete phase topology, every captured device pointer, resident-cache
allocation generation, and input/output mirror identities still match.
Generation, topology, pointer, or host-mirror changes reject replay and rebuild
through the ordinary collected path. Singleton phases and phases interrupted by
dependent beta, BFAST, or cylindrical corrections run their ordinary structured
kernels instead. Topologies known to require those direct corrections are
rejected before fingerprinting or operation collection, avoiding per-step Graph
bookkeeping on a path that cannot produce a complete executable graph. Graph
creation failure likewise falls back to those kernels;
graph launch failure is fail-closed and propagates because completion may be
uncertain. Graph destruction reselects the executable's creation device and
performs a device-wide completion fence before releasing the CUDA executable.
If selection, fencing, or destruction fails, gpmeep does not free the graph,
its plan metadata, or any captured resident allocation: the complete lifetime
set is quarantined until process exit, remains unusable, and stays included in
the live-plan gauge.

The opt-in must be the exact value `1`. Presence of
`MEEP_GPU_DISABLE_STRUCTURED_CURL_GRAPH` wins regardless of its value. If
`MEEP_GPU_ENABLE_PHASE_BATCHED_CURL=1` explicitly forces the historical fused
experiment, that control takes precedence and the graph path stays inactive.
`MEEP_GPU_DISABLE_CURL_PHASE_REPLAY` disables stable replay while retaining
collection-time graph submissions. `mp.gpu.statistics()["structured_curl_graph"]`
reports graph submissions, represented logical kernel operations, lifecycle,
fallback, failure, and generation/mirror rejection counters. In particular,
`structured_curl_graph_plan_destroy_failures` is resettable causal evidence,
while `structured_curl_graph_quarantined_plans` is a non-resetting process-long
gauge; neither a failed destroy nor a quarantined executable is reported as a
successful destroy. This path remains an ablation until sealed numerical and
timing evidence establishes a favorable crossover; enabling it is not yet a
general speed guarantee.

### DFT monitors and scientific reductions

DFT samples are interleaved-complex FP32 arrays with retained device
descriptors. Monitor updates accumulate on the GPU, and consumers avoid
publishing the full point-by-frequency data when a smaller result is enough.
CUDA implementations cover resident norm and scale operations, flux and
complex-flux spectra, energy, force, LDOS field/source contraction, eigenmode
overlap, DFT checkpoint staging, requested-array materialization, and supported
forward and adjoint Near2Far contractions. For scalar/spectral reductions such
as flux, energy, force, and LDOS, the kernels return rank-local compact results
that are combined by the required MPI collective.
Requested-array materialization and checkpoint/output instead use bounded
staging for the requested payload and are not compact scalar reductions.

There are deliberate host boundaries. HDF5 file I/O is host work, though its
CUDA path stages only the required DFT ranges. MPB samples eigenmode profiles
on the host and uploads them for the device overlap reduction. Geometry-derived
Dielectric/Permeability arrays are synthetic host queries rather than DFT
fallback. An unsupported postprocessing shape may use its independently
eligible CPU path after publishing the needed resident data; required CUDA's
fail-closed statement applies to the validated FDTD time step, not to every
output API.

### MPI and multi-GPU execution

Multi-GPU execution extends Meep's existing spatial MPI decomposition. Each
MPI rank advances its rank-local chunks on one selected physical GPU or MIG
instance. Absent an explicit `MEEP_GPU_DEVICE` or
`mp.gpu.select_device(...)` selection, node-local rank chooses among ordinals
exposed by `CUDA_VISIBLE_DEVICES`; the resulting stable physical/MIG identities
are then validated collectively across ranks and independent subcommunicators.
Distinct devices are required unless oversubscription is explicitly enabled;
one Python rank does not transparently divide one domain over every GPU.

`MEEP_GPU_MPI_TRANSPORT=auto` negotiates a safe common transport on all ranks.
Direct device-buffer `MPI_Isend`/`MPI_Irecv` is used only when every rank and
the Open MPI runtime report CUDA-aware support. Otherwise gpmeep uses retained
page-locked host buffers and asynchronous D2H/H2D staging. The default
completion policy is `waitsome`; `waitall` is available for qualified
CUDA-aware runs. Eligible schedules can overlap halo traffic with an H/E
update or an interior D curl and execute the shell after the receive, while
the general retained boundary path remains available when overlap is unsafe.

### FP32, reductions, and numerical accuracy

The v1 package builds both its CPU and CUDA lanes with Meep single precision;
the comparison is not FP64 CPU versus FP32 GPU. Fields, material coefficients,
time-step kernels, and complex DFT storage are FP32, and CUDA fast math is
disabled. Selected numerically sensitive consumers use deterministic FP64
accumulation or finalization, including DFT pair reductions and LDOS, while
Near2Far can retry a cancellation-sensitive contraction with a higher-precision
CUDA path. This improves reduction stability without turning the stored FDTD
state into FP64. CPU/CUDA acceptance is based on scientific tolerances, not
bitwise identity or equivalence to a separate upstream FP64 build.

### Adjoint, CW, and host-side work

Forward and adjoint FDTD simulations enter the same CUDA time-step and monitor
paths. Supported eigenmode and Near2Far adjoint contractions also have device
implementations. The complete inverse-design program is not GPU-resident:
Python orchestration, JAX/Autograd/NLopt, geometry/design mutation, and MPB
profile generation remain host-side, and a design mutation establishes a
coherence boundary. The continuous-wave solver has a qualified resident FP32
CUDA BiCGSTAB-L path, but legacy flux-monitor restrictions and FP32 convergence
diagnostics still apply; its host solver remains available.

Likewise, selecting CUDA does not move Python callbacks, geometry construction,
file I/O, or every possible third-party/custom extension onto the GPU. Small,
short, monitor-heavy, host-heavy, or excessively decomposed jobs can remain
slower than CPU. Use `mp.gpu.active_backend`, `mp.gpu.backend_diagnostic`, the
selected device identifier, and `mp.gpu.statistics()` to prove which kernels,
transfers, reductions, and MPI transport actually ran, then measure complete
end-to-end time for the real workload.

### Implementation source map

| Layer | Primary source |
|---|---|
| Python controller and telemetry | [`python/meep.i`](python/meep.i), [`src/meep/gpu.hpp`](src/meep/gpu.hpp) |
| backend selection, device mapping, resident cache, and coherence | [`src/gpu_backend.cpp`](src/gpu_backend.cpp), [`src/gpu_backend_internal.hpp`](src/gpu_backend_internal.hpp) |
| time-step order and boundary/MPI scheduling | [`src/step.cpp`](src/step.cpp) |
| B/D curl and topology-aware phase-batch policy | [`src/step_db.cpp`](src/step_db.cpp) |
| H/E and polarization updates | [`src/update_eh.cpp`](src/update_eh.cpp), [`src/update_pols.cpp`](src/update_pols.cpp), [`src/susceptibility.cpp`](src/susceptibility.cpp), [`src/multilevel-atom.cpp`](src/multilevel-atom.cpp) |
| DFT, LDOS, eigenmode, and Near2Far consumers | [`src/dft.cpp`](src/dft.cpp), [`src/dft_ldos.cpp`](src/dft_ldos.cpp), [`src/near2far.cpp`](src/near2far.cpp) |
| MPI transport negotiation | [`src/mympi.cpp`](src/mympi.cpp) |
| CUDA API and kernels | [`cuda/include/meep_cuda/runtime.hpp`](cuda/include/meep_cuda/runtime.hpp), [`cuda/src/runtime.cu`](cuda/src/runtime.cu) |

The exact validated coverage and known limits remain normative in the
[feature matrix](doc/docs/GPMEEP_FEATURE_MATRIX.md); crossover behavior and
measurements are in the [CPU/automatic/CUDA guide](doc/docs/GPMEEP_GPU_CROSSOVER.md).

## Numerical and environment migration notes

- Check `mp.is_single_precision()` and use tolerances appropriate for FP32.
  Results should satisfy the workload's scientific tolerance, but they need
  not be bit-for-bit identical to a separate FP64 Meep build.
- Do not install official `pymeep` and `gpmeep` into the same environment.
  Both provide the `meep` module and native `libmeep` ABI. Keep separate Conda
  or Micromamba prefixes and compare outputs across processes/files.
- The Conda package version (`gpmeep 1.1.0`, etc.) is the distribution version.
  `mp.__version__` remains the embedded upstream Meep source version, so use
  the Conda record and gpmeep provenance tools when identifying a release.
- `meep.mpb` and `meep.adjoint` retain their Python imports. In the v1 package,
  the MPB solver and packaged JAX are CPU-side (and MPB is non-MPI); only the
  supported FDTD portions follow the selected gpmeep backend.
- `mp.GDSII_vol`, `mp.GDSII_prisms`, and `mp.GDSII_layers` names may exist as
  disabled native-library stubs when `mp.with_libGDSII()` is false. Calling
  them is not supported; parse GDSII with the included `gdstk` instead.
- v1.0.3 retains the sm86 runtime qualification. The default AUTO package
  contains CUDA 12.4's native sm60+ targets plus newest-architecture PTX.
  During packaging, CUDA 12.4 `cuobjdump` enumerates the actual native cubins
  and embedded PTX in `libmeep`; the resulting hash- and binary-bound inventory
  must equal the requested AUTO or LIST policy. Custom LIST builds therefore
  require at least one native `-real` target, record and verify only what they
  compile, and claim sm86 runtime
  qualification only when native sm86 code is present. Run the installed
  self-check on each new machine before production use.
- Most validated Python functionality is CUDA accelerated, including broad
  FDTD, DFT, MPI, and adjoint families, but the 42-feature release contract is
  not a proof of every upstream callback or parameter combination. See the
  feature matrix for exact claims and known limits.

The v1.0.3 line was opened for confirmed binary compatibility and packaging
defects observed in a production deployment. It does not introduce a
speculative kernel or MPI transport rewrite. Structural GPU changes remain
separate explicitly approved v1.1.0 candidates.

The active v1.1.0 development branch is governed by a fail-closed
[development plan](doc/docs/GPMEEP_V1_1_0_DEVELOPMENT_PLAN.md). In that line,
backend choice remains the user's: the recommendation system may stop a
mismatched first attempt and require acknowledgement, but it may not switch
CPU/CUDA automatically. GPU-slower-than-CPU cases must be classified with
profiling evidence; gpmeep engine/design overhead is an optimization and
release-gate issue, not a condition that can be closed by a warning.
Candidate construction and publication are separate authorities: the receipt
inside a candidate package can authorize only the build, while an external
post-package receipt binds the exact qualified package SHA-256 before any tag
or upload. See the [v1.1.0 release-trust procedure](doc/docs/GPMEEP_V1_1_0_RELEASE_TRUST.md).

The full development history and authoritative source release remain in the
private `iverydj/gpmeep` repository. Validated public releases are published
through
[`iverydj/gpmeep-releases`](https://github.com/iverydj/gpmeep-releases).
Each release tag there points to a source snapshot authorized from the private
development tree and accompanies checksum-bound immutable release assets. The
public repository does not run the private development workflow or qualify a
candidate; it publishes already accepted source, packages, documentation, and
release evidence.

Public publication additionally follows the
[privacy-clean release policy](doc/docs/GPMEEP_PUBLIC_RELEASE_POLICY.md).
Legacy identical-tree mirror controllers must not be used for this public
target: raw evidence remains private, and all public refs/assets must pass a
privacy gate independently of numerical/package qualification.

The v1 release target is a checksum-locked Linux x86-64 Conda package and a
one-command installer into a dedicated environment. Runtime release validation
currently covers RTX 3090 Ti (sm86); the CUDA 12.4 portable build policy emits
native sm60+ targets exposed by nvcc and newest-architecture PTX. Scheme and
co-installation with official `pymeep` are intentionally unsupported.

- [Installation and runtime checks](doc/docs/GPMEEP_INSTALLATION.md)
- [Validated feature matrix and limits](doc/docs/GPMEEP_FEATURE_MATRIX.md)
- [Correctness and performance summary](doc/docs/GPMEEP_PERFORMANCE.md)
- [Choosing CPU, automatic, or CUDA execution](doc/docs/GPMEEP_GPU_CROSSOVER.md)
- [v1.0.3 field-feedback compatibility release notes](doc/docs/GPMEEP_V1_0_3_RELEASE_NOTES.md)
- [v1.1.0 structural-performance release notes](doc/docs/GPMEEP_V1_1_0_RELEASE_NOTES.md)
- [v1.0.2 crossover-guidance release notes](doc/docs/GPMEEP_V1_0_2_RELEASE_NOTES.md)
- [v1.0.1 provenance correction](doc/docs/GPMEEP_V1_0_1_RELEASE_NOTES.md)
- [License, attribution, and provenance](doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md)
- [v1 release gates](doc/docs/GPMEEP_V1_RELEASE_PLAN_KO.md)
- [v1.1.0 development contract](doc/docs/GPMEEP_V1_1_0_DEVELOPMENT_PLAN.md)
- [v1.1.0 candidate and release trust procedure](doc/docs/GPMEEP_V1_1_0_RELEASE_TRUST.md)
- [future portability TODO](doc/docs/GPMEEP_FUTURE_TODO.md)

After downloading the v1 Conda package and its `.sha256` sidecar into a
`downloads/` directory that is separate from the builder's `dist/` output:

```sh
/bin/bash -p scripts/install-gpmeep.sh \
  --prefix /absolute/path/to/gpmeep-v1 \
  --package downloads/gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0.conda \
  --mode gpu1 --gpu-devices 0
```

The installer verifies the package checksum, exact dependency lock,
package/source provenance, installed CPU/CUDA numerical equivalence, strict
CUDA work, and zero CPU fallback. `--mode gpu2 --gpu-devices 0,1` adds the
two-rank/two-device transport check.

## License

gpmeep is distributed under the GNU General Public License, version 2 or
later, the license of Meep, from which it is derived ([`LICENSE`](LICENSE),
[`COPYRIGHT`](COPYRIGHT)). Upstream copyright notices are retained; see the
[license and provenance notes](doc/docs/GPMEEP_LICENSE_AND_PROVENANCE.md).

## Citing gpmeep

If you use gpmeep in published work, please cite this software. The metadata,
including the release version, is in [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button exports it as APA or BibTeX):

```bibtex
@software{yi_gpmeep,
  author  = {Yi, Dong-Joon},
  title   = {gpmeep: CUDA-accelerated, multi-GPU Meep distribution},
  url     = {https://github.com/iverydj/gpmeep-releases},
  license = {GPL-2.0-or-later}
}
```

gpmeep executes Meep's FDTD formulation, so please also cite the Meep paper
listed in `CITATION.cff`: A. F. Oskooi *et al.*, Comput. Phys. Commun. **181**,
687 (2010), doi:[10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).

The general Meep interface (geometry, sources, materials, monitors) is
documented in the [Meep manual](https://meep.readthedocs.io/en/latest/); the
gpmeep-specific documents are listed above.
