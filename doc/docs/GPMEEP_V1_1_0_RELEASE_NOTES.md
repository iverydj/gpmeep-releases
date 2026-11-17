# gpmeep v1.1.0 release notes

gpmeep v1.1.0 is the structural-performance and release-trust line built on
the frozen v1.0.3 compatibility release. It preserves the normal Python
namespace (`import meep as mp`) and the validated Meep feature surface while
requiring the user to choose CPU or CUDA explicitly. It never silently changes
the requested backend. A recommendation mismatch is a guarded warning that
requires acknowledgement, not an automatic selector.

The CUDA backend keeps FP32 fields resident across the ordinary Meep time-step
pipeline and accelerates supported curl, source, boundary, polarization, DFT,
Near2Far, adjoint, reduction, and MPI paths. Multi-GPU execution remains one
MPI rank per selected physical or MIG device. Unsupported required-CUDA
combinations fail during preflight instead of falling back to CPU.

The recommendation guard builds each CUDA device inventory from one native
snapshot instead of enumerating the same devices once per reported property.
This removes avoidable fixed overhead from short or repeated `Simulation.run`
calls without caching device state across policy attempts. A qualification-only
legacy control is retained for same-binary performance and numerical A/B
evidence.

The first mismatch still fails before mutation and requires the exact
reason-bearing, one-shot challenge. After its unchanged retry succeeds, the
default reason scope may be reused by the same `Simulation` object for an
identical process-local policy class, avoiding the deep Python workload
fingerprint and exception/retry cycle on every later run. A separate simulation
must acknowledge its own first mismatch unless the caller explicitly chooses
`scope="process"` for an intentionally accepted batch of equivalent decisions.
The simulation and process caches are separately bounded to 32 entries and bind
backend direction, classification, policy version, MPI topology, and threshold
facts; forked children cannot inherit authorization. Environment/device errors
remain non-acknowledgeable and are checked every time. Qualification can disable reuse with
`MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE=1` for same-binary overhead evidence.
Every MPI reuse retains an exact collective workload fingerprint so
rank-divergent geometry, sources, or callbacks still fail before mutation; only
a one-rank reuse may skip that deep consistency work.

MPI exact-fingerprint construction now verifies the complete immutable Meep
package dependency graph once before and once after each bounded canonical
traversal, rather than re-walking overlapping transitive dependency subgraphs
for every repeated class and function reference. Cheap import-snapshot checks
still bracket every trusted node whose state is read into a token, closing a
concurrent mutate-and-restore interval. Original trusted function definitions
are emitted from their authenticated import-time references rather than from
live replaceable function attributes; temporary replacement therefore either
leaves the token unchanged or is rejected by the node/final check. Canonical coverage, within-version
collective rank comparison, fail-before-run behavior, and mutation rejection
are retained. Opaque workload/native digest values intentionally bind the
policy and runtime identity and can therefore change across the v23-to-v24
boundary. This removes Python policy overhead that was material for short,
repeated multi-GPU runs without relaxing the warning or acknowledgement
contract.

Successful guard validation still has a fixed cost that can be noticeable when
a workload makes many short `Simulation.run` calls. CPU MPI diagnostics suggest
a residual scale of roughly 0.07–0.13 seconds per run for the sampled workloads,
including the associated Python wrapper and reporting work. No warnings or
acknowledgement retries occurred in those successful runs; this is not a
measurement of warning-emission time.

That residual estimate combines an earlier guard ON/OFF comparison with two
reverse-order pairs and a separate, later guard-optimization comparison with
one observation per condition. It is a cross-window estimate, not a directly
measured constant, confidence interval, or guarantee for other workloads.
The diagnostics do not establish how much of the remaining cost could safely
be removed. Validation results are not newly reused across runs: the existing
workload, mutation, MPI agreement, nonce, and acknowledgement checks remain
required. These observations do not qualify a release or relax any release
gate.

The archived `performance-diagnosis-20261107/gpu-regression-final-20261108`
record retains an unresolved, cumulative GPU `multilayer_opt` regression.
With observation enabled, the previous A build took 62.817139 s overall and
24.317343 s in Boundaries, versus 60.196760 s and 20.661346 s for the v1.0.3
release-environment rebuild: about 4.35% longer overall and 3.655996 s more in
Boundaries. Single observations across four commit intervals support a
cumulative regression; no single culprit or precise causal shares were
established. The ineffective `ac70c5c4` trial is excluded. These are archived
diagnostics, not measurements of the newly merged candidate.

CPU differences beyond the explained guard work also remain unresolved.
With matched dependencies, CPU-only gpmeep exceeded upstream by 7.068064 s
for `test_adjoint_cyl`, 4.469261 s for `parallel-wvgs-force`, and 6.104078 s
for `3rd-harm-1d`. Those totals still include the successful guard. Subtracting
the separate historical 3.082444 s guard estimate from the last total leaves
about 3.021634 s; this cross-window arithmetic is not a direct measurement of
guard-free cost. The archive records the raw paths, single-sample limitations,
and corrections to earlier summaries. Existing qualification criteria remain
unchanged.

The receipt-bound performance harness still obeys the explicit-choice guard.
When a sealed comparison intentionally uses a non-recommended backend, it
acknowledges only the typed challenge and retries the unchanged run; later
events explicitly distinguish a new acknowledgement from audited scope reuse.
The timing includes every attempt. OFF/ON qualification requires identical run
counts and verifies the expected difference in warning/acknowledgement counts. Probe
attestation time is kept outside the workload-only primary measurement, and a
digest proves that both arms returned the same public inventory. Normal
applications still require one explicit acknowledgement per distinct policy
class, use simulation scope by default (or opt into process scope explicitly),
and never have their CPU/CUDA choice changed automatically.

This release adds a default-off structured-curl CUDA Graph optimization for
eligible repeated phase topologies. It changes launch submission only; the
existing kernels, arithmetic order, and Yee update order are unchanged.
Topology, allocation generation, mirror identity, device, and captured
lifetime checks guard every replay. Uncertain launch or destruction failures
are fail-closed and quarantine the complete captured lifetime set. Enable the
qualified ablation explicitly with:

```sh
MEEP_GPU_ENABLE_STRUCTURED_CURL_GRAPH=1 python simulation.py
```

On the development RTX 3090 Ti host, a position-balanced two-cycle Near2Far
adjoint check improved from 21.559882 s to 19.533670 s on average (1.1037x),
with exact equality for every projected numerical value. An ineligible
parallel-waveguide force case bypassed all Graph bookkeeping and differed by
only 0.116%, with the two measured order pairs changing direction. These are
bounded implementation measurements, not a promise that every simulation is
faster.

Packaging uses a CUDA 12.4 Linux x86-64 Conda build against a glibc 2.17
sysroot. The development package has an installable `_development.conda` name
and a byte-bound `.NON_PUBLISHABLE` companion marker; only the exact
candidate package whose external final receipt says `RELEASE_AUTHORIZED` may
be tagged and distributed. The final release assets include the complete raw
CPU, frozen-v1.0.3 CUDA, and v1.1.0 CUDA timing tables, numerical comparisons,
multi-GPU results, portability witnesses, and embedded report plots.

Scheme, native GDSII helpers, co-installation with `pymeep`, and non-Linux
binary packages remain outside the v1.1.0 release contract. See
`GPMEEP_FEATURE_MATRIX.md`, `GPMEEP_INSTALLATION.md`, and the repository's
v1.1.0 release-trust document for exact scope and verification commands.
