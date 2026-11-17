# gpmeep C3 sparse Lorentzian/Drude plan

This document defines the fail-closed C3 design boundary for the gpmeep 1.1.0
M1 performance milestone. It authorizes implementation and measurement on
`dev/v1.1.0`; it does not by itself authorize a release or a default policy
change. Historical measurements below are inputs to the current same-binary
ablation, not claims about the v1.1.0 candidate.

## Motivation and measured boundary

The C2 same-binary canary proved that omitting bit-identical zero-state stores
is correct for its selected fixed-window observables, but it improved the
topology-geometric-mean FDTD time by only 4.752% and end-to-end time by 1.618%.
It therefore ended in `NO_PROMOTION`.

The installed-v1 AuNP trace reports 48,000 standard Lorentzian/Drude dispatches
and 101,577,600,000 structured point visits on GPU1 in the 2,000-step structure
phase: 24 recorded standard dispatch calls per structure timestep. The
reference phase reports exactly zero polarization calls. The current C1 helper
still evaluates the complete recurrence at every owned
point; it only suppresses two stores after proving that the result is positive
zero. C3 instead tests whether stable inactive points can be removed from the
structure-phase launch domain.

This has a higher per-polarization-call ceiling than C1 because the Au
structure occupies only an estimated 14.8% of the complete 2D cell area. The
percentage is geometry evidence, not an assumed Yee-grid active fraction or an
end-to-end speedup prediction. C3 must measure the exact device-side active
count before choosing a sparse launch. Empty active sets remain required
correctness coverage but are not part of the measured AuNP performance claim.

## Immutable safety boundary

- The feature is off by default and enabled only by exact
  `MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN=1`.
- Presence of `MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN` always wins.
- Unset, empty, malformed, and non-`1` enable values use the dense v1 path.
- Each resident cache records an effective control token. Any transition away
  from enabled immediately makes every sparse plan and field guard in that
  cache execution-ineligible. Their device storage is released only after a
  checked quiescence of the owning stream at the normal safe phase boundary;
  until then it remains quarantined and cannot execute. Re-enabling always
  restarts at absent, performs one dense observation, and rebuilds from current
  state.
- The C1 zero-state-elision toggle remains independent. Sparse A/B runs keep
  the C1 setting identical on both sides so the new effect is isolated.
- CPU, gyrotropic, multilevel, noisy, anisotropic, non-FP32, and unsupported
  coefficient paths are unchanged.
- Only the exact standard Lorentzian/Drude CUDA dispatch may opt in.
- Any ambiguous identity, state lifetime, allocation, mutation, or validation
  case uses the dense path or propagates the pre-existing runtime error.
- Every identity or lifecycle invalidation is a two-stage retirement, including
  control, host upload/address, coefficient/`fields.dt`/options, cache reset,
  device migration, and epoch/generation exhaustion. The old plan, guard, and
  index become execution-ineligible before the next dispatch; no device
  allocation is physically freed until checked quiescence of its owning stream.
  Quiescence failure retains an execution-ineligible quarantine and dense-locks
  the cache. No mismatch path may call `cudaFree` based only on host identity.

## Reuse of the existing numerical kernel

The CUDA runtime already exposes `update_lorentzian_fp32` and
`update_lorentzian_increment_fp32`, which accept a device-resident sorted array
of explicit field indices. C3 must reuse their indexed-kernel implementation
and exact recurrence for sparse work. A guarded conditional launch variant may
be added, but it must not introduce a different recurrence. The existing
structured implementation remains the authoritative dense path and guarded
fallback.

## Exact inactive predicate

For one owned Yee point to be omitted, all of the following must hold when the
plan is built:

1. diagonal sigma has the exact FP32 positive-zero bit pattern;
2. polarization state has the exact FP32 positive-zero bit pattern;
3. auxiliary previous/increment state has the exact FP32 positive-zero bit
   pattern;
4. both off-diagonal sigma pointers are absent;
5. every recurrence coefficient is finite; and
6. a runtime coefficient-tuple predicate evaluates the existing recurrence
   for the exhaustive finite-field sign classes (`+0`, `-0`, positive, and
   negative) and accepts only tuples that return bit-identical positive-zero
   state; host/device tests must prove predicate agreement, and the device
   predicate result must gate compaction so disagreement becomes dense.

Negative zero, nonzero, NaN, and infinity in sigma or either state are active,
not inactive. The device compaction predicate uses bit inspection, not
floating-point equality.

## Non-finite field guard

Multiplying positive-zero sigma by a NaN or infinity is not a zero-state
invariant. Sparse execution therefore requires a device-resident sticky guard:

1. once per unique field/index-space in a resident polarization phase, scan
   the complete owned field domain for non-finite values;
2. set a device flag if any value is non-finite and never clear that flag for
   the current effective-control generation and guard lifetime;
3. the sparse indexed kernel executes only when the flag is zero;
4. the structured dense kernel executes only when the flag is nonzero; and
5. after the flag is set, every later update remains dense until that guard's
   effective-control generation ends or its owning cache is reset.

The scan, conditional sparse launch, and conditional dense launch stay in one
CUDA stream. No host readback is allowed on the steady-state path. A non-finite
field therefore receives the same dense recurrence instead of silently
preserving omitted zero state.

## Resident-plan state machine

| State | Action | Next state |
|---|---|---|
| absent | record exact mirror/index-space identity; run dense | observed |
| observed, identity unchanged | build active counts and sorted indices transactionally | sparse or dense-locked |
| observed, identity changed | discard observation; run dense | observed |
| sparse, identity unchanged | field guard plus conditional indexed/dense launch | sparse or poisoned |
| sparse, host upload/device address changed | logically retire and quarantine plan; run dense; physically release only after checked owning-stream quiescence | observed |
| dense-locked | run dense without rescanning | dense-locked |
| poisoned field guard | run dense for every dependent plan | poisoned |
| any enabled-to-disabled/malformed transition | logically retire and quarantine all plans/guards immediately; run dense; physically release only after checked owning-stream quiescence | absent-disabled |
| absent-disabled, exact enable becomes active | record new control token; run dense | observed |
| any coefficient/`fields.dt`/options identity mismatch | logically retire and quarantine old plan immediately; run dense; physically release only after checked owning-stream quiescence | observed |

Requiring one dense observation before device compaction prevents a
material-phasing cache that is destroyed every timestep from paying an O(N)
plan-build cost. It also allows compaction to inspect the authoritative device
state after the first update rather than a stale host copy.

## Plan identity and invalidation

One plan identity includes:

- host and device pointers for sigma, polarization, and auxiliary state;
- the complete index-space descriptor and array size;
- previous-state versus increment-state representation;
- the exact FP32 bit patterns of `gamma_inverse`, `gamma_previous`,
  `omega_dt_squared`, and `omega_dt_squared_denominator`;
- the exact device-material option bits, including the independent C1 option;
- the three mirrors' upload epochs; and
- the resident cache allocation generation, CUDA device ordinal, and effective
  sparse-control generation.

Normal device-side polarization updates do not invalidate the plan because the
omitted positive-zero state is an invariant under the guarded finite-field
contract. A host upload or mirror replacement must be detected by the stored
upload epoch and device-address identity before reuse. Cache reset, device
migration, material change, structure phase-in, `zero_fields`, real/complex
conversion, susceptibility removal, and checkpoint load already destroy or
clear the owning resident cache. C3 must add its allocations to every such
clear/reset and leak-accounting path and test them explicitly. Existing
checkpoint-save rejection for live polarization state must remain unchanged.
Changing public `fields.dt` may change the coefficient tuple without changing
the state representation. A tuple or option mismatch therefore invalidates the
old plan before dispatch, runs the complete dense recurrence once, and returns
to `observed`. The old allocation remains quarantined and ineligible under the
new identity until the common checked owning-stream-quiescence rule permits its
physical release.

Upload epochs and every C3 plan-identity allocation, control, guard, and cache-
lifetime generation are checked, saturating counters rather than wrapping
counters. Before an increment would overflow, the cache makes every C3 plan and
guard execution-ineligible, waits for their owning stream at the normal safe
phase boundary, releases them, and permanently dense-locks sparse execution for
the remaining cache lifetime. An epoch or identity generation value is never
reset and reused within that lifetime, so a wrapped value cannot create an ABA
identity match.

A reconstructed cache may reuse the same numeric host or device addresses. It
becomes a new lifetime only after checked quiescence of every owning stream,
complete destruction of the old plans, guards, mirrors, ledgers, snapshots, and
identity records, and removal of every dependency that could reference them;
address uniqueness is not a safety assumption. Quiescence failure leaves the
old storage retained in an execution-ineligible quarantine, permanently
dense-locks that cache, marks exact statistics/teardown incomplete, and reports
the error from checked operations rather than risking reuse or an asynchronous
use-after-free.

## Deterministic device compaction

Plan construction uses two device passes over the structured owned domain:

1. count active points in fixed 256-point logical blocks;
2. perform a checked device-to-host copy of the small block-count vector;
3. compute checked 64-bit exclusive offsets on the host;
4. upload the complete offset vector to temporary device storage;
5. allocate an exact-sized device index array;
6. fill it with a per-block deterministic prefix so indices remain in
   structured order; and
7. perform a checked event or synchronization after the fill before committing
   metadata.

The active count, offsets, byte sizes, and launch dimensions require checked
overflow handling. Temporary count/offset buffers are released before commit.
An identity mismatch makes an old plan execution-ineligible before replacement
work begins. Its allocation may remain privately owned until the new identity
is committed, but it cannot execute. A count-kernel, count-copy, prefix,
offset-upload, allocation, fill-kernel, or completion failure runs the new
identity densely and must not publish a partial or replaced plan. Tests inject
failures independently at every asynchronous boundary.

No index array is allocated for an empty active set. An all-active set or a
candidate outside the automatic-policy thresholds becomes dense-locked.

## Conservative automatic policy

Even with the environment opt-in, sparse execution is selected only when:

- the structured owned domain has at least 65,536 points;
- the active fraction is at most 35%;
- the plan passed its second-observation stability check; and
- the exact inactive and coefficient-invariance requirements pass; and
- adding its exact index bytes keeps the sum of retained sparse indices in the
  owning cache at or below 5% of the cache's resident-mirror bytes measured
  after the first complete dense observation.

These constants are experimental and fixed before the AuNP canary. They may be
changed only by a separately measured and audited policy experiment, never
after viewing one favorable timing sample.

## Memory and multi-GPU contract

Each locally owned `fields_chunk` has its own resident cache. That cache holds
a tuple-keyed plan map for every local susceptibility/component/complex-part
identity and a field/index-space-keyed guard map with explicit dependent-plan
links. A rank may own multiple chunks and every chunk may hold multiple poles
and real/imaginary tuples. No cache or rank shares device pointers or plan
metadata. MPI collectives and boundary exchange are unchanged.

A retained plan uses `active_count * sizeof(ptrdiff_t)` bytes plus bounded
metadata and one four-byte sticky guard per unique field/index-space. Rejected
plans retain no index allocation. The 5%-of-mirror per-cache retained-index
budget composes across multiple local caches; temporary compaction storage is
still included in measured peaks. Peak host RSS, host RSS by rank and total,
and attributed memory on each selected GPU remain separate gates; no value may
increase by more than 10% without rejecting the experiment.

GPU2 evidence must aggregate every local cache into each rank's total, active,
and dense-fallback point counts while retaining per-cache diagnostics. Reducing
work only on the already busy dispersive rank is valid, but it must not be
described as solving MPI imbalance unless a controlled result shows that.

## Causal counters

The experimental statistics must distinguish at least:

- opt-in calls and points;
- eligible and ineligible calls and points;
- first-observation dense calls;
- plan builds, hits, invalidations, and dense-locked decisions;
- exact total and active plan points;
- sparse indexed calls and launched points;
- finite-field guard scans and scanned points;
- poisoned/dense-fallback calls and points; and
- CPU polarization fallback calls and points.

Submission counters are host-side. Actual sparse versus conditional-dense
execution is recorded by a per-cache device `uint64` ledger updated inside the
branch that executes. The steady-state path performs no counter readback. A
separate explicit statistics query synchronizes, performs checked ledger
readback for all live local caches, folds normally destroyed-cache ledgers into
process totals exactly once, and reports an error if a noexcept teardown could
not preserve an exact ledger. Multiple queries use per-cache snapshots and do
not double-count. Counter overflow is saturating and reported.

The new statistics API has a separate checked reset. Query and reset are legal
only while no resident phase is active; reset synchronizes all live ledgers,
zeros them and the generation-tagged destroyed-cache totals, and establishes a
new generation for caches created later. The existing noexcept dispatch reset
is unchanged and does not silently perform device work. Query/reset,
live-cache destruction, discard, multi-cache aggregation, empty plans, and
failure paths require explicit tests.

The statistics reset generation is also a checked, saturating, non-reusing
counter. If a reset would exhaust it, the checked reset returns an explicit
error before zeroing any ledger, total, or snapshot and leaves the current
generation and queryable totals unchanged. That reset and all later resets
remain refused for the process lifetime; no destroyed-cache total or live
ledger may be reattributed to a reused generation.

Queries do not allocate or increment a query/snapshot generation. Each cache
stores its last complete absolute device-ledger snapshot under its non-reused
cache-lifetime identity. A query computes the checked nonnegative delta from
that snapshot and updates the snapshot only after the complete checked readback
succeeds. Cache destruction similarly folds the final absolute delta exactly
once before deleting the cache identity. Query/readback failure leaves the
prior snapshot and totals byte-for-byte unchanged and queryable by a later
successful call. Thus repeated queries have no query-generation value that can
wrap, be exhausted, or create ABA attribution.

## Public compatibility boundary

C3 adds a separate `sparse_lorentzian_statistics` structure plus checked getter
and reset. It does not append fields to the existing
`polarization_statistics`, change its layout, or change the existing eight
Python polarization keys. The existing exported indexed/structured
Lorentzian function signatures, default descriptor semantics, statistics
getter, and noexcept reset remain unchanged; guarded scan/launch and compaction
functions use new additive symbols. Python exposes the new statistics and
reset through separate methods. Pre-C3 header consumers must compile and link
unchanged against the new library, and existing Python exact-key tests must
remain unchanged.

## Required validation before a performance canary

1. Host and device formula tests exhaust field sign classes, both state forms,
   Drude/Lorentz coefficients, negative zero, negative coefficients, and
   NaN/infinity boundaries. They also change `fields.dt` within the same state
   kind across safe-to-unsafe, signed-zero, negative, and non-finite coefficient
   tuples and require dense invalidation.
2. Compaction tests prove exact active counts and sorted indices for linear and
   irregular 2D/3D index spaces, including empty, all-active, threshold-edge,
   and overflow cases.
3. Same-device A/B tests compare every selected state bit for previous and
   increment forms, real and complex fields, empty and mixed media, and
   poisoned non-finite fallback.
4. Production dispatch tests cover exact environment parsing, disable
   precedence, one-dense-observation behavior, enable-to-disable/malformed-to-
   re-enable invalidation with changing state/coefficients/field, counter
   semantics, and zero CPU fallback.
5. Lifecycle tests cover material phase-in, host reupload, cache reset,
   backend/device change, `zero_fields`, checkpoint-load invalidation,
   unchanged checkpoint-save rejection, real-field conversion, injected
   allocation/copy/commit failure, and leak gauges.
6. Counter-exhaustion tests inject every upload epoch and plan/control/guard/
   cache-lifetime generation one increment below saturation. They prove the
   next increment makes all prior identities immediately ineligible, keeps the
   current and every future dispatch dense for that cache lifetime, never
   reuses the saturated value, and cannot recreate an ABA match. Separate
   near-saturation tests for the statistics reset generation prove that
   exhaustion rejects reset before mutation, preserves queryable totals exactly
   once, and never reattributes a destroyed or live ledger to a reused
   generation. Repeated-query, failed-readback, and cache-destruction tests prove
   absolute snapshots update only after success and fold each delta exactly
   once without any query-generation counter.
7. Async-lifetime tests queue an old sparse kernel, then independently inject
   effective-control, host upload/address, `fields.dt`, coefficient, options,
   cache-reset, device-migration, and epoch/generation-exhaustion invalidations
   before it completes. Every later dispatch must be immediately dense while
   the retired plans and guards remain quarantined. Physical free is allowed
   only after checked owning-stream quiescence. An injected quiescence failure
   must retain the quarantine, permanently dense-lock the cache, surface
   checked-operation/statistics incompleteness, and show no use-after-free or
   leak-gauge undercount.
8. Multi-cache tests cover multiple local chunks, multiple poles, and real and
   imaginary tuples on one rank, with exact rank aggregation and independent
   invalidation.
9. Pre-C3 C++ header consumers compile/link without source changes; the exact
   existing Python polarization keys and exported runtime signatures remain
   unchanged.
10. The full single-GPU `gpu-step-db`, installed Python GPU API, MPI GPU1/GPU2
   smoke, DFT, CW, cylindrical sparse/fallback paths, adjoint, and existing
   coverage suites remain PASS.
11. An adversarial reviewer accepts the exact source, binary, tests, and runtime
   attestation before any AuNP timing is run.

## Canary and advancement gates

The C3 canary reuses the C2 fixed-window AuNP TM harness with two
order-reversed samples each for GPU1 and GPU2. The baseline and candidate use
the same binary and identical C1 setting. It records exact flux/field-energy
signatures, rank/device identity, all causal counters, host/GPU memory,
temperature, power, clocks, and throttle reasons.

C3 remains eligible for the combined v1.1.0 candidate only if all predeclared
correctness and non-regression gates pass:

- topology-geometric-mean FDTD and end-to-end changes are reported without an
  arbitrary minimum-improvement threshold;
- any claimed gain is positive and repeatable in the paired measurements;
- no topology metric regression over 3%;
- GPU2/GPU1 FDTD scaling at least 1.1x;
- repeat max/min at most 1.20;
- bit-identical paired observables;
- zero CPU polarization fallback;
- host rank/total and each GPU memory increase at most 10%; and
- comparable start temperatures, maximum temperature at most 90 C, and the
  predeclared strict throttle policy.

A positive short canary authorizes a longer 20-time/`5e-8` decay validation
and broad regression; it does not itself prove long-duration scientific
equivalence or authorize v1.1.0. A failed correctness, stability, provenance,
or material-regression gate excludes C3 from the release candidate. A safe,
repeatable smaller gain may remain eligible and is judged in the combined
same-binary M1 ablation.
