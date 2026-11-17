# gpmeep v1.1.0 C4-A2 static-material initialization plan

Status: M1 candidate contract. This document does not authorize release or
enable the feature by default.

## Objective

C4-A2 reduces host-side `init_sim/_set_materials` time for geometries whose
material evaluation is provably immutable and callback-free. It uses the
existing OpenMP material loops in Meep; it does not move geometry evaluation
to CUDA and does not alter the FDTD recurrence.

The motivating v1.0.x AuNP profile attributed about 55.7% of GPU1 end-to-end
time to material initialization. That percentage is historical evidence, not
a v1.1.0 performance claim. M1 measures C4 independently and in the combined
OFF/C1/C3/C1+C3/C4 ablation.

## Exact controls and legacy path

- C4 is enabled only by exact `MEEP_C4_MATERIAL_OMP_ENABLE=1`.
- `MEEP_C4_MATERIAL_OMP_DISABLE=1` explicitly selects the legacy serial path.
- Unset controls select the legacy serial path.
- Empty, malformed, non-`1`, or simultaneously present controls fail before
  material arrays are allocated.
- When C4 is disabled, C4-only ownership and epoch checks do not change the
  legacy compatibility surface.
- The measurement harness disables C1 and C3 in both C4 lanes so timing and
  causal counters remain attributable to C4.

## Eligibility boundary

Parallel evaluation is allowed only when every selectable material leaf is an
immutable built-in `MEDIUM` or `PERFECT_METAL` value. The default material,
every geometry leaf (including compound-object descendants), and every extra
material must pass the same check.

The following cases remain serial:

- Python or C user material callbacks;
- material grids, material files, and other mutable/custom subclasses;
- an unknown or malformed geometry/material subtype;
- a build without OpenMP; and
- any case whose safety cannot be established from the frozen geometry tree.

Epsilon and susceptibility-sigma evaluation use separate virtual safety hooks.
A future material implementation cannot opt into one by accidentally
inheriting approval for the other.

## Process-global geometry serialization

libctl geometry state is process-global. A recursive process-wide transaction
therefore surrounds Meep geometry construction, material array generation,
fragment statistics, material gradients, epsilon-grid extraction, cooperating
MPB initialization/evaluation entry points, and GDSII lattice mutation.

Every cooperating global mutation advances a monotonic geometry epoch. A
candidate `geom_epsilon` records the epoch of its frozen tree. With C4 enabled,
material evaluation requires both transaction ownership and an exact epoch
match. A stale tree or missing owner fails before parallel work begins. With
C4 disabled, the existing serial behavior is retained.

Prism membership has a libctl one-time boundary-policy initialization. The
first fixed prism is evaluated serially inside the transaction before any
parallel prism evaluation. Causal counters prove that this preflight occurred.

## OpenMP placement contract

The C4 canary initially uses four physical cores per MPI rank with:

- `OMP_NUM_THREADS=4`;
- `OMP_DYNAMIC=FALSE`;
- `OMP_PROC_BIND=close`;
- `OMP_PLACES=cores`; and
- Open MPI `--map-by core:PE=4 --bind-to core`.

This is a reproducible starting point, not the release optimum. M1 sweeps
2/4/8/16 threads where the host topology permits and retains the smallest
stable setting near the performance optimum. GPU1 and GPU2 measurements must
not oversubscribe the physical cores assigned to their ranks.

## Bitwise material oracle

`structure.material_arrays_manifest()` emits a read-only canonical inventory
for every local structure chunk:

- FP32 `realnum` width;
- chunk ownership and exact element counts;
- every `chi1inv[component][direction]` array;
- every susceptibility-sigma array in field-type and linked-list order;
- null/trivial flags, exact byte counts, and SHA-256 of non-null raw bytes.

The SHA-256 implementation self-tests the empty and `abc` vectors before its
first use. OFF and ON manifests must be identical on every rank. Fixed-window
field and observable comparisons remain mandatory; the manifest is not a
substitute for scientific validation.

## Required validation

Before C4 can enter the combined M1 candidate:

1. malformed, conflicting, unset, disabled, callback, and static-material
   controls follow the paths above;
2. Prism plus Lorentzian OFF/ON epsilon arrays, fields, and complete material
   manifests are bit-identical;
3. causal counters prove real parallel epsilon and sigma work with at least two
   observed threads and zero guard/stale/overflow events;
4. MPB regression passes after adding the shared transaction;
5. CUDA C1/C3 regression remains clean;
6. GPU1/GPU2 AuNP runs use two order-reversed repeats and the same binary;
7. initialization, FDTD, and end-to-end time are reported separately without
   an arbitrary minimum-improvement threshold; and
8. an adversarial milestone audit accepts the exact source, binary, manifests,
   numeric evidence, and timing provenance.

The current local smoke proves buildability and bitwise identity on a bounded
2D Prism case only. It does not yet establish AuNP or release performance.
