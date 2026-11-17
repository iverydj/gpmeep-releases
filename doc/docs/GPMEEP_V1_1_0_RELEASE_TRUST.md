# gpmeep v1.1.0 Candidate and Release Trust

Status: normative M4-A operator contract. This document does not declare that
v1.1.0 has passed qualification or been released.

## Why authority is split

A package cannot contain proof that tests performed after that package was
built have passed. gpmeep therefore uses two receipts:

1. The package-owned candidate receipt grants only
   `CANDIDATE_BUILD_AUTHORIZED`. It proves that the source-side M4 option-A
   gates passed before construction. Its `package_publication` and `git_tag`
   fields are always false.
2. The external final receipt grants `RELEASE_AUTHORIZED` only after installed
   qualification. It binds one exact `.conda` byte stream by file name, byte
   count, SHA-256, candidate-receipt ID, and source commit. It is never inserted
   into or used to rebuild the qualified package.

The split prevents the old M3 evidence-only receipt from being interpreted as
release authority and avoids the circular claim that a pre-build artifact has
already passed post-build tests.

## Stage 1: measure a package, close every inversion, then construct candidate bytes

The M4 candidate manifest uses schema
`gpmeep-v110-m4-candidate-input-v1`. It contains exactly these top-level keys:
`schema`, `state`, `validated_engine_commit`, `release_source_commit`, `option`,
`findings`, `validation`, and `evidence`. `option` is `A_ALL_11`; all eleven
recorded M4 findings must appear
in both the required and closed lists, and the material-open count must be
zero. Validation binds the targeted/common/full suites, 4 host cases, 5
specialized cases, 42 features, 32 routes, real multi-GPU without CPU fallback,
and the AuNP source candidate.

The 70-case paired correctness controller requires an external GPU-process
observation even for cases whose CUDA work lasts less than one sampling
interval. Its CUDA child retains its existing context at interpreter exit
until both original collectors have recorded that child, or the bounded wait
fails. The wait performs no additional CUDA work. The child records the exact
host/GPU row offsets, lengths and hashes with its PID, start time and run nonce;
paired replay checks those rows in the finalized streams and still applies the
complete original telemetry gates. Missing observations are never replaced by
native counters or a child success flag. CPU and ordinary non-opt-in runs do
not wait. Paired process durations include observation overhead and remain
ineligible for performance claims; M4 performance invocations do not enable
this correctness-only option.

Each evidence reference is relative to one external evidence root and binds
role, safe relative path, byte count, SHA-256, schema/media type, and content
ID. JSON without its own receipt/report/content/plan ID uses its canonical JSON
SHA-256 as the content ID. Symlinks, parent traversal, missing bytes, duplicate
JSON keys, non-finite values, schema changes, and hash changes fail closed.

M4 first freezes a measured-source commit, builds, installs, and measures an
exact `--development-build` package from that commit. Each
child process uses the Python, site-packages, MPI launcher, `_meep` extension,
and `libmeep` re-derived from that package's installed-provenance report; it
does not measure the nearby source build. The package-owned verifier is replayed
and every rank records the loaded runtime paths and SHA-256 values.

Before launching the 180-task campaign, reseal the upstream CPU baseline at
the exact clean measured-source commit and run the controller's zero-work
preflight. A receipt made before a later commit is stale even if no runtime
binary changed. Preflight validates the complete 180-task/120-measured-task
contract, both physical GPU UUIDs, installed package provenance, executable
identities, the complete Conda-v2 payload/`paths_data` correspondence, and
disk/host/cgroup capacity without creating an output tree or launching a
workload:

```sh
python3 -I scripts/python-validation/run_m4_equal_work_matrix.py \
  --preflight-only \
  --repo /absolute/measured/source --output /absolute/planned/m4-output \
  --python /absolute/v110/bin/python --build-python /absolute/build/bin/python \
  --install-prefix /absolute/v110 --build-receipt /absolute/v110/receipt.json \
  --mpiexec /absolute/v110/bin/mpiexec --upstream-receipt /absolute/upstream.json \
  --v103-install-prefix /absolute/v103 \
  --v103-provenance-report /absolute/v103/provenance.json \
  --v103-package /absolute/v103.conda \
  --v110-provenance-report /absolute/v110/provenance.json \
  --v110-package /absolute/v110-development.conda \
  --gpu-devices GPU-UUID-0,GPU-UUID-1
```

The controller seals every executable or normative input used to interpret
the campaign, including the inversion registry validator and the dedicated
inversion-experiment producer. Resume and final verification reject a byte
change to any sealed input.

All 13 observed inversions must then have an evidence-backed closed state:
proved hardware crossover, optimized-and-remeasured engine overhead,
or environment-fixed-and-remeasured. User-authored scope exclusions never
close a release gate.
Low-impact or nonrepresentative `open` records are not treated as closed.
`close-v110-inversion-registry.py` seals the 13-record result without overwrite.

The official M4 numerical replay is itself a typed evidence producer. It must
contain the exact 141 comparison-row identity census derived from the sealed
plan; 94 rows are measured cycles and the rest are warm-up/reference
comparisons. For the waveguide-crossing optimizer smoke, CUDA lanes are
strictly compared with the same-cycle gpmeep v1.1.0 CPU lane. The separately
built upstream CPU lane remains the reference for fixed-point physics and
structural optimizer invariants, while continuous NLopt trajectory values are
diagnostic because upstream repeat cycles can take different valid paths from
roundoff-scale objective changes. Both sides must independently pass bounded
design, progress, callback/evaluation, termination, and persistence oracles;
this exception cannot convert a gpmeep CPU/CUDA mismatch into a pass.
Every waveguide result must contain the canonical 950-key inventory and its
sealed key digest: 893 continuous optimizer metrics, 15 structural optimizer
metrics, 22 fixed-physics metrics, and 20 exact-work timestep scalars (12
fixed-work runs and 8 optimizer runs). The upstream/gpmeep CPU comparison
treats only the 893 continuous optimizer metrics as diagnostic and keeps 57
metrics verdict-bearing. Same-cycle gpmeep CPU/CUDA comparisons keep all 950
metrics strict. Missing metrics and same-count substituted metrics are
rejected. The two reported design-step norms are independently
recomputed from the initial 0.5 vector, the epoch-0 design, and the final
design.
Non-numerical timing, size/run crossover, profiler/ablation,
same-binary ablation, phase, and environment evidence is replayed from raw
sample terminals. The dedicated experiment producer
(`v110-inversion-experiment-sample.py`, producer schema v9 and child schema
v10) accepts no arbitrary child command. It can launch only the fixed package-owned
`v110-inversion-qualified-child.py`, using the Python bound by installed
provenance. Before the matrix starts, and again before every supplemental
sample, the exact package must parse as a bounded Conda-v2 archive; archive
payload ownership, installed `paths_data`, module, extension, libmeep,
manifest, verifier, launcher, and child bytes must agree. The package-owned
verifier and launcher must also equal their exact committed source files; the
verified package-owned verifier then re-derives the retained live
installed-provenance report and hashes every package-owned installed path.
The current package is approximately 22.4 MiB unpacked; each compressed outer
member is nevertheless allowed up to 128 MiB and each decoded tar stream up to
512 MiB. Decoding uses a temporary file and a child-process `RLIMIT_FSIZE` hard
stop, so a compression bomb cannot make the verifier buffer unbounded decoder
output in RAM. Timeout, decoder failure, compressed-input overflow, and
decoded-output overflow are distinct fail-closed results.
Every supplemental sample also performs a bounded authority replay of the
sealed `M4_COMPLETE` tree and must match its exact package SHA-256, provenance
report ID, candidate receipt ID, runtime paths, live-verification result,
archive-verification result, and measured-source commit. The bounded replay
re-derives the exact contract from the tracked source and installed runtimes,
then checks all 180 task commands, process logs, observer logs, telemetry
replays, child `COMPLETE` markers and reports, and every canonical file record
embedded in those reports. It also re-derives the performance table and all 13
inversion records from the child reports, validates the numerical PASS-row
identity census, checks the terminal-bound report/journal/checkpoint, and
replays the complete descriptor-rooted member census and byte hashes of every
resume archive. It deliberately does not repeat the expensive raw numerical
comparisons before each sample: the full semantic `verify_complete` replay
remains mandatory when sealing M4 and at the final release gate. The per-sample
replay runs in a fresh `python -I` subprocess using the provenance-bound
runtime Python; the complete evidence-tree metadata is compared before and
after so replay cannot mutate its own authority. Unit
fixtures exercise failure paths but are never release authority: the zero-work
preflight above must pass against the actual rebuilt package, installed prefix,
and installed-provenance report before the matrix can start. Installed and
committed child bytes must be identical,
and both the workload and child must be tracked at the exact clean source
commit. The child independently imports the
recorded meep module, derives the provenance report ID, queries requested and
active backend, physical GPU UUID, and native counter deltas, then publishes a
nonce-bound terminal with zero opposite-backend fallback and phase timings.
Only allowlisted experiment/environment controls are accepted. A CPU child
presented as CUDA, a wrong receipt or UUID, nonzero fallback, `/bin/true`, a
forging script, command drift, or changed raw artifact fails closed.

Supplemental samples use the same deterministic workload invocation as M4:
the fixed validation arguments, the two-branch Absorber/PML adapter, the
`run_k_point` observer for material dispersion, and four-test discovery for
cylindrical adjoint validation. The phase-in-material and cylindrical adjoint
fixed-work modes remain enabled. Source support files are bound to the exact
committed bytes and retained in each sample's workload contract. The child
preserves `sys.argv` identity and restores the original Simulation methods
while the backend guard authenticates and invokes them. Each sample has a
separate writable working directory; the existing user `HOME` is preserved
for MPI initialization. A callable decay criterion cannot be replaced with
an arbitrary numeric duration to manufacture a size/run sweep.

The size/run sweep is deliberately a supplemental single-rank (`cpu1` versus
`cuda1`) diagnostic; it cannot decide the user-facing CPU/GPU recommendation
alone. A hardware-crossover closure must additionally bind the two raw final
M4 cycles for the workload's sealed practical CPU topology (`cpu8` except the
plan's explicitly justified `material-dispersion` `cpu1` case) and GPU1.
Same-binary ablation includes an explicit `curl-phase-replay=off/on` profile,
alongside the other allowlisted engine controls, so the current M4 candidate
can be evaluated with reversed order in both cycles. Both arms force the same
phase-batched curl path and vary only replay; retained native counter deltas
must prove zero replay activity when off and successful replay hits when on.

The `device-inventory-snapshot=off/on` profile is the corresponding
same-binary control for recommendation-guard device discovery. Each child
performs a bounded probe before the workload and records native availability
and enumeration counters. The ON arm must use exactly one enumeration; the
legacy OFF arm must use exactly `1 + 10 * visible_device_count`, while both
return the same public inventory. Workload timing begins only after this probe
is reset. The authoritative `primary_seconds` is the child's sum of all wrapped
`Simulation.run` wall times; the producer retains subprocess wall time only as
non-primary diagnostic data. A deliberately delayed probe therefore cannot
manufacture an intervention benefit. The child also records a canonical digest
of the complete public inventory, and closure requires the OFF and ON digests
to match.

The `backend-ack-scope-cache=off/on` profile isolates Python guard overhead in
the same binary. OFF forces the exact warning/acknowledgement handshake for
every mismatched run. ON keeps the exact first handshake but permits subsequent
runs in the qualification batch to skip the exception/retry through the
explicit `scope="process"` opt-in. A one-rank reuse also skips the deep
rank-consistency fingerprint, while every MPI reuse retains that exact
collective check. Reuse across
separate `Simulation` objects is permitted only for an identical policy class:
backend pair, recommendation reason and confidence, policy version, MPI
topology, and threshold-category facts. A different class or topology cannot
inherit the suppression, fork-inherited state is rejected, and environment or
device-mapping failures are always rechecked. The workload must contain
multiple runs, use reversed arm order across two cycles, and retain the same
physics and run count.

Because the inversion experiment intentionally exercises a receipt-selected
backend, the package-owned child catches only the typed recommendation-mismatch
exception, submits a fixed reason-bearing acknowledgement for that exact
challenge, and repeats the unchanged run. Its measured wall time includes the
failed guard attempt, acknowledgement, and retry, and the child terminal records
both the acknowledgement count and a validated v3 policy-telemetry projection
after every successful run. Same-binary closure requires identical run counts
in every OFF/ON arm. Ordinary engine controls also require identical
acknowledgement counts. The dedicated
`backend-ack-scope-cache=off/on` control instead requires more than one run,
one exact acknowledgement per run when OFF, and exactly one acknowledgement
followed by audited process-scope reuse when ON. The registry independently
requires one shared policy version, backend pair, reason, scope digest, and MPI
topology; OFF records must all contain exact fingerprints, whereas ON records
after the first must explicitly report process-scope reuse and the originating
challenge cohort. One-rank reuse has no deep fingerprint; MPI reuse must retain
its exact rank-consistency fingerprint. These sequence and count differences
prove that the intervention changed only repeated warning work rather than
bypassing the guard.

The v24 policy implementation preserves exact MPI fingerprint evaluation while
deduplicating trusted-package integrity work within a single traversal. It
checks the complete registered Meep dependency graph before the first trusted
package token and again after the complete workload or rank-local token has
been built. Import-snapshot checks also bracket each trusted node whose state
is serialized, so a transient mutate-and-restore interval cannot alter a token
without rejection; original trusted function definitions are serialized from
the authenticated import-time references, so an entirely restored temporary
replacement does not enter the token. Corresponding ranks must produce the same digest within
one policy/runtime identity; the opaque digest value may change across policy
versions. Qualification must retain the MPI rank-divergence, transient
mutation, production-callsite, and traversal-end negative tests alongside the
performance result.
This qualification behavior
never changes the selected backend and is not part of the user-facing
automatic-selection policy.

Create each supplemental raw sample from a clean, exact source commit and the
already verified installed package. For example, one CUDA ablation sample is:

```sh
python3 -I scripts/v110-inversion-experiment-sample.py --create \
  --output /absolute/evidence/curl-phase-replay-on-cycle-1 \
  --repo /absolute/clean/source \
  --source-commit COMMIT \
  --runtime-provenance /absolute/v110/installed-provenance.json \
  --package /absolute/v110-development.conda \
  --install-prefix /absolute/v110 \
  --m4-output /absolute/m4/equal-work-complete \
  --workload-id python/examples/refl-quartz.py \
  --cycle-index 1 --topology cuda1 --gpu-device GPU-UUID \
  --environment-label baseline \
  --gpmeep-ablation curl-phase-replay=on
```

The size/run diagnostic accepts `--topology cpu1` or `cuda1` and the paired
`--gpmeep-sweep-size` and `--gpmeep-sweep-run-length` options. It does not
replace the practical CPU8/GPU1 M4 comparison used for a recommendation.

Before the final M4 source is accepted, the repository must already be
versioned as 1.1.0. An independent audit then freezes the source policy and the
exact verifier bytes outside the repository. The release-source commit must be
the exact M4 measured-source commit: no tracked source, documentation, test,
script, packaging, or metadata change is permitted after measurement. All
later evidence and reports live outside the repository. The candidate is
constructed from that same commit with the independently supplied SHA-256
pins, an external mode-0600 M0 attestation, and the closed registry:

```sh
/bin/bash -p scripts/build-conda-package.sh \
  --candidate-build \
  --output-dir /absolute/clean/output \
  --candidate-evidence-root /absolute/m4/evidence \
  --candidate-attestation /absolute/m0/attestation.json \
  --candidate-attestation-sha256 SHA256 \
  --m4-candidate-manifest /absolute/m4/evidence/candidate.json \
  --source-transition-policy /absolute/audit/source-policy.json \
  --source-transition-policy-sha256 POLICY_SHA256 \
  --source-transition-verifier /absolute/audit/source-verifier.py \
  --source-transition-verifier-sha256 VERIFIER_SHA256 \
  --candidate-inversion-terminal /absolute/m4/evidence/CLOSED_INVERSION_REGISTRY.json \
  --candidate-inversion-terminal-sha256 CLOSED_REGISTRY_SHA256 \
  --candidate-inversion-terminal-id CLOSED_REGISTRY_TERMINAL_ID \
  --candidate-inversion-records-directory /absolute/m4/evidence/closed-records
```

The build script validates the manifest-bound source-transition report with
the independently pinned verifier directly, using `/usr/bin/python3 -I -S`,
an empty environment, and `/` as its working directory. This happens before
any Python program from the candidate repository is imported or executed.
The publication script repeats the same external preflight before invoking
repository asset tooling. A tracked or untracked import-shadow file in the
candidate tree therefore cannot become the first authority that approves its
own source.

The package owns `share/gpmeep/v110-candidate-build-receipt.json`. That receipt
authorizes candidate construction only. A development build instead uses
`--development-build`, receives `DEVELOPMENT_ONLY`, retains an installable
`*_development.conda` name, and receives a read-only `.NON_PUBLISHABLE`
companion marker bound to its name, size, SHA-256, and source commit. The final
verifier compares the development package used by
M4 against the candidate package. Every installed payload path and byte must be
identical except the embedded candidate-receipt file. Both archives are fully
reparsed, and `info/paths.json` ownership hashes, sizes, path types, and
installation semantics must match. Prefix replacement and binary/text
relocation metadata are rejected. Thus a recipe change or install-time
transformation cannot silently alter the engine between measurement and
release.

## Stage 2: qualify the exact package

The external qualification manifest uses schema
`gpmeep-v110-post-package-qualification-v1` and contains exactly `schema`,
`state`, `source_commit`, `candidate_receipt_id`, `package`,
`installed_provenance`, `qualification`, `performance`, and `report_plots`.

It must bind:

- the exact candidate file name, size, SHA-256, Conda format/name/version/build,
  Linux subdir, source commit, and embedded candidate-receipt ID;
- fresh installed provenance from two distinct native environments:
  Ubuntu 22.04 and Ubuntu 24.04;
- development-package M4 numerical correctness and real multi-GPU UUID use,
  followed by exact runtime-payload byte equivalence to the candidate and a
  fresh candidate-bound GPU2 self-check in each Ubuntu installation; plus a
  direct installed-candidate TM-only AuNP contract (`minimum_run=20`,
  `dft_field_decay=5e-8`), all with zero CPU fallback;
- a raw ELF census that records every audited ELF byte count/SHA-256 and every
  required GLIBC symbol version. Candidate package ELF bytes must remain at
  the portable sysroot target (at most GLIBC 2.17); installed qualification
  rejects any overall requirement newer than GLIBC 2.35;
- a final adversarial audit with zero open critical/high/medium findings;
- Markdown and HTML final reports whose visible tables contain every retained
  raw second, mean/ratio, and all 141 numerical rows (including the 94 measured
  rows). Hidden JSON alone is insufficient. The HTML contains one canonical
  embedded PNG speedup plot; its chunks, pixels, dimensions, bytes, and source
  data hash are deterministically re-derived from the M4 report; and
- two equal-work repetitions with actual seconds retained for all six lanes:
  upstream Meep CPU, gpmeep v1.1.0 CPU, frozen v1.0.3 CUDA GPU1/GPU2, and
  v1.1.0 CUDA GPU1/GPU2. The v1.0.3 commit and package SHA-256 are fixed in the
  validator; every lane retains two positive finite wall-second samples rather
  than ratios alone. Numerical oracles and no-fallback must pass, no material
  v1.1.0 regression may remain, and at least one structural optimization
  benefit must be demonstrated.

The two Ubuntu-environment witnesses are not platform labels copied into a
manifest. Run the witness producer inside each installed target environment:

```sh
python3 -I scripts/v110-native-witness.py --create \
  --prefix /absolute/native/prefix \
  --installed-report /absolute/native/installed-provenance.json \
  --source-commit COMMIT --package-sha256 PACKAGE_SHA256 \
  --candidate-receipt-id RECEIPT_ID \
  --image-identity "sha256:${PREPARED_UBUNTU_IMAGE_SHA256}" \
  --output /absolute/evidence/native-witness.json
```

It derives `/etc/os-release`, kernel, libc, cgroup and namespace identity,
installed-prefix identity, and the installed report hash from the live
environment. Qualification requires distinct prefixes, report bytes,
environment IDs, and namespace IDs for native Ubuntu 22.04 and 24.04 runs.
An isolated Ubuntu root filesystem is acceptable only when it runs in a fresh
mount/PID/user namespace, records an immutable prepared-image digest, and uses
the guest `/etc/os-release` and glibc. It must never claim `native-host`.
Ubuntu 22.04 and 24.04 must use distinct namespaces, prefixes, and installed
reports. A userland-only PRoot execution that shares the host namespaces is not
a valid witness. Since Linux may reuse namespace inode numbers after a short
process exits, retain the first bwrap namespace until the second environment's
live witness has also been created and verified; sequential short-lived runs
can otherwise appear to have the same namespace ID. The current Ubuntu 26.04 development host may be used as an
additional smoke test but cannot replace either required witness. Wider non-Ubuntu Unix
portability remains a future compatibility item, not a v1.1.0 release claim.

The independent final-audit role likewise references the raw immutable audit
producer terminal, pinned by an external SHA-256, rather than trusting a
handwritten wrapper summary. Its exact F-01 through F-13 finding inventory and
critical/high/medium open count are replayed; release requires zero material
open findings.

Create the final external receipt without modifying the package:

```sh
.envs/gpmeep-v1.1-test/bin/python3.11 -I scripts/v110-final-release-receipt.py --create \
  --package /absolute/gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0.conda \
  --candidate-receipt /absolute/candidate-build-receipt.json \
  --candidate-tool scripts/v110-release-gate-receipt.py \
  --contract packaging/v110-development-contract.json \
  --inversion-schema packaging/v110-performance-inversion-record.schema.json \
  --inventory packaging/v110-performance-inversion-inventory.json \
  --qualification-manifest /absolute/release/evidence/qualification.json \
  --evidence-root /absolute/release/evidence \
  --candidate-contract-gate /absolute/m4/contract-gate.json \
  --candidate-registry-gate /absolute/m4/registry-gate.json \
  --candidate-attestation /absolute/m0/attestation.json \
  --m4-candidate-manifest /absolute/m4/evidence/candidate.json \
  --candidate-evidence-root /absolute/m4/evidence \
  --source-transition-policy /absolute/audit/source-policy.json \
  --expected-source-transition-policy-sha256 POLICY_SHA256 \
  --source-transition-verifier /absolute/audit/source-verifier.py \
  --expected-source-transition-verifier-sha256 VERIFIER_SHA256 \
  --final-audit-authority /absolute/audit/final-audit-authority.json \
  --expected-final-audit-authority-sha256 FINAL_AUDIT_SHA256 \
  --output /absolute/release/final-release-receipt.json
```

Release-critical receipts, transition reports, assets, native witnesses, and
installed-provenance reports refuse overwrite, verify the published bytes and mode, and fsync both
the file and parent directory. A failed publication removes an uncommitted
destination instead of leaving a valid-looking partial terminal.

## Tag and upload

Run the publication controller without `--execute` first. This performs no
network, tag, or release mutation and fully re-derives the final receipt, but it
does create and verify a new local evidence bundle and asset manifest in the
supplied empty asset directory:

```sh
/bin/bash -p scripts/publish-v110-release.sh \
  --repo /absolute/clean/v1.1.0/worktree \
  --package /absolute/gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0.conda \
  --final-receipt /absolute/release/final-release-receipt.json \
  --candidate-receipt /absolute/candidate-build-receipt.json \
  --qualification-manifest /absolute/release/evidence/qualification.json \
  --evidence-root /absolute/release/evidence \
  --candidate-contract-gate /absolute/m4/contract-gate.json \
  --candidate-registry-gate /absolute/m4/registry-gate.json \
  --candidate-attestation /absolute/m0/attestation.json \
  --m4-candidate-manifest /absolute/m4/evidence/candidate.json \
  --candidate-evidence-root /absolute/m4/evidence \
  --source-transition-policy /absolute/audit/source-policy.json \
  --source-transition-policy-sha256 POLICY_SHA256 \
  --source-transition-verifier /absolute/audit/source-verifier.py \
  --source-transition-verifier-sha256 VERIFIER_SHA256 \
  --final-audit-authority /absolute/audit/final-audit-authority.json \
  --final-audit-authority-sha256 FINAL_AUDIT_SHA256 \
  --asset-directory /absolute/release/assets \
  --repository OWNER/REPOSITORY
```

Only after preflight passes may the identical command be repeated with
`--execute`. It requires a clean worktree at the authorized source commit,
creates and pushes annotated tag `v1.1.0`, opens a private GitHub draft, uploads
an exact asset manifest plus a portable integrity-only evidence archive (not
a semantic release authority), downloads
and re-verifies every asset, then publishes the draft. It never overwrites an
existing asset. If a network operation is interrupted, repeat the identical
command with `--execute --resume`; exact tag, draft, and already-published states
are replayed without deleting or rewriting remote state.

The repository's immutable-releases setting must be enabled: the controller
checks `GET /repos/OWNER/REPO/immutable-releases` before creating or pushing a
tag or release, and still requires the published release to report
`isImmutable=true`. Every downloaded asset set must contain the exact
tag-bound manifest ID. The annotated tag message binds the
source commit, package SHA-256, candidate receipt ID, final receipt ID, and
asset-manifest ID, and resume compares the exact remote tag object. This
protects honest crash/retry operation and later asset mutation. Compromise of a
repository administrator or GitHub account remains outside this trust model;
external verifiers should retain the annotated tag object and compare all five
embedded identities.

## Public release snapshot

The private source publisher above remains restricted to `iverydj/gpmeep`.
After that exact source/package release is published and immutable,
`scripts/publish-v110-public.py` publishes the reviewed distribution to the
existing public `iverydj/gpmeep-releases` repository. Its numeric repository
identity, existing main commit and unchanged v1.0.3 source-only tag are pinned.
The former organization migration controllers are retired for this destination.

The public snapshot extends only public history. Retained runtime/build source
must match the accepted original commit; allowed documentation, notebook-output
and synthetic-fixture changes undergo the separate content and privacy checks.
The original binary source commit and package SHA-256 remain authoritative.
All package members and public history are inspected, while private raw evidence
and private receipt files stay out of the public upload list.

Publication requires an independently pinned audit of the exact plan, source
and artifact inspections, implementation, documentation/media and remote history.
The new release is immutable and each public asset is verified against GitHub's
signed release attestation. The controller never rebuilds a package or replaces
an existing tag/asset. Interrupted publication resumes only against the same
reviewed inputs and exact remote state. See
`GPMEEP_V1_1_0_PUBLIC_PUBLICATION.md` for the full procedure and plan schema.

## Scope of the M4 terminal

`M4_COMPLETE` proves only the frozen equal-work measurement campaign. It is
not by itself permission to publish. After it exists, operators must replay and
close all 13 inversion records, construct and byte-compare the candidate,
install and qualify that exact package in both required native environments,
produce the GLIBC/report/final-audit evidence, and create the external final
receipt. Only `RELEASE_AUTHORIZED` in that final receipt permits tagging and
publication.

The AuNP measurement remains a separate fixed production check: TM only,
`minimum_run=20`, and `dft_field_decay=5e-8`. Its historical 0.25 performance
window is a workload-specific disclosure and is not silently mixed into the
general 10% v1.1.0-versus-v1.0.3 regression rule.
