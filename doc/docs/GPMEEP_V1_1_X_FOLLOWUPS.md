# gpmeep v1.1.x follow-up work

Open items carried forward after the v1.1.0 release (2026-11-20). Most of them
are already listed in the v1.1.0 release notes; this file tracks them in one
place.

## Performance and correctness

1. GPU `multilayer_opt` cumulative regression remains UNRESOLVED. On the
   equal-step runs (run1 and run3) the v1.0.3/v1.1.0 time ratio is 0.945, so
   v1.1.0 is roughly 5-6% slower. GPU multilayer host overhead is part of this
   item.
2. Equal-work or fixed-step `multilayer_opt` validation. v1.1.0 applied a
   single-case release-gate exception that evaluates only the equal-step runs;
   the goal is to remove that exception.
3. CPU/GPU adjoint run2 stopping divergence. After the FilteredSource
   correction, v1.1.0 GPU run2 takes 605359 steps versus 206857 in v1.0.3.
4. Residual CPU slowdown beyond the guard cost.
5. Installed-package runner mode.
6. Unittest manifest registration.
7. `material_omp` runner and OpenMP configuration conflict.
8. Fixed-length build prefixes, so ELF bytes do not depend on the build output
   path length.
9. Open MPI shared-memory files (`/dev/shm/sm_segment.*`) are left behind by
   validation workers that import meep without `mpirun`, even after a normal
   exit. Reproduce, find the cause, and make runs clean up after themselves.

## Validation tooling

10. Reconcile installed files, links and ownership against the package archive,
    and reject unowned extra files.
11. Negative test for the obsolete `_0_release` build string.
12. Bind numerical-regression evidence to the actual campaign and M4 report IDs
    and raw data.
13. Derive feature and route counts from producer evidence, and strengthen the
    binding of host and specialized results.
14. Replace the fallback-counter aggregation, which can sum missing keys, with
    actual CUDA fallback evidence.
15. Strengthen the trust boundary of non-live replays and caller-declared image
    identity; bind actual install and namespace creation evidence.
16. Extend the media census beyond `<img>` elements.
17. Design signing and verification for annotated release tags.
18. Document that `validated_engine_commit` is an ancestor anchor; the shipped
    engine is the binary source commit recorded in `PUBLIC_RELEASE.json`.
19. Let audit records carry new finding IDs with an open or closed state.
20. Validate Ubuntu 22.04 and 24.04 compatibility on separate hosts, in addition
    to the namespace-isolated checks.
21. Reduce the build receipt's dependence on host system tool hashes, which
    change with routine operating-system security updates.

## Release tooling

22. Merge the release-tooling corrections made after the v1.1.0 source freeze
    into the source tree. The tagged v1.1.0 source contains the release tooling
    as it was at the freeze.
23. Retry briefly after GitHub write operations (repository settings, release
    publication) before verifying them, instead of failing on a delayed read.
