# gpmeep Future Portability TODO

These items are intentionally outside the minimum v1.1.0 release claim unless
they are completed and added to the native validation matrix. The list prevents
an untested "all Unix" promise while preserving that direction.

- [ ] Add native clean-install and runtime qualification for further Ubuntu
  releases after the required Ubuntu 22.04 plus one-newer baseline.
- [ ] Test glibc-based Debian, RHEL, and Rocky Linux hosts.
- [ ] Publish the minimum NVIDIA driver and a driver/toolkit forward-
  compatibility matrix.
- [ ] Runtime-qualify real NVIDIA GPUs beyond sm86 and record native SASS/PTX
  selection on each architecture.
- [ ] Investigate Linux aarch64 with NVIDIA GPU packaging and MPI.
- [ ] Validate system-MPI versus Conda-MPI policy and CUDA-aware MPI by MPI
  implementation and distribution.
- [ ] Investigate musl/Alpine separately from the glibc Conda package.
- [ ] Define macOS and BSD CPU-only source-build behavior, including whether the
  gpmeep Python API should degrade gracefully when CUDA is unavailable.
- [ ] Design a backend abstraction before considering AMD ROCm/HIP or Intel GPU
  implementations.
- [ ] Revisit universal pip wheels only after CUDA, MPI, HDF5, MPB, and native
  ABI distribution constraints have a reproducible solution.

None of these items permits silent backend substitution. On a future platform,
the user still chooses CPU or an available accelerator and acknowledges a
recommendation mismatch explicitly.
