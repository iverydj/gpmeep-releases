import contextlib
import json
import os
import unittest

import meep as mp
import meep._meep as _meep
import numpy as np
from meep import backend_policy


@contextlib.contextmanager
def environment_override(**changes):
    saved = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_with_explicit_qualification_acknowledgement(simulation, **kwargs):
    """Run once, explicitly acknowledging an expected test-only mismatch."""
    mismatch = None
    try:
        simulation.run(**kwargs)
    except backend_policy.BackendRecommendationMismatch as error:
        mismatch = error
        mp.gpu.acknowledge_backend_mismatch(
            error.challenge,
            "gpmeep CUDA qualification intentionally covers this crossover",
        )
        simulation.run(**kwargs)
    if mp.gpu.requested_backend == "cuda" and mismatch is None:
        raise AssertionError(
            "compact CUDA qualification did not exercise the recommendation guard"
        )


class TestGpuBackend(unittest.TestCase):
    def tearDown(self):
        mp.gpu.set_backend("cpu")

    def require_cuda(self):
        reason = None
        if not mp.gpu.compiled or not mp.gpu.runtime_available:
            reason = mp.gpu.runtime_diagnostic or "CUDA is unavailable"
        else:
            compatible = [
                device for device in mp.gpu.devices() if device["compatible"]
            ]
            if not compatible:
                reason = "no compatible CUDA device"
        if reason and os.environ.get("GPMEEP_REQUIRE_CUDA_TEST") == "1":
            self.fail(reason)
        if reason:
            self.skipTest(reason)

    def _run_sparse_public_simulation(self, backend, susceptibility_kind):
        enable = "1" if backend == "cuda" else None
        disable = None if backend == "cuda" else "1"
        with environment_override(
            MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN=enable,
            MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN=disable,
            MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION=None,
            MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION=None,
        ):
            mp.gpu.set_backend(backend)
            mp.gpu.reset_statistics()
            mp.gpu.reset_sparse_lorentzian_statistics()
            if susceptibility_kind == "lorentzian":
                susceptibility = mp.LorentzianSusceptibility(
                    frequency=0.29, gamma=0.06, sigma=0.3
                )
            elif susceptibility_kind == "drude":
                susceptibility = mp.DrudeSusceptibility(
                    frequency=0.29, gamma=0.06, sigma=0.3
                )
            else:
                raise ValueError(
                    "unknown sparse susceptibility fixture: "
                    + susceptibility_kind
                )
            medium = mp.Medium(
                epsilon=1.7, E_susceptibilities=[susceptibility]
            )
            simulation = mp.Simulation(
                # 6*6*64^2 gives each rank more than the fixed 65,536-point
                # eligibility floor in the mandatory two-rank qualification.
                cell_size=mp.Vector3(6.0, 6.0),
                resolution=64,
                geometry=[
                    mp.Block(
                        size=mp.Vector3(0.8, 0.8, mp.inf),
                        material=medium,
                    )
                ],
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.3, fwidth=0.12),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            run_with_explicit_qualification_acknowledgement(
                simulation, until=0.08
            )
            sparse = mp.gpu.sparse_lorentzian_statistics()
            field = simulation.get_array(component=mp.Ez).copy()
            simulation.reset_meep()
            released = mp.gpu.sparse_lorentzian_statistics()
            self.assertEqual(released["live_plans"], 0)
            self.assertEqual(released["live_guards"], 0)
            self.assertEqual(released["retained_index_bytes"], 0)
            return field, sparse

    def test_backend_control_and_inventory(self):
        mp.gpu.set_backend("cpu")
        self.assertEqual(mp.gpu.requested_backend, "cpu")
        self.assertEqual(mp.gpu.active_backend, "cpu")
        self.assertIsInstance(mp.gpu.compiled_architectures, str)
        self.assertIsInstance(mp.gpu.runtime_diagnostic, str)

        with self.assertRaises(ValueError):
            mp.gpu.set_backend("not-a-backend")

        runtime_was_available = mp.gpu.runtime_available
        mp.gpu.reset_statistics()
        devices = mp.gpu.devices()
        runtime_statistics = mp.gpu.statistics()["runtime"]
        self.assertEqual(
            runtime_statistics["runtime_availability_probes"],
            1 if mp.gpu.compiled else 0,
        )
        self.assertEqual(
            runtime_statistics["runtime_device_enumerations"],
            1 if runtime_was_available else 0,
        )
        self.assertIsInstance(devices, list)
        if not runtime_was_available:
            self.assertEqual(devices, [])
        for index, device in enumerate(devices):
            self.assertEqual(
                {
                    "ordinal",
                    "compute_capability",
                    "multiprocessor_count",
                    "max_threads_per_block",
                    "compatible",
                    "global_memory_bytes",
                    "memory_bandwidth_bytes_per_second",
                    "identifier",
                    "name",
                },
                set(device),
            )
            # Preserve the exact public values returned by the former
            # per-property implementation while proving that devices() itself
            # used only the single native inventory snapshot asserted above.
            self.assertEqual(
                {
                    "ordinal": _meep._gpu_device_integer_property(index, 0),
                    "compute_capability": (
                        _meep._gpu_device_integer_property(index, 1),
                        _meep._gpu_device_integer_property(index, 2),
                    ),
                    "multiprocessor_count": (
                        _meep._gpu_device_integer_property(index, 3)
                    ),
                    "max_threads_per_block": (
                        _meep._gpu_device_integer_property(index, 4)
                    ),
                    "compatible": bool(
                        _meep._gpu_device_integer_property(index, 5)
                    ),
                    "global_memory_bytes": _meep._gpu_device_memory(index),
                    "memory_bandwidth_bytes_per_second": (
                        _meep._gpu_device_memory_bandwidth(index)
                    ),
                    "identifier": _meep._gpu_device_identifier(index),
                    "name": _meep._gpu_device_name(index),
                },
                device,
            )

        if runtime_was_available:
            with environment_override(
                MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT="1"
            ):
                mp.gpu.reset_statistics()
                legacy_devices = mp.gpu.devices()
                legacy_runtime = mp.gpu.statistics()["runtime"]
            self.assertEqual(devices, legacy_devices)
            self.assertEqual(
                legacy_runtime["runtime_availability_probes"], 1
            )
            self.assertEqual(
                legacy_runtime["runtime_device_enumerations"],
                1 + 10 * len(devices),
            )

    def test_preflighted_fields_pin_backend_until_run_scope_ends(self):
        mp.gpu.set_backend("cpu")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.0),
            resolution=10,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.2, fwidth=0.1),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        simulation.init_sim()
        simulation.fields.gpmeep_begin_backend_operation()
        simulation.fields.step()

        # Reasserting the exact choice is an idempotent no-op. A different
        # backend must fail before resident state or the generation changes.
        mp.gpu.set_backend("cpu")
        with self.assertRaisesRegex(
            RuntimeError, "preflighted fields operation is active"
        ):
            mp.gpu.set_backend("cuda")
        self.assertEqual(mp.gpu.active_backend, "cpu")

        simulation.fields.gpmeep_release_backend_pin()
        try:
            mp.gpu.set_backend("cuda")
        except RuntimeError as error:
            # A CPU-only build or unavailable driver may still reject CUDA,
            # but the released fields owner must no longer be the reason.
            self.assertNotIn(
                "preflighted fields operation is active", str(error)
            )
        else:
            self.assertEqual(mp.gpu.active_backend, "cuda")

        # The high-level Python run path releases this same pin in a finally
        # block even though Simulation intentionally retains its fields.
        mp.gpu.set_backend("cpu")
        simulation.run(until=0.1)
        try:
            mp.gpu.set_backend("cuda")
        except RuntimeError as error:
            self.assertNotIn(
                "preflighted fields operation is active", str(error)
            )
        else:
            self.assertEqual(mp.gpu.active_backend, "cuda")

        # Other high-level Python operations that can enter a native step
        # preflight are operation-scoped as well. Direct low-level fields
        # calls remain independently bracketable with the same begin/release.
        mp.gpu.set_backend("cpu")
        simulation.initialize_field(mp.Ez, lambda _point: 0.0)
        try:
            mp.gpu.set_backend("cuda")
        except RuntimeError as error:
            self.assertNotIn(
                "preflighted fields operation is active", str(error)
            )
        else:
            self.assertEqual(mp.gpu.active_backend, "cuda")

        # Exceptions from a user step callback unwind through the same
        # finally block. The callback is delayed until one native step has
        # completed, proving that a pin was actually established first.
        mp.gpu.set_backend("cpu")
        initial_timestep = simulation.timestep()

        def fail_after_native_step(active_simulation):
            if active_simulation.timestep() > initial_timestep:
                raise RuntimeError("intentional run-scope failure")

        with self.assertRaisesRegex(RuntimeError, "intentional run-scope failure"):
            simulation.run(fail_after_native_step, until=0.1)
        try:
            mp.gpu.set_backend("cuda")
        except RuntimeError as error:
            self.assertNotIn(
                "preflighted fields operation is active", str(error)
            )
        else:
            self.assertEqual(mp.gpu.active_backend, "cuda")

        # The finally block must release the exact owner that began the
        # operation. A callback can reset the Simulation wrapper while an
        # external reference keeps its old native fields owner alive.
        mp.gpu.set_backend("cpu")
        simulation.init_sim()
        retained_fields = []
        initial_timestep = simulation.timestep()

        def reset_wrapper_after_native_step(active_simulation):
            if active_simulation.timestep() > initial_timestep:
                retained_fields.append(active_simulation.fields)
                active_simulation.reset_meep()
                raise RuntimeError("intentional wrapper reset")

        with self.assertRaisesRegex(RuntimeError, "intentional wrapper reset"):
            simulation.run(reset_wrapper_after_native_step, until=0.1)
        self.assertEqual(len(retained_fields), 1)
        try:
            mp.gpu.set_backend("cuda")
        except RuntimeError as error:
            self.assertNotIn(
                "preflighted fields operation is active", str(error)
            )
        else:
            self.assertEqual(mp.gpu.active_backend, "cuda")
        retained_fields.clear()

    def test_sourcedata_python_state_invalidates_acknowledgement(self):
        self.require_cuda()
        backend_policy._reset_for_testing()
        mp.gpu.set_backend("cuda")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.0, 1.0),
            resolution=8,
        )
        source_data = mp.sourcedata()
        source_data.user_scale = 1

        def stopping_condition(_simulation):
            return source_data.user_scale > 0

        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as first:
            backend_policy.guard_run(
                mp,
                simulation,
                stopping_condition,
                None,
                (stopping_condition,),
            )
        mp.gpu.acknowledge_backend_mismatch(
            first.exception.challenge,
            "sourcedata Python state mutation must invalidate acknowledgement",
        )
        source_data.user_scale = 2
        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as changed:
            backend_policy.guard_run(
                mp,
                simulation,
                stopping_condition,
                None,
                (stopping_condition,),
            )
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)

    def test_statistics_have_stable_groups(self):
        mp.gpu.reset_sparse_lorentzian_statistics()
        sparse_before_dispatch_reset = (
            mp.gpu.sparse_lorentzian_statistics()
        )
        mp.gpu.reset_statistics()
        sparse_after_dispatch_reset = mp.gpu.sparse_lorentzian_statistics()
        self.assertEqual(
            sparse_before_dispatch_reset["reset_generation"],
            sparse_after_dispatch_reset["reset_generation"],
        )
        statistics = mp.gpu.statistics()
        self.assertEqual(
            {
                "runtime",
                "dispatch",
                "resident",
                "field_updates",
                "polarizations",
                "sources",
                "boundaries",
                "boundary_execution",
                "resident_point_reads",
                "material_phasing",
                "dfts",
                "dft_batches",
                "dft_reductions",
                "dft_materializations",
                "dft_checkpoints",
                "dft_scales",
                "eigenmode_overlaps",
                "ldos",
                "near2far",
                "multi_gpu",
                "mpi_completion",
                "boundary_eh_overlap",
                "halo_curl_overlap",
                "tile_coalescing",
                "phase_batch_policy",
                "structured_curl_graph",
            },
            set(statistics),
        )
        self.assertEqual(
            {
                "runtime_availability_probes",
                "runtime_device_enumerations",
                "runtime_device_selections",
            },
            set(statistics["runtime"]),
        )
        polarization_keys = {
            "cpu_polarization_calls",
            "cpu_polarization_points",
            "cuda_polarization_calls",
            "cuda_polarization_points",
            "cuda_zero_state_elision_enabled_calls",
            "cuda_zero_state_elision_enabled_points",
            "cuda_zero_state_elision_inspectable_calls",
            "cuda_zero_state_elision_inspectable_points",
        }
        self.assertEqual(polarization_keys, set(statistics["polarizations"]))
        self.assertEqual(
            {key: 0 for key in polarization_keys},
            statistics["polarizations"],
        )
        sparse_lorentzian_keys = {
            "opt_in_calls",
            "opt_in_points",
            "eligible_calls",
            "eligible_points",
            "ineligible_calls",
            "ineligible_points",
            "first_observation_dense_calls",
            "first_observation_dense_points",
            "plan_build_attempts",
            "plan_build_successes",
            "plan_build_failures",
            "plan_hits",
            "plan_invalidations",
            "dense_locked_decisions",
            "accepted_total_points",
            "accepted_active_points",
            "sparse_submission_calls",
            "sparse_submission_points",
            "guard_scan_submissions",
            "guard_scan_submission_points",
            "cpu_fallback_calls",
            "cpu_fallback_points",
            "device_guard_scan_calls",
            "device_guard_scan_points",
            "device_sparse_update_calls",
            "device_sparse_update_points",
            "device_dense_fallback_calls",
            "device_dense_fallback_points",
            "live_plans",
            "live_guards",
            "retained_index_bytes",
            "reset_generation",
            "counter_overflow",
            "teardown_incomplete",
        }
        self.assertEqual(
            sparse_lorentzian_keys,
            set(sparse_after_dispatch_reset),
        )
        self.assertGreaterEqual(
            sparse_after_dispatch_reset["reset_generation"], 1
        )
        self.assertEqual(
            0, sparse_after_dispatch_reset["counter_overflow"]
        )
        self.assertEqual(
            0, sparse_after_dispatch_reset["teardown_incomplete"]
        )
        self.assertEqual(
            {
                "dft_batch_calls",
                "dft_submitted_updates",
                "dft_phase_preparation_launches",
                "dft_phase_reuses",
                "dft_update_kernel_launches",
                "dft_maximum_batch_size",
                "dft_multi_monitor_automatic_checks",
                "dft_multi_monitor_automatic_selected",
                "dft_multi_monitor_automatic_rejected",
                "dft_multi_monitor_forced_batches",
                "dft_multi_monitor_batched_updates",
                "dft_multi_monitor_unbatched_updates",
                "dft_multi_monitor_plan_uploads",
                "dft_multi_monitor_plan_reuses",
                "dft_multi_monitor_metadata_host_to_device_bytes",
            },
            set(statistics["dft_batches"]),
        )
        self.assertEqual(
            {
                "cpu_dft_reduction_calls",
                "cpu_dft_reduction_pairs",
                "cpu_dft_reduction_terms",
                "cuda_dft_reduction_calls",
                "cuda_dft_reduction_pairs",
                "cuda_dft_reduction_terms",
                "cuda_dft_reduction_descriptor_uploads",
                "cuda_dft_reduction_plan_reuses",
                "cuda_dft_reduction_kernel_launches",
                "cuda_dft_reduction_result_device_to_host_bytes",
                "dft_reduction_full_dft_device_to_host_bytes_avoided",
                "dft_reduction_mpi_allreduce_calls",
                "dft_reduction_mpi_allreduce_bytes",
            },
            set(statistics["dft_reductions"]),
        )
        self.assertEqual(
            {
                "cpu_dft_array_materialization_calls",
                "cpu_dft_array_materialization_points",
                "cuda_dft_array_materialization_calls",
                "cuda_dft_array_materialization_points",
                "host_synthetic_material_array_calls",
                "host_synthetic_material_array_points",
                "cpu_dft_output_calls",
                "cpu_dft_output_points",
                "cuda_dft_output_calls",
                "cuda_dft_output_points",
                "cuda_dft_output_staging_calls",
                "cuda_dft_output_staging_points",
                "cuda_dft_output_staging_frequencies",
                "cuda_dft_output_staging_descriptor_uploads",
                "cuda_dft_output_staging_plan_reuses",
                "cuda_dft_output_staging_kernel_launches",
                "cuda_dft_output_staging_result_device_to_host_bytes",
                (
                    "cuda_dft_output_staging_full_dft_"
                    "device_to_host_bytes_avoided"
                ),
                "cuda_dft_output_staging_workspace_ceiling_bytes",
                "cuda_dft_materialization_kernel_launches",
                "cuda_dft_materialization_result_device_to_host_bytes",
                "dft_materialization_full_dft_device_to_host_bytes_avoided",
                "dft_array_mpi_allreduce_calls",
                "dft_array_mpi_allreduce_bytes",
            },
            set(statistics["dft_materializations"]),
        )
        self.assertEqual(
            {
                "cpu_dft_scale_calls",
                "cpu_dft_scale_values",
                "cuda_dft_scale_calls",
                "cuda_dft_scale_values",
                "cuda_dft_scale_kernel_launches",
                "cuda_dft_scale_host_to_device_bytes",
            },
            set(statistics["dft_scales"]),
        )
        self.assertEqual(
            {
                "cpu_eigenmode_overlap_calls",
                "cpu_eigenmode_overlap_terms",
                "cuda_eigenmode_overlap_calls",
                "cuda_eigenmode_overlap_terms",
                "cuda_eigenmode_mode_flux_calls",
                "cuda_eigenmode_mode_mode_calls",
                "cuda_eigenmode_submitted_pairs",
                "cuda_eigenmode_descriptor_uploads",
                "cuda_eigenmode_plan_reuses",
                "cuda_eigenmode_kernel_launches",
                "cuda_eigenmode_result_device_to_host_bytes",
                "eigenmode_full_dft_device_to_host_bytes_avoided",
                "host_mode_profile_sampling_calls",
                "host_mode_profile_sampling_points",
                "eigenmode_zero_rank_channels_skipped",
                "host_mode_profile_host_to_device_bytes",
                "eigenmode_mpi_allreduce_calls",
                "eigenmode_mpi_allreduce_bytes",
            },
            set(statistics["eigenmode_overlaps"]),
        )
        self.assertEqual(
            0,
            statistics["dft_materializations"][
                "host_synthetic_material_array_calls"
            ],
        )
        self.assertEqual(
            0,
            statistics["dft_materializations"][
                "host_synthetic_material_array_points"
            ],
        )
        self.assertEqual(
            {"mpi_waitsome_executions", "mpi_waitall_executions"},
            set(statistics["mpi_completion"]),
        )
        self.assertEqual(
            {
                "cpu_ldos_reduction_calls",
                "cpu_ldos_source_points",
                "cuda_ldos_reduction_calls",
                "cuda_ldos_submitted_profiles",
                "cuda_ldos_source_points",
                "cuda_ldos_descriptor_uploads",
                "cuda_ldos_kernel_launches",
                "cuda_ldos_result_device_to_host_bytes",
                "ldos_full_field_device_to_host_bytes_avoided",
            },
            set(statistics["ldos"]),
        )
        self.assertEqual(
            {
                "cpu_near2far_transform_calls",
                "cpu_near2far_terms",
                "cuda_near2far_transform_calls",
                "cuda_near2far_terms",
                "cuda_near2far_submitted_chunks",
                "cuda_near2far_source_points",
                "cuda_near2far_output_points",
                "cuda_near2far_frequencies",
                "cuda_near2far_periodic_copies",
                "cuda_near2far_fast_precision_calls",
                "cuda_near2far_mixed_precision_calls",
                "cuda_near2far_cancellation_retries",
                "cuda_near2far_target_tiles",
                "cuda_near2far_frequency_tiles",
                "cuda_near2far_operation_tiles",
                "cuda_near2far_maximum_workspace_bytes",
                "cuda_near2far_descriptor_uploads",
                "cuda_near2far_kernel_launches",
                "cuda_near2far_result_device_to_host_bytes",
                "cuda_near2far_condition_device_to_host_bytes",
                "near2far_dft_device_to_host_bytes_avoided",
                "near2far_mpi_allreduce_calls",
                "near2far_mpi_allreduce_bytes",
                "cpu_near2far_adjoint_calls",
                "cpu_near2far_adjoint_terms",
                "cuda_near2far_adjoint_calls",
                "cuda_near2far_adjoint_terms",
                "cuda_near2far_adjoint_submitted_chunks",
                "cuda_near2far_adjoint_source_points",
                "cuda_near2far_adjoint_far_points",
                "cuda_near2far_adjoint_frequencies",
                "cuda_near2far_adjoint_periodic_copies",
                "cuda_near2far_adjoint_fast_precision_calls",
                "cuda_near2far_adjoint_mixed_precision_calls",
                "cuda_near2far_adjoint_cancellation_retries",
                "cuda_near2far_adjoint_maximum_workspace_bytes",
                "cuda_near2far_adjoint_descriptor_uploads",
                "cuda_near2far_adjoint_kernel_launches",
                "cuda_near2far_adjoint_host_to_device_bytes",
                "cuda_near2far_adjoint_result_device_to_host_bytes",
                "cuda_near2far_adjoint_condition_device_to_host_bytes",
            },
            set(statistics["near2far"]),
        )
        self.assertEqual(
            {
                "boundary_eh_overlap_checks",
                "boundary_eh_overlap_eligible",
                "boundary_eh_overlap_launched_h",
                "boundary_eh_overlap_launched_e",
                "boundary_eh_overlap_skipped_disabled",
                "boundary_eh_overlap_skipped_unsupported_schedule",
                "boundary_eh_overlap_skipped_no_remote",
                "boundary_eh_overlap_skipped_cold_topology",
                "boundary_eh_overlap_rejected",
            },
            set(statistics["boundary_eh_overlap"]),
        )
        self.assertEqual(
            {
                "halo_curl_overlap_checks",
                "halo_curl_overlap_eligible",
                "halo_curl_overlap_launches",
                "halo_curl_overlap_skipped_disabled",
                "halo_curl_overlap_skipped_unsupported_schedule",
                "halo_curl_overlap_skipped_no_remote",
                "halo_curl_overlap_skipped_cold_topology",
                "halo_curl_overlap_rejected_feature",
                "halo_curl_overlap_rejected_small",
                "halo_curl_overlap_full_points",
                "halo_curl_overlap_interior_points",
                "halo_curl_overlap_shell_points",
            },
            set(statistics["halo_curl_overlap"]),
        )
        self.assertEqual(
            {
                "tile_coalesced_curl_chunk_phases",
                "tile_coalesced_curl_input_tiles",
                "tile_coalesced_update_eh_chunk_phases",
                "tile_coalesced_update_eh_input_tiles",
            },
            set(statistics["tile_coalescing"]),
        )
        self.assertEqual(
            {
                "phase_curl_automatic_checks",
                "phase_curl_automatic_selected",
                "phase_curl_automatic_rejected",
                "phase_curl_forced_batches",
                "phase_curl_batched_operations",
                "phase_curl_unbatched_operations",
                "phase_curl_replay_checks",
                "phase_curl_replay_hits",
                "phase_curl_replay_unready",
                "phase_curl_replay_generation_misses",
                "phase_curl_replay_mirror_misses",
                "phase_update_eh_automatic_checks",
                "phase_update_eh_automatic_selected",
                "phase_update_eh_automatic_rejected",
                "phase_update_eh_forced_batches",
                "phase_update_eh_batched_operations",
                "phase_update_eh_unbatched_operations",
            },
            set(statistics["phase_batch_policy"]),
        )
        self.assertEqual(
            {
                "structured_curl_graph_collection_checks",
                "structured_curl_graph_collection_launches",
                "structured_curl_graph_plan_creations",
                "structured_curl_graph_plan_rebuilds",
                "structured_curl_graph_plan_reuses",
                "structured_curl_graph_plan_destroys",
                "structured_curl_graph_plan_destroy_failures",
                "structured_curl_graph_live_plans",
                "structured_curl_graph_quarantined_plans",
                "structured_curl_graph_logical_operations",
                "structured_curl_graph_unsupported_fallbacks",
                "structured_curl_graph_creation_fallbacks",
                "structured_curl_graph_launch_failures",
                "structured_curl_graph_replay_checks",
                "structured_curl_graph_replay_launches",
                "structured_curl_graph_replay_unready",
                "structured_curl_graph_replay_generation_misses",
                "structured_curl_graph_replay_mirror_misses",
            },
            set(statistics["structured_curl_graph"]),
        )
        self.assertTrue(
            all(
                isinstance(value, int)
                for group in statistics.values()
                for value in group.values()
            )
        )

    def test_internal_mpi_overlap_state_is_not_python_api(self):
        internal_names = (
            "comms_overlap_statistics",
            "comms_supports_cuda_device_buffers",
            "comms_start_cuda_device_receives",
            "comms_start_cuda_device_sends",
            "reset_comms_overlap_statistics",
            "get_comms_overlap_statistics",
            "comms_physical_message_count",
            "comms_finish",
        )
        for name in internal_names:
            self.assertFalse(hasattr(_meep, name), name)

    def test_zero_state_elision_statistics_follow_opt_in(self):
        self.require_cuda()

        def run_dispersive(enable, disable):
            with environment_override(
                MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION=enable,
                MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION=disable,
            ):
                mp.gpu.reset_statistics()
                mp.gpu.set_backend("cuda")
                medium = mp.Medium(
                    epsilon=1.5,
                    E_susceptibilities=[
                        mp.DrudeSusceptibility(
                            frequency=0.31, gamma=0.07, sigma=0.2
                        )
                    ],
                )
                simulation = mp.Simulation(
                    cell_size=mp.Vector3(1.2, 1.0),
                    resolution=10,
                    default_material=medium,
                    sources=[
                        mp.Source(
                            mp.GaussianSource(0.3, fwidth=0.1),
                            component=mp.Ez,
                            center=mp.Vector3(),
                        )
                    ],
                )
                run_with_explicit_qualification_acknowledgement(
                    simulation, until=0.5
                )
                statistics = mp.gpu.statistics()["polarizations"]
                simulation.reset_meep()
                return statistics

        enabled = run_dispersive("1", None)
        self.assertGreater(enabled["cuda_polarization_calls"], 0)
        self.assertEqual(
            enabled["cuda_polarization_calls"],
            enabled["cuda_zero_state_elision_enabled_calls"],
        )
        self.assertEqual(
            enabled["cuda_polarization_points"],
            enabled["cuda_zero_state_elision_enabled_points"],
        )
        self.assertEqual(
            enabled["cuda_polarization_calls"],
            enabled["cuda_zero_state_elision_inspectable_calls"],
        )
        self.assertEqual(
            enabled["cuda_polarization_points"],
            enabled["cuda_zero_state_elision_inspectable_points"],
        )

        disabled = run_dispersive("1", "1")
        self.assertGreater(disabled["cuda_polarization_calls"], 0)
        for key in (
            "cuda_zero_state_elision_enabled_calls",
            "cuda_zero_state_elision_enabled_points",
            "cuda_zero_state_elision_inspectable_calls",
            "cuda_zero_state_elision_inspectable_points",
        ):
            self.assertEqual(disabled[key], 0)

    def test_required_cuda_sparse_lorentzian_production_path(self):
        self.require_cuda()
        cpu_field, _ = self._run_sparse_public_simulation(
            "cpu", "lorentzian"
        )
        cuda_field, sparse = self._run_sparse_public_simulation(
            "cuda", "lorentzian"
        )
        np.testing.assert_allclose(
            cuda_field, cpu_field, rtol=3e-4, atol=3e-6
        )
        self.assertGreater(sparse["plan_build_successes"], 0)
        self.assertGreater(sparse["sparse_submission_calls"], 0)
        self.assertGreater(sparse["device_guard_scan_calls"], 0)
        self.assertGreater(sparse["device_sparse_update_calls"], 0)
        self.assertEqual(sparse["device_dense_fallback_calls"], 0)
        self.assertEqual(sparse["cpu_fallback_calls"], 0)
        self.assertGreater(sparse["accepted_total_points"], 0)
        self.assertLessEqual(
            100 * sparse["accepted_active_points"],
            35 * sparse["accepted_total_points"],
        )

    def test_required_cuda_sparse_drude_increment_production_path(self):
        self.require_cuda()
        cpu_field, _ = self._run_sparse_public_simulation("cpu", "drude")
        cuda_field, sparse = self._run_sparse_public_simulation(
            "cuda", "drude"
        )
        np.testing.assert_allclose(
            cuda_field, cpu_field, rtol=3e-4, atol=3e-6
        )
        self.assertGreater(sparse["plan_build_successes"], 0)
        self.assertGreater(sparse["plan_hits"], 0)
        self.assertGreater(sparse["sparse_submission_calls"], 0)
        self.assertGreater(sparse["device_sparse_update_calls"], 0)
        self.assertEqual(sparse["device_dense_fallback_calls"], 0)
        self.assertEqual(sparse["cpu_fallback_calls"], 0)

    def test_automatic_backend_selection_is_rejected(self):
        with environment_override(
            MEEP_GPU_DEVICE=None,
            MEEP_GPU_ALLOW_OVERSUBSCRIBE=None,
            MEEP_GPU_AUTO_MIN_CELLS=None,
        ):
            mp.gpu.reset_statistics()
            mp.gpu.set_backend("cpu")
            with self.assertRaisesRegex(ValueError, "automatic selection is disabled"):
                mp.gpu.set_backend("auto")

            statistics = mp.gpu.statistics()
            self.assertEqual(mp.gpu.requested_backend, "cpu")
            self.assertEqual(mp.gpu.active_backend, "cpu")
            self.assertEqual(
                {
                    "runtime_availability_probes": 0,
                    "runtime_device_enumerations": 0,
                    "runtime_device_selections": 0,
                },
                statistics["runtime"],
            )
            self.assertEqual(statistics["dispatch"]["cpu_curl_calls"], 0)
            self.assertEqual(statistics["dispatch"]["cuda_curl_calls"], 0)

    def test_low_level_automatic_backend_token_is_rejected(self):
        with environment_override(
            MEEP_GPU_DEVICE=None,
            MEEP_GPU_ALLOW_OVERSUBSCRIBE=None,
            MEEP_GPU_AUTO_MIN_CELLS=None,
        ):
            mp.gpu.reset_statistics()
            mp.gpu.set_backend("cpu")
            with self.assertRaisesRegex(RuntimeError, "automatic backend selection is disabled"):
                mp._gpu_set_backend(1)
            with self.assertRaisesRegex(
                RuntimeError,
                "backend must be 'cpu' or 'cuda'; automatic selection is disabled",
            ):
                mp._gpu_set_backend(99)
            statistics = mp.gpu.statistics()
            self.assertEqual(mp.gpu.requested_backend, "cpu")
            self.assertEqual(mp.gpu.active_backend, "cpu")
            self.assertEqual(statistics["runtime"]["runtime_availability_probes"], 0)
            self.assertEqual(statistics["runtime"]["runtime_device_enumerations"], 0)
            self.assertEqual(statistics["runtime"]["runtime_device_selections"], 0)
            self.assertEqual(statistics["dispatch"]["cuda_curl_calls"], 0)
            self.assertEqual(statistics["dispatch"]["cpu_curl_calls"], 0)

    def test_boundary_execution_statistics_cpu_backend(self):
        mp.gpu.set_backend("cpu")
        mp.gpu.reset_statistics()
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.6, 1.4), resolution=10,
            k_point=mp.Vector3(0.12, 0.07),
            sources=[mp.Source(mp.ContinuousSource(0.3), component=mp.Ez,
                               center=mp.Vector3())],
        )
        try:
            run_with_explicit_qualification_acknowledgement(simulation, until=0.5)
            simulation.get_field_point(mp.Ez, mp.Vector3())
            statistics = mp.gpu.statistics()
            self.assertEqual(statistics["material_phasing"],
                             dict.fromkeys(("checks", "reused", "disabled", "ineligible"), 0))
            self.assertEqual(len(statistics["resident_point_reads"]), 7)
            self.assertTrue(all(value == 0 for value in statistics["resident_point_reads"].values()))
            self.assertGreater(statistics["boundaries"]["cpu_boundary_calls"], 0)
            self.assertEqual(statistics["boundaries"]["cuda_boundary_calls"], 0)
            self.assertEqual(len(statistics["boundary_execution"]), 20)
            self.assertTrue(all(value == 0 for value in statistics["boundary_execution"].values()))
            mp.gpu.reset_statistics()
            self.assertTrue(all(
                value == 0 for value in mp.gpu.statistics()["boundary_execution"].values()
            ))
        finally:
            simulation.reset_meep()

    def test_required_cuda_boundary_replay_observations(self):
        self.require_cuda()
        if mp.count_processors() != 1:
            self.skipTest("local boundary replay requires a single MPI rank")
        outcomes = (
            "local_disabled", "local_unready", "local_prepare_invalidations",
            "local_replay_empty", "local_replay_work", "local_failures",
        )

        def run(backend, disabled=False, kernel="default", span_cache_disabled=False):
            with environment_override(
                MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY="1" if disabled else None,
                MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE="1" if span_cache_disabled else None,
                MEEP_GPU_DISABLE_BOUNDARY_PLAN_CACHE=None,
                MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO=(
                    "1" if kernel != "default" else None
                ),
                MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY=(
                    "1" if kernel != "default" else None
                ),
                MEEP_GPU_DISABLE_BOUNDARY_GRAPH="1" if kernel == "staged" else None,
            ):
                mp.gpu.set_backend(backend)
                mp.gpu.reset_statistics()
                self.assertTrue(all(
                    value == 0 for value in mp.gpu.statistics()["boundary_execution"].values()
                ))
                simulation = mp.Simulation(
                    cell_size=mp.Vector3(1.6, 1.4),
                    resolution=10,
                    k_point=mp.Vector3(0.12, 0.07),
                    sources=[mp.Source(
                        mp.ContinuousSource(0.3), component=mp.Ez,
                        center=mp.Vector3(),
                    )],
                )
                try:
                    run_with_explicit_qualification_acknowledgement(simulation, until=2)
                    statistics = mp.gpu.statistics()
                    counters = statistics["boundary_execution"]
                    field = simulation.get_array(component=mp.Ez).copy()
                    self.assertTrue(np.isfinite(field).all())
                finally:
                    simulation.reset_meep()
                if backend == "cpu":
                    self.assertTrue(all(value == 0 for value in counters.values()))
                else:
                    self.assertEqual(counters["local_checks"], sum(counters[n] for n in outcomes))
                    self.assertEqual(counters["local_failures"], 0)
                    self.assertEqual(counters["prepare_failures"], 0)
                    self.assertGreater(counters["prepare_calls"], 0)
                    self.assertGreater(counters["prepared_arrays"], 0)
                    if span_cache_disabled:
                        self.assertGreater(counters["span_cache_bypasses"], 0)
                        self.assertEqual(counters["span_cache_hits"], 0)
                        self.assertEqual(counters["span_cache_misses"], 0)
                    else:
                        self.assertGreater(counters["span_cache_hits"], 0)
                        self.assertGreater(counters["span_cache_misses"], 0)
                        self.assertEqual(counters["span_cache_bypasses"], 0)
                    self.assertGreater(counters["prepare_host_nanoseconds"], 0)
                    self.assertEqual(
                        counters["local_descriptor_rebuild_attempts"],
                        counters["local_disabled"] + counters["local_unready"]
                        + counters["local_prepare_invalidations"],
                    )
                    self.assertEqual(statistics["boundaries"]["cpu_boundary_calls"], 0)
                    self.assertGreater(statistics["boundaries"]["cuda_boundary_calls"], 0)
                    if disabled:
                        self.assertGreater(counters["local_disabled"], 0)
                        self.assertEqual(counters["local_replay_work"], 0)
                        self.assertEqual(counters["local_replay_empty"], 0)
                    else:
                        self.assertEqual(counters["local_disabled"], 0)
                        self.assertGreater(counters["local_replay_work"], 0)
                    if kernel in {"graph", "staged"}:
                        self.assertEqual(counters["plan_direct_zero"], 0)
                        self.assertEqual(counters["plan_direct_local"], 0)
                        self.assertGreater(counters["plan_" + kernel], 0)
                    if kernel == "staged":
                        self.assertEqual(counters["plan_graph"], 0)
                mp.gpu.reset_statistics()
                self.assertTrue(all(
                    value == 0 for value in mp.gpu.statistics()["boundary_execution"].values()
                ))
                return field

        cpu = run("cpu")
        for disabled, kernel, span_cache_disabled in (
            (False, "default", False), (False, "default", True),
            (True, "default", False), (False, "graph", False), (False, "staged", False),
        ):
            with self.subTest(disabled=disabled, kernel=kernel,
                              span_cache_disabled=span_cache_disabled):
                field = run("cuda", disabled, kernel, span_cache_disabled)
                self.assertGreater(float(np.linalg.norm(field)), 0)
                np.testing.assert_allclose(field, cpu, rtol=3e-4, atol=3e-6)

    def test_required_cuda_python_step(self):
        self.require_cuda()
        compatible = [device for device in mp.gpu.devices() if device["compatible"]]

        # In an MPI build the runtime's node-local-rank policy must choose a
        # distinct device for each rank.  Explicitly selecting ordinal zero on
        # every rank defeats that policy and correctly trips the physical-GPU
        # claim table.  The singleton test still covers the public selector;
        # dedicated MPI tests cover default and explicit per-rank mappings.
        if mp.count_processors() == 1:
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.6, 1.4),
            resolution=10,
            boundary_layers=[mp.PML(0.2)],
            sources=[
                mp.Source(
                    mp.ContinuousSource(0.3),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        run_with_explicit_qualification_acknowledgement(simulation, until=2)

        statistics = mp.gpu.statistics()
        self.assertEqual(mp.gpu.active_backend, "cuda")
        self.assertGreater(statistics["dispatch"]["cuda_curl_calls"], 0)
        self.assertEqual(statistics["dispatch"]["cpu_curl_calls"], 0)
        self.assertGreater(statistics["field_updates"]["cuda_update_eh_calls"], 0)
        self.assertEqual(statistics["field_updates"]["cpu_update_eh_calls"], 0)
        # A point source is owned by one MPI rank, while every rank owns curl
        # work.  The distributed C++ suite aggregates and proves nonzero
        # source dispatch; this public-binding test must not demand the same
        # rank-local source count from a peer that owns no source point.
        if mp.count_processors() == 1:
            self.assertGreater(statistics["sources"]["cuda_source_calls"], 0)
        self.assertEqual(statistics["sources"]["cpu_source_calls"], 0)
        self.assertTrue(simulation.fields.gpu_cuda_execution_selected())
        self.assertIn("CUDA", simulation.fields.gpu_execution_diagnostic())

    def test_required_cuda_resident_dft_scale_statistics(self):
        self.require_cuda()
        compatible = [device for device in mp.gpu.devices() if device["compatible"]]
        if mp.count_processors() == 1:
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")

        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.6, 1.4),
            resolution=10,
            sources=[
                mp.Source(
                    mp.ContinuousSource(0.3),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        flux = simulation.add_flux(
            0.3,
            0.0,
            1,
            mp.FluxRegion(center=mp.Vector3(0.2), size=mp.Vector3(0, 0.8)),
            decimation_factor=1,
        )
        run_with_explicit_qualification_acknowledgement(simulation, until=1)

        mp.gpu.reset_statistics()
        flux.scale_dfts(complex(-0.25, 0.125))
        statistics = mp.gpu.statistics()["dft_scales"]
        from mpi4py import MPI

        # The monitor plane can be owned entirely by one MPI rank.  Statistics
        # are deliberately process-local, so require zero fallback locally but
        # evaluate positive CUDA work over the distributed simulation instead
        # of manufacturing a no-op kernel on a rank with no DFT allocation.
        rank_statistics = MPI.COMM_WORLD.allgather(statistics)
        for rank_statistic in rank_statistics:
            self.assertEqual(rank_statistic["cpu_dft_scale_calls"], 0)
            self.assertEqual(rank_statistic["cpu_dft_scale_values"], 0)
            self.assertEqual(
                rank_statistic["cuda_dft_scale_host_to_device_bytes"], 0
            )
            if rank_statistic["cuda_dft_scale_calls"] == 0:
                self.assertEqual(rank_statistic["cuda_dft_scale_values"], 0)
                self.assertEqual(
                    rank_statistic["cuda_dft_scale_kernel_launches"], 0
                )
        global_statistics = {
            key: sum(rank_statistic[key] for rank_statistic in rank_statistics)
            for key in statistics
        }
        self.assertGreater(global_statistics["cuda_dft_scale_calls"], 0)
        self.assertGreater(global_statistics["cuda_dft_scale_values"], 0)
        self.assertGreater(global_statistics["cuda_dft_scale_kernel_launches"], 0)

    def test_required_cuda_centered_grid_unequal_dft_reduction(self):
        """Centered Yee monitors may store more H than E points."""
        self.require_cuda()
        from mpi4py import MPI

        def make_simulation():
            simulation = mp.Simulation(
                cell_size=mp.Vector3(2.0, 2.0),
                resolution=16,
                sources=[
                    mp.Source(
                        mp.ContinuousSource(0.3),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            monitor = simulation.add_mode_monitor(
                [0.27, 0.33],
                mp.ModeRegion(
                    center=mp.Vector3(0.5, 0.0),
                    size=mp.Vector3(0.0, 1.2),
                ),
                yee_grid=True,
                decimation_factor=1,
            )
            return simulation, monitor

        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        simulation, monitor = make_simulation()
        run_with_explicit_qualification_acknowledgement(
            simulation, until=2.0
        )

        electric = monitor.E
        magnetic = monitor.H
        local_unequal_pairs = 0
        local_pairs = 0
        while electric and magnetic:
            self.assertLessEqual(electric.N, magnetic.N)
            local_unequal_pairs += int(electric.N != magnetic.N)
            local_pairs += 1
            electric = electric.next_in_dft
            magnetic = magnetic.next_in_dft
        self.assertFalse(bool(electric) or bool(magnetic))
        self.assertGreater(
            MPI.COMM_WORLD.allreduce(local_pairs, op=MPI.SUM), 0
        )
        self.assertGreater(
            MPI.COMM_WORLD.allreduce(local_unequal_pairs, op=MPI.SUM), 0
        )

        values = np.asarray(mp.get_fluxes(monitor))
        statistics = mp.gpu.statistics()
        reductions = statistics["dft_reductions"]

        def global_sum(value):
            return MPI.COMM_WORLD.allreduce(value, op=MPI.SUM)

        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(mp.gpu.active_backend, "cuda")
        self.assertTrue(simulation.fields.gpu_cuda_execution_selected())
        cuda_reduction_calls = global_sum(
            reductions["cuda_dft_reduction_calls"]
        )
        self.assertGreater(cuda_reduction_calls, 0)
        self.assertGreater(
            global_sum(reductions["cuda_dft_reduction_pairs"]), 0
        )
        self.assertGreater(
            global_sum(reductions["cuda_dft_reduction_terms"]), 0
        )
        self.assertEqual(
            global_sum(reductions["cuda_dft_reduction_kernel_launches"]),
            2 * cuda_reduction_calls,
        )
        self.assertGreater(
            global_sum(
                reductions[
                    "cuda_dft_reduction_result_device_to_host_bytes"
                ]
            ),
            0,
        )
        self.assertEqual(
            global_sum(reductions["cpu_dft_reduction_calls"]), 0
        )
        self.assertGreater(
            global_sum(statistics["dfts"]["cuda_dft_calls"]), 0
        )
        self.assertEqual(
            global_sum(statistics["dfts"]["cpu_dft_calls"]), 0
        )
        simulation.reset_meep()

        mp.gpu.set_backend("cpu")
        mp.gpu.reset_statistics()
        cpu_simulation, cpu_monitor = make_simulation()
        run_with_explicit_qualification_acknowledgement(
            cpu_simulation, until=2.0
        )
        cpu_values = np.asarray(mp.get_fluxes(cpu_monitor))
        self.assertGreater(np.max(np.abs(cpu_values)), 1.0e-3)
        np.testing.assert_allclose(
            values, cpu_values, rtol=3.0e-4, atol=1.0e-8
        )
        cpu_reductions = mp.gpu.statistics()["dft_reductions"]
        self.assertGreater(
            global_sum(cpu_reductions["cpu_dft_reduction_calls"]), 0
        )
        self.assertEqual(
            global_sum(cpu_reductions["cuda_dft_reduction_calls"]), 0
        )
        cpu_simulation.reset_meep()

    def test_initialized_native_step_invalidates_acknowledgement(self):
        self.require_cuda()
        backend_policy._reset_for_testing()
        # This test exercises the strict exact-per-attempt diagnostic path.
        # Normal production reuse is intentionally owned by one Simulation,
        # so a same-object repeat would otherwise retain the user's explicit
        # backend intent and bypass the exact native-state challenge.
        cache_control = environment_override(
            MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE="1"
        )
        cache_control.__enter__()
        self.addCleanup(cache_control.__exit__, None, None, None)
        mp.gpu.set_backend("cuda")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.0, 1.0),
            resolution=8,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.3, fwidth=0.1),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        monitor = simulation.add_flux(
            0.3,
            0.0,
            1,
            mp.FluxRegion(center=mp.Vector3(0.2), size=mp.Vector3(0, 0.5)),
        )
        simulation.init_sim()
        simulation._evaluate_dft_objects()
        initial_manifest = json.loads(
            simulation.fields.gpmeep_state_manifest()
        )
        self.assertEqual(initial_manifest["schema"], "gpmeep-fields-state-v2")

        arguments = (mp, simulation, 0.1, None, (), {"entrypoint": "run"})
        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as first:
            backend_policy.guard_run(*arguments)
        mp.gpu.acknowledge_backend_mismatch(
            first.exception.challenge,
            "unchanged initialized native state is intentional qualification",
        )
        accepted = backend_policy.guard_run(*arguments)
        self.assertTrue(accepted["acknowledgement"])

        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as pending:
            backend_policy.guard_run(*arguments)
        mp.gpu.acknowledge_backend_mismatch(
            pending.exception.challenge,
            "native step mutation must invalidate this exact acknowledgement",
        )
        simulation.fields.step()
        stepped_manifest = json.loads(
            simulation.fields.gpmeep_state_manifest()
        )
        self.assertGreater(stepped_manifest["epoch"], initial_manifest["epoch"])
        self.assertGreater(stepped_manifest["t"], initial_manifest["t"])
        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as changed:
            backend_policy.guard_run(*arguments)
        self.assertNotEqual(
            pending.exception.challenge, changed.exception.challenge
        )

        mp.gpu.acknowledge_backend_mismatch(
            changed.exception.challenge,
            "native structure mutation must invalidate this acknowledgement",
        )
        structure_before = json.loads(
            simulation.structure.gpmeep_state_manifest()
        )
        simulation.set_epsilon(lambda _point: 2.0)
        structure_after = json.loads(
            simulation.structure.gpmeep_state_manifest()
        )
        self.assertGreater(structure_after["epoch"], structure_before["epoch"])
        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as material:
            backend_policy.guard_run(*arguments)
        self.assertNotEqual(changed.exception.challenge, material.exception.challenge)

        mp.gpu.acknowledge_backend_mismatch(
            material.exception.challenge,
            "native DFT scale must invalidate this exact acknowledgement",
        )
        epoch_before_scale = json.loads(
            simulation.fields.gpmeep_state_manifest()
        )["epoch"]
        monitor.scale_dfts(2.0)
        epoch_after_scale = json.loads(
            simulation.fields.gpmeep_state_manifest()
        )["epoch"]
        # A distributed monitor has DFT chunks only on the ranks intersecting
        # its volume.  The backend policy binds the ordered native-state
        # digest from every rank, so a direct monitor mutation is exact when
        # at least one owning rank advances; empty ranks are not required to
        # invent a local accumulator mutation.
        epoch_records = [
            json.loads(record)
            for record in mp._gpu_collective_rank_records(
                json.dumps(
                    {
                        "before": epoch_before_scale,
                        "after": epoch_after_scale,
                    },
                    sort_keys=True,
                )
            )
        ]
        self.assertTrue(
            any(
                record["after"] > record["before"]
                for record in epoch_records
            )
        )
        with self.assertRaises(backend_policy.BackendRecommendationMismatch) as dft:
            backend_policy.guard_run(*arguments)
        self.assertNotEqual(material.exception.challenge, dft.exception.challenge)
        simulation.reset_meep()

    def test_initialized_public_native_mutators_invalidate_acknowledgement(self):
        self.require_cuda()
        backend_policy._reset_for_testing()
        mp.gpu.set_backend("cuda")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.0, 1.0),
            resolution=8,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.3, fwidth=0.1),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        monitor = simulation.add_flux(
            0.3,
            0.0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0.2), size=mp.Vector3(0, 0.5)
            ),
        )
        simulation.init_sim()
        simulation._evaluate_dft_objects()
        arguments = (mp, simulation, 0.1, None, (), {"entrypoint": "run"})

        def assert_invalidates(label, mutation, *, require_epoch=True):
            with self.assertRaises(
                backend_policy.BackendRecommendationMismatch
            ) as pending:
                backend_policy.guard_run(*arguments)
            mp.gpu.acknowledge_backend_mismatch(
                pending.exception.challenge,
                f"{label} direct native mutation is intentional qualification",
            )
            epoch_before = json.loads(
                simulation.fields.gpmeep_state_manifest()
            )["epoch"]
            result = mutation()
            epoch_after = json.loads(
                simulation.fields.gpmeep_state_manifest()
            )["epoch"]
            epoch_records = [
                json.loads(record)
                for record in mp._gpu_collective_rank_records(
                    json.dumps(
                        {
                            "before": epoch_before,
                            "after": epoch_after,
                        },
                        sort_keys=True,
                    )
                )
            ]
            if require_epoch:
                self.assertTrue(
                    any(
                        record["after"] > record["before"]
                        for record in epoch_records
                    ),
                    label,
                )
            with self.assertRaises(
                backend_policy.BackendRecommendationMismatch
            ) as changed:
                backend_policy.guard_run(*arguments)
            self.assertNotEqual(
                pending.exception.challenge,
                changed.exception.challenge,
                label,
            )
            return result

        epoch_before_hidden_constructor = json.loads(
            simulation.fields.gpmeep_state_manifest()
        )["epoch"]
        with self.assertRaises(AttributeError):
            mp.flux_vol(
                simulation.fields,
                mp.X,
                simulation.fields.total_volume(),
            )
        self.assertEqual(
            json.loads(simulation.fields.gpmeep_state_manifest())["epoch"],
            epoch_before_hidden_constructor,
        )

        legacy = assert_invalidates(
            "fields.add_flux_vol",
            lambda: simulation.fields.add_flux_vol(
                mp.X, simulation.fields.total_volume()
            ),
        )
        assert_invalidates("flux_vol.update_half", legacy.update_half)
        assert_invalidates("flux_vol.update", legacy.update)
        assert_invalidates(
            "fields.step_boundaries",
            lambda: simulation.fields.step_boundaries(mp.E_stuff),
        )
        assert_invalidates(
            "fields.require_source_components",
            simulation.fields.require_source_components,
        )
        assert_invalidates(
            "dft_chunk.update_dft",
            lambda: (
                monitor.E.update_dft(simulation.fields.time())
                if monitor.E is not None
                else None
            ),
        )
        native_source = simulation.fields.sources
        with self.assertRaises(AttributeError):
            native_source.next = native_source
        self.assertFalse(
            hasattr(mp._meep, "src_time_next_set"),
            "raw native source ownership-link replacement must not be exported",
        )
        assert_invalidates(
            "src_time.set_frequency",
            lambda: native_source.set_frequency(0.41),
            require_epoch=False,
        )
        assert_invalidates(
            "src_time.set_fwidth",
            lambda: native_source.set_fwidth(0.13),
            require_epoch=False,
        )
        assert_invalidates(
            "src_time.update",
            lambda: native_source.update(
                simulation.fields.time() + simulation.fields.dt,
                simulation.fields.dt,
            ),
            require_epoch=False,
        )
        assert_invalidates(
            "src_time.is_integrated",
            lambda: setattr(
                native_source,
                "is_integrated",
                not native_source.is_integrated,
            ),
            require_epoch=False,
        )

        callback_state = {"amplitude": 1.0}

        def direct_native_callback(_time):
            return complex(callback_state["amplitude"], 0.0)

        direct_native_source = mp.custom_py_src_time(
            direct_native_callback,
            0.0,
            100.0,
            0.3,
            0.1,
        )
        simulation.fields.add_point_source(
            mp.Ez,
            direct_native_source,
            mp.vec(0.0, 0.0),
        )
        assert_invalidates(
            "custom_py_src_time callback closure",
            lambda: callback_state.__setitem__("amplitude", 7.0),
            require_epoch=False,
        )

        with self.assertRaises(AttributeError):
            monitor.E.scale = 2.0 + 0.0j
        with self.assertRaises(AttributeError):
            monitor.swigobj.E = None
        with self.assertRaises(AttributeError):
            legacy.next = None
        with self.assertRaises(AttributeError):
            simulation.fields.S = mp.identity()
        with self.assertRaises(AttributeError):
            simulation.structure.Courant = 0.25
        with self.assertRaises(AttributeError):
            simulation.structure.effort = simulation.structure.effort
        self.assertFalse(
            hasattr(mp._meep, "structure_effort_set"),
            "raw structure effort replacement must not be exported",
        )
        self.assertFalse(
            hasattr(mp.fields, "process_incoming_chunk_data"),
            "internal MPI halo transport must not be callable from Python",
        )
        simulation.reset_meep()

    def test_required_cuda_cylindrical_cartesian_near2far(self):
        self.require_cuda()
        import meep.adjoint as mpa

        mp.gpu.set_backend("cuda")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.2, 0.0, 1.2),
            resolution=9,
            dimensions=mp.CYLINDRICAL,
            m=2,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.31, fwidth=0.12),
                    component=mp.Ep,
                    center=mp.Vector3(0.35, 0.0, -0.12),
                )
            ],
        )
        regions = [
            mp.Near2FarRegion(
                center=mp.Vector3(0.38, 0.0, 0.32),
                size=mp.Vector3(0.76, 0.0, 0.0),
                weight=1.0,
            )
        ]
        observation_points = [
            mp.Vector3(-2.2, 1.7, -0.13),
            mp.Vector3(2.4, -1.1, 0.29),
        ]
        far_fields = mpa.Near2FarFields(
            simulation,
            regions,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        near2far = far_fields.register_monitors(np.array([0.31]))
        run_with_explicit_qualification_acknowledgement(
            simulation, until=12.0
        )

        mp.gpu.reset_statistics()
        cuda_batch = simulation.get_farfields_points(
            near2far,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        cuda_scalars = np.asarray(
            [
                simulation.get_farfield(
                    near2far,
                    point,
                    greencyl_tol=1.0e-5,
                    cartesian=True,
                )
                for point in observation_points
            ]
        ).reshape(cuda_batch.shape)
        cuda_grid = simulation.get_farfields(
            near2far,
            resolution=5.0,
            center=mp.Vector3(-2.2, 1.5, 0.0),
            size=mp.Vector3(0.5, 0.5, 0.5),
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        objective_cuda = far_fields()
        cuda_statistics = mp.gpu.statistics()["near2far"]

        np.testing.assert_allclose(
            cuda_scalars, cuda_batch, rtol=5.0e-6, atol=5.0e-8
        )
        np.testing.assert_allclose(
            objective_cuda, cuda_batch, rtol=5.0e-6, atol=5.0e-8
        )
        self.assertEqual(cuda_batch.shape, (2, 1, 6))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            self.assertEqual(np.asarray(cuda_grid[component]).size, 8)
        self.assertGreater(
            cuda_statistics["cuda_near2far_transform_calls"]
            + cuda_statistics["near2far_mpi_allreduce_calls"],
            0,
        )
        self.assertEqual(cuda_statistics["cpu_near2far_transform_calls"], 0)

        mp.gpu.set_backend("cpu")
        mp.gpu.reset_statistics()
        cpu_batch = simulation.get_farfields_points(
            near2far,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        cpu_grid = simulation.get_farfields(
            near2far,
            resolution=5.0,
            center=mp.Vector3(-2.2, 1.5, 0.0),
            size=mp.Vector3(0.5, 0.5, 0.5),
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        legacy = np.asarray(
            simulation.get_farfield(
                near2far,
                observation_points[1],
                greencyl_tol=1.0e-5,
            )
        )
        cpu_statistics = mp.gpu.statistics()["near2far"]
        np.testing.assert_allclose(
            cuda_batch, cpu_batch, rtol=4.0e-3, atol=3.0e-8
        )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_allclose(
                cuda_grid[component],
                cpu_grid[component],
                rtol=4.0e-3,
                atol=3.0e-8,
            )
        self.assertGreater(
            np.max(np.abs(legacy - cpu_batch[1, 0])),
            1.0e-7 * max(1.0, np.max(np.abs(cpu_batch[1, 0]))),
        )
        self.assertGreater(cpu_statistics["cpu_near2far_transform_calls"], 0)
        self.assertEqual(cpu_statistics["cuda_near2far_transform_calls"], 0)

        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        gradient = np.empty((2, 1, 6), dtype=np.complex128)
        for index in range(gradient.size):
            gradient.flat[index] = 0.11 + 0.003j * index
        adjoint_sources = far_fields.place_adjoint_source(gradient)
        adjoint_statistics = mp.gpu.statistics()["near2far"]
        if adjoint_sources:
            self.assertEqual(
                adjoint_statistics["cuda_near2far_adjoint_calls"], 1
            )
            self.assertGreater(
                adjoint_statistics["cuda_near2far_adjoint_kernel_launches"], 0
            )
        else:
            self.assertEqual(
                adjoint_statistics["cuda_near2far_adjoint_calls"], 0
            )
        self.assertEqual(adjoint_statistics["cpu_near2far_adjoint_calls"], 0)
        simulation.reset_meep()


if __name__ == "__main__":
    unittest.main()
