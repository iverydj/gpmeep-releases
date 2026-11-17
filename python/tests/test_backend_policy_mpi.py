import inspect
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import meep as mp
import meep.simulation as simulation_module
import numpy as np
from meep import backend_policy as policy


_RANK_CALLBACK_STATE = {"value": 0}
_RANK_DESCRIPTOR_STATE = {"value": 0}


def _rank_callback_helper():
    return _RANK_CALLBACK_STATE["value"]


def _rank_callback():
    return _rank_callback_helper()


class _RankDescriptorState:
    @staticmethod
    def value():
        return _RANK_DESCRIPTOR_STATE["value"]


def _rank_descriptor_callback():
    return _RankDescriptorState.value()


def _rank_numpy_dispatcher_callback():
    return np.where([True], [1], [0])[0]


class BackendPolicyMpiIntegrationTest(unittest.TestCase):
    def setUp(self):
        policy._reset_for_testing()
        mp.gpu.set_backend("cpu")

    def tearDown(self):
        policy._reset_for_testing()
        mp.gpu.set_backend("cpu")

    @staticmethod
    def simulation(resolution=8):
        return mp.Simulation(
            cell_size=mp.Vector3(1, 1),
            resolution=resolution,
        )

    def test_collective_cpu_recommendation_precedes_initialization(self):
        simulation = self.simulation()
        self.assertIsNone(simulation.fields)
        simulation.run(until=0.02)
        telemetry = mp.gpu.recommendation
        self.assertEqual(telemetry["selected_backend"], "cpu")
        self.assertEqual(telemetry["recommended_backend"], "cpu")
        self.assertFalse(telemetry["acknowledgement"])
        self.assertEqual(
            len(telemetry["mpi_gpu_topology"]), mp.count_processors()
        )

    @unittest.skipUnless(mp.count_processors() > 1, "fusion requires MPI peers")
    def test_fused_matching_nonce_records_are_exact_and_workload_is_fresh(self):
        simulation = self.simulation()
        process_nonce = f"{mp.my_rank() + 1:064x}"
        proposal = f"{mp.my_rank() + 17:064x}"
        old_state = _RANK_CALLBACK_STATE["value"]
        try:
            with mock.patch.object(policy, "_PROCESS_NONCE", process_nonce), mock.patch.object(
                policy.secrets, "token_hex", return_value=proposal
            ), mock.patch.object(policy, "_collective_records", wraps=policy._collective_records) as gather:
                first = policy.guard_run(mp, simulation, 0.02, None, (_rank_callback,))
                self.assertEqual(gather.call_count, 3)
                self.assertEqual(first["run_nonce"], f"{17:064x}")
                self.assertEqual(
                    first["process_nonce"],
                    policy._sha256_json([f"{rank + 1:064x}" for rank in range(mp.count_processors())]),
                )
                _RANK_CALLBACK_STATE["value"] += 1
                second = policy.guard_run(mp, simulation, 0.02, None, (_rank_callback,))
                self.assertEqual(gather.call_count, 6)
                self.assertNotEqual(first["workload_fingerprint"], second["workload_fingerprint"])
            self.assertIsNone(simulation.fields)
        finally:
            _RANK_CALLBACK_STATE["value"] = old_state

    @unittest.skipUnless(mp.count_processors() > 1, "fusion requires MPI peers")
    def test_fused_one_rank_entropy_failure_is_collectively_bounded(self):
        simulation = self.simulation()

        def nonce(_length):
            if mp.my_rank() == 0:
                raise OSError("injected rank-local entropy failure")
            return "a" * 64

        with mock.patch.object(policy.secrets, "token_hex", side_effect=nonce):
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "distributed gpmeep exact workload preflight failed.*rank-local entropy failure",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
        self.assertIsNone(simulation.fields)

    @unittest.skipUnless(mp.count_processors() > 1, "fusion requires MPI peers")
    def test_fused_one_rank_malformed_or_oversized_nonce_is_collectively_bounded(self):
        for bad in (None, "x" * 5000):
            with self.subTest(value_type=type(bad).__name__):
                simulation = self.simulation()
                proposal = bad if mp.my_rank() == 0 else "a" * 64
                with mock.patch.object(policy.secrets, "token_hex", return_value=proposal):
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "distributed gpmeep exact workload preflight failed.*nonce proposal is malformed",
                    ):
                        policy.guard_run(mp, simulation, 0.02, None)
                self.assertIsNone(simulation.fields)

    def test_package_run_helpers_are_sealed_by_simulation_class_token(self):
        baseline = policy._sha256_json(
            policy._class_token(mp.Simulation, policy._CanonicalTraversal())
        )

        original_fix_dft_args = simulation_module.fix_dft_args.__code__
        try:
            def changed_fix_dft_args(args, index):
                return ("changed",)

            simulation_module.fix_dft_args.__code__ = (
                changed_fix_dft_args.__code__
            )
            with self.assertRaises(policy.BackendRecommendationError):
                policy._class_token(
                    mp.Simulation, policy._CanonicalTraversal()
                )
        finally:
            simulation_module.fix_dft_args.__code__ = original_fix_dft_args

        original_eval_step_func = simulation_module._eval_step_func.__code__
        try:
            def changed_eval_step_func(simulation, function, todo):
                return None

            simulation_module._eval_step_func.__code__ = (
                changed_eval_step_func.__code__
            )
            with self.assertRaises(policy.BackendRecommendationError):
                policy._class_token(
                    mp.Simulation, policy._CanonicalTraversal()
                )
        finally:
            simulation_module._eval_step_func.__code__ = (
                original_eval_step_func
            )

        self.assertEqual(
            baseline,
            policy._sha256_json(
                policy._class_token(
                    mp.Simulation, policy._CanonicalTraversal()
                )
            ),
        )

    @unittest.skipUnless(
        mp.count_processors() == 2,
        "trusted dependency collective failure requires exactly two ranks",
    )
    def test_one_rank_trusted_dependency_mutation_fails_before_initialization(self):
        simulation = self.simulation()
        original_code = simulation_module.fix_dft_args.__code__

        def changed_fix_dft_args(args, index):
            return ("changed",)

        try:
            if mp.my_rank() == 0:
                simulation_module.fix_dft_args.__code__ = (
                    changed_fix_dft_args.__code__
                )
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "distributed gpmeep exact workload preflight failed.*"
                "package dependency was modified",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)
        finally:
            simulation_module.fix_dft_args.__code__ = original_code

    def test_collective_fixed_records_are_exact_under_repetition(self):
        ranks = mp.count_processors()
        for sequence in range(256):
            local = {
                "payload": "x" * (2048 + (17 * mp.my_rank() + sequence) % 512),
                "rank": mp.my_rank(),
                "sequence": sequence,
            }
            encoded = json.dumps(local, sort_keys=True, separators=(",", ":"))
            records = [
                json.loads(record)
                for record in mp._gpu_collective_rank_records(encoded)
            ]
            self.assertEqual(
                records,
                [
                    {
                        "payload": "x" * (2048 + (17 * rank + sequence) % 512),
                        "rank": rank,
                        "sequence": sequence,
                    }
                    for rank in range(ranks)
                ],
            )

    def test_native_chunk_layout_manifest_is_rank_common_and_exact(self):
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.5, 1.0),
            resolution=8,
            split_chunks_evenly=False,
        )
        simulation.init_sim()
        try:
            encoded = simulation.structure.gpmeep_chunk_layout_manifest()
            rank_manifests = mp._gpu_collective_rank_records(encoded)
            self.assertEqual(len(rank_manifests), mp.count_processors())
            self.assertEqual(len(set(rank_manifests)), 1)

            manifest = json.loads(encoded)
            self.assertEqual(manifest["schema"], "gpmeep-chunk-layout-v1")
            self.assertEqual(manifest["num_chunks"], len(manifest["chunks"]))
            self.assertGreater(manifest["num_chunks"], 0)
            for chunk_index, chunk in enumerate(manifest["chunks"]):
                self.assertEqual(chunk["index"], chunk_index)
                self.assertGreaterEqual(chunk["owner"], 0)
                self.assertLess(chunk["owner"], mp.count_processors())
                self.assertGreater(float.fromhex(chunk["a"]), 0.0)
                self.assertGreater(float.fromhex(chunk["inva"]), 0.0)
                self.assertEqual(len(chunk["directions"]), 3)
                self.assertEqual(
                    [direction["slot"] for direction in chunk["directions"]],
                    [0, 1, 2],
                )
                self.assertEqual(
                    len(
                        {
                            direction["direction"]
                            for direction in chunk["directions"]
                        }
                    ),
                    3,
                )
                for direction in chunk["directions"]:
                    self.assertGreater(direction["num"], 0)
                    self.assertGreaterEqual(
                        direction["big"], direction["little"]
                    )
        finally:
            simulation.reset_meep()

    def test_materialized_file_owners_are_bound_before_first_run(self):
        local_root = tempfile.mkdtemp(prefix="gpmeep-owner-") if mp.am_master() else ""
        roots = mp._gpu_collective_rank_records(local_root)
        root = pathlib.Path(next(item for item in roots if item))
        structure_file = str(root / "structure.h5")
        fields_file = str(root / "fields.h5")
        producer = mp.Simulation(
            cell_size=mp.Vector3(1, 1),
            resolution=8,
            default_material=mp.Medium(epsilon=2),
        )
        donor = mp.Simulation(
            cell_size=mp.Vector3(1, 1),
            resolution=8,
            default_material=mp.Medium(epsilon=7),
        )
        target = self.simulation()
        try:
            producer.init_sim()
            producer.dump_structure(structure_file)
            producer.dump_fields(fields_file)
            donor.init_sim()

            target.load_structure_file = structure_file
            target.load_fields_file = fields_file
            target.init_sim()
            original_structure = target.structure
            original_fields = target.fields

            target.structure = donor.structure
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "changed native owner"
            ):
                policy._workload_record(target, 0.02, None, (), None)
            target.structure = original_structure

            target.fields = donor.fields
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "changed native owner"
            ):
                policy._workload_record(target, 0.02, None, (), None)
            target.fields = original_fields
            self.assertTrue(
                policy._sha256_json(
                    policy._workload_record(target, 0.02, None, (), None)
                )
            )
        finally:
            target.reset_meep()
            donor.reset_meep()
            producer.reset_meep()
            mp.all_wait()
            if mp.am_master():
                shutil.rmtree(root)
            mp.all_wait()

    def test_zero_dimensional_material_cell_is_a_valid_policy_domain(self):
        simulation = mp.Simulation(cell_size=mp.Vector3(), resolution=20)
        accepted = policy.guard_run(mp, simulation, None, 1.0)
        self.assertEqual(
            accepted["rank_local_cells"], [1] * mp.count_processors()
        )
        self.assertEqual(accepted["recommended_backend"], "cpu")
        self.assertIsNone(simulation.fields)

    def test_run_k_point_guard_precedes_k_and_field_mutation(self):
        class GuardStopped(RuntimeError):
            pass

        simulation = mp.Simulation(
            cell_size=mp.Vector3(1, 1),
            resolution=8,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.3, fwidth=0.1),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        original_k = simulation.k_point
        with mock.patch.object(
            mp.gpu, "_guard_run", side_effect=GuardStopped("guarded")
        ) as guarded:
            with self.assertRaisesRegex(GuardStopped, "guarded"):
                simulation.run_k_point(1.0, mp.Vector3(0.1, 0.0))
        self.assertIsNone(simulation.fields)
        self.assertEqual(simulation.k_point, original_k)
        self.assertEqual(guarded.call_count, 1)
        self.assertEqual(
            guarded.call_args.args[-1]["entrypoint"], "run_k_point"
        )

    def test_cw_guards_precede_solver_mutation(self):
        class GuardStopped(RuntimeError):
            pass

        class Fields:
            def __init__(self):
                self.called = False

            def solve_cw(self, *args):
                self.called = True
                return True

        for method_name in ("solve_cw", "solve_eigfreq"):
            with self.subTest(method=method_name):
                simulation = self.simulation()
                fields = Fields()
                simulation.fields = fields
                with mock.patch.object(
                    mp.gpu, "_guard_run", side_effect=GuardStopped("guarded")
                ) as guarded:
                    with self.assertRaisesRegex(GuardStopped, "guarded"):
                        getattr(simulation, method_name)()
                self.assertFalse(fields.called)
                self.assertEqual(guarded.call_count, 1)
                self.assertEqual(
                    guarded.call_args.args[-1]["entrypoint"], method_name
                )

    @unittest.skipUnless(
        mp.count_processors() == 2,
        "subcommunicator policy regression requires exactly two ranks",
    )
    def test_divided_subcommunicators_use_the_active_meep_communicator(self):
        subgroup = mp.divide_parallel_processes(2)
        simulation = None
        try:
            self.assertIn(subgroup, (0, 1))
            self.assertEqual(mp.count_processors(), 1)
            simulation = self.simulation()
            simulation.run(until=0.02)
            telemetry = mp.gpu.recommendation
            self.assertEqual(len(telemetry["mpi_gpu_topology"]), 1)
            self.assertEqual(
                telemetry["mpi_gpu_topology"][0]["global_rank"],
                mp.my_global_rank(),
            )
        finally:
            if simulation is not None:
                simulation.reset_meep()
            policy._reset_for_testing()
            mp.end_divide_parallel()
        world_records = [
            json.loads(record)
            for record in mp._gpu_collective_rank_records(
                json.dumps(
                    {"global_rank": mp.my_global_rank()},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]
        self.assertEqual(
            world_records,
            [{"global_rank": rank} for rank in range(2)],
        )

    def test_rank_divergent_workload_fails_before_initialization(self):
        simulation = self.simulation(8 + mp.my_rank())
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "workload fingerprints differ across MPI ranks",
        ):
            simulation.run(until=0.02)
        self.assertIsNone(simulation.fields)

    def test_rank_local_indexed_source_count_is_valid_policy_input(self):
        simulation = self.simulation()
        if mp.my_rank() == 0:
            indexed = object.__new__(mp.IndexedSource)
            indexed.src = None
            indexed.num_pts = 1
            indexed.srcdata = None
            indexed.amp_arr = np.asarray([1.0 + 0.0j])
            indexed.needs_boundary_fix = False
            simulation.sources = [indexed]
        else:
            simulation.sources = []

        telemetry = policy.guard_run(mp, simulation, 0.02, None)
        self.assertEqual(telemetry["recommended_backend"], "cpu")
        self.assertTrue(telemetry["workload_fingerprint_evaluated"])
        source_counts = sorted(
            int(value)
            for value in mp._gpu_collective_rank_records(
                str(telemetry["source_count"])
            )
        )
        self.assertEqual(source_counts, [0, 1])
        self.assertIsNone(simulation.fields)

    @unittest.skipUnless(
        mp.count_processors() == 2,
        "rank-local preflight regression requires exactly two ranks",
    )
    def test_one_rank_preflight_exception_is_collectively_bounded(self):
        class RankLocalFailure(RuntimeError):
            def __str__(self):
                raise self

        class BrokenMonitor:
            @property
            def args(self):
                raise RankLocalFailure()

        simulation = self.simulation()
        if mp.my_rank() == 1:
            simulation.dft_objects = [BrokenMonitor()]
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "distributed gpmeep workload preflight failed.*<unprintable>",
        ):
            policy.guard_run(mp, simulation, 0.02, None)
        self.assertIsNone(simulation.fields)

    def test_rank_divergent_transitive_callable_state_fails_closed(self):
        _RANK_CALLBACK_STATE["value"] = mp.my_rank()
        simulation = self.simulation()
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "workload fingerprints differ across MPI ranks",
        ):
            policy.guard_run(
                mp,
                simulation,
                _rank_callback,
                None,
                (_rank_callback,),
            )
        self.assertIsNone(simulation.fields)

    def test_rank_divergent_bound_builtin_owner_state_fails_closed(self):
        items = list(range(mp.my_rank() + 1))
        callback = items.__len__
        simulation = self.simulation()
        with mock.patch.object(
            policy,
            "_policy_decision",
            return_value=("cuda", "hardware_crossover", "high"),
        ):
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp,
                    simulation,
                    callback,
                    None,
                    (callback,),
                )
        self.assertIsNone(simulation.fields)

    def test_rank_divergent_descriptor_and_annotation_state_fail_closed(self):
        _RANK_DESCRIPTOR_STATE["value"] = mp.my_rank()
        simulation = self.simulation()
        with mock.patch.object(
            policy,
            "_policy_decision",
            return_value=("cuda", "hardware_crossover", "high"),
        ):
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp,
                    simulation,
                    _rank_descriptor_callback,
                    None,
                    (_rank_descriptor_callback,),
                )

        class RankAnnotation:
            value = mp.my_rank()

        def annotation_callback():
            return annotation_callback.__annotations__["return"].value

        annotation_callback.__annotations__ = {"return": RankAnnotation}
        with mock.patch.object(
            policy,
            "_policy_decision",
            return_value=("cuda", "hardware_crossover", "high"),
        ):
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp,
                    simulation,
                    annotation_callback,
                    None,
                    (annotation_callback,),
                )
        self.assertIsNone(simulation.fields)

    def test_rank_divergent_sourcedata_python_state_fails_closed(self):
        source_data = mp.sourcedata()
        source_data.user_scale = mp.my_rank()

        def callback():
            return source_data.user_scale

        simulation = self.simulation()
        with mock.patch.object(
            policy,
            "_policy_decision",
            return_value=("cuda", "hardware_crossover", "high"),
        ):
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp,
                    simulation,
                    callback,
                    None,
                    (callback,),
                )
        self.assertIsNone(simulation.fields)

    def test_rank_divergent_original_numpy_dispatcher_state_fails_closed(self):
        dispatcher = np.where
        missing = object()
        original_signature = getattr(dispatcher, "__signature__", missing)
        try:
            dispatcher.__signature__ = inspect.Signature(
                [
                    inspect.Parameter(
                        "condition",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=mp.my_rank(),
                    )
                ]
            )
            simulation = self.simulation()
            with mock.patch.object(
                policy,
                "_policy_decision",
                return_value=("cuda", "hardware_crossover", "high"),
            ):
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "workload fingerprints differ across MPI ranks",
                ):
                    policy.guard_run(
                        mp,
                        simulation,
                        _rank_numpy_dispatcher_callback,
                        None,
                        (_rank_numpy_dispatcher_callback,),
                    )
            self.assertIsNone(simulation.fields)
        finally:
            if original_signature is missing:
                del dispatcher.__signature__
            else:
                dispatcher.__signature__ = original_signature

    def test_collective_exact_acknowledgement_then_scope_reuse_is_consistent(self):
        simulation = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(mp, simulation, 0.02, None)
            challenge = caught.exception.challenge
            self.assertEqual(
                len(set(mp._gpu_collective_rank_records(challenge))), 1
            )

            if mp.my_rank() == 0:
                policy.acknowledge_backend_mismatch(
                    challenge, "rank zero only acknowledgement"
                )
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "acknowledgement differs across MPI ranks",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)

            policy.acknowledge_backend_mismatch(
                challenge, "collective intentional CPU qualification"
            )
            accepted = policy.guard_run(mp, simulation, 0.02, None)
            self.assertTrue(accepted["acknowledgement"])
            self.assertEqual(accepted["acknowledgement_scope"], "simulation")
            self.assertEqual(
                accepted["acknowledgement_origin_cohort_digest"], challenge
            )
            self.assertEqual(
                accepted["acknowledgement_reason"],
                "collective intentional CPU qualification",
            )

            reused = policy.guard_run(mp, simulation, 0.02, None)
            self.assertTrue(reused["acknowledgement"])
            self.assertTrue(reused["acknowledgement_reused"])
            self.assertEqual(reused["acknowledgement_scope"], "simulation")
            self.assertEqual(
                reused["acknowledgement_origin_cohort_digest"], challenge
            )
            self.assertTrue(reused["workload_fingerprint_evaluated"])
            self.assertEqual(
                reused["acknowledgement_scope_digest"],
                accepted["acknowledgement_scope_digest"],
            )

            independent = self.simulation()
            with self.assertRaises(policy.BackendRecommendationMismatch) as fresh:
                policy.guard_run(mp, independent, 0.02, None)
            self.assertEqual(
                len(
                    set(
                        mp._gpu_collective_rank_records(
                            fresh.exception.challenge
                        )
                    )
                ),
                1,
            )

            if mp.my_rank() == 0:
                policy._ACKNOWLEDGED_POLICY_SCOPES.clear()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "scope cache differs across MPI ranks",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_simulation_scope_reuse_rejects_rank_divergent_callback_state(self):
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            _RANK_CALLBACK_STATE["value"] = 0
            simulation = self.simulation()
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(
                    mp, simulation, 0.02, None, (_rank_callback,)
                )
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "simulation scope must retain MPI semantic validation",
            )
            policy.guard_run(
                mp, simulation, 0.02, None, (_rank_callback,)
            )

            _RANK_CALLBACK_STATE["value"] = mp.my_rank()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp, simulation, 0.02, None, (_rank_callback,)
                )
            self.assertIsNone(simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_collective_process_scope_reuses_across_simulations(self):
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            first = self.simulation()
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(mp, first, 0.02, None)
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "collective process batch qualification",
                scope="process",
            )
            accepted = policy.guard_run(mp, first, 0.02, None)
            self.assertTrue(accepted["acknowledgement"])
            self.assertTrue(accepted["workload_fingerprint_evaluated"])
            self.assertEqual(accepted["acknowledgement_scope"], "process")
            self.assertEqual(
                accepted["acknowledgement_origin_cohort_digest"],
                caught.exception.challenge,
            )

            second = self.simulation()
            reused = policy.guard_run(mp, second, 0.02, None)
            self.assertTrue(reused["acknowledgement"])
            self.assertTrue(reused["acknowledgement_reused"])
            self.assertTrue(reused["workload_fingerprint_evaluated"])
            self.assertEqual(reused["acknowledgement_scope"], "process")
            self.assertEqual(
                reused["acknowledgement_origin_cohort_digest"],
                caught.exception.challenge,
            )

            if mp.my_rank() == 0:
                policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES.clear()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "scope cache differs across MPI ranks",
            ):
                policy.guard_run(mp, self.simulation(), 0.02, None)
        finally:
            policy._policy_decision = original_decision

    def test_process_scope_reuse_rejects_rank_divergent_second_simulation(self):
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            _RANK_CALLBACK_STATE["value"] = 0
            first = self.simulation()
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(
                    mp, first, 0.02, None, (_rank_callback,)
                )
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "collective process batch semantic regression",
                scope="process",
            )
            policy.guard_run(mp, first, 0.02, None, (_rank_callback,))

            _RANK_CALLBACK_STATE["value"] = mp.my_rank()
            second = self.simulation()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "workload fingerprints differ across MPI ranks",
            ):
                policy.guard_run(
                    mp, second, 0.02, None, (_rank_callback,)
                )
            self.assertIsNone(second.fields)
        finally:
            policy._policy_decision = original_decision

    def test_collective_acknowledgement_scope_disagreement_fails_closed(self):
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            simulation = self.simulation()
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(mp, simulation, 0.02, None)
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "rank scope disagreement must fail collectively",
                scope="process" if mp.my_rank() == 0 else "simulation",
            )
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "acknowledgement differs across MPI ranks",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_rank_local_scope_cohort_mismatch_fails_collectively(self):
        simulation = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(mp, simulation, 0.02, None)
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "collective cohort identity regression qualification",
            )
            policy.guard_run(mp, simulation, 0.02, None)
            if mp.my_rank() == 0:
                policy._ACKNOWLEDGED_POLICY_SCOPES[simulation][
                    "cohort_digest"
                ] = "0" * 64
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "scope differs across MPI ranks",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_rank_local_natural_cache_cohort_mix_fails_collectively(self):
        simulation_a = self.simulation()
        simulation_b = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            for simulation, reason in (
                (simulation_a, "collectively accept cache cohort A"),
                (simulation_b, "collectively accept cache cohort B"),
            ):
                with self.assertRaises(
                    policy.BackendRecommendationMismatch
                ) as pending:
                    policy.guard_run(mp, simulation, 0.02, None)
                policy.acknowledge_backend_mismatch(
                    pending.exception.challenge, reason
                )
                policy.guard_run(mp, simulation, 0.02, None)
            mixed_simulation = (
                simulation_a if mp.my_rank() == 0 else simulation_b
            )
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "scope differs across MPI ranks",
            ):
                policy.guard_run(mp, mixed_simulation, 0.02, None)
            self.assertIsNone(simulation_a.fields)
            self.assertIsNone(simulation_b.fields)
        finally:
            policy._policy_decision = original_decision

    def test_rank_local_cache_sequence_damage_on_exact_retry_fails_collectively(self):
        cached_simulation = self.simulation()
        pending_simulation = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            with self.assertRaises(policy.BackendRecommendationMismatch) as cached:
                policy.guard_run(mp, cached_simulation, 0.02, None)
            policy.acknowledge_backend_mismatch(
                cached.exception.challenge,
                "seed a valid Simulation-owned cache entry",
            )
            policy.guard_run(mp, cached_simulation, 0.02, None)

            with self.assertRaises(policy.BackendRecommendationMismatch) as pending:
                policy.guard_run(mp, pending_simulation, 0.02, None)
            policy.acknowledge_backend_mismatch(
                pending.exception.challenge,
                "exact retry must preflight cache state collectively",
            )
            if mp.my_rank() == 0:
                policy._ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE = "malformed"
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "distributed gpmeep workload preflight failed.*"
                "scope cache sequence is malformed",
            ):
                policy.guard_run(mp, pending_simulation, 0.02, None)
            self.assertIsNone(pending_simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_rank_local_malformed_pending_state_fails_collectively(self):
        simulation = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            with self.assertRaises(policy.BackendRecommendationMismatch):
                policy.guard_run(mp, simulation, 0.02, None)
            if mp.my_rank() == 0:
                policy._PENDING_BY_SIMULATION[simulation].pop("stable_key")
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "distributed gpmeep workload preflight failed.*"
                "pending record is malformed",
            ):
                policy.guard_run(mp, simulation, 0.02, None)
            self.assertIsNone(simulation.fields)
        finally:
            policy._policy_decision = original_decision

    def test_rank_local_malformed_scope_cache_fails_collectively(self):
        simulation = self.simulation()
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
                policy.guard_run(mp, simulation, 0.02, None)
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "collective cache corruption regression qualification",
            )
            accepted = policy.guard_run(mp, simulation, 0.02, None)
            self.assertTrue(accepted["acknowledgement"])
            self.assertEqual(len(policy._ACKNOWLEDGED_POLICY_SCOPES), 1)

            if mp.my_rank() == 0:
                cached_simulation = next(iter(policy._ACKNOWLEDGED_POLICY_SCOPES))
                policy._ACKNOWLEDGED_POLICY_SCOPES[cached_simulation] = "malformed"
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "distributed gpmeep workload preflight failed.*"
                "scope cache entry is malformed",
            ):
                policy.guard_run(mp, self.simulation(), 0.02, None)
        finally:
            policy._policy_decision = original_decision


if __name__ == "__main__":
    unittest.main()
