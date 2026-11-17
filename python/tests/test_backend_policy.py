import abc
import ast
import collections
import collections.abc
import contextvars
import copyreg
import enum
import functools
import gc
import importlib.util
import inspect
import os
import pathlib
import json
import random
import re
import tempfile
import threading
import types
import typing
import unittest
from unittest import mock
import weakref
import sys

import numpy as np


_POLICY_PATH = pathlib.Path(__file__).resolve().parents[1] / "backend_policy.py"
_POLICY_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_test_backend_policy", _POLICY_PATH
)
if _POLICY_SPEC is None or _POLICY_SPEC.loader is None:
    raise RuntimeError(f"could not load backend policy from {_POLICY_PATH}")
policy = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(policy)


_CALLBACK_GLOBAL = {"value": 1}
_TRANSITIVE_CALLBACK_GLOBAL = {"value": 1}
_CALLBACK_MODULE = types.ModuleType("gpmeep_backend_policy_test_module")
_CALLBACK_MODULE.threshold = 1
_SPOOFED_MEEP_CLASS_GLOBAL = {"value": 1}
_PACKAGE_GENERATED_GLOBAL_STATE = {"value": 1}


def _production_gaussian_noise_source_class():
    example = (
        pathlib.Path(__file__).resolve().parents[1]
        / "examples/stochastic_emitter.py"
    )
    tree = ast.parse(example.read_text(encoding="utf-8"))
    callback_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianNoiseSource"
    )
    namespace = {"__name__": "gpmeep_test_stochastic_emitter_callback"}
    module = ast.Module(body=[callback_class], type_ignores=[])
    exec(compile(module, str(example), "exec"), namespace)
    return namespace["GaussianNoiseSource"]


def _global_state_callback():
    return _CALLBACK_GLOBAL["value"]


def _transitive_state_helper():
    return _TRANSITIVE_CALLBACK_GLOBAL["value"]


def _transitive_state_callback():
    return _transitive_state_helper()


def _module_state_callback():
    return _CALLBACK_MODULE.threshold


def _package_generated_global_callback():
    return _PACKAGE_GENERATED_GLOBAL_STATE["value"]


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class SourceTime:
    def __init__(self, end_time=10):
        self.end_time = end_time


class Source:
    def __init__(self, end_time=10):
        self.src = SourceTime(end_time)


class Simulation:
    def __init__(self, cells_per_axis=100, resolution=10):
        self.cell_size = Vector(cells_per_axis / resolution, cells_per_axis / resolution, 0)
        self.resolution = resolution
        self.geometry = []
        self.sources = [Source()]
        self.eps_averaging = True
        self.dimensions = 2
        self.boundary_layers = []
        self.symmetries = []
        self.force_complex_fields = False
        self.default_material = None
        self.m = 0
        self.bfast_scaled_k = None
        self.k_point = False
        self.kz_2d = "complex"
        self.extra_materials = []
        self.material_function = None
        self.epsilon_func = None
        self.subpixel_tol = 1e-4
        self.subpixel_maxeval = 100000
        self.loop_tile_base_db = 0
        self.loop_tile_base_eh = 0
        self.ensure_periodicity = True
        self.num_chunks = 0
        self.Courant = 0.5
        self.accurate_fields_near_cylorigin = False
        self.geometry_center = Vector()
        self.force_all_components = False
        self.split_chunks_evenly = True
        self.chunk_layout = None
        self.dft_objects = []


class NativeFields:
    def __init__(self):
        self.this = object()
        self.epoch = 0

    def gpmeep_state_manifest(self):
        return json.dumps(
            {
                "schema": "gpmeep-fields-state-v2",
                "epoch": self.epoch,
                "t": self.epoch,
                "external_source_count": 0,
                "sources": [],
            },
            sort_keys=True,
        )

    def mutate(self):
        self.epoch += 1


class NativeStructure:
    def __init__(self):
        self.this = object()
        self.epoch = 0

    def gpmeep_state_manifest(self):
        return json.dumps(
            {
                "schema": "gpmeep-structure-state-v1",
                "epoch": self.epoch,
            },
            sort_keys=True,
        )

    def mutate(self):
        self.epoch += 1


class LayoutStructure(NativeStructure):
    def __init__(self, owner=0, little=0):
        super().__init__()
        self.owner = owner
        self.little = little
        self.manifest_override = None

    def gpmeep_chunk_layout_manifest(self):
        if self.manifest_override is not None:
            return self.manifest_override
        directions = [
            {
                "slot": slot,
                "direction": slot,
                "little": self.little if slot == 0 else 0,
                "big": self.little + 18 if slot == 0 else 0,
                "num": 10 if slot == 0 else 1,
            }
            for slot in range(3)
        ]
        return json.dumps(
            {
                "schema": "gpmeep-chunk-layout-v1",
                "num_chunks": 1,
                "chunks": [
                    {
                        "index": 0,
                        "owner": self.owner,
                        "dimension": 1,
                        "a": "0x1.4000000000000p+3",
                        "inva": "0x1.999999999999ap-4",
                        "directions": directions,
                    }
                ],
            },
            sort_keys=True,
        )


MeepStructure = type(
    "structure",
    (LayoutStructure,),
    {
        "__module__": "meep",
        "gpmeep_state_manifest": NativeStructure.gpmeep_state_manifest,
        "gpmeep_chunk_layout_manifest": (
            LayoutStructure.gpmeep_chunk_layout_manifest
        ),
        "get_chunk_volumes": lambda self: [self.little],
        "get_chunk_owners": lambda self: [self.owner],
    },
)


MeepFields = type(
    "fields",
    (NativeFields,),
    {
        "__module__": "meep",
        "gpmeep_state_manifest": NativeFields.gpmeep_state_manifest,
    },
)


MeepSimulation = type(
    "Simulation", (Simulation,), {"__module__": "meep.simulation"}
)


class IndexedSource:
    __module__ = "meep.source"

    def __init__(self, payload):
        self.payload = payload


class SpoofedMeepClass:
    __module__ = "meep.user_extension"

    @staticmethod
    def semantic_value():
        return _SPOOFED_MEEP_CLASS_GLOBAL["value"]


class TrustedMeepGeometry:
    @staticmethod
    def semantic_value():
        return 1


class TrustedDftLdos:
    __module__ = "meep"
    __qualname__ = "dft_ldos"

    def __init__(self, frequencies):
        self.this = object()
        self._frequencies = list(frequencies)

    @property
    def freq(self):
        return list(self._frequencies)


class GPU:
    def __init__(self, backend="cpu", device_identifier="GPU-a"):
        self.requested_backend = backend
        self.active_backend = backend
        self.runtime_diagnostic = "runtime unavailable"
        self.device_identifier = device_identifier
        self.selected_device = 0
        self.selected_device_identifier = device_identifier
        self._backend_generation = 0
        self.devices_calls = 0

    def devices(self):
        self.devices_calls += 1
        return [
            {
                "ordinal": 0,
                "compatible": True,
                "identifier": self.device_identifier,
                "global_memory_bytes": 24 * 1024**3,
                "memory_bandwidth_bytes_per_second": 1_000_000_000_000,
            }
        ]


class MP:
    def __init__(self, backend="cpu", ranks=1, transform=None, identifier="GPU-a"):
        self.gpu = GPU(backend, identifier)
        self.ranks = ranks
        self.transform = transform

    def count_processors(self):
        return self.ranks

    def my_rank(self):
        return 0

    def my_global_rank(self):
        return 0

    def my_node_rank(self):
        return 0

    def count_node_processors(self):
        return self.ranks

    def _gpu_collective_rank_records(self, record):
        records = [record for _ in range(self.ranks)]
        return self.transform(record, records) if self.transform else records


class BackendPolicyTest(unittest.TestCase):
    def test_generator_backed_callable_instance_is_exact_and_stateful(self):
        callback_class = _production_gaussian_noise_source_class()
        first = callback_class(np.random.default_rng(8675309))
        second = callback_class(np.random.default_rng(8675309))
        initial_digest = policy._sha256_json(policy._canonical(first))
        self.assertEqual(
            initial_digest,
            policy._sha256_json(policy._canonical(second)),
        )

        first_sample = first(0.0)
        second_sample = second(0.0)
        self.assertEqual(first_sample, second_sample)
        post_call_digest = policy._sha256_json(policy._canonical(first))
        self.assertNotEqual(initial_digest, post_call_digest)
        self.assertEqual(
            post_call_digest,
            policy._sha256_json(policy._canonical(second)),
        )

        rng_only = callback_class(np.random.default_rng(8675309))
        rng_only.rng.standard_normal()
        self.assertNotEqual(
            initial_digest,
            policy._sha256_json(policy._canonical(rng_only)),
        )

        count_only = callback_class(np.random.default_rng(8675309))
        count_only.calls += 1
        self.assertNotEqual(
            initial_digest,
            policy._sha256_json(policy._canonical(count_only)),
        )

    def test_exact_pcg64_generator_state_is_canonical_and_sequence_exact(self):
        seed = 8675309
        first = np.random.default_rng(seed)
        second = np.random.default_rng(seed)
        first_token = policy._canonical(first)
        second_token = policy._canonical(second)
        self.assertEqual(
            policy._sha256_json(first_token),
            policy._sha256_json(second_token),
        )
        self.assertNotEqual(
            policy._sha256_json(first_token),
            policy._sha256_json(policy._canonical(np.random.default_rng(seed + 1))),
        )

        expected = first.standard_normal(4096)
        observed = second.standard_normal(4096)
        np.testing.assert_array_equal(observed, expected)
        self.assertNotEqual(
            policy._sha256_json(first_token),
            policy._sha256_json(policy._canonical(first)),
        )

        state = first.bit_generator.state
        clone_bit_generator = np.random.PCG64()
        clone_bit_generator.state = state
        clone = np.random.Generator(clone_bit_generator)
        np.testing.assert_array_equal(
            clone.standard_normal(4096),
            first.standard_normal(4096),
        )

    def test_numpy_rng_rejects_unsupported_subclass_and_export_mutation(self):
        for bit_generator_type in (
            np.random.MT19937,
            np.random.Philox,
            np.random.SFC64,
            np.random.PCG64DXSM,
        ):
            with self.subTest(bit_generator=bit_generator_type.__name__):
                unsupported = np.random.Generator(bit_generator_type(17))
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError, "only exact NumPy PCG64"
                ):
                    policy._canonical(unsupported)

        class GeneratorSubclass(np.random.Generator):
            pass

        subclassed = GeneratorSubclass(np.random.PCG64(17))
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "unsupported NumPy Generator"
        ):
            policy._canonical(subclassed)

        original = np.random.Generator
        try:
            np.random.Generator = object
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "RNG surface differs",
            ):
                policy._canonical(np.random.default_rng(17))
        finally:
            np.random.Generator = original

        original_pcg64 = np.random.PCG64
        try:
            np.random.PCG64 = object
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "RNG surface differs",
            ):
                policy._canonical(np.random.default_rng(17))
        finally:
            np.random.PCG64 = original_pcg64

    def test_pcg64_alias_topology_is_canonicalized(self):
        shared_bit_generator = np.random.PCG64(17)
        shared_generator = np.random.Generator(shared_bit_generator)

        first_clone_bit_generator = np.random.PCG64()
        first_clone_bit_generator.state = shared_bit_generator.state
        second_clone_bit_generator = np.random.PCG64()
        second_clone_bit_generator.state = shared_bit_generator.state
        cloned_generator = np.random.Generator(first_clone_bit_generator)

        shared = (shared_generator, shared_bit_generator)
        independent = (cloned_generator, second_clone_bit_generator)
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(shared)),
            policy._sha256_json(policy._canonical(independent)),
        )

        second_shared_generator = np.random.Generator(shared_bit_generator)
        first_clone = np.random.PCG64()
        first_clone.state = shared_bit_generator.state
        second_clone = np.random.PCG64()
        second_clone.state = shared_bit_generator.state
        split_generators = (
            np.random.Generator(first_clone),
            np.random.Generator(second_clone),
        )
        self.assertNotEqual(
            policy._sha256_json(
                policy._canonical((shared_generator, second_shared_generator))
            ),
            policy._sha256_json(policy._canonical(split_generators)),
        )

        shared_seed_sequence = np.random.SeedSequence(123)
        shared_bit_generators = (
            np.random.PCG64(shared_seed_sequence),
            np.random.PCG64(shared_seed_sequence),
        )
        cloned_bit_generators = (
            np.random.PCG64(np.random.SeedSequence(123)),
            np.random.PCG64(np.random.SeedSequence(123)),
        )
        for bit_generator in cloned_bit_generators:
            bit_generator.state = shared_bit_generators[0].state
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(shared_bit_generators)),
            policy._sha256_json(policy._canonical(cloned_bit_generators)),
        )

    def test_pcg64_seed_sequence_and_spawn_state_are_canonicalized(self):
        first = np.random.PCG64(17)
        second = np.random.PCG64(18)
        second.state = first.state
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first)),
            policy._sha256_json(policy._canonical(second)),
        )

        before = policy._sha256_json(policy._canonical(first))
        numerical_state = first.state
        first.spawn(1)
        self.assertEqual(first.state, numerical_state)
        self.assertNotEqual(
            before, policy._sha256_json(policy._canonical(first))
        )

        class SeedSequenceSubclass(np.random.SeedSequence):
            pass

        unsupported = np.random.PCG64(SeedSequenceSubclass(17))
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "exact stable SeedSequence"
        ):
            policy._canonical(unsupported)

    def test_seed_sequence_pool_is_canonicalized(self):
        first = np.random.PCG64([1, 2, 3])
        second = np.random.PCG64([1, 2, 3])
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.seed_seq.state, second.seed_seq.state)
        self.assertTrue(
            np.array_equal(
                first.seed_seq.generate_state(16),
                second.seed_seq.generate_state(16),
            )
        )

        before = policy._sha256_json(policy._canonical(first))
        second.seed_seq.pool[0] ^= np.uint32(1)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.seed_seq.state, second.seed_seq.state)
        self.assertFalse(
            np.array_equal(
                first.seed_seq.generate_state(16),
                second.seed_seq.generate_state(16),
            )
        )
        self.assertNotEqual(
            before, policy._sha256_json(policy._canonical(second))
        )

    def test_numpy_rng_version_and_seed_sequence_exports_are_exact(self):
        original_version = np.__version__
        original_generator = np.random.Generator

        class MutatingVersion:
            called = False

            def __str__(self):
                type(self).called = True
                np.random.Generator = object
                return original_version

        try:
            np.__version__ = MutatingVersion()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.PCG64(17))
            self.assertFalse(MutatingVersion.called)
            self.assertIs(np.random.Generator, original_generator)
        finally:
            np.__version__ = original_version
            np.random.Generator = original_generator

        class VersionSubclass(str):
            pass

        try:
            np.__version__ = VersionSubclass(original_version)
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.PCG64(17))
        finally:
            np.__version__ = original_version

        original_seed_sequence = np.random.SeedSequence
        try:
            np.random.SeedSequence = object
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.PCG64(17))
        finally:
            np.random.SeedSequence = original_seed_sequence

    def test_seed_sequence_generate_state_semantics_are_sealed(self):
        method = np.random.SeedSequence.__dict__["generate_state"]
        original_code = method.__code__

        def replacement_factory():
            func = object()
            owner = object()

            def replacement(*args, **kwargs):
                if func is owner:
                    return ()
                return (0, 0, 0, 0)

            return replacement

        try:
            method.__code__ = replacement_factory().__code__
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.SeedSequence(17))
        finally:
            method.__code__ = original_code

        closure_cell = method.__closure__[0]
        original_value = closure_cell.cell_contents
        try:
            closure_cell.cell_contents = object()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.SeedSequence(17))
        finally:
            closure_cell.cell_contents = original_value

        owner = dict(zip(method.__code__.co_freevars, method.__closure__))[
            "self"
        ].cell_contents
        original_all = owner._all
        try:
            owner._all = "not-a-valid-error-policy"
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "RNG surface differs"
            ):
                policy._canonical(np.random.SeedSequence(17))
        finally:
            owner._all = original_all

    def test_pcg64_spawn_invalidates_pending_acknowledgement(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        generator = np.random.default_rng(17)

        def callback():
            return generator.standard_normal()

        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        old_challenge = first.exception.challenge
        policy.acknowledge_backend_mismatch(
            old_challenge,
            "SeedSequence spawn state must bind this acknowledgement",
        )
        numerical_state = generator.bit_generator.state
        generator.bit_generator.spawn(1)
        self.assertEqual(generator.bit_generator.state, numerical_state)
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        self.assertNotEqual(old_challenge, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old_challenge,
                "stale spawn acknowledgement must not be reusable",
            )

    def test_seed_sequence_pool_mutation_invalidates_acknowledgement(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        generator = np.random.default_rng([1, 2, 3])

        def callback():
            return generator.standard_normal()

        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        old_challenge = first.exception.challenge
        policy.acknowledge_backend_mismatch(
            old_challenge,
            "SeedSequence pool state must bind this acknowledgement",
        )
        numerical_state = generator.bit_generator.state
        seed_state = generator.bit_generator.seed_seq.state
        generator.bit_generator.seed_seq.pool[0] ^= np.uint32(1)
        self.assertEqual(generator.bit_generator.state, numerical_state)
        self.assertEqual(generator.bit_generator.seed_seq.state, seed_state)
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        self.assertNotEqual(old_challenge, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old_challenge,
                "stale pool acknowledgement must not be reusable",
            )

    def test_pcg64_draw_invalidates_pending_acknowledgement(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        generator = np.random.default_rng(17)

        def callback():
            return generator.standard_normal()

        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        old_challenge = first.exception.challenge
        policy.acknowledge_backend_mismatch(
            old_challenge,
            "exact RNG state must bind this acknowledgement",
        )

        generator.standard_normal()
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None, (callback,))
        self.assertNotEqual(old_challenge, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old_challenge,
                "stale RNG acknowledgement must not be reusable",
            )

    def setUp(self):
        policy._reset_for_testing()
        self._trusted_meep_classes = policy._TRUSTED_MEEP_CLASSES
        self._trusted_meep_class_members = policy._TRUSTED_MEEP_CLASS_MEMBERS
        self._trusted_function_annotation_snapshots = (
            policy._TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS
        )
        policy._TRUSTED_MEEP_CLASSES = frozenset(
            (
                *self._trusted_meep_classes,
                MeepSimulation,
                MeepStructure,
                MeepFields,
                IndexedSource,
                TrustedMeepGeometry,
                TrustedDftLdos,
            )
        )
        policy._TRUSTED_MEEP_CLASS_MEMBERS = {
            **self._trusted_meep_class_members,
            MeepSimulation: dict(MeepSimulation.__dict__),
            MeepStructure: dict(MeepStructure.__dict__),
            MeepFields: dict(MeepFields.__dict__),
            IndexedSource: dict(IndexedSource.__dict__),
            TrustedMeepGeometry: dict(TrustedMeepGeometry.__dict__),
            TrustedDftLdos: dict(TrustedDftLdos.__dict__),
        }

    def tearDown(self):
        policy._TRUSTED_MEEP_CLASSES = self._trusted_meep_classes
        policy._TRUSTED_MEEP_CLASS_MEMBERS = self._trusted_meep_class_members
        policy._TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS = (
            self._trusted_function_annotation_snapshots
        )
        _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 1
        _PACKAGE_GENERATED_GLOBAL_STATE["value"] = 1

    def test_small_cpu_matches_without_acknowledgement(self):
        result = policy.guard_run(MP("cpu"), Simulation(100), 10, None)
        self.assertEqual(result["selected_backend"], "cpu")
        self.assertEqual(result["recommended_backend"], "cpu")
        self.assertFalse(result["acknowledgement"])
        self.assertEqual(result["reason_class"], "hardware_crossover")

    def test_guard_reuses_positive_device_inventory_for_repeated_runs(self):
        mp = MP("cpu")
        simulation = Simulation(1200)
        original_decision = policy._policy_decision
        policy._policy_decision = lambda _records: (
            "cpu",
            "hardware_crossover",
            "high",
        )
        try:
            for _ in range(20):
                result = policy.guard_run(mp, simulation, 10, None)
                self.assertEqual(result["selected_backend"], "cpu")
        finally:
            policy._policy_decision = original_decision
        self.assertEqual(mp.gpu.devices_calls, 1)

    def test_device_inventory_cache_returns_mutation_isolated_records(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        first = policy._compatible_device_inventory(mp, owner)
        first[0]["identifier"] = "tampered"
        first[0]["global_memory_bytes"] = 1
        second = policy._compatible_device_inventory(mp, owner)
        self.assertEqual(second[0]["identifier"], "GPU-a")
        self.assertEqual(second[0]["global_memory_bytes"], 24 * 1024**3)
        self.assertEqual(mp.gpu.devices_calls, 1)

    def test_device_inventory_cache_is_bounded_to_one_live_simulation(self):
        mp = MP("cuda")
        first_owner = Simulation(1200)
        second_owner = Simulation(1200)
        policy._compatible_device_inventory(mp, first_owner)
        policy._compatible_device_inventory(mp, first_owner)
        self.assertEqual(mp.gpu.devices_calls, 1)
        policy._compatible_device_inventory(mp, second_owner)
        self.assertEqual(mp.gpu.devices_calls, 2)
        policy._compatible_device_inventory(mp)
        policy._compatible_device_inventory(mp)
        self.assertEqual(mp.gpu.devices_calls, 4)

    def test_gpu_controller_transition_is_serialized_and_invalidates_cache(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        policy._compatible_device_inventory(mp, owner)
        entered = threading.Event()
        finished = threading.Event()

        def transition(value):
            entered.set()
            mp.gpu.requested_backend = value
            mp.gpu.active_backend = value
            mp.gpu._backend_generation += 1

        def run_transition():
            policy._run_gpu_controller_transition(
                mp.gpu, transition, "cpu"
            )
            finished.set()

        worker = threading.Thread(target=run_transition)
        with policy._LOCK:
            worker.start()
            self.assertFalse(entered.wait(0.05))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 2)

        policy._compatible_device_inventory(mp, owner)

        def rejected(_value):
            raise RuntimeError("native transition failure")

        with self.assertRaisesRegex(RuntimeError, "native transition failure"):
            policy._run_gpu_controller_transition(
                mp.gpu, rejected, "cuda"
            )
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 3)

    def test_device_inventory_cache_invalidates_on_state_and_environment(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        policy._compatible_device_inventory(mp, owner)
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 1)

        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "gpmeep-unit-cache-change"},
            clear=False,
        ):
            policy._compatible_device_inventory(mp, owner)
            policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 2)

        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 3)
        mp.gpu._backend_generation += 1
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 4)
        mp.gpu.selected_device = 1
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 5)
        mp.ranks = 2
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 6)

    def test_device_inventory_cache_does_not_cache_exception_or_empty(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        calls = {"count": 0}
        device = {
            "ordinal": 0,
            "compatible": True,
            "identifier": "GPU-a",
            "global_memory_bytes": 24 * 1024**3,
            "memory_bandwidth_bytes_per_second": 1_000_000_000_000,
        }

        def transient_devices():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient discovery failure")
            if calls["count"] == 2:
                return []
            return [dict(device)]

        mp.gpu.devices = transient_devices
        with self.assertRaisesRegex(RuntimeError, "transient discovery"):
            policy._compatible_device_inventory(mp, owner)
        self.assertEqual(policy._compatible_device_inventory(mp, owner), [])
        self.assertEqual(policy._compatible_device_inventory(mp, owner), [device])
        self.assertEqual(policy._compatible_device_inventory(mp, owner), [device])
        self.assertEqual(calls["count"], 3)

    def test_device_inventory_cache_serializes_concurrent_discovery(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        started = threading.Event()
        release = threading.Event()
        calls = {"count": 0}
        results = []

        def slow_devices():
            calls["count"] += 1
            started.set()
            if not release.wait(5):
                raise RuntimeError("test discovery release timed out")
            return GPU.devices(mp.gpu)

        mp.gpu.devices = slow_devices
        workers = [
            threading.Thread(
                target=lambda: results.append(
                    policy._compatible_device_inventory(mp, owner)
                )
            )
            for _ in range(2)
        ]
        workers[0].start()
        self.assertTrue(started.wait(5))
        workers[1].start()
        release.set()
        for worker in workers:
            worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 2)
        self.assertEqual(calls["count"], 1)

    def test_device_inventory_cache_reset_and_callable_replacement_miss(self):
        mp = MP("cuda")
        owner = Simulation(1200)
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 1)
        policy._reset_after_fork_child()
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(mp.gpu.devices_calls, 2)

        replacement_calls = {"count": 0}

        def replacement_devices():
            replacement_calls["count"] += 1
            return GPU.devices(mp.gpu)

        mp.gpu.devices = replacement_devices
        policy._compatible_device_inventory(mp, owner)
        policy._compatible_device_inventory(mp, owner)
        self.assertEqual(replacement_calls["count"], 1)

    def test_small_cuda_requires_exact_first_acknowledgement_then_reuses_scope(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        challenge = caught.exception.challenge
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(challenge, "short")
        policy.acknowledge_backend_mismatch(challenge, "intentional CUDA qualification")
        accepted = policy.guard_run(mp, simulation, 10, None)
        self.assertTrue(accepted["acknowledgement"])
        self.assertEqual(accepted["acknowledgement_scope"], "simulation")
        self.assertEqual(
            accepted["acknowledgement_origin_cohort_digest"], challenge
        )
        self.assertEqual(
            accepted["acknowledgement_reason"], "intentional CUDA qualification"
        )
        with mock.patch.object(
            policy,
            "_workload_record",
            side_effect=AssertionError("same-Simulation reuse traversed workload"),
        ):
            repeated = policy.guard_run(mp, simulation, 10, None)
        self.assertTrue(repeated["acknowledgement"])
        self.assertTrue(repeated["acknowledgement_reused"])
        self.assertEqual(repeated["acknowledgement_scope"], "simulation")
        self.assertEqual(
            repeated["acknowledgement_origin_cohort_digest"], challenge
        )
        self.assertFalse(repeated["workload_fingerprint_evaluated"])
        self.assertEqual(
            repeated["acknowledgement_reason"], "intentional CUDA qualification"
        )
        with self.assertRaises(policy.BackendRecommendationMismatch) as independent:
            policy.guard_run(mp, Simulation(100), 10, None)
        self.assertNotEqual(challenge, independent.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                challenge, "replay of consumed acknowledgement"
            )

    def test_process_scope_requires_explicit_opt_in_and_reuses_across_simulations(self):
        mp = MP("cuda")
        first_simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, first_simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            caught.exception.challenge,
            "one deliberate decision covers this simulation batch",
            scope="process",
        )
        accepted = policy.guard_run(mp, first_simulation, 10, None)
        self.assertTrue(accepted["acknowledgement"])
        self.assertFalse(accepted["acknowledgement_reused"])
        self.assertEqual(accepted["acknowledgement_scope"], "process")
        self.assertEqual(
            accepted["acknowledgement_origin_cohort_digest"],
            caught.exception.challenge,
        )
        self.assertEqual(len(policy._ACKNOWLEDGED_POLICY_SCOPES), 0)
        self.assertEqual(len(policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES), 1)

        independent = Simulation(100)
        with mock.patch.object(
            policy,
            "_workload_record",
            side_effect=AssertionError("process-scope reuse traversed workload"),
        ):
            reused = policy.guard_run(mp, independent, 10, None)
        self.assertTrue(reused["acknowledgement"])
        self.assertTrue(reused["acknowledgement_reused"])
        self.assertEqual(reused["acknowledgement_scope"], "process")
        self.assertEqual(
            reused["acknowledgement_origin_cohort_digest"],
            caught.exception.challenge,
        )
        self.assertFalse(reused["workload_fingerprint_evaluated"])

    def test_process_scope_value_is_strict_and_does_not_change_default(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "scope must be 'simulation' or 'process'",
        ):
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge,
                "invalid scope must fail before granting reuse",
                scope="global",
            )
        policy.acknowledge_backend_mismatch(
            caught.exception.challenge,
            "default remains bound to one Simulation object",
        )
        policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(len(policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES), 0)
        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, Simulation(100), 10, None)

    def test_scope_cache_can_be_disabled_for_strict_same_binary_ablation(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with mock.patch.dict(
            os.environ,
            {"MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE": "1"},
            clear=False,
        ):
            with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                policy.guard_run(mp, simulation, 10, None)
            policy.acknowledge_backend_mismatch(
                first.exception.challenge, "strict per-run comparison control"
            )
            accepted = policy.guard_run(mp, simulation, 10, None)
            self.assertFalse(accepted["acknowledgement_scope_cache_enabled"])
            with self.assertRaises(policy.BackendRecommendationMismatch) as repeated:
                policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(first.exception.challenge, repeated.exception.challenge)

    def test_acknowledgement_scope_is_owned_by_one_simulation(self):
        mp = MP("cuda")
        pending = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, pending, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "pending exact retry must retain priority over scope reuse",
        )

        learner = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as other:
            policy.guard_run(mp, learner, 10, None)
        policy.acknowledge_backend_mismatch(
            other.exception.challenge,
            "independent simulation learns only its own policy scope",
        )
        learned = policy.guard_run(mp, learner, 10, None)
        self.assertFalse(learned["acknowledgement_reused"])
        reused = policy.guard_run(mp, learner, 10, None)
        self.assertTrue(reused["acknowledgement_reused"])

        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, Simulation(100), 10, None)

        pending.sources.append(7)
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, pending, 10, None)
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)

    def test_simulation_scope_cache_is_weak_and_globally_bounded(self):
        mp = MP("cuda")
        simulations = []
        for _index in range(policy.MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES + 3):
            simulation = Simulation(100)
            simulations.append(simulation)
            with self.assertRaises(policy.BackendRecommendationMismatch) as mismatch:
                policy.guard_run(mp, simulation, 10, None)
            policy.acknowledge_backend_mismatch(
                mismatch.exception.challenge,
                "bounded Simulation-owned acknowledgement cache",
            )
            policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(
            len(policy._ACKNOWLEDGED_POLICY_SCOPES),
            policy.MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES,
        )
        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, simulations[0], 10, None)
        self.assertTrue(
            policy.guard_run(mp, simulations[-1], 10, None)[
                "acknowledgement_reused"
            ]
        )

        retained = simulations[-1]
        retained_reference = weakref.ref(retained)
        simulations.clear()
        del simulation
        del retained
        gc.collect()
        self.assertIsNone(retained_reference())
        self.assertEqual(len(policy._ACKNOWLEDGED_POLICY_SCOPES), 0)
        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, Simulation(100), 10, None)

    def test_process_scope_cache_is_bounded_and_fails_closed_on_corruption(self):
        mp = MP("cuda")
        simulations = []
        for index in range(policy.MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES + 3):
            simulation = Simulation(100 + index)
            simulations.append(simulation)
            with self.assertRaises(policy.BackendRecommendationMismatch) as mismatch:
                policy.guard_run(mp, simulation, 10, None)
            policy.acknowledge_backend_mismatch(
                mismatch.exception.challenge,
                "bounded explicit process acknowledgement cache",
                scope="process",
            )
            policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(
            len(policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES),
            policy.MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES,
        )
        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, simulations[0], 10, None)
        self.assertTrue(
            policy.guard_run(mp, simulations[-1], 10, None)[
                "acknowledgement_reused"
            ]
        )

        scope_digest = next(iter(policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES))
        record = policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES[scope_digest]
        record["owner_pid"] = os.getpid() + 1
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "process-scope acknowledgement entry is malformed",
        ):
            policy.guard_run(mp, Simulation(1000), 10, None)
        record["owner_pid"] = os.getpid()
        policy._ACKNOWLEDGED_PROCESS_POLICY_SCOPES[scope_digest] = "malformed"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "process-scope acknowledgement entry is malformed",
        ):
            policy.guard_run(mp, Simulation(1000), 10, None)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_child_rejects_inherited_pending_and_scope_capabilities(self):
        mp = MP("cuda")

        def fork_observation(simulation, inherited_challenge):
            read_descriptor, write_descriptor = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_descriptor)
                result = {}
                try:
                    try:
                        policy.acknowledge_backend_mismatch(
                            inherited_challenge,
                            "fork child must reject its inherited challenge",
                        )
                    except policy.BackendRecommendationError:
                        result["inherited_challenge_rejected"] = True
                    else:
                        result["inherited_challenge_rejected"] = False
                    try:
                        policy.guard_run(mp, simulation, 10, None)
                    except policy.BackendRecommendationMismatch as mismatch:
                        result.update(
                            {
                                "outcome": "new_mismatch",
                                "challenge": mismatch.challenge,
                                "process_nonce": mismatch.telemetry[
                                    "process_nonce"
                                ],
                            }
                        )
                    else:
                        result["outcome"] = "unexpected_success"
                    payload = json.dumps(result, sort_keys=True).encode("utf-8")
                    os.write(write_descriptor, payload)
                finally:
                    os.close(write_descriptor)
                    os._exit(0)
            os.close(write_descriptor)
            chunks = []
            while True:
                chunk = os.read(read_descriptor, 4096)
                if not chunk:
                    break
                chunks.append(chunk)
            os.close(read_descriptor)
            waited_pid, status = os.waitpid(pid, 0)
            self.assertEqual(pid, waited_pid)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(0, os.WEXITSTATUS(status))
            return json.loads(b"".join(chunks).decode("utf-8"))

        pending = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, pending, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "parent reason acknowledgement must not cross a fork",
        )
        pending_child = fork_observation(pending, first.exception.challenge)
        self.assertTrue(pending_child["inherited_challenge_rejected"])
        self.assertEqual("new_mismatch", pending_child["outcome"])
        self.assertNotEqual(first.exception.challenge, pending_child["challenge"])
        self.assertNotEqual(
            first.exception.telemetry["process_nonce"],
            pending_child["process_nonce"],
        )

        policy._reset_for_testing()
        cached = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as second:
            policy.guard_run(mp, cached, 10, None)
        policy.acknowledge_backend_mismatch(
            second.exception.challenge,
            "parent policy scope must not cross a fork boundary",
            scope="process",
        )
        policy.guard_run(mp, cached, 10, None)
        cached_child = fork_observation(cached, second.exception.challenge)
        self.assertTrue(cached_child["inherited_challenge_rejected"])
        self.assertEqual("new_mismatch", cached_child["outcome"])
        self.assertNotEqual(second.exception.challenge, cached_child["challenge"])

    def test_large_cpu_is_guarded_when_cuda_is_recommended(self):
        mp = MP("cpu")
        simulation = Simulation(1200)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 20, None)
        telemetry = caught.exception.telemetry
        self.assertEqual(telemetry["recommended_backend"], "cuda")
        self.assertEqual(telemetry["rank_local_cells"], [1_440_000])
        self.assertGreater(telemetry["kernel_launch_count"], 0)

    def test_changed_workload_invalidates_old_challenge(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        old = first.exception.challenge
        simulation.sources.append(Source(20))
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(old, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(old, "old workload is no longer valid")

    def test_changed_step_function_invalidates_old_challenge(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        first_step = lambda: None
        second_step = lambda: 1
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None, (first_step,))
        old = first.exception.challenge
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None, (second_step,))
        self.assertNotEqual(old, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old, "old step function is no longer valid"
            )

    def test_changed_callable_object_and_global_state_invalidate_challenge(self):
        class StatefulStep:
            def __init__(self):
                self.value = 1

            def __call__(self):
                return self.value

            def bound(self):
                return self.value

        mp = MP("cuda")
        simulation = Simulation(100)
        step = StatefulStep()
        for callback, mutate in (
            (step, lambda: setattr(step, "value", 2)),
            (step.bound, lambda: setattr(step, "value", 3)),
            (_global_state_callback, lambda: _CALLBACK_GLOBAL.update(value=2)),
        ):
            with self.subTest(callback=callback):
                policy._reset_for_testing()
                step.value = 1
                _CALLBACK_GLOBAL["value"] = 1
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(mp, simulation, 10, None, (callback,))
                old = first.exception.challenge
                mutate()
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(mp, simulation, 10, None, (callback,))
                self.assertNotEqual(old, changed.exception.challenge)

        def handled_exception():
            try:
                return 1 // 0
            except ZeroDivisionError:
                return 1

        if hasattr(handled_exception.__code__, "co_exceptiontable"):
            changed_code = handled_exception.__code__.replace(
                co_exceptiontable=b""
            )
            changed_exception = types.FunctionType(
                changed_code, handled_exception.__globals__
            )
            self.assertNotEqual(
                policy._sha256_json(policy._canonical(handled_exception)),
                policy._sha256_json(policy._canonical(changed_exception)),
            )

        def two_arguments(first, second):
            return first

        changed_argument_count = types.FunctionType(
            two_arguments.__code__.replace(co_argcount=1),
            two_arguments.__globals__,
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(two_arguments)),
            policy._sha256_json(policy._canonical(changed_argument_count)),
        )

        formatted_state = {"value": 1}

        def formatted_callback():
            print(f"value={formatted_state['value']}")

        formatted_before = policy._sha256_json(
            policy._canonical(formatted_callback)
        )
        formatted_state["value"] = 2
        formatted_after = policy._sha256_json(
            policy._canonical(formatted_callback)
        )
        self.assertNotEqual(formatted_before, formatted_after)

        captured_formats = []

        def effectful_formatted_callback():
            print(captured_formats.append(f"{object()}"))

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "FORMAT_VALUE"
        ):
            policy._canonical(effectful_formatted_callback)

        def returned_identity_format():
            return f"{object()}"

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "FORMAT_VALUE"
        ):
            policy._canonical(returned_identity_format)

        class FormattedWriter:
            def write(self, value):
                captured_formats.append(value)

            def flush(self):
                pass

        formatted_writer = FormattedWriter()

        def redirected_identity_format():
            print(f"{object()}", file=formatted_writer)

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "FORMAT_VALUE"
        ):
            policy._canonical(redirected_identity_format)

        original_print = policy.builtins.print

        def replacement_print(*args, **kwargs):
            return None

        try:
            policy.builtins.print = replacement_print
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "FORMAT_VALUE"
            ):
                policy._canonical(formatted_callback)
        finally:
            policy.builtins.print = original_print

        def builtin_lookup():
            return len(())

        first_builtin = types.FunctionType(
            builtin_lookup.__code__,
            {"__builtins__": {"len": lambda value: 1}},
        )
        second_builtin = types.FunctionType(
            builtin_lookup.__code__,
            {"__builtins__": {"len": lambda value: 2}},
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_builtin)),
            policy._sha256_json(policy._canonical(second_builtin)),
        )

        dynamic_namespace = {"x": 1}
        exec(
            "def dynamic_global_lookup():\n"
            "    return globals()['x']\n",
            dynamic_namespace,
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "dynamic namespace"
        ):
            policy._canonical(dynamic_namespace["dynamic_global_lookup"])

        def imported_module():
            import math

            return math.pi

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "implicit import"
        ):
            policy._canonical(imported_module)

        first_default_module = types.ModuleType("same_default_module")
        second_default_module = types.ModuleType("same_default_module")
        first_default_module.x = 1
        second_default_module.x = 2

        def default_module_callback(module=None):
            return module.x

        def nested_default_module_callback(module=None):
            def inner():
                return module.x

            return inner()

        for template in (
            default_module_callback,
            nested_default_module_callback,
        ):
            first_default = types.FunctionType(
                template.__code__,
                template.__globals__,
                argdefs=(first_default_module,),
            )
            second_default = types.FunctionType(
                template.__code__,
                template.__globals__,
                argdefs=(second_default_module,),
            )
            first_default.__signature__ = inspect.Signature()
            second_default.__signature__ = inspect.Signature()
            self.assertNotEqual(
                policy._sha256_json(policy._canonical(first_default)),
                policy._sha256_json(policy._canonical(second_default)),
            )

        first_metadata = types.FunctionType(
            builtin_lookup.__code__, builtin_lookup.__globals__, "first_name"
        )
        second_metadata = types.FunctionType(
            builtin_lookup.__code__, builtin_lookup.__globals__, "second_name"
        )
        second_metadata.__qualname__ = first_metadata.__qualname__
        second_metadata.__module__ = first_metadata.__module__
        second_metadata.__doc__ = "different documentation"
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_metadata)),
            policy._sha256_json(policy._canonical(second_metadata)),
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "hidden extension-base state"
        ):
            policy._canonical(random.Random(1))

        identity_value = types.SimpleNamespace(value=1)
        identity_callbacks = (
            lambda value=identity_value: id(value),
            lambda value=identity_value: hash(value),
            lambda value=identity_value: repr(value),
            lambda value=identity_value: value.__hash__(),
        )
        for callback in identity_callbacks:
            with self.subTest(
                identity_callback=callback
            ), self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(callback)

        for direct_identity_callable in (id, hash):
            with self.subTest(
                direct_identity_callable=direct_identity_callable
            ), self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(direct_identity_callable)
        bound_identity_value = object()
        for bound_identity_callable in (
            bound_identity_value.__hash__,
            bound_identity_value.__repr__,
            bound_identity_value.__str__,
        ):
            with self.subTest(
                bound_identity_callable=bound_identity_callable
            ), self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(bound_identity_callable)

        def dynamic_set():
            return next(iter(set(["aa", "bb", "cc"])))

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "nondeterministic builtin|BUILD_SET",
        ):
            policy._canonical(dynamic_set)

    def test_static_closure_owner_projection_binds_only_used_attributes(self):
        class Owner:
            def __init__(self):
                self.value = 1
                self.unrelated_opaque_state = object()

        owner = Owner()

        def callback():
            return owner.value

        first = policy._sha256_json(policy._canonical(callback))
        owner.unrelated_opaque_state = object()
        self.assertEqual(
            first, policy._sha256_json(policy._canonical(callback))
        )
        owner.value = 2
        self.assertNotEqual(
            first, policy._sha256_json(policy._canonical(callback))
        )

        def bare_owner_callback():
            return owner

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "object state cannot be canonicalized exactly",
        ):
            policy._canonical(bare_owner_callback)

    def test_module_dependency_scope_respects_bytecode_binding_kind(self):
        collision = types.ModuleType("gpmeep_scope_collision")
        namespace = {"collision": collision}
        exec(
            "def local_parameter_callback(collision):\n"
            "    return collision.missing_from_global_module\n",
            namespace,
        )
        local_parameter_callback = namespace["local_parameter_callback"]
        policy._canonical(local_parameter_callback)

        default_module = types.ModuleType("gpmeep_default_module")
        default_module.threshold = 1

        def default_module_callback(value=default_module):
            return value.threshold

        first = policy._sha256_json(policy._canonical(default_module_callback))
        default_module.threshold = 2
        self.assertNotEqual(
            first,
            policy._sha256_json(policy._canonical(default_module_callback)),
        )

    def test_immutable_singletons_and_signatures_are_semantic(self):
        self.assertEqual(
            policy._canonical(Ellipsis),
            {"singleton": "builtins.Ellipsis"},
        )
        self.assertEqual(
            policy._canonical(NotImplemented),
            {"singleton": "builtins.NotImplemented"},
        )
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    "value",
                    inspect.Parameter.POSITIONAL_ONLY,
                    default=Ellipsis,
                    annotation=int,
                )
            ],
            return_annotation=NotImplemented,
        )
        changed = signature.replace(return_annotation=str)
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(signature)),
            policy._sha256_json(policy._canonical(changed)),
        )

        self.assertNotEqual(
            policy._sha256_json(policy._canonical(re.compile("first"))),
            policy._sha256_json(policy._canonical(re.compile("second"))),
        )

        def enum_member(semantic_value):
            class LocalEnum(enum.Enum):
                MEMBER = 1

                def semantic(self):
                    return semantic_value

            return LocalEnum.MEMBER

        self.assertNotEqual(
            policy._sha256_json(policy._canonical(enum_member(1))),
            policy._sha256_json(policy._canonical(enum_member(2))),
        )

        class PathSubclass(type(pathlib.Path())):
            pass

        PathSubclass.__module__ = "pathlib"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "Path subclasses"
        ):
            policy._canonical(PathSubclass("same-path"))
        context_value = contextvars.ContextVar("gpmeep-test", default=1)
        context_before = policy._sha256_json(policy._canonical(context_value))
        reset_token = context_value.set(2)
        try:
            self.assertNotEqual(
                context_before,
                policy._sha256_json(policy._canonical(context_value)),
            )
        finally:
            context_value.reset(reset_token)

    def test_context_var_default_and_alias_topology_are_semantic(self):
        no_default = contextvars.ContextVar("gpmeep-context-alias")
        with_default = contextvars.ContextVar(
            "gpmeep-context-alias", default=7
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(no_default)),
            policy._sha256_json(policy._canonical(with_default)),
        )

        first = contextvars.ContextVar("gpmeep-context-alias")
        second = contextvars.ContextVar("gpmeep-context-alias")
        self.assertNotEqual(
            policy._sha256_json(policy._canonical([first, first])),
            policy._sha256_json(policy._canonical([first, second])),
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "locale-dependent"
        ):
            policy._canonical(re.compile(b"a", re.LOCALE))
        self.assertTrue(policy._sha256_json(policy._canonical(np.errstate)))

    def test_nested_fast_binding_does_not_reuse_outer_default(self):
        default_module = types.ModuleType("gpmeep_nested_default_module")

        def callback(value=default_module):
            def nested(value):
                return value.attribute_selected_at_runtime

            return nested

        policy._canonical(callback)

    def test_numpy_ufunc_error_context_is_bound_for_explicit_and_operator_use(self):
        numerator = np.array([1.0])
        denominator = np.array([0.0])

        def explicit_callback():
            return np.divide(numerator, denominator)

        def operator_callback():
            return numerator / denominator

        previous = np.seterr(divide="ignore")
        previous_bufsize = np.getbufsize()
        try:
            explicit_ignore = policy._sha256_json(
                policy._canonical(explicit_callback)
            )
            operator_ignore = policy._sha256_json(
                policy._canonical(operator_callback)
            )
            self.assertTrue(np.isinf(operator_callback()[0]))

            np.seterr(divide="raise")
            with self.assertRaises(FloatingPointError):
                operator_callback()
            self.assertNotEqual(
                explicit_ignore,
                policy._sha256_json(policy._canonical(explicit_callback)),
            )
            self.assertNotEqual(
                operator_ignore,
                policy._sha256_json(policy._canonical(operator_callback)),
            )

            np.seterr(divide="ignore")
            np.setbufsize(8192)
            buffer_8192 = policy._sha256_json(
                policy._canonical(operator_callback)
            )
            np.setbufsize(16384)
            self.assertNotEqual(
                buffer_8192,
                policy._sha256_json(policy._canonical(operator_callback)),
            )
        finally:
            np.setbufsize(previous_bufsize)
            np.seterr(**previous)

        helper_globals = policy._ORIGINAL_NUMPY_GETERR.__globals__
        original_helper = helper_globals["_get_extobj_dict"]
        original_extobj = original_helper()
        try:
            helper_globals["_get_extobj_dict"] = lambda: dict(original_extobj)
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "ufunc context differs",
            ):
                policy._canonical(operator_callback)
        finally:
            helper_globals["_get_extobj_dict"] = original_helper

    def test_numpy_array_layout_mutability_and_dtype_are_semantic(self):
        c_order = np.array([[1, 2], [3, 4]], order="C")
        f_order = np.array([[1, 2], [3, 4]], order="F")
        self.assertEqual(c_order.tolist(), f_order.tolist())
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(c_order)),
            policy._sha256_json(policy._canonical(f_order)),
        )

        writable = np.array([1, 2, 3])
        readonly = writable.copy()
        readonly.setflags(write=False)
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(writable)),
            policy._sha256_json(policy._canonical(readonly)),
        )

        first_dtype = np.dtype([("x", "<i4"), ("pad", "<i4")])
        second_dtype = np.dtype([("x", "<i8")])
        first = np.array([(1, 2)], dtype=first_dtype)
        second = np.array([(8589934593,)], dtype=second_dtype)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first)),
            policy._sha256_json(policy._canonical(second)),
        )

        scalar_x = np.array([(1,)], dtype=[("x", "<i4")])[0]
        scalar_y = np.array([(1,)], dtype=[("y", "<i4")])[0]
        self.assertEqual(scalar_x.item(), scalar_y.item())
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(scalar_x)),
            policy._sha256_json(policy._canonical(scalar_y)),
        )

        original_ascontiguousarray = np.ascontiguousarray
        try:
            np.ascontiguousarray = lambda value: np.zeros(
                value.shape, dtype=value.dtype
            )
            self.assertNotEqual(
                policy._sha256_json(policy._canonical(np.array([1]))),
                policy._sha256_json(policy._canonical(np.array([2]))),
            )
        finally:
            np.ascontiguousarray = original_ascontiguousarray

        string_dtype = getattr(np.dtypes, "StringDType", None)
        if string_dtype is not None:
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "dtype is malformed"
            ):
                policy._canonical(np.array(["value"], dtype=string_dtype()))

        class ScalarSubclass(np.int64):
            __slots__ = ()

        ScalarSubclass.__module__ = "numpy.fake"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "scalar subclasses"
        ):
            policy._canonical(ScalarSubclass(1))

        first_nan = np.array([0x7FF8000000000001], dtype=np.uint64).view(
            np.float64
        )[0]
        second_nan = np.array([0x7FF8000000000002], dtype=np.uint64).view(
            np.float64
        )[0]
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_nan)),
            policy._sha256_json(policy._canonical(second_nan)),
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(float("nan"))),
            policy._sha256_json(policy._canonical(-float("nan"))),
        )

    def test_numpy_array_view_alias_topology_is_semantic(self):
        shared_base = np.array([0, 0])
        shared = (shared_base[:1], shared_base[:1])
        independent = (np.array([0, 0])[:1], np.array([0, 0])[:1])
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(shared)),
            policy._sha256_json(policy._canonical(independent)),
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "non-ndarray exporters"
        ):
            policy._canonical(np.frombuffer(bytearray(8), dtype=np.uint8))

    def test_mutable_byte_buffers_fail_closed_and_bytes_are_exact(self):
        self.assertTrue(
            policy._sha256_json(policy._canonical(bytes([1])))
        )
        for value in (bytearray([1]), memoryview(bytearray([1]))):
            with self.subTest(type=type(value).__name__), self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "mutable buffer exporter state",
            ):
                policy._canonical(value)

    def test_recursive_receiver_projection_falls_back_without_collision(self):
        class Owner:
            def selected(self):
                return self.selected.__self__.value

        owner = Owner()
        owner.value = 1

        def callback():
            return owner.selected()

        first = policy._sha256_json(policy._canonical(callback))
        owner.value = 2
        self.assertNotEqual(
            first, policy._sha256_json(policy._canonical(callback))
        )

    def test_only_import_time_numpy_dispatchers_are_accepted(self):
        original = np.where
        first = policy._sha256_json(policy._canonical(original))
        old_signature = getattr(original, "__signature__", None)
        had_signature = hasattr(original, "__signature__")
        try:
            original.__signature__ = inspect.Signature(
                [inspect.Parameter("condition", inspect.Parameter.POSITIONAL_ONLY)]
            )
            self.assertNotEqual(
                first, policy._sha256_json(policy._canonical(original))
            )
        finally:
            if had_signature:
                original.__signature__ = old_signature
            else:
                del original.__signature__

        runtime_ufunc = np.frompyfunc(lambda value: value + 1, 1, 1)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "not an original import-time"
        ):
            policy._canonical(runtime_ufunc)

    def test_numpy_hidden_dispatcher_and_export_are_sealed(self):
        original = np.zeros_like
        snapshot = policy._ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHERS[
            id(original)
        ]
        hidden = snapshot["hidden"]
        self.assertIsNotNone(hidden)

        def replacement(a, dtype=None, order=None, subok=None, shape=None):
            return (a,)

        old_code = hidden["function"].__code__
        try:
            hidden["function"].__code__ = replacement.__code__
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "hidden dispatcher differs"
            ):
                policy._canonical(original)
        finally:
            hidden["function"].__code__ = old_code

        old_export = np.zeros_like
        try:
            np.zeros_like = np.ones_like
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "export differs"
            ):
                policy._canonical(original)
        finally:
            np.zeros_like = old_export

        dispatcher_type = type(original)

        def implementation(*args, **kwargs):
            return args, kwargs

        def first_hidden_dispatcher(*args, **kwargs):
            return ()

        def second_hidden_dispatcher(*args, **kwargs):
            return (0,)

        forged = (
            dispatcher_type(first_hidden_dispatcher, implementation),
            dispatcher_type(second_hidden_dispatcher, implementation),
        )
        for candidate in forged:
            candidate.__module__ = "numpy"
            candidate.__name__ = "where"
            candidate.__qualname__ = "where"
            candidate.__wrapped__ = implementation
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "not an original import-time dependency",
            ):
                policy._canonical(candidate)

    def test_transitive_module_and_iterator_state_invalidate_challenge(self):
        iterator = iter([1, 2, 3])
        generator = (value for value in (1, 2, 3))
        closure_module = types.ModuleType(
            "gpmeep_backend_policy_closure_module"
        )
        closure_module.threshold = 1

        def iterator_callback():
            return next(iterator)

        def generator_callback():
            return next(generator)

        def closure_module_callback():
            return closure_module.threshold

        mp = MP("cuda")
        simulation = Simulation(100)
        cases = (
            (
                _transitive_state_callback,
                lambda: _TRANSITIVE_CALLBACK_GLOBAL.update(value=2),
            ),
            (
                _module_state_callback,
                lambda: setattr(_CALLBACK_MODULE, "threshold", 2),
            ),
            (iterator_callback, lambda: next(iterator)),
            (
                closure_module_callback,
                lambda: setattr(closure_module, "threshold", 2),
            ),
        )
        for callback, mutate in cases:
            with self.subTest(callback=callback):
                policy._reset_for_testing()
                _TRANSITIVE_CALLBACK_GLOBAL["value"] = 1
                _CALLBACK_MODULE.threshold = 1
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(mp, simulation, callback, None, (callback,))
                old = first.exception.challenge
                policy.acknowledge_backend_mismatch(
                    old, "mutation must invalidate this exact acknowledgement"
                )
                mutate()
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(mp, simulation, callback, None, (callback,))
                self.assertNotEqual(old, changed.exception.challenge)

        policy._reset_for_testing()
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(
                mp, simulation, generator_callback, None, (generator_callback,)
            )
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "suspended generator must fail closed",
        )
        next(generator)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "suspended or running generator"
        ):
            policy.guard_run(
                mp, simulation, generator_callback, None, (generator_callback,)
            )

    def test_opaque_extension_style_callable_fails_closed(self):
        class OpaqueCallable:
            __slots__ = ()

            def __call__(self):
                return True

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "callable state cannot be canonicalized exactly",
        ):
            policy.guard_run(
                MP("cuda"),
                Simulation(100),
                OpaqueCallable(),
                None,
            )

    def test_bound_builtin_owner_state_invalidates_acknowledgement(self):
        factories = (
            (
                "list.__len__",
                lambda: (
                    (items := [1]).__len__,
                    lambda: items.append(2),
                ),
            ),
            (
                "dict.copy",
                lambda: (
                    (items := {"value": 1}).copy,
                    lambda: items.update(value=2),
                ),
            ),
            (
                "deque.copy",
                lambda: (
                    (items := collections.deque([1])).copy,
                    lambda: items.append(2),
                ),
            ),
        )
        for name, factory in factories:
            with self.subTest(callback=name):
                policy._reset_for_testing()
                callback, mutate = factory()
                mp = MP("cuda")
                simulation = Simulation(100)
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "bound owner mutation must invalidate acknowledgement",
                )
                mutate()
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

    def test_class_slots_function_and_subclass_state_invalidate_acknowledgement(self):
        class ClassState:
            threshold = 1

        def class_callback():
            return ClassState.threshold

        class CallableClassState:
            threshold = 1

            def __call__(self):
                return self.threshold

        class BaseCallableState:
            threshold = 1

        class InheritedCallableState(BaseCallableState):
            def __call__(self):
                return self.threshold

        class MixedSlotsCallable:
            __slots__ = ("hidden", "__dict__")

            def __init__(self):
                self.hidden = 1

            def __call__(self):
                return self.hidden

        class SlottedDependency:
            __slots__ = ("value",)

            def __init__(self):
                self.value = 1

        slotted = SlottedDependency()

        def slotted_callback():
            return slotted.value

        def function_state_callback():
            return function_state_callback.state

        function_state_callback.state = 1

        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class IntSubclass(int):
            pass

        class PartialSubclass(functools.partial):
            __slots__ = ()

        dict_subclass = DictSubclass(value=1)
        dict_subclass.bias = 0
        list_subclass = ListSubclass([1])
        list_subclass.bias = 0
        int_subclass = IntSubclass(1)
        int_subclass.bias = 0

        cases = (
            (class_callback, lambda: setattr(ClassState, "threshold", 2)),
            (
                CallableClassState(),
                lambda: setattr(CallableClassState, "threshold", 2),
            ),
            (
                InheritedCallableState(),
                lambda: setattr(BaseCallableState, "threshold", 2),
            ),
            (
                (mixed := MixedSlotsCallable()),
                lambda: setattr(mixed, "hidden", 2),
            ),
            (slotted_callback, lambda: setattr(slotted, "value", 2)),
            (
                function_state_callback,
                lambda: setattr(function_state_callback, "state", 2),
            ),
        )
        for callback, mutate in cases:
            with self.subTest(callback=callback):
                policy._reset_for_testing()
                mp = MP("cuda")
                simulation = Simulation(100)
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "object state mutation must invalidate acknowledgement",
                )
                mutate()
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "dict subclasses"
        ):
            policy._canonical(dict_subclass)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "partial subclasses"
        ):
            policy._canonical(PartialSubclass(int))
        for subclass_value in (list_subclass, int_subclass):
            with self.subTest(
                unsupported_subclass=type(subclass_value).__name__
            ), self.assertRaisesRegex(
                policy.BackendRecommendationError, "subclasses"
            ):
                policy._canonical(subclass_value)
        class BytearraySubclass(bytearray):
            pass

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "mutable buffer exporter"
        ):
            policy._canonical(BytearraySubclass(b"value"))

        class HiddenDictionary:
            def __getattribute__(self, name):
                if name == "__dict__":
                    return {}
                return object.__getattribute__(self, name)

        first_hidden = HiddenDictionary()
        second_hidden = HiddenDictionary()
        first_hidden.value = 1
        second_hidden.value = 2
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_hidden)),
            policy._sha256_json(policy._canonical(second_hidden)),
        )

        @functools.lru_cache()
        def cached(value):
            return value

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "hidden extension state"
        ):
            policy._canonical(cached)

        first_order = type("OrderedClass", (), {"a": 1, "b": 2})
        second_order = type("OrderedClass", (), {"b": 2, "a": 1})
        first_order.__module__ = second_order.__module__ = "same_module"
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_order)),
            policy._sha256_json(policy._canonical(second_order)),
        )

        def class_with_metaclass(semantic_value):
            class LocalMetaclass(type):
                def __getattribute__(cls, name):
                    if name == "semantic_value":
                        return semantic_value
                    return super().__getattribute__(name)

            return LocalMetaclass(
                "MetaclassOwner", (), {"semantic_value": 0}
            )

        self.assertNotEqual(
            policy._sha256_json(
                policy._canonical(class_with_metaclass(1))
            ),
            policy._sha256_json(
                policy._canonical(class_with_metaclass(2))
            ),
        )

        class HidingMetaclass(type):
            def __getattribute__(cls, name):
                if name == "__dict__":
                    return types.MappingProxyType({})
                return type.__getattribute__(cls, name)

        first_hidden_class = HidingMetaclass(
            "HiddenClass", (), {"value": 1}
        )
        second_hidden_class = HidingMetaclass(
            "HiddenClass", (), {"value": 2}
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_hidden_class)),
            policy._sha256_json(policy._canonical(second_hidden_class)),
        )

        self.assertTrue(policy._canonical(collections.abc.Sequence))

        class LocalABC(metaclass=abc.ABCMeta):
            pass

        class AbstractPair(metaclass=abc.ABCMeta):
            @abc.abstractmethod
            def second(self):
                pass

            @abc.abstractmethod
            def first(self):
                pass

        class VirtualSequence:
            pass

        abc_before = policy._sha256_json(policy._canonical(LocalABC))
        abstract_pair = policy._class_token(
            AbstractPair, policy._CanonicalTraversal()
        )
        abstract_names_token = next(
            attribute["value"]
            for attribute in abstract_pair["attributes"]
            if attribute["name"] == "__abstractmethods__"
        )
        self.assertEqual(
            abstract_names_token["value"]["set"]["items"],
            ["first", "second"],
        )

        abstract_names_alias = AbstractPair.__abstractmethods__
        abstract_names_copy = frozenset(list(abstract_names_alias))
        self.assertIsNot(abstract_names_alias, abstract_names_copy)

        def identity_callback(candidate):
            def callback():
                return candidate is AbstractPair.__abstractmethods__

            return callback

        self.assertNotEqual(
            policy._sha256_json(
                policy._canonical(identity_callback(abstract_names_alias))
            ),
            policy._sha256_json(
                policy._canonical(identity_callback(abstract_names_copy))
            ),
        )
        isinstance(VirtualSequence(), LocalABC)
        self.assertEqual(
            abc_before,
            policy._sha256_json(policy._canonical(LocalABC)),
        )
        LocalABC.register(VirtualSequence)
        self.assertNotEqual(
            abc_before,
            policy._sha256_json(policy._canonical(LocalABC)),
        )
        original_abc_impl = LocalABC.__dict__["_abc_impl"]
        LocalABC._abc_impl = object()
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "ABC registry implementation cannot be authenticated exactly",
        ):
            policy._canonical(LocalABC)
        LocalABC._abc_impl = original_abc_impl

        enum_before = policy._sha256_json(policy._canonical(enum.Enum))
        original_enum_call_code = enum.EnumType.__call__.__code__
        try:
            def replacement_enum_call(cls, value, names=None, **kwargs):
                return 123

            enum.EnumType.__call__.__code__ = replacement_enum_call.__code__
            with self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(enum.Enum)
        finally:
            enum.EnumType.__call__.__code__ = original_enum_call_code
        self.assertEqual(
            enum_before,
            policy._sha256_json(policy._canonical(enum.Enum)),
        )

        original_enum_dict_setitem_code = enum._EnumDict.__setitem__.__code__
        try:
            class ReplacementEnumDict:
                def replacement(self, key, value):
                    __class__
                    return dict.__setitem__(self, key, value)

            enum._EnumDict.__setitem__.__code__ = (
                ReplacementEnumDict.replacement.__code__
            )
            with self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(enum.Enum)
        finally:
            enum._EnumDict.__setitem__.__code__ = (
                original_enum_dict_setitem_code
            )

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "must be captured through a statically projected method",
        ):
            policy._canonical(unittest.TestCase)

        original_assert_true_code = unittest.TestCase.assertTrue.__code__
        try:
            def replacement_assert_true(self, expression, message=None):
                return "modified"

            unittest.TestCase.assertTrue.__code__ = (
                replacement_assert_true.__code__
            )
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "trusted standard-library class .* was modified",
            ):
                policy._canonical(unittest.TestCase)
        finally:
            unittest.TestCase.assertTrue.__code__ = original_assert_true_code

        original_safe_repr_code = unittest.case.safe_repr.__code__
        assert_true_before = policy._sha256_json(
            policy._canonical(unittest.TestCase.assertTrue)
        )
        try:
            def replacement_safe_repr(value, short=False):
                return "modified"

            unittest.case.safe_repr.__code__ = replacement_safe_repr.__code__
            self.assertNotEqual(
                assert_true_before,
                policy._sha256_json(
                    policy._canonical(unittest.TestCase.assertTrue)
                ),
            )
        finally:
            unittest.case.safe_repr.__code__ = original_safe_repr_code

        assert_regex_before = policy._sha256_json(
            policy._canonical(unittest.TestCase.assertRegex)
        )
        original_escape = re._parser.ESCAPES["\\n"]
        try:
            re._parser.ESCAPES["\\n"] = (
                re._constants.LITERAL,
                ord("X"),
            )
            self.assertNotEqual(
                assert_regex_before,
                policy._sha256_json(
                    policy._canonical(unittest.TestCase.assertRegex)
                ),
            )
        finally:
            re._parser.ESCAPES["\\n"] = original_escape

        assert_regex_before_cache_fill = policy._sha256_json(
            policy._canonical(unittest.TestCase.assertRegex)
        )
        re.compile("gpmeep-unrelated-cache-entry-unique-20260901")
        self.assertEqual(
            assert_regex_before_cache_fill,
            policy._sha256_json(
                policy._canonical(unittest.TestCase.assertRegex)
            ),
        )

        # Warning-capture helpers resolve mutable ambient ``sys.modules``
        # state. They are intentionally outside the exact callback contract;
        # numerical TestCase assertions with statically projectable receiver
        # state remain supported. Dynamic assertion dispatch (assertEqual)
        # and warning-capture helpers are intentionally fail-closed.
        with self.assertRaises(policy.BackendRecommendationError):
            policy._canonical(unittest.TestCase.assertWarns)
        with self.assertRaises(policy.BackendRecommendationError):
            policy._canonical(unittest.TestCase().assertEqual)

        unordered = policy._trusted_dependency_value_token(
            frozenset({"zeta", "alpha", "middle"}),
            policy._CanonicalTraversal(),
        )
        unordered_items = unordered["value"]["frozenset"]
        self.assertEqual(
            unordered_items,
            sorted(
                unordered_items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            ),
        )

    def test_authentic_lazy_slotnames_cache_is_stable_and_tampering_rejects(self):
        class Slotted:
            __slots__ = ("value", "__hidden")

        snapshot = dict(Slotted.__dict__)
        self.assertTrue(
            policy._class_members_match_dependency_snapshot(
                Slotted, Slotted.__dict__, snapshot
            )
        )
        self.assertEqual(
            copyreg._slotnames(Slotted), ["value", "_Slotted__hidden"]
        )
        self.assertTrue(
            policy._class_members_match_dependency_snapshot(
                Slotted, Slotted.__dict__, snapshot
            )
        )
        Slotted.__slotnames__.append("forged")
        self.assertFalse(
            policy._class_members_match_dependency_snapshot(
                Slotted, Slotted.__dict__, snapshot
            )
        )

    def test_lazy_graph_extension_does_not_retrust_existing_mutation(self):
        class Existing:
            def value(self):
                return "original"

        class LateLoaded(Existing):
            def extra(self):
                return "late"

        graph = policy._trusted_dependency_graph_snapshot((Existing,))
        original = Existing.value

        def observer(self):
            return "instrumented"

        Existing.value = observer
        try:
            extended = policy._extend_trusted_dependency_graph(
                graph, (LateLoaded,)
            )
            for function, state in graph["functions"].items():
                self.assertIs(extended["functions"][function], state)
            for candidate, state in graph["classes"].items():
                self.assertIs(extended["classes"][candidate], state)
            self.assertEqual(
                extended["bindings"][: len(graph["bindings"])],
                graph["bindings"],
            )
            for owner, dependencies in graph["edges"].items():
                self.assertIs(extended["edges"][owner], dependencies)
            self.assertIn(LateLoaded, extended["classes"])
            self.assertIn(Existing, extended["edges"][LateLoaded])
            self.assertIs(
                extended["classes"][Existing]["members"]["value"],
                original,
            )
            self.assertFalse(
                policy._trusted_dependency_graph_is_unchanged(
                    extended, root=Existing
                )
            )
            self.assertFalse(
                policy._trusted_dependency_graph_is_unchanged(
                    extended, root=LateLoaded
                )
            )
        finally:
            Existing.value = original
        self.assertTrue(
            policy._trusted_dependency_graph_is_unchanged(
                extended, root=Existing
            )
        )
        self.assertTrue(
            policy._trusted_dependency_graph_is_unchanged(
                extended, root=LateLoaded
            )
        )

    def test_descriptor_and_annotation_state_invalidate_acknowledgement(self):
        descriptor_state = {"value": 1}

        class DescriptorState:
            @staticmethod
            def static_value():
                return descriptor_state["value"]

            @classmethod
            def class_value(cls):
                return descriptor_state["value"]

            @property
            def property_value(self):
                return descriptor_state["value"]

        descriptor_object = DescriptorState()
        callbacks = (
            lambda: DescriptorState.static_value(),
            lambda: DescriptorState.class_value(),
            lambda: descriptor_object.property_value,
        )
        for callback in callbacks:
            with self.subTest(descriptor_callback=callback):
                policy._reset_for_testing()
                descriptor_state["value"] = 1
                mp = MP("cuda")
                simulation = Simulation(100)
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(mp, simulation, callback, None, (callback,))
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "descriptor dependency mutation must invalidate acknowledgement",
                )
                descriptor_state["value"] = 2
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(mp, simulation, callback, None, (callback,))
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

        class MutableAnnotation:
            value = 1

        def annotated_callback():
            return annotated_callback.__annotations__["return"].value

        annotated_callback.__annotations__ = {"return": MutableAnnotation}
        policy._reset_for_testing()
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(
                mp,
                simulation,
                annotated_callback,
                None,
                (annotated_callback,),
            )
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "annotation state mutation must invalidate acknowledgement",
        )
        MutableAnnotation.value = 2
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(
                mp,
                simulation,
                annotated_callback,
                None,
                (annotated_callback,),
            )
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)

        first_annotation = type(
            "SameAnnotation", (), {"semantic_value": 1}
        )
        second_annotation = type(
            "SameAnnotation", (), {"semantic_value": 2}
        )
        first_annotation.__module__ = "same_annotation_module"
        second_annotation.__module__ = "same_annotation_module"

        def annotation_holder():
            return None

        annotation_holder.__annotations__ = {"return": first_annotation}
        first_annotation_token = policy._sha256_json(
            policy._canonical(annotation_holder)
        )
        annotation_holder.__annotations__ = {"return": second_annotation}
        self.assertNotEqual(
            first_annotation_token,
            policy._sha256_json(policy._canonical(annotation_holder)),
        )
        annotation_holder.__annotations__ = {1: first_annotation}
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "annotation keys"
        ):
            policy._canonical(annotation_holder)

        def trusted_annotated_member(
            value: typing.Optional[int],
            handler: typing.Callable[[float], float] | None = None,
        ) -> typing.Union[str, bytes]:
            return value

        trusted_forward_ref = typing.ForwardRef("TrustedThing")
        trusted_annotated_member.__annotations__["forward"] = (
            trusted_forward_ref
        )
        trusted_snapshot = policy._trusted_function_annotation_snapshots(
            ({"member": trusted_annotated_member},)
        )
        policy._TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS = {
            **policy._TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS,
            **trusted_snapshot,
        }
        policy._shallow_function_token(
            trusted_annotated_member,
            policy._CanonicalTraversal(),
            transitive=False,
        )
        trusted_alias = trusted_annotated_member.__annotations__["return"]
        trusted_alias_arguments = trusted_alias.__args__
        try:
            trusted_alias.__args__ = (int, int)
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "import-time state"
            ):
                policy._shallow_function_token(
                    trusted_annotated_member,
                    policy._CanonicalTraversal(),
                    transitive=False,
                )
        finally:
            trusted_alias.__args__ = trusted_alias_arguments
        class SpoofedFloat:
            pass

        SpoofedFloat.__module__ = "builtins"
        SpoofedFloat.__qualname__ = "float"
        try:
            trusted_alias.__args__ = (SpoofedFloat, SpoofedFloat)
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "import-time state"
            ):
                policy._shallow_function_token(
                    trusted_annotated_member,
                    policy._CanonicalTraversal(),
                    transitive=False,
                )
        finally:
            trusted_alias.__args__ = trusted_alias_arguments
        try:
            trusted_forward_ref.__forward_evaluated__ = True
            trusted_forward_ref.__forward_value__ = int
            with self.assertRaisesRegex(
                policy.BackendRecommendationError, "import-time state"
            ):
                policy._shallow_function_token(
                    trusted_annotated_member,
                    policy._CanonicalTraversal(),
                    transitive=False,
                )
        finally:
            trusted_forward_ref.__forward_evaluated__ = False
            trusted_forward_ref.__forward_value__ = None
        trusted_annotated_member.__annotations__["return"] = bytes
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "import-time state"
        ):
            policy._shallow_function_token(
                trusted_annotated_member,
                policy._CanonicalTraversal(),
                transitive=False,
            )

        class PropertySubclass(property):
            pass

        class DescriptorOwner:
            value = PropertySubclass(lambda self: 1)

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "property subclasses"
        ):
            policy._canonical(DescriptorOwner)

        class SpoofedIterator:
            __slots__ = ()

            def __iter__(self):
                return self

            def __next__(self):
                return 1

            def __reduce__(self):
                return iter, ((),)

        SpoofedIterator.__module__ = "builtins"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "iterator type"
        ):
            policy._canonical(SpoofedIterator())

        class SlotOwner:
            __slots__ = ("x", "y")

        first_descriptor_owner = type(
            "DescriptorAlias", (SlotOwner,), {"d": SlotOwner.x}
        )
        second_descriptor_owner = type(
            "DescriptorAlias", (SlotOwner,), {"d": SlotOwner.y}
        )
        self.assertNotEqual(
            policy._sha256_json(
                policy._canonical(first_descriptor_owner)
            ),
            policy._sha256_json(
                policy._canonical(second_descriptor_owner)
            ),
        )

        class SlotHidingMetaclass(type):
            def __getattribute__(cls, name):
                if name == "__dict__":
                    return types.MappingProxyType({})
                if name in ("__module__", "__qualname__"):
                    return "spoofed_slot_owner"
                return type.__getattribute__(cls, name)

        class HiddenSlot(metaclass=SlotHidingMetaclass):
            __slots__ = ("value",)

            def __init__(self, value):
                self.value = value

        self.assertNotEqual(
            policy._sha256_json(policy._canonical(HiddenSlot(1))),
            policy._sha256_json(policy._canonical(HiddenSlot(2))),
        )

        SemanticRange = collections.namedtuple(
            "SemanticRange", ("minimum", "maximum")
        )
        self.assertNotEqual(
            policy._sha256_json(
                policy._canonical(SemanticRange(1.0, 2.0))
            ),
            policy._sha256_json(
                policy._canonical(SemanticRange(1.0, 3.0))
            ),
        )

        ClosureRange = collections.namedtuple(
            "ClosureRange", ("minimum", "maximum")
        )
        original_make = ClosureRange._make.__func__

        def closure_cell(value):
            def capture():
                return value

            return capture.__closure__[0]

        forged_make = types.FunctionType(
            original_make.__code__,
            original_make.__globals__,
            "_make",
            None,
            (
                closure_cell(len),
                closure_cell(object()),
                closure_cell(tuple.__new__),
            ),
        )
        forged_make.__qualname__ = "ClosureRange._make"
        forged_make.__module__ = "collections"
        ClosureRange._make = classmethod(forged_make)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "formatted member closure is not authentic",
        ):
            policy._canonical(ClosureRange)

        def identity_repr(self):
            return f"{object()}"

        SemanticRange.__repr__ = identity_repr
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "FORMAT_VALUE"
        ):
            policy._canonical(SemanticRange)

        def wrapped_member():
            return 1

        first_static = staticmethod(wrapped_member)
        second_static = staticmethod(wrapped_member)
        second_static.semantic_state = 2
        first_wrapper_owner = type(
            "WrapperOwner", (), {"member": first_static}
        )
        second_wrapper_owner = type(
            "WrapperOwner", (), {"member": second_static}
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_wrapper_owner)),
            policy._sha256_json(policy._canonical(second_wrapper_owner)),
        )

        first_property_owner = type(
            "PropertyOwner",
            (),
            {"member": property(wrapped_member, doc="first documentation")},
        )
        second_property_owner = type(
            "PropertyOwner",
            (),
            {"member": property(wrapped_member, doc="second documentation")},
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_property_owner)),
            policy._sha256_json(policy._canonical(second_property_owner)),
        )

    def test_simulation_semantic_state_and_file_bytes_are_bound(self):
        cases = (
            ("init_sim_hooks", [], [lambda: None]),
            ("load_single_parallel_file", True, False),
            ("output_single_precision", False, True),
            ("output_volume", None, Vector(1, 1, 0)),
            ("progress_interval", 4, 8),
            ("global_d_conductivity", 0, 0.25),
            ("global_b_conductivity", 0, 0.5),
        )
        for name, before, after in cases:
            with self.subTest(simulation_state=name):
                policy._reset_for_testing()
                mp = MP("cuda")
                simulation = Simulation(100)
                setattr(simulation, name, before)
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(mp, simulation, 10, None)
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "Simulation state mutation must invalidate acknowledgement",
                )
                setattr(simulation, name, after)
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(mp, simulation, 10, None)
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "epsilon.h5"
            path.write_bytes(b"scientific state")
            self.assertNotEqual(
                policy._file_input_manifest(
                    "epsilon_input_file", str(path)
                ),
                policy._file_input_manifest(
                    "epsilon_input_file", f"{path}:"
                ),
            )

        for name in (
            "epsilon_input_file",
            "load_fields_file",
            "load_structure_file",
        ):
            with self.subTest(file_input=name), tempfile.TemporaryDirectory() as root:
                policy._reset_for_testing()
                path = pathlib.Path(root) / "state.h5"
                path.write_bytes(b"first scientific state")
                mp = MP("cuda")
                simulation = Simulation(100)
                setattr(simulation, name, str(path))
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(mp, simulation, 10, None)
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "file content mutation must invalidate acknowledgement",
                )
                path.write_bytes(b"second scientific state")
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(mp, simulation, 10, None)
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

    def test_initialized_native_epoch_is_exact_on_retry_then_policy_scope_reuses(self):
        policy._reset_for_testing()
        mp = MP("cuda")
        simulation = MeepSimulation(100)
        simulation._is_initialized = True
        simulation.fields = MeepFields()
        simulation.structure = MeepStructure()
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "unchanged initialized native state is an intentional CUDA run",
        )
        simulation.fields.mutate()
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                first.exception.challenge,
                "mutated native state cannot consume stale acknowledgement",
            )
        policy.acknowledge_backend_mismatch(
            changed.exception.challenge,
            "changed native state remains an intentional CUDA run",
        )
        accepted = policy.guard_run(mp, simulation, 10, None)
        self.assertTrue(accepted["acknowledgement"])
        self.assertFalse(accepted["acknowledgement_reused"])

        simulation.fields.mutate()
        reused = policy.guard_run(mp, simulation, 10, None)
        self.assertTrue(reused["acknowledgement"])
        self.assertTrue(reused["acknowledgement_reused"])
        self.assertFalse(reused["workload_fingerprint_evaluated"])

    def test_direct_native_handle_python_state_is_bound_by_collective_digest(self):
        for owner_name in ("fields", "structure"):
            with self.subTest(owner=owner_name):
                policy._reset_for_testing()
                mp = MP("cuda")
                simulation = MeepSimulation(100)
                simulation._is_initialized = True
                simulation.fields = MeepFields()
                simulation.structure = MeepStructure()
                owner = getattr(simulation, owner_name)
                owner.user_semantic = 1

                def make_callback(root, selected_owner):
                    if selected_owner == "fields":
                        def callback():
                            return root.fields.user_semantic
                    else:
                        def callback():
                            return root.structure.user_semantic
                    return callback

                callback = make_callback(simulation, owner_name)
                with self.assertRaises(
                    policy.BackendRecommendationMismatch
                ) as first:
                    policy.guard_run(
                        mp, simulation, 10, None, (callback,)
                    )
                owner.user_semantic = 2
                with self.assertRaises(
                    policy.BackendRecommendationMismatch
                ) as changed:
                    policy.guard_run(
                        mp, simulation, 10, None, (callback,)
                    )
                self.assertNotEqual(
                    first.exception.challenge,
                    changed.exception.challenge,
                )

    def test_initialized_native_source_cycle_manifest_fails_closed(self):
        class CyclicNativeFields:
            @staticmethod
            def gpmeep_state_manifest():
                return json.dumps(
                    {
                        "schema": "gpmeep-fields-state-v2",
                        "epoch": 1,
                        "external_source_count": 0,
                        "sources": [{"cycle": True}],
                    },
                    sort_keys=True,
                )

        simulation = types.SimpleNamespace(fields=CyclicNativeFields())
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "native source list is malformed",
        ):
            policy._native_fields_state(simulation)

    def test_root_native_state_is_not_duplicated_through_self_reference(self):
        root = MeepSimulation(100)
        root.fields = MeepFields()
        root.structure = MeepStructure()
        root.geometry.append(root)
        before = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        root.fields.mutate()
        after = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        self.assertEqual(before, after)

    def test_root_native_state_is_not_duplicated_through_static_projection(self):
        root = MeepSimulation(100)
        root.fields = MeepFields()
        root.structure = MeepStructure()

        def hook():
            return root.fields

        root.init_sim_hooks = [hook]
        before = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        root.fields.mutate()
        after = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        self.assertEqual(before, after)

        root._gpmeep_callback_state = {"semantic": 1}

        class Holder:
            def __init__(self):
                self.simulation = root
                self.unrelated = object()

        holder = Holder()

        def nested_holder_hook():
            return holder.simulation.fields

        policy._workload_record(
            root, 10, None, (nested_holder_hook,), None
        )

        root.fields.user_semantic = 1

        def nested_native_state_hook():
            return holder.simulation.fields.user_semantic

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "escaped nested static callback projection",
        ):
            policy._workload_record(
                root, 10, None, (nested_native_state_hook,), None
            )

        def aliased_native_state_hook():
            fields = holder.simulation.fields
            return fields.user_semantic

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "escaped nested static callback projection",
        ):
            policy._workload_record(
                root, 10, None, (aliased_native_state_hook,), None
            )

        def escaped_holder_hook():
            return holder.simulation

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "escaped nested static callback projection",
        ):
            policy._workload_record(
                root, 10, None, (escaped_holder_hook,), None
            )

        def internal_hook():
            return root._gpmeep_callback_state["semantic"]

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "runtime state that is not bound",
        ):
            policy._workload_record(root, 10, None, (internal_hook,), None)

        for name in sorted(policy._SIMULATION_UNBOUND_RUNTIME_FIELDS):
            with self.subTest(runtime_field=name):
                setattr(root, name, {"semantic": 1})

                def dynamic_runtime_hook(simulation=root, field=name):
                    return getattr(simulation, field)["semantic"]

                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "cannot be projected exactly",
                ):
                    policy._workload_record(
                        root, 10, None, (dynamic_runtime_hook,), None
                    )

        nested_root = [root]

        def nested_root_hook(container=nested_root):
            return container[0].absorber_vols["semantic"]

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "escaped static callback projection",
        ):
            policy._workload_record(root, 10, None, (nested_root_hook,), None)

        root.alias = root

        def alias_hook():
            return root.alias.absorber_vols["semantic"]

        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "escaped static callback projection",
        ):
            policy._workload_record(root, 10, None, (alias_hook,), None)

        def runtime_method(self):
            return self.absorber_vols["semantic"]

        MeepSimulation.runtime_method = runtime_method
        try:
            def method_hook():
                return root.runtime_method()

            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "runtime state that is not bound",
            ):
                policy._workload_record(root, 10, None, (method_hook,), None)
        finally:
            del MeepSimulation.runtime_method

    def test_inherited_method_receiver_projects_only_used_state(self):
        class Base:
            def selected(self):
                return self.value

        class Owner(Base):
            pass

        owner = Owner()
        owner.value = 1
        owner.unrelated = object()

        def callback():
            return owner.selected()

        first = policy._sha256_json(policy._canonical(callback))
        owner.unrelated = object()
        self.assertEqual(first, policy._sha256_json(policy._canonical(callback)))
        owner.value = 2
        self.assertNotEqual(first, policy._sha256_json(policy._canonical(callback)))

    def test_exact_meep_file_inputs_preserve_common_and_rank_local_routes(self):
        with tempfile.TemporaryDirectory() as root:
            first_path = pathlib.Path(root) / "rank00" / "structure.h5"
            second_path = pathlib.Path(root) / "rank01" / "structure.h5"
            first_path.parent.mkdir()
            second_path.parent.mkdir()
            first_path.write_bytes(b"rank zero scientific shard")
            second_path.write_bytes(b"rank one scientific shard")
            first = MeepSimulation(100)
            second = MeepSimulation(100)
            first.epsilon_input_file = str(first_path)
            second.epsilon_input_file = str(second_path)
            self.assertNotEqual(
                policy._sha256_json(
                    policy._workload_record(first, 10, None, (), None)
                ),
                policy._sha256_json(
                    policy._workload_record(second, 10, None, (), None)
                ),
            )
            del first.epsilon_input_file
            del second.epsilon_input_file
            first.load_structure_file = str(first_path)
            second.load_structure_file = str(second_path)
            first.load_single_parallel_file = True
            second.load_single_parallel_file = True
            self.assertNotEqual(
                policy._sha256_json(
                    policy._workload_record(first, 10, None, (), None)
                ),
                policy._sha256_json(
                    policy._workload_record(second, 10, None, (), None)
                ),
            )
            first.load_single_parallel_file = False
            second.load_single_parallel_file = False
            self.assertEqual(
                policy._sha256_json(
                    policy._workload_record(first, 10, None, (), None)
                ),
                policy._sha256_json(
                    policy._workload_record(second, 10, None, (), None)
                ),
            )
            first_native = policy._sha256_json(
                policy._native_simulation_state(first)
            )
            second_native = policy._sha256_json(
                policy._native_simulation_state(second)
            )
            self.assertNotEqual(first_native, second_native)
            first_path.write_bytes(b"mutated rank zero scientific shard")
            self.assertNotEqual(
                first_native,
                policy._sha256_json(policy._native_simulation_state(first)),
            )

    def test_missing_simulation_file_input_fails_collectively(self):
        simulation = MeepSimulation(100)
        simulation.load_fields_file = "/definitely/missing/gpmeep-fields.h5"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "distributed gpmeep workload preflight failed",
        ):
            policy.guard_run(MP("cpu"), simulation, 10, None)

    def test_file_and_chunk_layout_string_subclasses_fail_closed(self):
        class PathText(str):
            pass

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "state.h5"
            path.write_bytes(b"sealed scientific state")
            for field in (
                "epsilon_input_file",
                "load_fields_file",
                "load_structure_file",
                "chunk_layout",
            ):
                with self.subTest(field=field):
                    simulation = MeepSimulation(100)
                    setattr(simulation, field, PathText(str(path)))
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "exact string",
                    ):
                        policy._workload_record(
                            simulation, 10, None, (), None
                        )

    def test_materialized_file_input_uses_preinitialization_seal(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "epsilon.h5"
            path.write_bytes(b"materialized epsilon state")
            simulation = MeepSimulation(100)
            simulation.epsilon_input_file = str(path)
            before = policy._sha256_json(
                policy._workload_record(simulation, 10, None, (), None)
            )
            self.assertTrue(before)
            simulation.structure = MeepStructure()
            policy.bind_simulation_file_input_owners(
                simulation, ("epsilon_input_file",)
            )
            path.unlink()
            materialized = policy._workload_record(
                simulation, 10, None, (), None
            )
            self.assertIn("materialized_file_input", str(materialized))
            materialized_digest = policy._sha256_json(materialized)
            simulation.epsilon_input_file = ""
            self.assertEqual(
                materialized_digest,
                policy._sha256_json(
                    policy._workload_record(
                        simulation, 10, None, (), None
                    )
                ),
            )

    def test_collective_preinit_file_seal_supports_init_before_run_and_reset(self):
        class UnconfiguredSimulationSubclass(MeepSimulation):
            pass

        unconfigured = UnconfiguredSimulationSubclass(100)
        policy.seal_simulation_file_inputs(MP("cpu"), unconfigured)
        policy.bind_simulation_file_input_owners(
            unconfigured,
            ("epsilon_input_file", "load_structure_file"),
        )
        unconfigured.load_fields_file = "/not/allowed/on/a/subclass.h5"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "configured file inputs require an exact Meep Simulation",
        ):
            policy.seal_simulation_file_inputs(MP("cpu"), unconfigured)

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "epsilon.h5"
            path.write_bytes(b"pre-init material input")
            simulation = MeepSimulation(100)
            simulation.epsilon_input_file = str(path)
            policy.seal_simulation_file_inputs(MP("cpu"), simulation)
            simulation.structure = MeepStructure()
            policy.bind_simulation_file_input_owners(
                simulation, ("epsilon_input_file",)
            )
            path.unlink()
            self.assertIn(
                "materialized_file_input",
                str(policy._workload_record(simulation, 10, None, (), None)),
            )

            alternate = pathlib.Path(root) / "alternate-epsilon.h5"
            alternate.write_bytes(b"different pre-init material input")
            changed = MeepSimulation(100)
            changed.epsilon_input_file = str(path)
            path.write_bytes(b"pre-init material input")
            policy.seal_simulation_file_inputs(MP("cpu"), changed)
            changed.epsilon_input_file = str(alternate)
            changed.structure = MeepStructure()
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "changed after its pre-initialization seal",
            ):
                policy.bind_simulation_file_input_owners(
                    changed, ("epsilon_input_file",)
                )
            path.unlink()

            policy.clear_simulation_file_input_seals(simulation)
            simulation.structure = None
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "file-input sealing failed",
            ):
                policy.seal_simulation_file_inputs(MP("cpu"), simulation)

    def test_materialized_file_owner_is_bound_at_native_load(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "state.h5"
            path.write_bytes(b"owner-bound scientific state")
            for field, owner_field in (
                ("epsilon_input_file", "structure"),
                ("load_structure_file", "structure"),
                ("load_fields_file", "fields"),
            ):
                with self.subTest(field=field):
                    simulation = MeepSimulation(100)
                    setattr(simulation, field, str(path))
                    policy.seal_simulation_file_inputs(MP("cpu"), simulation)
                    setattr(simulation, owner_field, (
                        MeepFields()
                        if owner_field == "fields"
                        else MeepStructure()
                    ))
                    if owner_field == "fields":
                        simulation._is_initialized = True
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "no owner bound at native materialization",
                    ):
                        policy._workload_record(
                            simulation, 10, None, (), None
                        )
                    policy.bind_simulation_file_input_owners(
                        simulation, (field,)
                    )
                    self.assertTrue(
                        policy._sha256_json(
                            policy._workload_record(
                                simulation, 10, None, (), None
                            )
                        )
                    )
                    setattr(simulation, owner_field, (
                        MeepFields()
                        if owner_field == "fields"
                        else MeepStructure()
                    ))
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "changed native owner",
                    ):
                        policy._workload_record(
                            simulation, 10, None, (), None
                        )

    def test_collective_preinit_common_file_seal_mismatch_fails_closed(self):
        def transform(record, records):
            value = json.loads(record)
            if value.get("ok") is True and "common_digest" in value:
                remote = dict(value)
                remote["common_digest"] = "f" * 64
                records[1] = json.dumps(
                    remote, sort_keys=True, separators=(",", ":")
                )
            return records

        simulation = MeepSimulation(100)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "file-input seals differ",
        ):
            policy.seal_simulation_file_inputs(
                MP("cpu", ranks=2, transform=transform), simulation
            )

    def test_native_proxy_python_state_changes_native_digest(self):
        simulation = MeepSimulation(100)
        simulation.fields = MeepFields()
        simulation.structure = MeepStructure()
        simulation.fields.user_semantic = 1
        simulation.structure.user_semantic = 1
        simulation.fields.owner_alias = simulation.fields
        simulation.structure.owner_alias = simulation.structure
        for native_state in (
            policy._native_fields_state(simulation),
            policy._native_structure_state(simulation),
        ):
            self.assertIn(
                '"object_reference": 0',
                json.dumps(native_state, sort_keys=True),
            )
        first = policy._sha256_json(
            policy._native_simulation_state(simulation)
        )
        simulation.fields.user_semantic = 2
        fields_changed = policy._sha256_json(
            policy._native_simulation_state(simulation)
        )
        self.assertNotEqual(first, fields_changed)
        simulation.structure.user_semantic = 2
        self.assertNotEqual(
            fields_changed,
            policy._sha256_json(policy._native_simulation_state(simulation)),
        )

    def test_native_state_manifest_shadowing_fails_closed(self):
        simulation = MeepSimulation(100)
        simulation.fields = MeepFields()
        simulation.structure = MeepStructure()
        simulation.fields.gpmeep_state_manifest = lambda: "{}"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "cannot be canonicalized exactly",
        ) as shadowed:
            policy._native_fields_state(simulation)
        self.assertIn("accessor is shadowed", str(shadowed.exception.__cause__))
        del simulation.fields.gpmeep_state_manifest

        original = MeepStructure.gpmeep_state_manifest
        try:
            MeepStructure.gpmeep_state_manifest = lambda self: "{}"
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "cannot be canonicalized exactly",
            ):
                policy._native_structure_state(simulation)
        finally:
            MeepStructure.gpmeep_state_manifest = original

    def test_preflight_exception_rendering_is_bounded_and_nonthrowing(self):
        class Toxic(Exception):
            def __str__(self):
                raise self

        class ToxicMonitor:
            @property
            def args(self):
                raise Toxic()

        simulation = Simulation(100)
        simulation.dft_objects = [ToxicMonitor()]
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "<unprintable>"
        ):
            policy.guard_run(MP("cpu"), simulation, 10, None)

    def test_nonexact_root_does_not_project_chunk_layout_simulation(self):
        source = MeepSimulation(100)
        source.fields = MeepFields()
        source.structure = MeepStructure()
        root = Simulation(100)
        root.chunk_layout = source
        before = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        source.fields.mutate()
        self.assertNotEqual(
            before,
            policy._sha256_json(
                policy._workload_record(root, 10, None, (), None)
            ),
        )

    def test_rank_local_indexed_sources_are_ordered_native_state_only(self):
        first = MeepSimulation(100)
        second = MeepSimulation(100)
        first.sources = [IndexedSource({"rank": 0, "amplitude": [1, 2]})]
        second.sources = [IndexedSource({"rank": 1, "amplitude": [3]})]

        first_workload = policy._sha256_json(
            policy._workload_record(first, 10, None, (), None)
        )
        second_workload = policy._sha256_json(
            policy._workload_record(second, 10, None, (), None)
        )
        self.assertEqual(first_workload, second_workload)

        first_local = policy._sha256_json(
            policy._native_simulation_state(first)
        )
        second_local = policy._sha256_json(
            policy._native_simulation_state(second)
        )
        self.assertNotEqual(first_local, second_local)
        first.sources[0].payload["amplitude"][0] = 9
        self.assertEqual(
            first_workload,
            policy._sha256_json(
                policy._workload_record(first, 10, None, (), None)
            ),
        )
        self.assertNotEqual(
            first_local,
            policy._sha256_json(policy._native_simulation_state(first)),
        )

    def test_simulation_chunk_layout_binds_layout_not_rank_local_runtime(self):
        first_layout = MeepSimulation(100)
        first_layout.fields = MeepFields()
        first_layout.structure = MeepStructure()
        second_layout = MeepSimulation(100)
        second_layout.fields = MeepFields()
        second_layout.fields.mutate()
        second_layout.structure = MeepStructure()

        first = MeepSimulation(100)
        first.chunk_layout = first_layout
        first._chunk_layout_original = first_layout
        second = MeepSimulation(100)
        second.chunk_layout = second_layout
        second._chunk_layout_original = second_layout

        first_fingerprint = policy._sha256_json(
            policy._workload_record(first, 10, None, (), None)
        )
        second_fingerprint = policy._sha256_json(
            policy._workload_record(second, 10, None, (), None)
        )
        self.assertEqual(first_fingerprint, second_fingerprint)

        second_layout.structure.little = 2
        self.assertNotEqual(
            first_fingerprint,
            policy._sha256_json(
                policy._workload_record(second, 10, None, (), None)
            ),
        )

    def test_materialized_chunk_layout_binds_target_not_source_lifecycle(self):
        source = MeepSimulation(100)
        source.fields = MeepFields()
        source.structure = MeepStructure(owner=0, little=0)
        target = MeepSimulation(100)
        target.chunk_layout = source
        target._chunk_layout_original = source
        target.structure = MeepStructure(owner=0, little=0)

        materialized = policy._sha256_json(
            policy._workload_record(target, 10, None, (), None)
        )
        source.structure = None
        self.assertEqual(
            materialized,
            policy._sha256_json(
                policy._workload_record(target, 10, None, (), None)
            ),
        )
        target.structure.little = 2
        self.assertNotEqual(
            materialized,
            policy._sha256_json(
                policy._workload_record(target, 10, None, (), None)
            ),
        )
        target.structure = None
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "chunk-layout source must be initialized",
        ):
            policy._workload_record(target, 10, None, (), None)

    def test_simulation_chunk_layout_manifest_fails_closed(self):
        root = MeepSimulation(100)
        source = MeepSimulation(100)
        root.chunk_layout = source
        root._chunk_layout_original = source
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "chunk-layout source must be initialized",
        ):
            policy._workload_record(root, 10, None, (), None)

        source.structure = MeepStructure()
        source.structure.manifest_override = json.dumps(
            {"schema": "gpmeep-chunk-layout-v1", "num_chunks": 1}
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "chunk-layout manifest is malformed",
        ):
            policy._workload_record(root, 10, None, (), None)

        source.structure.manifest_override = None
        for name in (
            "gpmeep_chunk_layout_manifest",
            "get_chunk_volumes",
            "get_chunk_owners",
        ):
            with self.subTest(shadowed_native_accessor=name):
                setattr(source.structure, name, lambda: [])
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "native accessors are shadowed",
                ):
                    policy._workload_record(root, 10, None, (), None)
                delattr(source.structure, name)

    def test_spoofed_indexed_source_remains_in_uniform_workload(self):
        spoof_type = type(
            "IndexedSource", (), {"__module__": "meep.source"}
        )
        first = MeepSimulation(100)
        second = MeepSimulation(100)
        first.sources = [spoof_type()]
        second.sources = [spoof_type()]
        first.sources[0].payload = 1
        second.sources[0].payload = 2
        self.assertNotEqual(
            policy._sha256_json(
                policy._workload_record(first, 10, None, (), None)
            ),
            policy._sha256_json(
                policy._workload_record(second, 10, None, (), None)
            ),
        )

    def test_rank_local_digest_change_invalidates_collective_acknowledgement(self):
        remote_digest = {"value": "a" * 64}

        def transform(record, records):
            value = json.loads(record)
            if "device" in value and "topology" in value:
                remote = dict(value)
                remote["device"] = {
                    **remote["device"],
                    "identifier": "GPU-b",
                }
                remote["topology"] = {
                    **remote["topology"],
                    "rank": 1,
                    "global_rank": 1,
                    "node_rank": 1,
                    "device_identifier": "GPU-b",
                    "device_ordinal": 1,
                }
                records[1] = json.dumps(
                    remote, sort_keys=True, separators=(",", ":")
                )
            elif "native_state_fingerprint" in value:
                remote = dict(value)
                remote["rank"] = 1
                remote["native_state_fingerprint"] = remote_digest["value"]
                records[1] = json.dumps(
                    remote, sort_keys=True, separators=(",", ":")
                )
            return records

        mp = MP("cpu", ranks=2, transform=transform)
        simulation = MeepSimulation(1200)
        simulation.sources = [IndexedSource({"rank": 0})]
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 300, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "rank-local source digest is intentionally bound collectively",
        )
        remote_digest["value"] = "b" * 64
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 300, None)
        self.assertNotEqual(
            first.exception.challenge, changed.exception.challenge
        )

    def test_distinct_nested_simulation_native_state_remains_bound(self):
        root = MeepSimulation(100)
        root.fields = MeepFields()
        root.structure = MeepStructure()
        nested = MeepSimulation(50)
        nested.fields = MeepFields()
        nested.structure = MeepStructure()
        root.geometry.append(nested)
        before = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        nested.fields.mutate()
        after = policy._sha256_json(
            policy._workload_record(root, 10, None, (), None)
        )
        self.assertNotEqual(before, after)

    def test_initialized_native_owner_replacement_invalidates_retry(self):
        mp = MP("cuda")
        simulation = MeepSimulation(100)
        simulation._is_initialized = True
        simulation.fields = MeepFields()
        simulation.structure = MeepStructure()
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "native owner replacement must invalidate this acknowledgement",
        )
        simulation.fields = MeepFields()
        with self.assertRaises(policy.BackendRecommendationMismatch) as replaced:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(first.exception.challenge, replaced.exception.challenge)

    def test_initialized_native_structure_mutation_and_owner_replacement_invalidate(self):
        mp = MP("cuda")
        simulation = MeepSimulation(100)
        simulation._is_initialized = True
        simulation.fields = MeepFields()
        simulation.structure = MeepStructure()
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "native structure mutation must invalidate this acknowledgement",
        )
        simulation.structure.mutate()
        with self.assertRaises(policy.BackendRecommendationMismatch) as mutated:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(first.exception.challenge, mutated.exception.challenge)

        policy.acknowledge_backend_mismatch(
            mutated.exception.challenge,
            "native structure replacement must invalidate this acknowledgement",
        )
        simulation.structure = MeepStructure()
        with self.assertRaises(policy.BackendRecommendationMismatch) as replaced:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(mutated.exception.challenge, replaced.exception.challenge)

    def test_spoofed_meep_module_class_is_fully_canonicalized(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        simulation.geometry.append(SpoofedMeepClass())
        _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 1
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "spoofed Meep module names must not bypass transitive state",
        )
        _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 2
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)

    def test_exact_dft_ldos_ignores_pointer_but_binds_frequency_and_python_state(self):
        first = TrustedDftLdos([1.0, 2.0])
        same = TrustedDftLdos([1.0, 2.0])
        self.assertIsNot(first.this, same.this)
        self.assertEqual(policy._canonical(first), policy._canonical(same))

        changed_frequency = TrustedDftLdos([1.0, 2.5])
        self.assertNotEqual(
            policy._canonical(first), policy._canonical(changed_frequency)
        )
        same.user_state = {"weight": 3}
        self.assertNotEqual(policy._canonical(first), policy._canonical(same))

    def test_exact_dft_ldos_malformed_proxy_fails_closed(self):
        value = TrustedDftLdos([1.0])
        del value.this
        with self.assertRaisesRegex(
            policy.BackendRecommendationError,
            "dft_ldos Python proxy state is malformed",
        ):
            policy._canonical(value)

    def test_dft_ldos_projection_rejects_spoof_and_member_mutation(self):
        spoof_type = type(
            "dft_ldos",
            (),
            {
                "__module__": "meep",
                "freq": property(lambda self: [1.0]),
            },
        )
        spoof = spoof_type()
        spoof.this = object()
        with self.assertRaises(policy.BackendRecommendationError):
            policy._canonical(spoof)

        value = TrustedDftLdos([1.0])
        original = TrustedDftLdos.__dict__["freq"]
        try:
            TrustedDftLdos.freq = property(lambda self: [9.0])
            with self.assertRaises(policy.BackendRecommendationError):
                policy._canonical(value)
        finally:
            TrustedDftLdos.freq = original

    def test_late_package_class_registration_rejects_unlisted_class(self):
        module_name = "meep.adjoint.gpmeep_policy_registration_test"
        module = types.ModuleType(module_name)
        module.__file__ = __file__

        class LazyFilteredSource:
            def semantic_value(self):
                return self.value

        LazyFilteredSource.__module__ = module_name
        LazyFilteredSource.__qualname__ = "LazyFilteredSource"
        module.LazyFilteredSource = LazyFilteredSource
        sys.modules[module_name] = module
        try:
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "registration inventory",
            ):
                policy._register_trusted_meep_class(LazyFilteredSource)
        finally:
            sys.modules.pop(module_name, None)

    def test_late_package_class_registration_rejects_external_caller(self):
        module_name = "meep.adjoint.filter_source"
        module = types.ModuleType(module_name)
        module_file = (
            pathlib.Path(__file__).resolve().parents[1]
            / "adjoint"
            / "filter_source.py"
        )
        module.__file__ = str(module_file)
        module.__spec__ = importlib.util.spec_from_file_location(
            module_name, module_file
        )
        module.__package__ = "meep.adjoint"

        class FilteredSource:
            pass

        FilteredSource.__module__ = module_name
        FilteredSource.__qualname__ = "FilteredSource"
        module.FilteredSource = FilteredSource
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "did not originate",
            ):
                policy._register_trusted_meep_class(FilteredSource)
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

    def test_late_package_class_registration_rejects_fabricated_definition(
        self,
    ):
        module_name = "meep.adjoint.filter_source"
        module_file = (
            pathlib.Path(__file__).resolve().parents[1]
            / "adjoint"
            / "filter_source.py"
        )
        module = types.ModuleType(module_name)
        module.__file__ = str(module_file)
        module.__spec__ = importlib.util.spec_from_file_location(
            module_name, module_file
        )
        module.__package__ = "meep.adjoint"
        module._policy = policy
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        fabricated = compile(
            "class FilteredSource:\n"
            "    pass\n"
            "_policy._register_trusted_meep_class(FilteredSource)\n",
            str(module_file),
            "exec",
        )
        try:
            with self.assertRaisesRegex(
                policy.BackendRecommendationError,
                "definition differs from package source",
            ):
                exec(fabricated, module.__dict__)
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

    def test_package_path_callable_does_not_bypass_mutable_global_state(self):
        module_name = "meep.adjoint.gpmeep_policy_generated_test"
        module = types.ModuleType(module_name)
        module.__file__ = __file__
        generated = _package_generated_global_callback
        original_module = generated.__module__
        original_qualname = generated.__qualname__
        generated.__module__ = module_name
        generated.__qualname__ = "factory.<locals>.generated"
        sys.modules[module_name] = module
        try:
            _PACKAGE_GENERATED_GLOBAL_STATE["value"] = 1
            first = policy._canonical(
                generated, policy._CanonicalTraversal()
            )
            _PACKAGE_GENERATED_GLOBAL_STATE["value"] = 2
            changed = policy._canonical(
                generated, policy._CanonicalTraversal()
            )
            self.assertNotEqual(first, changed)
        finally:
            generated.__module__ = original_module
            generated.__qualname__ = original_qualname
            sys.modules.pop(module_name, None)

    def test_package_path_callable_global_mutation_invalidates_acknowledgement(
        self,
    ):
        module_name = "meep.adjoint.gpmeep_policy_generated_guard_test"
        module = types.ModuleType(module_name)
        module.__file__ = __file__
        generated = _package_generated_global_callback
        original_module = generated.__module__
        original_qualname = generated.__qualname__
        generated.__module__ = module_name
        generated.__qualname__ = "factory.<locals>.generated"
        module.generated = generated
        sys.modules[module_name] = module
        try:
            mp = MP("cuda")
            simulation = Simulation(100)
            simulation.geometry.append(generated)
            _PACKAGE_GENERATED_GLOBAL_STATE["value"] = 1
            with self.assertRaises(
                policy.BackendRecommendationMismatch
            ) as first:
                policy.guard_run(mp, simulation, 10, None)
            policy.acknowledge_backend_mismatch(
                first.exception.challenge,
                "package callback global mutation must invalidate approval",
            )
            _PACKAGE_GENERATED_GLOBAL_STATE["value"] = 2
            with self.assertRaises(
                policy.BackendRecommendationMismatch
            ) as changed:
                policy.guard_run(mp, simulation, 10, None)
            self.assertNotEqual(
                first.exception.challenge, changed.exception.challenge
            )
        finally:
            generated.__module__ = original_module
            generated.__qualname__ = original_qualname
            sys.modules.pop(module_name, None)

    def test_monkey_patched_trusted_meep_member_is_fully_canonicalized(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        simulation.geometry.append(TrustedMeepGeometry())
        original = TrustedMeepGeometry.__dict__["semantic_value"]

        with self.assertRaises(policy.BackendRecommendationMismatch) as initial:
            policy.guard_run(mp, simulation, 10, None)

        def semantic_value():
            return _SPOOFED_MEEP_CLASS_GLOBAL["value"]

        try:
            TrustedMeepGeometry.semantic_value = staticmethod(semantic_value)
            _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 1
            with self.assertRaises(policy.BackendRecommendationMismatch) as patched:
                policy.guard_run(mp, simulation, 10, None)
            self.assertNotEqual(initial.exception.challenge, patched.exception.challenge)
            policy.acknowledge_backend_mismatch(
                patched.exception.challenge,
                "patched trusted members must bind transitive user state",
            )
            _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 2
            with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                policy.guard_run(mp, simulation, 10, None)
            self.assertNotEqual(
                patched.exception.challenge, changed.exception.challenge
            )
        finally:
            TrustedMeepGeometry.semantic_value = original

    def test_bytecode_memo_is_local_and_preserves_canonical_tokens(self):
        state = {"value": 1}
        original_global = _CALLBACK_GLOBAL["value"]
        self.addCleanup(_CALLBACK_GLOBAL.__setitem__, "value", original_global)

        def callback(default=2):
            return state["value"] + default + _CALLBACK_GLOBAL["value"]

        traversal = policy._CanonicalTraversal()
        with mock.patch.object(
            policy.dis, "get_instructions", wraps=policy.dis.get_instructions
        ) as disassemble:
            policy._referenced_global_names(callback.__code__, traversal)
            policy._reject_implicit_namespace_operations(
                callback.__code__, seen=traversal
            )
            policy._referenced_global_names(callback.__code__, traversal)
            self.assertEqual(disassemble.call_count, 1)
            policy._referenced_global_names(
                callback.__code__, policy._CanonicalTraversal()
            )
            self.assertEqual(disassemble.call_count, 2)

        baseline = None
        for mutation in ("initial", "closure", "defaults", "globals"):
            with self.subTest(mutation=mutation):
                if mutation == "closure":
                    state["value"] += 1
                elif mutation == "defaults":
                    callback.__defaults__ = (3,)
                elif mutation == "globals":
                    _CALLBACK_GLOBAL["value"] += 1
                memoized = policy._canonical(callback, policy._CanonicalTraversal())
                with mock.patch.object(
                    policy, "_code_instructions",
                    side_effect=lambda code, seen=None: policy.dis.get_instructions(code),
                ):
                    fresh = policy._canonical(callback, policy._CanonicalTraversal())
                self.assertEqual(memoized, fresh)
                if baseline is not None:
                    self.assertNotEqual(memoized, baseline)
                baseline = memoized

    def test_bytecode_memo_keeps_format_permissions_and_print_binding_live(self):
        namespace = {"__builtins__": __builtins__}
        exec("def callback(value):\n    print(f'{value}')\n", namespace)
        callback = namespace["callback"]
        traversal = policy._CanonicalTraversal()
        self.assertTrue(policy._format_values_are_builtin_print_only(callback, traversal))
        policy._reject_implicit_namespace_operations(
            callback.__code__, allow_format_value=True, seen=traversal
        )
        with self.assertRaisesRegex(policy.BackendRecommendationError, "FORMAT_VALUE"):
            policy._reject_implicit_namespace_operations(callback.__code__, seen=traversal)
        namespace["print"] = lambda value: value
        self.assertFalse(policy._format_values_are_builtin_print_only(callback, traversal))

    def test_bytecode_memo_does_not_freeze_mutable_constants(self):
        def constant():
            return 1

        value = [1]
        code = constant.__code__.replace(co_consts=(None, value))
        traversal = policy._CanonicalTraversal()
        before = policy._code_token(code, policy._CanonicalTraversal())
        list(policy._code_instructions(code, traversal))
        value.append(2)
        after = policy._code_token(code, policy._CanonicalTraversal())
        self.assertNotEqual(before, after)
        self.assertEqual(
            list(policy._code_instructions(code, traversal)),
            list(policy.dis.get_instructions(code)),
        )
        self.assertNotIn(id(code), traversal.bytecode_instructions)

    def test_bytecode_memo_preserves_custom_constant_repr_calls_and_errors(self):
        calls = []

        class Constant:
            def __repr__(self):
                calls.append(len(calls))
                if len(calls) == 2:
                    raise ValueError("second constant repr")
                return "constant"

        def constant():
            return 1

        code = constant.__code__.replace(co_consts=(None, Constant()))
        traversal = policy._CanonicalTraversal()
        list(policy._code_instructions(code, traversal))
        with self.assertRaisesRegex(ValueError, "second constant repr"):
            list(policy._code_instructions(code, traversal))
        self.assertEqual(len(calls), 2)
        self.assertNotIn(id(code), traversal.bytecode_instructions)

    def test_memoized_trusted_dependency_graph_rechecks_at_traversal_end(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot(
            (TrustedMeepGeometry,)
        )
        dependency = next(iter(graph["functions"]))
        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            baseline_token = policy._class_token(
                TrustedMeepGeometry, policy._CanonicalTraversal()
            )
            memoized_token_traversal = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            memoized_token = policy._class_token(
                TrustedMeepGeometry, memoized_token_traversal
            )
            policy._finalize_trusted_meep_dependency_integrity(
                memoized_token_traversal, "class token"
            )
            self.assertEqual(baseline_token, memoized_token)

            traversal = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            self.assertTrue(
                policy._trusted_meep_dependency_is_unchanged(
                    TrustedMeepGeometry, traversal
                )
            )
            self.assertTrue(traversal.trusted_meep_dependency_graph_checked)

            original_doc = dependency.__doc__
            try:
                dependency.__doc__ = "gpmeep intentional traversal mutation"
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "changed during workload canonicalization",
                ):
                    policy._finalize_trusted_meep_dependency_integrity(
                        traversal, "workload"
                    )
            finally:
                dependency.__doc__ = original_doc
            policy._finalize_trusted_meep_dependency_integrity(
                traversal, "workload"
            )
        finally:
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_memoized_trusted_dependency_node_rejects_transient_mutation(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot(
            (TrustedMeepGeometry,)
        )
        dependency = next(iter(graph["functions"]))
        original_code = dependency.__code__

        def replacement():
            return 2

        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            traversal = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            self.assertTrue(
                policy._trusted_meep_dependency_is_unchanged(
                    TrustedMeepGeometry, traversal
                )
            )
            original_code_token = policy._code_token
            mutated = False

            def mutate_after_token(code, seen):
                nonlocal mutated
                token = original_code_token(code, seen)
                if code is original_code and not mutated:
                    dependency.__code__ = replacement.__code__
                    mutated = True
                return token

            try:
                with mock.patch.object(
                    policy,
                    "_code_token",
                    side_effect=mutate_after_token,
                ):
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "package dependency (?:was modified|changed during "
                        "canonicalization)",
                    ):
                        policy._class_token(
                            TrustedMeepGeometry, traversal
                        )
                self.assertTrue(mutated)
            finally:
                dependency.__code__ = original_code
            policy._finalize_trusted_meep_dependency_integrity(
                traversal, "workload"
            )
        finally:
            dependency.__code__ = original_code
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_import_snapshot_prevents_restored_mutation_from_altering_token(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot(
            (TrustedMeepGeometry,)
        )
        dependency = next(iter(graph["functions"]))
        original_code = dependency.__code__

        def replacement():
            return 2

        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            baseline_traversal = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            baseline = policy._class_token(
                TrustedMeepGeometry, baseline_traversal
            )
            policy._finalize_trusted_meep_dependency_integrity(
                baseline_traversal, "baseline"
            )

            original_code_token = policy._code_token
            mutation_observed = False

            def mutate_only_while_serializing_snapshot(code, seen):
                nonlocal mutation_observed
                if code is original_code and not mutation_observed:
                    mutation_observed = True
                    dependency.__code__ = replacement.__code__
                    try:
                        return original_code_token(code, seen)
                    finally:
                        dependency.__code__ = original_code
                return original_code_token(code, seen)

            traversal = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            with mock.patch.object(
                policy,
                "_code_token",
                side_effect=mutate_only_while_serializing_snapshot,
            ):
                observed = policy._class_token(
                    TrustedMeepGeometry, traversal
                )
            policy._finalize_trusted_meep_dependency_integrity(
                traversal, "restored mutation"
            )
            self.assertTrue(mutation_observed)
            self.assertEqual(observed, baseline)
        finally:
            dependency.__code__ = original_code
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_workload_record_finalizes_trusted_dependency_integrity(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot(
            (TrustedMeepGeometry,)
        )
        dependency = next(iter(graph["functions"]))
        original_code = dependency.__code__

        def replacement():
            return 2

        simulation = Simulation(100)
        simulation.geometry.append(TrustedMeepGeometry())
        operation = "".join(("gpmeep-", self.id(), "-operation"))
        original_canonical = policy._canonical
        mutated = False

        def mutate_after_final_operation(value, seen=None):
            nonlocal mutated
            token = original_canonical(value, seen)
            if (
                value is operation
                and isinstance(seen, policy._CanonicalTraversal)
                and seen.trusted_meep_dependency_graph_checked
                and not mutated
            ):
                dependency.__code__ = replacement.__code__
                mutated = True
            return token

        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            with mock.patch.object(
                policy, "_canonical", side_effect=mutate_after_final_operation
            ):
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "changed during workload canonicalization",
                ):
                    policy._workload_record(
                        simulation, 10, None, (), operation
                    )
            self.assertTrue(mutated)
        finally:
            dependency.__code__ = original_code
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_python_rank_local_state_finalizes_trusted_dependency_integrity(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot((IndexedSource,))
        dependency = next(iter(graph["functions"]))
        original_code = dependency.__code__

        def replacement(self, payload):
            self.payload = ("changed", payload)

        simulation = Simulation(100)
        simulation.sources = [IndexedSource("rank-local")]
        original_canonical = policy._canonical
        mutated = False

        def mutate_after_indexed_sources(value, seen=None):
            nonlocal mutated
            token = original_canonical(value, seen)
            is_indexed_source_list = (
                type(value) is list
                and len(value) == 1
                and type(value[0]) is dict
                and set(value[0]) == {"position", "source"}
            )
            if (
                is_indexed_source_list
                and isinstance(seen, policy._CanonicalTraversal)
                and seen.trusted_meep_dependency_graph_checked
                and not mutated
            ):
                dependency.__code__ = replacement.__code__
                mutated = True
            return token

        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            with mock.patch.object(
                policy, "_canonical", side_effect=mutate_after_indexed_sources
            ):
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "changed during Python rank-local state canonicalization",
                ):
                    policy._python_rank_local_state(simulation)
            self.assertTrue(mutated)
        finally:
            dependency.__code__ = original_code
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_trusted_dependency_memo_is_local_and_ordinary_checks_are_rooted(self):
        original_graph = policy._TRUSTED_MEEP_DEPENDENCY_GRAPH
        graph = policy._trusted_dependency_graph_snapshot(
            (TrustedMeepGeometry,)
        )
        dependency = next(iter(graph["functions"]))
        original_code = dependency.__code__

        def replacement():
            return 2

        policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = graph
        try:
            first = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            self.assertTrue(
                policy._trusted_meep_dependency_is_unchanged(
                    TrustedMeepGeometry, first
                )
            )
            second = policy._CanonicalTraversal(
                memoize_trusted_meep_dependency_graph=True
            )
            self.assertFalse(second.trusted_meep_dependency_graph_checked)
            try:
                dependency.__code__ = replacement.__code__
                self.assertFalse(
                    policy._trusted_meep_dependency_is_unchanged(
                        TrustedMeepGeometry, second
                    )
                )
            finally:
                dependency.__code__ = original_code

            with mock.patch.object(
                policy,
                "_trusted_dependency_graph_is_unchanged",
                wraps=policy._trusted_dependency_graph_is_unchanged,
            ) as graph_check:
                policy._class_token(
                    TrustedMeepGeometry, policy._CanonicalTraversal()
                )
            rooted_calls = [
                call
                for call in graph_check.call_args_list
                if call.args and call.args[0] is graph
            ]
            self.assertGreaterEqual(len(rooted_calls), 2)
            self.assertTrue(
                all(
                    call.kwargs.get("root") is TrustedMeepGeometry
                    for call in rooted_calls
                )
            )
        finally:
            dependency.__code__ = original_code
            policy._TRUSTED_MEEP_DEPENDENCY_GRAPH = original_graph

    def test_monkey_patched_root_simulation_member_is_bound_without_dft(self):
        mp = MP("cuda")
        simulation = MeepSimulation(100)

        with self.assertRaises(policy.BackendRecommendationMismatch) as initial:
            policy.guard_run(mp, simulation, 10, None)

        def semantic_value():
            return _SPOOFED_MEEP_CLASS_GLOBAL["value"]

        try:
            MeepSimulation.semantic_value = staticmethod(semantic_value)
            _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 1
            with self.assertRaises(policy.BackendRecommendationMismatch) as patched:
                policy.guard_run(mp, simulation, 10, None)
            self.assertNotEqual(initial.exception.challenge, patched.exception.challenge)
            policy.acknowledge_backend_mismatch(
                patched.exception.challenge,
                "patched root Simulation must bind transitive user state",
            )
            _SPOOFED_MEEP_CLASS_GLOBAL["value"] = 2
            with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                policy.guard_run(mp, simulation, 10, None)
            self.assertNotEqual(
                patched.exception.challenge, changed.exception.challenge
            )
        finally:
            del MeepSimulation.semantic_value

    def test_user_fields_are_not_treated_as_simulation_ephemeral_state(self):
        names = tuple(sorted(policy._EPHEMERAL_OBJECT_FIELDS)) + (
            "_gpmeep_user_state",
        )
        for name in names:
            with self.subTest(attribute=name):
                policy._reset_for_testing()

                class Holder:
                    pass

                holder = Holder()
                setattr(holder, name, 1)

                def callback():
                    return getattr(holder, name)

                mp = MP("cuda")
                simulation = Simulation(100)
                with self.assertRaises(policy.BackendRecommendationMismatch) as first:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "user field mutation must invalidate acknowledgement",
                )
                setattr(holder, name, 2)
                with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
                    policy.guard_run(
                        mp, simulation, callback, None, (callback,)
                    )
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

    def test_spoofed_simulation_reserved_fields_are_never_ephemeral(self):
        spoof_type = type(
            "Simulation", (), {"__module__": "meep.simulation"}
        )
        names = tuple(sorted(policy._EPHEMERAL_OBJECT_FIELDS)) + (
            "_gpmeep_user_state",
        )
        for name in names:
            with self.subTest(attribute=name):
                policy._reset_for_testing()
                spoof = spoof_type()
                setattr(spoof, name, {"semantic": 1})
                simulation = Simulation(100)
                simulation.geometry.append(spoof)
                mp = MP("cuda")
                with self.assertRaises(
                    policy.BackendRecommendationMismatch
                ) as first:
                    policy.guard_run(mp, simulation, 10, None)
                policy.acknowledge_backend_mismatch(
                    first.exception.challenge,
                    "spoofed Simulation reserved state must remain semantic",
                )
                setattr(spoof, name, {"semantic": 2})
                with self.assertRaises(
                    policy.BackendRecommendationMismatch
                ) as changed:
                    policy.guard_run(mp, simulation, 10, None)
                self.assertNotEqual(
                    first.exception.challenge, changed.exception.challenge
                )

    def test_spoofed_verbosity_cvars_is_never_ephemeral(self):
        spoof_type = type(
            "Verbosity", (), {"__module__": "meep.verbosity_mgr"}
        )
        spoof = spoof_type()
        spoof._cvars = {"semantic": 1}
        simulation = Simulation(100)
        simulation.geometry.append(spoof)
        mp = MP("cuda")
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "spoofed Verbosity state must remain semantic",
        )
        spoof._cvars = {"semantic": 2}
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(
            first.exception.challenge, changed.exception.challenge
        )

    def test_nested_module_state_fails_closed(self):
        module = types.ModuleType("gpmeep_nested_module")
        module.threshold = 1
        container = [module]

        class Holder:
            pass

        holder = Holder()
        holder.module = module

        def container_callback():
            return container[0].threshold

        def holder_callback():
            return holder.module.threshold

        for callback in (container_callback, holder_callback):
            with self.subTest(callback=callback):
                policy._reset_for_testing()
                with self.assertRaisesRegex(
                    policy.BackendRecommendationError,
                    "nested module state cannot be canonicalized exactly",
                ):
                    policy.guard_run(
                        MP("cuda"),
                        Simulation(100),
                        callback,
                        None,
                        (callback,),
                    )

    def test_alias_shared_subgraph_and_cycle_topology_are_bound(self):
        class Empty:
            pass

        shared = Empty()
        aliases = [shared, shared]

        def alias_callback():
            return aliases[0] is aliases[1]

        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, alias_callback, None, (alias_callback,))
        policy.acknowledge_backend_mismatch(
            first.exception.challenge,
            "alias topology mutation must invalidate acknowledgement",
        )
        aliases[1] = Empty()
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, alias_callback, None, (alias_callback,))
        self.assertNotEqual(first.exception.challenge, changed.exception.challenge)

        policy._reset_for_testing()
        leaf = Empty()
        graph = {"left": [leaf], "right": [leaf]}
        leaf.parent = graph

        def graph_callback():
            return graph["left"][0] is graph["right"][0]

        with self.assertRaises(policy.BackendRecommendationMismatch) as graph_first:
            policy.guard_run(mp, simulation, graph_callback, None, (graph_callback,))
        policy.acknowledge_backend_mismatch(
            graph_first.exception.challenge,
            "shared cycle mutation must invalidate acknowledgement",
        )
        replacement = Empty()
        replacement.parent = graph
        graph["right"][0] = replacement
        with self.assertRaises(policy.BackendRecommendationMismatch) as graph_changed:
            policy.guard_run(mp, simulation, graph_callback, None, (graph_callback,))
        self.assertNotEqual(
            graph_first.exception.challenge, graph_changed.exception.challenge
        )

        first_mapping = {"a": 1, "b": 2}
        second_mapping = {"b": 2, "a": 1}
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_mapping)),
            policy._sha256_json(policy._canonical(second_mapping)),
        )

        class CollidingItem:
            def __init__(self, name):
                self.name = name

            def __hash__(self):
                return 0

        first_items = [CollidingItem("a"), CollidingItem("b")]
        second_items = [CollidingItem("a"), CollidingItem("b")]
        first_set = set(first_items)
        second_set = set(reversed(second_items))
        self.assertNotEqual(
            [item.name for item in first_set],
            [item.name for item in second_set],
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(first_set)),
            policy._sha256_json(policy._canonical(second_set)),
        )
        self.assertNotEqual(
            policy._sha256_json(policy._canonical(set(first_items))),
            policy._sha256_json(policy._canonical(frozenset(first_items))),
        )
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "mapping views"
        ):
            policy._canonical(types.MappingProxyType({"value": 1}))
        for iterator_value in (iter({"value": 1}), iter({1})):
            with self.subTest(
                iterator_type=type(iterator_value).__name__
            ), self.assertRaisesRegex(
                policy.BackendRecommendationError, "iterator type"
            ):
                policy._canonical(iterator_value)

    def test_canonical_traversal_retains_objects_until_fingerprint_finishes(self):
        class Ephemeral:
            pass

        value = Ephemeral()
        reference = weakref.ref(value)
        identity = id(value)
        traversal = policy._CanonicalTraversal()
        token = policy._canonical(value, traversal)

        self.assertIs(traversal.object_values[identity], value)
        del value
        gc.collect()
        self.assertIsNotNone(reference())

        del traversal
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsInstance(token, dict)

    def test_fused_matching_records_preserve_pending_fallback_bindings(self):
        records = []

        def observe(record, peers):
            records.append(json.loads(record))
            return peers

        mp = MP("cpu", ranks=2, transform=observe)
        simulation = Simulation(100)
        with mock.patch.object(policy.secrets, "token_hex", return_value="a" * 64):
            fused = policy.guard_run(mp, simulation, 10, None)
            self.assertEqual(len(records), 3)
            self.assertIn("run_nonce_proposal", records[-1])
            with mock.patch.object(
                policy, "_policy_decision",
                return_value=("cuda", "hardware_crossover", "high"),
            ):
                with self.assertRaises(policy.BackendRecommendationMismatch):
                    policy.guard_run(mp, simulation, 10, None)
            records.clear()
            fallback = policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(fused, fallback)
        self.assertEqual(len(records), 6)
        self.assertTrue(records[1]["pending_present"])
        self.assertNotIn("run_nonce_proposal", records[2])
        self.assertNotIn(simulation, policy._PENDING_BY_SIMULATION)

    def test_fused_matching_runs_recheck_mutable_workload(self):
        original_global = _CALLBACK_GLOBAL["value"]
        self.addCleanup(_CALLBACK_GLOBAL.__setitem__, "value", original_global)
        mp = MP("cpu", ranks=2)
        simulation = Simulation(100)
        first = policy.guard_run(mp, simulation, 10, None, (_global_state_callback,))
        _CALLBACK_GLOBAL["value"] += 1
        second = policy.guard_run(mp, simulation, 10, None, (_global_state_callback,))
        self.assertTrue(first["workload_fingerprint_evaluated"])
        self.assertTrue(second["workload_fingerprint_evaluated"])
        self.assertNotEqual(first["workload_fingerprint"], second["workload_fingerprint"])
        self.assertNotEqual(first["run_nonce"], second["run_nonce"])

    def test_fused_nonce_failures_and_control_disagreement_fail_closed(self):
        records = []

        def observe(record, peers):
            records.append(json.loads(record))
            return peers

        for nonce in ("x" * 5000, None):
            with self.subTest(nonce_type=type(nonce).__name__):
                records.clear()
                with mock.patch.object(policy.secrets, "token_hex", return_value=nonce):
                    with self.assertRaisesRegex(
                        policy.BackendRecommendationError,
                        "distributed gpmeep exact workload preflight failed.*nonce proposal",
                    ):
                        policy.guard_run(MP("cpu", ranks=2, transform=observe), Simulation(100), 10, None)
                self.assertEqual(len(records), 3)
                self.assertFalse(records[-1]["ok"])
                self.assertLess(len(json.dumps(records[-1]).encode()), 4096)

        def corrupt(record, peers):
            local = json.loads(record)
            if "pending_present" in local:
                local["pending_present"] = True
                peers[1] = json.dumps(local)
            return peers

        with mock.patch.object(policy.secrets, "token_hex") as entropy:
            with self.assertRaisesRegex(policy.BackendRecommendationError, "pending state differs"):
                policy.guard_run(MP("cpu", ranks=2, transform=corrupt), Simulation(100), 10, None)
            entropy.assert_not_called()

    def test_fused_received_nonce_record_is_validated(self):
        def corrupt(record, peers):
            local = json.loads(record)
            if "run_nonce_proposal" in local:
                local.pop("run_nonce_proposal")
                peers[1] = json.dumps(local)
            return peers

        with self.assertRaisesRegex(policy.BackendRecommendationError, "exact nonce record is malformed"):
            policy.guard_run(MP("cpu", ranks=2, transform=corrupt), Simulation(100), 10, None)

    def test_changed_backend_invalidates_old_challenge(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        old = first.exception.challenge
        mp.gpu.requested_backend = "cpu"
        accepted = policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(accepted["selected_backend"], "cpu")
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old, "old backend choice is no longer valid"
            )

    def test_changed_recommendation_invalidates_old_challenge(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        old = first.exception.challenge
        original_decision = policy._policy_decision
        policy._policy_decision = lambda records: (
            "cuda",
            "hardware_crossover",
            "high",
        )
        try:
            accepted = policy.guard_run(mp, simulation, 10, None)
            self.assertEqual(accepted["recommended_backend"], "cuda")
        finally:
            policy._policy_decision = original_decision
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old, "old recommendation is no longer valid"
            )

    def test_changed_mpi_topology_invalidates_old_challenge(self):
        mp = MP("cuda", ranks=1)
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as first:
            policy.guard_run(mp, simulation, 10, None)
        old = first.exception.challenge
        mp.ranks = 2
        mp.transform = lambda record, records: [
            records[0],
            records[1]
            .replace('"device_identifier":"GPU-a"', '"device_identifier":"GPU-b"')
            .replace('"identifier":"GPU-a"', '"identifier":"GPU-b"'),
        ]
        with self.assertRaises(policy.BackendRecommendationMismatch) as changed:
            policy.guard_run(mp, simulation, 10, None)
        self.assertNotEqual(old, changed.exception.challenge)
        with self.assertRaises(policy.BackendRecommendationError):
            policy.acknowledge_backend_mismatch(
                old, "old MPI topology is no longer valid"
            )

    def test_backend_rank_disagreement_fails_closed(self):
        def transform(record, records):
            if '"selected_backend"' in record:
                records[1] = record.replace('"selected_backend":"cpu"', '"selected_backend":"cuda"')
            return records

        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "selections differ"
        ):
            policy.guard_run(MP("cpu", ranks=2, transform=transform), Simulation(100), 10, None)

    def test_acknowledgement_rank_disagreement_fails_closed(self):
        state = {"mutate_ack": False}

        def transform(record, records):
            if state["mutate_ack"] and '"challenge"' in record:
                records[1] = record.replace(
                    '"reason":"intentional CUDA qualification"',
                    '"reason":"different reason across another rank"',
                )
            return records

        mp = MP("cuda", ranks=2, transform=transform, identifier="GPU-a")
        # Give the second synthetic rank a distinct device identifier so the
        # recommendation remains a workload crossover rather than environment.
        def distinct_devices(record, records):
            records = transform(record, records)
            if '"device_identifier":"GPU-a"' in record:
                records[1] = records[1].replace(
                    '"device_identifier":"GPU-a"', '"device_identifier":"GPU-b"'
                ).replace('"identifier":"GPU-a"', '"identifier":"GPU-b"')
            return records

        mp.transform = distinct_devices
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            caught.exception.challenge, "intentional CUDA qualification"
        )
        state["mutate_ack"] = True
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "differs across MPI ranks"
        ):
            policy.guard_run(mp, simulation, 10, None)

    def test_consumed_state_rank_disagreement_fails_closed(self):
        state = {"mutate_consumed": False}

        def transform(record, records):
            if '"device_identifier":"GPU-a"' in record:
                records[1] = records[1].replace(
                    '"device_identifier":"GPU-a"', '"device_identifier":"GPU-b"'
                ).replace('"identifier":"GPU-a"', '"identifier":"GPU-b"')
            if state["mutate_consumed"] and '"consumed":false' in record:
                records[1] = record.replace('"consumed":false', '"consumed":true')
            return records

        mp = MP("cuda", ranks=2, transform=transform)
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        policy.acknowledge_backend_mismatch(
            caught.exception.challenge, "intentional CUDA qualification"
        )
        state["mutate_consumed"] = True
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "already consumed"
        ):
            policy.guard_run(mp, simulation, 10, None)

    def test_acknowledgement_inputs_are_strictly_bounded_utf8(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        challenge = caught.exception.challenge
        for malformed in ("", "0" * 63, "0" * 65, "G" * 64):
            with self.subTest(challenge=malformed):
                with self.assertRaises(policy.BackendRecommendationError):
                    policy.acknowledge_backend_mismatch(malformed, "valid reason")
        for reason in ("1234567", "가가", "x" * 513):
            with self.subTest(reason_bytes=len(reason.encode("utf-8"))):
                with self.assertRaises(policy.BackendRecommendationError):
                    policy.acknowledge_backend_mismatch(challenge, reason)
        policy.acknowledge_backend_mismatch(challenge, "가가가")
        accepted = policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(accepted["acknowledgement_reason"], "가가가")

    def test_corrupted_pending_reason_fails_closed_before_consumption(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch):
            policy.guard_run(mp, simulation, 10, None)
        policy._PENDING_BY_SIMULATION[simulation]["reason"] = "short"
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "8 to 512"
        ):
            policy.guard_run(mp, simulation, 10, None)
        self.assertFalse(policy._CONSUMED_RUN_NONCES)

    def test_abandoned_simulation_does_not_leak_challenge_index(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        challenge = caught.exception.challenge
        reference = weakref.ref(simulation)
        del simulation
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(challenge, policy._PENDING_BY_CHALLENGE)

    def test_duplicate_rank_device_mapping_is_non_acknowledgeable_environment_error(self):
        mp = MP("cuda", ranks=2)
        simulation = Simulation(1200)
        with self.assertRaises(policy.BackendEnvironmentError) as caught:
            policy.guard_run(mp, simulation, 20, None)
        self.assertEqual(caught.exception.telemetry["recommended_backend"], "cpu")
        self.assertEqual(caught.exception.telemetry["reason_class"], "environment")
        self.assertFalse(caught.exception.telemetry["acknowledgement_allowed"])

    def test_cpu_duplicate_rank_device_mapping_runs_without_acknowledgement(self):
        mp = MP("cpu", ranks=2)
        simulation = Simulation(1200)
        accepted = policy.guard_run(mp, simulation, 20, None)
        self.assertEqual(accepted["selected_backend"], "cpu")
        self.assertEqual(accepted["recommended_backend"], "cpu")
        self.assertEqual(accepted["reason_class"], "environment")
        self.assertFalse(accepted["acknowledgement_allowed"])
        self.assertFalse(accepted["acknowledgement"])
        self.assertEqual(accepted["acknowledgement_reason"], "")
        self.assertIsNone(policy.pending_recommendation())

    def test_small_cuda_duplicate_device_is_environment_not_crossover(self):
        mp = MP("cuda", ranks=2)
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendEnvironmentError) as caught:
            policy.guard_run(mp, simulation, 10, None)
        self.assertEqual(caught.exception.telemetry["recommended_backend"], "cpu")
        self.assertEqual(caught.exception.telemetry["reason_class"], "environment")

    def test_missing_device_identity_is_non_acknowledgeable_environment_error(self):
        with self.assertRaises(policy.BackendEnvironmentError) as caught:
            policy.guard_run(MP("cuda", identifier=""), Simulation(100), 10, None)
        self.assertEqual(caught.exception.telemetry["reason_class"], "environment")
        self.assertIsNone(policy.pending_recommendation())

    def test_nonfinite_cell_extent_fails_closed(self):
        for dimensions in (1, -2, 2, 3):
            for component in ("x", "y", "z"):
                for value in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(
                        dimensions=dimensions, component=component, value=value
                    ):
                        simulation = Simulation(100)
                        simulation.dimensions = dimensions
                        setattr(simulation.cell_size, component, value)
                        with self.assertRaisesRegex(
                            policy.BackendRecommendationError,
                            "extents must be finite",
                        ):
                            policy.guard_run(MP("cpu"), simulation, 10, None)

    def test_zero_dimensional_cell_counts_as_one_spatial_sample(self):
        simulation = Simulation(100)
        simulation.cell_size = Vector()
        simulation.dimensions = 3
        accepted = policy.guard_run(MP("cpu"), simulation, 10, None)
        self.assertEqual(accepted["rank_local_cells"], [1])
        self.assertEqual(accepted["recommended_backend"], "cpu")

    def test_extent_resolution_overflow_is_normalized(self):
        simulation = Simulation(100)
        simulation.cell_size.x = 1e308
        simulation.resolution = 1e308
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "product is out of range"
        ):
            policy.guard_run(MP("cpu"), simulation, 10, None)

    def test_acknowledgement_reason_rejects_invalid_unicode(self):
        mp = MP("cuda")
        simulation = Simulation(100)
        with self.assertRaises(policy.BackendRecommendationMismatch) as caught:
            policy.guard_run(mp, simulation, 10, None)
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "valid UTF-8"
        ):
            policy.acknowledge_backend_mismatch(
                caught.exception.challenge, "valid text then surrogate \ud800"
            )

    def test_automatic_choice_is_rejected(self):
        with self.assertRaisesRegex(
            policy.BackendRecommendationError, "automatic backend selection is disabled"
        ):
            policy.guard_run(MP("auto"), Simulation(100), 10, None)


if __name__ == "__main__":
    unittest.main()
