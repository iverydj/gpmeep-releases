"""Self-tests for the generic unchanged-example physical oracle."""

from __future__ import annotations

import contextlib
import ast
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ORACLE_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_example_oracle.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_example_oracle_test", ORACLE_PATH)
assert SPEC and SPEC.loader
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def fake_recommendation(
    selected: str,
    recommended: str,
    *,
    acknowledgement: bool = False,
    reason: str = "",
    acknowledgement_scope: str = "simulation",
) -> dict[str, object]:
    exact = selected != recommended
    telemetry: dict[str, object] = {
        "schema": ORACLE.qualification_guard.TELEMETRY_SCHEMA,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "reason_class": "hardware_crossover",
        "recommendation_confidence": "high",
        "rank_local_cells": [1000],
        "run_length": {
            "kind": "until",
            "numeric_duration": 5.0,
            "source_end_time": None,
            "expected_timesteps": 100,
            "dynamic_condition": False,
        },
        "kernel_launch_count": 400,
        "transfer_bytes": 1024000,
        "monitor_count": 0,
        "monitor_frequency_count": 0,
        "source_count": 1,
        "mpi_gpu_topology": [{
            "rank": 0,
            "global_rank": 0,
            "active_size": 1,
            "node_rank": 0,
            "node_size": 1,
            "device_identifier": "fixture",
            "device_ordinal": 0,
        }],
        "profiler_or_ablation_id": ORACLE.qualification_guard.PROFILER_OR_ABLATION_ID,
        "acknowledgement": acknowledgement,
        "acknowledgement_allowed": True,
        "acknowledgement_reason": reason if acknowledgement else "",
        "acknowledgement_reused": False,
        "acknowledgement_scope": (
            acknowledgement_scope if acknowledgement else ""
        ),
        "acknowledgement_origin_cohort_digest": "",
        "acknowledgement_scope_cache_enabled": True,
        "acknowledgement_scope_digest": "6" * 64,
        "workload_fingerprint_evaluated": exact,
        "workload_fingerprint": "1" * 64 if exact else "",
        "recommendation_policy_version": ORACLE.qualification_guard.POLICY_VERSION,
        "mpi_topology_digest": "2" * 64,
        "native_state_digest": "3" * 64 if exact else "",
        "process_nonce": "4" * 64 if exact else "",
        "run_nonce": "5" * 64 if exact else "",
        "acknowledgement_digest": "",
    }
    if exact:
        telemetry["acknowledgement_digest"] = (
            ORACLE.qualification_guard._acknowledgement_digest(telemetry)
        )
    if acknowledgement:
        telemetry["acknowledgement_origin_cohort_digest"] = telemetry[
            "acknowledgement_digest"
        ]
    return telemetry


class FakeFields:
    def __init__(self, simulation: "FakeSimulation") -> None:
        self.simulation = simulation

    def dft_norm(self) -> float:
        return 2.0 * self.simulation.runs

    @property
    def t(self) -> int:
        return 10 * self.simulation.runs

    def total_volume(self) -> object:
        return self

    def gpu_execution_diagnostic(self) -> str:
        return "synthetic execution diagnostic"


class FakeSimulation:
    gpu = None

    def __init__(self) -> None:
        self.runs = 0
        self.fields = FakeFields(self)

    def run(self, *step_funcs, **kwargs):
        self.runs += int(kwargs["until"])
        assert self.gpu is not None
        self.gpu.advance()

    def run_k_point(self, t=None, k=None):
        # Mirror the public operation's intentional field restart so the
        # oracle must account each k point from zero, not the previous run.
        self.runs = 0
        self.run(until=t)
        return (t, k)

    def solve_cw(self, *args, **kwargs):
        return (args, kwargs)

    def solve_eigfreq(self, *args, **kwargs):
        return complex(1.0, -0.1)

    def timestep(self) -> int:
        return 10 * self.runs

    def round_time(self) -> float:
        return 0.5 * self.runs

    def field_energy_in_box(self, box=None) -> float:
        return 1.5 * self.runs


class UpstreamFakeSimulation(FakeSimulation):
    def run(self, *step_funcs, **kwargs):
        self.runs += int(kwargs["until"])


class FakeGpu:
    def __init__(self, active_backend: str = "cpu") -> None:
        self.active_backend = active_backend
        self.requested_backend = active_backend
        self.cpu_calls = 0
        self.cuda_calls = 0
        self.recommendation = self.matching_recommendation()

    def matching_recommendation(self):
        return fake_recommendation(self.active_backend, self.active_backend)

    def advance(self) -> None:
        if self.active_backend == "cpu":
            self.cpu_calls += 3
        else:
            self.cuda_calls += 3
        self.recommendation = self.matching_recommendation()

    def statistics(self):
        return {
            "runtime": {
                "runtime_availability_probes": 0,
                "runtime_device_enumerations": 0,
                "runtime_device_selections": 0,
            },
            "dispatch": {
                "cpu_curl_calls": self.cpu_calls,
                "cuda_curl_calls": self.cuda_calls,
                "host_to_device_bytes": 0,
            }
        }


class FakeBackendRecommendationMismatch(RuntimeError):
    def __init__(self, challenge: str, telemetry: dict[str, object]) -> None:
        super().__init__("synthetic backend recommendation mismatch")
        self.challenge = challenge
        self.telemetry = dict(telemetry)


class AcknowledgingFakeGpu(FakeGpu):
    def __init__(self, active_backend: str = "cuda") -> None:
        super().__init__(active_backend)
        self.acknowledged = False
        self.acknowledgements: list[tuple[str, str, str]] = []
        self.acknowledgement_reason = ""
        self.acknowledgement_scope = "simulation"
        self.recommendation = None

    def acknowledge_backend_mismatch(
        self, challenge: str, reason: str, scope: str = "simulation"
    ) -> None:
        self.acknowledgements.append((challenge, reason, scope))
        self.acknowledgement_reason = reason
        self.acknowledgement_scope = scope
        self.acknowledged = True

    def advance(self) -> None:
        if self.active_backend == "cpu":
            self.cpu_calls += 3
        else:
            self.cuda_calls += 3


class GuardedFakeSimulation(FakeSimulation):
    def _guarded_auxiliary(self, name: str):
        assert isinstance(self.gpu, AcknowledgingFakeGpu)
        if not self.gpu.acknowledged:
            telemetry = fake_recommendation("cuda", "cpu")
            raise FakeBackendRecommendationMismatch(
                str(telemetry["acknowledgement_digest"]), telemetry
            )
        self.gpu.acknowledged = False
        self.gpu.recommendation = fake_recommendation(
            "cuda",
            "cpu",
            acknowledgement=True,
            reason=self.gpu.acknowledgement_reason,
            acknowledgement_scope=self.gpu.acknowledgement_scope,
        )
        return name

    def solve_cw(self, *args, **kwargs):
        return self._guarded_auxiliary("solve_cw")

    def solve_eigfreq(self, *args, **kwargs):
        return self._guarded_auxiliary("solve_eigfreq")

    def run(self, *step_funcs, **kwargs):
        assert isinstance(self.gpu, AcknowledgingFakeGpu)
        if not self.gpu.acknowledged:
            telemetry = fake_recommendation("cuda", "cpu")
            raise FakeBackendRecommendationMismatch(
                str(telemetry["acknowledgement_digest"]), telemetry
            )
        self.gpu.acknowledged = False
        self.gpu.recommendation = fake_recommendation(
            "cuda",
            "cpu",
            acknowledgement=True,
            reason=self.gpu.acknowledgement_reason,
            acknowledgement_scope=self.gpu.acknowledgement_scope,
        )
        super().run(*step_funcs, **kwargs)


class MutatingGuardedFakeSimulation(GuardedFakeSimulation):
    def run(self, *step_funcs, **kwargs):
        assert isinstance(self.gpu, AcknowledgingFakeGpu)
        if not self.gpu.acknowledged:
            self.runs += 1
        super().run(*step_funcs, **kwargs)


class PristineClassFakeSimulation(FakeSimulation):
    def run(self, *step_funcs, **kwargs):
        if type(self).run is not PristineClassFakeSimulation.run:
            raise RuntimeError("oracle observer leaked into the workload class")
        super().run(*step_funcs, **kwargs)


class RunMutatingFakeSimulation(FakeSimulation):
    def run(self, *step_funcs, **kwargs):
        type(self).run = lambda *_args, **_kwargs: None
        super().run(*step_funcs, **kwargs)


def fake_meep_module(
    simulation_class=FakeSimulation, active_backend: str = "cpu"
) -> types.ModuleType:
    fake_meep = types.ModuleType("meep")
    fake_meep.gpu = FakeGpu(active_backend)
    simulation_class.gpu = fake_meep.gpu
    fake_meep.Simulation = simulation_class
    fake_meep.backend_policy = types.SimpleNamespace(
        BackendRecommendationMismatch=FakeBackendRecommendationMismatch
    )
    return fake_meep


def guarded_fake_meep_module(
    simulation_class=GuardedFakeSimulation,
) -> types.ModuleType:
    fake_meep = types.ModuleType("meep")
    fake_meep.gpu = AcknowledgingFakeGpu()
    simulation_class.gpu = fake_meep.gpu
    fake_meep.Simulation = simulation_class
    fake_meep.backend_policy = types.SimpleNamespace(
        BackendRecommendationMismatch=FakeBackendRecommendationMismatch
    )
    return fake_meep


def upstream_fake_meep_module() -> types.ModuleType:
    fake_meep = types.ModuleType("meep")
    fake_meep.Simulation = UpstreamFakeSimulation
    return fake_meep


def contract_args(example: pathlib.Path, *extra: str) -> list[str]:
    return [
        "--expected-run-count",
        "2",
        "--expected-final-timestep",
        "50",
        "--expected-total-timestep-delta",
        "50",
        *extra,
        str(example),
    ]


class ExampleOracleTests(unittest.TestCase):
    def test_disabled_field_timing_snapshot_never_invokes_observer(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "example.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n",
                    encoding="utf-8",
                )
                with mock.patch.object(
                    ORACLE,
                    "field_timing_snapshot",
                    side_effect=AssertionError("intrusive timing observer ran"),
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(
                            ORACLE.main(
                                contract_args(
                                    example,
                                    "--disable-field-timing-snapshots",
                                )
                            ),
                            0,
                        )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_backend_policy_is_resolved_from_installed_submodule(self) -> None:
        package = types.ModuleType("synthetic_meep")
        package.__path__ = []
        policy = types.ModuleType("synthetic_meep.backend_policy")
        policy.BackendRecommendationMismatch = FakeBackendRecommendationMismatch
        previous_package = sys.modules.get("synthetic_meep")
        previous_policy = sys.modules.get("synthetic_meep.backend_policy")
        sys.modules["synthetic_meep"] = package
        sys.modules["synthetic_meep.backend_policy"] = policy
        try:
            self.assertFalse(hasattr(package, "backend_policy"))
            self.assertIs(
                ORACLE.backend_recommendation_mismatch_type(package),
                FakeBackendRecommendationMismatch,
            )
        finally:
            if previous_policy is None:
                sys.modules.pop("synthetic_meep.backend_policy", None)
            else:
                sys.modules["synthetic_meep.backend_policy"] = previous_policy
            if previous_package is None:
                sys.modules.pop("synthetic_meep", None)
            else:
                sys.modules["synthetic_meep"] = previous_package

    def test_backend_policy_exception_type_fails_closed(self) -> None:
        fake_meep = types.ModuleType("meep")
        fake_meep.backend_policy = types.SimpleNamespace(
            BackendRecommendationMismatch="not-an-exception"
        )
        previous_meep = sys.modules.get("meep")
        previous_policy = sys.modules.pop("meep.backend_policy", None)
        sys.modules["meep"] = fake_meep
        try:
            with self.assertRaisesRegex(
                RuntimeError, "exception type is unavailable"
            ):
                ORACLE.backend_recommendation_mismatch_type(fake_meep)
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
            if previous_policy is not None:
                sys.modules["meep.backend_policy"] = previous_policy

    def test_upstream_cpu_baseline_has_no_gpmeep_gpu_dependency(self) -> None:
        fake_meep = upstream_fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "upstream.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--upstream-cpu-baseline",
                                *contract_args(example),
                            ]
                        ),
                        0,
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        audit_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith(ORACLE.AUDIT_PREFIX)
        )
        runs = json.loads(audit_line[len(ORACLE.AUDIT_PREFIX) :])["runs"]
        self.assertEqual([record["active_backend"] for record in runs], ["cpu", "cpu"])
        self.assertEqual(
            [record["execution_diagnostic"] for record in runs],
            ["upstream-meep-cpu-baseline", "upstream-meep-cpu-baseline"],
        )
        self.assertEqual([record["statistics_delta"] for record in runs], [{}, {}])
        self.assertEqual(
            [record["run_phase_calls"] for record in runs],
            [{"cpu": 0, "cuda": 0}, {"cpu": 0, "cuda": 0}],
        )

    def test_upstream_cpu_baseline_refuses_gpmeep_or_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ORACLE.parse_args(
                    [
                        "--upstream-cpu-baseline",
                        "--expected-run-count",
                        "1",
                        "--expected-final-timestep",
                        "1",
                        "--acknowledge-backend-mismatch-reason",
                        "M3 explicit comparison run",
                        str(example),
                    ]
                )

        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "gpmeep.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "refuses a package"):
                    ORACLE.main(
                        [
                            "--upstream-cpu-baseline",
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_acknowledgement_option_abbreviations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            for option in ("--ack", "--acknowledge-backend-mismatch-r"):
                with self.subTest(option=option), self.assertRaises(SystemExit):
                    ORACLE.parse_args(
                        [
                            "--expected-run-count",
                            "1",
                            option,
                            "embedded old reason",
                            str(example),
                        ]
                    )

    def test_statistics_delta_allows_only_live_gauge_decreases(self) -> None:
        before = {
            "resident": {
                "live_resident_device_buffers": 4,
                "device_buffer_allocations": 7,
            }
        }
        after = {
            "resident": {
                "live_resident_device_buffers": 2,
                "device_buffer_allocations": 9,
            }
        }
        delta = ORACLE.statistics_delta(before, after)
        self.assertEqual(delta["resident"]["live_resident_device_buffers"], -2)
        self.assertEqual(delta["resident"]["device_buffer_allocations"], 2)
        after["resident"]["device_buffer_allocations"] = 6
        with self.assertRaisesRegex(RuntimeError, "statistics reset"):
            ORACLE.statistics_delta(before, after)
        missing = {
            "resident": {"live_resident_device_buffers": 2},
        }
        with self.assertRaisesRegex(RuntimeError, "disappeared or changed type"):
            ORACLE.statistics_delta(before, missing)
        with self.assertRaisesRegex(RuntimeError, "disappeared or changed type"):
            ORACLE.statistics_delta(before, {})
        changed_type = {
            "resident": {
                "live_resident_device_buffers": 2,
                "device_buffer_allocations": "nine",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "disappeared or changed type"):
            ORACLE.statistics_delta(before, changed_type)

    def test_explicit_backend_mismatch_acknowledgement_is_audited(self) -> None:
        fake_meep = guarded_fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "guarded.py"
                example.write_text(
                    "import meep as mp\n"
                    "mp.Simulation().run(until=5)\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "50",
                                "--acknowledge-backend-mismatch-reason",
                                "M3 forced CUDA classification run",
                                str(example),
                            ]
                        ),
                        0,
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        audit_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith(ORACLE.AUDIT_PREFIX)
        )
        audit = json.loads(audit_line[len(ORACLE.AUDIT_PREFIX) :])
        guard = audit["runs"][0]["backend_guard"]
        explicit = audit["runs"][0]["explicit_backend_guard"]
        self.assertEqual(guard["attempts"], 2)
        self.assertEqual(guard["acknowledgements"], 1)
        self.assertGreaterEqual(guard["warning_wall_seconds"], 0.0)
        self.assertEqual(
            guard["warning_telemetry"]["recommended_backend"], "cpu"
        )
        self.assertTrue(guard["final_telemetry"]["acknowledgement"])
        self.assertEqual(explicit["schema"], ORACLE.qualification_guard.SCHEMA)
        self.assertEqual(explicit["attempts"], 2)
        self.assertEqual(len(fake_meep.gpu.acknowledgements), 1)
        self.assertEqual(
            fake_meep.gpu.acknowledgements[0][1],
            "M3 forced CUDA classification run",
        )
        self.assertEqual(
            fake_meep.gpu.acknowledgements[0][0],
            explicit["final_telemetry"]["acknowledgement_digest"],
        )

    def test_shared_oracle_exposes_no_unprovenanced_historical_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ORACLE.parse_args(
                    [
                        "--legacy-gpmeep-baseline",
                        "--expected-run-count",
                        "1",
                        "--expected-final-timestep",
                        "1",
                        str(example),
                    ]
                )

    def test_frozen_runtime_capability_rejects_forged_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid frozen"):
                ORACLE.main(
                    contract_args(example),
                    _frozen_v103_runtime_token=object(),
                )

    def test_auxiliary_guarded_entrypoints_are_acknowledged_and_audited(self) -> None:
        fake_meep = guarded_fake_meep_module()
        original_solve_cw = fake_meep.Simulation.solve_cw
        original_solve_eigfreq = fake_meep.Simulation.solve_eigfreq
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "guarded_auxiliary.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "assert sim.solve_cw() == 'solve_cw'\n"
                    "assert sim.solve_eigfreq() == 'solve_eigfreq'\n"
                    "sim.run(until=5)\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "50",
                                "--acknowledge-backend-mismatch-reason",
                                "M3 forced CUDA classification run",
                                str(example),
                            ]
                        ),
                        0,
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertIs(fake_meep.Simulation.solve_cw, original_solve_cw)
        self.assertIs(fake_meep.Simulation.solve_eigfreq, original_solve_eigfreq)
        auxiliary_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith(ORACLE.AUXILIARY_GUARD_PREFIX)
        )
        auxiliary = json.loads(
            auxiliary_line[len(ORACLE.AUXILIARY_GUARD_PREFIX) :]
        )
        self.assertEqual(
            [event["entrypoint"] for event in auxiliary["events"]],
            ["solve_cw", "solve_eigfreq"],
        )
        self.assertTrue(
            all(event["guard"]["attempts"] == 2 for event in auxiliary["events"])
        )
        self.assertEqual(len(fake_meep.gpu.acknowledgements), 3)

    def test_backend_mismatch_is_not_silently_acknowledged(self) -> None:
        fake_meep = guarded_fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "guarded.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=5)\n",
                    encoding="utf-8",
                )
                with self.assertRaises(FakeBackendRecommendationMismatch):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "50",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertEqual(fake_meep.gpu.acknowledgements, [])

    def test_backend_mismatch_mutation_is_never_acknowledged(self) -> None:
        fake_meep = guarded_fake_meep_module(MutatingGuardedFakeSimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "mutating_guard.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=5)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "mutated simulation state"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "50",
                            "--acknowledge-backend-mismatch-reason",
                            "M3 forced CUDA classification run",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertEqual(fake_meep.gpu.acknowledgements, [])

    def test_backend_acknowledgement_reason_length_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            base = [
                "--expected-run-count",
                "1",
                "--expected-final-timestep",
                "1",
                "--acknowledge-backend-mismatch-reason",
            ]
            for reason in ("short", "x" * 513):
                with self.subTest(length=len(reason)), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, reason, str(example)])

    def test_original_run_observes_pristine_simulation_class(self) -> None:
        original_run = PristineClassFakeSimulation.run
        fake_meep = fake_meep_module(PristineClassFakeSimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "pristine_class.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=5)\n",
                    encoding="utf-8",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "50",
                                str(example),
                            ]
                        ),
                        0,
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertIs(PristineClassFakeSimulation.run, original_run)

    def test_run_k_point_observer_records_and_restores_selected_method(self) -> None:
        original_run = FakeSimulation.run
        original_run_k_point = FakeSimulation.run_k_point
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "run_k_points.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run_k_point(2, 'first')\n"
                    "sim.run_k_point(3, 'second')\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--observe-operation",
                                "run_k_point",
                                "--expected-run-count",
                                "2",
                                "--expected-final-timestep",
                                "30",
                                "--expected-total-timestep-delta",
                                "50",
                                str(example),
                            ]
                        ),
                        0,
                    )
                metrics_line = next(
                    line
                    for line in output.getvalue().splitlines()
                    if line.startswith(ORACLE.METRIC_PREFIX)
                )
                metrics = json.loads(metrics_line.removeprefix(ORACLE.METRIC_PREFIX))
                self.assertEqual(metrics["run_count"], 2)
                self.assertEqual(metrics["per_run"]["timestep_delta"], [20, 30])
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertIs(FakeSimulation.run, original_run)
        self.assertIs(FakeSimulation.run_k_point, original_run_k_point)

    def test_original_run_class_mutation_fails_closed_and_is_restored(self) -> None:
        original_run = RunMutatingFakeSimulation.run
        fake_meep = fake_meep_module(RunMutatingFakeSimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "mutating_class.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=5)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "changed while the original implementation was active",
                ):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "50",
                            str(example),
                        ]
                    )
        finally:
            RunMutatingFakeSimulation.run = original_run
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        self.assertIs(RunMutatingFakeSimulation.run, original_run)

    def test_unchanged_example_emits_aggregated_run_metrics(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "example.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(ORACLE.main(contract_args(example)), 0)
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        lines = output.getvalue().splitlines()
        checkpoints = [
            json.loads(line[len(ORACLE.CHECKPOINT_PREFIX) :])
            for line in lines
            if line.startswith(ORACLE.CHECKPOINT_PREFIX)
        ]
        self.assertEqual([item["run_index"] for item in checkpoints], [0, 1])
        self.assertEqual(
            checkpoints[0]["record"]["timestep_delta"],
            20,
        )
        audit_line = next(
            line for line in lines if line.startswith(ORACLE.AUDIT_PREFIX)
        )
        audit = json.loads(audit_line[len(ORACLE.AUDIT_PREFIX) :])
        self.assertEqual(audit["runs"][0]["run_phase_calls"]["cpu"], 3)
        self.assertEqual(audit["runs"][0]["measurement_phase_calls"]["cpu"], 0)
        self.assertGreater(audit["runs"][0]["run_wall_seconds"], 0)
        self.assertEqual(audit["runs"][0]["requested_backend"], "cpu")
        self.assertEqual(
            audit["runs"][0]["execution_diagnostic"],
            "synthetic execution diagnostic",
        )
        self.assertIsNone(audit["runs"][0]["field_timing_delta"])
        self.assertEqual(
            audit["runs"][0]["statistics_delta"]["dispatch"][
                "cpu_curl_calls"
            ],
            3,
        )
        self.assertEqual(
            audit["runs"][0]["runtime_counters"][
                "runtime_availability_probes"
            ],
            0,
        )
        line = next(
            item for item in lines if item.startswith(ORACLE.METRIC_PREFIX)
        )
        self.assertTrue(line.startswith(ORACLE.METRIC_PREFIX))
        metrics = json.loads(line[len(ORACLE.METRIC_PREFIX) :])
        self.assertEqual(metrics["run_count"], 2)
        self.assertEqual(metrics["timestep"]["final"], 50)
        self.assertEqual(metrics["timestep"]["sum"], 70)
        self.assertEqual(metrics["timestep_delta"]["sum"], 50)
        self.assertEqual(metrics["field_energy"]["weighted_checksum"], 18.0)
        self.assertEqual(metrics["dft_norm"]["l2"], (4.0**2 + 10.0**2) ** 0.5)
        self.assertEqual(metrics["per_run"]["timestep"], [20, 50])
        self.assertEqual(metrics["per_run"]["timestep_delta"], [20, 30])
        self.assertEqual(metrics["per_run"]["field_energy"], [3.0, 7.5])

    def test_example_without_run_is_rejected(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "empty.py"
                example.write_text("import meep\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "without calling"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "1",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_zero_timestep_run_is_rejected_even_if_measurement_dispatches(self) -> None:
        class EmptySimulation(FakeSimulation):
            def run(self, *step_funcs, **kwargs):
                return None

            def field_energy_in_box(self, box=None) -> float:
                assert self.gpu is not None
                self.gpu.advance()
                return 0.0

        fake_meep = fake_meep_module(EmptySimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "empty_run.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=0)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "zero timesteps"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "0",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_one_zero_work_run_cannot_hide_behind_a_later_run(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "partial_noop.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=0)\n"
                    "sim.run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "record indices \\[0\\]"
                ):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "2",
                            "--expected-final-timestep",
                            "10",
                            "--expected-total-timestep-delta",
                            "10",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_progress_without_run_dispatch_is_rejected(self) -> None:
        class NoDispatchSimulation(FakeSimulation):
            def run(self, *step_funcs, **kwargs):
                self.runs += int(kwargs["until"])

        fake_meep = fake_meep_module(NoDispatchSimulation)
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "no_dispatch.py"
                example.write_text(
                    "import meep as mp\nmp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "without selected-backend"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(example),
                        ]
                    )
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_contract_requires_expected_count_and_work_or_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ORACLE.parse_args([str(example)])
            with self.assertRaises(SystemExit):
                ORACLE.parse_args(
                    ["--expected-run-count", "1", str(example)]
                )

    def test_verbose_unittest_discovery_preserves_canonical_identities(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        module_name = "test_gpmeep_oracle_discovery_fixture"
        previous_fixture = sys.modules.pop(module_name, None)
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / f"{module_name}.py"
                example.write_text(
                    "import unittest\n"
                    "import meep as mp\n"
                    "class InstrumentedTests(unittest.TestCase):\n"
                    "    def test_first(self):\n"
                    "        mp.Simulation().run(until=2)\n"
                    "    def test_second(self):\n"
                    "        mp.Simulation().run(until=3)\n",
                    encoding="utf-8",
                )
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--unittest-discover",
                                "--expected-run-count",
                                "2",
                                "--expected-total-timestep-delta",
                                "50",
                                str(example),
                            ]
                        ),
                        0,
                    )
        finally:
            sys.modules.pop(module_name, None)
            if previous_fixture is not None:
                sys.modules[module_name] = previous_fixture
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        verbose = stderr.getvalue()
        self.assertIn(
            "test_first "
            "(test_gpmeep_oracle_discovery_fixture.InstrumentedTests.test_first)",
            verbose,
        )
        self.assertIn(
            "test_second "
            "(test_gpmeep_oracle_discovery_fixture.InstrumentedTests.test_second)",
            verbose,
        )
        self.assertIn("Ran 2 tests", verbose)
        self.assertTrue(verbose.rstrip().endswith("OK"))
        audit_line = next(
            line
            for line in stdout.getvalue().splitlines()
            if line.startswith(ORACLE.AUDIT_PREFIX)
        )
        runs = json.loads(audit_line[len(ORACLE.AUDIT_PREFIX) :])["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual([item["timestep_delta"] for item in runs], [20, 30])

    def test_unittest_discovery_cli_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "test_fixture.py"
            example.write_text("import unittest\n", encoding="utf-8")
            base = [
                "--unittest-discover",
                "--expected-run-count",
                "1",
                "--expected-final-timestep",
                "1",
            ]
            for extra in (
                [str(example), "--", "--unexpected"],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1",
                    str(example),
                ],
            ):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, *extra])

    def test_per_run_signal_contract_rejects_one_zero_record(self) -> None:
        records = [
            {
                "timestep": 10,
                "timestep_delta": 10,
                "field_energy": 0.0,
                "dft_norm": 2.0,
            },
            {
                "timestep": 20,
                "timestep_delta": 10,
                "field_energy": 3.0,
                "dft_norm": 4.0,
            },
        ]
        args = types.SimpleNamespace(
            expected_run_count=2,
            expected_final_timestep=20,
            expected_total_timestep_delta=20,
            min_field_energy=1.0,
            min_dft_norm=None,
            min_each_field_energy=1.0,
            min_each_dft_norm=None,
        )
        with self.assertRaisesRegex(RuntimeError, "record indices \\[0\\]"):
            ORACLE.validate_contract(records, args)

    def test_result_vectors_are_captured_with_direct_complex_values(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "results.py"
                example.write_text(
                    "import meep as mp\n"
                    "sim = mp.Simulation()\n"
                    "sim.run(until=2)\n"
                    "sim.run(until=3)\n"
                    "spectrum = [[1.0, 2.0 + 3.0j], [4.0, -5.0j]]\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                args = contract_args(
                    example,
                    "--result-vector",
                    "spectrum",
                    "--expected-result-shape",
                    "spectrum=2,2",
                    "--min-result-l2",
                    "spectrum=1",
                )
                with contextlib.redirect_stdout(output):
                    self.assertEqual(ORACLE.main(args), 0)
        finally:
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep
        metric_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith(ORACLE.METRIC_PREFIX)
        )
        metrics = json.loads(metric_line[len(ORACLE.METRIC_PREFIX) :])
        result = metrics["example_results"]["spectrum"]
        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["real"], [1.0, 2.0, 4.0, 0.0])
        self.assertEqual(result["imag"], [0.0, 3.0, 0.0, -5.0])
        scalar = ORACLE.numeric_result_vector(0.25, "scalar")
        self.assertEqual(scalar["rank"], 0)
        self.assertNotIn("shape", scalar)

    def test_result_vector_contract_rejects_invalid_results(self) -> None:
        cases = (
            ("missing", {}, "did not define"),
            ("ragged", {"ragged": [[1.0], [2.0, 3.0]]}, "ragged"),
            ("bad", {"bad": [float("inf")]}, "non-finite"),
            ("weak", {"weak": [1e-9]}, "below required signal"),
            ("large", {"large": [2.0]}, "exceeds required limit"),
        )
        for name, namespace, expected in cases:
            with self.subTest(name=name):
                args = types.SimpleNamespace(
                    result_vector=[name],
                    expected_result_shape=[(name, (1,))],
                    min_result_l2=[(name, 1e-6)],
                    max_result_abs=[(name, 1.0)],
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    ORACLE.collect_example_results(namespace, args)

    def test_result_vector_cli_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            base = [
                "--expected-run-count",
                "1",
                "--expected-final-timestep",
                "1",
            ]
            for extra in (
                ["--result-vector", "bad.name"],
                ["--result-vector", "value", "--result-vector", "value"],
                ["--min-result-l2", "value=1"],
                ["--result-vector", "value"],
                ["--expected-result-shape", "value=1"],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1",
                    "--expected-result-shape",
                    "value=2",
                ],
                ["--result-vector", "value", "--expected-result-shape", "value=0"],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1,bad",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--min-result-l2", "other=1",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--min-result-l2", "value=inf",
                ],
                [
                    "--result-vector", "value", "--expected-result-shape", "value=1",
                    "--max-result-abs", "other=1",
                ],
                [
                    "--result-vector",
                    "value",
                    "--expected-result-shape",
                    "value=1",
                    "--max-result-abs",
                    "value=1",
                    "--max-result-abs",
                    "value=2",
                ],
            ):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, *extra, str(example)])

    def test_result_shape_contract_rejects_shrink_and_rank_change(self) -> None:
        cases = (
            ([1.0], (2,), "shape"),
            (1.0, (1,), "shape"),
            ([1.0], (), "shape"),
        )
        for value, expected_shape, message in cases:
            with self.subTest(value=value, expected_shape=expected_shape):
                args = types.SimpleNamespace(
                    result_vector=["value"],
                    expected_result_shape=[("value", expected_shape)],
                    min_result_l2=[],
                    max_result_abs=[],
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    ORACLE.collect_example_results({"value": value}, args)

    def test_per_run_timestep_ranges_are_fail_closed(self) -> None:
        records = [
            {
                "timestep": 12,
                "timestep_delta": 10,
                "field_energy": 1.0,
                "dft_norm": 1.0,
            },
            {
                "timestep": 25,
                "timestep_delta": 20,
                "field_energy": 1.0,
                "dft_norm": 1.0,
            },
        ]
        args = types.SimpleNamespace(
            expected_run_count=2,
            expected_final_timestep=None,
            expected_total_timestep_delta=None,
            expected_run_timestep_range=[(10, 15), (20, 30)],
            expected_run_timestep_delta_range=[(8, 12), (18, 22)],
            min_field_energy=None,
            min_dft_norm=None,
            min_each_field_energy=None,
            min_each_dft_norm=None,
        )
        ORACLE.validate_contract(records, args)
        args.expected_run_timestep_range[1] = (26, 30)
        with self.assertRaisesRegex(RuntimeError, "outside required range"):
            ORACLE.validate_contract(records, args)

        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "example.py"
            example.write_text("pass\n", encoding="utf-8")
            base = ["--expected-run-count", "2"]
            invalid = (
                ["--expected-run-timestep-range", "1:2"],
                [
                    "--expected-final-timestep", "1",
                    "--expected-run-timestep-range", "1:2",
                    "--expected-run-timestep-range", "1:2",
                ],
                ["--expected-run-timestep-range", "2:1"],
                ["--expected-run-timestep-range", "0:1"],
                ["--expected-run-timestep-delta-range", "0:1"],
            )
            for extra in invalid:
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    ORACLE.parse_args([*base, *extra, str(example)])

    def test_oversized_array_is_rejected_before_tolist(self) -> None:
        class HugeArray:
            shape = (10**12,)
            size = 10**12
            tolist_called = False

            def tolist(self):
                self.tolist_called = True
                raise AssertionError("tolist must not be called")

        value = HugeArray()
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            ORACLE.numeric_result_vector(value, "huge")
        self.assertFalse(value.tolist_called)

    def test_binary_grating_acos_rejects_large_domain_violation(self) -> None:
        source_path = ORACLE_PATH.parents[2] / "python/examples/binary_grating.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "bounded_direction_cosine_acos"
        )
        eps = 1.1920928955078125e-7
        fake_numpy = types.SimpleNamespace(
            float32=object(),
            finfo=lambda _: types.SimpleNamespace(eps=eps),
        )
        namespace = {
            "math": __import__("math"),
            "np": fake_numpy,
            "direction_cosine_max_overshoot": 0.0,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(source_path), "exec"), namespace)
        bounded_acos = namespace["bounded_direction_cosine_acos"]
        bounded_acos(1.0 + 16 * eps)
        self.assertGreater(namespace["direction_cosine_max_overshoot"], 0.0)
        for invalid in (1.0 + 64 * eps, -1.0 - 64 * eps, float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                bounded_acos(invalid)

    def test_system_exit_and_run_exception_restore_process_state(self) -> None:
        fake_meep = fake_meep_module()
        original_run = fake_meep.Simulation.run
        previous_meep = sys.modules.get("meep")
        previous_argv = sys.argv
        previous_path = list(sys.path)
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory_path = pathlib.Path(directory)
                successful = directory_path / "successful.py"
                successful.write_text(
                    "import meep as mp\nmp.Simulation().run(until=1)\n"
                    "raise SystemExit(0)\n",
                    encoding="utf-8",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "10",
                                str(successful),
                            ]
                        ),
                        0,
                    )
                failing_exit = directory_path / "failing_exit.py"
                failing_exit.write_text("raise SystemExit(7)\n", encoding="utf-8")
                with self.assertRaises(SystemExit) as raised:
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(failing_exit),
                        ]
                    )
                self.assertEqual(raised.exception.code, 7)
                failing_run = directory_path / "failing_run.py"
                failing_run.write_text(
                    "import meep as mp\n"
                    "def fail(self, *args, **kwargs): raise RuntimeError('run failed')\n"
                    "mp.Simulation.run = fail\n"
                    "mp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "run failed"):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(failing_run),
                        ]
                    )
        finally:
            self.assertIs(fake_meep.Simulation.run, original_run)
            self.assertIs(sys.argv, previous_argv)
            self.assertEqual(sys.path, previous_path)
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep

    def test_original_run_and_metric_exceptions_restore_wrapper(self) -> None:
        class RaisingRunSimulation(FakeSimulation):
            def run(self, *args, **kwargs):
                raise RuntimeError("original run failed")

        class RaisingMetricSimulation(FakeSimulation):
            def field_energy_in_box(self, box=None):
                raise RuntimeError("metric failed")

        for label, simulation_class, expected in (
            ("run", RaisingRunSimulation, "original run failed"),
            ("metric", RaisingMetricSimulation, "metric failed"),
        ):
            with self.subTest(label=label):
                fake_meep = fake_meep_module(simulation_class)
                original_run = fake_meep.Simulation.run
                previous_meep = sys.modules.get("meep")
                previous_argv = sys.argv
                previous_path = list(sys.path)
                sys.modules["meep"] = fake_meep
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        example = pathlib.Path(directory) / "failing.py"
                        example.write_text(
                            "import meep as mp\n"
                            "mp.Simulation().run(until=1)\n",
                            encoding="utf-8",
                        )
                        with self.assertRaisesRegex(RuntimeError, expected):
                            ORACLE.main(
                                [
                                    "--expected-run-count",
                                    "1",
                                    "--expected-final-timestep",
                                    "10",
                                    str(example),
                                ]
                            )
                finally:
                    self.assertIs(fake_meep.Simulation.run, original_run)
                    self.assertIs(sys.argv, previous_argv)
                    self.assertEqual(sys.path, previous_path)
                    if previous_meep is None:
                        sys.modules.pop("meep", None)
                    else:
                        sys.modules["meep"] = previous_meep

    def test_sys_argv_object_and_contents_restore_after_example_rebind(self) -> None:
        fake_meep = fake_meep_module()
        previous_meep = sys.modules.get("meep")
        previous_argv = sys.argv
        previous_argv_contents = list(previous_argv)
        sys.modules["meep"] = fake_meep
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory_path = pathlib.Path(directory)
                successful = directory_path / "rebind_success.py"
                successful.write_text(
                    "import sys\n"
                    "import meep as mp\n"
                    "sys.argv = ['example-rebound']\n"
                    "mp.Simulation().run(until=1)\n",
                    encoding="utf-8",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        ORACLE.main(
                            [
                                "--expected-run-count",
                                "1",
                                "--expected-final-timestep",
                                "10",
                                str(successful),
                            ]
                        ),
                        0,
                    )
                self.assertIs(sys.argv, previous_argv)
                self.assertEqual(sys.argv, previous_argv_contents)

                failing = directory_path / "rebind_failure.py"
                failing.write_text(
                    "import sys\n"
                    "sys.argv = ['failure-rebound']\n"
                    "raise RuntimeError('failure after argv rebind')\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "failure after argv rebind"
                ):
                    ORACLE.main(
                        [
                            "--expected-run-count",
                            "1",
                            "--expected-final-timestep",
                            "10",
                            str(failing),
                        ]
                    )
                self.assertIs(sys.argv, previous_argv)
                self.assertEqual(sys.argv, previous_argv_contents)
        finally:
            sys.argv = previous_argv
            previous_argv[:] = previous_argv_contents
            if previous_meep is None:
                sys.modules.pop("meep", None)
            else:
                sys.modules["meep"] = previous_meep


if __name__ == "__main__":
    unittest.main()
