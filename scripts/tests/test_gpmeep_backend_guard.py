#!/usr/bin/env python3
"""Tests for reason-bound explicit-backend qualification acknowledgement."""

from __future__ import annotations

import copy
import pathlib
import sys
import types
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpmeep_backend_guard as guard  # noqa: E402


REASON = "release validation intentionally compares explicit CPU and CUDA"


class Mismatch(RuntimeError):
    def __init__(self, challenge, telemetry):
        super().__init__("backend mismatch")
        self.challenge = challenge
        self.telemetry = telemetry


class FakeGpu:
    def __init__(
        self,
        selected="cuda",
        recommended="cpu",
        *,
        reason_class="hardware_crossover",
        acknowledgement_allowed=True,
    ):
        self.selected = selected
        self.recommended = recommended
        self.reason_class = reason_class
        self.acknowledgement_allowed = acknowledgement_allowed
        self.acknowledged = False
        self.reused = False
        self.acknowledgement_scope = "process"
        self.scope_cache_enabled = True
        self.acks = []
        self.recommendation = None

    def telemetry(self):
        mismatch = self.selected != self.recommended
        acknowledgement = self.acknowledged or self.reused
        exact = mismatch and not self.reused
        telemetry = {
            "schema": guard.TELEMETRY_SCHEMA,
            "selected_backend": self.selected,
            "recommended_backend": self.recommended,
            "reason_class": self.reason_class,
            "recommendation_confidence": "high",
            "rank_local_cells": [1000],
            "run_length": {
                "kind": "until",
                "numeric_duration": 10.0,
                "source_end_time": None,
                "expected_timesteps": 200,
                "dynamic_condition": False,
            },
            "kernel_launch_count": 800,
            "transfer_bytes": 1024000,
            "monitor_count": 0,
            "monitor_frequency_count": 0,
            "source_count": 1,
            "mpi_gpu_topology": [
                {
                    "rank": 0,
                    "global_rank": 0,
                    "active_size": 1,
                    "node_rank": 0,
                    "node_size": 1,
                    "device_identifier": "fake-device",
                    "device_ordinal": 0,
                }
            ],
            "profiler_or_ablation_id": guard.PROFILER_OR_ABLATION_ID,
            "acknowledgement": acknowledgement,
            "acknowledgement_allowed": self.acknowledgement_allowed,
            "acknowledgement_reason": REASON if acknowledgement else "",
            "acknowledgement_reused": self.reused,
            "acknowledgement_scope": (
                self.acknowledgement_scope if acknowledgement else ""
            ),
            "acknowledgement_origin_cohort_digest": (
                "7" * 64 if self.reused else ""
            ),
            "acknowledgement_scope_cache_enabled": self.scope_cache_enabled,
            "acknowledgement_scope_digest": "6" * 64,
            "workload_fingerprint_evaluated": exact,
            "workload_fingerprint": "1" * 64 if exact else "",
            "recommendation_policy_version": guard.POLICY_VERSION,
            "mpi_topology_digest": "2" * 64,
            "native_state_digest": "3" * 64 if exact else "",
            "process_nonce": "4" * 64 if exact else "",
            "run_nonce": "5" * 64 if exact else "",
            "acknowledgement_digest": "",
        }
        if exact:
            telemetry["acknowledgement_digest"] = guard._acknowledgement_digest(
                telemetry
            )
        if acknowledgement and not self.reused:
            telemetry["acknowledgement_origin_cohort_digest"] = telemetry[
                "acknowledgement_digest"
            ]
        return telemetry

    def acknowledge_backend_mismatch(
        self, challenge, reason, scope="simulation"
    ):
        self.acks.append((challenge, reason, scope))
        self.acknowledgement_scope = scope
        self.acknowledged = True


class FakeSimulation:
    def __init__(self, gpu):
        self.gpu = gpu
        self.fields = None
        self.calls = 0

    def run(self):
        self.calls += 1
        telemetry = self.gpu.telemetry()
        if (
            self.gpu.selected != self.gpu.recommended
            and not self.gpu.acknowledged
            and not self.gpu.reused
        ):
            raise Mismatch(telemetry["acknowledgement_digest"], telemetry)
        self.gpu.recommendation = telemetry
        self.fields = types.SimpleNamespace(t=10)
        return "PASS"


def fake_mp(gpu):
    return types.SimpleNamespace(
        gpu=gpu,
        backend_policy=types.SimpleNamespace(BackendRecommendationMismatch=Mismatch),
    )


class BackendGuardTests(unittest.TestCase):
    def test_acknowledgement_reason_utf8_bounds_are_fail_closed(self):
        for invalid in ("x" * 7, "x" * 513, "\ud800"):
            with self.subTest(repr=repr(invalid)), self.assertRaises(
                guard.BackendGuardEvidenceError
            ):
                guard.validate_reason(invalid)
        self.assertEqual(guard.validate_reason("가가가"), "가가가")

    def test_pristine_class_dispatch_restores_observer_after_failure(self):
        class Simulation:
            def run(self):
                raise ValueError("expected failure")

        original = Simulation.run

        def observer(self):
            return "observer"

        Simulation.run = observer
        with self.assertRaisesRegex(ValueError, "expected failure"):
            guard.invoke_pristine_class_method(
                Simulation, "run", observer, original, Simulation(), (), {}
            )
        self.assertIs(Simulation.run, observer)

    def test_pristine_class_dispatch_rejects_third_party_mutation(self):
        class Simulation:
            pass

        def replacement(self):
            return "replacement"

        def original(self):
            Simulation.run = replacement
            return "result"

        def observer(self):
            return "observer"

        Simulation.run = observer
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "changed while the original"
        ):
            guard.invoke_pristine_class_method(
                Simulation, "run", observer, original, Simulation(), (), {}
            )
        self.assertIs(Simulation.run, observer)

    def test_pristine_class_dispatch_restores_complete_instrumented_group(self):
        class Simulation:
            def run(self):
                self.saw_pristine_group = (
                    Simulation.run is original_run
                    and Simulation.solve_cw is original_solve_cw
                )
                return "result"

            def solve_cw(self):
                return "solve"

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def run_observer(self):
            return "run observer"

        def solve_cw_observer(self):
            return "solve observer"

        originals = {
            "run": original_run,
            "solve_cw": original_solve_cw,
        }
        observers = {
            "run": run_observer,
            "solve_cw": solve_cw_observer,
        }
        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        simulation = Simulation()
        result = guard.invoke_pristine_class_method(
            Simulation,
            "run",
            run_observer,
            original_run,
            simulation,
            (),
            {},
            original_methods=originals,
            guarded_methods=observers,
        )
        self.assertEqual(result, "result")
        self.assertTrue(simulation.saw_pristine_group)
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)

    def test_pristine_class_dispatch_rejects_sibling_method_mutation(self):
        class Simulation:
            def run(self):
                Simulation.solve_cw = replacement
                return "result"

            def solve_cw(self):
                return "solve"

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def replacement(self):
            return "replacement"

        def run_observer(self):
            return "run observer"

        def solve_cw_observer(self):
            return "solve observer"

        originals = {
            "run": original_run,
            "solve_cw": original_solve_cw,
        }
        observers = {
            "run": run_observer,
            "solve_cw": solve_cw_observer,
        }
        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError,
            "methods changed while the original.*solve_cw",
        ):
            guard.invoke_pristine_class_method(
                Simulation,
                "run",
                run_observer,
                original_run,
                Simulation(),
                (),
                {},
                original_methods=originals,
                guarded_methods=observers,
            )
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)

    def test_pristine_class_dispatch_freezes_method_maps_before_call(self):
        class Simulation:
            def run(self):
                originals["solve_cw"] = replacement
                observers["solve_cw"] = replacement
                return "result"

            def solve_cw(self):
                return "solve"

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def replacement(self):
            return "replacement"

        def run_observer(self):
            return "run observer"

        def solve_cw_observer(self):
            return "solve observer"

        originals = {
            "run": original_run,
            "solve_cw": original_solve_cw,
        }
        observers = {
            "run": run_observer,
            "solve_cw": solve_cw_observer,
        }
        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        result = guard.invoke_pristine_class_method(
            Simulation,
            "run",
            run_observer,
            original_run,
            Simulation(),
            (),
            {},
            original_methods=originals,
            guarded_methods=observers,
        )
        self.assertEqual(result, "result")
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)

    def test_matching_explicit_backend_runs_once_without_ack(self):
        gpu = FakeGpu("cpu", "cpu")
        simulation = FakeSimulation(gpu)
        result, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        self.assertEqual(result, "PASS")
        self.assertEqual(simulation.calls, 1)
        self.assertEqual(gpu.acks, [])
        self.assertFalse(evidence["mismatch_observed"])
        guard.validate_evidence(
            evidence, expected_backend="cpu", expected_reason=REASON
        )

    def test_matching_fingerprint_mode_is_lightweight_only_for_single_rank(self):
        singleton = FakeGpu("cpu", "cpu").telemetry()
        singleton["workload_fingerprint_evaluated"] = True
        for index, name in enumerate(
            (
                "workload_fingerprint",
                "native_state_digest",
                "process_nonce",
                "run_nonce",
            ),
            start=1,
        ):
            singleton[name] = str(index) * 64
        singleton["acknowledgement_digest"] = guard._acknowledgement_digest(
            singleton
        )
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError,
            "matching recommendation fingerprint mode differs",
        ):
            guard._telemetry(singleton, "singleton matching exact")

        mpi = copy.deepcopy(singleton)
        mpi["rank_local_cells"].append(mpi["rank_local_cells"][0])
        mpi["mpi_gpu_topology"][0]["active_size"] = 2
        mpi["mpi_gpu_topology"][0]["node_size"] = 2
        second_rank = copy.deepcopy(mpi["mpi_gpu_topology"][0])
        second_rank.update(
            {
                "rank": 1,
                "global_rank": 1,
                "node_rank": 1,
                "device_identifier": "fake-device-2",
                "device_ordinal": 1,
            }
        )
        mpi["mpi_gpu_topology"].append(second_rank)
        retained = guard._telemetry(mpi, "MPI matching exact")
        self.assertTrue(retained["workload_fingerprint_evaluated"])

        mpi_lightweight = copy.deepcopy(mpi)
        mpi_lightweight["workload_fingerprint_evaluated"] = False
        for name in (
            "workload_fingerprint",
            "native_state_digest",
            "process_nonce",
            "run_nonce",
            "acknowledgement_digest",
        ):
            mpi_lightweight[name] = ""
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError,
            "matching recommendation fingerprint mode differs",
        ):
            guard._telemetry(mpi_lightweight, "MPI matching lightweight")

    def test_matching_cpu_environment_recommendation_runs_without_ack(self):
        gpu = FakeGpu(
            "cpu",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        simulation = FakeSimulation(gpu)
        result, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        self.assertEqual(result, "PASS")
        self.assertFalse(
            evidence["final_telemetry"]["acknowledgement_allowed"]
        )
        guard.validate_evidence(
            evidence, expected_backend="cpu", expected_reason=REASON
        )

    def test_reused_acknowledgement_fingerprint_mode_binds_rank_count(self):
        def as_two_rank(telemetry):
            telemetry["rank_local_cells"].append(
                telemetry["rank_local_cells"][0]
            )
            telemetry["mpi_gpu_topology"][0]["active_size"] = 2
            telemetry["mpi_gpu_topology"][0]["node_size"] = 2
            second_rank = copy.deepcopy(telemetry["mpi_gpu_topology"][0])
            second_rank.update(
                {
                    "rank": 1,
                    "global_rank": 1,
                    "node_rank": 1,
                    "device_identifier": "fake-device-2",
                    "device_ordinal": 1,
                }
            )
            telemetry["mpi_gpu_topology"].append(second_rank)
            return telemetry

        for scope in ("simulation", "process"):
            with self.subTest(scope=scope, forged_mode="serial-exact"):
                gpu = FakeGpu()
                gpu.reused = True
                gpu.acknowledgement_scope = scope
                serial_exact = gpu.telemetry()
                serial_exact["workload_fingerprint_evaluated"] = True
                for index, name in enumerate(
                    (
                        "workload_fingerprint",
                        "native_state_digest",
                        "process_nonce",
                        "run_nonce",
                    ),
                    start=1,
                ):
                    serial_exact[name] = str(index) * 64
                serial_exact["acknowledgement_digest"] = (
                    guard._acknowledgement_digest(serial_exact)
                )
                with self.assertRaisesRegex(
                    guard.BackendGuardEvidenceError,
                    "reuse fingerprint mode differs",
                ):
                    guard._telemetry(serial_exact, "forged serial reuse")

            with self.subTest(scope=scope, forged_mode="mpi-empty"):
                gpu = FakeGpu()
                gpu.reused = True
                gpu.acknowledgement_scope = scope
                mpi_empty = as_two_rank(gpu.telemetry())
                with self.assertRaisesRegex(
                    guard.BackendGuardEvidenceError,
                    "reuse fingerprint mode differs",
                ):
                    guard._telemetry(mpi_empty, "forged MPI reuse")

    def test_environment_recommendation_cannot_recommend_cuda(self):
        gpu = FakeGpu(
            "cpu",
            "cuda",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "environment state differs"
        ):
            guard._telemetry(gpu.telemetry(), "invalid")

    def test_cuda_environment_failure_can_use_lightweight_telemetry(self):
        gpu = FakeGpu(
            "cuda",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        telemetry = gpu.telemetry()
        telemetry["workload_fingerprint_evaluated"] = False
        for name in (
            "workload_fingerprint",
            "native_state_digest",
            "process_nonce",
            "run_nonce",
            "acknowledgement_digest",
        ):
            telemetry[name] = ""
        retained = guard._telemetry(telemetry, "environment failure")
        self.assertEqual("environment", retained["reason_class"])
        self.assertFalse(retained["acknowledgement_allowed"])

    def test_invalid_environment_states_are_rejected(self):
        invalid = []

        recommended_cuda = FakeGpu(
            "cpu",
            "cuda",
            reason_class="environment",
            acknowledgement_allowed=False,
        ).telemetry()
        invalid.append(("recommended_cuda", recommended_cuda))

        acknowledgement_allowed = FakeGpu(
            "cpu",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=True,
        ).telemetry()
        invalid.append(("acknowledgement_allowed", acknowledgement_allowed))

        acknowledged_gpu = FakeGpu(
            "cuda",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        acknowledged_gpu.acknowledged = True
        invalid.append(("acknowledged", acknowledged_gpu.telemetry()))

        unacknowledged_reason = FakeGpu(
            "cpu",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        ).telemetry()
        unacknowledged_reason["acknowledgement_reason"] = REASON
        unacknowledged_reason["acknowledgement_digest"] = (
            guard._acknowledgement_digest(unacknowledged_reason)
        )
        invalid.append(("unacknowledged_reason", unacknowledged_reason))

        for name, telemetry in invalid:
            with self.subTest(name=name), self.assertRaises(
                guard.BackendGuardEvidenceError
            ):
                guard._telemetry(telemetry, "invalid")

    def test_non_acknowledgeable_cuda_environment_mismatch_is_not_acknowledged(self):
        gpu = FakeGpu(
            "cuda",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        simulation = FakeSimulation(gpu)
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "telemetry is inconsistent"
        ):
            guard.invoke_with_acknowledgement(
                fake_mp(gpu), simulation, simulation.run, REASON
            )
        self.assertEqual(simulation.calls, 1)
        self.assertEqual(gpu.acks, [])

    def test_non_acknowledged_cuda_environment_evidence_is_rejected(self):
        gpu = FakeGpu(
            "cpu",
            "cpu",
            reason_class="environment",
            acknowledgement_allowed=False,
        )
        simulation = FakeSimulation(gpu)
        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        evidence["selected_backend"] = "cuda"
        evidence["recommended_backend"] = "cpu"
        evidence["mismatch_observed"] = True
        evidence["attempts"] = 2
        evidence["acknowledgements"] = 1
        evidence["warning_telemetry"] = dict(evidence["final_telemetry"])
        for telemetry in (
            evidence["warning_telemetry"],
            evidence["final_telemetry"],
        ):
            telemetry["selected_backend"] = "cuda"
            telemetry["acknowledgement_digest"] = (
                guard._acknowledgement_digest(telemetry)
            )
        with self.assertRaises(guard.BackendGuardEvidenceError):
            guard.validate_evidence(
                evidence, expected_backend="cuda", expected_reason=REASON
            )

    def test_mismatch_warns_before_mutation_then_acknowledges_once(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        result, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        self.assertEqual(result, "PASS")
        self.assertEqual(simulation.calls, 2)
        self.assertEqual(len(gpu.acks), 1)
        self.assertEqual(gpu.acks[0][1], REASON)
        self.assertEqual(gpu.acks[0][2], "simulation")
        self.assertEqual(evidence["attempts"], 2)
        self.assertEqual(evidence["acknowledgements"], 1)
        guard.validate_evidence(
            evidence, expected_backend="cuda", expected_reason=REASON
        )

    def test_process_scope_requires_and_forwards_explicit_opt_in(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        result, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu),
            simulation,
            simulation.run,
            REASON,
            "process",
        )
        self.assertEqual(result, "PASS")
        self.assertEqual(gpu.acks[0][2], "process")
        self.assertEqual(
            evidence["final_telemetry"]["acknowledgement_scope"],
            "process",
        )

        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "scope must be"
        ):
            guard.invoke_with_acknowledgement(
                fake_mp(gpu), simulation, simulation.run, REASON, "global"
            )

    def test_reused_scope_runs_once_and_remains_audited(self):
        gpu = FakeGpu("cuda", "cpu")
        gpu.reused = True
        simulation = FakeSimulation(gpu)
        result, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        self.assertEqual(result, "PASS")
        self.assertEqual(simulation.calls, 1)
        self.assertEqual(gpu.acks, [])
        self.assertTrue(evidence["mismatch_observed"])
        self.assertTrue(evidence["acknowledgement_reused"])
        self.assertEqual(evidence["attempts"], 1)
        self.assertEqual(evidence["acknowledgements"], 0)
        self.assertFalse(
            evidence["final_telemetry"]["workload_fingerprint_evaluated"]
        )
        guard.validate_evidence(
            evidence, expected_backend="cuda", expected_reason=REASON
        )

    def test_non_acknowledgeable_warning_is_rejected(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)

        def run():
            telemetry = gpu.telemetry()
            telemetry["acknowledgement_allowed"] = False
            raise Mismatch("a" * 64, telemetry)

        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "acknowledgement permission differs"
        ):
            guard.invoke_with_acknowledgement(fake_mp(gpu), simulation, run, REASON)
        self.assertEqual(gpu.acks, [])

    def test_malformed_challenge_is_rejected_before_acknowledgement(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)

        def run():
            telemetry = gpu.telemetry()
            telemetry["acknowledgement_digest"] = "g" * 64
            raise Mismatch("g" * 64, telemetry)

        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "acknowledgement_digest differs"
        ):
            guard.invoke_with_acknowledgement(fake_mp(gpu), simulation, run, REASON)
        self.assertEqual(gpu.acks, [])

    def test_retry_mismatch_propagates_without_a_third_attempt(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)

        def always_mismatch():
            simulation.calls += 1
            telemetry = gpu.telemetry()
            raise Mismatch(telemetry["acknowledgement_digest"], telemetry)

        with self.assertRaises(Mismatch):
            guard.invoke_with_acknowledgement(
                fake_mp(gpu), simulation, always_mismatch, REASON
            )
        self.assertEqual(simulation.calls, 2)
        self.assertEqual(len(gpu.acks), 1)
        self.assertEqual(gpu.acks[0][1], REASON)

    def test_absent_reason_preserves_original_warning(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        with self.assertRaises(Mismatch):
            guard.invoke_with_acknowledgement(
                fake_mp(gpu), simulation, simulation.run, None
            )
        self.assertEqual(simulation.calls, 1)
        self.assertIsNone(simulation.fields)
        self.assertEqual(gpu.acks, [])

    def test_mutation_before_warning_is_rejected(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)

        def corrupt_run():
            simulation.fields = types.SimpleNamespace(t=1)
            raise Mismatch("a" * 64, gpu.telemetry())

        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "mutated simulation state"
        ):
            guard.invoke_with_acknowledgement(
                fake_mp(gpu), simulation, corrupt_run, REASON
            )
        self.assertEqual(gpu.acks, [])

    def test_tampered_evidence_is_rejected(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        evidence["selected_backend"] = "cpu"
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "selected backend differs"
        ):
            guard.validate_evidence(
                evidence, expected_backend="cuda", expected_reason=REASON
            )

    def test_tampered_retained_digest_is_rejected(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        evidence["warning_telemetry"]["acknowledgement_digest"] = "g" * 64
        evidence["final_telemetry"]["acknowledgement_digest"] = "g" * 64
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "acknowledgement_digest differs"
        ):
            guard.validate_evidence(
                evidence, expected_backend="cuda", expected_reason=REASON
            )

    def test_boolean_attempt_inventory_is_rejected(self):
        gpu = FakeGpu("cpu", "cpu")
        simulation = FakeSimulation(gpu)
        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        evidence["attempts"] = True
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "attempt inventory differs"
        ):
            guard.validate_evidence(
                evidence, expected_backend="cpu", expected_reason=REASON
            )

    def test_minimal_or_binding_tampered_telemetry_is_rejected(self):
        gpu = FakeGpu("cuda", "cpu")
        simulation = FakeSimulation(gpu)
        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu), simulation, simulation.run, REASON
        )
        evidence["final_telemetry"] = {
            "selected_backend": "cuda",
            "recommended_backend": "cpu",
        }
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "telemetry schema differs"
        ):
            guard.validate_evidence(
                evidence, expected_backend="cuda", expected_reason=REASON
            )

        _, evidence = guard.invoke_with_acknowledgement(
            fake_mp(gpu := FakeGpu("cuda", "cpu")),
            simulation := FakeSimulation(gpu),
            simulation.run,
            REASON,
        )
        evidence["final_telemetry"]["workload_fingerprint"] = "9" * 64
        with self.assertRaisesRegex(
            guard.BackendGuardEvidenceError, "acknowledgement digest differs"
        ):
            guard.validate_evidence(
                evidence, expected_backend="cuda", expected_reason=REASON
            )


if __name__ == "__main__":
    unittest.main()
