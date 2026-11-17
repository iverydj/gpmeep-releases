#!/usr/bin/env python3
"""Subprocess tests for the validation import hook's backend guard."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
HOOK = ROOT / "scripts" / "python-validation"
SCRIPTS = ROOT / "scripts"


FAKE_MEEP = r'''
import hashlib
import json
import types


class BackendRecommendationMismatch(RuntimeError):
    def __init__(self, challenge, telemetry):
        super().__init__("backend mismatch")
        self.challenge = challenge
        self.telemetry = telemetry


backend_policy = types.SimpleNamespace(
    BackendRecommendationMismatch=BackendRecommendationMismatch
)


class NativeExtension:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: 0


_meep = NativeExtension()


class Gpu:
    active_backend = "cuda"
    requested_backend = "cuda"
    selected_device = 0
    selected_device_identifier = "fixture"

    def __init__(self):
        self.acknowledged = False
        self.recommendation = None

    def telemetry(self):
        telemetry = {
            "schema": "gpmeep-v110-backend-recommendation-v3",
            "selected_backend": "cuda",
            "recommended_backend": "cpu",
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
            "profiler_or_ablation_id": "v110-m1-m2-aunp-crossover-calibration",
            "acknowledgement": self.acknowledged,
            "acknowledgement_allowed": True,
            "acknowledgement_reason": (
                "validation intentionally compares explicit CPU and CUDA"
                if self.acknowledged else ""
            ),
            "acknowledgement_reused": False,
            "acknowledgement_scope": (
                "simulation" if self.acknowledged else ""
            ),
            "acknowledgement_origin_cohort_digest": "",
            "acknowledgement_scope_cache_enabled": True,
            "acknowledgement_scope_digest": "6" * 64,
            "workload_fingerprint_evaluated": True,
            "workload_fingerprint": "1" * 64,
            "recommendation_policy_version": "gpmeep-v110-explicit-backend-guard-v24",
            "mpi_topology_digest": "2" * 64,
            "native_state_digest": "3" * 64,
            "process_nonce": "4" * 64,
            "run_nonce": "5" * 64,
            "acknowledgement_digest": "",
        }
        bindings = {
            key: telemetry[key]
            for key in (
                "selected_backend", "recommended_backend", "workload_fingerprint",
                "recommendation_policy_version", "mpi_topology_digest",
                "native_state_digest", "process_nonce", "run_nonce",
            )
        }
        telemetry["acknowledgement_digest"] = hashlib.sha256(
            json.dumps(
                {"schema": "gpmeep-v110-backend-ack-v1", **bindings},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if self.acknowledged:
            telemetry["acknowledgement_origin_cohort_digest"] = telemetry[
                "acknowledgement_digest"
            ]
        return telemetry

    def acknowledge_backend_mismatch(self, challenge, reason, scope="simulation"):
        assert challenge == self.telemetry()["acknowledgement_digest"]
        assert reason == "validation intentionally compares explicit CPU and CUDA"
        assert scope == "simulation"
        self.acknowledged = True


gpu = Gpu()


class Simulation:
    def __init__(self, fail_retry=False):
        self.fields = None
        self.calls = 0
        self.fail_retry = fail_retry

    def _execute(self, method_name, original):
        assert getattr(Simulation, method_name) is original
        assert Simulation.run is ORIGINAL_RUN
        assert Simulation.run_k_point is ORIGINAL_RUN_K_POINT
        assert Simulation.solve_cw is ORIGINAL_SOLVE_CW
        assert Simulation.solve_eigfreq is ORIGINAL_SOLVE_EIGFREQ
        self.calls += 1
        if not gpu.acknowledged:
            telemetry = gpu.telemetry()
            raise BackendRecommendationMismatch(
                telemetry["acknowledgement_digest"], telemetry
            )
        if self.fail_retry:
            raise ValueError("retry failure")
        self.fields = types.SimpleNamespace(t=3)
        gpu.recommendation = gpu.telemetry()
        return "PASS"

    def run(self):
        return self._execute("run", ORIGINAL_RUN)

    def run_k_point(self):
        return self._execute("run_k_point", ORIGINAL_RUN_K_POINT)

    def solve_cw(self):
        return self._execute("solve_cw", ORIGINAL_SOLVE_CW)

    def solve_eigfreq(self):
        return self._execute("solve_eigfreq", ORIGINAL_SOLVE_EIGFREQ)


ORIGINAL_RUN = Simulation.run
ORIGINAL_RUN_K_POINT = Simulation.run_k_point
ORIGINAL_SOLVE_CW = Simulation.solve_cw
ORIGINAL_SOLVE_EIGFREQ = Simulation.solve_eigfreq
'''


CHILD = r'''
import json
import sitecustomize
import meep

wrapped = meep.Simulation.run
first = meep.Simulation()
assert first.run() == "PASS"
assert first.calls == 2
assert meep.Simulation.run is wrapped
assert len(sitecustomize._BACKEND_GUARD_EVENTS) == 1

meep.gpu.acknowledged = False
meep.gpu.recommendation = None
second = meep.Simulation(fail_retry=True)
try:
    second.run()
except ValueError as error:
    assert str(error) == "retry failure"
else:
    raise AssertionError("retry failure was not propagated")
assert second.calls == 2
assert meep.Simulation.run is wrapped
assert len(sitecustomize._BACKEND_GUARD_EVENTS) == 1

auxiliary = {}
for method_name in ("run_k_point", "solve_cw", "solve_eigfreq"):
    wrapped_method = getattr(meep.Simulation, method_name)
    meep.gpu.acknowledged = False
    meep.gpu.recommendation = None
    failed = meep.Simulation(fail_retry=True)
    try:
        getattr(failed, method_name)()
    except ValueError as error:
        assert str(error) == "retry failure"
    else:
        raise AssertionError(f"{method_name} retry failure was not propagated")
    assert failed.calls == 2
    assert getattr(meep.Simulation, method_name) is wrapped_method
    assert len(sitecustomize._BACKEND_GUARD_EVENTS) == 1 + len(auxiliary)

    meep.gpu.acknowledged = False
    meep.gpu.recommendation = None
    simulation = meep.Simulation()
    assert getattr(simulation, method_name)() == "PASS"
    assert simulation.calls == 2
    auxiliary[method_name] = simulation.calls
assert len(sitecustomize._BACKEND_GUARD_EVENTS) == 4
print(json.dumps({
    "first_calls": first.calls,
    "second_calls": second.calls,
    "auxiliary": auxiliary,
}))
'''


class SitecustomizeBackendGuardTests(unittest.TestCase):
    def test_import_hook_uses_pristine_dispatch_and_restores_observer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            package = root / "meep"
            package.mkdir()
            (package / "__init__.py").write_text(
                textwrap.dedent(FAKE_MEEP), encoding="utf-8"
            )
            child = root / "child.py"
            child.write_text(textwrap.dedent(CHILD), encoding="utf-8")
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "GPMEEP_VALIDATION_BACKEND_ACKNOWLEDGEMENT_REASON": (
                    "validation intentionally compares explicit CPU and CUDA"
                ),
                "PYTHONPATH": os.pathsep.join(
                    (str(HOOK), str(SCRIPTS), str(root))
                ),
            }
            completed = subprocess.run(
                [sys.executable, str(child)],
                cwd=root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertIn('"first_calls": 2', completed.stdout)
            self.assertIn('"second_calls": 2', completed.stdout)
            self.assertIn('"solve_eigfreq": 2', completed.stdout)


if __name__ == "__main__":
    unittest.main()
