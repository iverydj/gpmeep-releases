#!/usr/bin/env python3
"""Regression tests for reason-bound retries in installed CUDA examples."""

from __future__ import annotations

import contextlib
import copy
import io
import pathlib
import runpy
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "python/examples/gpmeep"


class Mismatch(RuntimeError):
    def __init__(self, challenge: str, telemetry: dict[str, object]):
        super().__init__("fixture recommendation mismatch")
        self.challenge = challenge
        self.telemetry = telemetry


class Fields:
    def gpu_cuda_execution_selected(self) -> bool:
        return True


class Gpu:
    compiled = True
    requested_backend = ""
    active_backend = ""
    selected_device_identifier = "fixture-gpu-0"

    def __init__(self) -> None:
        self.acknowledged = False
        self.acknowledgements: list[tuple[str, str, str]] = []
        self.simulation = None
        self.recommendation = None

    def set_backend(self, backend: str) -> None:
        if backend != "cuda":
            raise AssertionError("installed GPU examples must explicitly select CUDA")
        self.requested_backend = backend
        self.active_backend = backend

    def reset_statistics(self) -> None:
        return None

    def statistics(self) -> dict[str, object]:
        return {"dispatch": {"cuda_curl_calls": 1, "cpu_curl_calls": 0}}

    def acknowledge_backend_mismatch(
        self, challenge: str, reason: str, scope: str = "simulation"
    ) -> None:
        if challenge != "a" * 64 or "below crossover" not in reason:
            raise AssertionError("installed example acknowledgement is not exact")
        if self.simulation is None or self.simulation.fields is not None:
            raise AssertionError("simulation mutated before acknowledgement")
        self.acknowledgements.append((challenge, reason, scope))
        self.acknowledged = True


class Simulation:
    gpu: Gpu

    def __init__(self, **_kwargs: object) -> None:
        self.fields = None
        self.calls: list[dict[str, object]] = []
        self.gpu.simulation = self

    def run(self, **kwargs: object) -> None:
        self.calls.append(copy.deepcopy(kwargs))
        if not self.gpu.acknowledged:
            raise Mismatch(
                "a" * 64,
                {"selected_backend": "cuda", "recommended_backend": "cpu"},
            )
        self.fields = Fields()
        self.gpu.recommendation = {
            "selected_backend": "cuda",
            "recommended_backend": "cpu",
            "acknowledgement": True,
            "acknowledgement_reason": self.gpu.acknowledgements[0][1],
        }

    def timestep(self) -> int:
        return 20

    def reset_meep(self) -> None:
        return None


class Communicator:
    size = 2
    rank = 0

    def Barrier(self) -> None:
        return None

    def gather(self, local: dict[str, object], root: int = 0):
        if root != 0:
            raise AssertionError("fixture gather root differs")
        remote = copy.deepcopy(local)
        remote["rank"] = 1
        remote["selected_device_identifier"] = "fixture-gpu-1"
        return [local, remote]


def fake_modules(*, ranks: int) -> tuple[types.ModuleType, Gpu]:
    gpu = Gpu()
    meep = types.ModuleType("meep")
    meep.__path__ = []
    meep.gpu = gpu
    Simulation.gpu = gpu
    meep.Simulation = Simulation
    meep.Vector3 = lambda *args, **kwargs: (args, kwargs)
    meep.PML = lambda *args, **kwargs: (args, kwargs)
    meep.Source = lambda *args, **kwargs: (args, kwargs)
    meep.GaussianSource = lambda *args, **kwargs: (args, kwargs)
    meep.Ez = "Ez"
    meep.count_processors = lambda: ranks
    policy = types.ModuleType("meep.backend_policy")
    policy.BackendRecommendationMismatch = Mismatch
    return meep, gpu


class InstalledGpuExamplesTests(unittest.TestCase):
    def run_example(self, name: str, *, ranks: int) -> Gpu:
        meep, gpu = fake_modules(ranks=ranks)
        replacements = {
            "meep": meep,
            "meep.backend_policy": types.SimpleNamespace(
                BackendRecommendationMismatch=Mismatch
            ),
        }
        if ranks == 2:
            replacements["mpi4py"] = types.SimpleNamespace(
                MPI=types.SimpleNamespace(COMM_WORLD=Communicator())
            )
        previous = {key: sys.modules.get(key) for key in replacements}
        sys.modules.update(replacements)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(EXAMPLES / name), run_name="__main__")
        finally:
            for key, value in previous.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        return gpu

    def assert_exact_retry(self, gpu: Gpu) -> None:
        simulation = gpu.simulation
        self.assertIsNotNone(simulation)
        self.assertEqual(simulation.calls, [simulation.calls[0], simulation.calls[0]])
        self.assertEqual(len(gpu.acknowledgements), 1)
        self.assertTrue(gpu.recommendation["acknowledgement"])
        self.assertEqual(
            gpu.recommendation["acknowledgement_reason"],
            gpu.acknowledgements[0][1],
        )

    def test_single_gpu_example_warns_then_retries_once(self) -> None:
        self.assert_exact_retry(self.run_example("installed_gpu1.py", ranks=1))

    def test_two_gpu_example_warns_then_retries_once_per_rank_contract(self) -> None:
        self.assert_exact_retry(self.run_example("installed_gpu2.py", ranks=2))


if __name__ == "__main__":
    unittest.main()
