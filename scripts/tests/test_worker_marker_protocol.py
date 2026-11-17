from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOUNDARY = load(
    "gpmeep_boundary_marker_protocol",
    "compare-boundary-eh-overlap-observables.py",
)
DFT = load("gpmeep_dft_marker_protocol", "benchmark-dft-phase-sharing.py")
MGB = load("gpmeep_multi_gpu_marker_protocol", "multi_gpu_benchmark.py")
DIAGNOSTIC = load(
    "gpmeep_scaling_diagnostic_marker_protocol",
    "run-multi-gpu-scaling-diagnostic.py",
)


class WorkerMarkerProtocolTests(unittest.TestCase):
    def test_all_base_consumers_use_the_same_exact_protocol(self) -> None:
        inventories = (
            tuple(BOUNDARY.WORKER_GPMEEP_PREFIXES),
            tuple(DFT.WORKER_GPMEEP_PREFIXES),
            tuple(MGB.WORKER_STDOUT_GPMEEP_PREFIXES),
        )
        expected = set(inventories[0])
        self.assertTrue(expected)
        for inventory in inventories:
            self.assertEqual(len(inventory), len(set(inventory)))
            self.assertEqual(set(inventory), expected)
        self.assertIn("gpmeep-structured-curl-graph-v1:", expected)
        self.assertNotIn("gpmeep-structured-curl-graph-v2:", expected)

    def test_worker_emission_inventory_matches_base_plus_dft_extension(self) -> None:
        source = (REPO / "tests/gpu-mpi-performance.cpp").read_text(
            encoding="utf-8", errors="strict"
        )
        emitted = set(re.findall(r"gpmeep-[a-z0-9-]+-v[0-9]+:", source))
        expected = set(MGB.WORKER_STDOUT_GPMEEP_PREFIXES)
        expected.add("gpmeep-dft-workload-v1:")
        self.assertEqual(emitted, expected)

    def test_scaling_diagnostic_covers_the_exact_non_process_protocol(self) -> None:
        expected = set(MGB.WORKER_STDOUT_GPMEEP_PREFIXES)
        expected.remove("gpmeep-multi-process-v1:")
        actual = tuple(DIAGNOSTIC.KNOWN_GPMEEP_PREFIXES)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)
        self.assertIn(DIAGNOSTIC.STRUCTURED_CURL_GRAPH_PREFIX, actual)
        self.assertNotIn("gpmeep-structured-curl-graph-v2:", actual)


if __name__ == "__main__":
    unittest.main()
