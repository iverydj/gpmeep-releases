"""Self-tests for the perturbation polarization branch wrapper."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


WRAPPER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "perturbation_branch_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_perturbation_branch_matrix_test", WRAPPER_PATH
)
assert SPEC and SPEC.loader
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class PerturbationBranchMatrixTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "__main__"
        perpendicular = "-perpendicular" in sys.argv
        scale = 2.0 if perpendicular else 1.0
        return {
            "perturbation_frequencies": np.asarray([0.2, 0.199]),
            "perturbation_derivatives": np.asarray([-0.1 * scale, -0.11 * scale]),
            "perturbation_integrals": np.asarray([-0.5 * scale, 3.0]),
            "perturbation_relative_error": np.asarray([1 / 11]),
            "perturbation_mode_decays": np.asarray([-1e-4, -1.1e-4]),
            "perturbation_mode_q": np.asarray([1000.0, 904.5454545454545]),
            "perturbation_mode_amplitudes": np.asarray([1 + 2j, 2 - 1j]),
            "perturbation_mode_errors": np.asarray([1e-8, 2e-8]),
            "perturbation_polarization": np.asarray([int(perpendicular)]),
        }

    def test_both_polarizations_are_pinned_and_argv_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])
            previous_argv = sys.argv
            previous_contents = list(previous_argv)

            def identity_run_path(path: str, *, run_name: str):
                self.assertIs(sys.argv, previous_argv)
                return self.fake_run_path(path, run_name=run_name)

            result = WRAPPER.run_branch_matrix(
                args, run_path=identity_run_path
            )
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)
        np.testing.assert_array_equal(result["branch_polarizations"], [0, 1])
        self.assertEqual(result["branch_perturbation_frequencies"].shape, (2, 2))
        self.assertEqual(result["branch_perturbation_derivatives"].shape, (2, 2))
        self.assertEqual(result["branch_perturbation_integrals"].shape, (2, 2))
        self.assertEqual(
            result["branch_perturbation_relative_errors"].shape, (2,)
        )
        self.assertEqual(result["branch_harminv_decays"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_q"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_amplitudes"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_errors"].shape, (2, 2))

    def test_bad_finite_difference_closure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["perturbation_relative_error"] = np.asarray([0.5])
                return namespace

            with self.assertRaisesRegex(RuntimeError, "finite-difference closure"):
                WRAPPER.run_branch_matrix(args, run_path=bad_run_path)

    def test_bad_harminv_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["perturbation_mode_errors"] = np.asarray([1e-8, 2e-3])
                return namespace

            with self.assertRaisesRegex(RuntimeError, "inaccurate Harminv mode"):
                WRAPPER.run_branch_matrix(args, run_path=bad_run_path)

    def test_exception_restores_exact_argv_object_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])
            previous_argv = sys.argv
            previous_contents = list(previous_argv)

            def failing_run_path(_path: str, *, run_name: str):
                self.assertEqual(run_name, "__main__")
                self.assertIs(sys.argv, previous_argv)
                raise RuntimeError("synthetic nested failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic nested failure"):
                WRAPPER.run_branch_matrix(args, run_path=failing_run_path)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)

    def test_each_branch_restores_argv_after_nested_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])
            previous_argv = sys.argv
            previous_contents = list(previous_argv)
            calls = 0

            def rebinding_run_path(path: str, *, run_name: str):
                nonlocal calls
                self.assertIs(sys.argv, previous_argv)
                namespace = self.fake_run_path(path, run_name=run_name)
                calls += 1
                sys.argv = ["nested-rebind"]
                return namespace

            result = WRAPPER.run_branch_matrix(
                args, run_path=rebinding_run_path
            )
        self.assertEqual(calls, 2)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)
        np.testing.assert_array_equal(result["branch_polarizations"], [0, 1])

    def test_nested_rebinding_exception_restores_argv_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])
            previous_argv = sys.argv
            previous_contents = list(previous_argv)

            def failing_run_path(_path: str, *, run_name: str):
                self.assertEqual(run_name, "__main__")
                sys.argv = ["nested-rebind"]
                raise RuntimeError("nested rebind failure")

            with self.assertRaisesRegex(RuntimeError, "nested rebind failure"):
                WRAPPER.run_branch_matrix(args, run_path=failing_run_path)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)


if __name__ == "__main__":
    unittest.main()
