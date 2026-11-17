"""Self-tests for the stochastic-emitter branch-matrix wrapper."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


WRAPPER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "stochastic_branch_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_stochastic_branch_matrix_test", WRAPPER_PATH
)
assert SPEC and SPEC.loader
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class StochasticBranchMatrixTests(unittest.TestCase):
    def make_args(self, example: pathlib.Path):
        return WRAPPER.parse_args(
            [
                str(example),
                "--resolution",
                "50",
                "--trials",
                "2",
                "--dipoles",
                "2",
                "--frequencies",
                "3",
                "--seed",
                "17",
            ]
        )

    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "__main__"
        assert "--no-save" in sys.argv
        method = int(sys.argv[sys.argv.index("-method") + 1])
        textured = "-textured" in sys.argv
        seed = int(sys.argv[sys.argv.index("-seed") + 1])
        scale = float(method * 3 + textured)
        fluxes = np.asarray(
            [[scale, scale + 1], [scale + 2, scale + 3], [scale + 4, scale + 5]]
        )
        return {
            "freqs": np.asarray([0.9, 1.0, 1.1]),
            "fluxes": fluxes,
            "mean_flux": np.mean(fluxes, axis=1),
            "standard_deviation_flux": np.std(fluxes, axis=1),
            "trial_flux_l2": np.linalg.norm(fluxes, axis=0),
            "source_callback_counts": (
                np.asarray([5, 6, 7, 8], dtype=np.int64)
                if method == 1
                else np.asarray([], dtype=np.int64)
            ),
            "validation_seed": seed,
        }

    def test_all_four_branches_are_pinned_and_argv_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "stochastic.py"
            example.touch()
            args = self.make_args(example)
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
        np.testing.assert_array_equal(result["branch_codes"], [10, 11, 20, 21])
        self.assertEqual(result["branch_fluxes"].shape, (4, 3, 2))
        self.assertEqual(result["branch_frequencies"].shape, (4, 3))
        self.assertEqual(result["branch_trial_flux_l2"].shape, (4, 2))
        np.testing.assert_array_equal(
            result["branch_active_callback_counts"], [4, 4, 0, 0]
        )
        np.testing.assert_array_equal(
            result["branch_validation_seeds"], [17, 100017, 200017, 300017]
        )
        self.assertTrue(np.all(result["texture_relative_differences"] > 0))
        self.assertTrue(np.all(result["method_relative_differences"] > 0))

    def test_nonfinite_branch_is_rejected_and_argv_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "stochastic.py"
            example.touch()
            args = self.make_args(example)

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["fluxes"][0, 0] = np.nan
                return namespace

            previous_argv = sys.argv
            previous_contents = list(previous_argv)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                WRAPPER.run_branch_matrix(args, run_path=bad_run_path)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)

    def test_trials_and_dipoles_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "stochastic.py"
            example.touch()
            with self.assertRaisesRegex(RuntimeError, "must match"):
                WRAPPER.parse_args(
                    [str(example), "--trials", "2", "--dipoles", "3"]
                )

    def test_each_branch_restores_argv_after_nested_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "stochastic.py"
            example.touch()
            args = self.make_args(example)
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
        self.assertEqual(calls, 4)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.argv, previous_contents)
        np.testing.assert_array_equal(result["branch_codes"], [10, 11, 20, 21])

    def test_nested_rebinding_exception_restores_argv_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "stochastic.py"
            example.touch()
            args = self.make_args(example)
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
