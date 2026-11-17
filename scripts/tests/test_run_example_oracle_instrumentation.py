#!/usr/bin/env python3
"""Tests for pristine-class dispatch in the generic example oracle."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_example_oracle.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_run_example_oracle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class ExampleOracleInstrumentationTests(unittest.TestCase):
    def test_complete_instrumentation_set_is_pristine_during_dispatch(self) -> None:
        observations: list[tuple[bool, bool]] = []

        class Simulation:
            def run(self, value: int, *, scale: int) -> int:
                observations.append(
                    (
                        Simulation.run is original_run,
                        Simulation.solve_cw is original_solve_cw,
                    )
                )
                return value * scale

            def solve_cw(self) -> str:
                return "original"

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def run_observer(self, *args, **kwargs):
            return None

        def solve_cw_observer(self, *args, **kwargs):
            return None

        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        instrumentation = {
            "run": (original_run, run_observer),
            "solve_cw": (original_solve_cw, solve_cw_observer),
        }
        result = ORACLE.invoke_original_run_with_pristine_class(
            types.SimpleNamespace(Simulation=Simulation),
            "run",
            instrumentation,
            Simulation(),
            (7,),
            {"scale": 6},
        )
        self.assertEqual(result, 42)
        self.assertEqual(observations, [(True, True)])
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)

    def test_complete_instrumentation_set_is_restored_after_failure(self) -> None:
        class Simulation:
            def run(self) -> None:
                self.assertions.append(Simulation.solve_cw is original_solve_cw)
                raise ValueError("expected failure")

            def solve_cw(self) -> None:
                return None

            def __init__(self) -> None:
                self.assertions: list[bool] = []

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def run_observer(self):
            return None

        def solve_cw_observer(self):
            return None

        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        simulation = Simulation()
        with self.assertRaisesRegex(ValueError, "expected failure"):
            ORACLE.invoke_original_run_with_pristine_class(
                types.SimpleNamespace(Simulation=Simulation),
                "run",
                {
                    "run": (original_run, run_observer),
                    "solve_cw": (original_solve_cw, solve_cw_observer),
                },
                simulation,
                (),
                {},
            )
        self.assertEqual(simulation.assertions, [True])
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)

    def test_mutation_during_original_dispatch_fails_closed(self) -> None:
        class Simulation:
            def run(self) -> str:
                Simulation.solve_cw = replacement
                return "unexpected"

            def solve_cw(self) -> None:
                return None

        original_run = Simulation.run
        original_solve_cw = Simulation.solve_cw

        def run_observer(self):
            return None

        def solve_cw_observer(self):
            return None

        def replacement(self):
            return None

        Simulation.run = run_observer
        Simulation.solve_cw = solve_cw_observer
        with self.assertRaisesRegex(
            RuntimeError, "changed while the original implementation was active"
        ):
            ORACLE.invoke_original_run_with_pristine_class(
                types.SimpleNamespace(Simulation=Simulation),
                "run",
                {
                    "run": (original_run, run_observer),
                    "solve_cw": (original_solve_cw, solve_cw_observer),
                },
                Simulation(),
                (),
                {},
            )
        self.assertIs(Simulation.run, run_observer)
        self.assertIs(Simulation.solve_cw, solve_cw_observer)


if __name__ == "__main__":
    unittest.main()
