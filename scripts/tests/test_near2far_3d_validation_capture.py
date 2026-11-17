#!/usr/bin/env python3
"""Argument-contract tests for the Near2Far upstream timing adapter."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "python/examples/adjoint_optimization/near2far_3d_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_near2far_3d_validation", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXAMPLE
AUTOGRAD = types.ModuleType("autograd")
AUTOGRAD.numpy = np
MEEP = types.ModuleType("meep")
MEEP.__path__ = []


def vector3_stub(*coordinates: float) -> tuple[float, ...]:
    return coordinates


def verbosity_stub(_level: int) -> None:
    return None


MEEP.Vector3 = vector3_stub
MEEP.Near2FarRegion = object
MEEP.verbosity = verbosity_stub
MEEP_ADJOINT = types.ModuleType("meep.adjoint")
with mock.patch.dict(
    sys.modules,
    {
        "autograd": AUTOGRAD,
        "meep": MEEP,
        "meep.adjoint": MEEP_ADJOINT,
    },
):
    SPEC.loader.exec_module(EXAMPLE)


class Near2FarUpstreamPerformanceCaptureTests(unittest.TestCase):
    def test_default_execution_remains_strict(self) -> None:
        self.assertFalse(EXAMPLE.upstream_performance_capture_requested([]))

    def test_exact_capture_argument_is_accepted(self) -> None:
        self.assertTrue(
            EXAMPLE.upstream_performance_capture_requested(
                [EXAMPLE.UPSTREAM_PERFORMANCE_CAPTURE_ARGUMENT]
            )
        )

    def test_unknown_or_combined_arguments_fail_closed(self) -> None:
        for arguments in (
            ["--unknown"],
            [EXAMPLE.UPSTREAM_PERFORMANCE_CAPTURE_ARGUMENT, "--unknown"],
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                RuntimeError, "unsupported"
            ):
                EXAMPLE.upstream_performance_capture_requested(arguments)


if __name__ == "__main__":
    unittest.main()
