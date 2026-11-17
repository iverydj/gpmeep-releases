#!/usr/bin/env python3
"""Import and structurally exercise one unittest workload without simulation.

M4 intentionally measures installed runtimes rather than the developer
environment.  This preflight catches missing test-only imports and changed test
identities before the controller creates its evidence directory or launches a
simulation workload.  The optional invocation contract exercises only thin
test wrappers after replacing their numerical helper with a recorder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import unittest
from typing import Any


SCHEMA = "gpmeep-m4-unittest-import-preflight-v1"
OUTPUT_PREFIX = "gpmeep-m4-unittest-preflight:"
MAXIMUM_JSON_ARGUMENT_BYTES = 16 * 1024
GUARDED_SIMULATION_ENTRYPOINTS = (
    "init_sim",
    "run",
    "run_k_point",
    "run_k_points",
    "solve_cw",
)


class PreflightError(RuntimeError):
    pass


def _reject_constant(value: str) -> None:
    raise PreflightError(f"non-finite JSON value {value!r}")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_argument(value: str) -> Any:
    if len(value.encode("utf-8")) > MAXIMUM_JSON_ARGUMENT_BYTES:
        raise argparse.ArgumentTypeError("JSON argument exceeds its size bound")
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, PreflightError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _canonical(
    path: pathlib.Path, *, directory: bool, label: str
) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"{label} is unavailable: {error}") from error
    if (
        lexical.is_symlink()
        or resolved != lexical
        or directory != lexical.is_dir()
        or (not directory and not lexical.is_file())
        or metadata.st_size < 0
    ):
        raise PreflightError(f"{label} path differs")
    return resolved


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten(item))
        elif isinstance(item, unittest.TestCase):
            tests.append(item)
        else:
            raise PreflightError("unittest discovery returned an unknown item")
    return tests


def _json_value(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded, parse_constant=_reject_constant)
    except (TypeError, ValueError, PreflightError) as error:
        raise PreflightError(f"{label} is not finite JSON data") from error


def _vector(vector: Any) -> list[float]:
    try:
        values = [float(vector.x), float(vector.y), float(vector.z)]
    except (AttributeError, TypeError, ValueError) as error:
        raise PreflightError("wrapper invocation far point differs") from error
    if any(not math.isfinite(value) for value in values):
        raise PreflightError("wrapper invocation far point is non-finite")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", required=True, type=pathlib.Path)
    parser.add_argument("--meep-site-packages", required=True, type=pathlib.Path)
    parser.add_argument("--expected-meep-module", required=True, type=pathlib.Path)
    parser.add_argument("--expected-test-id", action="append", default=[])
    parser.add_argument("--case-contract-variable")
    parser.add_argument("--expected-case-contract", type=_json_argument)
    parser.add_argument("--wrapper-helper")
    parser.add_argument(
        "--expected-wrapper-invocation",
        action="append",
        default=[],
        type=_json_argument,
    )
    parser.add_argument("--expected-simulation-run-count", required=True, type=int)
    args = parser.parse_args(argv)
    if not args.expected_test_id or len(set(args.expected_test_id)) != len(
        args.expected_test_id
    ):
        parser.error("expected unittest identities must be nonempty and unique")
    optional = (
        args.case_contract_variable,
        args.expected_case_contract,
        args.wrapper_helper,
        args.expected_wrapper_invocation,
    )
    if any(bool(value) for value in optional) and not all(
        bool(value) for value in optional
    ):
        parser.error("wrapper contract options must be supplied together")
    if args.expected_simulation_run_count <= 0:
        parser.error("expected simulation run count must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = _canonical(args.case, directory=False, label="unittest case")
    site_packages = _canonical(
        args.meep_site_packages,
        directory=True,
        label="Meep site-packages",
    )
    expected_meep = _canonical(
        args.expected_meep_module,
        directory=False,
        label="expected Meep module",
    )
    if case.suffix != ".py" or case.parent == site_packages:
        raise PreflightError("unittest case layout differs")

    sys.path.insert(0, str(site_packages))
    import meep as mp  # pylint: disable=import-outside-toplevel

    actual_meep = _canonical(
        pathlib.Path(mp.__file__), directory=False, label="imported Meep module"
    )
    if actual_meep != expected_meep:
        raise PreflightError("unittest preflight imported a different Meep module")
    if "parameterized" in sys.modules:
        raise PreflightError("unittest preflight inherited parameterized")

    original_entrypoints: dict[str, Any] = {}

    def reject_workload(*_args: Any, **_kwargs: Any) -> None:
        raise PreflightError("unittest preflight attempted a simulation workload")

    for name in GUARDED_SIMULATION_ENTRYPOINTS:
        method = getattr(mp.Simulation, name, None)
        if callable(method):
            original_entrypoints[name] = method
            setattr(mp.Simulation, name, reject_workload)

    module_name = case.stem
    previous = sys.modules.pop(module_name, None)
    try:
        suite = unittest.defaultTestLoader.discover(
            str(case.parent), pattern=case.name, top_level_dir=str(case.parent)
        )
        tests = _flatten(suite)
        identities = [test.id() for test in tests]
        if identities != args.expected_test_id:
            raise PreflightError("unittest identities differ")
        module = sys.modules.get(module_name)
        if module is None or _canonical(
            pathlib.Path(module.__file__), directory=False, label="unittest module"
        ) != case:
            raise PreflightError("unittest module origin differs")
        if "parameterized" in sys.modules:
            raise PreflightError("unittest workload imported parameterized")

        case_contract = None
        invocations: list[dict[str, Any]] = []
        if args.case_contract_variable is not None:
            case_contract = _json_value(
                getattr(module, args.case_contract_variable, None),
                "unittest case contract",
            )
            if case_contract != args.expected_case_contract:
                raise PreflightError("unittest case parameters differ")
            helper_name = args.wrapper_helper
            classes = {test.__class__ for test in tests}
            originals: dict[type[unittest.TestCase], Any] = {}
            try:
                for test_class in classes:
                    helper = getattr(test_class, helper_name, None)
                    if not callable(helper):
                        raise PreflightError("unittest wrapper helper is absent")
                    originals[test_class] = helper

                    def recorder(
                        instance: unittest.TestCase,
                        mode: Any,
                        far_points: Any,
                    ) -> None:
                        if (
                            not isinstance(far_points, list)
                            or len(far_points) != 1
                        ):
                            raise PreflightError(
                                "wrapper invocation far-point list differs"
                            )
                        invocations.append(
                            {
                                "test_id": instance.id(),
                                "mode": mode,
                                "far_point": _vector(far_points[0]),
                            }
                        )

                    setattr(test_class, helper_name, recorder)
                for test in tests:
                    test.debug()
            finally:
                for test_class, helper in originals.items():
                    setattr(test_class, helper_name, helper)
            if invocations != args.expected_wrapper_invocation:
                raise PreflightError("unittest wrapper invocations differ")

        result = {
            "schema": SCHEMA,
            "state": "PASS",
            "case_path": str(case),
            "python_executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
            "meep_module": str(actual_meep),
            "test_count": len(identities),
            "test_identities": identities,
            "case_contract": case_contract,
            "wrapper_invocations": invocations,
            "expected_simulation_run_count": args.expected_simulation_run_count,
            "simulation_workloads_launched": 0,
            "parameterized_imported": False,
            "guarded_simulation_entrypoints": sorted(original_entrypoints),
        }
        print(
            OUTPUT_PREFIX
            + json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 0
    finally:
        for name, method in original_entrypoints.items():
            setattr(mp.Simulation, name, method)
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PreflightError, ValueError, json.JSONDecodeError) as error:
        print(f"M4 unittest preflight error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
