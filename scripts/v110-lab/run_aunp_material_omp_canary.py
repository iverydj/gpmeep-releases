#!/usr/bin/env python3
"""Run and replay the C4 paired GPU1/GPU2 material-OpenMP canary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

import run_aunp_material_omp_sample as material_sample


_BASE_CONTROLLER_PATH = pathlib.Path(__file__).with_name(
    "run_aunp_zero_state_canary.py"
)
_BASE_MODULE_NAME = "_gpmeep_c4_zero_state_canary_base"
_BASE_SPEC = importlib.util.spec_from_file_location(
    _BASE_MODULE_NAME, _BASE_CONTROLLER_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private C4 controller base module")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)


CANDIDATE_FEATURE = "static-material-openmp-initialization"
CONTROLLER_PATH = pathlib.Path(__file__).resolve()
WORKER_PATH = CONTROLLER_PATH.parent / "run_aunp_material_omp_sample.py"

base.sample = material_sample
base.SCHEMA = "gpmeep-v110-m1-aunp-material-omp-canary-v1"
base.COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-material-omp-canary-complete-v1"
base.RUN_SCHEMA = "gpmeep-v110-m1-aunp-material-omp-canary-run-v1"
base.CHECKPOINT_SCHEMA = "gpmeep-v110-m1-aunp-material-omp-canary-checkpoint-v1"
base.TELEMETRY_SCHEMA = "gpmeep-v110-m1-aunp-material-omp-telemetry-v1"
base.RUNTIME_ATTESTATION_SCHEMA = (
    "gpmeep-v110-m1-material-omp-runtime-attestation-v1"
)
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.CONTROLLER_PATH = CONTROLLER_PATH
base.WORKER_PATH = WORKER_PATH
base.ATTESTATION_VALIDATION_KEY = "c4_validation"
base.ATTESTATION_AUDIT_KEY = "c4_adversarial_audit"
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base.SUPPORT_SOURCE_PATHS = {
    "base_controller": _BASE_CONTROLLER_PATH,
    "base_worker": material_sample._BASE_PATH,
}
base.IMPLEMENTATION_ALLOWED_DESCENDANTS = (
    "scripts/v110-lab/",
    "scripts/tests/test_v110_aunp_material_omp_canary.py",
)

_base_environment = base._environment


def _environment(*args, **kwargs) -> dict[str, str]:
    environment = _base_environment(*args, **kwargs)
    environment.pop(material_sample.C1_ENABLE, None)
    environment[material_sample.C1_DISABLE] = "1"
    environment.pop(material_sample.C3_ENABLE, None)
    environment[material_sample.C3_DISABLE] = "1"
    return environment


base._environment = _environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)

    execute = subparsers.add_parser("run")
    execute.add_argument("--archive", required=True, type=pathlib.Path)
    execute.add_argument("--output", required=True, type=pathlib.Path)
    execute.add_argument("--python", required=True, type=pathlib.Path)
    execute.add_argument("--mpiexec", required=True, type=pathlib.Path)
    execute.add_argument("--pythonpath", required=True, type=pathlib.Path)
    execute.add_argument("--library-path", required=True)
    execute.add_argument("--implementation-commit", required=True)
    execute.add_argument(
        "--runtime-attestation", required=True, type=pathlib.Path
    )
    execute.add_argument("--gpu-devices", default="0,1")
    execute.add_argument("--resume", action="store_true")
    execute.set_defaults(function=base.run)

    replay = subparsers.add_parser("validate")
    replay.add_argument("--output", required=True, type=pathlib.Path)
    replay.set_defaults(function=lambda args: (base.validate(args.output), 0)[1])

    attest = subparsers.add_parser("attest-runtime")
    attest.add_argument("--output", required=True, type=pathlib.Path)
    attest.add_argument("--python", required=True, type=pathlib.Path)
    attest.add_argument("--pythonpath", required=True, type=pathlib.Path)
    attest.add_argument("--library-path", required=True)
    attest.add_argument("--implementation-commit", required=True)
    attest.add_argument(
        "--c4-validation",
        dest="c1_validation",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--c4-adversarial-audit",
        dest="c1_adversarial_audit",
        required=True,
        type=pathlib.Path,
    )
    attest.set_defaults(function=base.attest_runtime)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.CanaryError,
        base.common.WorkloadError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"C4 AuNP canary FAILED: {error}", file=base.sys.stderr, flush=True)
        raise SystemExit(1)
