#!/usr/bin/env python3
"""Run and replay the v1.1.0 M2 C1/C3/C4 composition ablation."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from typing import Any

import run_aunp_m2_composition_sample as composition_sample


_M1_PATH = pathlib.Path(__file__).with_name("run_aunp_m1_ablation.py").resolve()
_M1_MODULE_NAME = "_gpmeep_v110_m2_composition_controller_base"
_M1_SPEC = importlib.util.spec_from_file_location(_M1_MODULE_NAME, _M1_PATH)
if _M1_SPEC is None or _M1_SPEC.loader is None:
    raise ImportError("cannot load the private M2 composition controller base")
m1 = importlib.util.module_from_spec(_M1_SPEC)
sys.modules[_M1_MODULE_NAME] = m1
_M1_SPEC.loader.exec_module(m1)


_m1_derive_results = m1.derive_results
MODES = composition_sample.MODES
CANDIDATE_FEATURE = composition_sample.CANDIDATE_FEATURE
CONTROLLER_PATH = pathlib.Path(__file__).resolve()
WORKER_PATH = CONTROLLER_PATH.parent / "run_aunp_m2_composition_sample.py"
VALIDATION_GATE_SCHEMA = "gpmeep-v110-m2-composition-pre-performance-validation-gate-v1"
AUDIT_GATE_SCHEMA = (
    "gpmeep-v110-m2-composition-pre-performance-adversarial-audit-gate-v1"
)
VALIDATION_GATE_KEY = "m2_composition_validation_gate"
AUDIT_GATE_KEY = "m2_composition_adversarial_audit_gate"
BUILD_RECEIPT_KEY = "m2_build_receipt"
BUILD_IDENTITY_KEY = "m2_build_identity"
QUALIFICATION_FONTCONFIG_KEY = "m2_qualification_fontconfig"


base = m1.base
m1.MODES = MODES
m1.ablation_sample = composition_sample
m1.CANDIDATE_FEATURE = CANDIDATE_FEATURE
m1.CONTROLLER_PATH = CONTROLLER_PATH
m1.WORKER_PATH = WORKER_PATH
m1.VALIDATION_GATE_SCHEMA = VALIDATION_GATE_SCHEMA
m1.AUDIT_GATE_SCHEMA = AUDIT_GATE_SCHEMA
m1.VALIDATION_GATE_KEY = VALIDATION_GATE_KEY
m1.AUDIT_GATE_KEY = AUDIT_GATE_KEY
m1.BUILD_RECEIPT_KEY = BUILD_RECEIPT_KEY
m1.BUILD_IDENTITY_KEY = BUILD_IDENTITY_KEY
m1.QUALIFICATION_FONTCONFIG_KEY = QUALIFICATION_FONTCONFIG_KEY

base.sample = composition_sample
base.SCHEMA = "gpmeep-v110-m2-aunp-composition-canary-v1"
base.COMPLETE_SCHEMA = "gpmeep-v110-m2-aunp-composition-canary-complete-v1"
base.RUN_SCHEMA = "gpmeep-v110-m2-aunp-composition-canary-run-v1"
base.CHECKPOINT_SCHEMA = "gpmeep-v110-m2-aunp-composition-checkpoint-v1"
base.TELEMETRY_SCHEMA = "gpmeep-v110-m2-aunp-composition-telemetry-v1"
base.RUNTIME_ATTESTATION_SCHEMA = "gpmeep-v110-m2-composition-runtime-attestation-v1"
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.CONTROLLER_PATH = CONTROLLER_PATH
base.WORKER_PATH = WORKER_PATH
base.ATTESTATION_VALIDATION_KEY = "m2_composition_validation"
base.ATTESTATION_AUDIT_KEY = "m2_composition_adversarial_audit"
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base.SUPPORT_SOURCE_PATHS = {
    "m1_controller": pathlib.Path(m1.__file__).resolve(),
    "m1_worker": composition_sample._BASE_PATH,
    "base_controller": m1._BASE_CONTROLLER_PATH,
    "base_worker": pathlib.Path(composition_sample.m1._BASE_PATH).resolve(),
    "c3_worker": pathlib.Path(composition_sample.m1.c3.__file__).resolve(),
    "c4_worker": pathlib.Path(composition_sample.m1.c4.__file__).resolve(),
}
base.IMPLEMENTATION_ALLOWED_DESCENDANTS = (
    "scripts/v110-lab/",
    "scripts/tests/test_v110_aunp_m1_ablation.py",
    "scripts/tests/test_v110_aunp_m2_composition.py",
)


def _omp_threads(task: dict[str, Any]) -> int:
    mode = task.get("feature_mode")
    if not isinstance(mode, str):
        raise base.CanaryError("M2 task mode is invalid")
    try:
        return composition_sample._mode_configuration(mode)[3]
    except base.common.WorkloadError as exc:
        raise base.CanaryError(str(exc)) from exc


def _pair_comparison(
    means: dict[str, Any], baseline_mode: str, candidate_mode: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for topology in ("gpu1", "gpu2"):
        result[topology] = {}
        for metric in m1.METRICS:
            baseline = float(means[topology][baseline_mode][metric])
            candidate = float(means[topology][candidate_mode][metric])
            result[topology][metric] = {
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                "baseline_mean_seconds": baseline,
                "candidate_mean_seconds": candidate,
                "candidate_to_baseline_ratio": candidate / baseline,
                "improvement_fraction": 1.0 - candidate / baseline,
            }
    return result


def derive_results(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    result = _m1_derive_results(tasks)
    means = result["means_seconds"]
    composition = {
        "t2_combined_vs_c4": _pair_comparison(
            means, "c4_t2", "c1+c3+c4_t2"
        ),
        "t4_combined_vs_c4": _pair_comparison(
            means, "c4_t4", "c1+c3+c4_t4"
        ),
        "c4_t2_to_t4": _pair_comparison(means, "c4_t2", "c4_t4"),
        "combined_t2_to_t4": _pair_comparison(
            means, "c1+c3+c4_t2", "c1+c3+c4_t4"
        ),
    }
    composition_no_regression = all(
        comparison[topology][metric][
            "candidate_to_baseline_ratio"
        ]
        <= 1.0 + base.MAXIMUM_BLOCKER_REGRESSION
        for name, comparison in composition.items()
        if name in {"t2_combined_vs_c4", "t4_combined_vs_c4"}
        for topology in ("gpu1", "gpu2")
        for metric in ("fdtd_wall_seconds", "end_to_end_seconds")
    )
    result["composition_analysis"] = composition
    result["gates"][
        "combined_lane_no_fdtd_or_end_to_end_regression_over_3_percent_vs_c4"
    ] = composition_no_regression
    result["promotion_outcome"] = (
        "M2_COMPOSITION_ACCEPTED"
        if all(result["gates"].values())
        else "M2_COMPOSITION_BLOCKED"
    )
    return result


m1._omp_threads = _omp_threads
m1.derive_results = derive_results
base.task_plan = m1.task_plan
base._omp_threads = _omp_threads
base._environment = m1._environment
base.derive_results = derive_results
base.ATTESTATION_EXTENSION_BUILDER = m1._attestation_extension_builder
base.ATTESTATION_EXTENSION_VALIDATOR = m1._attestation_extension_validator


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
    attest.add_argument("--build-receipt", required=True, type=pathlib.Path)
    attest.add_argument(
        "--m2-validation",
        dest="c1_validation",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--m2-adversarial-audit",
        dest="c1_adversarial_audit",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--m2-validation-gate",
        dest="m1_validation_gate",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--m2-adversarial-audit-gate",
        dest="m1_adversarial_audit_gate",
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
    ) as error:
        print(f"M2 composition FAILED: {error}", file=base.sys.stderr)
        raise SystemExit(2)
