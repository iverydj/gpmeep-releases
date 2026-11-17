#!/usr/bin/env python3
"""Validate and, at release, close the gpmeep v1.1.0 inversion registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "packaging/v110-development-contract.json"
PHASES = {
    "initialization",
    "fdtd_timestep",
    "source",
    "boundary",
    "dispersive_ade",
    "dft_reduction",
    "host_device_transfer_and_sync",
    "mpi",
    "python_adjoint_mpb",
    "io",
}
LANES = {
    "upstream_meep_cpu",
    "gpmeep_v110_cpu",
    "gpmeep_v103_cuda_gpu1",
    "gpmeep_v110_cuda_gpu1",
    "gpmeep_v103_cuda_gpu2",
    "gpmeep_v110_cuda_gpu2",
}
GPU2_LANES = {"gpmeep_v103_cuda_gpu2", "gpmeep_v110_cuda_gpu2"}
GPMEEP_CPU_REFERENCE_CASES = {
    "python/examples/waveguide_crossing.py",
    "python/examples/adjoint_optimization/near2far_3d_validation.py",
    "python/tests/test_adjoint_cyl.py",
}
ABSORBER_CASE = "python/examples/absorber-1d.py"
PHASE_IN_MATERIAL_CASE = "python/examples/phase_in_material.py"
PHASE_IN_MATERIAL_FIXED_WORK = {
    "environment": "GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK",
    "value": "1",
    "phase_time": 200.0,
    "run_count": 2,
    "final_timestep": 4000,
    "total_timestep_delta": 8000,
    "purpose": "stable timing and one-second external GPU process observability",
}
WAVEGUIDE_CROSSING_CASE = "python/examples/waveguide_crossing.py"
NEAR2FAR_3D_CASE = "python/examples/adjoint_optimization/near2far_3d_validation.py"
ADJOINT_CYL_CASE = "python/tests/test_adjoint_cyl.py"
WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT = 893
WAVEGUIDE_OPTIMIZER_DESIGN_SIZE = 441
WAVEGUIDE_FIXED_SLICE_RUN_COUNT = 12
WAVEGUIDE_OPTIMIZER_RUN_COUNT = 8
WAVEGUIDE_FIXED_SLICE_TIMESTEP = 20_001
WAVEGUIDE_OPTIMIZER_TIMESTEP = 10_001
WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES = {
    "optimizer_smoke.epoch0_design[[]*]",
    "optimizer_smoke.final_design[[]*]",
    "optimizer_smoke.*_design_step_l2",
    "optimizer_smoke.final_design_l2",
    "optimizer_smoke.*_gradient_l2_*",
    "optimizer_smoke.*_objective_*",
}
WAVEGUIDE_EXPECTED_METRIC_COUNT = 950
WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256 = (
    "a156e185a0f5f16bc5f8c9713bae9d6cd700732dc34e8560ac6f4ce6e75d6e6e"
)
ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR = 1.0e-4
ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR = 1.0e-5
ADJOINT_CYL_EXPECTED_TIMESTEPS = tuple(
    value for _case in range(4) for value in (48001, 48001, 51100, 51100)
)
ADJOINT_CYL_EXPECTED_MEEP_TIMES = tuple(
    value for _case in range(4) for value in (1200.025, 1200.025, 1277.5, 1277.5)
)
EMPTY_JSON_FAILURE_RECORDS_SHA256 = hashlib.sha256(
    b"gpmeep-json-failure-records-v1\0"
).hexdigest()
RATIO_KEYS = {
    "upstream_cpu_over_v103_cuda",
    "upstream_cpu_over_v110_cuda",
    "v103_cuda_over_v110_cuda",
    "v110_cpu_over_v110_cuda",
}
LANE_CONTRACTS = {
    "upstream_meep_cpu": ("upstream_meep", "cpu", {"cpu1", "cpu8"}, 0, "upstream_sample"),
    "gpmeep_v110_cpu": ("gpmeep_v110", "cpu", {"cpu1", "cpu8"}, 0, "v110_sample"),
    "gpmeep_v103_cuda_gpu1": ("gpmeep_v103", "cuda", {"gpu1"}, 1, "v103_sample"),
    "gpmeep_v110_cuda_gpu1": ("gpmeep_v110", "cuda", {"gpu1"}, 1, "v110_sample"),
    "gpmeep_v103_cuda_gpu2": ("gpmeep_v103", "cuda", {"gpu2"}, 2, "v103_sample"),
    "gpmeep_v110_cuda_gpu2": ("gpmeep_v110", "cuda", {"gpu2"}, 2, "v110_sample"),
}
SAMPLE_SCHEMAS = {
    "upstream_sample": "gpmeep-v110-upstream-performance-sample-v1",
    "v103_sample": "gpmeep-v110-frozen-v103-performance-sample-v1",
    "v110_sample": "gpmeep-m3-performance-sample-v1",
    "experiment_sample": "gpmeep-v110-inversion-experiment-sample-v9",
}
CLASSES = {"hardware_crossover", "engine_overhead", "environment", "unknown"}
EVIDENCE_KINDS = {
    "timing",
    "phase_profile",
    "size_run_sweep",
    "profiler_or_ablation",
    "same_binary_ablation",
    "numerical_regression",
    "environment_diagnostic",
}
EVIDENCE_SCHEMAS = {
    "timing": "gpmeep-v110-inversion-timing-evidence-v2",
    "phase_profile": "gpmeep-v110-inversion-phase-profile-evidence-v2",
    "size_run_sweep": "gpmeep-v110-inversion-size-run-sweep-evidence-v2",
    "profiler_or_ablation": "gpmeep-v110-inversion-profiler-ablation-evidence-v2",
    "same_binary_ablation": "gpmeep-v110-inversion-same-binary-ablation-evidence-v2",
    "numerical_regression": "gpmeep-v110-m4-numerical-replay-v1",
    "environment_diagnostic": "gpmeep-v110-inversion-environment-diagnostic-evidence-v2",
}


class RegistryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise RegistryError(f"non-finite JSON constant: {value}")


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == keys, f"{name} key inventory differs")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
    )


def _expected_m4_numerical_method(case_path: str, candidate_lane: str) -> str:
    if case_path == ABSORBER_CASE:
        return "manifest-plus-scale-aware-trace"
    if case_path == WAVEGUIDE_CROSSING_CASE:
        return (
            "upstream-fixed-slices-plus-diagnostic-optimizer-trajectory-"
            "and-self-oracles"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-optimizer-self-oracles"
        )
    if case_path == NEAR2FAR_3D_CASE:
        return (
            "upstream-forward-fd-plus-candidate-self-oracle"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-candidate-self-oracle"
        )
    if case_path == ADJOINT_CYL_CASE:
        return (
            "upstream-forward-fd-plus-fixed-work-self-oracle"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-fixed-work-self-oracle"
        )
    return "manifest-json-metrics"


def _verify_m4_compact_comparison(value: Any, name: str) -> None:
    result = _exact(
        value,
        {
            "outcome", "reason", "metric_evidence", "difference_summary",
            "failure_count", "failures_reported", "failures_truncated",
            "failure_records_sha256", "failures",
        },
        name,
    )
    _require(
        result["outcome"] == "PASS"
        and isinstance(result["reason"], str)
        and bool(result["reason"])
        and result["failure_count"] == 0
        and type(result["failure_count"]) is int
        and result["failures_reported"] == 0
        and type(result["failures_reported"]) is int
        and result["failures_truncated"] is False
        and result["failure_records_sha256"] == EMPTY_JSON_FAILURE_RECORDS_SHA256
        and result["failures"] == []
        and isinstance(result["difference_summary"], dict),
        f"{name} verdict differs",
    )
    evidence = result["metric_evidence"]
    base_keys = {
        "schema", "count", "passed_count", "failure_count",
        "metric_keys_sha256", "cpu_sha256", "cuda_sha256",
        "comparison_records_sha256", "inline_limit", "inline_byte_limit",
        "flattened_storage_bytes", "inlined",
    }
    optional_keys = (
        frozenset(set(evidence) - base_keys)
        if isinstance(evidence, dict)
        else frozenset()
    )
    _require(
        optional_keys
        in {
            frozenset(),
            frozenset({"diagnostic_only_count", "diagnostic_only_out_of_tolerance_count"}),
            frozenset({"digest_order", "comparison_strategy"}),
            frozenset({
                "diagnostic_only_count", "diagnostic_only_out_of_tolerance_count",
                "digest_order", "comparison_strategy",
            }),
        },
        f"{name} metric evidence key inventory differs",
    )
    evidence = _exact(evidence, base_keys | optional_keys, f"{name} metric evidence")
    _require(
        evidence["schema"]
        in {"gpmeep-json-metric-evidence-v2", "gpmeep-json-metric-evidence-v3-direct"}
        and type(evidence["count"]) is int
        and evidence["count"] > 0
        and evidence["passed_count"] == evidence["count"]
        and type(evidence["passed_count"]) is int
        and evidence["failure_count"] == 0
        and type(evidence["failure_count"]) is int
        and all(
            _hex64(evidence[key])
            for key in (
                "metric_keys_sha256", "cpu_sha256", "cuda_sha256",
                "comparison_records_sha256",
            )
        )
        and type(evidence["inline_limit"]) is int
        and evidence["inline_limit"] > 0
        and type(evidence["inline_byte_limit"]) is int
        and evidence["inline_byte_limit"] > 0
        and type(evidence["flattened_storage_bytes"]) is int
        and evidence["flattened_storage_bytes"] >= 0
        and type(evidence["inlined"]) is bool,
        f"{name} metric evidence differs",
    )
    direct = evidence["schema"] == "gpmeep-json-metric-evidence-v3-direct"
    _require(
        ({"digest_order", "comparison_strategy"} <= optional_keys) == direct,
        f"{name} direct metric evidence fields differ",
    )
    if direct:
        _require(
            evidence["digest_order"] == "document-order"
            and evidence["comparison_strategy"] == "aligned-direct",
            f"{name} direct metric evidence strategy differs",
        )
    if "diagnostic_only_count" in optional_keys:
        _require(
            type(evidence["diagnostic_only_count"]) is int
            and 0 < evidence["diagnostic_only_count"] <= evidence["count"]
            and type(evidence["diagnostic_only_out_of_tolerance_count"]) is int
            and 0 <= evidence["diagnostic_only_out_of_tolerance_count"]
            <= evidence["diagnostic_only_count"],
            f"{name} diagnostic metric census differs",
        )


def _verify_m4_waveguide_diagnostic_summary(
    pairwise: Any, diagnostic_optimizer: bool, name: str
) -> None:
    _require(isinstance(pairwise, dict), f"{name} must be an object")
    evidence = pairwise.get("metric_evidence")
    summary = pairwise.get("difference_summary")
    _require(isinstance(evidence, dict), f"{name} evidence must be an object")
    _require(isinstance(summary, dict), f"{name} summary must be an object")
    diagnostic_fields = {
        "diagnostic_only", "diagnostic_only_out_of_tolerance_count"
    }
    marked: set[str] = set()
    metric_count = 0
    out_of_tolerance_count = 0
    for rule, value in summary.items():
        _require(isinstance(rule, str), f"{name} diagnostic rule differs")
        _require(isinstance(value, dict), f"{name} rule {rule} must be an object")
        leaf = value
        present = diagnostic_fields & set(leaf)
        if not present:
            continue
        _require(
            present == diagnostic_fields
            and leaf.get("diagnostic_only") is True
            and type(leaf.get("diagnostic_only_out_of_tolerance_count")) is int
            and leaf["diagnostic_only_out_of_tolerance_count"] >= 0
            and type(leaf.get("metric_count")) is int
            and leaf["metric_count"] > 0
            and leaf["diagnostic_only_out_of_tolerance_count"]
            <= leaf["metric_count"],
            f"{name} diagnostic leaf differs",
        )
        marked.add(rule)
        metric_count += leaf["metric_count"]
        out_of_tolerance_count += leaf[
            "diagnostic_only_out_of_tolerance_count"
        ]
    if diagnostic_optimizer:
        _require(
            marked == WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES
            and metric_count == WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
            and evidence.get("diagnostic_only_count") == metric_count
            and evidence.get("diagnostic_only_out_of_tolerance_count")
            == out_of_tolerance_count,
            f"{name} diagnostic rule scope differs",
        )
    else:
        _require(not marked, f"{name} CUDA diagnostic rules were hidden")


def _verify_m4_absorber_scale(value: Any, name: str) -> None:
    result = _exact(
        value,
        {
            "outcome", "maximum_peak_normalized_error",
            "maximum_rms_peak_normalized_error", "observed_peak",
            "observed_maximum_absolute_error",
            "observed_peak_normalized_maximum_error",
            "observed_rms_peak_normalized_error", "components",
        },
        name,
    )
    _require(result["outcome"] == "PASS", f"{name} verdict differs")
    numeric_keys = set(result) - {"outcome", "components"}
    _require(
        all(_finite_number(result[key]) for key in numeric_keys)
        and result["maximum_peak_normalized_error"]
        == ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR
        and result["maximum_rms_peak_normalized_error"]
        == ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR
        and result["observed_peak"] >= 0.0
        and result["observed_maximum_absolute_error"] >= 0.0
        and 0.0 <= result["observed_peak_normalized_maximum_error"]
        <= ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR
        and 0.0 <= result["observed_rms_peak_normalized_error"]
        <= ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR,
        f"{name} bounds differ",
    )
    expected_normalized = result["observed_maximum_absolute_error"] / max(
        result["observed_peak"], 1.0e-30
    )
    _require(
        math.isclose(
            result["observed_peak_normalized_maximum_error"],
            expected_normalized,
            rel_tol=1.0e-14,
            abs_tol=1.0e-15,
        ),
        f"{name} normalized maximum is inconsistent",
    )
    components = _exact(result["components"], {"real", "imag"}, f"{name} components")
    maxima = []
    for component in ("real", "imag"):
        item = _exact(
            components[component],
            {"count", "maximum_absolute_error"},
            f"{name} {component} component",
        )
        _require(
            type(item["count"]) is int
            and item["count"] > 0
            and _finite_number(item["maximum_absolute_error"])
            and item["maximum_absolute_error"] >= 0.0,
            f"{name} {component} component differs",
        )
        maxima.append(item["maximum_absolute_error"])
    _require(
        result["observed_maximum_absolute_error"] == max(maxima),
        f"{name} component maximum is inconsistent",
    )


def _verify_m4_cyl_self_oracle(value: Any, name: str) -> None:
    result = _exact(value, {"outcome", "bounds"}, name)
    bounds = result["bounds"]
    _require(
        result["outcome"] == "PASS" and isinstance(bounds, list) and len(bounds) == 4,
        f"{name} verdict/census differs",
    )
    for index, item in enumerate(bounds):
        item = _exact(
            item,
            {
                "case_index", "objective_relative_error", "objective_limit",
                "relative_derivative_error", "reported_relative_derivative_error",
                "derivative_limit",
            },
            f"{name} bound[{index}]",
        )
        expected_derivative_limit = 0.6 if index < 2 else 0.4
        _require(
            item["case_index"] == index
            and type(item["case_index"]) is int
            and item["objective_limit"] == 1.0e-3
            and item["derivative_limit"] == expected_derivative_limit
            and _finite_number(item["objective_relative_error"])
            and 0.0 <= item["objective_relative_error"] <= item["objective_limit"]
            and _finite_number(item["relative_derivative_error"])
            and 0.0 <= item["relative_derivative_error"] <= item["derivative_limit"]
            and _finite_number(item["reported_relative_derivative_error"])
            and math.isclose(
                item["relative_derivative_error"],
                item["reported_relative_derivative_error"],
                rel_tol=1.0e-14,
                abs_tol=1.0e-15,
            ),
            f"{name} bound[{index}] differs",
        )


def _verify_m4_waveguide_optimizer_self_oracle(value: Any, name: str) -> None:
    result = _exact(
        value,
        {
            "outcome", "design_size", "design_minimum", "design_maximum",
            "computed_epoch0_design_step_l2",
            "reported_epoch0_design_step_l2",
            "computed_epoch1_design_step_l2",
            "reported_epoch1_design_step_l2",
            "computed_final_design_l2", "reported_final_design_l2",
            "structural_metrics",
        },
        name,
    )
    expected_structure = {
        **{
            f"fixed_work.run_timesteps[{index}]": float(
                WAVEGUIDE_FIXED_SLICE_TIMESTEP
            )
            for index in range(WAVEGUIDE_FIXED_SLICE_RUN_COUNT)
        },
        **{
            f"optimizer_smoke.run_timesteps[{index}]": float(
                WAVEGUIDE_OPTIMIZER_TIMESTEP
            )
            for index in range(WAVEGUIDE_OPTIMIZER_RUN_COUNT)
        },
        "optimizer_smoke.beta_initial": 4.0,
        "optimizer_smoke.beta_final": 8.0,
        "optimizer_smoke.epoch_count": 2.0,
        "optimizer_smoke.epoch0_beta": 4.0,
        "optimizer_smoke.epoch1_beta": 8.0,
        "optimizer_smoke.epoch0_callback_count": 2.0,
        "optimizer_smoke.epoch1_callback_count": 2.0,
        "optimizer_smoke.epoch0_nlopt_evaluations": 2.0,
        "optimizer_smoke.epoch1_nlopt_evaluations": 2.0,
        "optimizer_smoke.epoch0_result_code": 5.0,
        "optimizer_smoke.epoch1_result_code": 5.0,
        "optimizer_smoke.checkpoint_callback_count": 4.0,
        "optimizer_smoke.checkpoint_completed_epochs": 2.0,
        "optimizer_smoke.checkpoint_roundtrip_max_abs_difference": 0.0,
        "optimizer_smoke.result_roundtrip_max_abs_difference": 0.0,
    }
    progress_pairs = (
        (
            "computed_epoch0_design_step_l2",
            "reported_epoch0_design_step_l2",
        ),
        (
            "computed_epoch1_design_step_l2",
            "reported_epoch1_design_step_l2",
        ),
    )
    _require(
        result["outcome"] == "PASS"
        and result["design_size"] == WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        and type(result["design_size"]) is int
        and _finite_number(result["design_minimum"])
        and _finite_number(result["design_maximum"])
        and 0.0 <= result["design_minimum"] <= result["design_maximum"] <= 1.0
        and all(
            _finite_number(result[computed])
            and result[computed] > 1.0e-10
            and _finite_number(result[reported])
            and math.isclose(
                result[computed],
                result[reported],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for computed, reported in progress_pairs
        )
        and _finite_number(result["computed_final_design_l2"])
        and result["computed_final_design_l2"] > 0.0
        and _finite_number(result["reported_final_design_l2"])
        and math.isclose(
            result["computed_final_design_l2"],
            result["reported_final_design_l2"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and result["structural_metrics"] == expected_structure,
        f"{name} waveguide optimizer evidence differs",
    )


def _verify_m4_cyl_fixed_work(value: Any, name: str) -> None:
    result = _exact(
        value,
        {"outcome", "run_count", "timesteps", "timestep_deltas", "meep_times"},
        name,
    )
    _require(
        result["outcome"] == "PASS"
        and result["run_count"] == 16
        and type(result["run_count"]) is int
        and isinstance(result["timesteps"], list)
        and tuple(result["timesteps"]) == ADJOINT_CYL_EXPECTED_TIMESTEPS
        and isinstance(result["timestep_deltas"], list)
        and tuple(result["timestep_deltas"]) == ADJOINT_CYL_EXPECTED_TIMESTEPS
        and isinstance(result["meep_times"], list)
        and len(result["meep_times"]) == 16
        and all(_finite_number(value) for value in result["meep_times"])
        and all(
            abs(observed - expected) <= 0.05
            for observed, expected in zip(
                result["meep_times"], ADJOINT_CYL_EXPECTED_MEEP_TIMES, strict=True
            )
        ),
        f"{name} fixed-work evidence differs",
    )


def _verify_m4_numerical_result(
    case_path: str, candidate_lane: str, method: Any, value: Any, name: str
) -> None:
    expected_method = _expected_m4_numerical_method(case_path, candidate_lane)
    _require(method == expected_method, f"{name} method differs")
    if case_path == ABSORBER_CASE:
        result = _exact(
            value,
            {"outcome", "failure_count", "pairwise", "scale_aware_trace"},
            name,
        )
        _require(
            result["outcome"] == "PASS"
            and result["failure_count"] == 0
            and type(result["failure_count"]) is int,
            f"{name} verdict differs",
        )
        _verify_m4_compact_comparison(result["pairwise"], f"{name} pairwise")
        _verify_m4_absorber_scale(result["scale_aware_trace"], f"{name} scale gate")
        return
    if case_path == WAVEGUIDE_CROSSING_CASE:
        result = _exact(
            value,
            {
                "outcome", "failure_count", "pairwise",
                "reference_self_oracle", "candidate_self_oracle",
            },
            name,
        )
        _require(
            result["outcome"] == "PASS"
            and result["failure_count"] == 0
            and type(result["failure_count"]) is int,
            f"{name} verdict differs",
        )
        _verify_m4_compact_comparison(result["pairwise"], f"{name} pairwise")
        evidence = result["pairwise"]["metric_evidence"]
        _require(
            evidence["count"] == WAVEGUIDE_EXPECTED_METRIC_COUNT
            and evidence["passed_count"] == WAVEGUIDE_EXPECTED_METRIC_COUNT
            and evidence["metric_keys_sha256"]
            == WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256,
            f"{name} waveguide metric inventory differs",
        )
        if candidate_lane == "gpmeep_v110_cpu":
            _require(
                evidence.get("diagnostic_only_count")
                == WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
                and type(evidence.get("diagnostic_only_out_of_tolerance_count"))
                is int,
                f"{name} upstream diagnostic scope differs",
            )
        else:
            _require(
                "diagnostic_only_count" not in evidence
                and "diagnostic_only_out_of_tolerance_count" not in evidence,
                f"{name} CUDA comparison was weakened",
            )
        _verify_m4_waveguide_diagnostic_summary(
            result["pairwise"],
            candidate_lane == "gpmeep_v110_cpu",
            f"{name} pairwise",
        )
        for field in ("reference_self_oracle", "candidate_self_oracle"):
            _verify_m4_waveguide_optimizer_self_oracle(
                result[field], f"{name} {field}"
            )
        return
    if case_path == ADJOINT_CYL_CASE:
        result = _exact(
            value,
            {
                "outcome", "failure_count", "pairwise", "reference_self_oracle",
                "candidate_self_oracle", "reference_fixed_work",
                "candidate_fixed_work",
            },
            name,
        )
        _require(
            result["outcome"] == "PASS"
            and result["failure_count"] == 0
            and type(result["failure_count"]) is int,
            f"{name} verdict differs",
        )
        _verify_m4_compact_comparison(result["pairwise"], f"{name} pairwise")
        for field in ("reference_self_oracle", "candidate_self_oracle"):
            _verify_m4_cyl_self_oracle(result[field], f"{name} {field}")
        for field in ("reference_fixed_work", "candidate_fixed_work"):
            _verify_m4_cyl_fixed_work(result[field], f"{name} {field}")
        return
    _verify_m4_compact_comparison(value, name)


def _validate_json_schema(record: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as error:
        raise RegistryError(
            "record validation requires the pinned jsonschema dependency"
        ) from error
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    except Exception as error:
        raise RegistryError(f"normative inversion schema is invalid: {error}") from error
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "record"
        raise RegistryError(f"record violates normative JSON Schema at {location}: {first.message}")


def _evidence(
    value: Any,
    name: str,
    evidence_root: pathlib.Path | None,
    required_kind: str | None = None,
    expected_workload_id: str | None = None,
    expected_campaign_id: str | None = None,
    expected_inventory_sha256: str | None = None,
    expected_classification: str | None = None,
) -> str:
    item = _exact(value, {"kind", "path", "sha256"}, name)
    _require(item["kind"] in EVIDENCE_KINDS, f"{name}.kind is invalid")
    if required_kind is not None:
        _require(item["kind"] == required_kind, f"{name}.kind must be {required_kind}")
    _require(_hex64(item["sha256"]), f"{name}.sha256 is invalid")
    relative = pathlib.PurePosixPath(item["path"])
    _require(
        item["path"] not in {"", "."}
        and not relative.is_absolute()
        and ".." not in relative.parts,
        f"{name}.path is unsafe",
    )
    if evidence_root is not None:
        lexical = evidence_root / pathlib.Path(*relative.parts)
        _require(not lexical.is_symlink(), f"{name}.path cannot be a symlink")
        path = lexical.resolve(strict=True)
        _require(path.is_relative_to(evidence_root), f"{name}.path escapes evidence root")
        _require(path == lexical, f"{name}.path cannot contain symlinks")
        _require(path.is_file(), f"{name}.path is not a file")
        _require(_sha256(path) == item["sha256"], f"{name} evidence hash differs")
        _verify_typed_evidence(
            path,
            evidence_root,
            expected_kind=item["kind"],
            expected_workload_id=expected_workload_id,
            expected_campaign_id=expected_campaign_id,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_classification=expected_classification,
        )
    return item["kind"]


def _verify_typed_evidence(
    path: pathlib.Path,
    evidence_root: pathlib.Path,
    *,
    expected_kind: str,
    expected_workload_id: str | None,
    expected_campaign_id: str | None,
    expected_inventory_sha256: str | None,
    expected_classification: str | None,
    derive_result: bool = False,
    payload_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load(path) if payload_override is None else payload_override
    _require(isinstance(payload, dict), "typed inversion evidence payload differs")
    _require(
        payload.get("schema") == EVIDENCE_SCHEMAS[expected_kind],
        "typed inversion evidence producer schema differs",
    )
    if expected_kind == "numerical_regression":
        payload = _exact(
            payload,
            {"schema", "outcome", "comparison_count", "comparisons"},
            "M4 numerical regression evidence",
        )
        comparisons = payload["comparisons"]
        _require(
            payload["outcome"] == "PASS"
            and type(payload["comparison_count"]) is int
            and payload["comparison_count"] == 141
            and isinstance(comparisons, list)
            and len(comparisons) == 141,
            "M4 numerical regression census differs",
        )
        plan = _load(ROOT / "packaging/v110-m4-equal-work-plan.json")
        cases = plan.get("cases")
        _require(
            plan.get("schema") == "gpmeep-v110-m4-equal-work-plan-v1"
            and isinstance(cases, list)
            and len(cases) == 13,
            "sealed M4 numerical plan differs",
        )
        expected_rows: set[tuple[int, str, str, str]] = set()
        for case in cases:
            expected_case_keys = {"path", "cpu_ranks", "gpu2", "timing"}
            if case.get("path") == PHASE_IN_MATERIAL_CASE:
                expected_case_keys.add("fixed_work")
            _exact(case, expected_case_keys, "M4 numerical plan case")
            if case["path"] == PHASE_IN_MATERIAL_CASE:
                _require(
                    case["fixed_work"] == PHASE_IN_MATERIAL_FIXED_WORK,
                    "M4 phase_in_material fixed-work contract differs",
                )
            candidate_lanes = {
                "gpmeep_v110_cpu",
                "gpmeep_v103_cuda_gpu1",
                "gpmeep_v110_cuda_gpu1",
            }
            if case["gpu2"] is True:
                candidate_lanes.update(
                    {"gpmeep_v103_cuda_gpu2", "gpmeep_v110_cuda_gpu2"}
                )
            else:
                _require(case["gpu2"] is False, "M4 numerical plan GPU2 policy differs")
            expected_rows.update(
                (cycle, case["path"], lane, "warmup" if cycle == 0 else "measured")
                for cycle in (0, 1, 2)
                for lane in candidate_lanes
            )
        _require(len(expected_rows) == 141, "sealed M4 numerical row census differs")
        observed_rows: set[tuple[int, str, str, str]] = set()
        for index, row in enumerate(comparisons):
            row = _exact(
                row,
                {
                    "cycle_index", "sample_kind", "case_path", "reference_lane",
                    "candidate_lane", "method", "result",
                },
                f"M4 numerical comparison[{index}]",
            )
            result = row["result"]
            expected_reference_lane = (
                "gpmeep_v110_cpu"
                if (
                    row.get("case_path") in GPMEEP_CPU_REFERENCE_CASES
                    and row.get("candidate_lane") != "gpmeep_v110_cpu"
                )
                else "upstream_meep_cpu"
            )
            _require(
                row["reference_lane"] == expected_reference_lane
                and row["sample_kind"] in {"warmup", "measured"}
                and type(row["cycle_index"]) is int
                and row["cycle_index"] in {0, 1, 2}
                and isinstance(row["candidate_lane"], str)
                and row["candidate_lane"] in LANES - {"upstream_meep_cpu"}
                and isinstance(result, dict)
                and result.get("outcome") == "PASS"
                and result.get("failure_count") == 0,
                "M4 numerical comparison result differs",
            )
            _verify_m4_numerical_result(
                row["case_path"],
                row["candidate_lane"],
                row["method"],
                result,
                f"M4 numerical comparison[{index}] result",
            )
            identity = (
                row["cycle_index"],
                row["case_path"],
                row["candidate_lane"],
                row["sample_kind"],
            )
            _require(
                identity not in observed_rows,
                "M4 numerical comparison identity repeats",
            )
            observed_rows.add(identity)
        _require(
            observed_rows == expected_rows,
            "M4 numerical comparison lane/cycle census differs",
        )
        matching = [row for row in comparisons if row["case_path"] == expected_workload_id]
        expected_matching_count = sum(
            identity[1] == expected_workload_id for identity in expected_rows
        )
        _require(
            expected_workload_id is not None
            and len(matching) == expected_matching_count
            and expected_matching_count in {9, 15}
            and {row["cycle_index"] for row in matching} == {0, 1, 2},
            "M4 numerical evidence lacks the expected workload/cycles",
        )
        return payload
    payload = _exact(
        payload,
        {
            "schema", "state", "kind", "workload_id", "source_campaign_id",
            "source_inventory_sha256", "classification", "arguments", "inputs",
            "result", "evidence_id",
        },
        "typed inversion evidence",
    )
    unsigned = dict(payload)
    evidence_id = unsigned.pop("evidence_id")
    _require(
        payload["state"] == "PASS"
        and payload["kind"] == expected_kind
        and evidence_id == hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "typed inversion evidence identity differs",
    )
    _require(
        expected_workload_id is not None
        and payload["workload_id"] == expected_workload_id,
        "typed inversion evidence workload differs",
    )
    _require(
        expected_campaign_id is not None
        and payload["source_campaign_id"] == expected_campaign_id,
        "typed inversion evidence campaign differs",
    )
    _require(
        expected_inventory_sha256 is not None
        and payload["source_inventory_sha256"] == expected_inventory_sha256,
        "typed inversion evidence inventory differs",
    )
    _require(
        expected_classification is not None
        and payload["classification"] == expected_classification,
        "typed inversion evidence classification differs",
    )
    expected_arguments = {
        "timing": {"comparison": "cpu_cuda_equal_work", "minimum_cycles": 2},
        "phase_profile": {"phase_inventory": sorted(PHASES), "minimum_cycles": 2},
        "size_run_sweep": {
            "size_flag": "--gpmeep-sweep-size",
            "run_length_flag": "--gpmeep-sweep-run-length",
            "minimum_sizes": 3,
            "minimum_run_lengths": 2,
            "require_observed_crossover": True,
        },
        "environment_diagnostic": {
            "diagnostic_keys": ["environment_sha256", "runtime_id"],
            "minimum_cycles": 2,
        },
    }
    if expected_kind in {"profiler_or_ablation", "same_binary_ablation"}:
        arguments = _exact(
            payload["arguments"],
            {"intervention_flag", "intervention", "minimum_cycles", "require_reversed_order"},
            f"{expected_kind} arguments",
        )
        _require(
            arguments["intervention_flag"] == "--gpmeep-ablation"
            and isinstance(arguments["intervention"], str)
            and re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", arguments["intervention"])
            and arguments["minimum_cycles"] == 2
            and arguments["require_reversed_order"] is True,
            f"{expected_kind} arguments differ",
        )
    else:
        _require(
            payload["arguments"] == expected_arguments[expected_kind],
            f"{expected_kind} arguments differ",
        )
    inputs = payload["inputs"]
    minimum_inputs = {
        "timing": 4,
        "phase_profile": 4,
        "size_run_sweep": 24,
        "profiler_or_ablation": 4,
        "same_binary_ablation": 4,
        "environment_diagnostic": 4,
    }[expected_kind]
    _require(
        isinstance(inputs, list) and len(inputs) >= minimum_inputs,
        f"{expected_kind} evidence lacks its required producer census",
    )
    reports: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(inputs):
        record = _exact(
            value,
            {"kind", "schema", "cycle_index", "path", "sha256"},
            f"typed inversion evidence input[{index}]",
        )
        _require(record["kind"] in SAMPLE_SCHEMAS, "typed inversion evidence producer kind differs")
        if expected_kind in {
            "size_run_sweep",
            "profiler_or_ablation",
            "same_binary_ablation",
            "environment_diagnostic",
        }:
            _require(
                record["kind"] == "experiment_sample",
                f"{expected_kind} requires the dedicated experiment producer",
            )
        cycle = record["cycle_index"]
        _require(type(cycle) is int and cycle in {1, 2}, "typed inversion evidence cycle differs")
        key = (record["path"], record["sha256"])
        _require(key not in seen, "typed inversion evidence repeats producer bytes")
        seen.add(key)
        report = _sample_evidence(
            record,
            f"typed inversion evidence input[{index}]",
            evidence_root,
            expected_kind=record["kind"],
            expected_cycle=cycle,
        )
        _require(report is not None, "typed inversion evidence producer replay is absent")
        _require(report.get("case_path") == expected_workload_id, "typed inversion producer workload differs")
        reports.append(report)
        command = report.get("lane", {}).get("command")
        _require(
            isinstance(command, list)
            and command
            and all(isinstance(argument, str) for argument in command),
            "typed inversion producer command differs",
        )
        identities.append(
            {
                "kind": record["kind"],
                "cycle_index": cycle,
                "sha256": record["sha256"],
                "report_id": report.get("report_id")
                or hashlib.sha256(
                    json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "topology": report.get("topology"),
                "gpu_devices": report.get("gpu_devices", []),
                "lane_devices": report.get("lane", {}).get("spec", {}).get("devices", []),
                "environment_sha256": report.get("lane", {}).get("environment_sha256"),
                "primary_seconds": report.get("timing", {}).get("primary_seconds"),
                "timing_contract": report.get("timing", {}).get("contract"),
                "run_count": report.get("timing", {}).get("run_count"),
                "command_sha256": hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    cycles = sorted({item["cycle_index"] for item in identities})
    _require(cycles == [1, 2], "typed inversion evidence lacks two measured cycles")
    topologies = {report.get("topology") for report in reports}
    _require(
        all(isinstance(item, str) and item for item in topologies),
        f"{expected_kind} producer topology differs",
    )
    if expected_kind in {"timing", "phase_profile", "size_run_sweep"}:
        _require(
            any(item.startswith("cpu") for item in topologies)
            and any(item.startswith("cuda") for item in topologies),
            f"{expected_kind} evidence lacks CPU/CUDA producers",
        )
        if expected_kind == "size_run_sweep":
            _require(
                topologies == {"cpu1", "cuda1"},
                "size-run sweep is a single-rank supplemental diagnostic",
            )
        if expected_kind in {"timing", "phase_profile"}:
            for cycle in (1, 2):
                cycle_topologies = {
                    item["topology"] for item in identities if item["cycle_index"] == cycle
                }
                _require(
                    any(item.startswith("cpu") for item in cycle_topologies)
                    and any(item.startswith("cuda") for item in cycle_topologies),
                    f"{expected_kind} evidence lacks same-cycle CPU/CUDA producers",
                )
            experiment_inputs = [
                item for item in identities if item["kind"] == "experiment_sample"
            ]
            if experiment_inputs:
                _require(
                    len(experiment_inputs) == len(identities)
                    and len({item["command_sha256"] for item in identities}) == 1
                    and len({report["receipt_id"] for report in reports}) == 1,
                    f"{expected_kind} experiment producers are not equal-work/same-runtime",
                )
                for cycle in (1, 2):
                    cycle_inputs = [
                        item for item in identities if item["cycle_index"] == cycle
                    ]
                    _require(
                        {
                            item["timing_contract"] for item in cycle_inputs
                        }
                        == {"sum-all-simulation-run-wall-seconds"}
                        and len({item["run_count"] for item in cycle_inputs}) == 1
                        and type(cycle_inputs[0]["run_count"]) is int
                        and cycle_inputs[0]["run_count"] > 0,
                        f"{expected_kind} experiment CPU/CUDA workload count differs",
                    )
    elif expected_kind in {"profiler_or_ablation", "same_binary_ablation"}:
        _require(
            all(item.startswith("cuda") for item in topologies),
            f"{expected_kind} evidence is not a CUDA same-binary experiment",
        )
    _require(
        all(
            _finite_number(item["primary_seconds"])
            and item["primary_seconds"] > 0
            for item in identities
        ),
        f"{expected_kind} timing facts differ",
    )
    semantics: dict[str, Any]
    if expected_kind == "timing":
        semantics = {
            "topologies": sorted(topologies),
            "mean_primary_seconds": {
                topology: math.fsum(
                    item["primary_seconds"] for item in identities if item["topology"] == topology
                )
                / sum(item["topology"] == topology for item in identities)
                for topology in sorted(topologies)
            },
        }
    elif expected_kind == "phase_profile":
        phase_records = []
        for report in reports:
            phases = report.get("phase_profile", {}).get("phases")
            _require(
                isinstance(phases, dict) and set(phases) == PHASES,
                "phase-profile evidence phase inventory differs",
            )
            available_seconds = {
                name: phases[name]["seconds"]
                for name in sorted(PHASES)
                if phases[name].get("available") is True
            }
            _require(
                available_seconds
                and all(
                    _finite_number(value) and value >= 0
                    for value in available_seconds.values()
                ),
                "phase-profile evidence lacks measured phase seconds",
            )
            phase_records.append(
                {
                    "report_id": report.get("receipt_id"),
                    "available_seconds": available_seconds,
                }
            )
        semantics = {"phase_records": phase_records}
    elif expected_kind == "size_run_sweep":
        size_flag = payload["arguments"]["size_flag"]
        run_flag = payload["arguments"]["run_length_flag"]
        points: dict[tuple[float, float, str], dict[int, float]] = {}
        point_run_counts: dict[tuple[float, float, int], set[int]] = {}
        normalized_commands: set[str] = set()
        runtime_ids: set[str] = set()
        timing_contracts: set[str] = set()
        for report, identity in zip(reports, identities, strict=True):
            command = report["lane"]["command"]
            _require(
                command.count(size_flag) == 1
                and command.count(run_flag) == 1
                and command.index(size_flag) + 1 < len(command)
                and command.index(run_flag) + 1 < len(command),
                "size-run sweep command lacks canonical coordinates",
            )
            try:
                size_value = float(command[command.index(size_flag) + 1])
                run_value = float(command[command.index(run_flag) + 1])
            except ValueError as error:
                raise RegistryError("size-run sweep coordinate is invalid") from error
            normalized = list(command)
            normalized[normalized.index(size_flag) + 1] = "<size>"
            normalized[normalized.index(run_flag) + 1] = "<run-length>"
            normalized_commands.add(json.dumps(normalized, separators=(",", ":")))
            runtime_ids.add(report["receipt_id"])
            _require(
                math.isfinite(size_value) and size_value > 0
                and math.isfinite(run_value) and run_value > 0,
                "size-run sweep coordinate is invalid",
            )
            backend = "cpu" if str(identity["topology"]).startswith("cpu") else "cuda"
            point = points.setdefault((size_value, run_value, backend), {})
            _require(
                identity["cycle_index"] not in point,
                "size-run sweep repeats a coordinate/cycle",
            )
            point[identity["cycle_index"]] = identity["primary_seconds"]
            run_count = identity["run_count"]
            _require(
                type(run_count) is int and run_count > 0,
                "size-run sweep workload count differs",
            )
            point_run_counts.setdefault(
                (size_value, run_value, identity["cycle_index"]), set()
            ).add(run_count)
            timing_contracts.add(identity["timing_contract"])
        sizes = sorted({key[0] for key in points})
        runs = sorted({key[1] for key in points})
        _require(
            len(sizes) >= 3
            and len(runs) >= 2
            and all(
                set(points.get((size, run, backend), {})) == {1, 2}
                for size in sizes for run in runs for backend in ("cpu", "cuda")
            ),
            "size-run sweep coverage differs",
        )
        _require(
            timing_contracts == {"sum-all-simulation-run-wall-seconds"}
            and all(
                len(point_run_counts[(size, run, cycle)]) == 1
                for size in sizes for run in runs for cycle in (1, 2)
            ),
            "size-run sweep CPU/CUDA workload count differs",
        )
        _require(
            len(normalized_commands) == 1
            and len(runtime_ids) == 1
            and all(_hex64(item) for item in runtime_ids),
            "size-run sweep is not an equal-work same-runtime experiment",
        )
        ratios = {
            f"{size:g}:{run:g}": math.fsum(points[(size, run, "cpu")].values())
            / len(points[(size, run, "cpu")])
            / (
                math.fsum(points[(size, run, "cuda")].values())
                / len(points[(size, run, "cuda")])
            )
            for size in sizes for run in runs
        }
        _require(
            min(ratios.values()) < 1.0 < max(ratios.values()),
            "size-run sweep does not demonstrate an observed crossover",
        )
        semantics = {
            "sizes": sizes,
            "run_lengths": runs,
            "cpu_over_cuda": ratios,
            "runtime_id": next(iter(runtime_ids)),
            "normalized_command_sha256": hashlib.sha256(
                next(iter(normalized_commands)).encode()
            ).hexdigest(),
        }
        semantics.update(_verify_sweep_family(
            expected_workload_id, inputs, reports, identities, evidence_root
        ))
    elif expected_kind in {"profiler_or_ablation", "same_binary_ablation"}:
        flag = payload["arguments"]["intervention_flag"]
        intervention = payload["arguments"]["intervention"]
        states: dict[str, list[float]] = {"off": [], "on": []}
        cycle_states: dict[int, list[str]] = {1: [], 2: []}
        cycle_seconds: dict[int, dict[str, float]] = {1: {}, 2: {}}
        runtime_ids: set[str] = set()
        normalized_commands: set[str] = set()
        experiment_topologies: set[str] = set()
        gpu_identities: set[str] = set()
        environment_ids: set[str] = set()
        timing_contracts: set[str] = set()
        run_counts: set[int] = set()
        acknowledgement_counts: dict[str, set[int]] = {
            "off": set(),
            "on": set(),
        }
        policy_runs_by_state: dict[str, list[list[dict[str, Any]]]] = {
            "off": [],
            "on": [],
        }
        inventory_digests: set[str] = set()
        inventory_device_counts: set[int] = set()
        boundary_execution_by_cycle: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
        material_phasing_by_cycle: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
        for report, identity in zip(reports, identities, strict=True):
            command = report["lane"]["command"]
            _require(
                command.count(flag) == 1 and command.index(flag) + 1 < len(command),
                f"{expected_kind} producer lacks intervention argument",
            )
            setting = command[command.index(flag) + 1]
            _require(
                setting in {f"{intervention}=off", f"{intervention}=on"},
                f"{expected_kind} intervention setting differs",
            )
            state = setting.rsplit("=", 1)[1]
            normalized = list(command)
            normalized[normalized.index(flag) + 1] = f"{intervention}=<state>"
            normalized_commands.add(
                json.dumps(normalized, separators=(",", ":"))
            )
            experiment_topologies.add(identity["topology"])
            gpu_identities.add(
                json.dumps(
                    {
                        "gpu_devices": identity["gpu_devices"],
                        "lane_devices": identity["lane_devices"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            environment_id = identity["environment_sha256"]
            _require(_hex64(environment_id), f"{expected_kind} environment identity differs")
            environment_ids.add(environment_id)
            states[state].append(identity["primary_seconds"])
            cycle = identity["cycle_index"]
            _require(
                state not in cycle_seconds[cycle],
                f"{expected_kind} repeats an intervention state/cycle",
            )
            cycle_states[cycle].append(state)
            cycle_seconds[cycle][state] = identity["primary_seconds"]
            runtime_id = report.get("runtime_snapshot", {}).get("runtime_contract", {}).get("receipt_id")
            _require(_hex64(runtime_id), f"{expected_kind} runtime identity differs")
            runtime_ids.add(runtime_id)
            timing = report.get("timing")
            _require(
                isinstance(timing, dict)
                and timing.get("contract")
                == "sum-all-simulation-run-wall-seconds"
                and type(timing.get("run_count")) is int
                and timing["run_count"] > 0,
                f"{expected_kind} workload timing contract differs",
            )
            timing_contracts.add(timing["contract"])
            run_counts.add(timing["run_count"])
            features = report.get("runtime_snapshot", {}).get("feature_counters")
            acknowledgements = (
                features.get("backend_mismatch_acknowledgements")
                if isinstance(features, dict)
                else None
            )
            _require(
                type(acknowledgements) is int
                and 0 <= acknowledgements <= timing["run_count"],
                f"{expected_kind} acknowledgement evidence differs",
            )
            acknowledgement_counts[state].add(acknowledgements)
            policy_runs = report.get("runtime_snapshot", {}).get(
                "backend_policy_runs"
            )
            _require(
                isinstance(policy_runs, list)
                and len(policy_runs) == timing["run_count"]
                and all(isinstance(item, dict) for item in policy_runs),
                f"{expected_kind} backend policy/run evidence differs",
            )
            policy_runs_by_state[state].append(policy_runs)
            if intervention == "phase-material-mirror-reuse":
                _require(report["topology"] == "cuda1",
                         f"{expected_kind} material phasing requires CUDA1")
                observations = report["runtime_snapshot"]["run_observations"]
                totals = {
                    name: sum(item["statistics_after"]["material_phasing"][name]
                              - item["statistics_before"]["material_phasing"][name]
                              for item in observations)
                    for name in ("checks", "reused", "disabled", "ineligible")
                }
                _require(totals["checks"] > 0
                         and totals["checks"] == totals["reused"] + totals["disabled"] + totals["ineligible"]
                         and ((state == "off" and totals["disabled"] == totals["checks"]
                               and totals["reused"] == totals["ineligible"] == 0)
                              or (state == "on" and totals["reused"] > 0 and totals["disabled"] == 0)),
                         f"{expected_kind} material phasing arm differs")
                material_phasing_by_cycle[str(cycle)][state] = totals
            if intervention in {"boundary-plan-replay", "boundary-span-cache"}:
                # Dedicated producer replay above has already validated every
                # raw snapshot and arm. Bind the run-only native path census to
                # the typed same-binary semantics as well.
                _require(report["topology"] == "cuda1",
                         f"{expected_kind} boundary replay requires CUDA1")
                observations = report["runtime_snapshot"]["run_observations"]
                totals = {
                    name: sum(item["statistics_after"]["boundary_execution"][name]
                              - item["statistics_before"]["boundary_execution"][name]
                              for item in observations)
                    for name in observations[0]["statistics_after"]["boundary_execution"]
                }
                _require(totals["local_checks"] > 0
                         and totals["local_failures"] == 0
                         and totals["prepare_calls"] > 0
                         and totals["prepare_failures"] == 0,
                         f"{expected_kind} boundary replay work differs")
                _require(sum(totals[key] for key in (
                    "span_cache_hits", "span_cache_misses", "span_cache_bypasses"
                )) <= totals["prepared_arrays"],
                         f"{expected_kind} boundary span cache accounting differs")
                if intervention == "boundary-span-cache":
                    _require(
                        (state == "off" and totals["span_cache_bypasses"] > 0
                         and totals["span_cache_hits"] == 0 and totals["span_cache_misses"] == 0)
                        or (state == "on" and totals["span_cache_hits"] > 0
                            and totals["span_cache_misses"] > 0 and totals["span_cache_bypasses"] == 0),
                        f"{expected_kind} boundary span cache arm differs",
                    )
                else:
                    _require(
                        (state == "off" and totals["local_disabled"] > 0
                         and totals["local_replay_work"] == 0
                         and totals["local_replay_empty"] == 0
                         and totals["local_descriptor_rebuild_attempts"] > 0)
                        or (state == "on" and totals["local_disabled"] == 0
                            and totals["local_replay_work"] > 0),
                        f"{expected_kind} boundary replay arm differs",
                    )
                boundary_execution_by_cycle[str(cycle)][state] = totals
            inventory = (
                features.get("device_inventory_probe")
                if isinstance(features, dict)
                else None
            )
            if intervention == "device-inventory-snapshot":
                _require(
                    isinstance(inventory, dict)
                    and _hex64(inventory.get("inventory_sha256"))
                    and type(inventory.get("device_count")) is int
                    and inventory["device_count"] > 0,
                    f"{expected_kind} device inventory identity differs",
                )
                inventory_digests.add(inventory["inventory_sha256"])
                inventory_device_counts.add(inventory["device_count"])
            else:
                _require(
                    inventory is None,
                    f"{expected_kind} has an unexpected device inventory probe",
                )
        _require(
            len(runtime_ids) == 1
            and len(states["off"]) >= 2
            and len(states["on"]) >= 2
            and len(normalized_commands) == 1
            and len(experiment_topologies) == 1
            and len(gpu_identities) == 1
            and len(environment_ids) == 1,
            f"{expected_kind} is not a same-binary two-cycle experiment",
        )
        _require(
            timing_contracts == {"sum-all-simulation-run-wall-seconds"}
            and len(run_counts) == 1,
            f"{expected_kind} OFF/ON workload path differs",
        )
        run_count = next(iter(run_counts))
        backend_policy_reuse = None
        if intervention == "backend-ack-scope-cache":
            _require(
                run_count > 1
                and acknowledgement_counts["off"] == {run_count}
                and acknowledgement_counts["on"] == {1},
                f"{expected_kind} backend acknowledgement cache invariant differs",
            )
            flattened = [
                item
                for state in ("off", "on")
                for sample in policy_runs_by_state[state]
                for item in sample
            ]
            identity = {
                json.dumps(
                    {
                        "schema": item.get("schema"),
                        "selected_backend": item.get("selected_backend"),
                        "recommended_backend": item.get("recommended_backend"),
                        "reason_class": item.get("reason_class"),
                        "recommendation_confidence": item.get(
                            "recommendation_confidence"
                        ),
                        "acknowledgement_reason": item.get(
                            "acknowledgement_reason"
                        ),
                        "acknowledgement_scope": item.get(
                            "acknowledgement_scope"
                        ),
                        "acknowledgement_scope_digest": item.get(
                            "acknowledgement_scope_digest"
                        ),
                        "recommendation_policy_version": item.get(
                            "recommendation_policy_version"
                        ),
                        "mpi_topology_digest": item.get(
                            "mpi_topology_digest"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in flattened
            }
            _require(
                len(identity) == 1,
                f"{expected_kind} backend acknowledgement reuse identity differs",
            )
            reuse_identity = json.loads(next(iter(identity)))
            _require(
                reuse_identity
                == {
                    "schema": "gpmeep-v110-backend-recommendation-v3",
                    "selected_backend": "cuda",
                    "recommended_backend": "cpu",
                    "reason_class": "hardware_crossover",
                    "recommendation_confidence": "high",
                    "acknowledgement_reason": (
                        "receipt-bound inversion experiment intentionally "
                        "exercises the selected backend"
                    ),
                    "acknowledgement_scope": "process",
                    "acknowledgement_scope_digest": reuse_identity.get(
                        "acknowledgement_scope_digest"
                    ),
                    "recommendation_policy_version": (
                        "gpmeep-v110-explicit-backend-guard-v24"
                    ),
                    "mpi_topology_digest": reuse_identity.get(
                        "mpi_topology_digest"
                    ),
                }
                and _hex64(reuse_identity["acknowledgement_scope_digest"])
                and _hex64(reuse_identity["mpi_topology_digest"]),
                f"{expected_kind} backend acknowledgement reuse policy differs",
            )
            for sample in policy_runs_by_state["off"]:
                _require(
                    all(
                        item.get("acknowledgement") is True
                        and item.get("acknowledgement_allowed") is True
                        and item.get("acknowledgement_reused") is False
                        and item.get("acknowledgement_scope_cache_enabled")
                        is False
                        and item.get("workload_fingerprint_evaluated") is True
                        and _hex64(item.get("workload_fingerprint"))
                        and _hex64(item.get("acknowledgement_digest"))
                        and item.get("acknowledgement_origin_cohort_digest")
                        == item.get("acknowledgement_digest")
                        for item in sample
                    ),
                    f"{expected_kind} backend cache OFF did not use exact checks",
                )
            for sample in policy_runs_by_state["on"]:
                first, *reused = sample
                _require(
                    first.get("acknowledgement") is True
                    and first.get("acknowledgement_allowed") is True
                    and first.get("acknowledgement_reused") is False
                    and first.get("acknowledgement_scope_cache_enabled") is True
                    and first.get("workload_fingerprint_evaluated") is True
                    and _hex64(first.get("workload_fingerprint"))
                    and _hex64(first.get("acknowledgement_digest"))
                    and first.get("acknowledgement_origin_cohort_digest")
                    == first.get("acknowledgement_digest")
                    and all(
                        item.get("acknowledgement") is True
                        and item.get("acknowledgement_allowed") is True
                        and item.get("acknowledgement_reused") is True
                        and item.get("acknowledgement_scope_cache_enabled") is True
                        and item.get("acknowledgement_scope") == "process"
                        and item.get("acknowledgement_origin_cohort_digest")
                        == first.get("acknowledgement_digest")
                        and item.get("workload_fingerprint_evaluated") is False
                        and item.get("workload_fingerprint") == ""
                        and item.get("acknowledgement_digest") == ""
                        for item in reused
                    ),
                    f"{expected_kind} backend cache ON did not use audited reuse",
                )
            backend_policy_reuse = {
                "schema": reuse_identity["schema"],
                "policy_version": reuse_identity[
                    "recommendation_policy_version"
                ],
                "scope_digest": reuse_identity[
                    "acknowledgement_scope_digest"
                ],
                "mpi_topology_digest": reuse_identity[
                    "mpi_topology_digest"
                ],
                "reason_sha256": hashlib.sha256(
                    reuse_identity["acknowledgement_reason"].encode("utf-8")
                ).hexdigest(),
                "off": {"exact": run_count, "reused": 0},
                "on": {"exact": 1, "reused": run_count - 1},
            }
        else:
            _require(
                len(acknowledgement_counts["off"]) == 1
                and acknowledgement_counts["off"]
                == acknowledgement_counts["on"],
                f"{expected_kind} OFF/ON acknowledgement path differs",
            )
        if intervention == "device-inventory-snapshot":
            _require(
                len(inventory_digests) == 1
                and len(inventory_device_counts) == 1,
                f"{expected_kind} OFF/ON public device inventory differs",
            )
        _require(
            all(set(cycle_states[cycle]) == {"off", "on"} for cycle in (1, 2))
            and cycle_states[1] == list(reversed(cycle_states[2])),
            f"{expected_kind} does not use reversed intervention order",
        )
        off_mean = math.fsum(states["off"]) / len(states["off"])
        on_mean = math.fsum(states["on"]) / len(states["on"])
        _require(
            on_mean < off_mean
            and all(
                cycle_seconds[cycle]["on"] < cycle_seconds[cycle]["off"]
                for cycle in (1, 2)
            ),
            f"{expected_kind} intervention has no repeatable measured benefit",
        )
        semantics = {
            "intervention": intervention,
            "runtime_id": next(iter(runtime_ids)),
            "topology": next(iter(experiment_topologies)),
            "gpu_identity": json.loads(next(iter(gpu_identities))),
            "environment_sha256": next(iter(environment_ids)),
            "normalized_command_sha256": hashlib.sha256(
                next(iter(normalized_commands)).encode()
            ).hexdigest(),
            "timing_contract": next(iter(timing_contracts)),
            "run_count": run_count,
            "backend_mismatch_acknowledgements": {
                state: next(iter(acknowledgement_counts[state]))
                for state in ("off", "on")
            },
            "off_seconds": states["off"],
            "on_seconds": states["on"],
            "relative_improvement": (off_mean - on_mean) / off_mean,
            "paired_relative_improvement": [
                (
                    cycle_seconds[cycle]["off"]
                    - cycle_seconds[cycle]["on"]
                )
                / cycle_seconds[cycle]["off"]
                for cycle in (1, 2)
            ],
        }
        if intervention == "device-inventory-snapshot":
            semantics["device_inventory"] = {
                "device_count": next(iter(inventory_device_counts)),
                "inventory_sha256": next(iter(inventory_digests)),
            }
        if backend_policy_reuse is not None:
            semantics["backend_policy_reuse"] = backend_policy_reuse
        if intervention == "phase-material-mirror-reuse":
            semantics["material_phasing_by_cycle"] = material_phasing_by_cycle
        if intervention in {"boundary-plan-replay", "boundary-span-cache"}:
            semantics["boundary_execution_by_cycle"] = boundary_execution_by_cycle
    else:
        environment_ids = {
            report.get("lane", {}).get("environment_sha256") for report in reports
        }
        runtime_ids = {
            report.get("runtime_snapshot", {}).get("runtime_contract", {}).get("receipt_id")
            for report in reports
        }
        normalized_commands = {
            json.dumps(report["lane"]["command"], separators=(",", ":"))
            for report in reports
        }
        diagnostic_topologies = {report["topology"] for report in reports}
        _require(
            len(environment_ids) >= 2
            and len(runtime_ids) == 1
            and len(normalized_commands) == 1
            and len(diagnostic_topologies) == 1
            and all(_hex64(item) for item in environment_ids | runtime_ids),
            "environment diagnostic identity differs",
        )
        semantics = {
            "environment_sha256": sorted(environment_ids),
            "runtime_ids": sorted(runtime_ids),
            "topology": next(iter(diagnostic_topologies)),
            "command_sha256": hashlib.sha256(
                next(iter(normalized_commands)).encode()
            ).hexdigest(),
        }
    expected_result = {
        "producer_input_count": len(identities),
        "cycles": cycles,
        "producer_reports": identities,
        "semantics": semantics,
    }
    if derive_result:
        # The production evidence compiler uses the exact verifier path to
        # derive this field, then publishes a fresh canonical evidence_id and
        # invokes the normal (non-derivation) replay below.  No classification
        # policy or raw-sample interpretation is duplicated in the creator.
        return expected_result
    _require(payload["result"] == expected_result, "typed inversion evidence result was not re-derived")
    return payload


def _sweep_family_verifier(workload_id: str) -> tuple[str, Any]:
    # A nominal size/length matrix and a timing crossover do not establish
    # distinct native work or numerical equivalence. Each accepted family must
    # bind its actual grids/steps, complete original results and paired oracle.
    # Other cases remain open until their family verifier is implemented; raw
    # experimental collection alone does not authorize hardware classification.
    if workload_id == "python/examples/phase_in_material.py":
        return "phase_numerical_pairs", _verify_phase_sweep_numerics
    raise RegistryError(
        "size-run sweep lacks a source-bound actual-work and numerical family verifier"
    )


def _verify_sweep_family(
    workload_id: str, inputs: list[dict[str, Any]], reports: list[dict[str, Any]],
    identities: list[dict[str, Any]], evidence_root: pathlib.Path,
) -> dict[str, Any]:
    key, verifier = _sweep_family_verifier(workload_id)
    return {key: verifier(inputs, reports, identities, evidence_root)}


def _verify_phase_sweep_numerics(
    inputs: list[dict[str, Any]], reports: list[dict[str, Any]],
    identities: list[dict[str, Any]], evidence_root: pathlib.Path,
) -> list[dict[str, Any]]:
    import types
    relative = "scripts/python-validation/phase_material_contract.py"
    helper_path = ROOT / relative
    helper_source = helper_path.read_bytes()
    helper_sha256 = hashlib.sha256(helper_source).hexdigest()
    for report in reports:
        authority = report["runtime_authority"]
        helpers = authority["workload_contract"]["support_files"]
        _require(len(helpers) == 1 and helpers[0] == {
            "path": str(pathlib.Path(authority["repo"]) / relative),
            "size_bytes": len(helper_source), "sha256": helper_sha256,
        }, "phase numerical comparator differs from the source-bound producer helper")
    helper = types.ModuleType("gpmeep_registry_phase_material_contract")
    helper.__file__ = str(helper_path)
    # Compile exactly the measured helper bytes; never consult timestamp .pyc files.
    exec(compile(helper_source, str(helper_path), "exec"), helper.__dict__)
    points: dict[tuple[float, float, int, str], tuple[dict[str, Any], pathlib.Path]] = {}
    actual_sizes: dict[float, tuple[int, ...]] = {}
    actual_lengths: dict[float, dict[float, int]] = {}
    for reference, report, identity in zip(inputs, reports, identities, strict=True):
        command = report["lane"]["command"]
        size = float(command[command.index("--gpmeep-sweep-size") + 1])
        duration = float(command[command.index("--gpmeep-sweep-run-length") + 1])
        numerical = report["runtime_snapshot"].get("phase_numerical")
        _require(isinstance(numerical, dict), "phase sweep lacks full numerical evidence")
        contract = helper.work_contract(size, duration, fixed_work=True)
        _require(numerical.get("contract") == contract, "phase sweep numerical coordinates differ")
        grid = tuple(contract["grid_shape"])
        _require(actual_sizes.setdefault(size, grid) == grid, "phase sweep grid differs within a size")
        lengths = actual_lengths.setdefault(size, {})
        steps = contract["steps_per_run"]
        _require(lengths.setdefault(duration, steps) == steps,
                 "phase sweep step count differs within a coordinate")
        backend = "cpu" if identity["topology"].startswith("cpu") else "cuda"
        key = (size, duration, identity["cycle_index"], backend)
        _require(key not in points, "phase sweep repeats numerical coordinates")
        sample_root = (evidence_root / reference["path"]).parent
        points[key] = (numerical, sample_root / "phase-numerical.npy")
    _require(len(set(actual_sizes.values())) == len(actual_sizes),
             "phase sweep coordinates do not produce distinct actual spatial sizes")
    _require(all(len(set(lengths.values())) == len(lengths) for lengths in actual_lengths.values()),
             "phase sweep run coordinates do not produce distinct actual step counts")
    comparisons = []
    for size in sorted(actual_sizes):
        for duration in sorted(actual_lengths[size]):
            for cycle in (1, 2):
                _require((size, duration, cycle, "cpu") in points
                         and (size, duration, cycle, "cuda") in points,
                         "phase sweep numerical CPU/CUDA pair is missing")
                cpu, cpu_path = points[(size, duration, cycle, "cpu")]
                cuda, cuda_path = points[(size, duration, cycle, "cuda")]
                try:
                    result = helper.compare_numerical(cpu, cpu_path, cuda, cuda_path)
                except (ValueError, TypeError, KeyError, OSError) as error:
                    raise RegistryError(f"phase sweep numerical comparison differs: {error}") from error
                comparisons.append({"size": size, "run_length": duration,
                                    "cycle_index": cycle, **result})
    return comparisons


def _load_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def _sample_evidence(
    value: Any,
    name: str,
    evidence_root: pathlib.Path | None,
    *,
    expected_kind: str,
    expected_cycle: int,
) -> dict[str, Any] | None:
    item = _exact(
        value,
        {"kind", "schema", "cycle_index", "path", "sha256"},
        name,
    )
    _require(item["kind"] == expected_kind, f"{name}.kind differs")
    _require(item["schema"] == SAMPLE_SCHEMAS[expected_kind], f"{name}.schema differs")
    _require(item["cycle_index"] == expected_cycle, f"{name}.cycle differs")
    _require(_hex64(item["sha256"]), f"{name}.sha256 is invalid")
    relative = pathlib.PurePosixPath(item["path"])
    _require(
        item["path"] not in {"", "."}
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.name == "report.json",
        f"{name}.path is unsafe or not a sample report",
    )
    if evidence_root is None:
        return None
    lexical = evidence_root / pathlib.Path(*relative.parts)
    _require(not lexical.is_symlink(), f"{name}.path cannot be a symlink")
    path = lexical.resolve(strict=True)
    _require(path.is_relative_to(evidence_root), f"{name}.path escapes evidence root")
    _require(path == lexical, f"{name}.path cannot contain symlinks")
    _require(path.is_file() and _sha256(path) == item["sha256"], f"{name} bytes differ")
    if expected_kind == "upstream_sample":
        module = _load_module(
            "gpmeep_registry_upstream_sample",
            "scripts/python-validation/run_v110_upstream_performance_sample.py",
        )
    elif expected_kind == "v103_sample":
        module = _load_module(
            "gpmeep_registry_v103_sample",
            "scripts/python-validation/run_v110_v103_performance_sample.py",
        )
    elif expected_kind == "v110_sample":
        module = _load_module(
            "gpmeep_registry_v110_sample",
            "scripts/python-validation/run_m3_performance_sample.py",
        )
    else:
        module = _load_module(
            "gpmeep_registry_inversion_experiment_sample",
            "scripts/v110-inversion-experiment-sample.py",
        )
    report = module.verify_complete(path.parent)
    _require(report.get("sample_kind") == "measured", f"{name} is not measured")
    _require(report.get("cycle_index") == expected_cycle, f"{name} report cycle differs")
    _require(report.get("schema") == item["schema"], f"{name} report schema differs")
    return report


def _lane(
    value: Any,
    lane_name: str,
    workload_id: str,
    evidence_root: pathlib.Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = f"comparison.lanes.{lane_name}"
    lane = _exact(
        value,
        {
            "runtime_id",
            "runtime",
            "backend",
            "topology",
            "devices",
            "measured_seconds",
            "mean_seconds",
            "samples",
        },
        name,
    )
    runtime, backend, topologies, device_count, sample_kind = LANE_CONTRACTS[lane_name]
    _require(_hex64(lane["runtime_id"]), f"{name}.runtime_id is invalid")
    _require(lane["runtime"] == runtime, f"{name}.runtime differs")
    _require(lane["backend"] == backend, f"{name}.backend differs")
    _require(lane["topology"] in topologies, f"{name}.topology differs")
    devices = lane["devices"]
    _require(
        isinstance(devices, list)
        and len(devices) == device_count
        and len(set(devices)) == len(devices)
        and all(isinstance(item, str) and item for item in devices),
        f"{name}.devices differs",
    )
    seconds = lane["measured_seconds"]
    _require(
        isinstance(seconds, list)
        and len(seconds) == 2
        and all(_finite_number(item) and item > 0 for item in seconds),
        f"{name}.measured_seconds differs",
    )
    expected_mean = math.fsum(float(item) for item in seconds) / 2.0
    _require(
        _finite_number(lane["mean_seconds"])
        and lane["mean_seconds"] > 0
        and math.isclose(
            float(lane["mean_seconds"]), expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ),
        f"{name}.mean_seconds differs",
    )
    samples = lane["samples"]
    _require(isinstance(samples, list) and len(samples) == 2, f"{name}.samples differs")
    reports = []
    for index, sample in enumerate(samples, start=1):
        report = _sample_evidence(
            sample,
            f"{name}.samples[{index - 1}]",
            evidence_root,
            expected_kind=sample_kind,
            expected_cycle=index,
        )
        if report is not None:
            reports.append(report)
    if reports:
        _require(len(reports) == 2, f"{name} live sample count differs")
        _require(
            all(report.get("case_path") == workload_id for report in reports),
            f"{name} sample workload differs",
        )
        expected_topologies = {
            "cpu1": {"cpu1", "upstream-cpu1"},
            "cpu8": {"cpu8", "upstream-cpu8"},
            "gpu1": {"cuda1", "v103-cuda1"},
            "gpu2": {"cuda2", "v103-cuda2"},
        }[lane["topology"]]
        _require(
            all(report.get("topology") in expected_topologies for report in reports),
            f"{name} sample topology differs",
        )
        _require(
            [report["timing"]["primary_seconds"] for report in reports] == seconds,
            f"{name} sample seconds differ",
        )
        _require(
            all(report.get("receipt_id") == lane["runtime_id"] for report in reports),
            f"{name} sample runtime identity differs",
        )
        expected_devices = [
            report["lane"]["spec"]["devices"] for report in reports
        ]
        _require(
            all(item == devices for item in expected_devices),
            f"{name} sample device identity differs",
        )
    return lane, reports


def _ratio(
    value: Any,
    name: str,
    numerator_name: str,
    denominator_name: str,
    lanes: dict[str, Any],
) -> None:
    ratio = _exact(
        value,
        {
            "numerator_lane",
            "denominator_lane",
            "paired_cycle_values",
            "ratio_of_means",
        },
        name,
    )
    _require(ratio["numerator_lane"] == numerator_name, f"{name}.numerator differs")
    _require(ratio["denominator_lane"] == denominator_name, f"{name}.denominator differs")
    paired = ratio["paired_cycle_values"]
    numerator = lanes[numerator_name]
    denominator = lanes[denominator_name]
    expected_paired = [
        float(left) / float(right)
        for left, right in zip(
            numerator["measured_seconds"], denominator["measured_seconds"], strict=True
        )
    ]
    _require(
        isinstance(paired, list)
        and len(paired) == 2
        and all(_finite_number(item) and item > 0 for item in paired)
        and all(
            math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(paired, expected_paired, strict=True)
        ),
        f"{name}.paired_cycle_values differs",
    )
    expected_mean_ratio = float(numerator["mean_seconds"]) / float(
        denominator["mean_seconds"]
    )
    _require(
        _finite_number(ratio["ratio_of_means"])
        and ratio["ratio_of_means"] > 0
        and math.isclose(
            float(ratio["ratio_of_means"]),
            expected_mean_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{name}.ratio_of_means differs",
    )


def _ratio_set(
    value: Any,
    label: str,
    suffix: str,
    lanes: dict[str, Any],
) -> None:
    ratios = _exact(value, RATIO_KEYS, label)
    v103 = f"gpmeep_v103_cuda_{suffix}"
    v110 = f"gpmeep_v110_cuda_{suffix}"
    pairs = {
        "upstream_cpu_over_v103_cuda": ("upstream_meep_cpu", v103),
        "upstream_cpu_over_v110_cuda": ("upstream_meep_cpu", v110),
        "v103_cuda_over_v110_cuda": (v103, v110),
        "v110_cpu_over_v110_cuda": ("gpmeep_v110_cpu", v110),
    }
    for ratio_name, (numerator, denominator) in pairs.items():
        _ratio(
            ratios[ratio_name],
            f"{label}.{ratio_name}",
            numerator,
            denominator,
            lanes,
        )


def _validate_record(
    record: dict[str, Any],
    observation: dict[str, Any],
    inventory_path: str,
    inventory_sha256: str,
    campaign_id: str,
    evidence_root: pathlib.Path | None,
) -> tuple[str, bool, bool]:
    _exact(
        record,
        {
            "schema_version",
            "record_id",
            "workload_id",
            "observed_gpu_slower_than_cpu",
            "source_inventory",
            "representative",
            "comparison",
            "phase_decomposition",
            "primary_class",
            "classification_evidence",
            "confidence",
            "owner",
            "resolution",
            "release_blocking",
        },
        "record",
    )
    _require(type(record["schema_version"]) is int and record["schema_version"] == 2, "record schema version differs")
    _require(
        isinstance(record["record_id"], str)
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]+", record["record_id"]) is not None,
        "record_id is invalid",
    )
    _require(record["workload_id"] == observation["workload_id"], "record workload differs")
    _require(record["observed_gpu_slower_than_cpu"] is True, "record is not an observed inversion")

    source = _exact(
        record["source_inventory"],
        {"campaign_id", "inventory_path", "inventory_sha256", "case_digest"},
        "source_inventory",
    )
    _require(source["campaign_id"] == campaign_id, "source campaign differs")
    _require(source["inventory_path"] == inventory_path, "source inventory path differs")
    _require(source["inventory_sha256"] == inventory_sha256, "source inventory hash differs")
    _require(
        source["case_digest"]
        == observation["workload_identity"]["case_contract_sha256"],
        "source case digest differs",
    )

    representative = _exact(
        record["representative"],
        {"value", "impact", "selection_policy_version", "rationale"},
        "representative",
    )
    _require(
        representative["value"] == observation["representative"],
        "representative.value differs from inventory",
    )
    for key in ("impact", "rationale"):
        _require(representative[key] == observation[key], f"representative.{key} differs from inventory")
    _require(representative["selection_policy_version"] == "v110-m0-1", "representative policy differs")

    comparison = _exact(
        record["comparison"],
        {
            "physics_fingerprint",
            "precision",
            "stopping_condition",
            "cpu_topology",
            "lanes",
            "ratios",
        },
        "comparison",
    )
    _require(_hex64(comparison["physics_fingerprint"]), "physics fingerprint is invalid")
    for key in ("precision", "stopping_condition", "cpu_topology"):
        _require(isinstance(comparison[key], str) and comparison[key], f"comparison.{key} is empty")
    _require(comparison["precision"] == "fp32", "comparison precision must be fp32")
    lanes = _exact(comparison["lanes"], LANES, "comparison.lanes")
    lane_reports: dict[str, list[dict[str, Any]]] = {}
    for lane_name in LANES - GPU2_LANES:
        lanes[lane_name], lane_reports[lane_name] = _lane(
            lanes[lane_name], lane_name, record["workload_id"], evidence_root
        )
    for lane_name in GPU2_LANES:
        if lanes[lane_name] is not None:
            lanes[lane_name], lane_reports[lane_name] = _lane(
                lanes[lane_name], lane_name, record["workload_id"], evidence_root
            )
    if observation["source_lane"] == "mpi_cpu8_vs_gpu1_gpu2":
        _require(
            all(lanes[lane_name] is not None for lane_name in GPU2_LANES),
            "historical MPI inversion requires both versioned GPU2 comparison lanes",
        )
    ratios = _exact(comparison["ratios"], {"gpu1", "gpu2"}, "comparison.ratios")
    _ratio_set(ratios["gpu1"], "comparison.ratios.gpu1", "gpu1", lanes)
    gpu2_required = observation["source_lane"] == "mpi_cpu8_vs_gpu1_gpu2"
    if gpu2_required:
        _require(ratios["gpu2"] is not None, "MPI inversion lacks GPU2 ratios")
        _ratio_set(ratios["gpu2"], "comparison.ratios.gpu2", "gpu2", lanes)
    else:
        _require(ratios["gpu2"] is None, "non-GPU2 inversion has GPU2 ratios")

    phase_by_lane = _exact(record["phase_decomposition"], LANES, "phase_decomposition")
    for lane_name in LANES:
        phases = phase_by_lane[lane_name]
        if lanes[lane_name] is None:
            _require(phases is None, f"phase_decomposition.{lane_name} must be null")
            continue
        _require(phases is not None, f"phase_decomposition.{lane_name} is absent")
        phases = _exact(phases, PHASES, f"phase_decomposition.{lane_name}")
        reports = lane_reports.get(lane_name, [])
        for phase_name, phase_value in phases.items():
            label = f"phase_decomposition.{lane_name}.{phase_name}"
            _require(isinstance(phase_value, dict), f"{label} must be an object")
            available = phase_value.get("available")
            _require(type(available) is bool, f"{label}.available is invalid")
            _require(phase_value.get("cycle_indices") == [1, 2], f"{label}.cycles differ")
            basis = phase_value.get("basis")
            sinks = phase_value.get("native_sinks")
            _require(
                isinstance(basis, list)
                and len(basis) == 2
                and all(isinstance(item, str) and item for item in basis),
                f"{label}.basis differs",
            )
            _require(
                isinstance(sinks, list)
                and len(sinks) == 2
                and all(
                    isinstance(items, list)
                    and len(set(items)) == len(items)
                    and all(isinstance(item, str) and item for item in items)
                    for items in sinks
                ),
                f"{label}.native_sinks differs",
            )
            measured = phase_value.get("measured_seconds")
            if available:
                phase = _exact(
                    phase_value,
                    {
                        "available",
                        "measured_seconds",
                        "mean_seconds",
                        "basis",
                        "native_sinks",
                        "cycle_indices",
                    },
                    label,
                )
                _require(
                    isinstance(measured, list)
                    and len(measured) == 2
                    and all(_finite_number(item) and item >= 0 for item in measured),
                    f"{label}.measured_seconds differs",
                )
                expected_mean = math.fsum(float(item) for item in measured) / 2.0
                _require(
                    _finite_number(phase["mean_seconds"])
                    and phase["mean_seconds"] >= 0
                    and math.isclose(
                        float(phase["mean_seconds"]),
                        expected_mean,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    f"{label}.mean_seconds differs",
                )
            else:
                phase = _exact(
                    phase_value,
                    {
                        "available",
                        "measured_seconds",
                        "mean_seconds",
                        "reason_code",
                        "basis",
                        "native_sinks",
                        "cycle_indices",
                    },
                    label,
                )
                _require(measured == [None, None], f"{label}.measured_seconds must be null")
                _require(phase["mean_seconds"] is None, f"{label}.mean_seconds must be null")
                allowed_reasons = {
                    "no_comparable_native_timer",
                    "initialization_not_separable",
                    "phase_not_exercised_in_both_samples",
                }
                _require(phase["reason_code"] in allowed_reasons, f"{label}.reason differs")
            if reports:
                raw_phases = [report["phase_profile"]["phases"][phase_name] for report in reports]
                raw_available = [item["available"] is True for item in raw_phases]
                expected_available = all(raw_available)
                _require(available is expected_available, f"{label} availability differs from samples")
                expected_seconds = (
                    [item["seconds"] for item in raw_phases]
                    if expected_available
                    else [None, None]
                )
                _require(measured == expected_seconds, f"{label} seconds differ from samples")
                _require(basis == [item["basis"] for item in raw_phases], f"{label} basis differs from samples")
                _require(sinks == [item["native_sinks"] for item in raw_phases], f"{label} sinks differ from samples")
                if not expected_available:
                    if phase_name == "initialization":
                        expected_reason = "initialization_not_separable"
                    elif any(item["native_sinks"] for item in raw_phases):
                        expected_reason = "phase_not_exercised_in_both_samples"
                    else:
                        expected_reason = "no_comparable_native_timer"
                    _require(
                        phase["reason_code"] == expected_reason,
                        f"{label} reason differs from samples",
                    )

    primary_class = record["primary_class"]
    _require(primary_class in CLASSES, "primary class is invalid")
    evidence = record["classification_evidence"]
    _require(isinstance(evidence, list) and evidence, "classification evidence is empty")
    evidence_kinds = {
        _evidence(
            item,
            f"classification_evidence[{index}]",
            evidence_root,
            expected_workload_id=record["workload_id"],
            expected_campaign_id=campaign_id,
            expected_inventory_sha256=inventory_sha256,
            expected_classification=primary_class,
        )
        for index, item in enumerate(evidence)
    }
    _require(
        _finite_number(record["confidence"])
        and 0 <= record["confidence"] <= 1,
        "confidence is invalid",
    )
    _require(isinstance(record["owner"], str) and record["owner"], "record owner is empty")

    resolution = _exact(
        record["resolution"],
        {
            "state",
            "same_binary_ablation",
            "numerical_regression",
            "performance_claim_eligible",
            "notes",
        },
        "resolution",
    )
    _require(isinstance(resolution["notes"], str) and resolution["notes"], "resolution notes are empty")
    _require(
        type(resolution["performance_claim_eligible"]) is bool,
        "resolution performance-claim eligibility is invalid",
    )
    for key, kind in (
        ("same_binary_ablation", "same_binary_ablation"),
        ("numerical_regression", "numerical_regression"),
    ):
        if resolution[key] is not None:
            _evidence(
                resolution[key],
                f"resolution.{key}",
                evidence_root,
                kind,
                expected_workload_id=record["workload_id"],
                expected_campaign_id=campaign_id,
                expected_inventory_sha256=inventory_sha256,
                expected_classification=primary_class,
            )

    state = resolution["state"]
    closed = False
    if state == "user_approved_scope_exclusion":
        raise RegistryError(
            "user-approved scope exclusion cannot close a v1.1 release inversion"
        )
    if primary_class == "hardware_crossover":
        # Schema-only inspection must not count an unregistered family as
        # proved merely because evidence bytes were not requested in this view.
        _sweep_family_verifier(record["workload_id"])
        sealed_plan = _load(ROOT / "packaging/v110-m4-equal-work-plan.json")
        planned_cases = [
            item
            for item in sealed_plan.get("cases", [])
            if item.get("path") == record["workload_id"]
        ]
        _require(len(planned_cases) == 1, "hardware crossover workload is absent from M4 plan")
        expected_cpu_topology = (
            "cpu1" if planned_cases[0].get("cpu_ranks") == 1 else "cpu8"
        )
        _require(
            state == "proved_hardware_crossover",
            "unproved crossover must remain Unknown rather than hardware_crossover",
        )
        _require(
            {"size_run_sweep", "profiler_or_ablation"} <= evidence_kinds,
            "hardware crossover proof is incomplete",
        )
        _require(
            lanes["gpmeep_v110_cpu"]["topology"] == expected_cpu_topology
            and lanes["gpmeep_v110_cuda_gpu1"]["topology"] == "gpu1"
            and (
                evidence_root is None
                or (
                    len(lane_reports["gpmeep_v110_cpu"]) == 2
                    and len(lane_reports["gpmeep_v110_cuda_gpu1"]) == 2
                )
            ),
            "hardware crossover lacks the sealed practical M4 CPU/GPU1 anchor",
        )
        closed = True
    elif primary_class == "engine_overhead":
        _require(state in {"open", "optimized_and_remeasured"}, "engine resolution state is invalid")
        _require(
            {"phase_profile", "profiler_or_ablation"} & evidence_kinds,
            "engine classification lacks phase/profile evidence",
        )
        if state == "optimized_and_remeasured":
            _require(resolution["same_binary_ablation"] is not None, "engine fix lacks same-binary ablation")
            _require(resolution["numerical_regression"] is not None, "engine fix lacks numerical regression")
            closed = True
    elif primary_class == "environment":
        _require(state in {"open", "environment_fixed_and_remeasured"}, "environment resolution state is invalid")
        _require("environment_diagnostic" in evidence_kinds, "environment class lacks diagnostic")
        if state == "environment_fixed_and_remeasured":
            _require("environment_diagnostic" in evidence_kinds, "environment fix lacks diagnostic")
            _require(resolution["numerical_regression"] is not None, "environment fix lacks remeasurement")
            closed = True
    else:
        _require(state == "open", "Unknown inversion cannot be closed without scope approval")

    _require(
        resolution["performance_claim_eligible"] is True,
        "evaluated inversion must remain performance-claim eligible",
    )
    if closed:
        _require(
            resolution["numerical_regression"] is not None,
            "closed inversion lacks mandatory numerical regression",
        )

    should_block = not closed and (
        (
            representative["value"] is True
            and representative["impact"] == "high"
        )
        or primary_class == "environment"
    )
    _require(type(record["release_blocking"]) is bool, "release_blocking must be boolean")
    _require(record["release_blocking"] is should_block, "release_blocking disagrees with policy")
    return record["record_id"], should_block, closed


def validate_registry(
    *,
    require_closed: bool,
    evidence_root: pathlib.Path | None,
    records_directory: pathlib.Path | None = None,
) -> dict[str, Any]:
    contract = _load(CONTRACT)
    registry = contract["performance_inversion_policy"]["registry"]
    schema_path = ROOT / registry["schema"]
    inventory_path = ROOT / registry["initial_inventory"]
    _require(_sha256(schema_path) == registry["schema_sha256"], "record schema hash differs")
    inventory_sha256 = _sha256(inventory_path)
    _require(inventory_sha256 == registry["initial_inventory_sha256"], "inventory hash differs")
    inventory = _load(inventory_path)
    schema = _load(schema_path)
    _require(inventory.get("immutable") is True, "inventory is not immutable")
    observations = inventory.get("observations")
    _require(isinstance(observations, list) and observations, "inventory observations are empty")
    by_workload = {item["workload_id"]: item for item in observations}
    _require(len(by_workload) == len(observations), "inventory repeats workloads")

    default_records_directory = records_directory is None
    if records_directory is None:
        records_directory = ROOT / registry["records_directory"]
    lexical_records = records_directory.absolute()
    _require(not lexical_records.is_symlink(), "registry records directory cannot be a symlink")
    records_directory = lexical_records.resolve(strict=True)
    _require(records_directory == lexical_records, "registry records directory path cannot contain symlinks")
    _require(records_directory.is_dir(), "registry records directory is absent")
    records_by_workload: dict[str, pathlib.Path] = {}
    blockers: list[str] = []
    open_workloads: list[str] = []
    record_ids: set[str] = set()
    if evidence_root is not None:
        lexical_evidence = evidence_root.absolute()
        _require(not lexical_evidence.is_symlink(), "evidence root cannot be a symlink")
        evidence_root = lexical_evidence.resolve(strict=True)
        _require(evidence_root == lexical_evidence, "evidence root path cannot contain symlinks")
        _require(evidence_root.is_dir(), "evidence root is not a directory")
    directory_entries = sorted(records_directory.iterdir())
    for path in directory_entries:
        if default_records_directory and path.name == ".gitkeep":
            _require(path.is_file() and not path.is_symlink(), "registry .gitkeep is unsafe")
            continue
        _require(not path.is_symlink(), f"registry record cannot be a symlink: {path.name}")
        _require(path.is_file() and path.suffix == ".json", f"unexpected registry directory member: {path.name}")
        record = _load(path)
        _validate_json_schema(record, schema)
        workload_id = record.get("workload_id")
        _require(workload_id in by_workload, f"record is not in immutable inventory: {path.name}")
        _require(workload_id not in records_by_workload, f"duplicate workload record: {workload_id}")
        record_id, blocks, closed = _validate_record(
            record,
            by_workload[workload_id],
            registry["initial_inventory"],
            inventory_sha256,
            inventory["source_campaign"]["id"],
            evidence_root,
        )
        _require(record_id not in record_ids, f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        records_by_workload[workload_id] = path
        if blocks:
            blockers.append(workload_id)
        if not closed:
            open_workloads.append(workload_id)

    missing = sorted(set(by_workload) - set(records_by_workload))
    if require_closed:
        _require(evidence_root is not None, "--require-closed needs --evidence-root")
        _require(not missing, f"unregistered observed inversions: {missing}")
        _require(not blockers, f"open representative release blockers: {blockers}")
        _require(
            not open_workloads,
            f"open observed inversions require an evidence-backed resolution: {sorted(open_workloads)}",
        )
    state = "PASS" if not missing and not open_workloads else "OPEN"
    return {
        "schema_version": 1,
        "state": state,
        "inventory_sha256": inventory_sha256,
        "observed_count": len(by_workload),
        "registered_count": len(records_by_workload),
        "missing_count": len(missing),
        "blocking_count": len(blockers),
        "closed_count": len(records_by_workload) - len(open_workloads),
        "open_count": len(open_workloads),
        "missing_workloads": missing,
        "blocking_workloads": sorted(blockers),
        "open_workloads": sorted(open_workloads),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-open", action="store_true")
    mode.add_argument("--require-closed", action="store_true")
    parser.add_argument("--evidence-root", type=pathlib.Path)
    parser.add_argument(
        "--records-directory",
        type=pathlib.Path,
        help="validate an exact externally produced record directory against the normative registry",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_registry(
            require_closed=args.require_closed,
            evidence_root=args.evidence_root,
            records_directory=args.records_directory,
        )
    except (KeyError, TypeError, OSError, json.JSONDecodeError, RegistryError) as error:
        if args.json:
            print(json.dumps({"schema_version": 1, "state": "FAIL", "error": str(error)}, sort_keys=True))
        else:
            print(f"v1.1.0 inversion registry: FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True) if args.json else f"v1.1.0 inversion registry: {result['state']} ({result['registered_count']}/{result['observed_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
