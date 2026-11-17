#!/usr/bin/env python3
"""Run one provenance-bound v1.1.0 M1 OFF/C1/C3/C1+C3/C4 sample."""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
from typing import Any

import run_aunp_material_omp_sample as c4
import run_aunp_sparse_lorentzian_sample as c3


_BASE_PATH = pathlib.Path(__file__).with_name("run_aunp_zero_state_sample.py")
_BASE_MODULE_NAME = "_gpmeep_v110_m1_ablation_sample_base"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private M1 ablation sample base module")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)
_base_validate_feature_counters = base._validate_feature_counters


MODES = ("off", "c1", "c3", "c1+c3", "c4")
CANDIDATE_FEATURE = "v110-m1-c1-c3-c4-ablation"
SCHEMA = "gpmeep-v110-m1-aunp-ablation-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-ablation-sample-complete-v1"


def _expected_feature_environment(mode: str) -> dict[str, str | None]:
    if mode not in MODES:
        raise base.common.WorkloadError(f"unknown M1 ablation mode: {mode!r}")
    c1_enabled = mode in {"c1", "c1+c3"}
    c3_enabled = mode in {"c3", "c1+c3"}
    c4_enabled = mode == "c4"
    omp_threads = c4.OMP_THREADS if c4_enabled else 1
    return {
        c3.C1_ENABLE: "1" if c1_enabled else None,
        c3.C1_DISABLE: None if c1_enabled else "1",
        c3.FEATURE_ENABLE: "1" if c3_enabled else None,
        c3.FEATURE_DISABLE: None if c3_enabled else "1",
        c4.FEATURE_ENABLE: "1" if c4_enabled else None,
        c4.FEATURE_DISABLE: None if c4_enabled else "1",
        "OMP_NUM_THREADS": str(omp_threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }


def _feature_environment(mode: str) -> dict[str, str | None]:
    expected = _expected_feature_environment(mode)
    observed = {name: base.os.environ.get(name) for name in expected}
    if observed != expected:
        raise base.common.WorkloadError(
            f"M1 ablation feature environment differs: {observed!r}"
        )
    return observed


def _summary_feature_environment_matches(summary: dict[str, Any]) -> bool:
    mode = summary.get("feature_mode")
    return mode in MODES and summary.get(
        "feature_environment"
    ) == _expected_feature_environment(mode)


def _reset_candidate_statistics(mp: Any) -> None:
    mp.gpu.reset_sparse_lorentzian_statistics()
    mp.reset_material_omp_statistics()


def _material_statistics(mp: Any) -> dict[str, Any]:
    return c4._candidate_statistics(mp)


def _candidate_statistics(mp: Any) -> dict[str, Any]:
    return {
        "sparse": c3._candidate_statistics(mp),
        "material": _material_statistics(mp),
    }


def _candidate_material_manifest(simulation: Any) -> dict[str, Any]:
    return c4._candidate_material_manifest(simulation)


def _component_records(
    rank_records: list[dict[str, Any]], component: str
) -> list[dict[str, Any]]:
    transformed = copy.deepcopy(rank_records)
    for rank_record in transformed:
        for record in rank_record.get("records", []):
            for suffix in ("before", "after"):
                name = f"candidate_statistics_{suffix}"
                value = record.get(name)
                if not isinstance(value, dict) or component not in value:
                    raise base.common.WorkloadError(
                        f"M1 {component} candidate statistics are absent"
                    )
                record[name] = value[component]
        final = rank_record.get("final_candidate_statistics")
        if not isinstance(final, dict) or component not in final:
            raise base.common.WorkloadError(
                f"M1 final {component} statistics are absent"
            )
        rank_record["final_candidate_statistics"] = final[component]
    return transformed


def _isolate_c3_ordinary_counters(
    rank_records: list[dict[str, Any]], c1_enabled: bool
) -> list[dict[str, Any]]:
    transformed = _component_records(rank_records, "sparse")
    if c1_enabled:
        for rank_record in transformed:
            for record in rank_record["records"]:
                polarization = record["statistics_delta"]["polarizations"]
                for name in base.FEATURE_COUNTERS:
                    polarization[name] = 0
    return transformed


def _validate_feature_counters(
    rank_records: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    if mode not in MODES:
        raise base.common.WorkloadError("M1 ablation mode is invalid")
    c1_enabled = mode in {"c1", "c1+c3"}
    c3_enabled = mode in {"c3", "c1+c3"}
    c4_enabled = mode == "c4"
    c1_totals = _base_validate_feature_counters(
        rank_records, "enabled" if c1_enabled else "disabled"
    )
    c3_totals = c3._validate_feature_counters(
        _isolate_c3_ordinary_counters(rank_records, c1_enabled),
        "enabled" if c3_enabled else "disabled",
    )
    c4_totals = c4._validate_feature_counters(
        _component_records(rank_records, "material"),
        "enabled" if c4_enabled else "disabled",
    )
    return {"c1": c1_totals, "c3": c3_totals, "c4": c4_totals}


base.SCHEMA = SCHEMA
base.COMPLETE_SCHEMA = COMPLETE_SCHEMA
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base.VALID_FEATURE_MODES = MODES
base._expected_feature_environment = _expected_feature_environment
base._feature_environment = _feature_environment
base._summary_feature_environment_matches = _summary_feature_environment_matches
base._reset_candidate_statistics = _reset_candidate_statistics
base._candidate_statistics = _candidate_statistics
base._candidate_material_manifest = _candidate_material_manifest
base._validate_feature_counters = _validate_feature_counters
base.__file__ = str(pathlib.Path(__file__).resolve())

validate_sample = base.validate_sample
parse_args = base.parse_args
main = base.main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.common.WorkloadError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"M1 AuNP ablation sample FAILED: {error}", file=base.sys.stderr)
        raise SystemExit(2)
