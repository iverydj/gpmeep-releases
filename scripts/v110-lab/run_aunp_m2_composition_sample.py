#!/usr/bin/env python3
"""Run one provenance-bound M2 C1/C3/C4 composition sample."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import run_aunp_m1_ablation_sample as m1


_BASE_PATH = pathlib.Path(m1.__file__).resolve()
_BASE_MODULE_NAME = "_gpmeep_v110_m2_composition_sample_base"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private M2 composition sample base")
base_module = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base_module
_BASE_SPEC.loader.exec_module(base_module)
base = base_module.base


MODES = (
    "off",
    "c1+c3",
    "c4_t2",
    "c4_t4",
    "c1+c3+c4_t2",
    "c1+c3+c4_t4",
)
CANDIDATE_FEATURE = "v110-m2-c1-c3-c4-composition"
SCHEMA = "gpmeep-v110-m2-aunp-composition-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m2-aunp-composition-sample-complete-v1"


def _mode_configuration(mode: str) -> tuple[bool, bool, bool, int]:
    if mode not in MODES:
        raise base.common.WorkloadError(
            f"unknown M2 composition mode: {mode!r}"
        )
    c1_enabled = mode.startswith("c1+c3")
    c3_enabled = c1_enabled
    c4_enabled = "c4_" in mode
    if mode.endswith("_t2"):
        omp_threads = 2
    elif mode.endswith("_t4"):
        omp_threads = 4
    else:
        omp_threads = 1
    if c4_enabled != (omp_threads in (2, 4)):
        raise base.common.WorkloadError(
            "M2 C4 mode and OpenMP thread count disagree"
        )
    return c1_enabled, c3_enabled, c4_enabled, omp_threads


def _expected_feature_environment(mode: str) -> dict[str, str | None]:
    c1_enabled, c3_enabled, c4_enabled, omp_threads = _mode_configuration(mode)
    return {
        m1.c3.C1_ENABLE: "1" if c1_enabled else None,
        m1.c3.C1_DISABLE: None if c1_enabled else "1",
        m1.c3.FEATURE_ENABLE: "1" if c3_enabled else None,
        m1.c3.FEATURE_DISABLE: None if c3_enabled else "1",
        m1.c4.FEATURE_ENABLE: "1" if c4_enabled else None,
        m1.c4.FEATURE_DISABLE: None if c4_enabled else "1",
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
            f"M2 composition feature environment differs: {observed!r}"
        )
    return observed


def _summary_feature_environment_matches(summary: dict[str, Any]) -> bool:
    mode = summary.get("feature_mode")
    return mode in MODES and summary.get(
        "feature_environment"
    ) == _expected_feature_environment(mode)


def _validate_feature_counters(
    rank_records: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    c1_enabled, c3_enabled, c4_enabled, omp_threads = _mode_configuration(mode)
    c1_totals = m1._base_validate_feature_counters(
        rank_records, "enabled" if c1_enabled else "disabled"
    )
    c3_totals = m1.c3._validate_feature_counters(
        m1._isolate_c3_ordinary_counters(rank_records, c1_enabled),
        "enabled" if c3_enabled else "disabled",
    )
    material_records = m1._component_records(rank_records, "material")
    c4_totals = m1.c4._validate_feature_counters(
        material_records,
        "enabled" if c4_enabled else "disabled",
    )
    expected_threads = omp_threads if c4_enabled else 0
    for rank_index, rank_record in enumerate(material_records):
        final = (
            rank_record.get("final_candidate_statistics")
            if isinstance(rank_record, dict)
            else None
        )
        if not isinstance(final, dict):
            raise base.common.WorkloadError(
                f"M2 C4 rank {rank_index} final statistics are absent"
            )
        for name in ("configured_threads", "observed_threads"):
            value = final.get(name)
            if type(value) is not int or value != expected_threads:
                raise base.common.WorkloadError(
                    f"M2 C4 rank {rank_index} {name} differs from requested "
                    "thread count"
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
base._reset_candidate_statistics = m1._reset_candidate_statistics
base._candidate_statistics = m1._candidate_statistics
base._candidate_material_manifest = m1._candidate_material_manifest
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
        print(f"M2 AuNP composition sample FAILED: {error}", file=base.sys.stderr)
        raise SystemExit(2)
