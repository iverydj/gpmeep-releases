#!/usr/bin/env python3
"""Run one provenance-bound C3 sparse-Lorentzian AuNP A/B sample.

The numerical workload, timing boundary, archive validation, and publication
transaction are inherited from the reviewed C2 worker.  This adapter replaces
only the candidate control and causal-counter contract.  C1 zero-state
elision is disabled identically in both lanes so the measured effect belongs
only to the resident sparse plan.
"""

from __future__ import annotations

import importlib.util
import pathlib
import struct
import sys
from typing import Any


_BASE_PATH = pathlib.Path(__file__).with_name("run_aunp_zero_state_sample.py")
_BASE_MODULE_NAME = "_gpmeep_c3_zero_state_sample_base"
_BASE_SPEC = importlib.util.spec_from_file_location(
    _BASE_MODULE_NAME, _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private C3 sample base module")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)


CANDIDATE_FEATURE = "resident-sparse-lorentzian-plan"
SCHEMA = "gpmeep-v110-m1-aunp-sparse-lorentzian-sample-v1"
COMPLETE_SCHEMA = (
    "gpmeep-v110-m1-aunp-sparse-lorentzian-sample-complete-v1"
)
FEATURE_ENABLE = "MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN"
FEATURE_DISABLE = "MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN"
C1_ENABLE = "MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION"
C1_DISABLE = "MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION"

COUNTERS = (
    "opt_in_calls",
    "opt_in_points",
    "eligible_calls",
    "eligible_points",
    "ineligible_calls",
    "ineligible_points",
    "first_observation_dense_calls",
    "first_observation_dense_points",
    "plan_build_attempts",
    "plan_build_successes",
    "plan_build_failures",
    "plan_hits",
    "plan_invalidations",
    "dense_locked_decisions",
    "accepted_total_points",
    "accepted_active_points",
    "sparse_submission_calls",
    "sparse_submission_points",
    "guard_scan_submissions",
    "guard_scan_submission_points",
    "cpu_fallback_calls",
    "cpu_fallback_points",
    "device_guard_scan_calls",
    "device_guard_scan_points",
    "device_sparse_update_calls",
    "device_sparse_update_points",
    "device_dense_fallback_calls",
    "device_dense_fallback_points",
)
GAUGES = ("live_plans", "live_guards", "retained_index_bytes")
FLAGS = ("counter_overflow", "teardown_incomplete")


def _expected_feature_environment(mode: str) -> dict[str, str | None]:
    return {
        FEATURE_ENABLE: "1" if mode == "enabled" else None,
        FEATURE_DISABLE: None if mode == "enabled" else "1",
        C1_ENABLE: None,
        C1_DISABLE: "1",
    }


def _feature_environment(mode: str) -> dict[str, str | None]:
    expected = _expected_feature_environment(mode)
    observed = {name: base.os.environ.get(name) for name in expected}
    if observed != expected:
        raise base.common.WorkloadError(
            f"C3 sparse feature environment differs: {observed!r}"
        )
    return observed


def _reset_candidate_statistics(mp: Any) -> None:
    mp.gpu.reset_sparse_lorentzian_statistics()


def _candidate_statistics(mp: Any) -> dict[str, Any]:
    return mp.gpu.sparse_lorentzian_statistics()


def _counter(group: Any, name: str) -> int:
    if not isinstance(group, dict):
        raise base.common.WorkloadError(
            "C3 sparse statistics group is absent"
        )
    value = group.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise base.common.WorkloadError(
            f"invalid C3 sparse counter {name}"
        )
    return value


def _delta(before: Any, after: Any) -> dict[str, int]:
    required = set(COUNTERS + GAUGES + FLAGS + ("reset_generation",))
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != required
        or set(after) != required
    ):
        raise base.common.WorkloadError(
            "C3 sparse statistics schema differs"
        )
    if (
        _counter(before, "reset_generation")
        != _counter(after, "reset_generation")
        or _counter(before, "reset_generation") == 0
    ):
        raise base.common.WorkloadError(
            "C3 sparse reset generation changed during one sample"
        )
    result: dict[str, int] = {}
    for name in COUNTERS:
        old = _counter(before, name)
        new = _counter(after, name)
        if new < old:
            raise base.common.WorkloadError(
                f"C3 sparse counter decreased: {name}"
            )
        result[name] = new - old
    for name in GAUGES + FLAGS:
        result[name] = _counter(after, name)
    return result


def _validate_feature_counters(
    rank_records: list[dict[str, Any]], mode: str
) -> dict[str, int]:
    aggregate = {
        "cuda_polarization_calls": 0,
        "cuda_polarization_points": 0,
        **{name: 0 for name in COUNTERS},
    }
    for rank_record in rank_records:
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != 2:
            raise base.common.WorkloadError(
                "C3 sparse sample phase inventory differs"
            )
        required = set(COUNTERS + GAUGES + FLAGS + ("reset_generation",))
        baseline = records[0].get("candidate_statistics_before")
        reference_after = records[0].get("candidate_statistics_after")
        structure_before = records[1].get("candidate_statistics_before")
        if (
            not isinstance(baseline, dict)
            or set(baseline) != required
            or _counter(baseline, "reset_generation") == 0
            or any(
                _counter(baseline, name)
                for name in COUNTERS + GAUGES + FLAGS
            )
            or reference_after != baseline
            or structure_before != baseline
        ):
            raise base.common.WorkloadError(
                "C3 sparse reset baseline or inter-phase continuity differs"
            )
        for run_index, record in enumerate(records):
            ordinary = record.get("statistics_delta")
            if not isinstance(ordinary, dict):
                raise base.common.WorkloadError(
                    "C3 ordinary statistics delta is absent"
                )
            polarization = ordinary.get("polarizations")
            if any(
                base._counter_value(polarization, name)
                for name in base.FEATURE_COUNTERS
            ):
                raise base.common.WorkloadError(
                    "C1 zero-state elision was not isolated from C3"
                )
            calls = base._counter_value(
                polarization, "cuda_polarization_calls"
            )
            points = base._counter_value(
                polarization, "cuda_polarization_points"
            )
            sparse = _delta(
                record.get("candidate_statistics_before"),
                record.get("candidate_statistics_after"),
            )
            if sparse["counter_overflow"] or sparse["teardown_incomplete"]:
                raise base.common.WorkloadError(
                    "C3 sparse sample reported overflow or incomplete teardown"
                )
            if run_index == 0:
                if calls or points or any(sparse[name] for name in COUNTERS):
                    raise base.common.WorkloadError(
                        "C3 reference phase unexpectedly updated polarization"
                    )
                if any(sparse[name] for name in GAUGES):
                    raise base.common.WorkloadError(
                        "C3 reference phase retained a sparse plan"
                    )
                continue
            if calls == 0:
                if points or any(sparse[name] for name in COUNTERS + GAUGES):
                    raise base.common.WorkloadError(
                        "zero-work C3 rank reported sparse work"
                    )
                continue
            if points <= 0:
                raise base.common.WorkloadError(
                    "C3 structure phase reported calls without points"
                )
            if mode == "disabled":
                if any(sparse[name] for name in COUNTERS + GAUGES):
                    raise base.common.WorkloadError(
                        "disabled C3 sample reported sparse activity"
                    )
            else:
                if (
                    sparse["opt_in_calls"] != calls
                    or sparse["opt_in_points"] != points
                    or sparse["eligible_calls"] != calls
                    or sparse["eligible_points"] != points
                    or sparse["eligible_points"]
                    < sparse["eligible_calls"] * 65536
                    or sparse["ineligible_calls"]
                    or sparse["ineligible_points"]
                    or sparse["first_observation_dense_calls"] <= 0
                    or sparse["plan_build_attempts"] <= 0
                    or sparse["first_observation_dense_calls"]
                    < sparse["plan_build_attempts"]
                    or sparse["first_observation_dense_points"]
                    < sparse["first_observation_dense_calls"] * 65536
                    or sparse["first_observation_dense_points"]
                    > sparse["eligible_points"]
                    or sparse["first_observation_dense_points"]
                    < sparse["accepted_total_points"]
                    or sparse["plan_build_successes"] <= 0
                    or sparse["plan_build_attempts"]
                    != sparse["plan_build_successes"]
                    + sparse["dense_locked_decisions"]
                    or sparse["plan_build_failures"]
                    or sparse["plan_hits"] <= 0
                    or sparse["accepted_total_points"] <= 0
                    or sparse["accepted_active_points"] <= 0
                    or sparse["accepted_total_points"]
                    < sparse["plan_build_successes"] * 65536
                    or sparse["accepted_active_points"]
                    > sparse["accepted_total_points"]
                    or sparse["accepted_active_points"] * 100
                    > sparse["accepted_total_points"] * 35
                    or sparse["sparse_submission_calls"] <= 0
                    or sparse["sparse_submission_calls"]
                    != sparse["plan_build_successes"] + sparse["plan_hits"]
                    or sparse["eligible_calls"]
                    < sparse["first_observation_dense_calls"]
                    + sparse["sparse_submission_calls"]
                    + sparse["dense_locked_decisions"]
                    or sparse["sparse_submission_points"] <= 0
                    or sparse["sparse_submission_points"] * 100
                    > sparse["eligible_points"] * 35
                    or sparse["device_sparse_update_calls"]
                    != sparse["sparse_submission_calls"]
                    or sparse["device_sparse_update_points"]
                    != sparse["sparse_submission_points"]
                    or sparse["guard_scan_submissions"] <= 0
                    or sparse["guard_scan_submission_points"] <= 0
                    or sparse["guard_scan_submission_points"]
                    < sparse["guard_scan_submissions"] * 65536
                    or sparse["guard_scan_submission_points"]
                    > sparse["eligible_points"]
                    or sparse["guard_scan_submissions"]
                    > sparse["sparse_submission_calls"]
                    or sparse["device_guard_scan_calls"]
                    != sparse["guard_scan_submissions"]
                    or sparse["device_guard_scan_points"]
                    != sparse["guard_scan_submission_points"]
                    or sparse["device_dense_fallback_calls"]
                    or sparse["device_dense_fallback_points"]
                    or sparse["cpu_fallback_calls"]
                    or sparse["cpu_fallback_points"]
                    or sparse["live_plans"] <= 0
                    or sparse["live_plans"]
                    > sparse["plan_build_successes"]
                    or sparse["live_plans"]
                    + sparse["plan_invalidations"]
                    > sparse["first_observation_dense_calls"]
                    or sparse["live_guards"] <= 0
                    or sparse["live_guards"] > sparse["live_plans"]
                    or sparse["retained_index_bytes"] <= 0
                    or sparse["retained_index_bytes"]
                    % struct.calcsize("n")
                    or sparse["retained_index_bytes"]
                    > sparse["accepted_active_points"]
                    * struct.calcsize("n")
                ):
                    raise base.common.WorkloadError(
                        "enabled C3 structure phase lacks exact sparse-plan "
                        f"causality: {sparse!r}"
                    )
            aggregate["cuda_polarization_calls"] += calls
            aggregate["cuda_polarization_points"] += points
            for name in COUNTERS:
                aggregate[name] += sparse[name]

        final_sparse = rank_record.get("final_candidate_statistics")
        structure_after = records[1].get("candidate_statistics_after")
        if (
            not isinstance(final_sparse, dict)
            or set(final_sparse) != required
            or not isinstance(structure_after, dict)
            or set(structure_after) != required
            or any(
                _counter(final_sparse, name)
                != _counter(structure_after, name)
                for name in COUNTERS + ("reset_generation",)
            )
            or any(
            _counter(final_sparse, name) for name in GAUGES + FLAGS
            )
        ):
            raise base.common.WorkloadError(
                "C3 sample teardown changed execution totals or retained "
                "sparse state"
            )

    if (
        aggregate["cuda_polarization_calls"] <= 0
        or aggregate["cuda_polarization_points"] <= 0
    ):
        raise base.common.WorkloadError(
            "C3 sample has no aggregate CUDA polarization work"
        )
    if mode == "enabled":
        if (
            aggregate["opt_in_calls"]
            != aggregate["cuda_polarization_calls"]
            or aggregate["opt_in_points"]
            != aggregate["cuda_polarization_points"]
            or aggregate["device_sparse_update_calls"] <= 0
            or aggregate["cpu_fallback_calls"]
        ):
            raise base.common.WorkloadError(
                "enabled C3 aggregate counters differ from CUDA work"
            )
    elif any(aggregate[name] for name in COUNTERS):
        raise base.common.WorkloadError(
            "disabled C3 aggregate reported sparse work"
        )
    return aggregate


base.SCHEMA = SCHEMA
base.COMPLETE_SCHEMA = COMPLETE_SCHEMA
base.FEATURE_ENABLE = FEATURE_ENABLE
base.FEATURE_DISABLE = FEATURE_DISABLE
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base._feature_environment = _feature_environment
base._expected_feature_environment = _expected_feature_environment
base._reset_candidate_statistics = _reset_candidate_statistics
base._candidate_statistics = _candidate_statistics
base._validate_feature_counters = _validate_feature_counters
# The inherited publication record must bind this executable adapter, not the
# imported implementation module.
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
        print(f"C3 AuNP sample FAILED: {error}", file=base.sys.stderr, flush=True)
        raise SystemExit(2)
