#!/usr/bin/env python3
"""Pure census and timing replay for the gpmeep v1.1.0 M4 matrix.

The executable controller owns files and process evidence.  This module owns the
meaning of the 180 planned observations: it rejects missing, duplicated, or
misidentified samples and derives paired, same-cycle timing ratios without
turning exploratory performance observations into release pass/fail gates.
"""

from __future__ import annotations

import math
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m4_equal_work_plan as equal_work_plan  # noqa: E402


SCHEMA = "gpmeep-v110-m4-equal-work-aggregation-v2"
TASK_COUNT = 180
MEASURED_TASK_COUNT = 120


class M4ReplayError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M4ReplayError(message)


def task_name(task: dict[str, Any]) -> str:
    """Return a stable filesystem-safe name for a sealed task."""
    _require(type(task.get("index")) is int, "M4 task index differs")
    task_id = task.get("task_id")
    _require(
        isinstance(task_id, str)
        and len(task_id) == 64
        and all(character in "0123456789abcdef" for character in task_id),
        "M4 task digest differs",
    )
    return f"{task['index'] + 1:03d}-{task_id[:16]}"


def _positive_seconds(value: Any) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0,
        "M4 primary seconds must be positive and finite",
    )
    return float(value)


def validate_sample_report(
    task: dict[str, Any], report: Any, timing: Any | None = None
) -> dict[str, Any]:
    """Validate the cross-runtime fields common to all three sample schemas."""
    _require(isinstance(report, dict), "M4 child sample report is not an object")
    expected = {
        "case_path": task["case_path"],
        "topology": task["topology"],
        "sample_kind": task["sample_kind"],
        "cycle_index": task["cycle_index"],
    }
    _require(
        report.get("outcome") == "PASS"
        and all(report.get(name) == value for name, value in expected.items()),
        "M4 child sample identity differs",
    )
    derived_timing = report.get("timing")
    _require(
        isinstance(derived_timing, dict)
        and derived_timing.get("contract") == task["timing_contract"],
        "M4 child timing contract differs",
    )
    _positive_seconds(derived_timing.get("primary_seconds"))
    phase_profile = report.get("phase_profile")
    _require(
        isinstance(phase_profile, dict)
        and phase_profile.get("schema") == "gpmeep-m3-ten-phase-profile-v1"
        and isinstance(phase_profile.get("phases"), dict)
        and len(phase_profile["phases"]) == 10,
        "M4 child phase profile differs",
    )
    for phase in phase_profile["phases"].values():
        _require(
            isinstance(phase, dict)
            and type(phase.get("available")) is bool
            and (
                phase.get("seconds") is None
                if not phase["available"]
                else not isinstance(phase.get("seconds"), bool)
                and isinstance(phase.get("seconds"), (int, float))
                and math.isfinite(float(phase["seconds"]))
                and float(phase["seconds"]) >= 0.0
            ),
            "M4 child phase value differs",
        )
    if timing is not None:
        _require(timing == derived_timing, "M4 retained timing was not re-derived")
    return report


def _report_map(
    plan: dict[str, Any], reports: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tasks = equal_work_plan.build_tasks(plan)
    _require(
        len(tasks) == TASK_COUNT
        and sum(task["sample_kind"] == "measured" for task in tasks)
        == MEASURED_TASK_COUNT,
        "M4 plan census differs",
    )
    _require(len(reports) == TASK_COUNT, "M4 sample report census differs")
    by_id: dict[str, dict[str, Any]] = {}
    for task, report in zip(tasks, reports, strict=True):
        validate_sample_report(task, report)
        task_id = task["task_id"]
        _require(task_id not in by_id, "M4 sample task ID repeats")
        by_id[task_id] = report
    return tasks, by_id


def _paired_ratio(
    reference: list[float], candidate: list[float]
) -> dict[str, Any]:
    _require(
        len(reference) == len(candidate) == 2,
        "M4 measured timing pair is incomplete",
    )
    ratios = [left / right for left, right in zip(reference, candidate, strict=True)]
    _require(
        all(math.isfinite(value) and value > 0.0 for value in ratios),
        "M4 timing ratio differs",
    )
    return {
        "reference_seconds": reference,
        "candidate_seconds": candidate,
        "reference_median_seconds": statistics.median(reference),
        "candidate_median_seconds": statistics.median(candidate),
        "reference_average_seconds": math.fsum(reference) / 2.0,
        "candidate_average_seconds": math.fsum(candidate) / 2.0,
        "paired_ratios": ratios,
        "average_of_paired_ratios": math.fsum(ratios) / 2.0,
        "median_of_paired_ratios": statistics.median(ratios),
        "ratio_of_average_seconds": (math.fsum(reference) / 2.0)
        / (math.fsum(candidate) / 2.0),
        "ratio_of_medians": statistics.median(reference)
        / statistics.median(candidate),
    }


def derive_results(
    plan: dict[str, Any], reports: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate exact raw seconds and explicit GPU1/GPU2 paired ratios."""
    tasks, by_id = _report_map(plan, reports)
    cases = [item["path"] for item in plan["cases"]]
    by_observation: dict[tuple[str, str, int], float] = {}
    by_phase: dict[tuple[str, str, int], dict[str, float | None]] = {}
    raw: list[dict[str, Any]] = []
    for task in tasks:
        seconds = _positive_seconds(
            by_id[task["task_id"]]["timing"]["primary_seconds"]
        )
        key = (task["case_path"], task["lane_id"], task["cycle_index"])
        _require(key not in by_observation, "M4 observation identity repeats")
        by_observation[key] = seconds
        phase_values = {
            name: (float(value["seconds"]) if value["available"] else None)
            for name, value in by_id[task["task_id"]]["phase_profile"]["phases"].items()
        }
        by_phase[key] = phase_values
        raw.append(
            {
                "task_id": task["task_id"],
                "case_path": task["case_path"],
                "lane_id": task["lane_id"],
                "cycle_index": task["cycle_index"],
                "sample_kind": task["sample_kind"],
                "primary_seconds": seconds,
                "phase_seconds": phase_values,
            }
        )

    case_results: dict[str, Any] = {}
    comparison_specs = (
        ("upstream-cpu_to_v103-gpu1", "upstream_meep_cpu", "gpmeep_v103_cuda_gpu1"),
        ("upstream-cpu_to_v110-gpu1", "upstream_meep_cpu", "gpmeep_v110_cuda_gpu1"),
        ("v110-cpu_to_v103-gpu1", "gpmeep_v110_cpu", "gpmeep_v103_cuda_gpu1"),
        ("v110-cpu_to_v110-gpu1", "gpmeep_v110_cpu", "gpmeep_v110_cuda_gpu1"),
        ("v103-gpu1_to_v103-gpu2", "gpmeep_v103_cuda_gpu1", "gpmeep_v103_cuda_gpu2"),
        ("v110-gpu1_to_v110-gpu2", "gpmeep_v110_cuda_gpu1", "gpmeep_v110_cuda_gpu2"),
        ("v103-gpu1_to_v110-gpu1", "gpmeep_v103_cuda_gpu1", "gpmeep_v110_cuda_gpu1"),
        ("v103-gpu2_to_v110-gpu2", "gpmeep_v103_cuda_gpu2", "gpmeep_v110_cuda_gpu2"),
    )
    for case_path in cases:
        lane_timings: dict[str, Any] = {}
        applicable = {
            task["lane_id"] for task in tasks if task["case_path"] == case_path
        }
        for lane_id in equal_work_plan.EXPECTED_LANES:
            if lane_id not in applicable:
                continue
            values = [by_observation[(case_path, lane_id, cycle)] for cycle in (0, 1, 2)]
            phase_names = tuple(by_phase[(case_path, lane_id, 0)])
            _require(
                all(tuple(by_phase[(case_path, lane_id, cycle)]) == phase_names for cycle in (1, 2)),
                "M4 phase inventory changed across cycles",
            )
            phases = {}
            for phase_name in phase_names:
                phase_values = [
                    by_phase[(case_path, lane_id, cycle)][phase_name]
                    for cycle in (0, 1, 2)
                ]
                measured = phase_values[1:]
                phases[phase_name] = {
                    "warmup_seconds": phase_values[0],
                    "measured_seconds": measured,
                    "measured_average_seconds": (
                        math.fsum(measured) / 2.0
                        if all(value is not None for value in measured)
                        else None
                    ),
                }
            lane_timings[lane_id] = {
                "warmup_seconds": values[0],
                "measured_seconds": values[1:],
                "measured_average_seconds": math.fsum(values[1:]) / 2.0,
                "measured_median_seconds": statistics.median(values[1:]),
                "phases": phases,
            }
        ratios = {
            name: _paired_ratio(
                lane_timings[reference]["measured_seconds"],
                lane_timings[candidate]["measured_seconds"],
            )
            for name, reference, candidate in comparison_specs
            if reference in lane_timings and candidate in lane_timings
        }
        case_results[case_path] = {
            "lanes": lane_timings,
            "ratios": ratios,
        }

    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "task_count": len(tasks),
        "measured_task_count": sum(
            task["sample_kind"] == "measured" for task in tasks
        ),
        "raw_seconds": raw,
        "cases": case_results,
        "ratio_semantics": (
            "reference_seconds/candidate_seconds; values above one mean the "
            "candidate lane is faster"
        ),
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ratio_record(
    lanes: dict[str, Any], numerator: str, denominator: str
) -> dict[str, Any]:
    left = lanes[numerator]
    right = lanes[denominator]
    paired = [
        first / second
        for first, second in zip(
            left["measured_seconds"], right["measured_seconds"], strict=True
        )
    ]
    return {
        "numerator_lane": numerator,
        "denominator_lane": denominator,
        "paired_cycle_values": paired,
        "ratio_of_means": left["mean_seconds"] / right["mean_seconds"],
    }


def _ratio_set(lanes: dict[str, Any], suffix: str) -> dict[str, Any]:
    v103 = f"gpmeep_v103_cuda_{suffix}"
    v110 = f"gpmeep_v110_cuda_{suffix}"
    return {
        "upstream_cpu_over_v103_cuda": _ratio_record(
            lanes, "upstream_meep_cpu", v103
        ),
        "upstream_cpu_over_v110_cuda": _ratio_record(
            lanes, "upstream_meep_cpu", v110
        ),
        "v103_cuda_over_v110_cuda": _ratio_record(lanes, v103, v110),
        "v110_cpu_over_v110_cuda": _ratio_record(
            lanes, "gpmeep_v110_cpu", v110
        ),
    }


def derive_inversion_records(
    plan: dict[str, Any],
    reports: list[dict[str, Any]],
    sample_evidence: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
    inventory_path: str,
    inventory_sha256: str,
    numerical_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce the exact workload-level v2 shape consumed by the registry."""
    tasks, by_id = _report_map(plan, reports)
    _require(
        set(sample_evidence) == {task["task_id"] for task in tasks},
        "M4 sample evidence inventory differs",
    )
    observations = inventory.get("observations")
    campaign = inventory.get("source_campaign")
    _require(
        isinstance(observations, list)
        and isinstance(campaign, dict)
        and isinstance(campaign.get("id"), str)
        and len(inventory_sha256) == 64,
        "M4 inversion inventory differs",
    )
    by_workload = {
        item.get("workload_id"): item for item in observations if isinstance(item, dict)
    }
    _require(
        set(by_workload) == {case["path"] for case in plan["cases"]},
        "M4 inversion workload inventory differs",
    )
    runtime_names = {
        "upstream_meep_cpu": "upstream_meep",
        "gpmeep_v110_cpu": "gpmeep_v110",
        "gpmeep_v103_cuda_gpu1": "gpmeep_v103",
        "gpmeep_v110_cuda_gpu1": "gpmeep_v110",
        "gpmeep_v103_cuda_gpu2": "gpmeep_v103",
        "gpmeep_v110_cuda_gpu2": "gpmeep_v110",
    }
    sample_kinds = {
        "upstream": "upstream_sample",
        "v103": "v103_sample",
        "v110": "v110_sample",
    }
    records = []
    for case in plan["cases"]:
        case_path = case["path"]
        observation = by_workload[case_path]
        lane_records: dict[str, Any] = {}
        phase_by_lane: dict[str, Any] = {}
        for lane_id in equal_work_plan.EXPECTED_LANES:
            matching = [
                task
                for task in tasks
                if task["case_path"] == case_path
                and task["lane_id"] == lane_id
                and task["sample_kind"] == "measured"
            ]
            if not matching:
                lane_records[lane_id] = None
                phase_by_lane[lane_id] = None
                continue
            matching.sort(key=lambda task: task["cycle_index"])
            _require(
                [task["cycle_index"] for task in matching] == [1, 2],
                "M4 measured cycle inventory differs",
            )
            lane_reports = [by_id[task["task_id"]] for task in matching]
            seconds = [
                _positive_seconds(report["timing"]["primary_seconds"])
                for report in lane_reports
            ]
            devices = lane_reports[0]["lane"]["spec"]["devices"]
            _require(
                isinstance(devices, list)
                and all(report["lane"]["spec"]["devices"] == devices for report in lane_reports),
                "M4 measured lane device identity changed",
            )
            topology = (
                "cpu1"
                if matching[0]["topology"].endswith("cpu1")
                else "cpu8"
                if matching[0]["topology"].endswith("cpu8")
                else "gpu2"
                if matching[0]["topology"].endswith("cuda2")
                else "gpu1"
            )
            lane_records[lane_id] = {
                "runtime_id": lane_reports[0]["receipt_id"],
                "runtime": runtime_names[lane_id],
                "backend": matching[0]["backend"],
                "topology": topology,
                "devices": devices,
                "measured_seconds": seconds,
                "mean_seconds": math.fsum(seconds) / 2.0,
                "samples": [
                    {
                        "kind": sample_kinds[task["runtime"]],
                        "schema": report["schema"],
                        "cycle_index": task["cycle_index"],
                        **sample_evidence[task["task_id"]],
                    }
                    for task, report in zip(matching, lane_reports, strict=True)
                ],
            }
            phase_names = tuple(lane_reports[0]["phase_profile"]["phases"])
            phase_set = {}
            for phase_name in phase_names:
                raw_phases = [
                    report["phase_profile"]["phases"][phase_name]
                    for report in lane_reports
                ]
                available = all(item["available"] is True for item in raw_phases)
                common = {
                    "available": available,
                    "measured_seconds": (
                        [float(item["seconds"]) for item in raw_phases]
                        if available
                        else [None, None]
                    ),
                    "mean_seconds": (
                        math.fsum(float(item["seconds"]) for item in raw_phases) / 2.0
                        if available
                        else None
                    ),
                    "basis": [item["basis"] for item in raw_phases],
                    "native_sinks": [item["native_sinks"] for item in raw_phases],
                    "cycle_indices": [1, 2],
                }
                if not available:
                    common["reason_code"] = (
                        "initialization_not_separable"
                        if phase_name == "initialization"
                        else "phase_not_exercised_in_both_samples"
                        if any(item["native_sinks"] for item in raw_phases)
                        else "no_comparable_native_timer"
                    )
                phase_set[phase_name] = common
            phase_by_lane[lane_id] = phase_set
        _require(
            all(lane_records[name] is not None for name in equal_work_plan.EXPECTED_LANES[:4])
            and (case["gpu2"] is True)
            == all(lane_records[name] is not None for name in equal_work_plan.EXPECTED_LANES[4:]),
            "M4 lane applicability differs",
        )
        representative = observation.get("representative") is True
        impact = observation.get("impact")
        records.append(
            {
                "schema_version": 2,
                "record_id": "m4-" + observation["workload_name_sha256"][:24],
                "workload_id": case_path,
                "observed_gpu_slower_than_cpu": True,
                "source_inventory": {
                    "campaign_id": campaign["id"],
                    "inventory_path": inventory_path,
                    "inventory_sha256": inventory_sha256,
                    "case_digest": observation["workload_identity"]["case_contract_sha256"],
                },
                "representative": {
                    "value": representative,
                    "impact": impact,
                    "selection_policy_version": "v110-m0-1",
                    "rationale": observation["rationale"],
                },
                "comparison": {
                    "physics_fingerprint": _sha256_json(
                        {"source_policy": plan["source_policy"], "case": case}
                    ),
                    "precision": "fp32",
                    "stopping_condition": "exact-entrypoint-source-bytes",
                    "cpu_topology": "physical-cores",
                    "lanes": lane_records,
                    "ratios": {
                        "gpu1": _ratio_set(lane_records, "gpu1"),
                        "gpu2": _ratio_set(lane_records, "gpu2") if case["gpu2"] else None,
                    },
                },
                "phase_decomposition": phase_by_lane,
                "primary_class": "unknown",
                "classification_evidence": [
                    {"kind": "numerical_regression", **numerical_evidence}
                ],
                "confidence": 0.0,
                "owner": "gpmeep-v1.1.0-m4",
                "resolution": {
                    "state": "open",
                    "same_binary_ablation": None,
                    "numerical_regression": None,
                    "performance_claim_eligible": True,
                    "notes": "Equal-work evidence captured; classification remains open pending M4 analysis.",
                },
                "release_blocking": representative and impact == "high",
            }
        )
    return records


__all__ = [
    "M4ReplayError",
    "SCHEMA",
    "derive_results",
    "derive_inversion_records",
    "task_name",
    "validate_sample_report",
]
