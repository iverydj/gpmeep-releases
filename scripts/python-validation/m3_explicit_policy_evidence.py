#!/usr/bin/env python3
"""Derive feature-specific v1.1 explicit-backend policy evidence.

This is the authoritative replacement for the historical M3 automatic-backend
classification path.  It replays already sealed paired-example and AuNP
outputs, validates every reason-bound guard event, and proves both directions:
CPU-recommended/CUDA-selected and CUDA-recommended/CPU-selected.  It never
selects a backend and never treats timing from the official examples as a
release speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import archive_validation  # noqa: E402
import gpmeep_backend_guard as backend_guard  # noqa: E402
import run_example_oracle as oracle  # noqa: E402
import run_m3_paired_matrix as paired_matrix  # noqa: E402
import run_aunp_performance_workload as aunp_performance  # noqa: E402
import run_hybrid_aunp_matrix as aunp_matrix  # noqa: E402
from common import (  # noqa: E402
    AUNP_BACKEND_ACKNOWLEDGEMENT_REASON,
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-v110-explicit-policy-evidence-v2"
COMPLETE_SCHEMA = "gpmeep-m3-v110-explicit-policy-evidence-complete-v2"
CPU_RECOMMENDATION_CASES = (
    "python/examples/antenna_pec_ground_plane_1D.py",
    "python/examples/binary_grating_phasemap.py",
    "python/examples/planar_cavity_ldos.py",
)
CUDA_RECOMMENDATION_CASES = (
    "python/examples/grating2d_triangular_lattice.py",
)
LONG_HORIZON_CASES = (
    "python/examples/stochastic_emitter_line.py",
    "python/examples/cherenkov-radiation.py",
)
EXPECTED_FEATURES = (
    "policy.recommend_cpu",
    "policy.recommend_cuda",
    "policy.long_horizon_small",
)
DIRECTIONS = (
    "cpu_selected__cpu_recommended",
    "cpu_selected__cuda_recommended",
    "cuda_selected__cpu_recommended",
    "cuda_selected__cuda_recommended",
)
EVENT_VIEW_KEYS = {
    "event_index",
    "selected_backend",
    "recommended_backend",
    "direction",
    "reason_class",
    "recommendation_confidence",
    "mismatch_observed",
    "attempts",
    "acknowledgements",
    "acknowledgement_reused",
    "acknowledgement_scope",
    "acknowledgement_origin_cohort_digest",
    "acknowledgement_allowed",
    "acknowledgement_reason",
    "warning_present",
    "final_acknowledgement",
    "acknowledgement_scope_digest",
    "workload_fingerprint_evaluated",
    "workload_fingerprint",
    "recommendation_policy_version",
}
LANE_SUMMARY_KEYS = {
    "source",
    "selected_backend",
    "classification_unit",
    "case_level_recommendation",
    "event_count",
    "attempt_count",
    "acknowledgement_count",
    "mutation_before_acknowledgement",
    "events_sha256",
    "event_views_sha256",
    "events",
    "direction_counts",
    "reason_class_counts",
    "execution_wall_seconds",
    "warning_wall_seconds",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _safe_member(root: pathlib.Path, relative: Any, label: str) -> pathlib.Path:
    if not isinstance(relative, str) or not relative:
        raise WorkloadError(f"{label} path is absent")
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise WorkloadError(f"{label} path is unsafe")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    if root.resolve(strict=True) not in resolved.parents or not resolved.is_file():
        raise WorkloadError(f"{label} escaped its evidence root")
    return resolved


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > 512 * 1024**2:
            raise WorkloadError(f"{label} exceeds its size bound")
        with stable.file_object() as handle:
            try:
                value = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise WorkloadError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _oracle_events(stdout: pathlib.Path) -> list[dict[str, Any]]:
    prefix = oracle.AUDIT_PREFIX.encode("utf-8")
    auxiliary_prefix = oracle.AUXILIARY_GUARD_PREFIX.encode("utf-8")
    with StableFile(stdout, "explicit-policy oracle stdout") as stable:
        if stable.initial_stat.st_size > 256 * 1024**2:
            raise WorkloadError("explicit-policy oracle stdout exceeds its bound")
        with stable.file_object() as handle:
            retained_lines = list(handle)
    lines = [line for line in retained_lines if line.startswith(prefix)]
    if len(lines) != 1:
        raise WorkloadError("explicit-policy oracle audit inventory differs")
    try:
        audit = json.loads(lines[0][len(prefix) :])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkloadError("explicit-policy oracle audit is invalid") from exc
    runs = audit.get("runs") if isinstance(audit, dict) else None
    if not isinstance(runs, list) or not runs:
        raise WorkloadError("explicit-policy oracle emitted no run evidence")
    events: list[Any] = []
    auxiliary_lines = [
        line for line in retained_lines if line.startswith(auxiliary_prefix)
    ]
    if len(auxiliary_lines) > 1:
        raise WorkloadError("explicit-policy auxiliary guard inventory differs")
    if auxiliary_lines:
        try:
            auxiliary = json.loads(auxiliary_lines[0][len(auxiliary_prefix) :])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkloadError(
                "explicit-policy auxiliary guard audit is invalid"
            ) from exc
        auxiliary_events = (
            auxiliary.get("events") if isinstance(auxiliary, dict) else None
        )
        if not isinstance(auxiliary_events, list):
            raise WorkloadError("explicit-policy auxiliary guard inventory differs")
        for item in auxiliary_events:
            if (
                not isinstance(item, dict)
                or set(item) != {"entrypoint", "guard"}
                or item["entrypoint"] not in oracle.GUARDED_ENTRYPOINTS
                or not isinstance(item["guard"], dict)
            ):
                raise WorkloadError("explicit-policy auxiliary guard event differs")
            events.append(item["guard"])
    events.extend(record.get("explicit_backend_guard") for record in runs)
    if any(not isinstance(event, dict) for event in events):
        raise WorkloadError("explicit-policy oracle guard evidence is absent")
    return events


def _runner_lane_events(
    runner_root: pathlib.Path, run: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    backend = run.get("backend")
    if backend not in {"cpu", "cuda"}:
        raise WorkloadError("explicit-policy runner backend differs")
    stats = run.get("statistics")
    guard = stats.get("backend_guard") if isinstance(stats, dict) else None
    if (
        isinstance(guard, dict)
        and guard.get("mode") == "reason-bound-policy-scope"
    ):
        events = guard.get("events")
        source = "sitecustomize"
    else:
        stdout = _safe_member(
            runner_root, run.get("stdout_log"), "explicit-policy stdout"
        )
        events = _oracle_events(stdout)
        source = "generic-oracle"
    if not isinstance(events, list) or not events:
        raise WorkloadError("explicit-policy runner guard event inventory differs")
    return source, events


def _event_view(event: dict[str, Any], index: int) -> dict[str, Any]:
    final = event["final_telemetry"]
    direction = (
        f"{event['selected_backend']}_selected__"
        f"{event['recommended_backend']}_recommended"
    )
    return {
        "event_index": index,
        "selected_backend": event["selected_backend"],
        "recommended_backend": event["recommended_backend"],
        "direction": direction,
        "reason_class": final["reason_class"],
        "recommendation_confidence": final["recommendation_confidence"],
        "mismatch_observed": event["mismatch_observed"],
        "attempts": event["attempts"],
        "acknowledgements": event["acknowledgements"],
        "acknowledgement_reused": event["acknowledgement_reused"],
        "acknowledgement_scope": final["acknowledgement_scope"],
        "acknowledgement_origin_cohort_digest": final[
            "acknowledgement_origin_cohort_digest"
        ],
        "acknowledgement_allowed": final["acknowledgement_allowed"],
        "acknowledgement_reason": final["acknowledgement_reason"],
        "warning_present": event["warning_telemetry"] is not None,
        "final_acknowledgement": final["acknowledgement"],
        "acknowledgement_scope_digest": final[
            "acknowledgement_scope_digest"
        ],
        "workload_fingerprint_evaluated": final[
            "workload_fingerprint_evaluated"
        ],
        "workload_fingerprint": final["workload_fingerprint"],
        "recommendation_policy_version": final[
            "recommendation_policy_version"
        ],
    }


def _validate_event_view(
    value: Any, *, expected_index: int, expected_selected: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EVENT_VIEW_KEYS:
        raise WorkloadError("explicit-policy event view schema differs")
    selected = value.get("selected_backend")
    recommended = value.get("recommended_backend")
    mismatch = selected != recommended
    direction = f"{selected}_selected__{recommended}_recommended"
    reused = value.get("acknowledgement_reused")
    fingerprint_evaluated = value.get("workload_fingerprint_evaluated")
    fingerprint = value.get("workload_fingerprint")
    scope_digest = value.get("acknowledgement_scope_digest")
    acknowledgement_scope = value.get("acknowledgement_scope")
    origin_cohort_digest = value.get(
        "acknowledgement_origin_cohort_digest"
    )
    if (
        type(value.get("event_index")) is not int
        or value["event_index"] != expected_index
        or selected != expected_selected
        or recommended not in {"cpu", "cuda"}
        or direction not in DIRECTIONS
        or value.get("direction") != direction
        or value.get("reason_class")
        not in {"hardware_crossover", "environment"}
        or value.get("recommendation_confidence") not in {"medium", "high"}
        or type(value.get("mismatch_observed")) is not bool
        or value["mismatch_observed"] is not mismatch
        or type(value.get("attempts")) is not int
        or type(reused) is not bool
        or reused and not mismatch
        or value["attempts"] != (2 if mismatch and not reused else 1)
        or type(value.get("acknowledgements")) is not int
        or value["acknowledgements"] != (1 if mismatch and not reused else 0)
        or type(value.get("acknowledgement_allowed")) is not bool
        or type(value.get("warning_present")) is not bool
        or value["warning_present"] is not (mismatch and not reused)
        or type(value.get("final_acknowledgement")) is not bool
        or value["final_acknowledgement"] is not mismatch
        or type(fingerprint_evaluated) is not bool
        or (mismatch and not reused and fingerprint_evaluated is not True)
        or not isinstance(scope_digest, str)
        or len(scope_digest) != 64
        or any(character not in "0123456789abcdef" for character in scope_digest)
        or not isinstance(fingerprint, str)
        or len(fingerprint) != (64 if fingerprint_evaluated else 0)
        or fingerprint_evaluated and any(
            character not in "0123456789abcdef" for character in fingerprint
        )
        or value.get("recommendation_policy_version")
        != backend_guard.POLICY_VERSION
    ):
        raise WorkloadError("explicit-policy event view semantics differ")
    reason = value.get("acknowledgement_reason")
    if mismatch:
        if (
            acknowledgement_scope not in {"simulation", "process"}
            or not isinstance(origin_cohort_digest, str)
            or len(origin_cohort_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in origin_cohort_digest
            )
        ):
            raise WorkloadError(
                "explicit-policy acknowledgement audit differs"
            )
        try:
            backend_guard.validate_reason(reason)
        except backend_guard.BackendGuardEvidenceError as exc:
            raise WorkloadError(
                "explicit-policy event acknowledgement reason differs"
            ) from exc
    elif (
        reason != ""
        or acknowledgement_scope != ""
        or origin_cohort_digest != ""
    ):
        raise WorkloadError("explicit-policy matching event has a reason")
    if value["reason_class"] == "environment":
        if (
            selected != "cpu"
            or recommended != "cpu"
            or value["mismatch_observed"] is not False
            or value["attempts"] != 1
            or value["acknowledgements"] != 0
            or value["acknowledgement_reused"] is not False
            or value["acknowledgement_allowed"] is not False
            or value["acknowledgement_reason"] != ""
            or value["warning_present"] is not False
            or value["final_acknowledgement"] is not False
        ):
            raise WorkloadError("explicit-policy environment event differs")
    elif value["acknowledgement_allowed"] is not True:
        raise WorkloadError(
            "explicit-policy hardware event is not acknowledgeable"
        )
    return dict(value)


def summarize_lane(
    events: list[dict[str, Any]],
    *,
    selected_backend: str,
    reason: str,
    source: str,
) -> dict[str, Any]:
    if selected_backend not in {"cpu", "cuda"}:
        raise WorkloadError("explicit-policy lane backend is invalid")
    if not isinstance(source, str) or not source or not events:
        raise WorkloadError("explicit-policy lane source or events are absent")
    validated = []
    for index, event in enumerate(events):
        try:
            retained = backend_guard.validate_evidence(
                event,
                expected_backend=selected_backend,
                expected_reason=reason,
            )
        except backend_guard.BackendGuardEvidenceError as exc:
            raise WorkloadError(
                f"explicit-policy guard event {index} differs: {exc}"
            ) from exc
        mismatch = (
            retained["selected_backend"] != retained["recommended_backend"]
        )
        if (
            retained["mismatch_observed"] is not mismatch
            or retained["attempts"]
            != (2 if mismatch and not retained["acknowledgement_reused"] else 1)
            or retained["acknowledgements"]
            != (1 if mismatch and not retained["acknowledgement_reused"] else 0)
        ):
            raise WorkloadError(
                "explicit-policy event acknowledgement pattern differs"
            )
        validated.append(retained)
    event_views = [
        _event_view(event, index) for index, event in enumerate(validated)
    ]
    direction_counts = {direction: 0 for direction in DIRECTIONS}
    reason_class_counts: dict[str, int] = {}
    for event in validated:
        direction = (
            f"{event['selected_backend']}_selected__"
            f"{event['recommended_backend']}_recommended"
        )
        if direction not in direction_counts:
            raise WorkloadError("explicit-policy event direction differs")
        direction_counts[direction] += 1
        reason_class = event["final_telemetry"]["reason_class"]
        reason_class_counts[reason_class] = (
            reason_class_counts.get(reason_class, 0) + 1
        )
    return {
        "source": source,
        "selected_backend": selected_backend,
        "classification_unit": "guard-event",
        "case_level_recommendation": None,
        "event_count": len(validated),
        "attempt_count": sum(event["attempts"] for event in validated),
        "acknowledgement_count": sum(
            event["acknowledgements"] for event in validated
        ),
        "mutation_before_acknowledgement": False,
        "events_sha256": _canonical_sha256(validated),
        "event_views_sha256": _canonical_sha256(event_views),
        "events": event_views,
        "direction_counts": direction_counts,
        "reason_class_counts": dict(sorted(reason_class_counts.items())),
        "execution_wall_seconds": [
            event["execution_wall_seconds"] for event in validated
        ],
        "warning_wall_seconds": [
            event["warning_wall_seconds"] for event in validated
        ],
    }


def _paired_case(
    paired_output: pathlib.Path,
    paired_report: dict[str, Any],
    case_path: str,
    coverage_feature: str,
) -> dict[str, Any]:
    if coverage_feature not in EXPECTED_FEATURES:
        raise WorkloadError("explicit-policy coverage feature differs")
    matches = [
        record
        for record in paired_report.get("tasks", [])
        if record.get("task", {}).get("case_path") == case_path
    ]
    if len(matches) != 1:
        raise WorkloadError(f"explicit-policy paired case is not unique: {case_path}")
    archive_relative = matches[0].get("archive", {}).get("path")
    if not isinstance(archive_relative, str) or not archive_relative:
        raise WorkloadError("explicit-policy paired archive path is absent")
    archive_pure = pathlib.PurePosixPath(archive_relative)
    if (
        archive_pure.is_absolute()
        or archive_pure.as_posix() != archive_relative
        or any(part in {"", ".", ".."} for part in archive_pure.parts)
    ):
        raise WorkloadError("explicit-policy paired archive path is unsafe")
    archive = paired_output.joinpath(*archive_pure.parts).resolve(strict=True)
    if paired_output not in archive.parents or not archive.is_dir():
        raise WorkloadError("explicit-policy paired archive escaped its evidence root")
    archive_validation.verify_archive(archive)
    runner_root = archive.joinpath(*archive_validation.RUNNER_ROOT.parts)
    runner_report = _load_object(
        runner_root / "report.json", "explicit-policy paired runner report"
    )
    selected = [
        record
        for record in runner_report.get("results", [])
        if record.get("selected") is True
    ]
    if (
        len(selected) != 1
        or selected[0].get("path") != case_path
        or selected[0].get("outcome") != "PASS"
        or set(selected[0].get("runs", {})) != {"cpu", "cuda"}
    ):
        raise WorkloadError("explicit-policy paired runner result differs")
    lanes = {}
    durations = {}
    for backend in ("cpu", "cuda"):
        run = selected[0]["runs"][backend]
        if not str(run.get("outcome", "")).startswith("PASS"):
            raise WorkloadError("explicit-policy paired backend did not pass")
        source, events = _runner_lane_events(runner_root, run)
        lanes[backend] = summarize_lane(
            events,
            selected_backend=backend,
            reason=paired_matrix.BACKEND_ACKNOWLEDGEMENT_REASON,
            source=source,
        )
        duration = run.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise WorkloadError("explicit-policy diagnostic duration differs")
        durations[backend] = float(duration)
    return {
        "case_path": case_path,
        "coverage_feature": coverage_feature,
        "classification_unit": "guard-event",
        "case_level_recommendation": None,
        "lanes": lanes,
        "diagnostic_process_seconds": durations,
        "valid_for_speed_gate": False,
        "runner_report": file_record(runner_root / "report.json", paired_output),
    }


def _aunp_evidence(
    aunp_output: pathlib.Path, aunp_report: dict[str, Any]
) -> dict[str, Any]:
    lanes: dict[str, list[dict[str, Any]]] = {"cpu": [], "cuda": []}
    task_count = 0
    for task in aunp_report.get("tasks", []):
        spec = task.get("task", {})
        if spec.get("kind") != "performance":
            continue
        backend = spec.get("backend")
        if backend not in lanes:
            raise WorkloadError("AuNP explicit-policy backend differs")
        lane_output = _safe_member(
            aunp_output,
            f"{task['output']}/summary.json",
            "AuNP explicit-policy summary",
        ).parent
        summary = aunp_performance.validate_performance_output(
            lane_output,
            expected_backend=backend,
            expected_mpi_size=spec.get("ranks"),
        )
        events = [
            record["backend_guard"]
            for rank in summary["rank_records"]
            for record in rank["records"]
        ]
        lanes[backend].append(
            summarize_lane(
                events,
                selected_backend=backend,
                reason=AUNP_BACKEND_ACKNOWLEDGEMENT_REASON,
                source="aunp-performance-adapter",
            )
        )
        task_count += 1
    if task_count != 6 or len(lanes["cpu"]) != 2 or len(lanes["cuda"]) != 4:
        raise WorkloadError("AuNP explicit-policy performance inventory differs")
    return {
        "workload": "AuNP TM-only r4000 fixed-work release matrix",
        "classification_unit": "guard-event",
        "case_level_recommendation": None,
        "performance_task_count": task_count,
        "lanes": lanes,
        "valid_for_speed_gate": False,
        "reason": "policy evidence only; AuNP timing authority remains in M3 report",
    }


def aggregate_directions(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {direction: 0 for direction in DIRECTIONS}
    total = 0
    for lane in lanes:
        if not isinstance(lane, dict) or set(lane) != LANE_SUMMARY_KEYS:
            raise WorkloadError("explicit-policy lane summary schema differs")
        if (
            lane.get("classification_unit") != "guard-event"
            or lane.get("case_level_recommendation") is not None
            or lane.get("selected_backend") not in {"cpu", "cuda"}
            or not isinstance(lane.get("source"), str)
            or not lane["source"]
            or lane.get("mutation_before_acknowledgement") is not False
        ):
            raise WorkloadError("explicit-policy lane classification differs")
        lane_counts = lane.get("direction_counts")
        event_count = lane.get("event_count")
        views = lane.get("events")
        if (
            not isinstance(lane_counts, dict)
            or set(lane_counts) != set(DIRECTIONS)
            or type(event_count) is not int
            or event_count <= 0
            or any(
                type(lane_counts[item]) is not int or lane_counts[item] < 0
                for item in DIRECTIONS
            )
            or sum(lane_counts.values()) != event_count
            or not isinstance(views, list)
            or len(views) != event_count
        ):
            raise WorkloadError("explicit-policy lane direction summary differs")
        validated_views = [
            _validate_event_view(
                event,
                expected_index=index,
                expected_selected=lane["selected_backend"],
            )
            for index, event in enumerate(views)
        ]
        derived_lane_counts = {direction: 0 for direction in DIRECTIONS}
        reason_counts: dict[str, int] = {}
        for event in validated_views:
            derived_lane_counts[event["direction"]] += 1
            reason_class = event["reason_class"]
            reason_counts[reason_class] = reason_counts.get(reason_class, 0) + 1
        if (
            lane_counts != derived_lane_counts
            or lane.get("reason_class_counts")
            != dict(sorted(reason_counts.items()))
            or lane.get("attempt_count")
            != sum(event["attempts"] for event in validated_views)
            or lane.get("acknowledgement_count")
            != sum(event["acknowledgements"] for event in validated_views)
            or lane.get("event_views_sha256")
            != _canonical_sha256(validated_views)
            or not isinstance(lane.get("events_sha256"), str)
            or len(lane["events_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in lane["events_sha256"]
            )
            or not isinstance(lane.get("execution_wall_seconds"), list)
            or len(lane["execution_wall_seconds"]) != event_count
            or not isinstance(lane.get("warning_wall_seconds"), list)
            or len(lane["warning_wall_seconds"]) != event_count
        ):
            raise WorkloadError("explicit-policy lane event projection differs")
        for direction in DIRECTIONS:
            counts[direction] += lane_counts[direction]
        total += event_count
    if not lanes or any(counts[direction] <= 0 for direction in DIRECTIONS):
        raise WorkloadError(
            "explicit-policy evidence does not cover all selected/recommended directions"
        )
    return validate_aggregate_projection({
        "classification_unit": "guard-event",
        "total_event_count": total,
        "direction_counts": counts,
        "all_selected_recommended_directions_present": True,
    })


def validate_aggregate_projection(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "classification_unit",
            "total_event_count",
            "direction_counts",
            "all_selected_recommended_directions_present",
        }
        or value.get("classification_unit") != "guard-event"
        or type(value.get("total_event_count")) is not int
        or value["total_event_count"] <= 0
        or not isinstance(value.get("direction_counts"), dict)
        or set(value["direction_counts"]) != set(DIRECTIONS)
        or any(
            type(value["direction_counts"][direction]) is not int
            or value["direction_counts"][direction] <= 0
            for direction in DIRECTIONS
        )
        or sum(value["direction_counts"].values())
        != value["total_event_count"]
        or value.get("all_selected_recommended_directions_present") is not True
    ):
        raise WorkloadError("explicit-policy aggregate projection differs")
    return {
        "classification_unit": value["classification_unit"],
        "total_event_count": value["total_event_count"],
        "direction_counts": dict(value["direction_counts"]),
        "all_selected_recommended_directions_present": True,
    }


def policy_contract_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Validate and project the event-native, user-explicit policy contract."""

    features = report.get("features")
    if (
        report.get("outcome") != "PASS"
        or report.get("backend_selection") != "user-explicit-only"
        or report.get("automatic_backend_selection") is not False
        or report.get("classification_unit") != "guard-event"
        or report.get("case_level_recommendations_used") is not False
        or not isinstance(features, dict)
        or set(features) != set(EXPECTED_FEATURES)
    ):
        raise WorkloadError("explicit-policy top-level contract differs")

    lanes: list[dict[str, Any]] = []
    inventories = {
        "policy.recommend_cpu": CPU_RECOMMENDATION_CASES,
        "policy.recommend_cuda": CUDA_RECOMMENDATION_CASES,
        "policy.long_horizon_small": LONG_HORIZON_CASES,
    }
    for feature_id, expected_cases in inventories.items():
        feature = features.get(feature_id)
        cases = feature.get("cases") if isinstance(feature, dict) else None
        if (
            not isinstance(feature, dict)
            or feature.get("outcome") != "PASS"
            or not isinstance(cases, list)
            or tuple(case.get("case_path") for case in cases) != expected_cases
        ):
            raise WorkloadError("explicit-policy feature case inventory differs")
        for case in cases:
            if (
                not isinstance(case, dict)
                or case.get("coverage_feature") != feature_id
                or case.get("classification_unit") != "guard-event"
                or case.get("case_level_recommendation") is not None
                or not isinstance(case.get("lanes"), dict)
                or set(case["lanes"]) != {"cpu", "cuda"}
            ):
                raise WorkloadError("explicit-policy case contract differs")
            lanes.extend(case["lanes"].values())

    aunp = features["policy.recommend_cuda"].get("aunp")
    aunp_lanes = aunp.get("lanes") if isinstance(aunp, dict) else None
    if (
        not isinstance(aunp, dict)
        or aunp.get("classification_unit") != "guard-event"
        or aunp.get("case_level_recommendation") is not None
        or aunp.get("performance_task_count") != 6
        or not isinstance(aunp_lanes, dict)
        or set(aunp_lanes) != {"cpu", "cuda"}
        or not isinstance(aunp_lanes["cpu"], list)
        or len(aunp_lanes["cpu"]) != 2
        or not isinstance(aunp_lanes["cuda"], list)
        or len(aunp_lanes["cuda"]) != 4
    ):
        raise WorkloadError("explicit-policy AuNP contract differs")
    lanes.extend(aunp_lanes["cpu"])
    lanes.extend(aunp_lanes["cuda"])
    aggregate = aggregate_directions(lanes)
    if report.get("aggregate") != aggregate:
        raise WorkloadError("explicit-policy aggregate was not re-derived")
    return {
        "backend_selection": "user-explicit-only",
        "automatic_backend_selection": False,
        "classification_unit": "guard-event",
        "case_level_recommendations_used": False,
        "aggregate": aggregate,
    }


def derive(paired_output: pathlib.Path, aunp_output: pathlib.Path) -> dict[str, Any]:
    paired_output = paired_output.resolve(strict=True)
    aunp_output = aunp_output.resolve(strict=True)
    paired_report = paired_matrix.verify_complete(paired_output)
    aunp_report = aunp_matrix.verify_hybrid_complete(aunp_output)
    paired_build = paired_report.get("contract", {}).get("build_receipt", {})
    paired_receipt = paired_build.get("receipt_id")
    paired_source = paired_build.get("source_sha256")
    paired_git_head = paired_build.get("git_head")
    aunp_receipt = aunp_report.get("build", {}).get("receipt_id")
    aunp_git_head = aunp_report.get("build", {}).get("git_head")
    aunp_source = aunp_report.get("source_snapshot_sha256")
    if (
        not isinstance(paired_receipt, str)
        or len(paired_receipt) != 64
        or paired_receipt != aunp_receipt
        or not isinstance(paired_source, str)
        or len(paired_source) != 64
        or paired_source != aunp_source
        or not isinstance(paired_git_head, str)
        or len(paired_git_head) != 40
        or any(
            character not in "0123456789abcdef"
            for character in paired_git_head
        )
        or paired_git_head != aunp_git_head
    ):
        raise WorkloadError(
            "explicit-policy candidate receipt/source/Git identity differs"
        )
    recommend_cpu = [
        _paired_case(
            paired_output, paired_report, case, "policy.recommend_cpu"
        )
        for case in CPU_RECOMMENDATION_CASES
    ]
    recommend_cuda = [
        _paired_case(
            paired_output, paired_report, case, "policy.recommend_cuda"
        )
        for case in CUDA_RECOMMENDATION_CASES
    ]
    aunp = _aunp_evidence(aunp_output, aunp_report)
    long_horizon = [
        _paired_case(
            paired_output, paired_report, case, "policy.long_horizon_small"
        )
        for case in LONG_HORIZON_CASES
    ]
    features = {
        "policy.recommend_cpu": {
            "outcome": "PASS",
            "cases": recommend_cpu,
            "proof": "event-native CPU/CUDA recommendation, exact first warning, and audited scope-reuse coverage",
        },
        "policy.recommend_cuda": {
            "outcome": "PASS",
            "cases": recommend_cuda,
            "aunp": aunp,
            "proof": "event-native CPU/CUDA recommendation, exact first warning, and audited scope-reuse coverage",
        },
        "policy.long_horizon_small": {
            "outcome": "PASS",
            "cases": long_horizon,
            "valid_for_speed_gate": False,
            "classification": "diagnostic timing retained without backend switching",
        },
    }
    if tuple(features) != EXPECTED_FEATURES:
        raise WorkloadError("explicit-policy feature inventory differs")
    paired_cases = [*recommend_cpu, *recommend_cuda, *long_horizon]
    summarized_lanes = [
        lane for case in paired_cases for lane in case["lanes"].values()
    ] + [
        lane for group in aunp["lanes"].values() for lane in group
    ]
    aggregate = aggregate_directions(summarized_lanes)
    report = {
        "schema": SCHEMA,
        "outcome": "PASS",
        "receipt_id": paired_receipt,
        "source_sha256": paired_source,
        "git_head": paired_git_head,
        "backend_selection": "user-explicit-only",
        "automatic_backend_selection": False,
        "classification_unit": "guard-event",
        "case_level_recommendations_used": False,
        "aggregate": aggregate,
        "features": features,
        "inputs": {
            "paired_complete": absolute_file_record(
                paired_output / "PAIRED_COMPLETE", "paired COMPLETE"
            ),
            "aunp_complete": absolute_file_record(
                aunp_output / "COMPLETE", "AuNP COMPLETE"
            ),
        },
        "evidence_code": {
            "controller": absolute_file_record(pathlib.Path(__file__), "controller"),
            "guard": absolute_file_record(
                SCRIPT_DIR.parent / "gpmeep_backend_guard.py", "guard"
            ),
            "oracle": absolute_file_record(SCRIPT_DIR / "run_example_oracle.py", "oracle"),
            "sitecustomize": absolute_file_record(
                SCRIPT_DIR / "sitecustomize.py", "sitecustomize"
            ),
        },
    }
    policy_contract_projection(report)
    return report


def verify(output: pathlib.Path) -> dict[str, Any]:
    output = output.resolve(strict=True)
    complete = _load_object(output / "COMPLETE", "explicit-policy COMPLETE")
    if set(complete) != {"schema", "outcome", "report"} or (
        complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("outcome") != "PASS"
    ):
        raise WorkloadError("explicit-policy terminal marker differs")
    report_path = output / "report.json"
    if complete.get("report") != file_record(report_path, output):
        raise WorkloadError("explicit-policy report binding differs")
    report = _load_object(report_path, "explicit-policy report")
    inputs = report.get("inputs", {})
    paired_complete = pathlib.Path(inputs.get("paired_complete", {}).get("path", ""))
    aunp_complete = pathlib.Path(inputs.get("aunp_complete", {}).get("path", ""))
    if paired_complete.name != "PAIRED_COMPLETE" or aunp_complete.name != "COMPLETE":
        raise WorkloadError("explicit-policy retained input path differs")
    derived = derive(paired_complete.parent, aunp_complete.parent)
    if report != derived:
        raise WorkloadError("explicit-policy report was not re-derived")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--paired-output", required=True, type=pathlib.Path)
    parser.add_argument("--aunp-output", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("explicit-policy output must be fresh")
    output.mkdir(parents=True, mode=0o700)
    report = derive(args.paired_output, args.aunp_output)
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "COMPLETE",
        {
            "schema": COMPLETE_SCHEMA,
            "outcome": "PASS",
            "report": file_record(output / "report.json", output),
        },
    )
    verify(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, WorkloadError) as exc:
        print(f"M3 explicit-policy evidence failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
