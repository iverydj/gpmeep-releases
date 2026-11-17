#!/usr/bin/env python3
"""Prepare and replay a self-contained, evidence-only M3 audit bundle.

This receipt is deliberately distinct from the release-authorizing milestone
audit receipt.  It can preserve a reviewed MISMATCH and open material
findings for historical characterization, but it can never authorize a
candidate release, source mutation, tag, upload, or package publication.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
from collections import Counter
from typing import Any


SCHEMA = "gpmeep-evidence-only-adversarial-receipt-v1"
DRAFT_SCHEMA = "gpmeep-adversarial-audit-draft-v1"
RECEIPT_NAME = "EVIDENCE_ONLY_RECEIPT.json"
CASE_PATH = "python/examples/stochastic_emitter_reciprocity.py"
UNIT_ID = "production-mixed-07"
RULE = "example_results.reciprocity_forward_normalized_convergence_residual.*"
FAILURE_METRIC = f"{RULE[:-1]}real[0]"
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
SEVERITIES = {"critical", "high", "medium", "low", "info"}
STATUSES = {"open", "closed"}
MATERIAL_SEVERITIES = {"critical", "high", "medium"}
MAXIMUM_FILE_BYTES = 64 * 1024**2
EXPECTED_FINDING_IDS = (
    "M3-HIGH-01",
    "M3-MEDIUM-02",
    "M3-MEDIUM-03",
    "M3-HIGH-04",
    "M3-HIGH-05",
    "M3-HIGH-06",
    "M3-MEDIUM-07",
    "M3-LOW-08",
    "M3-INFO-09",
    "M3-MEDIUM-10",
    "M3-LOW-11",
    "M3-INFO-12",
)
EXPECTED_OPEN_MATERIAL_IDS = (
    "M3-HIGH-01",
    "M3-MEDIUM-02",
    "M3-MEDIUM-03",
    "M3-HIGH-04",
    "M3-HIGH-05",
    "M3-HIGH-06",
    "M3-MEDIUM-07",
)
EXPECTED_TASK56_OPTIONS = (
    "--repo",
    "--manifest",
    "--build-python",
    "--install-prefix",
    "--python",
    "--output",
    "--backend",
    "--acknowledge-backend-mismatch-reason",
    "--tier",
    "--case",
)
TASK56_ACKNOWLEDGEMENT = (
    "M3 paired correctness intentionally compares explicit CPU and CUDA"
)
METRIC_EVIDENCE_SCHEMA = "gpmeep-json-metric-evidence-v3-direct"
MAX_INLINE_JSON_METRICS = 4096
MAX_INLINE_JSON_BYTES = 1024 * 1024
CLAIM_BOUNDARY = {
    "included": [
        "M3 paired correctness characterization",
        "reviewed task-56 evidence-only recomparison",
        "historical evidence and audit reproducibility",
    ],
    "excluded": [
        "candidate release readiness",
        "candidate source changes",
        "component or aggregate release authorization",
        "version bump, tag, upload, or package publication",
        "official-example timing as a release performance claim",
    ],
}
ROLE_PATHS = {
    "audit_draft": "audit/M3_PAIRED_AUDIT_DRAFT.json",
    "audit_review": "audit/M3_PAIRED_INDEPENDENT_REVIEW.md",
    "paired_terminal": "paired/PAIRED_COMPLETE",
    "paired_report": "paired/report.json",
    "paired_checkpoint": "paired/CHECKPOINT.json",
    "paired_journal": "paired/JOURNAL.json",
    "task56_runner_report": "task56/original-runner-report.json",
    "task56_replay": "task56/replay.json",
    "task56_original_attempt_journal": "task56/original-attempt-JOURNAL.json",
    "task56_original_process_log": "task56/original-process.log",
    "task56_cpu_stdout": "task56/cpu-stdout.log",
    "task56_cuda_stdout": "task56/cuda-stdout.log",
    "task56_adjudication": "task56/adjudication/ADJUDICATION.json",
    "task56_adjudication_script": (
        "task56/adjudication/adjudicate_m3_task56_reciprocity.py"
    ),
    "task56_independent_review_json": (
        "task56/adjudication/INDEPENDENT_REVIEW.json"
    ),
    "task56_independent_review_md": (
        "task56/adjudication/INDEPENDENT_REVIEW.md"
    ),
    "task56_schema_review_json": "task56/adjudication/SCHEMA_REVIEW.json",
    "task56_schema_review_md": "task56/adjudication/SCHEMA_REVIEW.md",
    "source_manifest": "source/manifest.json",
    "source_validation_comparator": "source/run_validation.py",
    "source_reciprocity_wrapper": "source/stochastic_reciprocity_matrix.py",
    "control_verifier": "control/seal_evidence_only_adversarial_audit.py",
}
GIT_SOURCE_PATHS = {
    "source_manifest": "scripts/python-validation/manifest.json",
    "source_validation_comparator": "scripts/python-validation/run_validation.py",
    "source_reciprocity_wrapper": (
        "scripts/python-validation/stochastic_reciprocity_matrix.py"
    ),
}
ADJUDICATION_SOURCE_NAMES = {
    "task56_adjudication": "ADJUDICATION.json",
    "task56_adjudication_script": "adjudicate_m3_task56_reciprocity.py",
    "task56_independent_review_json": "INDEPENDENT_REVIEW.json",
    "task56_independent_review_md": "INDEPENDENT_REVIEW.md",
    "task56_schema_review_json": "SCHEMA_REVIEW.json",
    "task56_schema_review_md": "SCHEMA_REVIEW.md",
}


class EvidenceOnlyError(RuntimeError):
    """Raised when evidence-only provenance cannot be exactly replayed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceOnlyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceOnlyError(f"invalid JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceOnlyError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceOnlyError(f"{label} is not a JSON object")
    return value


def _stable_bytes(path: pathlib.Path, label: str) -> tuple[bytes, os.stat_result]:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        before = lexical.lstat()
    except OSError as exc:
        raise EvidenceOnlyError(f"{label} is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > MAXIMUM_FILE_BYTES
    ):
        raise EvidenceOnlyError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise EvidenceOnlyError(f"{label} could not be opened: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        chunks = []
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1024 * 1024, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
            if offset > MAXIMUM_FILE_BYTES:
                raise EvidenceOnlyError(f"{label} exceeds its size bound")
        payload = b"".join(chunks)
        after = lexical.lstat()
    except OSError as exc:
        raise EvidenceOnlyError(f"{label} could not be read: {exc}") from exc
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(before) != identity(after)
        or len(payload) != before.st_size
    ):
        raise EvidenceOnlyError(f"{label} changed while read")
    return payload, before


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = lexical.resolve(strict=True)
        metadata = lexical.lstat()
    except OSError as exc:
        raise EvidenceOnlyError(f"{label} is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != lexical
    ):
        raise EvidenceOnlyError(f"{label} is not a canonical directory")
    return lexical


def _safe_relative(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceOnlyError(f"{label} path is invalid")
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise EvidenceOnlyError(f"{label} path is unsafe")
    return relative


def _member(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    safe = _safe_relative(relative, label)
    current = root
    for part in safe.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise EvidenceOnlyError(f"{label} parent is unavailable: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceOnlyError(f"{label} parent is not a fixed directory")
    path = root.joinpath(*safe.parts)
    try:
        if path.resolve(strict=True) != path:
            raise EvidenceOnlyError(f"{label} path is not canonical")
    except OSError as exc:
        raise EvidenceOnlyError(f"{label} path is unavailable: {exc}") from exc
    return path


def _record(root: pathlib.Path, role: str) -> tuple[dict[str, Any], bytes]:
    relative = ROLE_PATHS[role]
    path = _member(root, relative, role)
    payload, metadata = _stable_bytes(path, role)
    return (
        {
            "path": relative,
            "size_bytes": metadata.st_size,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        payload,
    )


def _inventory(root: pathlib.Path, *, receipt_present: bool) -> None:
    expected = set(ROLE_PATHS.values())
    if receipt_present:
        expected.add(RECEIPT_NAME)
    actual: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        base = pathlib.Path(directory)
        for name in names:
            child = base / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise EvidenceOnlyError("bundle contains an unsafe directory")
        for name in files:
            child = base / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise EvidenceOnlyError("bundle contains an unsafe file")
            actual.add(child.relative_to(root).as_posix())
    if actual != expected:
        raise EvidenceOnlyError("bundle file inventory differs")


def _file_record_matches(
    record: Any, expected: dict[str, Any], label: str
) -> None:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise EvidenceOnlyError(f"{label} file record schema differs")
    if (
        record.get("path") != expected["path"]
        or record.get("size_bytes") != expected["size_bytes"]
        or record.get("sha256") != expected["sha256"]
    ):
        raise EvidenceOnlyError(f"{label} file record differs")


def _finding_inventory(
    draft: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int], list[str]]:
    expected_keys = {
        "schema",
        "milestone",
        "auditor",
        "source",
        "evidence",
        "scope",
        "findings",
        "review",
    }
    if (
        set(draft) != expected_keys
        or draft.get("schema") != DRAFT_SCHEMA
        or draft.get("milestone") != "M3"
    ):
        raise EvidenceOnlyError("audit draft identity differs")
    findings = draft.get("findings")
    if not isinstance(findings, list) or len(findings) != len(
        EXPECTED_FINDING_IDS
    ):
        raise EvidenceOnlyError("audit finding inventory differs")
    retained = []
    for value in findings:
        keys = {
            "id",
            "severity",
            "status",
            "summary",
            "resolution",
            "verification",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise EvidenceOnlyError("audit finding schema differs")
        if (
            SAFE_ID.fullmatch(str(value.get("id", ""))) is None
            or value.get("severity") not in SEVERITIES
            or value.get("status") not in STATUSES
            or any(
                not isinstance(value.get(field), str)
                or not value[field].strip()
                for field in keys
            )
        ):
            raise EvidenceOnlyError("audit finding value differs")
        retained.append(dict(value))
    if tuple(item["id"] for item in retained) != EXPECTED_FINDING_IDS:
        raise EvidenceOnlyError("audit finding order or identity differs")
    open_material = [
        item["id"]
        for item in retained
        if item["status"] == "open"
        and item["severity"] in MATERIAL_SEVERITIES
    ]
    if tuple(open_material) != EXPECTED_OPEN_MATERIAL_IDS:
        raise EvidenceOnlyError("audit open material inventory differs")
    counts = Counter(item["severity"] for item in retained)
    unresolved = Counter(
        item["severity"] for item in retained if item["status"] == "open"
    )
    return (
        retained,
        {name: counts[name] for name in sorted(SEVERITIES)},
        {name: unresolved[name] for name in sorted(SEVERITIES)},
        open_material,
    )


def _draft_source(draft: dict[str, Any]) -> tuple[str, str]:
    source = draft.get("source")
    if not isinstance(source, dict) or set(source) != {"repo", "commit"}:
        raise EvidenceOnlyError("audit source binding differs")
    repository = source.get("repo")
    commit = source.get("commit")
    if (
        not isinstance(repository, str)
        or not repository
        or not isinstance(commit, str)
        or COMMIT.fullmatch(commit) is None
    ):
        raise EvidenceOnlyError("audit source value differs")
    return repository, commit


def _validate_audit(
    draft: dict[str, Any],
    review_payload: bytes,
    records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    findings, counts, unresolved, open_material = _finding_inventory(draft)
    repository, commit = _draft_source(draft)
    evidence = draft.get("evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"root", "terminal_name"}
        or not isinstance(evidence.get("root"), str)
        or not pathlib.PurePath(evidence["root"]).is_absolute()
        or evidence.get("terminal_name") != "PAIRED_COMPLETE"
    ):
        raise EvidenceOnlyError("audit evidence binding differs")
    auditor = draft.get("auditor")
    if (
        not isinstance(auditor, dict)
        or set(auditor) != {"agent_id", "task_name"}
        or SAFE_ID.fullmatch(str(auditor.get("agent_id", ""))) is None
        or not isinstance(auditor.get("task_name"), str)
        or not auditor["task_name"]
    ):
        raise EvidenceOnlyError("audit reviewer identity differs")
    scope = draft.get("scope")
    if (
        not isinstance(scope, list)
        or not scope
        or len(scope) != len(set(scope))
        or any(not isinstance(item, str) or not item.strip() for item in scope)
    ):
        raise EvidenceOnlyError("audit scope differs")
    if not isinstance(draft.get("review"), str) or not draft["review"].strip():
        raise EvidenceOnlyError("audit summary review differs")
    try:
        full_review = review_payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise EvidenceOnlyError("full audit review is not UTF-8") from exc
    review_anchors = (
        f"Reviewer agent: `{auditor['agent_id']}`",
        f"Reviewer task: `{auditor['task_name']}`",
        f"Candidate: `{repository}`",
        f"Candidate commit: `{commit}`",
        f"Evidence root: `{evidence['root']}`",
        f"Terminal: `{evidence['terminal_name']}`",
        f"SHA-256 `{records['audit_draft']['sha256']}`",
        f"| `PAIRED_COMPLETE` | `{records['paired_terminal']['sha256']}` |",
        f"| `CHECKPOINT.json` | `{records['paired_checkpoint']['sha256']}` |",
        f"| `JOURNAL.json` | `{records['paired_journal']['sha256']}` |",
        f"| paired terminal report | `{records['paired_report']['sha256']}` |",
    )
    if (
        any(anchor not in full_review for anchor in review_anchors)
        or "GO" not in full_review
        or "BLOCKED" not in full_review
        or any(finding_id not in full_review for finding_id in EXPECTED_FINDING_IDS)
    ):
        raise EvidenceOnlyError("full audit review does not bind the draft")
    return (
        {
            "auditor": dict(auditor),
            "scope": list(scope),
            "findings": findings,
            "finding_counts": counts,
            "unresolved_counts": unresolved,
            "open_material_finding_ids": open_material,
            "review_summary": draft["review"],
        },
        repository,
        evidence["root"],
    )


def _validate_core(
    values: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    commit: str,
    source_repository: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    terminal = values["paired_terminal"]
    if (
        terminal.get("schema") != "gpmeep-m3-paired-complete-v1"
        or terminal.get("outcome") != "PASS"
        or set(terminal)
        != {"schema", "outcome", "checkpoint", "journal", "report"}
    ):
        raise EvidenceOnlyError("paired terminal differs")
    for key, role, name in (
        ("checkpoint", "paired_checkpoint", "CHECKPOINT.json"),
        ("journal", "paired_journal", "JOURNAL.json"),
        ("report", "paired_report", "report.json"),
    ):
        expected = dict(records[role])
        expected["path"] = name
        _file_record_matches(terminal[key], expected, f"terminal {key}")

    report = values["paired_report"]
    checkpoint = values["paired_checkpoint"]
    journal = values["paired_journal"]
    if (
        report.get("schema") != "gpmeep-m3-paired-matrix-v1"
        or report.get("outcome") != "PASS"
        or not isinstance(report.get("tasks"), list)
        or len(report["tasks"]) != 70
        or checkpoint.get("schema") != "gpmeep-m3-paired-checkpoint-v1"
        or not isinstance(checkpoint.get("completed_tasks"), list)
        or len(checkpoint["completed_tasks"]) != 70
        or journal.get("schema") != "gpmeep-m3-paired-journal-v1"
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise EvidenceOnlyError("paired core semantics differ")
    if report["tasks"] != checkpoint["completed_tasks"]:
        raise EvidenceOnlyError("paired report/checkpoint task vectors differ")
    contracts = [
        report.get("contract"),
        checkpoint.get("contract"),
        journal.get("contract"),
    ]
    for contract in contracts:
        if (
            not isinstance(contract, dict)
            or contract.get("build_receipt", {}).get("git_head") != commit
        ):
            raise EvidenceOnlyError("paired core source commit differs")
    if contracts[0] != contracts[1] or contracts[0] != contracts[2]:
        raise EvidenceOnlyError("paired core contracts differ")
    contract = contracts[0]
    paths = contract.get("paths")
    if not isinstance(paths, dict):
        raise EvidenceOnlyError("paired core path contract differs")
    required_paths = {
        "repo",
        "manifest",
        "build_python",
        "install_prefix",
        "python",
        "runner",
    }
    if any(
        not isinstance(paths.get(name), str)
        or not pathlib.PurePosixPath(paths[name]).is_absolute()
        for name in required_paths
    ):
        raise EvidenceOnlyError("paired core required paths differ")
    if (
        paths["repo"] != source_repository
        or paths["manifest"]
        != f"{source_repository}/scripts/python-validation/manifest.json"
        or paths["runner"]
        != f"{source_repository}/scripts/python-validation/run_validation.py"
    ):
        raise EvidenceOnlyError("paired core source path binding differs")

    task_keys = {
        "backends",
        "case_path",
        "covered_example",
        "index",
        "plan_launch_order",
        "purpose",
        "release_role",
        "section",
        "unit_id",
    }
    task_specs: list[dict[str, Any]] = []
    for position, item in enumerate(report["tasks"], start=1):
        specification = item.get("task") if isinstance(item, dict) else None
        if (
            not isinstance(specification, dict)
            or set(specification) != task_keys
            or specification.get("index") != position
            or not isinstance(specification.get("plan_launch_order"), int)
            or isinstance(specification.get("plan_launch_order"), bool)
            or specification["plan_launch_order"] <= 0
            or not isinstance(specification.get("unit_id"), str)
            or not specification["unit_id"]
            or not isinstance(specification.get("case_path"), str)
            or not specification["case_path"]
            or not isinstance(item.get("process"), dict)
            or not isinstance(item.get("archive", {}).get("archive_complete"), dict)
        ):
            raise EvidenceOnlyError("paired task vector semantics differ")
        task_specs.append(specification)
    if (
        len({item["unit_id"] for item in task_specs}) != 70
        or len({item["case_path"] for item in task_specs}) != 70
    ):
        raise EvidenceOnlyError("paired task identities are not unique")

    events = journal["events"]
    state_events = {
        state_name: [event for event in events if event.get("state") == state_name]
        for state_name in (
            "controller-started",
            "task-started",
            "task-process-ended",
            "task-validated",
        )
    }
    if (
        set(event.get("state") for event in events) != set(state_events)
        or len(state_events["controller-started"]) != 1
        or any(
            len(state_events[state_name]) != 70
            for state_name in (
                "task-started",
                "task-process-ended",
                "task-validated",
            )
        )
    ):
        raise EvidenceOnlyError("paired journal state inventory differs")
    for index, specification in enumerate(task_specs):
        started = state_events["task-started"][index]
        ended = state_events["task-process-ended"][index]
        validated = state_events["task-validated"][index]
        if (
            started.get("task") != specification
            or ended.get("task") != specification
            or validated.get("task") != specification
            or ended.get("process") != report["tasks"][index]["process"]
            or validated.get("archive_complete")
            != report["tasks"][index]["archive"]["archive_complete"]
        ):
            raise EvidenceOnlyError("paired journal task replay differs")
    matches = [
        task
        for task in report["tasks"]
        if task.get("task", {}).get("index") == 56
        and task.get("task", {}).get("unit_id") == UNIT_ID
        and task.get("task", {}).get("case_path") == CASE_PATH
    ]
    checkpoints = [
        task
        for task in checkpoint["completed_tasks"]
        if task.get("task", {}).get("index") == 56
        and task.get("task", {}).get("unit_id") == UNIT_ID
        and task.get("task", {}).get("case_path") == CASE_PATH
    ]
    if len(matches) != 1 or len(checkpoints) != 1:
        raise EvidenceOnlyError("paired task-56 record is not unique")
    task = matches[0]
    checkpoint_task = checkpoints[0]
    original_sha = records["task56_runner_report"]["sha256"]
    original_size = records["task56_runner_report"]["size_bytes"]
    replay_sha = records["task56_replay"]["sha256"]
    replay_size = records["task56_replay"]["size_bytes"]
    if (
        task.get("archive", {}).get("report_sha256") != original_sha
        or task.get("archive", {}).get("runner_report", {}).get("sha256")
        != original_sha
        or task.get("archive", {}).get("runner_report", {}).get("size_bytes")
        != original_size
        or checkpoint_task.get("archive", {}).get("report_sha256")
        != original_sha
        or checkpoint_task.get("archive", {})
        .get("runner_report", {})
        .get("sha256")
        != original_sha
        or checkpoint_task.get("archive", {})
        .get("runner_report", {})
        .get("size_bytes")
        != original_size
        or task.get("telemetry_replay", {}).get("sha256") != replay_sha
        or task.get("telemetry_replay", {}).get("size_bytes") != replay_size
        or checkpoint_task.get("telemetry_replay", {}).get("sha256")
        != replay_sha
        or checkpoint_task.get("telemetry_replay", {}).get("size_bytes")
        != replay_size
    ):
        raise EvidenceOnlyError("paired task-56 file binding differs")
    return report, task, contract


def _validate_original_report(
    report: dict[str, Any], records: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [item for item in report.get("results", []) if item.get("selected")]
    if len(selected) != 1:
        raise EvidenceOnlyError("task-56 selected result is not unique")
    result = selected[0]
    comparison = result.get("comparison", {})
    failures = comparison.get("failures")
    runs = result.get("runs")
    if (
        report.get("schema_version") != 4
        or report.get("exit_code") != 3
        or report.get("summary") != {"MISMATCH": 1, "NOT_SELECTED": 209}
        or result.get("path") != CASE_PATH
        or result.get("outcome") != "MISMATCH"
        or comparison.get("outcome") != "MISMATCH"
        or comparison.get("failure_count") != 1
        or comparison.get("metric_evidence", {}).get("count") != 5904
        or not isinstance(failures, list)
        or len(failures) != 1
        or failures[0].get("metric") != FAILURE_METRIC
        or not isinstance(runs, dict)
        or set(runs) != {"cpu", "cuda"}
        or any(
            not isinstance(value, dict)
            or not str(value.get("outcome", "")).startswith("PASS")
            for value in runs.values()
        )
        or report.get("configuration", {}).get("case_ids") != [CASE_PATH]
        or report.get("configuration", {}).get("backends") != ["cpu", "cuda"]
        or report.get("configuration", {}).get("tiers") != ["full"]
    ):
        raise EvidenceOnlyError("task-56 native MISMATCH semantics differ")
    for backend, role in (
        ("cpu", "task56_cpu_stdout"),
        ("cuda", "task56_cuda_stdout"),
    ):
        source = runs[backend].get("evidence_files", {}).get("stdout")
        if (
            not isinstance(source, dict)
            or source.get("available") is not True
            or source.get("sha256") != records[role]["sha256"]
            or source.get("size_bytes") != records[role]["size_bytes"]
        ):
            raise EvidenceOnlyError(f"task-56 {backend} stdout binding differs")
    return result, failures[0]


def _validate_original_execution(
    journal: dict[str, Any],
    log_record: dict[str, Any],
    contract: dict[str, Any],
    evidence_root: str,
) -> dict[str, Any]:
    events = [
        event
        for event in journal.get("events", [])
        if event.get("state") == "task-process-ended"
        and event.get("task", {}).get("index") == 56
        and event.get("task", {}).get("unit_id") == UNIT_ID
        and event.get("task", {}).get("case_path") == CASE_PATH
    ]
    if len(events) != 1:
        raise EvidenceOnlyError("original task-56 process event is not unique")
    process = events[0].get("process")
    if not isinstance(process, dict):
        raise EvidenceOnlyError("original task-56 process record differs")
    command = process.get("command")
    saved_log = process.get("log")
    if (
        not isinstance(command, list)
        or len(command) != 2 + 2 * len(EXPECTED_TASK56_OPTIONS)
        or not all(isinstance(item, str) and item for item in command)
        or tuple(command[2::2]) != EXPECTED_TASK56_OPTIONS
    ):
        raise EvidenceOnlyError("original task-56 full argv differs")
    option_values = dict(zip(command[2::2], command[3::2], strict=True))
    paths = contract["paths"]
    expected_output = (
        pathlib.PurePosixPath(evidence_root) / "raw" / "056-production-mixed-07"
    ).as_posix()
    if (
        process.get("returncode") != 3
        or process.get("timed_out") is not False
        or process.get("output_limited") is not False
        or command[0] != paths["python"]
        or command[1] != paths["runner"]
        or option_values["--repo"] != paths["repo"]
        or option_values["--manifest"] != paths["manifest"]
        or option_values["--build-python"] != paths["build_python"]
        or option_values["--install-prefix"] != paths["install_prefix"]
        or option_values["--python"] != paths["python"]
        or option_values["--output"] != expected_output
        or option_values["--backend"] != "both"
        or option_values["--acknowledge-backend-mismatch-reason"]
        != TASK56_ACKNOWLEDGEMENT
        or option_values["--tier"] != "full"
        or option_values["--case"] != CASE_PATH
        or not isinstance(saved_log, dict)
        or saved_log.get("sha256") != log_record["sha256"]
        or saved_log.get("size_bytes") != log_record["size_bytes"]
    ):
        raise EvidenceOnlyError("original task-56 execution semantics differ")
    return {
        "returncode": 3,
        "timed_out": False,
        "output_limited": False,
        "full_argv": list(command),
        "full_argv_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _task56_comparison(
    manifest: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    entries = [
        item
        for item in manifest.get("overrides", [])
        if isinstance(item, dict) and item.get("glob") == CASE_PATH
    ]
    if len(entries) != 1:
        raise EvidenceOnlyError("task-56 manifest override is not unique")
    comparison = entries[0].get("comparison")
    if (
        not isinstance(comparison, dict)
        or comparison.get("mode") != "json_metrics"
        or not isinstance(comparison.get("prefix"), str)
        or not comparison["prefix"]
        or not isinstance(comparison.get("tolerances"), dict)
    ):
        raise EvidenceOnlyError("task-56 source comparison differs")
    tolerances: dict[str, dict[str, Any]] = {}
    for name, value in comparison["tolerances"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, dict)
            or set(value) - {"atol", "rtol", "diagnostic_only"}
        ):
            raise EvidenceOnlyError("task-56 tolerance schema differs")
        retained: dict[str, Any] = {}
        for field in ("atol", "rtol"):
            number = value.get(field, 0.0)
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(number)
                or number < 0
            ):
                raise EvidenceOnlyError("task-56 tolerance value differs")
            retained[field] = number
        diagnostic = value.get("diagnostic_only", False)
        if not isinstance(diagnostic, bool):
            raise EvidenceOnlyError("task-56 tolerance diagnostic flag differs")
        if "diagnostic_only" in value:
            retained["diagnostic_only"] = diagnostic
        tolerances[name] = retained
    if not tolerances or all(
        value.get("diagnostic_only", False) for value in tolerances.values()
    ):
        raise EvidenceOnlyError("task-56 has no verdict-bearing tolerance")
    rule = tolerances.get(RULE)
    if rule != {"rtol": 0.01, "atol": 0.00001}:
        raise EvidenceOnlyError("task-56 source manifest rule differs")
    original = {name: dict(value) for name, value in tolerances.items()}
    changed = {name: dict(value) for name, value in tolerances.items()}
    changed[RULE] = {"rtol": 0.02, "atol": 0.00001}
    common = {"mode": "json_metrics", "prefix": comparison["prefix"]}
    return (
        dict(rule),
        {**common, "tolerances": original},
        {**common, "tolerances": changed},
    )


def _metric_payload(
    payload: bytes, comparison: dict[str, Any], label: str
) -> list[tuple[str, float]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise EvidenceOnlyError(f"{label} is not strict UTF-8") from exc
    prefix = comparison["prefix"]
    lines = [line[len(prefix) :].strip() for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise EvidenceOnlyError(f"{label} has {len(lines)} metric payloads")
    try:
        value = json.loads(
            lines[0],
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceOnlyError(f"invalid metric constant in {label}: {token}")
            ),
        )
    except (json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise EvidenceOnlyError(f"{label} metric payload is invalid: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise EvidenceOnlyError(f"{label} metric payload is not a nonempty object")
    metrics: list[tuple[str, float]] = []
    seen: set[str] = set()

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 128:
            raise EvidenceOnlyError(f"{label} metric nesting is excessive")
        if isinstance(item, bool):
            raise EvidenceOnlyError(f"{label} metric {path!r} is not numeric")
        if isinstance(item, (int, float)):
            try:
                number = float(item)
            except OverflowError as exc:
                raise EvidenceOnlyError(f"{label} metric {path!r} is not finite") from exc
            if not path or not math.isfinite(number) or path in seen:
                raise EvidenceOnlyError(f"{label} metric path/value differs")
            seen.add(path)
            metrics.append((path, number))
            return
        if isinstance(item, list):
            if not item:
                raise EvidenceOnlyError(f"{label} metric array is empty")
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
            return
        if isinstance(item, dict):
            if not item:
                raise EvidenceOnlyError(f"{label} metric object is empty")
            for key, child in item.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or any(marker in key for marker in (".", "[", "]"))
                ):
                    raise EvidenceOnlyError(f"{label} is not eligible for aligned-direct replay")
                visit(child, f"{path}.{key}" if path else key, depth + 1)
            return
        raise EvidenceOnlyError(f"{label} metric {path!r} is not numeric")

    visit(value, "", 1)
    return metrics


def _finite_sum(first: float, second: float) -> float | None:
    if first > sys.float_info.max - second:
        return None
    return first + second


def _finite_product(first: float, second: float) -> float | None:
    if first == 0.0 or second == 0.0:
        return 0.0
    if first > sys.float_info.max / second:
        return None
    return first * second


def _finite_difference(first: float, second: float) -> float | None:
    if (first >= 0.0) == (second >= 0.0):
        return abs(first - second)
    return _finite_sum(abs(first), abs(second))


def _digest_optional(digest: Any, value: float | None) -> None:
    if value is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(struct.pack(">d", value))


def _recompare_metrics(
    cpu_payload: bytes,
    cuda_payload: bytes,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    cpu = _metric_payload(cpu_payload, comparison, "task-56 CPU stdout")
    cuda = _metric_payload(cuda_payload, comparison, "task-56 CUDA stdout")
    if len(cpu) != len(cuda) or any(a[0] != b[0] for a, b in zip(cpu, cuda, strict=True)):
        raise EvidenceOnlyError("task-56 CPU/CUDA aligned metric paths differ")
    metric_digest_cpu = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
    metric_digest_cuda = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
    key_digest = hashlib.sha256(b"gpmeep-json-metric-keys-v1\0")
    comparison_digest = hashlib.sha256(b"gpmeep-json-comparison-records-v1\0")
    used: set[str] = set()
    failure_count = 0
    arithmetic_overflow_count = 0
    zero_tolerance_failure_count = 0
    diagnostic_count = 0
    diagnostic_out = 0
    failures: list[dict[str, Any]] = []
    storage_bytes = 0
    tolerances = comparison["tolerances"]
    for (metric, cpu_value), (_cuda_metric, cuda_value) in zip(cpu, cuda, strict=True):
        encoded = metric.encode("utf-8", errors="strict")
        storage_bytes += len(encoded) + 8
        key_digest.update(len(encoded).to_bytes(8, "big"))
        key_digest.update(encoded)
        for digest, value in ((metric_digest_cpu, cpu_value), (metric_digest_cuda, cuda_value)):
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(struct.pack(">d", value))
        tolerance_rule = metric if metric in tolerances else None
        if tolerance_rule is None:
            matches = [
                name
                for name in tolerances
                if name != "*" and fnmatch.fnmatchcase(metric, name)
            ]
            if len(matches) > 1:
                raise EvidenceOnlyError(f"task-56 metric {metric!r} has ambiguous tolerance")
            tolerance_rule = matches[0] if matches else "*" if "*" in tolerances else None
        if tolerance_rule is None:
            raise EvidenceOnlyError(f"task-56 metric {metric!r} has no tolerance")
        used.add(tolerance_rule)
        tolerance = tolerances[tolerance_rule]
        atol = float(tolerance.get("atol", 0.0))
        rtol = float(tolerance.get("rtol", 0.0))
        diagnostic = bool(tolerance.get("diagnostic_only", False))
        diagnostic_count += int(diagnostic)
        absolute = _finite_difference(cpu_value, cuda_value)
        relative = _finite_product(rtol, max(abs(cpu_value), abs(cuda_value)))
        allowed = _finite_sum(atol, relative) if relative is not None else None
        overflow = absolute is None or allowed is None
        within = (
            not overflow
            and absolute is not None
            and allowed is not None
            and absolute <= allowed
        )
        diagnostic_out += int(diagnostic and not overflow and not within)
        passed = not overflow and (within or diagnostic)
        arithmetic_overflow_count += int(overflow)
        failure_count += int(not passed)
        zero_tolerance_failure_count += int(not passed and allowed == 0)
        normalized_error: float | None = None
        normalized_overflow = False
        if absolute is not None and allowed is not None and allowed > 0:
            normalized_error = absolute / allowed
            if not math.isfinite(normalized_error):
                normalized_error = None
                normalized_overflow = True
        elif absolute == 0 and allowed == 0:
            normalized_error = 0.0
        if not passed:
            failures.append(
                {
                    "metric": metric,
                    "cpu": cpu_value,
                    "cuda": cuda_value,
                    "absolute_error": absolute,
                    "allowed_error": allowed,
                    "arithmetic_overflow": overflow,
                    "normalized_error": normalized_error,
                    "normalized_error_overflow": normalized_overflow,
                }
            )
        for text_value in (metric, tolerance_rule):
            encoded_value = text_value.encode("utf-8", errors="strict")
            comparison_digest.update(len(encoded_value).to_bytes(8, "big"))
            comparison_digest.update(encoded_value)
        for number in (cpu_value, cuda_value, atol, rtol):
            comparison_digest.update(struct.pack(">d", number))
        _digest_optional(comparison_digest, absolute)
        _digest_optional(comparison_digest, allowed)
        if diagnostic:
            comparison_digest.update(b"\x02diagnostic-only-v1\0")
            comparison_digest.update(b"\x01" if within else b"\x00")
        comparison_digest.update(b"\x01" if passed else b"\x00")
    unused = set(tolerances) - {"*"} - used
    if unused:
        raise EvidenceOnlyError("task-56 tolerances reference missing metrics")
    count = len(cpu)
    evidence: dict[str, Any] = {
        "schema": METRIC_EVIDENCE_SCHEMA,
        "digest_order": "document-order",
        "comparison_strategy": "aligned-direct",
        "count": count,
        "passed_count": count - failure_count,
        "failure_count": failure_count,
        "metric_keys_sha256": key_digest.hexdigest(),
        "cpu_sha256": metric_digest_cpu.hexdigest(),
        "cuda_sha256": metric_digest_cuda.hexdigest(),
        "comparison_records_sha256": comparison_digest.hexdigest(),
        "inline_limit": MAX_INLINE_JSON_METRICS,
        "inline_byte_limit": MAX_INLINE_JSON_BYTES,
        "flattened_storage_bytes": storage_bytes,
        "inlined": count <= MAX_INLINE_JSON_METRICS and 2 * storage_bytes <= MAX_INLINE_JSON_BYTES,
    }
    if diagnostic_count:
        evidence.update(
            {
                "diagnostic_only_count": diagnostic_count,
                "diagnostic_only_out_of_tolerance_count": diagnostic_out,
            }
        )
    return {
        "outcome": "PASS" if failure_count == 0 else "MISMATCH",
        "metric_evidence": evidence,
        "failure_count": failure_count,
        "failures": failures,
        "tolerance_group_count": len(used),
        "arithmetic_overflow_count": arithmetic_overflow_count,
        "zero_tolerance_failure_count": zero_tolerance_failure_count,
        "diagnostic_only_metric_count": diagnostic_count,
        "diagnostic_only_out_of_tolerance_count": diagnostic_out,
    }


def _validate_adjudication(
    values: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    commit: str,
    original_failure: dict[str, Any],
    recomparison: dict[str, Any],
) -> dict[str, Any]:
    adjudication = values["task56_adjudication"]
    independent = values["task56_independent_review_json"]
    schema_review = values["task56_schema_review_json"]
    comparison = adjudication.get("adjudication", {}).get("comparison", {})
    decision = adjudication.get("adjudication", {})
    source = adjudication.get("source_evidence", {})
    if (
        adjudication.get("schema")
        != "gpmeep-m3-task56-reciprocity-adjudication-v1"
        or adjudication.get("outcome") != "PASS"
        or adjudication.get("candidate", {}).get("commit") != commit
        or adjudication.get("candidate", {}).get("manifest_sha256")
        != records["source_manifest"]["sha256"]
        or source.get("report_sha256")
        != records["task56_runner_report"]["sha256"]
        or source.get("cpu_stdout", {}).get("sha256")
        != records["task56_cpu_stdout"]["sha256"]
        or source.get("cuda_stdout", {}).get("sha256")
        != records["task56_cuda_stdout"]["sha256"]
        or source.get("original_failure") != original_failure
        or source.get("backend_outcomes") != {"cpu": "PASS", "cuda": "PASS"}
        or decision.get("changed_rule_count") != 1
        or decision.get("rule") != RULE
        or decision.get("old_rule") != {"rtol": 0.01, "atol": 0.00001}
        or decision.get("new_rule") != {"rtol": 0.02, "atol": 0.00001}
        or comparison.get("outcome") != "PASS"
        or comparison.get("failure_count") != 0
        or comparison.get("metric_evidence") != recomparison["metric_evidence"]
        or recomparison["outcome"] != "PASS"
        or recomparison["failure_count"] != 0
        or recomparison["metric_evidence"]["count"] != 5904
        or adjudication.get("release_semantics", {}).get(
            "candidate_manifest_fix_required_for_release"
        )
        is not True
        or adjudication.get("release_semantics", {}).get(
            "candidate_source_modified"
        )
        is not False
    ):
        raise EvidenceOnlyError("task-56 adjudication semantics differ")
    evidence_hashes = independent.get("evidence_sha256", {})
    expected_hashes = {
        "original_report": records["task56_runner_report"]["sha256"],
        "cpu_stdout": records["task56_cpu_stdout"]["sha256"],
        "cuda_stdout": records["task56_cuda_stdout"]["sha256"],
        "candidate_manifest": records["source_manifest"]["sha256"],
        "validation_comparator": records["source_validation_comparator"][
            "sha256"
        ],
        "reciprocity_wrapper": records["source_reciprocity_wrapper"]["sha256"],
        "adjudication_script": records["task56_adjudication_script"]["sha256"],
        "adjudication_receipt": records["task56_adjudication"]["sha256"],
        "markdown_review": records["task56_independent_review_md"]["sha256"],
    }
    if (
        independent.get("schema")
        != "gpmeep-m3-task56-independent-adjudication-review-v1"
        or independent.get("candidate", {}).get("commit") != commit
        or independent.get("verdict", {}).get("m3_resume") != "GO"
        or independent.get("verdict", {}).get("candidate_release") != "BLOCKED"
        or independent.get("full_recomparison", {}).get("outcome") != "PASS"
        or independent.get("full_recomparison", {}).get("metric_count") != 5904
        or independent.get("full_recomparison", {}).get(
            "verdict_failure_count"
        )
        != 0
        or independent.get("full_recomparison", {}).get("tolerance_group_count")
        != recomparison["tolerance_group_count"]
        or independent.get("full_recomparison", {}).get(
            "arithmetic_overflow_count"
        )
        != recomparison["arithmetic_overflow_count"]
        or independent.get("full_recomparison", {}).get(
            "zero_tolerance_failure_count"
        )
        != recomparison["zero_tolerance_failure_count"]
        or independent.get("full_recomparison", {}).get(
            "diagnostic_only_metric_count"
        )
        != recomparison["diagnostic_only_metric_count"]
        or independent.get("full_recomparison", {}).get(
            "diagnostic_only_out_of_tolerance_count"
        )
        != recomparison["diagnostic_only_out_of_tolerance_count"]
        or independent.get("full_recomparison", {}).get(
            "comparison_records_sha256"
        )
        != recomparison["metric_evidence"]["comparison_records_sha256"]
        or any(evidence_hashes.get(key) != value for key, value in expected_hashes.items())
    ):
        raise EvidenceOnlyError("task-56 independent review differs")
    alternative = schema_review.get(
        "narrow_derive_archive_override_alternative", {}
    )
    if (
        schema_review.get("current_strategy_verdict", {}).get("m3_resume")
        != "NO_GO"
        or schema_review.get("current_strategy_verdict", {}).get(
            "candidate_release"
        )
        != "BLOCKED"
        or alternative.get("m3_characterization") != "CONDITIONAL_GO"
        or alternative.get("candidate_release") != "BLOCKED"
        or schema_review.get("original_evidence", {}).get("report_sha256")
        != records["task56_runner_report"]["sha256"]
        or schema_review.get("original_evidence", {}).get("runner_exit_code") != 3
        or schema_review.get("original_evidence", {}).get("selected_outcome")
        != "MISMATCH"
    ):
        raise EvidenceOnlyError("task-56 schema review differs")
    return {
        "changed_rule_count": 1,
        "rule": RULE,
        "old_rule": {"rtol": 0.01, "atol": 0.00001},
        "new_rule": {"rtol": 0.02, "atol": 0.00001},
        "metric_count": 5904,
        "failure_count": 0,
        "independently_recomputed": True,
        "metric_evidence": dict(recomparison["metric_evidence"]),
        "new_meep_execution": False,
        "candidate_manifest_fix_required_for_release": True,
        "independent_review_verdict": {
            "m3_resume": "GO",
            "candidate_release": "BLOCKED",
        },
    }


def _validate_replay(
    replay: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    marker = replay.get("adjudicated_evidence_replay")
    if (
        not isinstance(marker, dict)
        or marker.get("schema") != "gpmeep-m3-task56-telemetry-replay-v1"
        or marker.get("archived_attempt") != "resume-history/attempt-0006"
        or marker.get("original_report_sha256")
        != records["task56_runner_report"]["sha256"]
        or marker.get("adjudication_receipt_sha256")
        != records["task56_adjudication"]["sha256"]
        or marker.get("independent_review_sha256")
        != records["task56_independent_review_json"]["sha256"]
        or marker.get("schema_review_sha256")
        != records["task56_schema_review_json"]["sha256"]
        or marker.get("new_meep_execution") is not False
        or marker.get("candidate_manifest_fix_required_for_release") is not True
        or marker.get("independent_review_verdict")
        != {"m3_resume": "GO", "candidate_release": "BLOCKED"}
    ):
        raise EvidenceOnlyError("task-56 replay marker differs")
    self_contained = marker.get("self_contained_adjudication")
    if not isinstance(self_contained, dict) or set(self_contained) != set(
        ADJUDICATION_SOURCE_NAMES.values()
    ):
        raise EvidenceOnlyError("task-56 adjudication inventory differs")
    inverse = {name: role for role, name in ADJUDICATION_SOURCE_NAMES.items()}
    for name, source_record in self_contained.items():
        role = inverse[name]
        if (
            not isinstance(source_record, dict)
            or source_record.get("sha256") != records[role]["sha256"]
            or source_record.get("size_bytes") != records[role]["size_bytes"]
        ):
            raise EvidenceOnlyError("task-56 adjudication member differs")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive(root: pathlib.Path) -> dict[str, Any]:
    root = _canonical_directory(root, "evidence-only bundle")
    _inventory(root, receipt_present=(root / RECEIPT_NAME).exists())
    records: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for role in sorted(ROLE_PATHS):
        records[role], payloads[role] = _record(root, role)
    running_path = pathlib.Path(__file__)
    running_payload, running_metadata = _stable_bytes(
        running_path, "running evidence-only verifier"
    )
    if hashlib.sha256(running_payload).hexdigest() != records[
        "control_verifier"
    ]["sha256"]:
        raise EvidenceOnlyError("running verifier differs from the bundled verifier")

    json_roles = {
        "audit_draft",
        "paired_terminal",
        "paired_report",
        "paired_checkpoint",
        "paired_journal",
        "task56_runner_report",
        "task56_replay",
        "task56_original_attempt_journal",
        "task56_adjudication",
        "task56_independent_review_json",
        "task56_schema_review_json",
        "source_manifest",
    }
    values = {
        role: _json_payload(payloads[role], role) for role in json_roles
    }
    audit, original_repository, evidence_root = _validate_audit(
        values["audit_draft"],
        payloads["audit_review"],
        records,
    )
    _repository, commit = _draft_source(values["audit_draft"])
    _report, _task, contract = _validate_core(
        values, records, commit, original_repository
    )
    original_result, original_failure = _validate_original_report(
        values["task56_runner_report"], records
    )
    original_execution = _validate_original_execution(
        values["task56_original_attempt_journal"],
        records["task56_original_process_log"],
        contract,
        evidence_root,
    )
    source_rule, source_comparison, changed_comparison = _task56_comparison(
        values["source_manifest"]
    )
    source_recomparison = _recompare_metrics(
        payloads["task56_cpu_stdout"],
        payloads["task56_cuda_stdout"],
        source_comparison,
    )
    if (
        source_recomparison["outcome"] != "MISMATCH"
        or source_recomparison["failure_count"] != 1
        or source_recomparison["failures"] != [original_failure]
        or original_result.get("comparison", {}).get("metric_evidence")
        != source_recomparison["metric_evidence"]
    ):
        raise EvidenceOnlyError(
            "task-56 native comparison is not reproduced from bundled stdout"
        )
    recomparison = _recompare_metrics(
        payloads["task56_cpu_stdout"],
        payloads["task56_cuda_stdout"],
        changed_comparison,
    )
    adjudication = _validate_adjudication(
        values, records, commit, original_failure, recomparison
    )
    _validate_replay(values["task56_replay"], records)
    after_records = {role: _record(root, role)[0] for role in sorted(ROLE_PATHS)}
    if after_records != records:
        raise EvidenceOnlyError("bundle changed during replay")
    running_after, running_after_metadata = _stable_bytes(
        running_path, "running evidence-only verifier"
    )
    running_identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        running_after != running_payload
        or running_identity(running_after_metadata)
        != running_identity(running_metadata)
    ):
        raise EvidenceOnlyError("running verifier changed during replay")
    _inventory(root, receipt_present=(root / RECEIPT_NAME).exists())
    return {
        "schema": SCHEMA,
        "verdicts": {
            "paired_evidence": "GO",
            "candidate_release": "BLOCKED",
            "release_authorized": False,
            "m4_or_release_mutation": "NOT_AUTHORIZED",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "source": {
            "historical_repository_label": original_repository,
            "commit": commit,
        },
        "audit": {
            **audit,
            "historical_evidence": {
                "root_label": evidence_root,
                "terminal_name": "PAIRED_COMPLETE",
            },
        },
        "task56": {
            "case_path": CASE_PATH,
            "unit_id": UNIT_ID,
            "native_runner_outcome": "MISMATCH",
            "native_runner_exit_code": 3,
            "native_failure_count": 1,
            "native_failure_metric": FAILURE_METRIC,
            "source_manifest_rule": source_rule,
            "original_execution": original_execution,
            "adjudication": adjudication,
        },
        "bundle": {
            "portable_relative_paths": True,
            "external_files_required_for_verify": False,
            "file_count": len(records),
            "files": records,
            "bundle_id": _canonical_sha256(records),
        },
    }


def verify(receipt: pathlib.Path) -> dict[str, Any]:
    receipt = pathlib.Path(os.path.abspath(os.fspath(receipt)))
    if receipt.name != RECEIPT_NAME:
        raise EvidenceOnlyError("evidence-only receipt name differs")
    root = _canonical_directory(receipt.parent, "evidence-only bundle")
    receipt_payload, receipt_metadata = _stable_bytes(
        receipt, "evidence-only receipt"
    )
    retained = _json_payload(receipt_payload, "receipt")
    expected = derive(root)
    final_payload, final_metadata = _stable_bytes(
        receipt, "evidence-only receipt"
    )
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        final_payload != receipt_payload
        or identity(final_metadata) != identity(receipt_metadata)
    ):
        raise EvidenceOnlyError("evidence-only receipt changed during replay")
    if retained != expected:
        raise EvidenceOnlyError("evidence-only receipt was not exactly re-derived")
    if (
        retained["verdicts"]["candidate_release"] != "BLOCKED"
        or retained["verdicts"]["release_authorized"] is not False
        or not retained["audit"]["open_material_finding_ids"]
    ):
        raise EvidenceOnlyError("evidence-only release boundary differs")
    return retained


def _write_exclusive(path: pathlib.Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_fixed(source: pathlib.Path, destination: pathlib.Path, label: str) -> None:
    payload, _metadata = _stable_bytes(source, label)
    _write_exclusive(destination, payload)


def _git_blob(repo: pathlib.Path, commit: str, path: str) -> bytes:
    if COMMIT.fullmatch(commit) is None or path not in GIT_SOURCE_PATHS.values():
        raise EvidenceOnlyError("historical source request differs")
    try:
        process = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=repo,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceOnlyError(f"historical source could not be read: {exc}") from exc
    if process.returncode != 0 or len(process.stdout) > MAXIMUM_FILE_BYTES:
        raise EvidenceOnlyError(
            f"historical source {path} is unavailable: "
            + process.stderr.decode("utf-8", errors="replace")[-2048:]
        )
    return process.stdout


def _source_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload, _metadata = _stable_bytes(path, label)
    return _json_payload(payload, label), payload


def prepare(
    *,
    repo: pathlib.Path,
    paired_root: pathlib.Path,
    audit_draft: pathlib.Path,
    audit_review: pathlib.Path,
    output: pathlib.Path,
) -> pathlib.Path:
    repo = _canonical_directory(repo, "source repository")
    paired_root = _canonical_directory(paired_root, "paired evidence root")
    draft, draft_payload = _source_json(audit_draft, "audit draft")
    _finding_inventory(draft)
    repository_label, commit = _draft_source(draft)
    evidence = draft.get("evidence", {})
    if (
        repository_label != repo.as_posix()
        or evidence.get("root") != paired_root.as_posix()
    ):
        raise EvidenceOnlyError("prepare inputs do not match audit draft bindings")
    review_payload, _review_metadata = _stable_bytes(
        audit_review, "full audit review"
    )
    output = pathlib.Path(os.path.abspath(os.fspath(output)))
    parent = _canonical_directory(output.parent, "bundle output parent")
    if (
        output.parent != parent
        or output.exists()
        or output.is_symlink()
        or output.is_relative_to(repo)
        or output.is_relative_to(paired_root)
    ):
        raise EvidenceOnlyError("bundle output must be a fresh canonical path")
    temporary = parent / (
        f".{output.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    temporary.mkdir(mode=0o700)
    try:
        _write_exclusive(
            temporary / ROLE_PATHS["audit_draft"], draft_payload
        )
        _write_exclusive(
            temporary / ROLE_PATHS["audit_review"],
            review_payload,
        )
        for role, name in (
            ("paired_terminal", "PAIRED_COMPLETE"),
            ("paired_report", "report.json"),
            ("paired_checkpoint", "CHECKPOINT.json"),
            ("paired_journal", "JOURNAL.json"),
        ):
            _copy_fixed(
                paired_root / name,
                temporary / ROLE_PATHS[role],
                role,
            )
        original_report_source = (
            paired_root
            / "archives/056-production-mixed-07/runner-output/report.json"
        )
        original_report, original_report_payload = _source_json(
            original_report_source, "task-56 original runner report"
        )
        _write_exclusive(
            temporary / ROLE_PATHS["task56_runner_report"],
            original_report_payload,
        )
        replay_source = (
            paired_root / "telemetry/056-production-mixed-07/replay.json"
        )
        replay, replay_payload = _source_json(replay_source, "task-56 replay")
        _write_exclusive(
            temporary / ROLE_PATHS["task56_replay"], replay_payload
        )
        marker = replay.get("adjudicated_evidence_replay", {})
        attempt_relative = _safe_relative(
            marker.get("archived_attempt"), "task-56 archived attempt"
        )
        attempt = paired_root.joinpath(*attempt_relative.parts)
        attempt_journal_source = attempt / "JOURNAL.json"
        attempt_journal, attempt_journal_payload = _source_json(
            attempt_journal_source, "task-56 original attempt journal"
        )
        _write_exclusive(
            temporary / ROLE_PATHS["task56_original_attempt_journal"],
            attempt_journal_payload,
        )
        events = [
            event
            for event in attempt_journal.get("events", [])
            if event.get("state") == "task-process-ended"
            and event.get("task", {}).get("index") == 56
            and event.get("task", {}).get("unit_id") == UNIT_ID
        ]
        if len(events) != 1:
            raise EvidenceOnlyError("task-56 original journal event differs")
        log_relative = _safe_relative(
            events[0].get("process", {}).get("log", {}).get("path"),
            "task-56 original process log",
        )
        _copy_fixed(
            attempt.joinpath(*log_relative.parts),
            temporary / ROLE_PATHS["task56_original_process_log"],
            "task-56 original process log",
        )
        selected = [
            item
            for item in original_report.get("results", [])
            if item.get("selected")
        ]
        if len(selected) != 1:
            raise EvidenceOnlyError("task-56 selected result differs")
        for backend, role in (
            ("cpu", "task56_cpu_stdout"),
            ("cuda", "task56_cuda_stdout"),
        ):
            run = selected[0].get("runs", {}).get(backend, {})
            stdout_relative = _safe_relative(
                run.get("stdout_log"), f"task-56 {backend} stdout"
            )
            raw = paired_root / "raw/056-production-mixed-07"
            source = raw.joinpath(*stdout_relative.parts)
            payload, metadata = _stable_bytes(source, f"task-56 {backend} stdout")
            expected = run.get("evidence_files", {}).get("stdout", {})
            if (
                expected.get("sha256") != hashlib.sha256(payload).hexdigest()
                or expected.get("size_bytes") != metadata.st_size
            ):
                raise EvidenceOnlyError(f"task-56 {backend} stdout differs")
            _write_exclusive(temporary / ROLE_PATHS[role], payload)
        adjudication_source = (
            paired_root / "telemetry/056-production-mixed-07/adjudication"
        )
        if not adjudication_source.is_dir() or adjudication_source.is_symlink():
            raise EvidenceOnlyError("task-56 adjudication directory differs")
        if {path.name for path in adjudication_source.iterdir()} != set(
            ADJUDICATION_SOURCE_NAMES.values()
        ):
            raise EvidenceOnlyError("task-56 adjudication source inventory differs")
        for role, name in ADJUDICATION_SOURCE_NAMES.items():
            _copy_fixed(
                adjudication_source / name,
                temporary / ROLE_PATHS[role],
                role,
            )
        for role, source_path in GIT_SOURCE_PATHS.items():
            _write_exclusive(
                temporary / ROLE_PATHS[role],
                _git_blob(repo, commit, source_path),
            )
        _copy_fixed(
            pathlib.Path(__file__),
            temporary / ROLE_PATHS["control_verifier"],
            "evidence-only verifier source",
        )
        _inventory(temporary, receipt_present=False)
        receipt = derive(temporary)
        receipt_payload = (
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        _write_exclusive(temporary / RECEIPT_NAME, receipt_payload)
        verify(temporary / RECEIPT_NAME)
        os.replace(temporary, output)
        verify(output / RECEIPT_NAME)
        return output / RECEIPT_NAME
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    create = operations.add_parser("prepare", allow_abbrev=False)
    create.add_argument("--repo", required=True, type=pathlib.Path)
    create.add_argument("--paired-root", required=True, type=pathlib.Path)
    create.add_argument("--audit-draft", required=True, type=pathlib.Path)
    create.add_argument("--audit-review", required=True, type=pathlib.Path)
    create.add_argument("--output", required=True, type=pathlib.Path)
    replay = operations.add_parser("verify", allow_abbrev=False)
    replay.add_argument("--input", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "prepare":
        prepare(
            repo=args.repo,
            paired_root=args.paired_root,
            audit_draft=args.audit_draft,
            audit_review=args.audit_review,
            output=args.output,
        )
    else:
        verify(args.input)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceOnlyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"evidence-only audit error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
