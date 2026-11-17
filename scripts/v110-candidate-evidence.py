#!/usr/bin/env python3
"""Replay role-specific candidate evidence from authoritative raw terminals."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMAS = {
    "targeted": "gpmeep-v110-m4-targeted-validation-v1",
    "full_common": "gpmeep-v110-m4-common-validation-v1",
    "full_validation": "gpmeep-v110-m4-full-validation-v1",
    "feature_route_closure": "gpmeep-v110-m4-feature-route-closure-v1",
    "real_multi_gpu": "gpmeep-v110-m4-real-multi-gpu-v1",
    "aunp_source_candidate": "gpmeep-v110-m4-aunp-source-candidate-v1",
}
PRODUCER = {
    "targeted": "m4_equal_work",
    "full_common": "m4_equal_work",
    "full_validation": "m4_equal_work",
    "feature_route_closure": "m4_equal_work",
    "real_multi_gpu": "m4_equal_work",
    "aunp_source_candidate": "hybrid_aunp",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    _require(isinstance(value, dict), "candidate evidence root must be an object")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "evidence producer is absent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _values(value: Any, key_fragment: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key_fragment in key.lower():
                found.append(item)
            found.extend(_values(item, key_fragment))
    elif isinstance(value, list):
        for item in value:
            found.extend(_values(item, key_fragment))
    return found


def _safe_input(wrapper: pathlib.Path, record: Any, role: str) -> tuple[pathlib.Path, str]:
    _require(
        isinstance(record, dict)
        and set(record) == {"producer", "path", "sha256", "schema", "content_id"},
        f"{role} raw terminal schema differs",
    )
    _require(record["producer"] == PRODUCER[role], f"{role} producer differs")
    relative = pathlib.PurePosixPath(record["path"])
    _require(
        relative.parts
        and not relative.is_absolute()
        and ".." not in relative.parts
        and str(relative) == record["path"],
        f"{role} raw terminal path is unsafe",
    )
    lexical = wrapper.parent.joinpath(*relative.parts)
    _require(not lexical.is_symlink(), f"{role} raw terminal is a symlink")
    path = lexical.resolve(strict=True)
    _require(path == lexical and path.is_file(), f"{role} raw terminal is unsafe")
    _require(HEX64.fullmatch(record["sha256"] or "") is not None and _sha256(path) == record["sha256"], f"{role} raw terminal bytes differ")
    return path, record["content_id"]


def _replay_raw(role: str, path: pathlib.Path) -> dict[str, Any]:
    if PRODUCER[role] == "m4_equal_work":
        producer = _module(
            "gpmeep_v110_candidate_m4_equal_work",
            ROOT / "scripts/python-validation/run_m4_equal_work_matrix.py",
        )
        _require(path.name in {"M4_COMPLETE", "PENDING_M4_COMPLETE"}, "M4 terminal name differs")
        return producer.verify_complete(path.parent, path.name)
    producer = _module(
        "gpmeep_v110_candidate_aunp",
        ROOT / "scripts/user-workloads/run_hybrid_aunp_matrix.py",
    )
    _require(path.name in {"COMPLETE", "PENDING_COMPLETE"}, "AuNP terminal name differs")
    return producer.verify_hybrid_complete(path.parent, path.name)


def _package_receipt_id(role: str, report: dict[str, Any]) -> str:
    if role == "aunp_source_candidate":
        receipt_id = report.get("build", {}).get("receipt_id")
    else:
        receipt_id = report.get("contract", {}).get("runtime_inputs", {}).get(
            "v110_receipt_id"
        )
    _require(
        isinstance(receipt_id, str) and HEX64.fullmatch(receipt_id) is not None,
        f"{role} raw producer package receipt ID differs",
    )
    return receipt_id


def _summary(role: str, report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("performance", {}).get("cases", {})
    comparisons = report.get("numerical", {}).get("comparisons", [])
    uuids = sorted(
        {
            item
            for value in _values(report, "uuid")
            for item in (value if isinstance(value, list) else [value])
            if isinstance(item, str) and item
        }
    )
    fallback_values = [
        value
        for value in _values(report, "cpu_fallback_count")
        if type(value) is int
    ]
    summary = {
        "producer_report_id": report.get("report_id") or _canonical_sha256(report),
        "workload_count": len(cases) if isinstance(cases, dict) else 0,
        "numerical_comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
        "gpu_uuids": uuids,
        "cpu_fallback_count": sum(fallback_values),
    }
    if role in {"full_common", "full_validation", "feature_route_closure"}:
        _require(summary["workload_count"] == 13, f"{role} workload census differs")
    if role in {"targeted", "full_validation", "feature_route_closure"}:
        _require(summary["numerical_comparison_count"] == 141, f"{role} numerical census differs")
    if role == "real_multi_gpu":
        _require(len(uuids) >= 2 and summary["cpu_fallback_count"] == 0, "real multi-GPU proof differs")
    if role == "aunp_source_candidate":
        # ``verify_hybrid_complete`` returns the replayed report rather than
        # its checkpoint contract.  The report's exact public key is
        # ``qualification_profile``; accepting a generic ``profile`` alias
        # would make real AuNP evidence impossible to replay and would also
        # weaken the fixed TM-only contract.
        profile = report.get("qualification_profile")
        _require(
            isinstance(profile, dict)
            and profile.get("mode") == "TM-only"
            and profile.get("dft_minimum_run_time") == 20.0
            and profile.get("dft_decay_tolerance") == 5e-8,
            "AuNP source-candidate profile differs",
        )
    return summary


def verify(role: str, path: pathlib.Path, release_source_commit: str) -> dict[str, Any]:
    _require(role in SCHEMAS, "candidate evidence role has no registered producer")
    lexical = path.absolute()
    _require(not lexical.is_symlink(), "candidate evidence cannot be a symlink")
    path = lexical.resolve(strict=True)
    _require(path == lexical and path.is_file(), "candidate evidence path is unsafe")
    payload = _load(path)
    _require(
        set(payload)
        == {
            "schema", "state", "role", "source_commit", "package_receipt_id",
            "raw_terminal", "summary", "report_id",
        },
        f"{role} evidence schema differs",
    )
    _require(
        payload["schema"] == SCHEMAS[role]
        and payload["state"] == "PASS"
        and payload["role"] == role
        and payload["source_commit"] == release_source_commit
        and HEX40.fullmatch(payload["source_commit"] or "") is not None
        and HEX64.fullmatch(payload["package_receipt_id"] or "") is not None,
        f"{role} evidence identity differs",
    )
    raw_path, raw_content_id = _safe_input(path, payload["raw_terminal"], role)
    report = _replay_raw(role, raw_path)
    _require(
        payload["package_receipt_id"] == _package_receipt_id(role, report),
        f"{role} package receipt ID was not re-derived",
    )
    _require(
        raw_content_id == report.get("terminal_id", report.get("report_id", _canonical_sha256(report))),
        f"{role} raw producer identity differs",
    )
    _require(payload["summary"] == _summary(role, report), f"{role} summary was not re-derived")
    unsigned = dict(payload)
    report_id = unsigned.pop("report_id")
    _require(report_id == _canonical_sha256(unsigned), f"{role} report ID differs")
    return payload
