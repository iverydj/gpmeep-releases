#!/usr/bin/env python3
"""Create or verify the package-owned v1.1.0 candidate-build receipt.

This receipt can authorize construction of candidate bytes.  It deliberately
cannot authorize a Git tag, upload, or package publication.  Publication is
authorized only by an external post-package receipt produced after the exact
package bytes have passed installed qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 5
VALIDATED_ENGINE_COMMIT = "245917cfcd7311d859ade299ed756ea2e4d3d313"
CANDIDATE_MANIFEST_SCHEMA = "gpmeep-v110-m4-candidate-input-v1"
CLOSED_REGISTRY_SCHEMA = "gpmeep-v110-closed-inversion-registry-v1"
M3_EVIDENCE_ONLY_SCHEMA = "gpmeep-evidence-only-adversarial-receipt-v1"
M4_FINDING_IDS = (
    "M3-HIGH-01",
    "M3-MEDIUM-02",
    "M3-MEDIUM-03",
    "M3-HIGH-04",
    "M3-HIGH-05",
    "M3-HIGH-06",
    "M3-MEDIUM-07",
    "M3-COMP-MEDIUM-01",
    "M3-COMP-MEDIUM-02",
    "M3-COMP-MEDIUM-03",
    "M3-COMP-HIGH-04",
)
REQUIRED_EVIDENCE_ROLES = (
    "historical_m3",
    "targeted",
    "full_common",
    "full_validation",
    "host_four",
    "specialized_five",
    "feature_route_closure",
    "real_multi_gpu",
    "aunp_source_candidate",
    "three_way_performance_plan",
    "m4_source_audit",
)
POST_PACKAGE_REQUIREMENTS = (
    "exact_package_provenance",
    "installed_correctness",
    "installed_multi_gpu",
    "installed_aunp",
    "three_way_performance",
    "ubuntu_22_04_native",
    "newer_ubuntu_native",
    "glibc_floor",
    "final_adversarial_audit",
    "final_report",
)
ROLE_SCHEMAS = {
    "historical_m3": M3_EVIDENCE_ONLY_SCHEMA,
    "targeted": "gpmeep-v110-m4-targeted-validation-v1",
    "full_common": "gpmeep-v110-m4-common-validation-v1",
    "full_validation": "gpmeep-v110-m4-full-validation-v1",
    "host_four": "gpmeep-m3-host-matrix-complete-v1",
    "specialized_five": "gpmeep-m3-specialized-complete-v1",
    "feature_route_closure": "gpmeep-v110-m4-feature-route-closure-v1",
    "real_multi_gpu": "gpmeep-v110-m4-real-multi-gpu-v1",
    "aunp_source_candidate": "gpmeep-v110-m4-aunp-source-candidate-v1",
    "three_way_performance_plan": "gpmeep-v110-m4-equal-work-plan-v1",
    "m4_source_audit": "gpmeep-v110-release-source-transition-v3",
}
TYPED_CANDIDATE_ROLES = {
    "targeted",
    "full_common",
    "full_validation",
    "feature_route_closure",
    "real_multi_gpu",
    "aunp_source_candidate",
}


class ReceiptError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _require_isolated_authority() -> None:
    _require(sys.flags.isolated == 1, "candidate authority requires Python isolated mode (-I)")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ReceiptError(f"non-finite JSON constant: {value}")


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == keys, f"{label} schema differs")
    return value


def _regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(resolved.is_file(), f"{label} is not a regular file")
    return resolved


def _regular_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(resolved.is_dir(), f"{label} is not a directory")
    return resolved


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    path = _regular_file(path, "gate input")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _portable_file_record(
    path: pathlib.Path, evidence_root: pathlib.Path
) -> dict[str, Any]:
    path = _regular_file(path, "closed inversion terminal")
    root = _regular_directory(evidence_root, "candidate evidence root")
    _require(path.is_relative_to(root), "closed inversion terminal escaped evidence root")
    relative = path.relative_to(root).as_posix()
    _require(relative not in {"", "."}, "closed inversion terminal path is not portable")
    return {
        "bundle_relative_path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _identity(path: pathlib.Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _run_isolated_source_verifier(
    *,
    report_path: pathlib.Path,
    report_payload: dict[str, Any],
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
) -> dict[str, Any]:
    """Run the external verifier before any repository producer is imported."""
    interpreter = pathlib.Path(sys.executable).resolve(strict=True)
    _require(interpreter.is_absolute() and interpreter.is_file(), "isolated Python interpreter is unsafe")
    interpreter_identity = _identity(interpreter)
    source_verifier_path = _regular_file(source_verifier_path, "external M4 source verifier")
    _require(not source_verifier_path.is_relative_to(ROOT), "M4 source verifier must be outside the release repository")
    verifier_identity = _identity(source_verifier_path)
    verifier_bytes = source_verifier_path.read_bytes()
    _require(
        hashlib.sha256(verifier_bytes).hexdigest() == expected_source_verifier_sha256,
        "external M4 source verifier SHA-256 differs",
    )
    policy_payload = _load(source_policy_path)
    policy_verifier = _exact(
        policy_payload.get("verifier"), {"path", "mode", "blob"}, "external policy verifier"
    )
    _require(
        policy_verifier["path"] == "scripts/verify-v110-release-source-transition.py"
        and policy_verifier["mode"] in {"100644", "100755"}
        and policy_verifier["blob"] == _git_blob_sha1(verifier_bytes),
        "external source verifier bytes differ from independently pinned policy",
    )
    _require_isolated_authority()
    command = [
        str(interpreter), "-I", "-S", str(source_verifier_path),
        "--verify", str(report_path), "--repo", str(ROOT),
        "--authority-policy", str(source_policy_path),
        "--authority-policy-sha256", expected_source_policy_sha256,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd="/",
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ReceiptError("isolated M4 source verifier timed out") from error
    _require(
        completed.returncode == 0,
        "isolated M4 source verifier rejected the transition: "
        + completed.stderr.strip()[:1000],
    )
    try:
        terminal = json.loads(completed.stdout, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ReceiptError("isolated M4 source verifier output is invalid") from error
    _require(
        terminal
        == {
            "schema": ROLE_SCHEMAS["m4_source_audit"],
            "state": "PASS",
            "report_id": report_payload.get("report_id"),
        },
        "isolated M4 source verifier terminal differs",
    )
    _require(
        _identity(interpreter) == interpreter_identity
        and _identity(source_verifier_path) == verifier_identity
        and source_verifier_path.read_bytes() == verifier_bytes,
        "isolated source verification authority changed during replay",
    )
    return report_payload


def _preflight_source_authority(
    *,
    release_source_commit: str,
    candidate_manifest_path: pathlib.Path,
    evidence_root: pathlib.Path,
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
) -> dict[str, Any]:
    """Validate the source boundary before loading mutable repository producers."""
    evidence_root = _regular_directory(evidence_root, "candidate evidence root")
    manifest_path = _regular_file(candidate_manifest_path, "M4 candidate manifest")
    _require(manifest_path.is_relative_to(evidence_root), "M4 candidate manifest is outside evidence root")
    manifest = _load(manifest_path)
    _require(
        manifest.get("release_source_commit") == release_source_commit
        and isinstance(manifest.get("evidence"), dict),
        "M4 preflight manifest identity differs",
    )
    record = _exact(
        manifest["evidence"].get("m4_source_audit"),
        {"role", "bundle_relative_path", "size_bytes", "sha256", "schema", "content_id"},
        "M4 source preflight evidence",
    )
    relative = pathlib.PurePosixPath(record["bundle_relative_path"])
    _require(
        record["role"] == "m4_source_audit"
        and record["schema"] == ROLE_SCHEMAS["m4_source_audit"]
        and relative.parts
        and not relative.is_absolute()
        and ".." not in relative.parts,
        "M4 source preflight reference differs",
    )
    report_path = _regular_file(
        evidence_root.joinpath(*relative.parts), "M4 source preflight report"
    )
    _require(report_path.is_relative_to(evidence_root), "M4 source preflight report escaped evidence root")
    _require(
        type(record["size_bytes"]) is int
        and report_path.stat().st_size == record["size_bytes"]
        and _sha256(report_path) == record["sha256"],
        "M4 source preflight report bytes differ",
    )
    report = _load(report_path)
    _require(
        report.get("schema") == ROLE_SCHEMAS["m4_source_audit"]
        and report.get("release_source_commit") == release_source_commit
        and report.get("report_id") == record["content_id"],
        "M4 source preflight report identity differs",
    )
    return _run_isolated_source_verifier(
        report_path=report_path,
        report_payload=report,
        source_policy_path=source_policy_path,
        expected_source_policy_sha256=expected_source_policy_sha256,
        source_verifier_path=source_verifier_path,
        expected_source_verifier_sha256=expected_source_verifier_sha256,
    )


def _gate(path: pathlib.Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _regular_file(path, label)
    value = _load(resolved)
    _require(value.get("state") in {"PASS", "OPEN"}, f"{label} state is invalid")
    return value, _file_record(resolved)


def _replay_native_evidence(
    role: str,
    path: pathlib.Path,
    payload: dict[str, Any],
    release_source_commit: str,
    *,
    source_policy_path: pathlib.Path | None,
    expected_source_policy_sha256: str | None,
    source_verifier_path: pathlib.Path | None,
    expected_source_verifier_sha256: str | None,
) -> str:
    """Replay the authoritative producer where one exists.

    The remaining summary records are accepted only as inputs to the independently
    replayed M4 audit and only when they have their role-specific schema.  They are
    never accepted as generic ``state=PASS`` documents.
    """
    if role == "historical_m3":
        producer = _load_module(
            "gpmeep_v110_historical_m3_replay",
            ROOT / "scripts/python-validation/seal_evidence_only_adversarial_audit.py",
        )
        replayed = producer.verify(path)
        _require(replayed == payload, "historical M3 receipt was not exactly re-derived")
        return _canonical_sha256(replayed)
    if role == "three_way_performance_plan":
        producer = _load_module(
            "gpmeep_v110_m4_plan_replay",
            ROOT / "scripts/python-validation/m4_equal_work_plan.py",
        )
        replayed = producer.load_plan(ROOT)
        _require(replayed == payload, "three-way performance plan differs from the official plan")
        return _canonical_sha256(replayed)
    if role == "host_four":
        producer = _load_module(
            "gpmeep_v110_m3_host_replay",
            ROOT / "scripts/python-validation/run_m3_host_matrix.py",
        )
        report = producer.verify_complete(path.parent, path.name)
        _require(len(report["tasks"]) == 4, "host-four producer census differs")
        # The role references the verified COMPLETE artifact, not its report.
        return _canonical_sha256(payload)
    if role == "specialized_five":
        producer = _load_module(
            "gpmeep_v110_m3_specialized_replay",
            ROOT / "scripts/python-validation/run_m3_specialized_matrix.py",
        )
        report = producer.verify_complete(path.parent, path.name)
        _require(len(report["tasks"]) == 5, "specialized-five producer census differs")
        return _canonical_sha256(payload)
    if role == "m4_source_audit":
        _require(source_policy_path is not None, "M4 source replay lacks independently supplied policy")
        _require(
            HEX64.fullmatch(expected_source_policy_sha256 or "") is not None,
            "M4 source replay lacks independently supplied policy SHA-256",
        )
        _require(source_verifier_path is not None, "M4 source replay lacks independently supplied verifier")
        _require(
            HEX64.fullmatch(expected_source_verifier_sha256 or "") is not None,
            "M4 source replay lacks independently supplied verifier SHA-256",
        )
        retained_policy = _exact(
            payload.get("external_policy"),
            {"path", "mode", "size_bytes", "sha256", "policy_id"},
            "M4 external source policy",
        )
        source_policy_path = _regular_file(source_policy_path, "external M4 source policy")
        _require(
            str(source_policy_path) == retained_policy["path"],
            "M4 source report policy path differs from independent authority",
        )
        _require(
            _sha256(source_policy_path) == expected_source_policy_sha256
            and retained_policy["sha256"] == expected_source_policy_sha256,
            "M4 source report policy hash differs from independent authority",
        )
        replayed = _run_isolated_source_verifier(
            report_path=path,
            report_payload=payload,
            source_policy_path=source_policy_path,
            expected_source_policy_sha256=expected_source_policy_sha256,
            source_verifier_path=source_verifier_path,
            expected_source_verifier_sha256=expected_source_verifier_sha256,
        )
        _require(replayed == payload, "M4 source audit was not exactly re-derived")
        _require(
            replayed.get("validated_engine_commit") == VALIDATED_ENGINE_COMMIT
            and replayed.get("release_source_commit") == release_source_commit
            and HEX40.fullmatch(replayed.get("measured_source_commit", "")) is not None
            and replayed.get("ancestor") is True
            and replayed.get("protected_changes") == [],
            "M4 source transition boundary differs",
        )
        return replayed["report_id"]
    if role in TYPED_CANDIDATE_ROLES:
        producer = _load_module(
            "gpmeep_v110_typed_candidate_evidence",
            ROOT / "scripts/v110-candidate-evidence.py",
        )
        try:
            replayed = producer.verify(role, path, release_source_commit)
        except Exception as error:
            raise ReceiptError(
                f"evidence.{role} producer replay failed: {error}"
            ) from error
        _require(replayed == payload, f"evidence.{role} was not exactly re-derived")
        return replayed["report_id"]
    raise ReceiptError(f"evidence.{role} has no registered producer replay")


def _validate_artifact_ref(
    value: Any,
    role: str,
    evidence_root: pathlib.Path,
    attested_files: set[tuple[str, int, str]],
    validated_engine_commit: str,
    release_source_commit: str,
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
) -> str | None:
    record = _exact(
        value,
        {"role", "bundle_relative_path", "size_bytes", "sha256", "schema", "content_id"},
        f"evidence.{role}",
    )
    _require(record["role"] == role, f"evidence.{role} role differs")
    relative = pathlib.PurePosixPath(record["bundle_relative_path"])
    _require(
        not relative.is_absolute()
        and relative.parts
        and ".." not in relative.parts
        and str(relative) == record["bundle_relative_path"],
        f"evidence.{role} path is unsafe",
    )
    _require(type(record["size_bytes"]) is int and record["size_bytes"] > 0, f"evidence.{role} size is invalid")
    _require(HEX64.fullmatch(record["sha256"] or "") is not None, f"evidence.{role} hash is invalid")
    _require(record["schema"] == ROLE_SCHEMAS[role], f"evidence.{role} producer schema differs")
    _require(HEX64.fullmatch(record["content_id"] or "") is not None, f"evidence.{role} content ID is invalid")
    path = evidence_root.joinpath(*relative.parts)
    resolved = _regular_file(path, f"evidence.{role}")
    artifact_identity = _identity(resolved)
    _require(resolved.is_relative_to(evidence_root), f"evidence.{role} escaped evidence root")
    _require(resolved.stat().st_size == record["size_bytes"], f"evidence.{role} size differs")
    _require(_sha256(resolved) == record["sha256"], f"evidence.{role} hash differs")
    _require(
        (str(resolved), record["size_bytes"], record["sha256"]) in attested_files,
        f"evidence.{role} is absent from the replayed attestation",
    )
    payload = _load(resolved)
    _require(payload.get("schema") == record["schema"], f"evidence.{role} schema differs")
    package_receipt_id = None
    if role in TYPED_CANDIDATE_ROLES:
        _require(payload.get("source_commit") == release_source_commit, f"evidence.{role} release source commit differs")
        package_receipt_id = payload.get("package_receipt_id")
        _require(
            isinstance(package_receipt_id, str)
            and HEX64.fullmatch(package_receipt_id) is not None,
            f"evidence.{role} development package receipt ID is invalid",
        )
    identity = _replay_native_evidence(
        role,
        resolved,
        payload,
        release_source_commit,
        source_policy_path=source_policy_path,
        expected_source_policy_sha256=expected_source_policy_sha256,
        source_verifier_path=source_verifier_path,
        expected_source_verifier_sha256=expected_source_verifier_sha256,
    )
    _require(identity == record["content_id"], f"evidence.{role} content ID differs")
    if role == "historical_m3":
        _require(record["schema"] == M3_EVIDENCE_ONLY_SCHEMA, "historical M3 schema differs")
        verdicts = payload.get("verdicts")
        _require(
            isinstance(verdicts, dict)
            and verdicts.get("paired_evidence") == "GO"
            and verdicts.get("candidate_release") == "BLOCKED"
            and verdicts.get("release_authorized") is False,
            "historical M3 release boundary differs",
        )
    _require(
        _identity(resolved) == artifact_identity,
        f"evidence.{role} changed during verification",
    )
    return package_receipt_id


def _validate_candidate_manifest(
    value: Any,
    release_source_commit: str,
    evidence_root: pathlib.Path,
    attested_files: set[tuple[str, int, str]],
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
) -> None:
    manifest = _exact(
        value,
        {"schema", "state", "validated_engine_commit", "release_source_commit", "option", "findings", "validation", "evidence"},
        "M4 candidate manifest",
    )
    _require(manifest["schema"] == CANDIDATE_MANIFEST_SCHEMA, "M4 candidate manifest schema differs")
    _require(manifest["state"] == "PASS", "M4 candidate manifest did not pass")
    _require(manifest["validated_engine_commit"] == VALIDATED_ENGINE_COMMIT, "M4 validated engine commit differs")
    _require(manifest["release_source_commit"] == release_source_commit, "M4 release source commit differs")
    _require(manifest["option"] == "A_ALL_11", "M4 option A was not fully selected")
    findings = _exact(
        manifest["findings"],
        {"required_ids", "closed_ids", "open_critical_high_medium"},
        "M4 findings",
    )
    expected_findings = list(M4_FINDING_IDS)
    _require(findings["required_ids"] == expected_findings, "M4 finding inventory differs")
    _require(findings["closed_ids"] == expected_findings, "not every M4 option-A finding is closed")
    _require(findings["open_critical_high_medium"] == 0, "M4 has an open material finding")
    validation = _exact(
        manifest["validation"],
        {
            "targeted",
            "full_common",
            "full_validation",
            "host_case_count",
            "specialized_case_count",
            "feature_count",
            "route_count",
            "real_multi_gpu",
            "no_cpu_fallback",
            "aunp_source_candidate",
        },
        "M4 validation summary",
    )
    for key in ("targeted", "full_common", "full_validation", "real_multi_gpu", "no_cpu_fallback", "aunp_source_candidate"):
        _require(validation[key] is True, f"M4 validation {key} did not pass")
    _require(validation["host_case_count"] == 4, "M4 host coverage differs")
    _require(validation["specialized_case_count"] == 5, "M4 specialized coverage differs")
    _require(validation["feature_count"] == 42, "M4 feature coverage differs")
    _require(validation["route_count"] == 32, "M4 route coverage differs")
    evidence = _exact(manifest["evidence"], set(REQUIRED_EVIDENCE_ROLES), "M4 evidence")
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    typed_package_receipt_ids: set[str] = set()
    for role in REQUIRED_EVIDENCE_ROLES:
        record = evidence[role]
        _require(record["bundle_relative_path"] not in seen_paths, "M4 evidence aliases a path")
        _require(record["sha256"] not in seen_hashes, "M4 evidence reuses bytes across roles")
        seen_paths.add(record["bundle_relative_path"])
        seen_hashes.add(record["sha256"])
        package_receipt_id = _validate_artifact_ref(
            record,
            role,
            evidence_root,
            attested_files,
            VALIDATED_ENGINE_COMMIT,
            release_source_commit,
            source_policy_path,
            expected_source_policy_sha256,
            source_verifier_path,
            expected_source_verifier_sha256,
        )
        if package_receipt_id is not None:
            typed_package_receipt_ids.add(package_receipt_id)
    _require(
        len(typed_package_receipt_ids) == 1,
        "typed candidate evidence does not bind one development package receipt ID",
    )


def _replay_contract_gate(path: pathlib.Path) -> dict[str, Any]:
    producer = _load_module(
        "gpmeep_v110_candidate_contract_replay",
        ROOT / "scripts/validate-v110-development-contract.py",
    )
    replayed = producer.validate(
        ROOT / "packaging/v110-development-contract.json",
        workspace_root=ROOT.parent,
        require_user_inputs=False,
        validate_source_files=True,
    )
    _require(_load(path) == replayed, "contract gate was not exactly re-derived")
    return replayed


def _replay_registry_gate(
    path: pathlib.Path,
    evidence_root: pathlib.Path,
    records_directory: pathlib.Path,
) -> dict[str, Any]:
    producer = _load_module(
        "gpmeep_v110_candidate_registry_replay",
        ROOT / "scripts/validate-v110-inversion-registry.py",
    )
    replayed = producer.validate_registry(
        require_closed=True,
        evidence_root=evidence_root,
        records_directory=records_directory,
    )
    _require(_load(path) == replayed, "registry gate was not exactly re-derived")
    return replayed


def _embedded_closed_registry(
    value: Any,
    *,
    inventory_sha256: str,
) -> dict[str, Any]:
    authority = _exact(value, {"file", "terminal"}, "closed inversion registry")
    file_record = _exact(
        authority["file"],
        {"bundle_relative_path", "sha256", "size_bytes"},
        "closed inversion terminal file",
    )
    _require(
        isinstance(file_record["bundle_relative_path"], str),
        "closed inversion terminal file path is invalid",
    )
    relative_file = pathlib.PurePosixPath(file_record["bundle_relative_path"])
    _require(
        relative_file.parts
        and not relative_file.is_absolute()
        and ".." not in relative_file.parts
        and str(relative_file) == file_record["bundle_relative_path"]
        and isinstance(file_record["sha256"], str)
        and HEX64.fullmatch(file_record["sha256"] or "") is not None
        and type(file_record["size_bytes"]) is int
        and file_record["size_bytes"] > 0,
        "closed inversion terminal file record is invalid",
    )
    terminal = _exact(
        authority["terminal"],
        {
            "schema", "state", "records_directory", "records",
            "registry_result", "terminal_id",
        },
        "closed inversion terminal",
    )
    unsigned = dict(terminal)
    terminal_id = unsigned.pop("terminal_id", None)
    _require(
        terminal["schema"] == CLOSED_REGISTRY_SCHEMA
        and terminal["state"] == "PASS"
        and isinstance(terminal_id, str)
        and HEX64.fullmatch(terminal_id or "") is not None
        and _canonical_sha256(unsigned) == terminal_id,
        "closed inversion terminal identity differs",
    )
    _require(
        isinstance(terminal["records_directory"], str),
        "closed inversion records directory is invalid",
    )
    relative_records = pathlib.PurePosixPath(terminal["records_directory"])
    _require(
        terminal["records_directory"] not in {"", "."}
        and relative_records.parts
        and not relative_records.is_absolute()
        and ".." not in relative_records.parts
        and str(relative_records) == terminal["records_directory"],
        "closed inversion records directory is unsafe",
    )
    records = terminal["records"]
    _require(
        isinstance(records, list) and len(records) == 13,
        "closed inversion record census differs",
    )
    expected_paths: list[str] = []
    seen_hashes: set[str] = set()
    for index, item in enumerate(records):
        record = _exact(
            item,
            {"path", "size_bytes", "sha256"},
            f"closed inversion record[{index}]",
        )
        _require(
            isinstance(record["path"], str),
            f"closed inversion record[{index}] path is invalid",
        )
        relative = pathlib.PurePosixPath(record["path"])
        _require(
            relative.parts
            and not relative.is_absolute()
            and ".." not in relative.parts
            and str(relative) == record["path"]
            and relative.parent == relative_records
            and relative.suffix == ".json"
            and type(record["size_bytes"]) is int
            and record["size_bytes"] > 0
            and isinstance(record["sha256"], str)
            and HEX64.fullmatch(record["sha256"] or "") is not None,
            f"closed inversion record[{index}] is invalid",
        )
        _require(record["sha256"] not in seen_hashes, "closed inversion records reuse bytes")
        expected_paths.append(record["path"])
        seen_hashes.add(record["sha256"])
    _require(
        expected_paths == sorted(expected_paths)
        and len(set(expected_paths)) == 13
        and file_record["bundle_relative_path"] not in set(expected_paths),
        "closed inversion record path inventory differs",
    )
    result = _exact(
        terminal["registry_result"],
        {
            "schema_version", "state", "inventory_sha256", "observed_count",
            "registered_count", "missing_count", "blocking_count",
            "closed_count", "open_count", "missing_workloads",
            "blocking_workloads", "open_workloads",
        },
        "closed inversion registry result",
    )
    _require(
        result
        == {
            "schema_version": 1,
            "state": "PASS",
            "inventory_sha256": inventory_sha256,
            "observed_count": 13,
            "registered_count": 13,
            "missing_count": 0,
            "blocking_count": 0,
            "closed_count": 13,
            "open_count": 0,
            "missing_workloads": [],
            "blocking_workloads": [],
            "open_workloads": [],
        },
        "closed inversion registry result differs",
    )
    return authority


def _replay_closed_registry(
    *,
    terminal_path: pathlib.Path,
    evidence_root: pathlib.Path,
    records_directory: pathlib.Path,
    expected_sha256: str,
    expected_terminal_id: str,
    inventory_sha256: str,
) -> tuple[dict[str, Any], pathlib.Path]:
    _require(
        terminal_path.is_absolute() and records_directory.is_absolute(),
        "closed inversion terminal and records directory must be absolute",
    )
    _require(
        isinstance(expected_sha256, str)
        and isinstance(expected_terminal_id, str)
        and HEX64.fullmatch(expected_sha256) is not None
        and HEX64.fullmatch(expected_terminal_id) is not None,
        "closed inversion terminal pins are invalid",
    )
    root = _regular_directory(evidence_root, "candidate evidence root")
    terminal_path = _regular_file(terminal_path, "closed inversion terminal")
    records_directory = _regular_directory(
        records_directory, "closed inversion records directory"
    )
    _require(
        terminal_path.is_relative_to(root) and terminal_path != root,
        "closed inversion terminal escaped evidence root",
    )
    _require(
        records_directory.is_relative_to(root) and records_directory != root,
        "closed inversion records directory escaped evidence root",
    )
    terminal_identity = _identity(terminal_path)
    records_identity = _identity(records_directory)
    _require(_sha256(terminal_path) == expected_sha256, "closed inversion terminal SHA-256 differs")
    closer = _load_module(
        "gpmeep_v110_candidate_closed_registry_replay",
        ROOT / "scripts/close-v110-inversion-registry.py",
    )
    try:
        terminal = closer.verify(terminal_path, root)
    except Exception as error:
        raise ReceiptError(f"closed inversion terminal replay failed: {error}") from error
    _require(terminal.get("terminal_id") == expected_terminal_id, "closed inversion terminal ID differs")
    relative_records = pathlib.PurePosixPath(terminal.get("records_directory", ""))
    _require(
        relative_records.parts
        and not relative_records.is_absolute()
        and ".." not in relative_records.parts
        and str(relative_records) == terminal.get("records_directory"),
        "closed inversion terminal records directory is unsafe",
    )
    derived_records = _regular_directory(
        root.joinpath(*relative_records.parts),
        "terminal-derived closed inversion records directory",
    )
    _require(
        derived_records == records_directory,
        "explicit closed inversion records directory differs from terminal",
    )
    record_identities: dict[str, tuple[int, int, int, int, int]] = {}
    for index, record in enumerate(terminal.get("records", [])):
        relative_record = pathlib.PurePosixPath(record.get("path", ""))
        record_path = _regular_file(
            root.joinpath(*relative_record.parts),
            f"closed inversion record[{index}]",
        )
        _require(
            record_path.is_relative_to(records_directory)
            and record_path.parent == records_directory
            and record_path.stat().st_size == record.get("size_bytes")
            and _sha256(record_path) == record.get("sha256"),
            f"closed inversion record[{index}] bytes differ",
        )
        record_identities[record["path"]] = _identity(record_path)
    authority = {
        "file": _portable_file_record(terminal_path, root),
        "terminal": terminal,
    }
    _embedded_closed_registry(authority, inventory_sha256=inventory_sha256)
    _require(
        _identity(terminal_path) == terminal_identity
        and _identity(records_directory) == records_identity
        and _sha256(terminal_path) == expected_sha256,
        "closed inversion authority changed during replay",
    )
    for index, record in enumerate(terminal["records"]):
        record_path = _regular_file(
            root.joinpath(*pathlib.PurePosixPath(record["path"]).parts),
            f"closed inversion record[{index}]",
        )
        _require(
            _identity(record_path) == record_identities[record["path"]]
            and record_path.stat().st_size == record["size_bytes"]
            and _sha256(record_path) == record["sha256"],
            f"closed inversion record[{index}] changed during replay",
        )
    return authority, records_directory


def _require_closed_registry_attested(
    authority: dict[str, Any],
    evidence_root: pathlib.Path,
    attested_files: set[tuple[str, int, str]],
) -> None:
    root = _regular_directory(evidence_root, "candidate evidence root")
    terminal_file = authority["file"]
    terminal_path = _regular_file(
        root.joinpath(*pathlib.PurePosixPath(terminal_file["bundle_relative_path"]).parts),
        "closed inversion terminal",
    )
    _require(
        (str(terminal_path), terminal_file["size_bytes"], terminal_file["sha256"])
        in attested_files,
        "closed inversion terminal is absent from the replayed attestation",
    )
    for index, record in enumerate(authority["terminal"]["records"]):
        path = _regular_file(
            root.joinpath(*pathlib.PurePosixPath(record["path"]).parts),
            f"closed inversion record[{index}]",
        )
        _require(
            (str(path), record["size_bytes"], record["sha256"]) in attested_files,
            f"closed inversion record[{index}] is absent from the replayed attestation",
        )


def _replay_attestation_gate(
    path: pathlib.Path, expected_sha256: str
) -> dict[str, Any]:
    producer = _load_module(
        "gpmeep_v110_candidate_attestation_replay",
        ROOT / "scripts/attest-v110-m0.py",
    )
    attested = producer._verify(path, expected_sha256)
    _require(_load(path) == attested, "attestation payload was not exactly re-derived")
    return attested


def create(
    *,
    mode: str,
    source_commit: str,
    contract_gate_path: pathlib.Path,
    registry_gate_path: pathlib.Path,
    attestation_gate_path: pathlib.Path | None = None,
    expected_attestation_sha256: str | None = None,
    candidate_manifest_path: pathlib.Path | None = None,
    evidence_root: pathlib.Path | None = None,
    source_policy_path: pathlib.Path | None = None,
    expected_source_policy_sha256: str | None = None,
    source_verifier_path: pathlib.Path | None = None,
    expected_source_verifier_sha256: str | None = None,
    closed_registry_terminal_path: pathlib.Path | None = None,
    expected_closed_registry_sha256: str | None = None,
    expected_closed_registry_terminal_id: str | None = None,
    closed_registry_records_directory: pathlib.Path | None = None,
) -> dict[str, Any]:
    _require(mode in {"development", "candidate"}, "build mode is invalid")
    _require(HEX40.fullmatch(source_commit) is not None, "source commit is invalid")
    preflight_source: dict[str, Any] | None = None
    if mode == "candidate":
        _require_isolated_authority()
        _require(
            candidate_manifest_path is not None
            and evidence_root is not None
            and source_policy_path is not None
            and source_verifier_path is not None,
            "candidate source preflight inputs are incomplete",
        )
        _require(
            closed_registry_terminal_path is not None
            and expected_closed_registry_sha256 is not None
            and expected_closed_registry_terminal_id is not None
            and closed_registry_records_directory is not None,
            "candidate build lacks exact closed inversion registry authority",
        )
        _require(
            HEX64.fullmatch(expected_source_policy_sha256 or "") is not None
            and HEX64.fullmatch(expected_source_verifier_sha256 or "") is not None,
            "candidate source preflight hashes are invalid",
        )
        source_policy_path = _regular_file(source_policy_path, "external M4 source policy")
        source_verifier_path = _regular_file(source_verifier_path, "external M4 source verifier")
        _require(
            not source_policy_path.is_relative_to(ROOT)
            and not source_verifier_path.is_relative_to(ROOT),
            "candidate source authority must be outside the release repository",
        )
        _require(
            _sha256(source_policy_path) == expected_source_policy_sha256
            and _sha256(source_verifier_path) == expected_source_verifier_sha256,
            "candidate source authority SHA-256 differs",
        )
        preflight_source = _preflight_source_authority(
            release_source_commit=source_commit,
            candidate_manifest_path=candidate_manifest_path,
            evidence_root=evidence_root,
            source_policy_path=source_policy_path,
            expected_source_policy_sha256=expected_source_policy_sha256,
            source_verifier_path=source_verifier_path,
            expected_source_verifier_sha256=expected_source_verifier_sha256,
        )
    contract, contract_file = _gate(contract_gate_path, "contract gate")
    registry, registry_file = _gate(registry_gate_path, "registry gate")
    _require(_replay_contract_gate(contract_gate_path) == contract, "contract producer replay differs")
    _require(contract.get("state") == "PASS", "contract gate did not pass")
    _require(contract.get("release_train") == "1.1.0", "contract release train differs")
    for key in ("contract_sha256", "contract_canonical_sha256"):
        _require(HEX64.fullmatch(contract.get(key, "")) is not None, f"contract gate {key} is invalid")
    _require(HEX64.fullmatch(registry.get("inventory_sha256", "")) is not None, "registry inventory hash is invalid")

    attestation = None
    attestation_file = None
    manifest = None
    manifest_file = None
    source_policy_record = None
    source_verifier_record = None
    measured_source_commit = None
    closed_registry = None
    if mode == "candidate":
        _require(evidence_root is not None, "candidate build lacks an evidence root")
        evidence_root = _regular_directory(evidence_root, "candidate evidence root")
        closed_registry, closed_registry_records_directory = _replay_closed_registry(
            terminal_path=closed_registry_terminal_path,
            evidence_root=evidence_root,
            records_directory=closed_registry_records_directory,
            expected_sha256=expected_closed_registry_sha256,
            expected_terminal_id=expected_closed_registry_terminal_id,
            inventory_sha256=registry["inventory_sha256"],
        )
        _require(
            _replay_registry_gate(
                registry_gate_path,
                evidence_root,
                closed_registry_records_directory,
            )
            == registry,
            "registry producer replay differs",
        )
        _require(registry.get("state") == "PASS", "candidate registry is not closed")
        _require(registry.get("missing_count") == 0, "candidate registry has missing records")
        _require(registry.get("blocking_count") == 0, "candidate registry has blockers")
        _require(registry.get("closed_count") == 13, "candidate registry is not fully closed")
        _require(registry.get("open_count") == 0, "candidate registry has open records")
        _require(registry.get("open_workloads") == [], "candidate registry open inventory differs")
        _require(attestation_gate_path is not None, "candidate build lacks an attestation gate")
        _require(
            expected_attestation_sha256 is not None
            and HEX64.fullmatch(expected_attestation_sha256) is not None,
            "candidate build lacks an exact attestation SHA-256",
        )
        attestation_path = _regular_file(attestation_gate_path, "attestation gate")
        attestation_identity = _identity(attestation_path)
        _require(_sha256(attestation_path) == expected_attestation_sha256, "attestation does not bind the expected SHA-256")
        attestation = _replay_attestation_gate(attestation_path, expected_attestation_sha256)
        attestation_file = _file_record(attestation_path)
        repository = attestation.get("repository", {})
        _require(HEX40.fullmatch(repository.get("m0_commit", "")) is not None, "attested M0 commit is invalid")
        _require(repository.get("current_head") == source_commit, "attestation source commit differs")
        _require(repository.get("clean") is True, "attested source tree is dirty")
        _require(candidate_manifest_path is not None, "candidate build lacks an M4 candidate manifest")
        _require(source_policy_path is not None, "candidate build lacks an external M4 source policy")
        _require(
            HEX64.fullmatch(expected_source_policy_sha256 or "") is not None,
            "candidate build lacks an exact external M4 source policy SHA-256",
        )
        source_policy_path = _regular_file(source_policy_path, "external M4 source policy")
        _require(
            _sha256(source_policy_path) == expected_source_policy_sha256,
            "external M4 source policy does not bind the expected SHA-256",
        )
        _require(
            not source_policy_path.is_relative_to(ROOT),
            "external M4 source policy must be outside the release repository",
        )
        _require(source_verifier_path is not None, "candidate build lacks an external M4 source verifier")
        _require(
            HEX64.fullmatch(expected_source_verifier_sha256 or "") is not None,
            "candidate build lacks an exact external M4 source verifier SHA-256",
        )
        source_verifier_path = _regular_file(source_verifier_path, "external M4 source verifier")
        _require(not source_verifier_path.is_relative_to(ROOT), "external M4 source verifier must be outside the release repository")
        _require(_sha256(source_verifier_path) == expected_source_verifier_sha256, "external M4 source verifier SHA-256 differs")
        manifest_path = _regular_file(candidate_manifest_path, "M4 candidate manifest")
        _require(manifest_path.is_relative_to(evidence_root), "M4 candidate manifest is outside evidence root")
        attested_files = {
            (item.get("path"), item.get("size_bytes"), item.get("sha256"))
            for item in attestation.get("external_evidence", [])
            if isinstance(item, dict)
        }
        _require(
            (str(manifest_path), manifest_path.stat().st_size, _sha256(manifest_path)) in attested_files,
            "M4 candidate manifest is absent from the replayed attestation",
        )
        _require_closed_registry_attested(
            closed_registry,
            evidence_root,
            attested_files,
        )
        manifest = _load(manifest_path)
        _validate_candidate_manifest(
            manifest,
            source_commit,
            evidence_root,
            attested_files,
            source_policy_path,
            expected_source_policy_sha256,
            source_verifier_path,
            expected_source_verifier_sha256,
        )
        source_report_record = manifest["evidence"]["m4_source_audit"]
        source_report = _load(evidence_root / source_report_record["bundle_relative_path"])
        _require(source_report == preflight_source, "M4 source report changed after isolated preflight")
        measured_source_commit = source_report.get("measured_source_commit")
        _require(HEX40.fullmatch(measured_source_commit or "") is not None, "M4 measured source anchor is invalid")
        source_policy_record = {
            "path": str(source_policy_path),
            **_file_record(source_policy_path),
        }
        source_verifier_record = {
            "path": str(source_verifier_path),
            **_file_record(source_verifier_path),
        }
        manifest_file = {
            **_file_record(manifest_path),
            "payload_sha256": _canonical_sha256(manifest),
        }
        replayed_closed_registry, replayed_records_directory = _replay_closed_registry(
            terminal_path=closed_registry_terminal_path,
            evidence_root=evidence_root,
            records_directory=closed_registry_records_directory,
            expected_sha256=expected_closed_registry_sha256,
            expected_terminal_id=expected_closed_registry_terminal_id,
            inventory_sha256=registry["inventory_sha256"],
        )
        _require(
            replayed_closed_registry == closed_registry
            and replayed_records_directory == closed_registry_records_directory,
            "closed inversion authority changed during candidate replay",
        )
        _require(_identity(attestation_path) == attestation_identity, "attestation changed during candidate replay")
    else:
        _require(attestation_gate_path is None, "development build cannot carry attestation authority")
        _require(expected_attestation_sha256 is None, "development build cannot carry attestation SHA-256")
        _require(candidate_manifest_path is None, "development build cannot carry M4 candidate authority")
        _require(evidence_root is None, "development build cannot carry an evidence root")
        _require(source_policy_path is None, "development build cannot carry source-policy authority")
        _require(expected_source_policy_sha256 is None, "development build cannot carry a source-policy SHA-256")
        _require(source_verifier_path is None, "development build cannot carry source-verifier authority")
        _require(expected_source_verifier_sha256 is None, "development build cannot carry a source-verifier SHA-256")
        _require(closed_registry_terminal_path is None, "development build cannot carry a closed inversion terminal")
        _require(expected_closed_registry_sha256 is None, "development build cannot carry a closed inversion terminal SHA-256")
        _require(expected_closed_registry_terminal_id is None, "development build cannot carry a closed inversion terminal ID")
        _require(closed_registry_records_directory is None, "development build cannot carry a closed inversion records directory")

    state = "CANDIDATE_BUILD_AUTHORIZED" if mode == "candidate" else "DEVELOPMENT_ONLY"
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_train": "1.1.0",
        "build_mode": mode,
        "state": state,
        "source_commit": source_commit,
        "validated_engine_commit": VALIDATED_ENGINE_COMMIT,
        "measured_source_commit": measured_source_commit,
        "authorization": {
            "candidate_build": mode == "candidate",
            "package_publication": False,
            "git_tag": False,
            "package_mutation_after_build": False,
        },
        "contract_sha256": contract["contract_sha256"],
        "contract_canonical_sha256": contract["contract_canonical_sha256"],
        "inventory_sha256": registry["inventory_sha256"],
        "registry_state": registry["state"],
        "registry_observed_count": registry.get("observed_count"),
        "registry_missing_count": registry.get("missing_count"),
        "registry_blocking_count": registry.get("blocking_count"),
        "m0_attestation_sha256": expected_attestation_sha256,
        "m0_commit": None if attestation is None else attestation["repository"]["m0_commit"],
        "m4_candidate_manifest": manifest,
        "m4_candidate_manifest_record": manifest_file,
        "m4_source_policy": source_policy_record,
        "m4_source_verifier": source_verifier_record,
        "closed_inversion_registry": closed_registry,
        "pending_post_package": list(POST_PACKAGE_REQUIREMENTS),
        "gate_records": {
            "contract": contract_file,
            "registry": registry_file,
            "attestation": attestation_file,
        },
    }
    receipt["receipt_id"] = _canonical_sha256(receipt)
    return receipt


def _validate_embedded_candidate_manifest(value: Any, source_commit: str) -> None:
    """Validate package-owned structure without granting operational authority."""
    manifest = _exact(
        value,
        {"schema", "state", "validated_engine_commit", "release_source_commit", "option", "findings", "validation", "evidence"},
        "embedded M4 candidate manifest",
    )
    _require(
        manifest["schema"] == CANDIDATE_MANIFEST_SCHEMA
        and manifest["state"] == "PASS"
        and manifest["validated_engine_commit"] == VALIDATED_ENGINE_COMMIT
        and manifest["release_source_commit"] == source_commit
        and manifest["option"] == "A_ALL_11",
        "embedded M4 candidate manifest identity differs",
    )
    findings = _exact(
        manifest["findings"],
        {"required_ids", "closed_ids", "open_critical_high_medium"},
        "embedded M4 findings",
    )
    _require(
        findings == {
            "required_ids": list(M4_FINDING_IDS),
            "closed_ids": list(M4_FINDING_IDS),
            "open_critical_high_medium": 0,
        },
        "embedded M4 finding closure differs",
    )
    validation = _exact(
        manifest["validation"],
        {
            "targeted", "full_common", "full_validation", "host_case_count",
            "specialized_case_count", "feature_count", "route_count",
            "real_multi_gpu", "no_cpu_fallback", "aunp_source_candidate",
        },
        "embedded M4 validation summary",
    )
    _require(
        validation == {
            "targeted": True,
            "full_common": True,
            "full_validation": True,
            "host_case_count": 4,
            "specialized_case_count": 5,
            "feature_count": 42,
            "route_count": 32,
            "real_multi_gpu": True,
            "no_cpu_fallback": True,
            "aunp_source_candidate": True,
        },
        "embedded M4 validation summary differs",
    )
    evidence = _exact(manifest["evidence"], set(REQUIRED_EVIDENCE_ROLES), "embedded M4 evidence")
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for role in REQUIRED_EVIDENCE_ROLES:
        record = _exact(
            evidence[role],
            {"role", "bundle_relative_path", "size_bytes", "sha256", "schema", "content_id"},
            f"embedded evidence.{role}",
        )
        relative = pathlib.PurePosixPath(record["bundle_relative_path"])
        _require(
            record["role"] == role
            and record["schema"] == ROLE_SCHEMAS[role]
            and type(record["size_bytes"]) is int
            and record["size_bytes"] > 0
            and HEX64.fullmatch(record["sha256"] or "") is not None
            and HEX64.fullmatch(record["content_id"] or "") is not None
            and not relative.is_absolute()
            and relative.parts
            and ".." not in relative.parts
            and str(relative) == record["bundle_relative_path"],
            f"embedded evidence.{role} is invalid",
        )
        _require(record["bundle_relative_path"] not in seen_paths, "embedded evidence aliases a path")
        _require(record["sha256"] not in seen_hashes, "embedded evidence reuses bytes")
        seen_paths.add(record["bundle_relative_path"])
        seen_hashes.add(record["sha256"])


def verify_embedded(
    *,
    receipt_path: pathlib.Path,
    contract_path: pathlib.Path,
    schema_path: pathlib.Path,
    inventory_path: pathlib.Path,
    require_candidate: bool,
) -> dict[str, Any]:
    receipt = _load(_regular_file(receipt_path, "candidate receipt"))
    expected_keys = {
        "schema_version", "release_train", "build_mode", "state", "source_commit",
        "validated_engine_commit", "measured_source_commit",
        "authorization", "contract_sha256", "contract_canonical_sha256",
        "inventory_sha256", "registry_state", "registry_observed_count",
        "registry_missing_count", "registry_blocking_count", "m0_attestation_sha256",
        "m0_commit", "m4_candidate_manifest", "m4_candidate_manifest_record",
        "m4_source_policy", "m4_source_verifier",
        "closed_inversion_registry", "pending_post_package", "gate_records",
        "receipt_id",
    }
    _require(set(receipt) == expected_keys, "receipt key inventory differs")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    _require(HEX64.fullmatch(receipt_id or "") is not None, "receipt identity is invalid")
    _require(_canonical_sha256(unsigned) == receipt_id, "receipt identity differs")
    _require(receipt["schema_version"] == SCHEMA_VERSION and receipt["release_train"] == "1.1.0", "receipt schema differs")
    _require(HEX40.fullmatch(receipt["source_commit"] or "") is not None, "receipt source commit is invalid")
    _require(receipt["validated_engine_commit"] == VALIDATED_ENGINE_COMMIT, "receipt validated engine commit differs")
    _require(receipt["build_mode"] in {"development", "candidate"}, "receipt build mode is invalid")
    expected_state = "CANDIDATE_BUILD_AUTHORIZED" if receipt["build_mode"] == "candidate" else "DEVELOPMENT_ONLY"
    _require(receipt["state"] == expected_state, "receipt state differs")
    authorization = _exact(
        receipt["authorization"],
        {"candidate_build", "package_publication", "git_tag", "package_mutation_after_build"},
        "receipt authorization",
    )
    _require(authorization["candidate_build"] is (receipt["build_mode"] == "candidate"), "candidate-build authority differs")
    for key in ("package_publication", "git_tag", "package_mutation_after_build"):
        _require(authorization[key] is False, f"package-owned receipt illegally grants {key}")
    _require(receipt["pending_post_package"] == list(POST_PACKAGE_REQUIREMENTS), "post-package requirement inventory differs")
    for key in ("registry_observed_count", "registry_missing_count", "registry_blocking_count"):
        _require(type(receipt[key]) is int and receipt[key] >= 0, f"receipt {key} is invalid")
    _require(receipt["registry_observed_count"] == 13, "receipt inversion inventory count differs")
    records = _exact(receipt["gate_records"], {"contract", "registry", "attestation"}, "gate records")
    for name in ("contract", "registry"):
        record = _exact(records[name], {"sha256", "size_bytes"}, f"{name} gate record")
        _require(HEX64.fullmatch(record["sha256"] or "") is not None and type(record["size_bytes"]) is int and record["size_bytes"] > 0, f"{name} gate record is invalid")
    contract_path = _regular_file(contract_path, "bundled contract")
    schema_path = _regular_file(schema_path, "bundled inversion schema")
    inventory_path = _regular_file(inventory_path, "bundled inversion inventory")
    _require(_sha256(contract_path) == receipt["contract_sha256"], "bundled contract differs")
    _require(_sha256(inventory_path) == receipt["inventory_sha256"], "bundled inventory differs")
    contract = _load(contract_path)
    _require(_canonical_sha256(contract) == receipt["contract_canonical_sha256"], "bundled canonical contract differs")
    registry = contract["performance_inversion_policy"]["registry"]
    _require(_sha256(schema_path) == registry["schema_sha256"], "bundled inversion schema differs")
    _require(receipt["inventory_sha256"] == registry["initial_inventory_sha256"], "receipt inventory is not contract-bound")
    if require_candidate:
        _require(receipt["build_mode"] == "candidate", "package lacks candidate-build authority")
    if receipt["build_mode"] == "candidate":
        _require(HEX40.fullmatch(receipt["measured_source_commit"] or "") is not None, "candidate measured-source anchor is invalid")
        _require(receipt["registry_state"] == "PASS" and receipt["registry_missing_count"] == 0 and receipt["registry_blocking_count"] == 0, "candidate registry gate is open")
        _require(HEX64.fullmatch(receipt["m0_attestation_sha256"] or "") is not None, "candidate lacks M0 attestation authority")
        _require(HEX40.fullmatch(receipt["m0_commit"] or "") is not None, "candidate lacks an attested M0 commit")
        _embedded_closed_registry(
            receipt["closed_inversion_registry"],
            inventory_sha256=receipt["inventory_sha256"],
        )
        _validate_embedded_candidate_manifest(receipt["m4_candidate_manifest"], receipt["source_commit"])
        record = _exact(receipt["m4_candidate_manifest_record"], {"sha256", "size_bytes", "payload_sha256"}, "candidate manifest record")
        _require(
            HEX64.fullmatch(record["sha256"] or "") is not None
            and type(record["size_bytes"]) is int
            and record["size_bytes"] > 0
            and record["payload_sha256"] == _canonical_sha256(receipt["m4_candidate_manifest"]),
            "candidate manifest record is invalid",
        )
        attestation_record = _exact(records["attestation"], {"sha256", "size_bytes"}, "attestation gate record")
        _require(HEX64.fullmatch(attestation_record["sha256"] or "") is not None and type(attestation_record["size_bytes"]) is int and attestation_record["size_bytes"] > 0, "attestation gate record is invalid")
        source_policy_record = _exact(receipt["m4_source_policy"], {"path", "sha256", "size_bytes"}, "M4 source policy record")
        _require(
            isinstance(source_policy_record["path"], str)
            and pathlib.Path(source_policy_record["path"]).is_absolute()
            and HEX64.fullmatch(source_policy_record["sha256"] or "") is not None
            and type(source_policy_record["size_bytes"]) is int
            and source_policy_record["size_bytes"] > 0,
            "M4 source policy record is invalid",
        )
        source_verifier_record = _exact(receipt["m4_source_verifier"], {"path", "sha256", "size_bytes"}, "M4 source verifier record")
        _require(
            isinstance(source_verifier_record["path"], str)
            and pathlib.Path(source_verifier_record["path"]).is_absolute()
            and HEX64.fullmatch(source_verifier_record["sha256"] or "") is not None
            and type(source_verifier_record["size_bytes"]) is int
            and source_verifier_record["size_bytes"] > 0,
            "M4 source verifier record is invalid",
        )
    else:
        _require(receipt["registry_state"] in {"OPEN", "PASS"}, "development registry state is invalid")
        _require(records["attestation"] is None, "development receipt carries attestation authority")
        _require(receipt["m0_attestation_sha256"] is None and receipt["m0_commit"] is None, "development receipt carries M0 authority")
        _require(receipt["m4_candidate_manifest"] is None and receipt["m4_candidate_manifest_record"] is None, "development receipt carries M4 authority")
        _require(receipt["m4_source_policy"] is None, "development receipt carries source-policy authority")
        _require(receipt["m4_source_verifier"] is None, "development receipt carries source-verifier authority")
        _require(receipt["measured_source_commit"] is None, "development receipt carries measured-source authority")
        _require(receipt["closed_inversion_registry"] is None, "development receipt carries closed inversion authority")
    return receipt


def verify(
    *,
    receipt_path: pathlib.Path,
    contract_path: pathlib.Path,
    schema_path: pathlib.Path,
    inventory_path: pathlib.Path,
    require_candidate: bool,
    contract_gate_path: pathlib.Path | None = None,
    registry_gate_path: pathlib.Path | None = None,
    attestation_gate_path: pathlib.Path | None = None,
    candidate_manifest_path: pathlib.Path | None = None,
    evidence_root: pathlib.Path | None = None,
    source_policy_path: pathlib.Path | None = None,
    expected_source_policy_sha256: str | None = None,
    source_verifier_path: pathlib.Path | None = None,
    expected_source_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    """Fully replay candidate authority against external producer evidence.

    This is deliberately distinct from :func:`verify_embedded`, which can only
    check package-owned structure and can never establish candidate authority.
    """
    receipt = verify_embedded(
        receipt_path=receipt_path,
        contract_path=contract_path,
        schema_path=schema_path,
        inventory_path=inventory_path,
        require_candidate=require_candidate,
    )
    if receipt["build_mode"] != "candidate":
        return receipt
    _require(contract_gate_path is not None, "candidate replay lacks the actual contract gate")
    _require(registry_gate_path is not None, "candidate replay lacks the actual registry gate")
    _require(attestation_gate_path is not None, "candidate replay lacks the actual attestation")
    _require(candidate_manifest_path is not None, "candidate replay lacks the actual M4 manifest")
    _require(evidence_root is not None, "candidate replay lacks the actual evidence root")
    _require(source_policy_path is not None, "candidate replay lacks independently supplied source policy")
    _require(expected_source_policy_sha256 is not None, "candidate replay lacks independently supplied source policy SHA-256")
    _require(source_verifier_path is not None, "candidate replay lacks independently supplied source verifier")
    _require(expected_source_verifier_sha256 is not None, "candidate replay lacks independently supplied source verifier SHA-256")
    evidence_root = _regular_directory(evidence_root, "candidate evidence root")
    closed_registry = _embedded_closed_registry(
        receipt["closed_inversion_registry"],
        inventory_sha256=receipt["inventory_sha256"],
    )
    terminal_relative = pathlib.PurePosixPath(
        closed_registry["file"]["bundle_relative_path"]
    )
    records_relative = pathlib.PurePosixPath(
        closed_registry["terminal"]["records_directory"]
    )
    closed_registry_terminal_path = evidence_root.joinpath(*terminal_relative.parts)
    closed_registry_records_directory = evidence_root.joinpath(*records_relative.parts)
    replayed = create(
        mode="candidate",
        source_commit=receipt["source_commit"],
        contract_gate_path=contract_gate_path,
        registry_gate_path=registry_gate_path,
        attestation_gate_path=attestation_gate_path,
        expected_attestation_sha256=receipt["m0_attestation_sha256"],
        candidate_manifest_path=candidate_manifest_path,
        evidence_root=evidence_root,
        source_policy_path=source_policy_path,
        expected_source_policy_sha256=expected_source_policy_sha256,
        source_verifier_path=source_verifier_path,
        expected_source_verifier_sha256=expected_source_verifier_sha256,
        closed_registry_terminal_path=closed_registry_terminal_path,
        expected_closed_registry_sha256=closed_registry["file"]["sha256"],
        expected_closed_registry_terminal_id=closed_registry["terminal"]["terminal_id"],
        closed_registry_records_directory=closed_registry_records_directory,
    )
    _require(replayed == receipt, "candidate receipt was not exactly re-derived from actual evidence")
    return receipt


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "receipt parent must already exist")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, "receipt parent cannot contain symlinks")
    destination = parent / destination.name
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite receipt")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = pathlib.Path(temporary_name)
    linked = False
    committed = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        linked = True
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, "receipt mode differs")
        _require(destination.read_bytes() == payload, "receipt bytes differ after publication")
        temporary.unlink()
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        committed = True
    finally:
        temporary.unlink(missing_ok=True)
        if linked and not committed:
            destination.unlink(missing_ok=True)
            try:
                directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", action="store_true")
    operation.add_argument("--source-preflight-only", action="store_true")
    parser.add_argument("--mode", choices=("development", "candidate"))
    parser.add_argument("--source-commit")
    parser.add_argument("--contract-gate", type=pathlib.Path)
    parser.add_argument("--registry-gate", type=pathlib.Path)
    parser.add_argument("--attestation-gate", type=pathlib.Path)
    parser.add_argument("--expected-attestation-sha256")
    parser.add_argument("--m4-candidate-manifest", type=pathlib.Path)
    parser.add_argument("--evidence-root", type=pathlib.Path)
    parser.add_argument("--source-transition-policy", type=pathlib.Path)
    parser.add_argument("--expected-source-transition-policy-sha256")
    parser.add_argument("--source-transition-verifier", type=pathlib.Path)
    parser.add_argument("--expected-source-transition-verifier-sha256")
    parser.add_argument("--closed-registry-terminal", type=pathlib.Path)
    parser.add_argument("--expected-closed-registry-sha256")
    parser.add_argument("--expected-closed-registry-terminal-id")
    parser.add_argument("--closed-registry-records-directory", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--contract", type=pathlib.Path)
    parser.add_argument("--schema", type=pathlib.Path)
    parser.add_argument("--inventory", type=pathlib.Path)
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument(
        "--embedded-only",
        action="store_true",
        help="structural package-owned check only; never establishes external candidate authority",
    )
    args = parser.parse_args(argv)
    try:
        if args.source_preflight_only:
            _require(
                args.source_commit
                and args.m4_candidate_manifest
                and args.evidence_root
                and args.source_transition_policy
                and args.expected_source_transition_policy_sha256
                and args.source_transition_verifier
                and args.expected_source_transition_verifier_sha256,
                "source preflight arguments are incomplete",
            )
            result = _preflight_source_authority(
                release_source_commit=args.source_commit,
                candidate_manifest_path=args.m4_candidate_manifest,
                evidence_root=args.evidence_root,
                source_policy_path=args.source_transition_policy,
                expected_source_policy_sha256=args.expected_source_transition_policy_sha256,
                source_verifier_path=args.source_transition_verifier,
                expected_source_verifier_sha256=args.expected_source_transition_verifier_sha256,
            )
            print(
                json.dumps(
                    {
                        "schema": result["schema"],
                        "state": "PASS",
                        "report_id": result["report_id"],
                        "isolated": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.create:
            _require(args.mode is not None and args.source_commit is not None, "--create requires mode and source commit")
            _require(args.contract_gate and args.registry_gate and args.output, "--create lacks gate/output arguments")
            receipt = create(
                mode=args.mode,
                source_commit=args.source_commit,
                contract_gate_path=args.contract_gate,
                registry_gate_path=args.registry_gate,
                attestation_gate_path=args.attestation_gate,
                expected_attestation_sha256=args.expected_attestation_sha256,
                candidate_manifest_path=args.m4_candidate_manifest,
                evidence_root=args.evidence_root,
                source_policy_path=args.source_transition_policy,
                expected_source_policy_sha256=args.expected_source_transition_policy_sha256,
                source_verifier_path=args.source_transition_verifier,
                expected_source_verifier_sha256=args.expected_source_transition_verifier_sha256,
                closed_registry_terminal_path=args.closed_registry_terminal,
                expected_closed_registry_sha256=args.expected_closed_registry_sha256,
                expected_closed_registry_terminal_id=args.expected_closed_registry_terminal_id,
                closed_registry_records_directory=args.closed_registry_records_directory,
            )
            _write_new(args.output, receipt)
        else:
            _require(args.receipt and args.contract and args.schema and args.inventory, "--verify lacks receipt/contract/schema/inventory")
            _require(
                not any(
                    (
                        args.closed_registry_terminal,
                        args.expected_closed_registry_sha256,
                        args.expected_closed_registry_terminal_id,
                        args.closed_registry_records_directory,
                    )
                ),
                "--verify derives closed inversion authority from the receipt and evidence root",
            )
            if args.embedded_only:
                _require(
                    not any((args.contract_gate, args.registry_gate, args.attestation_gate, args.m4_candidate_manifest, args.evidence_root, args.source_transition_policy, args.expected_source_transition_policy_sha256, args.source_transition_verifier, args.expected_source_transition_verifier_sha256, args.closed_registry_terminal, args.expected_closed_registry_sha256, args.expected_closed_registry_terminal_id, args.closed_registry_records_directory)),
                    "--embedded-only cannot accept external authority inputs",
                )
                receipt = verify_embedded(
                    receipt_path=args.receipt,
                    contract_path=args.contract,
                    schema_path=args.schema,
                    inventory_path=args.inventory,
                    require_candidate=args.require_candidate,
                )
            else:
                receipt = verify(
                    receipt_path=args.receipt,
                    contract_path=args.contract,
                    schema_path=args.schema,
                    inventory_path=args.inventory,
                    require_candidate=args.require_candidate,
                    contract_gate_path=args.contract_gate,
                    registry_gate_path=args.registry_gate,
                    attestation_gate_path=args.attestation_gate,
                    candidate_manifest_path=args.m4_candidate_manifest,
                    evidence_root=args.evidence_root,
                    source_policy_path=args.source_transition_policy,
                    expected_source_policy_sha256=args.expected_source_transition_policy_sha256,
                    source_verifier_path=args.source_transition_verifier,
                    expected_source_verifier_sha256=args.expected_source_transition_verifier_sha256,
                )
        print(json.dumps({"schema_version": SCHEMA_VERSION, "state": "PASS", "receipt_id": receipt["receipt_id"], "build_mode": receipt["build_mode"]}, sort_keys=True))
        return 0
    except (OSError, ReceiptError, ValueError) as error:
        print(f"v1.1.0 candidate receipt: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
