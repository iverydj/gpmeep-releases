#!/usr/bin/env python3
"""Create deterministic v1.1.0 candidate and final-release evidence.

This program is deliberately a compiler for already authoritative raw
terminals.  It never launches a simulation and never grants release authority;
the candidate and final receipt verifiers replay every byte produced here.

Operational precondition: the evidence root is a direct, caller-owned mode
0700 directory and no same-UID process may rename or mutate it concurrently.
The durable writer pins every input before/after publication, but this local
single-writer ownership rule remains the boundary for parent-directory races.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import html
import importlib.util
import json
import math
import os
import pathlib
import re
import stat
import sys
import tempfile
from typing import Any, Callable, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUDIT_INPUT_SCHEMA = "gpmeep-v110-independent-final-audit-input-v1"
HUMAN_DECISION_SCHEMA = "gpmeep-v110-human-release-decision-v1"
INVERSION_PLAN_SCHEMA = "gpmeep-v110-inversion-closure-plan-v1"


class EvidenceCreationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceCreationError(message)


def _require_isolated() -> None:
    _require(sys.flags.isolated == 1, "release evidence creation requires Python isolated mode (-I)")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceCreationError(f"non-finite JSON constant: {value}")


def _load(path: pathlib.Path, label: str = "JSON input") -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise EvidenceCreationError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: pathlib.Path) -> tuple[int, int, int, int, int, str]:
    details = path.stat()
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        _sha256(path),
    )


def _direct_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_file(), f"{label} is not a direct regular file")
    return resolved


def _direct_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_dir(), f"{label} is not a direct directory")
    return resolved


def _evidence_root(path: pathlib.Path, label: str) -> pathlib.Path:
    root = _direct_directory(path, label)
    details = root.stat()
    _require(details.st_uid == os.geteuid(), f"{label} is not caller-owned")
    _require(stat.S_IMODE(details.st_mode) == 0o700, f"{label} must be mode 0700")
    return root


def _inside(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    path = _direct_file(path, label)
    _require(path.is_relative_to(root), f"{label} is outside the evidence root")
    return path


def _module(name: str, path: pathlib.Path) -> Any:
    path = _direct_file(path, f"{name} module")
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"{name} module is unavailable")
    prior = sys.modules.get(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
    return module


def _candidate_module() -> Any:
    return _module("gpmeep_v110_evidence_candidate", ROOT / "scripts/v110-candidate-evidence.py")


def _candidate_gate_module() -> Any:
    return _module("gpmeep_v110_evidence_candidate_gate", ROOT / "scripts/v110-release-gate-receipt.py")


def _final_module() -> Any:
    return _module("gpmeep_v110_evidence_final", ROOT / "scripts/v110-final-release-receipt.py")


def _m4_module() -> Any:
    return _module(
        "gpmeep_v110_evidence_m4",
        ROOT / "scripts/python-validation/run_m4_equal_work_matrix.py",
    )


def _aunp_module() -> Any:
    return _module(
        "gpmeep_v110_evidence_aunp",
        ROOT / "scripts/user-workloads/run_installed_aunp_performance_matrix.py",
    )


def _native_module() -> Any:
    return _module("gpmeep_v110_evidence_native", ROOT / "scripts/v110-native-witness.py")


def _registry_module() -> Any:
    return _module(
        "gpmeep_v110_evidence_registry",
        ROOT / "scripts/validate-v110-inversion-registry.py",
    )


def _closure_module() -> Any:
    return _module(
        "gpmeep_v110_evidence_closure",
        ROOT / "scripts/close-v110-inversion-registry.py",
    )


def _fsync_directory(parent: pathlib.Path) -> None:
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _destination(path: pathlib.Path, root: pathlib.Path | None, label: str) -> pathlib.Path:
    destination = path.absolute()
    _require(destination.parent.exists(), f"{label} parent is absent")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, f"{label} parent contains a symlink")
    destination = parent / destination.name
    if root is not None:
        _require(destination.is_relative_to(root), f"{label} is outside the evidence root")
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"refusing to overwrite {label}",
    )
    return destination


def _write_new_bytes(
    path: pathlib.Path,
    payload: bytes,
    *,
    root: pathlib.Path | None = None,
    label: str = "release evidence",
    verify: Callable[[pathlib.Path], Any] | None = None,
) -> pathlib.Path:
    destination = _destination(path, root, label)
    parent = destination.parent
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
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, f"{label} mode differs")
        _require(destination.read_bytes() == payload, f"{label} bytes differ after publication")
        if verify is not None:
            verify(destination)
        temporary.unlink()
        _fsync_directory(parent)
        committed = True
        return destination
    finally:
        temporary.unlink(missing_ok=True)
        if linked and not committed:
            destination.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                _fsync_directory(parent)


def _write_new_json(
    path: pathlib.Path,
    value: dict[str, Any],
    *,
    root: pathlib.Path | None = None,
    label: str = "release evidence",
    verify: Callable[[pathlib.Path], Any] | None = None,
) -> pathlib.Path:
    return _write_new_bytes(
        path,
        _canonical_bytes(value),
        root=root,
        label=label,
        verify=verify,
    )


def _schema_and_content_id(payload: dict[str, Any]) -> tuple[str, str]:
    schema = payload.get("schema")
    if schema is None and type(payload.get("schema_version")) is int:
        schema = f"schema_version:{payload['schema_version']}"
    _require(isinstance(schema, str) and schema, "artifact schema is absent")
    content_id = next(
        (
            payload.get(key)
            for key in ("receipt_id", "report_id", "content_id", "plan_id", "terminal_id")
            if payload.get(key) is not None
        ),
        _canonical_sha256(payload),
    )
    _require(HEX64.fullmatch(content_id or "") is not None, "artifact content ID is invalid")
    return schema, content_id


def artifact_reference(root: pathlib.Path, role: str, path: pathlib.Path) -> dict[str, Any]:
    root = _evidence_root(root, "evidence root")
    path = _inside(path, root, f"{role} artifact")
    before = _identity(path)
    payload = _load(path, f"{role} artifact")
    schema, content_id = _schema_and_content_id(payload)
    result = {
        "role": role,
        "bundle_relative_path": path.relative_to(root).as_posix(),
        "size_bytes": before[3],
        "sha256": before[5],
        "schema": schema,
        "content_id": content_id,
    }
    _require(_identity(path) == before, f"{role} artifact changed while referenced")
    return result


def text_reference(
    root: pathlib.Path, role: str, path: pathlib.Path, media_type: str
) -> dict[str, Any]:
    root = _evidence_root(root, "evidence root")
    path = _inside(path, root, f"{role} report")
    before = _identity(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceCreationError(f"{role} report is not UTF-8") from error
    _require(text.strip() != "", f"{role} report is empty")
    digest = before[5]
    result = {
        "role": role,
        "bundle_relative_path": path.relative_to(root).as_posix(),
        "size_bytes": before[3],
        "sha256": digest,
        "schema": media_type,
        "content_id": digest,
    }
    _require(_identity(path) == before, f"{role} report changed while referenced")
    return result


def producer_reference(root: pathlib.Path, path: pathlib.Path) -> dict[str, Any]:
    reference = artifact_reference(root, "producer", path)
    reference.pop("role")
    return reference


def _candidate_context(
    package_path: pathlib.Path, candidate_receipt_path: pathlib.Path
) -> tuple[Any, pathlib.Path, pathlib.Path, dict[str, Any], dict[str, Any], dict[str, bytes], dict[str, Any]]:
    final = _final_module()
    package = _direct_file(package_path, "candidate package")
    candidate_receipt = _direct_file(candidate_receipt_path, "candidate receipt")
    package_identity = _identity(package)
    receipt_identity = _identity(candidate_receipt)
    candidate = _load(candidate_receipt, "candidate receipt")
    source_commit = candidate.get("source_commit")
    candidate_id = candidate.get("receipt_id")
    _require(HEX40.fullmatch(source_commit or "") is not None, "candidate source commit is invalid")
    _require(HEX64.fullmatch(candidate_id or "") is not None, "candidate receipt ID is invalid")
    metadata, critical, payload = final._inspect_conda_v2(
        package, candidate_receipt, candidate_id
    )
    _require(_identity(package) == package_identity, "candidate package changed during inspection")
    _require(_identity(candidate_receipt) == receipt_identity, "candidate receipt changed during inspection")
    return final, package, candidate_receipt, candidate, metadata, critical, payload


def create_candidate_wrapper(
    *,
    role: str,
    raw_terminal: pathlib.Path,
    release_source_commit: str,
    evidence_root: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    candidate = _candidate_module()
    _require(role in candidate.SCHEMAS, "candidate evidence role is invalid")
    _require(HEX40.fullmatch(release_source_commit or "") is not None, "release source commit is invalid")
    root = _evidence_root(evidence_root, "candidate evidence root")
    output = output.absolute()
    _require(output.parent.resolve(strict=True) == root, "candidate wrapper must be directly inside its evidence root")
    raw = _inside(raw_terminal, root, "candidate raw terminal")
    raw_identity = _identity(raw)
    report = candidate._replay_raw(role, raw)
    package_receipt_id = candidate._package_receipt_id(role, report)
    raw_payload = _load(raw, "candidate raw terminal")
    raw_schema = raw_payload.get("schema")
    _require(isinstance(raw_schema, str) and raw_schema, "candidate raw terminal schema is absent")
    raw_content_id = report.get("terminal_id") or report.get("report_id") or _canonical_sha256(report)
    value: dict[str, Any] = {
        "schema": candidate.SCHEMAS[role],
        "state": "PASS",
        "role": role,
        "source_commit": release_source_commit,
        "package_receipt_id": package_receipt_id,
        "raw_terminal": {
            "producer": candidate.PRODUCER[role],
            "path": raw.relative_to(root).as_posix(),
            "sha256": _sha256(raw),
            "schema": raw_schema,
            "content_id": raw_content_id,
        },
        "summary": candidate._summary(role, report),
    }
    value["report_id"] = _canonical_sha256(value)
    verify = lambda path: candidate.verify(role, path, release_source_commit)
    _write_new_json(output, value, root=root, label=f"candidate {role} wrapper", verify=verify)
    _require(_identity(raw) == raw_identity, "candidate raw terminal changed during creation")
    return value


def _values(value: Any, fragment: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if fragment in key.lower():
                result.append(item)
            result.extend(_values(item, fragment))
    elif isinstance(value, list):
        for item in value:
            result.extend(_values(item, fragment))
    return result


def _report_model(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[tuple[str, Any]]]]:
    performance = report["performance"]
    numerical = report["numerical"]
    report_id = report.get("report_id") or _canonical_sha256(report)
    measured = sum(
        isinstance(row, dict) and row.get("sample_kind") == "measured"
        for row in numerical["comparisons"]
    )
    _require(
        len(performance["raw_seconds"]) == 180
        and len(performance["cases"]) == 13
        and numerical.get("comparison_count") == 141
        and len(numerical["comparisons"]) == 141
        and measured == 94,
        "M4 report census differs",
    )
    data = {
        "schema": "gpmeep-v110-final-report-data-v1",
        "m4_report_id": report_id,
        "actual_seconds": performance["raw_seconds"],
        "cases": performance["cases"],
        "numerical_comparison_count": 141,
        "measured_cycle_numerical_comparison_count": 94,
        "numerical_comparisons": numerical["comparisons"],
    }
    rows = {
        "actual-seconds": [(str(index), item) for index, item in enumerate(performance["raw_seconds"])],
        "means-ratios": [(key, performance["cases"][key]) for key in sorted(performance["cases"])],
        "numerical-results": [(str(index), item) for index, item in enumerate(numerical["comparisons"])],
    }
    return data, rows


def render_reports(report: dict[str, Any]) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    final = _final_module()
    data, row_sets = _report_model(report)
    comparison_html, comparison_markdown = final._actual_time_comparison_tables(
        report
    )
    headings = {
        "actual-seconds": "Actual seconds",
        "means-ratios": "Means and ratios",
        "numerical-results": "Numerical results",
    }
    html_tables: list[str] = []
    markdown_tables: list[str] = []
    for table_id, rows in row_sets.items():
        html_rows = "".join(
            "<tr><td>"
            + html.escape(key, quote=True)
            + "</td><td><code>"
            + html.escape(json.dumps(item, sort_keys=True, separators=(",", ":")), quote=True)
            + "</code></td></tr>"
            for key, item in rows
        )
        html_tables.append(
            f'<h2>{headings[table_id]}</h2><table id="{table_id}"><thead><tr><th>Key</th><th>Canonical value</th>'
            f"</tr></thead><tbody>{html_rows}</tbody></table>"
        )
        markdown_rows = "\n".join(
            "| "
            + key.replace("|", "\\|")
            + " | "
            + json.dumps(item, sort_keys=True, separators=(",", ":")).replace("|", r"\|")
            + " |"
            for key, item in rows
        )
        markdown_tables.append(
            f"## {headings[table_id]}\n\n| Key | Canonical value |\n|---|---|\n{markdown_rows}"
        )
    png = final._expected_speedup_png(report)
    width, height = final._decode_png(png)
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    plot = {
        "id": "speedup",
        "mime_type": "image/png",
        "width_px": width,
        "height_px": height,
        "size_bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
        "data_uri_sha256": hashlib.sha256(data_uri.encode("ascii")).hexdigest(),
        "source_data_sha256": _canonical_sha256(report["performance"]["cases"]),
    }
    html_text = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>gpmeep v1.1.0 final report</title>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}th,td{border:1px solid #999;padding:3px;vertical-align:top}code{white-space:pre-wrap}</style>"
        "</head><body><h1>gpmeep v1.1.0 final qualification report</h1>"
        f'<p>M4 report ID: <code>{data["m4_report_id"]}</code></p>'
        f'<img data-plot-id="speedup" alt="deterministic speedup plot" src="{data_uri}">'
        + comparison_html
        + "".join(html_tables)
        + '<script id="gpmeep-v110-report-data" type="application/json">'
        + json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "</script></body></html>\n"
    )
    markdown_text = (
        "# gpmeep v1.1.0 final qualification report\n\n"
        f'M4 report ID: `{data["m4_report_id"]}`\n\n'
        + comparison_markdown
        + "\n\n"
        + "\n\n".join(markdown_tables)
        + "\n"
    )
    final._verify_report_semantics(html_text, markdown_text, report)
    final._verify_embedded_plots(html_text, [plot], report)
    return markdown_text.encode("utf-8"), html_text.encode("utf-8"), [plot]


def create_reports(
    *, m4_output: pathlib.Path, markdown_output: pathlib.Path, html_output: pathlib.Path,
    evidence_root: pathlib.Path,
) -> list[dict[str, Any]]:
    root = _evidence_root(evidence_root, "qualification evidence root")
    m4_directory = _direct_directory(m4_output, "M4 output")
    m4_terminal = _direct_file(m4_directory / "M4_COMPLETE", "M4 terminal")
    m4_identity = _identity(m4_terminal)
    report = _m4_module().verify_complete(m4_directory)
    markdown, html_bytes, plots = render_reports(report)
    markdown_path = _destination(markdown_output, root, "final Markdown report")
    html_path = _destination(html_output, root, "final HTML report")
    _require(markdown_path != html_path, "final report destinations alias")
    created: list[pathlib.Path] = []
    final = _final_module()
    try:
        _write_new_bytes(markdown_path, markdown, root=root, label="final Markdown report")
        created.append(markdown_path)
        _write_new_bytes(html_path, html_bytes, root=root, label="final HTML report")
        created.append(html_path)
        final._verify_report_semantics(
            html_path.read_text(encoding="utf-8"),
            markdown_path.read_text(encoding="utf-8"),
            report,
        )
        final._verify_embedded_plots(html_path.read_text(encoding="utf-8"), plots, report)
        _require(_identity(m4_terminal) == m4_identity, "M4 terminal changed during report creation")
        return plots
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            _fsync_directory(markdown_path.parent)
        if html_path.parent != markdown_path.parent:
            with contextlib.suppress(OSError):
                _fsync_directory(html_path.parent)
        raise


def _qualification_common(
    role: str,
    source_commit: str,
    package_sha256: str,
    candidate_receipt_id: str,
    producer: dict[str, Any],
    extras: dict[str, Any],
    final: Any,
) -> dict[str, Any]:
    value = {
        "schema": final.QUALIFICATION_ROLE_SCHEMAS[role],
        "state": "PASS",
        "role": role,
        "source_commit": source_commit,
        "package_sha256": package_sha256,
        "candidate_receipt_id": candidate_receipt_id,
        "producer_terminal": producer,
        **extras,
    }
    value["report_id"] = _canonical_sha256(value)
    return value


def create_qualification_wrapper(
    *, role: str, raw_terminal: pathlib.Path, package: pathlib.Path,
    candidate_receipt: pathlib.Path, evidence_root: pathlib.Path,
    output: pathlib.Path, installed_reports: Iterable[pathlib.Path] = (),
) -> dict[str, Any]:
    final, package_path, _receipt_path, candidate, metadata, _critical, candidate_payload = _candidate_context(
        package, candidate_receipt
    )
    _require(role in {
        "installed_correctness", "installed_multi_gpu", "installed_aunp",
        "ubuntu_22_04_native", "newer_ubuntu_native", "glibc_floor",
        "final_adversarial_audit",
    }, "qualification role is invalid")
    root = _evidence_root(evidence_root, "qualification evidence root")
    raw = _inside(raw_terminal, root, f"{role} raw terminal")
    raw_identity = _identity(raw)
    raw_payload = _load(raw, f"{role} raw terminal")
    producer = producer_reference(root, raw)
    source_commit = candidate["source_commit"]
    candidate_id = candidate["receipt_id"]
    package_sha256 = metadata["sha256"]
    extras: dict[str, Any]
    if role in {"installed_correctness", "installed_multi_gpu"}:
        _require(raw.name == "M4_COMPLETE", "final qualification requires M4_COMPLETE")
        report = final._replay_equal_work(raw.parent)
        report_id = report.get("report_id") or _canonical_sha256(report)
        development_receipt_id = (
            report.get("contract", {})
            .get("runtime_inputs", {})
            .get("v110_receipt_id")
        )
        _require(
            HEX64.fullmatch(development_receipt_id or "") is not None,
            "M4 development package receipt ID differs",
        )
        if role == "installed_correctness":
            extras = {
                "numerical_oracles_pass": report.get("numerical", {}).get("outcome") == "PASS",
                "comparison_count": report.get("numerical", {}).get("comparison_count"),
                "m4_report_id": report_id,
                "development_package_receipt_id": development_receipt_id,
            }
        else:
            uuids = sorted({
                item
                for value in _values(report, "uuid")
                for item in (value if isinstance(value, list) else [value])
                if isinstance(item, str) and item
            })
            fallbacks = sum(
                value for value in _values(report, "cpu_fallback_count") if type(value) is int
            )
            extras = {
                "gpu_uuids": uuids,
                "distinct_gpu_uuids": len(uuids) >= 2,
                "cpu_fallback_count": fallbacks,
                "m4_report_id": report_id,
                "development_package_receipt_id": development_receipt_id,
            }
    elif role == "installed_aunp":
        _require(raw.name == "COMPLETE", "installed AuNP producer terminal name differs")
        report = _aunp_module().validate(raw.parent)
        profile = report.get("qualification_profile", {})
        provenance_path = _direct_file(
            pathlib.Path(report.get("package_provenance", {}).get("path", "")),
            "installed AuNP package provenance",
        )
        provenance = _load(provenance_path, "installed AuNP package provenance")
        provenance_report_id = provenance.get("report_id")
        installed_prefix = provenance.get("prefix")
        _require(
            HEX64.fullmatch(provenance_report_id or "") is not None
            and isinstance(installed_prefix, str)
            and pathlib.Path(installed_prefix).is_absolute(),
            "installed AuNP package provenance identity differs",
        )
        extras = {
            "polarization": "TM" if profile.get("mode") == "TM-only" else None,
            "minimum_run": profile.get("dft_minimum_run_time"),
            "dft_field_decay": profile.get("dft_decay_tolerance"),
            "physics_fingerprint": _canonical_sha256(report),
            "installed_prefix": installed_prefix,
            "package_provenance_report_id": provenance_report_id,
        }
    elif role in {"ubuntu_22_04_native", "newer_ubuntu_native"}:
        report = _native_module().verify(raw, require_live=False)
        extras = {
            "platform": report.get("platform"),
            "native_install": True,
            "environment_id": report.get("environment_id"),
            "prefix": report.get("prefix"),
            "os_release_sha256": report.get("os_release", {}).get("sha256"),
            "installed_report_id": report.get("installed_report_id"),
            "namespace_id": report.get("namespace_id"),
            "image_identity": report.get("image_identity"),
            "kernel": report.get("kernel"),
            "libc": report.get("libc"),
        }
    elif role == "glibc_floor":
        embedded = final._replay_candidate_glibc(candidate_payload, raw_payload)
        report_ids = []
        for path in installed_reports:
            retained = _load(_inside(path, root, "installed provenance report"), "installed provenance report")
            report_ids.append(retained.get("report_id"))
        extras = {
            "overall_maximum_required": embedded["overall_maximum_required"],
            "audited_elf_sha256": sorted(item["sha256"] for item in embedded["elf_files"]),
            "installed_report_ids": sorted(report_ids),
        }
    else:
        extras = {
            "findings": raw_payload.get("findings"),
            "open_critical_high_medium": sum(
                isinstance(item, dict)
                and item.get("severity") in {"CRITICAL", "HIGH", "MEDIUM"}
                and item.get("state") != "CLOSED"
                for item in raw_payload.get("findings", [])
            ),
            "human_release_decision": raw_payload.get("human_release_decision"),
        }
    value = _qualification_common(
        role, source_commit, package_sha256, candidate_id, producer, extras, final
    )
    verify_value = lambda: final._verify_qualification_payload(
        role,
        value,
        source_commit=source_commit,
        package_sha256=package_sha256,
        candidate_receipt_id=candidate_id,
        evidence_root=root,
        candidate_payload=candidate_payload,
    )
    verify_value()
    def verify_published(path: pathlib.Path) -> None:
        retained = _load(path, f"{role} wrapper")
        _require(retained == value, f"{role} wrapper bytes differ")
        final._verify_qualification_payload(
            role, retained, source_commit=source_commit,
            package_sha256=package_sha256, candidate_receipt_id=candidate_id,
            evidence_root=root, candidate_payload=candidate_payload,
        )
    _write_new_json(output, value, root=root, label=f"{role} wrapper", verify=verify_published)
    _require(_identity(raw) == raw_identity, f"{role} raw terminal changed during creation")
    return value


def extract_glibc_terminal(
    *, package: pathlib.Path, candidate_receipt: pathlib.Path,
    evidence_root: pathlib.Path, output: pathlib.Path,
) -> dict[str, Any]:
    final, _package, _receipt, _candidate, _metadata, _critical, payload = _candidate_context(
        package, candidate_receipt
    )
    member = payload.get(final.GLIBC_AUDIT_PATH)
    _require(member is not None, "candidate GLIBC audit is absent")
    raw = final._member_bytes(member, "candidate GLIBC audit")
    value = final._json_bytes(raw, "candidate GLIBC audit")
    _write_new_bytes(
        output, raw, root=_evidence_root(evidence_root, "qualification evidence root"),
        label="candidate GLIBC terminal",
        verify=lambda path: _require(_load(path, "candidate GLIBC terminal") == value, "GLIBC bytes differ"),
    )
    return value


def create_final_audit(
    *, findings_input: pathlib.Path, human_decision: pathlib.Path,
    package: pathlib.Path, candidate_receipt: pathlib.Path,
    evidence_root: pathlib.Path, output: pathlib.Path,
) -> dict[str, Any]:
    final, _package, _receipt, candidate, metadata, _critical, candidate_payload = _candidate_context(
        package, candidate_receipt
    )
    root = _evidence_root(evidence_root, "qualification evidence root")
    findings_path = _direct_file(findings_input, "independent audit findings")
    decision_path = _direct_file(human_decision, "human release decision")
    findings_identity = _identity(findings_path)
    decision_identity = _identity(decision_path)
    _require(stat.S_IMODE(decision_path.stat().st_mode) == 0o600, "human release decision must be mode 0600")
    inputs = _load(findings_path, "independent audit findings")
    decision_input = _load(decision_path, "human release decision")
    _require(
        set(inputs) == {"schema", "auditor", "source_commit", "package_sha256", "candidate_receipt_id", "findings", "input_id"}
        and inputs["schema"] == AUDIT_INPUT_SCHEMA,
        "independent audit input schema differs",
    )
    unsigned_input = dict(inputs)
    input_id = unsigned_input.pop("input_id")
    _require(input_id == _canonical_sha256(unsigned_input), "independent audit input ID differs")
    _require(
        inputs["source_commit"] == candidate["source_commit"]
        and inputs["package_sha256"] == metadata["sha256"]
        and inputs["candidate_receipt_id"] == candidate["receipt_id"],
        "independent audit exact identity differs",
    )
    _require(isinstance(inputs["auditor"], str) and inputs["auditor"].strip(), "independent auditor is empty")
    findings = inputs["findings"]
    _require(
        isinstance(findings, list)
        and [item.get("id") for item in findings] == list(final.FINAL_AUDIT_FINDINGS),
        "independent audit finding census differs",
    )
    normalized = []
    for item in findings:
        _require(
            isinstance(item, dict)
            and set(item) == {"id", "severity", "state", "evidence"}
            and item["severity"] in {"CRITICAL", "HIGH", "MEDIUM"}
            and item["state"] == "CLOSED",
            "independent audit finding differs",
        )
        evidence = item["evidence"]
        _require(isinstance(evidence, dict) and set(evidence) == {"path", "sha256"}, "audit evidence reference differs")
        relative = pathlib.PurePosixPath(evidence["path"])
        _require(relative.parts and not relative.is_absolute() and ".." not in relative.parts and str(relative) == evidence["path"], "audit evidence path is unsafe")
        path = _inside(root.joinpath(*relative.parts), root, "audit finding evidence")
        _require(_sha256(path) == evidence["sha256"], "audit finding evidence hash differs")
        normalized.append({
            "id": item["id"], "severity": item["severity"], "state": "CLOSED",
            "evidence_sha256": evidence["sha256"],
        })
    _require(
        set(decision_input) == {"schema", "decision", "approver", "finding_ids", "package_sha256", "decision_id"}
        and decision_input["schema"] == HUMAN_DECISION_SCHEMA,
        "human release decision input schema differs",
    )
    decision = {key: decision_input[key] for key in ("decision", "approver", "finding_ids", "package_sha256", "decision_id")}
    unsigned_decision = dict(decision)
    decision_id = unsigned_decision.pop("decision_id")
    _require(
        decision["decision"] == "APPROVE_RELEASE"
        and isinstance(decision["approver"], str) and decision["approver"].strip()
        and decision["finding_ids"] == list(final.FINAL_AUDIT_FINDINGS)
        and decision["package_sha256"] == metadata["sha256"]
        and decision_id == _canonical_sha256(unsigned_decision),
        "human release decision differs",
    )
    value: dict[str, Any] = {
        "schema": final.FINAL_AUDIT_PRODUCER_SCHEMA,
        "state": "PASS",
        "source_commit": candidate["source_commit"],
        "package_sha256": metadata["sha256"],
        "candidate_receipt_id": candidate["receipt_id"],
        "auditor": inputs["auditor"],
        "scope": {
            "independent": True,
            "exact_source_commit": candidate["source_commit"],
            "exact_package_sha256": metadata["sha256"],
            "finding_ids": list(final.FINAL_AUDIT_FINDINGS),
        },
        "findings": normalized,
        "human_release_decision": decision,
    }
    value["terminal_id"] = _canonical_sha256(value)
    def verify_terminal(path: pathlib.Path) -> None:
        producer = producer_reference(root, path)
        wrapper = _qualification_common(
            "final_adversarial_audit", candidate["source_commit"], metadata["sha256"],
            candidate["receipt_id"], producer,
            {"findings": normalized, "open_critical_high_medium": 0, "human_release_decision": decision},
            final,
        )
        final._verify_qualification_payload(
            "final_adversarial_audit", wrapper,
            source_commit=candidate["source_commit"], package_sha256=metadata["sha256"],
            candidate_receipt_id=candidate["receipt_id"], evidence_root=root,
            candidate_payload=candidate_payload,
        )
    _write_new_json(output, value, root=root, label="independent final audit", verify=verify_terminal)
    _require(_identity(findings_path) == findings_identity, "independent audit findings changed during creation")
    _require(_identity(decision_path) == decision_identity, "human release decision changed during creation")
    return value


def _parse_role_files(values: list[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        _require(separator == "=" and role and path, "role file must use ROLE=PATH")
        _require(role not in result, f"duplicate role file: {role}")
        result[role] = pathlib.Path(path)
    return result


def create_candidate_manifest(
    *, role_files: dict[str, pathlib.Path], release_source_commit: str,
    evidence_root: pathlib.Path, output: pathlib.Path,
    source_policy: pathlib.Path, source_policy_sha256: str,
    source_verifier: pathlib.Path, source_verifier_sha256: str,
) -> dict[str, Any]:
    gate = _candidate_gate_module()
    root = _evidence_root(evidence_root, "candidate evidence root")
    _require(set(role_files) == set(gate.REQUIRED_EVIDENCE_ROLES), "candidate evidence role inventory differs")
    evidence = {
        role: artifact_reference(root, role, role_files[role])
        for role in gate.REQUIRED_EVIDENCE_ROLES
    }
    _require(len({item["bundle_relative_path"] for item in evidence.values()}) == len(evidence), "candidate evidence paths repeat")
    _require(len({item["sha256"] for item in evidence.values()}) == len(evidence), "candidate evidence hashes repeat")
    typed_receipts = {
        _load(_inside(role_files[role], root, f"candidate {role}"), f"candidate {role}").get("package_receipt_id")
        for role in gate.TYPED_CANDIDATE_ROLES
    }
    _require(len(typed_receipts) == 1 and HEX64.fullmatch(next(iter(typed_receipts)) or "") is not None, "typed candidate package receipt IDs differ")
    value = {
        "schema": gate.CANDIDATE_MANIFEST_SCHEMA,
        "state": "PASS",
        "validated_engine_commit": gate.VALIDATED_ENGINE_COMMIT,
        "release_source_commit": release_source_commit,
        "option": "A_ALL_11",
        "findings": {
            "required_ids": list(gate.M4_FINDING_IDS),
            "closed_ids": list(gate.M4_FINDING_IDS),
            "open_critical_high_medium": 0,
        },
        "validation": {
            "targeted": True, "full_common": True, "full_validation": True,
            "host_case_count": 4, "specialized_case_count": 5,
            "feature_count": 42, "route_count": 32,
            "real_multi_gpu": True, "no_cpu_fallback": True,
            "aunp_source_candidate": True,
        },
        "evidence": evidence,
    }
    attested = {
        (str(_inside(path, root, f"candidate {role}")), _inside(path, root, f"candidate {role}").stat().st_size, _sha256(_inside(path, root, f"candidate {role}")))
        for role, path in role_files.items()
    }
    policy = _direct_file(source_policy, "source transition policy")
    verifier = _direct_file(source_verifier, "source transition verifier")
    policy_identity = _identity(policy)
    verifier_identity = _identity(verifier)
    _require(_sha256(policy) == source_policy_sha256, "source transition policy hash differs")
    _require(_sha256(verifier) == source_verifier_sha256, "source transition verifier hash differs")
    def validate(retained: dict[str, Any]) -> None:
        gate._validate_candidate_manifest(
            retained, release_source_commit, root, attested,
            policy, source_policy_sha256, verifier, source_verifier_sha256,
        )
    validate(value)
    _write_new_json(
        output, value, root=root, label="M4 candidate manifest",
        verify=lambda path: validate(_load(path, "M4 candidate manifest")),
    )
    _require(_identity(policy) == policy_identity, "source transition policy changed during creation")
    _require(_identity(verifier) == verifier_identity, "source transition verifier changed during creation")
    return value


def create_inversion_records(
    *, m4_output: pathlib.Path, closure_plan: pathlib.Path,
    evidence_root: pathlib.Path, output_directory: pathlib.Path,
) -> dict[str, Any]:
    root = _evidence_root(evidence_root, "inversion evidence root")
    m4_directory = _direct_directory(m4_output, "M4 output")
    m4_terminal = _direct_file(m4_directory / "M4_COMPLETE", "M4 terminal")
    m4_identity = _identity(m4_terminal)
    report = _m4_module().verify_complete(m4_directory)
    bases = report.get("performance", {}).get("inversion_records")
    _require(isinstance(bases, list) and len(bases) == 13, "M4 inversion record census differs")
    plan = _load(_direct_file(closure_plan, "inversion closure plan"), "inversion closure plan")
    _require(set(plan) == {"schema", "m4_report_id", "entries", "plan_id"} and plan["schema"] == INVERSION_PLAN_SCHEMA, "inversion closure plan schema differs")
    unsigned = dict(plan)
    plan_id = unsigned.pop("plan_id")
    _require(plan_id == _canonical_sha256(unsigned), "inversion closure plan ID differs")
    report_id = report.get("report_id") or _canonical_sha256(report)
    _require(plan["m4_report_id"] == report_id, "inversion closure plan M4 report differs")
    entries = plan["entries"]
    _require(isinstance(entries, list) and len(entries) == 13, "inversion closure entry census differs")
    by_workload = {item.get("workload_id"): item for item in entries if isinstance(item, dict)}
    _require(len(by_workload) == 13 and set(by_workload) == {item["workload_id"] for item in bases}, "inversion closure workload inventory differs")
    records = []
    for base in bases:
        entry = by_workload[base["workload_id"]]
        _require(set(entry) == {"workload_id", "primary_class", "classification_evidence", "confidence", "owner", "resolution"}, "inversion closure entry schema differs")
        record = json.loads(json.dumps(base))
        record["primary_class"] = entry["primary_class"]
        record["classification_evidence"] = entry["classification_evidence"]
        record["confidence"] = entry["confidence"]
        record["owner"] = entry["owner"]
        record["resolution"] = entry["resolution"]
        state = entry["resolution"].get("state") if isinstance(entry["resolution"], dict) else None
        record["release_blocking"] = state not in {
            "proved_hardware_crossover", "optimized_and_remeasured", "environment_fixed_and_remeasured"
        }
        records.append(record)
    requested_destination = output_directory.absolute()
    destination_parent = _direct_directory(
        requested_destination.parent, "inversion record destination parent"
    )
    _require(
        destination_parent.is_relative_to(root),
        "inversion record directory is outside evidence root",
    )
    destination = destination_parent / requested_destination.name
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite inversion record directory")
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)
    _require(
        stat.S_IMODE(destination.stat().st_mode) == 0o700,
        "inversion record directory mode differs",
    )
    created = True
    try:
        for record in records:
            _write_new_json(destination / f"{record['record_id']}.json", record, root=root, label="closed inversion record")
        _fsync_directory(destination)
        result = _registry_module().validate_registry(
            require_closed=True, evidence_root=root, records_directory=destination
        )
        _require(result.get("closed_count") == 13 and result.get("open_count") == 0, "closed inversion registry result differs")
        _require(_identity(m4_terminal) == m4_identity, "M4 terminal changed during inversion closure")
        _fsync_directory(destination.parent)
        created = False
        return result
    finally:
        if created:
            for path in destination.iterdir() if destination.exists() else ():
                path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                destination.rmdir()
            with contextlib.suppress(OSError):
                _fsync_directory(destination.parent)


def _sample_spec(value: str, root: pathlib.Path, registry: Any) -> dict[str, Any]:
    kind, separator, remainder = value.partition(":")
    cycle_text, separator2, path_text = remainder.partition(":")
    _require(separator == ":" and separator2 == ":" and path_text, "sample must use KIND:CYCLE:PATH")
    _require(kind in registry.SAMPLE_SCHEMAS, "sample kind is invalid")
    try:
        cycle = int(cycle_text)
    except ValueError as error:
        raise EvidenceCreationError("sample cycle is invalid") from error
    _require(cycle in {1, 2}, "sample cycle must be 1 or 2")
    path = _inside(pathlib.Path(path_text), root, "inversion sample")
    before = _identity(path)
    payload = _load(path, "inversion sample")
    _require(payload.get("schema") == registry.SAMPLE_SCHEMAS[kind], "inversion sample schema differs")
    result = {
        "kind": kind,
        "schema": payload["schema"],
        "cycle_index": cycle,
        "path": path.relative_to(root).as_posix(),
        "sha256": before[5],
    }
    _require(_identity(path) == before, "inversion sample changed while referenced")
    return result


def create_inversion_evidence(
    *, kind: str, workload_id: str, classification: str,
    m4_output: pathlib.Path, samples: list[str], intervention: str | None,
    evidence_root: pathlib.Path, output: pathlib.Path,
) -> dict[str, Any]:
    registry = _registry_module()
    _require(kind in registry.EVIDENCE_KINDS - {"numerical_regression"}, "inversion evidence kind is invalid")
    _require(classification in registry.CLASSES - {"unknown"}, "inversion classification is invalid")
    root = _evidence_root(evidence_root, "inversion evidence root")
    m4_directory = _direct_directory(m4_output, "M4 output")
    m4_terminal = _direct_file(m4_directory / "M4_COMPLETE", "M4 terminal")
    m4_identity = _identity(m4_terminal)
    m4_report = _m4_module().verify_complete(m4_directory)
    contract = m4_report.get("contract", {})
    repo = _direct_directory(pathlib.Path(contract.get("paths", {}).get("repo", "")), "M4 source repository")
    inventory_path = _direct_file(
        repo / "packaging/v110-performance-inversion-inventory.json",
        "normative inversion inventory",
    )
    inventory_identity = _identity(inventory_path)
    inventory = _load(inventory_path, "normative inversion inventory")
    observations = inventory.get("observations")
    _require(
        isinstance(observations, list)
        and sum(isinstance(item, dict) and item.get("workload_id") == workload_id for item in observations) == 1,
        "inversion workload is absent or repeated",
    )
    input_records = [_sample_spec(value, root, registry) for value in samples]
    arguments = {
        "timing": {"comparison": "cpu_cuda_equal_work", "minimum_cycles": 2},
        "phase_profile": {"phase_inventory": sorted(registry.PHASES), "minimum_cycles": 2},
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
    if kind in {"profiler_or_ablation", "same_binary_ablation"}:
        _require(
            isinstance(intervention, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", intervention) is not None,
            "ablation intervention is invalid",
        )
        retained_arguments = {
            "intervention_flag": "--gpmeep-ablation",
            "intervention": intervention,
            "minimum_cycles": 2,
            "require_reversed_order": True,
        }
    else:
        _require(intervention is None, "intervention is only valid for ablation evidence")
        retained_arguments = arguments[kind]
    campaign_id = inventory.get("source_campaign", {}).get("id")
    inventory_sha256 = _sha256(inventory_path)
    value: dict[str, Any] = {
        "schema": registry.EVIDENCE_SCHEMAS[kind],
        "state": "PASS",
        "kind": kind,
        "workload_id": workload_id,
        "source_campaign_id": campaign_id,
        "source_inventory_sha256": inventory_sha256,
        "classification": classification,
        "arguments": retained_arguments,
        "inputs": input_records,
        "result": {},
    }
    value["evidence_id"] = _canonical_sha256(value)
    expected_result = registry._verify_typed_evidence(
        pathlib.Path("<in-memory>"),
        root,
        expected_kind=kind,
        expected_workload_id=workload_id,
        expected_campaign_id=campaign_id,
        expected_inventory_sha256=inventory_sha256,
        expected_classification=classification,
        derive_result=True,
        payload_override=value,
    )
    value["result"] = expected_result
    value.pop("evidence_id")
    value["evidence_id"] = _canonical_sha256(value)
    def verify(path: pathlib.Path) -> None:
        registry._verify_typed_evidence(
            path, root,
            expected_kind=kind,
            expected_workload_id=workload_id,
            expected_campaign_id=campaign_id,
            expected_inventory_sha256=inventory_sha256,
            expected_classification=classification,
        )
    _write_new_json(output, value, root=root, label=f"{kind} inversion evidence", verify=verify)
    _require(_identity(m4_terminal) == m4_identity, "M4 terminal changed during inversion evidence creation")
    _require(_identity(inventory_path) == inventory_identity, "normative inversion inventory changed during creation")
    return value


def create_closed_registry(
    *, records_directory: pathlib.Path, evidence_root: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    root = _evidence_root(evidence_root, "closure evidence root")
    records = _direct_directory(records_directory, "closed inversion records")
    _require(records.is_relative_to(root), "closed inversion records escape evidence root")
    closure = _closure_module()
    value = closure.derive(records, root)
    _write_new_json(
        output, value, root=root, label="closed inversion registry terminal",
        verify=lambda path: closure.verify(path, root),
    )
    return value


def create_qualification_manifest(
    *,
    package: pathlib.Path,
    candidate_receipt: pathlib.Path,
    evidence_root: pathlib.Path,
    m4_terminal: pathlib.Path,
    ubuntu22_prefix: pathlib.Path,
    ubuntu22_report: pathlib.Path,
    ubuntu22_witness: pathlib.Path,
    ubuntu22_self_check: pathlib.Path,
    ubuntu24_prefix: pathlib.Path,
    ubuntu24_report: pathlib.Path,
    ubuntu24_witness: pathlib.Path,
    ubuntu24_self_check: pathlib.Path,
    role_files: dict[str, pathlib.Path],
    final_audit_authority: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    final, package_path, _receipt_path, candidate, metadata, critical, candidate_payload = _candidate_context(
        package, candidate_receipt
    )
    root = _evidence_root(evidence_root, "qualification evidence root")
    expected_roles = set(final.QUALIFICATION_ROLES)
    _require(set(role_files) == expected_roles, "qualification role inventory differs")
    m4_path = _inside(m4_terminal, root, "three-way performance terminal")
    _require(m4_path.name == "M4_COMPLETE", "three-way performance terminal must be M4_COMPLETE")
    m4_identity = _identity(m4_path)
    equal_work = final._replay_equal_work(m4_path.parent)
    source_commit = candidate["source_commit"]
    candidate_id = candidate["receipt_id"]
    package_sha256 = metadata["sha256"]

    native_inputs = (
        (
            "ubuntu-22.04", ubuntu22_prefix, ubuntu22_report,
            ubuntu22_witness, ubuntu22_self_check,
        ),
        (
            "ubuntu-24.04", ubuntu24_prefix, ubuntu24_report,
            ubuntu24_witness, ubuntu24_self_check,
        ),
    )
    installed: list[dict[str, Any]] = []
    native = _native_module()
    for index, (platform, prefix_value, report_value, witness_value, self_check_value) in enumerate(native_inputs):
        prefix = _direct_directory(prefix_value, f"{platform} installed prefix")
        report_path = _inside(report_value, root, f"{platform} installed provenance")
        witness_path = _inside(witness_value, root, f"{platform} native witness")
        self_check_path = _inside(self_check_value, root, f"{platform} installed self-check")
        report_payload = _load(report_path, f"{platform} installed provenance")
        witness_payload = native.verify(witness_path, require_live=False)
        self_check_payload = final._replay_installed_self_check(
            self_check_path,
            prefix=prefix,
            source_commit=source_commit,
            installed_runtime=report_payload.get("runtime"),
        )
        _require(
            witness_payload.get("platform") == platform
            and witness_payload.get("prefix") == str(prefix)
            and report_payload.get("prefix") == str(prefix)
            and report_payload.get("package_sha256") == package_sha256
            and report_payload.get("manifest", {}).get("source_commit") == source_commit
            and report_payload.get("v110_candidate_gate", {}).get("receipt_id") == candidate_id,
            f"{platform} installed identity differs",
        )
        report_ref = artifact_reference(root, f"installed_provenance_{index}", report_path)
        witness_ref = artifact_reference(root, f"native_witness_{index}", witness_path)
        self_check_ref = artifact_reference(root, f"self_check_{index}", self_check_path)
        _require(
            self_check_payload.get("release", {}).get("source_commit") == source_commit,
            f"{platform} self-check source differs",
        )
        installed.append(
            {
                "environment_id": witness_payload["environment_id"],
                "platform": platform,
                "prefix": str(prefix),
                "native_witness": witness_ref,
                "self_check": self_check_ref,
                "package_sha256": package_sha256,
                "source_commit": source_commit,
                "candidate_receipt_id": candidate_id,
                "report": report_ref,
            }
        )
    _require(
        len({item["prefix"] for item in installed}) == 2
        and len({item["environment_id"] for item in installed}) == 2
        and len({item["report"]["sha256"] for item in installed}) == 2
        and len({item["native_witness"]["sha256"] for item in installed}) == 2
        and len({item["self_check"]["sha256"] for item in installed}) == 2,
        "native installed evidence is not independent",
    )

    qualification: dict[str, Any] = {}
    for role in final.QUALIFICATION_ROLES:
        path = role_files[role]
        if role == "final_report_markdown":
            qualification[role] = text_reference(root, role, path, "text/markdown")
        elif role == "final_report_html":
            qualification[role] = text_reference(root, role, path, "text/html")
        else:
            qualification[role] = artifact_reference(root, role, path)

    expected_markdown, expected_html, plots = render_reports(equal_work)
    markdown_path = _inside(role_files["final_report_markdown"], root, "final Markdown report")
    html_path = _inside(role_files["final_report_html"], root, "final HTML report")
    _require(
        markdown_path.read_bytes() == expected_markdown
        and html_path.read_bytes() == expected_html,
        "final reports were not produced deterministically from M4",
    )
    performance_terminal = artifact_reference(root, "three_way_performance", m4_path)
    material_ok, structural_benefit, equivalence = final._validate_equal_work_report(
        equal_work,
        source_commit=source_commit,
        candidate_package=package_path,
        candidate_payload=candidate_payload,
    )
    raw_seconds = equal_work.get("performance", {}).get("raw_seconds")
    numerical = equal_work.get("numerical", {})
    no_fallback = sum(
        item for item in _values(equal_work, "cpu_fallback_count") if type(item) is int
    ) == 0
    value: dict[str, Any] = {
        "schema": final.QUALIFICATION_SCHEMA,
        "state": "PASS",
        "source_commit": source_commit,
        "candidate_receipt_id": candidate_id,
        "package": metadata,
        "installed_provenance": installed,
        "qualification": qualification,
        "performance": {
            "state": "PASS",
            "report": performance_terminal,
            "lanes": list(final.PERFORMANCE_LANES),
            "repetitions": 2,
            "equal_work": equal_work.get("performance", {}).get("outcome") == "PASS",
            "actual_seconds_retained": isinstance(raw_seconds, list)
            and len(raw_seconds) == 180,
            "numerical_oracles_pass": numerical.get("outcome") == "PASS"
            and numerical.get("comparison_count") == 141,
            "no_cpu_fallback": no_fallback,
            "no_material_v110_regression": material_ok,
            "structural_optimization_benefit": structural_benefit,
            "frozen_v103": {
                "version": "1.0.3",
                "source_commit": final.FROZEN_V103_COMMIT,
                "package_sha256": final.FROZEN_V103_PACKAGE_SHA256,
            },
            "v110_package_sha256": equivalence["measured_package_sha256"],
        },
        "report_plots": plots,
    }
    audit_path = _inside(final_audit_authority, root, "independent final audit authority")
    audit_hash = _sha256(audit_path)

    def verify(retained: dict[str, Any]) -> None:
        final._verify_qualification(
            retained,
            evidence_root=root,
            source_commit=source_commit,
            candidate_receipt_id=candidate_id,
            package_name=package_path.name,
            package_size=package_path.stat().st_size,
            package_sha256=package_sha256,
            package_path=package_path,
            archive_members=critical,
            candidate_payload=candidate_payload,
            final_audit_authority_path=audit_path,
            expected_final_audit_authority_sha256=audit_hash,
        )

    verify(value)
    _write_new_json(
        output, value, root=root, label="post-package qualification manifest",
        verify=lambda path: verify(_load(path, "post-package qualification manifest")),
    )
    _require(_identity(m4_path) == m4_identity, "M4 terminal changed during qualification creation")
    return value


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    candidate = sub.add_parser("candidate-wrapper")
    candidate.add_argument("--role", required=True)
    candidate.add_argument("--raw-terminal", type=pathlib.Path, required=True)
    candidate.add_argument("--release-source-commit", required=True)
    candidate.add_argument("--evidence-root", type=pathlib.Path, required=True)
    candidate.add_argument("--output", type=pathlib.Path, required=True)

    reports = sub.add_parser("reports")
    reports.add_argument("--m4-output", type=pathlib.Path, required=True)
    reports.add_argument("--evidence-root", type=pathlib.Path, required=True)
    reports.add_argument("--markdown-output", type=pathlib.Path, required=True)
    reports.add_argument("--html-output", type=pathlib.Path, required=True)

    wrapper = sub.add_parser("qualification-role")
    wrapper.add_argument("--role", required=True)
    wrapper.add_argument("--raw-terminal", type=pathlib.Path, required=True)
    wrapper.add_argument("--package", type=pathlib.Path, required=True)
    wrapper.add_argument("--candidate-receipt", type=pathlib.Path, required=True)
    wrapper.add_argument("--evidence-root", type=pathlib.Path, required=True)
    wrapper.add_argument("--installed-report", type=pathlib.Path, action="append", default=[])
    wrapper.add_argument("--output", type=pathlib.Path, required=True)

    glibc = sub.add_parser("extract-glibc")
    glibc.add_argument("--package", type=pathlib.Path, required=True)
    glibc.add_argument("--candidate-receipt", type=pathlib.Path, required=True)
    glibc.add_argument("--evidence-root", type=pathlib.Path, required=True)
    glibc.add_argument("--output", type=pathlib.Path, required=True)

    audit = sub.add_parser("final-audit")
    audit.add_argument("--audit-findings", type=pathlib.Path, required=True)
    audit.add_argument("--human-decision", type=pathlib.Path, required=True)
    audit.add_argument("--package", type=pathlib.Path, required=True)
    audit.add_argument("--candidate-receipt", type=pathlib.Path, required=True)
    audit.add_argument("--evidence-root", type=pathlib.Path, required=True)
    audit.add_argument("--output", type=pathlib.Path, required=True)

    manifest = sub.add_parser("candidate-manifest")
    manifest.add_argument("--role-file", action="append", default=[], required=True)
    manifest.add_argument("--release-source-commit", required=True)
    manifest.add_argument("--evidence-root", type=pathlib.Path, required=True)
    manifest.add_argument("--source-transition-policy", type=pathlib.Path, required=True)
    manifest.add_argument("--source-transition-policy-sha256", required=True)
    manifest.add_argument("--source-transition-verifier", type=pathlib.Path, required=True)
    manifest.add_argument("--source-transition-verifier-sha256", required=True)
    manifest.add_argument("--output", type=pathlib.Path, required=True)

    inversion = sub.add_parser("inversion-records")
    inversion.add_argument("--m4-output", type=pathlib.Path, required=True)
    inversion.add_argument("--closure-plan", type=pathlib.Path, required=True)
    inversion.add_argument("--evidence-root", type=pathlib.Path, required=True)
    inversion.add_argument("--output-directory", type=pathlib.Path, required=True)

    inversion_evidence = sub.add_parser("inversion-evidence")
    inversion_evidence.add_argument("--kind", required=True)
    inversion_evidence.add_argument("--workload-id", required=True)
    inversion_evidence.add_argument("--classification", required=True)
    inversion_evidence.add_argument("--m4-output", type=pathlib.Path, required=True)
    inversion_evidence.add_argument("--sample", action="append", default=[], required=True)
    inversion_evidence.add_argument("--intervention")
    inversion_evidence.add_argument("--evidence-root", type=pathlib.Path, required=True)
    inversion_evidence.add_argument("--output", type=pathlib.Path, required=True)

    close_registry = sub.add_parser("close-registry")
    close_registry.add_argument("--records-directory", type=pathlib.Path, required=True)
    close_registry.add_argument("--evidence-root", type=pathlib.Path, required=True)
    close_registry.add_argument("--output", type=pathlib.Path, required=True)

    qualification = sub.add_parser("qualification-manifest")
    qualification.add_argument("--package", type=pathlib.Path, required=True)
    qualification.add_argument("--candidate-receipt", type=pathlib.Path, required=True)
    qualification.add_argument("--evidence-root", type=pathlib.Path, required=True)
    qualification.add_argument("--m4-terminal", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu22-prefix", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu22-report", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu22-witness", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu22-self-check", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu24-prefix", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu24-report", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu24-witness", type=pathlib.Path, required=True)
    qualification.add_argument("--ubuntu24-self-check", type=pathlib.Path, required=True)
    qualification.add_argument("--role-file", action="append", default=[], required=True)
    qualification.add_argument("--final-audit-authority", type=pathlib.Path, required=True)
    qualification.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        _require_isolated()
        if args.command == "candidate-wrapper":
            value = create_candidate_wrapper(
                role=args.role, raw_terminal=args.raw_terminal,
                release_source_commit=args.release_source_commit,
                evidence_root=args.evidence_root, output=args.output,
            )
            identity = value["report_id"]
        elif args.command == "reports":
            plots = create_reports(
                m4_output=args.m4_output, markdown_output=args.markdown_output,
                html_output=args.html_output, evidence_root=args.evidence_root,
            )
            identity = plots[0]["sha256"]
        elif args.command == "qualification-role":
            value = create_qualification_wrapper(
                role=args.role, raw_terminal=args.raw_terminal, package=args.package,
                candidate_receipt=args.candidate_receipt, evidence_root=args.evidence_root,
                output=args.output, installed_reports=args.installed_report,
            )
            identity = value["report_id"]
        elif args.command == "extract-glibc":
            value = extract_glibc_terminal(
                package=args.package, candidate_receipt=args.candidate_receipt,
                evidence_root=args.evidence_root, output=args.output,
            )
            identity = _canonical_sha256(value)
        elif args.command == "final-audit":
            value = create_final_audit(
                findings_input=args.audit_findings, human_decision=args.human_decision,
                package=args.package, candidate_receipt=args.candidate_receipt,
                evidence_root=args.evidence_root, output=args.output,
            )
            identity = value["terminal_id"]
        elif args.command == "candidate-manifest":
            value = create_candidate_manifest(
                role_files=_parse_role_files(args.role_file),
                release_source_commit=args.release_source_commit,
                evidence_root=args.evidence_root, output=args.output,
                source_policy=args.source_transition_policy,
                source_policy_sha256=args.source_transition_policy_sha256,
                source_verifier=args.source_transition_verifier,
                source_verifier_sha256=args.source_transition_verifier_sha256,
            )
            identity = _canonical_sha256(value)
        elif args.command == "inversion-records":
            value = create_inversion_records(
                m4_output=args.m4_output, closure_plan=args.closure_plan,
                evidence_root=args.evidence_root, output_directory=args.output_directory,
            )
            identity = value.get("inventory_sha256") or _canonical_sha256(value)
        elif args.command == "inversion-evidence":
            value = create_inversion_evidence(
                kind=args.kind, workload_id=args.workload_id,
                classification=args.classification, m4_output=args.m4_output,
                samples=args.sample, intervention=args.intervention,
                evidence_root=args.evidence_root, output=args.output,
            )
            identity = value["evidence_id"]
        elif args.command == "close-registry":
            value = create_closed_registry(
                records_directory=args.records_directory,
                evidence_root=args.evidence_root,
                output=args.output,
            )
            identity = value["terminal_id"]
        else:
            value = create_qualification_manifest(
                package=args.package,
                candidate_receipt=args.candidate_receipt,
                evidence_root=args.evidence_root,
                m4_terminal=args.m4_terminal,
                ubuntu22_prefix=args.ubuntu22_prefix,
                ubuntu22_report=args.ubuntu22_report,
                ubuntu22_witness=args.ubuntu22_witness,
                ubuntu22_self_check=args.ubuntu22_self_check,
                ubuntu24_prefix=args.ubuntu24_prefix,
                ubuntu24_report=args.ubuntu24_report,
                ubuntu24_witness=args.ubuntu24_witness,
                ubuntu24_self_check=args.ubuntu24_self_check,
                role_files=_parse_role_files(args.role_file),
                final_audit_authority=args.final_audit_authority,
                output=args.output,
            )
            identity = _canonical_sha256(value)
        print(json.dumps({"schema": "gpmeep-v110-release-evidence-result-v1", "state": "PASS", "command": args.command, "content_id": identity}, sort_keys=True))
        return 0
    except (EvidenceCreationError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"v1.1.0 release evidence: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
