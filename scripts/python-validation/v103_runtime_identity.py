#!/usr/bin/env python3
"""Verify the exact frozen gpmeep v1.0.3 runtime used by M4.

This module is deliberately independent of the v1.1 package verifier.  The
historical package is never rebuilt: its package bytes, release manifest,
installed provenance report, Python module, extension, and mapped libmeep are
all checked against the one accepted v1.0.3 artifact before its unguarded
historical Simulation methods may be called.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
from typing import Any


FROZEN_PACKAGE_NAME = "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda"
FROZEN_PACKAGE_SHA256 = (
    "fd834ca51019ce2f18b26cb1f7252933205e008cdf1b2d9598b54e58ea37ebf3"
)
FROZEN_SOURCE_COMMIT = "b48c9b678f2316cefa95e48a661f7a877547a000"
FROZEN_PROVENANCE_REPORT_SHA256 = (
    "c8598f275ce328d9c02a971061e29fc2864f6707a49f39cda0d3c7901158a0cd"
)
FROZEN_PROVENANCE_REPORT_ID = (
    "6c5d6e8266a33f8b6b85fa8b5b54c539e5519d74a0ed0cc58eb25b20e0f1cb17"
)
FROZEN_VERSION = "1.0.3"
REPORT_KEYS = {
    "schema_version",
    "state",
    "prefix",
    "conda_record",
    "package_url",
    "package_sha256",
    "ownership",
    "manifest",
    "runtime",
    "report_id",
}
RUNTIME_RECORDS = {
    "python": "python_executable",
    "module": "meep_module",
    "extension": "extension",
    "libmeep": "libmeep",
}
ENV_REPORT = "GPMEEP_V103_PROVENANCE_REPORT"
ENV_PACKAGE = "GPMEEP_V103_PACKAGE"
ENV_REPORT_ID = "GPMEEP_V103_REPORT_ID"


class V103IdentityError(RuntimeError):
    """Raised when the frozen-runtime identity is incomplete or different."""


def _reject_constant(value: str) -> None:
    raise V103IdentityError(f"v1.0.3 provenance contains non-finite value {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V103IdentityError(f"v1.0.3 provenance duplicates key {key!r}")
        result[key] = value
    return result


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_path(path: pathlib.Path, *, directory: bool, label: str) -> pathlib.Path:
    provided = pathlib.Path(os.path.abspath(path))
    try:
        metadata = provided.lstat()
    except OSError as exc:
        raise V103IdentityError(f"{label} is unavailable: {exc}") from exc
    if provided.is_symlink():
        raise V103IdentityError(f"{label} must not be a symlink")
    if directory != provided.is_dir() or not directory and not provided.is_file():
        raise V103IdentityError(f"{label} has the wrong path type")
    if provided.resolve(strict=True) != provided:
        raise V103IdentityError(f"{label} is not canonical")
    if metadata.st_size < 0:
        raise V103IdentityError(f"{label} has invalid metadata")
    return provided


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V103IdentityError(f"cannot read v1.0.3 provenance: {exc}") from exc
    if not isinstance(value, dict):
        raise V103IdentityError("v1.0.3 provenance root must be an object")
    return value


def _record_path(record: Any, label: str) -> pathlib.Path:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["sha256"])
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
    ):
        raise V103IdentityError(f"{label} file record differs")
    path = _canonical_path(pathlib.Path(record["path"]), directory=False, label=label)
    if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
        raise V103IdentityError(f"{label} bytes differ from provenance")
    return path


def _package_from_url(value: Any) -> pathlib.Path:
    if not isinstance(value, str):
        raise V103IdentityError("v1.0.3 package URL is absent")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise V103IdentityError("v1.0.3 package URL is not a local file URL")
    return pathlib.Path(urllib.parse.unquote(parsed.path))


def validate_report(
    report_path: pathlib.Path,
    package_path: pathlib.Path,
    install_prefix: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, pathlib.Path]]:
    """Validate a saved installed-verifier report and all bound live bytes."""
    report_path = _canonical_path(report_path, directory=False, label="provenance report")
    package_path = _canonical_path(package_path, directory=False, label="v1.0.3 package")
    install_prefix = _canonical_path(
        install_prefix, directory=True, label="v1.0.3 install prefix"
    )
    if sha256_file(report_path) != FROZEN_PROVENANCE_REPORT_SHA256:
        raise V103IdentityError("frozen v1.0.3 provenance report bytes differ")
    report = _load_json(report_path)
    if set(report) != REPORT_KEYS or report.get("schema_version") != 1 or report.get("state") != "PASS":
        raise V103IdentityError("v1.0.3 provenance schema/state differs")
    retained_id = report.get("report_id")
    without_id = {key: value for key, value in report.items() if key != "report_id"}
    expected_id = hashlib.sha256(_canonical_json(without_id)).hexdigest()
    if retained_id != FROZEN_PROVENANCE_REPORT_ID or retained_id != expected_id:
        raise V103IdentityError("v1.0.3 provenance report ID differs")
    if pathlib.Path(str(report.get("prefix", ""))).resolve() != install_prefix:
        raise V103IdentityError("v1.0.3 provenance prefix differs")
    if (
        package_path.name != FROZEN_PACKAGE_NAME
        or report.get("package_sha256") != FROZEN_PACKAGE_SHA256
        or sha256_file(package_path) != FROZEN_PACKAGE_SHA256
        or _package_from_url(report.get("package_url")).resolve(strict=True) != package_path
    ):
        raise V103IdentityError("frozen v1.0.3 package identity differs")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or (
        manifest.get("distribution") != "gpmeep"
        or manifest.get("distribution_version") != FROZEN_VERSION
        or manifest.get("source_commit") != FROZEN_SOURCE_COMMIT
        or manifest.get("precision") != "fp32"
        or manifest.get("mpi") != "openmpi"
        or manifest.get("cuda_toolkit") != "12.4"
        or manifest.get("cuda_fast_math") is not False
        or manifest.get("scheme") is not False
    ):
        raise V103IdentityError("frozen v1.0.3 release manifest differs")
    paths: dict[str, pathlib.Path] = {
        "provenance_report": report_path,
        "package": package_path,
        "install_prefix": install_prefix,
        "conda_record": _record_path(report.get("conda_record"), "conda record"),
        "release_manifest": _record_path(manifest.get("file"), "release manifest"),
    }
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("single_precision") is not True
        or runtime.get("mpi_enabled") is not True
        or runtime.get("cuda_compiled") is not True
    ):
        raise V103IdentityError("frozen v1.0.3 runtime capability differs")
    for source_name, path_name in RUNTIME_RECORDS.items():
        paths[path_name] = _record_path(runtime.get(source_name), path_name)
        try:
            paths[path_name].relative_to(install_prefix)
        except ValueError as exc:
            raise V103IdentityError(f"{path_name} escaped the install prefix") from exc
    if paths["meep_module"].parent.name != "meep":
        raise V103IdentityError("frozen v1.0.3 module layout differs")
    if paths["extension"].parent != paths["meep_module"].parent:
        raise V103IdentityError("frozen v1.0.3 extension layout differs")
    return report, paths


def runtime_contract(report: dict[str, Any], paths: dict[str, pathlib.Path]) -> dict[str, Any]:
    """Map installed v1.0.3 provenance into the validation-hook contract."""
    fontconfig = (paths["install_prefix"] / "etc/fonts/fonts.conf").resolve(strict=True)
    if not fontconfig.is_file():
        raise V103IdentityError("frozen v1.0.3 fontconfig file is absent")
    return {
        "receipt_id": report["report_id"],
        "python_executable": report["runtime"]["python"],
        "meep_module": report["runtime"]["module"],
        "extension": report["runtime"]["extension"],
        "libmeep": report["runtime"]["libmeep"],
        "installed_environment": str(paths["install_prefix"]),
        "fontconfig_file": {
            "path": str(fontconfig),
            "sha256": sha256_file(fontconfig),
            "size_bytes": fontconfig.stat().st_size,
        },
    }


def _mapped_libmeep_paths() -> set[pathlib.Path]:
    maps = pathlib.Path("/proc/self/maps")
    try:
        lines = maps.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise V103IdentityError(f"cannot inspect mapped libraries: {exc}") from exc
    result = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "libmeep.so" in pathlib.Path(fields[5]).name:
            result.add(pathlib.Path(fields[5]).resolve(strict=True))
    return result


def verify_current_process(mp: Any) -> dict[str, Any]:
    """Fail closed unless this process loaded the exact frozen v1.0.3 runtime."""
    raw_report = os.environ.get(ENV_REPORT)
    raw_package = os.environ.get(ENV_PACKAGE)
    retained_id = os.environ.get(ENV_REPORT_ID)
    if not raw_report or not raw_package or not retained_id:
        raise V103IdentityError("frozen v1.0.3 identity environment is incomplete")
    prefix = pathlib.Path(sys.prefix).resolve(strict=True)
    report, paths = validate_report(
        pathlib.Path(raw_report), pathlib.Path(raw_package), prefix
    )
    if retained_id != report["report_id"]:
        raise V103IdentityError("frozen v1.0.3 report ID environment differs")
    loaded_module = pathlib.Path(mp.__file__).resolve(strict=True)
    extension_module = __import__("meep._meep", fromlist=["_meep"])
    loaded_extension = pathlib.Path(extension_module.__file__).resolve(strict=True)
    loaded_python = pathlib.Path(sys.executable).resolve(strict=True)
    if (
        loaded_python != paths["python_executable"]
        or loaded_module != paths["meep_module"]
        or loaded_extension != paths["extension"]
    ):
        raise V103IdentityError("loaded frozen v1.0.3 Python runtime paths differ")
    mapped = _mapped_libmeep_paths()
    if mapped != {paths["libmeep"]}:
        raise V103IdentityError("mapped frozen v1.0.3 libmeep differs")
    # Rehash after the import/mapping checks so path replacement cannot hide
    # behind the saved provenance report.
    for name in ("python_executable", "meep_module", "extension", "libmeep"):
        record_name = next(key for key, value in RUNTIME_RECORDS.items() if value == name)
        record = report["runtime"][record_name]
        if sha256_file(paths[name]) != record["sha256"]:
            raise V103IdentityError(f"loaded {name} bytes changed")
    return {
        "schema": "gpmeep-frozen-v103-process-identity-v1",
        "report_id": report["report_id"],
        "package_sha256": FROZEN_PACKAGE_SHA256,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "python": str(paths["python_executable"]),
        "module": str(paths["meep_module"]),
        "extension": str(paths["extension"]),
        "libmeep": str(paths["libmeep"]),
    }


__all__ = [
    "ENV_PACKAGE",
    "ENV_REPORT",
    "ENV_REPORT_ID",
    "FROZEN_PACKAGE_NAME",
    "FROZEN_PACKAGE_SHA256",
    "FROZEN_PROVENANCE_REPORT_ID",
    "FROZEN_PROVENANCE_REPORT_SHA256",
    "FROZEN_SOURCE_COMMIT",
    "FROZEN_VERSION",
    "V103IdentityError",
    "runtime_contract",
    "sha256_file",
    "validate_report",
    "verify_current_process",
]
