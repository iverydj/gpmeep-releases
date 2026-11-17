#!/usr/bin/env python3
"""Bind M4 measurements to one exact installed gpmeep v1.1.0 package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Any


PACKAGE_NAME = re.compile(
    r"^gpmeep-1\.1\.0-[A-Za-z0-9_.-]+\.conda$"
)
DEVELOPMENT_PACKAGE_NAME = re.compile(
    r"^gpmeep-1\.1\.0-[A-Za-z0-9_.-]+_development\.conda$"
)
NONPUBLISHABLE_MARKER_SUFFIX = ".NON_PUBLISHABLE"
NONPUBLISHABLE_MARKER_SCHEMA = "gpmeep-non-publishable-development-artifact-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
    "v110_candidate_gate",
    "report_id",
}
RUNTIME_KEYS = {
    "python",
    "module",
    "extension",
    "libmeep",
    "meep_version",
    "cuda_compiled",
    "mpi_enabled",
    "single_precision",
}
RUNTIME_RECORDS = {
    "python": "python_executable",
    "module": "meep_module",
    "extension": "extension",
    "libmeep": "libmeep",
}
ARCHIVE_INSPECTOR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "v110-final-release-receipt.py"
)


class V110IdentityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V110IdentityError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"installed provenance duplicates key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise V110IdentityError(f"installed provenance contains non-finite value {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_archive_records(
    package_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read exact install semantics from the external package archive.

    The installed ``conda-meta`` record is useful evidence, but it is writable
    state inside the measured prefix.  Link destinations and modes therefore
    come from the independently retained ``.conda`` archive.
    """
    name = "gpmeep_v110_runtime_archive_inspector"
    inspector_path = pathlib.Path(os.path.abspath(ARCHIVE_INSPECTOR))
    try:
        metadata = inspector_path.lstat()
        working_source = inspector_path.read_bytes()
    except OSError as error:
        raise V110IdentityError(
            f"v1.1.0 archive inspector is unavailable: {error}"
        ) from error
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and inspector_path.resolve(strict=True) == inspector_path
        and len(working_source) == metadata.st_size
        and len(working_source) <= 4 * 1024 * 1024,
        "v1.1.0 archive inspector path differs",
    )
    repo = inspector_path.parents[1]
    try:
        relative = inspector_path.relative_to(repo).as_posix()
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(repo), "show", f"HEAD:{relative}"],
            cwd="/",
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise V110IdentityError(
            f"v1.1.0 committed archive inspector is unavailable: {error}"
        ) from error
    _require(
        completed.returncode == 0
        and len(completed.stdout) <= 4 * 1024 * 1024
        and completed.stdout == working_source,
        "v1.1.0 archive inspector differs from the exact source commit",
    )
    spec = importlib.util.spec_from_file_location(name, inspector_path)
    _require(
        spec is not None and spec.loader is not None,
        "v1.1.0 archive inspector is unavailable",
    )
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        code = compile(completed.stdout, str(inspector_path), "exec")
        exec(code, module.__dict__)
        index, payload, semantics = module._conda_payload_files(
            package_path, "M4 exact v1.1.0 package"
        )
    except Exception as error:
        raise V110IdentityError(
            f"v1.1.0 package archive semantics differ: {error}"
        ) from error
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    records: dict[str, dict[str, Any]] = {}
    for relative, semantic in semantics.items():
        member = payload.get(relative)
        _require(
            member is not None
            and type(member.mode) is int
            and 0 <= member.mode <= 0o7777,
            f"v1.1.0 archive member metadata differs: {relative}",
        )
        if semantic.get("path_type") == "hardlink":
            _require(
                member.member_type == "regular"
                and semantic.get("sha256") == member.sha256
                and semantic.get("size_in_bytes") == member.size_bytes,
                f"v1.1.0 archive regular-file semantics differ: {relative}",
            )
            records[relative] = {
                "path_type": "hardlink",
                "sha256": semantic["sha256"],
                "size_bytes": semantic["size_in_bytes"],
                "mode": member.mode,
                "content_mode": member.mode,
            }
        else:
            target_member = payload.get(semantic.get("link_target"))
            while target_member is not None and target_member.link_target is not None:
                target_member = payload.get(target_member.link_target)
            _require(
                semantic.get("path_type") == "softlink"
                and member.member_type == "symlink"
                and target_member is not None
                and target_member.member_type == "regular"
                and isinstance(semantic.get("link_target"), str)
                and HEX64.fullmatch(str(semantic.get("target_sha256", "")))
                is not None
                and type(semantic.get("target_size_in_bytes")) is int
                and semantic["target_size_in_bytes"] >= 0,
                f"v1.1.0 archive softlink semantics differ: {relative}",
            )
            records[relative] = {
                "path_type": "softlink",
                "link_target": semantic["link_target"],
                "sha256": semantic["target_sha256"],
                "size_bytes": semantic["target_size_in_bytes"],
                "mode": member.mode,
                "content_mode": target_member.mode,
            }
    _require(
        set(records) == set(payload),
        "v1.1.0 archive ownership inventory differs",
    )
    return index, records


def _normalized_link_target(relative: pathlib.PurePosixPath, target: str) -> str:
    _require(target not in {"", "."}, "v1.1.0 conda softlink target is unsafe")
    candidate = pathlib.PurePosixPath(target)
    _require(
        not candidate.is_absolute() and ".." not in candidate.parts,
        "v1.1.0 conda softlink target is unsafe",
    )
    normalized = pathlib.PurePosixPath(relative.parent, candidate)
    parts: list[str] = []
    for part in normalized.parts:
        if part in {"", "."}:
            continue
        _require(part != "..", "v1.1.0 conda softlink target is unsafe")
        parts.append(part)
    _require(bool(parts), "v1.1.0 conda softlink target is unsafe")
    return pathlib.PurePosixPath(*parts).as_posix()


def _canonical_path(
    path: pathlib.Path, *, directory: bool, label: str
) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path))
    try:
        metadata = lexical.lstat()
    except OSError as error:
        raise V110IdentityError(f"{label} is unavailable: {error}") from error
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    _require(
        lexical.is_dir() if directory else lexical.is_file(),
        f"{label} has the wrong path type",
    )
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(metadata.st_size >= 0, f"{label} metadata differs")
    return resolved


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V110IdentityError(f"cannot read installed provenance: {error}") from error
    _require(isinstance(value, dict), "installed provenance root must be an object")
    return value


def _record_path(
    record: Any, *, prefix: pathlib.Path | None, label: str
) -> pathlib.Path:
    _require(
        isinstance(record, dict)
        and set(record) == {"path", "sha256", "size_bytes"}
        and isinstance(record.get("path"), str)
        and HEX64.fullmatch(str(record.get("sha256", ""))) is not None
        and type(record.get("size_bytes")) is int
        and record["size_bytes"] >= 0,
        f"{label} file record differs",
    )
    path = _canonical_path(pathlib.Path(record["path"]), directory=False, label=label)
    if prefix is not None:
        try:
            path.relative_to(prefix)
        except ValueError as error:
            raise V110IdentityError(f"{label} escaped the installed prefix") from error
    _require(path.stat().st_size == record["size_bytes"], f"{label} size differs")
    _require(sha256_file(path) == record["sha256"], f"{label} bytes differ")
    return path


def _package_from_url(value: Any) -> pathlib.Path:
    _require(isinstance(value, str), "installed provenance package URL is absent")
    parsed = urllib.parse.urlparse(value)
    _require(
        parsed.scheme == "file" and parsed.netloc in ("", "localhost"),
        "installed provenance package URL is not a local file URL",
    )
    return pathlib.Path(urllib.parse.unquote(parsed.path))


def _validate_nonpublishable_marker(
    package_path: pathlib.Path, package_sha256: str, source_commit: str
) -> pathlib.Path:
    marker = _canonical_path(
        pathlib.Path(f"{package_path}{NONPUBLISHABLE_MARKER_SUFFIX}"),
        directory=False,
        label="v1.1.0 non-publishable marker",
    )
    metadata = marker.stat()
    _require(
        stat.S_IMODE(metadata.st_mode) == 0o400 and metadata.st_size <= 4096,
        "v1.1.0 non-publishable marker metadata differs",
    )
    try:
        value = json.loads(
            marker.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V110IdentityError(
            f"cannot read v1.1.0 non-publishable marker: {error}"
        ) from error
    _require(
        value
        == {
            "schema": NONPUBLISHABLE_MARKER_SCHEMA,
            "state": "NON_PUBLISHABLE",
            "source_commit": source_commit,
            "package": {
                "file_name": package_path.name,
                "size_bytes": package_path.stat().st_size,
                "sha256": package_sha256,
            },
        },
        "v1.1.0 non-publishable marker content differs",
    )
    return marker


def validate_report(
    report_path: pathlib.Path,
    package_path: pathlib.Path,
    install_prefix: pathlib.Path,
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, pathlib.Path]]:
    """Validate saved provenance, package bytes, and all measured runtime files."""
    _require(
        HEX40.fullmatch(expected_source_commit) is not None,
        "expected source commit is invalid",
    )
    report_path = _canonical_path(report_path, directory=False, label="v1.1.0 provenance report")
    package_path = _canonical_path(package_path, directory=False, label="v1.1.0 package")
    install_prefix = _canonical_path(install_prefix, directory=True, label="v1.1.0 install prefix")
    report = _load(report_path)
    _require(
        set(report) == REPORT_KEYS
        and report.get("schema_version") == 1
        and report.get("state") == "PASS",
        "v1.1.0 provenance schema/state differs",
    )
    unsigned = dict(report)
    report_id = unsigned.pop("report_id")
    _require(
        isinstance(report_id, str)
        and HEX64.fullmatch(report_id) is not None
        and hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() == report_id,
        "v1.1.0 provenance report ID differs",
    )
    _require(
        pathlib.Path(str(report.get("prefix", ""))) == install_prefix,
        "v1.1.0 provenance prefix differs",
    )
    package_sha256 = sha256_file(package_path)
    _require(PACKAGE_NAME.fullmatch(package_path.name) is not None, "v1.1.0 package name differs")
    _require(report.get("package_sha256") == package_sha256, "v1.1.0 package hash differs")
    _require(
        _package_from_url(report.get("package_url")).resolve(strict=True) == package_path,
        "v1.1.0 package URL differs from the measured package",
    )
    manifest = report.get("manifest")
    _require(
        isinstance(manifest, dict)
        and manifest.get("distribution") == "gpmeep"
        and manifest.get("distribution_version") == "1.1.0"
        and manifest.get("source_commit") == expected_source_commit
        and manifest.get("precision") == "fp32"
        and manifest.get("mpi") == "openmpi"
        and manifest.get("cuda_fast_math") is False
        and manifest.get("scheme") is False,
        "v1.1.0 release manifest differs",
    )
    gate = report.get("v110_candidate_gate")
    build_mode = gate.get("build_mode") if isinstance(gate, dict) else None
    expected_state = {
        "development": "DEVELOPMENT_ONLY",
        "candidate": "CANDIDATE_BUILD_AUTHORIZED",
    }.get(build_mode)
    _require(
        isinstance(gate, dict)
        and expected_state is not None
        and gate.get("state") == expected_state
        and gate.get("source_commit") == expected_source_commit
        and HEX64.fullmatch(str(gate.get("receipt_id", ""))) is not None,
        "M4 requires an authorized development or candidate v1.1.0 package",
    )
    marker_path = pathlib.Path(f"{package_path}{NONPUBLISHABLE_MARKER_SUFFIX}")
    if build_mode == "development":
        _require(
            DEVELOPMENT_PACKAGE_NAME.fullmatch(package_path.name) is not None,
            "v1.1.0 development package name differs",
        )
        _validate_nonpublishable_marker(
            package_path, package_sha256, expected_source_commit
        )
    else:
        _require(
            DEVELOPMENT_PACKAGE_NAME.fullmatch(package_path.name) is None,
            "v1.1.0 candidate package carries a development name",
        )
        _require(
            not marker_path.exists() and not marker_path.is_symlink(),
            "v1.1.0 candidate package carries a non-publishable marker",
        )
    runtime = report.get("runtime")
    _require(
        isinstance(runtime, dict)
        and set(runtime) == RUNTIME_KEYS
        and runtime.get("cuda_compiled") is True
        and runtime.get("mpi_enabled") is True
        and runtime.get("single_precision") is True,
        "v1.1.0 runtime capability differs",
    )
    paths: dict[str, pathlib.Path] = {
        "provenance_report": report_path,
        "package": package_path,
        "install_prefix": install_prefix,
        "conda_record": _record_path(report.get("conda_record"), prefix=install_prefix, label="conda record"),
        "release_manifest": _record_path(manifest.get("file"), prefix=install_prefix, label="release manifest"),
    }
    for source_name, path_name in RUNTIME_RECORDS.items():
        paths[path_name] = _record_path(
            runtime.get(source_name), prefix=install_prefix, label=path_name
        )
    _require(paths["meep_module"].parent.name == "meep", "v1.1.0 module layout differs")
    _require(paths["extension"].parent == paths["meep_module"].parent, "v1.1.0 extension layout differs")
    try:
        report_path.relative_to(install_prefix)
    except ValueError:
        pass
    else:
        raise V110IdentityError("M4 provenance report must be prefix-external")
    try:
        package_path.relative_to(install_prefix)
    except ValueError:
        pass
    else:
        raise V110IdentityError("M4 package must be prefix-external")
    return report, paths


def package_owned_records(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> tuple[dict[str, Any], ...]:
    """Independently validate every file owned by the installed package.

    The package-shipped provenance executable is useful defence in depth, but
    it cannot be the trust root for its own bytes.  The externally validated
    conda record supplies an immutable inventory and per-file hashes that this
    controller checks before invoking that executable.
    """
    prefix = paths["install_prefix"]
    conda_record = _load(paths["conda_record"])
    _require(
        conda_record.get("name") == "gpmeep"
        and conda_record.get("version") == "1.1.0"
        and conda_record.get("sha256") == report["package_sha256"]
        and conda_record.get("fn") == paths["package"].name,
        "v1.1.0 conda package identity differs",
    )
    archive_index, archive_records = _package_archive_records(paths["package"])
    expected_build = {
        "development": "cuda124_mpi_openmpi_py311_0_development",
        "candidate": "cuda124_mpi_openmpi_py311_0",
    }[report["v110_candidate_gate"]["build_mode"]]
    _require(
        archive_index.get("name") == "gpmeep"
        and archive_index.get("version") == "1.1.0"
        and archive_index.get("subdir") == "linux-64"
        and archive_index.get("build") == expected_build
        and conda_record.get("build") == expected_build
        and conda_record.get("subdir") == "linux-64",
        "v1.1.0 conda archive identity differs",
    )
    paths_data = conda_record.get("paths_data")
    _require(
        isinstance(paths_data, dict)
        and set(paths_data) == {"paths", "paths_version"}
        and paths_data.get("paths_version") == 1
        and isinstance(paths_data.get("paths"), list),
        "v1.1.0 conda paths inventory differs",
    )
    entries = paths_data["paths"]
    files = conda_record.get("files")
    _require(
        isinstance(files, list)
        and all(isinstance(item, str) for item in files)
        and len(files) == len(set(files)) == len(entries),
        "v1.1.0 conda file census differs",
    )
    _require(
        set(files) == set(archive_records),
        "v1.1.0 installed and archived ownership inventories differ",
    )
    records: list[dict[str, Any]] = []
    inventory_paths: list[str] = []
    for entry in entries:
        _require(isinstance(entry, dict), "v1.1.0 conda path entry differs")
        path_type = entry.get("path_type")
        expected_keys = {
            "_path",
            "path_type",
            "sha256_in_prefix",
            "size_in_bytes",
        }
        if path_type == "hardlink":
            expected_keys.add("sha256")
        _require(
            path_type in {"hardlink", "softlink"}
            and set(entry) == expected_keys
            and isinstance(entry.get("_path"), str)
            and HEX64.fullmatch(str(entry.get("sha256_in_prefix", "")))
            is not None
            and type(entry.get("size_in_bytes")) is int
            and entry["size_in_bytes"] >= 0,
            "v1.1.0 conda path record differs",
        )
        if path_type == "hardlink":
            _require(
                entry.get("sha256") == entry["sha256_in_prefix"],
                "v1.1.0 conda hardlink hashes differ",
            )
        raw_relative = entry["_path"]
        relative = pathlib.PurePosixPath(raw_relative)
        _require(
            raw_relative == relative.as_posix()
            and not relative.is_absolute()
            and relative.parts
            and all(part not in {"", ".", ".."} for part in relative.parts),
            "v1.1.0 conda owned path is unsafe",
        )
        archived = archive_records.get(raw_relative)
        _require(
            archived is not None and archived["path_type"] == path_type,
            "v1.1.0 installed path type differs from exact archive",
        )
        _require(
            entry["sha256_in_prefix"] == archived["sha256"]
            and entry["size_in_bytes"] == archived["size_bytes"],
            "v1.1.0 installed ownership bytes differ from exact archive",
        )
        lexical = pathlib.Path(os.path.abspath(prefix.joinpath(*relative.parts)))
        try:
            lexical.relative_to(prefix)
        except ValueError as error:
            raise V110IdentityError(
                "v1.1.0 conda owned path escaped the prefix"
            ) from error
        _require(
            lexical.parent.resolve(strict=True) == lexical.parent,
            "v1.1.0 conda owned path contains a symlink ancestor",
        )
        try:
            metadata = lexical.lstat()
        except OSError as error:
            raise V110IdentityError(
                f"v1.1.0 conda owned path is unavailable: {error}"
            ) from error
        _require(
            stat.S_IMODE(metadata.st_mode) == archived["mode"],
            "v1.1.0 conda owned path mode differs from exact archive",
        )
        link_target: str | None = None
        content_path = lexical
        if path_type == "hardlink":
            _require(
                stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                "v1.1.0 conda hardlink path type differs",
            )
        else:
            _require(
                stat.S_ISLNK(metadata.st_mode),
                "v1.1.0 conda softlink path type differs",
            )
            try:
                link_target = os.readlink(lexical)
                normalized_target = _normalized_link_target(relative, link_target)
                content_path = lexical.resolve(strict=True)
                content_path.relative_to(prefix)
            except (OSError, ValueError) as error:
                raise V110IdentityError(
                    "v1.1.0 conda softlink target is unsafe"
                ) from error
            _require(
                normalized_target == archived["link_target"],
                "v1.1.0 conda softlink target differs from exact archive",
            )
            try:
                exact_archive_target = pathlib.PurePosixPath(
                    archived["link_target"]
                ).relative_to(relative.parent).as_posix()
            except ValueError as error:
                raise V110IdentityError(
                    "v1.1.0 archive softlink target is not canonical"
                ) from error
            _require(
                link_target == exact_archive_target,
                "v1.1.0 conda softlink raw target differs from exact archive",
            )
            _require(
                content_path.is_file() and not content_path.is_symlink(),
                "v1.1.0 conda softlink target type differs",
            )
        _require(
            content_path.stat().st_size == entry["size_in_bytes"]
            and sha256_file(content_path) == entry["sha256_in_prefix"],
            "v1.1.0 conda owned file bytes differ",
        )
        inventory_paths.append(raw_relative)
        records.append(
            {
                "path": lexical,
                "path_type": path_type,
                "sha256": entry["sha256_in_prefix"],
                "size_bytes": entry["size_in_bytes"],
                "link_target": link_target,
                "archive_link_target": archived.get("link_target"),
                "mode": archived["mode"],
                "content_mode": archived["content_mode"],
                "content_path": content_path,
            }
        )
    _require(files == inventory_paths, "v1.1.0 conda file ordering differs")
    _require(
        report.get("ownership")
        == {
            "owned_paths": len(records),
            "regular_files": sum(
                item["path_type"] == "hardlink" for item in records
            ),
            "symlinks": sum(
                item["path_type"] == "softlink" for item in records
            ),
            "content_hashes_verified": len(records),
            "content_sizes_verified": len(records),
            "prefix_rewritten_sizes": 0,
        },
        "v1.1.0 package ownership census differs",
    )
    return tuple(records)


def runtime_paths(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> dict[str, pathlib.Path]:
    prefix = paths["install_prefix"]
    site_packages = paths["meep_module"].parent.parent
    candidates = {
        "python": paths["python_executable"],
        "build_python": site_packages,
        "install_prefix": prefix,
        "build_receipt": paths["provenance_report"],
        "mpiexec": (prefix / "bin/mpirun").resolve(strict=True),
        "package": paths["package"],
        "provenance_report": paths["provenance_report"],
        "verifier": (prefix / "bin/gpmeep-verify-provenance").resolve(strict=True),
    }
    _require(site_packages.is_dir(), "v1.1.0 site-packages directory is absent")
    for name in ("python", "mpiexec", "verifier"):
        path = candidates[name]
        _require(path.is_file() and os.access(path, os.X_OK), f"v1.1.0 {name} is not executable")
        try:
            path.relative_to(prefix)
        except ValueError as error:
            raise V110IdentityError(f"v1.1.0 {name} escaped the prefix") from error
    return candidates


def runtime_contract(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> dict[str, Any]:
    fontconfig = (paths["install_prefix"] / "etc/fonts/fonts.conf").resolve(strict=True)
    _require(fontconfig.is_file(), "v1.1.0 fontconfig file is absent")
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


def live_verify(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> dict[str, Any]:
    """Run the verifier shipped by the exact package and compare its full report."""
    executable = runtime_paths(report, paths)["verifier"]
    prefix = paths["install_prefix"]
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise V110IdentityError(f"verification account home is unavailable: {error}") from error
    environment = {
        "HOME": str(account_home),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(prefix / "lib"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MEEP_GPU_BACKEND": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with tempfile.TemporaryDirectory(prefix="gpmeep-v110-live-provenance-") as raw:
        output = pathlib.Path(raw).resolve() / "report.json"
        command = [
            str(executable),
            "--prefix",
            str(prefix),
            "--expected-source-commit",
            report["manifest"]["source_commit"],
            "--expected-package-sha256",
            report["package_sha256"],
            "--report",
            str(output),
        ]
        try:
            process = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise V110IdentityError("installed provenance verifier timed out") from error
        _require(
            process.returncode == 0
            and len(process.stdout) <= 16 * 1024**2
            and len(process.stderr) <= 16 * 1024**2,
            "installed provenance verifier failed or exceeded output bounds",
        )
        live = _load(output)
    _require(live == report, "live installed provenance differs from retained report")
    return {
        "report_id": report["report_id"],
        "candidate_receipt_id": report["v110_candidate_gate"]["receipt_id"],
        "build_mode": report["v110_candidate_gate"]["build_mode"],
        "package_sha256": report["package_sha256"],
        "verifier": {
            "path": str(executable),
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
    }


def snapshot(
    repo: pathlib.Path,
    report: dict[str, Any],
    paths: dict[str, pathlib.Path],
    contract: dict[str, Any],
    live: dict[str, Any],
    *,
    captured_at_utc: str,
) -> dict[str, Any]:
    """Return an M3-compatible snapshot backed by installed package evidence."""
    return {
        "captured_at_utc": captured_at_utc,
        "available": True,
        "source_snapshot": __import__("run_validation").source_snapshot(repo),
        "build_extension": {
            "available": True,
            **report["runtime"]["extension"],
        },
        "build_receipt": {
            "available": True,
            "path": str(paths["provenance_report"]),
            "sha256": sha256_file(paths["provenance_report"]),
            "receipt_id": report["report_id"],
            "candidate_receipt_id": report["v110_candidate_gate"]["receipt_id"],
            "package_sha256": report["package_sha256"],
            "live_verification": live,
        },
        "runtime_contract": contract,
        "problems": [],
    }
