#!/usr/bin/env python3
"""Seal or replay external v1.1.0 release authority for exact package bytes."""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import hashlib
import html as html_module
import importlib.util
import io
import json
import math
import os
import pathlib
import posixpath
import pwd
import re
import resource
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from html.parser import HTMLParser
from typing import Any


SCHEMA = "gpmeep-v110-final-release-v1"
QUALIFICATION_SCHEMA = "gpmeep-v110-post-package-qualification-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_NAME = re.compile(r"^gpmeep-1\.1\.0-[A-Za-z0-9_.-]+\.conda$")
MEASURED_PACKAGE_BUILDS = {
    "cuda124_mpi_openmpi_py311_0_development",
    "cuda124_mpi_openmpi_py311_0",
}
PAYLOAD_DIFFERENCE_ALLOWLIST = {
    "share/gpmeep/v110-candidate-build-receipt.json",
}
# The current package is approximately 22.4 MiB unpacked. These fixed limits
# retain ample release headroom while bounding compressed input and decoder
# output independently.
MAX_CONDA_COMPRESSED_MEMBER_BYTES = 128 * 1024 * 1024
MAX_CONDA_DECOMPRESSED_MEMBER_BYTES = 512 * 1024 * 1024
MAX_CONDA_FILES = 200_000
CANDIDATE_RECEIPT_PATH = "share/gpmeep/v110-candidate-build-receipt.json"
GLIBC_AUDIT_PATH = "share/gpmeep/glibc-compatibility.json"
INSTALLED_VERIFIER_PATH = "share/gpmeep/libexec/gpmeep-verify-provenance.py"
INSTALLED_LAUNCHER_PATH = "bin/gpmeep-verify-provenance"
INSTALLED_SELF_CHECK_PATH = "share/gpmeep/libexec/gpmeep-self-check.py"
INSTALLED_RELEASE_MANIFEST_PATH = "share/gpmeep/release.json"
FROZEN_V103_COMMIT = "b48c9b678f2316cefa95e48a661f7a877547a000"
FROZEN_V103_PACKAGE_SHA256 = "fd834ca51019ce2f18b26cb1f7252933205e008cdf1b2d9598b54e58ea37ebf3"
PERFORMANCE_LANES = (
    "upstream_meep_cpu",
    "gpmeep_v110_cpu",
    "gpmeep_v103_cuda_gpu1",
    "gpmeep_v110_cuda_gpu1",
    "gpmeep_v103_cuda_gpu2",
    "gpmeep_v110_cuda_gpu2",
)
M4_COMPLETE_SCHEMA = "gpmeep-v110-m4-equal-work-complete-v1"
FINAL_AUDIT_PRODUCER_SCHEMA = "gpmeep-v110-independent-final-audit-v1"
MAX_V110_V103_REGRESSION_FRACTION = 0.10
QUALIFICATION_ROLES = (
    "installed_correctness",
    "installed_multi_gpu",
    "installed_aunp",
    "ubuntu_22_04_native",
    "newer_ubuntu_native",
    "glibc_floor",
    "final_report_markdown",
    "final_report_html",
    "final_adversarial_audit",
    "closed_inversion_registry",
)
TEXT_ARTIFACT_SCHEMAS = {
    "final_report_markdown": "text/markdown",
    "final_report_html": "text/html",
}
QUALIFICATION_ROLE_SCHEMAS = {
    "installed_correctness": "gpmeep-v110-final-installed-correctness-v1",
    "installed_multi_gpu": "gpmeep-v110-final-installed-multi-gpu-v1",
    "installed_aunp": "gpmeep-v110-final-installed-aunp-v1",
    "ubuntu_22_04_native": "gpmeep-v110-native-environment-v1",
    "newer_ubuntu_native": "gpmeep-v110-native-environment-v1",
    "glibc_floor": "gpmeep-v110-final-glibc-floor-v1",
    "final_adversarial_audit": "gpmeep-v110-final-adversarial-audit-v1",
    "closed_inversion_registry": "gpmeep-v110-closed-inversion-registry-v1",
    "installed_provenance_0": "schema_version:1",
    "installed_provenance_1": "schema_version:1",
    "native_witness_0": "gpmeep-v110-native-witness-v1",
    "native_witness_1": "gpmeep-v110-native-witness-v1",
    "self_check_0": "schema_version:2",
    "self_check_1": "schema_version:2",
    "three_way_performance": M4_COMPLETE_SCHEMA,
}
FINAL_AUDIT_FINDINGS = tuple(f"F-{index:02d}" for index in range(1, 14))
EXPECTED_REPORT_PLOT_IDS = ("speedup",)


class FinalReleaseError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalReleaseError(message)


def _require_isolated_release_runtime() -> None:
    _require(sys.flags.isolated == 1, "final release authority requires Python isolated mode (-I)")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FinalReleaseError(f"non-finite JSON constant: {value}")


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise FinalReleaseError(f"invalid JSON: {path}") from error
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


def _identity(path: pathlib.Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _record(path: pathlib.Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_candidate_tool(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("gpmeep_v110_candidate_receipt", path)
    _require(spec is not None and spec.loader is not None, "candidate receipt verifier cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"producer cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def _safe_archive_name(value: str, label: str) -> str:
    path = pathlib.PurePosixPath(value)
    _require(
        value not in {"", "."}
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value,
        f"{label} has an unsafe path",
    )
    return value


def _zstd_decompress(payload: bytes, label: str) -> bytes:
    _require(pathlib.Path("/usr/bin/zstd").is_file(), "fixed zstd decoder is unavailable")
    _require(
        len(payload) <= MAX_CONDA_COMPRESSED_MEMBER_BYTES,
        f"{label} compressed member exceeds the 128 MiB limit",
    )
    # RLIMIT_FSIZE prevents the fixed decoder from growing the temporary file
    # beyond the accepted boundary even when the input is a compression bomb.
    # A zero return code distinguishes an exact-boundary stream from a decoder
    # killed (or failed) on its next write beyond that boundary.
    hard_limit = MAX_CONDA_DECOMPRESSED_MEMBER_BYTES

    def limit_decoder_output() -> None:
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        resource.setrlimit(resource.RLIMIT_FSIZE, (hard_limit, hard_limit))

    with tempfile.TemporaryFile(
        mode="w+b", prefix="gpmeep-v110-zstd-"
    ) as decoded:
        try:
            result = subprocess.run(
                ["/usr/bin/zstd", "-d", "-q", "-c"],
                input=payload,
                stdout=decoded,
                stderr=subprocess.PIPE,
                check=False,
                timeout=120,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                preexec_fn=limit_decoder_output,
            )
        except subprocess.TimeoutExpired as error:
            raise FinalReleaseError(f"{label} zstd decoding timed out") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise FinalReleaseError(f"{label} zstd decoder could not start") from error
        decoded_size = os.fstat(decoded.fileno()).st_size
        _require(
            decoded_size <= MAX_CONDA_DECOMPRESSED_MEMBER_BYTES
            and not (
                result.returncode != 0
                and decoded_size == MAX_CONDA_DECOMPRESSED_MEMBER_BYTES
            ),
            f"{label} zstd output exceeds the 512 MiB limit",
        )
        _require(result.returncode == 0, f"{label} zstd decoder failed")
        decoded.seek(0)
        value = decoded.read(MAX_CONDA_DECOMPRESSED_MEMBER_BYTES + 1)
    _require(
        len(value) == decoded_size
        and len(value) <= MAX_CONDA_DECOMPRESSED_MEMBER_BYTES,
        f"{label} zstd output exceeds the 512 MiB limit",
    )
    return value


@dataclasses.dataclass(frozen=True)
class ArchiveMember:
    """A conda payload member, including the metadata that affects installation."""

    path: str
    member_type: str
    mode: int
    size_bytes: int
    sha256: str | None
    link_target: str | None
    data: bytes | None = dataclasses.field(compare=False, repr=False)


def _safe_link_target(name: str, target: str, label: str, *, hardlink: bool) -> str:
    _require(target not in {"", "."}, f"{label} has an empty link target")
    candidate = pathlib.PurePosixPath(target)
    _require(not candidate.is_absolute(), f"{label} has an absolute link target")
    _require(".." not in candidate.parts, f"{label} link target contains '..'")
    base = "" if hardlink else str(pathlib.PurePosixPath(name).parent)
    resolved = posixpath.normpath(posixpath.join(base, target))
    _safe_archive_name(resolved, f"{label} link target")
    if hardlink:
        canonical_target = resolved
    else:
        try:
            canonical_target = pathlib.PurePosixPath(resolved).relative_to(
                pathlib.PurePosixPath(name).parent
            ).as_posix()
        except ValueError as error:
            raise FinalReleaseError(
                f"{label} link target is not canonical"
            ) from error
    _require(
        target == canonical_target,
        f"{label} link target is not canonical",
    )
    return resolved


def _validate_link_graph(files: dict[str, ArchiveMember], label: str) -> None:
    for name, member in files.items():
        if member.link_target is None:
            continue
        _require(member.link_target in files, f"{label} link {name} is dangling")
        visited = {name}
        cursor = member.link_target
        while files[cursor].link_target is not None:
            _require(cursor not in visited, f"{label} link graph contains a cycle")
            visited.add(cursor)
            cursor = files[cursor].link_target  # type: ignore[assignment]
            _require(cursor in files, f"{label} link {name} is dangling")
        _require(cursor not in visited, f"{label} link graph contains a cycle")
        _require(
            files[cursor].member_type in {"regular", "hardlink"},
            f"{label} link {name} does not resolve to a file",
        )


def _tar_files(payload: bytes, label: str) -> dict[str, ArchiveMember]:
    files: dict[str, ArchiveMember] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            _require(len(members) <= MAX_CONDA_FILES, f"{label} contains too many members")
            for member in members:
                name = _safe_archive_name(member.name, label)
                _require(name not in files, f"{label} duplicates member {name}")
                if member.isdir():
                    continue
                mode = stat.S_IMODE(member.mode)
                _require(0 <= mode <= 0o7777, f"{label} member mode is invalid")
                if member.issym() or member.islnk():
                    kind = "symlink" if member.issym() else "hardlink"
                    target = _safe_link_target(
                        name, member.linkname, f"{label} member {name}", hardlink=member.islnk()
                    )
                    files[name] = ArchiveMember(name, kind, mode, 0, None, target, None)
                    continue
                _require(member.isfile(), f"{label} contains a special member")
                _require(0 <= member.size <= MAX_CONDA_DECOMPRESSED_MEMBER_BYTES, f"{label} member is too large")
                stream = archive.extractfile(member)
                _require(stream is not None, f"{label} member cannot be read")
                data = stream.read(MAX_CONDA_DECOMPRESSED_MEMBER_BYTES + 1)
                _require(len(data) == member.size and len(data) <= MAX_CONDA_DECOMPRESSED_MEMBER_BYTES, f"{label} member size differs")
                files[name] = ArchiveMember(
                    name,
                    "regular",
                    mode,
                    len(data),
                    hashlib.sha256(data).hexdigest(),
                    None,
                    data,
                )
    except (tarfile.TarError, OSError) as error:
        raise FinalReleaseError(f"{label} is not a valid tar archive") from error
    _validate_link_graph(files, label)
    return files


def _member_bytes(member: ArchiveMember, label: str) -> bytes:
    _require(member.member_type == "regular" and member.data is not None, f"{label} is not a regular file")
    return member.data


def _member_identity(member: ArchiveMember) -> tuple[str, int, int, str | None, str | None]:
    return (
        member.member_type,
        member.mode,
        member.size_bytes,
        member.sha256,
        member.link_target,
    )


def _glibc_version_key(value: Any) -> tuple[int, ...]:
    _require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is not None,
        "GLIBC version is invalid",
    )
    return tuple(int(part) for part in value.split("."))


def _replay_candidate_glibc(
    candidate_payload: dict[str, ArchiveMember],
    producer_payload: dict[str, Any],
) -> dict[str, Any]:
    """Derive the complete GLIBC claim from the exact candidate archive bytes."""
    _require(GLIBC_AUDIT_PATH in candidate_payload, "candidate GLIBC audit is absent")
    embedded = _json_bytes(
        _member_bytes(candidate_payload[GLIBC_AUDIT_PATH], "candidate GLIBC audit"),
        "candidate GLIBC audit",
    )
    _require(producer_payload == embedded, "GLIBC producer differs from candidate package bytes")
    _exact(
        embedded,
        {
            "schema", "state", "maximum_allowed", "overall_maximum_required",
            "elf_count", "elf_files",
        },
        "candidate GLIBC audit",
    )
    _require(
        embedded["schema"] == "gpmeep-glibc-compatibility-v1"
        and embedded["state"] == "PASS"
        and embedded["maximum_allowed"] == "2.17",
        "candidate GLIBC audit policy differs",
    )
    records = embedded["elf_files"]
    _require(
        isinstance(records, list)
        and records
        and type(embedded["elf_count"]) is int
        and embedded["elf_count"] == len(records),
        "candidate GLIBC ELF inventory differs",
    )
    actual_elf_paths = {
        path
        for path, member in candidate_payload.items()
        if member.member_type == "regular"
        and member.data is not None
        and member.data.startswith(b"\x7fELF")
    }
    seen: set[str] = set()
    maxima: list[str] = []
    for record in records:
        _exact(
            record,
            {"path", "size_bytes", "sha256", "required_versions", "maximum_required"},
            "candidate GLIBC ELF record",
        )
        relative = record["path"]
        _require(
            isinstance(relative, str)
            and relative in candidate_payload
            and relative not in seen,
            "candidate GLIBC ELF path differs",
        )
        seen.add(relative)
        member = candidate_payload[relative]
        data = _member_bytes(member, f"candidate GLIBC ELF {relative}")
        versions = record["required_versions"]
        _require(
            data.startswith(b"\x7fELF")
            and record["size_bytes"] == len(data)
            and record["sha256"] == hashlib.sha256(data).hexdigest()
            and isinstance(versions, list)
            and versions
            and versions == sorted(set(versions), key=_glibc_version_key)
            and record["maximum_required"] == versions[-1]
            and _glibc_version_key(record["maximum_required"]) <= (2, 17),
            f"candidate GLIBC ELF identity differs: {relative}",
        )
        maxima.append(record["maximum_required"])
    _require(seen == actual_elf_paths, "candidate GLIBC audit does not cover every ELF")
    _require(
        embedded["overall_maximum_required"] == max(maxima, key=_glibc_version_key),
        "candidate GLIBC overall maximum differs",
    )
    return embedded


def _json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalReleaseError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _conda_install_semantics(
    info_files: dict[str, ArchiveMember],
    package_files: dict[str, ArchiveMember],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Validate and normalize every conda operation that can alter installed bytes."""
    _require(
        "info/paths.json" in info_files and "info/files" in info_files,
        f"{label} ownership metadata is incomplete",
    )
    paths_payload = _json_bytes(
        _member_bytes(info_files["info/paths.json"], f"{label} paths.json"),
        f"{label} paths.json",
    )
    _require(
        set(paths_payload) == {"paths_version", "paths"}
        and paths_payload["paths_version"] == 1
        and isinstance(paths_payload["paths"], list),
        f"{label} paths.json schema differs",
    )
    try:
        legacy_paths = [
            line
            for line in _member_bytes(info_files["info/files"], f"{label} info/files")
            .decode("utf-8")
            .splitlines()
            if line
        ]
    except UnicodeDecodeError as error:
        raise FinalReleaseError(f"{label} info/files is not UTF-8") from error
    _require(
        legacy_paths == sorted(set(legacy_paths))
        and set(legacy_paths) == set(package_files),
        f"{label} payload and legacy ownership inventories differ",
    )
    allowed_keys = {
        "_path", "path_type", "sha256", "sha256_in_prefix", "size_in_bytes",
        "no_link", "prefix_placeholder", "file_mode",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for entry in paths_payload["paths"]:
        _require(
            isinstance(entry, dict)
            and set(entry).issubset(allowed_keys)
            and isinstance(entry.get("_path"), str),
            f"{label} ownership entry schema differs",
        )
        name = entry["_path"]
        _safe_archive_name(name, f"{label} ownership")
        _require(name not in normalized and name in package_files, f"{label} ownership path differs: {name}")
        _require(
            "prefix_placeholder" not in entry
            and "file_mode" not in entry
            and "sha256_in_prefix" not in entry,
            f"{label} enables an unsupported install-time payload transform: {name}",
        )
        no_link = entry.get("no_link", False)
        _require(type(no_link) is bool, f"{label} no_link is invalid: {name}")
        member = package_files[name]
        path_type = entry.get("path_type")
        if path_type == "softlink":
            _require(member.member_type == "symlink", f"{label} softlink metadata differs: {name}")
            metadata_keys = set(entry) & {"sha256", "size_in_bytes"}
            _require(
                set(entry).issubset(
                    {"_path", "path_type", "sha256", "size_in_bytes", "no_link"}
                )
                and metadata_keys in (set(), {"sha256", "size_in_bytes"}),
                f"{label} softlink byte metadata schema differs: {name}",
            )
            target_name = member.link_target
            _require(
                isinstance(target_name, str),
                f"{label} softlink target is absent: {name}",
            )
            target = package_files[target_name]
            while target.link_target is not None:
                target = package_files[target.link_target]
            _require(
                target.member_type == "regular"
                and isinstance(target.sha256, str)
                and type(target.size_bytes) is int,
                f"{label} softlink target bytes are invalid: {name}",
            )
            if metadata_keys:
                _require(
                    entry.get("sha256") == target.sha256
                    and type(entry.get("size_in_bytes")) is int
                    and entry["size_in_bytes"] == target.size_bytes,
                    f"{label} softlink target hash/size differs: {name}",
                )
            normalized[name] = {
                "path_type": path_type,
                "link_target": member.link_target,
                "target_sha256": target.sha256,
                "target_size_in_bytes": target.size_bytes,
                "no_link": no_link,
            }
            continue
        _require(path_type == "hardlink" and member.member_type == "regular", f"{label} hardlink metadata differs: {name}")
        _require(
            set(entry).issubset({"_path", "path_type", "sha256", "size_in_bytes", "no_link"})
            and entry.get("sha256") == member.sha256
            and type(entry.get("size_in_bytes")) is int
            and entry["size_in_bytes"] == member.size_bytes,
            f"{label} ownership hash/size differs or is absent: {name}",
        )
        normalized[name] = {
            "path_type": path_type,
            "sha256": member.sha256,
            "size_in_bytes": member.size_bytes,
            "no_link": no_link,
        }
    _require(set(normalized) == set(package_files), f"{label} ownership inventory differs")
    return normalized


def _inspect_conda_v2(
    package_path: pathlib.Path,
    candidate_receipt_path: pathlib.Path,
    candidate_receipt_id: str,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, ArchiveMember]]:
    """Inspect, but never extract, the exact conda-v2 package bytes."""
    try:
        with zipfile.ZipFile(package_path, mode="r") as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            _require(len(names) == len(set(names)), "conda-v2 outer archive duplicates members")
            for item in members:
                _safe_archive_name(item.filename, "conda-v2 outer archive")
                _require(
                    item.file_size <= MAX_CONDA_COMPRESSED_MEMBER_BYTES,
                    "conda-v2 outer member exceeds the 128 MiB compressed limit",
                )
            info_names = [name for name in names if name.startswith("info-") and name.endswith(".tar.zst")]
            pkg_names = [name for name in names if name.startswith("pkg-") and name.endswith(".tar.zst")]
            _require(
                set(names) == {"metadata.json", *info_names, *pkg_names}
                and len(info_names) == 1
                and len(pkg_names) == 1,
                "conda-v2 outer member inventory differs",
            )
            metadata = _json_bytes(archive.read("metadata.json"), "conda-v2 metadata")
            _require(metadata == {"conda_pkg_format_version": 2}, "conda-v2 metadata differs")
            info_files = _tar_files(
                _zstd_decompress(archive.read(info_names[0]), "conda-v2 info member"),
                "conda-v2 info tar",
            )
            package_files = _tar_files(
                _zstd_decompress(archive.read(pkg_names[0]), "conda-v2 package member"),
                "conda-v2 package tar",
            )
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise FinalReleaseError("candidate package is not a valid conda-v2 archive") from error

    _require("info/index.json" in info_files and "info/paths.json" in info_files and "info/files" in info_files, "conda package metadata inventory is incomplete")
    index = _json_bytes(
        _member_bytes(info_files["info/index.json"], "conda info/index.json"),
        "conda info/index.json",
    )
    _require(
        index.get("name") == "gpmeep"
        and index.get("version") == "1.1.0"
        and index.get("build") == "cuda124_mpi_openmpi_py311_0"
        and index.get("subdir") == "linux-64",
        "conda package index identity differs",
    )
    install_semantics = _conda_install_semantics(info_files, package_files, "conda package")
    owned_paths = list(install_semantics)
    legacy_paths = sorted(package_files)
    critical = {
        CANDIDATE_RECEIPT_PATH,
        INSTALLED_VERIFIER_PATH,
        INSTALLED_LAUNCHER_PATH,
        INSTALLED_SELF_CHECK_PATH,
    }
    _require(critical.issubset(set(owned_paths)) and critical.issubset(set(legacy_paths)), "conda ownership omits release-trust files")
    _require(critical.issubset(package_files), "conda payload omits release-trust files")
    candidate_bytes = candidate_receipt_path.read_bytes()
    embedded_candidate = _member_bytes(
        package_files[CANDIDATE_RECEIPT_PATH], "embedded candidate receipt"
    )
    _require(embedded_candidate == candidate_bytes, "embedded candidate receipt bytes differ from the external receipt")
    embedded = _json_bytes(embedded_candidate, "embedded candidate receipt")
    _require(embedded.get("receipt_id") == candidate_receipt_id, "embedded candidate receipt ID differs")
    package_identity = {
        "file_name": package_path.name,
        "size_bytes": package_path.stat().st_size,
        "sha256": _sha256(package_path),
        "format": "conda-v2",
        "name": index["name"],
        "version": index["version"],
        "build": index["build"],
        "subdir": index["subdir"],
        "embedded_candidate_receipt_id": candidate_receipt_id,
    }
    return (
        package_identity,
        {
            "candidate_receipt": embedded_candidate,
            "installed_verifier": _member_bytes(package_files[INSTALLED_VERIFIER_PATH], "installed verifier"),
            "installed_launcher": _member_bytes(package_files[INSTALLED_LAUNCHER_PATH], "installed launcher"),
            "installed_self_check": _member_bytes(
                package_files[INSTALLED_SELF_CHECK_PATH], "installed self-check"
            ),
        },
        package_files,
    )


def _conda_payload_files(
    package_path: pathlib.Path, label: str
) -> tuple[dict[str, Any], dict[str, ArchiveMember], dict[str, dict[str, Any]]]:
    """Read one bounded conda-v2 payload without trusting its info metadata."""
    package_path = _regular_file(package_path, label)
    try:
        with zipfile.ZipFile(package_path, mode="r") as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            _require(len(names) == len(set(names)), f"{label} duplicates outer members")
            for item in members:
                _safe_archive_name(item.filename, label)
                _require(
                    item.file_size <= MAX_CONDA_COMPRESSED_MEMBER_BYTES,
                    f"{label} outer member exceeds the 128 MiB compressed limit",
                )
            info_names = [
                name
                for name in names
                if name.startswith("info-") and name.endswith(".tar.zst")
            ]
            pkg_names = [
                name
                for name in names
                if name.startswith("pkg-") and name.endswith(".tar.zst")
            ]
            _require(
                set(names) == {"metadata.json", *info_names, *pkg_names}
                and len(info_names) == 1
                and len(pkg_names) == 1,
                f"{label} outer inventory differs",
            )
            _require(
                _json_bytes(archive.read("metadata.json"), label)
                == {"conda_pkg_format_version": 2},
                f"{label} metadata differs",
            )
            info = _tar_files(
                _zstd_decompress(archive.read(info_names[0]), label), label
            )
            payload = _tar_files(
                _zstd_decompress(archive.read(pkg_names[0]), label), label
            )
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise FinalReleaseError(f"{label} is not a valid conda-v2 archive") from error
    _require("info/index.json" in info, f"{label} omits package index")
    index = _json_bytes(_member_bytes(info["info/index.json"], f"{label} index"), f"{label} index")
    _require(
        index.get("name") == "gpmeep"
        and index.get("version") == "1.1.0"
        and index.get("subdir") == "linux-64"
        and index.get("build") in MEASURED_PACKAGE_BUILDS,
        f"{label} package identity differs",
    )
    install_semantics = _conda_install_semantics(info, payload, label)
    return index, payload, install_semantics


def _runtime_payload_equivalence(
    measured_package: pathlib.Path,
    candidate_package: pathlib.Path,
    candidate_payload: dict[str, ArchiveMember],
) -> dict[str, Any]:
    """Prove candidate runtime bytes equal those used by M4.

    The candidate authorization receipt is the sole permitted payload change.
    Conda info metadata may differ because the development and candidate build
    strings differ, but it is not installed runtime payload.
    """
    measured_package = _regular_file(measured_package, "M4 measured v1.1.0 package")
    candidate_package = _regular_file(candidate_package, "candidate v1.1.0 package")
    _require(
        measured_package != candidate_package,
        "candidate package cannot be its own M4 measurement",
    )
    measured_sha256 = _sha256(measured_package)
    candidate_sha256 = _sha256(candidate_package)
    _require(
        measured_sha256 != candidate_sha256,
        "development and candidate package bytes must be distinct",
    )
    measured_index, measured_payload, measured_install_semantics = _conda_payload_files(
        measured_package, "M4 measured v1.1.0 package"
    )
    candidate_index, independently_read_candidate, candidate_install_semantics = _conda_payload_files(
        candidate_package, "candidate v1.1.0 package"
    )
    _require(
        measured_index["build"] == "cuda124_mpi_openmpi_py311_0_development"
        and candidate_index["build"] == "cuda124_mpi_openmpi_py311_0",
        "development-to-candidate conda build transition differs",
    )
    _require(
        independently_read_candidate == candidate_payload,
        "candidate payload changed during equivalence replay",
    )
    _require(
        set(measured_payload) == set(candidate_payload),
        "measured and candidate payload inventories differ",
    )
    changed = sorted(
        path
        for path in measured_payload
        if _member_identity(measured_payload[path]) != _member_identity(candidate_payload[path])
    )
    _require(
        set(changed) == PAYLOAD_DIFFERENCE_ALLOWLIST,
        f"candidate payload transition differs: {changed}",
    )
    _require(
        {
            path: value
            for path, value in measured_install_semantics.items()
            if path not in PAYLOAD_DIFFERENCE_ALLOWLIST
        }
        == {
            path: value
            for path, value in candidate_install_semantics.items()
            if path not in PAYLOAD_DIFFERENCE_ALLOWLIST
        },
        "development-to-candidate conda install semantics differ",
    )
    measured_receipt = _json_bytes(
        _member_bytes(
            measured_payload[CANDIDATE_RECEIPT_PATH],
            "measured development receipt",
        ),
        "measured development receipt",
    )
    candidate_receipt = _json_bytes(
        _member_bytes(candidate_payload[CANDIDATE_RECEIPT_PATH], "candidate receipt"),
        "candidate receipt",
    )
    _require(
        measured_receipt.get("build_mode") == "development"
        and candidate_receipt.get("build_mode") == "candidate",
        "embedded receipt build-mode transition differs",
    )
    _require(
        HEX40.fullmatch(candidate_receipt.get("measured_source_commit") or "") is not None
        and measured_receipt.get("source_commit")
        == candidate_receipt.get("measured_source_commit")
        and measured_receipt.get("receipt_id")
        != candidate_receipt.get("receipt_id"),
        "embedded receipt measured-source/identity transition differs",
    )
    runtime_paths = sorted(set(measured_payload) - PAYLOAD_DIFFERENCE_ALLOWLIST)
    _require(
        any(path.startswith("lib/libmeep.so") for path in runtime_paths)
        and any(
            path.startswith("lib/python3.11/site-packages/meep/")
            and pathlib.PurePosixPath(path).name.startswith("_meep")
            for path in runtime_paths
        ),
        "package equivalence omitted core libmeep/Python extension payload",
    )
    digest = hashlib.sha256()
    for path in runtime_paths:
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        identity = json.dumps(
            _member_identity(measured_payload[path]), separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
    return {
        "schema": "gpmeep-v110-runtime-payload-equivalence-v2",
        "state": "PASS",
        "measured_package_sha256": measured_sha256,
        "candidate_package_sha256": candidate_sha256,
        "measured_build": measured_index["build"],
        "candidate_build": candidate_index["build"],
        "measured_receipt_id": measured_receipt.get("receipt_id"),
        "candidate_receipt_id": candidate_receipt.get("receipt_id"),
        "measured_source_commit": candidate_receipt.get("measured_source_commit"),
        "runtime_file_count": len(runtime_paths),
        "runtime_payload_sha256": digest.hexdigest(),
        "allowed_changed_paths": changed,
    }


def _verify_installed_prefix(
    *,
    prefix: pathlib.Path,
    retained_report: dict[str, Any],
    archive_members: dict[str, bytes],
    source_commit: str,
    package_sha256: str,
    candidate_receipt_id: str,
) -> None:
    prefix = _regular_directory(prefix, "installed qualification prefix")
    verifier = _regular_file(prefix / INSTALLED_VERIFIER_PATH, "installed provenance verifier")
    launcher = _regular_file(prefix / INSTALLED_LAUNCHER_PATH, "installed provenance launcher")
    self_check = _regular_file(prefix / INSTALLED_SELF_CHECK_PATH, "installed self-check")
    _require(verifier.read_bytes() == archive_members["installed_verifier"], "installed verifier bytes differ from the package archive")
    _require(launcher.read_bytes() == archive_members["installed_launcher"], "installed launcher bytes differ from the package archive")
    _require(
        self_check.read_bytes() == archive_members["installed_self_check"],
        "installed self-check bytes differ from the package archive",
    )
    _require(os.access(launcher, os.X_OK), "installed provenance launcher is not executable")
    # MPI initialization needs a real home even when the outer receipt runs
    # with an empty environment. Do not inherit a caller-controlled HOME.
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise FinalReleaseError(f"verification account home is unavailable: {error}") from error
    _require(account_home.is_dir(), "verification account home is not a directory")
    with tempfile.TemporaryDirectory(prefix="gpmeep-v110-live-provenance-") as directory:
        report_path = pathlib.Path(directory) / "report.json"
        try:
            result = subprocess.run(
                [
                    str(launcher),
                    "--prefix", str(prefix),
                    "--expected-source-commit", source_commit,
                    "--expected-package-sha256", package_sha256,
                    "--report", str(report_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=300,
                env={
                    "HOME": str(account_home),
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONNOUSERSITE": "1",
                },
            )
        except subprocess.TimeoutExpired as error:
            raise FinalReleaseError("installed provenance replay timed out") from error
        _require(result.returncode == 0, "installed provenance verifier replay failed")
        _require(report_path.is_file() and not report_path.is_symlink(), "installed provenance replay omitted its report")
        replayed = _load(report_path)
    _require(replayed == retained_report, "installed provenance report was not exactly re-derived")
    _require(replayed.get("state") == "PASS", "installed provenance replay did not pass")
    _require(replayed.get("prefix") == str(prefix), "installed provenance prefix differs")
    _require(replayed.get("package_sha256") == package_sha256, "installed provenance package differs")
    _require(replayed.get("manifest", {}).get("source_commit") == source_commit, "installed provenance source differs")
    _require(
        replayed.get("v110_candidate_gate", {}).get("receipt_id") == candidate_receipt_id,
        "installed provenance candidate receipt differs",
    )


def _replay_equal_work(output: pathlib.Path) -> dict[str, Any]:
    producer = _load_module(
        "gpmeep_v110_final_equal_work_replay",
        pathlib.Path(__file__).resolve().parent
        / "python-validation/run_m4_equal_work_matrix.py",
    )
    try:
        return producer.verify_complete(output)
    except Exception as error:
        raise FinalReleaseError(f"official equal-work replay failed: {error}") from error


def _replay_equal_work_registry(output: pathlib.Path) -> dict[str, Any]:
    validator = _load_module(
        "gpmeep_v110_final_normative_inversion_registry",
        pathlib.Path(__file__).resolve().parent
        / "validate-v110-inversion-registry.py",
    )
    try:
        result = validator.validate_registry(
            require_closed=False,
            evidence_root=output,
            records_directory=output / "inversion-records",
        )
    except Exception as error:
        raise FinalReleaseError(
            f"normative equal-work inversion registry replay failed: {error}"
        ) from error
    expected = {
        "schema_version": 1,
        "observed_count": 13,
        "registered_count": 13,
        "missing_count": 0,
        "missing_workloads": [],
    }
    _require(
        all(result.get(key) == value for key, value in expected.items()),
        "normative equal-work inversion registry result differs",
    )
    blockers = result.get("blocking_workloads")
    open_workloads = result.get("open_workloads")
    _require(
        result.get("state") in {"PASS", "OPEN"}
        and isinstance(blockers, list)
        and blockers == sorted(set(blockers))
        and isinstance(open_workloads, list)
        and open_workloads == sorted(set(open_workloads))
        and set(blockers).issubset(open_workloads)
        and result.get("blocking_count") == len(blockers)
        and result.get("open_count") == len(open_workloads)
        and result.get("closed_count") + result.get("open_count") == 13
        and (result.get("state") == "PASS") == (not open_workloads),
        "normative equal-work inversion closure computation differs",
    )
    return result


def _replay_closed_inversion_registry(
    terminal_path: pathlib.Path, evidence_root: pathlib.Path
) -> dict[str, Any]:
    producer = _load_module(
        "gpmeep_v110_final_closed_inversion_registry",
        pathlib.Path(__file__).resolve().parent
        / "close-v110-inversion-registry.py",
    )
    try:
        result = producer.verify(terminal_path, evidence_root)
    except Exception as error:
        raise FinalReleaseError(
            f"closed inversion registry replay failed: {error}"
        ) from error
    _require(
        result.get("schema") == "gpmeep-v110-closed-inversion-registry-v1"
        and result.get("state") == "PASS"
        and result.get("registry_result", {}).get("state") == "PASS"
        and result.get("registry_result", {}).get("blocking_count") == 0
        and result.get("registry_result", {}).get("closed_count") == 13
        and result.get("registry_result", {}).get("open_count") == 0
        and result.get("registry_result", {}).get("open_workloads") == [],
        "closed inversion registry did not prove closure",
    )
    return result


def _validate_equal_work_report(
    report: dict[str, Any], *, source_commit: str,
    candidate_package: pathlib.Path,
    candidate_payload: dict[str, ArchiveMember],
) -> tuple[bool, bool, dict[str, Any]]:
    _require(report.get("schema") == "gpmeep-v110-m4-equal-work-matrix-v1", "equal-work report schema differs")
    _require(report.get("outcome") == "PASS", "equal-work report did not pass")
    performance = report.get("performance")
    numerical = report.get("numerical")
    _require(
        isinstance(performance, dict)
        and performance.get("schema") == "gpmeep-v110-m4-equal-work-aggregation-v2"
        and performance.get("outcome") == "PASS"
        and performance.get("task_count") == 180
        and performance.get("measured_task_count") == 120
        and isinstance(performance.get("raw_seconds"), list)
        and len(performance["raw_seconds"]) == 180
        and isinstance(performance.get("cases"), dict)
        and len(performance["cases"]) == 13,
        "equal-work performance replay census differs",
    )
    _require(
        isinstance(numerical, dict)
        and numerical.get("schema") == "gpmeep-v110-m4-numerical-replay-v1"
        and numerical.get("outcome") == "PASS"
        and numerical.get("comparison_count") == 141
        and isinstance(numerical.get("comparisons"), list)
        and len(numerical["comparisons"]) == 141,
        "equal-work numerical replay census differs",
    )
    contract = report.get("contract")
    _require(isinstance(contract, dict), "equal-work controller contract is absent")
    official_plan = _load_module(
        "gpmeep_v110_final_equal_work_plan",
        pathlib.Path(__file__).resolve().parent / "python-validation/m4_equal_work_plan.py",
    ).load_plan(pathlib.Path(__file__).resolve().parents[1])
    _require(contract.get("plan") == official_plan, "equal-work plan differs from the official contract")
    inputs = contract.get("runtime_inputs", {})
    candidate_receipt = _json_bytes(
        _member_bytes(candidate_payload[CANDIDATE_RECEIPT_PATH], "candidate receipt"),
        "candidate receipt",
    )
    measured_source_commit = candidate_receipt.get("measured_source_commit")
    _require(
        HEX40.fullmatch(measured_source_commit or "") is not None,
        "candidate receipt measured-source anchor differs",
    )
    measured_package = _regular_file(
        pathlib.Path(contract.get("paths", {}).get("v110_package", "")),
        "equal-work measured v1.1.0 package",
    )
    measured_package_sha256 = _sha256(measured_package)
    _require(
        inputs.get("v103_package", {}).get("sha256") == FROZEN_V103_PACKAGE_SHA256,
        "equal-work frozen v1.0.3 package differs",
    )
    _require(
        inputs.get("v110_source_identity", {}).get("git_head") == measured_source_commit,
        "equal-work v1.1.0 measured-source commit differs",
    )
    _require(
        inputs.get("v110_package_sha256") == measured_package_sha256,
        "equal-work controller is not bound to its exact measured v1.1.0 package",
    )
    equivalence = _runtime_payload_equivalence(
        measured_package, candidate_package, candidate_payload
    )
    material_ok = True
    structural_benefit = False
    for case_path, case in performance["cases"].items():
        _require(isinstance(case, dict), f"equal-work case differs: {case_path}")
        ratios = case.get("ratios", {})
        gpu1 = ratios.get("v103-gpu1_to_v110-gpu1", {})
        ratio1 = gpu1.get("ratio_of_average_seconds")
        _require(type(ratio1) in {int, float} and math.isfinite(ratio1) and ratio1 > 0, "equal-work GPU1 ratio differs")
        material_ok = material_ok and ratio1 >= 1.0 - MAX_V110_V103_REGRESSION_FRACTION
        structural_benefit = structural_benefit or ratio1 > 1.0
        if "gpmeep_v110_cuda_gpu2" in case.get("lanes", {}):
            gpu2 = ratios.get("v103-gpu2_to_v110-gpu2", {})
            ratio2 = gpu2.get("ratio_of_average_seconds")
            _require(type(ratio2) in {int, float} and math.isfinite(ratio2) and ratio2 > 0, "equal-work GPU2 ratio differs")
            material_ok = material_ok and ratio2 >= 1.0 - MAX_V110_V103_REGRESSION_FRACTION
            structural_benefit = structural_benefit or ratio2 > 1.0
    return material_ok, structural_benefit, equivalence


def _artifact_ref(
    value: Any,
    role: str,
    evidence_root: pathlib.Path,
) -> tuple[pathlib.Path, Any]:
    record = _exact(
        value,
        {"role", "bundle_relative_path", "size_bytes", "sha256", "schema", "content_id"},
        f"qualification.{role}",
    )
    _require(record["role"] == role, f"qualification.{role} role differs")
    relative = pathlib.PurePosixPath(record["bundle_relative_path"])
    _require(
        not relative.is_absolute()
        and relative.parts
        and ".." not in relative.parts
        and str(relative) == record["bundle_relative_path"],
        f"qualification.{role} path is unsafe",
    )
    _require(type(record["size_bytes"]) is int and record["size_bytes"] > 0, f"qualification.{role} size is invalid")
    _require(HEX64.fullmatch(record["sha256"] or "") is not None, f"qualification.{role} hash is invalid")
    expected_schema = TEXT_ARTIFACT_SCHEMAS.get(role) or QUALIFICATION_ROLE_SCHEMAS.get(role)
    _require(expected_schema is not None, f"qualification.{role} has no fixed producer schema")
    _require(record["schema"] == expected_schema, f"qualification.{role} producer schema differs")
    _require(HEX64.fullmatch(record["content_id"] or "") is not None, f"qualification.{role} content ID is invalid")
    path = _regular_file(evidence_root.joinpath(*relative.parts), f"qualification.{role}")
    artifact_identity = _identity(path)
    _require(path.is_relative_to(evidence_root), f"qualification.{role} escaped evidence root")
    _require(path.stat().st_size == record["size_bytes"], f"qualification.{role} size differs")
    _require(_sha256(path) == record["sha256"], f"qualification.{role} hash differs")
    if role in TEXT_ARTIFACT_SCHEMAS:
        _require(
            record["schema"] == TEXT_ARTIFACT_SCHEMAS[role],
            f"qualification.{role} media type differs",
        )
        _require(
            record["content_id"] == record["sha256"],
            f"qualification.{role} content ID differs",
        )
        try:
            payload = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise FinalReleaseError(
                f"qualification.{role} is not UTF-8 text"
            ) from error
        _require(payload.strip() != "", f"qualification.{role} is empty")
        if role == "final_report_html":
            lower = payload.lower()
            _require("<html" in lower, "final HTML report lacks an html document")
            _require("data:image/" in lower, "final HTML report does not embed its images")
        _require(
            _identity(path) == artifact_identity,
            f"qualification.{role} changed during verification",
        )
        return path, payload
    payload = _load(path)
    payload_schema = payload.get("schema")
    if payload_schema is None and type(payload.get("schema_version")) is int:
        payload_schema = f"schema_version:{payload['schema_version']}"
    _require(payload_schema == expected_schema, f"qualification.{role} schema differs")
    identity = next(
        (
            payload.get(key)
            for key in ("receipt_id", "report_id", "content_id", "plan_id", "terminal_id")
            if payload.get(key) is not None
        ),
        _canonical_sha256(payload),
    )
    _require(identity == record["content_id"], f"qualification.{role} content ID differs")
    _require(
        _identity(path) == artifact_identity,
        f"qualification.{role} changed during verification",
    )
    return path, payload


class _ImageCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        values = dict(attrs)
        plot_id = values.get("data-plot-id")
        source = values.get("src")
        if not isinstance(plot_id, str) or not isinstance(source, str):
            raise FinalReleaseError("final HTML image lacks data-plot-id or src")
        self.images.append({"id": plot_id, "src": source})


def _decode_png(payload: bytes) -> tuple[int, int]:
    """Strictly decode the bounded PNG container and its compressed pixels."""
    _require(payload.startswith(b"\x89PNG\r\n\x1a\n"), "embedded PNG magic differs")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        _require(offset + 12 <= len(payload), "embedded PNG is truncated")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        kind = payload[offset + 4 : offset + 8]
        _require(length <= 64 * 1024 * 1024, "embedded PNG chunk is too large")
        end = offset + 12 + length
        _require(end <= len(payload), "embedded PNG chunk is truncated")
        data = payload[offset + 8 : offset + 8 + length]
        retained_crc = int.from_bytes(payload[offset + 8 + length : end], "big")
        computed_crc = binascii.crc32(kind + data) & 0xFFFFFFFF
        _require(retained_crc == computed_crc, "embedded PNG CRC differs")
        chunks.append((kind, data))
        offset = end
        if kind == b"IEND":
            break
    _require(offset == len(payload), "embedded PNG has trailing bytes")
    _require(chunks and chunks[0][0] == b"IHDR", "embedded PNG omits leading IHDR")
    _require(chunks[-1] == (b"IEND", b""), "embedded PNG omits terminal IEND")
    _require(sum(kind == b"IHDR" for kind, _ in chunks) == 1, "embedded PNG repeats IHDR")
    _require(sum(kind == b"IEND" for kind, _ in chunks) == 1, "embedded PNG repeats IEND")
    idat_indices = [index for index, (kind, _data) in enumerate(chunks) if kind == b"IDAT"]
    _require(
        idat_indices
        and idat_indices == list(range(idat_indices[0], idat_indices[-1] + 1)),
        "embedded PNG IDAT chunks are absent or nonconsecutive",
    )
    known_critical = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
    _require(
        all(kind in known_critical or kind[0] & 0x20 for kind, _data in chunks),
        "embedded PNG has an unknown critical chunk",
    )
    ihdr = chunks[0][1]
    _require(len(ihdr) == 13, "embedded PNG IHDR differs")
    width = int.from_bytes(ihdr[0:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    _require(0 < width <= 32768 and 0 < height <= 32768, "embedded PNG dimensions are invalid")
    _require(ihdr[10:13] == b"\x00\x00\x00", "embedded PNG compression/filter/interlace differs")
    bit_depth = ihdr[8]
    color_type = ihdr[9]
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    _require(
        color_type in valid_depths and bit_depth in valid_depths[color_type],
        "embedded PNG color type/bit depth differs",
    )
    plte_indices = [index for index, (kind, _data) in enumerate(chunks) if kind == b"PLTE"]
    _require(len(plte_indices) <= 1, "embedded PNG repeats PLTE")
    if color_type == 3:
        _require(
            plte_indices and plte_indices[0] < idat_indices[0],
            "embedded indexed PNG lacks a leading PLTE",
        )
    if plte_indices:
        palette = chunks[plte_indices[0]][1]
        palette_entries = len(palette) // 3
        _require(
            color_type not in {0, 4}
            and plte_indices[0] < idat_indices[0]
            and len(palette) % 3 == 0
            and 0 < len(palette) <= 3 * 256,
            "embedded PNG palette differs",
        )
        if color_type == 3:
            _require(
                palette_entries <= 2**bit_depth,
                "embedded indexed PNG palette exceeds its bit depth",
            )
    idat = b"".join(data for kind, data in chunks if kind == b"IDAT")
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = (width * samples * bit_depth + 7) // 8
    expected_pixel_bytes = height * (row_bytes + 1)
    _require(
        0 < expected_pixel_bytes <= 128 * 1024 * 1024,
        "embedded PNG decoded pixel size exceeds its bound",
    )
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(idat, 128 * 1024 * 1024 + 1)
        _require(
            len(pixels) == expected_pixel_bytes
            and decoder.eof
            and not decoder.unused_data
            and not decoder.unconsumed_tail,
            "embedded PNG pixel stream length differs or is incomplete",
        )
    except (zlib.error, TypeError) as error:
        raise FinalReleaseError("embedded PNG pixel stream is invalid") from error
    _require(
        all(pixels[row * (row_bytes + 1)] <= 4 for row in range(height)),
        "embedded PNG scanline filter differs",
    )
    return width, height


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        len(data).to_bytes(4, "big")
        + kind
        + data
        + (binascii.crc32(kind + data) & 0xFFFFFFFF).to_bytes(4, "big")
    )


def _expected_speedup_png(equal_work_report: dict[str, Any]) -> bytes:
    """Render the sole accepted plot deterministically from replayed ratios."""
    cases = equal_work_report["performance"]["cases"]
    ratios: list[float] = []
    for case_name in sorted(cases):
        for ratio_name in sorted(cases[case_name].get("ratios", {})):
            value = cases[case_name]["ratios"][ratio_name].get(
                "ratio_of_average_seconds"
            )
            _require(
                type(value) in {int, float}
                and math.isfinite(value)
                and value > 0,
                "speedup plot ratio is invalid",
            )
            ratios.append(float(value))
    _require(ratios, "speedup plot has no replayed ratios")
    width = max(96, 16 + 8 * len(ratios))
    height = 96
    _require(width <= 4096, "speedup plot ratio census is too large")
    pixels = bytearray([255] * (width * height * 3))

    def set_pixel(x: int, y: int, rgb: tuple[int, int, int]) -> None:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(rgb)

    baseline_y = height - 12
    for x in range(8, width - 4):
        set_pixel(x, baseline_y, (0, 0, 0))
    scale_max = max(1.0, max(ratios))
    for index, value in enumerate(ratios):
        x0 = 10 + index * 8
        bar_height = max(1, round((height - 20) * min(value, scale_max) / scale_max))
        color = (32, 112, 208) if value >= 1.0 else (208, 72, 64)
        for x in range(x0, min(x0 + 5, width - 1)):
            for y in range(baseline_y - bar_height, baseline_y):
                set_pixel(x, y, color)
    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3])
        for row in range(height)
    )
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, 2, 0, 0, 0))
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _verify_embedded_plots(
    html: str, retained: Any, equal_work_report: dict[str, Any]
) -> None:
    _require(isinstance(retained, list) and retained, "embedded plot manifest is empty")
    collector = _ImageCollector()
    try:
        collector.feed(html)
        collector.close()
    except (ValueError, FinalReleaseError) as error:
        raise FinalReleaseError(f"final HTML plot parsing failed: {error}") from error
    _require(len(collector.images) == len(retained), "embedded plot census differs")
    seen_ids: set[str] = set()
    for index, (image, expected) in enumerate(zip(collector.images, retained, strict=True)):
        record = _exact(
            expected,
            {
                "id", "mime_type", "width_px", "height_px", "size_bytes",
                "sha256", "data_uri_sha256", "source_data_sha256",
            },
            f"embedded plot manifest[{index}]",
        )
        plot_id = image["id"]
        _require(re.fullmatch(r"[A-Za-z0-9_.-]+", plot_id) is not None, "embedded plot ID is invalid")
        _require(plot_id not in seen_ids and record["id"] == plot_id, "embedded plot ID differs or repeats")
        seen_ids.add(plot_id)
        match = re.fullmatch(r"data:(image/png);base64,([A-Za-z0-9+/]*={0,2})", image["src"])
        _require(match is not None, "final HTML contains a noncanonical or external image source")
        try:
            png_payload = base64.b64decode(match.group(2), validate=True)
        except ValueError as error:
            raise FinalReleaseError("final HTML plot base64 is invalid") from error
        width, height = _decode_png(png_payload)
        expected_png = _expected_speedup_png(equal_work_report)
        _require(
            record["mime_type"] == match.group(1)
            and record["width_px"] == width
            and record["height_px"] == height
            and type(record["size_bytes"]) is int
            and record["size_bytes"] == len(png_payload)
            and record["sha256"] == hashlib.sha256(png_payload).hexdigest()
            and record["data_uri_sha256"] == hashlib.sha256(image["src"].encode("ascii")).hexdigest(),
            "embedded plot bytes or manifest hash differ",
        )
        _require(
            plot_id == "speedup"
            and png_payload == expected_png
            and record["source_data_sha256"]
            == _canonical_sha256(equal_work_report["performance"]["cases"]),
            "embedded plot is not deterministically derived from replayed performance",
        )
    _require(tuple(seen_ids) == EXPECTED_REPORT_PLOT_IDS, "embedded plot identity census differs")


def _actual_time_comparison_tables(
    equal_work_report: dict[str, Any],
) -> tuple[str, str]:
    headers = tuple(PERFORMANCE_LANES)
    rows: list[tuple[str, list[str]]] = []
    for case_name in sorted(equal_work_report["performance"]["cases"]):
        lanes = equal_work_report["performance"]["cases"][case_name].get(
            "lanes", {}
        )
        values: list[str] = []
        for lane in headers:
            lane_record = lanes.get(lane)
            if lane_record is None:
                values.append("—")
                continue
            seconds = lane_record.get(
                "measured_average_seconds", lane_record.get("mean_seconds")
            )
            _require(
                type(seconds) in {int, float}
                and math.isfinite(seconds)
                and seconds > 0,
                f"final report actual time differs for {case_name}/{lane}",
            )
            values.append(format(float(seconds), ".9g"))
        rows.append((case_name, values))
    title = "CPU vs v1.0.3 CUDA vs v1.1.0 CUDA measured mean seconds"
    html_table = (
        f"<h2>{title}</h2>"
        '<table id="actual-time-comparison"><thead><tr><th>Case</th>'
        + "".join(f"<th>{html_module.escape(lane)}</th>" for lane in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr><td>"
            + html_module.escape(case_name, quote=True)
            + "</td>"
            + "".join(
                f"<td>{html_module.escape(value)}</td>" for value in values
            )
            + "</tr>"
            for case_name, values in rows
        )
        + "</tbody></table>"
    )
    markdown_table = (
        f"## {title}\n\n| Case | "
        + " | ".join(headers)
        + " |\n|---|"
        + "---|" * len(headers)
        + "\n"
        + "\n".join(
            "| "
            + case_name.replace("|", r"\|")
            + " | "
            + " | ".join(values)
            + " |"
            for case_name, values in rows
        )
    )
    return html_table, markdown_table


def _verify_report_semantics(
    html: str, markdown: str, equal_work_report: dict[str, Any]
) -> None:
    """Bind every displayed timing/numerical table to the replayed M4 report."""
    performance = equal_work_report["performance"]
    numerical = equal_work_report["numerical"]
    report_id = equal_work_report.get("report_id") or _canonical_sha256(equal_work_report)
    measured_numerical_count = sum(
        isinstance(row, dict) and row.get("sample_kind") == "measured"
        for row in numerical["comparisons"]
    )
    _require(
        numerical["comparison_count"] == 141
        and len(numerical["comparisons"]) == 141
        and measured_numerical_count == 94,
        "final report numerical total/measured census differs",
    )
    expected = {
        "schema": "gpmeep-v110-final-report-data-v1",
        "m4_report_id": report_id,
        "actual_seconds": performance["raw_seconds"],
        "cases": performance["cases"],
        "numerical_comparison_count": numerical["comparison_count"],
        "measured_cycle_numerical_comparison_count": measured_numerical_count,
        "numerical_comparisons": numerical["comparisons"],
    }
    matches = re.findall(
        r'<script id="gpmeep-v110-report-data" type="application/json">([^<]+)</script>',
        html,
    )
    _require(len(matches) == 1, "final HTML lacks the exact deterministic report-data payload")
    try:
        retained = json.loads(
            matches[0], object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise FinalReleaseError("final HTML report-data payload is invalid") from error
    _require(retained == expected, "final HTML seconds/means/ratios/numerical data differ from M4 replay")
    expected_comparison_html, expected_comparison_markdown = (
        _actual_time_comparison_tables(equal_work_report)
    )
    _require(
        html.count(expected_comparison_html) == 1
        and markdown.count(expected_comparison_markdown) == 1,
        "final readable CPU/v1.0.3/v1.1.0 actual-time comparison differs",
    )
    row_sets = {
        "actual-seconds": [
            (str(index), value) for index, value in enumerate(performance["raw_seconds"])
        ],
        "means-ratios": [
            (key, performance["cases"][key]) for key in sorted(performance["cases"])
        ],
        "numerical-results": [
            (str(index), value) for index, value in enumerate(numerical["comparisons"])
        ],
    }
    markdown_headings = {
        "actual-seconds": "Actual seconds",
        "means-ratios": "Means and ratios",
        "numerical-results": "Numerical results",
    }
    for table_id, rows in row_sets.items():
        html_rows = "".join(
            "<tr><td>"
            + html_module.escape(key, quote=True)
            + "</td><td><code>"
            + html_module.escape(
                json.dumps(value, sort_keys=True, separators=(",", ":")), quote=True
            )
            + "</code></td></tr>"
            for key, value in rows
        )
        expected_table = (
            f'<table id="{table_id}"><thead><tr><th>Key</th><th>Canonical value</th>'
            f"</tr></thead><tbody>{html_rows}</tbody></table>"
        )
        _require(
            html.count(expected_table) == 1,
            f"final HTML visible table differs: {table_id}",
        )
        markdown_rows = "\n".join(
            "| " + key.replace("|", "\\|") + " | "
            + json.dumps(value, sort_keys=True, separators=(",", ":")).replace("|", r"\|")
            + " |"
            for key, value in rows
        )
        expected_markdown = (
            f"## {markdown_headings[table_id]}\n\n"
            "| Key | Canonical value |\n|---|---|\n"
            + markdown_rows
        )
        _require(
            markdown.count(expected_markdown) == 1,
            f"final Markdown visible table differs: {table_id}",
        )
    _require(
        f"M4 report ID: `{report_id}`" in markdown
        and all(len(rows) > 0 for rows in row_sets.values()),
        "final Markdown report semantic census differs",
    )


def _verify_qualification_payload(
    role: str,
    payload: dict[str, Any],
    *,
    source_commit: str,
    package_sha256: str,
    candidate_receipt_id: str,
    evidence_root: pathlib.Path,
    candidate_payload: dict[str, ArchiveMember],
) -> None:
    """Reject self-described PASS files; each role has one exact terminal shape."""
    common = {
        "schema",
        "state",
        "role",
        "source_commit",
        "package_sha256",
        "candidate_receipt_id",
        "producer_terminal",
        "report_id",
    }
    extras = {
        "installed_correctness": {"numerical_oracles_pass", "comparison_count", "m4_report_id", "development_package_receipt_id"},
        "installed_multi_gpu": {"gpu_uuids", "distinct_gpu_uuids", "cpu_fallback_count", "m4_report_id", "development_package_receipt_id"},
        "installed_aunp": {"polarization", "minimum_run", "dft_field_decay", "physics_fingerprint", "installed_prefix", "package_provenance_report_id"},
        "ubuntu_22_04_native": {"platform", "native_install", "environment_id", "prefix", "os_release_sha256", "installed_report_id", "namespace_id", "image_identity", "kernel", "libc"},
        "newer_ubuntu_native": {"platform", "native_install", "environment_id", "prefix", "os_release_sha256", "installed_report_id", "namespace_id", "image_identity", "kernel", "libc"},
        "glibc_floor": {"overall_maximum_required", "audited_elf_sha256", "installed_report_ids"},
        "final_adversarial_audit": {"findings", "open_critical_high_medium", "human_release_decision"},
    }
    _require(role in extras, f"qualification {role} has no producer-specific verifier")
    _exact(payload, common | extras[role], f"qualification {role} terminal")
    unsigned = dict(payload)
    report_id = unsigned.pop("report_id")
    _require(
        payload["schema"] == QUALIFICATION_ROLE_SCHEMAS[role]
        and payload["state"] == "PASS"
        and payload["role"] == role
        and payload["source_commit"] == source_commit
        and payload["package_sha256"] == package_sha256
        and payload["candidate_receipt_id"] == candidate_receipt_id
        and report_id == _canonical_sha256(unsigned),
        f"qualification {role} producer identity differs",
    )
    producer = _exact(
        payload["producer_terminal"],
        {"schema", "content_id", "sha256", "size_bytes", "bundle_relative_path"},
        f"qualification {role} producer terminal",
    )
    _require(
        isinstance(producer["schema"], str)
        and HEX64.fullmatch(producer["content_id"] or "") is not None
        and HEX64.fullmatch(producer["sha256"] or "") is not None
        and type(producer["size_bytes"]) is int
        and producer["size_bytes"] > 0,
        f"qualification {role} producer terminal identity differs",
    )
    relative = pathlib.PurePosixPath(producer["bundle_relative_path"])
    _require(
        relative.parts
        and not relative.is_absolute()
        and ".." not in relative.parts
        and str(relative) == producer["bundle_relative_path"],
        f"qualification {role} producer terminal path is unsafe",
    )
    producer_path = _regular_file(
        evidence_root.joinpath(*relative.parts),
        f"qualification {role} producer terminal",
    )
    _require(producer_path.is_relative_to(evidence_root), f"qualification {role} producer terminal escaped evidence root")
    _require(
        producer_path.stat().st_size == producer["size_bytes"]
        and _sha256(producer_path) == producer["sha256"],
        f"qualification {role} producer terminal bytes differ",
    )
    producer_payload = _load(producer_path)
    producer_schema = producer_payload.get("schema")
    if producer_schema is None and type(producer_payload.get("schema_version")) is int:
        producer_schema = f"schema_version:{producer_payload['schema_version']}"
    _require(producer_schema == producer["schema"], f"qualification {role} producer schema differs")
    producer_content_id = next(
        (
            producer_payload.get(key)
            for key in ("terminal_id", "report_id", "receipt_id", "content_id")
            if producer_payload.get(key) is not None
        ),
        _canonical_sha256(producer_payload),
    )
    _require(producer_content_id == producer["content_id"], f"qualification {role} producer content ID differs")

    producer_report: dict[str, Any] | None = None
    if role in {"installed_correctness", "installed_multi_gpu"}:
        _require(
            producer["schema"] == M4_COMPLETE_SCHEMA
            and producer_path.name == "M4_COMPLETE",
            f"qualification {role} is not backed by the official M4 terminal",
        )
        producer_report = _replay_equal_work(producer_path.parent)
        expected_report_id = producer_report.get("report_id") or _canonical_sha256(producer_report)
        _require(payload["m4_report_id"] == expected_report_id, f"qualification {role} M4 report identity differs")
    elif role == "installed_aunp":
        aunp_producer = _load_module(
            "gpmeep_v110_final_aunp_replay",
            pathlib.Path(__file__).resolve().parent
            / "user-workloads/run_installed_aunp_performance_matrix.py",
        )
        _require(
            producer["schema"] == aunp_producer.COMPLETE_SCHEMA
            and producer_path.name == "COMPLETE",
            "installed AuNP producer terminal differs",
        )
        try:
            producer_report = aunp_producer.validate(producer_path.parent)
        except Exception as error:
            raise FinalReleaseError(f"installed AuNP producer replay failed: {error}") from error
    elif role in {"ubuntu_22_04_native", "newer_ubuntu_native"}:
        native_producer = _load_module(
            "gpmeep_v110_native_witness_replay",
            pathlib.Path(__file__).resolve().parent / "v110-native-witness.py",
        )
        _require(
            producer["schema"] == native_producer.SCHEMA,
            f"qualification {role} native witness schema differs",
        )
        try:
            producer_report = native_producer.verify(producer_path, require_live=False)
        except Exception as error:
            raise FinalReleaseError(
                f"qualification {role} native witness replay failed: {error}"
            ) from error
        _require(
            producer_report == producer_payload,
            f"qualification {role} native witness bytes differ",
        )
    if role == "installed_correctness":
        numerical = producer_report.get("numerical", {}) if producer_report else {}
        development_receipt_id = (
            producer_report.get("contract", {})
            .get("runtime_inputs", {})
            .get("v110_receipt_id")
            if producer_report
            else None
        )
        _require(
            payload["numerical_oracles_pass"] is True
            and payload["comparison_count"] == 141
            and numerical.get("outcome") == "PASS"
            and numerical.get("comparison_count") == 141,
            "installed correctness census differs",
        )
        _require(
            HEX64.fullmatch(development_receipt_id or "") is not None
            and payload["development_package_receipt_id"]
            == development_receipt_id,
            "installed correctness development package receipt differs",
        )
    elif role == "installed_multi_gpu":
        def values(value: Any, fragment: str) -> list[Any]:
            found: list[Any] = []
            if isinstance(value, dict):
                for key, item in value.items():
                    if fragment in key.lower():
                        found.append(item)
                    found.extend(values(item, fragment))
            elif isinstance(value, list):
                for item in value:
                    found.extend(values(item, fragment))
            return found

        replayed_uuids = sorted({
            item
            for value in values(producer_report, "uuid")
            for item in (value if isinstance(value, list) else [value])
            if isinstance(item, str) and item
        })
        replayed_fallbacks = sum(
            value for value in values(producer_report, "cpu_fallback_count") if type(value) is int
        )
        development_receipt_id = (
            producer_report.get("contract", {})
            .get("runtime_inputs", {})
            .get("v110_receipt_id")
            if producer_report
            else None
        )
        uuids = payload["gpu_uuids"]
        _require(
            isinstance(uuids, list)
            and len(uuids) >= 2
            and uuids == sorted(set(uuids))
            and payload["distinct_gpu_uuids"] is True
            and payload["cpu_fallback_count"] == 0
            and uuids == replayed_uuids
            and replayed_fallbacks == 0
            and HEX64.fullmatch(development_receipt_id or "") is not None
            and payload["development_package_receipt_id"]
            == development_receipt_id,
            "installed multi-GPU producer result differs",
        )
    elif role == "installed_aunp":
        profile = producer_report.get("qualification_profile", {}) if producer_report else {}
        provenance_path = _regular_file(
            pathlib.Path(producer_report.get("package_provenance", {}).get("path", "")),
            "installed AuNP package provenance",
        )
        provenance = _load(provenance_path)
        provenance_report_id = provenance.get("report_id")
        installed_prefix = provenance.get("prefix")
        _require(
            payload["polarization"] == "TM"
            and payload["minimum_run"] == 20
            and payload["dft_field_decay"] == 5e-8
            and profile.get("mode") == "TM-only"
            and profile.get("dft_minimum_run_time") == 20.0
            and profile.get("dft_decay_tolerance") == 5e-8
            and HEX64.fullmatch(payload["physics_fingerprint"] or "") is not None
            and payload["physics_fingerprint"]
            == _canonical_sha256(producer_report)
            and producer_report.get("source_commit") == source_commit
            and producer_report.get("package_sha256") == package_sha256
            and HEX64.fullmatch(provenance_report_id or "") is not None
            and payload["package_provenance_report_id"] == provenance_report_id
            and payload["installed_prefix"] == installed_prefix,
            "installed AuNP producer result differs",
        )
    elif role in {"ubuntu_22_04_native", "newer_ubuntu_native"}:
        native = producer_report or {}
        _require(
            payload["native_install"] is True
            and payload["platform"]
            == ("ubuntu-22.04" if role == "ubuntu_22_04_native" else "ubuntu-24.04")
            and HEX64.fullmatch(payload["namespace_id"] or "") is not None
            and (
                payload["image_identity"] == "native-host"
                or re.fullmatch(r"sha256:[0-9a-f]{64}", payload["image_identity"] or "")
                is not None
            )
            and isinstance(payload["kernel"], dict)
            and isinstance(payload["libc"], str),
            f"qualification {role} live native identity differs",
        )
        expected_native = {
            "platform": native.get("platform"),
            "environment_id": native.get("environment_id"),
            "prefix": native.get("prefix"),
            "os_release_sha256": native.get("os_release", {}).get("sha256"),
            "installed_report_id": native.get("installed_report_id"),
            "namespace_id": native.get("namespace_id"),
            "image_identity": native.get("image_identity"),
            "kernel": native.get("kernel"),
            "libc": native.get("libc"),
        }
        _require(
            all(payload[key] == value for key, value in expected_native.items()),
            f"qualification {role} differs from replayed native witness",
        )
    elif role == "glibc_floor":
        glibc_report = _replay_candidate_glibc(candidate_payload, producer_payload)
        _require(
            payload["overall_maximum_required"]
            == glibc_report["overall_maximum_required"]
            and _glibc_version_key(payload["overall_maximum_required"]) <= (2, 17)
            and isinstance(payload["audited_elf_sha256"], list)
            and payload["audited_elf_sha256"] == sorted(set(payload["audited_elf_sha256"]))
            and payload["audited_elf_sha256"]
            and isinstance(payload["installed_report_ids"], list)
            and len(payload["installed_report_ids"]) == 2
            and payload["installed_report_ids"] == sorted(set(payload["installed_report_ids"]))
            and all(HEX64.fullmatch(item or "") is not None for item in payload["installed_report_ids"]),
            "GLIBC producer inventory differs",
        )
        _require(
            payload["audited_elf_sha256"]
            == sorted(record["sha256"] for record in glibc_report["elf_files"]),
            "GLIBC wrapper differs from candidate ELF bytes",
        )
    elif role == "final_adversarial_audit":
        _exact(
            producer_payload,
            {
                "schema", "state", "source_commit", "package_sha256",
                "candidate_receipt_id", "auditor", "scope", "findings",
                "human_release_decision", "terminal_id",
            },
            "independent final audit producer",
        )
        producer_unsigned = dict(producer_payload)
        producer_terminal_id = producer_unsigned.pop("terminal_id")
        scope = _exact(
            producer_payload["scope"],
            {"independent", "exact_source_commit", "exact_package_sha256", "finding_ids"},
            "independent final audit scope",
        )
        _require(
            producer["schema"] == FINAL_AUDIT_PRODUCER_SCHEMA
            and producer_payload["schema"] == FINAL_AUDIT_PRODUCER_SCHEMA
            and producer_payload["state"] == "PASS"
            and producer_payload["source_commit"] == source_commit
            and producer_payload["package_sha256"] == package_sha256
            and producer_payload["candidate_receipt_id"] == candidate_receipt_id
            and isinstance(producer_payload["auditor"], str)
            and producer_payload["auditor"].strip()
            and scope
            == {
                "independent": True,
                "exact_source_commit": source_commit,
                "exact_package_sha256": package_sha256,
                "finding_ids": list(FINAL_AUDIT_FINDINGS),
            }
            and producer_terminal_id == _canonical_sha256(producer_unsigned),
            "independent final audit producer identity differs",
        )
        findings = payload["findings"]
        _require(
            isinstance(findings, list)
            and [item.get("id") for item in findings] == list(FINAL_AUDIT_FINDINGS)
            and all(
                isinstance(item, dict)
                and set(item) == {"id", "severity", "state", "evidence_sha256"}
                and item["severity"] in {"CRITICAL", "HIGH", "MEDIUM"}
                and item["state"] == "CLOSED"
                and HEX64.fullmatch(item["evidence_sha256"] or "") is not None
                for item in findings
            ),
            "final adversarial audit finding inventory differs",
        )
        _require(
            findings == producer_payload["findings"]
            and payload["open_critical_high_medium"]
            == sum(
                item["state"] != "CLOSED"
                and item["severity"] in {"CRITICAL", "HIGH", "MEDIUM"}
                for item in producer_payload["findings"]
            )
            == 0,
            "final adversarial audit open finding census differs",
        )
        decision = _exact(
            payload["human_release_decision"],
            {"decision", "approver", "finding_ids", "package_sha256", "decision_id"},
            "human release decision",
        )
        unsigned_decision = dict(decision)
        decision_id = unsigned_decision.pop("decision_id")
        _require(
            decision["decision"] == "APPROVE_RELEASE"
            and isinstance(decision["approver"], str)
            and decision["approver"].strip()
            and decision["finding_ids"] == list(FINAL_AUDIT_FINDINGS)
            and decision["package_sha256"] == package_sha256
            and decision_id == _canonical_sha256(unsigned_decision),
            "explicit human release decision differs",
        )
        _require(
            decision == producer_payload["human_release_decision"],
            "human release decision differs from independent producer",
        )
    else:
        raise FinalReleaseError(f"qualification {role} verifier dispatch differs")


def _replay_installed_self_check(
    path: pathlib.Path,
    *,
    prefix: pathlib.Path,
    source_commit: str,
    installed_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Replay one installed gpu2 self-check and bind every runtime byte live."""
    self_check = _load_module(
        "gpmeep_v110_final_installed_self_check_replay",
        pathlib.Path(__file__).resolve().parent / "gpmeep-self-check.py",
    )
    payload = _load(path)
    _exact(
        payload,
        {
            "schema_version", "state", "mode", "release", "lanes",
            "comparisons", "wall_seconds", "valid_for_speed_gate",
            "speed_gate_reason", "report_id",
        },
        "installed self-check terminal",
    )
    unsigned = dict(payload)
    report_id = unsigned.pop("report_id")
    _require(
        payload["schema_version"] == 2
        and payload["state"] == "PASS"
        and payload["mode"] == "gpu2"
        and type(payload["wall_seconds"]) in {int, float}
        and math.isfinite(payload["wall_seconds"])
        and payload["wall_seconds"] > 0
        and payload["valid_for_speed_gate"] is False
        and payload["speed_gate_reason"]
        == "small installed-runtime correctness smoke"
        and report_id == _canonical_sha256(unsigned),
        "installed self-check terminal identity differs",
    )
    release = _exact(
        payload["release"],
        set(self_check.RELEASE_MANIFEST_FIELDS) | {"source_commit", "file"},
        "installed self-check release manifest",
    )
    release_without_file = dict(release)
    release_file = _exact(
        release_without_file.pop("file"),
        {"path", "size_bytes", "sha256"},
        "installed self-check release-manifest file",
    )
    try:
        self_check.validate_release_manifest(
            release_without_file, expected_source_commit=source_commit
        )
    except Exception as error:
        raise FinalReleaseError(
            f"installed self-check release manifest replay failed: {error}"
        ) from error
    expected_release_path = prefix / INSTALLED_RELEASE_MANIFEST_PATH
    live_release_path = _regular_file(
        expected_release_path, "installed self-check live release manifest"
    )
    _require(
        pathlib.Path(release_file["path"]) == live_release_path
        and release_file["size_bytes"] == live_release_path.stat().st_size
        and release_file["sha256"] == _sha256(live_release_path),
        "installed self-check release-manifest bytes differ",
    )
    lanes = _exact(payload["lanes"], {"cpu", "gpu1", "gpu2"}, "installed self-check lanes")
    try:
        validated = {
            lane: self_check._validate_lane(lanes[lane], lane)
            for lane in ("cpu", "gpu1", "gpu2")
        }
        expected_comparisons = {
            "cpu_vs_gpu1": self_check._compare_observables(
                validated["cpu"][0], validated["gpu1"][0]
            ),
            "cpu_vs_gpu2": self_check._compare_observables(
                validated["cpu"][0], validated["gpu2"][0]
            ),
            "gpu1_vs_gpu2": self_check._compare_observables(
                validated["gpu1"][0], validated["gpu2"][0]
            ),
        }
    except Exception as error:
        raise FinalReleaseError(
            f"installed self-check lane replay failed: {error}"
        ) from error
    _require(
        payload["comparisons"] == expected_comparisons,
        "installed self-check comparisons were not re-derived",
    )
    provenance_runtime = _exact(
        installed_runtime,
        {
            "python", "module", "extension", "libmeep", "meep_version",
            "cuda_compiled", "mpi_enabled", "single_precision",
        },
        "installed provenance runtime",
    )
    for lane in ("cpu", "gpu1", "gpu2"):
        for record in validated[lane]:
            for key in ("python", "extension", "libmeep"):
                retained = _exact(
                    record[key], {"path", "size_bytes", "sha256"},
                    f"installed self-check {lane} {key}",
                )
                runtime_path = _regular_file(
                    pathlib.Path(retained["path"]),
                    f"installed self-check live {lane} {key}",
                )
                _require(
                    runtime_path.is_relative_to(prefix)
                    and retained["size_bytes"] == runtime_path.stat().st_size
                    and retained["sha256"] == _sha256(runtime_path),
                    f"installed self-check {lane} {key} escaped or changed",
                )
                _require(
                    retained == provenance_runtime[key],
                    f"installed self-check {lane} {key} differs from installed provenance runtime",
                )
    return payload


def _verify_qualification(
    value: Any,
    *,
    evidence_root: pathlib.Path,
    source_commit: str,
    candidate_receipt_id: str,
    package_name: str,
    package_size: int,
    package_sha256: str,
    package_path: pathlib.Path,
    archive_members: dict[str, bytes],
    candidate_payload: dict[str, ArchiveMember],
    final_audit_authority_path: pathlib.Path,
    expected_final_audit_authority_sha256: str,
) -> dict[str, Any]:
    manifest = _exact(
        value,
        {
            "schema", "state", "source_commit", "candidate_receipt_id", "package",
            "installed_provenance", "qualification", "performance", "report_plots",
        },
        "post-package qualification manifest",
    )
    _require(manifest["schema"] == QUALIFICATION_SCHEMA, "qualification manifest schema differs")
    _require(manifest["state"] == "PASS", "post-package qualification did not pass")
    _require(manifest["source_commit"] == source_commit, "qualification source commit differs")
    _require(manifest["candidate_receipt_id"] == candidate_receipt_id, "qualification candidate receipt differs")
    package = _exact(
        manifest["package"],
        {"file_name", "size_bytes", "sha256", "format", "name", "version", "build", "subdir", "embedded_candidate_receipt_id"},
        "qualified package",
    )
    _require(
        package
        == {
            "file_name": package_name,
            "size_bytes": package_size,
            "sha256": package_sha256,
            "format": "conda-v2",
            "name": "gpmeep",
            "version": "1.1.0",
            "build": "cuda124_mpi_openmpi_py311_0",
            "subdir": "linux-64",
            "embedded_candidate_receipt_id": candidate_receipt_id,
        },
        "qualified package identity differs",
    )
    installed = manifest["installed_provenance"]
    _require(isinstance(installed, list) and len(installed) == 2, "two native installed-provenance reports are required")
    platforms: set[str] = set()
    environment_ids: set[str] = set()
    prefixes: set[pathlib.Path] = set()
    report_paths: set[pathlib.Path] = set()
    report_hashes: set[str] = set()
    native_witness_paths: set[pathlib.Path] = set()
    native_witness_hashes: set[str] = set()
    self_check_paths: set[pathlib.Path] = set()
    self_check_hashes: set[str] = set()
    self_check_report_ids: set[str] = set()
    namespace_ids: set[str] = set()
    installed_report_ids: set[str] = set()
    environment_bindings: dict[str, dict[str, Any]] = {}
    native_witness_tool = _load_module(
        "gpmeep_v110_native_witness_manifest_replay",
        pathlib.Path(__file__).resolve().parent / "v110-native-witness.py",
    )
    for index, item in enumerate(installed):
        entry = _exact(
            item,
            {"environment_id", "platform", "prefix", "native_witness", "self_check", "package_sha256", "source_commit", "candidate_receipt_id", "report"},
            f"installed_provenance[{index}]",
        )
        prefix = _regular_directory(pathlib.Path(entry["prefix"]), f"installed_provenance[{index}].prefix")
        _require(prefix not in prefixes, "installed provenance prefixes are duplicated")
        prefixes.add(prefix)
        expected_platform = "ubuntu-22.04" if index == 0 else "ubuntu-24.04"
        _require(entry["platform"] == expected_platform, "installed platform is invalid")
        platforms.add(entry["platform"])
        _require(entry["package_sha256"] == package_sha256, "installed provenance package hash differs")
        _require(entry["source_commit"] == source_commit, "installed provenance source differs")
        _require(entry["candidate_receipt_id"] == candidate_receipt_id, "installed provenance candidate receipt differs")
        report_path, payload = _artifact_ref(entry["report"], f"installed_provenance_{index}", evidence_root)
        _require(report_path not in report_paths, "installed provenance reports are duplicated")
        _require(entry["report"]["sha256"] not in report_hashes, "installed provenance report bytes are reused")
        report_paths.add(report_path)
        report_hashes.add(entry["report"]["sha256"])
        _require(payload.get("state") == "PASS", "installed provenance report did not pass")
        _require(payload.get("prefix") == str(prefix), "installed provenance report prefix differs")
        unsigned = dict(payload)
        report_id = unsigned.pop("report_id", None)
        _require(
            HEX64.fullmatch(report_id or "") is not None
            and _canonical_sha256(unsigned) == report_id,
            "installed provenance report identity differs",
        )
        installed_report_ids.add(report_id)
        _require(payload.get("package_sha256") == package_sha256, "installed provenance report package differs")
        _require(payload.get("manifest", {}).get("source_commit") == source_commit, "installed provenance report source differs")
        _require(
            payload.get("v110_candidate_gate", {}).get("receipt_id")
            == candidate_receipt_id,
            "installed provenance report candidate differs",
        )
        self_check_path, self_check_payload = _artifact_ref(
            entry["self_check"], f"self_check_{index}", evidence_root
        )
        _require(self_check_path not in self_check_paths, "installed self-check paths are duplicated")
        _require(entry["self_check"]["sha256"] not in self_check_hashes, "installed self-check bytes are reused")
        self_check_paths.add(self_check_path)
        self_check_hashes.add(entry["self_check"]["sha256"])
        replayed_self_check = _replay_installed_self_check(
            self_check_path,
            prefix=prefix,
            source_commit=source_commit,
            installed_runtime=payload.get("runtime"),
        )
        _require(
            replayed_self_check == self_check_payload,
            "installed self-check bytes differ from replay",
        )
        self_check_report_id = self_check_payload.get("report_id")
        _require(
            HEX64.fullmatch(self_check_report_id or "") is not None
            and self_check_report_id not in self_check_report_ids,
            "installed self-check report IDs are invalid or duplicated",
        )
        self_check_report_ids.add(self_check_report_id)
        witness_path, witness_payload = _artifact_ref(
            entry["native_witness"], f"native_witness_{index}", evidence_root
        )
        _require(witness_path not in native_witness_paths, "native witness paths are duplicated")
        _require(entry["native_witness"]["sha256"] not in native_witness_hashes, "native witness bytes are reused")
        native_witness_paths.add(witness_path)
        native_witness_hashes.add(entry["native_witness"]["sha256"])
        try:
            replayed_witness = native_witness_tool.verify(
                witness_path, require_live=False
            )
        except Exception as error:
            raise FinalReleaseError(
                f"installed native witness replay failed: {error}"
            ) from error
        _require(replayed_witness == witness_payload, "installed native witness bytes differ")
        _require(
            witness_payload["platform"] == expected_platform
            and witness_payload["prefix"] == str(prefix)
            and witness_payload["source_commit"] == source_commit
            and witness_payload["package_sha256"] == package_sha256
            and witness_payload["candidate_receipt_id"] == candidate_receipt_id
            and witness_payload["installed_report_id"] == report_id
            and witness_payload["installed_report_sha256"] == entry["report"]["sha256"],
            "installed native witness binding differs",
        )
        _require(entry["environment_id"] == witness_payload["environment_id"], "installed environment ID was not derived")
        _require(entry["environment_id"] not in environment_ids, "installed environment IDs are duplicated")
        environment_ids.add(entry["environment_id"])
        _require(witness_payload["namespace_id"] not in namespace_ids, "native namespaces are duplicated")
        namespace_ids.add(witness_payload["namespace_id"])
        environment_bindings[entry["platform"]] = {
            "environment_id": entry["environment_id"],
            "prefix": str(prefix),
            "os_release_sha256": witness_payload["os_release"]["sha256"],
            "installed_report_id": report_id,
            "namespace_id": witness_payload["namespace_id"],
            "image_identity": witness_payload["image_identity"],
            "kernel": witness_payload["kernel"],
            "libc": witness_payload["libc"],
        }
        _verify_installed_prefix(
            prefix=prefix,
            retained_report=payload,
            archive_members=archive_members,
            source_commit=source_commit,
            package_sha256=package_sha256,
            candidate_receipt_id=candidate_receipt_id,
        )
    _require(platforms == {"ubuntu-22.04", "ubuntu-24.04"}, "native portability environment set differs")

    qualification = _exact(manifest["qualification"], set(QUALIFICATION_ROLES), "qualification evidence")
    payloads: dict[str, dict[str, Any]] = {}
    for role in QUALIFICATION_ROLES:
        _path, payload = _artifact_ref(qualification[role], role, evidence_root)
        if role not in TEXT_ARTIFACT_SCHEMAS:
            payloads[role] = payload
            _require(payload.get("state") == "PASS", f"qualification {role} did not pass")
            if role != "closed_inversion_registry":
                _verify_qualification_payload(
                    role,
                    payload,
                    source_commit=source_commit,
                    package_sha256=package_sha256,
                    candidate_receipt_id=candidate_receipt_id,
                    evidence_root=evidence_root,
                    candidate_payload=candidate_payload,
                )
    html_payload = _artifact_ref(
        qualification["final_report_html"], "final_report_html", evidence_root
    )[1]
    markdown_payload = _artifact_ref(
        qualification["final_report_markdown"], "final_report_markdown", evidence_root
    )[1]
    audit = payloads["final_adversarial_audit"]
    audit_path = _regular_file(final_audit_authority_path, "independent final audit authority")
    audit_reference = audit["producer_terminal"]
    audit_relative = pathlib.PurePosixPath(audit_reference["bundle_relative_path"])
    _require(
        HEX64.fullmatch(expected_final_audit_authority_sha256 or "") is not None
        and _sha256(audit_path) == expected_final_audit_authority_sha256
        and audit_path == evidence_root.joinpath(*audit_relative.parts)
        and audit_reference["sha256"] == expected_final_audit_authority_sha256,
        "final adversarial audit is not independently SHA-256 pinned",
    )
    _require(audit.get("open_critical_high_medium") == 0, "final adversarial audit has an open material finding")
    closed_terminal_path, closed_terminal = _artifact_ref(
        qualification["closed_inversion_registry"],
        "closed_inversion_registry",
        evidence_root,
    )
    replayed_closure = _replay_closed_inversion_registry(
        closed_terminal_path, evidence_root
    )
    _require(
        replayed_closure == closed_terminal,
        "closed inversion registry artifact differs from producer replay",
    )
    correctness = payloads["installed_correctness"]
    _require(
        correctness.get("numerical_oracles_pass") is True,
        "installed correctness numerical oracles did not pass",
    )
    multi_gpu = payloads["installed_multi_gpu"]
    _require(multi_gpu.get("distinct_gpu_uuids") is True and multi_gpu.get("cpu_fallback_count") == 0, "installed multi-GPU proof differs")
    aunp = payloads["installed_aunp"]
    _require(
        aunp.get("polarization") == "TM"
        and aunp.get("minimum_run") == 20
        and aunp.get("dft_field_decay") == 5e-8,
        "installed AuNP qualification contract differs",
    )
    development_receipt_ids = {
        payloads[role].get("development_package_receipt_id")
        for role in (
            "installed_correctness",
            "installed_multi_gpu",
        )
    }
    _require(
        len(development_receipt_ids) == 1
        and HEX64.fullmatch(next(iter(development_receipt_ids)) or "") is not None,
        "installed qualification terminals do not bind one development package receipt",
    )
    _require(
        aunp.get("installed_prefix")
        in {binding["prefix"] for binding in environment_bindings.values()}
        and aunp.get("package_provenance_report_id") in installed_report_ids,
        "installed AuNP is not bound to a native candidate package provenance",
    )
    for role, platform in (
        ("ubuntu_22_04_native", "ubuntu-22.04"),
        ("newer_ubuntu_native", "ubuntu-24.04"),
    ):
        proof = payloads[role]
        binding = environment_bindings[platform]
        _require(
            proof.get("platform") == platform
            and proof.get("native_install") is True,
            f"qualification {role} is not a native platform proof",
        )
        for key, expected in binding.items():
            _require(proof.get(key) == expected, f"qualification {role} {key} differs")
    glibc = payloads["glibc_floor"]
    maximum_glibc = glibc.get("overall_maximum_required")
    _require(
        isinstance(maximum_glibc, str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", maximum_glibc) is not None
        and tuple(int(part) for part in maximum_glibc.split(".")) <= (2, 35),
        "GLIBC floor exceeds 2.35 or is invalid",
    )
    _require(
        glibc.get("installed_report_ids") == sorted(installed_report_ids),
        "GLIBC qualification installed report bindings differ",
    )

    performance = _exact(
        manifest["performance"],
        {
            "state", "report", "lanes", "repetitions", "equal_work",
            "actual_seconds_retained", "numerical_oracles_pass", "no_cpu_fallback",
            "no_material_v110_regression", "structural_optimization_benefit",
            "frozen_v103", "v110_package_sha256",
        },
        "three-way performance",
    )
    _require(performance["state"] == "PASS", "three-way performance did not pass")
    _require(performance["lanes"] == list(PERFORMANCE_LANES), "three-way performance lane inventory differs")
    _require(performance["repetitions"] == 2, "three-way performance must retain two repetitions")
    for key in ("equal_work", "actual_seconds_retained", "numerical_oracles_pass", "no_cpu_fallback"):
        _require(performance[key] is True, f"three-way performance {key} did not pass")
    frozen = _exact(performance["frozen_v103"], {"version", "source_commit", "package_sha256"}, "frozen v1.0.3")
    _require(
        frozen
        == {
            "version": "1.0.3",
            "source_commit": FROZEN_V103_COMMIT,
            "package_sha256": FROZEN_V103_PACKAGE_SHA256,
        },
        "frozen v1.0.3 identity differs",
    )
    performance_path, performance_terminal = _artifact_ref(
        performance["report"], "three_way_performance", evidence_root
    )
    _require(
        performance_path.name == "M4_COMPLETE"
        and performance_terminal.get("schema") == M4_COMPLETE_SCHEMA
        and performance_terminal.get("outcome") == "PASS",
        "three-way performance reference is not an official M4 terminal",
    )
    equal_work_report = _replay_equal_work(performance_path.parent)
    _verify_embedded_plots(html_payload, manifest["report_plots"], equal_work_report)
    _verify_report_semantics(html_payload, markdown_payload, equal_work_report)
    _replay_equal_work_registry(performance_path.parent)
    material_ok, structural_benefit, equivalence = _validate_equal_work_report(
        equal_work_report,
        source_commit=source_commit,
        candidate_package=package_path,
        candidate_payload=candidate_payload,
    )
    m4_report_id = equal_work_report.get("report_id") or _canonical_sha256(
        equal_work_report
    )
    _require(
        correctness.get("m4_report_id") == m4_report_id
        and multi_gpu.get("m4_report_id") == m4_report_id,
        "installed qualification M4 report differs from performance authority",
    )
    _require(
        next(iter(development_receipt_ids))
        == equivalence["measured_receipt_id"],
        "installed qualification development receipt differs from measured package",
    )
    _require(
        performance["v110_package_sha256"]
        == equivalence["measured_package_sha256"],
        "performance report measured package hash differs",
    )
    _require(
        performance["no_material_v110_regression"] is material_ok and material_ok,
        "official equal-work replay contains a material v1.1.0 regression",
    )
    _require(
        performance["structural_optimization_benefit"] is structural_benefit
        and structural_benefit,
        "official equal-work replay does not demonstrate a structural benefit",
    )
    return manifest


def create(
    *,
    package_path: pathlib.Path,
    candidate_receipt_path: pathlib.Path,
    candidate_tool_path: pathlib.Path,
    contract_path: pathlib.Path,
    inversion_schema_path: pathlib.Path,
    inventory_path: pathlib.Path,
    qualification_manifest_path: pathlib.Path,
    evidence_root: pathlib.Path,
    candidate_contract_gate_path: pathlib.Path,
    candidate_registry_gate_path: pathlib.Path,
    candidate_attestation_path: pathlib.Path,
    candidate_manifest_path: pathlib.Path,
    candidate_evidence_root: pathlib.Path,
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
    final_audit_authority_path: pathlib.Path,
    expected_final_audit_authority_sha256: str,
) -> dict[str, Any]:
    _require_isolated_release_runtime()
    package_path = _regular_file(package_path, "candidate package")
    _require(PACKAGE_NAME.fullmatch(package_path.name) is not None, "candidate package file name is invalid")
    package_file_identity = _identity(package_path)
    package_sha256 = _sha256(package_path)
    package_size = package_path.stat().st_size
    candidate_receipt_path = _regular_file(candidate_receipt_path, "candidate receipt")
    candidate_identity = _identity(candidate_receipt_path)
    candidate_record = _record(candidate_receipt_path)
    candidate_tool_path = _regular_file(candidate_tool_path, "candidate receipt verifier")
    candidate_tool = _load_candidate_tool(candidate_tool_path)
    try:
        candidate = candidate_tool.verify(
            receipt_path=candidate_receipt_path,
            contract_path=contract_path,
            schema_path=inversion_schema_path,
            inventory_path=inventory_path,
            require_candidate=True,
            contract_gate_path=candidate_contract_gate_path,
            registry_gate_path=candidate_registry_gate_path,
            attestation_gate_path=candidate_attestation_path,
            candidate_manifest_path=candidate_manifest_path,
            evidence_root=candidate_evidence_root,
            source_policy_path=source_policy_path,
            expected_source_policy_sha256=expected_source_policy_sha256,
            source_verifier_path=source_verifier_path,
            expected_source_verifier_sha256=expected_source_verifier_sha256,
        )
    except Exception as error:
        raise FinalReleaseError(f"candidate receipt replay failed: {error}") from error
    source_commit = candidate["source_commit"]
    candidate_receipt_id = candidate["receipt_id"]
    package_metadata, package_members, candidate_payload = _inspect_conda_v2(
        package_path, candidate_receipt_path, candidate_receipt_id
    )
    _require(package_metadata["sha256"] == package_sha256, "inspected package digest differs")
    evidence_root = _regular_directory(evidence_root, "qualification evidence root")
    qualification_manifest_path = _regular_file(qualification_manifest_path, "qualification manifest")
    qualification_identity = _identity(qualification_manifest_path)
    qualification_record = _record(qualification_manifest_path)
    _require(qualification_manifest_path.is_relative_to(evidence_root), "qualification manifest is outside evidence root")
    qualification = _load(qualification_manifest_path)
    _verify_qualification(
        qualification,
        evidence_root=evidence_root,
        source_commit=source_commit,
        candidate_receipt_id=candidate_receipt_id,
        package_name=package_path.name,
        package_size=package_size,
        package_sha256=package_sha256,
        package_path=package_path,
        archive_members=package_members,
        candidate_payload=candidate_payload,
        final_audit_authority_path=final_audit_authority_path,
        expected_final_audit_authority_sha256=expected_final_audit_authority_sha256,
    )
    _require(qualification["package"] == package_metadata, "qualification package metadata was not archive-derived")
    _require(_identity(package_path) == package_file_identity and _sha256(package_path) == package_sha256, "candidate package changed during qualification replay")
    _require(
        _identity(candidate_receipt_path) == candidate_identity
        and _record(candidate_receipt_path) == candidate_record,
        "candidate receipt changed during qualification replay",
    )
    _require(
        _identity(qualification_manifest_path) == qualification_identity
        and _record(qualification_manifest_path) == qualification_record,
        "qualification manifest changed during replay",
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "release_train": "1.1.0",
        "state": "RELEASE_AUTHORIZED",
        "candidate_authority": {
            "receipt_id": candidate_receipt_id,
            "source_commit": source_commit,
            "file": candidate_record,
        },
        "package": qualification["package"],
        "qualification_manifest": {
            **qualification_record,
            "content_sha256": _canonical_sha256(qualification),
        },
        "final_audit_authority": {
            "file": _record(_regular_file(final_audit_authority_path, "independent final audit authority")),
            "sha256": expected_final_audit_authority_sha256,
        },
        "gate_summary": {
            "candidate_replayed": True,
            "exact_bytes_installed": True,
            "all_release_gates_pass": True,
            "open_critical_high_medium": 0,
        },
        "tag_intent": {"name": "v1.1.0", "target_commit": source_commit},
        "authorization": {"release_publication": True, "git_tag": True, "package_mutation": False},
    }
    receipt["receipt_id"] = _canonical_sha256(receipt)
    return receipt


def verify(
    *,
    receipt_path: pathlib.Path,
    package_path: pathlib.Path,
    candidate_receipt_path: pathlib.Path,
    candidate_tool_path: pathlib.Path,
    contract_path: pathlib.Path,
    inversion_schema_path: pathlib.Path,
    inventory_path: pathlib.Path,
    qualification_manifest_path: pathlib.Path,
    evidence_root: pathlib.Path,
    candidate_contract_gate_path: pathlib.Path,
    candidate_registry_gate_path: pathlib.Path,
    candidate_attestation_path: pathlib.Path,
    candidate_manifest_path: pathlib.Path,
    candidate_evidence_root: pathlib.Path,
    source_policy_path: pathlib.Path,
    expected_source_policy_sha256: str,
    source_verifier_path: pathlib.Path,
    expected_source_verifier_sha256: str,
    final_audit_authority_path: pathlib.Path,
    expected_final_audit_authority_sha256: str,
) -> dict[str, Any]:
    receipt = _load(_regular_file(receipt_path, "final release receipt"))
    expected = {
        "schema", "release_train", "state", "candidate_authority", "package",
        "qualification_manifest", "final_audit_authority", "gate_summary", "tag_intent", "authorization", "receipt_id",
    }
    _require(set(receipt) == expected, "final receipt schema differs")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id", None)
    _require(HEX64.fullmatch(receipt_id or "") is not None and _canonical_sha256(unsigned) == receipt_id, "final receipt identity differs")
    recreated = create(
        package_path=package_path,
        candidate_receipt_path=candidate_receipt_path,
        candidate_tool_path=candidate_tool_path,
        contract_path=contract_path,
        inversion_schema_path=inversion_schema_path,
        inventory_path=inventory_path,
        qualification_manifest_path=qualification_manifest_path,
        evidence_root=evidence_root,
        candidate_contract_gate_path=candidate_contract_gate_path,
        candidate_registry_gate_path=candidate_registry_gate_path,
        candidate_attestation_path=candidate_attestation_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_evidence_root=candidate_evidence_root,
        source_policy_path=source_policy_path,
        expected_source_policy_sha256=expected_source_policy_sha256,
        source_verifier_path=source_verifier_path,
        expected_source_verifier_sha256=expected_source_verifier_sha256,
        final_audit_authority_path=final_audit_authority_path,
        expected_final_audit_authority_sha256=expected_final_audit_authority_sha256,
    )
    _require(receipt == recreated, "final release receipt was not exactly re-derived")
    _require(receipt["authorization"] == {"release_publication": True, "git_tag": True, "package_mutation": False}, "final release authority differs")
    return receipt


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "receipt parent must already exist")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, "receipt parent cannot contain symlinks")
    destination = parent / destination.name
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite final receipt")
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
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, "final receipt mode differs")
        _require(destination.read_bytes() == payload, "final receipt bytes differ after publication")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", action="store_true")
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-tool", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--inversion-schema", type=pathlib.Path, required=True)
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--qualification-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--evidence-root", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-contract-gate", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-registry-gate", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-attestation", type=pathlib.Path, required=True)
    parser.add_argument("--m4-candidate-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-evidence-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-transition-policy", type=pathlib.Path, required=True)
    parser.add_argument("--expected-source-transition-policy-sha256", required=True)
    parser.add_argument("--source-transition-verifier", type=pathlib.Path, required=True)
    parser.add_argument("--expected-source-transition-verifier-sha256", required=True)
    parser.add_argument("--final-audit-authority", type=pathlib.Path, required=True)
    parser.add_argument("--expected-final-audit-authority-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        keywords = {
            "package_path": args.package,
            "candidate_receipt_path": args.candidate_receipt,
            "candidate_tool_path": args.candidate_tool,
            "contract_path": args.contract,
            "inversion_schema_path": args.inversion_schema,
            "inventory_path": args.inventory,
            "qualification_manifest_path": args.qualification_manifest,
            "evidence_root": args.evidence_root,
            "candidate_contract_gate_path": args.candidate_contract_gate,
            "candidate_registry_gate_path": args.candidate_registry_gate,
            "candidate_attestation_path": args.candidate_attestation,
            "candidate_manifest_path": args.m4_candidate_manifest,
            "candidate_evidence_root": args.candidate_evidence_root,
            "source_policy_path": args.source_transition_policy,
            "expected_source_policy_sha256": args.expected_source_transition_policy_sha256,
            "source_verifier_path": args.source_transition_verifier,
            "expected_source_verifier_sha256": args.expected_source_transition_verifier_sha256,
            "final_audit_authority_path": args.final_audit_authority,
            "expected_final_audit_authority_sha256": args.expected_final_audit_authority_sha256,
        }
        if args.create:
            _require(args.output is not None and args.receipt is None, "--create requires only --output")
            receipt = create(**keywords)
            _write_new(args.output, receipt)
        else:
            _require(args.receipt is not None and args.output is None, "--verify requires only --receipt")
            receipt = verify(receipt_path=args.receipt, **keywords)
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "receipt_id": receipt["receipt_id"], "package_sha256": receipt["package"]["sha256"]}, sort_keys=True))
        return 0
    except (OSError, FinalReleaseError, ValueError) as error:
        print(f"v1.1.0 final release receipt: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
