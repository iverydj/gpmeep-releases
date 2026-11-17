#!/usr/bin/env python3
"""Create and verify the receipt for the isolated M3 upstream CPU baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import pwd
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "gpmeep-v110-upstream-cpu-baseline-receipt-v1"
CONTRACT_RELATIVE = pathlib.Path("packaging/v110-upstream-cpu-baseline.json")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PACKAGE_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^\s#]+#[0-9a-f]{64}\Z")
LOCK_GENERATOR_HEADER = re.compile(
    r"# Generated with Micromamba [0-9]+\.[0-9]+\.[0-9]+ "
    r"from (?P<source>[A-Za-z0-9_./-]+\.ya?ml)\Z"
)
PINNED_MICROMAMBA_SHA256 = (
    "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
)
RECEIPT_FIELDS = {
    "artifacts",
    "baseline_id",
    "builder",
    "configure_arguments",
    "contract",
    "created_at_utc",
    "dependency_lock",
    "micromamba",
    "mpiexec",
    "python",
    "qualification",
    "repository",
    "schema",
    "source_archive",
    "state",
    "upstream_commit",
}


class ReceiptError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ReceiptError(f"non-finite JSON constant in {label}: {item}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot load {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _sha256(path: pathlib.Path) -> str:
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
    _require(not path.is_symlink(), f"symbolic-link input is forbidden: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    _require(identity(metadata) == identity(after), f"file changed while hashing: {path}")
    return digest.hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(os.path.abspath(path))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(repo: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.rstrip("\n")


def _contract(repo: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    path = repo / CONTRACT_RELATIVE
    contract = _load(path, "upstream baseline contract")
    _require(contract.get("schema_version") == 1, "contract schema differs")
    source = contract.get("source")
    dependency = contract.get("dependency_environment")
    build = contract.get("build")
    _require(isinstance(source, dict), "contract source is invalid")
    _require(isinstance(dependency, dict), "contract dependency environment is invalid")
    _require(isinstance(build, dict), "contract build is invalid")
    _require(
        set(dependency) == {"lock_path", "lock_sha256", "prefix_path"},
        "contract dependency environment fields differ",
    )
    _require(
        isinstance(source.get("upstream_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", source["upstream_commit"]) is not None,
        "contract upstream commit is invalid",
    )
    _require(
        build.get("required_cxx_optimization_flag") == "-O2",
        "contracted C++ optimization flag differs",
    )
    for value, label in (
        (source.get("archive_sha256"), "archive digest"),
        (dependency.get("lock_sha256"), "lock digest"),
    ):
        _require(isinstance(value, str) and HEX64.fullmatch(value), f"{label} is invalid")
    return path, contract


def _strict_repo_path(
    repo: pathlib.Path, relative: Any, label: str, *, directory: bool
) -> pathlib.Path:
    _require(isinstance(relative, str) and relative, f"{label} path is invalid")
    pure = pathlib.PurePosixPath(relative)
    _require(
        pure.as_posix() == relative
        and not pure.is_absolute()
        and "." not in pure.parts
        and ".." not in pure.parts,
        f"{label} path is not a canonical repository-relative path",
    )
    unresolved = repo / pathlib.Path(relative)
    resolved = unresolved.resolve(strict=True)
    _require(resolved.is_relative_to(repo), f"{label} path escapes repository")
    _require(resolved == unresolved, f"{label} path contains a symbolic link")
    metadata = unresolved.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    _require(expected_type(metadata.st_mode), f"{label} path type differs")
    return resolved


def _artifact_map(values: list[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        _require(separator == "=" and name and raw_path, "artifact must use NAME=PATH")
        _require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, "artifact name is invalid")
        _require(name not in result, f"duplicate artifact: {name}")
        result[name] = pathlib.Path(raw_path)
    _require(
        set(result) == {
            "libmeep",
            "python_extension",
            "mpb_extension",
            "libpympb",
        },
        "artifact inventory differs",
    )
    return result


def _qualification_map(
    values: list[str], contract: dict[str, Any]
) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    expected = contract["build"]["qualification"]
    _require(isinstance(expected, dict) and expected, "qualification contract is invalid")
    for value in values:
        name, separator, raw_path = value.partition("=")
        _require(separator == "=" and name and raw_path, "qualification must use NAME=PATH")
        _require(name not in result, f"duplicate qualification: {name}")
        result[name] = pathlib.Path(raw_path)
    _require(set(result) == set(expected), "qualification inventory differs")
    return result


def _validate_tool_paths(
    repo: pathlib.Path,
    contract: dict[str, Any],
    python: pathlib.Path,
    mpiexec: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    prefix = (repo / contract["dependency_environment"]["prefix_path"]).resolve(
        strict=True
    )
    expected_python = (prefix / "bin/python").resolve(strict=True)
    expected_mpiexec = (prefix / "bin/mpiexec").resolve(strict=True)
    actual_python = python.resolve(strict=True)
    actual_mpiexec = mpiexec.resolve(strict=True)
    _require(actual_python == expected_python, "Python is not the contracted exact-lock tool")
    _require(actual_mpiexec == expected_mpiexec, "mpiexec is not the contracted exact-lock tool")
    return actual_python, actual_mpiexec


def _validate_artifact_paths(
    repo: pathlib.Path,
    contract: dict[str, Any],
    artifacts: dict[str, pathlib.Path],
) -> dict[str, pathlib.Path]:
    install = (repo / contract["build"]["install_prefix"]).resolve(strict=True)
    resolved = {
        name: path.resolve(strict=True) for name, path in artifacts.items()
    }
    for name, path in resolved.items():
        _require(path.is_relative_to(install), f"artifact escapes install prefix: {name}")
    install_lib = (install / "lib").resolve(strict=True)
    for name, stem in (("libmeep", "libmeep.so"), ("libpympb", "libpympb.so")):
        path = resolved[name]
        _require(
            path.parent == install_lib and path.name.startswith(stem),
            f"artifact path differs: {name}",
        )
    relative_python = resolved["python_extension"].relative_to(install).parts
    relative_mpb = resolved["mpb_extension"].relative_to(install).parts
    _require(
        len(relative_python) == 5
        and relative_python[0] == "lib"
        and re.fullmatch(r"python[0-9]+\.[0-9]+", relative_python[1]) is not None
        and relative_python[2:4] == ("site-packages", "meep")
        and re.fullmatch(r"_meep\.so(?:\.[0-9]+)*", relative_python[4]) is not None,
        "artifact path differs: python_extension",
    )
    _require(
        len(relative_mpb) == 6
        and relative_mpb[:2] == relative_python[:2]
        and relative_mpb[2:5] == ("site-packages", "meep", "mpb")
        and re.fullmatch(r"_mpb\.so(?:\.[0-9]+)*", relative_mpb[5]) is not None,
        "artifact path differs: mpb_extension",
    )
    return resolved


def _validate_qualification_paths(
    repo: pathlib.Path,
    contract: dict[str, Any],
    qualification: dict[str, pathlib.Path],
) -> dict[str, pathlib.Path]:
    build = (repo / contract["build"]["build_directory"]).resolve(strict=True)
    expected = contract["build"]["qualification"]
    resolved = {
        name: path.resolve(strict=True) for name, path in qualification.items()
    }
    _require(set(resolved) == set(expected), "qualification path inventory differs")
    for name, relative in expected.items():
        path = resolved[name]
        _require(
            path == (build / pathlib.Path(relative)).resolve(strict=True),
            f"qualification path differs: {name}",
        )
        _require(path.stat().st_size > 0, f"qualification is empty: {name}")
    return resolved


def _validate_build_configuration(
    contract: dict[str, Any], qualification: dict[str, pathlib.Path]
) -> None:
    configuration = qualification["upstream_build_configuration"]
    try:
        lines = configuration.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(f"cannot read upstream build configuration: {error}") from error
    cxxflags = [line.partition("=")[2].split() for line in lines if line.startswith("CXXFLAGS =")]
    _require(len(cxxflags) == 1, "upstream CXXFLAGS record is not unique")
    required = contract["build"]["required_cxx_optimization_flag"]
    _require(required in cxxflags[0], "upstream C++ optimization flag is absent")


def _validate_fontconfig_configuration(
    repo: pathlib.Path,
    contract: dict[str, Any],
    qualification: dict[str, pathlib.Path],
) -> None:
    prefix = (repo / contract["dependency_environment"]["prefix_path"]).resolve(
        strict=True
    )
    configuration = qualification["upstream_fontconfig_configuration"]
    cache = configuration.parent / "qualification-cache/fontconfig"
    expected = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
        "<fontconfig>\n"
        "  <dir>/usr/share/fonts</dir>\n"
        f"  <dir>{prefix}/share/fonts</dir>\n"
        f"  <dir>{prefix}/fonts</dir>\n"
        f'  <include ignore_missing="yes">{prefix}/etc/fonts/conf.d</include>\n'
        f"  <cachedir>{cache}</cachedir>\n"
        "</fontconfig>\n"
    )
    try:
        actual = configuration.read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(f"cannot read upstream Fontconfig file: {error}") from error
    _require(actual == expected, "upstream Fontconfig configuration differs")


def _dependency_package_entries(lock: pathlib.Path) -> list[str]:
    try:
        payload = lock.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(f"cannot read dependency lock: {error}") from error
    _require(payload.endswith("\n"), "dependency lock lacks a terminal newline")
    lines = payload.splitlines()
    header = LOCK_GENERATOR_HEADER.fullmatch(lines[0]) if lines else None
    _require(header is not None, "dependency lock generator header differs")
    source = pathlib.PurePosixPath(header.group("source"))
    _require(
        not source.is_absolute()
        and "." not in source.parts
        and ".." not in source.parts
        and source.as_posix() == header.group("source"),
        "dependency lock generator source differs",
    )
    _require(
        len(lines) >= 4
        and lines[1] == "# platform: linux-64"
        and lines[2] == "@EXPLICIT",
        "dependency lock explicit header differs",
    )
    entries = lines[3:]
    _require(entries, "dependency lock package set is empty")
    for entry in entries:
        _require(
            PACKAGE_URL.fullmatch(entry) is not None,
            "dependency lock package URL is invalid",
        )
    _require(len(entries) == len(set(entries)), "dependency lock has duplicate packages")
    package_urls = [entry.rsplit("#", 1)[0] for entry in entries]
    _require(
        len(package_urls) == len(set(package_urls)),
        "dependency lock has duplicate package URLs",
    )
    return sorted(entries)


def _validate_micromamba(repo: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    micromamba = _strict_repo_path(
        repo, ".tools/micromamba", "Micromamba", directory=False
    )
    _require(os.access(micromamba, os.X_OK), "Micromamba is not executable")
    record = _file_record(micromamba)
    _require(
        record["sha256"] == PINNED_MICROMAMBA_SHA256,
        "Micromamba binary digest differs",
    )
    return micromamba, record


def _dependency_paths(
    repo: pathlib.Path, contract: dict[str, Any]
) -> tuple[pathlib.Path, pathlib.Path]:
    dependency = contract["dependency_environment"]
    lock = _strict_repo_path(
        repo, dependency["lock_path"], "dependency lock", directory=False
    )
    prefix = _strict_repo_path(
        repo, dependency["prefix_path"], "dependency prefix", directory=True
    )
    return lock, prefix


def _validate_dependency_lock(repo: pathlib.Path, contract: dict[str, Any]) -> pathlib.Path:
    lock, _ = _dependency_paths(repo, contract)
    dependency = contract["dependency_environment"]
    _require(
        _sha256(lock) == dependency["lock_sha256"],
        "dependency lock digest differs",
    )
    _dependency_package_entries(lock)
    return lock


def _live_dependency_package_entries(
    repo: pathlib.Path, contract: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    _, prefix = _dependency_paths(repo, contract)
    micromamba, before = _validate_micromamba(repo)
    root = _strict_repo_path(
        repo, ".micromamba", "Micromamba root", directory=True
    )
    cache = _strict_repo_path(
        repo, ".micromamba/cache", "Micromamba cache", directory=True
    )
    try:
        user_home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError as error:
        raise ReceiptError("launch user account is unavailable") from error
    _require(os.path.isabs(user_home), "launch user home is invalid")
    try:
        result = subprocess.run(
            [
                str(micromamba),
                "--no-rc",
                "list",
                "--root-prefix",
                str(root),
                "--prefix",
                str(prefix),
                "--json",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
            env={
                "HOME": user_home,
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "XDG_CACHE_HOME": str(cache),
            },
        )
    except UnicodeDecodeError as error:
        raise ReceiptError("Micromamba live prefix inventory encoding differs") from error
    _, after = _validate_micromamba(repo)
    _require(before == after, "Micromamba binary changed during live inventory")
    _require(result.returncode == 0, "Micromamba live prefix inventory failed")
    _require(
        len(result.stdout.encode("utf-8")) <= 4 * 1024 * 1024
        and len(result.stderr.encode("utf-8")) <= 1024 * 1024,
        "Micromamba live prefix inventory output is too large",
    )
    try:
        records = json.loads(
            result.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ReceiptError(
                    f"non-finite JSON constant in live prefix inventory: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("Micromamba live prefix inventory JSON differs") from error
    _require(
        isinstance(records, list) and records,
        "Micromamba live prefix package set is empty",
    )
    packages: list[dict[str, str]] = []
    package_names: list[str] = []
    for record in records:
        _require(isinstance(record, dict), "Micromamba package record is invalid")
        name = record.get("name")
        version = record.get("version")
        build_string = record.get("build_string")
        dist_name = record.get("dist_name")
        url = record.get("url")
        sha256 = record.get("sha256")
        _require(
            isinstance(name, str)
            and re.fullmatch(r"[a-z0-9_.-]+", name) is not None,
            "Micromamba package identity is invalid",
        )
        _require(
            isinstance(version, str)
            and version
            and isinstance(build_string, str)
            and build_string
            and isinstance(dist_name, str)
            and dist_name == f"{name}-{version}-{build_string}",
            "Micromamba package distribution identity differs",
        )
        _require(
            isinstance(url, str)
            and isinstance(sha256, str)
            and HEX64.fullmatch(sha256) is not None,
            "Micromamba package URL or digest is invalid",
        )
        filename = url.rsplit("/", 1)[-1]
        if filename.endswith(".conda"):
            filename_identity = filename.removesuffix(".conda")
        elif filename.endswith(".tar.bz2"):
            filename_identity = filename.removesuffix(".tar.bz2")
        else:
            raise ReceiptError("Micromamba package URL suffix differs")
        canonical_parts = filename_identity.rsplit("-", 2)
        _require(
            len(canonical_parts) == 3
            and canonical_parts == [name, version, build_string]
            and filename_identity == dist_name,
            "Micromamba package URL identity differs from package metadata",
        )
        entry = f"{url}#{sha256}"
        _require(
            PACKAGE_URL.fullmatch(entry) is not None,
            "Micromamba live prefix package URL is invalid",
        )
        package_names.append(name)
        packages.append({"entry": entry, "name": name})
    _require(
        len(packages) == len({package["entry"] for package in packages}),
        "Micromamba live prefix inventory has duplicate packages",
    )
    _require(
        len(package_names) == len(set(package_names)),
        "Micromamba live prefix inventory has duplicate package identities",
    )
    return sorted(packages, key=lambda package: package["entry"]), before


def _dependency_package_set_payload(
    repo: pathlib.Path,
    contract: dict[str, Any],
    packages: list[dict[str, str]],
    micromamba: dict[str, Any],
) -> str:
    dependency = contract["dependency_environment"]
    lock, prefix = _dependency_paths(repo, contract)
    lock_relative = pathlib.PurePosixPath(dependency["lock_path"]).as_posix()
    return "\n".join(
        [
            "schema=gpmeep-upstream-dependency-package-set-v1",
            f"prefix={prefix}",
            f"lock={lock_relative}",
            f"lock_sha256={_sha256(lock)}",
            f"micromamba={micromamba['path']}",
            f"micromamba_sha256={micromamba['sha256']}",
            f"micromamba_size_bytes={micromamba['size_bytes']}",
            f"package_count={len(packages)}",
            *[
                f"package={package['name']}\t{package['entry']}"
                for package in packages
            ],
            "",
        ]
    )


def _expected_dependency_package_set(
    repo: pathlib.Path, contract: dict[str, Any]
) -> str:
    lock = _validate_dependency_lock(repo, contract)
    expected_entries = _dependency_package_entries(lock)
    packages, micromamba = _live_dependency_package_entries(repo, contract)
    _require(
        [package["entry"] for package in packages] == expected_entries,
        "live dependency prefix package set differs from exact lock",
    )
    return _dependency_package_set_payload(repo, contract, packages, micromamba)


def _validate_dependency_package_set(
    repo: pathlib.Path,
    contract: dict[str, Any],
    qualification: dict[str, pathlib.Path],
) -> dict[str, Any]:
    lock = _validate_dependency_lock(repo, contract)
    expected_entries = _dependency_package_entries(lock)
    packages, micromamba = _live_dependency_package_entries(repo, contract)
    _require(
        [package["entry"] for package in packages] == expected_entries,
        "live dependency prefix package set differs from exact lock",
    )
    package_set = qualification["upstream_dependency_package_set"]
    try:
        actual = package_set.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(f"cannot read dependency package set: {error}") from error
    expected = _dependency_package_set_payload(
        repo, contract, packages, micromamba
    )
    _require(actual == expected, "upstream dependency package set differs")
    return micromamba


def create(args: argparse.Namespace) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    _require(repo.is_dir(), "repository is not a directory")
    contract_path, contract = _contract(repo)
    archive = pathlib.Path(os.path.abspath(args.archive))
    lock = repo / contract["dependency_environment"]["lock_path"]
    builder = repo / contract["build"]["builder"]
    artifacts = _artifact_map(args.artifact)
    qualification = _qualification_map(args.qualification, contract)
    python, mpiexec = _validate_tool_paths(
        repo, contract, pathlib.Path(args.python), pathlib.Path(args.mpiexec)
    )
    artifacts = _validate_artifact_paths(repo, contract, artifacts)
    qualification = _validate_qualification_paths(repo, contract, qualification)
    _validate_build_configuration(contract, qualification)
    _validate_fontconfig_configuration(repo, contract, qualification)
    _require(
        _sha256(archive) == contract["source"]["archive_sha256"],
        "upstream source archive digest differs",
    )
    micromamba = _validate_dependency_package_set(repo, contract, qualification)
    status = _git(repo, "status", "--short")
    _require(status == "", "repository must be clean before sealing the baseline")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_id": contract["baseline_id"],
        "repository": {
            "path": str(repo),
            "git_head": _git(repo, "rev-parse", "HEAD"),
        },
        "contract": _file_record(contract_path),
        "source_archive": _file_record(archive),
        "upstream_commit": contract["source"]["upstream_commit"],
        "dependency_lock": _file_record(lock),
        "micromamba": micromamba,
        "builder": _file_record(builder),
        "configure_arguments": contract["build"]["configure_arguments"],
        "python": _file_record(python),
        "mpiexec": _file_record(mpiexec),
        "artifacts": {
            name: _file_record(path.resolve(strict=True))
            for name, path in sorted(artifacts.items())
        },
        "qualification": {
            name: _file_record(path.resolve(strict=True))
            for name, path in sorted(qualification.items())
        },
    }
    receipt["receipt_id"] = _canonical_digest(receipt)
    return receipt


def verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    receipt_path = pathlib.Path(os.path.abspath(args.receipt))
    receipt = _load(receipt_path, "upstream baseline receipt")
    _require(
        set(receipt) == RECEIPT_FIELDS | {"receipt_id"},
        "receipt field inventory differs",
    )
    receipt_id = receipt.pop("receipt_id", None)
    _require(isinstance(receipt_id, str) and HEX64.fullmatch(receipt_id), "receipt ID is invalid")
    _require(_canonical_digest(receipt) == receipt_id, "receipt ID differs")
    receipt["receipt_id"] = receipt_id
    _require(receipt.get("schema") == SCHEMA, "receipt schema differs")
    _require(receipt.get("state") == "complete", "receipt is not complete")
    created_at = receipt.get("created_at_utc")
    _require(isinstance(created_at, str), "receipt timestamp is invalid")
    try:
        parsed_created_at = dt.datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ReceiptError("receipt timestamp is invalid") from error
    _require(
        parsed_created_at.tzinfo == dt.timezone.utc
        and parsed_created_at.isoformat() == created_at,
        "receipt timestamp is not canonical UTC",
    )
    contract_path, contract = _contract(repo)
    _require(receipt.get("baseline_id") == contract["baseline_id"], "baseline ID differs")
    _require(
        receipt.get("upstream_commit") == contract["source"]["upstream_commit"],
        "upstream commit differs",
    )
    _require(receipt.get("contract") == _file_record(contract_path), "contract record differs")
    _require(
        receipt.get("dependency_lock")
        == _file_record(repo / contract["dependency_environment"]["lock_path"]),
        "dependency lock record differs",
    )
    _validate_dependency_lock(repo, contract)
    _require(
        receipt.get("builder") == _file_record(repo / contract["build"]["builder"]),
        "builder record differs",
    )
    _require(
        receipt.get("source_archive")
        == _file_record(pathlib.Path(receipt["source_archive"]["path"])),
        "source archive record differs",
    )
    _require(
        receipt["source_archive"]["sha256"] == contract["source"]["archive_sha256"],
        "source archive is not the contracted archive",
    )
    _require(
        receipt.get("configure_arguments") == contract["build"]["configure_arguments"],
        "configure arguments differ",
    )
    _require(
        receipt.get("repository")
        == {"path": str(repo), "git_head": _git(repo, "rev-parse", "HEAD")},
        "repository source identity differs",
    )
    _require(_git(repo, "status", "--short") == "", "repository is dirty")
    expected_python, expected_mpiexec = _validate_tool_paths(
        repo,
        contract,
        pathlib.Path(receipt["python"]["path"]),
        pathlib.Path(receipt["mpiexec"]["path"]),
    )
    expected_tools = {"python": expected_python, "mpiexec": expected_mpiexec}
    for name in ("python", "mpiexec"):
        record = receipt.get(name)
        _require(isinstance(record, dict), f"{name} record is invalid")
        _require(record == _file_record(expected_tools[name]), f"{name} record differs")
    artifacts = receipt.get("artifacts")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == {"libmeep", "python_extension", "mpb_extension", "libpympb"},
        "artifact receipt inventory differs",
    )
    artifact_paths = _validate_artifact_paths(
        repo,
        contract,
        {
            name: pathlib.Path(record["path"])
            for name, record in artifacts.items()
            if isinstance(record, dict) and isinstance(record.get("path"), str)
        },
    )
    for name, record in artifacts.items():
        _require(
            isinstance(record, dict)
            and record == _file_record(artifact_paths[name]),
            f"artifact record differs: {name}",
        )
    qualification = receipt.get("qualification")
    _require(isinstance(qualification, dict), "qualification receipt is invalid")
    qualification_paths = _validate_qualification_paths(
        repo,
        contract,
        {
            name: pathlib.Path(record["path"])
            for name, record in qualification.items()
            if isinstance(record, dict) and isinstance(record.get("path"), str)
        },
    )
    _validate_build_configuration(contract, qualification_paths)
    _validate_fontconfig_configuration(repo, contract, qualification_paths)
    micromamba = _validate_dependency_package_set(
        repo, contract, qualification_paths
    )
    _require(receipt.get("micromamba") == micromamba, "Micromamba record differs")
    for name, record in qualification.items():
        _require(
            isinstance(record, dict)
            and record == _file_record(qualification_paths[name]),
            f"qualification record differs: {name}",
        )
    return {
        "schema": "gpmeep-v110-upstream-cpu-baseline-verification-v1",
        "state": "PASS",
        "receipt_id": receipt_id,
        "baseline_id": receipt["baseline_id"],
        "upstream_commit": receipt["upstream_commit"],
    }


def verify_environment(args: argparse.Namespace) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    _, contract = _contract(repo)
    lock = _validate_dependency_lock(repo, contract)
    expected_entries = _dependency_package_entries(lock)
    packages, micromamba = _live_dependency_package_entries(repo, contract)
    _require(
        [package["entry"] for package in packages] == expected_entries,
        "live dependency prefix package set differs from exact lock",
    )
    return {
        "schema": "gpmeep-v110-upstream-dependency-environment-verification-v1",
        "state": "PASS",
        "lock_sha256": _sha256(lock),
        "micromamba": micromamba,
        "package_count": len(packages),
        "package_set_id": _canonical_digest({"packages": packages}),
    }


def capture_environment(args: argparse.Namespace) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    _, contract = _contract(repo)
    payload = _expected_dependency_package_set(repo, contract)
    output = pathlib.Path(os.path.abspath(args.output))
    _write_text(output, payload)
    record = _file_record(output)
    return {
        "schema": "gpmeep-v110-upstream-dependency-environment-capture-v1",
        "state": "PASS",
        "package_set": record,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create_parser = subparsers.add_parser("create", allow_abbrev=False)
    create_parser.add_argument("--repo", required=True, type=pathlib.Path)
    create_parser.add_argument("--archive", required=True, type=pathlib.Path)
    create_parser.add_argument("--python", required=True, type=pathlib.Path)
    create_parser.add_argument("--mpiexec", required=True, type=pathlib.Path)
    create_parser.add_argument("--artifact", action="append", default=[])
    create_parser.add_argument("--qualification", action="append", default=[])
    create_parser.add_argument("--output", required=True, type=pathlib.Path)
    verify_parser = subparsers.add_parser("verify", allow_abbrev=False)
    verify_parser.add_argument("--repo", required=True, type=pathlib.Path)
    verify_parser.add_argument("--receipt", required=True, type=pathlib.Path)
    environment_parser = subparsers.add_parser(
        "verify-environment", allow_abbrev=False
    )
    environment_parser.add_argument("--repo", required=True, type=pathlib.Path)
    capture_parser = subparsers.add_parser(
        "capture-environment", allow_abbrev=False
    )
    capture_parser.add_argument("--repo", required=True, type=pathlib.Path)
    capture_parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def _write(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = pathlib.Path(os.path.abspath(path))
    _require(not path.exists() and not path.is_symlink(), "receipt output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: pathlib.Path, value: str) -> None:
    path = pathlib.Path(os.path.abspath(path))
    _require(not path.exists() and not path.is_symlink(), "text output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "create":
        value = create(args)
        _write(args.output, value)
    elif args.operation == "capture-environment":
        value = capture_environment(args)
    elif args.operation == "verify-environment":
        value = verify_environment(args)
    else:
        value = verify(args)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ReceiptError, subprocess.SubprocessError) as error:
        print(f"upstream CPU baseline receipt error: {error}", file=sys.stderr)
        raise SystemExit(2)
