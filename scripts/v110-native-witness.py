#!/usr/bin/env python3
"""Capture or verify a typed v1.1.0 native qualification witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import tempfile
from typing import Any


SCHEMA = "gpmeep-v110-native-witness-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class WitnessError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WitnessError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    _require(isinstance(value, dict), "native witness root must be an object")
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


def _direct_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_file(), f"{label} is unsafe")
    return resolved


def _direct_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_dir(), f"{label} is unsafe")
    return resolved


def _live_environment() -> dict[str, Any]:
    lexical = pathlib.Path("/etc/os-release")
    _require(lexical.is_file(), "live /etc/os-release is absent")
    resolved = lexical.resolve(strict=True)
    raw = resolved.read_bytes()
    _require(len(raw) <= 1024 * 1024, "live /etc/os-release is too large")
    values: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _require("=" in line, "live /etc/os-release line differs")
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    _require(
        values.get("ID") == "ubuntu"
        and re.fullmatch(r"[0-9]+\.[0-9]+", values.get("VERSION_ID", "")) is not None,
        "native witness OS is not a supported Ubuntu identity",
    )
    namespaces = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "pid", "user")}
    uname = os.uname()
    environment = {
        "capture_source": "/etc/os-release",
        "os_release": {
            "id": values["ID"],
            "version_id": values["VERSION_ID"],
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "namespaces": namespaces,
        "kernel": {"sysname": uname.sysname, "release": uname.release, "machine": uname.machine},
        "libc": os.confstr("CS_GNU_LIBC_VERSION"),
        "cgroup_sha256": hashlib.sha256(pathlib.Path("/proc/self/cgroup").read_bytes()).hexdigest(),
    }
    environment["namespace_id"] = _canonical_sha256(namespaces)
    return environment


def create(
    *,
    prefix: pathlib.Path,
    installed_report_path: pathlib.Path,
    source_commit: str,
    package_sha256: str,
    candidate_receipt_id: str,
    image_identity: str,
) -> dict[str, Any]:
    prefix = _direct_directory(prefix, "installed prefix")
    installed_report_path = _direct_file(installed_report_path, "installed provenance report")
    _require(HEX40.fullmatch(source_commit or "") is not None, "source commit is invalid")
    _require(HEX64.fullmatch(package_sha256 or "") is not None, "package SHA-256 is invalid")
    _require(HEX64.fullmatch(candidate_receipt_id or "") is not None, "candidate receipt ID is invalid")
    _require(
        image_identity == "native-host"
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_identity or "") is not None,
        "image identity must be native-host or an immutable digest",
    )
    installed = _load(installed_report_path)
    _require(
        installed.get("state") == "PASS"
        and installed.get("prefix") == str(prefix)
        and installed.get("package_sha256") == package_sha256
        and installed.get("manifest", {}).get("source_commit") == source_commit
        and installed.get("v110_candidate_gate", {}).get("receipt_id") == candidate_receipt_id,
        "installed provenance binding differs",
    )
    unsigned_report = dict(installed)
    installed_report_id = unsigned_report.pop("report_id", None)
    _require(
        installed_report_id == _canonical_sha256(unsigned_report),
        "installed provenance report ID differs",
    )
    live = _live_environment()
    witness: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "PASS",
        "platform": f"ubuntu-{live['os_release']['version_id']}",
        "prefix": str(prefix),
        "source_commit": source_commit,
        "package_sha256": package_sha256,
        "candidate_receipt_id": candidate_receipt_id,
        "installed_report_id": installed_report_id,
        "installed_report_sha256": _sha256(installed_report_path),
        "image_identity": image_identity,
        **live,
    }
    witness["environment_id"] = _canonical_sha256(witness)
    witness["terminal_id"] = _canonical_sha256(witness)
    return witness


def verify(path: pathlib.Path, *, require_live: bool = False) -> dict[str, Any]:
    path = _direct_file(path, "native witness")
    witness = _load(path)
    expected_keys = {
        "schema", "state", "platform", "prefix", "source_commit", "package_sha256",
        "candidate_receipt_id", "installed_report_id", "installed_report_sha256",
        "image_identity", "capture_source", "os_release", "namespaces", "namespace_id",
        "kernel", "libc", "cgroup_sha256", "environment_id", "terminal_id",
    }
    _require(set(witness) == expected_keys, "native witness schema differs")
    unsigned = dict(witness)
    terminal_id = unsigned.pop("terminal_id")
    _require(terminal_id == _canonical_sha256(unsigned), "native witness terminal ID differs")
    environment_unsigned = dict(unsigned)
    environment_id = environment_unsigned.pop("environment_id")
    _require(environment_id == _canonical_sha256(environment_unsigned), "native witness environment ID differs")
    os_release = witness["os_release"]
    _require(
        witness["schema"] == SCHEMA
        and witness["state"] == "PASS"
        and witness["capture_source"] == "/etc/os-release"
        and isinstance(witness["prefix"], str)
        and pathlib.Path(witness["prefix"]).is_absolute()
        and isinstance(os_release, dict)
        and set(os_release) == {"id", "version_id", "size_bytes", "sha256"}
        and os_release["id"] == "ubuntu"
        and re.fullmatch(r"[0-9]+\.[0-9]+", os_release["version_id"] or "") is not None
        and type(os_release["size_bytes"]) is int
        and 0 < os_release["size_bytes"] <= 1024 * 1024
        and HEX64.fullmatch(os_release["sha256"] or "") is not None
        and witness["platform"] == f"ubuntu-{os_release['version_id']}"
        and HEX40.fullmatch(witness["source_commit"] or "") is not None
        and all(HEX64.fullmatch(witness[key] or "") is not None for key in ("package_sha256", "candidate_receipt_id", "installed_report_id", "installed_report_sha256", "namespace_id", "cgroup_sha256"))
        and isinstance(witness["namespaces"], dict)
        and set(witness["namespaces"]) == {"mnt", "pid", "user"}
        and all(
            isinstance(witness["namespaces"][key], str)
            and re.fullmatch(rf"{key}:\[[0-9]+\]", witness["namespaces"][key]) is not None
            for key in ("mnt", "pid", "user")
        )
        and witness["namespace_id"] == _canonical_sha256(witness["namespaces"])
        and isinstance(witness["kernel"], dict)
        and set(witness["kernel"]) == {"sysname", "release", "machine"}
        and all(isinstance(value, str) and value for value in witness["kernel"].values())
        and isinstance(witness["libc"], str)
        and re.fullmatch(r"glibc [0-9]+(?:\.[0-9]+)+", witness["libc"]) is not None
        and (witness["image_identity"] == "native-host" or re.fullmatch(r"sha256:[0-9a-f]{64}", witness["image_identity"] or "") is not None),
        "native witness identity differs",
    )
    if require_live:
        live = _live_environment()
        for key in ("capture_source", "os_release", "namespaces", "namespace_id", "kernel", "libc", "cgroup_sha256"):
            _require(witness[key] == live[key], f"native witness live {key} differs")
    return witness


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "native witness parent is absent")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite native witness")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--prefix", type=pathlib.Path)
    parser.add_argument("--installed-report", type=pathlib.Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--package-sha256")
    parser.add_argument("--candidate-receipt-id")
    parser.add_argument("--image-identity")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--require-live", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.create:
            _require(all((args.prefix, args.installed_report, args.source_commit, args.package_sha256, args.candidate_receipt_id, args.image_identity, args.output)), "native witness create arguments are incomplete")
            result = create(
                prefix=args.prefix,
                installed_report_path=args.installed_report,
                source_commit=args.source_commit,
                package_sha256=args.package_sha256,
                candidate_receipt_id=args.candidate_receipt_id,
                image_identity=args.image_identity,
            )
            _write_new(args.output, result)
        else:
            result = verify(args.verify, require_live=args.require_live)
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "terminal_id": result["terminal_id"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, WitnessError) as error:
        print(f"v1.1.0 native witness: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
