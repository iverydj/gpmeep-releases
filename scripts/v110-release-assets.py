#!/usr/bin/env python3
"""Create and replay integrity-only gpmeep v1.1.0 release assets.

The bundle proves only a relocated file inventory.  It is deliberately not a
semantic release authority; semantic replay requires the fixed producer tools
and native environments referenced by the final receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import tarfile
import tempfile
from typing import Any, Iterable


BUNDLE_SCHEMA = "gpmeep-v110-release-evidence-integrity-only-v2"
ASSET_SCHEMA = "gpmeep-v110-release-asset-manifest-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SAFE_ASSET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$")
MAX_MEMBERS = 200_000
MAX_MEMBER_BYTES = 128 * 1024**3


class ReleaseAssetError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAssetError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    _require(lexical.is_file(), f"{label} is not a regular file")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(stat.S_ISREG(resolved.stat().st_mode), f"{label} is not regular")
    return resolved


def _regular_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    _require(lexical.is_dir(), f"{label} is not a directory")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    return resolved


def _safe_relative(value: str) -> bool:
    path = pathlib.PurePosixPath(value)
    return (
        bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def _record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_new(path: pathlib.Path, payload: bytes) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "output parent must already exist")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, "output parent cannot contain symlinks")
    destination = parent / destination.name
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"refusing to overwrite output: {destination.name}",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=parent
    )
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
        _require(
            stat.S_IMODE(destination.stat().st_mode) == 0o600,
            "output mode differs",
        )
        _require(destination.read_bytes() == payload, "output bytes differ after publication")
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


def _parse_role_path(items: Iterable[str]) -> dict[str, pathlib.Path]:
    parsed: dict[str, pathlib.Path] = {}
    for item in items:
        role, separator, raw_path = item.partition("=")
        _require(separator == "=" and raw_path, "role input must be ROLE=PATH")
        _require(SAFE_ROLE.fullmatch(role) is not None, f"unsafe role: {role}")
        _require(role not in parsed, f"duplicate role: {role}")
        parsed[role] = pathlib.Path(raw_path)
    return parsed


def _evidence_files(
    root: pathlib.Path, tree_role: str
) -> list[tuple[str, pathlib.Path]]:
    records: list[tuple[str, pathlib.Path]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        for name in directories:
            _require(
                not (current_path / name).is_symlink(),
                f"evidence directory cannot be a symlink: {name}",
            )
        for name in files:
            path = current_path / name
            _require(not path.is_symlink(), f"evidence file cannot be a symlink: {name}")
            _require(stat.S_ISREG(path.stat().st_mode), f"evidence member is not regular: {name}")
            relative = path.relative_to(root).as_posix()
            _require(_safe_relative(relative), f"unsafe evidence path: {relative}")
            records.append((f"evidence/{tree_role}/{relative}", path))
    records.sort(key=lambda item: item[0])
    _require(bool(records), "evidence root is empty")
    _require(len(records) <= MAX_MEMBERS, "evidence root contains too many files")
    return records


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def create_bundle(
    evidence_roots: dict[str, pathlib.Path],
    authorities: dict[str, pathlib.Path],
    output: pathlib.Path,
) -> dict[str, Any]:
    manifest, members = _derive_bundle_manifest(evidence_roots, authorities)
    manifest_payload = _canonical_bytes(manifest) + b"\n"

    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w:", format=tarfile.PAX_FORMAT) as archive:
            archive.addfile(
                _tar_info("bundle-manifest.json", len(manifest_payload)),
                io.BytesIO(manifest_payload),
            )
            for archive_path, path, _role in members:
                with path.open("rb") as source:
                    archive.addfile(_tar_info(archive_path, path.stat().st_size), source)
    _write_new(output, stream.getvalue())
    verified = verify_bundle(output)
    _require(verified == manifest, "new evidence bundle did not replay exactly")
    return verified


def _derive_bundle_manifest(
    evidence_roots: dict[str, pathlib.Path],
    authorities: dict[str, pathlib.Path],
) -> tuple[dict[str, Any], list[tuple[str, pathlib.Path, str]]]:
    _require(bool(evidence_roots), "at least one evidence tree is required")
    _require(bool(authorities), "at least one authority file is required")
    members: list[tuple[str, pathlib.Path, str]] = []
    for tree_role, raw_root in sorted(evidence_roots.items()):
        _require(SAFE_ROLE.fullmatch(tree_role) is not None, f"unsafe evidence tree role: {tree_role}")
        root = _regular_directory(raw_root, f"evidence tree {tree_role}")
        members.extend(
            (archive_path, path, "evidence")
            for archive_path, path in _evidence_files(root, tree_role)
        )
    for role, raw_path in sorted(authorities.items()):
        path = _regular_file(raw_path, f"authority {role}")
        name = path.name
        _require(_safe_relative(name) and "/" not in name, "unsafe authority file name")
        members.append((f"authority/{role}/{name}", path, role))
    archive_names = [item[0] for item in members]
    _require(len(archive_names) == len(set(archive_names)), "duplicate archive member")
    manifest_files: list[dict[str, Any]] = []
    for archive_path, path, role in members:
        record = _record(path)
        _require(record["size_bytes"] <= MAX_MEMBER_BYTES, "evidence member is too large")
        manifest_files.append(
            {"path": archive_path, "role": role, **record}
        )
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "portable_integrity_only": True,
        "semantic_replay": False,
        "release_authority": False,
        "files": manifest_files,
    }
    manifest["bundle_id"] = _canonical_sha256(manifest)
    return manifest, members


def verify_bundle_inputs(
    archive_path: pathlib.Path,
    evidence_roots: dict[str, pathlib.Path],
    authorities: dict[str, pathlib.Path],
) -> dict[str, Any]:
    retained = verify_bundle(archive_path)
    expected, _members = _derive_bundle_manifest(evidence_roots, authorities)
    _require(retained == expected, "evidence bundle differs from current release inputs")
    return retained


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ReleaseAssetError(f"{label} contains non-finite JSON: {value}")
        ),
    )
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def verify_bundle(archive_path: pathlib.Path) -> dict[str, Any]:
    path = _regular_file(archive_path, "evidence archive")
    seen: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        _require(len(members) <= MAX_MEMBERS + 1, "archive contains too many members")
        for member in members:
            _require(_safe_relative(member.name), f"unsafe archive member: {member.name}")
            _require(member.name not in seen, f"duplicate archive member: {member.name}")
            _require(member.isfile(), f"archive member is not a regular file: {member.name}")
            _require(0 <= member.size <= MAX_MEMBER_BYTES, "archive member is too large")
            extracted = archive.extractfile(member)
            _require(extracted is not None, "archive member could not be read")
            payload = extracted.read(MAX_MEMBER_BYTES + 1)
            _require(len(payload) == member.size, "archive member size differs")
            seen[member.name] = payload
    _require("bundle-manifest.json" in seen, "bundle manifest is missing")
    manifest = _load_json_bytes(seen.pop("bundle-manifest.json"), "bundle manifest")
    _require(
        set(manifest)
        == {
            "schema",
            "portable_integrity_only",
            "semantic_replay",
            "release_authority",
            "files",
            "bundle_id",
        },
        "bundle manifest schema differs",
    )
    _require(manifest["schema"] == BUNDLE_SCHEMA, "bundle schema differs")
    _require(
        manifest["portable_integrity_only"] is True
        and manifest["semantic_replay"] is False
        and manifest["release_authority"] is False,
        "bundle integrity-only scope differs",
    )
    unsigned = dict(manifest)
    bundle_id = unsigned.pop("bundle_id")
    _require(
        isinstance(bundle_id, str)
        and HEX64.fullmatch(bundle_id) is not None
        and _canonical_sha256(unsigned) == bundle_id,
        "bundle identity differs",
    )
    files = manifest["files"]
    _require(isinstance(files, list) and files, "bundle file inventory is empty")
    expected_names: list[str] = []
    authority_roles: set[str] = set()
    for record in files:
        _require(
            isinstance(record, dict)
            and set(record) == {"path", "role", "size_bytes", "sha256"},
            "bundle file record schema differs",
        )
        name = record["path"]
        role = record["role"]
        _require(isinstance(name, str) and _safe_relative(name), "bundle file path is unsafe")
        _require(isinstance(role, str) and (role == "evidence" or SAFE_ROLE.fullmatch(role)), "bundle role differs")
        _require(name in seen, f"bundled file is missing: {name}")
        payload = seen[name]
        _require(record["size_bytes"] == len(payload), f"bundled file size differs: {name}")
        _require(record["sha256"] == _sha256_bytes(payload), f"bundled file hash differs: {name}")
        if role == "evidence":
            _require(name.startswith("evidence/"), "evidence path prefix differs")
        else:
            _require(name.startswith(f"authority/{role}/"), "authority path prefix differs")
            _require(role not in authority_roles, f"duplicate authority role: {role}")
            authority_roles.add(role)
        expected_names.append(name)
    _require(expected_names == sorted(expected_names, key=lambda name: (not name.startswith("evidence/"), name)), "bundle inventory order differs")
    _require(len(expected_names) == len(set(expected_names)), "bundle inventory contains duplicates")
    _require(set(expected_names) == set(seen), "archive member inventory differs")
    _require(bool(authority_roles), "bundle contains no authority files")
    return manifest


def create_asset_manifest(
    assets: dict[str, pathlib.Path], output: pathlib.Path
) -> dict[str, Any]:
    _require(bool(assets), "at least one release asset is required")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for role, raw_path in sorted(assets.items()):
        path = _regular_file(raw_path, f"release asset {role}")
        _require(SAFE_ASSET_NAME.fullmatch(path.name) is not None, f"unsafe release asset name: {path.name}")
        _require(path.name not in names, f"duplicate release asset name: {path.name}")
        names.add(path.name)
        records.append({"role": role, "name": path.name, **_record(path)})
    manifest: dict[str, Any] = {"schema": ASSET_SCHEMA, "assets": records}
    manifest["manifest_id"] = _canonical_sha256(manifest)
    _write_new(output, _canonical_bytes(manifest) + b"\n")
    return verify_asset_manifest(output)


def verify_asset_manifest(path: pathlib.Path) -> dict[str, Any]:
    source = _regular_file(path, "release asset manifest")
    manifest = _load_json_bytes(source.read_bytes(), "release asset manifest")
    _require(set(manifest) == {"schema", "assets", "manifest_id"}, "asset manifest schema differs")
    _require(manifest["schema"] == ASSET_SCHEMA, "asset manifest schema differs")
    unsigned = dict(manifest)
    manifest_id = unsigned.pop("manifest_id")
    _require(
        isinstance(manifest_id, str)
        and HEX64.fullmatch(manifest_id) is not None
        and _canonical_sha256(unsigned) == manifest_id,
        "asset manifest identity differs",
    )
    records = manifest["assets"]
    _require(isinstance(records, list) and records, "asset inventory is empty")
    roles: set[str] = set()
    names: set[str] = set()
    for record in records:
        _require(
            isinstance(record, dict)
            and set(record) == {"role", "name", "size_bytes", "sha256"},
            "asset record schema differs",
        )
        role, name = record["role"], record["name"]
        _require(isinstance(role, str) and SAFE_ROLE.fullmatch(role), "asset role differs")
        _require(isinstance(name, str) and SAFE_ASSET_NAME.fullmatch(name) is not None, "asset name differs")
        _require(role not in roles and name not in names, "duplicate asset role or name")
        _require(isinstance(record["size_bytes"], int) and record["size_bytes"] >= 0, "asset size differs")
        _require(isinstance(record["sha256"], str) and HEX64.fullmatch(record["sha256"]), "asset hash differs")
        roles.add(role)
        names.add(name)
    _require([item["role"] for item in records] == sorted(roles), "asset inventory order differs")
    return manifest


def verify_asset_sources(
    manifest_path: pathlib.Path, assets: dict[str, pathlib.Path]
) -> dict[str, Any]:
    retained = verify_asset_manifest(manifest_path)
    _require(
        set(assets) == {record["role"] for record in retained["assets"]},
        "local release asset roles differ",
    )
    expected: list[dict[str, Any]] = []
    for role, raw_path in sorted(assets.items()):
        path = _regular_file(raw_path, f"release asset {role}")
        expected.append({"role": role, "name": path.name, **_record(path)})
    _require(retained["assets"] == expected, "release assets differ from manifest")
    return retained


def verify_asset_directory(
    manifest_path: pathlib.Path,
    directory: pathlib.Path,
    expected_manifest_id: str,
) -> dict[str, Any]:
    manifest = verify_asset_manifest(manifest_path)
    _require(
        isinstance(expected_manifest_id, str)
        and HEX64.fullmatch(expected_manifest_id) is not None
        and manifest["manifest_id"] == expected_manifest_id,
        "downloaded asset manifest differs from the tag-bound manifest ID",
    )
    root = _regular_directory(directory, "download directory")
    expected = {pathlib.Path(manifest_path).name}
    for record in manifest["assets"]:
        expected.add(record["name"])
        path = _regular_file(root / record["name"], f"downloaded asset {record['role']}")
        _require(path.parent == root, "downloaded asset escaped directory")
        _require(path.stat().st_size == record["size_bytes"], f"downloaded asset size differs: {record['name']}")
        _require(_sha256(path) == record["sha256"], f"downloaded asset hash differs: {record['name']}")
    actual = {item.name for item in root.iterdir()}
    _require(actual == expected, "downloaded release asset set differs")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create-bundle", action="store_true")
    operation.add_argument("--verify-bundle", action="store_true")
    operation.add_argument("--create-asset-manifest", action="store_true")
    operation.add_argument("--verify-asset-sources", action="store_true")
    operation.add_argument("--verify-asset-directory", action="store_true")
    parser.add_argument("--evidence-tree", action="append", default=[])
    parser.add_argument("--authority", action="append", default=[])
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--archive", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--directory", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expected-manifest-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.create_bundle:
            _require(args.evidence_tree and args.output is not None, "bundle creation requires --evidence-tree and --output")
            value = create_bundle(
                _parse_role_path(args.evidence_tree),
                _parse_role_path(args.authority),
                args.output,
            )
        elif args.verify_bundle:
            _require(args.archive is not None, "bundle verification requires --archive")
            if args.evidence_tree or args.authority:
                _require(args.evidence_tree and args.authority, "input-bound bundle verification requires both --evidence-tree and --authority")
                value = verify_bundle_inputs(
                    args.archive,
                    _parse_role_path(args.evidence_tree),
                    _parse_role_path(args.authority),
                )
            else:
                value = verify_bundle(args.archive)
        elif args.create_asset_manifest:
            _require(args.output is not None, "asset manifest creation requires --output")
            value = create_asset_manifest(_parse_role_path(args.asset), args.output)
        elif args.verify_asset_sources:
            _require(args.manifest is not None and args.asset, "asset source verification requires --manifest and --asset")
            value = verify_asset_sources(
                args.manifest, _parse_role_path(args.asset)
            )
        else:
            _require(args.manifest is not None and args.directory is not None and args.expected_manifest_id is not None, "asset-directory verification requires --manifest, --directory, and --expected-manifest-id")
            value = verify_asset_directory(args.manifest, args.directory, args.expected_manifest_id)
        identity = value.get("bundle_id", value.get("manifest_id"))
        print(json.dumps({"state": "PASS", "id": identity}, sort_keys=True))
        return 0
    except (OSError, ReleaseAssetError, ValueError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"v1.1.0 release assets: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
