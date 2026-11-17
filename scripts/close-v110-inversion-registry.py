#!/usr/bin/env python3
"""Seal the evidence-backed closure of the v1.1.0 inversion registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
from typing import Any


SCHEMA = "gpmeep-v110-closed-inversion-registry-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ClosureError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


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


def _direct_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_dir(), f"{label} is not a direct directory")
    return resolved


def _direct_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_file(), f"{label} is not a direct regular file")
    return resolved


def _load(path: pathlib.Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _require(key not in value, f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ClosureError(f"non-finite JSON constant: {item}")
        ),
    )
    _require(isinstance(value, dict), "closure terminal must be a JSON object")
    return value


def _validator() -> Any:
    path = pathlib.Path(__file__).resolve().parent / "validate-v110-inversion-registry.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_v110_closure_normative_registry", path
    )
    _require(spec is not None and spec.loader is not None, "normative registry validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def derive(
    records_directory: pathlib.Path, evidence_root: pathlib.Path
) -> dict[str, Any]:
    root = _direct_directory(evidence_root, "closure evidence root")
    records = _direct_directory(records_directory, "closure records directory")
    _require(records.is_relative_to(root), "closure records directory escapes evidence root")
    try:
        relative_directory = records.relative_to(root).as_posix()
    except ValueError as error:
        raise ClosureError("closure records directory is not portable") from error
    _require(relative_directory not in {"", "."}, "closure records need a dedicated directory")
    validator = _validator()
    result = validator.validate_registry(
        require_closed=True,
        evidence_root=root,
        records_directory=records,
    )
    expected = {
        "schema_version": 1,
        "state": "PASS",
        "observed_count": 13,
        "registered_count": 13,
        "missing_count": 0,
        "blocking_count": 0,
        "closed_count": 13,
        "open_count": 0,
        "missing_workloads": [],
        "blocking_workloads": [],
        "open_workloads": [],
    }
    _require(
        all(result.get(key) == value for key, value in expected.items()),
        "normative closed registry result differs",
    )
    file_records = []
    for path in sorted(records.iterdir()):
        path = _direct_file(path, "closed inversion record")
        _require(path.suffix == ".json", "closed inversion record suffix differs")
        file_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _require(len(file_records) == 13, "closed inversion record census differs")
    terminal: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "PASS",
        "records_directory": relative_directory,
        "records": file_records,
        "registry_result": result,
    }
    terminal["terminal_id"] = _canonical_sha256(terminal)
    return terminal


def verify(
    terminal_path: pathlib.Path, evidence_root: pathlib.Path
) -> dict[str, Any]:
    path = _direct_file(terminal_path, "closed inversion terminal")
    retained = _load(path)
    _require(
        set(retained)
        == {
            "schema",
            "state",
            "records_directory",
            "records",
            "registry_result",
            "terminal_id",
        },
        "closure terminal schema differs",
    )
    _require(retained["schema"] == SCHEMA and retained["state"] == "PASS", "closure terminal is not PASS")
    unsigned = dict(retained)
    terminal_id = unsigned.pop("terminal_id")
    _require(
        isinstance(terminal_id, str)
        and HEX64.fullmatch(terminal_id) is not None
        and _canonical_sha256(unsigned) == terminal_id,
        "closure terminal identity differs",
    )
    relative = pathlib.PurePosixPath(retained["records_directory"])
    _require(
        not relative.is_absolute()
        and relative.parts
        and ".." not in relative.parts
        and str(relative) == retained["records_directory"],
        "closure records directory path is unsafe",
    )
    root = _direct_directory(evidence_root, "closure evidence root")
    expected = derive(root.joinpath(*relative.parts), root)
    _require(retained == expected, "closure terminal was not exactly re-derived")
    return retained


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "closure terminal parent is absent")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, "closure terminal parent contains a symlink")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite closure terminal")
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
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
            "closure terminal mode differs",
        )
        _require(_load(destination) == value, "closure terminal bytes differ")
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
                directory_descriptor = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                # Preserve the original publication error.  The caller still
                # fails closed, and the destination unlink has been attempted.
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", action="store_true")
    parser.add_argument("--records-directory", type=pathlib.Path)
    parser.add_argument("--evidence-root", type=pathlib.Path, required=True)
    parser.add_argument("--terminal", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        if args.create:
            _require(args.records_directory is not None and args.output is not None and args.terminal is None, "closure create arguments differ")
            value = derive(args.records_directory, args.evidence_root)
            _write_new(args.output, value)
        else:
            _require(args.terminal is not None and args.records_directory is None and args.output is None, "closure verify arguments differ")
            value = verify(args.terminal, args.evidence_root)
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "terminal_id": value["terminal_id"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, ClosureError, json.JSONDecodeError) as error:
        print(f"v1.1.0 inversion closure: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
