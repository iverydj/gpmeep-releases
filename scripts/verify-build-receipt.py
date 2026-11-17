#!/usr/bin/env python3
"""Verify a gpmeep build receipt and emit its stable identity."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpmeep_provenance import ProvenanceError, verify_build_receipt  # noqa: E402


def _valid_source_bytecode_records(
    records: object,
    *,
    ownership: str,
    python_version: object,
) -> bool:
    if not isinstance(python_version, str) or re.fullmatch(
        r"[0-9]+\.[0-9]+", python_version
    ) is None:
        return False
    if ownership == "relocated":
        exact_keys = {
            "archive_sha256",
            "current_sha256",
            "current_state",
            "derived_sha256",
            "derived_size_bytes",
            "installed_source_hash_hex",
            "installed_source_sha256",
            "path",
            "source_path",
        }
        expected_state = "relocated-source-compiled"
    elif ownership == "generated":
        exact_keys = {
            "current_sha256",
            "current_state",
            "derived_sha256",
            "derived_size_bytes",
            "installed_source_hash_hex",
            "installed_source_sha256",
            "path",
            "source_path",
        }
        expected_state = "source-compiled"
    else:  # pragma: no cover - internal programming invariant
        return False
    if not isinstance(records, list) or not records:
        return False
    cache_tag = python_version.replace(".", "")
    target_pattern = re.compile(
        rf"(.+)/__pycache__/([^/]+)\.cpython-{re.escape(cache_tag)}"
        r"(?:\.opt-[0-9]+)?\.pyc"
    )
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != exact_keys:
            return False
        path = record["path"]
        source_path = record["source_path"]
        if not isinstance(path, str) or not isinstance(source_path, str):
            return False
        target = pathlib.PurePosixPath(path)
        source = pathlib.PurePosixPath(source_path)
        match = target_pattern.fullmatch(path)
        if (
            target.is_absolute()
            or source.is_absolute()
            or not target.parts
            or not source.parts
            or any(part in {".", ".."} for part in (*target.parts, *source.parts))
            or "\\" in path
            or "\\" in source_path
            or target.as_posix() != path
            or source.as_posix() != source_path
            or match is None
            or source_path != f"{match.group(1)}/{match.group(2)}.py"
            or record["current_state"] != expected_state
            or re.fullmatch(r"[0-9a-f]{64}", str(record["current_sha256"]))
            is None
            or record["current_sha256"] != record["derived_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(record["derived_sha256"]))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(record["installed_source_sha256"])
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{16}", str(record["installed_source_hash_hex"])
            )
            is None
            or not isinstance(record["derived_size_bytes"], int)
            or isinstance(record["derived_size_bytes"], bool)
            or record["derived_size_bytes"] < 16
            or (
                ownership == "relocated"
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(record["archive_sha256"])
                )
                is None
            )
        ):
            return False
        paths.append(path)
    return paths == sorted(paths) and len(paths) == len(set(paths))


def verify_conda_prefix_against_archives(
    receipt: dict[str, object], repo: pathlib.Path
) -> None:
    if receipt.get("build_kind") not in {
        "cpu-mpi-python-fp64",
        "cuda-mpi-python-fp32",
    }:
        return
    configuration_files = receipt.get("configuration_files")
    if not isinstance(configuration_files, dict):
        raise RuntimeError("build receipt conda-prefix audit record is absent")
    record = configuration_files.get("conda_prefix_content_audit")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError("build receipt conda-prefix audit record is absent")
    audit_path = pathlib.Path(record["path"])
    if not audit_path.is_absolute():
        audit_path = repo / audit_path
    try:
        retained = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("build receipt conda-prefix audit is unreadable") from error
    relocated_bytecode = (
        retained.get("relocated_source_bytecode")
        if isinstance(retained, dict)
        else None
    )
    generated_bytecode = (
        retained.get("generated_source_bytecode")
        if isinstance(retained, dict)
        else None
    )
    python_version = retained.get("python_version") if isinstance(retained, dict) else None
    relocated_paths = (
        [item.get("path") for item in relocated_bytecode]
        if _valid_source_bytecode_records(
            relocated_bytecode,
            ownership="relocated",
            python_version=python_version,
        )
        else None
    )
    generated_paths = (
        [item.get("path") for item in generated_bytecode]
        if _valid_source_bytecode_records(
            generated_bytecode,
            ownership="generated",
            python_version=python_version,
        )
        else None
    )
    if (
        not isinstance(retained, dict)
        or retained.get("schema_version") != 4
        or retained.get("pass") is not True
        or not isinstance(retained.get("environment_prefix"), str)
        or not isinstance(retained.get("lock"), dict)
        or not isinstance(retained["lock"].get("path"), str)
        or not isinstance(retained.get("package_cache"), dict)
        or not isinstance(retained["package_cache"].get("path"), str)
        or retained.get("archive_snapshot")
        != {
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "hash_and_parse_same_private_snapshot": True,
            "descriptor_retained_until_terminal_verification": True,
            "pathname_fingerprint_rechecked": True,
        }
        or relocated_paths is None
        or retained.get("relocated_source_bytecode_count")
        != len(relocated_bytecode)
        or generated_paths is None
        or retained.get("generated_source_bytecode_count")
        != len(generated_bytecode)
        or retained.get("declared_generated_bytecode_path_count")
        != len(generated_bytecode)
        or bool(set(relocated_paths) & set(generated_paths))
    ):
        raise RuntimeError("build receipt conda-prefix audit is invalid")
    with tempfile.TemporaryDirectory(
        prefix="gpmeep-receipt-prefix-audit-", dir="/tmp"
    ) as temporary:
        output = pathlib.Path(temporary) / "audit.json"
        command = [
            "/usr/bin/python3",
            "-S",
            "-P",
            "-B",
            str(repo / "scripts" / "audit-conda-prefix.py"),
            "--prefix",
            retained["environment_prefix"],
            "--lock",
            retained["lock"]["path"],
            "--package-cache",
            retained["package_cache"]["path"],
            "--output",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repo,
                env={
                    "HOME": "/tmp",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "cannot reproduce the build receipt conda-prefix audit"
            ) from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "current exact-archive conda-prefix audit failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            reproduced = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("reproduced conda-prefix audit is invalid") from error
    if reproduced != retained:
        raise RuntimeError(
            "build receipt conda-prefix audit differs from the current exact archives"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    parser.add_argument("--expected-build-kind")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = verify_build_receipt(args.receipt, args.repo, verify_source=True)
        if (
            args.expected_build_kind is not None
            and value.get("build_kind") != args.expected_build_kind
        ):
            raise ProvenanceError(
                "build kind mismatch: expected "
                f"{args.expected_build_kind!r}, observed {value.get('build_kind')!r}"
            )
        verify_conda_prefix_against_archives(value, args.repo.resolve())
    except (OSError, ProvenanceError, RuntimeError, ValueError) as exc:
        print(f"gpmeep build receipt verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": "gpmeep-build-receipt-verification-v1",
                "build_kind": value["build_kind"],
                "build_input_id": value["build_input_id"],
                "artifact_set_id": value["artifact_set_id"],
                "receipt_id": value["receipt_id"],
                "source_sha256": value["source_end"]["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
