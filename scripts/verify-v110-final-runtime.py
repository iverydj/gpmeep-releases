#!/usr/bin/env python3
"""Verify the contracted Python environment before final evidence replay."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

_spec = importlib.util.spec_from_file_location("gpmeep_runtime_source", Path(__file__).with_name("verify-public-source.py"))
SOURCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SOURCE)
require = SOURCE.require
digest = SOURCE.digest

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def file_bytes(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "runtime file must be absolute without symlinks")
    data = path.read_bytes()
    require(digest(data) == expected, "runtime file SHA-256 differs")
    return data

def file_record(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "runtime Python pin fields differ")
    return file_bytes(value["path"], value["sha256"])

def directory(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(), "runtime directory must be absolute without symlinks")
    return path

def verify_final_runtime(source_repo, source_commit, python_pin):
    """Use the contracted test interpreter and M0-level exact package lock."""
    require(isinstance(source_commit, str) and SOURCE.HEX40.fullmatch(source_commit), "invalid runtime source commit")
    file_record(python_pin)
    contract = SOURCE.read_json(SOURCE.git(source_repo, "show", f"{source_commit}:packaging/v110-development-contract.json"))
    isolation = contract["development_isolation"]
    prefix = directory(str(source_repo / isolation["test_prefix_relative"]))
    expected_python = prefix / "bin/python3.11"
    require(Path(python_pin["path"]) == expected_python and Path(sys.executable).resolve(strict=True) == expected_python,
            "run publication with the pinned contracted test Python")
    lock_bytes = SOURCE.git(source_repo, "show", f"{source_commit}:{isolation['test_lock']}")
    require(digest(lock_bytes) == isolation["test_lock_sha256"], "committed test environment lock differs")
    mamba = source_repo / ".tools/micromamba"
    file_bytes(mamba, "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82")
    result = subprocess.run([str(mamba), "--no-rc", "list", "--root-prefix", str(source_repo / ".micromamba"),
                             "--prefix", str(prefix), "--explicit", "--sha256"], capture_output=True,
                            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}, timeout=120)
    require(result.returncode == 0, "test environment package inventory failed")
    def packages(data):
        return sorted(line.strip() for line in data.decode().splitlines() if "://" in line)
    require(packages(lock_bytes) and packages(result.stdout) == packages(lock_bytes),
            "test environment differs from the exact committed lock")
    return {"python": python_pin, "lock_sha256": digest(lock_bytes),
            "package_set_sha256": digest(canonical(packages(lock_bytes)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify_final_runtime(directory(str(args.repo)), args.source_commit,
            {"path": str(Path(sys.executable).resolve(strict=True)), "sha256": args.python_sha256})
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"final runtime verification failed: {exc}\n")

if __name__ == "__main__":
    main()
