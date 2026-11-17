#!/usr/bin/env python3
"""Seal/replay the exact measured-source identity used by the v1.1.0 M4 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import tempfile
from typing import Any


SCHEMA = "gpmeep-v110-release-source-transition-v3"
POLICY_SCHEMA = "gpmeep-v110-external-source-policy-v2"
VALIDATED_ENGINE_COMMIT = "245917cfcd7311d859ade299ed756ea2e4d3d313"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TransitionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransitionError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            TransitionError(f"non-finite JSON constant: {value}")
        ),
    )
    _require(isinstance(value, dict), "transition report root must be an object")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == keys, f"{label} schema differs")
    return value


def _run(repo: pathlib.Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    if check and result.returncode != 0:
        raise TransitionError(f"git command failed: {' '.join(arguments)}")
    return result.stdout.rstrip("\n")


def _git_entry(repo: pathlib.Path, commit: str, path: str) -> dict[str, str] | None:
    value = _run(repo, "ls-tree", commit, "--", path, check=False)
    if not value:
        return None
    fields = value.split(None, 3)
    _require(len(fields) == 4 and fields[3] == path, f"Git tree entry differs: {path}")
    mode, kind, blob, _name = fields
    _require(kind == "blob" and mode in {"100644", "100755"}, f"Git object type/mode is unsafe: {path}")
    _require(HEX40.fullmatch(blob) is not None, f"Git blob identity is invalid: {path}")
    return {"mode": mode, "blob": blob}


def _policy(
    path: pathlib.Path,
    *,
    repo: pathlib.Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), "external transition policy cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_file(), "external transition policy is unsafe")
    _require(not resolved.is_relative_to(repo), "transition policy must be external to the release repository")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    _require(mode & 0o022 == 0, "external transition policy is group/world writable")
    identity = (resolved.stat().st_dev, resolved.stat().st_ino, resolved.stat().st_size, resolved.stat().st_mtime_ns)
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(HEX64.fullmatch(expected_sha256 or "") is not None, "exact external policy SHA-256 is required")
    _require(digest == expected_sha256, "external transition policy SHA-256 differs")
    value = json.loads(payload, object_pairs_hook=_unique)
    _require(isinstance(value, dict), "external transition policy root must be an object")
    _require(
        set(value)
        == {
            "schema", "decision", "validated_engine_commit", "measured_source_commit", "release_source_commit",
            "verifier", "approved_changes", "policy_id",
        },
        "external transition policy schema differs",
    )
    unsigned = dict(value)
    policy_id = unsigned.pop("policy_id", None)
    _require(value["schema"] == POLICY_SCHEMA and value["decision"] == "APPROVE_EXACT_TRANSITION", "external transition decision differs")
    _require(policy_id == _canonical_sha256(unsigned), "external transition policy ID differs")
    _require(
        identity
        == (resolved.stat().st_dev, resolved.stat().st_ino, resolved.stat().st_size, resolved.stat().st_mtime_ns)
        and resolved.read_bytes() == payload,
        "external transition policy changed during verification",
    )
    return value, {
        "path": str(resolved),
        "mode": f"{mode:04o}",
        "size_bytes": len(payload),
        "sha256": digest,
        "policy_id": policy_id,
    }


def derive(
    repo: pathlib.Path,
    release_source_commit: str,
    authority_policy_path: pathlib.Path,
    authority_policy_sha256: str,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    _require((repo / ".git").exists(), "source repository has no Git metadata")
    _require(HEX40.fullmatch(release_source_commit) is not None, "release source commit is invalid")
    head = _run(repo, "rev-parse", "HEAD")
    _require(head == release_source_commit, "release source is not repository HEAD")
    _require(_run(repo, "status", "--porcelain", "--untracked-files=all") == "", "release source worktree is not clean")
    policy, policy_record = _policy(
        authority_policy_path,
        repo=repo,
        expected_sha256=authority_policy_sha256,
    )
    measured_source_commit = policy["measured_source_commit"]
    _require(HEX40.fullmatch(measured_source_commit or "") is not None, "external policy measured source commit is invalid")
    _require(policy["validated_engine_commit"] == VALIDATED_ENGINE_COMMIT, "external policy engine commit differs")
    _require(policy["release_source_commit"] == release_source_commit, "external policy release commit differs")
    _require(
        measured_source_commit == release_source_commit,
        "release source commit must exactly equal the M4 measured source commit",
    )
    ancestor = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "merge-base", "--is-ancestor", VALIDATED_ENGINE_COMMIT, measured_source_commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    _require(ancestor.returncode == 0, "validated engine commit is not an ancestor of measured source")
    measured_ancestor = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "merge-base", "--is-ancestor", measured_source_commit, release_source_commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    _require(measured_ancestor.returncode == 0, "measured source commit is not an ancestor of release source")
    changed_raw = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "diff", "--no-renames", "--name-only", "-z", measured_source_commit, release_source_commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    _require(changed_raw.returncode == 0, "source transition diff failed")
    try:
        paths = changed_raw.stdout.decode("utf-8", errors="strict").split("\0")
    except UnicodeDecodeError as error:
        raise TransitionError("source transition contains a non-UTF-8 path") from error
    paths = [path for path in paths if path]
    _require(paths == sorted(set(paths)), "source transition path inventory differs")
    _require(paths == [], "release source tree differs from the M4 measured source")
    changed = [
        {
            "path": path,
            "measured_source": _git_entry(repo, measured_source_commit, path),
            "release_source": _git_entry(repo, release_source_commit, path),
        }
        for path in paths
    ]
    _require(
        policy["approved_changes"] == changed == [],
        "exact-source release policy must contain no approved changes",
    )
    verifier_entry = _git_entry(
        repo, measured_source_commit, "scripts/verify-v110-release-source-transition.py"
    )
    _require(
        policy["verifier"]
        == {
            "path": "scripts/verify-v110-release-source-transition.py",
            **(verifier_entry or {}),
        },
        "external policy does not pin the verifier Git blob and mode",
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "PASS",
        "validated_engine_commit": VALIDATED_ENGINE_COMMIT,
        "measured_source_commit": measured_source_commit,
        "release_source_commit": release_source_commit,
        "validated_engine_tree": _run(repo, "rev-parse", f"{VALIDATED_ENGINE_COMMIT}^{{tree}}"),
        "measured_source_tree": _run(repo, "rev-parse", f"{measured_source_commit}^{{tree}}"),
        "release_source_tree": _run(repo, "rev-parse", f"{release_source_commit}^{{tree}}"),
        "validated_engine_merge_base": _run(repo, "merge-base", VALIDATED_ENGINE_COMMIT, measured_source_commit),
        "measured_source_merge_base": _run(repo, "merge-base", measured_source_commit, release_source_commit),
        "ancestor": True,
        "clean_head": True,
        "external_policy": policy_record,
        "changed_files": changed,
        "changed_file_count": len(changed),
        "protected_changes": [],
    }
    report["report_id"] = _canonical_sha256(report)
    return report


def verify(
    path: pathlib.Path,
    repo: pathlib.Path,
    *,
    authority_policy_path: pathlib.Path,
    authority_policy_sha256: str,
) -> dict[str, Any]:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), "transition report cannot be a symlink")
    path = lexical.resolve(strict=True)
    _require(path == lexical and path.is_file(), "transition report is not a regular direct path")
    retained = _load(path)
    _require(retained.get("schema") == SCHEMA and retained.get("state") == "PASS", "transition report identity differs")
    unsigned = dict(retained)
    report_id = unsigned.pop("report_id", None)
    _require(report_id == _canonical_sha256(unsigned), "transition report ID differs")
    retained_policy = _exact(
        retained.get("external_policy"),
        {"path", "mode", "size_bytes", "sha256", "policy_id"},
        "retained external policy",
    )
    _require(
        HEX64.fullmatch(authority_policy_sha256 or "") is not None,
        "an independently supplied external policy SHA-256 is required",
    )
    _require(str(authority_policy_path.absolute()) == retained_policy["path"], "external policy path differs")
    _require(authority_policy_sha256 == retained_policy["sha256"], "external policy expected hash differs")
    expected = derive(
        repo,
        retained.get("release_source_commit"),
        authority_policy_path,
        authority_policy_sha256,
    )
    _require(retained == expected, "transition report was not exactly re-derived")
    return retained


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "transition report parent is absent")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent, "transition report parent contains a symlink")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite transition report")
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
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, "transition report mode differs")
        _require(destination.read_bytes() == payload, "transition report bytes differ")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--release-source-commit")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--authority-policy", type=pathlib.Path)
    parser.add_argument("--authority-policy-sha256")
    args = parser.parse_args(argv)
    try:
        if args.create:
            _require(
                args.release_source_commit is not None
                and args.output is not None
                and args.authority_policy is not None
                and args.authority_policy_sha256 is not None,
                "create arguments are incomplete",
            )
            result = derive(
                args.repo,
                args.release_source_commit,
                args.authority_policy,
                args.authority_policy_sha256,
            )
            _write_new(args.output, result)
        else:
            _require(args.release_source_commit is None and args.output is None, "verify arguments differ")
            _require(
                args.authority_policy is not None
                and args.authority_policy_sha256 is not None,
                "verify requires an externally supplied policy path and SHA-256",
            )
            result = verify(
                args.verify,
                args.repo,
                authority_policy_path=args.authority_policy,
                authority_policy_sha256=args.authority_policy_sha256,
            )
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "report_id": result["report_id"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, TransitionError) as error:
        print(f"v1.1.0 release source transition: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
