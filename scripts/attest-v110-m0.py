#!/usr/bin/env python3
"""Create or verify the external, checksum-bound gpmeep v1.1.0 M0 attestation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONTRACT_PATH = ROOT / "packaging/v110-development-contract.json"
VALIDATOR_PATH = ROOT / "scripts/validate-v110-development-contract.py"
MAMBA = ROOT / ".tools/micromamba"
EXPECTED_BASE = "b48c9b678f2316cefa95e48a661f7a877547a000"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_ATTESTATION_BYTES = 16 * 1024 * 1024


class AttestationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestationError(message)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise AttestationError(f"non-finite JSON constant: {value}")


def _load_json(raw: bytes | str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: pathlib.Path) -> dict[str, Any]:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"attested file must not be a symlink: {lexical}")
    resolved = lexical.resolve(strict=True)
    _require(resolved.is_file(), f"attested path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "mode": stat.S_IMODE(resolved.stat().st_mode),
        "sha256": _sha256(resolved),
    }


def _run(*arguments: str) -> str:
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
        },
    )
    return result.stdout.strip()


def _git(*arguments: str) -> str:
    return _run("/usr/bin/git", "-C", str(ROOT), *arguments)


def _load_contract_validator():
    spec = importlib.util.spec_from_file_location("v110_contract_attestation", VALIDATOR_PATH)
    _require(spec is not None and spec.loader is not None, "cannot load development-contract validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package_lines(text: str) -> list[str]:
    lines = sorted({line.strip() for line in text.splitlines() if "://" in line})
    _require(lines, "explicit package set is empty")
    for line in lines:
        _require("#" in line and len(line.rsplit("#", 1)[1]) == 64, "package URL lacks SHA-256")
    return lines


def _set_digest(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _environment_record(
    contract: dict[str, Any],
    prefix_key: str,
    lock_key: str,
    lock_sha_key: str,
) -> dict[str, Any]:
    isolation = contract["development_isolation"]
    prefix = (ROOT / isolation[prefix_key]).resolve(strict=True)
    _require(prefix.is_dir() and prefix.is_relative_to(ROOT), f"{prefix_key} escaped repository")
    history = prefix / "conda-meta/history"
    _require(history.is_file(), f"environment history is absent: {prefix}")
    lock = (ROOT / isolation[lock_key]).resolve(strict=True)
    lock_record = _record(lock)
    _require(lock_record["sha256"] == isolation[lock_sha_key], f"{lock_key} hash differs")
    locked = _package_lines(lock.read_text(encoding="utf-8"))
    installed_text = _run(
        str(MAMBA),
        "--no-rc",
        "list",
        "--root-prefix",
        str(ROOT / ".micromamba"),
        "--prefix",
        str(prefix),
        "--explicit",
        "--sha256",
    )
    installed = _package_lines(installed_text)
    _require(installed == locked, f"environment differs from exact lock: {prefix}")
    return {
        "prefix": str(prefix),
        "history": _record(history),
        "lock": lock_record,
        "package_count": len(locked),
        "exact_url_sha256_set_digest": _set_digest(locked),
        "exact_lock_match": True,
    }


def _repository_record(contract: dict[str, Any], m0_commit: str | None) -> dict[str, Any]:
    isolation = contract["development_isolation"]
    _require(ROOT.name == isolation["repository_relative_to_workspace"], "repository directory differs")
    _require((ROOT / ".git").is_dir(), "repository does not have an independent .git directory")
    _require(not (ROOT / ".git").is_symlink(), "repository .git directory is a symlink")
    git_dir = pathlib.Path(_git("rev-parse", "--absolute-git-dir")).resolve(strict=True)
    common_dir = pathlib.Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir")).resolve(strict=True)
    expected_git_dir = (ROOT / ".git").resolve(strict=True)
    _require(git_dir == expected_git_dir and common_dir == expected_git_dir, "Git object store is shared or redirected")
    alternates = git_dir / "objects/info/alternates"
    _require(not alternates.exists(), "Git alternates file is forbidden")
    _require(not os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES"), "Git alternate-object environment is forbidden")
    _require(_git("status", "--porcelain", "--untracked-files=all") == "", "source worktree is not clean")
    _require(_git("branch", "--show-current") == isolation["required_branch"], "development branch differs")
    _require(_git("remote", "get-url", "origin") == isolation["origin_url"], "origin URL differs")
    _require(_git("rev-parse", "--is-shallow-repository") == "false", "shallow repository is forbidden")
    head = _git("rev-parse", "HEAD")
    candidate = m0_commit or head
    _require(HEX40.fullmatch(candidate) is not None, "M0 commit must be a full lowercase SHA-1")
    _require(_git("cat-file", "-t", candidate) == "commit", "M0 object is not a commit")
    _require(_git("rev-parse", f"{candidate}^") == EXPECTED_BASE, "M0 first parent is not frozen v1.0.3")
    _require(_git("merge-base", candidate, EXPECTED_BASE) == EXPECTED_BASE, "M0 merge base differs")
    _require(_git("merge-base", "--is-ancestor", candidate, head) == "", "M0 is not an ancestor of current HEAD")
    _require(_git("rev-parse", "v1.0.3^{}") == EXPECTED_BASE, "v1.0.3 tag target differs")
    return {
        "path": str(ROOT),
        "workspace": str(WORKSPACE),
        "branch": isolation["required_branch"],
        "current_head": head,
        "m0_commit": candidate,
        "m0_tree": _git("rev-parse", f"{candidate}^{{tree}}"),
        "m0_first_parent": EXPECTED_BASE,
        "merge_base": EXPECTED_BASE,
        "v103_tag_object": _git("rev-parse", "v1.0.3"),
        "v103_tag_target": EXPECTED_BASE,
        "origin": isolation["origin_url"],
        "git_dir": str(git_dir),
        "common_dir": str(common_dir),
        "independent_git_object_store": True,
        "git_alternates_absent": True,
        "shallow": False,
        "clean": True,
    }


def collect_attestation(
    *,
    evidence_paths: list[pathlib.Path],
    m0_commit: str | None = None,
) -> dict[str, Any]:
    validator = _load_contract_validator()
    contract = validator._load_json(CONTRACT_PATH.read_text(encoding="utf-8"))
    validator.validate_contract_data(contract)
    validator.validate_bound_files(contract)
    user_inputs = validator.validate_user_inputs(contract, WORKSPACE)
    evidence: list[dict[str, Any]] = []
    for path in sorted(evidence_paths, key=lambda item: str(item)):
        record = _record(path)
        resolved = pathlib.Path(record["path"])
        _require(not resolved.is_relative_to(ROOT), "M0 evidence must remain outside repository")
        _require(record["mode"] == 0o600, f"external evidence must have mode 0600: {resolved}")
        evidence.append(record)
    _require(evidence, "at least one external evidence file is required")
    expected_campaign_hashes = {
        item["sha256"]
        for item in _load_contract_validator()
        ._load_json((ROOT / contract["performance_inversion_policy"]["registry"]["initial_inventory"]).read_bytes())["source_campaign"]
        .values()
        if isinstance(item, dict) and set(item) == {"workspace_relative_path", "sha256"}
    }
    attested_hashes = {item["sha256"] for item in evidence}
    _require(
        expected_campaign_hashes <= attested_hashes,
        "external evidence omits the checksum-bound historical campaign reports",
    )
    contract_validation = validator.validate(require_user_inputs=True)
    return {
        "schema_version": 1,
        "state": "PASS",
        "release_train": "1.1.0",
        "repository": _repository_record(contract, m0_commit),
        "contract": {
            "file": _record(CONTRACT_PATH),
            "canonical_sha256": contract_validation["contract_canonical_sha256"],
            "validator": _record(VALIDATOR_PATH),
            "inversion_schema": _record(ROOT / contract["performance_inversion_policy"]["registry"]["schema"]),
            "inversion_inventory": _record(ROOT / contract["performance_inversion_policy"]["registry"]["initial_inventory"]),
        },
        "micromamba": {
            "file": _record(MAMBA),
            "version": _run(str(MAMBA), "--version"),
        },
        "environments": {
            "builder": _environment_record(contract, "builder_prefix_relative", "builder_lock", "builder_lock_sha256"),
            "test": _environment_record(contract, "test_prefix_relative", "test_lock", "test_lock_sha256"),
        },
        "user_inputs": user_inputs,
        "external_evidence": evidence,
    }


def _write_atomic_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.exists(), "attestation report parent must already exist")
    resolved_parent = destination.parent.resolve(strict=True)
    _require(destination.parent == resolved_parent, "attestation report parent cannot contain symlinks")
    destination = resolved_parent / destination.name
    _require(not destination.is_relative_to(ROOT), "attestation report must be outside repository")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite attestation")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _verify(path: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    _require(HEX64.fullmatch(expected_sha256) is not None, "expected attestation SHA-256 is invalid")
    lexical = path.absolute()
    _require(not lexical.is_symlink(), "attestation report cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, "attestation report path cannot contain symlinks")
    _require(not resolved.is_relative_to(ROOT), "attestation report must be external")
    _require(resolved.is_file(), "attestation report must be a regular file")
    _require(stat.S_IMODE(resolved.stat().st_mode) == 0o600, "attestation report mode must be 0600")
    _require(resolved.stat().st_size <= MAX_ATTESTATION_BYTES, "attestation report is too large")
    _require(_sha256(resolved) == expected_sha256, "attestation report SHA-256 differs")
    expected = _load_json(resolved.read_bytes())
    _require(isinstance(expected, dict), "attestation report root must be an object")
    _require(expected.get("schema_version") == 1 and expected.get("state") == "PASS", "attestation identity is invalid")
    evidence_paths = [pathlib.Path(item["path"]) for item in expected["external_evidence"]]
    current = collect_attestation(
        evidence_paths=evidence_paths,
        m0_commit=expected["repository"]["m0_commit"],
    )
    # HEAD may advance after M0 while the attested M0 commit remains its ancestor.
    expected = json.loads(json.dumps(expected))
    expected["repository"]["current_head"] = current["repository"]["current_head"]
    _require(current == expected, "attestation is stale or an attested byte/state changed")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", type=pathlib.Path)
    mode.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--evidence", action="append", type=pathlib.Path, default=[])
    parser.add_argument(
        "--expected-sha256",
        help="exact outer SHA-256 required by --verify",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.report is not None:
            _require(args.evidence, "--report requires at least one --evidence")
            _require(args.expected_sha256 is None, "--expected-sha256 is valid only with --verify")
            result = collect_attestation(evidence_paths=args.evidence)
            _write_atomic_new(args.report, result)
            report_path = args.report.absolute()
        else:
            _require(not args.evidence, "--evidence is valid only with --report")
            _require(args.expected_sha256 is not None, "--verify requires --expected-sha256")
            result = _verify(args.verify, args.expected_sha256)
            report_path = args.verify.absolute()
    except (AttestationError, OSError, ValueError, subprocess.CalledProcessError) as error:
        if args.json:
            print(json.dumps({"schema_version": 1, "state": "FAIL", "error": str(error)}, sort_keys=True))
        else:
            print(f"v1.1.0 M0 attestation: FAIL: {error}", file=sys.stderr)
        return 2
    output = {
        "schema_version": 1,
        "state": "PASS",
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "m0_commit": result["repository"]["m0_commit"],
    }
    print(json.dumps(output, sort_keys=True) if args.json else f"v1.1.0 M0 attestation: PASS ({output['m0_commit']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
