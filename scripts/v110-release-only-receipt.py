#!/usr/bin/env python3
"""Construct and verify the history-confidential v1.1.0 release mirror.

This module deliberately uses Git objects rather than a checkout.  The source
tree is attached to a new parentless commit in an alternate-backed staging
repository, moved through a bundle, and cloned into a repository which has no
access to the development object store.  Only that isolated repository is
eligible to become a publisher.

The JSON formats in this file are strict, canonical and intentionally small.
The organization snapshot is a distribution identity; ``source_commit``
continues to identify the private commit which was measured and built.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "gpmeep-v110-release-only-mirror-v1"
SNAPSHOT_SCHEMA = "gpmeep-v110-release-only-snapshot-v1"
SOURCE_AUTHORITY_SCHEMA = "gpmeep-v110-release-only-source-authority-v1"
TARGET_PREFLIGHT_SCHEMA = "gpmeep-v110-release-only-target-preflight-v1"
ASSET_SCHEMA = "gpmeep-v110-release-only-assets-v1"
TAG_ATTESTATION_SCHEMA = "gpmeep-v110-release-only-tag-v1"
TARGET_HISTORY_SCHEMA = "gpmeep-v110-release-only-target-history-attestation-v2"
EXTERNAL_AUDIT_SCHEMA = "gpmeep-v110-release-only-independent-audit-v2"
PRIOR_MIGRATION_RECEIPT_SCHEMA = "gpmeep-v103-release-only-migration-receipt-v2"
PRIOR_MIGRATION_MANIFEST_SCHEMA = "gpmeep-v103-release-only-migration-assets-v2"
PRIOR_MIGRATION_AUDIT_SCHEMA = "gpmeep-v103-release-only-independent-audit-v1"
PRIOR_MIGRATION_RUN_INTENT_SCHEMA = "gpmeep-v103-release-only-migration-run-intent-v1"
PRIOR_MIGRATION_TERMINAL_SCHEMA = "gpmeep-v103-release-only-migration-terminal-v1"
PRIOR_MIGRATION_INCIDENT_SCHEMA = "gpmeep-v103-release-only-migration-incident-v1"
RUN_INTENT_SCHEMA = "gpmeep-v110-release-only-run-intent-v2"
PUBLICATION_TERMINAL_SCHEMA = "gpmeep-v110-release-only-publication-terminal-v2"
METHOD = "parentless-identical-git-tree-v1"
TERMINAL_STATE = "MIRROR_AUTHORIZED"
RELEASE_TRAIN = "1.1.0"
DEFAULT_TAG = "v1.1.0"
TARGET_REPOSITORY = "example-organization/gpmeep-releases"
V103_SOURCE_REPOSITORY = "iverydj/gpmeep"
V103_SOURCE_REPOSITORY_ID = 1346912823
V103_SOURCE_REPOSITORY_NODE_ID = "R_kgDOUEhCNw"
V103_SOURCE_RELEASE_ID = 378328014
V103_SOURCE_RELEASE_NODE_ID = "RE_kwDOUEhCN84WjNPO"
V103_LEGACY_TARGET_REPOSITORY_ID = 1900000001
V103_LEGACY_TARGET_REPOSITORY_NODE_ID = "R_publicfixture_1900000001"
V103_SOURCE_COMMIT = "b48c9b678f2316cefa95e48a661f7a877547a000"
V103_SOURCE_TREE = "1c85bcd9925dfd28ad8d763a1caeb6532de79752"
V103_SOURCE_TAG_OBJECT = "256bb9bf50f23833f805a1e0e6bb638c551a3373"
V103_SOURCE_TAG_PAYLOAD_SHA256 = "4a14b7af7341313a452aebee81d1cc37e90e5e3fc0c24aee1f9f42eb7a5610b2"
V103_SNAPSHOT_COMMIT = "121a5a87059f8f830420f2e5ec09003c383599fc"
V103_TARGET_TAG_OBJECT = "cf9e47e4ce35a0828654acbade08d55f5a97e2da"
V103_TARGET_TAG_PAYLOAD_SHA256 = "fc18764038a43ae141c4051097c1ae5b2f84170f6609f7e9007905d556d62fa9"
V103_SNAPSHOT_COMMIT_PAYLOAD_SHA256 = "c7a111e5df72c28235053a1c56415880eb756d16dbcac49c84804435eeee042c"
V103_RELEASE_ASSET_COUNT = 18
V103_MIGRATED_RELEASE_ASSET_COUNT = 20
PRIOR_MIGRATION_RECEIPT_NAME = "gpmeep-v1.0.3-release-only-migration-receipt.json"
PRIOR_MIGRATION_MANIFEST_NAME = "gpmeep-v1.0.3-release-only-migration-assets.json"
PRIOR_RELEASE_METADATA_NAME = "release-metadata.json"
PRIOR_RELEASE_ATTESTATION_NAME = "release-attestation.json"
DISTRIBUTION_MANIFEST_NAME = "gpmeep-v1.1.0-release-only-assets.json"
SYSTEM_GIT = pathlib.Path("/usr/bin/git")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
SAFE_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
TRANSPORT_REF = "refs/tags/__gpmeep_snapshot_transport__"
DEPENDABOT_POLICY_SHA256 = "fdc9440a866a6e08995f276fe141f0d2e36e04a7706451c9c8c8ed68f5971357"
AUDITED_RECEIPT_TOOL_PATH = pathlib.Path(__file__)
AUDITED_V103_MIGRATION_PUBLISHER_PATH = pathlib.Path(__file__).with_name("publish-v103-release-only-migration.sh")
AUDITED_GITHUB_CLI_VERSION = "2.97.0"
AUDITED_GITHUB_CLI_SHA256 = "84bf91e50cbcea686e7407d7b6b09f742d66498504425d4a190f816cc073bdd1"
APPROVED_GITHUB_CLI_SHA256 = AUDITED_GITHUB_CLI_SHA256


class ReleaseOnlyError(ValueError):
    """A release-only authority or isolation invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseOnlyError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: pathlib.Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _stable_file_snapshot(paths: Iterable[pathlib.Path], label: str) -> dict[str, tuple[tuple[int, int, int, int, int], str]]:
    snapshot: dict[str, tuple[tuple[int, int, int, int, int], str]] = {}
    for candidate in paths:
        path = _regular_file(candidate, label)
        key = str(path)
        _require(key not in snapshot, f"{label} contains duplicate paths")
        snapshot[key] = (_file_identity(path), _sha256_file(path))
    return snapshot


def _stable_directory_identity(path: pathlib.Path, label: str) -> tuple[int, int, int]:
    directory = _regular_directory(path, label)
    value = directory.stat()
    return value.st_dev, value.st_ino, stat.S_IMODE(value.st_mode)


def _regular_tree_snapshot(path: pathlib.Path, label: str) -> list[tuple[str, tuple[int, int, int, int, int], str]]:
    root = _regular_directory(path, label)
    records: list[tuple[str, tuple[int, int, int, int, int], str]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        for name in directories:
            child = current_path / name
            _require(not child.is_symlink(), f"{label} contains a symlink")
        for name in files:
            child = _regular_file(current_path / name, label)
            records.append((child.relative_to(root).as_posix(), _file_identity(child), _sha256_file(child)))
    records.sort(key=lambda record: record[0])
    return records


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ReleaseOnlyError(f"non-finite JSON constant: {value}")


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(
            source.read_bytes(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseOnlyError(f"invalid JSON: {label}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == keys, f"{label} schema differs")
    return value


def _hex40(value: Any, label: str) -> str:
    _require(isinstance(value, str) and HEX40.fullmatch(value) is not None, f"{label} is invalid")
    return value


def _hex64(value: Any, label: str) -> str:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} is invalid")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} is invalid")
    return value


def _positive_int(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} is invalid")
    return value


def _regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(resolved.is_file() and stat.S_ISREG(resolved.stat().st_mode), f"{label} is not a regular file")
    return resolved


def _regular_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} path cannot contain symlinks")
    _require(resolved.is_dir(), f"{label} is not a directory")
    return resolved


def _new_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(lexical.parent.exists(), f"{label} parent must already exist")
    parent = lexical.parent.resolve(strict=True)
    _require(parent == lexical.parent and not lexical.is_symlink(), f"{label} parent cannot contain symlinks")
    try:
        lexical.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ReleaseOnlyError(f"refusing to reuse {label}") from error
    _require(stat.S_IMODE(lexical.stat().st_mode) == 0o700, f"{label} mode differs")
    _fsync_directory(parent)
    return lexical


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(
    path: pathlib.Path,
    payload: bytes,
    label: str,
    *,
    require_private_parent: bool = True,
) -> pathlib.Path:
    destination = path.absolute()
    _require(destination.parent.exists(), f"{label} parent must already exist")
    parent = destination.parent.resolve(strict=True)
    _require(parent == destination.parent and not destination.is_symlink(), f"{label} parent cannot contain symlinks")
    parent_stat = parent.stat()
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    if require_private_parent:
        _require(stat.S_IMODE(parent_stat.st_mode) == 0o700, f"{label} parent mode must be 0700")
    _require(not destination.exists(), f"refusing to overwrite {label}")
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
        current_parent = parent.stat()
        _require(
            (current_parent.st_dev, current_parent.st_ino) == parent_identity
            and (not require_private_parent or stat.S_IMODE(current_parent.st_mode) == 0o700),
            f"{label} parent changed during publication",
        )
        os.link(temporary, destination)
        linked = True
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, f"{label} mode differs")
        _require(destination.read_bytes() == payload, f"{label} changed during publication")
        temporary.unlink()
        _fsync_directory(parent)
        final_parent = parent.stat()
        _require(
            (final_parent.st_dev, final_parent.st_ino) == parent_identity
            and (not require_private_parent or stat.S_IMODE(final_parent.st_mode) == 0o700),
            f"{label} parent changed during publication",
        )
        committed = True
    finally:
        temporary.unlink(missing_ok=True)
        if linked and not committed:
            destination.unlink(missing_ok=True)
            try:
                _fsync_directory(parent)
            except OSError:
                pass
    return destination


def _write_new_json(path: pathlib.Path, value: Mapping[str, Any], label: str) -> pathlib.Path:
    return _write_new_bytes(path, _canonical_bytes(value) + b"\n", label)


def _harden_tree(root: pathlib.Path) -> None:
    """Make persistent controller state private without following symlinks."""

    root = _regular_directory(root, "release-only state directory")
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = pathlib.Path(directory)
        _require(not current.is_symlink(), "release-only state contains a symlinked directory")
        os.chmod(current, 0o700)
        for name in names:
            child = current / name
            _require(not child.is_symlink(), "release-only state contains a symlinked directory")
        for name in files:
            child = current / name
            _require(not child.is_symlink(), "release-only state contains a symlink")
            _require(child.is_file(), "release-only state contains a non-regular file")
            os.chmod(child, 0o600)


def _git_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "XDG_CONFIG_HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
    }
    if extra:
        environment.update(extra)
    return environment


def _git(
    arguments: Sequence[str],
    *,
    repo: pathlib.Path | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
    extra_environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    _require(SYSTEM_GIT.is_file(), "fixed system Git is unavailable")
    command = [str(SYSTEM_GIT), "-c", "core.hooksPath=/dev/null"]
    if repo is not None:
        command.extend(["--git-dir", str(repo)])
    command.extend(arguments)
    result = subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_git_environment(extra_environment),
        timeout=300,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReleaseOnlyError(f"fixed Git command failed ({' '.join(arguments)}): {detail}")
    return result


def _git_text(arguments: Sequence[str], *, repo: pathlib.Path | None = None) -> str:
    return _git(arguments, repo=repo).stdout.decode("ascii").strip()


def _object_exists(repo: pathlib.Path, oid: str) -> bool:
    return _git(["cat-file", "-e", oid], repo=repo, check=False).returncode == 0


def _repository_object_directory(source_repo: pathlib.Path) -> pathlib.Path:
    source = _regular_directory(source_repo, "source repository")
    git_dir_raw = _git(["-C", str(source), "rev-parse", "--absolute-git-dir"]).stdout
    git_dir = pathlib.Path(git_dir_raw.decode("utf-8").strip())
    git_dir = _regular_directory(git_dir, "source Git directory")
    objects_raw = _git(["--git-dir", str(git_dir), "rev-parse", "--git-path", "objects"]).stdout
    objects = pathlib.Path(objects_raw.decode("utf-8").strip())
    if not objects.is_absolute():
        objects = git_dir / objects
    return _regular_directory(objects, "source object directory")


def _source_git(source_repo: pathlib.Path, arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    source = _regular_directory(source_repo, "source repository")
    return _git(["-C", str(source), *arguments], check=check)


def _source_tree(source_repo: pathlib.Path, source_commit: str) -> str:
    _hex40(source_commit, "source commit")
    kind = _source_git(source_repo, ["cat-file", "-t", source_commit]).stdout.decode("ascii").strip()
    _require(kind == "commit", "source authority is not a commit")
    return _source_git(source_repo, ["rev-parse", f"{source_commit}^{{tree}}"]).stdout.decode("ascii").strip()


def _tag_state(source_repo: pathlib.Path, tag_name: str | None) -> tuple[str, str] | None:
    if tag_name is None:
        return None
    _require(tag_name == DEFAULT_TAG, "source tag name differs")
    ref = f"refs/tags/{tag_name}"
    probe = _source_git(source_repo, ["show-ref", "--hash", "--verify", ref], check=False)
    _require(probe.returncode == 0, "authoritative source tag is absent")
    tag_oid = probe.stdout.decode("ascii").strip()
    _hex40(tag_oid, "authoritative source tag object")
    _require(
        _source_git(source_repo, ["cat-file", "-t", tag_oid]).stdout.decode("ascii").strip() == "tag",
        "authoritative source tag is not annotated",
    )
    peeled = _source_git(source_repo, ["rev-parse", f"{ref}^{{}}"]).stdout.decode("ascii").strip()
    _hex40(peeled, "authoritative source tag target")
    return tag_oid, peeled


def _parse_ls_tree(payload: bytes) -> list[tuple[str, str, str, int | None, bytes]]:
    records: list[tuple[str, str, str, int | None, bytes]] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        header, separator, path = raw.partition(b"\t")
        _require(separator == b"\t" and path, "Git tree record is malformed")
        # ``ls-tree -l`` right-aligns sizes and can emit multiple spaces.
        fields = header.split()
        _require(len(fields) == 4, "Git tree record header is malformed")
        mode = fields[0].decode("ascii")
        kind = fields[1].decode("ascii")
        oid = fields[2].decode("ascii")
        size_raw = fields[3].decode("ascii")
        _require(re.fullmatch(r"[0-7]{6}", mode) is not None, "Git tree mode is malformed")
        _hex40(oid, "Git tree object")
        size = None if size_raw == "-" else int(size_raw)
        records.append((mode, kind, oid, size, path))
    return records


def tree_manifest(repo: pathlib.Path, tree_oid: str) -> dict[str, Any]:
    """Return a filter-independent manifest for every tree/blob entry."""

    tree_oid = _hex40(tree_oid, "source tree")
    _require(_git_text(["cat-file", "-t", tree_oid], repo=repo) == "tree", "source tree object is absent")
    payload = _git(
        ["ls-tree", "-r", "-t", "-z", "-l", "--full-tree", tree_oid], repo=repo
    ).stdout
    entries: list[dict[str, Any]] = []
    for mode, kind, oid, size, path in _parse_ls_tree(payload):
        _require(kind in {"tree", "blob", "commit"}, "unsupported Git tree object type")
        _require(mode != "160000" and kind != "commit", "submodules are unsupported by the release-only mirror")
        blob_sha256: str | None = None
        if kind == "blob":
            contents = _git(["cat-file", "blob", oid], repo=repo).stdout
            _require(size == len(contents), "Git blob size changed during tree replay")
            _require(not contents.startswith(LFS_HEADER), "Git LFS pointers are unsupported by the release-only mirror")
            blob_sha256 = _sha256_bytes(contents)
        entries.append(
            {
                "path_base64": base64.b64encode(path).decode("ascii"),
                "mode": mode,
                "git_type": kind,
                "object_id": oid,
                "size_bytes": size,
                "blob_sha256": blob_sha256,
            }
        )
    _require(entries, "source tree manifest is empty")
    manifest = {"object_format": "sha1", "tree": tree_oid, "entries": entries}
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def verify_dependabot_suppression(repo: pathlib.Path, tree_oid: str) -> dict[str, int]:
    payload = _git(["show", f"{_hex40(tree_oid, 'source tree')}:.github/dependabot.yml"], repo=repo).stdout
    # YAML has aliases, tags, explicit keys and escape forms which are unsafe
    # to approximate with regular expressions.  This release-specific mirror
    # therefore accepts only the reviewed policy bytes already present in the
    # measured v1.1.0 source tree.  Any semantic or comment change requires a
    # new independent review and a new pinned digest.
    _require(
        _sha256_bytes(payload) == DEPENDABOT_POLICY_SHA256,
        "Dependabot policy bytes differ from the reviewed canonical policy",
    )
    return {"github-actions": 0, "pip": 0}


def object_inventory(repo: pathlib.Path, revision: str) -> dict[str, Any]:
    listed = _git(["rev-list", "--objects", "--no-object-names", revision], repo=repo).stdout
    oids = sorted({line.decode("ascii") for line in listed.splitlines() if line})
    _require(bool(oids), "reachable object inventory is empty")
    request = b"".join(oid.encode("ascii") + b"\n" for oid in oids)
    response = _git(
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        repo=repo,
        input_bytes=request,
    ).stdout
    records: list[dict[str, Any]] = []
    counts = {"blob": 0, "commit": 0, "tag": 0, "tree": 0}
    for line in response.splitlines():
        fields = line.decode("ascii").split(" ")
        _require(len(fields) == 3, "reachable object inventory response is malformed")
        oid, kind, size_raw = fields
        _require(oid in oids and kind in counts, "reachable object inventory differs")
        size = int(size_raw)
        counts[kind] += 1
        records.append({"object_id": oid, "git_type": kind, "size_bytes": size})
    records.sort(key=lambda item: item["object_id"])
    _require(len(records) == len(oids), "reachable object inventory is incomplete")
    return {
        "scope": "reachable-from-snapshot-commit",
        "objects": records,
        "object_counts": counts,
        "inventory_sha256": _canonical_sha256(records),
    }


def _verify_no_repository_extensions(
    repo: pathlib.Path,
    *,
    expected_detached_head: str | None = None,
    allow_unborn_symbolic_head: bool = False,
) -> None:
    repository = _regular_directory(repo, "isolated repository")
    # Reject every symlink in repository storage before asking Git to inspect
    # any config, refs or objects.  Selected-path checks are insufficient
    # because Git may acquire new extension points over time.
    for current, directories, files in os.walk(repository, followlinks=False):
        current_path = pathlib.Path(current)
        for name in [*directories, *files]:
            candidate = current_path / name
            _require(not stat.S_ISLNK(candidate.lstat().st_mode), "isolated repository storage contains a symlink")
    _require(_git_text(["rev-parse", "--is-bare-repository"], repo=repository) == "true", "publisher repository is not bare")
    _require(_git_text(["rev-parse", "--show-object-format"], repo=repository) == "sha1", "repository object format is not sha1")
    forbidden_paths = {
        "objects/info/alternates": "alternate object store",
        "objects/info/http-alternates": "HTTP alternate object store",
        "shallow": "shallow boundary",
        "info/grafts": "grafts",
        "commondir": "linked common directory",
        "index": "index state",
        "FETCH_HEAD": "FETCH_HEAD state",
        "ORIG_HEAD": "ORIG_HEAD state",
        "MERGE_HEAD": "merge state",
        "MERGE_MSG": "merge message state",
        "AUTO_MERGE": "automatic merge state",
        "SQUASH_MSG": "squash state",
        "CHERRY_PICK_HEAD": "cherry-pick state",
        "REVERT_HEAD": "revert state",
        "REBASE_HEAD": "rebase head state",
        "rebase-apply": "rebase-apply state",
        "rebase-merge": "rebase-merge state",
        "sequencer": "sequencer state",
        "rr-cache": "rerere state",
        "logs": "reflog state",
        "worktrees": "linked worktree state",
    }
    for relative, label in forbidden_paths.items():
        path = repository / relative
        _require(not path.exists() and not path.is_symlink(), f"isolated repository has {label}")
    for path in repository.iterdir():
        _require(
            not path.name.startswith("BISECT_") and not path.name.startswith("CHERRY_PICK_"),
            "isolated repository has bisect or cherry-pick state",
        )
    hooks = repository / "hooks"
    _require(not hooks.is_symlink(), "isolated repository hooks directory is a symlink")
    if hooks.exists():
        _require(hooks.is_dir() and not any(hooks.iterdir()), "isolated repository has hooks")
    head = _regular_file(repository / "HEAD", "isolated repository HEAD").read_bytes()
    if expected_detached_head is not None:
        expected = f"{_hex40(expected_detached_head, 'expected detached HEAD')}\n".encode("ascii")
        _require(head == expected, "isolated repository HEAD differs from the expected detached snapshot")
    else:
        _require(allow_unborn_symbolic_head, "isolated repository HEAD policy is unspecified")
        match = re.fullmatch(rb"ref: (refs/heads/[A-Za-z0-9._/-]+)\n", head)
        _require(match is not None, "prior release repository HEAD is not an unborn symbolic branch")
        symbolic_ref = match.group(1).decode("ascii")
        _require(
            _git(["show-ref", "--verify", "--quiet", symbolic_ref], repo=repository, check=False).returncode != 0,
            "prior release repository has a live default branch",
        )
    replacements = _git(["for-each-ref", "--format=%(refname)", "refs/replace"], repo=repository).stdout
    _require(not replacements.strip(), "isolated repository has replace refs")
    config = _git(["config", "--local", "--null", "--list"], repo=repository).stdout
    actual_config: dict[str, str] = {}
    for item in config.split(b"\0"):
        if not item:
            continue
        key, separator, raw_value = item.partition(b"\n")
        _require(separator == b"\n", "isolated repository config is malformed")
        name = key.decode("utf-8").lower()
        _require(name not in actual_config, "isolated repository config contains a duplicate key")
        actual_config[name] = raw_value.decode("utf-8")
    allowed_config = {
        "core.repositoryformatversion": "0",
        "core.filemode": "true",
        "core.bare": "true",
    }
    _require(actual_config == allowed_config, "isolated repository local config is not the exact safe allowlist")


def _commit_properties(repo: pathlib.Path, commit: str) -> tuple[str, list[str]]:
    payload = _git(["cat-file", "commit", commit], repo=repo).stdout
    header = payload.split(b"\n\n", 1)[0]
    tree: str | None = None
    parents: list[str] = []
    for line in header.splitlines():
        if line.startswith(b"tree "):
            tree = line[5:].decode("ascii")
        elif line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
    _require(tree is not None, "snapshot commit has no tree")
    return _hex40(tree, "snapshot tree"), [_hex40(item, "snapshot parent") for item in parents]


def verify_isolated_repository(
    repo: pathlib.Path,
    *,
    snapshot_commit: str,
    source_tree: str,
    source_commit: str,
    tag_name: str | None = None,
    tag_object: str | None = None,
    forbidden_objects: Iterable[str] = (),
    expected_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that a publishing repository contains one parentless release."""

    repository = _regular_directory(repo, "isolated repository")
    snapshot_commit = _hex40(snapshot_commit, "snapshot commit")
    source_tree = _hex40(source_tree, "source tree")
    source_commit = _hex40(source_commit, "private source commit")
    _verify_no_repository_extensions(repository, expected_detached_head=snapshot_commit)
    refs_raw = _git(["for-each-ref", "--format=%(refname)"], repo=repository).stdout
    refs = sorted(line.decode("utf-8") for line in refs_raw.splitlines() if line)
    if tag_name is None:
        _require(refs == [], "pre-tag isolated repository has unexpected refs")
        revision = snapshot_commit
    else:
        _require(tag_name == DEFAULT_TAG, "release tag name differs")
        expected_ref = f"refs/tags/{tag_name}"
        _require(refs == [expected_ref], "isolated repository ref inventory differs")
        actual_tag = _git_text(["rev-parse", expected_ref], repo=repository)
        _require(_git_text(["cat-file", "-t", actual_tag], repo=repository) == "tag", "release tag is not annotated")
        if tag_object is not None:
            _require(actual_tag == _hex40(tag_object, "tag object"), "annotated tag object differs")
        _require(_git_text(["rev-parse", f"{expected_ref}^{{}}"], repo=repository) == snapshot_commit, "release tag target differs")
        revision = expected_ref
    tree, parents = _commit_properties(repository, snapshot_commit)
    _require(tree == source_tree, "snapshot tree differs from authoritative source tree")
    _require(parents == [], "snapshot commit is not parentless")
    commit_count = int(_git_text(["rev-list", "--count", snapshot_commit], repo=repository))
    _require(commit_count == 1, "snapshot reachable commit count differs")
    _require(source_commit != snapshot_commit, "distribution commit substituted for source authority")
    _require(not _object_exists(repository, source_commit), "private source commit leaked into isolated repository")
    checked_forbidden: list[str] = []
    for oid in sorted(set(forbidden_objects)):
        checked = _hex40(oid, "forbidden development object")
        _require(not _object_exists(repository, checked), f"forbidden development object leaked: {checked}")
        checked_forbidden.append(checked)
    fsck = _git(["fsck", "--full", "--no-reflogs", "--unreachable", revision], repo=repository, check=False)
    _require(fsck.returncode == 0, "isolated repository fsck failed")
    findings = (fsck.stdout + fsck.stderr).decode("utf-8", "replace").strip()
    # A bare tag-only repository can report an unborn default HEAD as a notice.
    non_notice = [line for line in findings.splitlines() if not line.startswith("notice: HEAD points to an unborn branch")]
    _require(not non_notice, "isolated repository has fsck findings")
    inventory = object_inventory(repository, revision)
    if expected_inventory is not None:
        _require(inventory == expected_inventory, "reachable object inventory differs from receipt")
    return {
        "tree": tree,
        "parent_count": len(parents),
        "reachable_commit_count": commit_count,
        "forbidden_objects_absent": checked_forbidden,
        "inventory": inventory,
    }


def _validate_v103_migration_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "source_authority", "snapshot", "target", "publication_id"},
        "v1.0.3 migration receipt",
    )
    unsigned = dict(retained)
    publication_id = unsigned.pop("publication_id")
    _require(
        _canonical_sha256(unsigned) == _hex64(publication_id, "v1.0.3 migration publication identity"),
        "v1.0.3 migration receipt identity differs",
    )
    _require(retained["schema"] == PRIOR_MIGRATION_RECEIPT_SCHEMA, "v1.0.3 migration receipt schema differs")
    source = _exact(
        retained["source_authority"],
        {
            "repository", "legacy_target_repository", "source_release",
            "source_commit", "source_tree", "source_tag",
        },
        "v1.0.3 migration source authority",
    )
    source_repository = _exact(source["repository"], {"slug", "id", "node_id"}, "v1.0.3 source repository")
    _require(
        source_repository == {
            "slug": V103_SOURCE_REPOSITORY,
            "id": V103_SOURCE_REPOSITORY_ID,
            "node_id": V103_SOURCE_REPOSITORY_NODE_ID,
        },
        "v1.0.3 source repository authority differs",
    )
    legacy_repository = _exact(
        source["legacy_target_repository"], {"slug", "id", "node_id"},
        "v1.0.3 legacy target repository",
    )
    _require(
        legacy_repository == {
            "slug": TARGET_REPOSITORY,
            "id": V103_LEGACY_TARGET_REPOSITORY_ID,
            "node_id": V103_LEGACY_TARGET_REPOSITORY_NODE_ID,
        },
        "v1.0.3 legacy target repository authority differs",
    )
    source_release = _exact(
        source["source_release"],
        {"id", "node_id", "tag", "draft", "prerelease", "immutable", "asset_count", "assets_sha256"},
        "v1.0.3 private source release observation",
    )
    _require(
        source_release["id"] == V103_SOURCE_RELEASE_ID
        and source_release["node_id"] == V103_SOURCE_RELEASE_NODE_ID
        and source_release["tag"] == "v1.0.3"
        and source_release["draft"] is False and source_release["prerelease"] is False
        and source_release["immutable"] is False
        and source_release["asset_count"] == V103_RELEASE_ASSET_COUNT,
        "v1.0.3 private mutable source release observation differs",
    )
    _hex64(source_release["assets_sha256"], "v1.0.3 private source release asset census")
    _require(
        source["source_commit"] == V103_SOURCE_COMMIT and source["source_tree"] == V103_SOURCE_TREE,
        "v1.0.3 source commit or tree differs",
    )
    source_tag = _exact(
        source["source_tag"], {"name", "object_id", "target", "payload_sha256"},
        "v1.0.3 source tag",
    )
    _require(
        source_tag == {
            "name": "v1.0.3", "object_id": V103_SOURCE_TAG_OBJECT,
            "target": V103_SOURCE_COMMIT, "payload_sha256": V103_SOURCE_TAG_PAYLOAD_SHA256,
        },
        "v1.0.3 private source tag authority differs",
    )
    snapshot = _exact(
        retained["snapshot"],
        {
            "snapshot_commit", "snapshot_tree", "parent_count", "reachable_commit_count",
            "tree_manifest_sha256", "commit_payload_sha256", "tag_object", "tag_target",
            "tag_payload_sha256",
        },
        "v1.0.3 migration snapshot",
    )
    _require(
        snapshot["snapshot_commit"] == V103_SNAPSHOT_COMMIT
        and snapshot["snapshot_tree"] == V103_SOURCE_TREE
        and snapshot["parent_count"] == 0
        and snapshot["reachable_commit_count"] == 1
        and snapshot["commit_payload_sha256"] == V103_SNAPSHOT_COMMIT_PAYLOAD_SHA256
        and snapshot["tag_object"] == V103_TARGET_TAG_OBJECT
        and snapshot["tag_target"] == snapshot["snapshot_commit"]
        and snapshot["tag_payload_sha256"] == V103_TARGET_TAG_PAYLOAD_SHA256,
        "v1.0.3 parentless snapshot or annotated tag authority differs",
    )
    _hex64(snapshot["tree_manifest_sha256"], "v1.0.3 snapshot tree manifest")
    target = _exact(retained["target"], {"repository", "organization", "tag", "release"}, "v1.0.3 migration target")
    target_repository = _exact(target["repository"], {"slug", "id", "node_id"}, "v1.0.3 target repository")
    _require(
        target_repository["slug"] == TARGET_REPOSITORY
        and type(target_repository["id"]) is int and target_repository["id"] > 0
        and isinstance(target_repository["node_id"], str) and target_repository["node_id"],
        "v1.0.3 target repository authority differs",
    )
    _require(
        (target_repository["id"], target_repository["node_id"])
        != (legacy_repository["id"], legacy_repository["node_id"]),
        "v1.0.3 migration target did not move to a fresh repository identity",
    )
    organization = _exact(target["organization"], {"login", "id", "node_id"}, "v1.0.3 target organization")
    _require(
        organization["login"] == "example-organization"
        and type(organization["id"]) is int and organization["id"] > 0
        and isinstance(organization["node_id"], str) and organization["node_id"],
        "v1.0.3 target organization authority differs",
    )
    _require(target["tag"] == "v1.0.3", "v1.0.3 migration target tag differs")
    release = _exact(
        target["release"], {"id", "node_id", "published", "prerelease", "immutable"},
        "v1.0.3 target immutable release",
    )
    _require(
        type(release["id"]) is int and release["id"] > 0
        and isinstance(release["node_id"], str) and release["node_id"]
        and release["published"] is True and release["prerelease"] is False
        and release["immutable"] is True,
        "v1.0.3 target release is not final and immutable",
    )
    return dict(retained)


def _validate_v103_migration_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value, {"schema", "mirror_publication_id", "legacy_asset_count", "assets", "manifest_id"},
        "v1.0.3 migration asset manifest",
    )
    unsigned = dict(retained)
    manifest_id = unsigned.pop("manifest_id")
    _require(
        _canonical_sha256(unsigned) == _hex64(manifest_id, "v1.0.3 migration manifest identity"),
        "v1.0.3 migration manifest identity differs",
    )
    _require(retained["schema"] == PRIOR_MIGRATION_MANIFEST_SCHEMA, "v1.0.3 migration manifest schema differs")
    _hex64(retained["mirror_publication_id"], "v1.0.3 migration publication identity")
    assets = retained["assets"]
    _require(retained["legacy_asset_count"] == V103_RELEASE_ASSET_COUNT, "v1.0.3 legacy release asset count differs")
    _require(isinstance(assets, list) and len(assets) == V103_RELEASE_ASSET_COUNT + 1, "v1.0.3 migration manifest must bind 18 legacy assets plus its receipt")
    names: list[str] = []
    roles: list[str] = []
    for asset in assets:
        record = _exact(asset, {"role", "name", "size_bytes", "sha256"}, "v1.0.3 migration asset")
        _require(record["role"] in {"v103_legacy_release_asset", "migration_receipt"}, "v1.0.3 migration asset role differs")
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), "v1.0.3 migration asset name differs")
        _positive_int(record["size_bytes"], f"v1.0.3 migration asset size: {record['name']}")
        _hex64(record["sha256"], f"v1.0.3 migration asset digest: {record['name']}")
        names.append(record["name"])
        roles.append(record["role"])
    _require(names == sorted(set(names)), "v1.0.3 migration asset census is not canonical")
    _require(
        roles.count("v103_legacy_release_asset") == V103_RELEASE_ASSET_COUNT
        and roles.count("migration_receipt") == 1
        and next(item["name"] for item in assets if item["role"] == "migration_receipt") == PRIOR_MIGRATION_RECEIPT_NAME,
        "v1.0.3 migration manifest role census differs",
    )
    return dict(retained)


def _validate_v103_release_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value, {"schema", "repository", "organization", "release", "capture_id"},
        "v1.0.3 live release metadata",
    )
    unsigned = dict(retained)
    capture_id = unsigned.pop("capture_id")
    _require(_canonical_sha256(unsigned) == _hex64(capture_id, "v1.0.3 release capture identity"), "v1.0.3 release capture identity differs")
    _require(retained["schema"] == "gpmeep-v103-release-only-live-release-v1", "v1.0.3 release capture schema differs")
    repository = _exact(retained["repository"], {"slug", "id", "node_id"}, "v1.0.3 release repository")
    _require(repository["slug"] == TARGET_REPOSITORY, "v1.0.3 live release repository differs")
    _positive_int(repository["id"], "v1.0.3 live release repository ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "v1.0.3 live release repository node ID differs")
    organization = _exact(retained["organization"], {"login", "id", "node_id"}, "v1.0.3 release organization")
    _require(
        organization["login"] == "example-organization"
        and type(organization["id"]) is int and organization["id"] > 0
        and isinstance(organization["node_id"], str) and organization["node_id"],
        "v1.0.3 release organization differs",
    )
    release = _exact(
        retained["release"],
        {"id", "node_id", "tag", "title", "body", "draft", "prerelease", "immutable", "assets"},
        "v1.0.3 live release",
    )
    _positive_int(release["id"], "v1.0.3 live release ID")
    _require(
        isinstance(release["node_id"], str) and release["node_id"]
        and release["tag"] == "v1.0.3"
        and isinstance(release["title"], str) and release["title"]
        and isinstance(release["body"], str)
        and release["draft"] is False and release["prerelease"] is False and release["immutable"] is True,
        "v1.0.3 live release is not final and immutable",
    )
    assets = release["assets"]
    _require(isinstance(assets, list) and len(assets) == V103_MIGRATED_RELEASE_ASSET_COUNT, "v1.0.3 live release asset count differs")
    names: list[str] = []
    for asset in assets:
        record = _exact(asset, {"id", "node_id", "name", "size_bytes", "sha256"}, "v1.0.3 live release asset")
        _positive_int(record["id"], f"v1.0.3 live release asset ID: {record['name']}")
        _require(isinstance(record["node_id"], str) and record["node_id"], "v1.0.3 live release asset node ID differs")
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), "v1.0.3 live release asset name differs")
        _positive_int(record["size_bytes"], f"v1.0.3 live release asset size: {record['name']}")
        _hex64(record["sha256"], f"v1.0.3 live release asset digest: {record['name']}")
        names.append(record["name"])
    _require(names == sorted(set(names)), "v1.0.3 live release asset census is not canonical")
    return dict(retained)


def validate_v103_migration_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {
            "schema", "state", "source_authority", "snapshot", "target",
            "legacy_assets", "implementation", "github_cli", "findings", "auditor", "audit_id",
        },
        "v1.0.3 independent migration audit",
    )
    unsigned = dict(retained)
    audit_id = unsigned.pop("audit_id")
    _require(
        _canonical_sha256(unsigned) == _hex64(audit_id, "v1.0.3 migration audit identity"),
        "v1.0.3 migration audit identity differs",
    )
    _require(
        retained["schema"] == PRIOR_MIGRATION_AUDIT_SCHEMA and retained["state"] == "GO",
        "v1.0.3 independent migration audit is not GO",
    )
    target = _exact(
        retained["target"],
        {"repository", "organization", "never_imported_or_transferred", "empty_ref_inventory_sha256"},
        "v1.0.3 migration audit target",
    )
    _require(
        target["never_imported_or_transferred"] is True
        and target["empty_ref_inventory_sha256"] == _canonical_sha256([]),
        "v1.0.3 migration target is not independently audited as fresh and empty",
    )
    audit_snapshot = _exact(
        retained["snapshot"],
        {
            "snapshot_commit", "snapshot_tree", "parent_count", "reachable_commit_count",
            "tree_manifest_sha256", "commit_payload_sha256", "tag_object", "tag_target",
            "tag_payload_sha256", "reachable_inventory",
        },
        "v1.0.3 audited legacy snapshot",
    )
    receipt_snapshot = {key: item for key, item in audit_snapshot.items() if key != "reachable_inventory"}
    probe = {
        "schema": PRIOR_MIGRATION_RECEIPT_SCHEMA,
        "source_authority": retained["source_authority"],
        "snapshot": receipt_snapshot,
        "target": {
            "repository": target["repository"],
            "organization": target["organization"],
            "tag": "v1.0.3",
            "release": {"id": 1, "node_id": "migration-audit-probe", "published": True, "prerelease": False, "immutable": True},
        },
    }
    probe["publication_id"] = _canonical_sha256(probe)
    _validate_v103_migration_receipt(probe)
    inventory = _exact(audit_snapshot["reachable_inventory"], {"scope", "objects", "object_counts", "inventory_sha256"}, "v1.0.3 audited legacy inventory")
    _require(inventory["scope"] == "reachable-from-release-tag", "v1.0.3 audited legacy inventory scope differs")
    _require(_canonical_sha256(inventory["objects"]) == inventory["inventory_sha256"], "v1.0.3 audited legacy inventory identity differs")
    assets = retained["legacy_assets"]
    _require(isinstance(assets, list) and len(assets) == V103_RELEASE_ASSET_COUNT, "v1.0.3 audit legacy asset count differs")
    names: list[str] = []
    for asset in assets:
        record = _exact(asset, {"role", "name", "size_bytes", "sha256"}, "v1.0.3 audited legacy asset")
        _require(record["role"] == "v103_legacy_release_asset", "v1.0.3 audited legacy asset role differs")
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), "v1.0.3 audited legacy asset name differs")
        _positive_int(record["size_bytes"], f"v1.0.3 audited legacy asset size: {record['name']}")
        _hex64(record["sha256"], f"v1.0.3 audited legacy asset digest: {record['name']}")
        names.append(record["name"])
    _require(names == sorted(set(names)), "v1.0.3 audited legacy asset census is not canonical")
    _require(
        retained["source_authority"]["source_release"]["assets_sha256"] == _canonical_sha256(assets),
        "v1.0.3 mutable source release differs from audited legacy assets",
    )
    implementation = _exact(
        retained["implementation"],
        {"receipt_tool_sha256", "migration_publisher_sha256", "tests_sha256"},
        "v1.0.3 audited migration implementation",
    )
    for key, digest in implementation.items():
        _hex64(digest, f"v1.0.3 audited migration implementation {key}")
    _require(
        _exact(retained["github_cli"], {"version", "sha256"}, "v1.0.3 audited GitHub CLI")
        == {"version": AUDITED_GITHUB_CLI_VERSION, "sha256": AUDITED_GITHUB_CLI_SHA256},
        "v1.0.3 migration GitHub CLI authority differs",
    )
    _require(
        _exact(retained["findings"], {"open_critical", "open_high", "open_medium"}, "v1.0.3 migration findings")
        == {"open_critical": 0, "open_high": 0, "open_medium": 0},
        "v1.0.3 migration audit has open findings",
    )
    auditor = _exact(retained["auditor"], {"independent", "name", "completed_at_utc"}, "v1.0.3 migration auditor")
    _require(
        auditor["independent"] is True and isinstance(auditor["name"], str) and auditor["name"]
        and isinstance(auditor["completed_at_utc"], str) and auditor["completed_at_utc"].endswith("Z"),
        "v1.0.3 migration auditor identity differs",
    )
    return dict(retained)


def verify_v103_migration_audit_files(
    value: Mapping[str, Any], *, audit_path: pathlib.Path, trusted_sha256: str,
    legacy_repo: pathlib.Path, legacy_assets: pathlib.Path,
) -> dict[str, Any]:
    path = _regular_file(audit_path, "v1.0.3 independent migration audit")
    _require(stat.S_IMODE(path.stat().st_mode) == 0o600, "v1.0.3 migration audit mode must be 0600")
    _require(_sha256_file(path) == _hex64(trusted_sha256, "trusted v1.0.3 migration audit digest"), "v1.0.3 migration audit differs from its out-of-band trusted digest")
    retained = validate_v103_migration_audit(value)
    implementation_paths = {
        "receipt_tool_sha256": _regular_file(AUDITED_RECEIPT_TOOL_PATH, "v1.0.3 migration receipt tool"),
        "migration_publisher_sha256": _regular_file(AUDITED_V103_MIGRATION_PUBLISHER_PATH, "v1.0.3 migration publisher"),
        "tests_sha256": _regular_file(AUDITED_RECEIPT_TOOL_PATH.parent / "tests/test_v110_release_only_mirror.py", "v1.0.3 migration tests"),
    }
    for key, implementation_path in implementation_paths.items():
        _require(_sha256_file(implementation_path) == retained["implementation"][key], f"audited v1.0.3 migration implementation changed: {key}")
    repository = _regular_directory(legacy_repo, "v1.0.3 legacy snapshot repository")
    snapshot = retained["snapshot"]
    _require(_git_text(["rev-parse", "refs/tags/v1.0.3"], repo=repository) == snapshot["tag_object"], "audited legacy v1.0.3 tag object differs")
    _require(_git_text(["rev-parse", "refs/tags/v1.0.3^{}"], repo=repository) == snapshot["snapshot_commit"], "audited legacy v1.0.3 tag target differs")
    _require(
        _sha256_bytes(_git(["cat-file", "tag", snapshot["tag_object"]], repo=repository).stdout) == snapshot["tag_payload_sha256"]
        and _sha256_bytes(_git(["cat-file", "commit", snapshot["snapshot_commit"]], repo=repository).stdout) == snapshot["commit_payload_sha256"],
        "audited legacy v1.0.3 raw tag or commit payload differs",
    )
    _verify_no_repository_extensions(repository, allow_unborn_symbolic_head=True)
    refs = _git_text(["for-each-ref", "--format=%(refname)"], repo=repository).splitlines()
    _require(refs == ["refs/tags/v1.0.3"], "audited legacy v1.0.3 ref inventory differs")
    snapshot_tree, parents = _commit_properties(repository, snapshot["snapshot_commit"])
    _require(snapshot_tree == snapshot["snapshot_tree"] and not parents, "audited legacy v1.0.3 snapshot topology differs")
    _require(int(_git_text(["rev-list", "--count", snapshot["snapshot_commit"]], repo=repository)) == 1, "audited legacy v1.0.3 snapshot is not parentless")
    tree = tree_manifest(repository, snapshot["snapshot_tree"])
    _require(tree["manifest_sha256"] == snapshot["tree_manifest_sha256"], "audited legacy v1.0.3 tree manifest differs")
    inventory = {**object_inventory(repository, "refs/tags/v1.0.3"), "scope": "reachable-from-release-tag"}
    _require(inventory == snapshot["reachable_inventory"], "audited legacy v1.0.3 object inventory differs")
    root = _regular_directory(legacy_assets, "v1.0.3 legacy asset directory")
    actual_names: list[str] = []
    for candidate in root.iterdir():
        file_path = _regular_file(candidate, "v1.0.3 legacy release asset")
        actual_names.append(file_path.name)
    _require(sorted(actual_names) == [item["name"] for item in retained["legacy_assets"]], "v1.0.3 legacy asset directory census differs")
    for record in retained["legacy_assets"]:
        candidate = _regular_file(root / record["name"], f"v1.0.3 legacy asset {record['name']}")
        _require(candidate.stat().st_size == record["size_bytes"] and _sha256_file(candidate) == record["sha256"], f"v1.0.3 legacy asset bytes differ: {record['name']}")
    return retained


def create_v103_migration_receipt(
    audit: Mapping[str, Any], *, release_id: int, release_node_id: str,
) -> dict[str, Any]:
    retained = validate_v103_migration_audit(audit)
    _positive_int(release_id, "v1.0.3 migration release ID")
    _require(isinstance(release_node_id, str) and release_node_id, "v1.0.3 migration release node ID differs")
    snapshot = {
        key: item for key, item in retained["snapshot"].items() if key != "reachable_inventory"
    }
    value: dict[str, Any] = {
        "schema": PRIOR_MIGRATION_RECEIPT_SCHEMA,
        "source_authority": retained["source_authority"],
        "snapshot": snapshot,
        "target": {
            "repository": retained["target"]["repository"],
            "organization": retained["target"]["organization"],
            "tag": "v1.0.3",
            "release": {
                "id": release_id, "node_id": release_node_id, "published": True,
                "prerelease": False, "immutable": True,
            },
        },
    }
    value["publication_id"] = _canonical_sha256(value)
    return _validate_v103_migration_receipt(value)


def create_v103_migration_manifest(
    audit: Mapping[str, Any], *, receipt_path: pathlib.Path, legacy_assets: pathlib.Path,
) -> dict[str, Any]:
    retained = validate_v103_migration_audit(audit)
    receipt_file = _regular_file(receipt_path, "v1.0.3 migration receipt")
    receipt = _validate_v103_migration_receipt(_load_json(receipt_file, "v1.0.3 migration receipt"))
    root = _regular_directory(legacy_assets, "v1.0.3 legacy release assets")
    actual_names = sorted(path.name for path in root.iterdir())
    _require(actual_names == [item["name"] for item in retained["legacy_assets"]], "v1.0.3 legacy release asset census differs")
    for record in retained["legacy_assets"]:
        path = _regular_file(root / record["name"], f"v1.0.3 legacy release asset {record['name']}")
        _require(path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"], f"v1.0.3 legacy release asset bytes differ: {record['name']}")
    receipt_record = {
        "role": "migration_receipt", "name": receipt_file.name,
        "size_bytes": receipt_file.stat().st_size, "sha256": _sha256_file(receipt_file),
    }
    _require(receipt_file.name == PRIOR_MIGRATION_RECEIPT_NAME, "v1.0.3 migration receipt name differs")
    value: dict[str, Any] = {
        "schema": PRIOR_MIGRATION_MANIFEST_SCHEMA,
        "mirror_publication_id": receipt["publication_id"],
        "legacy_asset_count": V103_RELEASE_ASSET_COUNT,
        "assets": sorted([*retained["legacy_assets"], receipt_record], key=lambda item: item["name"]),
    }
    value["manifest_id"] = _canonical_sha256(value)
    return _validate_v103_migration_manifest(value)


def create_v103_release_metadata(
    *, receipt: Mapping[str, Any], manifest: Mapping[str, Any],
    repository_metadata: Mapping[str, Any], organization_metadata: Mapping[str, Any],
    release_metadata: Mapping[str, Any], manifest_path: pathlib.Path,
) -> dict[str, Any]:
    retained_receipt = _validate_v103_migration_receipt(receipt)
    retained_manifest = _validate_v103_migration_manifest(manifest)
    manifest_file = _regular_file(manifest_path, "v1.0.3 migration manifest")
    _require(
        manifest_file.name == PRIOR_MIGRATION_MANIFEST_NAME
        and manifest_file.stat().st_size > 0 and _sha256_file(manifest_file) == _sha256_bytes(_canonical_bytes(retained_manifest) + b"\n"),
        "v1.0.3 migration manifest file differs",
    )
    repository = {
        "slug": repository_metadata.get("full_name"),
        "id": repository_metadata.get("id"),
        "node_id": repository_metadata.get("node_id"),
    }
    organization = {
        "login": organization_metadata.get("login"),
        "id": organization_metadata.get("id"),
        "node_id": organization_metadata.get("node_id"),
    }
    _require(
        repository == retained_receipt["target"]["repository"]
        and organization == retained_receipt["target"]["organization"],
        "v1.0.3 live repository or organization differs from the migration receipt",
    )
    expected_assets = {
        item["name"]: (item["size_bytes"], item["sha256"])
        for item in retained_manifest["assets"]
    }
    expected_assets[manifest_file.name] = (manifest_file.stat().st_size, _sha256_file(manifest_file))
    raw_assets = release_metadata.get("assets")
    _require(isinstance(raw_assets, list), "v1.0.3 live release assets differ")
    assets: list[dict[str, Any]] = []
    for raw in raw_assets:
        _require(isinstance(raw, dict), "v1.0.3 live release asset projection differs")
        digest = raw.get("digest")
        _require(isinstance(digest, str) and digest.startswith("sha256:"), "v1.0.3 live release asset digest differs")
        assets.append({
            "id": raw.get("id"), "node_id": raw.get("node_id"), "name": raw.get("name"),
            "size_bytes": raw.get("size"), "sha256": digest.removeprefix("sha256:"),
        })
    assets.sort(key=lambda item: item["name"] if isinstance(item["name"], str) else "")
    _require(
        {item["name"]: (item["size_bytes"], item["sha256"]) for item in assets} == expected_assets,
        "v1.0.3 live release asset bytes differ from the migration manifest",
    )
    release = {
        "id": release_metadata.get("id"), "node_id": release_metadata.get("node_id"),
        "tag": release_metadata.get("tag_name"), "title": release_metadata.get("name"),
        "body": release_metadata.get("body"), "draft": release_metadata.get("draft"),
        "prerelease": release_metadata.get("prerelease"), "immutable": release_metadata.get("immutable"),
        "assets": assets,
    }
    _require(
        {key: release[key] for key in ("id", "node_id")} == {
            "id": retained_receipt["target"]["release"]["id"],
            "node_id": retained_receipt["target"]["release"]["node_id"],
        },
        "v1.0.3 live release identity differs from the migration receipt",
    )
    value: dict[str, Any] = {
        "schema": "gpmeep-v103-release-only-live-release-v1",
        "repository": repository, "organization": organization, "release": release,
    }
    value["capture_id"] = _canonical_sha256(value)
    return _validate_v103_release_metadata(value)


def create_v103_migration_run_intent(
    *, audit_path: pathlib.Path, trusted_audit_sha256: str, legacy_repo: pathlib.Path,
    legacy_assets: pathlib.Path, target_repo: pathlib.Path, gh_binary: pathlib.Path,
    require_empty_target: bool = True,
) -> dict[str, Any]:
    audit = verify_v103_migration_audit_files(
        _load_json(audit_path, "v1.0.3 independent migration audit"),
        audit_path=audit_path, trusted_sha256=trusted_audit_sha256,
        legacy_repo=legacy_repo, legacy_assets=legacy_assets,
    )
    if require_empty_target:
        verify_empty_target_repository(target_repo)
    else:
        _regular_directory(target_repo, "v1.0.3 migration target repository")
    binary = _regular_file(gh_binary, "v1.0.3 migration GitHub CLI")
    _require(os.access(binary, os.X_OK) and _sha256_file(binary) == APPROVED_GITHUB_CLI_SHA256, "v1.0.3 migration GitHub CLI differs")
    value: dict[str, Any] = {
        "schema": PRIOR_MIGRATION_RUN_INTENT_SCHEMA,
        "state": "MIGRATION_AUTHORIZED",
        "audit": {"sha256": trusted_audit_sha256, "audit_id": audit["audit_id"]},
        "source_authority": audit["source_authority"],
        "snapshot": audit["snapshot"],
        "target": audit["target"],
        "legacy_assets": audit["legacy_assets"],
        "implementation": audit["implementation"],
        "github_cli": audit["github_cli"],
        "transport": {
            "mode": "github",
            "gh_binary": {"path": str(binary), "size_bytes": binary.stat().st_size, "sha256": _sha256_file(binary)},
        },
    }
    value["intent_id"] = _canonical_sha256(value)
    return validate_v103_migration_run_intent(value)


def validate_v103_migration_run_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "state", "audit", "source_authority", "snapshot", "target", "legacy_assets", "implementation", "github_cli", "transport", "intent_id"},
        "v1.0.3 migration run intent",
    )
    unsigned = dict(retained)
    intent_id = unsigned.pop("intent_id")
    _require(_canonical_sha256(unsigned) == _hex64(intent_id, "v1.0.3 migration intent identity"), "v1.0.3 migration run intent identity differs")
    _require(retained["schema"] == PRIOR_MIGRATION_RUN_INTENT_SCHEMA and retained["state"] == "MIGRATION_AUTHORIZED", "v1.0.3 migration run intent is not authorized")
    audit_record = _exact(retained["audit"], {"sha256", "audit_id"}, "v1.0.3 migration intent audit")
    _hex64(audit_record["sha256"], "v1.0.3 migration trusted audit digest")
    _hex64(audit_record["audit_id"], "v1.0.3 migration audit identity")
    audit_probe = {
        "schema": PRIOR_MIGRATION_AUDIT_SCHEMA, "state": "GO",
        "source_authority": retained["source_authority"], "snapshot": retained["snapshot"],
        "target": retained["target"], "legacy_assets": retained["legacy_assets"],
        "implementation": retained["implementation"], "github_cli": retained["github_cli"],
        "findings": {"open_critical": 0, "open_high": 0, "open_medium": 0},
        "auditor": {"independent": True, "name": "intent semantic probe", "completed_at_utc": "1970-01-01T00:00:00Z"},
    }
    audit_probe["audit_id"] = _canonical_sha256(audit_probe)
    validate_v103_migration_audit(audit_probe)
    transport = _exact(retained["transport"], {"mode", "gh_binary"}, "v1.0.3 migration transport")
    binary = _exact(transport["gh_binary"], {"path", "size_bytes", "sha256"}, "v1.0.3 migration GitHub CLI binary")
    _require(transport["mode"] == "github" and pathlib.Path(binary["path"]).is_absolute(), "v1.0.3 migration transport differs")
    _positive_int(binary["size_bytes"], "v1.0.3 migration GitHub CLI size")
    _require(binary["sha256"] == APPROVED_GITHUB_CLI_SHA256, "v1.0.3 migration GitHub CLI digest differs")
    return dict(retained)


def _verify_prior_release_assets(
    authority: Mapping[str, Any], *, repo: pathlib.Path, assets_root: pathlib.Path
) -> dict[str, Any]:
    retained = validate_prior_release_authority(authority)
    tag = retained["tag"]
    root = _regular_directory(assets_root / tag, f"prior release {tag} asset directory")
    receipt_record = retained["mirror_receipt"]
    manifest_record = retained["distribution_manifest"]
    receipt_path = _regular_file(root / receipt_record["name"], f"prior release {tag} mirror receipt")
    manifest_path = _regular_file(root / manifest_record["name"], f"prior release {tag} distribution manifest")
    metadata_path = _regular_file(root / PRIOR_RELEASE_METADATA_NAME, f"prior release {tag} live release metadata")
    attestation_path = _regular_file(root / PRIOR_RELEASE_ATTESTATION_NAME, f"prior release {tag} signed release attestation")
    for record, path, label in (
        (receipt_record, receipt_path, "mirror receipt"),
        (manifest_record, manifest_path, "distribution manifest"),
    ):
        _require(path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"], f"prior release {tag} {label} bytes differ")
    receipt = _validate_v103_migration_receipt(_load_json(receipt_path, f"prior release {tag} mirror receipt"))
    publication_id = receipt["publication_id"]
    _require(publication_id == receipt_record["publication_id"], f"prior release {tag} mirror receipt identity differs")
    target = receipt["target"]
    snapshot = receipt["snapshot"]
    source = receipt["source_authority"]
    _require(
        snapshot["snapshot_commit"] == retained["snapshot_commit"]
        and snapshot["snapshot_tree"] == retained["snapshot_tree"]
        and snapshot["tree_manifest_sha256"] == retained["tree_manifest_sha256"]
        and snapshot["tag_object"] == retained["tag_object"]
        and snapshot["tag_payload_sha256"] == retained["tag_payload_sha256"]
        and snapshot["commit_payload_sha256"] == retained["snapshot_commit_payload_sha256"]
        and source["source_tree"] == retained["snapshot_tree"],
        f"prior release {tag} receipt tree authority differs",
    )
    manifest = _validate_v103_migration_manifest(_load_json(manifest_path, f"prior release {tag} distribution manifest"))
    manifest_id = manifest["manifest_id"]
    _require(manifest_id == manifest_record["manifest_id"], f"prior release {tag} distribution manifest identity differs")
    _require(manifest["mirror_publication_id"] == publication_id, f"prior release {tag} distribution authority differs")
    _require(
        source["source_release"]["assets_sha256"]
        == _canonical_sha256([item for item in manifest["assets"] if item["role"] == "v103_legacy_release_asset"]),
        f"prior release {tag} mutable source release asset census differs",
    )
    names: list[str] = []
    for asset in manifest["assets"]:
        path = _regular_file(root / asset["name"], f"prior release {tag} asset {asset['name']}")
        _require(path.parent == root and path.stat().st_size == asset["size_bytes"] and _sha256_file(path) == asset["sha256"], f"prior release {tag} asset bytes differ: {asset['name']}")
        names.append(asset["name"])
    expected_evidence_names = sorted([*names, manifest_path.name, metadata_path.name, attestation_path.name])
    actual_evidence_names: list[str] = []
    for path in root.iterdir():
        _require(not path.is_symlink() and path.is_file() and stat.S_ISREG(path.stat().st_mode), f"prior release {tag} evidence contains a non-regular entry")
        actual_evidence_names.append(path.name)
    _require(sorted(actual_evidence_names) == expected_evidence_names, f"prior release {tag} evidence directory census differs")
    metadata = _validate_v103_release_metadata(_load_json(metadata_path, f"prior release {tag} live release metadata"))
    _require(
        metadata["repository"] == target["repository"]
        and metadata["organization"] == target["organization"],
        f"prior release {tag} repository or organization authority differs",
    )
    live_release = metadata["release"]
    _require(
        {key: live_release[key] for key in ("id", "node_id", "prerelease", "immutable")}
        == {
            "id": target["release"]["id"], "node_id": target["release"]["node_id"],
            "prerelease": False, "immutable": True,
        }
        and live_release["draft"] is False,
        f"prior release {tag} immutable release identity differs",
    )
    live_assets = [
        {"id": item["id"], "node_id": item["node_id"], "name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in live_release["assets"]
    ]
    manifest_assets = [
        {"name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in manifest["assets"]
    ]
    manifest_assets.append(
        {"name": manifest_path.name, "size_bytes": manifest_path.stat().st_size, "sha256": _sha256_file(manifest_path)}
    )
    manifest_assets.sort(key=lambda item: item["name"])
    _require(
        [{key: item[key] for key in ("name", "size_bytes", "sha256")} for item in live_assets] == manifest_assets,
        f"prior release {tag} immutable release asset union differs",
    )
    tag_ref = f"refs/tags/{tag}"
    _require(_git_text(["rev-parse", tag_ref], repo=repo) == retained["tag_object"], f"prior release {tag} tag object differs")
    _require(_git_text(["rev-parse", f"{tag_ref}^{{}}"], repo=repo) == retained["snapshot_commit"], f"prior release {tag} tag target differs")
    tag_payload = _git(["cat-file", "tag", retained["tag_object"]], repo=repo).stdout
    commit_payload = _git(["cat-file", "commit", retained["snapshot_commit"]], repo=repo).stdout
    _require(
        _sha256_bytes(tag_payload) == retained["tag_payload_sha256"]
        and _sha256_bytes(commit_payload) == retained["snapshot_commit_payload_sha256"],
        f"prior release {tag} raw tag or commit payload differs",
    )
    tree, parents = _commit_properties(repo, retained["snapshot_commit"])
    _require(tree == retained["snapshot_tree"] and not parents, f"prior release {tag} snapshot root differs")
    computed_tree = tree_manifest(repo, tree)
    _require(computed_tree["manifest_sha256"] == retained["tree_manifest_sha256"], f"prior release {tag} tree manifest differs")
    inventory = object_inventory(repo, tag_ref)
    inventory = {**inventory, "scope": "reachable-from-release-tag"}
    _require(inventory == retained["reachable_inventory"], f"prior release {tag} reachable inventory differs")
    signed_release = _validate_github_release_attestation_output(
        output=_load_json(attestation_path, f"prior release {tag} signed release attestation"),
        repository_slug=metadata["repository"]["slug"],
        repository_id=metadata["repository"]["id"],
        owner_id=metadata["organization"]["id"],
        tag=tag,
        release_id=live_release["id"],
        tag_object=retained["tag_object"],
        asset_records={item["name"]: (item["size_bytes"], item["sha256"]) for item in manifest_assets},
    )
    _require(signed_release == retained["signed_release"], f"prior release {tag} signed immutable-release authority differs")
    _require(
        retained["release"] == {
            "id": live_release["id"], "node_id": live_release["node_id"],
            "published": True, "prerelease": False, "immutable": True,
            "title": live_release["title"], "body_sha256": _sha256_bytes(live_release["body"].encode("utf-8")),
            "assets": live_assets,
        },
        f"prior release {tag} live immutable release authority differs",
    )
    return retained


def audit_prior_release_repository(
    repo: pathlib.Path,
    *,
    history_attestation: Mapping[str, Any],
    prior_assets_root: pathlib.Path,
    external_audit_path: pathlib.Path,
    trusted_external_audit_sha256: str,
    forbidden_objects: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit the advertised closure of a non-empty release-only target.

    Previous release tags must be annotated and peel to disconnected root
    commits.  An optional branch is accepted only when it is immutable in
    content: ``refs/heads/release/<version>`` must point to one of those exact
    roots.  In particular a convenience ``main`` child which adds workflow or
    Dependabot metadata is not silently grandfathered into the release-only
    history model.
    """

    repository = _regular_directory(repo, "prior release audit repository")
    _verify_no_repository_extensions(repository, allow_unborn_symbolic_head=True)
    history = validate_target_history_attestation(history_attestation)
    _require(history["mode"] == "audited-prior-releases", "prior release audit mode differs")
    audit_record_value = external_audit_record(external_audit_path, trusted_external_audit_sha256)
    audit_authority = validate_external_mirror_audit(_load_json(external_audit_path, "independent mirror audit"))
    _require(
        history["external_audit"] == {key: audit_record_value[key] for key in ("name", "size_bytes", "sha256", "audit_id")},
        "prior release history differs from the out-of-band trusted audit",
    )
    expected = history["advertised_refs"]
    actual_raw = _git(["for-each-ref", "--format=%(refname)"], repo=repository).stdout
    actual = sorted(line.decode("utf-8") for line in actual_raw.splitlines() if line)
    _require(actual == expected, "prior release advertised ref inventory differs")
    tags = [ref for ref in actual if ref.startswith("refs/tags/")]
    branches = [ref for ref in actual if ref.startswith("refs/heads/")]
    _require(len(tags) == len(actual) and not branches and tags, "prior target must contain release tags only")
    tag_targets: dict[str, str] = {}
    for ref in tags:
        name = ref.removeprefix("refs/tags/")
        _require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", name) is not None and name != DEFAULT_TAG, "prior release tag name differs")
        tag_oid = _git_text(["rev-parse", ref], repo=repository)
        _require(_git_text(["cat-file", "-t", tag_oid], repo=repository) == "tag", f"prior release tag is not annotated: {ref}")
        target = _git_text(["rev-parse", f"{ref}^{{}}"], repo=repository)
        _tree, parents = _commit_properties(repository, target)
        _require(not parents and int(_git_text(["rev-list", "--count", target], repo=repository)) == 1, f"prior release tag is not parentless: {ref}")
        tag_targets[ref] = target
    accepted_roots = set(tag_targets.values())
    _require(len(accepted_roots) == len(tags), "prior release tags do not have unique disconnected roots")
    authorities = history["prior_releases"]
    _require([f"refs/tags/{item['tag']}" for item in authorities] == tags, "prior release authority tag union differs")
    _require(
        audit_authority["target_history"]["advertised_refs_sha256"]
        == _canonical_sha256([
            {"object_id": item["tag_object"], "ref": f"refs/tags/{item['tag']}"}
            for item in authorities
        ]),
        "prior release full ref union differs from the out-of-band trusted audit",
    )
    _require(
        audit_authority["target_history"]["prior_releases_sha256"] == _canonical_sha256(authorities),
        "prior release authorities differ from the out-of-band trusted audit",
    )
    for authority in authorities:
        _require(tag_targets[f"refs/tags/{authority['tag']}"] == authority["snapshot_commit"], f"prior release {authority['tag']} root differs")
        replayed_authority = _derive_prior_release_authority(
            repo=repository,
            tag=authority["tag"],
            release_directory=_regular_directory(prior_assets_root, "prior release assets root") / authority["tag"],
        )
        _require(replayed_authority == authority, f"prior release {authority['tag']} live immutable release authority differs")
    reachable_commits_raw = _git(["rev-list", "--all"], repo=repository).stdout
    reachable_commits = {line.decode("ascii") for line in reachable_commits_raw.splitlines() if line}
    _require(reachable_commits == accepted_roots, "prior release reachable commit union differs")
    checked_forbidden: list[str] = []
    for oid in sorted(set(forbidden_objects)):
        checked = _hex40(oid, "prior target forbidden object")
        _require(not _object_exists(repository, checked), f"forbidden development object exists in prior target: {checked}")
        checked_forbidden.append(checked)
    fsck = _git(["fsck", "--full", "--no-reflogs", "--unreachable", *tags], repo=repository, check=False)
    _require(fsck.returncode == 0, "prior release repository fsck failed")
    findings = (fsck.stdout + fsck.stderr).decode("utf-8", "replace").strip()
    non_notice = [line for line in findings.splitlines() if not line.startswith("notice: HEAD points to an unborn branch")]
    _require(not non_notice, "prior release repository has unreachable or corrupt objects")
    audit: dict[str, Any] = {
        "state": "PRIOR_RELEASE_TOPOLOGY_PASS",
        "refs": actual,
        "annotated_parentless_tags": tag_targets,
        "release_branches": [],
        "per_tag_release_authority": [item["tag"] for item in authorities],
        "reachable_commits": sorted(reachable_commits),
        "forbidden_objects_absent": checked_forbidden,
    }
    audit["audit_id"] = _canonical_sha256(audit)
    return audit


def _validate_identity(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    identity = _exact(value, {"name", "email", "epoch", "timezone"}, label)
    _require(
        isinstance(identity["name"], str)
        and identity["name"]
        and all(ord(character) >= 32 for character in identity["name"])
        and not {"<", ">", "\n", "\r", "\0"}.intersection(identity["name"]),
        f"{label} name differs",
    )
    _require(isinstance(identity["email"], str) and re.fullmatch(r"[^<>\s@]+@[^<>\s@]+", identity["email"]) is not None, f"{label} email differs")
    _nonnegative_int(identity["epoch"], f"{label} epoch")
    _require(identity["timezone"] == "+0000", f"{label} timezone must be +0000")
    return dict(identity)


def _identity_environment(identity: Mapping[str, Any]) -> dict[str, str]:
    retained = _validate_identity(identity, "snapshot identity")
    timestamp = f"{retained['epoch']} {retained['timezone']}"
    return {
        "GIT_AUTHOR_NAME": retained["name"],
        "GIT_AUTHOR_EMAIL": retained["email"],
        "GIT_AUTHOR_DATE": timestamp,
        "GIT_COMMITTER_NAME": retained["name"],
        "GIT_COMMITTER_EMAIL": retained["email"],
        "GIT_COMMITTER_DATE": timestamp,
    }


def _snapshot_message(source_commit: str, source_tree: str) -> bytes:
    return (
        "gpmeep v1.1.0 release-only source snapshot\n"
        f"authoritative-source-commit: {source_commit}\n"
        f"authoritative-source-tree: {source_tree}\n"
        f"method: {METHOD}\n"
    ).encode("utf-8")


def _snapshot_commit_payload(
    source_commit: str, source_tree: str, identity: Mapping[str, Any]
) -> bytes:
    retained = _validate_identity(identity, "snapshot identity")
    person = f"{retained['name']} <{retained['email']}> {retained['epoch']} {retained['timezone']}"
    return (
        f"tree {source_tree}\n"
        f"author {person}\n"
        f"committer {person}\n"
        "\n"
    ).encode("utf-8") + _snapshot_message(source_commit, source_tree)


def _git_object_id(kind: str, payload: bytes) -> str:
    _require(kind in {"commit", "tag"}, "unsupported deterministic Git object type")
    return hashlib.sha1(kind.encode("ascii") + b" " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def prepare_snapshot(
    *,
    source_repo: pathlib.Path,
    source_commit: str,
    source_tree: str,
    state_root: pathlib.Path,
    identity: Mapping[str, Any],
    source_tag: str | None = None,
    source_tag_object: str | None = None,
    forbidden_objects: Iterable[str] = (),
) -> dict[str, Any]:
    """Create a root commit and cross an object-isolation bundle boundary."""

    source_repo = _regular_directory(source_repo, "source repository")
    state_root = _regular_directory(state_root, "release-only state root")
    source_commit = _hex40(source_commit, "source commit")
    source_tree = _hex40(source_tree, "source tree")
    _require(_source_tree(source_repo, source_commit) == source_tree, "source commit tree differs from authority")
    tag_before = _tag_state(source_repo, source_tag)
    if source_tag_object is not None:
        _require(tag_before is not None and tag_before[0] == _hex40(source_tag_object, "source tag object"), "source tag object differs before snapshot")
        _require(tag_before[1] == source_commit, "source tag target differs before snapshot")
    source_objects = _repository_object_directory(source_repo)
    _require(b"\n" not in os.fsencode(source_objects), "source object directory contains a newline")
    source_object_identity = (source_objects.stat().st_dev, source_objects.stat().st_ino)
    staging = _new_directory(state_root / "alternate-staging.git", "alternate staging repository")
    _git(["init", "--bare", "--template=", "--object-format=sha1", str(staging)])
    alternate_file = staging / "objects/info/alternates"
    _write_new_bytes(
        alternate_file,
        os.fsencode(source_objects) + b"\n",
        "staging alternate file",
        require_private_parent=False,
    )
    message = _snapshot_message(source_commit, source_tree)
    commit = _git(
        ["commit-tree", source_tree],
        repo=staging,
        input_bytes=message,
        extra_environment=_identity_environment(identity),
    ).stdout.decode("ascii").strip()
    _hex40(commit, "snapshot commit")
    tree, parents = _commit_properties(staging, commit)
    _require(tree == source_tree and not parents, "constructed snapshot is not an identical-tree root commit")
    _git(["update-ref", TRANSPORT_REF, commit, "0" * 40], repo=staging)
    transport_bundle = state_root / "snapshot-transport.bundle"
    _require(not transport_bundle.exists(), "refusing to overwrite snapshot transport bundle")
    _git(["bundle", "create", str(transport_bundle), TRANSPORT_REF], repo=staging)
    isolated = state_root / "isolated-snapshot.git"
    _require(not isolated.exists(), "refusing to reuse isolated snapshot repository")
    _git(["-c", "protocol.file.allow=always", "clone", "--bare", "--template=", str(transport_bundle), str(isolated)])
    _git(["remote", "remove", "origin"], repo=isolated)
    _git(["update-ref", "-d", TRANSPORT_REF, commit], repo=isolated)
    # Detached HEAD prevents fsck's unborn-HEAD notice without adding a ref.
    _git(["update-ref", "--no-deref", "HEAD", commit], repo=isolated)
    # Every development ancestor commit is forbidden automatically.  Extra
    # OIDs let an audit pin deleted, otherwise-unreachable sentinel blobs too.
    development_history = _source_git(source_repo, ["rev-list", source_commit]).stdout
    development_commits = [line.decode("ascii") for line in development_history.splitlines() if line]
    _require(development_commits and development_commits[0] == source_commit, "source history traversal differs")
    forbidden = sorted(set([*forbidden_objects, *development_commits[1:]]))
    isolation = verify_isolated_repository(
        isolated,
        snapshot_commit=commit,
        source_tree=source_tree,
        source_commit=source_commit,
        forbidden_objects=forbidden,
    )
    manifest = tree_manifest(isolated, source_tree)
    dependabot_policy = verify_dependabot_suppression(isolated, source_tree)
    expected_commit_payload = _snapshot_commit_payload(source_commit, source_tree, identity)
    _require(_git_object_id("commit", expected_commit_payload) == commit, "snapshot commit identity is not deterministic")
    _require(_git(["cat-file", "commit", commit], repo=isolated).stdout == expected_commit_payload, "snapshot commit bytes differ")
    _require(_source_tree(source_repo, source_commit) == source_tree, "source commit changed during snapshot construction")
    tag_after = _tag_state(source_repo, source_tag)
    _require(tag_after == tag_before, "authoritative source tag changed during snapshot construction")
    _require((source_objects.stat().st_dev, source_objects.stat().st_ino) == source_object_identity, "source object directory changed during snapshot construction")
    evidence: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "method": METHOD,
        "object_format": "sha1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "snapshot_commit": commit,
        "snapshot_tree": tree,
        "parent_count": 0,
        "reachable_commit_count": 1,
        "identity": _validate_identity(identity, "snapshot identity"),
        "commit_message_sha256": _sha256_bytes(message),
        "tree_manifest_sha256": manifest["manifest_sha256"],
        "reachable_inventory_sha256": isolation["inventory"]["inventory_sha256"],
        "object_counts": isolation["inventory"]["object_counts"],
        "inventory_scope": isolation["inventory"]["scope"],
        "assertions": {
            "source_commit_absent": True,
            "development_parents_absent": True,
            "forbidden_objects_absent": sorted(set(forbidden)),
            "no_alternates": True,
            "no_promisor": True,
            "no_shallow": True,
            "no_grafts": True,
            "no_replace_refs": True,
            "no_unexpected_refs": True,
            "fsck_clean": True,
            "dependabot_pull_requests_suppressed": dependabot_policy == {"github-actions": 0, "pip": 0},
        },
        "tree_manifest": manifest,
        "reachable_inventory": isolation["inventory"],
    }
    evidence["snapshot_id"] = _canonical_sha256(evidence)
    _harden_tree(state_root)
    return validate_snapshot(evidence)


def validate_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "method", "object_format", "source_commit", "source_tree",
        "snapshot_commit", "snapshot_tree", "parent_count", "reachable_commit_count",
        "identity", "commit_message_sha256", "tree_manifest_sha256",
        "reachable_inventory_sha256", "object_counts", "inventory_scope", "assertions",
        "tree_manifest", "reachable_inventory", "snapshot_id",
    }
    retained = _exact(value, expected, "snapshot evidence")
    unsigned = dict(retained)
    snapshot_id = unsigned.pop("snapshot_id")
    _hex64(snapshot_id, "snapshot identity")
    _require(_canonical_sha256(unsigned) == snapshot_id, "snapshot evidence identity differs")
    _require(retained["schema"] == SNAPSHOT_SCHEMA and retained["method"] == METHOD, "snapshot method differs")
    _require(retained["object_format"] == "sha1", "snapshot object format differs")
    source_commit = _hex40(retained["source_commit"], "snapshot source commit")
    source_tree = _hex40(retained["source_tree"], "snapshot source tree")
    _hex40(retained["snapshot_commit"], "snapshot commit")
    _require(retained["snapshot_tree"] == source_tree, "snapshot tree identity differs")
    _require(retained["parent_count"] == 0 and retained["reachable_commit_count"] == 1, "snapshot history isolation differs")
    _validate_identity(retained["identity"], "snapshot identity")
    _hex64(retained["commit_message_sha256"], "snapshot message digest")
    manifest = _exact(retained["tree_manifest"], {"object_format", "tree", "entries", "manifest_sha256"}, "tree manifest")
    manifest_unsigned = dict(manifest)
    manifest_id = manifest_unsigned.pop("manifest_sha256")
    _require(_canonical_sha256(manifest_unsigned) == manifest_id == retained["tree_manifest_sha256"], "tree manifest identity differs")
    _require(manifest["tree"] == source_tree and manifest["object_format"] == "sha1", "tree manifest authority differs")
    _require(isinstance(manifest["entries"], list) and manifest["entries"], "tree manifest is empty")
    encoded_paths: list[str] = []
    for entry in manifest["entries"]:
        _exact(entry, {"path_base64", "mode", "git_type", "object_id", "size_bytes", "blob_sha256"}, "tree entry")
        try:
            path_bytes = base64.b64decode(entry["path_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise ReleaseOnlyError("tree entry path encoding differs") from error
        _require(bool(path_bytes), "tree entry path is empty")
        encoded_paths.append(entry["path_base64"])
        _hex40(entry["object_id"], "tree entry object")
        _require(entry["git_type"] in {"blob", "tree"} and entry["mode"] != "160000", "tree entry type differs")
        if entry["git_type"] == "blob":
            _nonnegative_int(entry["size_bytes"], "tree blob size")
            _hex64(entry["blob_sha256"], "tree blob digest")
        else:
            _require(entry["size_bytes"] is None and entry["blob_sha256"] is None, "tree record payload differs")
    _require(len(encoded_paths) == len(set(encoded_paths)), "tree manifest contains duplicate paths")
    inventory = _exact(retained["reachable_inventory"], {"scope", "objects", "object_counts", "inventory_sha256"}, "reachable inventory")
    _require(inventory["scope"] == retained["inventory_scope"] == "reachable-from-snapshot-commit", "inventory scope differs")
    _require(_canonical_sha256(inventory["objects"]) == inventory["inventory_sha256"] == retained["reachable_inventory_sha256"], "reachable inventory identity differs")
    _require(inventory["object_counts"] == retained["object_counts"], "reachable object counts differ")
    counts = _exact(inventory["object_counts"], {"blob", "commit", "tag", "tree"}, "reachable object counts")
    _require(all(type(item) is int and item >= 0 for item in counts.values()), "reachable object count is invalid")
    _require(isinstance(inventory["objects"], list) and inventory["objects"], "reachable object list is empty")
    inventory_oids: list[str] = []
    replayed_counts = {"blob": 0, "commit": 0, "tag": 0, "tree": 0}
    for record in inventory["objects"]:
        _exact(record, {"object_id", "git_type", "size_bytes"}, "reachable object record")
        inventory_oids.append(_hex40(record["object_id"], "reachable object identity"))
        _require(record["git_type"] in replayed_counts, "reachable object type differs")
        replayed_counts[record["git_type"]] += 1
        _nonnegative_int(record["size_bytes"], "reachable object size")
    _require(inventory_oids == sorted(set(inventory_oids)), "reachable object inventory order or uniqueness differs")
    _require(replayed_counts == counts, "reachable object counts do not replay")
    _require(source_commit != retained["snapshot_commit"], "source authority was substituted by snapshot")
    assertions = _exact(
        retained["assertions"],
        {"source_commit_absent", "development_parents_absent", "forbidden_objects_absent", "no_alternates", "no_promisor", "no_shallow", "no_grafts", "no_replace_refs", "no_unexpected_refs", "fsck_clean", "dependabot_pull_requests_suppressed"},
        "snapshot assertions",
    )
    _require(all(assertions[key] is True for key in assertions if key != "forbidden_objects_absent"), "snapshot assertion is not true")
    _require(isinstance(assertions["forbidden_objects_absent"], list), "forbidden object assertion differs")
    for oid in assertions["forbidden_objects_absent"]:
        _hex40(oid, "forbidden object assertion")
    _require(assertions["forbidden_objects_absent"] == sorted(set(assertions["forbidden_objects_absent"])), "forbidden object assertions are not canonical")
    return dict(retained)


def replay_snapshot(snapshot: Mapping[str, Any], snapshot_repo: pathlib.Path) -> dict[str, Any]:
    retained = validate_snapshot(snapshot)
    repository = _regular_directory(snapshot_repo, "isolated snapshot repository")
    manifest = tree_manifest(repository, retained["source_tree"])
    _require(manifest == retained["tree_manifest"], "tree content manifest does not replay")
    expected_commit = _snapshot_commit_payload(
        retained["source_commit"], retained["source_tree"], retained["identity"]
    )
    _require(_git_object_id("commit", expected_commit) == retained["snapshot_commit"], "deterministic snapshot commit identity differs")
    _require(_git(["cat-file", "commit", retained["snapshot_commit"]], repo=repository).stdout == expected_commit, "snapshot commit bytes do not replay")
    refs_raw = _git(["for-each-ref", "--format=%(refname)"], repo=repository).stdout
    refs = sorted(line.decode("utf-8") for line in refs_raw.splitlines() if line)
    # A resumed controller may replay the intermediate repository after the
    # deterministic final tag has already been sealed.  That exact tag is the
    # only permissible post-snapshot ref; the receipt inventory remains scoped
    # to the snapshot commit and therefore cannot acquire a tag hash cycle.
    tag_name = DEFAULT_TAG if refs == [f"refs/tags/{DEFAULT_TAG}"] else None
    isolation = verify_isolated_repository(
        repository,
        snapshot_commit=retained["snapshot_commit"],
        source_tree=retained["source_tree"],
        source_commit=retained["source_commit"],
        tag_name=tag_name,
        forbidden_objects=retained["assertions"]["forbidden_objects_absent"],
    )
    replayed_inventory = object_inventory(repository, retained["snapshot_commit"])
    _require(replayed_inventory == retained["reachable_inventory"], "snapshot inventory replay differs")
    return retained


def _run_pinned_final_verifier(
    *,
    verifier_path: pathlib.Path,
    expected_verifier_sha256: str,
    verifier_arguments: Sequence[str],
    final_receipt_path: pathlib.Path,
    package_path: pathlib.Path,
    candidate_receipt_path: pathlib.Path,
) -> dict[str, Any]:
    verifier = _regular_file(verifier_path, "private final receipt verifier")
    expected_digest = _hex64(expected_verifier_sha256, "private final verifier digest")
    _require(_sha256_file(verifier) == expected_digest, "private final verifier bytes differ")
    fixed_paths = [
        _regular_file(final_receipt_path, "private final receipt"),
        _regular_file(package_path, "private package"),
        _regular_file(candidate_receipt_path, "private candidate receipt"),
    ]
    forbidden = {"--create", "--output", "--verify", "--receipt", "--package", "--candidate-receipt"}
    _require(
        all(isinstance(item, str) and item and "\0" not in item for item in verifier_arguments),
        "private final verifier arguments differ",
    )
    _require(not any(item in forbidden for item in verifier_arguments), "private final verifier arguments override fixed authority inputs")
    before = {path: (_file_identity(path), _sha256_file(path)) for path in [verifier, *fixed_paths]}
    command = [
        sys.executable,
        "-I",
        str(verifier),
        *verifier_arguments,
        "--verify",
        "--receipt",
        str(fixed_paths[0]),
        "--package",
        str(fixed_paths[1]),
        "--candidate-receipt",
        str(fixed_paths[2]),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_git_environment(),
        timeout=1800,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReleaseOnlyError(f"pinned private final verifier rejected the authority chain: {detail}")
    try:
        terminal = json.loads(
            result.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReleaseOnlyError("private final verifier terminal is malformed") from error
    _exact(terminal, {"schema", "state", "receipt_id", "package_sha256"}, "private final verifier terminal")
    _require(
        terminal["schema"] == "gpmeep-v110-final-release-v1" and terminal["state"] == "PASS",
        "private final verifier did not produce the exact PASS terminal",
    )
    _hex64(terminal["receipt_id"], "private final verifier receipt identity")
    _hex64(terminal["package_sha256"], "private final verifier package digest")
    after = {path: (_file_identity(path), _sha256_file(path)) for path in [verifier, *fixed_paths]}
    _require(before == after, "private final verifier or fixed authority inputs changed during replay")
    return {
        "method": "pinned-final-verifier-replay-v1",
        "verifier_name": verifier.name,
        "verifier_sha256": expected_digest,
        "arguments_sha256": _canonical_sha256(list(verifier_arguments)),
        "terminal": terminal,
    }


def _expected_private_tag_message(
    *, source_commit: str, package_sha256: str, candidate_id: str, final_id: str, manifest_id: str
) -> str:
    return (
        f"gpmeep {DEFAULT_TAG}\n"
        f"source-commit: {source_commit}\n"
        f"package-sha256: {package_sha256}\n"
        f"candidate-receipt-id: {candidate_id}\n"
        f"final-receipt-id: {final_id}\n"
        f"asset-manifest-id: {manifest_id}"
    )


def _verify_private_source_tag(source_repo: pathlib.Path, source: Mapping[str, Any]) -> None:
    repository = _regular_directory(source_repo, "private source repository")
    state = _tag_state(repository, DEFAULT_TAG)
    _require(state is not None, "private source tag is absent")
    _require(state == (source["source_tag"]["object_id"], source["source_commit"]), "private source tag identity differs")
    raw = _source_git(repository, ["cat-file", "tag", state[0]]).stdout
    header, separator, body = raw.partition(b"\n\n")
    _require(separator == b"\n\n" and body.endswith(b"\n"), "private source annotated tag bytes are malformed")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, space, value = line.partition(b" ")
        name = key.decode("ascii")
        _require(space == b" " and name not in fields, "private source tag header differs")
        fields[name] = value.decode("utf-8")
    _require(set(fields) == {"object", "type", "tag", "tagger"}, "private source tag header schema differs")
    _require(
        fields["object"] == source["source_commit"]
        and fields["type"] == "commit"
        and fields["tag"] == DEFAULT_TAG
        and fields["tagger"],
        "private source tag header authority differs",
    )
    expected = _expected_private_tag_message(
        source_commit=source["source_commit"],
        package_sha256=source["package"]["sha256"],
        candidate_id=source["candidate_receipt"]["receipt_id"],
        final_id=source["final_receipt"]["receipt_id"],
        manifest_id=source["source_asset_manifest"]["manifest_id"],
    )
    _require(body == (expected + "\n").encode("utf-8"), "private source annotated tag message differs from source publisher authority")


def validate_source_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "repository", "source_tag", "source_commit", "source_tree", "final_receipt", "candidate_receipt", "package", "source_asset_manifest", "private_release", "verification"},
        "source authority",
    )
    _require(retained["schema"] == SOURCE_AUTHORITY_SCHEMA, "source authority schema differs")
    repository = _exact(retained["repository"], {"slug", "id", "node_id"}, "source repository identity")
    _require(repository["slug"] == "iverydj/gpmeep" and SAFE_SLUG.fullmatch(repository["slug"]), "source repository slug differs")
    _positive_int(repository["id"], "source repository numeric ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "source repository node ID differs")
    tag = _exact(retained["source_tag"], {"name", "object_id"}, "source tag authority")
    _require(tag["name"] == DEFAULT_TAG, "source tag name differs")
    _hex40(tag["object_id"], "source tag object")
    source_commit = _hex40(retained["source_commit"], "authority source commit")
    _hex40(retained["source_tree"], "authority source tree")
    for key in ("final_receipt", "candidate_receipt"):
        record = _exact(retained[key], {"name", "size_bytes", "sha256", "receipt_id"}, key)
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), f"{key} name differs")
        _positive_int(record["size_bytes"], f"{key} size")
        _hex64(record["sha256"], f"{key} digest")
        _hex64(record["receipt_id"], f"{key} identity")
    package = _exact(retained["package"], {"name", "size_bytes", "sha256", "source_commit", "conda_metadata"}, "package authority")
    _require(isinstance(package["name"], str) and package["name"].endswith(".conda") and SAFE_NAME.fullmatch(package["name"]), "package name differs")
    _positive_int(package["size_bytes"], "package size")
    _hex64(package["sha256"], "package digest")
    _require(package["source_commit"] == source_commit, "package source commit differs")
    _require(isinstance(package["conda_metadata"], dict) and package["conda_metadata"], "package Conda metadata is absent")
    manifest = _exact(retained["source_asset_manifest"], {"name", "size_bytes", "sha256", "manifest_id"}, "source asset manifest")
    _require(isinstance(manifest["name"], str) and SAFE_NAME.fullmatch(manifest["name"]), "source asset manifest name differs")
    _positive_int(manifest["size_bytes"], "source asset manifest size")
    _hex64(manifest["sha256"], "source asset manifest digest")
    _hex64(manifest["manifest_id"], "source asset manifest identity")
    release = _exact(retained["private_release"], {"id", "node_id", "published", "prerelease", "immutable"}, "private source release")
    _positive_int(release["id"], "private source release ID")
    _require(isinstance(release["node_id"], str) and release["node_id"], "private source release node ID differs")
    _require(
        release["published"] is True and release["prerelease"] is False and release["immutable"] is True,
        "private source release is not a final published immutable release",
    )
    verification = _exact(
        retained["verification"],
        {"method", "verifier_name", "verifier_sha256", "arguments_sha256", "terminal"},
        "private source authority verification",
    )
    _require(verification["method"] == "pinned-final-verifier-replay-v1", "private source verification method differs")
    _require(isinstance(verification["verifier_name"], str) and SAFE_NAME.fullmatch(verification["verifier_name"]), "private final verifier name differs")
    _hex64(verification["verifier_sha256"], "private final verifier digest")
    _hex64(verification["arguments_sha256"], "private final verifier arguments identity")
    terminal = _exact(verification["terminal"], {"schema", "state", "receipt_id", "package_sha256"}, "private final verifier terminal")
    _require(
        terminal == {
            "schema": "gpmeep-v110-final-release-v1",
            "state": "PASS",
            "receipt_id": retained["final_receipt"]["receipt_id"],
            "package_sha256": retained["package"]["sha256"],
        },
        "private final verifier terminal differs from source authority",
    )
    return dict(retained)


def verify_source_authority_files(
    source_authority: Mapping[str, Any],
    asset_directory: pathlib.Path,
    *,
    source_repo: pathlib.Path,
    final_verifier_path: pathlib.Path,
    final_verifier_sha256: str,
    final_verifier_arguments: Sequence[str],
    trusted_source_tag_object: str,
) -> dict[str, Any]:
    """Bind mirror metadata to the immutable private-release bytes.

    The full private final/candidate verifiers remain the source-release
    authority.  This replay additionally proves that the mirror descriptor did
    not substitute their IDs, source commit, package metadata, or asset bytes.
    """

    retained = validate_source_authority(source_authority)
    _require(
        retained["source_tag"]["object_id"] == _hex40(trusted_source_tag_object, "trusted private source tag object"),
        "private source tag differs from the out-of-band trusted object",
    )
    root = _regular_directory(asset_directory, "private release asset directory")
    paths: dict[str, pathlib.Path] = {}
    for key in ("final_receipt", "candidate_receipt", "package", "source_asset_manifest"):
        record = retained[key]
        path = _regular_file(root / record["name"], f"source authority {key}")
        _require(path.parent == root, f"source authority {key} escaped the asset directory")
        _require(path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"], f"source authority {key} bytes differ")
        paths[key] = path

    candidate = _load_json(paths["candidate_receipt"], "private candidate receipt")
    candidate_unsigned = dict(candidate)
    candidate_id = candidate_unsigned.pop("receipt_id", None)
    _require(_canonical_sha256(candidate_unsigned) == candidate_id == retained["candidate_receipt"]["receipt_id"], "private candidate receipt identity differs")
    _require(candidate.get("source_commit") == retained["source_commit"], "private candidate source commit differs")

    final = _load_json(paths["final_receipt"], "private final receipt")
    final_unsigned = dict(final)
    final_id = final_unsigned.pop("receipt_id", None)
    _require(_canonical_sha256(final_unsigned) == final_id == retained["final_receipt"]["receipt_id"], "private final receipt identity differs")
    _require(final.get("schema") == "gpmeep-v110-final-release-v1" and final.get("state") == "RELEASE_AUTHORIZED", "private final receipt authority differs")
    _require(final.get("tag_intent") == {"name": DEFAULT_TAG, "target_commit": retained["source_commit"]}, "private final tag intent differs")
    candidate_authority = final.get("candidate_authority")
    _require(isinstance(candidate_authority, dict) and candidate_authority.get("receipt_id") == candidate_id and candidate_authority.get("source_commit") == retained["source_commit"], "private final candidate authority differs")
    final_package = final.get("package")
    _require(isinstance(final_package, dict), "private final package authority is absent")
    expected_package = retained["package"]
    _require(final_package.get("file_name") == expected_package["name"], "private final package name differs")
    _require(final_package.get("size_bytes") == expected_package["size_bytes"] and final_package.get("sha256") == expected_package["sha256"], "private final package bytes differ")
    conda_metadata = dict(final_package)
    for key in ("file_name", "size_bytes", "sha256"):
        conda_metadata.pop(key, None)
    _require(conda_metadata == expected_package["conda_metadata"], "private final Conda metadata differs")
    _require(conda_metadata.get("embedded_candidate_receipt_id") == candidate_id, "private package candidate receipt identity differs")

    source_manifest = _load_json(paths["source_asset_manifest"], "private source asset manifest")
    _exact(source_manifest, {"schema", "assets", "manifest_id"}, "private source asset manifest")
    source_unsigned = dict(source_manifest)
    source_manifest_id = source_unsigned.pop("manifest_id")
    _require(source_manifest["schema"] == "gpmeep-v110-release-asset-manifest-v1" and _canonical_sha256(source_unsigned) == source_manifest_id == retained["source_asset_manifest"]["manifest_id"], "private source asset manifest identity differs")
    _require(isinstance(source_manifest["assets"], list) and source_manifest["assets"], "private source asset inventory is empty")
    roles: set[str] = set()
    names: set[str] = set()
    manifest_by_role: dict[str, dict[str, Any]] = {}
    for record in source_manifest["assets"]:
        _exact(record, {"role", "name", "size_bytes", "sha256"}, "private source asset record")
        role, name = record["role"], record["name"]
        _require(isinstance(role, str) and SAFE_NAME.fullmatch(role) and isinstance(name, str) and SAFE_NAME.fullmatch(name), "private source asset identity differs")
        _require(role not in roles and name not in names, "private source asset inventory contains duplicates")
        roles.add(role)
        names.add(name)
        _nonnegative_int(record["size_bytes"], "private source asset size")
        _hex64(record["sha256"], "private source asset digest")
        asset = _regular_file(root / name, f"private source asset {name}")
        _require(asset.parent == root and asset.stat().st_size == record["size_bytes"] and _sha256_file(asset) == record["sha256"], f"private source asset bytes differ: {name}")
        manifest_by_role[role] = record
    _require([record["role"] for record in source_manifest["assets"]] == sorted(roles), "private source asset inventory order differs")
    for role, key in (("candidate_receipt", "candidate_receipt"), ("final_receipt", "final_receipt"), ("package", "package")):
        record = manifest_by_role.get(role)
        authority = retained[key]
        _require(record is not None and record["name"] == authority["name"] and record["size_bytes"] == authority["size_bytes"] and record["sha256"] == authority["sha256"], f"private source manifest {role} authority differs")
    replay = _run_pinned_final_verifier(
        verifier_path=final_verifier_path,
        expected_verifier_sha256=final_verifier_sha256,
        verifier_arguments=final_verifier_arguments,
        final_receipt_path=paths["final_receipt"],
        package_path=paths["package"],
        candidate_receipt_path=paths["candidate_receipt"],
    )
    _require(replay == retained["verification"], "private final verifier replay differs from source authority")
    _verify_private_source_tag(source_repo, retained)
    _require(_source_tree(source_repo, retained["source_commit"]) == retained["source_tree"], "private source tree differs from source authority")
    return retained


def verify_private_live_assets(release_metadata, source_manifest, manifest_path):
    """Bind every uploaded private asset, including its manifest, to local bytes."""
    _exact(source_manifest, {"schema", "assets", "manifest_id"}, "private source asset manifest")
    unsigned = dict(source_manifest)
    identity = unsigned.pop("manifest_id")
    _require(source_manifest["schema"] == "gpmeep-v110-release-asset-manifest-v1" and
             _canonical_sha256(unsigned) == identity, "private source asset manifest identity differs")
    assets = source_manifest["assets"]
    _require(isinstance(assets, list) and assets, "private source asset manifest inventory is empty")
    expected = {}
    for record in assets:
        _exact(record, {"role", "name", "size_bytes", "sha256"}, "private source asset record")
        name = record["name"]
        _require(isinstance(name, str) and SAFE_NAME.fullmatch(name) and name not in expected,
                 "private source asset manifest names duplicate or invalid")
        _nonnegative_int(record["size_bytes"], "private source asset size")
        _hex64(record["sha256"], "private source asset digest")
        expected[name] = {"size_bytes": record["size_bytes"], "sha256": record["sha256"]}
    manifest_path = _regular_file(manifest_path, "private source asset manifest")
    _require(manifest_path.name not in expected, "private source manifest lists itself")
    expected[manifest_path.name] = {"size_bytes": manifest_path.stat().st_size,
                                    "sha256": _sha256_file(manifest_path)}
    live = release_metadata.get("assets")
    _require(isinstance(live, list), "private source release live asset inventory differs")
    seen = set()
    for item in live:
        _require(isinstance(item, dict) and isinstance(item.get("name"), str) and
                 item["name"] in expected and item["name"] not in seen,
                 "private source release live asset inventory differs")
        name = item["name"]
        seen.add(name)
        record = expected[name]
        _require(type(item.get("size")) is int and item["size"] == record["size_bytes"] and
                 item.get("digest") == "sha256:" + record["sha256"] and item.get("state") == "uploaded",
                 "private source release live asset bytes or state differ")
    _require(seen == set(expected), "private source release live asset inventory differs")
    return expected


def create_source_authority(
    *,
    source_repo: pathlib.Path,
    asset_directory: pathlib.Path,
    final_receipt_name: str,
    candidate_receipt_name: str,
    package_name: str,
    source_manifest_name: str,
    private_repository_metadata_path: pathlib.Path,
    private_release_metadata_path: pathlib.Path,
    final_verifier_path: pathlib.Path,
    final_verifier_sha256: str,
    final_verifier_arguments: Sequence[str],
    trusted_source_tag_object: str,
) -> dict[str, Any]:
    root = _regular_directory(asset_directory, "private release asset directory")
    names = [final_receipt_name, candidate_receipt_name, package_name, source_manifest_name]
    _require(all(SAFE_NAME.fullmatch(name) for name in names) and len(set(names)) == len(names), "private release asset names differ")
    final_path, candidate_path, package_path, manifest_path = [
        _regular_file(root / name, f"private release asset {name}") for name in names
    ]
    stable_paths = [
        final_path, candidate_path, package_path, manifest_path,
        private_repository_metadata_path, private_release_metadata_path, final_verifier_path,
    ]
    stable_before = _stable_file_snapshot(stable_paths, "source authority creator input")
    source_refs_before = _source_git(source_repo, ["for-each-ref", "--format=%(objectname)%09%(refname)"]).stdout
    repository_metadata = _load_json(private_repository_metadata_path, "private repository metadata")
    _require(
        repository_metadata.get("full_name") == "iverydj/gpmeep"
        and type(repository_metadata.get("id")) is int
        and repository_metadata["id"] > 0
        and isinstance(repository_metadata.get("node_id"), str)
        and repository_metadata["node_id"]
        and repository_metadata.get("private") is True,
        "private repository live metadata differs",
    )
    release_metadata = _load_json(private_release_metadata_path, "private source release metadata")
    _require(
        release_metadata.get("tag_name") == DEFAULT_TAG
        and release_metadata.get("draft") is False
        and release_metadata.get("prerelease") is False
        and release_metadata.get("immutable") is True
        and type(release_metadata.get("id")) is int
        and release_metadata["id"] > 0
        and isinstance(release_metadata.get("node_id"), str)
        and release_metadata["node_id"],
        "private source release live metadata differs",
    )
    live_manifest = _load_json(manifest_path, "private source asset manifest")
    live_records = verify_private_live_assets(release_metadata, live_manifest, manifest_path)
    _require(set(names).issubset(live_records), "private source asset manifest omits authority files")
    verification = _run_pinned_final_verifier(
        verifier_path=final_verifier_path,
        expected_verifier_sha256=final_verifier_sha256,
        verifier_arguments=final_verifier_arguments,
        final_receipt_path=final_path,
        package_path=package_path,
        candidate_receipt_path=candidate_path,
    )
    final = _load_json(final_path, "private final receipt")
    candidate = _load_json(candidate_path, "private candidate receipt")
    source_manifest = _load_json(manifest_path, "private source asset manifest")
    final_id = verification["terminal"]["receipt_id"]
    candidate_id = candidate.get("receipt_id")
    source_commit = final.get("tag_intent", {}).get("target_commit")
    _hex64(candidate_id, "private candidate receipt identity")
    _hex40(source_commit, "private source commit")
    source_tree = _source_tree(source_repo, source_commit)
    tag_state = _tag_state(source_repo, DEFAULT_TAG)
    _require(
        tag_state is not None
        and tag_state[0] == _hex40(trusted_source_tag_object, "trusted private source tag object")
        and tag_state[1] == source_commit,
        "private source tag differs from the out-of-band trusted object",
    )
    final_package = final.get("package")
    _require(isinstance(final_package, dict), "private final package authority is absent")
    source_manifest_unsigned = dict(source_manifest)
    source_manifest_id = source_manifest_unsigned.pop("manifest_id", None)
    _require(_canonical_sha256(source_manifest_unsigned) == source_manifest_id, "private source manifest identity differs")
    def record(path: pathlib.Path, identity_name: str, identity: str) -> dict[str, Any]:
        return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path), identity_name: identity}
    conda_metadata = dict(final_package)
    for key in ("file_name", "size_bytes", "sha256"):
        conda_metadata.pop(key, None)
    authority: dict[str, Any] = {
        "schema": SOURCE_AUTHORITY_SCHEMA,
        "repository": {
            "slug": repository_metadata["full_name"],
            "id": repository_metadata["id"],
            "node_id": repository_metadata["node_id"],
        },
        "source_tag": {"name": DEFAULT_TAG, "object_id": tag_state[0]},
        "source_commit": source_commit,
        "source_tree": source_tree,
        "final_receipt": record(final_path, "receipt_id", final_id),
        "candidate_receipt": record(candidate_path, "receipt_id", candidate_id),
        "package": {
            "name": package_path.name,
            "size_bytes": package_path.stat().st_size,
            "sha256": _sha256_file(package_path),
            "source_commit": source_commit,
            "conda_metadata": conda_metadata,
        },
        "source_asset_manifest": record(manifest_path, "manifest_id", source_manifest_id),
        "private_release": {
            "id": release_metadata["id"],
            "node_id": release_metadata["node_id"],
            "published": True,
            "prerelease": False,
            "immutable": True,
        },
        "verification": verification,
    }
    verify_source_authority_files(
        authority,
        root,
        source_repo=source_repo,
        final_verifier_path=final_verifier_path,
        final_verifier_sha256=final_verifier_sha256,
        final_verifier_arguments=final_verifier_arguments,
        trusted_source_tag_object=trusted_source_tag_object,
    )
    _require(
        _stable_file_snapshot(stable_paths, "source authority creator input") == stable_before
        and _source_git(source_repo, ["for-each-ref", "--format=%(objectname)%09%(refname)"]).stdout == source_refs_before,
        "source authority inputs changed during creation",
    )
    return authority


def validate_external_mirror_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {
            "schema", "state", "source_commit", "source_tree", "snapshot_id",
            "snapshot_commit", "source_tag_object", "source_repository", "private_release",
            "target_repository", "organization_access", "implementation",
            "github_cli", "source_verifier_sha256", "target_history", "findings", "auditor", "audit_id",
        },
        "independent mirror audit",
    )
    unsigned = dict(retained)
    audit_id = unsigned.pop("audit_id")
    _require(_canonical_sha256(unsigned) == _hex64(audit_id, "independent mirror audit identity"), "independent mirror audit identity differs")
    _require(retained["schema"] == EXTERNAL_AUDIT_SCHEMA and retained["state"] == "GO", "independent mirror audit is not GO")
    for key in ("source_commit", "source_tree", "snapshot_commit", "source_tag_object"):
        _hex40(retained[key], f"independent mirror audit {key}")
    _hex64(retained["snapshot_id"], "independent mirror audit snapshot identity")
    _hex64(retained["source_verifier_sha256"], "independent mirror audit source verifier digest")
    source_repository = _exact(retained["source_repository"], {"slug", "id", "node_id"}, "independent mirror audit source repository")
    _require(source_repository["slug"] == "iverydj/gpmeep", "independent mirror audit source repository slug differs")
    _positive_int(source_repository["id"], "independent mirror audit source repository ID")
    _require(isinstance(source_repository["node_id"], str) and source_repository["node_id"], "independent mirror audit source repository node ID differs")
    private_release = _exact(retained["private_release"], {"id", "node_id", "tag", "published", "prerelease", "immutable"}, "independent mirror audit private release")
    _positive_int(private_release["id"], "independent mirror audit private release ID")
    _require(
        isinstance(private_release["node_id"], str) and private_release["node_id"]
        and private_release["tag"] == DEFAULT_TAG and private_release["published"] is True
        and private_release["prerelease"] is False and private_release["immutable"] is True,
        "independent mirror audit private release differs",
    )
    repository = _exact(retained["target_repository"], {"slug", "id", "node_id"}, "independent mirror audit target")
    _require(repository["slug"] == TARGET_REPOSITORY, "independent mirror audit target slug differs")
    _positive_int(repository["id"], "independent mirror audit target ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "independent mirror audit target node ID differs")
    organization = _exact(retained["organization_access"], {"organization", "id", "node_id", "default_repository_permission", "all_members_can_read"}, "independent mirror audit organization access")
    _require(
        organization["organization"] == "example-organization"
        and type(organization["id"]) is int and organization["id"] > 0
        and isinstance(organization["node_id"], str) and organization["node_id"]
        and organization["default_repository_permission"] in {"read", "write", "admin"}
        and organization["all_members_can_read"] is True,
        "independent mirror audit organization member access differs",
    )
    implementation = _exact(retained["implementation"], {"receipt_tool_sha256", "publisher_sha256", "tests_sha256"}, "audited mirror implementation")
    for key, digest in implementation.items():
        _hex64(digest, f"independent mirror audit {key}")
    github_cli = _exact(retained["github_cli"], {"version", "sha256"}, "audited GitHub CLI")
    _require(
        github_cli == {
            "version": AUDITED_GITHUB_CLI_VERSION,
            "sha256": AUDITED_GITHUB_CLI_SHA256,
        },
        "independent mirror audit GitHub CLI authority differs",
    )
    history = _exact(
        retained["target_history"],
        {"never_imported_or_transferred", "advertised_refs_sha256", "prior_releases_sha256"},
        "independent mirror target history",
    )
    _require(history["never_imported_or_transferred"] is True, "independent mirror audit did not prove clean target lifecycle")
    _hex64(history["advertised_refs_sha256"], "independent mirror audited ref inventory")
    _hex64(history["prior_releases_sha256"], "independent mirror audited prior release authorities")
    findings = _exact(retained["findings"], {"open_critical", "open_high", "open_medium"}, "independent mirror audit findings")
    _require(findings == {"open_critical": 0, "open_high": 0, "open_medium": 0}, "independent mirror audit has open findings")
    auditor = _exact(retained["auditor"], {"independent", "name", "completed_at_utc"}, "independent mirror auditor")
    _require(auditor["independent"] is True and isinstance(auditor["name"], str) and auditor["name"], "mirror auditor independence differs")
    _require(isinstance(auditor["completed_at_utc"], str) and auditor["completed_at_utc"].endswith("Z"), "mirror audit completion time differs")
    return dict(retained)


def external_audit_record(path: pathlib.Path, trusted_sha256: str) -> dict[str, Any]:
    audit_path = _regular_file(path, "independent mirror audit")
    trusted_digest = _hex64(trusted_sha256, "trusted independent mirror audit digest")
    _require(
        stat.S_IMODE(audit_path.stat().st_mode) == 0o600,
        "independent mirror audit mode must be 0600",
    )
    _require(_sha256_file(audit_path) == trusted_digest, "independent mirror audit differs from the out-of-band trusted digest")
    audit = validate_external_mirror_audit(_load_json(audit_path, "independent mirror audit"))
    return {
        "name": audit_path.name,
        "size_bytes": audit_path.stat().st_size,
        "sha256": _sha256_file(audit_path),
        "audit_id": audit["audit_id"],
        "source_commit": audit["source_commit"],
        "source_tree": audit["source_tree"],
        "snapshot_id": audit["snapshot_id"],
        "snapshot_commit": audit["snapshot_commit"],
        "final_verifier_sha256": audit["source_verifier_sha256"],
    }


def verify_audited_implementation(audit: Mapping[str, Any]) -> None:
    retained = validate_external_mirror_audit(audit)
    audited_tool = _regular_file(AUDITED_RECEIPT_TOOL_PATH, "release-only receipt tool")
    script_directory = audited_tool.resolve(strict=True).parent
    expected_paths = {
        "receipt_tool_sha256": audited_tool,
        "publisher_sha256": _regular_file(script_directory / "publish-v110-release-only.sh", "release-only publisher"),
        "tests_sha256": _regular_file(script_directory / "tests/test_v110_release_only_mirror.py", "release-only tests"),
    }
    for key, path in expected_paths.items():
        _require(_sha256_file(path) == retained["implementation"][key], f"audited implementation bytes changed: {key}")


def verify_external_mirror_audit(
    target_preflight: Mapping[str, Any],
    *,
    asset_directory: pathlib.Path,
    source_authority: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    trusted_sha256: str,
) -> dict[str, Any]:
    target = validate_target_preflight(target_preflight)
    source = validate_source_authority(source_authority)
    snap = validate_snapshot(snapshot)
    expected = target["external_mirror_audit"]
    root = _regular_directory(asset_directory, "release asset directory")
    path = _regular_file(root / expected["name"], "independent mirror audit asset")
    _require(path.parent == root and external_audit_record(path, trusted_sha256) == expected, "independent mirror audit asset differs")
    audit = validate_external_mirror_audit(_load_json(path, "independent mirror audit"))
    verify_audited_implementation(audit)
    _require(
        audit["source_commit"] == source["source_commit"]
        and audit["source_tree"] == source["source_tree"]
        and audit["snapshot_id"] == snap["snapshot_id"]
        and audit["snapshot_commit"] == snap["snapshot_commit"],
        "independent mirror audit source or snapshot authority differs",
    )
    _require(audit["target_repository"] == target["repository"], "independent mirror audit target authority differs")
    _require(audit["source_repository"] == source["repository"], "independent mirror audit source repository authority differs")
    _require(audit["source_tag_object"] == source["source_tag"]["object_id"], "independent mirror audit source tag authority differs")
    _require(
        audit["private_release"] == {
            "id": source["private_release"]["id"],
            "node_id": source["private_release"]["node_id"],
            "tag": DEFAULT_TAG,
            "published": source["private_release"]["published"],
            "prerelease": source["private_release"]["prerelease"],
            "immutable": source["private_release"]["immutable"],
        },
        "independent mirror audit private release authority differs",
    )
    _require(audit["organization_access"] == target["organization_access"], "independent mirror audit organization authority differs")
    _require(audit["source_verifier_sha256"] == source["verification"]["verifier_sha256"], "independent mirror audit source verifier differs")
    return audit


def _validate_asset_record(value: Mapping[str, Any], label: str, *, identity_key: str) -> dict[str, Any]:
    retained = _exact(value, {"name", "size_bytes", "sha256", identity_key}, label)
    _require(isinstance(retained["name"], str) and SAFE_NAME.fullmatch(retained["name"]), f"{label} name differs")
    _positive_int(retained["size_bytes"], f"{label} size")
    _hex64(retained["sha256"], f"{label} digest")
    _hex64(retained[identity_key], f"{label} identity")
    return dict(retained)


def validate_prior_release_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {
            "tag", "tag_object", "snapshot_commit", "snapshot_tree",
            "tag_payload_sha256", "snapshot_commit_payload_sha256",
            "tree_manifest_sha256", "reachable_inventory", "mirror_receipt",
            "distribution_manifest", "release", "signed_release",
        },
        "prior release authority",
    )
    _require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", retained["tag"] or "") is not None and retained["tag"] != DEFAULT_TAG, "prior release version differs")
    for key in ("tag_object", "snapshot_commit", "snapshot_tree"):
        _hex40(retained[key], f"prior release {key}")
    for key in ("tag_payload_sha256", "snapshot_commit_payload_sha256", "tree_manifest_sha256"):
        _hex64(retained[key], f"prior release {key}")
    _require(
        retained["tag"] == "v1.0.3"
        and retained["tag_object"] == V103_TARGET_TAG_OBJECT
        and retained["snapshot_commit"] == V103_SNAPSHOT_COMMIT
        and retained["snapshot_tree"] == V103_SOURCE_TREE
        and retained["tag_payload_sha256"] == V103_TARGET_TAG_PAYLOAD_SHA256
        and retained["snapshot_commit_payload_sha256"] == V103_SNAPSHOT_COMMIT_PAYLOAD_SHA256,
        "prior v1.0.3 immutable Git authority differs",
    )
    inventory = _exact(retained["reachable_inventory"], {"scope", "objects", "object_counts", "inventory_sha256"}, "prior release inventory")
    _require(inventory["scope"] == "reachable-from-release-tag", "prior release inventory scope differs")
    _require(_canonical_sha256(inventory["objects"]) == inventory["inventory_sha256"], "prior release inventory identity differs")
    counts = _exact(inventory["object_counts"], {"blob", "commit", "tag", "tree"}, "prior release object counts")
    _require(all(type(item) is int and item >= 0 for item in counts.values()) and counts["commit"] == 1 and counts["tag"] == 1, "prior release object counts differ")
    _validate_asset_record(retained["mirror_receipt"], "prior mirror receipt", identity_key="publication_id")
    _validate_asset_record(retained["distribution_manifest"], "prior distribution manifest", identity_key="manifest_id")
    release = _exact(retained["release"], {"id", "node_id", "published", "prerelease", "immutable", "title", "body_sha256", "assets"}, "prior immutable release")
    _positive_int(release["id"], "prior release ID")
    _require(isinstance(release["node_id"], str) and release["node_id"], "prior release node ID differs")
    _require(
        release["published"] is True and release["prerelease"] is False and release["immutable"] is True,
        "prior release is not a final published immutable release",
    )
    _require(isinstance(release["title"], str) and release["title"] and isinstance(release["body_sha256"], str), "prior release title or body digest differs")
    _hex64(release["body_sha256"], "prior release body digest")
    assets = release["assets"]
    _require(isinstance(assets, list) and len(assets) == V103_MIGRATED_RELEASE_ASSET_COUNT, "prior release asset count differs")
    names: list[str] = []
    for asset in assets:
        record = _exact(asset, {"id", "node_id", "name", "size_bytes", "sha256"}, "prior immutable release asset")
        _positive_int(record["id"], "prior immutable release asset ID")
        _require(isinstance(record["node_id"], str) and record["node_id"], "prior immutable release asset node ID differs")
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), "prior immutable release asset name differs")
        _positive_int(record["size_bytes"], "prior immutable release asset size")
        _hex64(record["sha256"], "prior immutable release asset digest")
        names.append(record["name"])
    _require(names == sorted(set(names)), "prior release assets are not canonical")
    signed = _exact(
        retained["signed_release"],
        {
            "verification_output_sha256", "bundle_sha256", "statement_sha256",
            "certificate_sha256", "certificate_subject_alternative_name",
            "timestamp_sha256", "signature_sha256", "predicate_type", "predicate",
            "tag_subject", "asset_subjects",
        },
        "prior signed release",
    )
    for key in ("verification_output_sha256", "bundle_sha256", "statement_sha256", "certificate_sha256", "signature_sha256"):
        _hex64(signed[key], f"prior signed release {key}")
    _require(signed["certificate_subject_alternative_name"] == "https://dotcom.releases.github.com", "prior signed release certificate SAN differs")
    _require(
        isinstance(signed["timestamp_sha256"], list) and signed["timestamp_sha256"]
        and signed["timestamp_sha256"] == sorted(set(signed["timestamp_sha256"])),
        "prior signed release timestamp union differs",
    )
    for digest in signed["timestamp_sha256"]:
        _hex64(digest, "prior signed release timestamp digest")
    _require(signed["predicate_type"] == "https://in-toto.io/attestation/release/v0.2", "prior signed release predicate type differs")
    _require(
        signed["tag_subject"] == {
            "uri": f"pkg:github/{TARGET_REPOSITORY}@v1.0.3",
            "object_id": retained["tag_object"],
        },
        "prior signed release tag subject differs",
    )
    _require(
        signed["asset_subjects"] == [
            {"name": item["name"], "sha256": item["sha256"]} for item in assets
        ],
        "prior signed release asset subjects differ",
    )
    return dict(retained)


def validate_target_history_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "state", "mode", "never_imported_or_transferred", "repository", "advertised_refs", "prior_releases", "external_audit", "attestation_id"},
        "target history attestation",
    )
    unsigned = dict(retained)
    attestation_id = unsigned.pop("attestation_id")
    _require(_canonical_sha256(unsigned) == _hex64(attestation_id, "target history attestation identity"), "target history attestation identity differs")
    _require(retained["schema"] == TARGET_HISTORY_SCHEMA and retained["state"] == "HISTORY_CONFIDENTIALITY_ATTESTED", "target history attestation state differs")
    _require(retained["mode"] in {"empty-new-repository", "audited-prior-releases"} and retained["never_imported_or_transferred"] is True, "target history lifecycle authority differs")
    repository = _exact(retained["repository"], {"slug", "id", "node_id"}, "target history repository")
    _require(repository["slug"] == TARGET_REPOSITORY, "target history repository slug differs")
    _positive_int(repository["id"], "target history repository ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "target history repository node ID differs")
    refs = retained["advertised_refs"]
    _require(isinstance(refs, list) and refs == sorted(set(refs)) and all(re.fullmatch(r"refs/tags/v[0-9]+\.[0-9]+\.[0-9]+", ref) for ref in refs), "target history refs must be canonical tag-only releases")
    prior = [validate_prior_release_authority(item) for item in retained["prior_releases"]] if isinstance(retained["prior_releases"], list) else []
    _require(len(prior) == len(retained["prior_releases"]), "target prior release authority differs")
    _require([item["tag"] for item in prior] == sorted(set(item["tag"] for item in prior)), "target prior releases are not canonical")
    _require(refs == [f"refs/tags/{item['tag']}" for item in prior], "target prior release refs and authorities differ")
    if retained["mode"] == "empty-new-repository":
        _require(not refs and not prior, "empty target history contains prior releases")
    else:
        _require(bool(prior), "audited prior target has no per-tag release authorities")
    external = _exact(retained["external_audit"], {"name", "size_bytes", "sha256", "audit_id"}, "target history independent audit")
    _require(isinstance(external["name"], str) and SAFE_NAME.fullmatch(external["name"]), "target history audit name differs")
    _positive_int(external["size_bytes"], "target history audit size")
    _hex64(external["sha256"], "target history audit digest")
    _hex64(external["audit_id"], "target history audit identity")
    return dict(retained)


def validate_v110_target_history_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the one and only history from which v1.1.0 may be published."""

    retained = validate_target_history_attestation(value)
    _require(
        retained["mode"] == "audited-prior-releases"
        and retained["advertised_refs"] == ["refs/tags/v1.0.3"]
        and len(retained["prior_releases"]) == 1
        and retained["prior_releases"][0]["tag"] == "v1.0.3",
        "v1.1.0 requires the exact independently audited v1.0.3 prior release",
    )
    return retained


def _parse_advertised_refs(path: pathlib.Path) -> list[dict[str, str]]:
    source = _regular_file(path, "advertised ref capture")
    records: list[dict[str, str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        oid, separator, ref = line.partition("\t")
        _require(separator == "\t", "advertised ref capture is malformed")
        records.append({"object_id": _hex40(oid, "advertised ref object"), "ref": ref})
    _require(records == sorted(records, key=lambda item: item["ref"]), "advertised ref capture is not canonical")
    _require(len({item["ref"] for item in records}) == len(records), "advertised ref capture has duplicate names")
    return records


def _derive_prior_release_authority(
    *, repo: pathlib.Path, tag: str, release_directory: pathlib.Path
) -> dict[str, Any]:
    _require(tag == "v1.0.3", "only the audited v1.0.3 migration is accepted as prior history")
    root = _regular_directory(release_directory, f"prior release {tag} asset directory")
    receipt_path = _regular_file(root / PRIOR_MIGRATION_RECEIPT_NAME, f"prior release {tag} migration receipt")
    manifest_path = _regular_file(root / PRIOR_MIGRATION_MANIFEST_NAME, f"prior release {tag} migration manifest")
    receipt = _validate_v103_migration_receipt(_load_json(receipt_path, f"prior release {tag} migration receipt"))
    manifest = _validate_v103_migration_manifest(_load_json(manifest_path, f"prior release {tag} migration manifest"))
    _require(manifest["mirror_publication_id"] == receipt["publication_id"], f"prior release {tag} manifest authority differs")
    release = _validate_v103_release_metadata(
        _load_json(root / PRIOR_RELEASE_METADATA_NAME, f"prior release {tag} live release metadata")
    )
    tag_ref = f"refs/tags/{tag}"
    tag_object = _git_text(["rev-parse", tag_ref], repo=repo)
    snapshot_commit = _git_text(["rev-parse", f"{tag_ref}^{{}}"], repo=repo)
    snapshot_tree, parents = _commit_properties(repo, snapshot_commit)
    _require(not parents, f"prior release {tag} is not parentless")
    tag_payload_sha256 = _sha256_bytes(_git(["cat-file", "tag", tag_object], repo=repo).stdout)
    commit_payload_sha256 = _sha256_bytes(_git(["cat-file", "commit", snapshot_commit], repo=repo).stdout)
    tree = tree_manifest(repo, snapshot_tree)
    inventory = {**object_inventory(repo, tag_ref), "scope": "reachable-from-release-tag"}
    live_release = release["release"]
    signed_asset_records = {
        item["name"]: (item["size_bytes"], item["sha256"]) for item in manifest["assets"]
    }
    signed_asset_records[manifest_path.name] = (manifest_path.stat().st_size, _sha256_file(manifest_path))
    signed_release = _validate_github_release_attestation_output(
        output=_load_json(root / PRIOR_RELEASE_ATTESTATION_NAME, f"prior release {tag} signed release attestation"),
        repository_slug=release["repository"]["slug"],
        repository_id=release["repository"]["id"],
        owner_id=release["organization"]["id"],
        tag=tag,
        release_id=live_release["id"],
        tag_object=tag_object,
        asset_records=signed_asset_records,
    )
    authority = {
        "tag": tag,
        "tag_object": tag_object,
        "snapshot_commit": snapshot_commit,
        "snapshot_tree": snapshot_tree,
        "tag_payload_sha256": tag_payload_sha256,
        "snapshot_commit_payload_sha256": commit_payload_sha256,
        "tree_manifest_sha256": tree["manifest_sha256"],
        "reachable_inventory": inventory,
        "mirror_receipt": {
            "name": receipt_path.name,
            "size_bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "publication_id": receipt["publication_id"],
        },
        "distribution_manifest": {
            "name": manifest_path.name,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
            "manifest_id": manifest["manifest_id"],
        },
        "release": {
            "id": live_release["id"],
            "node_id": live_release["node_id"],
            "published": True,
            "prerelease": False,
            "immutable": True,
            "title": live_release["title"],
            "body_sha256": _sha256_bytes(live_release["body"].encode("utf-8")),
            "assets": [dict(item) for item in live_release["assets"]],
        },
        "signed_release": signed_release,
    }
    return _verify_prior_release_assets(authority, repo=repo, assets_root=root.parent)


def create_target_history_attestation(
    *,
    repo: pathlib.Path,
    repository_metadata_path: pathlib.Path,
    mode: str,
    advertised_refs_path: pathlib.Path,
    prior_assets_root: pathlib.Path | None,
    external_audit_path: pathlib.Path,
    trusted_external_audit_sha256: str,
) -> dict[str, Any]:
    repository = _regular_directory(repo, "target history repository")
    stable_paths = [repository_metadata_path, advertised_refs_path, external_audit_path]
    stable_before = _stable_file_snapshot(stable_paths, "target history creator input")
    refs_before = _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=repository).stdout
    prior_tree_before = (
        _regular_tree_snapshot(prior_assets_root, "prior release assets root")
        if prior_assets_root is not None else None
    )
    metadata = _load_json(repository_metadata_path, "target repository metadata")
    _require(
        metadata.get("full_name") == TARGET_REPOSITORY
        and type(metadata.get("id")) is int and metadata["id"] > 0
        and isinstance(metadata.get("node_id"), str) and metadata["node_id"]
        and metadata.get("private") is True and metadata.get("fork") is False
        and metadata.get("is_template") is False,
        "target repository metadata differs for history attestation",
    )
    captured = _parse_advertised_refs(advertised_refs_path)
    refs = [item["ref"] for item in captured]
    _require(all(re.fullmatch(r"refs/tags/v[0-9]+\.[0-9]+\.[0-9]+", ref) for ref in refs), "target history contains non-release or live branch refs")
    actual = _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=repository).stdout.decode("utf-8").splitlines()
    _require(actual == [f"{item['object_id']}\t{item['ref']}" for item in captured], "target history clone differs from captured full ref union")
    audit_path = _regular_file(external_audit_path, "independent mirror audit")
    _require(
        _sha256_file(audit_path) == _hex64(trusted_external_audit_sha256, "trusted independent mirror audit digest"),
        "independent mirror audit differs from the out-of-band trusted digest",
    )
    audit = validate_external_mirror_audit(_load_json(audit_path, "independent mirror audit"))
    _require(audit["target_repository"] == {"slug": metadata["full_name"], "id": metadata["id"], "node_id": metadata["node_id"]}, "history audit target differs")
    _require(audit["target_history"]["advertised_refs_sha256"] == _canonical_sha256(captured), "history audit ref union differs")
    prior: list[dict[str, Any]] = []
    if mode == "empty-new-repository":
        _require(not refs and prior_assets_root is None, "empty target history inputs differ")
        _verify_no_repository_extensions(repository, allow_unborn_symbolic_head=True)
        objects = _git(["count-objects", "-v"], repo=repository).stdout.decode("ascii")
        values = dict(line.split(": ", 1) for line in objects.splitlines() if ": " in line)
        _require(values.get("count") == "0" and values.get("in-pack") == "0", "empty target contains hidden Git objects")
    else:
        _require(mode == "audited-prior-releases" and prior_assets_root is not None and refs, "audited prior target inputs differ")
        assets = _regular_directory(prior_assets_root, "prior release assets root")
        for ref in refs:
            tag = ref.removeprefix("refs/tags/")
            prior.append(_derive_prior_release_authority(repo=repository, tag=tag, release_directory=assets / tag))
    _require(
        audit["target_history"]["prior_releases_sha256"] == _canonical_sha256(prior),
        "history prior release authorities differ from the out-of-band trusted audit",
    )
    history: dict[str, Any] = {
        "schema": TARGET_HISTORY_SCHEMA,
        "state": "HISTORY_CONFIDENTIALITY_ATTESTED",
        "mode": mode,
        "never_imported_or_transferred": audit["target_history"]["never_imported_or_transferred"],
        "repository": {"slug": metadata["full_name"], "id": metadata["id"], "node_id": metadata["node_id"]},
        "advertised_refs": refs,
        "prior_releases": prior,
        "external_audit": {
            "name": audit_path.name,
            "size_bytes": audit_path.stat().st_size,
            "sha256": _sha256_file(audit_path),
            "audit_id": audit["audit_id"],
        },
    }
    history["attestation_id"] = _canonical_sha256(history)
    retained = validate_target_history_attestation(history)
    _require(
        _stable_file_snapshot(stable_paths, "target history creator input") == stable_before
        and _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=repository).stdout == refs_before
        and (
            prior_assets_root is None
            or _regular_tree_snapshot(prior_assets_root, "prior release assets root") == prior_tree_before
        ),
        "target history inputs changed during creation",
    )
    return retained


def validate_target_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {
            "schema", "repository", "visibility", "fork", "template", "archived", "disabled",
            "actions_enabled", "immutable_releases_enabled", "advertised_refs",
            "bootstrap", "organization_access", "automation", "external_mirror_audit",
        },
        "target preflight",
    )
    _require(retained["schema"] == TARGET_PREFLIGHT_SCHEMA, "target preflight schema differs")
    repository = _exact(retained["repository"], {"slug", "id", "node_id"}, "target repository identity")
    _require(repository["slug"] == TARGET_REPOSITORY and SAFE_SLUG.fullmatch(repository["slug"]), "target repository slug differs")
    _positive_int(repository["id"], "target repository numeric ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "target repository node ID differs")
    _require(
        retained["visibility"] == "private"
        and retained["fork"] is False
        and retained["template"] is False
        and retained["archived"] is False
        and retained["disabled"] is False,
        "target must be private, active, non-fork and non-template",
    )
    _require(retained["actions_enabled"] is False, "GitHub Actions must be disabled before publication")
    _require(retained["immutable_releases_enabled"] is True, "immutable releases must be enabled")
    refs = retained["advertised_refs"]
    _require(isinstance(refs, list) and all(isinstance(item, str) and item.startswith("refs/") for item in refs), "target advertised refs differ")
    _require(refs == sorted(set(refs)), "target advertised refs are not canonical")
    _require(all(ref.startswith("refs/tags/v") for ref in refs), "release-only target must be tag-only with no live default branch")
    bootstrap = _exact(retained["bootstrap"], {"mode", "attestation_sha256", "attestation_id"}, "target bootstrap")
    _require(bootstrap["mode"] in {"empty-new-repository", "audited-prior-releases"}, "target bootstrap mode differs")
    _hex64(bootstrap["attestation_sha256"], "target bootstrap attestation")
    _hex64(bootstrap["attestation_id"], "target bootstrap attestation identity")
    if bootstrap["mode"] == "empty-new-repository":
        _require(refs == [], "new release-only target is not empty")
    access = _exact(retained["organization_access"], {"organization", "id", "node_id", "default_repository_permission", "all_members_can_read"}, "organization access")
    _require(access["organization"] == "example-organization", "target organization identity differs")
    _positive_int(access["id"], "target organization ID")
    _require(isinstance(access["node_id"], str) and access["node_id"], "target organization node ID differs")
    _require(access["default_repository_permission"] in {"read", "write", "admin"} and access["all_members_can_read"] is True, "organization-wide member read access is not proven")
    automation = _exact(
        retained["automation"],
        {
            "actions_enabled", "dependabot_security_updates_enabled",
            "dependabot_version_updates_enabled", "no_live_default_branch",
            "repository_administration_read_authorized", "source_dependabot_limits",
        },
        "target automation policy",
    )
    _require(
        automation == {
            "actions_enabled": False,
            "dependabot_security_updates_enabled": False,
            "dependabot_version_updates_enabled": False,
            "no_live_default_branch": True,
            "repository_administration_read_authorized": True,
            "source_dependabot_limits": {"github-actions": 0, "pip": 0},
        },
        "target automation suppression policy differs",
    )
    audit = _exact(
        retained["external_mirror_audit"],
        {"name", "size_bytes", "sha256", "audit_id", "source_commit", "source_tree", "snapshot_id", "snapshot_commit", "final_verifier_sha256"},
        "target independent mirror audit",
    )
    _require(isinstance(audit["name"], str) and SAFE_NAME.fullmatch(audit["name"]), "independent mirror audit asset name differs")
    _positive_int(audit["size_bytes"], "independent mirror audit asset size")
    for key in ("sha256", "audit_id", "snapshot_id", "final_verifier_sha256"):
        _hex64(audit[key], f"target independent mirror audit {key}")
    for key in ("source_commit", "source_tree", "snapshot_commit"):
        _hex40(audit[key], f"target independent mirror audit {key}")
    return dict(retained)


def validate_v110_target_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject an empty or multi-release bootstrap for the v1.1.0 publisher."""

    retained = validate_target_preflight(value)
    _require(
        retained["bootstrap"]["mode"] == "audited-prior-releases"
        and retained["advertised_refs"] == ["refs/tags/v1.0.3"],
        "v1.1.0 target preflight requires only the audited v1.0.3 prior tag",
    )
    return retained


def create_target_preflight(
    *,
    repository_metadata_path: pathlib.Path,
    actions_permissions_path: pathlib.Path,
    immutable_releases_path: pathlib.Path,
    organization_metadata_path: pathlib.Path,
    dependabot_security_updates_status: str,
    advertised_refs_path: pathlib.Path,
    history_attestation_path: pathlib.Path,
    external_audit_path: pathlib.Path,
    trusted_external_audit_sha256: str,
) -> dict[str, Any]:
    stable_paths = [
        repository_metadata_path, actions_permissions_path, immutable_releases_path,
        organization_metadata_path, advertised_refs_path, history_attestation_path,
        external_audit_path,
    ]
    stable_before = _stable_file_snapshot(stable_paths, "target preflight creator input")
    repository = _load_json(repository_metadata_path, "target repository metadata")
    actions = _load_json(actions_permissions_path, "target Actions permissions")
    immutable = _load_json(immutable_releases_path, "target immutable release settings")
    organization = _load_json(organization_metadata_path, "target organization metadata")
    history_path = _regular_file(history_attestation_path, "target history attestation")
    history = validate_target_history_attestation(_load_json(history_path, "target history attestation"))
    captured = _parse_advertised_refs(advertised_refs_path)
    refs = [item["ref"] for item in captured]
    _require(
        repository.get("full_name") == TARGET_REPOSITORY
        and type(repository.get("id")) is int and repository["id"] > 0
        and isinstance(repository.get("node_id"), str) and repository["node_id"]
        and repository.get("private") is True
        and repository.get("visibility") == "private"
        and repository.get("fork") is False
        and repository.get("is_template") is False
        and repository.get("archived") is False
        and repository.get("disabled") is False,
        "target live repository metadata differs",
    )
    permissions = repository.get("permissions")
    _require(
        isinstance(permissions, dict) and permissions.get("admin") is True,
        "target Administration(read) authority is not proven by live repository metadata",
    )
    _require(actions.get("enabled") is False, "target Actions are not disabled")
    _require(immutable.get("enabled") is True, "target immutable releases are not enabled")
    _require(
        organization.get("login") == "example-organization"
        and type(organization.get("id")) is int and organization["id"] > 0
        and isinstance(organization.get("node_id"), str) and organization["node_id"]
        and organization.get("default_repository_permission") in {"read", "write", "admin"},
        "organization-wide member read access is not proven by live metadata",
    )
    _require(dependabot_security_updates_status == "disabled", "Dependabot security updates are not proven disabled")
    _require(all(ref.startswith("refs/tags/v") for ref in refs), "target has a live branch or non-release ref")
    _require(history["advertised_refs"] == refs and history["repository"] == {"slug": repository["full_name"], "id": repository["id"], "node_id": repository["node_id"]}, "target history attestation differs from live target")
    audit_record = external_audit_record(external_audit_path, trusted_external_audit_sha256)
    audit = validate_external_mirror_audit(_load_json(external_audit_path, "independent mirror audit"))
    _require(
        audit["target_repository"] == {
            "slug": repository["full_name"], "id": repository["id"], "node_id": repository["node_id"]
        },
        "target preflight independent audit repository differs",
    )
    _require(
        audit["organization_access"] == {
            "organization": organization["login"],
            "id": organization["id"],
            "node_id": organization["node_id"],
            "default_repository_permission": organization["default_repository_permission"],
            "all_members_can_read": True,
        },
        "target preflight independent audit organization authority differs",
    )
    _require(
        history["external_audit"] == {key: audit_record[key] for key in ("name", "size_bytes", "sha256", "audit_id")},
        "target history and preflight independent audits differ",
    )
    preflight = {
        "schema": TARGET_PREFLIGHT_SCHEMA,
        "repository": {"slug": repository["full_name"], "id": repository["id"], "node_id": repository["node_id"]},
        "visibility": "private",
        "fork": False,
        "template": False,
        "archived": False,
        "disabled": False,
        "actions_enabled": False,
        "immutable_releases_enabled": True,
        "advertised_refs": refs,
        "bootstrap": {
            "mode": history["mode"],
            "attestation_sha256": _sha256_file(history_path),
            "attestation_id": history["attestation_id"],
        },
        "organization_access": {
            "organization": organization["login"],
            "id": organization["id"],
            "node_id": organization["node_id"],
            "default_repository_permission": organization["default_repository_permission"],
            "all_members_can_read": True,
        },
        "automation": {
            "actions_enabled": False,
            "dependabot_security_updates_enabled": False,
            "dependabot_version_updates_enabled": False,
            "no_live_default_branch": True,
            "repository_administration_read_authorized": True,
            "source_dependabot_limits": {"github-actions": 0, "pip": 0},
        },
        "external_mirror_audit": audit_record,
    }
    retained = validate_target_preflight(preflight)
    _require(
        _stable_file_snapshot(stable_paths, "target preflight creator input") == stable_before,
        "target preflight inputs changed during creation",
    )
    return retained


def _intent_file(path: pathlib.Path, label: str) -> dict[str, Any]:
    retained = _regular_file(path, label)
    return {"path": str(retained), "size_bytes": retained.stat().st_size, "sha256": _sha256_file(retained)}


def _intent_directory(path: pathlib.Path, label: str) -> dict[str, Any]:
    retained = _regular_directory(path, label)
    stat_value = retained.stat()
    return {"path": str(retained), "device": stat_value.st_dev, "inode": stat_value.st_ino}


def create_run_intent(
    *,
    source_repo: pathlib.Path,
    source_authority_path: pathlib.Path,
    target_preflight_path: pathlib.Path,
    target_history_attestation_path: pathlib.Path,
    external_audit_path: pathlib.Path,
    asset_directory: pathlib.Path,
    state_root: pathlib.Path,
    repository: str,
    identity: Mapping[str, Any],
    forbidden_objects: Sequence[str],
    local_bare_remote: pathlib.Path | None,
    gh_binary: pathlib.Path | None,
    gh_sha256: str | None,
    gh_capabilities_sha256: str | None,
    trusted_external_audit_sha256: str,
) -> dict[str, Any]:
    _require(repository == TARGET_REPOSITORY, "run intent target repository differs")
    stable_paths = [
        source_authority_path, target_preflight_path,
        target_history_attestation_path, external_audit_path,
    ]
    if gh_binary is not None:
        stable_paths.append(gh_binary)
    stable_before = _stable_file_snapshot(stable_paths, "run intent input")
    _require(
        stat.S_IMODE(_regular_file(external_audit_path, "run external mirror audit").stat().st_mode) == 0o600,
        "run external mirror audit mode must be 0600",
    )
    trusted_audit_digest = _hex64(trusted_external_audit_sha256, "trusted independent mirror audit digest")
    _require(
        _sha256_file(external_audit_path) == trusted_audit_digest,
        "independent mirror audit differs from the out-of-band trusted digest",
    )
    directory_paths = [source_repo, asset_directory, state_root]
    if local_bare_remote is not None:
        directory_paths.append(local_bare_remote)
    directories_before = {
        str(path.absolute()): _stable_directory_identity(path, "run intent directory")
        for path in directory_paths
    }
    assets_before = _regular_tree_snapshot(asset_directory, "run intent asset directory")
    source_refs_before = _source_git(source_repo, ["for-each-ref", "--format=%(objectname)%09%(refname)"]).stdout
    remote_repo = local_bare_remote
    remote_refs_before = (
        _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=remote_repo).stdout
        if remote_repo is not None else None
    )
    files = {
        "source_authority": _intent_file(source_authority_path, "run source authority"),
        "target_preflight": _intent_file(target_preflight_path, "run target preflight"),
        "target_history_attestation": _intent_file(target_history_attestation_path, "run target history attestation"),
        "external_audit": _intent_file(external_audit_path, "run external mirror audit"),
    }
    directories = {
        "source_repo": _intent_directory(source_repo, "run source repository"),
        "asset_directory": _intent_directory(asset_directory, "run asset directory"),
        "state_root": _intent_directory(state_root, "run state root"),
    }
    forbidden = sorted(set(_hex40(item, "run forbidden object") for item in forbidden_objects))
    if local_bare_remote is not None:
        _require(
            gh_binary is None and gh_sha256 is None and gh_capabilities_sha256 is None,
            "run intent transport is ambiguous",
        )
        transport = {"mode": "local-bare", **_intent_directory(local_bare_remote, "local bare remote")}
    else:
        _require(
            gh_binary is not None and gh_sha256 is not None and gh_capabilities_sha256 is not None,
            "GitHub run intent requires a pinned gh binary and capability identity",
        )
        binary = _regular_file(gh_binary, "pinned gh binary")
        digest = _hex64(gh_sha256, "pinned gh binary digest")
        capability_digest = _hex64(gh_capabilities_sha256, "pinned gh capability digest")
        _require(
            os.access(binary, os.X_OK)
            and _sha256_file(binary) == digest == APPROVED_GITHUB_CLI_SHA256,
            "pinned gh binary is not the approved audited GitHub CLI",
        )
        transport = {
            "mode": "github",
            "remote_url": f"git@github.com:{TARGET_REPOSITORY}.git",
            "gh_binary": {"path": str(binary), "size_bytes": binary.stat().st_size, "sha256": digest},
            "capabilities_sha256": capability_digest,
        }
    intent: dict[str, Any] = {
        "schema": RUN_INTENT_SCHEMA,
        "repository": repository,
        "files": files,
        "directories": directories,
        "identity": _validate_identity(identity, "run identity"),
        "forbidden_objects": forbidden,
        "transport": transport,
        "trusted_external_audit_sha256": trusted_audit_digest,
    }
    intent["intent_id"] = _canonical_sha256(intent)
    retained = validate_run_intent(intent)
    _require(
        _stable_file_snapshot(stable_paths, "run intent input") == stable_before
        and {
            str(path.absolute()): _stable_directory_identity(path, "run intent directory")
            for path in directory_paths
        } == directories_before
        and _regular_tree_snapshot(asset_directory, "run intent asset directory") == assets_before
        and _source_git(source_repo, ["for-each-ref", "--format=%(objectname)%09%(refname)"]).stdout == source_refs_before
        and (
            remote_repo is None
            or _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=remote_repo).stdout == remote_refs_before
        ),
        "run intent inputs changed during creation",
    )
    return retained


def validate_run_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "repository", "files", "directories", "identity", "forbidden_objects", "transport", "trusted_external_audit_sha256", "intent_id"},
        "release-only run intent",
    )
    unsigned = dict(retained)
    identity = unsigned.pop("intent_id")
    _require(_canonical_sha256(unsigned) == _hex64(identity, "run intent identity"), "release-only run intent identity differs")
    _require(retained["schema"] == RUN_INTENT_SCHEMA and retained["repository"] == TARGET_REPOSITORY, "release-only run intent authority differs")
    _hex64(retained["trusted_external_audit_sha256"], "trusted independent mirror audit digest")
    _require(set(retained["files"]) == {"source_authority", "target_preflight", "target_history_attestation", "external_audit"}, "run intent file set differs")
    for record in retained["files"].values():
        _exact(record, {"path", "size_bytes", "sha256"}, "run intent file")
        _require(isinstance(record["path"], str) and pathlib.Path(record["path"]).is_absolute(), "run intent file path differs")
        _nonnegative_int(record["size_bytes"], "run intent file size")
        _hex64(record["sha256"], "run intent file digest")
    _require(
        retained["files"]["external_audit"]["sha256"] == retained["trusted_external_audit_sha256"],
        "run intent external audit differs from the out-of-band trusted digest",
    )
    _require(set(retained["directories"]) == {"source_repo", "asset_directory", "state_root"}, "run intent directory set differs")
    for record in retained["directories"].values():
        _exact(record, {"path", "device", "inode"}, "run intent directory")
        _require(isinstance(record["path"], str) and pathlib.Path(record["path"]).is_absolute(), "run intent directory path differs")
        _nonnegative_int(record["device"], "run intent directory device")
        _positive_int(record["inode"], "run intent directory inode")
    _validate_identity(retained["identity"], "run intent identity")
    _require(retained["forbidden_objects"] == sorted(set(retained["forbidden_objects"])), "run intent forbidden objects differ")
    for oid in retained["forbidden_objects"]:
        _hex40(oid, "run intent forbidden object")
    transport = retained["transport"]
    _require(isinstance(transport, dict) and transport.get("mode") in {"local-bare", "github"}, "run intent transport differs")
    if transport["mode"] == "local-bare":
        _exact(transport, {"mode", "path", "device", "inode"}, "local run transport")
        _require(isinstance(transport["path"], str) and pathlib.Path(transport["path"]).is_absolute(), "local run transport path differs")
        _nonnegative_int(transport["device"], "local run transport device")
        _positive_int(transport["inode"], "local run transport inode")
    else:
        _exact(transport, {"mode", "gh_binary", "capabilities_sha256", "remote_url"}, "GitHub run transport")
        _require(transport["remote_url"] == f"git@github.com:{TARGET_REPOSITORY}.git", "GitHub run transport URL differs")
        _hex64(transport["capabilities_sha256"], "GitHub capability identity")
        binary = _exact(transport["gh_binary"], {"path", "size_bytes", "sha256"}, "GitHub CLI identity")
        _require(isinstance(binary["path"], str) and pathlib.Path(binary["path"]).is_absolute(), "GitHub CLI path differs")
        _positive_int(binary["size_bytes"], "GitHub CLI size")
        _require(
            _hex64(binary["sha256"], "GitHub CLI digest") == APPROVED_GITHUB_CLI_SHA256,
            "GitHub run intent does not pin the approved audited CLI",
        )
    return dict(retained)


def replay_run_intent(intent: Mapping[str, Any], **keywords: Any) -> dict[str, Any]:
    retained = validate_run_intent(intent)
    expected = create_run_intent(**keywords)
    _require(retained == expected, "release-only run intent differs from the exact dry-run arguments or transport")
    return retained


def create_receipt(
    source_authority: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    target_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_source_authority(source_authority)
    snap = validate_snapshot(snapshot)
    target = validate_v110_target_preflight(target_preflight)
    _require(snap["source_commit"] == source["source_commit"] and snap["source_tree"] == source["source_tree"], "snapshot source authority differs")
    audit = target["external_mirror_audit"]
    _require(
        audit["source_commit"] == source["source_commit"]
        and audit["source_tree"] == source["source_tree"]
        and audit["snapshot_id"] == snap["snapshot_id"]
        and audit["snapshot_commit"] == snap["snapshot_commit"]
        and audit["final_verifier_sha256"] == source["verification"]["verifier_sha256"],
        "independent mirror audit is not bound to the exact source and snapshot",
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "release_train": RELEASE_TRAIN,
        "state": TERMINAL_STATE,
        "source_authority": source,
        "snapshot": {key: snap[key] for key in (
            "method", "object_format", "snapshot_commit", "snapshot_tree", "parent_count",
            "reachable_commit_count", "identity", "commit_message_sha256", "tree_manifest_sha256",
            "reachable_inventory_sha256", "object_counts", "inventory_scope", "assertions", "snapshot_id",
        )},
        "target": {
            **target,
            "tag": DEFAULT_TAG,
        },
        "tag_construction": {
            "name": DEFAULT_TAG,
            "tagger": snap["identity"],
            "message_format": "gpmeep-v110-release-only-tag-message-v1",
            "distribution_manifest_id_bound_after_receipt": True,
        },
        "authorization": {
            "release_only_mirror_publication": True,
            "source_authority_replacement": False,
            "package_build_or_mutation": False,
            "ref_overwrite": False,
            "ref_deletion": False,
            "force_push": False,
            "asset_overwrite": False,
            "asset_deletion": False,
        },
    }
    receipt["publication_id"] = _canonical_sha256(receipt)
    return validate_receipt(receipt)


def validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(value, {"schema", "release_train", "state", "source_authority", "snapshot", "target", "tag_construction", "authorization", "publication_id"}, "mirror receipt")
    unsigned = dict(retained)
    publication_id = unsigned.pop("publication_id")
    _hex64(publication_id, "mirror publication identity")
    _require(_canonical_sha256(unsigned) == publication_id, "mirror publication identity differs")
    _require(retained["schema"] == SCHEMA and retained["release_train"] == RELEASE_TRAIN and retained["state"] == TERMINAL_STATE, "mirror receipt authority differs")
    source = validate_source_authority(retained["source_authority"])
    snapshot = _exact(retained["snapshot"], {"method", "object_format", "snapshot_commit", "snapshot_tree", "parent_count", "reachable_commit_count", "identity", "commit_message_sha256", "tree_manifest_sha256", "reachable_inventory_sha256", "object_counts", "inventory_scope", "assertions", "snapshot_id"}, "receipt snapshot")
    _require(snapshot["method"] == METHOD and snapshot["object_format"] == "sha1", "receipt snapshot method differs")
    _hex40(snapshot["snapshot_commit"], "receipt snapshot commit")
    _require(snapshot["snapshot_tree"] == source["source_tree"], "receipt snapshot tree differs")
    _require(snapshot["parent_count"] == 0 and snapshot["reachable_commit_count"] == 1, "receipt snapshot history differs")
    _validate_identity(snapshot["identity"], "receipt snapshot identity")
    for key in ("commit_message_sha256", "tree_manifest_sha256", "reachable_inventory_sha256", "snapshot_id"):
        _hex64(snapshot[key], f"receipt snapshot {key}")
    _require(snapshot["inventory_scope"] == "reachable-from-snapshot-commit", "receipt snapshot inventory scope differs")
    counts = _exact(snapshot["object_counts"], {"blob", "commit", "tag", "tree"}, "receipt snapshot object counts")
    _require(all(type(item) is int and item >= 0 for item in counts.values()) and counts["commit"] == 1 and counts["tag"] == 0, "receipt snapshot object counts differ")
    snapshot_assertions = _exact(
        snapshot["assertions"],
        {"source_commit_absent", "development_parents_absent", "forbidden_objects_absent", "no_alternates", "no_promisor", "no_shallow", "no_grafts", "no_replace_refs", "no_unexpected_refs", "fsck_clean", "dependabot_pull_requests_suppressed"},
        "receipt snapshot assertions",
    )
    _require(all(snapshot_assertions[key] is True for key in snapshot_assertions if key != "forbidden_objects_absent"), "receipt snapshot assertion differs")
    bound_forbidden = snapshot_assertions["forbidden_objects_absent"]
    _require(isinstance(bound_forbidden, list) and bound_forbidden == sorted(set(bound_forbidden)), "receipt forbidden object assertions differ")
    for oid in bound_forbidden:
        _hex40(oid, "receipt forbidden object")
    target_with_tag = _exact(retained["target"], {"schema", "repository", "visibility", "fork", "template", "archived", "disabled", "actions_enabled", "immutable_releases_enabled", "advertised_refs", "bootstrap", "organization_access", "automation", "external_mirror_audit", "tag"}, "receipt target")
    target = dict(target_with_tag)
    _require(target.pop("tag") == DEFAULT_TAG, "receipt target tag differs")
    validate_v110_target_preflight(target)
    tag = _exact(retained["tag_construction"], {"name", "tagger", "message_format", "distribution_manifest_id_bound_after_receipt"}, "receipt tag construction")
    _require(tag["name"] == DEFAULT_TAG and tag["message_format"] == "gpmeep-v110-release-only-tag-message-v1" and tag["distribution_manifest_id_bound_after_receipt"] is True, "receipt tag construction differs")
    _require(tag["tagger"] == snapshot["identity"], "receipt tagger identity differs")
    expected_authorization = {
        "release_only_mirror_publication": True,
        "source_authority_replacement": False,
        "package_build_or_mutation": False,
        "ref_overwrite": False,
        "ref_deletion": False,
        "force_push": False,
        "asset_overwrite": False,
        "asset_deletion": False,
    }
    _require(retained["authorization"] == expected_authorization, "mirror publication authorization differs")
    return dict(retained)


def replay_receipt(
    receipt: Mapping[str, Any],
    *,
    source_authority: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    target_preflight: Mapping[str, Any],
    snapshot_repo: pathlib.Path,
) -> dict[str, Any]:
    retained = validate_receipt(receipt)
    replay_snapshot(snapshot, snapshot_repo)
    expected = create_receipt(source_authority, snapshot, target_preflight)
    _require(retained == expected, "mirror receipt was not exactly re-derived")
    return retained


def create_distribution_manifest(
    *,
    source_manifest_path: pathlib.Path,
    asset_directory: pathlib.Path,
    mirror_receipt_path: pathlib.Path,
    target_history_attestation_path: pathlib.Path,
) -> dict[str, Any]:
    source_path = _regular_file(source_manifest_path, "private source asset manifest")
    root = _regular_directory(asset_directory, "release asset directory")
    receipt_path = _regular_file(mirror_receipt_path, "mirror receipt")
    history_path = _regular_file(target_history_attestation_path, "target history attestation")
    receipt = validate_receipt(_load_json(receipt_path, "mirror receipt"))
    history = validate_v110_target_history_attestation(_load_json(history_path, "target history attestation"))
    _require(
        receipt["target"]["bootstrap"]["attestation_sha256"] == _sha256_file(history_path)
        and receipt["target"]["bootstrap"]["attestation_id"] == history["attestation_id"],
        "target history attestation differs from mirror receipt",
    )
    source = _load_json(source_path, "private source asset manifest")
    _exact(source, {"schema", "assets", "manifest_id"}, "private source asset manifest")
    _require(source["schema"] == "gpmeep-v110-release-asset-manifest-v1", "private source asset manifest schema differs")
    unsigned = dict(source)
    manifest_id = unsigned.pop("manifest_id")
    _require(_canonical_sha256(unsigned) == _hex64(manifest_id, "private source asset manifest identity"), "private source asset manifest identity differs")
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_roles: set[str] = set()
    for record in source["assets"]:
        _exact(record, {"role", "name", "size_bytes", "sha256"}, "private release asset record")
        role = f"source_{record['role']}"
        name = record["name"]
        _require(isinstance(name, str) and SAFE_NAME.fullmatch(name), "private release asset name differs")
        path = _regular_file(root / name, f"private release asset {name}")
        _require(path.parent == root, "private release asset escaped asset directory")
        _require(path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"], f"private release asset bytes differ: {name}")
        _require(role not in seen_roles and name not in seen_names, "duplicate distribution asset")
        seen_roles.add(role)
        seen_names.add(name)
        records.append({"role": role, "name": name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    for role, path in (
        ("source_asset_manifest", source_path),
        ("mirror_receipt", receipt_path),
        ("target_history_attestation", history_path),
        (
            "independent_mirror_audit",
            _regular_file(root / receipt["target"]["external_mirror_audit"]["name"], "independent mirror audit asset"),
        ),
    ):
        _require(path.parent == root, f"{role} must be inside the release asset directory")
        _require(role not in seen_roles and path.name not in seen_names, "duplicate distribution asset")
        seen_roles.add(role)
        seen_names.add(path.name)
        records.append({"role": role, "name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    records.sort(key=lambda item: item["role"])
    manifest: dict[str, Any] = {
        "schema": ASSET_SCHEMA,
        "source_manifest_id": manifest_id,
        "mirror_publication_id": receipt["publication_id"],
        "assets": records,
    }
    manifest["manifest_id"] = _canonical_sha256(manifest)
    return validate_distribution_manifest(manifest)


def validate_distribution_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(value, {"schema", "source_manifest_id", "mirror_publication_id", "assets", "manifest_id"}, "distribution asset manifest")
    unsigned = dict(retained)
    manifest_id = unsigned.pop("manifest_id")
    _require(_canonical_sha256(unsigned) == _hex64(manifest_id, "distribution manifest identity"), "distribution manifest identity differs")
    _require(retained["schema"] == ASSET_SCHEMA, "distribution asset schema differs")
    _hex64(retained["source_manifest_id"], "source asset manifest identity")
    _hex64(retained["mirror_publication_id"], "mirror publication identity")
    _require(isinstance(retained["assets"], list) and retained["assets"], "distribution asset inventory is empty")
    roles: set[str] = set()
    names: set[str] = set()
    for record in retained["assets"]:
        _exact(record, {"role", "name", "size_bytes", "sha256"}, "distribution asset")
        _require(isinstance(record["role"], str) and SAFE_NAME.fullmatch(record["role"]), "distribution asset role differs")
        _require(isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]), "distribution asset name differs")
        _require(record["role"] not in roles and record["name"] not in names, "duplicate distribution asset")
        roles.add(record["role"])
        names.add(record["name"])
        _nonnegative_int(record["size_bytes"], "distribution asset size")
        _hex64(record["sha256"], "distribution asset digest")
    _require([item["role"] for item in retained["assets"]] == sorted(roles), "distribution asset inventory order differs")
    return dict(retained)


def verify_distribution_sources(manifest: Mapping[str, Any], asset_directory: pathlib.Path) -> list[pathlib.Path]:
    retained = validate_distribution_manifest(manifest)
    root = _regular_directory(asset_directory, "distribution asset directory")
    actual: list[pathlib.Path] = []
    for record in retained["assets"]:
        path = _regular_file(root / record["name"], f"distribution asset {record['name']}")
        _require(path.parent == root, "distribution asset escaped asset directory")
        _require(path.stat().st_size == record["size_bytes"] and _sha256_file(path) == record["sha256"], f"distribution asset bytes differ: {record['name']}")
        actual.append(path)
    return actual


def cross_validate_distribution_metadata(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    retained_manifest = validate_distribution_manifest(manifest)
    retained_receipt = validate_receipt(receipt)
    _require(retained_manifest["mirror_publication_id"] == retained_receipt["publication_id"], "distribution manifest mirror receipt identity differs")
    source = retained_receipt["source_authority"]
    _require(retained_manifest["source_manifest_id"] == source["source_asset_manifest"]["manifest_id"], "distribution source manifest identity differs from receipt")
    by_role = {record["role"]: record for record in retained_manifest["assets"]}
    expected_source_manifest = {
        "role": "source_asset_manifest",
        "name": source["source_asset_manifest"]["name"],
        "size_bytes": source["source_asset_manifest"]["size_bytes"],
        "sha256": source["source_asset_manifest"]["sha256"],
    }
    _require(by_role.get("source_asset_manifest") == expected_source_manifest, "distribution source manifest bytes differ from receipt")
    for source_role, authority_key in (("candidate_receipt", "candidate_receipt"), ("final_receipt", "final_receipt"), ("package", "package")):
        authority = source[authority_key]
        expected = {
            "role": f"source_{source_role}",
            "name": authority["name"],
            "size_bytes": authority["size_bytes"],
            "sha256": authority["sha256"],
        }
        _require(by_role.get(f"source_{source_role}") == expected, f"distribution {source_role} bytes differ from receipt")
    audit = retained_receipt["target"]["external_mirror_audit"]
    expected_audit = {
        "role": "independent_mirror_audit",
        "name": audit["name"],
        "size_bytes": audit["size_bytes"],
        "sha256": audit["sha256"],
    }
    _require(by_role.get("independent_mirror_audit") == expected_audit, "distribution independent mirror audit differs from receipt")
    history = by_role.get("target_history_attestation")
    _require(
        history is not None
        and history["sha256"] == retained_receipt["target"]["bootstrap"]["attestation_sha256"],
        "distribution target history bytes differ from receipt",
    )
    return retained_manifest


def cross_validate_distribution(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    asset_directory: pathlib.Path,
) -> dict[str, Any]:
    retained_manifest = cross_validate_distribution_metadata(manifest, receipt)
    retained_receipt = validate_receipt(receipt)
    root = _regular_directory(asset_directory, "distribution asset directory")
    verify_distribution_sources(retained_manifest, root)
    manifest_path = _regular_file(root / DISTRIBUTION_MANIFEST_NAME, "canonical distribution manifest")
    _require(
        _load_json(manifest_path, "canonical distribution manifest") == retained_manifest,
        "canonical distribution manifest bytes differ",
    )
    expected_names = sorted(
        [record["name"] for record in retained_manifest["assets"]] + [DISTRIBUTION_MANIFEST_NAME]
    )
    actual_entries = sorted(root.iterdir(), key=lambda path: path.name)
    _require(
        [path.name for path in actual_entries] == expected_names
        and all(not path.is_symlink() and path.is_file() for path in actual_entries),
        "distribution asset directory census differs",
    )
    source = retained_receipt["source_authority"]
    source_manifest = _load_json(root / source["source_asset_manifest"]["name"], "distributed source asset manifest")
    manifest_by_role = {record["role"]: record for record in source_manifest.get("assets", []) if isinstance(record, dict) and "role" in record}
    for source_role, authority_key in (("candidate_receipt", "candidate_receipt"), ("final_receipt", "final_receipt"), ("package", "package")):
        authority = source[authority_key]
        _require(
            manifest_by_role.get(source_role) == {
                "role": source_role,
                "name": authority["name"],
                "size_bytes": authority["size_bytes"],
                "sha256": authority["sha256"],
            },
            f"distributed source manifest {source_role} authority differs",
        )
    by_role = {record["role"]: record for record in retained_manifest["assets"]}
    history_record = by_role["target_history_attestation"]
    history = validate_v110_target_history_attestation(
        _load_json(root / history_record["name"], "distributed target history attestation")
    )
    _require(
        history["attestation_id"] == retained_receipt["target"]["bootstrap"]["attestation_id"]
        and history_record["sha256"] == retained_receipt["target"]["bootstrap"]["attestation_sha256"],
        "distributed target history authority differs from receipt",
    )
    return retained_manifest


def tag_message(receipt: Mapping[str, Any], distribution_manifest: Mapping[str, Any]) -> str:
    retained = validate_receipt(receipt)
    manifest = cross_validate_distribution_metadata(distribution_manifest, retained)
    source = retained["source_authority"]
    snapshot = retained["snapshot"]
    return (
        f"gpmeep {DEFAULT_TAG} release-only mirror\n"
        f"authoritative-source-commit: {source['source_commit']}\n"
        f"authoritative-source-tree: {source['source_tree']}\n"
        f"snapshot-commit: {snapshot['snapshot_commit']}\n"
        f"package-sha256: {source['package']['sha256']}\n"
        f"candidate-receipt-id: {source['candidate_receipt']['receipt_id']}\n"
        f"final-receipt-id: {source['final_receipt']['receipt_id']}\n"
        f"mirror-publication-id: {retained['publication_id']}\n"
        f"independent-mirror-audit-id: {retained['target']['external_mirror_audit']['audit_id']}\n"
        f"distribution-asset-manifest-id: {manifest['manifest_id']}"
    )


def _annotated_tag_payload(
    *, commit: str, tag_name: str, message: str, identity: Mapping[str, Any]
) -> bytes:
    retained = _validate_identity(identity, "tagger identity")
    _require(tag_name == DEFAULT_TAG, "annotated tag name differs")
    _require(message and not message.endswith("\n"), "annotated tag message must have one canonical trailing newline")
    return (
        f"object {commit}\n"
        "type commit\n"
        f"tag {tag_name}\n"
        f"tagger {retained['name']} <{retained['email']}> {retained['epoch']} {retained['timezone']}\n"
        "\n"
        f"{message}\n"
    ).encode("utf-8")


def seal_tag(
    *,
    snapshot_repo: pathlib.Path,
    output_root: pathlib.Path,
    receipt: Mapping[str, Any],
    distribution_manifest: Mapping[str, Any],
    forbidden_objects: Iterable[str] = (),
) -> dict[str, Any]:
    retained = validate_receipt(receipt)
    manifest = validate_distribution_manifest(distribution_manifest)
    snapshot_repo = _regular_directory(snapshot_repo, "isolated snapshot repository")
    output_root = _regular_directory(output_root, "release-only state root")
    snapshot = retained["snapshot"]
    source = retained["source_authority"]
    bound_forbidden = snapshot["assertions"]["forbidden_objects_absent"]
    all_forbidden = sorted(set([*bound_forbidden, *forbidden_objects]))
    verify_isolated_repository(
        snapshot_repo,
        snapshot_commit=snapshot["snapshot_commit"],
        source_tree=source["source_tree"],
        source_commit=source["source_commit"],
        forbidden_objects=all_forbidden,
    )
    tagging_repository = output_root / "tag-construction.git"
    _require(not tagging_repository.exists(), "refusing to reuse tag construction repository")
    for path in snapshot_repo.rglob("*"):
        _require(not path.is_symlink(), "isolated snapshot repository contains a symlink")
    shutil.copytree(snapshot_repo, tagging_repository, symlinks=False)
    verify_isolated_repository(
        tagging_repository,
        snapshot_commit=snapshot["snapshot_commit"],
        source_tree=source["source_tree"],
        source_commit=source["source_commit"],
        forbidden_objects=all_forbidden,
    )
    message = tag_message(retained, manifest)
    raw_tag = _annotated_tag_payload(
        commit=snapshot["snapshot_commit"],
        tag_name=DEFAULT_TAG,
        message=message,
        identity=retained["tag_construction"]["tagger"],
    )
    tag_oid = _git(["mktag"], repo=tagging_repository, input_bytes=raw_tag).stdout.decode("ascii").strip()
    _hex40(tag_oid, "annotated mirror tag object")
    tag_ref = f"refs/tags/{DEFAULT_TAG}"
    _git(["update-ref", tag_ref, tag_oid, "0" * 40], repo=tagging_repository)
    bundle_path = output_root / "release-only.bundle"
    _require(not bundle_path.exists(), "refusing to overwrite final release-only bundle")
    _git(["bundle", "create", str(bundle_path), tag_ref], repo=tagging_repository)
    publisher = output_root / "isolated-publisher.git"
    _require(not publisher.exists(), "refusing to reuse isolated publisher repository")
    _git(["-c", "protocol.file.allow=always", "clone", "--bare", "--template=", str(bundle_path), str(publisher)])
    _git(["remote", "remove", "origin"], repo=publisher)
    _git(["update-ref", "--no-deref", "HEAD", snapshot["snapshot_commit"]], repo=publisher)
    final_inventory = object_inventory(publisher, tag_ref)
    verification = verify_isolated_repository(
        publisher,
        snapshot_commit=snapshot["snapshot_commit"],
        source_tree=source["source_tree"],
        source_commit=source["source_commit"],
        tag_name=DEFAULT_TAG,
        tag_object=tag_oid,
        forbidden_objects=all_forbidden,
        expected_inventory=final_inventory,
    )
    attestation: dict[str, Any] = {
        "schema": TAG_ATTESTATION_SCHEMA,
        "tag": DEFAULT_TAG,
        "tag_object": tag_oid,
        "tag_target": snapshot["snapshot_commit"],
        "tag_message_sha256": _sha256_bytes((message + "\n").encode("utf-8")),
        "publication_id": retained["publication_id"],
        "distribution_manifest_id": manifest["manifest_id"],
        "source_commit": source["source_commit"],
        "source_tree": source["source_tree"],
        "parent_count": verification["parent_count"],
        "reachable_commit_count": verification["reachable_commit_count"],
        "reachable_inventory": final_inventory,
        "assertions": {
            "pushed_only_from_isolated_repository": True,
            "no_force": True,
            "no_delete": True,
            "no_alternates": True,
            "source_commit_absent": True,
        },
    }
    attestation["attestation_id"] = _canonical_sha256(attestation)
    _harden_tree(output_root)
    return attestation


def validate_tag_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(value, {"schema", "tag", "tag_object", "tag_target", "tag_message_sha256", "publication_id", "distribution_manifest_id", "source_commit", "source_tree", "parent_count", "reachable_commit_count", "reachable_inventory", "assertions", "attestation_id"}, "tag attestation")
    unsigned = dict(retained)
    identity = unsigned.pop("attestation_id")
    _require(_canonical_sha256(unsigned) == _hex64(identity, "tag attestation identity"), "tag attestation identity differs")
    _require(retained["schema"] == TAG_ATTESTATION_SCHEMA and retained["tag"] == DEFAULT_TAG, "tag attestation schema differs")
    for key in ("tag_object", "tag_target", "source_commit", "source_tree"):
        _hex40(retained[key], f"tag attestation {key}")
    for key in ("tag_message_sha256", "publication_id", "distribution_manifest_id"):
        _hex64(retained[key], f"tag attestation {key}")
    _require(retained["parent_count"] == 0 and retained["reachable_commit_count"] == 1, "tag attestation history differs")
    assertions = _exact(retained["assertions"], {"pushed_only_from_isolated_repository", "no_force", "no_delete", "no_alternates", "source_commit_absent"}, "tag attestation assertions")
    _require(all(item is True for item in assertions.values()), "tag attestation assertion differs")
    return dict(retained)


def replay_tag_attestation(
    *,
    receipt: Mapping[str, Any],
    distribution_manifest: Mapping[str, Any],
    attestation: Mapping[str, Any],
    publisher_repo: pathlib.Path,
    forbidden_objects: Iterable[str] = (),
) -> dict[str, Any]:
    retained_receipt = validate_receipt(receipt)
    retained_manifest = validate_distribution_manifest(distribution_manifest)
    retained = validate_tag_attestation(attestation)
    _require(retained["publication_id"] == retained_receipt["publication_id"], "tag attestation publication identity differs")
    _require(retained["distribution_manifest_id"] == retained_manifest["manifest_id"], "tag attestation distribution manifest differs")
    source = retained_receipt["source_authority"]
    snapshot = retained_receipt["snapshot"]
    _require(retained["source_commit"] == source["source_commit"] and retained["source_tree"] == source["source_tree"], "tag attestation source authority differs")
    _require(retained["tag_target"] == snapshot["snapshot_commit"], "tag attestation snapshot target differs")
    message = tag_message(retained_receipt, retained_manifest)
    raw_tag = _annotated_tag_payload(
        commit=snapshot["snapshot_commit"],
        tag_name=DEFAULT_TAG,
        message=message,
        identity=retained_receipt["tag_construction"]["tagger"],
    )
    expected_tag = _git_object_id("tag", raw_tag)
    _require(retained["tag_object"] == expected_tag, "deterministic annotated tag object differs")
    _require(retained["tag_message_sha256"] == _sha256_bytes((message + "\n").encode("utf-8")), "annotated tag message digest differs")
    repository = _regular_directory(publisher_repo, "isolated publisher repository")
    _require(_git(["cat-file", "tag", expected_tag], repo=repository).stdout == raw_tag, "annotated tag bytes do not replay")
    bound_forbidden = retained_receipt["snapshot"]["assertions"]["forbidden_objects_absent"]
    all_forbidden = sorted(set([*bound_forbidden, *forbidden_objects]))
    verification = verify_isolated_repository(
        repository,
        snapshot_commit=snapshot["snapshot_commit"],
        source_tree=source["source_tree"],
        source_commit=source["source_commit"],
        tag_name=DEFAULT_TAG,
        tag_object=expected_tag,
        forbidden_objects=all_forbidden,
        expected_inventory=retained["reachable_inventory"],
    )
    _require(verification["parent_count"] == retained["parent_count"] and verification["reachable_commit_count"] == retained["reachable_commit_count"], "tag attestation history replay differs")
    return retained


def verify_published_repository(
    *,
    repo: pathlib.Path,
    receipt: Mapping[str, Any],
    distribution_manifest: Mapping[str, Any],
    asset_directory: pathlib.Path,
    target_history_attestation: Mapping[str, Any],
    prior_assets_root: pathlib.Path | None,
    trusted_external_audit_sha256: str,
    forbidden_objects: Iterable[str] = (),
) -> dict[str, Any]:
    repository = _regular_directory(repo, "published release-only repository")
    retained_receipt = validate_receipt(receipt)
    retained_manifest = cross_validate_distribution(distribution_manifest, retained_receipt, asset_directory)
    history = validate_v110_target_history_attestation(target_history_attestation)
    _require(
        retained_receipt["target"]["bootstrap"]["attestation_id"] == history["attestation_id"],
        "published repository history attestation identity differs",
    )
    _verify_no_repository_extensions(repository, allow_unborn_symbolic_head=True)
    message = tag_message(retained_receipt, retained_manifest)
    snapshot = retained_receipt["snapshot"]
    source = retained_receipt["source_authority"]
    raw_tag = _annotated_tag_payload(
        commit=snapshot["snapshot_commit"],
        tag_name=DEFAULT_TAG,
        message=message,
        identity=retained_receipt["tag_construction"]["tagger"],
    )
    tag_object = _git_object_id("tag", raw_tag)
    expected_refs = [*history["advertised_refs"], f"refs/tags/{DEFAULT_TAG}"]
    _require(expected_refs == sorted(expected_refs) and len(expected_refs) == len(set(expected_refs)), "published expected ref union differs")
    actual_lines = _git(["for-each-ref", "--format=%(objectname)%09%(refname)"], repo=repository).stdout.decode("utf-8").splitlines()
    prior_by_ref = {f"refs/tags/{item['tag']}": item for item in history["prior_releases"]}
    expected_lines = [
        f"{tag_object if ref == f'refs/tags/{DEFAULT_TAG}' else prior_by_ref[ref]['tag_object']}\t{ref}"
        for ref in expected_refs
    ]
    _require(actual_lines == expected_lines, "published full remote ref union differs")
    current_ref = f"refs/tags/{DEFAULT_TAG}"
    _require(_git(["cat-file", "tag", tag_object], repo=repository).stdout == raw_tag, "published annotated tag bytes differ")
    _require(_git_text(["rev-parse", f"{current_ref}^{{}}"], repo=repository) == snapshot["snapshot_commit"], "published snapshot target differs")
    current_tree, current_parents = _commit_properties(repository, snapshot["snapshot_commit"])
    _require(current_tree == source["source_tree"] and not current_parents, "published current release is not the exact parentless source tree")
    _require(tree_manifest(repository, current_tree)["manifest_sha256"] == snapshot["tree_manifest_sha256"], "published current tree manifest differs")
    verify_dependabot_suppression(repository, current_tree)
    current_inventory = object_inventory(repository, snapshot["snapshot_commit"])
    _require(
        current_inventory["inventory_sha256"] == snapshot["reachable_inventory_sha256"]
        and current_inventory["object_counts"] == snapshot["object_counts"],
        "published current snapshot inventory differs",
    )
    roots = {snapshot["snapshot_commit"]}
    if history["prior_releases"]:
        _require(prior_assets_root is not None, "published prior release verification requires prior assets")
        for authority in history["prior_releases"]:
            replayed_prior = _derive_prior_release_authority(
                repo=repository,
                tag=authority["tag"],
                release_directory=_regular_directory(prior_assets_root, "published prior assets root") / authority["tag"],
            )
            _require(replayed_prior == authority, f"published prior release {authority['tag']} live authority differs")
            roots.add(authority["snapshot_commit"])
    else:
        _require(prior_assets_root is None, "unexpected prior asset root for an empty history")
    _require(len(roots) == len(expected_refs), "published release tags do not have unique disconnected roots")
    commits = {line.decode("ascii") for line in _git(["rev-list", "--all"], repo=repository).stdout.splitlines() if line}
    _require(commits == roots, "published reachable commit union differs")
    bound_forbidden = retained_receipt["snapshot"]["assertions"]["forbidden_objects_absent"]
    for oid in sorted(set([*bound_forbidden, *forbidden_objects])):
        checked = _hex40(oid, "published forbidden object")
        _require(not _object_exists(repository, checked), f"published forbidden object exists: {checked}")
    fsck = _git(["fsck", "--full", "--no-reflogs", "--unreachable", *expected_refs], repo=repository, check=False)
    _require(fsck.returncode == 0, "published repository fsck failed")
    findings = (fsck.stdout + fsck.stderr).decode("utf-8", "replace").strip()
    non_notice = [line for line in findings.splitlines() if not line.startswith("notice: HEAD points to an unborn branch")]
    _require(not non_notice, "published repository has unreachable or hidden objects")
    audit_record_value = retained_receipt["target"]["external_mirror_audit"]
    audit_path = _regular_file(pathlib.Path(asset_directory) / audit_record_value["name"], "published independent audit")
    _require(external_audit_record(audit_path, trusted_external_audit_sha256) == audit_record_value, "published independent audit bytes differ")
    audit = validate_external_mirror_audit(_load_json(audit_path, "published independent audit"))
    verify_audited_implementation(audit)
    baseline_ref_records = [
        {"object_id": item["tag_object"], "ref": f"refs/tags/{item['tag']}"}
        for item in history["prior_releases"]
    ]
    _require(
        audit["source_commit"] == source["source_commit"]
        and audit["source_tree"] == source["source_tree"]
        and audit["snapshot_id"] == snapshot["snapshot_id"]
        and audit["snapshot_commit"] == snapshot["snapshot_commit"]
        and audit["target_repository"] == retained_receipt["target"]["repository"]
        and audit["source_repository"] == source["repository"]
        and audit["source_tag_object"] == source["source_tag"]["object_id"]
        and audit["private_release"] == {
            "id": source["private_release"]["id"],
            "node_id": source["private_release"]["node_id"],
            "tag": DEFAULT_TAG,
            "published": True,
            "prerelease": False,
            "immutable": True,
        }
        and audit["organization_access"] == retained_receipt["target"]["organization_access"]
        and audit["source_verifier_sha256"] == source["verification"]["verifier_sha256"],
        "published independent audit semantic authority differs",
    )
    _require(
        audit["target_history"]["advertised_refs_sha256"] == _canonical_sha256(baseline_ref_records)
        and history["external_audit"] == {key: audit_record_value[key] for key in ("name", "size_bytes", "sha256", "audit_id")},
        "published independent audit baseline ref authority differs",
    )
    result: dict[str, Any] = {
        "state": "PUBLISHED_RELEASE_ONLY_VERIFIED",
        "publication_id": retained_receipt["publication_id"],
        "manifest_id": retained_manifest["manifest_id"],
        "tag_object": tag_object,
        "refs": expected_refs,
        "roots": sorted(roots),
    }
    result["verification_id"] = _canonical_sha256(result)
    return result


def verify_empty_target_repository(repo: pathlib.Path) -> dict[str, Any]:
    repository = _regular_directory(repo, "empty release-only target")
    _verify_no_repository_extensions(repository, allow_unborn_symbolic_head=True)
    refs = _git(["for-each-ref", "--format=%(refname)"], repo=repository).stdout
    _require(not refs.strip(), "empty release-only target has advertised refs")
    counts = _git(["count-objects", "-v"], repo=repository).stdout.decode("ascii")
    values = dict(line.split(": ", 1) for line in counts.splitlines() if ": " in line)
    _require(values.get("count") == "0" and values.get("in-pack") == "0", "empty release-only target contains hidden objects")
    return {"state": "EMPTY_TARGET_VERIFIED"}


def release_title() -> str:
    return "gpmeep v1.1.0 release-only mirror"


def release_notes(
    receipt: Mapping[str, Any], distribution_manifest: Mapping[str, Any]
) -> str:
    retained_receipt = validate_receipt(receipt)
    retained_manifest = cross_validate_distribution_metadata(
        distribution_manifest, retained_receipt
    )
    return (
        f"Authoritative private source commit: {retained_receipt['source_authority']['source_commit']}\n"
        f"Parentless distribution snapshot: {retained_receipt['snapshot']['snapshot_commit']}\n"
        f"Mirror publication ID: {retained_receipt['publication_id']}\n"
        f"Distribution asset manifest ID: {retained_manifest['manifest_id']}"
    )


def _approved_github_json(
    *,
    gh_binary: pathlib.Path,
    run_intent: Mapping[str, Any],
    arguments: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if run_intent.get("schema") == PRIOR_MIGRATION_RUN_INTENT_SCHEMA:
        retained_intent = validate_v103_migration_run_intent(run_intent)
    else:
        retained_intent = validate_run_intent(run_intent)
    transport = retained_intent["transport"]
    _require(transport["mode"] == "github", f"{label} requires GitHub transport")
    binary = _regular_file(gh_binary, "approved GitHub CLI")
    binary_identity = _file_identity(binary)
    binary_digest = _sha256_file(binary)
    _require(
        pathlib.Path(transport["gh_binary"]["path"]) == binary
        and binary.stat().st_size == transport["gh_binary"]["size_bytes"]
        and binary_digest
        == transport["gh_binary"]["sha256"]
        == APPROVED_GITHUB_CLI_SHA256,
        "publication terminal GitHub CLI is not the audited approved binary",
    )
    _require(os.access(binary, os.X_OK), "approved GitHub CLI is not executable")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    _require(
        isinstance(token, str) and token and "\x00" not in token,
        f"{label} requires an authentication token",
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "XDG_CONFIG_HOME": "/nonexistent",
        "GH_CONFIG_DIR": "/nonexistent",
        "GH_PAGER": "cat",
        "NO_COLOR": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GH_TOKEN": token,
    }
    completed = subprocess.run(
        [str(binary), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd="/",
        check=False,
        timeout=120,
    )
    _require(completed.returncode == 0, f"approved GitHub CLI did not verify {label}")
    _require(
        _file_identity(binary) == binary_identity
        and _sha256_file(binary) == APPROVED_GITHUB_CLI_SHA256,
        f"approved GitHub CLI changed during {label}",
    )
    _require(0 < len(completed.stdout) <= 16 * 1024 * 1024, f"{label} output size differs")
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseOnlyError(f"{label} output is not strict JSON") from error
    _require(isinstance(value, dict), f"{label} output is not a JSON object")
    return value


def _live_github_terminal_metadata(
    *, gh_binary: pathlib.Path, run_intent: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_repository = _approved_github_json(
        gh_binary=gh_binary,
        run_intent=run_intent,
        arguments=[
            "api", "-H", "Accept: application/vnd.github+json", "-H",
            "X-GitHub-Api-Version: 2026-03-10", f"/repos/{TARGET_REPOSITORY}",
        ],
        label="live target repository authority",
    )
    permissions = raw_repository.get("permissions")
    repository = {
        "full_name": raw_repository.get("full_name"),
        "id": raw_repository.get("id"),
        "node_id": raw_repository.get("node_id"),
        "private": raw_repository.get("private"),
        "visibility": raw_repository.get("visibility"),
        "fork": raw_repository.get("fork"),
        "is_template": raw_repository.get("is_template"),
        "archived": raw_repository.get("archived"),
        "disabled": raw_repository.get("disabled"),
        "permissions": {"admin": permissions.get("admin")} if isinstance(permissions, dict) else None,
    }
    raw_release = _approved_github_json(
        gh_binary=gh_binary,
        run_intent=run_intent,
        arguments=[
            "release", "view", DEFAULT_TAG, "--repo", TARGET_REPOSITORY,
            "--json", "apiUrl,assets,body,databaseId,id,isDraft,isImmutable,isPrerelease,name,tagName",
        ],
        label="live immutable release authority",
    )
    raw_release = _exact(
        raw_release,
        {"apiUrl", "assets", "body", "databaseId", "id", "isDraft", "isImmutable", "isPrerelease", "name", "tagName"},
        "live immutable release projection",
    )
    release_url = raw_release["apiUrl"]
    release_match = re.fullmatch(
        rf"https://api\.github\.com/repos/{re.escape(TARGET_REPOSITORY)}/releases/([1-9][0-9]*)",
        release_url if isinstance(release_url, str) else "",
    )
    _require(
        release_match is not None
        and int(release_match.group(1)) == raw_release["databaseId"],
        "live immutable release API identity differs",
    )
    raw_assets = raw_release["assets"]
    _require(isinstance(raw_assets, list), "live immutable release asset projection differs")
    assets: list[dict[str, Any]] = []
    for raw_asset in raw_assets:
        retained_asset = _exact(
            raw_asset,
            {"url", "apiUrl", "id", "name", "label", "size", "digest", "state", "createdAt", "updatedAt", "downloadCount", "contentType"},
            "live immutable release asset",
        )
        asset_url = retained_asset["apiUrl"]
        asset_match = re.fullmatch(
            rf"https://api\.github\.com/repos/{re.escape(TARGET_REPOSITORY)}/releases/assets/([1-9][0-9]*)",
            asset_url if isinstance(asset_url, str) else "",
        )
        _require(asset_match is not None, "live immutable release asset API identity differs")
        assets.append(
            {
                "id": int(asset_match.group(1)),
                "nodeId": retained_asset["id"],
                "name": retained_asset["name"],
                "sizeBytes": retained_asset["size"],
                "digest": retained_asset["digest"],
            }
        )
    release = {
        "id": raw_release["databaseId"],
        "nodeId": raw_release["id"],
        "tagName": raw_release["tagName"],
        "name": raw_release["name"],
        "body": raw_release["body"],
        "isDraft": raw_release["isDraft"],
        "isPrerelease": raw_release["isPrerelease"],
        "isImmutable": raw_release["isImmutable"],
        "assets": assets,
    }
    return repository, release


def _validate_github_release_attestation_output(
    *,
    output: Mapping[str, Any],
    repository_slug: str,
    repository_id: int,
    owner_id: int,
    tag: str,
    release_id: int,
    tag_object: str,
    asset_records: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    _require(SAFE_SLUG.fullmatch(repository_slug) is not None, "GitHub signed release repository differs")
    _positive_int(repository_id, "GitHub signed release repository ID")
    _positive_int(owner_id, "GitHub signed release owner ID")
    _require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is not None, "GitHub signed release tag differs")
    _positive_int(release_id, "attested release ID")
    tag_object = _hex40(tag_object, "GitHub signed release tag object")
    retained_output = _exact(
        output, {"attestation", "verificationResult"},
        "GitHub release attestation verification output",
    )
    attestation = _exact(
        retained_output["attestation"], {"bundle", "bundle_url", "initiator"},
        "GitHub release attestation",
    )
    _require(
        attestation["bundle_url"] == "" and attestation["initiator"] == "",
        "GitHub CLI returned unexpected unverified attestation metadata",
    )
    bundle = _exact(
        attestation["bundle"], {"mediaType", "verificationMaterial", "dsseEnvelope"},
        "GitHub release attestation bundle",
    )
    _require(
        bundle["mediaType"] == "application/vnd.dev.sigstore.bundle.v0.3+json",
        "GitHub release attestation bundle media type differs",
    )
    verification_material = _exact(
        bundle["verificationMaterial"], {"timestampVerificationData", "certificate"},
        "GitHub release attestation verification material",
    )
    certificate = _exact(
        verification_material["certificate"], {"rawBytes"},
        "GitHub release attestation certificate",
    )
    timestamp_data = _exact(
        verification_material["timestampVerificationData"], {"rfc3161Timestamps"},
        "GitHub release attestation timestamp material",
    )
    timestamps = timestamp_data["rfc3161Timestamps"]
    _require(isinstance(timestamps, list) and timestamps, "GitHub release attestation has no signed timestamp")
    timestamp_digests: list[str] = []
    for entry in timestamps:
        timestamp = _exact(entry, {"signedTimestamp"}, "GitHub signed release timestamp")
        try:
            raw_timestamp = base64.b64decode(timestamp["signedTimestamp"], validate=True)
        except (TypeError, ValueError) as error:
            raise ReleaseOnlyError("GitHub signed release timestamp is not canonical base64") from error
        _require(len(raw_timestamp) >= 64, "GitHub signed release timestamp is too small")
        timestamp_digests.append(_sha256_bytes(raw_timestamp))
    try:
        raw_certificate = base64.b64decode(certificate["rawBytes"], validate=True)
    except (TypeError, ValueError) as error:
        raise ReleaseOnlyError("GitHub release certificate is not canonical base64") from error
    _require(len(raw_certificate) >= 128, "GitHub release certificate is too small")
    envelope = _exact(
        bundle["dsseEnvelope"], {"payload", "payloadType", "signatures"},
        "GitHub release DSSE envelope",
    )
    _require(
        envelope["payloadType"] == "application/vnd.in-toto+json",
        "GitHub release DSSE payload type differs",
    )
    signatures = envelope["signatures"]
    _require(isinstance(signatures, list) and len(signatures) == 1, "GitHub release DSSE signature union differs")
    signature = _exact(signatures[0], {"sig"}, "GitHub release DSSE signature")
    try:
        raw_signature = base64.b64decode(signature["sig"], validate=True)
        raw_payload = base64.b64decode(envelope["payload"], validate=True)
        statement = json.loads(raw_payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseOnlyError("GitHub release DSSE envelope encoding differs") from error
    _require(len(raw_signature) >= 48, "GitHub release DSSE signature is too small")
    statement = _exact(
        statement, {"_type", "subject", "predicateType", "predicate"},
        "GitHub signed release statement",
    )
    verification = retained_output["verificationResult"]
    _require(
        isinstance(verification, dict)
        and set(verification).issubset({"mediaType", "statement", "signature", "verifiedTimestamps", "verifiedIdentity"})
        and {"mediaType", "statement", "signature", "verifiedTimestamps"}.issubset(verification),
        "GitHub release verification-result projection differs",
    )
    _require(
        verification["mediaType"] == "application/vnd.dev.sigstore.verificationresult+json;version=0.1"
        and verification["statement"] == statement
        and isinstance(verification["verifiedTimestamps"], list)
        and verification["verifiedTimestamps"],
        "GitHub CLI did not return a fully verified signed statement",
    )
    verified_signature = _exact(
        verification["signature"], {"certificate"},
        "GitHub release verified signature",
    )
    verified_certificate = _exact(
        verified_signature["certificate"], {"certificateIssuer", "subjectAlternativeName"},
        "GitHub release verified certificate",
    )
    _require(
        isinstance(verified_certificate["certificateIssuer"], str)
        and verified_certificate["certificateIssuer"]
        and verified_certificate["subjectAlternativeName"] == "https://dotcom.releases.github.com",
        "GitHub release verifier certificate identity differs",
    )
    verified_identity = _exact(
        verification.get("verifiedIdentity"), {"subjectAlternativeName", "issuer"},
        "GitHub release verified identity policy",
    )
    _require(
        _exact(
            verified_identity["subjectAlternativeName"],
            {"subjectAlternativeName", "regexp"},
            "GitHub release verified SAN policy",
        )
        == {
            "subjectAlternativeName": "",
            "regexp": r"^https://dotcom\.releases\.github\.com$",
        }
        and _exact(
            verified_identity["issuer"], {"issuer", "regexp"},
            "GitHub release verified issuer policy",
        )
        == {"issuer": "", "regexp": ".*"},
        "GitHub release verifier identity policy differs",
    )
    for verified_timestamp in verification["verifiedTimestamps"]:
        retained_timestamp = _exact(
            verified_timestamp, {"type", "uri", "timestamp"},
            "GitHub release verified timestamp",
        )
        _require(
            retained_timestamp["type"] == "TimestampAuthority"
            and retained_timestamp["uri"] == "timestamp.githubapp.com"
            and isinstance(retained_timestamp["timestamp"], str)
            and retained_timestamp["timestamp"].endswith("Z"),
            "GitHub release verified timestamp authority differs",
        )
    expected_purl = f"pkg:github/{repository_slug}@{tag}"
    expected_predicate = {
        "databaseId": str(release_id),
        "ownerId": str(owner_id),
        "packageId": str(repository_id),
        "purl": expected_purl,
        "repository": repository_slug,
        "repositoryId": str(repository_id),
        "tag": tag,
    }
    _require(
        statement["_type"] == "https://in-toto.io/Statement/v1"
        and statement["predicateType"] == "https://in-toto.io/attestation/release/v0.2"
        and statement["predicate"] == expected_predicate,
        "GitHub signed release predicate differs from publication authority",
    )
    subjects = statement["subject"]
    _require(isinstance(subjects, list) and subjects, "GitHub signed release subject union differs")
    tag_subjects = [subject for subject in subjects if isinstance(subject, dict) and "uri" in subject]
    asset_subjects = [subject for subject in subjects if isinstance(subject, dict) and "name" in subject]
    _require(
        len(tag_subjects) == 1
        and len(asset_subjects) + 1 == len(subjects)
        and _exact(
            tag_subjects[0], {"uri", "digest"},
            "GitHub signed release tag subject",
        ) == {"uri": expected_purl, "digest": {"sha1": tag_object}},
        "GitHub signed release tag object differs",
    )
    signed_assets: list[dict[str, str]] = []
    seen: set[str] = set()
    for subject in asset_subjects:
        retained_subject = _exact(subject, {"name", "digest"}, "GitHub signed release asset subject")
        name = retained_subject["name"]
        _require(isinstance(name, str) and name in asset_records and name not in seen, "GitHub signed release asset name differs")
        expected_size, expected_digest = asset_records[name]
        _require(
            retained_subject["digest"] == {"sha256": expected_digest},
            f"GitHub signed release asset digest differs: {name}",
        )
        _nonnegative_int(expected_size, f"GitHub signed release asset size: {name}")
        seen.add(name)
        signed_assets.append({"name": name, "sha256": expected_digest})
    _require(seen == set(asset_records), "GitHub signed release asset union differs")
    signed_assets.sort(key=lambda item: item["name"])
    return {
        "verification_output_sha256": _canonical_sha256(retained_output),
        "bundle_sha256": _canonical_sha256(bundle),
        "statement_sha256": _canonical_sha256(statement),
        "certificate_sha256": _sha256_bytes(raw_certificate),
        "certificate_subject_alternative_name": verified_certificate["subjectAlternativeName"],
        "timestamp_sha256": sorted(timestamp_digests),
        "signature_sha256": _sha256_bytes(raw_signature),
        "predicate_type": statement["predicateType"],
        "predicate": expected_predicate,
        "tag_subject": {"uri": expected_purl, "object_id": tag_object},
        "asset_subjects": signed_assets,
    }


def _verified_github_release_attestation(
    *,
    gh_binary: pathlib.Path,
    run_intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    distribution_manifest: Mapping[str, Any],
    release_metadata: Mapping[str, Any],
    asset_records: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    retained_intent = validate_run_intent(run_intent)
    retained_receipt = validate_receipt(receipt)
    retained_manifest = cross_validate_distribution_metadata(
        distribution_manifest, retained_receipt
    )
    output = _approved_github_json(
        gh_binary=gh_binary,
        run_intent=retained_intent,
        arguments=[
            "release", "verify", DEFAULT_TAG, "--repo", TARGET_REPOSITORY,
            "--format", "json",
        ],
        label="signed immutable-release attestation",
    )
    snapshot = retained_receipt["snapshot"]
    expected_tag_object = _git_object_id(
        "tag",
        _annotated_tag_payload(
            commit=snapshot["snapshot_commit"],
            tag_name=DEFAULT_TAG,
            message=tag_message(retained_receipt, retained_manifest),
            identity=retained_receipt["tag_construction"]["tagger"],
        ),
    )
    return _validate_github_release_attestation_output(
        output=output,
        repository_slug=TARGET_REPOSITORY,
        repository_id=retained_receipt["target"]["repository"]["id"],
        owner_id=retained_receipt["target"]["organization_access"]["id"],
        tag=DEFAULT_TAG,
        release_id=_positive_int(release_metadata["id"], "attested release ID"),
        tag_object=expected_tag_object,
        asset_records=asset_records,
    )


def _write_or_verify_canonical_json(path: pathlib.Path, value: Mapping[str, Any], label: str) -> pathlib.Path:
    payload = _canonical_bytes(value) + b"\n"
    if path.exists() or path.is_symlink():
        existing = _regular_file(path, label)
        _require(existing.read_bytes() == payload, f"{label} differs on resume")
        return existing
    return _write_new_bytes(path, payload, label)


def validate_v103_migration_terminal(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "state", "intent_id", "audit", "prior_release_authority", "live_evidence", "incident_resolution", "terminal_id"},
        "v1.0.3 migration terminal",
    )
    unsigned = dict(retained)
    terminal_id = unsigned.pop("terminal_id")
    _require(_canonical_sha256(unsigned) == _hex64(terminal_id, "v1.0.3 migration terminal identity"), "v1.0.3 migration terminal identity differs")
    _require(retained["schema"] == PRIOR_MIGRATION_TERMINAL_SCHEMA and retained["state"] == "IMMUTABLE_MIGRATION_VERIFIED", "v1.0.3 migration terminal is not verified")
    _hex64(retained["intent_id"], "v1.0.3 migration intent identity")
    audit = _exact(retained["audit"], {"sha256", "audit_id"}, "v1.0.3 terminal audit")
    _hex64(audit["sha256"], "v1.0.3 terminal audit digest")
    _hex64(audit["audit_id"], "v1.0.3 terminal audit identity")
    validate_prior_release_authority(retained["prior_release_authority"])
    evidence = _exact(retained["live_evidence"], {"repository_sha256", "organization_sha256", "release_sha256", "attestation_sha256"}, "v1.0.3 terminal live evidence")
    for key, digest in evidence.items():
        _hex64(digest, f"v1.0.3 terminal {key}")
    incident = retained["incident_resolution"]
    if incident is not None:
        incident = _exact(incident, {"name", "sha256", "incident_id", "resolved_by"}, "v1.0.3 terminal incident resolution")
        _require(
            incident["name"] == "v103-migration-incident.json"
            and incident["resolved_by"] == "SIGNED_IMMUTABLE_MIGRATION_TERMINAL",
            "v1.0.3 terminal incident resolution differs",
        )
        _hex64(incident["sha256"], "v1.0.3 resolved incident digest")
        _hex64(incident["incident_id"], "v1.0.3 resolved incident identity")
    return dict(retained)


def validate_v103_migration_incident(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {"schema", "state", "stage", "exit_status", "observed_at_utc", "audit_sha256", "intent_id", "release", "incident_id"},
        "v1.0.3 migration incident",
    )
    unsigned = dict(retained)
    incident_id = unsigned.pop("incident_id")
    _require(_canonical_sha256(unsigned) == _hex64(incident_id, "v1.0.3 migration incident identity"), "v1.0.3 migration incident identity differs")
    _require(retained["schema"] == PRIOR_MIGRATION_INCIDENT_SCHEMA and retained["state"] == "IMMUTABLE_MIGRATION_INCOMPLETE", "v1.0.3 migration incident state differs")
    _require(isinstance(retained["stage"], str) and retained["stage"] and all(ord(item) >= 32 for item in retained["stage"]), "v1.0.3 migration incident stage differs")
    _positive_int(retained["exit_status"], "v1.0.3 migration incident exit status")
    _require(isinstance(retained["observed_at_utc"], str) and retained["observed_at_utc"].endswith("Z"), "v1.0.3 migration incident timestamp differs")
    _hex64(retained["audit_sha256"], "v1.0.3 migration incident audit digest")
    _hex64(retained["intent_id"], "v1.0.3 migration incident intent identity")
    release = _exact(retained["release"], {"id", "node_id"}, "v1.0.3 migration incident release")
    _positive_int(release["id"], "v1.0.3 migration incident release ID")
    _require(isinstance(release["node_id"], str) and release["node_id"], "v1.0.3 migration incident release node ID differs")
    return dict(retained)


def create_v103_migration_incident(
    *, stage: str, exit_status: int, observed_at_utc: str, audit_sha256: str,
    intent_id: str, release_id: int, release_node_id: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": PRIOR_MIGRATION_INCIDENT_SCHEMA,
        "state": "IMMUTABLE_MIGRATION_INCOMPLETE",
        "stage": stage, "exit_status": exit_status,
        "observed_at_utc": observed_at_utc, "audit_sha256": audit_sha256,
        "intent_id": intent_id,
        "release": {"id": release_id, "node_id": release_node_id},
    }
    value["incident_id"] = _canonical_sha256(value)
    return validate_v103_migration_incident(value)


def create_v103_migration_terminal(
    *, run_intent: Mapping[str, Any], audit_path: pathlib.Path, trusted_audit_sha256: str,
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], manifest_path: pathlib.Path,
    target_repo: pathlib.Path, evidence_directory: pathlib.Path, gh_binary: pathlib.Path,
    incident_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    intent = validate_v103_migration_run_intent(run_intent)
    audit_file = _regular_file(audit_path, "v1.0.3 migration terminal audit")
    _require(
        _sha256_file(audit_file) == trusted_audit_sha256 == intent["audit"]["sha256"],
        "v1.0.3 terminal audit differs from run intent",
    )
    audit = validate_v103_migration_audit(_load_json(audit_file, "v1.0.3 migration terminal audit"))
    _require(audit["audit_id"] == intent["audit"]["audit_id"], "v1.0.3 terminal audit identity differs from run intent")
    retained_receipt = _validate_v103_migration_receipt(receipt)
    retained_manifest = _validate_v103_migration_manifest(manifest)
    _require(
        retained_receipt["source_authority"] == intent["source_authority"]
        and {**retained_receipt["snapshot"], "reachable_inventory": intent["snapshot"]["reachable_inventory"]} == intent["snapshot"]
        and retained_receipt["target"]["repository"] == intent["target"]["repository"]
        and retained_receipt["target"]["organization"] == intent["target"]["organization"],
        "v1.0.3 terminal receipt differs from migration intent",
    )
    root = _regular_directory(evidence_directory, "v1.0.3 migration evidence directory")
    manifest_file = _regular_file(manifest_path, "v1.0.3 migration manifest")
    _require(manifest_file.parent == root and _load_json(manifest_file, "v1.0.3 migration manifest") == retained_manifest, "v1.0.3 migration manifest evidence differs")
    repository_raw = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10", f"/repos/{TARGET_REPOSITORY}"],
        label="v1.0.3 live target repository",
    )
    organization_raw = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10", "/orgs/example-organization"],
        label="v1.0.3 live target organization",
    )
    release_raw = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10", f"/repos/{TARGET_REPOSITORY}/releases/tags/v1.0.3"],
        label="v1.0.3 live immutable release",
    )
    metadata = create_v103_release_metadata(
        receipt=retained_receipt, manifest=retained_manifest,
        repository_metadata=repository_raw, organization_metadata=organization_raw,
        release_metadata=release_raw, manifest_path=manifest_file,
    )
    attestation = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["release", "verify", "v1.0.3", "--repo", TARGET_REPOSITORY, "--format", "json"],
        label="v1.0.3 signed immutable-release attestation",
    )
    _write_or_verify_canonical_json(root / PRIOR_RELEASE_METADATA_NAME, metadata, "v1.0.3 live release evidence")
    _write_or_verify_canonical_json(root / PRIOR_RELEASE_ATTESTATION_NAME, attestation, "v1.0.3 signed release evidence")
    authority = _derive_prior_release_authority(repo=target_repo, tag="v1.0.3", release_directory=root)
    repository_after = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10", f"/repos/{TARGET_REPOSITORY}"],
        label="v1.0.3 final target repository recheck",
    )
    release_after = _approved_github_json(
        gh_binary=gh_binary, run_intent=intent,
        arguments=["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10", f"/repos/{TARGET_REPOSITORY}/releases/tags/v1.0.3"],
        label="v1.0.3 final immutable release recheck",
    )
    _require(repository_after == repository_raw and release_after == release_raw, "v1.0.3 live authority changed during terminal creation")
    incident_resolution: dict[str, Any] | None = None
    if incident_path is not None:
        incident_file = _regular_file(incident_path, "v1.0.3 migration incident")
        incident = validate_v103_migration_incident(_load_json(incident_file, "v1.0.3 migration incident"))
        _require(
            incident["audit_sha256"] == trusted_audit_sha256
            and incident["intent_id"] == intent["intent_id"]
            and incident["release"] == {
                "id": retained_receipt["target"]["release"]["id"],
                "node_id": retained_receipt["target"]["release"]["node_id"],
            },
            "v1.0.3 migration incident differs from the recovered terminal authority",
        )
        incident_resolution = {
            "name": incident_file.name, "sha256": _sha256_file(incident_file),
            "incident_id": incident["incident_id"],
            "resolved_by": "SIGNED_IMMUTABLE_MIGRATION_TERMINAL",
        }
    value: dict[str, Any] = {
        "schema": PRIOR_MIGRATION_TERMINAL_SCHEMA,
        "state": "IMMUTABLE_MIGRATION_VERIFIED",
        "intent_id": intent["intent_id"],
        "audit": intent["audit"],
        "prior_release_authority": authority,
        "live_evidence": {
            "repository_sha256": _canonical_sha256(repository_raw),
            "organization_sha256": _canonical_sha256(organization_raw),
            "release_sha256": _canonical_sha256(release_raw),
            "attestation_sha256": _canonical_sha256(attestation),
        },
        "incident_resolution": incident_resolution,
    }
    value["terminal_id"] = _canonical_sha256(value)
    return validate_v103_migration_terminal(value)


def create_publication_terminal(
    *,
    receipt: Mapping[str, Any],
    distribution_manifest: Mapping[str, Any],
    tag_attestation: Mapping[str, Any],
    run_intent: Mapping[str, Any],
    asset_directory: pathlib.Path,
    tag_object: str,
    gh_binary: pathlib.Path,
    gh_capabilities_sha256: str,
    trusted_external_audit_sha256: str,
) -> dict[str, Any]:
    retained_receipt = validate_receipt(receipt)
    retained_manifest = cross_validate_distribution(
        distribution_manifest, retained_receipt, asset_directory
    )
    retained_tag = validate_tag_attestation(tag_attestation)
    retained_intent = validate_run_intent(run_intent)
    trusted_audit_digest = _hex64(
        trusted_external_audit_sha256, "trusted independent mirror audit digest"
    )
    source = retained_receipt["source_authority"]
    snapshot = retained_receipt["snapshot"]
    _require(
        retained_receipt["target"]["external_mirror_audit"]["sha256"] == trusted_audit_digest
        and retained_intent["trusted_external_audit_sha256"] == trusted_audit_digest,
        "terminal trusted independent audit authority differs",
    )
    audit_path = pathlib.Path(retained_intent["files"]["external_audit"]["path"])
    _require(
        _intent_file(audit_path, "terminal independent mirror audit")
        == retained_intent["files"]["external_audit"],
        "terminal independent audit bytes differ from run intent",
    )
    audit = validate_external_mirror_audit(_load_json(audit_path, "terminal independent mirror audit"))
    _require(
        audit["github_cli"] == {
            "version": AUDITED_GITHUB_CLI_VERSION,
            "sha256": AUDITED_GITHUB_CLI_SHA256,
        },
        "terminal independent audit does not approve the fixed GitHub CLI",
    )
    _require(
        retained_intent["repository"] == TARGET_REPOSITORY
        and retained_intent["transport"]["mode"] == "github"
        and retained_intent["transport"]["remote_url"] == f"git@github.com:{TARGET_REPOSITORY}.git",
        "terminal run intent is not the fixed production GitHub transport",
    )
    repository_metadata, release_metadata = _live_github_terminal_metadata(
        gh_binary=gh_binary, run_intent=retained_intent
    )
    expected_tag_message = tag_message(retained_receipt, retained_manifest)
    expected_tag_object = _git_object_id(
        "tag",
        _annotated_tag_payload(
            commit=snapshot["snapshot_commit"],
            tag_name=DEFAULT_TAG,
            message=expected_tag_message,
            identity=retained_receipt["tag_construction"]["tagger"],
        ),
    )
    _require(
        retained_tag["tag_object"] == expected_tag_object == _hex40(tag_object, "terminal tag object")
        and retained_tag["tag_target"] == snapshot["snapshot_commit"]
        and retained_tag["source_commit"] == source["source_commit"]
        and retained_tag["source_tree"] == source["source_tree"]
        and retained_tag["publication_id"] == retained_receipt["publication_id"]
        and retained_tag["distribution_manifest_id"] == retained_manifest["manifest_id"]
        and retained_tag["tag_message_sha256"] == _sha256_bytes((expected_tag_message + "\n").encode("utf-8")),
        "terminal tag attestation differs from canonical release authority",
    )
    repository = retained_receipt["target"]["repository"]
    permissions = repository_metadata.get("permissions")
    _require(
        repository_metadata.get("full_name") == repository["slug"]
        and repository_metadata.get("id") == repository["id"]
        and repository_metadata.get("node_id") == repository["node_id"]
        and repository_metadata.get("private") is True
        and repository_metadata.get("visibility") == "private"
        and repository_metadata.get("fork") is False
        and repository_metadata.get("is_template") is False
        and repository_metadata.get("archived") is False
        and repository_metadata.get("disabled") is False
        and isinstance(permissions, dict)
        and permissions.get("admin") is True,
        "terminal target repository authority differs",
    )
    release = _exact(
        release_metadata,
        {
            "id", "nodeId", "tagName", "name", "body", "isDraft",
            "isPrerelease", "isImmutable", "assets",
        },
        "terminal live release projection",
    )
    _positive_int(release["id"], "terminal release ID")
    _require(isinstance(release["nodeId"], str) and release["nodeId"], "terminal release node ID differs")
    _require(
        release["tagName"] == DEFAULT_TAG
        and release["isDraft"] is False
        and release["isPrerelease"] is False
        and release["isImmutable"] is True
        and release["name"] == release_title()
        and release["body"] == release_notes(retained_receipt, retained_manifest),
        "terminal release is not final and immutable",
    )
    root = _regular_directory(asset_directory, "terminal release download")
    manifest_records = {
        record["name"]: (record["size_bytes"], record["sha256"])
        for record in retained_manifest["assets"]
    }
    manifest_path = _regular_file(root / DISTRIBUTION_MANIFEST_NAME, "terminal distribution manifest")
    manifest_records[DISTRIBUTION_MANIFEST_NAME] = (
        manifest_path.stat().st_size,
        _sha256_file(manifest_path),
    )
    live_assets = release["assets"]
    _require(isinstance(live_assets, list), "terminal live asset inventory differs")
    terminal_assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset in live_assets:
        retained_asset = _exact(
            asset, {"id", "nodeId", "name", "sizeBytes", "digest"},
            "terminal live release asset",
        )
        _positive_int(retained_asset["id"], "terminal asset ID")
        _require(isinstance(retained_asset["nodeId"], str) and retained_asset["nodeId"], "terminal asset node ID differs")
        name = retained_asset["name"]
        _require(isinstance(name, str) and name in manifest_records and name not in seen, "terminal asset name differs")
        expected_size, expected_sha = manifest_records[name]
        _require(
            retained_asset["sizeBytes"] == expected_size
            and retained_asset["digest"] == f"sha256:{expected_sha}",
            f"terminal asset bytes differ: {name}",
        )
        seen.add(name)
        terminal_assets.append(
            {
                "id": retained_asset["id"],
                "node_id": retained_asset["nodeId"],
                "name": name,
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )
    _require(seen == set(manifest_records), "terminal asset union differs")
    terminal_assets.sort(key=lambda item: item["name"])
    release_attestation = _verified_github_release_attestation(
        gh_binary=gh_binary,
        run_intent=retained_intent,
        receipt=retained_receipt,
        distribution_manifest=retained_manifest,
        release_metadata=release,
        asset_records=manifest_records,
    )
    final_repository_metadata, final_release_metadata = _live_github_terminal_metadata(
        gh_binary=gh_binary, run_intent=retained_intent
    )
    _require(
        final_repository_metadata == repository_metadata
        and final_release_metadata == release_metadata,
        "live GitHub repository or release authority changed during terminal creation",
    )
    terminal: dict[str, Any] = {
        "schema": PUBLICATION_TERMINAL_SCHEMA,
        "state": "ORGANIZATION_RELEASE_PUBLISHED_IMMUTABLE",
        "repository": {
            "slug": repository["slug"],
            "id": repository["id"],
            "node_id": repository["node_id"],
            "private": True,
            "visibility": "private",
            "archived": False,
            "disabled": False,
            "administration_read_authorized": True,
        },
        "release": {
            "id": release["id"],
            "node_id": release["nodeId"],
            "tag": DEFAULT_TAG,
            "title": release["name"],
            "body": release["body"],
            "draft": False,
            "prerelease": False,
            "immutable": True,
        },
        "tag": {"ref": f"refs/tags/{DEFAULT_TAG}", "object_id": _hex40(tag_object, "terminal tag object")},
        "publication_id": retained_receipt["publication_id"],
        "manifest_id": retained_manifest["manifest_id"],
        "source_commit": source["source_commit"],
        "snapshot_commit": snapshot["snapshot_commit"],
        "tag_attestation_id": retained_tag["attestation_id"],
        "trusted_external_audit_sha256": trusted_audit_digest,
        "run_intent": {
            "intent_id": retained_intent["intent_id"],
            "mode": "github",
            "remote_url": retained_intent["transport"]["remote_url"],
            "gh_sha256": retained_intent["transport"]["gh_binary"]["sha256"],
            "gh_capabilities_sha256": retained_intent["transport"]["capabilities_sha256"],
        },
        "github_cli": {
            "version": AUDITED_GITHUB_CLI_VERSION,
            "sha256": APPROVED_GITHUB_CLI_SHA256,
            "capabilities_sha256": _hex64(gh_capabilities_sha256, "terminal GitHub CLI capabilities"),
        },
        "github_release_attestation": release_attestation,
        "assets": terminal_assets,
    }
    terminal["terminal_id"] = _canonical_sha256(terminal)
    return validate_publication_terminal(terminal)


def validate_publication_terminal(value: Mapping[str, Any]) -> dict[str, Any]:
    retained = _exact(
        value,
        {
            "schema", "state", "repository", "release", "tag",
            "publication_id", "manifest_id", "source_commit", "snapshot_commit",
            "tag_attestation_id", "trusted_external_audit_sha256", "run_intent",
            "github_cli", "github_release_attestation", "assets", "terminal_id",
        },
        "publication terminal",
    )
    unsigned = dict(retained)
    terminal_id = unsigned.pop("terminal_id")
    _require(_canonical_sha256(unsigned) == _hex64(terminal_id, "publication terminal identity"), "publication terminal identity differs")
    _require(
        retained["schema"] == PUBLICATION_TERMINAL_SCHEMA
        and retained["state"] == "ORGANIZATION_RELEASE_PUBLISHED_IMMUTABLE",
        "publication terminal state differs",
    )
    repository = _exact(retained["repository"], {"slug", "id", "node_id", "private", "visibility", "archived", "disabled", "administration_read_authorized"}, "terminal repository")
    _require(repository["slug"] == TARGET_REPOSITORY and repository["private"] is True and repository["visibility"] == "private" and repository["archived"] is False and repository["disabled"] is False and repository["administration_read_authorized"] is True, "terminal repository policy differs")
    _positive_int(repository["id"], "terminal repository ID")
    _require(isinstance(repository["node_id"], str) and repository["node_id"], "terminal repository node ID differs")
    release = _exact(retained["release"], {"id", "node_id", "tag", "title", "body", "draft", "prerelease", "immutable"}, "terminal release")
    _positive_int(release["id"], "terminal release ID")
    _require(
        isinstance(release["node_id"], str) and release["node_id"]
        and release["tag"] == DEFAULT_TAG
        and release["title"] == release_title()
        and release["body"] == (
            f"Authoritative private source commit: {retained['source_commit']}\n"
            f"Parentless distribution snapshot: {retained['snapshot_commit']}\n"
            f"Mirror publication ID: {retained['publication_id']}\n"
            f"Distribution asset manifest ID: {retained['manifest_id']}"
        )
        and release["draft"] is False and release["prerelease"] is False and release["immutable"] is True,
        "terminal release policy differs",
    )
    tag = _exact(retained["tag"], {"ref", "object_id"}, "terminal tag")
    _require(tag["ref"] == f"refs/tags/{DEFAULT_TAG}", "terminal tag ref differs")
    _hex40(tag["object_id"], "terminal tag object")
    _hex64(retained["publication_id"], "terminal publication identity")
    _hex64(retained["manifest_id"], "terminal manifest identity")
    _hex40(retained["source_commit"], "terminal source commit")
    _hex40(retained["snapshot_commit"], "terminal snapshot commit")
    _hex64(retained["tag_attestation_id"], "terminal tag attestation identity")
    _hex64(retained["trusted_external_audit_sha256"], "terminal trusted independent audit digest")
    run = _exact(
        retained["run_intent"],
        {"intent_id", "mode", "remote_url", "gh_sha256", "gh_capabilities_sha256"},
        "terminal run intent",
    )
    _hex64(run["intent_id"], "terminal run intent identity")
    _require(
        run["mode"] == "github" and run["remote_url"] == f"git@github.com:{TARGET_REPOSITORY}.git",
        "terminal run intent transport differs",
    )
    _hex64(run["gh_sha256"], "terminal run intent GitHub CLI digest")
    _hex64(run["gh_capabilities_sha256"], "terminal run intent GitHub capability identity")
    cli = _exact(retained["github_cli"], {"version", "sha256", "capabilities_sha256"}, "terminal GitHub CLI")
    _require(
        cli["version"] == AUDITED_GITHUB_CLI_VERSION
        and _hex64(cli["sha256"], "terminal GitHub CLI digest") == APPROVED_GITHUB_CLI_SHA256,
        "terminal GitHub CLI is not the audited approved binary",
    )
    _hex64(cli["capabilities_sha256"], "terminal GitHub CLI capability identity")
    _require(
        cli == {
            "version": AUDITED_GITHUB_CLI_VERSION,
            "sha256": run["gh_sha256"],
            "capabilities_sha256": run["gh_capabilities_sha256"],
        },
        "terminal GitHub CLI differs from the authorized run intent",
    )
    release_attestation = _exact(
        retained["github_release_attestation"],
        {
            "verification_output_sha256", "bundle_sha256", "statement_sha256",
            "certificate_sha256", "certificate_subject_alternative_name",
            "timestamp_sha256", "signature_sha256", "predicate_type", "predicate",
            "tag_subject", "asset_subjects",
        },
        "terminal GitHub release attestation",
    )
    for field in (
        "verification_output_sha256", "bundle_sha256", "statement_sha256",
        "certificate_sha256", "signature_sha256",
    ):
        _hex64(release_attestation[field], f"terminal release attestation {field}")
    _require(
        release_attestation["certificate_subject_alternative_name"]
        == "https://dotcom.releases.github.com"
        and release_attestation["predicate_type"]
        == "https://in-toto.io/attestation/release/v0.2",
        "terminal GitHub release attestation identity differs",
    )
    timestamp_digests = release_attestation["timestamp_sha256"]
    _require(
        isinstance(timestamp_digests, list)
        and timestamp_digests
        and timestamp_digests == sorted(timestamp_digests),
        "terminal signed timestamp inventory differs",
    )
    for digest in timestamp_digests:
        _hex64(digest, "terminal signed timestamp digest")
    predicate = _exact(
        release_attestation["predicate"],
        {"databaseId", "ownerId", "packageId", "purl", "repository", "repositoryId", "tag"},
        "terminal signed release predicate",
    )
    _require(
        predicate["databaseId"] == str(release["id"])
        and predicate["packageId"] == predicate["repositoryId"] == str(repository["id"])
        and predicate["repository"] == TARGET_REPOSITORY
        and predicate["purl"] == f"pkg:github/{TARGET_REPOSITORY}@{DEFAULT_TAG}"
        and predicate["tag"] == DEFAULT_TAG
        and isinstance(predicate["ownerId"], str)
        and predicate["ownerId"].isdigit()
        and int(predicate["ownerId"]) > 0,
        "terminal signed release predicate differs",
    )
    tag_subject = _exact(
        release_attestation["tag_subject"], {"uri", "object_id"},
        "terminal signed tag subject",
    )
    _require(
        tag_subject == {
            "uri": f"pkg:github/{TARGET_REPOSITORY}@{DEFAULT_TAG}",
            "object_id": tag["object_id"],
        },
        "terminal signed tag object differs",
    )
    assets = retained["assets"]
    _require(isinstance(assets, list) and assets == sorted(assets, key=lambda item: item.get("name", "")), "terminal assets are not canonical")
    names: set[str] = set()
    for asset in assets:
        record = _exact(asset, {"id", "node_id", "name", "size_bytes", "sha256"}, "terminal asset")
        _positive_int(record["id"], "terminal asset ID")
        _require(isinstance(record["node_id"], str) and record["node_id"] and isinstance(record["name"], str) and SAFE_NAME.fullmatch(record["name"]) and record["name"] not in names, "terminal asset identity differs")
        _nonnegative_int(record["size_bytes"], "terminal asset size")
        _hex64(record["sha256"], "terminal asset digest")
        names.add(record["name"])
    signed_assets = release_attestation["asset_subjects"]
    _require(
        isinstance(signed_assets, list)
        and signed_assets == sorted(signed_assets, key=lambda item: item.get("name", "")),
        "terminal signed asset subjects are not canonical",
    )
    signed_by_name: dict[str, str] = {}
    for subject in signed_assets:
        record = _exact(subject, {"name", "sha256"}, "terminal signed asset subject")
        _require(
            isinstance(record["name"], str)
            and record["name"] not in signed_by_name,
            "terminal signed asset subject identity differs",
        )
        signed_by_name[record["name"]] = _hex64(
            record["sha256"], "terminal signed asset subject digest"
        )
    _require(
        signed_by_name == {record["name"]: record["sha256"] for record in assets},
        "terminal signed asset union differs from published release assets",
    )
    return dict(retained)


def _parse_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {"name": args.identity_name, "email": args.identity_email, "epoch": args.identity_epoch, "timezone": args.identity_timezone}


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--identity-name", required=True)
    parser.add_argument("--identity-email", required=True)
    parser.add_argument("--identity-epoch", required=True, type=int)
    parser.add_argument("--identity-timezone", default="+0000")


def _add_source_replay(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-repo", required=True, type=pathlib.Path)
    parser.add_argument("--final-verifier", required=True, type=pathlib.Path)
    parser.add_argument("--final-verifier-sha256", required=True)
    parser.add_argument("--final-verifier-arg", action="append", default=[])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)

    prepare = operations.add_parser("prepare-snapshot")
    prepare.add_argument("--source-repo", required=True, type=pathlib.Path)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--source-tree", required=True)
    prepare.add_argument("--source-tag", default=DEFAULT_TAG)
    prepare.add_argument("--source-tag-object")
    prepare.add_argument("--state-root", required=True, type=pathlib.Path)
    prepare.add_argument("--output", required=True, type=pathlib.Path)
    prepare.add_argument("--forbid-object", action="append", default=[])
    _add_identity(prepare)

    create = operations.add_parser("create")
    _add_source_replay(create)
    create.add_argument("--source-authority", required=True, type=pathlib.Path)
    create.add_argument("--snapshot", required=True, type=pathlib.Path)
    create.add_argument("--target-preflight", required=True, type=pathlib.Path)
    create.add_argument("--snapshot-repo", required=True, type=pathlib.Path)
    create.add_argument("--asset-directory", required=True, type=pathlib.Path)
    create.add_argument("--external-audit-sha256", required=True)
    create.add_argument("--output", required=True, type=pathlib.Path)

    verify = operations.add_parser("verify")
    _add_source_replay(verify)
    verify.add_argument("--receipt", required=True, type=pathlib.Path)
    verify.add_argument("--source-authority", required=True, type=pathlib.Path)
    verify.add_argument("--snapshot", required=True, type=pathlib.Path)
    verify.add_argument("--target-preflight", required=True, type=pathlib.Path)
    verify.add_argument("--snapshot-repo", required=True, type=pathlib.Path)
    verify.add_argument("--asset-directory", required=True, type=pathlib.Path)
    verify.add_argument("--external-audit-sha256", required=True)

    assets = operations.add_parser("create-assets")
    assets.add_argument("--source-manifest", required=True, type=pathlib.Path)
    assets.add_argument("--asset-directory", required=True, type=pathlib.Path)
    assets.add_argument("--mirror-receipt", required=True, type=pathlib.Path)
    assets.add_argument("--target-history-attestation", required=True, type=pathlib.Path)
    assets.add_argument("--output", required=True, type=pathlib.Path)

    verify_assets = operations.add_parser("verify-assets")
    verify_assets.add_argument("--manifest", required=True, type=pathlib.Path)
    verify_assets.add_argument("--asset-directory", required=True, type=pathlib.Path)
    verify_assets.add_argument("--receipt", required=True, type=pathlib.Path)

    seal = operations.add_parser("seal-tag")
    seal.add_argument("--snapshot-repo", required=True, type=pathlib.Path)
    seal.add_argument("--state-root", required=True, type=pathlib.Path)
    seal.add_argument("--receipt", required=True, type=pathlib.Path)
    seal.add_argument("--distribution-manifest", required=True, type=pathlib.Path)
    seal.add_argument("--output", required=True, type=pathlib.Path)
    seal.add_argument("--forbid-object", action="append", default=[])

    isolated = operations.add_parser("verify-isolated")
    isolated.add_argument("--repo", required=True, type=pathlib.Path)
    isolated.add_argument("--receipt", required=True, type=pathlib.Path)
    isolated.add_argument("--distribution-manifest", required=True, type=pathlib.Path)
    isolated.add_argument("--tag-attestation", required=True, type=pathlib.Path)
    isolated.add_argument("--forbid-object", action="append", default=[])

    prior = operations.add_parser("audit-prior")
    prior.add_argument("--repo", required=True, type=pathlib.Path)
    prior.add_argument("--target-preflight", required=True, type=pathlib.Path)
    prior.add_argument("--target-history-attestation", required=True, type=pathlib.Path)
    prior.add_argument("--prior-assets-root", required=True, type=pathlib.Path)
    prior.add_argument("--external-audit", required=True, type=pathlib.Path)
    prior.add_argument("--external-audit-sha256", required=True)
    prior.add_argument("--forbid-object", action="append", default=[])

    migration_audit = operations.add_parser("verify-v103-migration-audit")
    migration_audit.add_argument("--audit", required=True, type=pathlib.Path)
    migration_audit.add_argument("--audit-sha256", required=True)
    migration_audit.add_argument("--legacy-repo", required=True, type=pathlib.Path)
    migration_audit.add_argument("--legacy-assets", required=True, type=pathlib.Path)

    for operation_name in ("create-v103-migration-run-intent", "verify-v103-migration-run-intent"):
        migration_intent = operations.add_parser(operation_name)
        migration_intent.add_argument("--audit", required=True, type=pathlib.Path)
        migration_intent.add_argument("--audit-sha256", required=True)
        migration_intent.add_argument("--legacy-repo", required=True, type=pathlib.Path)
        migration_intent.add_argument("--legacy-assets", required=True, type=pathlib.Path)
        migration_intent.add_argument("--target-repo", required=True, type=pathlib.Path)
        migration_intent.add_argument("--gh-binary", required=True, type=pathlib.Path)
        migration_intent.add_argument("--intent", type=pathlib.Path)
        if operation_name.startswith("create-"):
            migration_intent.add_argument("--output", required=True, type=pathlib.Path)

    migration_receipt = operations.add_parser("create-v103-migration-receipt")
    migration_receipt.add_argument("--audit", required=True, type=pathlib.Path)
    migration_receipt.add_argument("--release-id", required=True, type=int)
    migration_receipt.add_argument("--release-node-id", required=True)
    migration_receipt.add_argument("--output", required=True, type=pathlib.Path)

    migration_manifest = operations.add_parser("create-v103-migration-manifest")
    migration_manifest.add_argument("--audit", required=True, type=pathlib.Path)
    migration_manifest.add_argument("--receipt", required=True, type=pathlib.Path)
    migration_manifest.add_argument("--legacy-assets", required=True, type=pathlib.Path)
    migration_manifest.add_argument("--output", required=True, type=pathlib.Path)

    migration_terminal = operations.add_parser("create-v103-migration-terminal")
    migration_terminal.add_argument("--intent", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--audit", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--audit-sha256", required=True)
    migration_terminal.add_argument("--receipt", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--manifest", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--target-repo", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--evidence-directory", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--gh-binary", required=True, type=pathlib.Path)
    migration_terminal.add_argument("--incident", type=pathlib.Path)
    migration_terminal.add_argument("--output", required=True, type=pathlib.Path)

    prior_live_metadata = operations.add_parser("capture-v103-release-metadata")
    prior_live_metadata.add_argument("--receipt", required=True, type=pathlib.Path)
    prior_live_metadata.add_argument("--manifest", required=True, type=pathlib.Path)
    prior_live_metadata.add_argument("--repository-metadata", required=True, type=pathlib.Path)
    prior_live_metadata.add_argument("--organization-metadata", required=True, type=pathlib.Path)
    prior_live_metadata.add_argument("--release-metadata", required=True, type=pathlib.Path)
    prior_live_metadata.add_argument("--output", required=True, type=pathlib.Path)

    migration_incident = operations.add_parser("create-v103-migration-incident")
    migration_incident.add_argument("--stage", required=True)
    migration_incident.add_argument("--exit-status", required=True, type=int)
    migration_incident.add_argument("--observed-at-utc", required=True)
    migration_incident.add_argument("--audit-sha256", required=True)
    migration_incident.add_argument("--intent-id", required=True)
    migration_incident.add_argument("--release-id", required=True, type=int)
    migration_incident.add_argument("--release-node-id", required=True)
    migration_incident.add_argument("--output", required=True, type=pathlib.Path)

    source_authority = operations.add_parser("create-source-authority")
    _add_source_replay(source_authority)
    source_authority.add_argument("--asset-directory", required=True, type=pathlib.Path)
    source_authority.add_argument("--final-receipt", required=True)
    source_authority.add_argument("--candidate-receipt", required=True)
    source_authority.add_argument("--package", required=True)
    source_authority.add_argument("--source-manifest", required=True)
    source_authority.add_argument("--private-repository-metadata", required=True, type=pathlib.Path)
    source_authority.add_argument("--private-release-metadata", required=True, type=pathlib.Path)
    source_authority.add_argument("--trusted-source-tag-object", required=True)
    source_authority.add_argument("--output", required=True, type=pathlib.Path)

    verify_source = operations.add_parser("verify-source-authority")
    _add_source_replay(verify_source)
    verify_source.add_argument("--source-authority", required=True, type=pathlib.Path)
    verify_source.add_argument("--asset-directory", required=True, type=pathlib.Path)
    verify_source.add_argument("--trusted-source-tag-object", required=True)

    history = operations.add_parser("create-target-history")
    history.add_argument("--repo", required=True, type=pathlib.Path)
    history.add_argument("--repository-metadata", required=True, type=pathlib.Path)
    history.add_argument("--mode", required=True, choices=("empty-new-repository", "audited-prior-releases"))
    history.add_argument("--advertised-refs", required=True, type=pathlib.Path)
    history.add_argument("--prior-assets-root", type=pathlib.Path)
    history.add_argument("--external-audit", required=True, type=pathlib.Path)
    history.add_argument("--external-audit-sha256", required=True)
    history.add_argument("--output", required=True, type=pathlib.Path)

    target_preflight = operations.add_parser("create-target-preflight")
    target_preflight.add_argument("--repository-metadata", required=True, type=pathlib.Path)
    target_preflight.add_argument("--actions-permissions", required=True, type=pathlib.Path)
    target_preflight.add_argument("--immutable-releases", required=True, type=pathlib.Path)
    target_preflight.add_argument("--organization-metadata", required=True, type=pathlib.Path)
    target_preflight.add_argument("--dependabot-security-updates-status", required=True, choices=("disabled", "enabled", "unverifiable"))
    target_preflight.add_argument("--advertised-refs", required=True, type=pathlib.Path)
    target_preflight.add_argument("--history-attestation", required=True, type=pathlib.Path)
    target_preflight.add_argument("--external-audit", required=True, type=pathlib.Path)
    target_preflight.add_argument("--external-audit-sha256", required=True)
    target_preflight.add_argument("--output", required=True, type=pathlib.Path)

    for operation_name in ("create-run-intent", "verify-run-intent"):
        intent = operations.add_parser(operation_name)
        intent.add_argument("--source-repo", required=True, type=pathlib.Path)
        intent.add_argument("--source-authority", required=True, type=pathlib.Path)
        intent.add_argument("--target-preflight", required=True, type=pathlib.Path)
        intent.add_argument("--target-history-attestation", required=True, type=pathlib.Path)
        intent.add_argument("--external-audit", required=True, type=pathlib.Path)
        intent.add_argument("--external-audit-sha256", required=True)
        intent.add_argument("--asset-directory", required=True, type=pathlib.Path)
        intent.add_argument("--state-root", required=True, type=pathlib.Path)
        intent.add_argument("--repository", required=True)
        intent.add_argument("--forbid-object", action="append", default=[])
        transport = intent.add_mutually_exclusive_group(required=True)
        transport.add_argument("--local-bare-remote", type=pathlib.Path)
        transport.add_argument("--gh-binary", type=pathlib.Path)
        intent.add_argument("--gh-sha256")
        intent.add_argument("--gh-capabilities-sha256")
        if operation_name == "create-run-intent":
            intent.add_argument("--output", required=True, type=pathlib.Path)
        else:
            intent.add_argument("--intent", required=True, type=pathlib.Path)
        _add_identity(intent)

    published = operations.add_parser("verify-published")
    published.add_argument("--repo", required=True, type=pathlib.Path)
    published.add_argument("--receipt", required=True, type=pathlib.Path)
    published.add_argument("--distribution-manifest", required=True, type=pathlib.Path)
    published.add_argument("--asset-directory", required=True, type=pathlib.Path)
    published.add_argument("--target-history-attestation", required=True, type=pathlib.Path)
    published.add_argument("--prior-assets-root", type=pathlib.Path)
    published.add_argument("--external-audit-sha256", required=True)
    published.add_argument("--forbid-object", action="append", default=[])

    empty_target = operations.add_parser("verify-empty-target")
    empty_target.add_argument("--repo", required=True, type=pathlib.Path)
    for operation_name in ("create-publication-terminal", "verify-publication-terminal"):
        terminal = operations.add_parser(operation_name)
        terminal.add_argument("--receipt", required=True, type=pathlib.Path)
        terminal.add_argument("--distribution-manifest", required=True, type=pathlib.Path)
        terminal.add_argument("--tag-attestation", required=True, type=pathlib.Path)
        terminal.add_argument("--run-intent", required=True, type=pathlib.Path)
        terminal.add_argument("--asset-directory", required=True, type=pathlib.Path)
        terminal.add_argument("--tag-object", required=True)
        terminal.add_argument("--gh-binary", required=True, type=pathlib.Path)
        terminal.add_argument("--gh-capabilities-sha256", required=True)
        terminal.add_argument("--external-audit-sha256", required=True)
        if operation_name == "create-publication-terminal":
            terminal.add_argument("--output", required=True, type=pathlib.Path)
        else:
            terminal.add_argument("--terminal", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "prepare-snapshot":
            value = prepare_snapshot(
                source_repo=args.source_repo,
                source_commit=args.source_commit,
                source_tree=args.source_tree,
                state_root=args.state_root,
                identity=_parse_identity(args),
                source_tag=args.source_tag,
                source_tag_object=args.source_tag_object,
                forbidden_objects=args.forbid_object,
            )
            _write_new_json(args.output, value, "snapshot evidence")
            result_id = value["snapshot_id"]
        elif args.operation == "create":
            source = _load_json(args.source_authority, "source authority")
            verify_source_authority_files(
                source,
                args.asset_directory,
                source_repo=args.source_repo,
                final_verifier_path=args.final_verifier,
                final_verifier_sha256=args.final_verifier_sha256,
                final_verifier_arguments=args.final_verifier_arg,
                trusted_source_tag_object=source["source_tag"]["object_id"],
            )
            snapshot = _load_json(args.snapshot, "snapshot evidence")
            target = _load_json(args.target_preflight, "target preflight")
            replay_snapshot(snapshot, args.snapshot_repo)
            verify_external_mirror_audit(target, asset_directory=args.asset_directory, source_authority=source, snapshot=snapshot, trusted_sha256=args.external_audit_sha256)
            value = create_receipt(source, snapshot, target)
            _write_new_json(args.output, value, "mirror receipt")
            result_id = value["publication_id"]
        elif args.operation == "verify":
            source = _load_json(args.source_authority, "source authority")
            verify_source_authority_files(
                source,
                args.asset_directory,
                source_repo=args.source_repo,
                final_verifier_path=args.final_verifier,
                final_verifier_sha256=args.final_verifier_sha256,
                final_verifier_arguments=args.final_verifier_arg,
                trusted_source_tag_object=source["source_tag"]["object_id"],
            )
            snapshot = _load_json(args.snapshot, "snapshot evidence")
            target = _load_json(args.target_preflight, "target preflight")
            verify_external_mirror_audit(target, asset_directory=args.asset_directory, source_authority=source, snapshot=snapshot, trusted_sha256=args.external_audit_sha256)
            value = replay_receipt(
                _load_json(args.receipt, "mirror receipt"),
                source_authority=source,
                snapshot=snapshot,
                target_preflight=target,
                snapshot_repo=args.snapshot_repo,
            )
            result_id = value["publication_id"]
        elif args.operation == "create-assets":
            value = create_distribution_manifest(
                source_manifest_path=args.source_manifest,
                asset_directory=args.asset_directory,
                mirror_receipt_path=args.mirror_receipt,
                target_history_attestation_path=args.target_history_attestation,
            )
            _write_new_json(args.output, value, "distribution asset manifest")
            result_id = value["manifest_id"]
        elif args.operation == "verify-assets":
            value = cross_validate_distribution(
                _load_json(args.manifest, "distribution asset manifest"),
                _load_json(args.receipt, "mirror receipt"),
                args.asset_directory,
            )
            result_id = value["manifest_id"]
        elif args.operation == "seal-tag":
            value = seal_tag(
                snapshot_repo=args.snapshot_repo,
                output_root=args.state_root,
                receipt=_load_json(args.receipt, "mirror receipt"),
                distribution_manifest=_load_json(args.distribution_manifest, "distribution asset manifest"),
                forbidden_objects=args.forbid_object,
            )
            _write_new_json(args.output, value, "tag attestation")
            result_id = value["attestation_id"]
        elif args.operation == "verify-isolated":
            receipt = validate_receipt(_load_json(args.receipt, "mirror receipt"))
            manifest = validate_distribution_manifest(_load_json(args.distribution_manifest, "distribution asset manifest"))
            attestation = validate_tag_attestation(_load_json(args.tag_attestation, "tag attestation"))
            replay_tag_attestation(
                receipt=receipt,
                distribution_manifest=manifest,
                attestation=attestation,
                publisher_repo=args.repo,
                forbidden_objects=args.forbid_object,
            )
            result_id = attestation["attestation_id"]
        elif args.operation == "audit-prior":
            target = validate_target_preflight(_load_json(args.target_preflight, "target preflight"))
            _require(target["bootstrap"]["mode"] == "audited-prior-releases", "prior audit requires audited-prior-releases mode")
            history_path = _regular_file(args.target_history_attestation, "target history attestation")
            history = validate_target_history_attestation(_load_json(history_path, "target history attestation"))
            _require(
                target["bootstrap"]["attestation_sha256"] == _sha256_file(history_path)
                and target["bootstrap"]["attestation_id"] == history["attestation_id"],
                "target history attestation differs from preflight",
            )
            value = audit_prior_release_repository(
                args.repo,
                history_attestation=history,
                prior_assets_root=args.prior_assets_root,
                external_audit_path=args.external_audit,
                trusted_external_audit_sha256=args.external_audit_sha256,
                forbidden_objects=args.forbid_object,
            )
            result_id = value["audit_id"]
        elif args.operation == "verify-v103-migration-audit":
            value = verify_v103_migration_audit_files(
                _load_json(args.audit, "v1.0.3 independent migration audit"),
                audit_path=args.audit, trusted_sha256=args.audit_sha256,
                legacy_repo=args.legacy_repo, legacy_assets=args.legacy_assets,
            )
            result_id = value["audit_id"]
        elif args.operation in {"create-v103-migration-run-intent", "verify-v103-migration-run-intent"}:
            value = create_v103_migration_run_intent(
                audit_path=args.audit, trusted_audit_sha256=args.audit_sha256,
                legacy_repo=args.legacy_repo, legacy_assets=args.legacy_assets,
                target_repo=args.target_repo, gh_binary=args.gh_binary,
                require_empty_target=args.operation == "create-v103-migration-run-intent",
            )
            if args.operation == "create-v103-migration-run-intent":
                _write_new_json(args.output, value, "v1.0.3 migration run intent")
            else:
                _require(args.intent is not None, "v1.0.3 migration intent is required for replay")
                _require(
                    _load_json(args.intent, "v1.0.3 migration run intent") == value,
                    "v1.0.3 migration run intent replay differs",
                )
            result_id = value["intent_id"]
        elif args.operation == "create-v103-migration-receipt":
            value = create_v103_migration_receipt(
                _load_json(args.audit, "v1.0.3 migration receipt audit"),
                release_id=args.release_id, release_node_id=args.release_node_id,
            )
            _write_new_json(args.output, value, "v1.0.3 migration receipt")
            result_id = value["publication_id"]
        elif args.operation == "create-v103-migration-manifest":
            value = create_v103_migration_manifest(
                _load_json(args.audit, "v1.0.3 migration manifest audit"),
                receipt_path=args.receipt, legacy_assets=args.legacy_assets,
            )
            _write_new_json(args.output, value, "v1.0.3 migration manifest")
            result_id = value["manifest_id"]
        elif args.operation == "create-v103-migration-terminal":
            value = create_v103_migration_terminal(
                run_intent=_load_json(args.intent, "v1.0.3 migration run intent"),
                audit_path=args.audit, trusted_audit_sha256=args.audit_sha256,
                receipt=_load_json(args.receipt, "v1.0.3 migration receipt"),
                manifest=_load_json(args.manifest, "v1.0.3 migration manifest"),
                manifest_path=args.manifest, target_repo=args.target_repo,
                evidence_directory=args.evidence_directory, gh_binary=args.gh_binary,
                incident_path=args.incident,
            )
            _write_new_json(args.output, value, "v1.0.3 migration terminal")
            result_id = value["terminal_id"]
        elif args.operation == "capture-v103-release-metadata":
            value = create_v103_release_metadata(
                receipt=_load_json(args.receipt, "v1.0.3 live capture receipt"),
                manifest=_load_json(args.manifest, "v1.0.3 live capture manifest"),
                repository_metadata=_load_json(args.repository_metadata, "v1.0.3 live repository metadata"),
                organization_metadata=_load_json(args.organization_metadata, "v1.0.3 live organization metadata"),
                release_metadata=_load_json(args.release_metadata, "v1.0.3 live release API metadata"),
                manifest_path=args.manifest,
            )
            _write_new_json(args.output, value, "v1.0.3 live release evidence")
            result_id = value["capture_id"]
        elif args.operation == "create-v103-migration-incident":
            value = create_v103_migration_incident(
                stage=args.stage, exit_status=args.exit_status,
                observed_at_utc=args.observed_at_utc, audit_sha256=args.audit_sha256,
                intent_id=args.intent_id, release_id=args.release_id,
                release_node_id=args.release_node_id,
            )
            _write_new_json(args.output, value, "v1.0.3 migration incident")
            result_id = value["incident_id"]
        elif args.operation == "create-source-authority":
            value = create_source_authority(
                source_repo=args.source_repo,
                asset_directory=args.asset_directory,
                final_receipt_name=args.final_receipt,
                candidate_receipt_name=args.candidate_receipt,
                package_name=args.package,
                source_manifest_name=args.source_manifest,
                private_repository_metadata_path=args.private_repository_metadata,
                private_release_metadata_path=args.private_release_metadata,
                final_verifier_path=args.final_verifier,
                final_verifier_sha256=args.final_verifier_sha256,
                final_verifier_arguments=args.final_verifier_arg,
                trusted_source_tag_object=args.trusted_source_tag_object,
            )
            _write_new_json(args.output, value, "source authority")
            result_id = value["final_receipt"]["receipt_id"]
        elif args.operation == "verify-source-authority":
            value = verify_source_authority_files(
                _load_json(args.source_authority, "source authority"),
                args.asset_directory,
                source_repo=args.source_repo,
                final_verifier_path=args.final_verifier,
                final_verifier_sha256=args.final_verifier_sha256,
                final_verifier_arguments=args.final_verifier_arg,
                trusted_source_tag_object=args.trusted_source_tag_object,
            )
            result_id = value["verification"]["terminal"]["receipt_id"]
        elif args.operation == "create-target-history":
            value = create_target_history_attestation(
                repo=args.repo,
                repository_metadata_path=args.repository_metadata,
                mode=args.mode,
                advertised_refs_path=args.advertised_refs,
                prior_assets_root=args.prior_assets_root,
                external_audit_path=args.external_audit,
                trusted_external_audit_sha256=args.external_audit_sha256,
            )
            _write_new_json(args.output, value, "target history attestation")
            result_id = value["attestation_id"]
        elif args.operation == "create-target-preflight":
            value = create_target_preflight(
                repository_metadata_path=args.repository_metadata,
                actions_permissions_path=args.actions_permissions,
                immutable_releases_path=args.immutable_releases,
                organization_metadata_path=args.organization_metadata,
                dependabot_security_updates_status=args.dependabot_security_updates_status,
                advertised_refs_path=args.advertised_refs,
                history_attestation_path=args.history_attestation,
                external_audit_path=args.external_audit,
                trusted_external_audit_sha256=args.external_audit_sha256,
            )
            _write_new_json(args.output, value, "target preflight")
            result_id = value["bootstrap"]["attestation_id"]
        elif args.operation in {"create-run-intent", "verify-run-intent"}:
            keywords = {
                "source_repo": args.source_repo,
                "source_authority_path": args.source_authority,
                "target_preflight_path": args.target_preflight,
                "target_history_attestation_path": args.target_history_attestation,
                "external_audit_path": args.external_audit,
                "asset_directory": args.asset_directory,
                "state_root": args.state_root,
                "repository": args.repository,
                "identity": _parse_identity(args),
                "forbidden_objects": args.forbid_object,
                "local_bare_remote": args.local_bare_remote,
                "gh_binary": args.gh_binary,
                "gh_sha256": args.gh_sha256,
                "gh_capabilities_sha256": args.gh_capabilities_sha256,
                "trusted_external_audit_sha256": args.external_audit_sha256,
            }
            value = create_run_intent(**keywords) if args.operation == "create-run-intent" else replay_run_intent(_load_json(args.intent, "run intent"), **keywords)
            if args.operation == "create-run-intent":
                _write_new_json(args.output, value, "run intent")
            result_id = value["intent_id"]
        elif args.operation == "verify-published":
            value = verify_published_repository(
                repo=args.repo,
                receipt=_load_json(args.receipt, "mirror receipt"),
                distribution_manifest=_load_json(args.distribution_manifest, "distribution asset manifest"),
                asset_directory=args.asset_directory,
                target_history_attestation=_load_json(args.target_history_attestation, "target history attestation"),
                prior_assets_root=args.prior_assets_root,
                trusted_external_audit_sha256=args.external_audit_sha256,
                forbidden_objects=args.forbid_object,
            )
            result_id = value["verification_id"]
        elif args.operation == "verify-empty-target":
            value = verify_empty_target_repository(args.repo)
            result_id = _canonical_sha256(value)
        elif args.operation in {"create-publication-terminal", "verify-publication-terminal"}:
            value = create_publication_terminal(
                receipt=_load_json(args.receipt, "terminal mirror receipt"),
                distribution_manifest=_load_json(args.distribution_manifest, "terminal distribution manifest"),
                tag_attestation=_load_json(args.tag_attestation, "terminal tag attestation"),
                run_intent=_load_json(args.run_intent, "terminal run intent"),
                asset_directory=args.asset_directory,
                tag_object=args.tag_object,
                gh_binary=args.gh_binary,
                gh_capabilities_sha256=args.gh_capabilities_sha256,
                trusted_external_audit_sha256=args.external_audit_sha256,
            )
            if args.operation == "create-publication-terminal":
                _write_new_json(args.output, value, "publication terminal")
            else:
                retained_terminal = validate_publication_terminal(
                    _load_json(args.terminal, "publication terminal")
                )
                _require(retained_terminal == value, "publication terminal live replay differs")
            result_id = value["terminal_id"]
        else:
            raise ReleaseOnlyError("unsupported operation")
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "id": result_id}, sort_keys=True))
        return 0
    except (OSError, ReleaseOnlyError, subprocess.SubprocessError, ValueError) as error:
        print(f"v1.1.0 release-only mirror: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
