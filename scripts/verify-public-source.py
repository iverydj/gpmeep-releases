#!/usr/bin/env python3
"""Verify a privacy-edited public tree against its original build commit.

This is a content check, not publication authority. Package qualification,
public-history inspection, and final release approval remain separate gates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
OMITTED_DIRECTORIES = (".github/", "artifacts/", "evidence-anchors/")
OMITTED_FILES = {"RESUME_CHECKPOINT.md", "monkeytype.sqlite3"}
EDITABLE_TEXT = {"AUTHORS", "CITATION.cff", ".gitignore"}
ADDED_METADATA = {"PUBLIC_RELEASE.json", "PUBLIC_RELEASE_POLICY.md", "CITATION.cff"}
MAX_BLOB = 128 * 1024 * 1024
MAX_TREE_BYTES = 512 * 1024 * 1024
SECRET_PATTERN = re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}")
PRIVATE_PATH_PATTERN = re.compile(rb"/(?:home|scratch)/[A-Za-z0-9_.-]+(?:/|\b)")
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
LEGACY_SYNTHETIC_UUIDS = {"GPU-11111111-2222-3333-4444-555555555555",
                        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}


class PublicSourceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicSourceError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(data: bytes) -> Any:
    def bad_constant(value):
        raise PublicSourceError("non-finite JSON value")
    try:
        return json.loads(data, object_pairs_hook=unique_object, parse_constant=bad_constant)
    except (ValueError, UnicodeError) as exc:
        raise PublicSourceError(f"invalid JSON: {exc}") from exc


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    # Do not allow replace refs, hooks, or caller-supplied Git environment to
    # change what the named object means. No commands execute checked-out code.
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_NO_REPLACE_OBJECTS": "1", "GIT_OPTIONAL_LOCKS": "0"}
    result = subprocess.run(["/usr/bin/git", "-C", str(repo), "-c",
                             "core.hooksPath=/dev/null", *args],
                            env=env, capture_output=True, timeout=60)
    require(result.returncode == 0, f"Git object read failed: {args[0]}")
    return result.stdout


def tree(repo: Path, commit: str) -> dict[str, tuple[str, str]]:
    require(isinstance(commit, str) and HEX40.fullmatch(commit) is not None,
            "use a full immutable commit ID")
    require(git(repo, "cat-file", "-t", commit).strip() == b"commit",
            "object must be a commit")
    entries = {}
    for row in git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        meta, raw_path = row.split(b"\t", 1)
        mode, kind, oid = meta.decode("ascii").split()
        name = raw_path.decode("utf-8")
        p = PurePosixPath(name)
        require(not p.is_absolute() and p.as_posix() == name and
                ".." not in p.parts and not any(ord(c) < 32 for c in name),
                "unsafe Git path")
        require(kind == "blob" and mode in {"100644", "100755", "120000"},
                f"unsupported tree entry: {name}")
        require(name not in entries, "duplicate Git path")
        entries[name] = (mode, oid)
    return entries


def omitted(name: str) -> bool:
    return name in OMITTED_FILES or name.startswith(OMITTED_DIRECTORIES)


def documentation(name: str) -> bool:
    return name.endswith(".md") or name in EDITABLE_TEXT


def notebook_projection(data: bytes) -> Any:
    obj = read_json(data)
    require(isinstance(obj, dict) and isinstance(obj.get("cells"), list),
            "invalid notebook structure")
    for cell in obj["cells"]:
        require(isinstance(cell, dict), "invalid notebook cell")
        require(isinstance(cell.get("source"), (str, list)), "missing notebook source")
        if isinstance(cell["source"], list):
            require(all(isinstance(line, str) for line in cell["source"]),
                    "invalid notebook source lines")
        if cell.get("cell_type") == "code":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    return obj


def validate_replacements(replacements: dict[str, str]) -> None:
    require(isinstance(replacements, dict), "fixture replacements must be an object")
    destinations = []
    for old, new in replacements.items():
        require(isinstance(old, str) and isinstance(new, str) and
                UUID_PATTERN.fullmatch(old) is not None and
                UUID_PATTERN.fullmatch(new) is not None,
                "fixture replacements must be complete GPU UUIDs")
        require(old != new, "fixture replacement must change the UUID")
        # Synthetic UUIDs are visibly labelled by a zero prefix, never another
        # hardware identifier. Keep the full format and a one-to-one mapping.
        require(new.startswith("GPU-00000000-0000-0000-0000-") or new in LEGACY_SYNTHETIC_UUIDS,
                "fixture destination must be a synthetic UUID")
        destinations.append(new)
    require(len(set(destinations)) == len(destinations),
            "fixture mapping must preserve distinct device identities")
    require(not set(replacements).intersection(destinations),
            "fixture replacement chains are forbidden")


def fixture_projection(data: bytes, replacements: dict[str, str]) -> str:
    try:
        # Compile the actual Python byte stream: an encoding cookie can turn
        # apparent comments into executable statements (e.g. unicode_escape).
        syntax = ast.parse(data, type_comments=True)
    except (SyntaxError, UnicodeError) as exc:
        raise PublicSourceError("fixture is not parseable Python") from exc

    # Compare interpreted constants, including \u escapes and implicit literal
    # concatenation, before replacing identifiers. Raw byte searches miss these.
    for node in ast.walk(syntax):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            for new in replacements.values():
                target = new.encode() if isinstance(node.value, bytes) else new
                require(target not in node.value,
                        "synthetic UUID collides with an original fixture identity")

    class Replace(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for old, new in replacements.items():
                    node.value = node.value.replace(old, new)
            elif isinstance(node.value, bytes):
                for old, new in replacements.items():
                    node.value = node.value.replace(old.encode(), new.encode())
            return node

    return ast.dump(Replace().visit(syntax), include_attributes=False)


def verify(source_repo: Path, source_commit: str, public_repo: Path, public_commit: str,
           *, forbidden_literals: list[str], fixture_replacements: dict[str, str]) -> dict:
    validate_replacements(fixture_replacements)
    require(isinstance(forbidden_literals, list) and
            all(isinstance(s, str) and len(s) >= 4 for s in forbidden_literals),
            "forbidden literals must be strings of at least four characters")
    source = tree(source_repo, source_commit)
    public = tree(public_repo, public_commit)
    missing = sorted(set(source) - set(public))
    require(all(omitted(name) for name in missing),
            "public tree omits protected source: " + ", ".join(n for n in missing if not omitted(n)))
    require(not any(omitted(name) for name in public), "public tree contains private evidence paths")
    added = sorted(set(public) - set(source))
    require(all(name in ADDED_METADATA or name.endswith(".md") for name in added),
            "public tree adds an unverified runtime/build file")
    # Metadata can be changed only as text, never by replacing it with a link
    # or executable. Existing unchanged files preserve their original modes.
    records = []
    total = 0
    forbidden = [s.encode() for s in forbidden_literals] + [s.encode() for s in fixture_replacements]
    for name, (mode, oid) in sorted(public.items()):
        size = int(git(public_repo, "cat-file", "-s", oid))
        total += size
        require(size <= MAX_BLOB and total <= MAX_TREE_BYTES, "public tree exceeds read limits")
        content = git(public_repo, "cat-file", "blob", oid)
        scan = name.encode() + b"\0" + content
        if name.endswith((".json", ".ipynb")):
            # Literal byte scans alone miss JSON Unicode escapes in notebook
            # outputs or publication metadata. Embedded media still needs review.
            scan += b"\0" + json.dumps(read_json(content), ensure_ascii=False).encode()
        require(not SECRET_PATTERN.search(scan), f"credential pattern in public file: {name}")
        require(not any(token in scan for token in forbidden),
                f"private literal remains in public file: {name}")
        # Generic sample paths can be legitimate fixture inputs. They still
        # require review, recorded below; explicit host literals are forbidden.
        path_matches = len(PRIVATE_PATH_PATTERN.findall(content))
        original = source.get(name)
        kind = "added_metadata"
        source_sha256 = None
        if original:
            require(mode == original[0], f"file mode changed: {name}")
            if oid == original[1]:
                kind = "identical"
                source_sha256 = digest(content)
            else:
                require(mode == "100644" or
                        (mode == "100755" and name.startswith(("scripts/tests/", "python/tests/"))),
                        f"changed executable or symlink: {name}")
                source_size = int(git(source_repo, "cat-file", "-s", original[1]))
                require(source_size <= MAX_BLOB, "source blob exceeds read limit")
                old = git(source_repo, "cat-file", "blob", original[1])
                source_sha256 = digest(old)
                if documentation(name):
                    old.decode("utf-8")
                    content.decode("utf-8")
                    kind = "reviewed_documentation_required"
                elif name.endswith(".ipynb"):
                    require(notebook_projection(old) == notebook_projection(content),
                            f"notebook source or metadata changed: {name}")
                    kind = "notebook_outputs_only"
                elif name.startswith(("scripts/tests/", "python/tests/")) and name.endswith(".py"):
                    require(not any(new.encode() in old for new in fixture_replacements.values()),
                            f"synthetic UUID collides with an original fixture identity: {name}")
                    require(fixture_projection(old, fixture_replacements) == fixture_projection(content, {}),
                            f"fixture logic changed beyond UUID substitution: {name}")
                    kind = "synthetic_fixture_uuids"
                else:
                    raise PublicSourceError(f"runtime/build source changed: {name}")
        else:
            require(mode == "100644", f"new metadata is executable or a symlink: {name}")
            content.decode("utf-8")
        if name == "PUBLIC_RELEASE.json":
            metadata = read_json(content)
            require(isinstance(metadata, dict) and metadata.get("binary_source_commit") == source_commit,
                    "public metadata substitutes the original binary source commit")
        if mode == "120000":
            target = content.decode("utf-8")
            normalized = os.path.normpath(str(PurePosixPath(name).parent / target))
            require(not target.startswith("/") and normalized != ".." and not normalized.startswith("../"),
                    f"public symlink escapes source tree: {name}")
        records.append({"path": name, "mode": mode, "public_sha256": digest(content),
                        "source_sha256": source_sha256, "classification": kind,
                        "generic_private_path_patterns_for_review": path_matches})
    report = {"schema": "gpmeep-public-source-content-v1", "state": "CONTENT_VERIFIED",
              "source_commit": source_commit, "public_commit": public_commit,
              "source_tree": git(source_repo, "rev-parse", source_commit + "^{tree}").decode().strip(),
              "public_tree": git(public_repo, "rev-parse", public_commit + "^{tree}").decode().strip(),
              "omitted_paths": missing, "files": records,
              "publication_authorized": False,
              "verifier_sha256": digest(Path(__file__).read_bytes()),
              "remaining_checks": ["public_history_and_refs", "documentation_and_media_review",
                                   "complete_package_privacy_scan", "final_release_authority"],
              "policy_sha256": digest(json.dumps({"forbidden_literals": forbidden_literals,
                                                   "fixture_replacements": fixture_replacements},
                                                  sort_keys=True).encode())}
    report["report_sha256"] = digest(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--public-repo", required=True, type=Path)
    parser.add_argument("--public-commit", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        policy = read_json(args.policy.read_bytes())
        require(isinstance(policy, dict) and set(policy) == {"forbidden_literals", "fixture_replacements"},
                "privacy policy fields differ")
        report = verify(args.source_repo, args.source_commit, args.public_repo, args.public_commit, **policy)
        # Never replace an earlier report, and never emit a partial success file.
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".public-source-",
                                             delete=False) as stream:
                staged = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, args.output)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        print(f"Public source content verified: {len(report['files'])} files; publication remains gated")
        return 0
    except (PublicSourceError, OSError, UnicodeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"public source verification failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
