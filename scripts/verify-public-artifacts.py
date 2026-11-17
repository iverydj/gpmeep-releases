#!/usr/bin/env python3
"""Inspect public Git history and release files without extracting or changing them.

Literal checks cannot establish that pictures or prose are safe to publish.
This report inventories those bytes for independent review, never authorizes
publication, and does not replace the final package qualification verifier.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


SOURCE = module("gpmeep_public_source", "verify-public-source.py")
PACKAGE = module("gpmeep_public_package", "v110-final-release-receipt.py")
require = SOURCE.require
digest = SOURCE.digest
MAX_FILE = 128 * 1024 * 1024
MAX_HISTORY = 512 * 1024 * 1024
MAX_COMMITS = 1000
SECRET = re.compile(rb"(?:xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")
GPU_UUID = re.compile(rb"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
DATA_URL = re.compile(rb"data:([^,\r\n]*?);base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def policy_read(path):
    value = SOURCE.read_json(path.read_bytes())
    require(isinstance(value, dict) and set(value) == {"forbidden_literals", "fixture_replacements"},
            "privacy policy fields differ")
    require(isinstance(value["forbidden_literals"], list) and value["forbidden_literals"] and
            all(isinstance(x, str) and len(x) >= 4 for x in value["forbidden_literals"]),
            "a nonempty private-literal inventory is required")
    SOURCE.validate_replacements(value["fixture_replacements"])
    return value


def regular(path):
    path = Path(os.path.abspath(path))
    require(path.resolve(strict=True) == path and path.is_file(), "file path contains a symlink")
    require(path.stat().st_size <= MAX_FILE, "file exceeds size limit")
    return path


def scan(name, data, policy):
    """Check raw and common text encodings; record opaque media for review."""
    require(len(data) <= MAX_HISTORY, "content exceeds scan limit")
    forbidden = [x.encode() for x in policy["forbidden_literals"]]
    forbidden += [x.encode() for x in policy["fixture_replacements"]]
    safe_uuids = {x.encode() for x in SOURCE.LEGACY_SYNTHETIC_UUIDS}
    safe_uuids.update(x.encode() for x in policy["fixture_replacements"].values())
    # Existing public unit tests use visually synthetic repeated-digit IDs.
    safe_uuids.update(("GPU-" + "-".join(c * n for n in (8, 4, 4, 4, 12))).encode()
                      for c in "0123456789abcdef")
    safe_uuids.update({b"GPU-00112233-4455-6677-8899-aabbccddeeff",
                       b"GPU-ffeeddcc-bbaa-9988-7766-554433221100"})
    views = [name.encode(), data]
    media = []
    if name.endswith((".json", ".ipynb")):
        decoded_json = SOURCE.read_json(data)
        views.append(canonical(decoded_json))
        pending = [decoded_json]
        while pending:
            value = pending.pop()
            if isinstance(value, str):
                views.append(value.encode())
            elif isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    if name.endswith((".md", ".html", ".txt", ".json", ".ipynb")):
        # Decode common prose/URL escapes. This is not a claim to interpret
        # arbitrary scripts, compressed files, or executable media.
        text_views = views[2:] if name.endswith((".json", ".ipynb")) else [data]
        decoded_views = [urllib.parse.unquote(html.unescape(view.decode("utf-8", "strict"))).encode()
                         for view in text_views]
        views.extend(decoded_views)
        seen_media = set()
        # Search interpreted JSON strings and HTML/URL-decoded views too;
        # escaped data: prefixes and base64 characters must not hide media.
        for view in decoded_views:
            for match in DATA_URL.finditer(view):
                try:
                    decoded = base64.b64decode(re.sub(rb"\s+", b"", match[2]), validate=True)
                except ValueError as exc:
                    raise SOURCE.PublicSourceError("invalid embedded data URL") from exc
                require(len(decoded) <= MAX_FILE, "embedded media exceeds limit")
                key = (match[1], digest(decoded))
                if key in seen_media:
                    continue
                seen_media.add(key)
                views.append(decoded)
                media.append({"mime_type": (match[1].split(b";", 1)[0] or b"text/plain").decode(), "sha256": key[1],
                              "size_bytes": len(decoded)})
    require(sum(map(len, views)) <= MAX_HISTORY * 4, "decoded views exceed scan limit")
    for view in views:
        require(not SECRET.search(view), f"credential pattern in {name}")
        require(not any(x in view for x in forbidden), f"private literal in {name}")
        for value in GPU_UUID.findall(view):
            require(value in safe_uuids or
                    value.startswith(b"GPU-00000000-0000-0000-0000-"),
                    f"unreviewed hardware UUID in {name}")
    return {"name": name, "sha256": digest(data), "size_bytes": len(data),
            "generic_private_path_count": len(SOURCE.PRIVATE_PATH_PATTERN.findall(data)),
            "embedded_media": media}


def inspect_conda(path, policy):
    """Inspect metadata, all tar headers, links and payload bytes in conda-v2."""
    path = regular(path)
    archive_bytes = path.read_bytes()
    before = digest(archive_bytes)
    # ZIP comments, extra fields, local headers and trailing bytes are also
    # distributed. Inspect them rather than only archive.read() payloads.
    scan(path.name, archive_bytes, policy)
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = archive.infolist()
        scan("zip-archive-comment", archive.comment, policy)
        for member in members:
            scan(member.filename + ":zip-header", member.comment + b"\0" + member.extra, policy)
        names = [x.filename for x in members]
        require(len(set(names)) == len(names), "duplicate outer package members")
        info = [n for n in names if n.startswith("info-") and n.endswith(".tar.zst")]
        pkg = [n for n in names if n.startswith("pkg-") and n.endswith(".tar.zst")]
        require(len(info) == len(pkg) == 1 and set(names) == {"metadata.json", *info, *pkg},
                "unexpected conda-v2 outer inventory")
        require(all(0 <= x.file_size <= MAX_FILE for x in members), "package member exceeds limit")
        require(SOURCE.read_json(archive.read("metadata.json")) == {"conda_pkg_format_version": 2},
                "invalid conda-v2 format metadata")
        scan(path.name, archive.read("metadata.json"), policy)
        for outer in info + pkg:
            scan(outer, b"", policy)
            compressed = archive.read(outer)
            # zstd skippable frames do not reach decoded tar output.
            scan(outer, compressed, policy)
            raw_tar = PACKAGE._zstd_decompress(compressed, "public package")
            # Also scan the raw tar: owner names, PAX headers, link names,
            # discarded directory headers and padding must not hide literals.
            scan(outer, raw_tar, policy)
            files = PACKAGE._tar_files(raw_tar, "public package")
            require(not any(n == "info/recipe" or n.startswith("info/recipe/") for n in files),
                    "rendered Conda recipe must not be distributed")
            require(all(n.startswith("info/") == (outer == info[0]) for n in files),
                    "package metadata/payload namespace differs")
            for name, member in sorted(files.items()):
                row = scan(name, member.data if member.data is not None else
                           (member.link_target or "").encode(), policy)
                row.update(archive=outer, member_type=member.member_type)
                rows.append(row)
    require(digest(path.read_bytes()) == before, "package changed during inspection")
    return {"name": path.name, "sha256": before, "members": rows}


def inspect_history(repo, private_repo, policy):
    """Inventory every local ref and every reachable public commit/blob/tag.

    Remote PR/cached refs are outside a local repository. The publisher must
    separately bind a fresh remote inventory and the independent history audit.
    """
    git = SOURCE.git
    require(git(repo, "rev-parse", "--is-shallow-repository").strip() == b"false",
            "shallow history is not reviewable")
    for component, message in (("objects/info/alternates", "object alternates are forbidden"),
                               ("info/grafts", "grafted history is forbidden")):
        # Resolve through the common Git directory for linked worktrees.
        path = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-path", component).decode().strip())
        require(not path.exists(), message)
    refs = {}
    for line in git(repo, "for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines():
        ref, oid = line.split()
        require(ref.startswith(("refs/heads/", "refs/tags/", "refs/remotes/")), "unexpected public ref")
        scan(ref, b"", policy)
        refs[ref] = oid
    require(bool(refs), "public repository has no refs")
    roots = sorted(set(refs.values()))
    commits = git(repo, "rev-list", *roots).decode().splitlines()
    require(0 < len(commits) <= MAX_COMMITS, "public commit inventory is empty or too large")
    private = set(git(private_repo, "rev-list", "--all").decode().splitlines())
    require(not private.intersection(commits), "private development commits reached public history")
    objects = {}
    for line in git(repo, "rev-list", "--objects", *roots).decode().splitlines():
        oid = line.split(" ", 1)[0]
        objects[oid] = git(repo, "cat-file", "-t", oid).decode().strip()
    total = 0
    records = []
    for oid, kind in sorted(objects.items()):
        size = int(git(repo, "cat-file", "-s", oid))
        total += size
        require(size <= MAX_FILE and total <= MAX_HISTORY, "public history exceeds size limit")
        data = git(repo, "cat-file", kind, oid)
        # Tree bytes contain all path components, including directories.
        # A ref/tag can point directly to a blob or tree outside any commit.
        # Every reachable object needs a raw scan, even without a path name.
        row = scan(f"{kind}:{oid}", data, policy)
        row["object_id"] = oid
        records.append(row)
    seen = set()
    for commit in commits:
        for name, (mode, oid) in SOURCE.tree(repo, commit).items():
            require(not SOURCE.omitted(name), "private evidence path in public history")
            key = (name, oid)
            if key in seen:
                continue
            seen.add(key)
            data = git(repo, "cat-file", "blob", oid)
            row = scan(name, data, policy)
            row.update(object_id=oid, mode=mode)
            records.append(row)
    require(refs == dict(line.split() for line in git(
        repo, "for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines()),
        "public refs changed during inspection")
    return {"refs": refs, "commits": sorted(commits), "objects": dict(sorted(objects.items())),
            "files": sorted(records, key=lambda row: (row["name"], row["object_id"]))}


def verify(policy, *, package=None, assets=(), public_repo=None, private_repo=None):
    report = {"schema": "gpmeep-public-artifact-inspection-v1", "state": "LITERALS_CHECKED",
              "publication_authorized": False, "policy_sha256": digest(canonical(policy)),
              "implementation": {name: digest(Path(__file__).with_name(name).read_bytes())
                                 for name in (Path(__file__).name, "verify-public-source.py",
                                              "v110-final-release-receipt.py")}}
    if package is not None:
        report["package"] = inspect_conda(package, policy)
    report["assets"] = []
    for asset in assets:
        asset = regular(asset)
        require(asset.suffix in {".md", ".html", ".json", ".txt", ".sha256", ".png"},
                "unsupported public asset format")
        report["assets"].append(scan(asset.name, asset.read_bytes(), policy))
    if public_repo is not None:
        require(private_repo is not None, "private repository required for history isolation")
        report["history"] = inspect_history(public_repo, private_repo, policy)
    require(package is not None or assets or public_repo is not None, "no public artifacts supplied")
    report["remaining_review"] = ["documentation and media privacy", "remote refs and cached commits",
                                  "exact-package final release authority", "independent publication audit"]
    report["report_id"] = digest(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--public-repo", type=Path)
    parser.add_argument("--private-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(policy_read(args.policy), package=args.package, assets=args.asset,
                        public_repo=args.public_repo, private_repo=args.private_repo)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
            staged = Path(stream.name)
            stream.write(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, args.output)
        finally:
            staged.unlink(missing_ok=True)
        print(json.dumps({"state": report["state"], "report_id": report["report_id"],
                          "publication_authorized": False}))
        return 0
    except (SOURCE.PublicSourceError, PACKAGE.FinalReleaseError, OSError, ValueError,
            subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"public artifact inspection failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
