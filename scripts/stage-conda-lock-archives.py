#!/usr/bin/env python3
"""Stage exact-lock package archives at stable URL-derived cache paths.

Micromamba stores downloaded archives at ``PKGS_DIR/<basename>``.  The prefix
auditor deliberately addresses each archive by its complete lock URL so that
two channels cannot silently alias one basename.  This helper bridges those
layouts after environment creation, while verifying the lock digest before a
byte is admitted to the URL-derived audit cache.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import stat
import tempfile
from dataclasses import dataclass


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SCHEME_RE = re.compile(r"[a-z][a-z0-9+.-]*")


@dataclass(frozen=True)
class LockedArchive:
    url: str
    sha256: str
    basename: str
    destination_parts: tuple[str, ...]


def canonical_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent: {lexical}") from error
    if not stat.S_ISREG(info.st_mode) or resolved != lexical:
        raise RuntimeError(f"{label} is not a canonical regular file: {lexical}")
    return lexical


def canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent: {lexical}") from error
    if not stat.S_ISDIR(info.st_mode) or resolved != lexical:
        raise RuntimeError(f"{label} is not a canonical directory: {lexical}")
    return lexical


def parse_lock(lock: pathlib.Path) -> list[LockedArchive]:
    lock = canonical_regular(lock, "exact lock")
    entries: list[LockedArchive] = []
    destinations: dict[tuple[str, ...], tuple[str, str]] = {}
    for line_number, raw in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line == "@EXPLICIT":
            continue
        url, separator, digest = line.rpartition("#")
        if not separator or SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(
                f"exact lock line {line_number} has no lowercase SHA-256 fragment"
            )
        scheme, url_separator, remainder = url.partition("://")
        parts = remainder.split("/")
        if (
            not url_separator
            or SCHEME_RE.fullmatch(scheme) is None
            or len(parts) < 2
            or any(not part or part in {".", ".."} for part in parts)
            or any(character in remainder for character in "?#\\")
        ):
            raise RuntimeError(f"exact lock line {line_number} has an unmappable URL")
        destination_parts = (scheme, *parts)
        identity = (url, digest)
        previous = destinations.setdefault(destination_parts, identity)
        if previous != identity:
            raise RuntimeError("exact lock URLs collide at one archive destination")
        entries.append(
            LockedArchive(url, digest, parts[-1], destination_parts)
        )
    if not entries:
        raise RuntimeError("exact lock contains no package URLs")
    return entries


def digest_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def open_verified(path: pathlib.Path, expected: str, label: str) -> int:
    path = canonical_regular(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("archive staging requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        fingerprint = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if fingerprint != (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ):
            raise RuntimeError(f"{label} changed before hashing: {path}")
        actual = digest_descriptor(descriptor)
        if actual != expected:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: {path.name}: {actual} != {expected}"
            )
        replay = os.fstat(descriptor)
        if fingerprint != (
            replay.st_dev,
            replay.st_ino,
            replay.st_mode,
            replay.st_size,
            replay.st_mtime_ns,
            replay.st_ctime_ns,
        ):
            raise RuntimeError(f"{label} changed while hashing: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def canonical_destination_parent(
    package_cache: pathlib.Path, parts: tuple[str, ...]
) -> pathlib.Path:
    parent = package_cache.joinpath(*parts[:-1])
    parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    return canonical_directory(parent, "URL-derived package-cache directory")


def stage_one(package_cache: pathlib.Path, entry: LockedArchive) -> bool:
    destination = package_cache.joinpath(*entry.destination_parts)
    if destination.exists() or destination.is_symlink():
        descriptor = open_verified(
            destination, entry.sha256, "URL-derived exact-lock package archive"
        )
        os.close(descriptor)
        return False

    source = package_cache / entry.basename
    source_descriptor = open_verified(
        source, entry.sha256, "Micromamba exact-lock package archive"
    )
    parent = canonical_destination_parent(package_cache, entry.destination_parts)
    temporary_descriptor = -1
    temporary_path: pathlib.Path | None = None
    try:
        temporary_descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{entry.basename}.", suffix=".staging", dir=parent
        )
        temporary_path = pathlib.Path(raw_temporary)
        copied_digest = hashlib.sha256()
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            copied_digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(temporary_descriptor, view)
                if written <= 0:
                    raise RuntimeError("cannot write staged exact-lock archive")
                view = view[written:]
        if copied_digest.hexdigest() != entry.sha256:
            raise RuntimeError(
                f"package archive changed while staging: {entry.basename}"
            )
        os.fsync(temporary_descriptor)
        os.fchmod(temporary_descriptor, 0o444)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError:
            descriptor = open_verified(
                destination,
                entry.sha256,
                "concurrently staged exact-lock package archive",
            )
            os.close(descriptor)
            return False
        return True
    finally:
        os.close(source_descriptor)
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def stage(lock: pathlib.Path, package_cache: pathlib.Path) -> tuple[int, int]:
    package_cache = canonical_directory(package_cache, "conda package cache")
    entries = parse_lock(lock)
    staged = sum(stage_one(package_cache, entry) for entry in entries)
    return staged, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=pathlib.Path)
    parser.add_argument("--package-cache", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        staged, total = stage(args.lock, args.package_cache)
    except (OSError, RuntimeError, UnicodeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Exact-lock archive staging verified: {staged} staged, {total} total")


if __name__ == "__main__":
    main()
