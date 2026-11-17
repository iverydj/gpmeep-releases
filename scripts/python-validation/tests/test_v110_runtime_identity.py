#!/usr/bin/env python3
"""Adversarial tests for exact v1.1.0 installed-package runtime binding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/python-validation/v110_runtime_identity.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_v110_identity_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IDENTITY
SPEC.loader.exec_module(IDENTITY)
COMMIT = "c" * 40


def record(path: pathlib.Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root.resolve()
        self.prefix = self.root / "prefix"
        module = self.prefix / "lib/python3.11/site-packages/meep/__init__.py"
        extension = module.parent / "_meep.so"
        python = self.prefix / "bin/python3.11"
        mpirun = self.prefix / "bin/mpirun"
        verifier = self.prefix / "bin/gpmeep-verify-provenance"
        libmeep = self.prefix / "lib/libmeep.so.38"
        conda_record = self.prefix / "conda-meta/gpmeep-1.1.0-0.json"
        manifest_file = self.prefix / "share/gpmeep/release.json"
        fontconfig = self.prefix / "etc/fonts/fonts.conf"
        for path, payload in (
            (module, b"module\n"),
            (extension, b"extension\n"),
            (python, b"#!/bin/sh\n"),
            (mpirun, b"#!/bin/sh\n"),
            (verifier, b"#!/bin/sh\n"),
            (libmeep, b"libmeep\n"),
            (conda_record, b"{}\n"),
            (manifest_file, b"{}\n"),
            (fontconfig, b"fonts\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        for path in (python, mpirun, verifier):
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        libmeep_alias = self.prefix / "lib/libmeep.so"
        libmeep_alias.symlink_to(libmeep.name)
        self.package = self.root / "gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0_release.conda"
        self.package.write_bytes(b"exact-package\n")
        owned = (
            module,
            extension,
            verifier,
            libmeep,
            libmeep_alias,
            manifest_file,
        )
        relative_owned = [path.relative_to(self.prefix).as_posix() for path in owned]
        path_records = []
        self.archive_records = {}
        for path in owned:
            relative = path.relative_to(self.prefix).as_posix()
            if path.is_symlink():
                target = path.resolve(strict=True)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                size = target.stat().st_size
                path_records.append(
                    {
                        "_path": relative,
                        "path_type": "softlink",
                        "sha256_in_prefix": digest,
                        "size_in_bytes": size,
                    }
                )
                self.archive_records[relative] = {
                    "path_type": "softlink",
                    "link_target": target.relative_to(self.prefix).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                    "mode": stat.S_IMODE(path.lstat().st_mode),
                    "content_mode": stat.S_IMODE(target.stat().st_mode),
                }
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                size = path.stat().st_size
                path_records.append(
                    {
                        "_path": relative,
                        "path_type": "hardlink",
                        "sha256": digest,
                        "sha256_in_prefix": digest,
                        "size_in_bytes": size,
                    }
                )
                self.archive_records[relative] = {
                    "path_type": "hardlink",
                    "sha256": digest,
                    "size_bytes": size,
                    "mode": stat.S_IMODE(path.lstat().st_mode),
                    "content_mode": stat.S_IMODE(path.stat().st_mode),
                }
        conda_record.write_text(
            json.dumps(
                {
                    "name": "gpmeep",
                    "version": "1.1.0",
                    "build": "cuda124_mpi_openmpi_py311_0",
                    "subdir": "linux-64",
                    "fn": self.package.name,
                    "sha256": hashlib.sha256(self.package.read_bytes()).hexdigest(),
                    "files": relative_owned,
                    "paths_data": {
                        "paths_version": 1,
                        "paths": path_records,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.report_path = self.root / "installed-provenance.json"
        self.report = {
            "schema_version": 1,
            "state": "PASS",
            "prefix": str(self.prefix),
            "conda_record": record(conda_record),
            "package_url": self.package.as_uri(),
            "package_sha256": hashlib.sha256(self.package.read_bytes()).hexdigest(),
            "ownership": {
                "owned_paths": 6,
                "regular_files": 5,
                "symlinks": 1,
                "content_hashes_verified": 6,
                "content_sizes_verified": 6,
                "prefix_rewritten_sizes": 0,
            },
            "manifest": {
                "distribution": "gpmeep",
                "distribution_version": "1.1.0",
                "source_commit": COMMIT,
                "precision": "fp32",
                "mpi": "openmpi",
                "cuda_toolkit": "12.4",
                "cuda_fast_math": False,
                "scheme": False,
                "file": record(manifest_file),
            },
            "runtime": {
                "python": record(python),
                "module": record(module),
                "extension": record(extension),
                "libmeep": record(libmeep),
                "meep_version": "1.31.0",
                "cuda_compiled": True,
                "mpi_enabled": True,
                "single_precision": True,
            },
            "v110_candidate_gate": {
                "build_mode": "candidate",
                "state": "CANDIDATE_BUILD_AUTHORIZED",
                "source_commit": COMMIT,
                "receipt_id": "a" * 64,
            },
        }
        self.reseal()

    def package_archive_records(
        self, package: pathlib.Path
    ) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
        if package.resolve() != self.package.resolve():
            raise AssertionError("unexpected package path")
        development = self.report["v110_candidate_gate"]["build_mode"] == "development"
        return (
            {
                "name": "gpmeep",
                "version": "1.1.0",
                "build": (
                    "cuda124_mpi_openmpi_py311_0_development"
                    if development
                    else "cuda124_mpi_openmpi_py311_0"
                ),
                "subdir": "linux-64",
            },
            {name: dict(value) for name, value in self.archive_records.items()},
        )

    def write_development_marker(self) -> pathlib.Path:
        marker = pathlib.Path(
            f"{self.package}{IDENTITY.NONPUBLISHABLE_MARKER_SUFFIX}"
        )
        marker.write_text(
            json.dumps(
                {
                    "schema": IDENTITY.NONPUBLISHABLE_MARKER_SCHEMA,
                    "state": "NON_PUBLISHABLE",
                    "source_commit": COMMIT,
                    "package": {
                        "file_name": self.package.name,
                        "size_bytes": self.package.stat().st_size,
                        "sha256": hashlib.sha256(
                            self.package.read_bytes()
                        ).hexdigest(),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        marker.chmod(0o400)
        return marker

    def make_development(self) -> pathlib.Path:
        development = (
            self.root
            / "gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0_development.conda"
        )
        self.package.rename(development)
        self.package = development
        self.report["package_url"] = self.package.as_uri()
        conda_record = pathlib.Path(self.report["conda_record"]["path"])
        metadata = json.loads(conda_record.read_text(encoding="utf-8"))
        metadata["fn"] = self.package.name
        metadata["build"] = "cuda124_mpi_openmpi_py311_0_development"
        conda_record.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        self.report["conda_record"] = record(conda_record)
        self.report["v110_candidate_gate"] = {
            "build_mode": "development",
            "state": "DEVELOPMENT_ONLY",
            "source_commit": COMMIT,
            "receipt_id": "a" * 64,
        }
        marker = self.write_development_marker()
        self.reseal()
        return marker

    def reseal(self) -> None:
        unsigned = {key: value for key, value in self.report.items() if key != "report_id"}
        self.report["report_id"] = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")


class V110RuntimeIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(pathlib.Path(self.temporary.name))
        self.archive_patch = mock.patch.object(
            IDENTITY,
            "_package_archive_records",
            side_effect=self.fixture.package_archive_records,
        )
        self.archive_patch.start()

    def tearDown(self) -> None:
        self.archive_patch.stop()
        self.temporary.cleanup()

    def validate(self):
        return IDENTITY.validate_report(
            self.fixture.report_path,
            self.fixture.package,
            self.fixture.prefix,
            COMMIT,
        )

    def test_report_binds_package_candidate_prefix_and_runtime_bytes(self) -> None:
        report_value, installed = self.validate()
        owned = IDENTITY.package_owned_records(report_value, installed)
        runtime = IDENTITY.runtime_paths(report_value, installed)
        contract = IDENTITY.runtime_contract(report_value, installed)
        self.assertEqual(runtime["python"], installed["python_executable"])
        self.assertEqual(runtime["build_receipt"], self.fixture.report_path)
        self.assertEqual(runtime["package"], self.fixture.package)
        self.assertEqual(contract["receipt_id"], report_value["report_id"])
        self.assertEqual(len(owned), 6)

    def test_archive_target_and_mode_are_enforced(self) -> None:
        report_value, installed = self.validate()
        alias = self.fixture.prefix / "lib/libmeep.so"
        original_target = os.readlink(alias)
        alias.unlink()
        alias.symlink_to(f"./{original_target}")
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "raw target differs from exact archive"
        ):
            IDENTITY.package_owned_records(report_value, installed)
        alias.unlink()
        alias.symlink_to((self.fixture.prefix / "lib/libmeep.so.38").resolve())
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "softlink target is unsafe"
        ):
            IDENTITY.package_owned_records(report_value, installed)
        alias.unlink()
        same_bytes = self.fixture.prefix / "lib/libmeep-copy.so"
        same_bytes.write_bytes((self.fixture.prefix / "lib/libmeep.so.38").read_bytes())
        alias.symlink_to(same_bytes.name)
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "softlink target differs from exact archive"
        ):
            IDENTITY.package_owned_records(report_value, installed)
        alias.unlink()
        alias.symlink_to(original_target)
        module = self.fixture.prefix / "lib/python3.11/site-packages/meep/__init__.py"
        original_mode = stat.S_IMODE(module.stat().st_mode)
        module.chmod(0o600)
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "mode differs from exact archive"
        ):
            IDENTITY.package_owned_records(report_value, installed)
        module.chmod(original_mode)

    def test_package_owned_python_module_is_verified_without_live_verifier(self) -> None:
        report_value, installed = self.validate()
        module = self.fixture.prefix / "lib/python3.11/site-packages/meep/__init__.py"
        module.write_bytes(b"same-size-mutant\n")
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "owned file bytes differ"
        ):
            IDENTITY.package_owned_records(report_value, installed)

    def test_same_name_or_self_declared_candidate_cannot_replace_exact_package(self) -> None:
        replacement = self.fixture.root / "other" / self.fixture.package.name
        replacement.parent.mkdir()
        replacement.write_bytes(self.fixture.package.read_bytes())
        with self.assertRaisesRegex(IDENTITY.V110IdentityError, "package URL differs"):
            IDENTITY.validate_report(
                self.fixture.report_path, replacement, self.fixture.prefix, COMMIT
            )
        marker = self.fixture.make_development()
        report_value, _installed = self.validate()
        self.assertEqual(report_value["v110_candidate_gate"]["build_mode"], "development")
        marker.chmod(0o600)
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "non-publishable marker metadata differs"
        ):
            self.validate()
        marker.chmod(0o400)
        marker.unlink()
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "non-publishable marker is unavailable"
        ):
            self.validate()
        marker = self.fixture.write_development_marker()
        marker.chmod(0o600)
        marker.write_text('{"state":"NON_PUBLISHABLE"}\n', encoding="utf-8")
        marker.chmod(0o400)
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError, "non-publishable marker content differs"
        ):
            self.validate()
        marker.chmod(0o600)
        marker.unlink()
        self.fixture.write_development_marker()
        self.fixture.report["v110_candidate_gate"]["state"] = "CANDIDATE_BUILD_AUTHORIZED"
        self.fixture.reseal()
        with self.assertRaisesRegex(IDENTITY.V110IdentityError, "authorized development"):
            self.validate()

    def test_candidate_rejects_nonpublishable_marker(self) -> None:
        marker = pathlib.Path(
            f"{self.fixture.package}{IDENTITY.NONPUBLISHABLE_MARKER_SUFFIX}"
        )
        marker.write_text("{}\n", encoding="utf-8")
        marker.chmod(0o400)
        with self.assertRaisesRegex(
            IDENTITY.V110IdentityError,
            "candidate package carries a non-publishable marker",
        ):
            self.validate()

    def test_runtime_mutation_and_source_mismatch_fail_closed(self) -> None:
        self.fixture.prefix.joinpath(
            "lib/python3.11/site-packages/meep/_meep.so"
        ).write_bytes(b"changed\n")
        with self.assertRaisesRegex(IDENTITY.V110IdentityError, "extension (size|bytes) differs"):
            self.validate()
        self.fixture = Fixture(self.fixture.root / "fresh")
        with self.assertRaisesRegex(IDENTITY.V110IdentityError, "release manifest differs"):
            IDENTITY.validate_report(
                self.fixture.report_path,
                self.fixture.package,
                self.fixture.prefix,
                "d" * 40,
            )

    def test_live_verifier_report_must_exactly_match_retained_report(self) -> None:
        report_value, installed = self.validate()

        def completed(command, **_kwargs):
            output = pathlib.Path(command[command.index("--report") + 1])
            output.write_text(json.dumps(report_value), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, b"PASS\n", b"")

        with mock.patch.object(IDENTITY.subprocess, "run", side_effect=completed):
            live = IDENTITY.live_verify(report_value, installed)
        self.assertEqual(live["report_id"], report_value["report_id"])
        self.assertEqual(live["package_sha256"], report_value["package_sha256"])

        def forged(command, **_kwargs):
            output = pathlib.Path(command[command.index("--report") + 1])
            changed = dict(report_value, state="FORGED")
            output.write_text(json.dumps(changed), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, b"PASS\n", b"")

        with mock.patch.object(IDENTITY.subprocess, "run", side_effect=forged):
            with self.assertRaisesRegex(
                IDENTITY.V110IdentityError, "live installed provenance differs"
            ):
                IDENTITY.live_verify(report_value, installed)


if __name__ == "__main__":
    unittest.main()
