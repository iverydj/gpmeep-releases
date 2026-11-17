#!/usr/bin/env python3
"""Tests for the frozen gpmeep v1.0.3 runtime-identity boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/python-validation/v103_runtime_identity.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_v103_runtime_identity_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def record(path: pathlib.Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


class V103RuntimeIdentityTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, str]:
        prefix = root / "prefix"
        meep = prefix / "lib/python3.11/site-packages/meep"
        meep.mkdir(parents=True)
        (prefix / "bin").mkdir()
        (prefix / "lib").mkdir(exist_ok=True)
        (prefix / "share/gpmeep").mkdir(parents=True)
        (prefix / "conda-meta").mkdir()
        files = {
            "python": prefix / "bin/python3.11",
            "module": meep / "__init__.py",
            "extension": meep / "_meep.so.38.0.0",
            "libmeep": prefix / "lib/libmeep.so.38.0.0",
            "manifest": prefix / "share/gpmeep/release.json",
            "conda": prefix / "conda-meta/gpmeep.json",
        }
        for name, path in files.items():
            path.write_bytes((name + "\n").encode())
        package = root / IDENTITY.FROZEN_PACKAGE_NAME
        package.write_bytes(b"frozen-package")
        package_sha = hashlib.sha256(package.read_bytes()).hexdigest()
        manifest = {
            "distribution": "gpmeep",
            "distribution_version": IDENTITY.FROZEN_VERSION,
            "source_commit": IDENTITY.FROZEN_SOURCE_COMMIT,
            "precision": "fp32",
            "mpi": "openmpi",
            "cuda_toolkit": "12.4",
            "cuda_fast_math": False,
            "scheme": False,
            "file": record(files["manifest"]),
        }
        report = {
            "schema_version": 1,
            "state": "PASS",
            "prefix": str(prefix),
            "conda_record": record(files["conda"]),
            "package_url": package.as_uri(),
            "package_sha256": package_sha,
            "ownership": {},
            "manifest": manifest,
            "runtime": {
                "python": record(files["python"]),
                "module": record(files["module"]),
                "extension": record(files["extension"]),
                "libmeep": record(files["libmeep"]),
                "single_precision": True,
                "mpi_enabled": True,
                "cuda_compiled": True,
            },
        }
        report["report_id"] = hashlib.sha256(IDENTITY._canonical_json(report)).hexdigest()
        report_path = root / "provenance.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, package, prefix, package_sha

    def test_report_binds_package_prefix_and_every_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report_path, package, prefix, package_sha = self.fixture(pathlib.Path(raw))
            report_sha = IDENTITY.sha256_file(report_path)
            report_id = json.loads(report_path.read_text(encoding="utf-8"))["report_id"]
            with (
                mock.patch.object(IDENTITY, "FROZEN_PACKAGE_SHA256", package_sha),
                mock.patch.object(
                    IDENTITY, "FROZEN_PROVENANCE_REPORT_SHA256", report_sha
                ),
                mock.patch.object(
                    IDENTITY, "FROZEN_PROVENANCE_REPORT_ID", report_id
                ),
            ):
                report, paths = IDENTITY.validate_report(report_path, package, prefix)
                self.assertEqual(report["package_sha256"], package_sha)
                self.assertEqual(paths["install_prefix"], prefix)
                paths["meep_module"].write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(Exception, "bytes differ"):
                    IDENTITY.validate_report(report_path, package, prefix)

    def test_report_id_and_file_url_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report_path, package, prefix, package_sha = self.fixture(pathlib.Path(raw))
            value = json.loads(report_path.read_text(encoding="utf-8"))
            value["report_id"] = "0" * 64
            report_path.write_text(json.dumps(value), encoding="utf-8")
            with (
                mock.patch.object(IDENTITY, "FROZEN_PACKAGE_SHA256", package_sha),
                mock.patch.object(
                    IDENTITY,
                    "FROZEN_PROVENANCE_REPORT_SHA256",
                    IDENTITY.sha256_file(report_path),
                ),
            ):
                with self.assertRaisesRegex(Exception, "report ID"):
                    IDENTITY.validate_report(report_path, package, prefix)

    def test_recomputed_report_cannot_authorize_a_forged_conda_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report_path, package, prefix, package_sha = self.fixture(pathlib.Path(raw))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            conda_record = prefix / "conda-meta/gpmeep.json"
            conda_record.write_text("forged ownership metadata\n", encoding="utf-8")
            report["conda_record"] = record(conda_record)
            without_id = {key: value for key, value in report.items() if key != "report_id"}
            report["report_id"] = hashlib.sha256(
                IDENTITY._canonical_json(without_id)
            ).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(IDENTITY, "FROZEN_PACKAGE_SHA256", package_sha):
                with self.assertRaisesRegex(Exception, "provenance report bytes"):
                    IDENTITY.validate_report(report_path, package, prefix)

    def test_current_process_requires_all_external_identity_inputs(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(Exception, "environment is incomplete"):
                IDENTITY.verify_current_process(object())

    def test_nonfinite_and_duplicate_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "report.json"
            path.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "non-finite"):
                IDENTITY._load_json(path)
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "duplicates"):
                IDENTITY._load_json(path)


if __name__ == "__main__":
    unittest.main()
