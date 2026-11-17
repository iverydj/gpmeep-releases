from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NATIVE = load_script("v110_native_witness_tests", ROOT / "scripts/v110-native-witness.py")


class V110NativeWitnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.prefix = self.root / "prefix"
        self.prefix.mkdir()
        self.source_commit = "c" * 40
        self.package_sha256 = "d" * 64
        self.candidate_receipt_id = "e" * 64
        self.report = self.root / "installed.json"
        installed = {
            "state": "PASS",
            "prefix": str(self.prefix.resolve()),
            "package_sha256": self.package_sha256,
            "manifest": {"source_commit": self.source_commit},
            "v110_candidate_gate": {"receipt_id": self.candidate_receipt_id},
        }
        installed["report_id"] = NATIVE._canonical_sha256(installed)
        self.report.write_text(json.dumps(installed, sort_keys=True))
        self.live = {
            "capture_source": "/etc/os-release",
            "os_release": {
                "id": "ubuntu",
                "version_id": "22.04",
                "size_bytes": 42,
                "sha256": "1" * 64,
            },
            "namespaces": {
                "mnt": "mnt:[100]",
                "pid": "pid:[200]",
                "user": "user:[300]",
            },
            "kernel": {"sysname": "Linux", "release": "test", "machine": "x86_64"},
            "libc": "glibc 2.35",
            "cgroup_sha256": "2" * 64,
        }
        self.live["namespace_id"] = NATIVE._canonical_sha256(self.live["namespaces"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create(self) -> dict[str, object]:
        with mock.patch.object(NATIVE, "_live_environment", return_value=self.live):
            return NATIVE.create(
                prefix=self.prefix,
                installed_report_path=self.report,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_receipt_id,
                image_identity="sha256:" + "3" * 64,
            )

    def test_witness_round_trip_binds_live_environment_and_installed_report(self):
        witness = self._create()
        path = self.root / "native-witness.json"
        NATIVE._write_new(path, witness)
        self.assertEqual(NATIVE.verify(path), witness)
        with mock.patch.object(NATIVE, "_live_environment", return_value=self.live):
            self.assertEqual(NATIVE.verify(path, require_live=True), witness)

    def test_live_host_round_trip_without_environment_mock(self):
        try:
            live = NATIVE._live_environment()
        except NATIVE.WitnessError as error:
            self.skipTest(str(error))
        witness = NATIVE.create(
            prefix=self.prefix,
            installed_report_path=self.report,
            source_commit=self.source_commit,
            package_sha256=self.package_sha256,
            candidate_receipt_id=self.candidate_receipt_id,
            image_identity="native-host",
        )
        self.assertEqual(witness["os_release"], live["os_release"])
        path = self.root / "native-live-witness.json"
        NATIVE._write_new(path, witness)
        self.assertEqual(NATIVE.verify(path, require_live=True), witness)

    def test_live_replay_rejects_different_namespace(self):
        witness = self._create()
        path = self.root / "native-witness.json"
        NATIVE._write_new(path, witness)
        changed = dict(self.live)
        changed["namespaces"] = dict(self.live["namespaces"], pid="pid:[999]")
        changed["namespace_id"] = NATIVE._canonical_sha256(changed["namespaces"])
        with mock.patch.object(NATIVE, "_live_environment", return_value=changed):
            with self.assertRaisesRegex(NATIVE.WitnessError, "live namespaces"):
                NATIVE.verify(path, require_live=True)

    def test_self_hashed_but_malformed_witness_is_rejected(self):
        witness = self._create()
        witness["namespaces"] = {"mnt": "attacker", "pid": "pid:[200]", "user": "user:[300]"}
        witness["namespace_id"] = NATIVE._canonical_sha256(witness["namespaces"])
        witness.pop("environment_id")
        witness.pop("terminal_id")
        witness["environment_id"] = NATIVE._canonical_sha256(witness)
        witness["terminal_id"] = NATIVE._canonical_sha256(witness)
        path = self.root / "malformed.json"
        path.write_text(json.dumps(witness, sort_keys=True))
        with self.assertRaisesRegex(NATIVE.WitnessError, "identity differs"):
            NATIVE.verify(path)


if __name__ == "__main__":
    unittest.main()
