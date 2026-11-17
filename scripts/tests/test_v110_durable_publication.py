#!/usr/bin/env python3
"""Fault-injection tests for release-critical non-clobber publication."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
USER_WORKLOADS = SCRIPTS / "user-workloads"
if str(USER_WORKLOADS) not in sys.path:
    sys.path.insert(0, str(USER_WORKLOADS))

import common  # noqa: E402


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULES = (
    (load("gpmeep_test_transition_durable", "verify-v110-release-source-transition.py"), {"kind": "transition"}),
    (load("gpmeep_test_gate_durable", "v110-release-gate-receipt.py"), {"kind": "gate"}),
    (load("gpmeep_test_final_durable", "v110-final-release-receipt.py"), {"kind": "final"}),
    (load("gpmeep_test_assets_durable", "v110-release-assets.py"), b"asset-bytes\n"),
    (load("gpmeep_test_provenance_durable", "verify-gpmeep-package.py"), {"kind": "provenance"}),
)


class V110DurablePublicationTest(unittest.TestCase):
    def test_matrix_terminal_first_fsync_failure_rolls_back_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            pending = root / "PENDING_M4_COMPLETE"
            complete = root / "M4_COMPLETE"
            pending.write_bytes(b"verified terminal\n")
            with mock.patch.object(
                common, "_fsync_directory", side_effect=OSError("injected fsync")
            ):
                with self.assertRaisesRegex(OSError, "injected fsync"):
                    common.durable_publish_terminal(pending, complete)
            self.assertTrue(pending.is_file())
            self.assertFalse(complete.exists())

    def test_matrix_terminal_cleanup_fsync_failure_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            pending = root / "PENDING_M4_COMPLETE"
            complete = root / "M4_COMPLETE"
            pending.write_bytes(b"verified terminal\n")
            real_fsync = common._fsync_directory
            calls = 0

            def fail_second(path: pathlib.Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected cleanup fsync")
                real_fsync(path)

            with mock.patch.object(common, "_fsync_directory", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected cleanup fsync"):
                    common.durable_publish_terminal(pending, complete)
            self.assertFalse(pending.exists())
            self.assertEqual(complete.read_bytes(), b"verified terminal\n")
            common.durable_publish_terminal(pending, complete)
            self.assertEqual(complete.read_bytes(), b"verified terminal\n")

    def test_matrix_terminal_never_clobbers_an_existing_complete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            pending = root / "PENDING_M4_COMPLETE"
            complete = root / "M4_COMPLETE"
            pending.write_bytes(b"new\n")
            complete.write_bytes(b"old\n")
            with self.assertRaisesRegex(Exception, "already exists"):
                common.durable_publish_terminal(pending, complete)
            self.assertEqual(complete.read_bytes(), b"old\n")

    def test_every_release_writer_fsyncs_parent_after_exact_publication(self) -> None:
        for module, payload in MODULES:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as raw:
                root = pathlib.Path(raw).resolve()
                destination = root / "terminal.json"
                real_fsync = module.os.fsync
                calls = 0

                def recording_fsync(descriptor: int) -> None:
                    nonlocal calls
                    calls += 1
                    real_fsync(descriptor)

                with mock.patch.object(module.os, "fsync", side_effect=recording_fsync):
                    module._write_new(destination, payload)
                self.assertGreaterEqual(calls, 2)
                self.assertTrue(destination.is_file())
                self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_parent_fsync_failure_leaves_no_apparently_committed_terminal(self) -> None:
        for module, payload in MODULES:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as raw:
                root = pathlib.Path(raw).resolve()
                destination = root / "terminal.json"
                real_fsync = module.os.fsync
                calls = 0

                def fail_parent_fsync(descriptor: int) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("injected parent fsync failure")
                    real_fsync(descriptor)

                with mock.patch.object(module.os, "fsync", side_effect=fail_parent_fsync):
                    with self.assertRaisesRegex(OSError, "injected parent fsync failure"):
                        module._write_new(destination, payload)
                self.assertFalse(destination.exists())
                self.assertEqual(list(root.glob(".terminal.json.*")), [])

    def test_existing_destination_is_never_replaced(self) -> None:
        for module, payload in MODULES:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as raw:
                destination = pathlib.Path(raw).resolve() / "terminal.json"
                destination.write_bytes(b"existing\n")
                with self.assertRaises(Exception):
                    module._write_new(destination, payload)
                self.assertEqual(destination.read_bytes(), b"existing\n")


if __name__ == "__main__":
    unittest.main()
