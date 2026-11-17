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


CLOSURE = load_script(
    "v110_inversion_closure", ROOT / "scripts/close-v110-inversion-registry.py"
)


def pass_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "state": "PASS",
        "inventory_sha256": "f" * 64,
        "observed_count": 13,
        "registered_count": 13,
        "missing_count": 0,
        "blocking_count": 0,
        "closed_count": 13,
        "open_count": 0,
        "missing_workloads": [],
        "blocking_workloads": [],
        "open_workloads": [],
    }


class V110InversionClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.records = self.root / "closed-records"
        self.records.mkdir()
        for index in range(13):
            (self.records / f"record-{index:02d}.json").write_text(
                json.dumps({"record": index}) + "\n", encoding="utf-8"
            )
        self.validator = type(
            "Validator",
            (),
            {"validate_registry": staticmethod(lambda **_kwargs: pass_result())},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_closure_terminal_is_exactly_replayed_from_all_record_bytes(self) -> None:
        terminal_path = self.root / "CLOSED_INVERSION_REGISTRY.json"
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
            terminal = CLOSURE.derive(self.records, self.root)
            CLOSURE._write_new(terminal_path, terminal)
            self.assertEqual(CLOSURE.verify(terminal_path, self.root), terminal)
            (self.records / "record-00.json").write_text(
                '{"record":"changed"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CLOSURE.ClosureError, "not exactly re-derived"
            ):
                CLOSURE.verify(terminal_path, self.root)

    def test_publication_fsyncs_parent_directory(self) -> None:
        terminal_path = self.root / "CLOSED_INVERSION_REGISTRY.json"
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
            terminal = CLOSURE.derive(self.records, self.root)
        real_fsync = CLOSURE.os.fsync
        calls: list[int] = []

        def recording_fsync(descriptor: int) -> None:
            calls.append(descriptor)
            real_fsync(descriptor)

        with mock.patch.object(CLOSURE.os, "fsync", side_effect=recording_fsync):
            CLOSURE._write_new(terminal_path, terminal)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(terminal_path.is_file())

    def test_parent_fsync_failure_removes_uncommitted_terminal(self) -> None:
        terminal_path = self.root / "CLOSED_INVERSION_REGISTRY.json"
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
            terminal = CLOSURE.derive(self.records, self.root)
        real_fsync = CLOSURE.os.fsync
        calls = 0

        def fail_publication_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected parent fsync failure")
            real_fsync(descriptor)

        with mock.patch.object(
            CLOSURE.os, "fsync", side_effect=fail_publication_fsync
        ):
            with self.assertRaisesRegex(OSError, "injected parent fsync failure"):
                CLOSURE._write_new(terminal_path, terminal)
        self.assertFalse(terminal_path.exists())
        self.assertEqual(list(self.root.glob(".CLOSED_INVERSION_REGISTRY.json.*")), [])

    def test_declared_pass_cannot_override_normative_computed_blocker(self) -> None:
        blocked = pass_result()
        blocked.update(
            state="OPEN",
            blocking_count=1,
            closed_count=12,
            open_count=1,
            blocking_workloads=["python/examples/waveguide_crossing.py"],
            open_workloads=["python/examples/waveguide_crossing.py"],
        )
        validator = type(
            "BlockedValidator",
            (),
            {"validate_registry": staticmethod(lambda **_kwargs: blocked)},
        )
        with mock.patch.object(CLOSURE, "_validator", return_value=validator):
            with self.assertRaisesRegex(
                CLOSURE.ClosureError, "normative closed registry result differs"
            ):
                CLOSURE.derive(self.records, self.root)

    def test_closure_rejects_record_directory_outside_evidence_root(self) -> None:
        outside = pathlib.Path(self.temporary.name).parent
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
            with self.assertRaisesRegex(CLOSURE.ClosureError, "escapes evidence"):
                CLOSURE.derive(outside, self.root)


if __name__ == "__main__":
    unittest.main()
