from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
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


RECEIPT = load_script(
    "v110_candidate_registry_bridge_receipt",
    ROOT / "scripts/v110-release-gate-receipt.py",
)
CLOSURE = load_script(
    "v110_candidate_registry_bridge_closure",
    ROOT / "scripts/close-v110-inversion-registry.py",
)
CONTRACT = ROOT / "packaging/v110-development-contract.json"
SCHEMA = ROOT / "packaging/v110-performance-inversion-record.schema.json"
INVENTORY = ROOT / "packaging/v110-performance-inversion-inventory.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V110CandidateRegistryBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.evidence = self.root / "evidence"
        self.records = self.evidence / "closed-records"
        self.records.mkdir(parents=True)
        for index in range(13):
            path = self.records / f"record-{index:02d}.json"
            path.write_text(json.dumps({"record": index}) + "\n", encoding="utf-8")
            path.chmod(0o600)
        self.inventory_sha256 = sha256(INVENTORY)
        self.registry_result = {
            "schema_version": 1,
            "state": "PASS",
            "inventory_sha256": self.inventory_sha256,
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
        self.validator = type(
            "ClosedValidator",
            (),
            {"validate_registry": staticmethod(lambda **_kwargs: self.registry_result)},
        )
        self.terminal = self.evidence / "CLOSED_INVERSION_REGISTRY.json"
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
            value = CLOSURE.derive(self.records, self.evidence)
            CLOSURE._write_new(self.terminal, value)
        self.terminal_value = value
        self.terminal_sha256 = sha256(self.terminal)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def replay(self, **overrides):
        arguments = {
            "terminal_path": self.terminal,
            "evidence_root": self.evidence,
            "records_directory": self.records,
            "expected_sha256": self.terminal_sha256,
            "expected_terminal_id": self.terminal_value["terminal_id"],
            "inventory_sha256": self.inventory_sha256,
        }
        arguments.update(overrides)
        with mock.patch.object(CLOSURE, "_validator", return_value=self.validator), mock.patch.object(
            RECEIPT, "_load_module", return_value=CLOSURE
        ):
            return RECEIPT._replay_closed_registry(**arguments)

    def test_exact_terminal_and_explicit_records_directory_replay(self) -> None:
        authority, records = self.replay()
        self.assertEqual(records, self.records)
        self.assertEqual(authority["terminal"], self.terminal_value)
        self.assertEqual(authority["file"]["sha256"], self.terminal_sha256)
        self.assertEqual(
            authority["file"]["bundle_relative_path"],
            "CLOSED_INVERSION_REGISTRY.json",
        )

    def test_terminal_sha_id_and_explicit_directory_are_independent_pins(self) -> None:
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "SHA-256 differs"):
            self.replay(expected_sha256="0" * 64)
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "terminal ID differs"):
            self.replay(expected_terminal_id="0" * 64)
        twin = self.evidence / "byte-identical-but-unsealed-directory"
        twin.mkdir()
        for source in self.records.iterdir():
            (twin / source.name).write_bytes(source.read_bytes())
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "differs from terminal"):
            self.replay(records_directory=twin)

    def test_paths_outside_root_and_symlink_components_fail(self) -> None:
        outside_records = self.root / "outside-records"
        outside_records.mkdir()
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "escaped evidence root"):
            self.replay(records_directory=outside_records)
        outside_terminal = self.root / "outside-terminal.json"
        outside_terminal.write_bytes(self.terminal.read_bytes())
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "escaped evidence root"):
            self.replay(
                terminal_path=outside_terminal,
                expected_sha256=sha256(outside_terminal),
            )
        alias = self.root / "evidence-alias"
        alias.symlink_to(self.evidence, target_is_directory=True)
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "path cannot contain symlinks"):
            self.replay(terminal_path=alias / self.terminal.name)

    def test_record_mutation_during_terminal_replay_fails(self) -> None:
        victim = self.records / "record-00.json"

        class MutatingCloser:
            @staticmethod
            def verify(path, root):
                with mock.patch.object(CLOSURE, "_validator", return_value=self.validator):
                    retained = CLOSURE.verify(path, root)
                victim.write_text('{"record":"changed-after-replay"}\n')
                return retained

        with mock.patch.object(RECEIPT, "_load_module", return_value=MutatingCloser):
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "bytes differ"):
                RECEIPT._replay_closed_registry(
                    terminal_path=self.terminal,
                    evidence_root=self.evidence,
                    records_directory=self.records,
                    expected_sha256=self.terminal_sha256,
                    expected_terminal_id=self.terminal_value["terminal_id"],
                    inventory_sha256=self.inventory_sha256,
                )

    def test_m0_attestation_must_list_terminal_and_every_record(self) -> None:
        authority, _ = self.replay()
        terminal_record = authority["file"]
        attested = {
            (
                str(self.terminal),
                terminal_record["size_bytes"],
                terminal_record["sha256"],
            )
        }
        for record in authority["terminal"]["records"]:
            attested.add(
                (
                    str(self.evidence / record["path"]),
                    record["size_bytes"],
                    record["sha256"],
                )
            )
        RECEIPT._require_closed_registry_attested(authority, self.evidence, attested)
        without_terminal = {
            item for item in attested if item[0] != str(self.terminal)
        }
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "terminal is absent"):
            RECEIPT._require_closed_registry_attested(
                authority, self.evidence, without_terminal
            )
        missing_record = set(attested)
        missing_record.remove(
            next(item for item in missing_record if item[0].endswith("record-00.json"))
        )
        with self.assertRaisesRegex(RECEIPT.ReceiptError, r"record\[0\] is absent"):
            RECEIPT._require_closed_registry_attested(
                authority, self.evidence, missing_record
            )

    def test_embedded_binding_rejects_rehashed_semantic_tampering(self) -> None:
        authority, _ = self.replay()
        RECEIPT._embedded_closed_registry(
            authority, inventory_sha256=self.inventory_sha256
        )
        for mutation, message, recompute_terminal_id in (
            (
                lambda value: value["terminal"].__setitem__("terminal_id", "0" * 64),
                "terminal identity differs",
                False,
            ),
            (
                lambda value: value["terminal"]["records"][0].__setitem__(
                    "path", "different/record-00.json"
                ),
                "record\\[0\\] is invalid",
                True,
            ),
            (
                lambda value: value["terminal"]["registry_result"].__setitem__(
                    "closed_count", 12
                ),
                "registry result differs",
                True,
            ),
        ):
            candidate = copy.deepcopy(authority)
            mutation(candidate)
            if recompute_terminal_id:
                unsigned = dict(candidate["terminal"])
                unsigned.pop("terminal_id")
                candidate["terminal"]["terminal_id"] = RECEIPT._canonical_sha256(
                    unsigned
                )
            with self.assertRaisesRegex(RECEIPT.ReceiptError, message):
                RECEIPT._embedded_closed_registry(
                    candidate, inventory_sha256=self.inventory_sha256
                )

    def test_registry_gate_replay_forwards_exact_nondefault_directory(self) -> None:
        gate = self.root / "registry-gate.json"
        gate.write_text(json.dumps(self.registry_result), encoding="utf-8")
        observed = {}

        class Registry:
            @staticmethod
            def validate_registry(**kwargs):
                observed.update(kwargs)
                return self.registry_result

        with mock.patch.object(RECEIPT, "_load_module", return_value=Registry):
            replayed = RECEIPT._replay_registry_gate(
                gate, self.evidence, self.records
            )
        self.assertEqual(replayed, self.registry_result)
        self.assertEqual(observed["records_directory"], self.records)
        self.assertIsNotNone(observed["records_directory"])
        self.assertTrue(observed["require_closed"])

    def test_full_receipt_replay_derives_paths_from_embedded_binding(self) -> None:
        authority, _ = self.replay()
        candidate = {
            "build_mode": "candidate",
            "source_commit": "c" * 40,
            "inventory_sha256": self.inventory_sha256,
            "m0_attestation_sha256": "a" * 64,
            "closed_inversion_registry": authority,
        }
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return candidate

        dummy = self.root / "dummy"
        with mock.patch.object(RECEIPT, "verify_embedded", return_value=candidate), mock.patch.object(
            RECEIPT, "create", side_effect=create
        ):
            result = RECEIPT.verify(
                receipt_path=dummy,
                contract_path=dummy,
                schema_path=dummy,
                inventory_path=dummy,
                require_candidate=True,
                contract_gate_path=dummy,
                registry_gate_path=dummy,
                attestation_gate_path=dummy,
                candidate_manifest_path=dummy,
                evidence_root=self.evidence,
                source_policy_path=dummy,
                expected_source_policy_sha256="1" * 64,
                source_verifier_path=dummy,
                expected_source_verifier_sha256="2" * 64,
            )
        self.assertIs(result, candidate)
        self.assertEqual(captured["closed_registry_terminal_path"], self.terminal)
        self.assertEqual(captured["closed_registry_records_directory"], self.records)
        self.assertEqual(
            captured["expected_closed_registry_terminal_id"],
            self.terminal_value["terminal_id"],
        )

    def test_builder_contract_is_candidate_only_and_retains_original_attestation(self) -> None:
        builder = ROOT / "scripts/build-conda-package.sh"
        text = builder.read_text(encoding="utf-8")
        for marker in (
            "--candidate-inversion-terminal",
            "--candidate-inversion-terminal-sha256",
            "--candidate-inversion-terminal-id",
            "--candidate-inversion-records-directory",
            '--records-directory "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}"',
            '--verify --terminal "${CANDIDATE_INVERSION_TERMINAL}"',
            '/usr/bin/install -m 0600 "${CANDIDATE_ATTESTATION}" "${V110_ATTESTATION_GATE}"',
            '/usr/bin/cmp -s -- "${CANDIDATE_ATTESTATION}" "${V110_ATTESTATION_GATE}"',
        ):
            self.assertIn(marker, text)
        self.assertNotIn(
            '--expected-sha256 "${CANDIDATE_ATTESTATION_SHA256}" --json \\\n+      >"${V110_ATTESTATION_GATE}"',
            text,
        )
        missing = subprocess.run(
            ["/bin/bash", "-p", str(builder), "--candidate-build", "--output-dir", str(self.root / "out")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("exact closed inversion terminal/records authority", missing.stderr)
        development = subprocess.run(
            [
                "/bin/bash", "-p", str(builder), "--development-build",
                "--candidate-inversion-terminal", str(self.terminal),
                "--output-dir", str(self.root / "out-development"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(development.returncode, 0)
        self.assertIn("development build cannot accept candidate authority", development.stderr)

    def test_development_receipt_is_schema_v5_and_carries_no_closure(self) -> None:
        contract_result = {
            "state": "PASS",
            "release_train": "1.1.0",
            "contract_sha256": sha256(CONTRACT),
            "contract_canonical_sha256": RECEIPT._canonical_sha256(
                json.loads(CONTRACT.read_text(encoding="utf-8"))
            ),
        }
        registry_result = dict(self.registry_result)
        contract_gate = self.root / "contract-gate.json"
        registry_gate = self.root / "registry-gate.json"
        contract_gate.write_text(json.dumps(contract_result), encoding="utf-8")
        registry_gate.write_text(json.dumps(registry_result), encoding="utf-8")
        with mock.patch.object(
            RECEIPT, "_replay_contract_gate", return_value=contract_result
        ):
            receipt = RECEIPT.create(
                mode="development",
                source_commit="c" * 40,
                contract_gate_path=contract_gate,
                registry_gate_path=registry_gate,
            )
        self.assertEqual(receipt["schema_version"], 5)
        self.assertIsNone(receipt["closed_inversion_registry"])
        with mock.patch.object(
            RECEIPT, "_replay_contract_gate", return_value=contract_result
        ):
            with self.assertRaisesRegex(
                RECEIPT.ReceiptError,
                "development build cannot carry a closed inversion terminal",
            ):
                RECEIPT.create(
                    mode="development",
                    source_commit="c" * 40,
                    contract_gate_path=contract_gate,
                    registry_gate_path=registry_gate,
                    closed_registry_terminal_path=self.terminal,
                )


if __name__ == "__main__":
    unittest.main()
