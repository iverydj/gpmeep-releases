"""Run the builder's contract branch without starting closure replay or a build.

The producer and receipt contract replay are real. Successful full-mode tests
copy the small, checksum-bound external inputs into a temporary workspace;
they skip explicitly when those external inputs are unavailable. Missing-input
and development-mode checks need only the source checkout.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-conda-package.sh"
CONTRACT = ROOT / "packaging/v110-development-contract.json"
VALIDATOR = ROOT / "scripts/validate-v110-development-contract.py"
SPEC = importlib.util.spec_from_file_location(
    "candidate_contract_gate_receipt_test", ROOT / "scripts/v110-release-gate-receipt.py"
)
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def contract_shell_branch() -> str:
    """Keep the real initialization and both mode branches up to their next gate."""
    text = BUILDER.read_text(encoding="utf-8")
    start = text.index('GPMEEP_V110_CANDIDATE_RECEIPT=""\n')
    candidate_end = text.index('    CLOSURE_RESULT="$(', start)
    development_start = text.index("\n  else\n", candidate_end)
    development_end = text.index(
        '    "${V110_GATE_PYTHON}" -I "${V110_REGISTRY_VALIDATOR}" --allow-open',
        development_start,
    )
    # A marker replaces each subsequent expensive stage, not the contract CLI.
    next_stage = '    printf "reached\\n" >"${NEXT_STAGE}"\n'
    return (
        text[:text.index('if [[ "$-"')]
        + text[start:candidate_end] + next_stage
        + text[development_start:development_end] + next_stage + "  fi\nfi\n"
    )


class CandidateContractGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.repo = self.workspace / "repo"
        (self.repo / "packaging").mkdir(parents=True)
        shutil.copyfile(CONTRACT, self.repo / "packaging" / CONTRACT.name)
        self.output = self.workspace / "output"
        self.output.mkdir()
        self.gates = self.output / ("v110-gates-" + "a" * 40)
        self.next_stage = self.workspace / "next-stage"

    def copy_external_inputs(self) -> tuple[list[Path], list[Path]]:
        contract = json.loads(CONTRACT.read_bytes())
        inventory = json.loads((ROOT / contract["performance_inversion_policy"]["registry"]
                                ["initial_inventory"]).read_bytes())
        user = [item["workspace_relative_path"]
                for item in contract["validation"]["user_input_artifacts"]]
        campaign = [inventory["source_campaign"][name]["workspace_relative_path"]
                    for name in ("summary_evidence", "controller_evidence")]
        for relative in user + campaign:
            source = ROOT.parent / relative
            if not source.is_file():
                self.skipTest("checksum-bound external contract inputs are unavailable")
            destination = self.workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return ([self.workspace / path for path in user],
                [self.workspace / path for path in campaign])

    def run_branch(self, mode: str = "candidate") -> subprocess.CompletedProcess:
        assignments = {
            "REPO_ROOT": str(self.repo), "BUILD_MODE": mode,
            "OUTPUT_DIRECTORY": str(self.output), "SOURCE_COMMIT": "a" * 40,
            "V110_GATE_PYTHON": str(Path(sys.executable).resolve(strict=True)),
            "V110_CONTRACT_VALIDATOR": str(VALIDATOR),
            "NEXT_STAGE": str(self.next_stage),
        }
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(self.workspace),
                       "TMPDIR": str(self.workspace)}
        # Variables are supplied before the extracted branch; its original
        # set -euo pipefail still determines whether a failing CLI may continue.
        script = "\n".join(key + "=" + shlex.quote(value)
                           for key, value in assignments.items()) + "\n" + contract_shell_branch()
        return subprocess.run(["/bin/bash", "-p", "-c", script], cwd=ROOT,
                              env=environment, capture_output=True, text=True, timeout=30)

    def assert_full_validation_stopped(self, result: subprocess.CompletedProcess):
        self.assertEqual(result.returncode, 2, result.stderr)
        full = json.loads((self.gates / "contract-full-inputs.json").read_bytes())
        self.assertEqual(full["state"], "FAIL")
        self.assertEqual((self.gates / "contract.json").read_bytes(), b"")
        self.assertFalse(self.next_stage.exists())
        return full

    def test_candidate_retains_full_validation_and_emits_replayable_contract_gate(self):
        self.copy_external_inputs()
        result = self.run_branch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.next_stage.is_file())
        full_path = self.gates / "contract-full-inputs.json"
        gate_path = self.gates / "contract.json"
        replayed = RECEIPT._replay_contract_gate(gate_path)
        self.assertEqual((replayed["user_inputs_checked"],
                          replayed["campaign_inputs_checked"]), (0, 0))
        full = json.loads(full_path.read_bytes())
        self.assertEqual((full["state"], full["user_inputs_checked"],
                          full["campaign_inputs_checked"]), ("PASS", 2, 2))
        for path in (full_path, gate_path):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        # Reproduce the old producer/consumer incompatibility without weakening
        # the receipt's exact comparison or manufacturing either JSON result.
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "not exactly re-derived"):
            RECEIPT._replay_contract_gate(full_path)

    def test_candidate_missing_external_input_stops_before_next_gate(self):
        self.assert_full_validation_stopped(self.run_branch())

    def test_candidate_changed_user_input_stops_before_next_gate(self):
        user, _ = self.copy_external_inputs()
        user[0].write_bytes(user[0].read_bytes() + b"\n# changed fixture\n")
        full = self.assert_full_validation_stopped(self.run_branch())
        self.assertIn("user input hash mismatch", full["error"])

    def test_candidate_changed_campaign_input_stops_before_next_gate(self):
        _, campaign = self.copy_external_inputs()
        campaign[0].write_bytes(campaign[0].read_bytes() + b"\n")
        full = self.assert_full_validation_stopped(self.run_branch())
        self.assertIn("source campaign evidence hash differs", full["error"])

    def test_development_contract_gate_needs_no_external_inputs(self):
        result = self.run_branch("development")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.next_stage.is_file())
        self.assertFalse((self.gates / "contract-full-inputs.json").exists())
        replayed = RECEIPT._replay_contract_gate(self.gates / "contract.json")
        self.assertEqual((replayed["user_inputs_checked"],
                          replayed["campaign_inputs_checked"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
