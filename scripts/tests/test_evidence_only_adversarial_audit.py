#!/usr/bin/env python3
"""Contract tests for the self-contained evidence-only audit receipt."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "scripts/python-validation/seal_evidence_only_adversarial_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_evidence_only_adversarial_audit", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
RELEASE_SOURCE = (
    ROOT / "scripts/python-validation/seal_milestone_adversarial_audit.py"
)
RELEASE_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_release_adversarial_audit", RELEASE_SOURCE
)
assert RELEASE_SPEC is not None and RELEASE_SPEC.loader is not None
RELEASE_AUDIT = importlib.util.module_from_spec(RELEASE_SPEC)
sys.modules[RELEASE_SPEC.name] = RELEASE_AUDIT
RELEASE_SPEC.loader.exec_module(RELEASE_AUDIT)

COMMIT = "1" * 40


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def record(payload: bytes, path: str = "fixture") -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def refresh_terminal_member(root: pathlib.Path, role: str, key: str, name: str) -> None:
    payload = (root / AUDIT.ROLE_PATHS[role]).read_bytes()
    terminal_path = root / AUDIT.ROLE_PATHS["paired_terminal"]
    terminal = json.loads(terminal_path.read_text())
    terminal[key] = record(payload, name)
    terminal_path.write_bytes(encoded(terminal))


def refresh_review_anchors(
    receipt: pathlib.Path, roles: tuple[str, ...]
) -> None:
    retained = json.loads(receipt.read_text())
    review_path = receipt.parent / AUDIT.ROLE_PATHS["audit_review"]
    review = review_path.read_text()
    for role in roles:
        old_digest = retained["bundle"]["files"][role]["sha256"]
        new_digest = hashlib.sha256(
            (receipt.parent / AUDIT.ROLE_PATHS[role]).read_bytes()
        ).hexdigest()
        review = review.replace(old_digest, new_digest)
    review_path.write_text(review)


class BundleBuilder:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.payloads: dict[str, bytes] = {}

    def write(self, role: str, payload: bytes) -> dict[str, object]:
        path = self.root / AUDIT.ROLE_PATHS[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.payloads[role] = payload
        return record(payload, AUDIT.ROLE_PATHS[role])

    def json(self, role: str, value: object) -> dict[str, object]:
        return self.write(role, encoded(value))


def finding(finding_id: str, severity: str, status: str) -> dict[str, str]:
    return {
        "id": finding_id,
        "severity": severity,
        "status": status,
        "summary": f"summary {finding_id}",
        "resolution": f"resolution {finding_id}",
        "verification": f"verification {finding_id}",
    }


def finding_values() -> list[dict[str, str]]:
    severities = {
        "M3-HIGH-01": ("high", "open"),
        "M3-MEDIUM-02": ("medium", "open"),
        "M3-MEDIUM-03": ("medium", "open"),
        "M3-HIGH-04": ("high", "open"),
        "M3-HIGH-05": ("high", "open"),
        "M3-HIGH-06": ("high", "open"),
        "M3-MEDIUM-07": ("medium", "open"),
        "M3-LOW-08": ("low", "open"),
        "M3-INFO-09": ("info", "closed"),
        "M3-MEDIUM-10": ("medium", "closed"),
        "M3-LOW-11": ("low", "closed"),
        "M3-INFO-12": ("info", "open"),
    }
    return [
        finding(finding_id, *severities[finding_id])
        for finding_id in AUDIT.EXPECTED_FINDING_IDS
    ]


def contract() -> dict[str, object]:
    repository = "/historical/repository"
    return {
        "build_receipt": {"git_head": COMMIT},
        "paths": {
            "repo": repository,
            "manifest": f"{repository}/scripts/python-validation/manifest.json",
            "build_python": f"{repository}/build/python",
            "install_prefix": f"{repository}/install/prefix",
            "python": f"{repository}/environment/bin/python",
            "runner": f"{repository}/scripts/python-validation/run_validation.py",
        },
    }


def task_identity(index: int = 56) -> dict[str, object]:
    special = index == 56
    return {
        "backends": ["cpu", "cuda"],
        "case_path": AUDIT.CASE_PATH if special else f"python/examples/fixture-{index:03d}.py",
        "covered_example": None,
        "index": index,
        "plan_launch_order": index,
        "purpose": f"fixture purpose {index}",
        "release_role": "fixture",
        "section": "fixture",
        "unit_id": AUDIT.UNIT_ID if special else f"fixture-{index:03d}",
    }


def make_bundle(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(mode=0o700)
    builder = BundleBuilder(root)
    builder.write("control_verifier", SOURCE.read_bytes())
    manifest = {
        "overrides": [
            {
                "glob": AUDIT.CASE_PATH,
                "comparison": {
                    "mode": "json_metrics",
                    "prefix": "fixture-metrics:",
                    "tolerances": {
                        AUDIT.RULE: {"rtol": 0.01, "atol": 0.00001},
                        "fixture.*": {"rtol": 0.0, "atol": 0.0},
                    }
                },
            }
        ]
    }
    manifest_record = builder.json("source_manifest", manifest)
    comparator_record = builder.write(
        "source_validation_comparator", b"# fixed comparator\n"
    )
    wrapper_record = builder.write(
        "source_reciprocity_wrapper", b"# fixed wrapper\n"
    )
    cpu_metrics = {
        "example_results": {
            "reciprocity_forward_normalized_convergence_residual": {
                "real": [-0.0052]
            }
        },
        "fixture": {"values": [0.0] * 5903},
    }
    cuda_metrics = copy.deepcopy(cpu_metrics)
    cuda_metrics["example_results"][
        "reciprocity_forward_normalized_convergence_residual"
    ]["real"][0] = -0.0053
    cpu_record = builder.write(
        "task56_cpu_stdout",
        b"fixture-metrics:"
        + json.dumps(cpu_metrics, separators=(",", ":"), sort_keys=True).encode()
        + b"\n",
    )
    cuda_record = builder.write(
        "task56_cuda_stdout",
        b"fixture-metrics:"
        + json.dumps(cuda_metrics, separators=(",", ":"), sort_keys=True).encode()
        + b"\n",
    )
    _old_rule, source_comparison, changed_comparison = AUDIT._task56_comparison(
        manifest
    )
    source_recomparison = AUDIT._recompare_metrics(
        builder.payloads["task56_cpu_stdout"],
        builder.payloads["task56_cuda_stdout"],
        source_comparison,
    )
    recomparison = AUDIT._recompare_metrics(
        builder.payloads["task56_cpu_stdout"],
        builder.payloads["task56_cuda_stdout"],
        changed_comparison,
    )
    process_log_record = builder.write(
        "task56_original_process_log", b"comparison mismatch\n"
    )
    failure = source_recomparison["failures"][0]
    original_report = {
        "schema_version": 4,
        "exit_code": 3,
        "summary": {"MISMATCH": 1, "NOT_SELECTED": 209},
        "configuration": {
            "case_ids": [AUDIT.CASE_PATH],
            "backends": ["cpu", "cuda"],
            "tiers": ["full"],
        },
        "results": [
            {
                "selected": True,
                "path": AUDIT.CASE_PATH,
                "outcome": "MISMATCH",
                "comparison": {
                    "outcome": "MISMATCH",
                    "failure_count": 1,
                    "metric_evidence": source_recomparison[
                        "metric_evidence"
                    ],
                    "failures": [failure],
                },
                "runs": {
                    "cpu": {
                        "outcome": "PASS",
                        "stdout_log": "runs/cpu/stdout.log",
                        "evidence_files": {
                            "stdout": {
                                "available": True,
                                **cpu_record,
                            }
                        },
                    },
                    "cuda": {
                        "outcome": "PASS_STRICT_CUDA",
                        "stdout_log": "runs/cuda/stdout.log",
                        "evidence_files": {
                            "stdout": {
                                "available": True,
                                **cuda_record,
                            }
                        },
                    },
                },
            }
        ],
    }
    original_report_record = builder.json(
        "task56_runner_report", original_report
    )
    original_journal = {
        "events": [
            {
                "state": "task-process-ended",
                "task": task_identity(),
                "process": {
                    "returncode": 3,
                    "timed_out": False,
                    "output_limited": False,
                    "command": [
                        "/historical/repository/environment/bin/python",
                        "/historical/repository/scripts/python-validation/run_validation.py",
                        "--repo",
                        "/historical/repository",
                        "--manifest",
                        "/historical/repository/scripts/python-validation/manifest.json",
                        "--build-python",
                        "/historical/repository/build/python",
                        "--install-prefix",
                        "/historical/repository/install/prefix",
                        "--python",
                        "/historical/repository/environment/bin/python",
                        "--output",
                        "/historical/evidence/raw/056-production-mixed-07",
                        "--backend",
                        "both",
                        "--acknowledge-backend-mismatch-reason",
                        AUDIT.TASK56_ACKNOWLEDGEMENT,
                        "--tier",
                        "full",
                        "--case",
                        AUDIT.CASE_PATH,
                    ],
                    "log": process_log_record,
                },
            }
        ]
    }
    builder.json("task56_original_attempt_journal", original_journal)

    adjudication = {
        "schema": "gpmeep-m3-task56-reciprocity-adjudication-v1",
        "outcome": "PASS",
        "candidate": {
            "commit": COMMIT,
            "manifest_sha256": manifest_record["sha256"],
        },
        "source_evidence": {
            "report_sha256": original_report_record["sha256"],
            "cpu_stdout": {"sha256": cpu_record["sha256"]},
            "cuda_stdout": {"sha256": cuda_record["sha256"]},
            "original_failure": failure,
            "backend_outcomes": {"cpu": "PASS", "cuda": "PASS"},
        },
        "adjudication": {
            "changed_rule_count": 1,
            "rule": AUDIT.RULE,
            "old_rule": {"rtol": 0.01, "atol": 0.00001},
            "new_rule": {"rtol": 0.02, "atol": 0.00001},
            "comparison": {
                "outcome": "PASS",
                "failure_count": 0,
                "metric_evidence": recomparison["metric_evidence"],
            },
        },
        "release_semantics": {
            "candidate_manifest_fix_required_for_release": True,
            "candidate_source_modified": False,
        },
    }
    adjudication_record = builder.json("task56_adjudication", adjudication)
    adjudication_script_record = builder.write(
        "task56_adjudication_script", b"# fixed adjudicator\n"
    )
    independent_md_record = builder.write(
        "task56_independent_review_md", b"GO for M3; BLOCKED for release\n"
    )
    independent = {
        "schema": "gpmeep-m3-task56-independent-adjudication-review-v1",
        "candidate": {"commit": COMMIT},
        "verdict": {"m3_resume": "GO", "candidate_release": "BLOCKED"},
        "full_recomparison": {
            "outcome": "PASS",
            "metric_count": 5904,
            "verdict_failure_count": 0,
            "tolerance_group_count": recomparison["tolerance_group_count"],
            "arithmetic_overflow_count": recomparison[
                "arithmetic_overflow_count"
            ],
            "zero_tolerance_failure_count": recomparison[
                "zero_tolerance_failure_count"
            ],
            "diagnostic_only_metric_count": recomparison[
                "diagnostic_only_metric_count"
            ],
            "diagnostic_only_out_of_tolerance_count": recomparison[
                "diagnostic_only_out_of_tolerance_count"
            ],
            "comparison_records_sha256": recomparison["metric_evidence"][
                "comparison_records_sha256"
            ],
        },
        "evidence_sha256": {
            "original_report": original_report_record["sha256"],
            "cpu_stdout": cpu_record["sha256"],
            "cuda_stdout": cuda_record["sha256"],
            "candidate_manifest": manifest_record["sha256"],
            "validation_comparator": comparator_record["sha256"],
            "reciprocity_wrapper": wrapper_record["sha256"],
            "adjudication_script": adjudication_script_record["sha256"],
            "adjudication_receipt": adjudication_record["sha256"],
            "markdown_review": independent_md_record["sha256"],
        },
    }
    independent_record = builder.json(
        "task56_independent_review_json", independent
    )
    schema_review = {
        "current_strategy_verdict": {
            "m3_resume": "NO_GO",
            "candidate_release": "BLOCKED",
        },
        "narrow_derive_archive_override_alternative": {
            "m3_characterization": "CONDITIONAL_GO",
            "candidate_release": "BLOCKED",
        },
        "original_evidence": {
            "report_sha256": original_report_record["sha256"],
            "runner_exit_code": 3,
            "selected_outcome": "MISMATCH",
        },
    }
    schema_review_record = builder.json(
        "task56_schema_review_json", schema_review
    )
    schema_review_md_record = builder.write(
        "task56_schema_review_md", b"conditional GO; release BLOCKED\n"
    )
    replay = {
        "adjudicated_evidence_replay": {
            "schema": "gpmeep-m3-task56-telemetry-replay-v1",
            "archived_attempt": "resume-history/attempt-0006",
            "original_report_sha256": original_report_record["sha256"],
            "adjudication_receipt_sha256": adjudication_record["sha256"],
            "independent_review_sha256": independent_record["sha256"],
            "schema_review_sha256": schema_review_record["sha256"],
            "new_meep_execution": False,
            "candidate_manifest_fix_required_for_release": True,
            "independent_review_verdict": {
                "m3_resume": "GO",
                "candidate_release": "BLOCKED",
            },
            "self_contained_adjudication": {
                "ADJUDICATION.json": adjudication_record,
                "adjudicate_m3_task56_reciprocity.py": (
                    adjudication_script_record
                ),
                "INDEPENDENT_REVIEW.json": independent_record,
                "INDEPENDENT_REVIEW.md": independent_md_record,
                "SCHEMA_REVIEW.json": schema_review_record,
                "SCHEMA_REVIEW.md": schema_review_md_record,
            },
        }
    }
    replay_record = builder.json("task56_replay", replay)

    task = {
        "task": task_identity(),
        "process": {"returncode": 0, "fixture": 56},
        "archive": {
            "archive_complete": record(
                b"archive 56", "archives/056/ARCHIVE_COMPLETE"
            ),
            "report_sha256": original_report_record["sha256"],
            "runner_report": original_report_record,
        },
        "telemetry_replay": {
            "sha256": replay_record["sha256"],
            "size_bytes": replay_record["size_bytes"],
        },
    }
    tasks = [
        {
            "task": task_identity(index),
            "process": {"returncode": 0, "fixture": index},
            "archive": {
                "archive_complete": record(
                    f"archive {index}".encode(),
                    f"archives/{index:03d}/ARCHIVE_COMPLETE",
                )
            },
        }
        for index in range(1, 71)
    ]
    tasks[55] = copy.deepcopy(task)
    paired_report = {
        "schema": "gpmeep-m3-paired-matrix-v1",
        "outcome": "PASS",
        "contract": contract(),
        "tasks": tasks,
    }
    paired_checkpoint = {
        "schema": "gpmeep-m3-paired-checkpoint-v1",
        "contract": contract(),
        "completed_tasks": copy.deepcopy(tasks),
    }
    paired_journal = {
        "schema": "gpmeep-m3-paired-journal-v1",
        "contract": contract(),
        "events": [
            {"state": "controller-started"},
            *[
                {"state": "task-started", "task": copy.deepcopy(item["task"])}
                for item in tasks
            ],
            *[
                {
                    "state": "task-process-ended",
                    "task": copy.deepcopy(item["task"]),
                    "process": copy.deepcopy(item["process"]),
                }
                for item in tasks
            ],
            *[
                {
                    "state": "task-validated",
                    "task": copy.deepcopy(item["task"]),
                    "archive_complete": copy.deepcopy(
                        item["archive"]["archive_complete"]
                    ),
                }
                for item in tasks
            ],
        ],
    }
    report_record = builder.json("paired_report", paired_report)
    checkpoint_record = builder.json("paired_checkpoint", paired_checkpoint)
    journal_record = builder.json("paired_journal", paired_journal)
    terminal_record = builder.json(
        "paired_terminal",
        {
            "schema": "gpmeep-m3-paired-complete-v1",
            "outcome": "PASS",
            "checkpoint": {
                **checkpoint_record,
                "path": "CHECKPOINT.json",
            },
            "journal": {**journal_record, "path": "JOURNAL.json"},
            "report": {**report_record, "path": "report.json"},
        },
    )
    findings = finding_values()
    draft = {
        "schema": AUDIT.DRAFT_SCHEMA,
        "milestone": "M3",
        "auditor": {"agent_id": "fixture-agent", "task_name": "/fixture"},
        "source": {"repo": "/historical/repository", "commit": COMMIT},
        "evidence": {"root": "/historical/evidence", "terminal_name": "PAIRED_COMPLETE"},
        "scope": ["M3 paired characterization", "No new Meep execution"],
        "findings": findings,
        "review": "GO for M3 evidence; candidate release BLOCKED",
    }
    draft_record = builder.json("audit_draft", draft)
    review = (
        "Reviewer agent: `fixture-agent`\n"
        "Reviewer task: `/fixture`\n"
        "Candidate: `/historical/repository`\n"
        f"Candidate commit: `{COMMIT}`\n"
        "Evidence root: `/historical/evidence`\n"
        "Terminal: `PAIRED_COMPLETE`\n"
        f"Draft SHA-256 `{draft_record['sha256']}`\n"
        f"| `PAIRED_COMPLETE` | `{terminal_record['sha256']}` |\n"
        f"| `CHECKPOINT.json` | `{checkpoint_record['sha256']}` |\n"
        f"| `JOURNAL.json` | `{journal_record['sha256']}` |\n"
        f"| paired terminal report | `{report_record['sha256']}` |\n"
        "GO for M3 evidence\nBLOCKED for release\n"
        + "\n".join(AUDIT.EXPECTED_FINDING_IDS)
        + "\n"
    ).encode("utf-8")
    builder.write("audit_review", review)
    receipt = AUDIT.derive(root)
    receipt_path = root / AUDIT.RECEIPT_NAME
    receipt_path.write_bytes(encoded(receipt))
    return receipt_path


class EvidenceOnlyAuditTests(unittest.TestCase):
    def bundle(self) -> tuple[tempfile.TemporaryDirectory, pathlib.Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name) / "bundle"
        return temporary, make_bundle(root)

    def test_valid_receipt_is_dual_verdict_and_never_release_authority(self) -> None:
        _temporary, receipt = self.bundle()
        value = AUDIT.verify(receipt)
        self.assertEqual(value["verdicts"]["paired_evidence"], "GO")
        self.assertEqual(value["verdicts"]["candidate_release"], "BLOCKED")
        self.assertFalse(value["verdicts"]["release_authorized"])
        self.assertEqual(value["task56"]["native_runner_outcome"], "MISMATCH")
        self.assertFalse(value["task56"]["adjudication"]["new_meep_execution"])
        self.assertEqual(value["bundle"]["file_count"], len(AUDIT.ROLE_PATHS))

    def test_evidence_only_receipt_is_rejected_by_release_audit_verifier(self) -> None:
        _temporary, receipt = self.bundle()
        with self.assertRaises(RELEASE_AUDIT.AuditError):
            RELEASE_AUDIT.verify(receipt)

    def test_bundle_replays_after_relocation_without_originals(self) -> None:
        temporary, receipt = self.bundle()
        relocated = pathlib.Path(temporary.name) / "relocated"
        shutil.copytree(receipt.parent, relocated)
        shutil.rmtree(receipt.parent)
        value = AUDIT.verify(relocated / AUDIT.RECEIPT_NAME)
        self.assertTrue(value["bundle"]["portable_relative_paths"])
        self.assertFalse(value["bundle"]["external_files_required_for_verify"])

    def test_file_change_is_rejected(self) -> None:
        _temporary, receipt = self.bundle()
        report = receipt.parent / AUDIT.ROLE_PATHS["task56_runner_report"]
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaises(AUDIT.EvidenceOnlyError):
            AUDIT.verify(receipt)

    def test_verdict_flip_is_rejected(self) -> None:
        _temporary, receipt = self.bundle()
        value = json.loads(receipt.read_text())
        value["verdicts"]["candidate_release"] = "GO"
        value["verdicts"]["release_authorized"] = True
        receipt.write_bytes(encoded(value))
        with self.assertRaisesRegex(
            AUDIT.EvidenceOnlyError, "not exactly re-derived"
        ):
            AUDIT.verify(receipt)

    def test_finding_omission_reordering_and_status_change_are_rejected(self) -> None:
        mutations = (
            lambda values: values.pop(),
            lambda values: values.__setitem__(
                slice(0, 2), [values[1], values[0]]
            ),
            lambda values: values[0].__setitem__("status", "closed"),
            lambda values: values[0].__setitem__("severity", "low"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                _temporary, receipt = self.bundle()
                draft_path = receipt.parent / AUDIT.ROLE_PATHS["audit_draft"]
                draft = json.loads(draft_path.read_text())
                mutate(draft["findings"])
                draft_path.write_bytes(encoded(draft))
                with self.assertRaises(AUDIT.EvidenceOnlyError):
                    AUDIT.derive(receipt.parent)

    def test_path_escape_and_unknown_file_are_rejected(self) -> None:
        _temporary, receipt = self.bundle()
        value = json.loads(receipt.read_text())
        value["bundle"]["files"]["paired_report"]["path"] = "../report.json"
        receipt.write_bytes(encoded(value))
        with self.assertRaises(AUDIT.EvidenceOnlyError):
            AUDIT.verify(receipt)

        _temporary, receipt = self.bundle()
        (receipt.parent / "unexpected").write_text("unexpected\n")
        with self.assertRaisesRegex(AUDIT.EvidenceOnlyError, "inventory"):
            AUDIT.verify(receipt)

    def test_symlink_substitution_is_rejected(self) -> None:
        _temporary, receipt = self.bundle()
        target = receipt.parent / AUDIT.ROLE_PATHS["source_manifest"]
        saved = receipt.parent.parent / "saved-manifest"
        saved.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(saved)
        with self.assertRaises(AUDIT.EvidenceOnlyError):
            AUDIT.verify(receipt)

    def test_external_hardlink_substitution_is_rejected(self) -> None:
        _temporary, receipt = self.bundle()
        target = receipt.parent / AUDIT.ROLE_PATHS["source_manifest"]
        saved = receipt.parent.parent / "saved-hardlink-manifest"
        saved.write_bytes(target.read_bytes())
        target.unlink()
        target.hardlink_to(saved)
        with self.assertRaisesRegex(AUDIT.EvidenceOnlyError, "regular file"):
            AUDIT.verify(receipt)

    def test_native_mismatch_and_new_execution_flags_cannot_be_relabelled(self) -> None:
        _temporary, receipt = self.bundle()
        report_path = receipt.parent / AUDIT.ROLE_PATHS["task56_runner_report"]
        report = json.loads(report_path.read_text())
        report["exit_code"] = 0
        report["results"][0]["outcome"] = "PASS"
        report_path.write_bytes(encoded(report))
        with self.assertRaises(AUDIT.EvidenceOnlyError):
            AUDIT.derive(receipt.parent)

        _temporary, receipt = self.bundle()
        replay_path = receipt.parent / AUDIT.ROLE_PATHS["task56_replay"]
        replay = json.loads(replay_path.read_text())
        replay["adjudicated_evidence_replay"]["new_meep_execution"] = True
        replay_path.write_bytes(encoded(replay))
        with self.assertRaises(AUDIT.EvidenceOnlyError):
            AUDIT.derive(receipt.parent)

    def test_full_paired_vectors_and_journal_are_cross_checked(self) -> None:
        _temporary, receipt = self.bundle()
        report_path = receipt.parent / AUDIT.ROLE_PATHS["paired_report"]
        report = json.loads(report_path.read_text())
        report["tasks"][0]["task"]["unit_id"] = "contradict-checkpoint"
        report_path.write_bytes(encoded(report))
        refresh_terminal_member(
            receipt.parent, "paired_report", "report", "report.json"
        )
        refresh_review_anchors(
            receipt, ("paired_report", "paired_terminal")
        )
        with self.assertRaisesRegex(
            AUDIT.EvidenceOnlyError, "report/checkpoint"
        ):
            AUDIT.derive(receipt.parent)

        _temporary, receipt = self.bundle()
        journal_path = receipt.parent / AUDIT.ROLE_PATHS["paired_journal"]
        journal = json.loads(journal_path.read_text())
        ended = next(
            event
            for event in journal["events"]
            if event.get("state") == "task-process-ended"
        )
        ended["process"]["returncode"] = 9
        journal_path.write_bytes(encoded(journal))
        refresh_terminal_member(
            receipt.parent, "paired_journal", "journal", "JOURNAL.json"
        )
        refresh_review_anchors(
            receipt, ("paired_journal", "paired_terminal")
        )
        with self.assertRaisesRegex(AUDIT.EvidenceOnlyError, "journal task"):
            AUDIT.derive(receipt.parent)

        _temporary, receipt = self.bundle()
        for role, array_name in (
            ("paired_report", "tasks"),
            ("paired_checkpoint", "completed_tasks"),
        ):
            path = receipt.parent / AUDIT.ROLE_PATHS[role]
            value = json.loads(path.read_text())
            value[array_name][0]["task"]["purpose"] = "coherent rewrite"
            path.write_bytes(encoded(value))
        journal_path = receipt.parent / AUDIT.ROLE_PATHS["paired_journal"]
        journal = json.loads(journal_path.read_text())
        for event in journal["events"]:
            if event.get("task", {}).get("index") == 1:
                event["task"]["purpose"] = "coherent rewrite"
        journal_path.write_bytes(encoded(journal))
        for role, key, name in (
            ("paired_report", "report", "report.json"),
            ("paired_checkpoint", "checkpoint", "CHECKPOINT.json"),
            ("paired_journal", "journal", "JOURNAL.json"),
        ):
            refresh_terminal_member(receipt.parent, role, key, name)
        with self.assertRaisesRegex(AUDIT.EvidenceOnlyError, "bind the draft"):
            AUDIT.derive(receipt.parent)

    def test_audit_review_and_original_argv_are_cross_bound(self) -> None:
        _temporary, receipt = self.bundle()
        draft_path = receipt.parent / AUDIT.ROLE_PATHS["audit_draft"]
        draft = json.loads(draft_path.read_text())
        draft["evidence"]["root"] = "/different/evidence"
        draft_path.write_bytes(encoded(draft))
        with self.assertRaisesRegex(AUDIT.EvidenceOnlyError, "bind the draft"):
            AUDIT.derive(receipt.parent)

        _temporary, receipt = self.bundle()
        journal_path = (
            receipt.parent
            / AUDIT.ROLE_PATHS["task56_original_attempt_journal"]
        )
        journal = json.loads(journal_path.read_text())
        command = journal["events"][0]["process"]["command"]
        command[command.index("--manifest") + 1] = "/different/manifest.json"
        journal_path.write_bytes(encoded(journal))
        with self.assertRaisesRegex(
            AUDIT.EvidenceOnlyError, "execution semantics"
        ):
            AUDIT.derive(receipt.parent)

    def test_metric_recomparison_is_computed_from_bundled_stdout(self) -> None:
        _temporary, receipt = self.bundle()
        value = AUDIT.verify(receipt)
        adjudication = value["task56"]["adjudication"]
        self.assertTrue(adjudication["independently_recomputed"])
        self.assertEqual(adjudication["metric_evidence"]["count"], 5904)
        manifest = json.loads(
            (receipt.parent / AUDIT.ROLE_PATHS["source_manifest"]).read_text()
        )
        _old, _source_comparison, comparison = AUDIT._task56_comparison(
            manifest
        )
        cpu = (
            receipt.parent / AUDIT.ROLE_PATHS["task56_cpu_stdout"]
        ).read_bytes()
        cuda = (
            receipt.parent / AUDIT.ROLE_PATHS["task56_cuda_stdout"]
        ).read_bytes()
        changed = cuda.replace(b"-0.0053", b"-0.5", 1)
        recomputed = AUDIT._recompare_metrics(cpu, changed, comparison)
        self.assertEqual(recomputed["outcome"], "MISMATCH")
        self.assertNotEqual(
            recomputed["metric_evidence"]["comparison_records_sha256"],
            adjudication["metric_evidence"]["comparison_records_sha256"],
        )

if __name__ == "__main__":
    unittest.main()
