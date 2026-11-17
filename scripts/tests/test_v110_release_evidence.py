#!/usr/bin/env python3
"""Production release-evidence compiler tests."""

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
import types
import unittest
from unittest import mock

from scripts.tests import test_gpmeep_self_check as SELF_CHECK_FIXTURE
from scripts.tests import test_v110_final_release_receipt as FINAL_FIXTURE

_write_conda_v2 = FINAL_FIXTURE._write_conda_v2
write_fixture_json = FINAL_FIXTURE._write_json


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVIDENCE = load(
    "gpmeep_v110_release_evidence_test",
    ROOT / "scripts/v110-release-evidence.py",
)
CANDIDATE = load(
    "gpmeep_v110_candidate_evidence_test",
    ROOT / "scripts/v110-candidate-evidence.py",
)
FINAL = load(
    "gpmeep_v110_final_release_evidence_test",
    ROOT / "scripts/v110-final-release-receipt.py",
)


def write_json(path: pathlib.Path, value: dict[str, object], mode: int = 0o600) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


def synthetic_m4_report() -> dict[str, object]:
    cases = {
        f"python/examples/case-{index:02d}.py": {
            "lanes": {
                "upstream_meep_cpu": {"mean_seconds": 10.0 + index},
                "gpmeep_v110_cpu": {"mean_seconds": 11.0 + index},
                "gpmeep_v103_cuda_gpu1": {"mean_seconds": 5.0 + index},
                "gpmeep_v110_cuda_gpu1": {"mean_seconds": 4.5 + index},
                "gpmeep_v103_cuda_gpu2": {"mean_seconds": 4.0 + index},
                "gpmeep_v110_cuda_gpu2": {"mean_seconds": 3.5 + index},
            },
            "ratios": {
                "v103-gpu1_to_v110-gpu1": {
                    "ratio_of_average_seconds": 1.1 + index / 100.0
                },
                "v103-gpu2_to_v110-gpu2": {
                    "ratio_of_average_seconds": 1.05 + index / 100.0
                },
            },
        }
        for index in range(13)
    }
    comparisons = [
        {
            "cycle_index": 1 if index < 94 else 0,
            "sample_kind": "measured" if index < 94 else "warmup",
            "case_path": f"python/examples/case-{index % 13:02d}.py",
            "reference_lane": "upstream_meep_cpu",
            "candidate_lane": "gpmeep_v110_cuda_gpu1",
            "method": "fixture",
            "result": {"outcome": "PASS", "failure_count": 0},
        }
        for index in range(141)
    ]
    return {
        "performance": {
            "raw_seconds": [
                {
                    "task_id": f"task-{index:03d}",
                    "lane": tuple(FINAL.PERFORMANCE_LANES)[index % 6],
                    "seconds": 1.0 + index / 10.0,
                }
                for index in range(180)
            ],
            "cases": cases,
        },
        "numerical": {
            "outcome": "PASS",
            "comparison_count": 141,
            "comparisons": comparisons,
        },
    }


class DurableWriterTests(unittest.TestCase):
    def test_mode_no_overwrite_and_semantic_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            output = root / "terminal.json"
            value = {"schema": "fixture-v1", "state": "PASS"}
            EVIDENCE._write_new_json(output, value, root=root)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            first = output.read_bytes()
            with self.assertRaisesRegex(Exception, "refusing to overwrite"):
                EVIDENCE._write_new_json(output, {"changed": True}, root=root)
            self.assertEqual(output.read_bytes(), first)
            failing = root / "failing.json"
            with self.assertRaisesRegex(Exception, "semantic failure"):
                EVIDENCE._write_new_json(
                    failing,
                    value,
                    root=root,
                    verify=lambda _path: (_ for _ in ()).throw(
                        RuntimeError("semantic failure")
                    ),
                )
            self.assertFalse(failing.exists())

    def test_symlinked_input_and_parent_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            actual = root / "actual.json"
            actual.write_text("{}\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(actual)
            with self.assertRaisesRegex(Exception, "symlink"):
                EVIDENCE._direct_file(link, "fixture")
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(Exception, "symlink"):
                EVIDENCE._write_new_json(linked_parent / "out.json", {}, root=root)

    def test_cli_requires_isolated_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/v110-release-evidence.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        # argparse help exits before the runtime gate; an actual operation must fail.
        operation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/v110-release-evidence.py"),
                "reports",
                "--m4-output", "/missing",
                "--evidence-root", "/missing",
                "--markdown-output", "/missing/report.md",
                "--html-output", "/missing/report.html",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(operation.returncode, 2)
        self.assertIn("isolated mode", operation.stderr)


class NativeCandidateRoleReferenceTests(unittest.TestCase):
    """Exercise the real compiler/gate boundary after native producer replay."""

    def check_role(self, root, role, count, *, returned_count=None, error=None):
        gate = EVIDENCE._candidate_gate_module()
        terminal_name = "HOST_COMPLETE" if role == "host_four" else "SPECIALIZED_COMPLETE"
        terminal = root / terminal_name
        payload = {
            "schema": gate.ROLE_SCHEMAS[role], "outcome": "PASS",
            "report": {"path": "report.json", "size_bytes": 123, "sha256": "a" * 64},
            "journal": {"path": "JOURNAL.json", "size_bytes": 123, "sha256": "b" * 64},
            "checkpoint": {"path": "CHECKPOINT.json", "size_bytes": 123, "sha256": "c" * 64},
        }
        write_json(terminal, payload)
        # The native producers return a matrix report, with tasks rather than
        # cases/results. Their full workload replay is covered by their suites.
        report = {
            "schema": "native-matrix-fixture", "outcome": "PASS",
            "contract": {}, "tasks": [{} for _ in range(count if returned_count is None else returned_count)],
            "hardware": {}, "capacity_initial": {}, "capacity_final": {},
            "resume_history": [], "checkpoint": payload["checkpoint"],
        }
        reference = EVIDENCE.artifact_reference(root, role, terminal)
        self.assertNotEqual(reference["content_id"], gate._canonical_sha256(report))
        producer = types.SimpleNamespace(verify_complete=mock.Mock(return_value=report, side_effect=error))
        with mock.patch.object(gate, "_load_module", return_value=producer):
            result = gate._validate_artifact_ref(
                reference, role, root,
                {(str(terminal), terminal.stat().st_size, EVIDENCE._sha256(terminal))},
                gate.VALIDATED_ENGINE_COMMIT, "c" * 40,
                root / "unused-policy", "d" * 64, root / "unused-verifier", "e" * 64,
            )
        producer.verify_complete.assert_called_once_with(root, terminal_name)
        self.assertIsNone(result)

    def test_native_roles_accept_official_terminal_references_after_report_replay(self):
        for role, count in (("host_four", 4), ("specialized_five", 5)):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                self.check_role(pathlib.Path(temporary).resolve(), role, count)

    def test_native_roles_reject_wrong_replayed_task_census(self):
        for role, count in (("host_four", 4), ("specialized_five", 5)):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(Exception, "producer census differs"):
                    self.check_role(pathlib.Path(temporary).resolve(), role, count, returned_count=count - 1)

    def test_native_roles_propagate_failed_producer_replay(self):
        for role, count in (("host_four", 4), ("specialized_five", 5)):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(RuntimeError, "native replay rejected"):
                    self.check_role(pathlib.Path(temporary).resolve(), role, count,
                                    error=RuntimeError("native replay rejected"))


class ReportProducerTests(unittest.TestCase):
    def test_reports_preserve_180_seconds_141_rows_and_png(self) -> None:
        report = synthetic_m4_report()
        markdown, html_bytes, plots = EVIDENCE.render_reports(report)
        markdown_text = markdown.decode("utf-8")
        html_text = html_bytes.decode("utf-8")
        self.assertIn("task-179", markdown_text)
        self.assertIn('id="actual-seconds"', html_text)
        self.assertIn('id="means-ratios"', html_text)
        self.assertIn('id="numerical-results"', html_text)
        self.assertIn('id="actual-time-comparison"', html_text)
        self.assertIn(
            "CPU vs v1.0.3 CUDA vs v1.1.0 CUDA measured mean seconds",
            markdown_text,
        )
        self.assertIn("data:image/png;base64,", html_text)
        FINAL._verify_report_semantics(html_text, markdown_text, report)
        FINAL._verify_embedded_plots(html_text, plots, report)
        self.assertEqual(len(plots), 1)
        visibly_wrong = html_text.replace("<td>10</td>", "<td>999</td>", 1)
        self.assertNotEqual(visibly_wrong, html_text)
        with self.assertRaisesRegex(Exception, "actual-time comparison differs"):
            FINAL._verify_report_semantics(visibly_wrong, markdown_text, report)

    def test_report_creation_is_real_durable_pair(self) -> None:
        report = synthetic_m4_report()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            m4 = root / "m4"
            m4.mkdir()
            (m4 / "M4_COMPLETE").write_text("{}\n", encoding="utf-8")
            fake_module = types.SimpleNamespace(verify_complete=lambda _path: report)
            with mock.patch.object(EVIDENCE, "_m4_module", return_value=fake_module):
                EVIDENCE.create_reports(
                    m4_output=m4,
                    markdown_output=root / "report.md",
                    html_output=root / "report.html",
                    evidence_root=root,
                )
            self.assertEqual(stat.S_IMODE((root / "report.md").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / "report.html").stat().st_mode), 0o600)
            with self.assertRaisesRegex(Exception, "overwrite"):
                with mock.patch.object(EVIDENCE, "_m4_module", return_value=fake_module):
                    EVIDENCE.create_reports(
                        m4_output=m4,
                        markdown_output=root / "report.md",
                        html_output=root / "report.html",
                        evidence_root=root,
                    )


class CandidateAuNPRegressionTests(unittest.TestCase):
    def test_actual_hybrid_qualification_profile_key_is_required(self) -> None:
        report = {
            "qualification_profile": {
                "mode": "TM-only",
                "dft_minimum_run_time": 20.0,
                "dft_decay_tolerance": 5e-8,
            }
        }
        summary = CANDIDATE._summary("aunp_source_candidate", report)
        self.assertEqual(summary["workload_count"], 0)
        wrong = json.loads(json.dumps(report))
        wrong["qualification_profile"]["mode"] = "TE-only"
        with self.assertRaisesRegex(Exception, "profile differs"):
            CANDIDATE._summary("aunp_source_candidate", wrong)
        old_shape = {"profile": report["qualification_profile"]}
        with self.assertRaisesRegex(Exception, "profile differs"):
            CANDIDATE._summary("aunp_source_candidate", old_shape)

    def test_wrapper_receipt_must_equal_each_replayed_raw_receipt(self) -> None:
        source_commit = "a" * 40
        real_receipt = "1" * 64
        reports = {
            "targeted": {
                "contract": {
                    "runtime_inputs": {"v110_receipt_id": real_receipt}
                },
                "numerical": {"comparisons": [{} for _ in range(141)]},
            },
            "aunp_source_candidate": {
                "build": {"receipt_id": real_receipt},
                "qualification_profile": {
                    "mode": "TM-only",
                    "dft_minimum_run_time": 20.0,
                    "dft_decay_tolerance": 5e-8,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            for role, report in reports.items():
                with self.subTest(role=role):
                    report["report_id"] = CANDIDATE._canonical_sha256(report)
                    raw_name = (
                        "COMPLETE"
                        if role == "aunp_source_candidate"
                        else "M4_COMPLETE"
                    )
                    raw = root / f"{role}-{raw_name}"
                    write_json(raw, {"schema": "fixture-raw-v1"})
                    wrapper = {
                        "schema": CANDIDATE.SCHEMAS[role],
                        "state": "PASS",
                        "role": role,
                        "source_commit": source_commit,
                        "package_receipt_id": real_receipt,
                        "raw_terminal": {
                            "producer": CANDIDATE.PRODUCER[role],
                            "path": raw.name,
                            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                            "schema": "fixture-raw-v1",
                            "content_id": report["report_id"],
                        },
                        "summary": CANDIDATE._summary(role, report),
                    }
                    wrapper["report_id"] = CANDIDATE._canonical_sha256(wrapper)
                    wrapper_path = root / f"{role}.json"
                    write_json(wrapper_path, wrapper)
                    with mock.patch.object(
                        CANDIDATE, "_replay_raw", return_value=report
                    ):
                        CANDIDATE.verify(role, wrapper_path, source_commit)

                    wrapper["package_receipt_id"] = "f" * 64
                    wrapper.pop("report_id")
                    wrapper["report_id"] = CANDIDATE._canonical_sha256(wrapper)
                    write_json(wrapper_path, wrapper)
                    with mock.patch.object(
                        CANDIDATE, "_replay_raw", return_value=report
                    ):
                        with self.assertRaisesRegex(
                            Exception, "package receipt ID was not re-derived"
                        ):
                            CANDIDATE.verify(role, wrapper_path, source_commit)


class RealTemporaryAuditEvidenceE2ETests(unittest.TestCase):
    def test_exact_conda_glibc_audit_and_external_human_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            source_commit = "c" * 40
            candidate = {"source_commit": source_commit}
            candidate["receipt_id"] = EVIDENCE._canonical_sha256(candidate)
            candidate_path = root / "candidate-receipt.json"
            write_json(candidate_path, candidate)
            package = root / "gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0.conda"
            _write_conda_v2(package, candidate_path.read_bytes())
            package_sha256 = EVIDENCE._sha256(package)

            glibc = root / "candidate-glibc.json"
            EVIDENCE.extract_glibc_terminal(
                package=package,
                candidate_receipt=candidate_path,
                evidence_root=root,
                output=glibc,
            )
            self.assertEqual(stat.S_IMODE(glibc.stat().st_mode), 0o600)

            installed_reports = []
            for index in range(2):
                report = {"schema_version": 1, "state": "PASS", "index": index}
                report["report_id"] = EVIDENCE._canonical_sha256(report)
                path = root / f"installed-{index}.json"
                write_json(path, report)
                installed_reports.append(path)
            glibc_wrapper = root / "glibc-wrapper.json"
            EVIDENCE.create_qualification_wrapper(
                role="glibc_floor",
                raw_terminal=glibc,
                package=package,
                candidate_receipt=candidate_path,
                evidence_root=root,
                output=glibc_wrapper,
                installed_reports=installed_reports,
            )
            self.assertEqual(stat.S_IMODE(glibc_wrapper.stat().st_mode), 0o600)

            finding_evidence = []
            for finding in FINAL.FINAL_AUDIT_FINDINGS:
                path = root / f"{finding}.txt"
                path.write_text(f"{finding} closed\n", encoding="utf-8")
                path.chmod(0o600)
                finding_evidence.append((finding, path))
            findings = {
                "schema": EVIDENCE.AUDIT_INPUT_SCHEMA,
                "auditor": "independent-adversarial-agent",
                "source_commit": source_commit,
                "package_sha256": package_sha256,
                "candidate_receipt_id": candidate["receipt_id"],
                "findings": [
                    {
                        "id": finding,
                        "severity": "HIGH",
                        "state": "CLOSED",
                        "evidence": {
                            "path": path.relative_to(root).as_posix(),
                            "sha256": EVIDENCE._sha256(path),
                        },
                    }
                    for finding, path in finding_evidence
                ],
            }
            findings["input_id"] = EVIDENCE._canonical_sha256(findings)
            findings_path = root / "audit-input.json"
            write_json(findings_path, findings)
            decision = {
                "decision": "APPROVE_RELEASE",
                "approver": "fixture-human",
                "finding_ids": list(FINAL.FINAL_AUDIT_FINDINGS),
                "package_sha256": package_sha256,
            }
            decision["decision_id"] = EVIDENCE._canonical_sha256(decision)
            decision_input = {"schema": EVIDENCE.HUMAN_DECISION_SCHEMA, **decision}
            decision_path = root / "human-decision.json"
            write_json(decision_path, decision_input)

            audit = root / "final-audit.json"
            EVIDENCE.create_final_audit(
                findings_input=findings_path,
                human_decision=decision_path,
                package=package,
                candidate_receipt=candidate_path,
                evidence_root=root,
                output=audit,
            )
            wrapper = root / "final-audit-wrapper.json"
            EVIDENCE.create_qualification_wrapper(
                role="final_adversarial_audit",
                raw_terminal=audit,
                package=package,
                candidate_receipt=candidate_path,
                evidence_root=root,
                output=wrapper,
            )
            self.assertEqual(stat.S_IMODE(audit.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(wrapper.stat().st_mode), 0o600)
            retained = json.loads(wrapper.read_text())
            self.assertEqual(retained["human_release_decision"], decision)
            self.assertEqual(retained["open_critical_high_medium"], 0)

            rejected = dict(decision_input)
            rejected["decision"] = "REJECT_RELEASE"
            rejected.pop("decision_id")
            unsigned = {key: rejected[key] for key in ("decision", "approver", "finding_ids", "package_sha256")}
            rejected["decision_id"] = EVIDENCE._canonical_sha256(unsigned)
            rejected_path = root / "human-rejected.json"
            write_json(rejected_path, rejected)
            with self.assertRaisesRegex(Exception, "human release decision differs"):
                EVIDENCE.create_final_audit(
                    findings_input=findings_path,
                    human_decision=rejected_path,
                    package=package,
                    candidate_receipt=candidate_path,
                    evidence_root=root,
                    output=root / "must-not-exist.json",
                )
            self.assertFalse((root / "must-not-exist.json").exists())


class CandidateSelfCheckBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FINAL_FIXTURE.V110FinalReleaseReceiptTest(
            methodName="test_final_receipt_is_external_mode_600_and_exactly_replayable"
        )
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def mutate_qualification_role(
        self, role: str, key: str, value: object
    ) -> None:
        reference = self.fixture.qualification["qualification"][role]
        path = self.fixture.root / reference["bundle_relative_path"]
        payload = json.loads(path.read_text())
        payload[key] = value
        write_fixture_json(path, payload)
        reference["size_bytes"] = path.stat().st_size
        reference["sha256"] = FINAL._sha256(path)
        reference["content_id"] = payload.get("report_id") or FINAL._canonical_sha256(
            payload
        )
        write_fixture_json(
            self.fixture.qualification_path, self.fixture.qualification
        )

    def test_both_native_entries_require_distinct_self_check_terminals(self) -> None:
        self.fixture.qualification["installed_provenance"][0].pop("self_check")
        write_fixture_json(
            self.fixture.qualification_path, self.fixture.qualification
        )
        with self.assertRaisesRegex(Exception, r"installed_provenance\[0\]"):
            self.fixture._create()

    def test_self_check_path_hash_and_content_id_are_bound(self) -> None:
        second = self.fixture.qualification["installed_provenance"][1]
        second["self_check"] = dict(
            self.fixture.qualification["installed_provenance"][0]["self_check"]
        )
        second["self_check"]["role"] = "self_check_1"
        write_fixture_json(
            self.fixture.qualification_path, self.fixture.qualification
        )
        with self.assertRaisesRegex(Exception, "self-check paths are duplicated"):
            self.fixture._create()

    def test_changed_self_check_bytes_fail_before_semantic_replay(self) -> None:
        reference = self.fixture.qualification["installed_provenance"][0][
            "self_check"
        ]
        path = self.fixture.root / reference["bundle_relative_path"]
        path.write_text('{"schema_version":2,"state":"FAIL"}\n')
        with self.assertRaisesRegex(Exception, "size differs|hash differs"):
            self.fixture._create()

    def test_installed_aunp_must_use_one_native_candidate_prefix(self) -> None:
        self.mutate_qualification_role(
            "installed_aunp", "installed_prefix", "/tmp/not-a-candidate-prefix"
        )
        with self.assertRaisesRegex(
            Exception, "not bound to a native candidate package provenance"
        ):
            self.fixture._create()

    def test_m4_role_identity_must_equal_performance_authority(self) -> None:
        self.mutate_qualification_role(
            "installed_correctness", "m4_report_id", "f" * 64
        )
        with self.assertRaisesRegex(
            Exception, "M4 report differs from performance authority"
        ):
            self.fixture._create()

    def test_m4_roles_cannot_share_the_wrong_development_receipt(self) -> None:
        for role in ("installed_correctness", "installed_multi_gpu"):
            self.mutate_qualification_role(
                role, "development_package_receipt_id", "f" * 64
            )
        with self.assertRaisesRegex(
            Exception, "development receipt differs from measured package"
        ):
            self.fixture._create()


class InstalledSelfCheckReplayTests(unittest.TestCase):
    def test_actual_replay_binds_live_candidate_runtime_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            prefix = root / "candidate-prefix"
            release_path = prefix / FINAL.INSTALLED_RELEASE_MANIFEST_PATH
            release_path.parent.mkdir(parents=True)
            source_commit = "d" * 40
            release = SELF_CHECK_FIXTURE.release_manifest(source_commit)
            write_json(release_path, release)

            runtime_paths = {
                "python": prefix / "bin/python",
                "extension": prefix / "lib/python/meep/_meep.so",
                "libmeep": prefix / "lib/libmeep.so",
            }
            for name, path in runtime_paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + "-candidate-bytes").encode("ascii"))

            def file_record(path: pathlib.Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            def runtime_record(
                backend: str,
                rank: int = 0,
                world_size: int = 1,
                identifier: str = "GPU-A",
            ) -> dict[str, object]:
                record = SELF_CHECK_FIXTURE.record(
                    backend, rank, world_size, identifier
                )
                record.update(
                    {name: file_record(path) for name, path in runtime_paths.items()}
                )
                return record

            lanes = {
                "cpu": {
                    "schema_version": 2,
                    "lane": "cpu",
                    "rank_records": [runtime_record("cpu")],
                },
                "gpu1": {
                    "schema_version": 2,
                    "lane": "gpu1",
                    "rank_records": [runtime_record("cuda")],
                },
                "gpu2": {
                    "schema_version": 2,
                    "lane": "gpu2",
                    "rank_records": [
                        runtime_record("cuda", 0, 2, "GPU-A"),
                        runtime_record("cuda", 1, 2, "GPU-B"),
                    ],
                },
            }
            validated = {
                lane: SELF_CHECK_FIXTURE.MODULE._validate_lane(value, lane)
                for lane, value in lanes.items()
            }
            comparisons = {
                "cpu_vs_gpu1": SELF_CHECK_FIXTURE.MODULE._compare_observables(
                    validated["cpu"][0], validated["gpu1"][0]
                ),
                "cpu_vs_gpu2": SELF_CHECK_FIXTURE.MODULE._compare_observables(
                    validated["cpu"][0], validated["gpu2"][0]
                ),
                "gpu1_vs_gpu2": SELF_CHECK_FIXTURE.MODULE._compare_observables(
                    validated["gpu1"][0], validated["gpu2"][0]
                ),
            }
            payload = {
                "schema_version": 2,
                "state": "PASS",
                "mode": "gpu2",
                "release": {**release, "file": file_record(release_path)},
                "lanes": lanes,
                "comparisons": comparisons,
                "wall_seconds": 1.0,
                "valid_for_speed_gate": False,
                "speed_gate_reason": "small installed-runtime correctness smoke",
            }
            payload["report_id"] = FINAL._canonical_sha256(payload)
            terminal = root / "installed-self-check.json"
            write_json(terminal, payload)
            installed_runtime = {
                **{name: file_record(path) for name, path in runtime_paths.items()},
                "module": file_record(runtime_paths["extension"]),
                "meep_version": "1.35.0-beta",
                "cuda_compiled": True,
                "mpi_enabled": True,
                "single_precision": True,
            }
            self.assertEqual(
                FINAL._replay_installed_self_check(
                    terminal,
                    prefix=prefix,
                    source_commit=source_commit,
                    installed_runtime=installed_runtime,
                ),
                payload,
            )

            original_payload = json.loads(json.dumps(payload))
            decoy = prefix / "lib/libmeep-decoy.so"
            decoy.write_bytes(b"valid-looking-decoy-bytes")
            decoy_record = file_record(decoy)
            for lane in payload["lanes"].values():
                for record in lane["rank_records"]:
                    record["libmeep"] = decoy_record
            payload.pop("report_id")
            payload["report_id"] = FINAL._canonical_sha256(payload)
            write_json(terminal, payload)
            with self.assertRaisesRegex(
                Exception, "differs from installed provenance runtime"
            ):
                FINAL._replay_installed_self_check(
                    terminal,
                    prefix=prefix,
                    source_commit=source_commit,
                    installed_runtime=installed_runtime,
                )

            payload = original_payload
            write_json(terminal, payload)
            runtime_paths["libmeep"].write_bytes(b"changed-after-self-check")
            with self.assertRaisesRegex(Exception, "escaped or changed"):
                FINAL._replay_installed_self_check(
                    terminal,
                    prefix=prefix,
                    source_commit=source_commit,
                    installed_runtime=installed_runtime,
                )


if __name__ == "__main__":
    unittest.main()
