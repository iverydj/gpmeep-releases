#!/usr/bin/env python3
"""Focused tests for one sealed M3 performance sample."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_performance_sample.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_performance_sample", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLE
SPEC.loader.exec_module(SAMPLE)


GPU0 = "GPU-00000000-0000-0000-0000-000000009001"
GPU1 = "GPU-00000000-0000-0000-0000-000000009002"


def audited_run(backend: str, seconds: float) -> dict[str, object]:
    other = "cuda" if backend == "cpu" else "cpu"
    return {
        "run_wall_seconds": seconds,
        "active_backend": backend,
        "run_phase_calls": {backend: 1, other: 0},
        "field_timing_delta": {
            "time_stepping": {
                "per_rank_seconds": [seconds * 0.8],
                "maximum_seconds": seconds * 0.8,
                "mean_seconds": seconds * 0.8,
            },
            "boundaries": {
                "per_rank_seconds": [seconds * 0.1],
                "maximum_seconds": seconds * 0.1,
                "mean_seconds": seconds * 0.1,
            },
            "fourier_transform": {
                "per_rank_seconds": [seconds * 0.05],
                "maximum_seconds": seconds * 0.05,
                "mean_seconds": seconds * 0.05,
            },
            "get_farfields": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
            "connecting_chunks": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
            "mpi_all_to_all": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
            "mpi_one_to_one": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
            "mpb": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
            "field_output": {
                "per_rank_seconds": [0.0],
                "maximum_seconds": 0.0,
                "mean_seconds": 0.0,
            },
        },
    }


class M3PerformanceSampleTests(unittest.TestCase):
    def test_evidence_code_seals_cylindrical_fixed_work_contract(self) -> None:
        records = SAMPLE._code_records(
            {
                "repo": ROOT,
                "launcher": ROOT / "scripts/python-validation/m3_mpi_rank_launcher.py",
                "manifest": ROOT / "scripts/python-validation/manifest.json",
            }
        )
        self.assertEqual(
            records["adjoint_cyl_fixed_work_contract"]["path"],
            str(
                (
                    ROOT / "python/tests/adjoint_cyl_fixed_work_contract.py"
                ).resolve()
            ),
        )
        installed_records = SAMPLE._code_records(
            {
                "repo": ROOT,
                "launcher": ROOT / "scripts/python-validation/m3_mpi_rank_launcher.py",
                "manifest": ROOT / "scripts/python-validation/manifest.json",
                "package": ROOT / "README.md",
            }
        )
        self.assertEqual(
            installed_records["conda_archive_inspector"]["path"],
            str((ROOT / "scripts/v110-final-release-receipt.py").resolve()),
        )

    @staticmethod
    def clean_git(commit: str):
        def run(_repo: pathlib.Path, *arguments: str) -> str:
            if arguments == ("rev-parse", "--verify", "HEAD"):
                return f"{commit}\n"
            if arguments == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--no-renames",
            ):
                return ""
            raise AssertionError(f"unexpected Git arguments: {arguments!r}")

        return run

    def test_installed_source_identity_uses_trusted_git_not_snapshot_field(self) -> None:
        commit = "a" * 40
        compact_snapshot = {
            "available": True,
            "algorithm": "sha256",
            "file_count": 1,
            "sha256": "b" * 64,
        }
        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=self.clean_git(commit),
        ) as git_output, mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value=compact_snapshot,
        ):
            self.assertEqual(
                SAMPLE.repository_source_identity(ROOT),
                (commit, compact_snapshot),
            )
        self.assertEqual(git_output.call_count, 4)

    def test_installed_source_identity_fails_if_head_changes_during_snapshot(self) -> None:
        commit_calls = iter(("a" * 40, "b" * 40))

        def changing_head(_repo: pathlib.Path, *arguments: str) -> str:
            if arguments[0] == "rev-parse":
                return next(commit_calls)
            return ""

        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=changing_head,
        ), mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value={"available": True, "sha256": "c" * 64},
        ):
            with self.assertRaisesRegex(SAMPLE.WorkloadError, "changed while captured"):
                SAMPLE.repository_source_identity(ROOT)

    def test_installed_source_identity_rejects_dirty_worktree(self) -> None:
        commit = "a" * 40

        def dirty_git(_repo: pathlib.Path, *arguments: str) -> str:
            if arguments[0] == "rev-parse":
                return commit
            return "?? untracked.py\n"

        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=dirty_git,
        ), mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value={"available": True, "sha256": "c" * 64},
        ):
            with self.assertRaisesRegex(SAMPLE.WorkloadError, "not clean"):
                SAMPLE.repository_source_identity(ROOT)

    def test_installed_source_identity_rejects_status_race(self) -> None:
        statuses = iter(("", " M changed.py\n"))

        def racing_status(_repo: pathlib.Path, *arguments: str) -> str:
            if arguments[0] == "rev-parse":
                return "a" * 40
            return next(statuses)

        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=racing_status,
        ), mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value={"available": True, "sha256": "c" * 64},
        ):
            with self.assertRaisesRegex(SAMPLE.WorkloadError, "changed while captured"):
                SAMPLE.repository_source_identity(ROOT)

    def test_installed_source_identity_rejects_malformed_commit(self) -> None:
        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=self.clean_git("not-a-commit"),
        ), mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value={"available": True, "sha256": "c" * 64},
        ):
            with self.assertRaisesRegex(SAMPLE.WorkloadError, "changed while captured"):
                SAMPLE.repository_source_identity(ROOT)

    def test_installed_source_identity_fails_if_snapshot_is_unavailable(self) -> None:
        with mock.patch.object(
            SAMPLE.provenance,
            "git_output",
            side_effect=self.clean_git("a" * 40),
        ), mock.patch.object(
            SAMPLE.validation,
            "source_snapshot",
            return_value={"available": False, "error": "inventory failed"},
        ):
            with self.assertRaisesRegex(SAMPLE.WorkloadError, "inventory failed"):
                SAMPLE.repository_source_identity(ROOT)

    def test_installed_runtime_paths_pass_independent_commit_to_provenance(self) -> None:
        commit = "a" * 40
        compact = {"available": True, "sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            manifest = repo / "scripts/python-validation/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.touch()
            prefix = root / "prefix"
            build_python = prefix / "lib/python3.11/site-packages"
            build_python.mkdir(parents=True)
            files = {
                name: root / name
                for name in ("python", "provenance.json", "mpiexec", "package.conda", "verifier")
            }
            for path in files.values():
                path.touch()
            args = mock.Mock(
                repo=repo,
                manifest=manifest,
                python=files["python"],
                build_python=build_python,
                install_prefix=prefix,
                build_receipt=files["provenance.json"],
                mpiexec=files["mpiexec"],
                package=files["package.conda"],
            )
            runtime = {
                "python": files["python"],
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": files["provenance.json"],
                "mpiexec": files["mpiexec"],
                "package": files["package.conda"],
                "provenance_report": files["provenance.json"],
                "verifier": files["verifier"],
            }
            with mock.patch.object(
                SAMPLE,
                "repository_source_identity",
                return_value=(commit, compact),
            ), mock.patch.object(
                SAMPLE.v110_identity,
                "validate_report",
                return_value=({"report_id": "report"}, {}),
            ) as validate, mock.patch.object(
                SAMPLE.v110_identity,
                "runtime_paths",
                return_value=runtime,
            ):
                paths = SAMPLE._runtime_paths(args)
            validate.assert_called_once_with(
                files["provenance.json"],
                files["package.conda"],
                prefix,
                commit,
            )
            self.assertEqual(paths["provenance_report"], files["provenance.json"])

    def test_installed_runtime_state_retains_independent_compact_snapshot(self) -> None:
        commit = "a" * 40
        compact = {"available": True, "sha256": "b" * 64}
        report = {"report_id": "report"}
        runtime = {
            name: pathlib.Path(f"/{name}")
            for name in (
                "python",
                "build_python",
                "install_prefix",
                "build_receipt",
                "mpiexec",
                "package",
                "provenance_report",
                "verifier",
            )
        }
        paths = {"repo": ROOT, **runtime}
        retained = {
            "available": True,
            "source_snapshot": compact,
            "runtime_contract": {"receipt_id": "report"},
        }
        with mock.patch.object(
            SAMPLE,
            "repository_source_identity",
            side_effect=((commit, compact), (commit, compact)),
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "validate_report",
            return_value=(report, {}),
        ) as validate, mock.patch.object(
            SAMPLE.v110_identity,
            "runtime_paths",
            return_value=runtime,
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "package_owned_records",
            return_value=(),
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "runtime_contract",
            return_value=retained["runtime_contract"],
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "live_verify",
            return_value={},
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "snapshot",
            return_value=retained,
        ):
            receipt_id, snapshot = SAMPLE._runtime_state(paths)
        validate.assert_called_once_with(
            runtime["provenance_report"],
            runtime["package"],
            runtime["install_prefix"],
            commit,
        )
        self.assertEqual((receipt_id, snapshot), ("report", retained))

    def test_installed_runtime_state_rejects_snapshot_substitution(self) -> None:
        compact = {"available": True, "sha256": "b" * 64}
        changed = {"available": True, "sha256": "c" * 64}
        runtime = {
            name: pathlib.Path(f"/{name}")
            for name in (
                "python",
                "build_python",
                "install_prefix",
                "build_receipt",
                "mpiexec",
                "package",
                "provenance_report",
                "verifier",
            )
        }
        with mock.patch.object(
            SAMPLE,
            "repository_source_identity",
            side_effect=(("a" * 40, compact), ("a" * 40, compact)),
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "validate_report",
            return_value=({"report_id": "report"}, {}),
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "runtime_paths",
            return_value=runtime,
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "package_owned_records",
            return_value=(),
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "runtime_contract",
            return_value={},
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "live_verify",
            return_value={},
        ), mock.patch.object(
            SAMPLE.v110_identity,
            "snapshot",
            return_value={"source_snapshot": changed},
        ):
            with self.assertRaisesRegex(
                SAMPLE.WorkloadError, "changed during runtime capture"
            ):
                SAMPLE._runtime_state({"repo": ROOT, **runtime})

    def test_runtime_verification_session_reuses_then_reverifies(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "lib/python3.11/site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            files = {}
            for name in (
                "manifest",
                "python",
                "receipt",
                "mpiexec",
                "package",
                "verifier",
                "extension",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                files[name] = path
            paths = {
                "repo": repo,
                "manifest": files["manifest"],
                "python": files["python"],
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": files["receipt"],
                "mpiexec": files["mpiexec"],
                "package": files["package"],
                "provenance_report": files["receipt"],
                "verifier": files["verifier"],
                "launcher": files["manifest"],
            }
            snapshot = {
                "available": True,
                "source_snapshot": {
                    "available": True,
                    "sha256": "b" * 64,
                },
                "build_extension": {
                    "available": True,
                    "path": str(files["extension"]),
                    "sha256": "c" * 64,
                    "size_bytes": files["extension"].stat().st_size,
                },
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }

            def git_output(_repo, command, *arguments):
                if command == "rev-parse":
                    return "a" * 40 + "\n"
                self.assertEqual(command, "status")
                return ""

            with mock.patch.object(
                SAMPLE,
                "_runtime_state",
                return_value=("report", snapshot),
            ) as capture, mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=git_output,
            ):
                session = SAMPLE.RuntimeVerificationSession()
                first = session.runtime_state(paths)
                first[1]["source_snapshot"]["sha256"] = "tampered"
                second = session.runtime_state(paths)
                third = session.runtime_state(paths)
                self.assertEqual(
                    second[1]["source_snapshot"]["sha256"], "b" * 64
                )
                self.assertEqual(third, second)
                self.assertEqual(capture.call_count, 1)
                session.finalize()
                self.assertEqual(capture.call_count, 2)

    def test_runtime_verification_session_rejects_anchored_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            anchored = root / "runtime"
            anchored.write_text("before", encoding="utf-8")
            paths = {
                "repo": repo,
                "manifest": anchored,
                "python": anchored,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": anchored,
                "mpiexec": anchored,
                "package": anchored,
                "provenance_report": anchored,
                "verifier": anchored,
                "launcher": anchored,
            }
            snapshot = {
                "available": True,
                "source_snapshot": {
                    "available": True,
                    "sha256": "b" * 64,
                },
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }
            with mock.patch.object(
                SAMPLE,
                "_runtime_state",
                return_value=("report", snapshot),
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                session.runtime_state(paths)
                anchored.write_text("after", encoding="utf-8")
                with self.assertRaisesRegex(
                    SAMPLE.WorkloadError, "session file changed"
                ):
                    session.runtime_state(paths)
                session.close(verify=False)

    def test_runtime_verification_session_anchors_full_identity_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            common = root / "runtime"
            conda_record = prefix / "conda-meta/gpmeep.json"
            release_manifest = prefix / "share/gpmeep/release.json"
            owned_module = build_python / "meep/backend_policy.py"
            for path in (common, conda_record, release_manifest, owned_module):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")
            paths = {
                "repo": repo,
                "manifest": common,
                "python": common,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": common,
                "mpiexec": common,
                "package": common,
                "provenance_report": common,
                "verifier": common,
                "launcher": common,
            }
            snapshot = {
                "available": True,
                "source_snapshot": {"available": True, "sha256": "b" * 64},
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }

            def runtime_state(
                _paths,
                *,
                identity_record_paths=None,
                identity_file_records=None,
            ):
                if identity_record_paths is not None:
                    identity_record_paths.update(
                        {conda_record, release_manifest}
                    )
                if identity_file_records is not None:
                    identity_file_records.append(
                        {
                            "path": owned_module,
                            "path_type": "hardlink",
                            "content_path": owned_module,
                            "sha256": hashlib.sha256(
                                owned_module.read_bytes()
                            ).hexdigest(),
                            "size_bytes": owned_module.stat().st_size,
                            "mode": stat.S_IMODE(owned_module.lstat().st_mode),
                            "content_mode": stat.S_IMODE(
                                owned_module.stat().st_mode
                            ),
                        }
                    )
                return "report", snapshot

            with mock.patch.object(
                SAMPLE, "_runtime_state", side_effect=runtime_state
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                for target in (
                    conda_record,
                    release_manifest,
                    owned_module,
                ):
                    with self.subTest(target=target.name):
                        original = target.read_text(encoding="utf-8")
                        session = SAMPLE.RuntimeVerificationSession()
                        session.runtime_state(paths)
                        target.write_text("mutated", encoding="utf-8")
                        with self.assertRaisesRegex(
                            SAMPLE.WorkloadError, "session file changed"
                        ):
                            session.runtime_state(paths)
                        session.close(verify=False)
                        target.write_text(original, encoding="utf-8")

    def test_runtime_verification_session_hashes_exact_opened_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            common = root / "runtime"
            common.write_text("runtime", encoding="utf-8")
            owned = build_python / "meep/backend_policy.py"
            owned.parent.mkdir(parents=True)
            owned.write_text("validated", encoding="utf-8")
            expected_sha256 = hashlib.sha256(owned.read_bytes()).hexdigest()
            expected_size = owned.stat().st_size
            paths = {
                "repo": repo,
                "manifest": common,
                "python": common,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": common,
                "mpiexec": common,
                "package": common,
                "provenance_report": common,
                "verifier": common,
                "launcher": common,
            }
            snapshot = {
                "available": True,
                "source_snapshot": {"available": True, "sha256": "b" * 64},
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }

            def runtime_state(
                _paths,
                *,
                identity_record_paths=None,
                identity_file_records=None,
            ):
                identity_file_records.append(
                    {
                        "path": owned,
                        "path_type": "hardlink",
                        "content_path": owned,
                        "sha256": expected_sha256,
                        "size_bytes": expected_size,
                        "mode": stat.S_IMODE(owned.lstat().st_mode),
                        "content_mode": stat.S_IMODE(owned.stat().st_mode),
                    }
                )
                owned.write_text("substituted", encoding="utf-8")
                return "report", snapshot

            with mock.patch.object(
                SAMPLE, "_runtime_state", side_effect=runtime_state
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                with self.assertRaisesRegex(
                    SAMPLE.WorkloadError,
                    "(expected file identity|anchored file bytes) differs",
                ):
                    session.runtime_state(paths)

    def test_runtime_verification_session_initial_failure_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            common = root / "runtime"
            common.write_text("runtime", encoding="utf-8")
            paths = {
                "repo": repo,
                "manifest": common,
                "python": common,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": common,
                "mpiexec": common,
                "package": common,
                "provenance_report": common,
                "verifier": common,
                "launcher": common,
            }
            snapshot = {
                "available": True,
                "source_snapshot": {"available": True, "sha256": "b" * 64},
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }
            calls = {"count": 0}

            def runtime_state(_paths, **_kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise SAMPLE.WorkloadError("transient verifier failure")
                return "report", snapshot

            with mock.patch.object(
                SAMPLE, "_runtime_state", side_effect=runtime_state
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                with self.assertRaisesRegex(
                    SAMPLE.WorkloadError, "transient verifier failure"
                ):
                    session.runtime_state(paths)
                self.assertEqual(session.runtime_state(paths)[0], "report")
                session.finalize()
                self.assertEqual(calls["count"], 3)

    def test_runtime_verification_session_final_mismatch_closes_and_reports_abort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            common = root / "runtime"
            common.write_text("runtime", encoding="utf-8")
            paths = {
                "repo": repo,
                "manifest": common,
                "python": common,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": common,
                "mpiexec": common,
                "package": common,
                "provenance_report": common,
                "verifier": common,
                "launcher": common,
            }
            initial = {
                "available": True,
                "source_snapshot": {"available": True, "sha256": "b" * 64},
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }
            changed = json.loads(json.dumps(initial))
            changed["source_snapshot"]["sha256"] = "c" * 64
            with mock.patch.object(
                SAMPLE,
                "_runtime_state",
                side_effect=(("report", initial), ("report", changed)),
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                session.runtime_state(paths)
                with self.assertRaisesRegex(
                    SAMPLE.WorkloadError, "final verification differed"
                ):
                    session.finalize()
                with self.assertRaisesRegex(
                    SAMPLE.WorkloadError, "no longer active"
                ):
                    session.runtime_state(paths)

            with mock.patch.object(
                SAMPLE,
                "_runtime_state",
                side_effect=(("report", initial), ("report", changed)),
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                session.runtime_state(paths)
                diagnostic = session.abort()
                self.assertIn("final verification differed", diagnostic)
                self.assertIsNone(session.abort())

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_runtime_verification_session_rejects_child_without_harming_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            prefix = root / "prefix"
            build_python = prefix / "site-packages"
            build_python.mkdir(parents=True)
            repo.mkdir()
            common = root / "runtime"
            common.write_text("runtime", encoding="utf-8")
            paths = {
                "repo": repo,
                "manifest": common,
                "python": common,
                "build_python": build_python,
                "install_prefix": prefix,
                "build_receipt": common,
                "mpiexec": common,
                "package": common,
                "provenance_report": common,
                "verifier": common,
                "launcher": common,
            }
            snapshot = {
                "available": True,
                "source_snapshot": {"available": True, "sha256": "b" * 64},
                "build_extension": {"available": False},
                "build_receipt": {"available": True},
                "runtime_contract": {"receipt_id": "report"},
                "problems": [],
            }
            with mock.patch.object(
                SAMPLE, "_runtime_state", return_value=("report", snapshot)
            ), mock.patch.object(
                SAMPLE.provenance,
                "git_output",
                side_effect=lambda _repo, command, *args: (
                    "a" * 40 + "\n" if command == "rev-parse" else ""
                ),
            ):
                session = SAMPLE.RuntimeVerificationSession()
                session.runtime_state(paths)
                child = os.fork()
                if child == 0:
                    try:
                        try:
                            session.runtime_state(paths)
                        except SAMPLE.WorkloadError:
                            session.close(verify=False)
                            os._exit(0)
                        os._exit(3)
                    except BaseException:
                        os._exit(4)
                waited, status = os.waitpid(child, 0)
                self.assertEqual(waited, child)
                self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                session.finalize()

    def test_json_metrics_comparison_replays_metric_extraction(self) -> None:
        comparison = {"mode": "json_metrics", "prefix": "metrics:"}
        with mock.patch.object(
            SAMPLE.validation, "extract_json_metrics"
        ) as extract:
            SAMPLE.validate_comparison_evidence("metrics:{}", comparison)
        extract.assert_called_once_with("metrics:{}", comparison)

    def test_embedded_oracle_comparison_needs_no_json_prefix(self) -> None:
        with mock.patch.object(
            SAMPLE.validation, "extract_json_metrics"
        ) as extract:
            SAMPLE.validate_comparison_evidence(
                "Ran 4 tests in 1.0s\n\nOK\n", {"mode": "embedded_oracle"}
            )
        extract.assert_not_called()

    def test_unknown_comparison_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            Exception, "unsupported M3 performance comparison mode"
        ):
            SAMPLE.validate_comparison_evidence("", {"mode": "future"})

    def test_complete_inversion_inventory_is_sealed(self) -> None:
        expected = {
            "python/examples/phase_in_material.py",
            "python/examples/material-dispersion.py",
            "python/examples/refl-quartz.py",
            "python/examples/waveguide_crossing.py",
            "python/examples/adjoint_optimization/multilayer_opt.py",
            "python/examples/absorber-1d.py",
            "python/examples/parallel-wvgs-force.py",
            "python/examples/antenna_pec_ground_plane_1D.py",
            "python/examples/3rd-harm-1d.py",
            "python/examples/adjoint_optimization/near2far_3d_validation.py",
            "python/examples/edge_emitter_3D.py",
            "python/examples/dipole_in_vacuum_cyl_off_axis.py",
            "python/tests/test_adjoint_cyl.py",
        }
        self.assertEqual(SAMPLE.INVERSION_CASES, expected)
        SAMPLE.validate_inversion_inventory(ROOT)
        self.assertTrue(expected <= set(SAMPLE.CASE_TIMING))

    def test_four_direct_cases_are_wrapped_by_the_exact_generic_oracle(self) -> None:
        manifest = SAMPLE.validation.load_json(
            ROOT / "scripts/python-validation/manifest.json"
        )
        cases = SAMPLE.validation.materialize_cases(manifest, ROOT)
        by_path = {case["path"]: case for case in cases}
        expected_counts = {
            "python/examples/waveguide_crossing.py": "20",
            "python/examples/adjoint_optimization/multilayer_opt.py": "3",
            "python/examples/adjoint_optimization/near2far_3d_validation.py": "10",
            "python/tests/test_adjoint_cyl.py": "16",
        }
        for path, count in expected_counts.items():
            with self.subTest(path=path):
                command = SAMPLE.performance_oracle_command(
                    by_path[path], pathlib.Path("/python"), ROOT
                )
                self.assertEqual(pathlib.Path(command[1]).name, "run_example_oracle.py")
                run_count = command.index("--expected-run-count")
                self.assertEqual(command[run_count + 1], count)
                self.assertEqual(
                    command[command.index("--min-field-energy") + 1], "1e-30"
                )
                if path == "python/examples/waveguide_crossing.py":
                    self.assertIn("--disable-field-timing-snapshots", command)
                else:
                    self.assertNotIn(
                        "--disable-field-timing-snapshots", command
                    )
                self.assertIn(str(ROOT / path), command)
                if path == "python/tests/test_adjoint_cyl.py":
                    self.assertIn("--unittest-discover", command)
                else:
                    self.assertNotIn("--unittest-discover", command)

    def test_waveguide_optimizer_trajectory_metrics_remain_blocking(self) -> None:
        manifest = SAMPLE.validation.load_json(
            ROOT / "scripts/python-validation/manifest.json"
        )
        cases = SAMPLE.validation.materialize_cases(manifest, ROOT)
        waveguide = next(
            case
            for case in cases
            if case["path"] == "python/examples/waveguide_crossing.py"
        )
        tolerances = waveguide["comparison"]["tolerances"]
        trajectory_rules = (
            "optimizer_smoke.*_design_step_l2",
            "optimizer_smoke.*_gradient_l2_*",
            "optimizer_smoke.*_objective_*",
        )
        for name in trajectory_rules:
            with self.subTest(name=name):
                self.assertIn(name, tolerances)
                self.assertFalse(tolerances[name].get("diagnostic_only", False))

    def test_direct_route_rejects_a_manifest_command_change(self) -> None:
        path = "python/examples/waveguide_crossing.py"
        case = {
            "path": path,
            "kind": "example",
            "command": ["{python}", f"{{repo}}/{path}", "--changed"],
        }
        with self.assertRaisesRegex(Exception, "direct example command differs"):
            SAMPLE.performance_oracle_command(case, pathlib.Path("/python"), ROOT)

    def test_sample_identity_is_canonical(self) -> None:
        SAMPLE.validate_sample_identity("warmup", 0)
        SAMPLE.validate_sample_identity("measured", 1)
        for kind, cycle in (("warmup", 1), ("measured", 0), ("other", 1)):
            with self.subTest(kind=kind, cycle=cycle):
                with self.assertRaisesRegex(Exception, "sample"):
                    SAMPLE.validate_sample_identity(kind, cycle)

    def test_topology_selection_binds_exact_gpu_inventory(self) -> None:
        devices = SAMPLE.normalized_devices(f"{GPU0},{GPU1}")
        cpu, requested, policy = SAMPLE.select_lane(
            "cpu8", devices, "python/examples/edge_emitter_3D.py"
        )
        self.assertEqual((cpu.backend, cpu.ranks, cpu.devices), ("cpu", 8, ()))
        self.assertEqual(requested, "cpu")
        self.assertIsNone(policy["active_backends"])
        cpu1, requested, _ = SAMPLE.select_lane(
            "cpu1", devices, "python/examples/metasurface_lens.py"
        )
        self.assertEqual((cpu1.backend, cpu1.ranks, cpu1.devices), ("cpu", 1, ()))
        self.assertEqual(requested, "cpu")
        cuda2, requested, _ = SAMPLE.select_lane(
            "cuda2", devices, "python/examples/edge_emitter_3D.py"
        )
        self.assertEqual((cuda2.backend, cuda2.ranks), ("cuda", 2))
        self.assertEqual(cuda2.devices, devices)
        self.assertEqual(requested, "cuda")
        automatic, requested, policy = SAMPLE.select_lane(
            "auto1", devices, "python/examples/metasurface_lens.py"
        )
        self.assertEqual(requested, "auto")
        self.assertEqual(automatic.devices, devices[:1])
        self.assertEqual(automatic.backend, "cuda")
        self.assertEqual(policy["active_backends"][-1], "cuda")
        with self.assertRaisesRegex(Exception, "distinct GPUs"):
            SAMPLE.normalized_devices(f"{GPU0},{GPU0.lower()}")

    def test_automatic_environment_removes_strict_cuda_marker(self) -> None:
        lane = SAMPLE.mpi_case.Lane("auto", "cuda", 1, (GPU0,))
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            output = root / "samples" / "task"
            runtime_tmp = root / "runtime" / "task" / "tmp"
            output.mkdir(parents=True)
            runtime_tmp.mkdir(parents=True)
            python = pathlib.Path(sys.executable).resolve(strict=True)
            with mock.patch.object(
                SAMPLE.mpi_case,
                "clean_environment",
                return_value={
                    "MEEP_GPU_BACKEND": "cuda",
                    "GPMEEP_VALIDATION_STRICT_CUDA": "1",
                },
            ) as cleaner, mock.patch.object(
                SAMPLE.validation,
                "isolated_process_environment",
                side_effect=AssertionError("verifier ambient must not be sampled"),
            ):
                environment = SAMPLE.sample_environment(
                    lane,
                    "auto",
                    case_path="python/examples/waveguide_crossing.py",
                    output=output,
                    python=python,
                    build_python=pathlib.Path("/build"),
                    runtime_contract={},
                    run_nonce="nonce",
                )
            self.assertEqual(
                cleaner.call_args.kwargs["producer_safe_ambient"],
                {
                    "PATH": f"{python.parent}:/usr/bin:/bin",
                    "TMPDIR": str(runtime_tmp),
                    "LC_ALL": "C",
                    "LANG": "C",
                },
            )
        self.assertEqual(environment["MEEP_GPU_BACKEND"], "auto")
        self.assertEqual(
            environment["GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND"],
            "auto",
        )
        self.assertNotIn("GPMEEP_VALIDATION_STRICT_CUDA", environment)
        self.assertNotIn(SAMPLE.ADJOINT_CYL_FIXED_WORK_ENV, environment)
        self.assertNotIn(SAMPLE.PHASE_IN_MATERIAL_FIXED_WORK_ENV, environment)

    def test_cylindrical_adjoint_performance_environment_is_fixed_work(self) -> None:
        lane = SAMPLE.mpi_case.Lane("cpu", "cpu", 1, ())
        with mock.patch.object(
            SAMPLE.mpi_case,
            "expected_performance_producer_safe_ambient",
            return_value={},
        ), mock.patch.object(SAMPLE.mpi_case, "clean_environment", return_value={}):
            environment = SAMPLE.sample_environment(
                lane,
                "cpu",
                case_path="python/tests/test_adjoint_cyl.py",
                output=pathlib.Path("/output"),
                python=pathlib.Path("/python"),
                build_python=pathlib.Path("/build"),
                runtime_contract={},
                run_nonce="nonce",
            )
        self.assertEqual(environment[SAMPLE.ADJOINT_CYL_FIXED_WORK_ENV], "1")

    def test_phase_in_material_performance_work_is_observable_and_sealed(self) -> None:
        lane = SAMPLE.mpi_case.Lane("cuda", "cuda", 1, (GPU0,))
        with mock.patch.object(
            SAMPLE.mpi_case,
            "expected_performance_producer_safe_ambient",
            return_value={},
        ), mock.patch.object(SAMPLE.mpi_case, "clean_environment", return_value={}):
            environment = SAMPLE.sample_environment(
                lane,
                "cuda",
                case_path=SAMPLE.PHASE_IN_MATERIAL_CASE,
                output=pathlib.Path("/output"),
                python=pathlib.Path("/python"),
                build_python=pathlib.Path("/build"),
                runtime_contract={},
                run_nonce="nonce",
            )
        self.assertEqual(
            environment[SAMPLE.PHASE_IN_MATERIAL_FIXED_WORK_ENV], "1"
        )
        manifest = SAMPLE.validation.load_json(
            ROOT / "scripts/python-validation/manifest.json"
        )
        cases = SAMPLE.validation.materialize_cases(manifest, ROOT)
        case = next(
            item for item in cases if item["path"] == SAMPLE.PHASE_IN_MATERIAL_CASE
        )
        ordinary = SAMPLE.validation.case_command(case, pathlib.Path("/python"), ROOT)
        fixed = SAMPLE.performance_oracle_command(case, pathlib.Path("/python"), ROOT)
        self.assertEqual(
            ordinary[ordinary.index("--expected-final-timestep") + 1], "40"
        )
        self.assertEqual(
            ordinary[ordinary.index("--expected-total-timestep-delta") + 1],
            "80",
        )
        self.assertEqual(
            fixed[fixed.index("--expected-final-timestep") + 1],
            str(SAMPLE.PHASE_IN_MATERIAL_FIXED_FINAL_TIMESTEP),
        )
        self.assertEqual(
            fixed[fixed.index("--expected-total-timestep-delta") + 1],
            str(SAMPLE.PHASE_IN_MATERIAL_FIXED_TOTAL_TIMESTEP_DELTA),
        )
        self.assertEqual(fixed[-2:], ["--", "--validation"])
        plan = SAMPLE.validation.load_json(
            ROOT / "packaging/v110-m4-equal-work-plan.json"
        )
        phase_contract = next(
            item["fixed_work"]
            for item in plan["cases"]
            if item["path"] == SAMPLE.PHASE_IN_MATERIAL_CASE
        )
        self.assertEqual(
            phase_contract,
            {
                "environment": SAMPLE.PHASE_IN_MATERIAL_FIXED_WORK_ENV,
                "value": "1",
                "phase_time": 200.0,
                "run_count": 2,
                "final_timestep": (
                    SAMPLE.PHASE_IN_MATERIAL_FIXED_FINAL_TIMESTEP
                ),
                "total_timestep_delta": (
                    SAMPLE.PHASE_IN_MATERIAL_FIXED_TOTAL_TIMESTEP_DELTA
                ),
                "purpose": (
                    "stable timing and one-second external GPU process "
                    "observability"
                ),
            },
        )

    def test_phase_in_material_fixed_work_command_fails_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "lacks exact"):
            SAMPLE._phase_in_material_fixed_work_command(["python", "oracle"])
        with self.assertRaisesRegex(Exception, "changed"):
            SAMPLE._phase_in_material_fixed_work_command(
                [
                    "python",
                    "oracle",
                    "--expected-final-timestep",
                    "41",
                    "--expected-total-timestep-delta",
                    "80",
                ]
            )

    def test_timing_contracts_are_rederived_from_every_run(self) -> None:
        summed = SAMPLE.derive_timing(
            "python/examples/edge_emitter_3D.py",
            [audited_run("cpu", 1.0), audited_run("cpu", 2.0)],
        )
        self.assertEqual(summed["primary_seconds"], 3.0)
        third_harmonic = SAMPLE.derive_timing(
            "python/examples/3rd-harm-1d.py",
            [audited_run("cuda", 2.0), audited_run("cuda", 3.0)],
        )
        self.assertEqual(third_harmonic["primary_seconds"], 5.0)
        runs = [audited_run("cpu", float(index + 1)) for index in range(20)]
        metasurface = SAMPLE.derive_timing(
            "python/examples/metasurface_lens.py", runs
        )
        self.assertEqual(metasurface["small_seconds"], 19.0)
        self.assertEqual(metasurface["large_seconds"], 20.0)
        self.assertEqual(metasurface["primary_seconds"], 20.0)
        with self.assertRaisesRegex(Exception, "exactly 20"):
            SAMPLE.derive_timing("python/examples/metasurface_lens.py", runs[:-1])
        with self.assertRaisesRegex(Exception, "not sealed"):
            SAMPLE.derive_timing("python/examples/other.py", runs)

    def test_ten_phase_profile_marks_unseparable_phases_explicitly(self) -> None:
        runs = [audited_run("cpu", 2.0), audited_run("cpu", 3.0)]
        profile = SAMPLE.derive_phase_profile(runs, 6.0)
        self.assertEqual(tuple(profile["phases"]), SAMPLE.PHASE_NAMES)
        self.assertFalse(profile["phases"]["initialization"]["available"])
        self.assertIsNone(profile["phases"]["initialization"]["seconds"])
        self.assertAlmostEqual(profile["outside_observed_runs_seconds"], 1.0)
        self.assertAlmostEqual(profile["phases"]["fdtd_timestep"]["seconds"], 4.0)
        self.assertFalse(profile["phases"]["source"]["available"])
        self.assertIsNone(profile["phases"]["source"]["seconds"])
        missing = [dict(runs[0]), dict(runs[1])]
        missing[1]["field_timing_delta"] = None
        unavailable = SAMPLE.derive_phase_profile(missing, 6.0)
        self.assertFalse(unavailable["phases"]["fdtd_timestep"]["available"])
        self.assertIsNone(unavailable["phases"]["fdtd_timestep"]["seconds"])

    def test_each_forced_backend_command_binds_the_exact_reason(self) -> None:
        case = {"path": "fixture.py"}
        command = [
            "/python",
            str(ROOT / "scripts/python-validation/run_example_oracle.py"),
            "fixture.py",
        ]
        with mock.patch.object(
            SAMPLE.validation, "case_command", return_value=command
        ):
            cpu = SAMPLE.case_command_for_lane(
                case, pathlib.Path("/python"), ROOT, "cpu"
            )
            cuda = SAMPLE.case_command_for_lane(
                case, pathlib.Path("/python"), ROOT, "cuda"
            )
        option = "--acknowledge-backend-mismatch-reason"
        self.assertEqual(cpu[cpu.index(option) + 1], SAMPLE.BACKEND_MISMATCH_REASON)
        self.assertEqual(cuda[cuda.index(option) + 1], SAMPLE.BACKEND_MISMATCH_REASON)
        for command in (cpu, cuda):
            scope_option = "--acknowledgement-scope"
            self.assertEqual(command[command.index(scope_option) + 1], "process")

    def test_forced_policy_requires_whole_owner_dispatch(self) -> None:
        lane = SAMPLE.mpi_case.Lane("cpu", "cpu", 8, ())
        policy = {
            "schema": "forced-v1",
            "classification": "forced",
            "active_backends": None,
        }
        result = SAMPLE.validate_policy(
            [audited_run("cpu", 1.0)], "cpu", lane, policy
        )
        self.assertEqual(result["active_backends"], ["cpu"])
        with self.assertRaisesRegex(Exception, "owner decisions"):
            SAMPLE.validate_policy(
                [audited_run("cuda", 1.0)], "cpu", lane, policy
            )

    def test_process_schema_rejects_boolean_sizes(self) -> None:
        valid = {
            "exit_code": 0,
            "timeout": False,
            "output_limit": None,
            "spawn_error": None,
            "stdout_size_bytes": 10,
            "stderr_size_bytes": 0,
            "duration_seconds": 1.0,
        }
        SAMPLE._validate_process(valid)
        invalid = dict(valid, stdout_size_bytes=False)
        with self.assertRaisesRegex(Exception, "process record"):
            SAMPLE._validate_process(invalid)
        invalid = dict(valid, exit_code=False)
        with self.assertRaisesRegex(Exception, "process record"):
            SAMPLE._validate_process(invalid)

    def test_pending_terminal_is_replayed_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "report.json").write_text("{}", encoding="utf-8")
            report = {
                "case_path": "fixture.py",
                "topology": "cpu8",
                "sample_kind": "warmup",
                "cycle_index": 0,
                "receipt_id": "receipt",
            }
            with mock.patch.object(SAMPLE, "verify_complete") as replay:
                SAMPLE.publish_terminal(output, report)
            replay.assert_called_once_with(output, "PENDING_COMPLETE")
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            marker = json.loads((output / "COMPLETE").read_text())
            self.assertEqual(marker["sample_kind"], "warmup")


if __name__ == "__main__":
    unittest.main()
