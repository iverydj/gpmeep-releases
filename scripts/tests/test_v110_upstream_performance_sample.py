#!/usr/bin/env python3
"""Focused tests for the sealed upstream-Meep CPU M3 sample runner."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "scripts/python-validation/run_v110_upstream_performance_sample.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_v110_upstream_performance_sample", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
SAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLE
SPEC.loader.exec_module(SAMPLE)


class UpstreamPerformanceSampleTests(unittest.TestCase):
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
            Exception, "unsupported upstream performance comparison mode"
        ):
            SAMPLE.validate_comparison_evidence("", {"mode": "future"})

    def test_case_command_inserts_upstream_mode_before_oracle_options(self) -> None:
        command = [
            "/python",
            str(ROOT / "scripts/python-validation/run_example_oracle.py"),
            "--expected-run-count",
            "1",
            "/example.py",
        ]
        with mock.patch.object(
            SAMPLE.performance, "performance_oracle_command", return_value=command
        ):
            result = SAMPLE.upstream_case_command(
                {"path": "fixture.py"}, pathlib.Path("/python"), ROOT
            )
        self.assertEqual(result[2], "--upstream-cpu-baseline")
        self.assertEqual(result[3:], command[2:])

    def test_near2far_case_appends_only_explicit_upstream_capture(self) -> None:
        command = [
            "/python",
            str(ROOT / "scripts/python-validation/run_example_oracle.py"),
            "--expected-run-count",
            "10",
            "/near2far_3d_validation.py",
        ]
        with mock.patch.object(
            SAMPLE.performance, "performance_oracle_command", return_value=command
        ):
            result = SAMPLE.upstream_case_command(
                {"path": SAMPLE.NEAR2FAR_ADJOINT_CASE},
                pathlib.Path("/python"),
                ROOT,
            )
        self.assertEqual(
            result,
            [
                command[0],
                command[1],
                "--upstream-cpu-baseline",
                *command[2:],
                SAMPLE.NEAR2FAR_UPSTREAM_CAPTURE_ARGUMENT,
            ],
        )

    def test_near2far_capture_cannot_be_injected_by_manifest(self) -> None:
        command = [
            "/python",
            str(ROOT / "scripts/python-validation/run_example_oracle.py"),
            "/near2far_3d_validation.py",
            SAMPLE.NEAR2FAR_UPSTREAM_CAPTURE_ARGUMENT,
        ]
        with mock.patch.object(
            SAMPLE.performance, "performance_oracle_command", return_value=command
        ), self.assertRaisesRegex(Exception, "already contains"):
            SAMPLE.upstream_case_command(
                {"path": SAMPLE.NEAR2FAR_ADJOINT_CASE},
                pathlib.Path("/python"),
                ROOT,
            )

    def test_environment_contains_only_upstream_python_and_cpu_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            output = root / "samples" / "task"
            runtime_tmp = root / "runtime" / "task" / "tmp"
            output.mkdir(parents=True)
            runtime_tmp.mkdir(parents=True)
            paths = {
                "python": pathlib.Path(sys.executable).resolve(strict=True),
                "site_packages": pathlib.Path("/baseline/site-packages"),
                "libmeep": pathlib.Path("/baseline/lib/libmeep.so"),
                "exact_lock_prefix": pathlib.Path("/exact-lock"),
                "fontconfig": pathlib.Path("/baseline/fontconfig.conf"),
            }
            with mock.patch.object(
                SAMPLE.validation,
                "isolated_process_environment",
                side_effect=AssertionError("verifier ambient must not be sampled"),
            ), mock.patch.object(SAMPLE.pwd, "getpwuid") as account:
                account.return_value.pw_dir = "/tmp"
                environment = SAMPLE.clean_environment(
                    paths,
                    "nonce",
                    output=output,
                    case_path="python/examples/waveguide_crossing.py",
                )
        self.assertEqual(environment["PYTHONPATH"], "/baseline/site-packages")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["GPMEEP_VALIDATION_EXPECTED_BACKEND"], "cpu")
        self.assertEqual(environment["OMPI_MCA_osc"], "^ucx")
        self.assertEqual(environment["TMPDIR"], str(runtime_tmp))
        self.assertEqual(environment["LANG"], "C")
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertNotIn("LC_CTYPE", environment)
        self.assertNotIn("TERM", environment)
        self.assertNotIn("MEEP_GPU_BACKEND", environment)
        self.assertNotIn("GPMEEP_VALIDATION_BUILD_RECEIPT_ID", environment)
        self.assertNotIn(SAMPLE.performance.ADJOINT_CYL_FIXED_WORK_ENV, environment)
        self.assertNotIn(
            SAMPLE.performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV, environment
        )

        with mock.patch.object(
            SAMPLE.mpi_case,
            "expected_performance_producer_safe_ambient",
            return_value={},
        ), mock.patch.object(SAMPLE.pwd, "getpwuid") as account:
            account.return_value.pw_dir = "/tmp"
            fixed = SAMPLE.clean_environment(
                paths,
                "nonce",
                output=output,
                case_path="python/tests/test_adjoint_cyl.py",
            )
        self.assertEqual(
            fixed[SAMPLE.performance.ADJOINT_CYL_FIXED_WORK_ENV], "1"
        )

        with mock.patch.object(
            SAMPLE.mpi_case,
            "expected_performance_producer_safe_ambient",
            return_value={},
        ), mock.patch.object(SAMPLE.pwd, "getpwuid") as account:
            account.return_value.pw_dir = "/tmp"
            phase = SAMPLE.clean_environment(
                paths,
                "nonce",
                output=output,
                case_path=SAMPLE.performance.PHASE_IN_MATERIAL_CASE,
            )
        self.assertEqual(
            phase[SAMPLE.performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV], "1"
        )

    def test_zero_dimensional_case_is_the_only_cpu1_upstream_lane(self) -> None:
        topology, lane = SAMPLE.topology_and_lane(
            "python/examples/material-dispersion.py"
        )
        self.assertEqual(topology, "upstream-cpu1")
        self.assertEqual((lane.backend, lane.ranks, lane.devices), ("cpu", 1, ()))
        topology, lane = SAMPLE.topology_and_lane(
            "python/examples/phase_in_material.py"
        )
        self.assertEqual(topology, "upstream-cpu8")
        self.assertEqual((lane.backend, lane.ranks, lane.devices), ("cpu", 8, ()))

    def test_rank_bundle_requires_identities_and_forbids_statistics(self) -> None:
        for lane in (SAMPLE.DEFAULT_LANE, SAMPLE.ZERO_DIMENSIONAL_LANE):
            with (
                self.subTest(ranks=lane.ranks),
                tempfile.TemporaryDirectory() as raw,
            ):
                root = pathlib.Path(raw)
                for name in ("identity", "statistics", "work"):
                    (root / name).mkdir()
                command = ["/python", "/oracle.py"]
                nonce = "a" * 32
                for rank in range(lane.ranks):
                    work = root / "work" / f"rank-{rank:05d}"
                    work.mkdir()
                    identity = {
                        "schema": SAMPLE.mpi_case.rank_launcher.SCHEMA,
                        "rank": rank,
                        "world_size": lane.ranks,
                        "local_rank": rank,
                        "pid": 1000 + rank,
                        "executable": command[0],
                        "command": command,
                        "run_nonce": nonce,
                        "requested_backend": "cpu",
                        "visible_devices": "",
                        "statistics": str(
                            root / "statistics" / f"rank-{rank:05d}.json"
                        ),
                        "work": str(work),
                    }
                    (root / "identity" / f"rank-{rank:05d}.json").write_text(
                        json.dumps(identity), encoding="utf-8"
                    )
                records = SAMPLE.validate_rank_bundle(
                    lane, root, {"kind": "example"}, command, nonce
                )
                self.assertEqual(len(records), lane.ranks)
                (root / "statistics/rank-00000.json").write_text(
                    "{}", encoding="utf-8"
                )
                with self.assertRaisesRegex(Exception, "inventory"):
                    SAMPLE.validate_rank_bundle(
                        lane, root, {"kind": "example"}, command, nonce
                    )

    def test_upstream_unittest_uses_exact_mpi_suite_validator(self) -> None:
        lane = SAMPLE.DEFAULT_LANE
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            for name in ("identity", "statistics", "work"):
                (root / name).mkdir()
            (root / "stdout.log").write_text("", encoding="utf-8")
            (root / "stderr.log").write_text("", encoding="utf-8")
            command = ["/python", "/oracle.py"]
            nonce = "b" * 32
            for rank in range(lane.ranks):
                work = root / "work" / f"rank-{rank:05d}"
                work.mkdir()
                identity = {
                    "schema": SAMPLE.mpi_case.rank_launcher.SCHEMA,
                    "rank": rank,
                    "world_size": lane.ranks,
                    "local_rank": rank,
                    "pid": 2000 + rank,
                    "executable": command[0],
                    "command": command,
                    "run_nonce": nonce,
                    "requested_backend": "cpu",
                    "visible_devices": "",
                    "statistics": str(
                        root / "statistics" / f"rank-{rank:05d}.json"
                    ),
                    "work": str(work),
                }
                (root / "identity" / f"rank-{rank:05d}.json").write_text(
                    json.dumps(identity), encoding="utf-8"
                )
            case = {"kind": "unittest", "expected_unittest": {}}
            with mock.patch.object(
                SAMPLE.mpi_case,
                "validate_unittest_mpi_output",
                return_value={"outcome": "PASS"},
            ) as exact_suite:
                SAMPLE.validate_rank_bundle(
                    lane, root, case, command, nonce
                )
            exact_suite.assert_called_once_with(lane, root, case)

    def test_pending_terminal_is_replayed_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "report.json").write_text("{}", encoding="utf-8")
            report = {
                "case_path": "fixture.py",
                "sample_kind": "measured",
                "cycle_index": 1,
                "receipt_id": "receipt",
            }
            with mock.patch.object(SAMPLE, "verify_complete") as replay:
                SAMPLE.publish_terminal(output, report)
            replay.assert_called_once_with(output, "PENDING_COMPLETE")
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            self.assertTrue((output / "COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
