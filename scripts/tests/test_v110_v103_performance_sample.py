#!/usr/bin/env python3
"""Focused tests for the frozen-v1.0.3 side of the v1.1 M4 matrix."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_v110_v103_performance_sample.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_v110_v103_sample_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLE
SPEC.loader.exec_module(SAMPLE)


class V103PerformanceSampleTests(unittest.TestCase):
    def test_evidence_code_seals_cylindrical_fixed_work_contract(self) -> None:
        records = SAMPLE._code_records(
            {
                "repo": ROOT,
                "launcher": ROOT / "scripts/python-validation/m3_mpi_rank_launcher.py",
                "manifest": ROOT / "scripts/python-validation/manifest.json",
            },
            "python/examples/phase_in_material.py",
        )
        self.assertEqual(
            records["adjoint_cyl_fixed_work_contract"]["path"],
            str(
                (
                    ROOT / "python/tests/adjoint_cyl_fixed_work_contract.py"
                ).resolve()
            ),
        )

    def _live_fixture(
        self, root: pathlib.Path, *, stderr: str = ""
    ) -> tuple[dict[str, object], dict[str, pathlib.Path], types.SimpleNamespace]:
        report: dict[str, object] = {
            "report_id": "frozen-report",
            "package_sha256": SAMPLE.identity.FROZEN_PACKAGE_SHA256,
        }
        prefix = root / "prefix"
        home = root / "home"
        prefix.mkdir()
        home.mkdir()
        paths = {
            "install_prefix": prefix,
            "verifier": prefix / "bin/gpmeep-verify-provenance",
        }
        process = types.SimpleNamespace(
            returncode=0,
            stdout="verifier summary\n" + json.dumps(report) + "\n",
            stderr=stderr,
        )
        return report, paths, process

    def test_live_verifier_excludes_only_osc_ucx_in_minimal_child_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            report, paths, process = self._live_fixture(root)
            original_osc = os.environ.get("OMPI_MCA_osc")
            with (
                mock.patch.object(
                    SAMPLE.validation,
                    "isolated_process_environment",
                    side_effect=AssertionError("verifier ambient must not be sampled"),
                ),
                mock.patch.object(
                    SAMPLE.pwd,
                    "getpwuid",
                    return_value=types.SimpleNamespace(pw_dir=str(root / "home")),
                ),
                mock.patch.object(
                    SAMPLE.subprocess, "run", return_value=process
                ) as run,
            ):
                result = SAMPLE.live_verify(report, paths)
            self.assertEqual(result["state"], "PASS")
            self.assertEqual(
                run.call_args.kwargs["env"],
                {
                    "HOME": str((root / "home").resolve()),
                    "PATH": f"{paths['install_prefix'] / 'bin'}:/usr/bin:/bin",
                    "LD_LIBRARY_PATH": str(paths["install_prefix"] / "lib"),
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "MEEP_GPU_BACKEND": "cpu",
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMPI_MCA_osc": "^ucx",
                },
            )
            self.assertEqual(os.environ.get("OMPI_MCA_osc"), original_osc)

    def test_lane_environment_uses_only_controller_derived_ambient(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            output = root / "samples" / "task"
            runtime_tmp = root / "runtime" / "task" / "tmp"
            output.mkdir(parents=True)
            runtime_tmp.mkdir(parents=True)
            python = pathlib.Path(sys.executable).resolve(strict=True)
            paths = {
                "python": python,
                "site_packages": root / "site-packages",
                "provenance_report": root / "provenance.json",
                "package": root / "package.conda",
            }
            lane = SAMPLE.mpi_case.Lane("cuda", "cuda", 1, ("GPU-a",))
            with mock.patch.object(
                SAMPLE.mpi_case,
                "clean_environment",
                side_effect=lambda *args, **kwargs: {},
            ) as cleaner, mock.patch.object(
                SAMPLE.validation,
                "isolated_process_environment",
                side_effect=AssertionError("verifier ambient must not be sampled"),
            ):
                environment = SAMPLE.clean_environment(
                    lane,
                    paths,
                    {},
                    {"report_id": "report"},
                    "nonce",
                    output=output,
                    case_path="python/examples/waveguide_crossing.py",
                )
                phase_environment = SAMPLE.clean_environment(
                    lane,
                    paths,
                    {},
                    {"report_id": "report"},
                    "nonce",
                    output=output,
                    case_path=SAMPLE.performance.PHASE_IN_MATERIAL_CASE,
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
            self.assertEqual(environment[SAMPLE.identity.ENV_REPORT_ID], "report")
            self.assertNotIn(
                SAMPLE.performance.ADJOINT_CYL_FIXED_WORK_ENV, environment
            )
            self.assertNotIn(
                SAMPLE.performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV,
                environment,
            )
            self.assertEqual(
                phase_environment[
                    SAMPLE.performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV
                ],
                "1",
            )

    def test_live_verifier_still_rejects_every_unrelated_stderr_byte(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            report, paths, process = self._live_fixture(
                root, stderr="unexpected verifier warning\n"
            )
            with (
                mock.patch.object(
                    SAMPLE.validation,
                    "isolated_process_environment",
                    return_value={},
                ),
                mock.patch.object(
                    SAMPLE.pwd,
                    "getpwuid",
                    return_value=types.SimpleNamespace(pw_dir=str(root / "home")),
                ),
                mock.patch.object(SAMPLE.subprocess, "run", return_value=process),
                self.assertRaisesRegex(Exception, "live provenance differs"),
            ):
                SAMPLE.live_verify(report, paths)

    def test_real_clean_v103_package_live_verifier_when_requested(self) -> None:
        values = {
            name: os.environ.get(name)
            for name in (
                "GPMEEP_V110_REAL_V103_PREFIX",
                "GPMEEP_V110_REAL_V103_REPORT",
                "GPMEEP_V110_REAL_V103_PACKAGE",
            )
        }
        if any(value is None for value in values.values()):
            self.skipTest("real frozen-v1.0.3 authority was not supplied")
        args = types.SimpleNamespace(
            repo=ROOT,
            manifest=ROOT / "scripts/python-validation/manifest.json",
            install_prefix=pathlib.Path(values["GPMEEP_V110_REAL_V103_PREFIX"]),
            provenance_report=pathlib.Path(
                values["GPMEEP_V110_REAL_V103_REPORT"]
            ),
            package=pathlib.Path(values["GPMEEP_V110_REAL_V103_PACKAGE"]),
        )
        report, paths, _contract = SAMPLE.runtime_paths(args)
        result = SAMPLE.live_verify(report, paths)
        self.assertEqual(
            result,
            {
                "command": [
                    str(paths["verifier"]),
                    "--prefix",
                    str(paths["install_prefix"]),
                    "--expected-source-commit",
                    SAMPLE.identity.FROZEN_SOURCE_COMMIT,
                    "--expected-package-sha256",
                    SAMPLE.identity.FROZEN_PACKAGE_SHA256,
                ],
                "report_id": report["report_id"],
                "package_sha256": SAMPLE.identity.FROZEN_PACKAGE_SHA256,
                "state": "PASS",
            },
        )

    def test_gpu2_is_limited_to_the_four_explicit_plan_cases(self) -> None:
        devices = ("GPU-a", "GPU-b")
        lane = SAMPLE.select_lane("v103-cuda1", devices, {"gpu2": False})
        self.assertEqual((lane.ranks, lane.devices), (1, ("GPU-a",)))
        lane = SAMPLE.select_lane("v103-cuda2", devices, {"gpu2": True})
        self.assertEqual((lane.ranks, lane.devices), (2, devices))
        with self.assertRaisesRegex(Exception, "pairing"):
            SAMPLE.select_lane("v103-cuda2", devices, {"gpu2": False})

    def test_historical_command_uses_only_provenance_checking_oracle(self) -> None:
        ordinary = ROOT / "scripts/python-validation/run_example_oracle.py"
        command = ["/prefix/bin/python3.11", str(ordinary), "--expected-run-count", "1", "/example.py"]
        with mock.patch.object(
            SAMPLE.performance, "performance_oracle_command", return_value=command
        ):
            result = SAMPLE.legacy_case_command(
                {"path": "fixture.py"}, pathlib.Path(command[0]), ROOT
            )
        self.assertEqual(
            pathlib.Path(result[1]).name, "run_v103_example_oracle.py"
        )
        self.assertNotIn("--acknowledge-backend-mismatch-reason", result)
        self.assertNotIn("--upstream-cpu-baseline", result)
        self.assertFalse(
            SAMPLE.mpi_case.site_guard_expected_for_command(result)
        )

    def test_only_generic_oracles_disable_reason_bound_site_guard(self) -> None:
        ordinary = ["python", "/repo/run_example_oracle.py", "/repo/case.py"]
        frozen = ["python", "/repo/run_v103_example_oracle.py", "/repo/case.py"]
        native = ["python", "/repo/custom_validation_driver.py", "/repo/case.py"]
        self.assertFalse(
            SAMPLE.mpi_case.site_guard_expected_for_command(ordinary)
        )
        self.assertFalse(
            SAMPLE.mpi_case.site_guard_expected_for_command(frozen)
        )
        self.assertTrue(
            SAMPLE.mpi_case.site_guard_expected_for_command(native)
        )

    def test_process_identity_requires_exactly_one_exact_payload(self) -> None:
        expected = {"schema": "identity", "report_id": "r"}
        prefix = SAMPLE.auto_case.oracle.V103_IDENTITY_PREFIX
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "stdout.log"
            output.write_text(prefix + json.dumps(expected) + "\n", encoding="utf-8")
            self.assertEqual(SAMPLE.parse_process_identity(output, expected), expected)
            output.write_text(
                (prefix + json.dumps(expected) + "\n") * 2, encoding="utf-8"
            )
            with self.assertRaisesRegex(Exception, "count"):
                SAMPLE.parse_process_identity(output, expected)

    def test_plan_rejects_any_frozen_package_rebuild_permission(self) -> None:
        plan = {
            "schema": "gpmeep-v110-m4-equal-work-plan-v1",
            "release_train": "1.1.0",
            "frozen_v103": {
                "package_name": SAMPLE.identity.FROZEN_PACKAGE_NAME,
                "package_sha256": SAMPLE.identity.FROZEN_PACKAGE_SHA256,
                "provenance_report_sha256": (
                    SAMPLE.identity.FROZEN_PROVENANCE_REPORT_SHA256
                ),
                "provenance_report_id": SAMPLE.identity.FROZEN_PROVENANCE_REPORT_ID,
                "source_commit": SAMPLE.identity.FROZEN_SOURCE_COMMIT,
                "rebuild_permitted": True,
                "fresh_dedicated_prefix_required": True,
            },
        }
        with mock.patch.object(SAMPLE.validation, "load_json", return_value=plan):
            with self.assertRaisesRegex(Exception, "plan differs"):
                SAMPLE._load_plan(ROOT)


if __name__ == "__main__":
    unittest.main()
