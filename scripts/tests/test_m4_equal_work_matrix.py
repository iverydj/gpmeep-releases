#!/usr/bin/env python3
"""Unit tests for the M4 controller without launching simulation workloads."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_m4_equal_work_matrix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m4_equal_work_matrix_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def waveguide_diagnostic_difference_summary(
    out_of_tolerance_count: int = 109,
) -> dict[str, object]:
    metric_counts = {
        "optimizer_smoke.epoch0_design[[]*]": 441,
        "optimizer_smoke.final_design[[]*]": 441,
        "optimizer_smoke.*_design_step_l2": 2,
        "optimizer_smoke.*_gradient_l2_*": 4,
        "optimizer_smoke.*_objective_*": 4,
        "optimizer_smoke.final_design_l2": 1,
    }
    summary = {}
    for index, (rule, metric_count) in enumerate(metric_counts.items()):
        summary[rule] = {
            "arithmetic_overflow_count": 0,
            "failure_count": 0,
            "max_absolute_error": 0.0,
            "max_absolute_error_metric": rule,
            "max_absolute_error_overflow": False,
            "max_absolute_error_record": None,
            "max_normalized_error": 0.0,
            "max_normalized_error_metric": rule,
            "max_normalized_error_overflow": False,
            "max_normalized_error_record": None,
            "metric_count": metric_count,
            "zero_tolerance_failure_count": 0,
            "diagnostic_only": True,
            "diagnostic_only_out_of_tolerance_count": (
                out_of_tolerance_count if index == 0 else 0
            ),
        }
    return summary


def contract() -> dict[str, object]:
    plan = MATRIX.equal_work_plan.load_plan(ROOT)
    return {
        "paths": {
            "python": "/runtime/python",
            "repo": str(ROOT),
            "manifest": str(ROOT / "scripts/python-validation/manifest.json"),
            "v110_runner": "/runner/v110.py",
            "upstream_runner": "/runner/upstream.py",
            "v103_runner": "/runner/v103.py",
            "upstream_receipt": "/receipts/upstream.json",
            "v103_install_prefix": "/runtime/v103",
            "v103_provenance_report": "/receipts/v103.json",
            "v103_package": "/packages/v103.tar.bz2",
            "build_python": "/build/python",
            "install_prefix": "/runtime/v110",
            "build_receipt": "/receipts/v110.json",
            "mpiexec": "/runtime/mpirun",
        },
        "plan": plan,
        "tasks": MATRIX.equal_work_plan.build_tasks(plan),
        "gpu_devices": [
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        ],
        "runtime_inputs": {
            "v110_installed_report_id": "installed-package-report",
            "v110_installed_runtime": {
                "python": "/installed/v110/bin/python",
                "build_python": "/installed/v110/lib/python3.11/site-packages",
                "install_prefix": "/installed/v110",
                "build_receipt": "/receipts/v110-installed.json",
                "mpiexec": "/installed/v110/bin/mpirun",
                "package": "/packages/v110.conda",
                "provenance_report": "/receipts/v110-installed.json",
                "verifier": "/installed/v110/bin/gpmeep-verify-provenance",
            },
        },
        "limits": {
            "task_timeout_seconds": 7200,
            "stdout_limit_bytes": 64 * 1024**2,
            "telemetry_interval_seconds": 1.0,
        },
    }


def sealed_authority_fixture(
    output: pathlib.Path,
    *,
    task_count: int = MATRIX.TASK_COUNT,
    numerical_outcome: str = "PASS",
) -> dict[str, pathlib.Path]:
    """Create a compact, internally sealed authority tree for bounded replay."""

    fixture_contract = {
        "schema": MATRIX.SCHEMA,
        "paths": {"repo": str(ROOT)},
        "plan": {},
        "tasks": [
            {"task_index": index, "task_id": f"{index:064x}"}
            for index in range(task_count)
        ],
    }
    completed = [
        {
            "task": {"task_index": index},
            "process": {"returncode": 0},
            "sample_replay": {
                "complete": {"sha256": f"{index:064x}"},
                "report": {
                    "path": f"synthetic/report-{index:03d}.json",
                    "sha256": f"{index + 1:064x}",
                },
            },
        }
        for index in range(task_count)
    ]
    first_event = {"state": "controller-started", "unix_seconds": 1.0}
    events = [first_event]
    for retained in completed:
        events.extend(
            [
                {"state": "task-started", "task": retained["task"]},
                {
                    "state": "task-process-ended",
                    "task": retained["task"],
                    "process": retained["process"],
                },
                {
                    "state": "task-validated",
                    "task": retained["task"],
                    "sample_complete": retained["sample_replay"]["complete"],
                },
            ]
        )
    journal = {
        "schema": MATRIX.JOURNAL_SCHEMA,
        "contract": fixture_contract,
        "events": events,
    }
    checkpoint = {
        "schema": MATRIX.CHECKPOINT_SCHEMA,
        "contract": fixture_contract,
        "hardware_before": {},
        "capacity_before": {},
        "completed_tasks": completed,
    }
    inversion_values = [
        {"schema": "fixture-inversion-v1", "case_index": index}
        for index in range(13)
    ]
    performance = {
        "outcome": "PASS",
        "task_count": MATRIX.TASK_COUNT,
        "measured_task_count": 120,
        "cases": {f"case-{index:02d}": {} for index in range(13)},
        "inversion_records": inversion_values,
    }
    comparisons = [
        {
            "comparison_index": index,
            "result": {
                "outcome": numerical_outcome if index == 0 else "PASS",
                "failure_count": 0 if numerical_outcome == "PASS" or index else 1,
            },
        }
        for index in range(141)
    ]
    numerical = {
        "outcome": "PASS",
        "comparison_count": 141,
        "comparisons": comparisons,
    }
    paths = {
        "journal": output / "JOURNAL.json",
        "checkpoint": output / "CHECKPOINT.json",
        "performance": output / "performance-replay.json",
        "numerical": output / "numerical-replay.json",
    }
    MATRIX.atomic_write_json(paths["journal"], journal)
    MATRIX.atomic_write_json(paths["checkpoint"], checkpoint)
    MATRIX.atomic_write_json(paths["performance"], performance)
    MATRIX.atomic_write_json(paths["numerical"], numerical)
    inversion_records = []
    for index, value in enumerate(inversion_values):
        path = output / "inversion-records" / f"case-{index:02d}.json"
        MATRIX.atomic_write_json(path, value)
        inversion_records.append(MATRIX.file_record(path, output))
    report = {
        "schema": MATRIX.SCHEMA,
        "outcome": "PASS",
        "contract": fixture_contract,
        "tasks": completed,
        "performance": performance,
        "numerical": numerical,
        "hardware": {"before": {}, "after": {}},
        "capacity_initial": {},
        "capacity_final": {},
        "resume_history": [],
        "checkpoint": MATRIX.file_record(paths["checkpoint"], output),
        "performance_replay": MATRIX.file_record(paths["performance"], output),
        "numerical_replay": MATRIX.file_record(paths["numerical"], output),
        "inversion_records": inversion_records,
    }
    paths["report"] = output / "report.json"
    MATRIX.atomic_write_json(paths["report"], report)
    paths["terminal"] = output / "M4_COMPLETE"
    MATRIX.atomic_write_json(paths["terminal"], MATRIX._terminal_payload(output, report))
    paths["inversion"] = output / "inversion-records" / "case-00.json"
    return paths


class M4EqualWorkMatrixTests(unittest.TestCase):
    def test_m4_telemetry_default_retains_one_second_cuda_observation(self) -> None:
        args = MATRIX.parse_args(["--output", "/tmp/m4-output"])
        self.assertEqual(args.telemetry_interval_seconds, 1.0)

    def test_nvidia_595_open_hmm_enabled_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            version = root / "version"
            hmm = root / "uvm_disable_hmm"
            version.write_text(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
                "595.71.05  Release Build\n",
                encoding="utf-8",
            )
            hmm.write_text("N\n", encoding="ascii")
            with self.assertRaisesRegex(
                MATRIX.WorkloadError,
                "refuses NVIDIA 595 open-driver with UVM HMM enabled",
            ):
                MATRIX.nvidia_uvm_hmm_binding(version, hmm)

    def test_nvidia_595_open_hmm_disabled_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            version = root / "version"
            hmm = root / "uvm_disable_hmm"
            version.write_text(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
                "595.91.07  Release Build\n",
                encoding="utf-8",
            )
            hmm.write_text("Y\n", encoding="ascii")
            self.assertEqual(
                MATRIX.nvidia_uvm_hmm_binding(version, hmm),
                {
                    "schema": "gpmeep-v110-nvidia-uvm-hmm-safety-v1",
                    "driver_version": "595.91.07",
                    "driver_module_flavor": "open",
                    "hmm_requirement": "must_be_disabled",
                    "uvm_disable_hmm": True,
                    "blocked_open_driver_major_versions": [595],
                },
            )

    def test_nvidia_proprietary_or_non595_is_not_overblocked(self) -> None:
        fixtures = (
            ("NVRM version: NVIDIA UNIX x86_64 Kernel Module 595.71.05\n", "N"),
            (
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 "
                "600.1.2\n",
                "N",
            ),
        )
        for version_text, hmm_text in fixtures:
            with self.subTest(version=version_text):
                with tempfile.TemporaryDirectory() as directory:
                    root = pathlib.Path(directory)
                    version = root / "version"
                    hmm = root / "uvm_disable_hmm"
                    version.write_text(version_text, encoding="utf-8")
                    hmm.write_text(hmm_text, encoding="ascii")
                    record = MATRIX.nvidia_uvm_hmm_binding(version, hmm)
                    self.assertIsNone(record["uvm_disable_hmm"])
                    self.assertEqual(record["hmm_requirement"], "not_required")

    def test_nonrisk_driver_records_absent_uvm_module_without_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            version = root / "version"
            version.write_text(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 "
                "600.1.2\n",
                encoding="utf-8",
            )
            record = MATRIX.nvidia_uvm_hmm_binding(
                version, root / "absent-uvm-disable-hmm"
            )
            self.assertIsNone(record["uvm_disable_hmm"])
            self.assertEqual(record["hmm_requirement"], "not_required")

    def test_unknown_nvidia_kernel_module_flavor_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            version = root / "version"
            version.write_text(
                "NVRM version: vendor wrapper 595.71.05\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "kernel-module flavor is unknown"
            ):
                MATRIX.nvidia_uvm_hmm_binding(version, root / "absent")

    def test_nvidia_uvm_hmm_malformed_or_missing_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            version = root / "version"
            hmm = root / "uvm_disable_hmm"
            version.write_text("no driver version here\n", encoding="utf-8")
            hmm.write_text("Y\n", encoding="ascii")
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "driver version is malformed"
            ):
                MATRIX.nvidia_uvm_hmm_binding(version, hmm)
            version.write_text(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 "
                "595.71.05\n",
                encoding="utf-8",
            )
            hmm.write_text("unknown\n", encoding="ascii")
            with self.assertRaisesRegex(MATRIX.WorkloadError, "state is malformed"):
                MATRIX.nvidia_uvm_hmm_binding(version, hmm)
            hmm.unlink()
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "cannot verify NVIDIA UVM/HMM"
            ):
                MATRIX.nvidia_uvm_hmm_binding(version, hmm)

    def test_build_contract_rejects_unsafe_host_before_any_runtime_probe(
        self,
    ) -> None:
        with (
            mock.patch.object(
                MATRIX,
                "host_binding",
                side_effect=MATRIX.WorkloadError("unsafe HMM fixture"),
            ),
            mock.patch.object(MATRIX, "_runtime_paths") as runtime_paths,
            mock.patch.object(MATRIX.provenance, "verify_build_receipt") as receipt,
            self.assertRaisesRegex(MATRIX.WorkloadError, "unsafe HMM fixture"),
        ):
            MATRIX.build_contract(
                types.SimpleNamespace(),
                ROOT,
                {},
                [],
                {},
                (
                    "GPU-00000000-0000-0000-0000-000000000001",
                    "GPU-00000000-0000-0000-0000-000000000002",
                ),
            )
        runtime_paths.assert_not_called()
        receipt.assert_not_called()

    def test_validate_contract_rejects_unsafe_host_before_live_replay(
        self,
    ) -> None:
        sealed = contract()
        sealed.update(
            {
                "schema": MATRIX.SCHEMA,
                "evidence_code": {},
                "host_binding": {},
                "nvidia_smi": {},
                "gates": {},
            }
        )
        with (
            mock.patch.object(
                MATRIX,
                "host_binding",
                side_effect=MATRIX.WorkloadError("unsafe replay host fixture"),
            ),
            mock.patch.object(MATRIX, "_v110_installed_identity") as v110_live,
            mock.patch.object(MATRIX, "_v103_installed_identity") as v103_live,
            mock.patch.object(MATRIX, "_unittest_import_preflight") as imports,
            self.assertRaisesRegex(
                MATRIX.WorkloadError, "unsafe replay host fixture"
            ),
        ):
            MATRIX.validate_contract(sealed)
        v110_live.assert_not_called()
        v103_live.assert_not_called()
        imports.assert_not_called()

    def test_controller_dependency_preflight_is_fail_closed(self) -> None:
        MATRIX.validate_controller_dependencies()
        with mock.patch.object(
            MATRIX.importlib, "import_module", side_effect=ImportError("absent")
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "requires the pinned jsonschema dependency"
            ):
                MATRIX.validate_controller_dependencies()
        with mock.patch.object(
            MATRIX.importlib, "import_module", return_value=types.SimpleNamespace()
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "jsonschema capability differs"
            ):
                MATRIX.validate_controller_dependencies()

    @staticmethod
    def bounded_authority_mocks(output: pathlib.Path) -> ExitStack:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        tasks = report["contract"]["tasks"]
        performance = dict(report["performance"])
        inversions = performance.pop("inversion_records")
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                MATRIX, "validate_contract", return_value=(ROOT, tasks, ("gpu-a", "gpu-b"))
            )
        )
        stack.enter_context(
            mock.patch.object(MATRIX, "_bounded_child_context", return_value={})
        )
        stack.enter_context(
            mock.patch.object(
                MATRIX,
                "_validate_completed_task_authority",
                side_effect=[{} for _task in tasks],
            )
        )
        stack.enter_context(
            mock.patch.object(
                MATRIX.equal_work_replay, "derive_results", return_value=performance
            )
        )
        stack.enter_context(
            mock.patch.object(
                MATRIX.equal_work_replay,
                "derive_inversion_records",
                return_value=inversions,
            )
        )
        stack.enter_context(
            mock.patch.object(MATRIX, "_validate_bounded_numerical_replay")
        )
        stack.enter_context(
            mock.patch.object(MATRIX, "validate_normative_inversion_registry")
        )
        stack.enter_context(
            mock.patch.object(MATRIX.matrix, "validate_hardware_evidence")
        )
        stack.enter_context(mock.patch.object(MATRIX, "validate_capacity"))
        return stack

    def test_experiment_authority_rejects_a_self_sealed_but_unvalidated_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output)
            with self.assertRaisesRegex(Exception, "sealed contract schema differs"):
                MATRIX.verify_experiment_authority(output)

    def test_experiment_authority_accepts_already_validated_transitive_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output)
            with self.bounded_authority_mocks(output):
                report = MATRIX.verify_experiment_authority(output)
            self.assertEqual(report["outcome"], "PASS")
            self.assertEqual(len(report["tasks"]), MATRIX.TASK_COUNT)

    def test_experiment_authority_rejects_terminal_bound_file_mutation(self) -> None:
        for name in ("report", "performance", "numerical", "inversion"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output = pathlib.Path(temporary).resolve()
                paths = sealed_authority_fixture(output)
                paths[name].write_text("{}\n", encoding="utf-8")
                if name == "report":
                    with self.assertRaises(Exception):
                        MATRIX.verify_experiment_authority(output)
                else:
                    with self.bounded_authority_mocks(output):
                        with self.assertRaises(Exception):
                            MATRIX.verify_experiment_authority(output)

    def test_experiment_authority_rejects_self_consistent_numerical_failure(self) -> None:
        tasks = [
            {
                "lane_id": "candidate",
                "case_path": f"case-{index:03d}",
                "cycle_index": 1,
                "sample_kind": "measured",
            }
            for index in range(141)
        ]
        rows = [
            {
                "cycle_index": task["cycle_index"],
                "sample_kind": task["sample_kind"],
                "case_path": task["case_path"],
                "reference_lane": "upstream_meep_cpu",
                "candidate_lane": task["lane_id"],
                "method": "fixture",
                "result": {
                    "outcome": "FAIL" if index == 0 else "PASS",
                    "failure_count": 1 if index == 0 else 0,
                },
            }
            for index, task in enumerate(tasks)
        ]
        numerical = {
            "schema": "gpmeep-v110-m4-numerical-replay-v1",
            "outcome": "PASS",
            "comparison_count": 141,
            "comparisons": rows,
        }
        with self.assertRaisesRegex(Exception, "numerical comparison differs"):
            MATRIX._validate_bounded_numerical_replay(numerical, tasks)

    def test_bounded_replay_accepts_waveguide_rows_and_rejects_census_attacks(
        self,
    ) -> None:
        tasks = [
            {
                "lane_id": "gpmeep_v110_cpu",
                "case_path": MATRIX.WAVEGUIDE_CROSSING_CASE,
                "cycle_index": 1,
                "sample_kind": "measured",
            },
            {
                "lane_id": "gpmeep_v110_cuda_gpu1",
                "case_path": MATRIX.WAVEGUIDE_CROSSING_CASE,
                "cycle_index": 1,
                "sample_kind": "measured",
            },
            *(
                {
                    "lane_id": "candidate",
                    "case_path": f"case-{index:03d}",
                    "cycle_index": 1,
                    "sample_kind": "measured",
                }
                for index in range(139)
            ),
        ]

        def compact(count: int = 1, digest: str = "1" * 64) -> dict:
            return {
                "outcome": "PASS",
                "reason": "within-tolerance",
                "metric_evidence": {
                    "schema": "gpmeep-json-metric-evidence-v2",
                    "count": count,
                    "passed_count": count,
                    "failure_count": 0,
                    "metric_keys_sha256": digest,
                    "cpu_sha256": "2" * 64,
                    "cuda_sha256": "3" * 64,
                    "comparison_records_sha256": "4" * 64,
                    "inline_limit": 4096,
                    "inline_byte_limit": 1048576,
                    "flattened_storage_bytes": 1,
                    "inlined": True,
                },
                "difference_summary": {},
                "failure_count": 0,
                "failures_reported": 0,
                "failures_truncated": False,
                "failure_records_sha256": hashlib.sha256(
                    b"gpmeep-json-failure-records-v1\0"
                ).hexdigest(),
                "failures": [],
            }

        self_oracle = {
            "outcome": "PASS",
            "design_size": MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE,
            "design_minimum": 0.4,
            "design_maximum": 0.6,
            "computed_epoch0_design_step_l2": 0.1,
            "reported_epoch0_design_step_l2": 0.1,
            "computed_epoch1_design_step_l2": 0.2,
            "reported_epoch1_design_step_l2": 0.2,
            "computed_final_design_l2": 10.5,
            "reported_final_design_l2": 10.5,
            "structural_metrics": {
                **{
                    f"fixed_work.run_timesteps[{index}]": float(
                        MATRIX.WAVEGUIDE_FIXED_SLICE_TIMESTEP
                    )
                    for index in range(
                        MATRIX.WAVEGUIDE_FIXED_SLICE_RUN_COUNT
                    )
                },
                **{
                    f"optimizer_smoke.run_timesteps[{index}]": float(
                        MATRIX.WAVEGUIDE_OPTIMIZER_TIMESTEP
                    )
                    for index in range(MATRIX.WAVEGUIDE_OPTIMIZER_RUN_COUNT)
                },
                "optimizer_smoke.beta_initial": 4.0,
                "optimizer_smoke.beta_final": 8.0,
                "optimizer_smoke.epoch_count": 2.0,
                "optimizer_smoke.epoch0_beta": 4.0,
                "optimizer_smoke.epoch1_beta": 8.0,
                "optimizer_smoke.epoch0_callback_count": 2.0,
                "optimizer_smoke.epoch1_callback_count": 2.0,
                "optimizer_smoke.epoch0_nlopt_evaluations": 2.0,
                "optimizer_smoke.epoch1_nlopt_evaluations": 2.0,
                "optimizer_smoke.epoch0_result_code": 5.0,
                "optimizer_smoke.epoch1_result_code": 5.0,
                "optimizer_smoke.checkpoint_callback_count": 4.0,
                "optimizer_smoke.checkpoint_completed_epochs": 2.0,
                "optimizer_smoke.checkpoint_roundtrip_max_abs_difference": 0.0,
                "optimizer_smoke.result_roundtrip_max_abs_difference": 0.0,
            },
        }
        rows = []
        for task in tasks:
            if task["case_path"] == MATRIX.WAVEGUIDE_CROSSING_CASE:
                pairwise = compact(
                    MATRIX.WAVEGUIDE_EXPECTED_METRIC_COUNT,
                    MATRIX.WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256,
                )
                if task["lane_id"] == "gpmeep_v110_cpu":
                    pairwise["metric_evidence"].update(
                        diagnostic_only_count=(
                            MATRIX.WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
                        ),
                        diagnostic_only_out_of_tolerance_count=109,
                    )
                    pairwise["difference_summary"] = (
                        waveguide_diagnostic_difference_summary()
                    )
                result = {
                    "outcome": "PASS",
                    "failure_count": 0,
                    "pairwise": pairwise,
                    "reference_self_oracle": copy.deepcopy(self_oracle),
                    "candidate_self_oracle": copy.deepcopy(self_oracle),
                }
            else:
                result = compact()
            rows.append(
                {
                    "cycle_index": task["cycle_index"],
                    "sample_kind": task["sample_kind"],
                    "case_path": task["case_path"],
                    "reference_lane": (
                        "gpmeep_v110_cpu"
                        if task["case_path"] == MATRIX.WAVEGUIDE_CROSSING_CASE
                        and task["lane_id"] != "gpmeep_v110_cpu"
                        else "upstream_meep_cpu"
                    ),
                    "candidate_lane": task["lane_id"],
                    "method": MATRIX._expected_numerical_method(
                        task["case_path"], task["lane_id"]
                    ),
                    "result": result,
                }
            )
        baseline = {
            "schema": "gpmeep-v110-m4-numerical-replay-v1",
            "outcome": "PASS",
            "comparison_count": 141,
            "comparisons": rows,
        }
        MATRIX._validate_bounded_numerical_replay(baseline, tasks)

        for field, value in (
            ("count", 929),
            ("passed_count", 929),
            ("metric_keys_sha256", "9" * 64),
        ):
            with self.subTest(field=field):
                attacked = copy.deepcopy(baseline)
                attacked["comparisons"][0]["result"]["pairwise"][
                    "metric_evidence"
                ][field] = value
                with self.assertRaises(Exception):
                    MATRIX._validate_bounded_numerical_replay(attacked, tasks)

        inconsistent_progress = copy.deepcopy(baseline)
        inconsistent_progress["comparisons"][1]["result"][
            "candidate_self_oracle"
        ]["reported_epoch1_design_step_l2"] = 0.3
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(
                inconsistent_progress, tasks
            )

        hidden_diagnostic = copy.deepcopy(baseline)
        hidden_pairwise = hidden_diagnostic["comparisons"][1]["result"][
            "pairwise"
        ]
        hidden_pairwise["difference_summary"] = (
            waveguide_diagnostic_difference_summary()
        )
        with self.assertRaisesRegex(Exception, "diagnostic rules were hidden"):
            MATRIX._validate_bounded_numerical_replay(
                hidden_diagnostic, tasks
            )

    def test_bounded_numerical_replay_rejects_equivalent_types_methods_and_nested_failures(self) -> None:
        tasks = [
            {
                "lane_id": "candidate",
                "case_path": f"case-{index:03d}",
                "cycle_index": 1,
                "sample_kind": "measured",
            }
            for index in range(141)
        ]

        def compact() -> dict:
            return {
                "outcome": "PASS",
                "reason": "within-tolerance",
                "metric_evidence": {
                    "schema": "gpmeep-json-metric-evidence-v2",
                    "count": 1,
                    "passed_count": 1,
                    "failure_count": 0,
                    "metric_keys_sha256": "1" * 64,
                    "cpu_sha256": "2" * 64,
                    "cuda_sha256": "3" * 64,
                    "comparison_records_sha256": "4" * 64,
                    "inline_limit": 4096,
                    "inline_byte_limit": 1048576,
                    "flattened_storage_bytes": 1,
                    "inlined": True,
                },
                "difference_summary": {},
                "failure_count": 0,
                "failures_reported": 0,
                "failures_truncated": False,
                "failure_records_sha256": hashlib.sha256(
                    b"gpmeep-json-failure-records-v1\0"
                ).hexdigest(),
                "failures": [],
            }

        rows = []
        for task in tasks:
            result = compact()
            rows.append(
                {
                    "cycle_index": task["cycle_index"],
                    "sample_kind": task["sample_kind"],
                    "case_path": task["case_path"],
                    "reference_lane": "upstream_meep_cpu",
                    "candidate_lane": task["lane_id"],
                    "method": MATRIX._expected_numerical_method(
                        task["case_path"], task["lane_id"]
                    ),
                    "result": result,
                }
            )
        baseline = {
            "schema": "gpmeep-v110-m4-numerical-replay-v1",
            "outcome": "PASS",
            "comparison_count": 141,
            "comparisons": rows,
        }
        MATRIX._validate_bounded_numerical_replay(baseline, tasks)

        malformed = copy.deepcopy(baseline)
        malformed["comparison_count"] = 141.0
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        malformed = copy.deepcopy(baseline)
        malformed["comparisons"][1]["method"] = "fabricated-method"
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        malformed = copy.deepcopy(baseline)
        malformed["comparisons"][0]["result"]["failure_count"] = False
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        malformed = copy.deepcopy(baseline)
        malformed["comparisons"][0]["result"]["difference_summary"] = {
            "nested": {"outcome": "FAIL", "failure_count": 99}
        }
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        malformed = copy.deepcopy(baseline)
        malformed["comparisons"][1]["result"]["diagnostic_only_count"] = False
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        malformed = copy.deepcopy(baseline)
        malformed["comparisons"][1]["result"][
            "diagnostic_only_count"
        ] = 1
        malformed["comparisons"][1]["result"][
            "diagnostic_only_out_of_tolerance_count"
        ] = 0.5
        with self.assertRaises(Exception):
            MATRIX._validate_bounded_numerical_replay(malformed, tasks)
        for nested in (
            {"metric_evidence": {"count": False}},
            {"difference_summary": {"rule": {"metric_count": 1.5}}},
            {"difference_summary": {"rule": {"arithmetic_overflow_count": 1}}},
            {"difference_summary": {"rule": {"zero_tolerance_failure_count": 2}}},
        ):
            with self.subTest(nested=nested):
                malformed = copy.deepcopy(baseline)
                malformed["comparisons"][1]["result"].update(nested)
                with self.assertRaises(Exception):
                    MATRIX._validate_bounded_numerical_replay(malformed, tasks)

    def test_embedded_record_rejects_symlinked_lane_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sample = output / "samples" / "sample"
            outside = output / "outside"
            sample.mkdir(parents=True)
            outside.mkdir()
            data = outside / "stdout.log"
            data.write_bytes(b"outside")
            (sample / "lane").symlink_to(outside, target_is_directory=True)
            record = {
                "path": "stdout.log",
                "size_bytes": data.stat().st_size,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(Exception, "symlink"):
                MATRIX._validate_embedded_file_records(
                    record,
                    output,
                    {},
                    "symlinked lane fixture",
                    pathlib.PurePosixPath("samples/sample/lane"),
                )

    def test_embedded_record_rejects_lane_replacement_during_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sample = output / "samples" / "sample"
            lane = sample / "lane"
            outside = output / "outside"
            lane.mkdir(parents=True)
            outside.mkdir()
            trusted = lane / "stdout.log"
            trusted.write_bytes(b"trusted")
            (outside / "stdout.log").write_bytes(b"outside")
            record = {
                "path": "stdout.log",
                "size_bytes": trusted.stat().st_size,
                "sha256": hashlib.sha256(trusted.read_bytes()).hexdigest(),
            }
            real_open = MATRIX.os.open
            swapped = False

            def swap_lane_before_file_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "stdout.log" and kwargs.get("dir_fd") is not None and not swapped:
                    swapped = True
                    lane.rename(sample / "lane-original")
                    lane.symlink_to(outside, target_is_directory=True)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                MATRIX.os, "open", side_effect=swap_lane_before_file_open
            ):
                with self.assertRaisesRegex(Exception, "ancestor changed"):
                    MATRIX._validate_embedded_file_records(
                        record,
                        output,
                        {},
                        "replaced lane fixture",
                        pathlib.PurePosixPath("samples/sample/lane"),
                    )

    def test_bounded_replay_rejects_persistent_stdout_byte_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            lane = output / "samples" / "sample" / "lane"
            lane.mkdir(parents=True)
            stdout = lane / "stdout.log"
            stdout.write_bytes(b"trusted-output")
            record = {
                "path": "stdout.log",
                "size_bytes": stdout.stat().st_size,
                "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
            }
            stdout.write_bytes(b"mutated-output")
            with self.assertRaisesRegex(Exception, "file identity differs"):
                MATRIX._validate_embedded_file_records(
                    record,
                    output,
                    {},
                    "post-validation stdout fixture",
                    pathlib.PurePosixPath("samples/sample/lane"),
                )

    def test_child_lane_semantics_rejects_failed_fabricated_lane(self) -> None:
        task = {"runtime": "upstream", "case_path": "case.py"}
        report = {"lane": {"process": {"exit_code": 1}, "spec": {"fabricated": True}}}
        with self.assertRaisesRegex(Exception, "lane schema differs"):
            MATRIX._validate_child_lane_semantics(
                task, report, pathlib.Path("/nonexistent"), {}, {"cases": {}}
            )

    def test_experiment_authority_rejects_wrong_task_census(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output, task_count=MATRIX.TASK_COUNT - 1)
            tasks = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )["contract"]["tasks"]
            with mock.patch.object(
                MATRIX, "validate_contract", return_value=(ROOT, tasks, ("gpu-a", "gpu-b"))
            ):
                with self.assertRaisesRegex(Exception, "task/contract inventory differs"):
                    MATRIX.verify_experiment_authority(output)

    def test_experiment_authority_propagates_resume_archive_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output)
            with self.bounded_authority_mocks(output):
                with mock.patch.object(
                    MATRIX.matrix,
                    "resume_history_records",
                    side_effect=MATRIX.WorkloadError(
                        "resume archive member inventory changed"
                    ),
                ):
                    with self.assertRaisesRegex(Exception, "member inventory changed"):
                        MATRIX.verify_experiment_authority(output)

    def test_experiment_authority_rejects_contradictory_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output)
            (output / "FAILED.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "contradictory state markers"):
                MATRIX.verify_experiment_authority(output)

    def test_absorber_scale_gate_uses_whole_trace_not_one_near_zero_point(self) -> None:
        comparison = {
            "mode": "json_metrics",
            "prefix": MATRIX.GENERIC_METRIC_PREFIX,
            "tolerances": {"*": {"rtol": 0.0, "atol": 0.0}},
        }
        reference = {
            "example_results": {
                "absorber_trace_fields": {
                    "real": [1.0, 1.0e-8, -0.5, 0.25],
                    "imag": [0.0, 0.0, 0.0, 0.0],
                }
            }
        }
        candidate = {
            "example_results": {
                "absorber_trace_fields": {
                    "real": [1.0, 6.0e-7, -0.5, 0.25],
                    "imag": [0.0, 0.0, 0.0, 0.0],
                }
            }
        }
        line = lambda value: MATRIX.GENERIC_METRIC_PREFIX + json.dumps(value)
        passed = MATRIX._absorber_scale_gate(
            line(reference), line(candidate), comparison
        )
        self.assertEqual(passed["outcome"], "PASS")
        candidate["example_results"]["absorber_trace_fields"]["real"][0] = 0.999
        with self.assertRaisesRegex(Exception, "scale-aware"):
            MATRIX._absorber_scale_gate(
                line(reference), line(candidate), comparison
            )

    def test_phase_snapshot_fp32_boundary_passes_without_masking_larger_drift(
        self,
    ) -> None:
        manifest = MATRIX.validation.load_json(
            ROOT / "scripts/python-validation/manifest.json"
        )
        cases = MATRIX.validation.materialize_cases(manifest, ROOT)
        case = next(
            item
            for item in cases
            if item["path"] == "python/examples/phase_in_material.py"
        )
        manifest_tolerances = case["comparison"]["tolerances"]
        observations = {
            "phase_snapshot_ez": (
                -0.00023185834288597107,
                -0.00022928323596715927,
            ),
            "phase_snapshot_dz": (
                -5.117710679769516e-5,
                -4.902482032775879e-5,
            ),
        }
        line = lambda value: MATRIX.GENERIC_METRIC_PREFIX + json.dumps(value)
        for field, (reference_value, candidate_value) in observations.items():
            tolerance = manifest_tolerances[f"example_results.{field}.*"]
            self.assertEqual(tolerance, {"rtol": 0.001, "atol": 3e-6})
            comparison = {
                "mode": "json_metrics",
                "prefix": MATRIX.GENERIC_METRIC_PREFIX,
                "tolerances": {f"example_results.{field}.*": tolerance},
            }
            reference = {"example_results": {field: {"real": [reference_value]}}}
            candidate = {"example_results": {field: {"real": [candidate_value]}}}
            observed = MATRIX.validation.compare_json_metrics(
                line(reference), line(candidate), comparison
            )
            self.assertEqual(observed["outcome"], "PASS")

            candidate["example_results"][field]["real"][0] = (
                reference_value + 4e-6
            )
            excessive = MATRIX.validation.compare_json_metrics(
                line(reference), line(candidate), comparison
            )
            self.assertEqual(excessive["outcome"], "MISMATCH")
            self.assertEqual(excessive["failure_count"], 1)

    def test_near2far_upstream_contract_diagnoses_only_invalid_adjoint_oracle(self) -> None:
        manifest = MATRIX.validation.load_json(
            ROOT / "scripts/python-validation/manifest.json"
        )
        cases = MATRIX.validation.materialize_cases(manifest, ROOT)
        case = next(item for item in cases if item["path"] == MATRIX.NEAR2FAR_3D_CASE)
        upstream = MATRIX._near2far_upstream_forward_contract(case)
        diagnostic = {
            name
            for name, tolerance in upstream["tolerances"].items()
            if tolerance.get("diagnostic_only")
        }
        self.assertEqual(
            diagnostic,
            {
                "gradient_l2",
                "gradient_by_frequency_l2*",
                "center_parameter_gradient",
                "derivative_checks*.adjoint_derivative",
                "derivative_checks*.adjoint_derivative_by_frequency*",
                "derivative_checks*.relative_derivative_error",
                "maximum_relative_derivative_error",
            },
        )
        self.assertFalse(upstream["tolerances"]["objective"].get("diagnostic_only", False))
        self.assertFalse(
            upstream["tolerances"]["derivative_checks*.finite_difference_derivative"].get(
                "diagnostic_only", False
            )
        )

    def test_cylindrical_self_oracle_and_fixed_work_are_independent_gates(self) -> None:
        cases = []
        derivative_error = abs(1.0e-6 - 1.1e-6) / 1.1e-6
        for index, (m, far_point) in enumerate(MATRIX.ADJOINT_CYL_EXPECTED_CASES):
            cases.append(
                {
                    "case_index": index,
                    "m": m,
                    "far_point": list(far_point),
                    "adjoint_objective": 1.0,
                    "forward_objective": 1.0,
                    "perturbed_forward_objective": 1.000001,
                    "adjoint_directional_derivative": 1.0e-6,
                    "finite_difference_derivative": 1.1e-6,
                    "relative_derivative_error": derivative_error,
                }
            )
        payload = MATRIX.ADJOINT_CYL_METRIC_PREFIX + json.dumps(
            {
                "schema_version": 1,
                "fixed_work": 1,
                "case_count": 4,
                "cases": cases,
            }
        )
        comparison = MATRIX._adjoint_cyl_contract(upstream_forward_only=False)
        self.assertEqual(
            MATRIX._adjoint_cyl_self_oracle(payload, comparison)["outcome"],
            "PASS",
        )
        cases[0]["finite_difference_derivative"] = 3.0e-6
        cases[0]["relative_derivative_error"] = abs(1.0e-6 - 3.0e-6) / 3.0e-6
        bad_payload = MATRIX.ADJOINT_CYL_METRIC_PREFIX + json.dumps(
            {
                "schema_version": 1,
                "fixed_work": 1,
                "case_count": 4,
                "cases": cases,
            }
        )
        with self.assertRaisesRegex(Exception, "finite-difference oracle"):
            MATRIX._adjoint_cyl_self_oracle(bad_payload, comparison)

        cases[0]["finite_difference_derivative"] = 1.1e-6
        for forged_error in (0.0, -1.0):
            cases[0]["relative_derivative_error"] = forged_error
            forged_payload = MATRIX.ADJOINT_CYL_METRIC_PREFIX + json.dumps(
                {
                    "schema_version": 1,
                    "fixed_work": 1,
                    "case_count": 4,
                    "cases": cases,
                }
            )
            with self.assertRaisesRegex(Exception, "reported derivative error"):
                MATRIX._adjoint_cyl_self_oracle(forged_payload, comparison)

        generic = MATRIX.GENERIC_METRIC_PREFIX + json.dumps(
            {
                "run_count": 16,
                "per_run": {
                    "timestep": list(MATRIX.ADJOINT_CYL_EXPECTED_TIMESTEPS),
                    "timestep_delta": list(MATRIX.ADJOINT_CYL_EXPECTED_TIMESTEPS),
                    "meep_time": list(MATRIX.ADJOINT_CYL_EXPECTED_MEEP_TIMES),
                },
            }
        )
        self.assertEqual(
            MATRIX._adjoint_cyl_fixed_work_gate(generic)["outcome"], "PASS"
        )
        self.assertEqual(
            MATRIX._adjoint_cyl_fixed_work_gate(generic)["timestep_deltas"],
            list(MATRIX.ADJOINT_CYL_EXPECTED_TIMESTEPS),
        )
        wrong = json.loads(generic.removeprefix(MATRIX.GENERIC_METRIC_PREFIX))
        wrong["per_run"]["timestep"][0] -= 1
        with self.assertRaisesRegex(Exception, "not fixed-work"):
            MATRIX._adjoint_cyl_fixed_work_gate(
                MATRIX.GENERIC_METRIC_PREFIX + json.dumps(wrong)
            )

    def test_cylindrical_fixed_work_metrics_emit_once_across_mpi_rank_roles(self) -> None:
        module_path = ROOT / "python/tests/adjoint_cyl_fixed_work_contract.py"
        module_name = "gpmeep_adjoint_cyl_emission_contract_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        rows = [
            {"case_index": index, "metric": float(index)}
            for index in range(len(MATRIX.ADJOINT_CYL_EXPECTED_CASES))
        ]
        emitted = [
            module.fixed_work_metrics_payload(
                rows,
                is_master=rank == 0,
                expected_case_count=len(MATRIX.ADJOINT_CYL_EXPECTED_CASES),
                prefix=MATRIX.ADJOINT_CYL_METRIC_PREFIX,
            )
            for rank in range(8)
        ]
        self.assertEqual(sum(value is not None for value in emitted), 1)
        self.assertTrue(emitted[0].startswith(MATRIX.ADJOINT_CYL_METRIC_PREFIX))
        with self.assertRaisesRegex(RuntimeError, "metric census"):
            module.fixed_work_metrics_payload(
                rows[:-1],
                is_master=False,
                expected_case_count=len(MATRIX.ADJOINT_CYL_EXPECTED_CASES),
                prefix=MATRIX.ADJOINT_CYL_METRIC_PREFIX,
            )

    def test_complete_replay_rejects_contradictory_running_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "RUNNING.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "contradictory state markers"):
                MATRIX.verify_complete(output)

    def test_complete_replay_reuses_each_verified_child_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sealed_authority_fixture(output)
            report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )
            tasks = report["contract"]["tasks"]
            completed = report["tasks"]
            sample_reports = [
                {"fixture_task_index": index} for index in range(MATRIX.TASK_COUNT)
            ]
            performance = dict(report["performance"])
            inversions = performance.pop("inversion_records")
            order = []
            validated = mock.patch.object(
                MATRIX,
                "_validate_completed_task_with_report",
                side_effect=[
                    (retained, sample_report)
                    for retained, sample_report in zip(
                        completed, sample_reports, strict=True
                    )
                ],
            )
            aggregate = mock.patch.object(
                MATRIX,
                "_derive_aggregate_from_reports",
                side_effect=lambda *_args: (
                    order.append("aggregate")
                    or (performance, report["numerical"])
                ),
            )
            inversion = mock.patch.object(
                MATRIX,
                "_derive_inversion_records_from_reports",
                side_effect=lambda *_args: (
                    order.append("inversion") or inversions
                ),
            )
            authority = mock.patch.object(
                MATRIX,
                "_revalidate_cached_reports_authority",
                side_effect=lambda *_args: order.append("authority"),
            )
            runtime_session = mock.Mock()
            runtime_session.finalize.side_effect = lambda: order.append(
                "finalize"
            )
            with (
                mock.patch.object(
                    MATRIX,
                    "validate_contract",
                    return_value=(ROOT, tasks, ("gpu-a", "gpu-b")),
                ),
                validated as validate_mock,
                aggregate as aggregate_mock,
                inversion as inversion_mock,
                authority as authority_mock,
                mock.patch.object(MATRIX, "validate_inversion_records"),
                mock.patch.object(MATRIX.matrix, "validate_hardware_evidence"),
                mock.patch.object(MATRIX, "validate_capacity"),
                mock.patch.object(
                    MATRIX.matrix, "resume_history_records", return_value=[]
                ),
                mock.patch.object(
                    MATRIX.v110_sample,
                    "RuntimeVerificationSession",
                    return_value=runtime_session,
                ),
            ):
                verified = MATRIX.verify_complete(output)
            self.assertEqual(verified, report)
            self.assertEqual(validate_mock.call_count, MATRIX.TASK_COUNT)
            runtime_session.finalize.assert_called_once_with()
            self.assertEqual(
                order, ["aggregate", "inversion", "authority", "finalize"]
            )
            self.assertTrue(
                all(
                    call.kwargs.get("v110_runtime_session")
                    is runtime_session
                    for call in validate_mock.call_args_list
                )
            )
            self.assertEqual(aggregate_mock.call_count, 1)
            self.assertEqual(inversion_mock.call_count, 1)
            reused = aggregate_mock.call_args.args[2]
            self.assertEqual(reused, sample_reports)
            self.assertIs(inversion_mock.call_args.args[3], reused)
            self.assertEqual(authority_mock.call_count, 1)
            self.assertIs(authority_mock.call_args.args[4], reused)

    def test_standalone_derivations_finalize_after_consuming_reports(self) -> None:
        for function_name, consumer_name in (
            ("derive_aggregate", "_derive_aggregate_from_reports"),
            ("derive_inversion_records", "_derive_inversion_records_from_reports"),
        ):
            with self.subTest(function=function_name):
                order = []
                runtime_session = mock.Mock()
                runtime_session.finalize.side_effect = lambda: order.append(
                    "finalize"
                )
                reports = [{"fixture": True}]
                patches = [
                    mock.patch.object(
                        MATRIX.v110_sample,
                        "RuntimeVerificationSession",
                        return_value=runtime_session,
                    ),
                    mock.patch.object(
                        MATRIX,
                        "_derive_verified_sample_reports",
                        side_effect=lambda *_args, **_kwargs: (
                            order.append("reports") or reports
                        ),
                    ),
                    mock.patch.object(
                        MATRIX,
                        consumer_name,
                        side_effect=lambda *_args: (
                            order.append("consumer") or {"derived": True}
                        ),
                    ),
                ]
                if function_name == "derive_inversion_records":
                    patches.append(
                        mock.patch.object(
                            MATRIX,
                            "_load_checkpoint",
                            return_value={"completed_tasks": []},
                        )
                    )
                with patches[0], patches[1], patches[2]:
                    if len(patches) == 4:
                        with patches[3]:
                            getattr(MATRIX, function_name)(
                                pathlib.Path("/fixture"), {}, []
                            ) if function_name == "derive_aggregate" else getattr(
                                MATRIX, function_name
                            )(pathlib.Path("/fixture"), {})
                    else:
                        getattr(MATRIX, function_name)(
                            pathlib.Path("/fixture"), {}, []
                        )
                self.assertEqual(
                    order, ["reports", "consumer", "finalize"]
                )

    def test_verified_report_reuse_retains_per_task_semantic_gate(self) -> None:
        tasks = [
            {"task_id": f"task-{index:03d}"}
            for index in range(MATRIX.TASK_COUNT)
        ]
        reports = [
            {"fixture_task_index": index} for index in range(MATRIX.TASK_COUNT)
        ]
        with mock.patch.object(
            MATRIX.equal_work_replay,
            "validate_sample_report",
            side_effect=[
                *([None] * 17),
                MATRIX.WorkloadError("mutated child report"),
            ],
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "mutated child report"
            ):
                MATRIX._derive_aggregate_from_reports(
                    pathlib.Path("/unused"), {"tasks": tasks}, reports
                )

    def test_verified_report_collection_rejects_retained_replay_drift(self) -> None:
        tasks = [
            {"task_id": f"task-{index:03d}"}
            for index in range(MATRIX.TASK_COUNT)
        ]
        completed = [
            {"sample_replay": {"fixture_task_index": index}}
            for index in range(MATRIX.TASK_COUNT)
        ]
        replays = [
            (
                {"fixture_task_index": index},
                {"fixture_report_index": index},
            )
            for index in range(MATRIX.TASK_COUNT)
        ]
        completed[73] = {"sample_replay": {"fixture_task_index": -1}}
        with mock.patch.object(
            MATRIX, "derive_sample_replay", side_effect=replays
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "retained child replay differs"
            ):
                MATRIX._derive_verified_sample_reports(
                    pathlib.Path("/unused"), {"tasks": tasks}, completed
                )

    def test_inversion_reuse_rejects_missing_retained_report_evidence(self) -> None:
        tasks = [
            {"task_id": f"task-{index:03d}"}
            for index in range(MATRIX.TASK_COUNT)
        ]
        completed = [
            {
                "sample_replay": {
                    "report": {
                        "path": f"raw/{index:03d}/report.json",
                        "sha256": f"{index:064x}",
                    }
                }
            }
            for index in range(MATRIX.TASK_COUNT)
        ]
        reports = [
            {"fixture_task_index": index} for index in range(MATRIX.TASK_COUNT)
        ]
        completed[91] = {"sample_replay": {}}
        with mock.patch.object(
            MATRIX.equal_work_replay, "validate_sample_report"
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError, "child report evidence differs"
            ):
                MATRIX._derive_inversion_records_from_reports(
                    pathlib.Path("/unused"),
                    {"tasks": tasks},
                    completed,
                    reports,
                )

    def test_cached_report_authority_rejects_post_validation_change(self) -> None:
        tasks = [
            {"task_id": f"task-{index:03d}"}
            for index in range(MATRIX.TASK_COUNT)
        ]
        completed = [
            {"fixture_task_index": index} for index in range(MATRIX.TASK_COUNT)
        ]
        cached = [
            {"fixture_report_index": index}
            for index in range(MATRIX.TASK_COUNT)
        ]
        replayed = copy.deepcopy(cached)
        replayed[117]["post_validation_mutation"] = True
        with (
            mock.patch.object(
                MATRIX, "_bounded_child_context", return_value={}
            ),
            mock.patch.object(
                MATRIX,
                "_validate_completed_task_authority",
                side_effect=replayed,
            ) as authority_mock,
        ):
            with self.assertRaisesRegex(
                MATRIX.WorkloadError,
                "cached child reports changed before acceptance",
            ):
                MATRIX._revalidate_cached_reports_authority(
                    pathlib.Path("/unused"),
                    {
                        "paths": {"repo": str(ROOT)},
                        "tasks": tasks,
                    },
                    completed,
                    ("gpu-a", "gpu-b"),
                    cached,
                )
        self.assertEqual(authority_mock.call_count, MATRIX.TASK_COUNT)

    def test_complete_replay_rejects_contradictory_failed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "FAILED.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "contradictory state markers"):
                MATRIX.verify_complete(output)

    def test_remove_running_marker_is_durable_and_required_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            marker = output / "RUNNING.json"
            marker.write_text("{}\n", encoding="utf-8")
            MATRIX.remove_running_marker(output, required=True)
            self.assertFalse(marker.exists())
            MATRIX.remove_running_marker(output, required=False)
            with self.assertRaisesRegex(Exception, "RUNNING marker is absent"):
                MATRIX.remove_running_marker(output, required=True)

    def test_verify_cli_needs_only_output(self) -> None:
        args = MATRIX.parse_args(["--output", "/tmp/result", "--verify-complete"])
        MATRIX.validate_args(args)
        args = MATRIX.parse_args(["--output", "/tmp/result"])
        with self.assertRaisesRegex(Exception, "every runtime path"):
            MATRIX.validate_args(args)

    def test_preflight_requires_full_runtime_authority_and_is_mutually_exclusive(self) -> None:
        args = MATRIX.parse_args(["--output", "/tmp/result", "--preflight-only"])
        with self.assertRaisesRegex(Exception, "every runtime path"):
            MATRIX.validate_args(args)
        args = MATRIX.parse_args(
            ["--output", "/tmp/result", "--preflight-only", "--resume"]
        )
        with self.assertRaisesRegex(Exception, "mutually exclusive"):
            MATRIX.validate_args(args)

    def test_main_routes_preflight_before_output_creation_or_task_execution(self) -> None:
        arguments = [
            "--repo", str(ROOT), "--output", "/tmp/gpmeep-preflight-do-not-create",
            "--preflight-only", "--python", "/runtime/python",
            "--build-python", "/runtime/site-packages", "--install-prefix", "/runtime/prefix",
            "--build-receipt", "/receipts/build.json", "--mpiexec", "/runtime/mpiexec",
            "--upstream-receipt", "/receipts/upstream.json",
            "--v103-install-prefix", "/runtime/v103",
            "--v103-provenance-report", "/receipts/v103.json",
            "--v103-package", "/packages/v103.conda",
            "--v110-provenance-report", "/receipts/v110.json",
            "--v110-package", "/packages/v110.conda",
            "--gpu-devices", "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002",
        ]
        terminal = {
            "schema": MATRIX.PREFLIGHT_SCHEMA,
            "state": "PASS",
            "workloads_launched": 0,
        }
        with (
            mock.patch.object(
                MATRIX, "nvidia_uvm_hmm_binding", return_value={}
            ) as hmm_guard,
            mock.patch.object(MATRIX, "preflight_contract", return_value=terminal) as replay,
            mock.patch.object(pathlib.Path, "mkdir") as mkdir,
            mock.patch("builtins.print") as output,
        ):
            self.assertEqual(MATRIX.main(arguments), 0)
        hmm_guard.assert_called_once_with()
        replay.assert_called_once()
        mkdir.assert_not_called()
        output.assert_called_once()

    def test_main_rejects_unsafe_hmm_before_preflight_or_output(self) -> None:
        with (
            mock.patch.object(
                MATRIX,
                "nvidia_uvm_hmm_binding",
                side_effect=MATRIX.WorkloadError("unsafe HMM fixture"),
            ) as hmm_guard,
            mock.patch.object(MATRIX, "preflight_contract") as preflight,
            mock.patch.object(pathlib.Path, "mkdir") as mkdir,
            self.assertRaisesRegex(MATRIX.WorkloadError, "unsafe HMM fixture"),
        ):
            MATRIX.main(["--output", "/tmp/m4-must-not-exist", "--verify-complete"])
        hmm_guard.assert_called_once_with()
        preflight.assert_not_called()
        mkdir.assert_not_called()

    def test_precreation_capacity_passes_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary).resolve()
            output = parent / "planned" / "m4"
            capacity = {
                "path": str(parent),
                "free_bytes": MATRIX.INITIAL_FREE_BYTES + 1,
                "total_bytes": MATRIX.INITIAL_FREE_BYTES + 2,
                "free_inodes": MATRIX.MINIMUM_FREE_INODES + 1,
                "total_inodes": MATRIX.MINIMUM_FREE_INODES + 2,
            }
            with mock.patch.object(
                MATRIX, "capacity_snapshot", return_value=capacity
            ) as snapshot:
                self.assertEqual(
                    MATRIX.validate_precreation_capacity(output), capacity
                )
            snapshot.assert_called_once_with(parent)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_precreation_capacity_fails_closed_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary).resolve()
            output = parent / "planned-m4"
            insufficient = {
                "path": str(parent),
                "free_bytes": MATRIX.INITIAL_FREE_BYTES - 1,
                "total_bytes": MATRIX.INITIAL_FREE_BYTES,
                "free_inodes": MATRIX.MINIMUM_FREE_INODES + 1,
                "total_inodes": MATRIX.MINIMUM_FREE_INODES + 2,
            }
            with (
                mock.patch.object(
                    MATRIX, "capacity_snapshot", return_value=insufficient
                ),
                self.assertRaisesRegex(Exception, "free-space/inode gate failed"),
            ):
                MATRIX.validate_precreation_capacity(output)
            self.assertFalse(output.exists())

    def test_preflight_checks_capacity_before_opening_runtime_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve() / "planned-m4"
            args = MATRIX.parse_args(
                [
                    "--repo", str(ROOT), "--output", str(output),
                    "--preflight-only", "--python", "/runtime/python",
                    "--build-python", "/runtime/build-python",
                    "--install-prefix", "/runtime/prefix",
                    "--build-receipt", "/receipts/build.json",
                    "--mpiexec", "/runtime/mpiexec",
                    "--upstream-receipt", "/receipts/upstream.json",
                    "--v103-install-prefix", "/runtime/v103",
                    "--v103-provenance-report", "/receipts/v103.json",
                    "--v103-package", "/packages/v103.conda",
                    "--v110-provenance-report", "/receipts/v110.json",
                    "--v110-package", "/packages/v110.conda",
                    "--gpu-devices", "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002",
                ]
            )
            with (
                mock.patch.object(MATRIX, "host_binding", return_value={}),
                mock.patch.object(
                    MATRIX,
                    "validate_precreation_capacity",
                    side_effect=MATRIX.WorkloadError("capacity fixture failure"),
                ) as capacity,
                mock.patch.object(MATRIX, "_code_records") as code_records,
                self.assertRaisesRegex(Exception, "capacity fixture failure"),
            ):
                MATRIX.preflight_contract(args, ROOT)
            capacity.assert_called_once_with(output)
            code_records.assert_not_called()
            self.assertFalse(output.exists())

    def test_fresh_main_checks_capacity_before_output_creation(self) -> None:
        arguments = [
            "--repo", str(ROOT), "--output", "/tmp/gpmeep-capacity-fail",
            "--python", "/runtime/python", "--build-python", "/runtime/build-python",
            "--install-prefix", "/runtime/prefix",
            "--build-receipt", "/receipts/build.json",
            "--mpiexec", "/runtime/mpiexec",
            "--upstream-receipt", "/receipts/upstream.json",
            "--v103-install-prefix", "/runtime/v103",
            "--v103-provenance-report", "/receipts/v103.json",
            "--v103-package", "/packages/v103.conda",
            "--v110-provenance-report", "/receipts/v110.json",
            "--v110-package", "/packages/v110.conda",
            "--gpu-devices", "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002",
        ]
        with (
            mock.patch.object(
                MATRIX, "nvidia_uvm_hmm_binding", return_value={}
            ),
            mock.patch.object(
                MATRIX,
                "validate_precreation_capacity",
                side_effect=MATRIX.WorkloadError("capacity fixture failure"),
            ) as capacity,
            mock.patch.object(pathlib.Path, "mkdir") as mkdir,
            self.assertRaisesRegex(Exception, "capacity fixture failure"),
        ):
            MATRIX.main(arguments)
        capacity.assert_called_once_with(pathlib.Path("/tmp/gpmeep-capacity-fail"))
        mkdir.assert_not_called()

    def test_fresh_main_rejects_unittest_import_before_output_creation(self) -> None:
        output = pathlib.Path("/tmp/gpmeep-unittest-preflight-fail")
        arguments = [
            "--repo", str(ROOT), "--output", str(output),
            "--python", "/runtime/python", "--build-python", "/runtime/build-python",
            "--install-prefix", "/runtime/prefix",
            "--build-receipt", "/receipts/build.json",
            "--mpiexec", "/runtime/mpiexec",
            "--upstream-receipt", "/receipts/upstream.json",
            "--v103-install-prefix", "/runtime/v103",
            "--v103-provenance-report", "/receipts/v103.json",
            "--v103-package", "/packages/v103.conda",
            "--v110-provenance-report", "/receipts/v110.json",
            "--v110-package", "/packages/v110.conda",
            "--gpu-devices", "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002",
        ]
        with (
            mock.patch.object(
                MATRIX, "nvidia_uvm_hmm_binding", return_value={}
            ),
            mock.patch.object(
                MATRIX,
                "preflight_contract",
                side_effect=MATRIX.WorkloadError(
                    "M4 gpmeep_v110 unittest import preflight failed"
                ),
            ) as preflight,
            mock.patch.object(pathlib.Path, "mkdir") as mkdir,
            self.assertRaisesRegex(Exception, "unittest import preflight failed"),
        ):
            MATRIX.main(arguments)
        preflight.assert_called_once()
        mkdir.assert_not_called()

    def test_unittest_preflight_streams_to_hard_caps_with_empty_pycache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary).resolve()
            site_packages = repo / "site-packages"
            site_packages.mkdir()
            meep_module = site_packages / "meep.py"
            meep_module.write_text("# fixture\n", encoding="utf-8")
            fontconfig = repo / "fonts.conf"
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            observed: dict[str, object] = {}

            def capped(command, **kwargs):
                observed["command"] = command
                observed["kwargs"] = kwargs
                option_index = command.index("-X")
                pycache_option = command[option_index + 1]
                self.assertTrue(pycache_option.startswith("pycache_prefix="))
                pycache = pathlib.Path(pycache_option.split("=", 1)[1])
                self.assertTrue(pycache.is_dir())
                self.assertEqual(list(pycache.iterdir()), [])
                kwargs["stdout_file"].write_bytes(
                    b"x" * MATRIX.UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES
                )
                kwargs["stderr_file"].write_bytes(b"")
                return {
                    "exit_code": -15,
                    "timeout": False,
                    "output_limit": "stdout",
                    "spawn_error": None,
                    "stdout_size_bytes": (
                        MATRIX.UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES
                    ),
                    "stderr_size_bytes": 0,
                }

            runtime = {
                "python": pathlib.Path(sys.executable).resolve(),
                "site_packages": site_packages,
                "meep_module": meep_module,
                "library_paths": [],
                "fontconfig": fontconfig,
                "gpmeep": False,
            }
            fixture_contract = {
                "expected_simulation_run_count": 16,
                "test_identities": ["fixture.Test.test_0"],
                "case_contract_variable": "CASES",
                "case_contract": [[0, [1, 2, 3]]],
                "wrapper_helper": "_check",
                "wrapper_invocations": [],
            }
            with (
                mock.patch.object(
                    MATRIX.validation,
                    "run_bounded_process",
                    side_effect=capped,
                ) as bounded,
                self.assertRaisesRegex(Exception, "import preflight failed"),
            ):
                MATRIX._run_unittest_import_preflight(
                    repo,
                    "test_fixture.py",
                    fixture_contract,
                    "fixture_runtime",
                    runtime,
                )
        bounded.assert_called_once()
        kwargs = observed["kwargs"]
        self.assertEqual(
            kwargs["stdout_limit"],
            MATRIX.UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
        )
        self.assertEqual(
            kwargs["stderr_limit"],
            MATRIX.UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
        )
        self.assertEqual(
            kwargs["combined_limit"],
            2 * MATRIX.UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
        )

    def test_repository_must_own_the_executing_controller(self) -> None:
        MATRIX.validate_repo_ownership(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Exception, "does not own"):
                MATRIX.validate_repo_ownership(pathlib.Path(temporary))

    def test_v110_identity_checks_exact_archive_before_live_verifier(self) -> None:
        paths = {
            "repo": str(ROOT),
            "v110_provenance_report": "/receipts/v110.json",
            "v110_package": "/packages/v110.conda",
        }
        raw = {"prefix": "/installed/v110"}
        report = {"report_id": "installed-package-report"}
        installed = {"install_prefix": pathlib.Path("/installed/v110")}
        runtime = {
            "python": pathlib.Path("/installed/v110/bin/python"),
            "build_python": pathlib.Path(
                "/installed/v110/lib/python3.11/site-packages"
            ),
            "install_prefix": pathlib.Path("/installed/v110"),
            "build_receipt": pathlib.Path("/receipts/build.json"),
            "mpiexec": pathlib.Path("/installed/v110/bin/mpirun"),
            "package": pathlib.Path("/packages/v110.conda"),
            "provenance_report": pathlib.Path("/receipts/v110.json"),
            "verifier": pathlib.Path(
                "/installed/v110/bin/gpmeep-verify-provenance"
            ),
        }
        archive = {"format": "conda-v2", "payload_path_count": 42}
        live = {"state": "PASS", "report_id": report["report_id"]}
        order: list[str] = []

        def archive_check(*_args, **_kwargs):
            order.append("archive")
            return archive

        def live_check(*_args, **_kwargs):
            order.append("live")
            return live

        with (
            mock.patch.object(MATRIX.validation, "load_json", return_value=raw),
            mock.patch.object(
                MATRIX.v110_identity,
                "validate_report",
                return_value=(report, installed),
            ),
            mock.patch.object(
                MATRIX.v110_identity, "runtime_paths", return_value=runtime
            ),
            mock.patch.object(
                MATRIX, "_v110_archive_authority", side_effect=archive_check
            ) as archive_replay,
            mock.patch.object(
                MATRIX.v110_identity, "live_verify", side_effect=live_check
            ) as live_replay,
        ):
            retained_report, retained_runtime, authority = (
                MATRIX._v110_installed_identity(paths, "a" * 40)
            )
        self.assertIs(retained_report, report)
        self.assertIs(retained_runtime, runtime)
        self.assertEqual(order, ["archive", "live"])
        self.assertEqual(authority["archive"], archive)
        self.assertEqual(authority["live"], live)
        self.assertEqual(
            authority["paths"], {name: str(value) for name, value in runtime.items()}
        )
        archive_replay.assert_called_once_with(paths, report, runtime)
        live_replay.assert_called_once_with(report, installed)

    def test_v103_identity_binds_frozen_package_prefix_and_live_records(self) -> None:
        paths = {
            "repo": str(ROOT),
            "manifest": str(ROOT / "scripts/python-validation/manifest.json"),
            "v103_provenance_report": "/receipts/v103.json",
            "v103_package": "/packages/v103.conda",
            "v103_install_prefix": "/installed/v103",
        }
        report = {
            "report_id": "frozen-report",
            "package_sha256": "a" * 64,
            "manifest": {"source_commit": "b" * 40},
        }
        installed = {
            "install_prefix": pathlib.Path("/installed/v103"),
            "python_executable": pathlib.Path("/installed/v103/bin/python"),
        }
        runtime = {"receipt_id": "frozen-report", "python_executable": {}}
        live_result = {"state": "PASS", "report_id": "frozen-report"}
        with (
            mock.patch.object(
                MATRIX.v103_sample,
                "runtime_paths",
                return_value=(report, installed, runtime),
            ) as validate,
            mock.patch.object(
                MATRIX.v103_sample, "live_verify", return_value=live_result
            ) as live,
        ):
            identity = MATRIX._v103_installed_identity(paths)
        namespace = validate.call_args.args[0]
        self.assertEqual(namespace.repo, pathlib.Path(paths["repo"]))
        self.assertEqual(namespace.manifest, pathlib.Path(paths["manifest"]))
        self.assertEqual(
            namespace.provenance_report,
            pathlib.Path(paths["v103_provenance_report"]),
        )
        self.assertEqual(namespace.package, pathlib.Path(paths["v103_package"]))
        self.assertEqual(
            namespace.install_prefix, pathlib.Path(paths["v103_install_prefix"])
        )
        live.assert_called_once_with(report, installed)
        self.assertEqual(identity["report_id"], "frozen-report")
        self.assertEqual(identity["package_sha256"], "a" * 64)
        self.assertEqual(identity["source_commit"], "b" * 40)
        self.assertEqual(identity["runtime_contract"], runtime)
        self.assertEqual(identity["live_verification"], live_result)

    def test_v103_identity_rejects_tampered_package_prefix_or_report(self) -> None:
        paths = {
            "repo": str(ROOT),
            "manifest": str(ROOT / "scripts/python-validation/manifest.json"),
            "v103_provenance_report": "/receipts/tampered-v103.json",
            "v103_package": "/packages/substitute-v103.conda",
            "v103_install_prefix": "/installed/unrelated-v103",
        }
        with (
            mock.patch.object(
                MATRIX.v103_sample,
                "runtime_paths",
                side_effect=MATRIX.WorkloadError(
                    "frozen v1.0.3 package identity differs"
                ),
            ),
            mock.patch.object(MATRIX.v103_sample, "live_verify") as live,
            self.assertRaisesRegex(
                MATRIX.WorkloadError,
                "frozen v1.0.3 installed package identity differs",
            ),
        ):
            MATRIX._v103_installed_identity(paths)
        live.assert_not_called()

    def test_build_contract_seals_replayed_v103_identity(self) -> None:
        args = types.SimpleNamespace(
            task_timeout_seconds=7200,
            stdout_limit_mib=64,
            telemetry_interval_seconds=1.0,
        )
        paths = {
            "repo": str(ROOT),
            "python": "/runtime/python",
            "build_python": "/build/python",
            "install_prefix": "/runtime/v110",
            "build_receipt": "/build/build-provenance.json",
            "mpiexec": "/runtime/mpiexec",
            "upstream_receipt": "/receipts/upstream.json",
            "v103_install_prefix": "/runtime/v103",
            "v103_provenance_report": "/receipts/v103.json",
            "v103_package": "/packages/v103.conda",
            "v110_provenance_report": "/receipts/v110.json",
            "v110_package": "/packages/v110.conda",
            "manifest": "/code/manifest.json",
            "v110_runner": "/code/v110.py",
            "upstream_runner": "/code/upstream.py",
            "v103_runner": "/code/v103.py",
        }
        receipt = {
            "build_kind": MATRIX.validation.EXPECTED_BUILD_KIND,
            "git_status_porcelain": [],
            "git_head": "a" * 40,
            "receipt_id": "b" * 64,
            "source_end": {"tree": "c" * 40},
        }
        v103 = {"report_id": "d" * 64, "package_sha256": "e" * 64}
        v110 = {
            "package_sha256": "f" * 64,
            "report_id": "1" * 64,
            "v110_candidate_gate": {
                "receipt_id": "2" * 64,
                "build_mode": "development",
            },
        }
        with (
            mock.patch.object(MATRIX, "_runtime_paths", return_value=paths),
            mock.patch.object(
                MATRIX.provenance, "verify_build_receipt", return_value=receipt
            ),
            mock.patch.object(
                MATRIX.upstream_sample,
                "verify_receipt",
                return_value=({"receipt_id": "3" * 64}, {"receipt_id": "3" * 64}),
            ),
            mock.patch.object(
                MATRIX,
                "_v110_installed_identity",
                return_value=(
                    v110,
                    {},
                    {"paths": {}, "live": {"state": "PASS"}, "archive": {"state": "PASS"}},
                ),
            ),
            mock.patch.object(
                MATRIX, "_v103_installed_identity", return_value=v103
            ) as replay_v103,
            mock.patch.object(
                MATRIX,
                "_unittest_import_preflight",
                return_value={"schema": "fixture", "state": "PASS"},
            ) as unittest_preflight,
            mock.patch.object(
                MATRIX, "_file_input_record", side_effect=lambda path, label: {
                    "path": str(path), "label": label
                }
            ),
            mock.patch.object(MATRIX, "host_binding", return_value={"host": "fixture"}),
            mock.patch.object(
                MATRIX, "absolute_file_record", return_value={"nvidia-smi": "fixture"}
            ),
        ):
            sealed = MATRIX.build_contract(
                args,
                ROOT,
                {"schema": "plan"},
                [],
                {},
                (
                    "GPU-00000000-0000-0000-0000-000000000001",
                    "GPU-00000000-0000-0000-0000-000000000002",
                ),
            )
        replay_v103.assert_called_once_with(paths)
        self.assertEqual(sealed["runtime_inputs"]["v103_installed_identity"], v103)
        unittest_preflight.assert_called_once()
        self.assertEqual(
            sealed["runtime_inputs"]["unittest_import_preflight"]["state"],
            "PASS",
        )

    def test_archive_preflight_restores_dynamic_module_on_failure(self) -> None:
        module_name = "gpmeep_m4_inversion_archive_preflight"
        previous = types.ModuleType(module_name)
        sys.modules[module_name] = previous
        try:
            with tempfile.TemporaryDirectory() as temporary:
                repo = pathlib.Path(temporary).resolve()
                source = repo / "scripts/v110-inversion-experiment-sample.py"
                source.parent.mkdir()
                source.write_text(
                    "def _installed_archive_authority(**kwargs):\n"
                    "    raise RuntimeError('archive fixture failure')\n"
                )
                with self.assertRaisesRegex(
                    MATRIX.WorkloadError, "archive fixture failure"
                ):
                    MATRIX._v110_archive_authority(
                        {"repo": str(repo)},
                        {},
                        {
                            "package": pathlib.Path("/package.conda"),
                            "install_prefix": pathlib.Path("/prefix"),
                        },
                    )
            self.assertIs(sys.modules[module_name], previous)
        finally:
            sys.modules.pop(module_name, None)

    def test_archive_preflight_removes_new_dynamic_module_on_success(self) -> None:
        module_name = "gpmeep_m4_inversion_archive_preflight"
        sys.modules.pop(module_name, None)
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary).resolve()
            source = repo / "scripts/v110-inversion-experiment-sample.py"
            source.parent.mkdir()
            source.write_text(
                "def _installed_archive_authority(**kwargs):\n"
                "    return ({'format': 'conda-v2'}, {}, {})\n"
            )
            authority = MATRIX._v110_archive_authority(
                {"repo": str(repo)},
                {},
                {
                    "package": pathlib.Path("/package.conda"),
                    "install_prefix": pathlib.Path("/prefix"),
                },
            )
        self.assertEqual(authority, {"format": "conda-v2"})
        self.assertNotIn(module_name, sys.modules)

    def test_normative_registry_validator_is_sealed_as_campaign_code(self) -> None:
        stables, records = MATRIX._code_records(ROOT)
        try:
            self.assertIn("unittest_preflight", stables)
            self.assertIn("unittest_adjoint_cyl_contract", stables)
            self.assertIn("inversion_validator", stables)
            self.assertIn("inversion_experiment_producer", stables)
            self.assertIn("inversion_qualified_child", stables)
            self.assertIn("inversion_archive_inspector", stables)
            self.assertIn("installed_provenance_verifier", stables)
            self.assertIn("installed_provenance_launcher", stables)
            self.assertEqual(
                records["unittest_preflight"]["path"],
                "scripts/python-validation/preflight_unittest_workload.py",
            )
            self.assertEqual(
                records["unittest_adjoint_cyl_contract"]["path"],
                "python/tests/adjoint_cyl_fixed_work_contract.py",
            )
            self.assertEqual(
                stables["inversion_validator"].path,
                (ROOT / "scripts/validate-v110-inversion-registry.py").resolve(),
            )
            self.assertEqual(
                records["inversion_validator"]["path"],
                "scripts/validate-v110-inversion-registry.py",
            )
            self.assertEqual(
                stables["inversion_experiment_producer"].path,
                (ROOT / "scripts/v110-inversion-experiment-sample.py").resolve(),
            )
            self.assertEqual(
                records["inversion_experiment_producer"]["path"],
                "scripts/v110-inversion-experiment-sample.py",
            )
            self.assertEqual(
                stables["inversion_qualified_child"].path,
                (ROOT / "scripts/v110-inversion-qualified-child.py").resolve(),
            )
            self.assertEqual(
                records["inversion_qualified_child"]["path"],
                "scripts/v110-inversion-qualified-child.py",
            )
            self.assertEqual(
                stables["inversion_archive_inspector"].path,
                (ROOT / "scripts/v110-final-release-receipt.py").resolve(),
            )
            self.assertEqual(
                records["inversion_archive_inspector"]["path"],
                "scripts/v110-final-release-receipt.py",
            )
            self.assertEqual(
                records["installed_provenance_verifier"]["path"],
                "scripts/verify-gpmeep-package.py",
            )
            self.assertEqual(
                records["installed_provenance_launcher"]["path"],
                "scripts/gpmeep-verify-provenance-launcher.sh",
            )
        finally:
            for stable in stables.values():
                stable.close(verify=False)

    def test_unittest_preflight_seals_four_cylindrical_adjoint_cases(self) -> None:
        plan = MATRIX.equal_work_plan.load_plan(ROOT)
        tasks = MATRIX.equal_work_plan.build_tasks(plan)
        contracts = MATRIX._selected_unittest_contracts(ROOT, tasks)
        self.assertEqual(set(contracts), {"python/tests/test_adjoint_cyl.py"})
        retained = contracts["python/tests/test_adjoint_cyl.py"]
        self.assertEqual(retained["expected_simulation_run_count"], 16)
        self.assertEqual(len(retained["test_identities"]), 4)
        self.assertEqual(
            [row["mode"] for row in retained["wrapper_invocations"]],
            [0, 0, -1, 1.2],
        )
        self.assertEqual(
            [row["far_point"] for row in retained["wrapper_invocations"]],
            [
                [5.0, 0.0, 20.0],
                [4.0, 0.0, 28.0],
                [5.0, 0.0, 20.0],
                [5.0, 0.0, 20.0],
            ],
        )

    def test_mutated_normative_registry_validator_breaks_stable_campaign_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            validator = root / "validate-v110-inversion-registry.py"
            validator.write_text("original validator\n")
            (root / "packaging").mkdir()
            (root / "scripts").mkdir()
            (root / "python/tests").mkdir(parents=True)
            (root / "python/tests/test_adjoint_cyl.py").write_text(
                "original unittest case\n"
            )
            (root / "python/tests/adjoint_cyl_fixed_work_contract.py").write_text(
                "original fixed-work contract\n"
            )
            (root / "packaging/v110-development-contract.json").write_text("{}\n")
            (root / "packaging/v110-m4-equal-work-plan.json").write_text("{}\n")
            (root / "packaging/v110-performance-inversion-inventory.json").write_text("{}\n")
            (root / "scripts/validate-v110-inversion-registry.py").write_text(
                "original validator\n"
            )
            (root / "scripts/v110-inversion-experiment-sample.py").write_text(
                "original experiment producer\n"
            )
            (root / "scripts/v110-inversion-qualified-child.py").write_text(
                "original qualified child\n"
            )
            (root / "scripts/v110-final-release-receipt.py").write_text(
                "original archive inspector\n"
            )
            (root / "scripts/verify-gpmeep-package.py").write_text(
                "original installed verifier\n"
            )
            (root / "scripts/gpmeep-verify-provenance-launcher.sh").write_text(
                "original installed launcher\n"
            )
            validator = root / "scripts/validate-v110-inversion-registry.py"
            with mock.patch.object(
                MATRIX,
                "_code_paths",
                return_value={},
            ):
                stables, _records = MATRIX._code_records(root)
            try:
                validator.write_text("changed during campaign\n")
                with self.assertRaisesRegex(Exception, "changed"):
                    stables["inversion_validator"].verify_unchanged()
            finally:
                for stable in stables.values():
                    stable.close(verify=False)
            validator.write_text("original validator\n")
            with mock.patch.object(MATRIX, "_code_paths", return_value={}):
                helper_stables, _records = MATRIX._code_records(root)
            try:
                helper = root / "python/tests/adjoint_cyl_fixed_work_contract.py"
                helper.write_text("changed during campaign\n")
                with self.assertRaisesRegex(Exception, "changed"):
                    helper_stables[
                        "unittest_adjoint_cyl_contract"
                    ].verify_unchanged()
            finally:
                for stable in helper_stables.values():
                    stable.close(verify=False)

    def test_every_task_routes_to_the_expected_runner(self) -> None:
        sealed = contract()
        output = pathlib.Path("/evidence")
        runners = {
            "upstream": "/runner/upstream.py",
            "v103": "/runner/v103.py",
            "v110": "/runner/v110.py",
        }
        for task in sealed["tasks"]:
            command = MATRIX.task_command(task, sealed, output)
            self.assertEqual(command[:2], ["/runtime/python", runners[task["runtime"]]])
            self.assertIn(task["case_path"], command)
            self.assertIn(task["sample_kind"], command)
            if task["runtime"] == "v103":
                self.assertIn("/packages/v103.tar.bz2", command)
            if task["runtime"] == "upstream":
                self.assertNotIn("--topology", command)
            if task["runtime"] == "v110":
                self.assertIn("/packages/v110.conda", command)
                self.assertIn("/installed/v110/bin/python", command)

    def test_observer_selection_matches_backend_and_gpu_count(self) -> None:
        sealed = contract()
        tasks = sealed["tasks"]
        cpu = next(task for task in tasks if task["backend"] == "cpu")
        gpu1 = next(task for task in tasks if task["topology"].endswith("cuda1"))
        gpu2 = next(task for task in tasks if task["topology"].endswith("cuda2"))
        cpu_specs = MATRIX.observer_specs(cpu, 123, pathlib.Path("/out"), sealed)
        self.assertEqual([spec.label for spec in cpu_specs], ["host", "gpu"])
        self.assertEqual(
            len([x for x in cpu_specs[1].command if x == "--selected-uuid"]), 2
        )
        one = MATRIX.observer_specs(gpu1, 123, pathlib.Path("/out"), sealed)
        two = MATRIX.observer_specs(gpu2, 123, pathlib.Path("/out"), sealed)
        self.assertEqual([spec.label for spec in one], ["host", "gpu"])
        self.assertEqual(len([x for x in one[1].command if x == "--selected-uuid"]), 1)
        self.assertEqual(len([x for x in two[1].command if x == "--selected-uuid"]), 2)

    def test_outer_environment_hides_gpus_from_cpu_and_limits_cuda1(self) -> None:
        sealed = contract()
        cpu = next(task for task in sealed["tasks"] if task["backend"] == "cpu")
        gpu1 = next(task for task in sealed["tasks"] if task["topology"].endswith("cuda1"))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu_environment = MATRIX.clean_environment(cpu, sealed, root / "cpu")
            gpu_environment = MATRIX.clean_environment(gpu1, sealed, root / "gpu")
        self.assertEqual(cpu_environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(cpu_environment["JAX_PLATFORMS"], "cpu")
        self.assertEqual(cpu_environment["JAX_PLATFORM_NAME"], "cpu")
        self.assertEqual(
            gpu_environment["CUDA_VISIBLE_DEVICES"], sealed["gpu_devices"][0]
        )
        self.assertNotIn("JAX_PLATFORMS", gpu_environment)

    def test_numerical_replay_has_one_reference_comparison_per_candidate(self) -> None:
        sealed = contract()
        cases = [
            {
                "path": item["path"],
                "comparison": {
                    "mode": "json_metrics",
                    "prefix": "metrics:",
                    "tolerances": {"*": {"rtol": 0.0, "atol": 0.0}},
                },
            }
            for item in sealed["plan"]["cases"]
        ]
        comparison = {
            "outcome": "PASS",
            "reason": "fixture",
            "metric_evidence": {"count": 1},
            "difference_summary": {},
            "failure_count": 0,
            "failures_reported": 0,
            "failures_truncated": False,
            "failure_records_sha256": "0" * 64,
            "failures": [],
        }
        with (
            mock.patch.object(MATRIX.validation, "load_json", return_value={}),
            mock.patch.object(MATRIX.validation, "materialize_cases", return_value=cases),
            mock.patch.object(MATRIX, "_read_stdout", return_value="metrics:{}"),
            mock.patch.object(
                MATRIX,
                "_numerical_comparison",
                return_value=(comparison, "fixture"),
            ),
        ):
            result = MATRIX.derive_numerical_results(pathlib.Path("/out"), sealed)
        self.assertEqual(result["comparison_count"], 141)
        self.assertEqual(
            {(row["cycle_index"], row["sample_kind"]) for row in result["comparisons"]},
            {(0, "warmup"), (1, "measured"), (2, "measured")},
        )
        for row in result["comparisons"]:
            expected = (
                "gpmeep_v110_cpu"
                if (
                    row["case_path"] in MATRIX.GPMEEP_CPU_REFERENCE_CASES
                    and row["candidate_lane"] != "gpmeep_v110_cpu"
                )
                else "upstream_meep_cpu"
            )
            self.assertEqual(row["reference_lane"], expected)

    def test_waveguide_cuda_uses_gpmeep_cpu_reference(self) -> None:
        self.assertIn(
            MATRIX.WAVEGUIDE_CROSSING_CASE,
            MATRIX.GPMEEP_CPU_REFERENCE_CASES,
        )

    def test_waveguide_upstream_contract_diagnoses_only_continuous_optimizer_path(
        self,
    ) -> None:
        cases = MATRIX.validation.materialize_cases(
            MATRIX.validation.load_json(
                ROOT / "scripts/python-validation/manifest.json"
            ),
            ROOT,
        )
        waveguide = next(
            case
            for case in cases
            if case["path"] == MATRIX.WAVEGUIDE_CROSSING_CASE
        )
        comparison = MATRIX._waveguide_upstream_optimizer_contract(waveguide)
        tolerances = comparison["tolerances"]
        for name in MATRIX.WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES:
            with self.subTest(name=name):
                self.assertTrue(tolerances[name]["diagnostic_only"])
        for name in (
            "optimizer_smoke.epoch_count",
            "optimizer_smoke.epoch0_callback_count",
            "optimizer_smoke.epoch1_nlopt_evaluations",
            "optimizer_smoke.checkpoint_roundtrip_max_abs_difference",
            "*.adjoint_directional_derivative",
            "*.objective",
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    tolerances[name].get("diagnostic_only", False)
                )

    def test_waveguide_optimizer_self_oracle_binds_structure_and_progress(self) -> None:
        epoch0_design = [0.5] * MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        epoch0_design[0] = 0.55
        final_design = list(epoch0_design)
        final_design[1] = 0.6
        epoch0_step = sum((value - 0.5) ** 2 for value in epoch0_design) ** 0.5
        epoch1_step = sum(
            (final - epoch0) ** 2
            for epoch0, final in zip(epoch0_design, final_design, strict=True)
        ) ** 0.5
        final_l2 = sum(value * value for value in final_design) ** 0.5
        optimizer = {
            "beta_initial": 4.0,
            "beta_final": 8.0,
            "epoch_count": 2,
            "epoch0_beta": 4.0,
            "epoch1_beta": 8.0,
            "epoch0_callback_count": 2,
            "epoch1_callback_count": 2,
            "epoch0_nlopt_evaluations": 2,
            "epoch1_nlopt_evaluations": 2,
            "epoch0_result_code": 5,
            "epoch1_result_code": 5,
            "checkpoint_callback_count": 4,
            "checkpoint_completed_epochs": 2,
            "checkpoint_roundtrip_max_abs_difference": 0.0,
            "result_roundtrip_max_abs_difference": 0.0,
            "epoch0_design": epoch0_design,
            "final_design": final_design,
            "epoch0_design_step_l2": epoch0_step,
            "epoch1_design_step_l2": epoch1_step,
            "epoch0_gradient_l2_initial": 0.3,
            "epoch0_gradient_l2_final": 0.4,
            "epoch1_gradient_l2_initial": 0.5,
            "epoch1_gradient_l2_final": 0.6,
            "epoch0_objective_initial": 0.7,
            "epoch0_objective_final": 0.6,
            "epoch1_objective_initial": 0.5,
            "epoch1_objective_final": 0.4,
            "final_design_l2": final_l2,
            "run_timesteps": [
                MATRIX.WAVEGUIDE_OPTIMIZER_TIMESTEP
            ]
            * MATRIX.WAVEGUIDE_OPTIMIZER_RUN_COUNT,
        }
        payload = {
            "fixed_work": {
                "run_timesteps": [
                    MATRIX.WAVEGUIDE_FIXED_SLICE_TIMESTEP
                ]
                * MATRIX.WAVEGUIDE_FIXED_SLICE_RUN_COUNT,
            },
            "optimizer_smoke": optimizer,
        }
        for name in (
            "smoothed_beta10",
            "smoothed_beta1000",
            "tanh_beta10",
            "tanh_beta1000",
        ):
            payload[name] = {
                metric.split(".", 1)[1]: 1.0
                for metric in MATRIX.WAVEGUIDE_FIXED_PHYSICS_METRICS
                if metric.startswith(f"{name}.")
            }
        comparison = {
            "mode": "json_metrics",
            "prefix": MATRIX.GENERIC_METRIC_PREFIX,
            "tolerances": {"*": {"rtol": 0.0, "atol": 0.0}},
        }

        def encoded(value: dict[str, object]) -> str:
            return MATRIX.GENERIC_METRIC_PREFIX + json.dumps(
                value, sort_keys=True
            )

        result = MATRIX._waveguide_optimizer_self_oracle(
            encoded(payload), comparison
        )
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(
            result["design_size"], MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        )

        invalid_structure = copy.deepcopy(payload)
        invalid_structure["optimizer_smoke"]["epoch1_result_code"] = 4
        with self.assertRaisesRegex(Exception, "structural oracle"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(invalid_structure), comparison
            )

        invalid_fixed_work = copy.deepcopy(payload)
        invalid_fixed_work["fixed_work"]["run_timesteps"][0] -= 1
        with self.assertRaisesRegex(Exception, "structural oracle"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(invalid_fixed_work), comparison
            )

        invalid_bounds = copy.deepcopy(payload)
        invalid_bounds["optimizer_smoke"]["final_design"][0] = 1.1
        with self.assertRaisesRegex(Exception, "design bounds"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(invalid_bounds), comparison
            )

        invalid_l2 = copy.deepcopy(payload)
        invalid_l2["optimizer_smoke"]["final_design_l2"] = final_l2 + 0.1
        with self.assertRaisesRegex(Exception, "L2 oracle"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(invalid_l2), comparison
            )

        no_progress = copy.deepcopy(payload)
        no_progress["optimizer_smoke"]["epoch0_design"] = [
            0.5
        ] * MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        no_progress["optimizer_smoke"]["final_design"] = [
            0.5
        ] * MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        no_progress["optimizer_smoke"]["final_design_l2"] = (
            0.5 * MATRIX.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE**0.5
        )
        with self.assertRaisesRegex(Exception, "design-step oracle"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(no_progress), comparison
            )

        missing = copy.deepcopy(payload)
        del missing["smoothed_beta10"]["gradient_l2"]
        with self.assertRaisesRegex(Exception, "metric inventory"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(missing), comparison
            )

        substituted = copy.deepcopy(missing)
        substituted["phantom"] = {"gradient_l2": 1.0}
        with self.assertRaisesRegex(Exception, "metric inventory"):
            MATRIX._waveguide_optimizer_self_oracle(
                encoded(substituted), comparison
            )

    def test_waveguide_comparison_routes_upstream_and_cuda_differently(self) -> None:
        case = {"path": MATRIX.WAVEGUIDE_CROSSING_CASE}
        passing = {
            "outcome": "PASS",
            "failure_count": 0,
            "failures_reported": 0,
            "failures_truncated": False,
            "failure_records_sha256": "0" * 64,
            "failures": [],
        }
        oracle = {"outcome": "PASS"}
        strict = {"mode": "strict"}
        diagnostic = {"mode": "diagnostic"}
        with (
            mock.patch.object(
                MATRIX,
                "_comparison_contract",
                return_value=(strict, "manifest-json-metrics"),
            ),
            mock.patch.object(
                MATRIX,
                "_waveguide_upstream_optimizer_contract",
                return_value=diagnostic,
            ),
            mock.patch.object(
                MATRIX.validation,
                "compare_json_metrics",
                return_value=passing,
            ) as compare,
            mock.patch.object(
                MATRIX,
                "_waveguide_optimizer_self_oracle",
                return_value=oracle,
            ),
            mock.patch.object(
                MATRIX,
                "_waveguide_pairwise_gate",
                return_value=passing,
            ),
        ):
            _result, cpu_method = MATRIX._numerical_comparison(
                case, "gpmeep_v110_cpu", "reference", "candidate"
            )
            self.assertIn("diagnostic-optimizer-trajectory", cpu_method)
            self.assertIs(compare.call_args.args[2], diagnostic)

            _result, cuda_method = MATRIX._numerical_comparison(
                case, "gpmeep_v110_cuda_gpu1", "reference", "candidate"
            )
            self.assertIn("gpmeep-cpu-cross-lane", cuda_method)
            self.assertIs(compare.call_args.args[2], strict)

    def test_cuda_sample_replay_rejects_missing_backend_contract(self) -> None:
        sealed = contract()
        sealed["runtime_inputs"].update(
            v110_installed_report_id="candidate-receipt",
            upstream_receipt_id="upstream-receipt",
        )
        task = next(
            item
            for item in sealed["tasks"]
            if item["runtime"] == "v110" and item["topology"] == "cuda1"
        )
        report = {
            "schema": "fixture",
            "outcome": "PASS",
            "case_path": task["case_path"],
            "topology": task["topology"],
            "sample_kind": task["sample_kind"],
            "cycle_index": task["cycle_index"],
            "receipt_id": "candidate-receipt",
            "gpu_devices": sealed["gpu_devices"],
            "timing": {
                "contract": task["timing_contract"],
                "primary_seconds": 1.0,
            },
            "phase_profile": {
                "schema": "gpmeep-m3-ten-phase-profile-v1",
                "phases": {
                    f"phase-{index}": {
                        "available": index > 0,
                        "seconds": None if index == 0 else 0.1,
                    }
                    for index in range(10)
                },
            },
            "lane": {
                "requested_backend": "cuda",
                "rank_records": [
                    {
                        "backend_contract": "PASS",
                        "selected_device_identifier": sealed["gpu_devices"][0],
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary).resolve()
            sample = output.joinpath(*MATRIX.task_sample_relative(task).parts)
            sample.mkdir(parents=True)
            (sample / "COMPLETE").write_text("{}", encoding="utf-8")
            (sample / "report.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(MATRIX, "_sample_verifier", return_value=lambda _root: report):
                retained, _ = MATRIX.derive_sample_replay(task, output, sealed)
                self.assertEqual(retained["timing"]["primary_seconds"], 1.0)
                report["lane"]["rank_records"][0]["backend_contract"] = "FAIL"
                with self.assertRaisesRegex(Exception, "fallback/device"):
                    MATRIX.derive_sample_replay(task, output, sealed)

    def test_normative_registry_replays_output_records_and_computes_blockers(self) -> None:
        calls = []

        class Validator:
            @staticmethod
            def validate_registry(**kwargs):
                calls.append(kwargs)
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

        output = pathlib.Path("/retained/equal-work")
        with mock.patch.object(
            MATRIX, "_load_normative_inversion_registry", return_value=Validator
        ):
            result = MATRIX.validate_normative_inversion_registry(output, ROOT)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(
            calls,
            [
                {
                    "require_closed": False,
                    "evidence_root": output,
                    "records_directory": output / "inversion-records",
                }
            ],
        )

        class OpenWithComputedBlocker(Validator):
            @staticmethod
            def validate_registry(**kwargs):
                value = Validator.validate_registry(**kwargs)
                value.update(
                    state="OPEN",
                    blocking_count=1,
                    closed_count=12,
                    open_count=1,
                    blocking_workloads=["python/examples/waveguide_crossing.py"],
                    open_workloads=["python/examples/waveguide_crossing.py"],
                )
                return value

        with mock.patch.object(
            MATRIX,
            "_load_normative_inversion_registry",
            return_value=OpenWithComputedBlocker,
        ):
            result = MATRIX.validate_normative_inversion_registry(output, ROOT)
        self.assertEqual(result["state"], "OPEN")
        self.assertEqual(result["blocking_count"], 1)

        class FalsePassWithBlocker(OpenWithComputedBlocker):
            @staticmethod
            def validate_registry(**kwargs):
                value = OpenWithComputedBlocker.validate_registry(**kwargs)
                value["state"] = "PASS"
                return value

        with mock.patch.object(
            MATRIX,
            "_load_normative_inversion_registry",
            return_value=FalsePassWithBlocker,
        ):
            with self.assertRaisesRegex(Exception, "closure computation differs"):
                MATRIX.validate_normative_inversion_registry(output, ROOT)


if __name__ == "__main__":
    unittest.main()
