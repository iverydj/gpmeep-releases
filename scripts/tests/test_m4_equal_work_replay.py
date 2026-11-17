#!/usr/bin/env python3
"""Tests for the pure M4 equal-work result replay."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PHASES = (
    "initialization",
    "fdtd_timestep",
    "source",
    "boundary",
    "dispersive_ade",
    "dft_reduction",
    "host_device_transfer_and_sync",
    "mpi",
    "python_adjoint_mpb",
    "io",
)
SOURCE = ROOT / "scripts/python-validation/m4_equal_work_replay.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m4_equal_work_replay_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)
REGISTRY_SOURCE = ROOT / "scripts/validate-v110-inversion-registry.py"
REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m4_registry_boundary_test", REGISTRY_SOURCE
)
assert REGISTRY_SPEC is not None and REGISTRY_SPEC.loader is not None
REGISTRY = importlib.util.module_from_spec(REGISTRY_SPEC)
sys.modules[REGISTRY_SPEC.name] = REGISTRY
REGISTRY_SPEC.loader.exec_module(REGISTRY)
VALIDATION_SOURCE = ROOT / "scripts/python-validation/run_validation.py"
VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m4_real_comparison_boundary_test", VALIDATION_SOURCE
)
assert VALIDATION_SPEC is not None and VALIDATION_SPEC.loader is not None
VALIDATION = importlib.util.module_from_spec(VALIDATION_SPEC)
sys.modules[VALIDATION_SPEC.name] = VALIDATION
VALIDATION_SPEC.loader.exec_module(VALIDATION)


def reports() -> list[dict[str, object]]:
    plan = REPLAY.equal_work_plan.load_plan(ROOT)
    result = []
    lane_seconds = {
        "upstream_meep_cpu": 12.0,
        "gpmeep_v110_cpu": 10.0,
        "gpmeep_v103_cuda_gpu1": 6.0,
        "gpmeep_v110_cuda_gpu1": 5.0,
        "gpmeep_v103_cuda_gpu2": 4.0,
        "gpmeep_v110_cuda_gpu2": 3.0,
    }
    for task in REPLAY.equal_work_plan.build_tasks(plan):
        devices = (
            ["GPU-one", "GPU-two"]
            if task["topology"].endswith("cuda2")
            else ["GPU-one"]
            if task["backend"] == "cuda"
            else []
        )
        result.append(
            {
                "schema": {
                    "upstream": "gpmeep-v110-upstream-performance-sample-v1",
                    "v103": "gpmeep-v110-frozen-v103-performance-sample-v1",
                    "v110": "gpmeep-m3-performance-sample-v1",
                }[task["runtime"]],
                "outcome": "PASS",
                "case_path": task["case_path"],
                "topology": task["topology"],
                "sample_kind": task["sample_kind"],
                "cycle_index": task["cycle_index"],
                "receipt_id": hashlib.sha256(task["runtime"].encode()).hexdigest(),
                "lane": {"spec": {"devices": devices}},
                "timing": {
                    "contract": task["timing_contract"],
                    "primary_seconds": lane_seconds[task["lane_id"]]
                    + 0.1 * task["cycle_index"],
                },
                "phase_profile": {
                    "schema": "gpmeep-m3-ten-phase-profile-v1",
                    "phases": {
                        phase_name: {
                            "available": index != 0,
                            "seconds": None if index == 0 else float(index),
                            "basis": "fixture basis",
                            "native_sinks": [] if index == 0 else [f"sink-{index}"],
                        }
                        for index, phase_name in enumerate(PHASES)
                    },
                },
            }
        )
    return result


def compact_numerical_result() -> dict[str, object]:
    return {
        "outcome": "PASS",
        "reason": "all JSON metrics are within tolerance",
        "metric_evidence": {
            "schema": "gpmeep-json-metric-evidence-v2",
            "count": 1,
            "passed_count": 1,
            "failure_count": 0,
            "metric_keys_sha256": "1" * 64,
            "cpu_sha256": "2" * 64,
            "cuda_sha256": "3" * 64,
            "comparison_records_sha256": "4" * 64,
            "inline_limit": 256,
            "inline_byte_limit": 1048576,
            "flattened_storage_bytes": 8,
            "inlined": False,
        },
        "difference_summary": {},
        "failure_count": 0,
        "failures_reported": 0,
        "failures_truncated": False,
        "failure_records_sha256": REGISTRY.EMPTY_JSON_FAILURE_RECORDS_SHA256,
        "failures": [],
    }


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


def waveguide_optimizer_self_oracle() -> dict[str, object]:
    return {
        "outcome": "PASS",
        "design_size": REGISTRY.WAVEGUIDE_OPTIMIZER_DESIGN_SIZE,
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
                    REGISTRY.WAVEGUIDE_FIXED_SLICE_TIMESTEP
                )
                for index in range(
                    REGISTRY.WAVEGUIDE_FIXED_SLICE_RUN_COUNT
                )
            },
            **{
                f"optimizer_smoke.run_timesteps[{index}]": float(
                    REGISTRY.WAVEGUIDE_OPTIMIZER_TIMESTEP
                )
                for index in range(REGISTRY.WAVEGUIDE_OPTIMIZER_RUN_COUNT)
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


def numerical_method_and_result(
    case_path: str, lane: str
) -> tuple[str, dict[str, object]]:
    method = REGISTRY._expected_m4_numerical_method(case_path, lane)
    if case_path == REGISTRY.ABSORBER_CASE:
        result = {
            "outcome": "PASS",
            "failure_count": 0,
            "pairwise": compact_numerical_result(),
            "scale_aware_trace": {
                "outcome": "PASS",
                "maximum_peak_normalized_error": 1.0e-4,
                "maximum_rms_peak_normalized_error": 1.0e-5,
                "observed_peak": 1.0,
                "observed_maximum_absolute_error": 1.0e-6,
                "observed_peak_normalized_maximum_error": 1.0e-6,
                "observed_rms_peak_normalized_error": 5.0e-7,
                "components": {
                    "real": {"count": 4, "maximum_absolute_error": 1.0e-6},
                    "imag": {"count": 4, "maximum_absolute_error": 2.0e-7},
                },
            },
        }
    elif case_path == REGISTRY.WAVEGUIDE_CROSSING_CASE:
        pairwise = compact_numerical_result()
        pairwise["metric_evidence"].update(
            count=REGISTRY.WAVEGUIDE_EXPECTED_METRIC_COUNT,
            passed_count=REGISTRY.WAVEGUIDE_EXPECTED_METRIC_COUNT,
            metric_keys_sha256=(
                REGISTRY.WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256
            ),
        )
        if lane == "gpmeep_v110_cpu":
            pairwise["metric_evidence"].update(
                diagnostic_only_count=(
                    REGISTRY.WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
                ),
                diagnostic_only_out_of_tolerance_count=109,
            )
            pairwise["difference_summary"] = (
                waveguide_diagnostic_difference_summary()
            )
        self_oracle = waveguide_optimizer_self_oracle()
        result = {
            "outcome": "PASS",
            "failure_count": 0,
            "pairwise": pairwise,
            "reference_self_oracle": json.loads(json.dumps(self_oracle)),
            "candidate_self_oracle": json.loads(json.dumps(self_oracle)),
        }
    elif case_path == REGISTRY.ADJOINT_CYL_CASE:
        self_oracle = {
            "outcome": "PASS",
            "bounds": [
                {
                    "case_index": index,
                    "objective_relative_error": 1.0e-6,
                    "objective_limit": 1.0e-3,
                    "relative_derivative_error": 0.1,
                    "reported_relative_derivative_error": 0.1,
                    "derivative_limit": 0.6 if index < 2 else 0.4,
                }
                for index in range(4)
            ],
        }
        fixed_work = {
            "outcome": "PASS",
            "run_count": 16,
            "timesteps": list(REGISTRY.ADJOINT_CYL_EXPECTED_TIMESTEPS),
            "timestep_deltas": list(REGISTRY.ADJOINT_CYL_EXPECTED_TIMESTEPS),
            "meep_times": list(REGISTRY.ADJOINT_CYL_EXPECTED_MEEP_TIMES),
        }
        result = {
            "outcome": "PASS",
            "failure_count": 0,
            "pairwise": compact_numerical_result(),
            "reference_self_oracle": json.loads(json.dumps(self_oracle)),
            "candidate_self_oracle": json.loads(json.dumps(self_oracle)),
            "reference_fixed_work": json.loads(json.dumps(fixed_work)),
            "candidate_fixed_work": json.loads(json.dumps(fixed_work)),
        }
    else:
        result = compact_numerical_result()
    return method, result


def numerical_comparisons(plan: dict[str, object]) -> list[dict[str, object]]:
    comparisons = []
    for case in plan["cases"]:
        lanes = [
            "gpmeep_v110_cpu",
            "gpmeep_v103_cuda_gpu1",
            "gpmeep_v110_cuda_gpu1",
        ]
        if case["gpu2"]:
            lanes.extend(["gpmeep_v103_cuda_gpu2", "gpmeep_v110_cuda_gpu2"])
        for cycle in (0, 1, 2):
            for lane in lanes:
                reference_lane = (
                    "gpmeep_v110_cpu"
                    if case["path"] in REGISTRY.GPMEEP_CPU_REFERENCE_CASES
                    and lane != "gpmeep_v110_cpu"
                    else "upstream_meep_cpu"
                )
                method, result = numerical_method_and_result(case["path"], lane)
                comparisons.append(
                    {
                        "cycle_index": cycle,
                        "sample_kind": "warmup" if cycle == 0 else "measured",
                        "case_path": case["path"],
                        "reference_lane": reference_lane,
                        "candidate_lane": lane,
                        "method": method,
                        "result": result,
                    }
                )
    return comparisons


class M4EqualWorkReplayTests(unittest.TestCase):
    def test_exact_census_raw_seconds_and_explicit_ratios(self) -> None:
        plan = REPLAY.equal_work_plan.load_plan(ROOT)
        result = REPLAY.derive_results(plan, reports())
        self.assertEqual(result["task_count"], 180)
        self.assertEqual(result["measured_task_count"], 120)
        self.assertEqual(len(result["raw_seconds"]), 180)
        edge = result["cases"]["python/examples/edge_emitter_3D.py"]
        self.assertIn("upstream_meep_cpu", edge["lanes"])
        self.assertIn("gpmeep_v110_cpu", edge["lanes"])
        self.assertIn("v110-gpu1_to_v110-gpu2", edge["ratios"])
        self.assertAlmostEqual(
            edge["lanes"]["gpmeep_v110_cpu"]["measured_average_seconds"],
            10.15,
        )
        self.assertIsNone(
            edge["lanes"]["gpmeep_v110_cpu"]["phases"]["initialization"][
                "measured_average_seconds"
            ]
        )
        self.assertEqual(
            edge["ratios"]["v110-cpu_to_v110-gpu1"]["paired_ratios"],
            [10.1 / 5.1, 10.2 / 5.2],
        )

    def test_non_gpu2_case_does_not_invent_gpu2_ratios(self) -> None:
        result = REPLAY.derive_results(
            REPLAY.equal_work_plan.load_plan(ROOT), reports()
        )
        phase = result["cases"]["python/examples/phase_in_material.py"]
        self.assertNotIn("gpmeep_v110_cuda_gpu2", phase["lanes"])
        self.assertNotIn("v110-gpu1_to_v110-gpu2", phase["ratios"])

    def test_rejects_missing_and_misidentified_samples(self) -> None:
        plan = REPLAY.equal_work_plan.load_plan(ROOT)
        values = reports()
        with self.assertRaisesRegex(Exception, "census"):
            REPLAY.derive_results(plan, values[:-1])
        values[0]["case_path"] = "wrong.py"
        with self.assertRaisesRegex(Exception, "identity"):
            REPLAY.derive_results(plan, values)

    def test_rejects_nonpositive_and_nonfinite_seconds(self) -> None:
        plan = REPLAY.equal_work_plan.load_plan(ROOT)
        for bad in (0.0, float("inf"), float("nan"), True):
            values = reports()
            values[0]["timing"]["primary_seconds"] = bad
            with self.assertRaisesRegex(Exception, "positive and finite"):
                REPLAY.derive_results(plan, values)

    def test_emits_registry_v2_records_with_two_live_sample_links(self) -> None:
        plan = REPLAY.equal_work_plan.load_plan(ROOT)
        values = reports()
        tasks = REPLAY.equal_work_plan.build_tasks(plan)
        evidence = {
            task["task_id"]: {
                "path": f"samples/{task['task_id']}/report.json",
                "sha256": hashlib.sha256(task["task_id"].encode()).hexdigest(),
            }
            for task in tasks
        }
        inventory_path = ROOT / "packaging/v110-performance-inversion-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        result = REPLAY.derive_inversion_records(
            plan,
            values,
            evidence,
            inventory,
            "packaging/v110-performance-inversion-inventory.json",
            hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            {"path": "numerical-replay.json", "sha256": "1" * 64},
        )
        self.assertEqual(len(result), 13)
        edge = next(
            record
            for record in result
            if record["workload_id"] == "python/examples/edge_emitter_3D.py"
        )
        lane = edge["comparison"]["lanes"]["gpmeep_v110_cuda_gpu2"]
        self.assertEqual(lane["measured_seconds"], [3.1, 3.2])
        self.assertEqual(len(lane["samples"]), 2)
        self.assertEqual(
            edge["comparison"]["ratios"]["gpu2"][
                "upstream_cpu_over_v110_cuda"
            ]["paired_cycle_values"],
            [12.1 / 3.1, 12.2 / 3.2],
        )
        self.assertEqual(
            edge["phase_decomposition"]["upstream_meep_cpu"]["initialization"][
                "reason_code"
            ],
            "initialization_not_separable",
        )

    def test_generated_numerical_evidence_crosses_real_registry_boundary(self) -> None:
        plan = REPLAY.equal_work_plan.load_plan(ROOT)
        values = reports()
        tasks = REPLAY.equal_work_plan.build_tasks(plan)
        sample_evidence = {
            task["task_id"]: {
                "path": f"samples/{task['task_id']}/report.json",
                "sha256": hashlib.sha256(task["task_id"].encode()).hexdigest(),
            }
            for task in tasks
        }
        inventory_path = ROOT / "packaging/v110-performance-inversion-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        comparisons = numerical_comparisons(plan)
        numerical = {
            "schema": "gpmeep-v110-m4-numerical-replay-v1",
            "outcome": "PASS",
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            numerical_path = root / "numerical-replay.json"
            numerical_path.write_text(json.dumps(numerical, sort_keys=True))
            numerical_ref = {
                "path": numerical_path.name,
                "sha256": hashlib.sha256(numerical_path.read_bytes()).hexdigest(),
            }
            records = REPLAY.derive_inversion_records(
                plan,
                values,
                sample_evidence,
                inventory,
                "packaging/v110-performance-inversion-inventory.json",
                inventory_sha,
                numerical_ref,
            )
            self.assertEqual(len(records), 13)
            for record in records:
                reference = record["classification_evidence"][0]
                self.assertEqual(reference["kind"], "numerical_regression")
                self.assertEqual(
                    REGISTRY._evidence(
                        reference,
                        "generated numerical evidence",
                        root,
                        expected_workload_id=record["workload_id"],
                        expected_campaign_id=inventory["source_campaign"]["id"],
                        expected_inventory_sha256=inventory_sha,
                        expected_classification="unknown",
                    ),
                    "numerical_regression",
                )

    def test_registry_rejects_forged_or_omitted_m4_numerical_gates(self) -> None:
        method, result = numerical_method_and_result(
            "python/examples/phase_in_material.py", "gpmeep_v110_cpu"
        )
        with self.assertRaisesRegex(REGISTRY.RegistryError, "method differs"):
            REGISTRY._verify_m4_numerical_result(
                "python/examples/phase_in_material.py",
                "gpmeep_v110_cpu",
                "fixture",
                result,
                "ordinary",
            )

        method, result = numerical_method_and_result(
            REGISTRY.ABSORBER_CASE, "gpmeep_v110_cuda_gpu1"
        )
        del result["scale_aware_trace"]
        with self.assertRaisesRegex(REGISTRY.RegistryError, "key inventory"):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.ABSORBER_CASE,
                "gpmeep_v110_cuda_gpu1",
                method,
                result,
                "absorber",
            )

        method, result = numerical_method_and_result(
            REGISTRY.ADJOINT_CYL_CASE, "gpmeep_v110_cuda_gpu1"
        )
        del result["candidate_fixed_work"]
        with self.assertRaisesRegex(REGISTRY.RegistryError, "key inventory"):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.ADJOINT_CYL_CASE,
                "gpmeep_v110_cuda_gpu1",
                method,
                result,
                "cylindrical",
            )

        method, result = numerical_method_and_result(
            REGISTRY.WAVEGUIDE_CROSSING_CASE, "gpmeep_v110_cuda_gpu1"
        )
        result["pairwise"]["metric_evidence"][
            "diagnostic_only_count"
        ] = REGISTRY.WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
        result["pairwise"]["metric_evidence"][
            "diagnostic_only_out_of_tolerance_count"
        ] = 0
        with self.assertRaisesRegex(REGISTRY.RegistryError, "weakened"):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.WAVEGUIDE_CROSSING_CASE,
                "gpmeep_v110_cuda_gpu1",
                method,
                result,
                "waveguide CUDA",
            )

        method, result = numerical_method_and_result(
            REGISTRY.WAVEGUIDE_CROSSING_CASE, "gpmeep_v110_cpu"
        )
        del result["pairwise"]["metric_evidence"]["diagnostic_only_count"]
        del result["pairwise"]["metric_evidence"][
            "diagnostic_only_out_of_tolerance_count"
        ]
        with self.assertRaisesRegex(
            REGISTRY.RegistryError, "diagnostic rules were hidden"
        ):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.WAVEGUIDE_CROSSING_CASE,
                "gpmeep_v110_cuda_gpu1",
                REGISTRY._expected_m4_numerical_method(
                    REGISTRY.WAVEGUIDE_CROSSING_CASE,
                    "gpmeep_v110_cuda_gpu1",
                ),
                result,
                "waveguide hidden CUDA diagnostic",
            )

        method, result = numerical_method_and_result(
            REGISTRY.WAVEGUIDE_CROSSING_CASE, "gpmeep_v110_cpu"
        )
        result["pairwise"]["metric_evidence"]["diagnostic_only_count"] -= 1
        with self.assertRaisesRegex(REGISTRY.RegistryError, "diagnostic scope"):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.WAVEGUIDE_CROSSING_CASE,
                "gpmeep_v110_cpu",
                method,
                result,
                "waveguide CPU",
            )

        for lane in ("gpmeep_v110_cpu", "gpmeep_v110_cuda_gpu1"):
            for field, value in (
                ("count", 929),
                ("passed_count", 929),
                ("metric_keys_sha256", "9" * 64),
            ):
                with self.subTest(lane=lane, field=field):
                    method, result = numerical_method_and_result(
                        REGISTRY.WAVEGUIDE_CROSSING_CASE, lane
                    )
                    result["pairwise"]["metric_evidence"][field] = value
                    with self.assertRaisesRegex(
                        REGISTRY.RegistryError, "metric (evidence|inventory)"
                    ):
                        REGISTRY._verify_m4_numerical_result(
                            REGISTRY.WAVEGUIDE_CROSSING_CASE,
                            lane,
                            method,
                            result,
                            "waveguide inventory",
                        )

        method, result = numerical_method_and_result(
            REGISTRY.WAVEGUIDE_CROSSING_CASE, "gpmeep_v110_cuda_gpu1"
        )
        result["candidate_self_oracle"][
            "reported_epoch1_design_step_l2"
        ] = 0.3
        with self.assertRaisesRegex(
            REGISTRY.RegistryError, "optimizer evidence"
        ):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.WAVEGUIDE_CROSSING_CASE,
                "gpmeep_v110_cuda_gpu1",
                method,
                result,
                "waveguide progress",
            )

    def test_real_json_comparison_passes_registry_boundary(self) -> None:
        comparison = {
            "mode": "json_metrics",
            "prefix": "gpmeep-real-boundary:",
            "tolerances": {"value": {"rtol": 0.0, "atol": 0.0}},
        }
        text = 'gpmeep-real-boundary:{"value":1.0}'
        produced = VALIDATION.compare_json_metrics(text, text, comparison)
        compact_keys = {
            "outcome", "reason", "metric_evidence", "difference_summary",
            "failure_count", "failures_reported", "failures_truncated",
            "failure_records_sha256", "failures",
        }
        compact = {key: produced[key] for key in compact_keys}
        REGISTRY._verify_m4_numerical_result(
            "python/examples/phase_in_material.py",
            "gpmeep_v110_cpu",
            "manifest-json-metrics",
            compact,
            "real producer result",
        )

        method, result = numerical_method_and_result(
            REGISTRY.ADJOINT_CYL_CASE, "gpmeep_v110_cuda_gpu1"
        )
        result["candidate_self_oracle"]["bounds"][0][
            "reported_relative_derivative_error"
        ] = -1.0
        with self.assertRaisesRegex(REGISTRY.RegistryError, r"bound\[0\] differs"):
            REGISTRY._verify_m4_numerical_result(
                REGISTRY.ADJOINT_CYL_CASE,
                "gpmeep_v110_cuda_gpu1",
                method,
                result,
                "cylindrical",
            )


if __name__ == "__main__":
    unittest.main()
