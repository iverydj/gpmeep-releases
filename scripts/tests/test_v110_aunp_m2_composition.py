from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[2]
LAB = REPO / "scripts" / "v110-lab"
for path in (REPO / "scripts", LAB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_aunp_m2_composition as controller
import run_aunp_m2_composition_sample as sample


class V110AuNPM2CompositionTest(unittest.TestCase):
    def test_modes_task_plan_and_thread_allocation_are_exact(self):
        self.assertEqual(
            (
                "off",
                "c1+c3",
                "c4_t2",
                "c4_t4",
                "c1+c3+c4_t2",
                "c1+c3+c4_t4",
            ),
            controller.MODES,
        )
        plan = controller.m1.task_plan()
        self.assertEqual(24, len(plan))
        self.assertEqual(24, len({task["task_id"] for task in plan}))
        for repeat in range(2):
            gpu1 = [
                task["feature_mode"]
                for task in plan
                if task["repeat"] == repeat and task["topology"] == "gpu1"
            ]
            gpu2 = [
                task["feature_mode"]
                for task in plan
                if task["repeat"] == repeat and task["topology"] == "gpu2"
            ]
            expected = list(controller.MODES)
            if repeat == 1:
                expected.reverse()
            self.assertEqual(expected, gpu1)
            self.assertEqual(list(reversed(expected)), gpu2)
        expected_threads = {
            "off": 1,
            "c1+c3": 1,
            "c4_t2": 2,
            "c4_t4": 4,
            "c1+c3+c4_t2": 2,
            "c1+c3+c4_t4": 4,
        }
        for mode, threads in expected_threads.items():
            task = {"feature_mode": mode}
            self.assertEqual(threads, controller._omp_threads(task))
            self.assertEqual(
                f"slot:PE={threads}", controller.base._mpi_map_by(task)
            )

    def test_mode_environment_is_exact_and_unknown_modes_fail(self):
        for mode in sample.MODES:
            c1, c3, c4, threads = sample._mode_configuration(mode)
            environment = sample._expected_feature_environment(mode)
            self.assertEqual(str(threads), environment["OMP_NUM_THREADS"])
            self.assertEqual("FALSE", environment["OMP_DYNAMIC"])
            self.assertEqual("close", environment["OMP_PROC_BIND"])
            self.assertEqual("cores", environment["OMP_PLACES"])
            self.assertEqual("1" if c1 else None, environment[sample.m1.c3.C1_ENABLE])
            self.assertEqual(None if c1 else "1", environment[sample.m1.c3.C1_DISABLE])
            self.assertEqual(
                "1" if c3 else None,
                environment[sample.m1.c3.FEATURE_ENABLE],
            )
            self.assertEqual(
                None if c3 else "1",
                environment[sample.m1.c3.FEATURE_DISABLE],
            )
            self.assertEqual(
                "1" if c4 else None,
                environment[sample.m1.c4.FEATURE_ENABLE],
            )
            self.assertEqual(
                None if c4 else "1",
                environment[sample.m1.c4.FEATURE_DISABLE],
            )
        with self.assertRaisesRegex(
            sample.base.common.WorkloadError, "unknown M2 composition mode"
        ):
            sample._mode_configuration("c4_t8")
        with self.assertRaisesRegex(controller.base.CanaryError, "mode is invalid"):
            controller._omp_threads({"feature_mode": 4})

    def test_counter_dispatch_and_exact_openmp_team(self):
        for mode in sample.MODES:
            c1_enabled, c3_enabled, c4_enabled, threads = (
                sample._mode_configuration(mode)
            )
            material_records = [
                {
                    "rank": rank,
                    "final_candidate_statistics": {
                        "configured_threads": threads if c4_enabled else 0,
                        "observed_threads": threads if c4_enabled else 0,
                    },
                }
                for rank in range(2)
            ]
            c4_totals = {
                "configured_threads": threads if c4_enabled else 0,
                "observed_threads": threads if c4_enabled else 0,
            }
            with (
                mock.patch.object(
                    sample.m1,
                    "_base_validate_feature_counters",
                    return_value={"component": "c1"},
                ) as c1,
                mock.patch.object(
                    sample.m1.c3,
                    "_validate_feature_counters",
                    return_value={"component": "c3"},
                ) as c3,
                mock.patch.object(
                    sample.m1.c4,
                    "_validate_feature_counters",
                    return_value=c4_totals,
                ) as c4,
                mock.patch.object(
                    sample.m1, "_isolate_c3_ordinary_counters", return_value=[]
                ),
                mock.patch.object(
                    sample.m1,
                    "_component_records",
                    return_value=material_records,
                ),
            ):
                totals = sample._validate_feature_counters([], mode)
            self.assertEqual(["c1", "c3", "c4"], sorted(totals))
            self.assertEqual(
                "enabled" if c1_enabled else "disabled", c1.call_args.args[1]
            )
            self.assertEqual(
                "enabled" if c3_enabled else "disabled", c3.call_args.args[1]
            )
            self.assertEqual(
                "enabled" if c4_enabled else "disabled", c4.call_args.args[1]
            )

    def test_heterogeneous_rank_openmp_team_is_rejected(self):
        for field in ("configured_threads", "observed_threads"):
            with self.subTest(field=field):
                material_records = [
                    {
                        "rank": rank,
                        "final_candidate_statistics": {
                            "configured_threads": 4,
                            "observed_threads": 4,
                        },
                    }
                    for rank in range(2)
                ]
                material_records[0]["final_candidate_statistics"][field] = 2
                with (
                    mock.patch.object(
                        sample.m1,
                        "_base_validate_feature_counters",
                        return_value={},
                    ),
                    mock.patch.object(
                        sample.m1.c3,
                        "_validate_feature_counters",
                        return_value={},
                    ),
                    mock.patch.object(
                        sample.m1.c4,
                        "_validate_feature_counters",
                        return_value={
                            "configured_threads": 4,
                            "observed_threads": 4,
                        },
                    ),
                    mock.patch.object(
                        sample.m1,
                        "_isolate_c3_ordinary_counters",
                        return_value=[],
                    ),
                    mock.patch.object(
                        sample.m1,
                        "_component_records",
                        return_value=material_records,
                    ),
                ):
                    with self.assertRaisesRegex(
                        sample.base.common.WorkloadError,
                        f"rank 0 {field} differs",
                    ):
                        sample._validate_feature_counters([], "c4_t4")

    def test_c4_team_mismatch_counter_fails_closed(self):
        c4 = sample.m1.c4
        snapshot = {name: 0 for name in c4.COUNTERS + c4.MAX_GAUGES}
        snapshot.update(
            {
                "feature_checks": 1,
                "enabled_checks": 1,
                "eligible_checks": 1,
                "epsilon_parallel_calls": 1,
                "epsilon_parallel_points": 1,
                "sigma_parallel_calls": 1,
                "sigma_parallel_points": 1,
                "team_observations": 1,
                "team_mismatch_observations": 1,
                "configured_threads": 4,
                "observed_threads": 4,
                "geometry_epoch": 1,
                "prism_preflight_calls": 1,
                "reset_generation": 1,
                "prism_preflight_complete": True,
                "counter_overflow": False,
            }
        )
        rank_records = [
            {
                "records": [
                    {
                        "candidate_statistics_before": dict(snapshot),
                        "candidate_statistics_after": dict(snapshot),
                    }
                    for _ in range(2)
                ],
                "final_candidate_statistics": dict(snapshot),
            }
        ]
        with mock.patch.object(
            c4,
            "_validate_material_rank_distribution",
            return_value={
                "material_array_count": 1,
                "material_array_bytes": 4,
                "susceptibility_array_count": 1,
            },
        ):
            with self.assertRaisesRegex(
                c4.base.common.WorkloadError,
                "fail-closed rejection occurred",
            ):
                c4._validate_feature_counters(rank_records, "enabled")

    @staticmethod
    def _synthetic_tasks() -> list[dict[str, object]]:
        factors = {
            "off": (1.0, 1.0, 1.0),
            "c1+c3": (1.0, 0.90, 0.95),
            "c4_t2": (0.70, 0.85, 0.75),
            "c4_t4": (0.50, 0.80, 0.70),
            "c1+c3+c4_t2": (0.70, 0.75, 0.70),
            "c1+c3+c4_t4": (0.50, 0.70, 0.65),
        }
        tasks: list[dict[str, object]] = []
        for planned in controller.m1.task_plan():
            task = dict(planned)
            topology_factor = 1.0 if task["topology"] == "gpu1" else 0.6
            repeat_factor = 1.0 + 0.01 * int(task["repeat"])
            initialization, fdtd, end_to_end = factors[str(task["feature_mode"])]
            ranks = int(task["ranks"])
            devices = {
                f"GPU-{index}": {
                    "peak_attributed_memory_mib": 100.0,
                    "start_temperature_c": 35.0,
                    "maximum_temperature_c": 60.0,
                    "throttle_clean": True,
                }
                for index in range(ranks)
            }
            task.update(
                {
                    "initialization_wall_seconds": 10
                    * initialization
                    * repeat_factor,
                    "fdtd_wall_seconds": 20
                    * fdtd
                    * topology_factor
                    * repeat_factor,
                    "end_to_end_seconds": 40
                    * end_to_end
                    * topology_factor
                    * repeat_factor,
                    "peak_host_rss_bytes_by_rank": [1000] * ranks,
                    "peak_host_rss_bytes_total": 1000 * ranks,
                    "observable_signature": [{"same": True}],
                    "runtime_artifacts": {
                        "libmeep": {"sha256": "a" * 64}
                    },
                    "telemetry": {"devices": devices},
                }
            )
            tasks.append(task)
        return tasks

    def test_result_derivation_reports_composition_and_thread_scaling(self):
        result = controller.derive_results(self._synthetic_tasks())
        self.assertEqual("M2_COMPOSITION_ACCEPTED", result["promotion_outcome"])
        self.assertTrue(all(result["gates"].values()))
        comparison = result["composition_analysis"]["t4_combined_vs_c4"][
            "gpu1"
        ]["end_to_end_seconds"]
        self.assertAlmostEqual(0.65 / 0.70, comparison["candidate_to_baseline_ratio"])
        thread = result["composition_analysis"]["c4_t2_to_t4"]["gpu2"][
            "initialization_wall_seconds"
        ]
        self.assertAlmostEqual(0.50 / 0.70, thread["candidate_to_baseline_ratio"])

    def test_composition_regression_blocks_even_when_better_than_off(self):
        gate = (
            "combined_lane_no_fdtd_or_end_to_end_regression_over_3_percent_vs_c4"
        )
        for metric in ("fdtd_wall_seconds", "end_to_end_seconds"):
            with self.subTest(metric=metric):
                tasks = self._synthetic_tasks()
                for task in tasks:
                    if task["feature_mode"] == "c1+c3+c4_t2":
                        matching = next(
                            candidate
                            for candidate in tasks
                            if candidate["repeat"] == task["repeat"]
                            and candidate["topology"] == task["topology"]
                            and candidate["feature_mode"] == "c4_t2"
                        )
                        task[metric] = float(matching[metric]) * 1.04
                result = controller.derive_results(tasks)
                self.assertFalse(result["gates"][gate])
                self.assertEqual(
                    "M2_COMPOSITION_BLOCKED", result["promotion_outcome"]
                )

    def test_attestation_cli_and_support_sources_are_m2_specific(self):
        args = controller.parse_args(
            [
                "attest-runtime",
                "--output",
                "/tmp/attestation.json",
                "--python",
                "/tmp/python",
                "--pythonpath",
                "/tmp/pythonpath",
                "--library-path",
                "/tmp/lib",
                "--implementation-commit",
                "a" * 40,
                "--build-receipt",
                "/tmp/receipt.json",
                "--m2-validation",
                "/tmp/validation.md",
                "--m2-adversarial-audit",
                "/tmp/audit.md",
                "--m2-validation-gate",
                "/tmp/validation-gate.json",
                "--m2-adversarial-audit-gate",
                "/tmp/audit-gate.json",
            ]
        )
        self.assertEqual(pathlib.Path("/tmp/validation.md"), args.c1_validation)
        self.assertEqual(
            pathlib.Path("/tmp/audit.md"), args.c1_adversarial_audit
        )
        self.assertEqual(
            pathlib.Path("/tmp/validation-gate.json"),
            args.m1_validation_gate,
        )
        self.assertEqual(
            pathlib.Path("/tmp/audit-gate.json"),
            args.m1_adversarial_audit_gate,
        )
        self.assertFalse(hasattr(args, "m2_validation_gate"))
        self.assertFalse(hasattr(args, "m2_adversarial_audit_gate"))
        self.assertEqual(
            controller.VALIDATION_GATE_SCHEMA, controller.m1.VALIDATION_GATE_SCHEMA
        )
        self.assertEqual(
            len(controller.base.SUPPORT_SOURCE_PATHS),
            len(set(controller.base.SUPPORT_SOURCE_PATHS.values())),
        )


if __name__ == "__main__":
    unittest.main()
