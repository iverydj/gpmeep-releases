import json
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
LAB = ROOT / "scripts" / "v110-lab"
USER_WORKLOADS = ROOT / "scripts" / "user-workloads"


class TestV110AuNPSparseLorentzianCanary(unittest.TestCase):
    def run_isolated(self, source):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(LAB), str(USER_WORKLOADS))
        )
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            self.fail(
                f"isolated C3 contract failed ({completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    def test_wrappers_expose_distinct_c3_contract(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_sparse_lorentzian_canary as module
            print(json.dumps({
                "candidate": module.base.CANDIDATE_FEATURE,
                "controller": str(module.base.CONTROLLER_PATH),
                "worker": str(module.base.WORKER_PATH),
                "schema": module.base.SCHEMA,
                "attestation": module.base.RUNTIME_ATTESTATION_SCHEMA,
                "legacy": module.base.LEGACY_CANDIDATE_FIELD_ALLOWED,
                "validation_key": module.base.ATTESTATION_VALIDATION_KEY,
                "audit_key": module.base.ATTESTATION_AUDIT_KEY,
                "support_sources": {
                    name: str(path) for name, path in
                    module.base.SUPPORT_SOURCE_PATHS.items()
                },
            }, sort_keys=True))
            """
        )
        contract = json.loads(output)
        self.assertEqual("resident-sparse-lorentzian-plan", contract["candidate"])
        self.assertTrue(contract["controller"].endswith(
            "run_aunp_sparse_lorentzian_canary.py"
        ))
        self.assertTrue(contract["worker"].endswith(
            "run_aunp_sparse_lorentzian_sample.py"
        ))
        self.assertIn("sparse-lorentzian", contract["schema"])
        self.assertIn("sparse-lorentzian", contract["attestation"])
        self.assertFalse(contract["legacy"])
        self.assertEqual("c3_validation", contract["validation_key"])
        self.assertEqual("c3_adversarial_audit", contract["audit_key"])
        self.assertEqual(
            {"base_controller", "base_worker"},
            set(contract["support_sources"]),
        )
        self.assertTrue(contract["support_sources"]["base_controller"].endswith(
            "run_aunp_zero_state_canary.py"
        ))
        self.assertTrue(contract["support_sources"]["base_worker"].endswith(
            "run_aunp_zero_state_sample.py"
        ))

    def test_c3_import_does_not_mutate_canonical_c2_modules(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_zero_state_canary as c2_controller
            import run_aunp_zero_state_sample as c2_sample
            before = {
                "controller_schema": c2_controller.SCHEMA,
                "controller_worker": str(c2_controller.WORKER_PATH),
                "sample_schema": c2_sample.SCHEMA,
                "sample_feature": c2_sample.FEATURE_ENABLE,
                "sample_file": c2_sample.__file__,
            }
            import run_aunp_sparse_lorentzian_canary
            after = {
                "controller_schema": c2_controller.SCHEMA,
                "controller_worker": str(c2_controller.WORKER_PATH),
                "sample_schema": c2_sample.SCHEMA,
                "sample_feature": c2_sample.FEATURE_ENABLE,
                "sample_file": c2_sample.__file__,
            }
            print(json.dumps({"unchanged": before == after}, sort_keys=True))
            """
        )
        self.assertTrue(json.loads(output)["unchanged"])

    def test_source_allowlist_distinguishes_files_from_directories(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_sparse_lorentzian_canary as module
            allowed_file = (
                "scripts/tests/test_v110_aunp_sparse_lorentzian_canary.py"
            )
            print(json.dumps({
                "exact": module.base._implementation_descendant_allowed(
                    allowed_file
                ),
                "prefix": module.base._implementation_descendant_allowed(
                    allowed_file + ".evil"
                ),
                "directory": module.base._implementation_descendant_allowed(
                    "scripts/v110-lab/new_harness.py"
                ),
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertTrue(result["exact"])
        self.assertFalse(result["prefix"])
        self.assertTrue(result["directory"])

    def test_environment_isolates_c3_from_c1(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import run_aunp_sparse_lorentzian_canary as module
            result = {}
            for mode in ("enabled", "disabled"):
                task = dict(module.base.task_plan()[0])
                task["feature_mode"] = mode
                result[mode] = module.base._environment(
                    task, pathlib.Path("/python"), pathlib.Path("/pythonpath"),
                    "/lib", ["gpu0", "gpu1"], pathlib.Path("/home")
                )
            print(json.dumps(result, sort_keys=True))
            """
        )
        environments = json.loads(output)
        enabled = environments["enabled"]
        disabled = environments["disabled"]
        self.assertEqual(
            "1", enabled["MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN"]
        )
        self.assertNotIn(
            "MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN", enabled
        )
        self.assertEqual(
            "1", disabled["MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN"]
        )
        self.assertNotIn(
            "MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN", disabled
        )
        for environment in environments.values():
            self.assertEqual(
                "1",
                environment[
                    "MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION"
                ],
            )
            self.assertNotIn(
                "MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION",
                environment,
            )

    def test_retained_environment_rejects_c1_tampering(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_sparse_lorentzian_sample as module
            expected = module._expected_feature_environment("enabled")
            tampered = dict(expected)
            tampered[module.C1_ENABLE] = "1"
            missing_disable = dict(expected)
            del missing_disable[module.C1_DISABLE]
            print(json.dumps({
                "expected": module.base._summary_feature_environment_matches({
                    "feature_mode": "enabled",
                    "feature_environment": expected,
                }),
                "c1_enable": module.base._summary_feature_environment_matches({
                    "feature_mode": "enabled",
                    "feature_environment": tampered,
                }),
                "missing_disable":
                    module.base._summary_feature_environment_matches({
                        "feature_mode": "enabled",
                        "feature_environment": missing_disable,
                    }),
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertTrue(result["expected"])
        self.assertFalse(result["c1_enable"])
        self.assertFalse(result["missing_disable"])

    def test_sparse_causal_counter_contract_fails_closed(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_sparse_lorentzian_sample as module

            names = set(module.COUNTERS + module.GAUGES + module.FLAGS)
            def snapshot(**changes):
                value = {name: 0 for name in names}
                value["reset_generation"] = 1
                value.update(changes)
                return value

            zero = snapshot()
            enabled = snapshot(
                opt_in_calls=100, opt_in_points=10000000,
                eligible_calls=100, eligible_points=10000000,
                first_observation_dense_calls=10,
                first_observation_dense_points=700000,
                plan_build_attempts=10, plan_build_successes=10,
                plan_hits=80, accepted_total_points=700000,
                accepted_active_points=70000,
                sparse_submission_calls=90,
                sparse_submission_points=630000,
                guard_scan_submissions=90,
                guard_scan_submission_points=6300000,
                device_guard_scan_calls=90,
                device_guard_scan_points=6300000,
                device_sparse_update_calls=90,
                device_sparse_update_points=630000,
                live_plans=10, live_guards=5,
                retained_index_bytes=560000,
            )
            final_enabled = dict(enabled)
            for name in module.GAUGES:
                final_enabled[name] = 0
            ordinary_zero = {"cuda_polarization_calls": 0,
                             "cuda_polarization_points": 0,
                             **{name: 0 for name in module.base.FEATURE_COUNTERS}}
            ordinary_work = {"cuda_polarization_calls": 100,
                             "cuda_polarization_points": 10000000,
                             **{name: 0 for name in module.base.FEATURE_COUNTERS}}
            rank = {
                "records": [
                    {"statistics_delta": {"polarizations": ordinary_zero},
                     "candidate_statistics_before": zero,
                     "candidate_statistics_after": zero},
                    {"statistics_delta": {"polarizations": ordinary_work},
                     "candidate_statistics_before": zero,
                     "candidate_statistics_after": enabled},
                ],
                "final_candidate_statistics": final_enabled,
            }
            accepted = module._validate_feature_counters([rank], "enabled")
            bad = json.loads(json.dumps(rank))
            bad["records"][1]["candidate_statistics_after"][
                "device_sparse_update_calls"
            ] = 89
            rejected = False
            try:
                module._validate_feature_counters([bad], "enabled")
            except Exception:
                rejected = True

            missing_guard = json.loads(json.dumps(rank))
            for name in (
                "guard_scan_submissions", "guard_scan_submission_points",
                "device_guard_scan_calls", "device_guard_scan_points",
            ):
                missing_guard["records"][1][
                    "candidate_statistics_after"
                ][name] = 0
            missing_guard_rejected = False
            try:
                module._validate_feature_counters(
                    [missing_guard], "enabled"
                )
            except Exception:
                missing_guard_rejected = True

            broken_transition = json.loads(json.dumps(rank))
            broken_transition["records"][1][
                "candidate_statistics_after"
            ]["plan_hits"] = 81
            broken_transition_rejected = False
            try:
                module._validate_feature_counters(
                    [broken_transition], "enabled"
                )
            except Exception:
                broken_transition_rejected = True

            broken_fold = json.loads(json.dumps(rank))
            broken_fold["final_candidate_statistics"][
                "device_sparse_update_calls"
            ] += 1
            broken_fold_rejected = False
            try:
                module._validate_feature_counters(
                    [broken_fold], "enabled"
                )
            except Exception:
                broken_fold_rejected = True

            unexpected_reset = json.loads(json.dumps(rank))
            unexpected_reset["records"][1][
                "candidate_statistics_before"
            ]["reset_generation"] = 2
            unexpected_reset["records"][1][
                "candidate_statistics_after"
            ]["reset_generation"] = 2
            unexpected_reset["final_candidate_statistics"][
                "reset_generation"
            ] = 2
            unexpected_reset_rejected = False
            try:
                module._validate_feature_counters(
                    [unexpected_reset], "enabled"
                )
            except Exception:
                unexpected_reset_rejected = True

            stale = snapshot(opt_in_calls=1)
            stale_disabled = {
                "records": [
                    {"statistics_delta": {"polarizations": ordinary_zero},
                     "candidate_statistics_before": stale,
                     "candidate_statistics_after": stale},
                    {"statistics_delta": {"polarizations": ordinary_work},
                     "candidate_statistics_before": stale,
                     "candidate_statistics_after": stale},
                ],
                "final_candidate_statistics": stale,
            }
            stale_baseline_rejected = False
            try:
                module._validate_feature_counters(
                    [stale_disabled], "disabled"
                )
            except Exception:
                stale_baseline_rejected = True

            c1_contaminated = json.loads(json.dumps(rank))
            c1_contaminated["records"][1]["statistics_delta"][
                "polarizations"
            ][module.base.FEATURE_COUNTERS[0]] = 1
            c1_contamination_rejected = False
            try:
                module._validate_feature_counters(
                    [c1_contaminated], "enabled"
                )
            except Exception:
                c1_contamination_rejected = True

            broken_gauge = json.loads(json.dumps(rank))
            broken_gauge["records"][1][
                "candidate_statistics_after"
            ]["live_plans"] = 11
            broken_gauge_rejected = False
            try:
                module._validate_feature_counters(
                    [broken_gauge], "enabled"
                )
            except Exception:
                broken_gauge_rejected = True

            # Production creates mirrors lazily during the first timestep.
            # A later allocation generation can therefore retire a mixture of
            # observed, dense-locked, and successful plans before the stable
            # live set is reached.  Historical build counters must not be
            # equated with the final live gauges.
            churned = snapshot(
                opt_in_calls=48000,
                opt_in_points=101577600000,
                eligible_calls=48000,
                eligible_points=101577600000,
                first_observation_dense_calls=60,
                first_observation_dense_points=137952000,
                plan_build_attempts=36,
                plan_build_successes=27,
                plan_hits=35088,
                plan_invalidations=36,
                dense_locked_decisions=9,
                accepted_total_points=59882400,
                accepted_active_points=14753828,
                sparse_submission_calls=35115,
                sparse_submission_points=10982864388,
                guard_scan_submissions=7990,
                guard_scan_submission_points=16906608000,
                device_guard_scan_calls=7990,
                device_guard_scan_points=16906608000,
                device_sparse_update_calls=35115,
                device_sparse_update_points=10982864388,
                live_plans=24,
                live_guards=4,
                retained_index_bytes=94463232,
            )
            churned_final = dict(churned)
            for name in module.GAUGES:
                churned_final[name] = 0
            churned_rank = {
                "records": [
                    {"statistics_delta": {"polarizations": ordinary_zero},
                     "candidate_statistics_before": zero,
                     "candidate_statistics_after": zero},
                    {"statistics_delta": {"polarizations": {
                         "cuda_polarization_calls": 48000,
                         "cuda_polarization_points": 101577600000,
                         **{name: 0 for name in
                            module.base.FEATURE_COUNTERS}}},
                     "candidate_statistics_before": zero,
                     "candidate_statistics_after": churned},
                ],
                "final_candidate_statistics": churned_final,
            }
            churned_accepted = module._validate_feature_counters(
                [churned_rank], "enabled"
            )

            impossible_plan_accounting = json.loads(
                json.dumps(churned_rank)
            )
            impossible_plan_accounting["records"][1][
                "candidate_statistics_after"
            ]["plan_invalidations"] = 37
            impossible_plan_accounting["final_candidate_statistics"][
                "plan_invalidations"
            ] = 37
            impossible_plan_accounting_rejected = False
            try:
                module._validate_feature_counters(
                    [impossible_plan_accounting], "enabled"
                )
            except Exception:
                impossible_plan_accounting_rejected = True

            unaligned_retained_bytes = json.loads(json.dumps(churned_rank))
            unaligned_retained_bytes["records"][1][
                "candidate_statistics_after"
            ]["retained_index_bytes"] = 94463233
            unaligned_retained_bytes_rejected = False
            try:
                module._validate_feature_counters(
                    [unaligned_retained_bytes], "enabled"
                )
            except Exception:
                unaligned_retained_bytes_rejected = True

            excess_retained_bytes = json.loads(json.dumps(churned_rank))
            excess_retained_bytes["records"][1][
                "candidate_statistics_after"
            ]["retained_index_bytes"] = 118030632
            excess_retained_bytes_rejected = False
            try:
                module._validate_feature_counters(
                    [excess_retained_bytes], "enabled"
                )
            except Exception:
                excess_retained_bytes_rejected = True

            eligible_floor = json.loads(json.dumps(rank))
            for stats in (
                eligible_floor["records"][1][
                    "candidate_statistics_after"
                ],
                eligible_floor["final_candidate_statistics"],
            ):
                stats["opt_in_points"] = 6553599
                stats["eligible_points"] = 6553599
            eligible_floor["records"][1]["statistics_delta"][
                "polarizations"
            ]["cuda_polarization_points"] = 6553599
            eligible_floor_rejected = False
            try:
                module._validate_feature_counters(
                    [eligible_floor], "enabled"
                )
            except Exception:
                eligible_floor_rejected = True

            first_observation_floor = json.loads(json.dumps(rank))
            for stats in (
                first_observation_floor["records"][1][
                    "candidate_statistics_after"
                ],
                first_observation_floor["final_candidate_statistics"],
            ):
                stats["opt_in_calls"] = 101
                stats["eligible_calls"] = 101
                stats["first_observation_dense_calls"] = 11
            first_observation_floor["records"][1]["statistics_delta"][
                "polarizations"
            ]["cuda_polarization_calls"] = 101
            first_observation_floor_rejected = False
            try:
                module._validate_feature_counters(
                    [first_observation_floor], "enabled"
                )
            except Exception:
                first_observation_floor_rejected = True

            first_observation_upper = json.loads(json.dumps(churned_rank))
            for stats in (
                first_observation_upper["records"][1][
                    "candidate_statistics_after"
                ],
                first_observation_upper["final_candidate_statistics"],
            ):
                stats["first_observation_dense_points"] = 101577600001
            first_observation_upper_rejected = False
            try:
                module._validate_feature_counters(
                    [first_observation_upper], "enabled"
                )
            except Exception:
                first_observation_upper_rejected = True

            guard_floor = json.loads(json.dumps(rank))
            for stats in (
                guard_floor["records"][1]["candidate_statistics_after"],
                guard_floor["final_candidate_statistics"],
            ):
                stats["guard_scan_submissions"] = 1
                stats["guard_scan_submission_points"] = 65535
                stats["device_guard_scan_calls"] = 1
                stats["device_guard_scan_points"] = 65535
            guard_floor_rejected = False
            try:
                module._validate_feature_counters([guard_floor], "enabled")
            except Exception:
                guard_floor_rejected = True

            guard_upper = json.loads(json.dumps(churned_rank))
            for stats in (
                guard_upper["records"][1]["candidate_statistics_after"],
                guard_upper["final_candidate_statistics"],
            ):
                stats["guard_scan_submission_points"] = 101577600001
                stats["device_guard_scan_points"] = 101577600001
            guard_upper_rejected = False
            try:
                module._validate_feature_counters([guard_upper], "enabled")
            except Exception:
                guard_upper_rejected = True

            sparse_fraction = json.loads(json.dumps(churned_rank))
            for stats in (
                sparse_fraction["records"][1][
                    "candidate_statistics_after"
                ],
                sparse_fraction["final_candidate_statistics"],
            ):
                stats["sparse_submission_points"] = 35552160001
                stats["device_sparse_update_points"] = 35552160001
            sparse_fraction_rejected = False
            try:
                module._validate_feature_counters(
                    [sparse_fraction], "enabled"
                )
            except Exception:
                sparse_fraction_rejected = True
            print(json.dumps({
                "accepted_calls": accepted["device_sparse_update_calls"],
                "churned_accepted_calls":
                    churned_accepted["device_sparse_update_calls"],
                "rejected": rejected,
                "missing_guard_rejected": missing_guard_rejected,
                "broken_transition_rejected": broken_transition_rejected,
                "broken_fold_rejected": broken_fold_rejected,
                "unexpected_reset_rejected": unexpected_reset_rejected,
                "stale_baseline_rejected": stale_baseline_rejected,
                "c1_contamination_rejected": c1_contamination_rejected,
                "broken_gauge_rejected": broken_gauge_rejected,
                "impossible_plan_accounting_rejected":
                    impossible_plan_accounting_rejected,
                "unaligned_retained_bytes_rejected":
                    unaligned_retained_bytes_rejected,
                "excess_retained_bytes_rejected":
                    excess_retained_bytes_rejected,
                "eligible_floor_rejected": eligible_floor_rejected,
                "first_observation_floor_rejected":
                    first_observation_floor_rejected,
                "first_observation_upper_rejected":
                    first_observation_upper_rejected,
                "guard_floor_rejected": guard_floor_rejected,
                "guard_upper_rejected": guard_upper_rejected,
                "sparse_fraction_rejected": sparse_fraction_rejected,
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual(90, result["accepted_calls"])
        self.assertEqual(35115, result["churned_accepted_calls"])
        self.assertTrue(result["rejected"])
        self.assertTrue(result["missing_guard_rejected"])
        self.assertTrue(result["broken_transition_rejected"])
        self.assertTrue(result["broken_fold_rejected"])
        self.assertTrue(result["unexpected_reset_rejected"])
        self.assertTrue(result["stale_baseline_rejected"])
        self.assertTrue(result["c1_contamination_rejected"])
        self.assertTrue(result["broken_gauge_rejected"])
        self.assertTrue(result["impossible_plan_accounting_rejected"])
        self.assertTrue(result["unaligned_retained_bytes_rejected"])
        self.assertTrue(result["excess_retained_bytes_rejected"])
        self.assertTrue(result["eligible_floor_rejected"])
        self.assertTrue(result["first_observation_floor_rejected"])
        self.assertTrue(result["first_observation_upper_rejected"])
        self.assertTrue(result["guard_floor_rejected"])
        self.assertTrue(result["guard_upper_rejected"])
        self.assertTrue(result["sparse_fraction_rejected"])

    def test_c3_keeps_all_c2_advancement_gates(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_sparse_lorentzian_canary as module
            tasks = []
            uuid0 = "GPU-00000000-0000-0000-0000-000000000000"
            uuid1 = "GPU-00000000-0000-0000-0000-000000000001"
            for planned in module.base.task_plan():
                enabled = planned["feature_mode"] == "enabled"
                factor = (0.8 if enabled else 1.0) * (
                    1.0 + 0.01 * planned["repeat"]
                )
                gpu1 = planned["topology"] == "gpu1"
                uuids = [uuid0] if gpu1 else [uuid0, uuid1]
                rss = [1050 if enabled else 1000] * planned["ranks"]
                tasks.append({
                    **planned,
                    "fdtd_wall_seconds": (10.0 if gpu1 else 6.0) * factor,
                    "end_to_end_seconds": (20.0 if gpu1 else 12.0) * factor,
                    "peak_host_rss_bytes_by_rank": rss,
                    "peak_host_rss_bytes_total": sum(rss),
                    "peak_host_rss_bytes": max(rss),
                    "observable_signature": [{"bitwise": "same"}],
                    "runtime_artifacts": {"binary": "same"},
                    "telemetry": {
                        "devices": {
                            uuid: {
                                "peak_attributed_memory_mib":
                                    105.0 if enabled else 100.0,
                                "start_temperature_c":
                                    42.0 if enabled else 40.0,
                                "maximum_temperature_c": 65.0,
                                "throttle_clean": True,
                            } for uuid in uuids
                        }
                    },
                })
            result = module.base.derive_results(tasks)
            print(json.dumps({
                "outcome": result["promotion_outcome"],
                "gates": result["gates"],
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual("POSITIVE_CANARY", result["outcome"])
        self.assertTrue(all(result["gates"].values()))


if __name__ == "__main__":
    unittest.main()
