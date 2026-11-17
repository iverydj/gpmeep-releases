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


class TestV110AuNPM1Ablation(unittest.TestCase):
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
                f"isolated M1 contract failed ({completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    def test_plan_has_two_balanced_repeats_and_twenty_tasks(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_m1_ablation as module
            plan = module.task_plan()
            inventory = {}
            for repeat in range(module.base.REPEATS):
                for topology in ("gpu1", "gpu2"):
                    inventory[f"{repeat}-{topology}"] = sorted(
                        task["feature_mode"] for task in plan
                        if task["repeat"] == repeat
                        and task["topology"] == topology
                    )
            print(json.dumps({
                "count": len(plan),
                "inventory": inventory,
                "ids": [task["task_id"] for task in plan],
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual(20, result["count"])
        expected = sorted(("off", "c1", "c3", "c1+c3", "c4"))
        self.assertTrue(
            all(inventory == expected for inventory in result["inventory"].values())
        )
        self.assertEqual(20, len(set(result["ids"])))

    def test_environment_is_exact_and_c4_uses_four_bound_cores(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import tempfile
            import run_aunp_m1_ablation as module
            result = {}
            with tempfile.TemporaryDirectory() as directory:
                fontconfig = pathlib.Path(directory) / "qualification.conf"
                fontconfig.write_text("<fontconfig/>\\n", encoding="utf-8")
                attestation = {
                    module.QUALIFICATION_FONTCONFIG_KEY:
                        module.base.common.absolute_file_record(
                            fontconfig, "test Fontconfig"
                        )
                }
                for mode in module.MODES:
                    task = {
                        "feature_mode": mode, "ranks": 2,
                        "topology": "gpu2", "repeat": 0, "sequence": 0,
                        "task_id": "test",
                    }
                    result[mode] = module.base._environment(
                        task, pathlib.Path("/prefix/bin/python"),
                        pathlib.Path("/installed/site-packages"),
                        "/lib", ["gpu0", "gpu1"], pathlib.Path("/home"),
                        attestation,
                    )
                result["fontconfig"] = str(fontconfig)
            print(json.dumps(result, sort_keys=True))
            """
        )
        environments = json.loads(output)
        fontconfig = environments.pop("fontconfig")
        for mode, environment in environments.items():
            self.assertEqual("FALSE", environment["OMP_DYNAMIC"])
            self.assertEqual("close", environment["OMP_PROC_BIND"])
            self.assertEqual("cores", environment["OMP_PLACES"])
            self.assertEqual("4" if mode == "c4" else "1", environment["OMP_NUM_THREADS"])
            self.assertEqual("/prefix", environment["CONDA_PREFIX"])
            self.assertEqual("/home/.cache", environment["XDG_CACHE_HOME"])
            self.assertEqual("/home/.config", environment["XDG_CONFIG_HOME"])
            self.assertEqual("/home", environment["TMPDIR"])
            self.assertEqual("Agg", environment["MPLBACKEND"])
            self.assertEqual(
                "/home/.config/matplotlib", environment["MPLCONFIGDIR"]
            )
            self.assertEqual(
                fontconfig,
                environment["FONTCONFIG_FILE"],
            )
            self.assertEqual("1", environment["CUDA_CACHE_DISABLE"])
            self.assertEqual("1", environment["MEEP_GPU_STRICT"])
            self.assertEqual("1", environment["OPENBLAS_NUM_THREADS"])
            self.assertEqual("1", environment["MKL_NUM_THREADS"])
            self.assertEqual("1", environment["NUMEXPR_NUM_THREADS"])
            self.assertEqual("pinned", environment["MEEP_GPU_MPI_TRANSPORT"])
            self.assertEqual("waitsome", environment["MEEP_GPU_MPI_COMPLETION"])
        self.assertEqual(
            "1", environments["c1+c3"]["MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION"]
        )
        self.assertEqual(
            "1", environments["c1+c3"]["MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN"]
        )
        self.assertEqual("1", environments["c4"]["MEEP_C4_MATERIAL_OMP_ENABLE"])

    def test_mpi_mapping_allocates_each_rank_its_openmp_cores(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_m1_ablation as module
            result = {}
            for mode in module.MODES:
                result[mode] = module.base._mpi_map_by({
                    "feature_mode": mode,
                })
            print(json.dumps(result, sort_keys=True))
            """
        )
        mappings = json.loads(output)
        for mode in ("off", "c1", "c3", "c1+c3"):
            self.assertEqual("slot:PE=1", mappings[mode])
        self.assertEqual("slot:PE=4", mappings["c4"])

    def test_environment_rejects_missing_or_changed_fontconfig_attestation(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import tempfile
            import run_aunp_m1_ablation as module

            task = {
                "feature_mode": "off", "ranks": 1,
                "topology": "gpu1", "repeat": 0, "sequence": 0,
                "task_id": "test",
            }
            arguments = (
                task, pathlib.Path("/prefix/bin/python"),
                pathlib.Path("/installed/site-packages"), "/lib",
                ["gpu0", "gpu1"], pathlib.Path("/home"),
            )
            outcomes = {}
            try:
                module.base._environment(*arguments, {})
            except module.base.CanaryError as error:
                outcomes["missing"] = str(error)
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "qualification.conf"
                path.write_text("original\\n", encoding="utf-8")
                attestation = {
                    module.QUALIFICATION_FONTCONFIG_KEY:
                        module.base.common.absolute_file_record(
                            path, "test Fontconfig"
                        )
                }
                path.write_text("changed\\n", encoding="utf-8")
                try:
                    module.base._environment(*arguments, attestation)
                except module.base.common.WorkloadError as error:
                    outcomes["changed"] = str(error)
            print(json.dumps(outcomes, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertIn("absolute file record differs", result["missing"])
        self.assertIn("SHA-256 mismatch", result["changed"])

    def test_receipt_bound_fontconfig_policy_is_exact_and_prefix_safe(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import tempfile
            import run_aunp_m1_ablation as module

            outcomes = {}
            with tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                module.base.REPO = root
                path = root / "build" / "qualification-fontconfig.conf"
                path.parent.mkdir()
                path.write_text(
                    '<fontconfig><cachedir prefix="xdg">'
                    'fontconfig</cachedir></fontconfig>\\n',
                    encoding="utf-8",
                )
                receipt = {
                    "configuration_files": {
                        "qualification_fontconfig":
                            module.base.common.file_record(path, root)
                    }
                }
                outcomes["safe"] = module._qualification_fontconfig_record(
                    receipt
                )
                path.write_text(
                    '<fontconfig><cachedir>/var/cache/fontconfig'
                    '</cachedir></fontconfig>\\n',
                    encoding="utf-8",
                )
                receipt["configuration_files"]["qualification_fontconfig"] = (
                    module.base.common.file_record(path, root)
                )
                try:
                    module._qualification_fontconfig_record(receipt)
                except module.base.CanaryError as error:
                    outcomes["unsafe"] = str(error)
            print(json.dumps(outcomes, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertTrue(pathlib.Path(result["safe"]["path"]).is_absolute())
        self.assertIn("cache policy is unsafe", result["unsafe"])

    def test_component_projection_isolates_c1_from_c3_only(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_m1_ablation_sample as module
            ordinary = {
                "cuda_polarization_calls": 2,
                "cuda_polarization_points": 4,
                **{name: 7 for name in module.base.FEATURE_COUNTERS},
            }
            records = [{
                "records": [{
                    "candidate_statistics_before": {
                        "sparse": {"token": 1}, "material": {"token": 2}
                    },
                    "candidate_statistics_after": {
                        "sparse": {"token": 3}, "material": {"token": 4}
                    },
                    "statistics_delta": {"polarizations": ordinary},
                }],
                "final_candidate_statistics": {
                    "sparse": {"token": 5}, "material": {"token": 6}
                },
            }]
            projected = module._isolate_c3_ordinary_counters(records, True)
            print(json.dumps({
                "projected": projected,
                "original": records,
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        projected = result["projected"][0]
        original = result["original"][0]
        group = projected["records"][0]["statistics_delta"]["polarizations"]
        self.assertEqual(2, group["cuda_polarization_calls"])
        self.assertTrue(
            all(
                group[name] == 0
                for name in (
                    "cuda_zero_state_elision_enabled_calls",
                    "cuda_zero_state_elision_enabled_points",
                    "cuda_zero_state_elision_inspectable_calls",
                    "cuda_zero_state_elision_inspectable_points",
                )
            )
        )
        self.assertEqual(7, original["records"][0]["statistics_delta"]["polarizations"][
            "cuda_zero_state_elision_enabled_calls"
        ])
        self.assertEqual({"token": 1}, projected["records"][0]["candidate_statistics_before"])

    def test_counter_validation_dispatches_to_saved_component_validators(self):
        output = self.run_isolated(
            """
            import json
            from unittest import mock
            import run_aunp_m1_ablation_sample as module

            results = {}
            for mode in module.MODES:
                with (
                    mock.patch.object(
                        module, "_base_validate_feature_counters",
                        return_value={"component": "c1"},
                    ) as c1,
                    mock.patch.object(
                        module.c3, "_validate_feature_counters",
                        return_value={"component": "c3"},
                    ) as c3,
                    mock.patch.object(
                        module.c4, "_validate_feature_counters",
                        return_value={"component": "c4"},
                    ) as c4,
                    mock.patch.object(
                        module, "_isolate_c3_ordinary_counters",
                        return_value=[],
                    ),
                    mock.patch.object(
                        module, "_component_records", return_value=[]
                    ),
                ):
                    value = module._validate_feature_counters([], mode)
                    results[mode] = {
                        "value": value,
                        "c1_mode": c1.call_args.args[1],
                        "c3_mode": c3.call_args.args[1],
                        "c4_mode": c4.call_args.args[1],
                    }
            print(json.dumps(results, sort_keys=True))
            """
        )
        results = json.loads(output)
        expected = {
            "off": ("disabled", "disabled", "disabled"),
            "c1": ("enabled", "disabled", "disabled"),
            "c3": ("disabled", "enabled", "disabled"),
            "c1+c3": ("enabled", "enabled", "disabled"),
            "c4": ("disabled", "disabled", "enabled"),
        }
        for mode, component_modes in expected.items():
            result = results[mode]
            self.assertEqual(
                ["c1", "c3", "c4"], sorted(result["value"])
            )
            self.assertEqual(
                component_modes,
                (result["c1_mode"], result["c3_mode"], result["c4_mode"]),
            )

    def test_synthetic_result_derivation_reports_absolute_times(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_m1_ablation as module
            mode_factor = {
                "off": (1.0, 1.0, 1.0),
                "c1": (1.0, 0.96, 0.98),
                "c3": (1.0, 0.94, 0.97),
                "c1+c3": (1.0, 0.90, 0.94),
                "c4": (0.50, 1.0, 0.75),
            }
            tasks = []
            for planned in module.task_plan():
                task = dict(planned)
                topology_factor = 1.0 if task["topology"] == "gpu1" else 0.6
                repeat_factor = 1.0 + 0.01 * task["repeat"]
                init, fdtd, e2e = mode_factor[task["feature_mode"]]
                ranks = task["ranks"]
                cold_start = (
                    (task["repeat"] == 0 and task["feature_mode"] == "off")
                    or (
                        task["repeat"] == 1
                        and task["feature_mode"] != "off"
                    )
                )
                devices = {
                    f"GPU-{index}": {
                        "peak_attributed_memory_mib": 100.0,
                        "start_temperature_c": 30.0 if cold_start else 50.0,
                        "maximum_temperature_c": 60.0,
                        "throttle_clean": True,
                    }
                    for index in range(ranks)
                }
                task.update({
                    "initialization_wall_seconds": 10 * init * repeat_factor,
                    "fdtd_wall_seconds": 20 * fdtd * topology_factor * repeat_factor,
                    "end_to_end_seconds": 40 * e2e * topology_factor * repeat_factor,
                    "peak_host_rss_bytes_by_rank": [1000] * ranks,
                    "peak_host_rss_bytes_total": 1000 * ranks,
                    "observable_signature": [{"same": True}],
                    "runtime_artifacts": {"libmeep": {"sha256": "a" * 64}},
                    "telemetry": {"devices": devices},
                })
                tasks.append(task)
            print(json.dumps(module.derive_results(tasks), sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual("M1_CANDIDATES_ACCEPTED", result["promotion_outcome"])
        c4 = result["comparisons_to_off"]["gpu1"]["c4"]
        self.assertAlmostEqual(0.5, c4["initialization_wall_seconds"]["candidate_to_off_ratio"])
        self.assertAlmostEqual(0.75, c4["end_to_end_seconds"]["candidate_to_off_ratio"])
        self.assertIn("candidate_mean_seconds", c4["end_to_end_seconds"])
        self.assertEqual(
            20.0,
            max(
                device["start_temperature_delta_c"]
                for check in result["resource_checks"]
                for device in check["devices"].values()
            ),
        )

    def test_compute_throttle_excludes_memory_resident_idle_samples(self):
        output = self.run_isolated(
            """
            import copy
            import json
            import run_aunp_m1_ablation as module

            uuid = "GPU-00000000-0000-0000-0000-000000000000"
            def record(
                *, utilization, memory, clock, pstate, idle,
                temperature=50.0
            ):
                return {
                    "index": 0,
                    "uuid": uuid,
                    "temperature_c": temperature,
                    "memory_used_mib": memory,
                    "utilization_percent": utilization,
                    "pstate": pstate,
                    "sm_clock_mhz": clock,
                    "power_w": 300.0 if utilization else 25.0,
                    "enforced_power_limit_w": 450.0,
                    "active_clock_event_mask": "0x0000000000000001" if idle else "0x0000000000000000",
                    "active_clock_event_mask_value": 1 if idle else 0,
                    "gpu_idle_active": idle,
                    "software_power_cap_active": False,
                    "software_thermal_slowdown_active": False,
                    "hardware_thermal_slowdown_active": False,
                    "applications_clocks_setting_active": False,
                    "hardware_slowdown_active": False,
                    "hardware_power_brake_slowdown_active": False,
                    "sync_boost_active": False,
                    "temperature_limit_c": None,
                }

            samples = [
                {"elapsed_seconds": 0.0, "devices": [record(
                    utilization=0.0, memory=1.0, clock=210.0,
                    pstate="P8", idle=True,
                )]},
                {"elapsed_seconds": 1.0, "devices": [record(
                    utilization=0.0, memory=2800.0, clock=210.0,
                    pstate="P8", idle=True,
                )]},
                {"elapsed_seconds": 2.0, "devices": [record(
                    utilization=80.0, memory=2800.0, clock=1920.0,
                    pstate="P2", idle=False,
                )]},
                {"elapsed_seconds": 3.0, "devices": [record(
                    utilization=0.0, memory=2800.0, clock=210.0,
                    pstate="P8", idle=True,
                )]},
            ]
            clean = module.base.summarize_telemetry(samples, 1)
            throttled = copy.deepcopy(samples)
            throttled[2]["devices"][0][
                "hardware_thermal_slowdown_active"
            ] = True
            dirty = module.base.summarize_telemetry(throttled, 1)
            at_temperature_limit = copy.deepcopy(samples)
            at_temperature_limit[2]["devices"][0]["temperature_c"] = 80.0
            at_limit = module.base.summarize_telemetry(
                at_temperature_limit, 1
            )
            over_temperature_limit = copy.deepcopy(samples)
            over_temperature_limit[2]["devices"][0][
                "temperature_c"
            ] = 80.0001
            over_limit = module.base.summarize_telemetry(
                over_temperature_limit, 1
            )
            utilization_errors = {}
            for name, value in (
                ("missing", None),
                ("bool", True),
                ("negative", -1.0),
                ("over_100", 100.0001),
                ("nan", float("nan")),
                ("infinite", float("inf")),
            ):
                malformed = copy.deepcopy(samples)
                malformed[1]["devices"][0]["utilization_percent"] = value
                try:
                    module.base.summarize_telemetry(malformed, 1)
                except module.base.CanaryError as error:
                    utilization_errors[name] = str(error)
            idle_only_error = None
            try:
                module.base.summarize_telemetry(samples[:2], 1)
            except module.base.CanaryError as error:
                idle_only_error = str(error)
            print(json.dumps({
                "clean": clean,
                "dirty": dirty,
                "at_limit": at_limit,
                "over_limit": over_limit,
                "utilization_errors": utilization_errors,
                "idle_only_error": idle_only_error,
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        uuid = "GPU-00000000-0000-0000-0000-000000000000"
        clean = result["clean"]["devices"][uuid]
        dirty = result["dirty"]["devices"][uuid]
        self.assertTrue(clean["throttle_clean"])
        self.assertEqual(1, clean["active_sample_count"])
        self.assertEqual([0], clean["active_clock_event_mask_values"])
        self.assertEqual(["P2"], clean["active_pstates"])
        self.assertEqual(2799.0, clean["peak_attributed_memory_mib"])
        self.assertFalse(clean["gpu_idle_observed"])
        self.assertFalse(dirty["throttle_clean"])
        self.assertTrue(dirty["hardware_thermal_slowdown_observed"])
        self.assertTrue(
            result["at_limit"]["devices"][uuid]["throttle_clean"]
        )
        self.assertFalse(
            result["over_limit"]["devices"][uuid]["throttle_clean"]
        )
        self.assertEqual(
            {"bool", "infinite", "missing", "nan", "negative", "over_100"},
            set(result["utilization_errors"]),
        )
        self.assertTrue(all(
            "utilization telemetry is invalid" in error
            for error in result["utilization_errors"].values()
        ))
        self.assertIn("no active workload sample", result["idle_only_error"])

    def test_source_allowlist_is_exact(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_m1_ablation as module
            allowed = "scripts/tests/test_v110_aunp_m1_ablation.py"
            print(json.dumps({
                "exact": module.base._implementation_descendant_allowed(allowed),
                "prefix": module.base._implementation_descendant_allowed(
                    allowed + ".evil"
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

    def test_machine_audit_gate_rejects_all_blocking_and_tampered_inputs(self):
        output = self.run_isolated(
            """
            import copy
            import json
            import pathlib
            import tempfile
            import run_aunp_m1_ablation as module

            source = "a" * 40
            implementation = "b" * 40
            validation_report = {
                "path": "/validation.md", "size_bytes": 10,
                "sha256": "1" * 64,
            }
            audit_report = {
                "path": "/audit.md", "size_bytes": 11,
                "sha256": "2" * 64,
            }
            validation_gate_record = {
                "path": "/validation-gate.json", "size_bytes": 12,
                "sha256": "3" * 64,
            }
            build_receipt = {
                "path": "/receipt.json", "size_bytes": 13,
                "sha256": "4" * 64,
            }
            build_identity = {
                "receipt_id": "5" * 64,
                "artifact_set_id": "6" * 64,
                "build_input_id": "7" * 64,
                "build_kind": "cuda-mpi-python-fp32",
                "git_head": implementation,
                "source_sha256": "8" * 64,
            }
            validation_gate = {
                "schema": module.VALIDATION_GATE_SCHEMA,
                "status": "PASS",
                "source_commit": source,
                "implementation_commit": implementation,
                "validation_report": validation_report,
                "build_receipt": build_receipt,
                "build_identity": build_identity,
            }
            audit_gate = {
                "schema": module.AUDIT_GATE_SCHEMA,
                "decision": "ACCEPT",
                "severity": {
                    "critical": 0, "high": 0, "medium": 0, "low": 1,
                },
                "source_commit": source,
                "implementation_commit": implementation,
                "validation_report": validation_report,
                "validation_gate": validation_gate_record,
                "audit_report": audit_report,
                "build_receipt": build_receipt,
                "build_identity": build_identity,
            }

            def accepts(vgate, agate, **overrides):
                arguments = {
                    "source_commit": source,
                    "implementation_commit": implementation,
                    "validation_report": validation_report,
                    "audit_report": audit_report,
                    "validation_gate_record": validation_gate_record,
                    "build_receipt": build_receipt,
                    "build_identity": build_identity,
                }
                arguments.update(overrides)
                try:
                    module._validate_gate_payloads(vgate, agate, **arguments)
                except module.base.CanaryError:
                    return False
                return True

            cases = {"valid": accepts(validation_gate, audit_gate)}
            rejected = copy.deepcopy(audit_gate)
            rejected["decision"] = "REJECT"
            cases["reject"] = accepts(validation_gate, rejected)
            medium = copy.deepcopy(audit_gate)
            medium["severity"]["medium"] = 1
            cases["medium"] = accepts(validation_gate, medium)
            wrong_head = copy.deepcopy(audit_gate)
            wrong_head["source_commit"] = "c" * 40
            cases["wrong_head"] = accepts(validation_gate, wrong_head)
            wrong_validation = copy.deepcopy(audit_gate)
            wrong_validation["validation_report"]["sha256"] = "9" * 64
            cases["wrong_validation"] = accepts(validation_gate, wrong_validation)
            cases["malformed"] = accepts({}, audit_gate)

            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "gate.json"
                empty = pathlib.Path(directory) / "empty.json"
                empty.write_text("", encoding="utf-8")
                empty_record = module.base.common.absolute_file_record(
                    empty, "empty test gate"
                )
                try:
                    module._load_bound_json(empty_record, "empty test gate")
                except Exception:
                    cases["empty"] = False
                else:
                    cases["empty"] = True
                path.write_text("{}", encoding="utf-8")
                record = module.base.common.absolute_file_record(path, "test gate")
                path.write_text('{"changed":true}', encoding="utf-8")
                try:
                    module._load_bound_json(record, "test gate")
                except Exception:
                    cases["post_attestation_tamper"] = False
                else:
                    cases["post_attestation_tamper"] = True
            print(json.dumps(cases, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertTrue(result.pop("valid"))
        self.assertTrue(result)
        self.assertTrue(all(value is False for value in result.values()))

    def test_empty_reference_manifest_is_valid_but_structure_is_not(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_material_omp_sample as module

            null_array = {
                "component": 0,
                "direction": 0,
                "trivial": True,
                "null": True,
                "elements": 0,
                "bytes": 0,
                "sha256": None,
            }
            chi1inv = []
            for component in range(20):
                for direction in range(5):
                    record = dict(null_array)
                    record["component"] = component
                    record["direction"] = direction
                    chi1inv.append(record)
            manifest = {
                "schema": "gpmeep-material-arrays-v1",
                "realnum_bytes": 4,
                "num_chunks": 1,
                "chunks": [{
                    "index": 0,
                    "owned": True,
                    "ntot": 0,
                    "chi1inv": chi1inv,
                    "susceptibilities": [],
                }],
            }
            result = {
                "reference": module._validate_material_manifest(
                    manifest, require_arrays=False
                )
            }
            try:
                module._validate_material_manifest(
                    manifest, require_arrays=True
                )
            except module.base.common.WorkloadError as error:
                result["structure_error"] = str(error)
            print(json.dumps(result, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual([0, 0, 0], result["reference"])
        self.assertIn("structure material manifest has no arrays", result[
            "structure_error"
        ])

    def test_material_ownership_may_be_sparse_across_mpi_ranks(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_material_omp_sample as module

            def array(component, direction, *, present):
                return {
                    "component": component,
                    "direction": direction,
                    "trivial": not present,
                    "null": not present,
                    "elements": 1 if present else 0,
                    "bytes": 4 if present else 0,
                    "sha256": "1" * 64 if present else None,
                }

            def manifest(*, material):
                chi1inv = [
                    array(component, direction, present=(
                        material and component == 0 and direction == 0
                    ))
                    for component in range(20)
                    for direction in range(5)
                ]
                susceptibilities = []
                if material:
                    susceptibilities.append({
                        "field_type": 0,
                        "order": 0,
                        "type": "lorentzian",
                        "ntot": 1,
                        "sigma": [
                            array(component, direction, present=(
                                component == 0 and direction == 0
                            ))
                            for component in range(20)
                            for direction in range(5)
                        ],
                    })
                return {
                    "schema": "gpmeep-material-arrays-v1",
                    "realnum_bytes": 4,
                    "num_chunks": 1,
                    "chunks": [{
                        "index": 0,
                        "owned": True,
                        "ntot": 1 if material else 0,
                        "chi1inv": chi1inv,
                        "susceptibilities": susceptibilities,
                    }],
                }

            empty = manifest(material=False)
            nonempty = manifest(material=True)
            ranks = [
                {"records": [
                    {"candidate_material_manifest": empty},
                    {"candidate_material_manifest": empty},
                ]},
                {"records": [
                    {"candidate_material_manifest": empty},
                    {"candidate_material_manifest": nonempty},
                ]},
            ]
            result = {
                "sparse": module._validate_material_rank_distribution(ranks)
            }
            ranks[1]["records"][1]["candidate_material_manifest"] = empty
            try:
                module._validate_material_rank_distribution(ranks)
            except module.base.common.WorkloadError as error:
                result["all_empty_error"] = str(error)
            no_susceptibility = manifest(material=True)
            no_susceptibility["chunks"][0]["susceptibilities"] = []
            ranks[1]["records"][1][
                "candidate_material_manifest"
            ] = no_susceptibility
            try:
                module._validate_material_rank_distribution(ranks)
            except module.base.common.WorkloadError as error:
                result["no_susceptibility_error"] = str(error)
            try:
                module._validate_material_rank_distribution([{
                    "records": [None, {
                        "candidate_material_manifest": nonempty,
                    }],
                }])
            except module.base.common.WorkloadError as error:
                result["phase_type_error"] = str(error)
            print(json.dumps(result, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual({
            "material_array_bytes": 8,
            "material_array_count": 2,
            "susceptibility_array_count": 1,
        }, result["sparse"])
        self.assertIn("no arrays across ranks", result["all_empty_error"])
        self.assertIn(
            "no susceptibility arrays across ranks",
            result["no_susceptibility_error"],
        )
        self.assertIn("phase record differs", result["phase_type_error"])

    def test_material_manifest_schema_is_exactly_typed_and_ordered(self):
        output = self.run_isolated(
            """
            import copy
            import json
            import run_aunp_material_omp_sample as module

            def records():
                return [{
                    "component": component,
                    "direction": direction,
                    "trivial": True,
                    "null": True,
                    "elements": 0,
                    "bytes": 0,
                    "sha256": None,
                } for component in range(20) for direction in range(5)]

            chunks = []
            for chunk_index in range(2):
                susceptibilities = []
                for order in range(2):
                    susceptibilities.append({
                        "field_type": 0,
                        "order": order,
                        "type": "lorentzian",
                        "ntot": 0,
                        "sigma": records(),
                    })
                chunks.append({
                    "index": chunk_index,
                    "owned": True,
                    "ntot": 0,
                    "chi1inv": records(),
                    "susceptibilities": susceptibilities,
                })
            valid = {
                "schema": "gpmeep-material-arrays-v1",
                "realnum_bytes": 4,
                "num_chunks": 2,
                "chunks": chunks,
            }

            def rejected(candidate):
                try:
                    module._validate_material_manifest(
                        candidate, require_arrays=False
                    )
                except module.base.common.WorkloadError:
                    return True
                return False

            cases = {}
            candidate = copy.deepcopy(valid)
            candidate["realnum_bytes"] = 4.0
            cases["realnum_float"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["realnum_bytes"] = True
            cases["realnum_bool"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][0]["index"] = False
            cases["first_chunk_bool"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][1]["index"] = 1.0
            cases["second_chunk_float"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][0]["susceptibilities"][0]["order"] = False
            cases["first_order_bool"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][0]["susceptibilities"][1]["order"] = 1.0
            cases["second_order_float"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][0]["chi1inv"][1] = copy.deepcopy(
                candidate["chunks"][0]["chi1inv"][0]
            )
            cases["chi_duplicate"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            candidate["chunks"][0]["chi1inv"][0:2] = reversed(
                candidate["chunks"][0]["chi1inv"][0:2]
            )
            cases["chi_reordered"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            sigma = candidate["chunks"][0]["susceptibilities"][0]["sigma"]
            sigma[1] = copy.deepcopy(sigma[0])
            cases["sigma_duplicate"] = rejected(candidate)
            candidate = copy.deepcopy(valid)
            sigma = candidate["chunks"][0]["susceptibilities"][0]["sigma"]
            sigma[0:2] = reversed(sigma[0:2])
            cases["sigma_reordered"] = rejected(candidate)
            print(json.dumps({
                "valid": module._validate_material_manifest(
                    valid, require_arrays=False
                ),
                "cases": cases,
            }, sort_keys=True))
            """
        )
        result = json.loads(output)
        self.assertEqual([0, 0, 0], result["valid"])
        self.assertEqual(10, len(result["cases"]))
        self.assertTrue(all(result["cases"].values()))


if __name__ == "__main__":
    unittest.main()
