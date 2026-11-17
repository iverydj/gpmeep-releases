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


class TestV110AuNPMaterialOpenMPCanary(unittest.TestCase):
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
                f"isolated C4 contract failed ({completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    def test_wrapper_exposes_distinct_c4_contract(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_material_omp_canary as module
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
        self.assertEqual(
            "static-material-openmp-initialization", contract["candidate"]
        )
        self.assertTrue(
            contract["controller"].endswith("run_aunp_material_omp_canary.py")
        )
        self.assertTrue(
            contract["worker"].endswith("run_aunp_material_omp_sample.py")
        )
        self.assertIn("material-omp", contract["schema"])
        self.assertIn("material-omp", contract["attestation"])
        self.assertFalse(contract["legacy"])
        self.assertEqual("c4_validation", contract["validation_key"])
        self.assertEqual("c4_adversarial_audit", contract["audit_key"])
        self.assertEqual(
            {"base_controller", "base_worker"}, set(contract["support_sources"])
        )

    def test_environment_binds_four_cores_and_disables_c1_c3(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import run_aunp_material_omp_canary as module
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
        for environment in environments.values():
            self.assertEqual("4", environment["OMP_NUM_THREADS"])
            self.assertEqual("FALSE", environment["OMP_DYNAMIC"])
            self.assertEqual("close", environment["OMP_PROC_BIND"])
            self.assertEqual("cores", environment["OMP_PLACES"])
            self.assertEqual(
                "1", environment["MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION"]
            )
            self.assertEqual(
                "1", environment["MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN"]
            )
        self.assertEqual("1", environments["enabled"]["MEEP_C4_MATERIAL_OMP_ENABLE"])
        self.assertEqual(
            "1", environments["disabled"]["MEEP_C4_MATERIAL_OMP_DISABLE"]
        )

    def test_invalid_candidate_thread_count_fails_closed(self):
        output = self.run_isolated(
            """
            import json
            import pathlib
            import run_aunp_material_omp_canary as module
            module.base.sample.OMP_THREADS = 0
            try:
                module.base._environment(
                    module.base.task_plan()[0], pathlib.Path("/python"),
                    pathlib.Path("/pythonpath"), "/lib", ["gpu0", "gpu1"],
                    pathlib.Path("/home")
                )
            except module.base.CanaryError as error:
                print(json.dumps({"error": str(error)}, sort_keys=True))
            else:
                raise SystemExit("invalid thread count was accepted")
            """
        )
        self.assertIn("thread count is invalid", json.loads(output)["error"])

    def test_source_allowlist_is_exact(self):
        output = self.run_isolated(
            """
            import json
            import run_aunp_material_omp_canary as module
            allowed_file = "scripts/tests/test_v110_aunp_material_omp_canary.py"
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


if __name__ == "__main__":
    unittest.main()
