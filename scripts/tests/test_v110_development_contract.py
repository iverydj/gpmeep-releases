from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validate-v110-development-contract.py"
CONTRACT = ROOT / "packaging/v110-development-contract.json"


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_script("v110_contract", SCRIPT)
REGISTRY = load_script(
    "v110_registry", ROOT / "scripts/validate-v110-inversion-registry.py"
)
ATTESTER = load_script("v110_attester", ROOT / "scripts/attest-v110-m0.py")
RECEIPT = load_script(
    "v110_receipt", ROOT / "scripts/v110-release-gate-receipt.py"
)
TRANSITION = load_script(
    "v110_source_transition",
    ROOT / "scripts/verify-v110-release-source-transition.py",
)


def _write_evidence(
    root: pathlib.Path, role: str, payload: dict[str, object]
) -> dict[str, object]:
    path = root / f"{role}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    content_id = RECEIPT._canonical_sha256(payload)
    return {
        "role": role,
        "bundle_relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema": payload["schema"],
        "content_id": content_id,
    }


def _compact_numerical_result_fixture() -> dict[str, object]:
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


def _waveguide_diagnostic_difference_summary() -> dict[str, object]:
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
            "diagnostic_only_out_of_tolerance_count": 109 if index == 0 else 0,
        }
    return summary


def _waveguide_optimizer_self_oracle_fixture() -> dict[str, object]:
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


def _numerical_result_fixture(
    case_path: str, lane: str
) -> tuple[str, dict[str, object]]:
    method = REGISTRY._expected_m4_numerical_method(case_path, lane)
    if case_path == REGISTRY.ABSORBER_CASE:
        result = {
            "outcome": "PASS",
            "failure_count": 0,
            "pairwise": _compact_numerical_result_fixture(),
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
        pairwise = _compact_numerical_result_fixture()
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
                _waveguide_diagnostic_difference_summary()
            )
            pairwise["difference_summary"] = (
                _waveguide_diagnostic_difference_summary()
            )
        self_oracle = _waveguide_optimizer_self_oracle_fixture()
        result = {
            "outcome": "PASS",
            "failure_count": 0,
            "pairwise": pairwise,
            "reference_self_oracle": copy.deepcopy(self_oracle),
            "candidate_self_oracle": copy.deepcopy(self_oracle),
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
            "pairwise": _compact_numerical_result_fixture(),
            "reference_self_oracle": copy.deepcopy(self_oracle),
            "candidate_self_oracle": copy.deepcopy(self_oracle),
            "reference_fixed_work": copy.deepcopy(fixed_work),
            "candidate_fixed_work": copy.deepcopy(fixed_work),
        }
    else:
        result = _compact_numerical_result_fixture()
    return method, result


def candidate_manifest_fixture(root: pathlib.Path, commit: str) -> pathlib.Path:
    evidence: dict[str, object] = {}
    for role in RECEIPT.REQUIRED_EVIDENCE_ROLES:
        if role == "historical_m3":
            payload = {
                "schema": RECEIPT.M3_EVIDENCE_ONLY_SCHEMA,
                "verdicts": {
                    "paired_evidence": "GO",
                    "candidate_release": "BLOCKED",
                    "release_authorized": False,
                },
            }
        else:
            payload = {
                "schema": RECEIPT.ROLE_SCHEMAS[role],
                "state": "PASS",
                "source_commit": (
                    commit
                    if role in RECEIPT.TYPED_CANDIDATE_ROLES
                    else RECEIPT.VALIDATED_ENGINE_COMMIT
                ),
                "report_id": hashlib.sha256(role.encode("utf-8")).hexdigest(),
            }
            if role in RECEIPT.TYPED_CANDIDATE_ROLES:
                payload["package_receipt_id"] = "d" * 64
            if role == "m4_source_audit":
                payload["measured_source_commit"] = commit
        evidence[role] = _write_evidence(root, role, payload)
    manifest = {
        "schema": RECEIPT.CANDIDATE_MANIFEST_SCHEMA,
        "state": "PASS",
        "validated_engine_commit": RECEIPT.VALIDATED_ENGINE_COMMIT,
        "release_source_commit": commit,
        "option": "A_ALL_11",
        "findings": {
            "required_ids": list(RECEIPT.M4_FINDING_IDS),
            "closed_ids": list(RECEIPT.M4_FINDING_IDS),
            "open_critical_high_medium": 0,
        },
        "validation": {
            "targeted": True,
            "full_common": True,
            "full_validation": True,
            "host_case_count": 4,
            "specialized_case_count": 5,
            "feature_count": 42,
            "route_count": 32,
            "real_multi_gpu": True,
            "no_cpu_fallback": True,
            "aunp_source_candidate": True,
        },
        "evidence": evidence,
    }
    path = root / "m4-candidate.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


class V110DevelopmentContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def assert_rejected(self, mutate):
        candidate = copy.deepcopy(self.contract)
        mutate(candidate)
        with self.assertRaises(VALIDATOR.ContractError):
            VALIDATOR.validate_contract_data(candidate)

    @staticmethod
    def evidence(kind: str, suffix: str = "evidence") -> dict[str, object]:
        return {
            "kind": kind,
            "path": f"{suffix}.json",
            "sha256": hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def sample_evidence(
        kind: str, suffix: str, cycle: int
    ) -> dict[str, object]:
        return {
            "kind": kind,
            "schema": REGISTRY.SAMPLE_SCHEMAS[kind],
            "cycle_index": cycle,
            "path": f"{suffix}/report.json",
            "sha256": hashlib.sha256(
                f"{suffix}/report.json".encode("utf-8")
            ).hexdigest(),
        }

    def lane(self, lane_name: str, index: int) -> dict[str, object]:
        runtime, backend, topologies, device_count, sample_kind = (
            REGISTRY.LANE_CONTRACTS[lane_name]
        )
        topology = sorted(topologies)[0]
        if topology == "cpu1":
            topology = "cpu8"
        devices = [f"GPU-{position}" for position in range(device_count)]
        seconds = [1.0, 1.0]
        return {
            "runtime_id": hashlib.sha256(
                f"{index}-{runtime}".encode("utf-8")
            ).hexdigest(),
            "runtime": runtime,
            "backend": backend,
            "topology": topology,
            "devices": devices,
            "measured_seconds": seconds,
            "mean_seconds": 1.0,
            "samples": [
                self.sample_evidence(
                    sample_kind, f"{index}-{lane_name}-cycle-{cycle}", cycle
                )
                for cycle in (1, 2)
            ],
        }

    @staticmethod
    def ratio(
        numerator: str, denominator: str
    ) -> dict[str, object]:
        return {
            "numerator_lane": numerator,
            "denominator_lane": denominator,
            "paired_cycle_values": [1.0, 1.0],
            "ratio_of_means": 1.0,
        }

    def ratio_set(self, suffix: str) -> dict[str, object]:
        v103 = f"gpmeep_v103_cuda_{suffix}"
        v110 = f"gpmeep_v110_cuda_{suffix}"
        return {
            "upstream_cpu_over_v103_cuda": self.ratio(
                "upstream_meep_cpu", v103
            ),
            "upstream_cpu_over_v110_cuda": self.ratio(
                "upstream_meep_cpu", v110
            ),
            "v103_cuda_over_v110_cuda": self.ratio(v103, v110),
            "v110_cpu_over_v110_cuda": self.ratio(
                "gpmeep_v110_cpu", v110
            ),
        }

    @staticmethod
    def phase_set() -> dict[str, object]:
        return {
            phase: {
                "available": True,
                "measured_seconds": [0.0, 0.0],
                "mean_seconds": 0.0,
                "basis": ["native phase timer", "native phase timer"],
                "native_sinks": [[], []],
                "cycle_indices": [1, 2],
            }
            for phase in REGISTRY.PHASES
        }

    def open_record(self, observation: dict[str, object], index: int) -> dict[str, object]:
        inventory_path = ROOT / "packaging/v110-performance-inversion-inventory.json"
        lanes: dict[str, object] = {}
        for lane in REGISTRY.LANES - REGISTRY.GPU2_LANES:
            lanes[lane] = self.lane(lane, index)
        if observation["source_lane"] == "mpi_cpu8_vs_gpu1_gpu2":
            for lane in REGISTRY.GPU2_LANES:
                lanes[lane] = self.lane(lane, index)
        else:
            for lane in REGISTRY.GPU2_LANES:
                lanes[lane] = None
        phase_decomposition = {
            lane: self.phase_set() if lane_value is not None else None
            for lane, lane_value in lanes.items()
        }
        gpu2 = (
            self.ratio_set("gpu2")
            if observation["source_lane"] == "mpi_cpu8_vs_gpu1_gpu2"
            else None
        )
        return {
            "schema_version": 2,
            "record_id": f"r-{index}",
            "workload_id": observation["workload_id"],
            "observed_gpu_slower_than_cpu": True,
            "source_inventory": {
                "campaign_id": "v1.0.1-final-report-386e3c4a",
                "inventory_path": "packaging/v110-performance-inversion-inventory.json",
                "inventory_sha256": REGISTRY._sha256(inventory_path),
                "case_digest": observation["workload_identity"]["case_contract_sha256"],
            },
            "representative": {
                "value": observation["representative"],
                "impact": observation["impact"],
                "selection_policy_version": "v110-m0-1",
                "rationale": observation["rationale"],
            },
            "comparison": {
                "physics_fingerprint": "1" * 64,
                "precision": "fp32",
                "stopping_condition": "fixed-work",
                "cpu_topology": "physical-cores",
                "lanes": lanes,
                "ratios": {
                    "gpu1": self.ratio_set("gpu1"),
                    "gpu2": gpu2,
                },
            },
            "phase_decomposition": phase_decomposition,
            "primary_class": "unknown",
            "classification_evidence": [self.evidence("timing", f"{index}-class")],
            "confidence": 0.0,
            "owner": "gpmeep",
            "resolution": {
                "state": "open",
                "same_binary_ablation": None,
                "numerical_regression": None,
                "performance_claim_eligible": True,
                "notes": "M0 seed remains unclassified pending equal-work measurement.",
            },
            "release_blocking": bool(
                observation["representative"] and observation["impact"] == "high"
            ),
        }

    def assert_schema_result(
        self, record: dict[str, object], schema: dict[str, object], accepted: bool
    ) -> None:
        if accepted:
            REGISTRY._validate_json_schema(record, schema)
        else:
            with self.assertRaises(REGISTRY.RegistryError):
                REGISTRY._validate_json_schema(record, schema)

    def test_registry_loader_restores_module_cache_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            success = root / "loader-success.py"
            failure = root / "loader-failure.py"
            success.write_text("value = 23\n", encoding="utf-8")
            failure.write_text(
                "raise RuntimeError('registry loader failure')\n", encoding="utf-8"
            )
            with mock.patch.object(REGISTRY, "ROOT", root):
                prior_name = "gpmeep_registry_loader_prior"
                prior = types.ModuleType(prior_name)
                sys.modules[prior_name] = prior
                try:
                    loaded = REGISTRY._load_module(prior_name, success.name)
                    self.assertEqual(loaded.value, 23)
                    self.assertIs(sys.modules[prior_name], prior)
                    with self.assertRaisesRegex(RuntimeError, "registry loader failure"):
                        REGISTRY._load_module(prior_name, failure.name)
                    self.assertIs(sys.modules[prior_name], prior)
                finally:
                    sys.modules.pop(prior_name, None)

                fresh_name = "gpmeep_registry_loader_fresh"
                sys.modules.pop(fresh_name, None)
                loaded = REGISTRY._load_module(fresh_name, success.name)
                self.assertEqual(loaded.value, 23)
                self.assertNotIn(fresh_name, sys.modules)
                with self.assertRaisesRegex(RuntimeError, "registry loader failure"):
                    REGISTRY._load_module(fresh_name, failure.name)
                self.assertNotIn(fresh_name, sys.modules)

    def test_source_only_gate_skips_only_external_user_bytes(self):
        result = VALIDATOR.validate(require_user_inputs=False)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["user_inputs_checked"], 0)
        VALIDATOR.validate_bound_files(self.contract)
        VALIDATOR.validate_gate_integration(self.contract)

    def test_active_distribution_identity_rejects_each_stale_source(self):
        relatives = (
            "packaging/conda/meta.yaml",
            "packaging/conda/release.json.in",
            "scripts/build-conda-package.sh",
            "scripts/gpmeep_release_manifest.py",
        )
        VALIDATOR.validate_active_distribution_identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            originals: dict[str, str] = {}
            for relative in relatives:
                text = (ROOT / relative).read_text(encoding="utf-8")
                originals[relative] = text
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
            with mock.patch.object(VALIDATOR, "ROOT", root):
                for relative in relatives:
                    with self.subTest(relative=relative):
                        path = root / relative
                        path.write_text(
                            originals[relative].replace("1.1.0", "1.0.3", 1),
                            encoding="utf-8",
                        )
                        with self.assertRaisesRegex(
                            VALIDATOR.ContractError,
                            f"active distribution version differs in {re.escape(relative)}",
                        ):
                            VALIDATOR.validate_active_distribution_identity()
                        path.write_text(originals[relative], encoding="utf-8")

    def test_every_normative_contract_leaf_is_fail_closed(self):
        leaves: list[tuple[tuple[object, ...], object]] = []

        def visit(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path + (key,))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, path + (index,))
            else:
                leaves.append((path, value))

        def replacement(value):
            if isinstance(value, bool):
                return not value
            if type(value) is int:
                return value + 1
            if isinstance(value, float):
                return value * 2 if value else 1.0
            if isinstance(value, str):
                return value + "-mutated"
            if value is None:
                return "non-null"
            raise AssertionError(f"unsupported leaf type: {type(value)}")

        visit(self.contract)
        self.assertGreater(len(leaves), 100)
        for path, original in leaves:
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.contract)
                parent = candidate
                for component in path[:-1]:
                    parent = parent[component]
                parent[path[-1]] = replacement(original)
                with self.assertRaises(VALIDATOR.ContractError):
                    VALIDATOR.validate_contract_data(candidate)

    def test_unknown_keys_are_rejected_at_root_and_nested_levels(self):
        self.assert_rejected(lambda value: value.__setitem__("escape_hatch", True))
        self.assert_rejected(
            lambda value: value["backend_policy"].__setitem__("fallback", "cpu")
        )
        self.assert_rejected(
            lambda value: value["backend_policy"]["acknowledgement"].__setitem__(
                "persistent", True
            )
        )

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        with self.assertRaises(VALIDATOR.ContractError):
            VALIDATOR._load_json('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(VALIDATOR.ContractError):
            VALIDATOR._load_json('{"value":NaN}')

    def test_backend_auto_switch_fallback_and_unbound_ack_are_rejected(self):
        for key in (
            "automatic_backend_selection",
            "runtime_backend_switching",
            "silent_cpu_fallback",
        ):
            self.assert_rejected(
                lambda value, key=key: value["backend_policy"].__setitem__(key, True)
            )
        self.assert_rejected(
            lambda value: value["backend_policy"].__setitem__(
                "mismatch_override_requires_acknowledgement", False
            )
        )
        self.assert_rejected(
            lambda value: value["backend_policy"]["acknowledgement"].__setitem__(
                "required_digest_bindings", ["selected_backend"]
            )
        )
        self.assert_rejected(
            lambda value: value["backend_policy"]["acknowledgement"].__setitem__(
                "replay_is_forbidden", False
            )
        )

    def test_inversion_lanes_phases_telemetry_and_release_blocks_are_locked(self):
        inversion = self.contract["performance_inversion_policy"]
        self.assertEqual(inversion["comparison_lanes"], VALIDATOR.EXPECTED_LANES)
        self.assertEqual(inversion["required_decomposition"], VALIDATOR.EXPECTED_PHASES)
        self.assertEqual(inversion["required_telemetry"], VALIDATOR.EXPECTED_TELEMETRY)
        self.assert_rejected(
            lambda value: value["performance_inversion_policy"]["comparison_lanes"].pop()
        )
        self.assert_rejected(
            lambda value: value["performance_inversion_policy"]["required_decomposition"].pop()
        )
        self.assert_rejected(
            lambda value: value["performance_inversion_policy"]["required_telemetry"].pop()
        )
        self.assert_rejected(
            lambda value: value["performance_inversion_policy"]["reason_classes"]
            ["engine_overhead"].__setitem__("warning_may_be_terminal", True)
        )
        self.assert_rejected(
            lambda value: value["performance_inversion_policy"]["reason_classes"]
            ["unknown"].__setitem__(
                "representative_high_impact_case_blocks_release", False
            )
        )

    def test_equal_work_plan_is_three_way_versioned_and_order_reversed(self):
        plan_path = ROOT / "packaging/v110-m4-equal-work-plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [lane["id"] for lane in plan["lanes"]],
            VALIDATOR.EXPECTED_LANES,
        )
        self.assertEqual(len(plan["cases"]), 13)
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [case["path"] for case in plan["cases"]],
            [item["workload_id"] for item in inventory["observations"]],
        )
        self.assertEqual(
            [case["path"] for case in plan["cases"] if case["gpu2"]],
            [
                "python/examples/adjoint_optimization/near2far_3d_validation.py",
                "python/examples/edge_emitter_3D.py",
                "python/examples/dipole_in_vacuum_cyl_off_axis.py",
                "python/tests/test_adjoint_cyl.py",
            ],
        )
        self.assertEqual(plan["cycles"][1]["case_order"], "forward")
        self.assertEqual(plan["cycles"][2]["case_order"], "reverse")
        self.assertEqual(plan["cycles"][2]["lane_order"], "reverse")
        self.assertEqual(plan["execution_counts"], {
            "per_cycle": 60,
            "all_cycles": 180,
            "measured": 120,
        })
        self.assertTrue(plan["reporting"]["unavailable_phase_seconds_are_null"])

    def test_installed_contract_layout_requires_exact_equal_work_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            installed_root = pathlib.Path(directory) / "share/gpmeep"
            libexec = installed_root / "libexec"
            packaging = installed_root / "packaging"
            libexec.mkdir(parents=True)
            packaging.mkdir(parents=True)
            validator = libexec / "validate-v110-development-contract.py"
            validator.write_bytes(SCRIPT.read_bytes())
            filenames = (
                "v110-development-contract.json",
                "v110-m4-equal-work-plan.json",
                "v110-performance-inversion-record.schema.json",
                "v110-performance-inversion-inventory.json",
            )
            for filename in filenames:
                (packaging / filename).write_bytes(
                    (ROOT / "packaging" / filename).read_bytes()
                )

            def run_installed() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        str(validator),
                        "--contract",
                        str(packaging / "v110-development-contract.json"),
                        "--installed-package",
                        "--json",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
                )

            completed = run_installed()
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["state"], "PASS")

            plan = packaging / "v110-m4-equal-work-plan.json"
            original = plan.read_bytes()
            plan.unlink()
            missing = run_installed()
            self.assertEqual(missing.returncode, 2)
            self.assertIn("equal-work plan is absent", missing.stdout)

            plan.write_bytes(original + b"\n")
            mutated = run_installed()
            self.assertEqual(mutated.returncode, 2)
            self.assertIn("equal-work plan hash mismatch", mutated.stdout)

    def test_inversion_inventory_is_immutable_complete_and_release_open(self):
        result = REGISTRY.validate_registry(require_closed=False, evidence_root=None)
        self.assertEqual(result["state"], "OPEN")
        self.assertEqual(result["observed_count"], 13)
        self.assertEqual(result["registered_count"], 0)
        self.assertEqual(result["missing_count"], 13)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(REGISTRY.RegistryError):
                REGISTRY.validate_registry(
                    require_closed=True, evidence_root=pathlib.Path(directory)
                )

    def test_inversion_schema_requires_all_lanes_and_typed_evidence(self):
        schema = json.loads(
            (ROOT / "packaging/v110-performance-inversion-record.schema.json").read_text(
                encoding="utf-8"
            )
        )
        required_lanes = schema["properties"]["comparison"]["properties"]["lanes"]["required"]
        self.assertEqual(set(required_lanes), REGISTRY.LANES)
        required_ratios = schema["$defs"]["ratio_set"]["required"]
        self.assertEqual(set(required_ratios), REGISTRY.RATIO_KEYS)
        lane = schema["$defs"]["lane"]
        self.assertEqual(
            set(lane["required"]),
            {
                "runtime_id",
                "runtime",
                "backend",
                "topology",
                "devices",
                "measured_seconds",
                "mean_seconds",
                "samples",
            },
        )
        evidence = schema["$defs"]["evidence"]
        self.assertEqual(set(evidence["required"]), {"kind", "path", "sha256"})
        self.assertEqual(set(evidence["properties"]["kind"]["enum"]), REGISTRY.EVIDENCE_KINDS)

    def test_unavailable_phase_is_explicit_null_not_fake_zero(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        schema = json.loads(
            (ROOT / "packaging/v110-performance-inversion-record.schema.json").read_text()
        )
        observation = inventory["observations"][0]
        record = self.open_record(observation, 70)
        phase_lane = "upstream_meep_cpu"
        record["phase_decomposition"][phase_lane]["source"] = {
            "available": False,
            "measured_seconds": [None, None],
            "mean_seconds": None,
            "reason_code": "no_comparable_native_timer",
            "basis": [
                "No comparable native source timer exists in this sample.",
                "No comparable native source timer exists in this sample.",
            ],
            "native_sinks": [[], []],
            "cycle_indices": [1, 2],
        }
        self.assert_schema_result(record, schema, True)
        REGISTRY._validate_record(
            record,
            observation,
            "packaging/v110-performance-inversion-inventory.json",
            REGISTRY._sha256(
                ROOT / "packaging/v110-performance-inversion-inventory.json"
            ),
            inventory["source_campaign"]["id"],
            None,
        )

        fake_zero = copy.deepcopy(record)
        fake_zero["phase_decomposition"][phase_lane]["source"][
            "measured_seconds"
        ] = [0.0, 0.0]
        self.assert_schema_result(fake_zero, schema, False)
        with self.assertRaises(REGISTRY.RegistryError):
            REGISTRY._validate_record(
                fake_zero,
                observation,
                "packaging/v110-performance-inversion-inventory.json",
                REGISTRY._sha256(
                    ROOT / "packaging/v110-performance-inversion-inventory.json"
                ),
                inventory["source_campaign"]["id"],
                None,
            )

    def test_mpi_inversion_requires_both_versioned_gpu2_lanes(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        observation = next(
            item
            for item in inventory["observations"]
            if item["source_lane"] == "mpi_cpu8_vs_gpu1_gpu2"
        )
        arguments = (
            observation,
            "packaging/v110-performance-inversion-inventory.json",
            REGISTRY._sha256(
                ROOT / "packaging/v110-performance-inversion-inventory.json"
            ),
            inventory["source_campaign"]["id"],
            None,
        )
        for lane in REGISTRY.GPU2_LANES:
            record = self.open_record(observation, 71)
            record["comparison"]["lanes"][lane] = None
            with self.subTest(lane=lane), self.assertRaises(REGISTRY.RegistryError):
                REGISTRY._validate_record(record, *arguments)

    def test_inversion_semantics_reject_exponent_overflow_numbers(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        observation = inventory["observations"][0]
        arguments = (
            observation,
            "packaging/v110-performance-inversion-inventory.json",
            REGISTRY._sha256(
                ROOT / "packaging/v110-performance-inversion-inventory.json"
            ),
            inventory["source_campaign"]["id"],
            None,
        )
        mutations = []
        lane = next(iter(REGISTRY.LANES - REGISTRY.GPU2_LANES))
        mutations.append(
            lambda record: record["comparison"]["lanes"][lane][
                "measured_seconds"
            ].__setitem__(0, float("inf"))
        )
        mutations.append(
            lambda record: record["phase_decomposition"][lane]["fdtd_timestep"][
                "measured_seconds"
            ].__setitem__(0, float("inf"))
        )
        mutations.append(
            lambda record: record.__setitem__("confidence", float("inf"))
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                record = self.open_record(observation, 80 + index)
                mutate(record)
                with self.assertRaises(REGISTRY.RegistryError):
                    REGISTRY._validate_record(record, *arguments)

    def test_inversion_semantics_recompute_means_ratios_and_lane_identity(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        observation = inventory["observations"][0]
        arguments = (
            observation,
            "packaging/v110-performance-inversion-inventory.json",
            REGISTRY._sha256(
                ROOT / "packaging/v110-performance-inversion-inventory.json"
            ),
            inventory["source_campaign"]["id"],
            None,
        )
        mutations = (
            lambda record: record["comparison"]["lanes"][
                "upstream_meep_cpu"
            ].__setitem__("mean_seconds", 2.0),
            lambda record: record["comparison"]["ratios"]["gpu1"][
                "v103_cuda_over_v110_cuda"
            ]["paired_cycle_values"].__setitem__(0, 2.0),
            lambda record: record["comparison"]["lanes"][
                "gpmeep_v103_cuda_gpu1"
            ].__setitem__("runtime", "gpmeep_v110"),
            lambda record: record["comparison"]["lanes"][
                "gpmeep_v110_cuda_gpu1"
            ]["samples"][0].__setitem__(
                "schema", REGISTRY.SAMPLE_SCHEMAS["v103_sample"]
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                record = self.open_record(observation, 90 + index)
                mutate(record)
                with self.assertRaises(REGISTRY.RegistryError):
                    REGISTRY._validate_record(record, *arguments)

    def test_all_thirteen_open_records_conform_without_validator_crash(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        schema = json.loads(
            (ROOT / "packaging/v110-performance-inversion-record.schema.json").read_text()
        )
        for index, observation in enumerate(inventory["observations"]):
            with self.subTest(workload=observation["workload_id"]):
                record = self.open_record(observation, index)
                self.assert_schema_result(record, schema, True)
                record_id, blocking, closed = REGISTRY._validate_record(
                    record,
                    observation,
                    "packaging/v110-performance-inversion-inventory.json",
                    REGISTRY._sha256(
                        ROOT / "packaging/v110-performance-inversion-inventory.json"
                    ),
                    inventory["source_campaign"]["id"],
                    None,
                )
                self.assertEqual(record_id, f"r-{index}")
                self.assertIs(blocking, record["release_blocking"])
                self.assertFalse(closed)

    def test_registry_override_recomputes_unknown_high_blockers(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            records = pathlib.Path(directory) / "records"
            records.mkdir()
            expected_blockers = []
            for index, observation in enumerate(inventory["observations"]):
                record = self.open_record(observation, index)
                (records / f"{record['record_id']}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                if observation["representative"] and observation["impact"] == "high":
                    expected_blockers.append(observation["workload_id"])
            result = REGISTRY.validate_registry(
                require_closed=False,
                evidence_root=None,
                records_directory=records,
            )
            self.assertEqual(result["registered_count"], 13)
            self.assertEqual(result["blocking_workloads"], sorted(expected_blockers))
            high_record_path = next(
                path
                for path in records.glob("*.json")
                if json.loads(path.read_text())["workload_id"] in expected_blockers
            )
            high_record = json.loads(high_record_path.read_text())
            high_record["release_blocking"] = False
            high_record_path.write_text(json.dumps(high_record), encoding="utf-8")
            with self.assertRaisesRegex(
                REGISTRY.RegistryError, "release_blocking disagrees with policy"
            ):
                REGISTRY.validate_registry(
                    require_closed=False,
                    evidence_root=None,
                    records_directory=records,
                )

    def test_require_closed_rejects_even_nonblocking_open_records(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory).resolve()
            records = evidence / "records"
            records.mkdir()
            for index, observation in enumerate(inventory["observations"]):
                (records / f"record-{index:02d}.json").write_text(
                    json.dumps({"workload_id": observation["workload_id"]}),
                    encoding="utf-8",
                )

            def open_record(record, *_args):
                return record["workload_id"], False, False

            with mock.patch.object(REGISTRY, "_validate_json_schema"), mock.patch.object(
                REGISTRY, "_validate_record", side_effect=open_record
            ):
                measured = REGISTRY.validate_registry(
                    require_closed=False,
                    evidence_root=evidence,
                    records_directory=records,
                )
                self.assertEqual(measured["state"], "OPEN")
                self.assertEqual(measured["blocking_count"], 0)
                self.assertEqual(measured["closed_count"], 0)
                self.assertEqual(measured["open_count"], 13)
                with self.assertRaisesRegex(
                    REGISTRY.RegistryError, "evidence-backed resolution"
                ):
                    REGISTRY.validate_registry(
                        require_closed=True,
                        evidence_root=evidence,
                        records_directory=records,
                    )

    def test_registry_override_rejects_uninventoried_directory_members(self):
        with tempfile.TemporaryDirectory() as directory:
            records = pathlib.Path(directory)
            (records / "claimed-pass.txt").write_text("PASS", encoding="utf-8")
            with self.assertRaisesRegex(
                REGISTRY.RegistryError, "unexpected registry directory member"
            ):
                REGISTRY.validate_registry(
                    require_closed=False,
                    evidence_root=None,
                    records_directory=records,
                )

    def test_normative_schema_rejects_bad_id_and_duplicate_evidence(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        schema = json.loads(
            (ROOT / "packaging/v110-performance-inversion-record.schema.json").read_text()
        )
        record = self.open_record(inventory["observations"][0], 0)
        record["record_id"] = "A A"
        self.assert_schema_result(record, schema, False)
        record = self.open_record(inventory["observations"][0], 0)
        record["classification_evidence"] *= 2
        self.assert_schema_result(record, schema, False)

    def test_schema_only_hardware_claims_require_registered_work_families(self):
        inventory_path = ROOT / "packaging/v110-performance-inversion-inventory.json"
        inventory = json.loads(inventory_path.read_text())
        unsupported = [item for item in inventory["observations"]
                       if item["workload_id"] != "python/examples/phase_in_material.py"]
        self.assertEqual(len(unsupported), 12)
        for index, observation in enumerate(unsupported):
            record = self.open_record(observation, 100 + index)
            record["primary_class"] = "hardware_crossover"
            record["release_blocking"] = False
            record["classification_evidence"] = [
                self.evidence("size_run_sweep", "sweep"),
                self.evidence("profiler_or_ablation", "profile"),
            ]
            record["resolution"]["state"] = "proved_hardware_crossover"
            record["resolution"]["numerical_regression"] = self.evidence(
                "numerical_regression", "numerical"
            )
            with self.subTest(workload=observation["workload_id"]), self.assertRaisesRegex(
                REGISTRY.RegistryError, "actual-work and numerical family verifier"
            ):
                REGISTRY._validate_record(
                    record, observation,
                    "packaging/v110-performance-inversion-inventory.json",
                    REGISTRY._sha256(inventory_path), inventory["source_campaign"]["id"], None,
                )

    def test_hardware_crossover_requires_closed_state_and_both_proofs(self):
        inventory = json.loads(
            (ROOT / "packaging/v110-performance-inversion-inventory.json").read_text()
        )
        observation = next(
            item
            for item in inventory["observations"]
            if item["workload_id"] == "python/examples/phase_in_material.py"
        )
        record = self.open_record(observation, 90)
        record["primary_class"] = "hardware_crossover"
        record["release_blocking"] = False
        record["classification_evidence"] = [
            self.evidence("size_run_sweep", "sweep"),
            self.evidence("profiler_or_ablation", "profile"),
        ]
        arguments = (
            observation,
            "packaging/v110-performance-inversion-inventory.json",
            REGISTRY._sha256(
                ROOT / "packaging/v110-performance-inversion-inventory.json"
            ),
            inventory["source_campaign"]["id"],
            None,
        )
        with self.assertRaises(REGISTRY.RegistryError):
            REGISTRY._validate_record(record, *arguments)
        record["resolution"]["state"] = "proved_hardware_crossover"
        record["resolution"]["numerical_regression"] = self.evidence(
            "numerical_regression", "numerical"
        )
        REGISTRY._validate_record(record, *arguments)
        wrong_anchor = copy.deepcopy(record)
        expected_cpu = next(
            item["cpu_ranks"]
            for item in json.loads(
                (ROOT / "packaging/v110-m4-equal-work-plan.json").read_text()
            )["cases"]
            if item["path"] == record["workload_id"]
        )
        wrong_anchor["comparison"]["lanes"]["gpmeep_v110_cpu"]["topology"] = (
            "cpu1" if expected_cpu == 8 else "cpu8"
        )
        with self.assertRaisesRegex(REGISTRY.RegistryError, "practical M4 CPU/GPU1 anchor"):
            REGISTRY._validate_record(wrong_anchor, *arguments)
        for kind in ("size_run_sweep", "profiler_or_ablation"):
            incomplete = copy.deepcopy(record)
            incomplete["classification_evidence"] = [
                item for item in incomplete["classification_evidence"] if item["kind"] != kind
            ]
            with self.assertRaises(REGISTRY.RegistryError):
                REGISTRY._validate_record(incomplete, *arguments)

    def test_release_closure_rejects_opaque_or_cross_workload_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            opaque = root / "opaque.json"
            opaque.write_text('{"state":"PASS"}\n')
            reference = {
                "kind": "numerical_regression",
                "path": opaque.name,
                "sha256": hashlib.sha256(opaque.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(REGISTRY.RegistryError, "schema differs"):
                REGISTRY._evidence(
                    reference,
                    "opaque",
                    root,
                    expected_workload_id="python/examples/other.py",
                    expected_campaign_id="campaign",
                    expected_inventory_sha256="a" * 64,
                    expected_classification="hardware_crossover",
                )

            plan = json.loads(
                (ROOT / "packaging/v110-m4-equal-work-plan.json").read_text()
            )
            comparisons = []
            for case in plan["cases"]:
                lanes = [
                    "gpmeep_v110_cpu",
                    "gpmeep_v103_cuda_gpu1",
                    "gpmeep_v110_cuda_gpu1",
                ]
                if case["gpu2"]:
                    lanes += [
                        "gpmeep_v103_cuda_gpu2",
                        "gpmeep_v110_cuda_gpu2",
                    ]
                for cycle in (0, 1, 2):
                    for lane in lanes:
                        reference_lane = (
                            "gpmeep_v110_cpu"
                            if (
                                case["path"] in REGISTRY.GPMEEP_CPU_REFERENCE_CASES
                                and lane != "gpmeep_v110_cpu"
                            )
                            else "upstream_meep_cpu"
                        )
                        method, result = _numerical_result_fixture(
                            case["path"], lane
                        )
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
            payload = {
                "schema": REGISTRY.EVIDENCE_SCHEMAS["numerical_regression"],
                "outcome": "PASS",
                "comparison_count": 141,
                "comparisons": comparisons,
            }
            cross = root / "cross.json"
            cross.write_text(json.dumps(payload, sort_keys=True))
            reference.update(
                path=cross.name,
                sha256=hashlib.sha256(cross.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(REGISTRY.RegistryError, "expected workload"):
                REGISTRY._evidence(
                    reference,
                    "cross",
                    root,
                    expected_workload_id="python/examples/other.py",
                    expected_campaign_id="campaign",
                    expected_inventory_sha256="a" * 64,
                    expected_classification="hardware_crossover",
                )

            payload["comparisons"][-1] = dict(payload["comparisons"][0])
            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps(payload, sort_keys=True))
            reference.update(
                path=duplicate.name,
                sha256=hashlib.sha256(duplicate.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(REGISTRY.RegistryError, "identity repeats"):
                REGISTRY._evidence(
                    reference,
                    "duplicate",
                    root,
                    expected_workload_id=plan["cases"][0]["path"],
                    expected_campaign_id="campaign",
                    expected_inventory_sha256="a" * 64,
                    expected_classification="unknown",
                )

    def test_fixed_sample_pair_cannot_impersonate_distinct_evidence_kinds(self):
        common = {
            "state": "PASS",
            "workload_id": "python/examples/refl-quartz.py",
            "source_campaign_id": "campaign",
            "source_inventory_sha256": "a" * 64,
            "classification": "hardware_crossover",
            "inputs": [{}, {}],
            "result": {},
        }
        arguments = {
            "size_run_sweep": {
                "size_flag": "--gpmeep-sweep-size",
                "run_length_flag": "--gpmeep-sweep-run-length",
                "minimum_sizes": 3,
                "minimum_run_lengths": 2,
                "require_observed_crossover": True,
            },
            "profiler_or_ablation": {
                "intervention_flag": "--gpmeep-ablation",
                "intervention": "launch-batching",
                "minimum_cycles": 2,
                "require_reversed_order": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for kind in ("size_run_sweep", "profiler_or_ablation"):
                with self.subTest(kind=kind):
                    payload = {
                        "schema": REGISTRY.EVIDENCE_SCHEMAS[kind],
                        "kind": kind,
                        "arguments": arguments[kind],
                        **common,
                    }
                    unsigned = dict(payload)
                    payload["evidence_id"] = hashlib.sha256(
                        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    path = root / f"{kind}.json"
                    path.write_text(json.dumps(payload, sort_keys=True))
                    with self.assertRaisesRegex(REGISTRY.RegistryError, "producer census"):
                        REGISTRY._verify_typed_evidence(
                            path,
                            root,
                            expected_kind=kind,
                            expected_workload_id="python/examples/refl-quartz.py",
                            expected_campaign_id="campaign",
                            expected_inventory_sha256="a" * 64,
                            expected_classification="hardware_crossover",
                        )

    def test_user_scope_exclusion_is_not_a_release_closing_schema_state(self):
        schema = json.loads(
            (ROOT / "packaging/v110-performance-inversion-record.schema.json").read_text()
        )
        states = schema["properties"]["resolution"]["properties"]["state"]["enum"]
        kinds = schema["$defs"]["evidence"]["properties"]["kind"]["enum"]
        self.assertNotIn("user_approved_scope_exclusion", states)
        self.assertNotIn("user_scope_approval", kinds)

    def test_registry_loader_rejects_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "record.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(REGISTRY.RegistryError):
                REGISTRY._load(path)
            path.write_text('{"value":NaN}')
            with self.assertRaises(REGISTRY.RegistryError):
                REGISTRY._load(path)

    def test_manual_workflow_has_no_automatic_trigger(self):
        workflow = (ROOT / ".github/workflows/v110-contract.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("workflow_call:", workflow)
        for trigger in ("pull_request:", "push:", "schedule:"):
            self.assertNotIn(trigger, workflow)
        self.assertIn("validate-v110-development-contract.py", workflow)
        self.assertIn("scripts/create-env.sh v110-test", workflow)
        self.assertIn(".envs/gpmeep-v1.1-test/bin/python", workflow)
        self.assertIn('version("jsonschema") == "4.26.0"', workflow)
        self.assertIn(
            "scripts.tests.test_v110_development_contract.V110DevelopmentContractTest",
            workflow,
        )
        self.assertNotIn(
            "scripts.tests.test_v110_development_contract\n",
            workflow,
        )
        self.assertEqual(
            {
                name
                for name in dir(V110ExternalEvidenceContractTest)
                if name.startswith("test_")
            },
            {
                "test_actual_user_inputs_and_archive_members_are_byte_bound",
                "test_repository_contract_documents_bound_files_and_inputs_pass",
            },
        )
        self.assertFalse(
            any(
                name in V110DevelopmentContractTest.__dict__
                for name in {
                    "test_actual_user_inputs_and_archive_members_are_byte_bound",
                    "test_repository_contract_documents_bound_files_and_inputs_pass",
                }
            )
        )
        host_python = "/usr/bin/" + "python3"
        self.assertNotIn(
            host_python,
            (ROOT / "scripts/tests/test_v110_development_contract.py").read_text(),
        )

    def test_builder_and_installed_verifier_are_connected_to_contract_gate(self):
        builder = (ROOT / "scripts/build-conda-package.sh").read_text(encoding="utf-8")
        package_build = (ROOT / "packaging/conda/build.sh").read_text(
            encoding="utf-8"
        )
        verifier = (ROOT / "scripts/verify-gpmeep-package.py").read_text(encoding="utf-8")
        runtime_identity = (
            ROOT / "scripts/python-validation/v110_runtime_identity.py"
        ).read_text(encoding="utf-8")
        self.assertIn("validate-v110-development-contract.py", builder)
        self.assertIn("validate-v110-inversion-registry.py", builder)
        self.assertIn("--require-closed", builder)
        self.assertIn("validate-v110-development-contract.py", verifier)
        self.assertIn("DISTRIBUTION_VERSION == \"1.1.0\"", verifier)
        self.assertIn('"${SRC_DIR}/packaging/v110-m4-equal-work-plan.json"', package_build)
        self.assertNotIn('/usr/bin/mv -- "${PACKAGE}"', builder)
        self.assertIn('NONPUBLISHABLE_MARKER="${PACKAGE}.NON_PUBLISHABLE"', builder)
        self.assertIn("gpmeep-non-publishable-development-artifact-v1", builder)
        self.assertIn("_validate_nonpublishable_marker", runtime_identity)
        self.assertIn("candidate package carries a non-publishable marker", runtime_identity)

    def test_attestation_atomic_writer_is_external_mode_600_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "attestation.json"
            ATTESTER._write_atomic_new(path, {"state": "PASS"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()), {"state": "PASS"})
            with self.assertRaises(ATTESTER.AttestationError):
                ATTESTER._write_atomic_new(path, {"state": "changed"})

    def test_attestation_verify_requires_exact_outer_hash_and_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "attestation.json"
            path.write_text('{"state":"PASS"}', encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(ATTESTER.AttestationError, "SHA-256 differs"):
                ATTESTER._verify(path, "0" * 64)
            duplicate = b'{"schema_version":1,"schema_version":1,"state":"PASS"}'
            path.write_bytes(duplicate)
            with self.assertRaisesRegex(ATTESTER.AttestationError, "duplicate JSON key"):
                ATTESTER._verify(path, hashlib.sha256(duplicate).hexdigest())
            nonfinite = b'{"schema_version":1,"state":"PASS","value":NaN}'
            path.write_bytes(nonfinite)
            with self.assertRaisesRegex(ATTESTER.AttestationError, "non-finite"):
                ATTESTER._verify(path, hashlib.sha256(nonfinite).hexdigest())

    def test_builder_behavior_fails_closed_without_explicit_mode_or_authority(self):
        builder = ROOT / "scripts/build-conda-package.sh"
        with tempfile.TemporaryDirectory() as directory:
            missing_mode = subprocess.run(
                ["/bin/bash", "-p", str(builder), "--output-dir", directory],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(missing_mode.returncode, 0)
            self.assertIn("explicit --development-build or --candidate-build", missing_mode.stderr)
            retired_release = subprocess.run(
                [
                    "/bin/bash",
                    "-p",
                    str(builder),
                    "--release-build",
                    "--output-dir",
                    directory,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(retired_release.returncode, 0)
            self.assertIn("release pre-build authority is retired", retired_release.stderr)
            missing_authority = subprocess.run(
                [
                    "/bin/bash",
                    "-p",
                    str(builder),
                    "--candidate-build",
                    "--output-dir",
                    directory,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(missing_authority.returncode, 0)
            self.assertIn("requires evidence root, attestation", missing_authority.stderr)
            attestation = pathlib.Path(directory) / "attestation.json"
            attestation.write_text("{}\n", encoding="utf-8")
            manifest = pathlib.Path(directory) / "candidate.json"
            manifest.write_text("{}\n", encoding="utf-8")
            manifest_link = pathlib.Path(directory) / "candidate-link.json"
            manifest_link.symlink_to(manifest)
            source_policy = pathlib.Path(directory) / "source-policy.json"
            source_policy.write_text("{}\n", encoding="utf-8")
            source_verifier = pathlib.Path(directory) / "source-verifier.py"
            source_verifier.write_text("# verifier\n", encoding="utf-8")
            inversion_records = pathlib.Path(directory) / "closed-records"
            inversion_records.mkdir()
            inversion_terminal = pathlib.Path(directory) / "closed-registry.json"
            inversion_terminal.write_text("{}\n", encoding="utf-8")
            symlink_manifest = subprocess.run(
                [
                    "/bin/bash",
                    "-p",
                    str(builder),
                    "--candidate-build",
                    "--output-dir",
                    directory,
                    "--candidate-evidence-root",
                    directory,
                    "--candidate-attestation",
                    str(attestation),
                    "--candidate-attestation-sha256",
                    "a" * 64,
                    "--m4-candidate-manifest",
                    str(manifest_link),
                    "--source-transition-policy",
                    str(source_policy),
                    "--source-transition-policy-sha256",
                    hashlib.sha256(source_policy.read_bytes()).hexdigest(),
                    "--source-transition-verifier",
                    str(source_verifier),
                    "--source-transition-verifier-sha256",
                    hashlib.sha256(source_verifier.read_bytes()).hexdigest(),
                    "--candidate-inversion-terminal",
                    str(inversion_terminal),
                    "--candidate-inversion-terminal-sha256",
                    hashlib.sha256(inversion_terminal.read_bytes()).hexdigest(),
                    "--candidate-inversion-terminal-id",
                    "b" * 64,
                    "--candidate-inversion-records-directory",
                    str(inversion_records),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(symlink_manifest.returncode, 0)
            self.assertIn("non-symlink", symlink_manifest.stderr)

    def test_candidate_receipt_binds_m4_authority_but_never_publication(self):
        contract_result = VALIDATOR.validate(require_user_inputs=False)
        registry_result = REGISTRY.validate_registry(
            require_closed=False, evidence_root=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            contract_gate = root / "contract.json"
            registry_gate = root / "registry.json"
            attestation_gate = root / "attestation.json"
            contract_gate.write_text(json.dumps(contract_result))
            registry_gate.write_text(json.dumps(registry_result))
            candidate_manifest = candidate_manifest_fixture(root, "c" * 40)
            source_policy = root / "independent-source-policy.json"
            source_policy.write_text('{"fixture":"operator-authority"}\n')
            source_policy_sha256 = hashlib.sha256(source_policy.read_bytes()).hexdigest()
            source_verifier = root / "independent-source-verifier.py"
            source_verifier.write_bytes(
                (ROOT / "scripts/verify-v110-release-source-transition.py").read_bytes()
            )
            source_verifier_sha256 = hashlib.sha256(source_verifier.read_bytes()).hexdigest()
            attestation_gate.write_text('{"fixture":"attestation"}\n')
            expected_attestation = hashlib.sha256(attestation_gate.read_bytes()).hexdigest()
            closed = dict(registry_result)
            closed.update(
                state="PASS",
                registered_count=13,
                missing_count=0,
                blocking_count=0,
                closed_count=13,
                open_count=0,
                missing_workloads=[],
                blocking_workloads=[],
                open_workloads=[],
            )
            registry_gate.write_text(json.dumps(closed))
            closed_records = root / "closed-records"
            closed_records.mkdir()
            closed_record_files = []
            closed_record_inventory = []
            for index in range(13):
                path = closed_records / f"record-{index:02d}.json"
                path.write_text(json.dumps({"record": index}) + "\n")
                closed_record_files.append(path)
                closed_record_inventory.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            closed_terminal_value = {
                "schema": RECEIPT.CLOSED_REGISTRY_SCHEMA,
                "state": "PASS",
                "records_directory": closed_records.relative_to(root).as_posix(),
                "records": closed_record_inventory,
                "registry_result": closed,
            }
            closed_terminal_value["terminal_id"] = RECEIPT._canonical_sha256(
                closed_terminal_value
            )
            closed_terminal = root / "closed-registry.json"
            closed_terminal.write_text(
                json.dumps(closed_terminal_value, sort_keys=True) + "\n"
            )
            closed_authority = {
                "file": RECEIPT._portable_file_record(closed_terminal, root),
                "terminal": closed_terminal_value,
            }
            evidence_files = [
                candidate_manifest,
                closed_terminal,
                *closed_record_files,
            ]
            evidence_files.extend(
                root / item["bundle_relative_path"]
                for item in json.loads(candidate_manifest.read_text())["evidence"].values()
            )
            attested = {
                "repository": {
                    "m0_commit": "b" * 40,
                    "current_head": "c" * 40,
                    "clean": True,
                },
                "external_evidence": [
                    {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in evidence_files
                ],
            }
            producer_patches = (
                mock.patch.object(RECEIPT, "_require_isolated_authority"),
                mock.patch.object(
                    RECEIPT,
                    "_preflight_source_authority",
                    side_effect=lambda **kwargs: json.loads(
                        (
                            pathlib.Path(kwargs["evidence_root"])
                            / json.loads(pathlib.Path(kwargs["candidate_manifest_path"]).read_text())["evidence"]["m4_source_audit"]["bundle_relative_path"]
                        ).read_text()
                    ),
                ),
                mock.patch.object(RECEIPT, "_replay_contract_gate", return_value=contract_result),
                mock.patch.object(RECEIPT, "_replay_registry_gate", return_value=closed),
                mock.patch.object(
                    RECEIPT,
                    "_replay_closed_registry",
                    return_value=(closed_authority, closed_records),
                ),
                mock.patch.object(RECEIPT, "_replay_attestation_gate", return_value=attested),
                mock.patch.object(
                    RECEIPT,
                    "_replay_native_evidence",
                    side_effect=lambda role, path, payload, release_source_commit, **kwargs: RECEIPT._canonical_sha256(payload),
                ),
            )
            with producer_patches[0], producer_patches[1], producer_patches[2], producer_patches[3], producer_patches[4], producer_patches[5], producer_patches[6]:
                receipt = RECEIPT.create(
                    mode="candidate",
                    source_commit="c" * 40,
                    contract_gate_path=contract_gate,
                    registry_gate_path=registry_gate,
                    attestation_gate_path=attestation_gate,
                    expected_attestation_sha256=expected_attestation,
                    candidate_manifest_path=candidate_manifest,
                    evidence_root=root,
                    source_policy_path=source_policy,
                    expected_source_policy_sha256=source_policy_sha256,
                    source_verifier_path=source_verifier,
                    expected_source_verifier_sha256=source_verifier_sha256,
                    closed_registry_terminal_path=closed_terminal,
                    expected_closed_registry_sha256=hashlib.sha256(
                        closed_terminal.read_bytes()
                    ).hexdigest(),
                    expected_closed_registry_terminal_id=closed_terminal_value[
                        "terminal_id"
                    ],
                    closed_registry_records_directory=closed_records,
                )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "actual contract gate"):
                RECEIPT.verify(
                    receipt_path=receipt_path,
                    contract_path=CONTRACT,
                    schema_path=ROOT / "packaging/v110-performance-inversion-record.schema.json",
                    inventory_path=ROOT / "packaging/v110-performance-inversion-inventory.json",
                    require_candidate=True,
                )
            with producer_patches[0], producer_patches[1], producer_patches[2], producer_patches[3], producer_patches[4], producer_patches[5], producer_patches[6]:
                verified = RECEIPT.verify(
                    receipt_path=receipt_path,
                    contract_path=CONTRACT,
                    schema_path=ROOT / "packaging/v110-performance-inversion-record.schema.json",
                    inventory_path=ROOT / "packaging/v110-performance-inversion-inventory.json",
                    require_candidate=True,
                    contract_gate_path=contract_gate,
                    registry_gate_path=registry_gate,
                    attestation_gate_path=attestation_gate,
                    candidate_manifest_path=candidate_manifest,
                    evidence_root=root,
                    source_policy_path=source_policy,
                    expected_source_policy_sha256=source_policy_sha256,
                    source_verifier_path=source_verifier,
                    expected_source_verifier_sha256=source_verifier_sha256,
                )
            self.assertEqual(verified["state"], "CANDIDATE_BUILD_AUTHORIZED")
            self.assertEqual(
                verified["authorization"],
                {
                    "candidate_build": True,
                    "package_publication": False,
                    "git_tag": False,
                    "package_mutation_after_build": False,
                },
            )

            tampered = copy.deepcopy(receipt)
            tampered["authorization"]["package_publication"] = True
            unsigned = dict(tampered)
            unsigned.pop("receipt_id")
            tampered["receipt_id"] = RECEIPT._canonical_sha256(unsigned)
            receipt_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(
                RECEIPT.ReceiptError, "illegally grants package_publication"
            ):
                RECEIPT.verify_embedded(
                    receipt_path=receipt_path,
                    contract_path=CONTRACT,
                    schema_path=ROOT
                    / "packaging/v110-performance-inversion-record.schema.json",
                    inventory_path=ROOT
                    / "packaging/v110-performance-inversion-inventory.json",
                    require_candidate=True,
                )

            tampered = copy.deepcopy(receipt)
            tampered["m4_candidate_manifest"]["validation"]["no_cpu_fallback"] = False
            tampered["m4_candidate_manifest_record"]["payload_sha256"] = (
                RECEIPT._canonical_sha256(tampered["m4_candidate_manifest"])
            )
            unsigned = dict(tampered)
            unsigned.pop("receipt_id")
            tampered["receipt_id"] = RECEIPT._canonical_sha256(unsigned)
            receipt_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "validation summary"):
                RECEIPT.verify_embedded(
                    receipt_path=receipt_path,
                    contract_path=CONTRACT,
                    schema_path=ROOT
                    / "packaging/v110-performance-inversion-record.schema.json",
                    inventory_path=ROOT
                    / "packaging/v110-performance-inversion-inventory.json",
                    require_candidate=True,
                )

    def test_candidate_manifest_rejects_mixed_development_package_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest_path = candidate_manifest_fixture(root, "c" * 40)
            manifest = json.loads(manifest_path.read_text())
            role = "real_multi_gpu"
            record = manifest["evidence"][role]
            evidence_path = root / record["bundle_relative_path"]
            payload = json.loads(evidence_path.read_text())
            payload["package_receipt_id"] = "e" * 64
            evidence_path.write_text(json.dumps(payload, sort_keys=True))
            record.update(
                size_bytes=evidence_path.stat().st_size,
                sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                content_id=RECEIPT._canonical_sha256(payload),
            )
            attested_files = {
                (
                    str((root / item["bundle_relative_path"]).resolve()),
                    item["size_bytes"],
                    item["sha256"],
                )
                for item in manifest["evidence"].values()
            }
            with mock.patch.object(
                RECEIPT,
                "_replay_native_evidence",
                side_effect=lambda _role, _path, value, _commit, **_kwargs: (
                    RECEIPT._canonical_sha256(value)
                ),
            ):
                with self.assertRaisesRegex(
                    RECEIPT.ReceiptError,
                    "does not bind one development package receipt ID",
                ):
                    RECEIPT._validate_candidate_manifest(
                        manifest,
                        "c" * 40,
                        root,
                        attested_files,
                        root / "source-policy.json",
                        "1" * 64,
                        root / "source-verifier.py",
                        "2" * 64,
                    )

    def test_development_receipt_cannot_be_promoted_to_candidate(self):
        contract_result = VALIDATOR.validate(require_user_inputs=False)
        registry_result = REGISTRY.validate_registry(
            require_closed=False, evidence_root=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            contract_gate = root / "contract.json"
            registry_gate = root / "registry.json"
            contract_gate.write_text(json.dumps(contract_result))
            registry_gate.write_text(json.dumps(registry_result))
            receipt = RECEIPT.create(
                mode="development",
                source_commit="c" * 40,
                contract_gate_path=contract_gate,
                registry_gate_path=registry_gate,
            )
            receipt_path = root / "receipt.json"
            RECEIPT._write_new(receipt_path, receipt)
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
            verified = RECEIPT.verify(
                receipt_path=receipt_path,
                contract_path=CONTRACT,
                schema_path=ROOT
                / "packaging/v110-performance-inversion-record.schema.json",
                inventory_path=ROOT
                / "packaging/v110-performance-inversion-inventory.json",
                require_candidate=False,
            )
            self.assertEqual(verified["state"], "DEVELOPMENT_ONLY")
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "lacks candidate"):
                RECEIPT.verify(
                    receipt_path=receipt_path,
                    contract_path=CONTRACT,
                    schema_path=ROOT
                    / "packaging/v110-performance-inversion-record.schema.json",
                    inventory_path=ROOT
                    / "packaging/v110-performance-inversion-inventory.json",
                    require_candidate=True,
                )

    def test_candidate_evidence_rejects_arbitrary_and_self_hashed_pass_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            payload = {
                "schema": "attacker-chosen-pass-v1",
                "state": "PASS",
                "source_commit": "c" * 40,
            }
            record = _write_evidence(root, "targeted", payload)
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "producer schema"):
                RECEIPT._validate_artifact_ref(
                    record,
                    "targeted",
                    root,
                    set(),
                    RECEIPT.VALIDATED_ENGINE_COMMIT,
                    "c" * 40,
                    root / "unused-policy",
                    "0" * 64,
                    root / "unused-verifier",
                    "0" * 64,
                )

            for role in sorted(RECEIPT.TYPED_CANDIDATE_ROLES):
                with self.subTest(role=role):
                    fabricated = {
                        "schema": RECEIPT.ROLE_SCHEMAS[role],
                        "state": "PASS",
                        "source_commit": "c" * 40,
                        "claim": "no producer evidence",
                    }
                    fabricated_path = root / f"fabricated-{role}.json"
                    fabricated_path.write_text(json.dumps(fabricated, sort_keys=True))
                    with self.assertRaisesRegex(
                        RECEIPT.ReceiptError, "producer replay failed"
                    ):
                        RECEIPT._replay_native_evidence(
                            role, fabricated_path, fabricated, "c" * 40,
                            source_policy_path=None,
                            expected_source_policy_sha256=None,
                            source_verifier_path=None,
                            expected_source_verifier_sha256=None,
                        )

            payload["schema"] = RECEIPT.ROLE_SCHEMAS["targeted"]
            record = _write_evidence(root, "targeted", payload)
            self.assertEqual(
                record["content_id"], RECEIPT._canonical_sha256(payload)
            )
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "replayed attestation"):
                RECEIPT._validate_artifact_ref(
                    record,
                    "targeted",
                    root,
                    set(),
                    RECEIPT.VALIDATED_ENGINE_COMMIT,
                    "c" * 40,
                    root / "unused-policy",
                    "0" * 64,
                    root / "unused-verifier",
                    "0" * 64,
                )

    def test_release_source_transition_has_no_path_allowlist(self):
        self.assertFalse(hasattr(TRANSITION, "ALLOWED_EXACT"))
        self.assertFalse(hasattr(TRANSITION, "ALLOWED_PREFIXES"))
        self.assertFalse(hasattr(TRANSITION, "_allowed"))

    def test_source_replay_rejects_substituted_verifier_without_independent_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trusted_bytes = b"# independently audited verifier\n"
            malicious_verifier = root / "substituted-verifier.py"
            malicious_verifier.write_bytes(b"def verify(*args, **kwargs): return kwargs\n")
            policy_path = root / "external-policy.json"
            policy = {
                "verifier": {
                    "path": "scripts/verify-v110-release-source-transition.py",
                    "mode": "100644",
                    "blob": RECEIPT._git_blob_sha1(trusted_bytes),
                }
            }
            policy_path.write_text(json.dumps(policy, sort_keys=True))
            policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            payload = {
                "external_policy": {
                    "path": str(policy_path),
                    "mode": "0444",
                    "size_bytes": policy_path.stat().st_size,
                    "sha256": policy_sha,
                    "policy_id": "0" * 64,
                }
            }
            with self.assertRaisesRegex(RECEIPT.ReceiptError, "verifier SHA-256 differs"):
                RECEIPT._replay_native_evidence(
                    "m4_source_audit",
                    root / "forged-report.json",
                    payload,
                    "c" * 40,
                    source_policy_path=policy_path,
                    expected_source_policy_sha256=policy_sha,
                    source_verifier_path=malicious_verifier,
                    expected_source_verifier_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                )

    def test_candidate_cli_isolated_mode_does_not_execute_script_path_shadows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            receipt = scripts / "v110-release-gate-receipt.py"
            receipt.write_bytes((ROOT / "scripts/v110-release-gate-receipt.py").read_bytes())
            marker = root / "shadow-executed"
            payload = f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
            (scripts / "subprocess.py").write_text(payload)
            (scripts / "hashlib.py").write_text(payload)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(receipt),
                    "--create",
                    "--mode",
                    "candidate",
                    "--source-commit",
                    "c" * 40,
                ],
                cwd=scripts,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(scripts)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("lacks gate/output", completed.stderr)
            self.assertFalse(marker.exists())
            build_script = (ROOT / "scripts/build-conda-package.sh").read_text()
            publish_script = (ROOT / "scripts/publish-v110-release.sh").read_text()
            host_python = "/usr/bin/" + "python3"
            self.assertIn('"${V110_GATE_PYTHON}" -I "${V110_RECEIPT_TOOL}"', build_script)
            self.assertIn('"$FINAL_PYTHON" -I "$FINAL_TOOL"', publish_script)
            self.assertLess(publish_script.index('independently pinned source-transition preflight failed'),
                            publish_script.index('verify-v110-final-runtime.py'))
            self.assertIn(host_python + ' -I -S "${SOURCE_TRANSITION_VERIFIER}"', build_script)
            self.assertIn(host_python + ' -I -S "$SOURCE_TRANSITION_VERIFIER"', publish_script)

    def test_source_preflight_isolated_process_ignores_tracked_and_untracked_shadows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            subprocess.run(
                ["/usr/bin/git", "clone", "-q", "--shared", str(ROOT), str(repo)],
                check=True,
            )

            def git(*arguments: str) -> str:
                completed = subprocess.run(
                    ["/usr/bin/git", "-C", str(repo), *arguments],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                return completed.stdout.strip()

            git("config", "user.name", "M4 isolated test")
            git("config", "user.email", "m4-isolated@example.invalid")
            marker = root / "shadow-executed"
            shadow = (
                f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
                "raise RuntimeError('repository shadow imported')\n"
            )
            (repo / "scripts/v110-release-gate-receipt.py").write_bytes(
                (ROOT / "scripts/v110-release-gate-receipt.py").read_bytes()
            )
            (repo / "scripts/subprocess.py").write_text(shadow)
            git("add", "scripts/subprocess.py", "scripts/v110-release-gate-receipt.py")
            git("commit", "-qm", "fixture measured source with inert shadow")
            measured = git("rev-parse", "HEAD")
            release = measured

            external_verifier = root / "external-source-verifier.py"
            external_verifier.write_bytes(
                (repo / "scripts/verify-v110-release-source-transition.py").read_bytes()
            )
            external_verifier.chmod(0o555)
            policy = {
                "schema": TRANSITION.POLICY_SCHEMA,
                "decision": "APPROVE_EXACT_TRANSITION",
                "validated_engine_commit": TRANSITION.VALIDATED_ENGINE_COMMIT,
                "measured_source_commit": measured,
                "release_source_commit": release,
                "verifier": {
                    "path": "scripts/verify-v110-release-source-transition.py",
                    **TRANSITION._git_entry(
                        repo,
                        measured,
                        "scripts/verify-v110-release-source-transition.py",
                    ),
                },
                "approved_changes": [],
            }
            policy["policy_id"] = TRANSITION._canonical_sha256(policy)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy, sort_keys=True))
            policy_path.chmod(0o444)
            policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            report = TRANSITION.derive(repo, release, policy_path, policy_sha)
            evidence = root / "evidence"
            evidence.mkdir(mode=0o700)
            report_path = evidence / "source-transition.json"
            report_path.write_text(json.dumps(report, sort_keys=True))
            report_path.chmod(0o600)
            report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
            manifest = {
                "release_source_commit": release,
                "evidence": {
                    "m4_source_audit": {
                        "role": "m4_source_audit",
                        "bundle_relative_path": report_path.name,
                        "size_bytes": report_path.stat().st_size,
                        "sha256": report_sha,
                        "schema": TRANSITION.SCHEMA,
                        "content_id": report["report_id"],
                    }
                },
            }
            manifest_path = evidence / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            command = [
                sys.executable,
                "-I",
                str(repo / "scripts/v110-release-gate-receipt.py"),
                "--source-preflight-only",
                "--source-commit",
                release,
                "--m4-candidate-manifest",
                str(manifest_path),
                "--evidence-root",
                str(evidence),
                "--source-transition-policy",
                str(policy_path),
                "--expected-source-transition-policy-sha256",
                policy_sha,
                "--source-transition-verifier",
                str(external_verifier),
                "--expected-source-transition-verifier-sha256",
                hashlib.sha256(external_verifier.read_bytes()).hexdigest(),
            ]
            completed = subprocess.run(
                command,
                cwd=repo / "scripts",
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "scripts")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())

            # Preflight and normal candidate replay must agree on the exact
            # source report ID produced by the independently pinned verifier.
            compiler = load_script(
                "v110_source_reference_compiler",
                ROOT / "scripts/v110-release-evidence.py",
            )
            reference = compiler.artifact_reference(evidence, "m4_source_audit", report_path)
            self.assertEqual(reference["content_id"], report["report_id"])
            self.assertNotEqual(reference["content_id"], RECEIPT._canonical_sha256(report))
            cloned_gate = load_script(
                "v110_source_reference_cloned_gate",
                repo / "scripts/v110-release-gate-receipt.py",
            )
            self.assertIsNone(cloned_gate._validate_artifact_ref(
                reference, "m4_source_audit", evidence,
                {(str(report_path), report_path.stat().st_size, report_sha)},
                RECEIPT.VALIDATED_ENGINE_COMMIT, release,
                policy_path, policy_sha, external_verifier,
                hashlib.sha256(external_verifier.read_bytes()).hexdigest(),
            ))
            self.assertFalse(marker.exists())

            (repo / "scripts/hashlib.py").write_text(shadow)
            rejected = subprocess.run(
                command,
                cwd=repo / "scripts",
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "scripts")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("rejected the transition", rejected.stderr)
            self.assertFalse(marker.exists())

    def test_publish_preflights_source_before_every_isolated_asset_operation(self):
        script = (ROOT / "scripts/publish-v110-release.sh").read_text()
        host_python = "/usr/bin/" + "python3"
        preflight = script.index(host_python + ' -I -S "$SOURCE_TRANSITION_VERIFIER"')
        first_asset = script.index('"$ASSET_TOOL"', preflight)
        self.assertLess(preflight, first_asset)
        self.assertNotIn('"$CANDIDATE_TOOL" --source-preflight-only', script)
        invocation_lines = [
            line.strip()
            for line in script.splitlines()
            if '"$ASSET_TOOL" --' in line
        ]
        self.assertGreaterEqual(len(invocation_lines), 8)
        self.assertTrue(
            all(line.startswith(host_python + ' -I -S "$ASSET_TOOL"') for line in invocation_lines)
        )

    def test_source_and_public_publishers_are_repository_separated(self):
        integration = json.loads(CONTRACT.read_text())["gate_integration"]
        source = (ROOT / integration["publication_controller"]).read_text()
        public = (ROOT / integration["public_publication_controller"]).read_text()
        self.assertIn(
            '[[ "$REPOSITORY" == iverydj/gpmeep ]]',
            source,
        )
        self.assertIn('SLUG = "iverydj/gpmeep-releases"', public)
        self.assertIn("1372662716", public)
        self.assertIn("--audit-sha256", public)
        self.assertIn("public-parent-preserved-runtime-source-v1", public)
        self.assertTrue(integration["public_package_must_be_byte_identical"])
        self.assertTrue(integration["public_requires_independent_publication_audit"])
        self.assertNotIn("release_only_prior_migration_controller", integration)

    def test_authoritative_tree_disables_all_scheduled_dependabot_prs(self):
        policy = (ROOT / ".github/dependabot.yml").read_bytes()
        self.assertEqual(
            hashlib.sha256(policy).hexdigest(),
            VALIDATOR.EXPECTED_DEPENDABOT_POLICY_SHA256,
        )
        VALIDATOR.validate_dependabot_policy_bytes(policy)
        unicode_escape_bypass = policy + (
            b'\n  - "package-\\u0065cosystem": npm\n'
            b'    directory: "/"\n'
            b'    schedule:\n'
            b'      interval: daily\n'
        )
        with self.assertRaisesRegex(
            VALIDATOR.ContractError, "Dependabot policy bytes differ"
        ):
            VALIDATOR.validate_dependabot_policy_bytes(unicode_escape_bypass)

    def test_candidate_builder_runs_external_source_authority_before_repo_code(self):
        script = (ROOT / "scripts/build-conda-package.sh").read_text()
        host_python = "/usr/bin/" + "python3"
        external = script.index(
            host_python + ' -I -S "${SOURCE_TRANSITION_VERIFIER}"'
        )
        first_contract = script.index(
            host_python + ' -I "${V110_CONTRACT_VALIDATOR}" --contract-only'
        )
        environment_creation = script.index(
            '/bin/bash -p "${SCRIPT_DIR}/create-env.sh"'
        )
        self.assertLess(external, first_contract)
        self.assertLess(external, environment_creation)
        self.assertIn(
            '"${V110_GATE_PYTHON}" -I "${V110_CONTRACT_VALIDATOR}"', script
        )
        self.assertIn(
            '"${V110_GATE_PYTHON}" -I "${V110_REGISTRY_VALIDATOR}"', script
        )
        self.assertIn('"${V110_GATE_PYTHON}" -I "${V110_ATTESTER}"', script)

    def test_external_transition_policy_pins_paths_blobs_modes_and_verifier(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            repo.mkdir()

            def git(*arguments: str) -> str:
                completed = subprocess.run(
                    ["/usr/bin/git", "-C", str(repo), *arguments],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                return completed.stdout.strip()

            git("init", "-q")
            git("config", "user.name", "M4 test")
            git("config", "user.email", "m4@example.invalid")
            (repo / "src").mkdir()
            (repo / "scripts").mkdir()
            (repo / "packaging/conda").mkdir(parents=True)
            (repo / "src/step.cpp").write_text("engine\n")
            verifier = repo / "scripts/verify-v110-release-source-transition.py"
            verifier.write_text("#!/usr/bin/env python3\n")
            verifier.chmod(0o755)
            (repo / "packaging/conda/build.sh").write_text("#!/bin/sh\n")
            (repo / "README.md").write_text("base\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            release = base
            changes = []
            policy = {
                "schema": TRANSITION.POLICY_SCHEMA,
                "decision": "APPROVE_EXACT_TRANSITION",
                "validated_engine_commit": base,
                "measured_source_commit": base,
                "release_source_commit": release,
                "verifier": {
                    "path": "scripts/verify-v110-release-source-transition.py",
                    **TRANSITION._git_entry(
                        repo, release, "scripts/verify-v110-release-source-transition.py"
                    ),
                },
                "approved_changes": changes,
            }
            policy["policy_id"] = TRANSITION._canonical_sha256(policy)
            policy_path = root / "external-policy.json"
            policy_path.write_text(json.dumps(policy, sort_keys=True))
            policy_path.chmod(0o444)
            policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            with mock.patch.object(TRANSITION, "VALIDATED_ENGINE_COMMIT", base):
                report = TRANSITION.derive(repo, release, policy_path, policy_sha)
                self.assertEqual(report["changed_files"], [])
                self.assertEqual(report["external_policy"]["sha256"], policy_sha)

                (repo / "README.md").write_text("release docs\n")
                git("add", "README.md")
                git("commit", "-qm", "attempt post-measurement documentation change")
                changed_release = git("rev-parse", "HEAD")
                readme_changes = [{
                    "path": "README.md",
                    "measured_source": TRANSITION._git_entry(repo, base, "README.md"),
                    "release_source": TRANSITION._git_entry(repo, changed_release, "README.md"),
                }]
                changed_policy = dict(policy)
                changed_policy.update(
                    release_source_commit=changed_release,
                    approved_changes=readme_changes,
                )
                changed_policy.pop("policy_id", None)
                changed_policy["policy_id"] = TRANSITION._canonical_sha256(changed_policy)
                changed_policy_path = root / "changed-policy.json"
                changed_policy_path.write_text(json.dumps(changed_policy, sort_keys=True))
                changed_policy_path.chmod(0o444)
                changed_policy_sha = hashlib.sha256(changed_policy_path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(
                    TRANSITION.TransitionError,
                    "must exactly equal the M4 measured source commit",
                ):
                    TRANSITION.derive(
                        repo, changed_release, changed_policy_path, changed_policy_sha
                    )

                git("reset", "--hard", release)
                verifier.write_text("#!/usr/bin/env python3\n# weakened\n")
                (repo / "src/step.cpp").write_text("mutated engine\n")
                git("add", ".")
                git("commit", "-qm", "self weaken and mutate engine")
                attacker = git("rev-parse", "HEAD")
                with self.assertRaises(TRANSITION.TransitionError):
                    TRANSITION.derive(repo, attacker, policy_path, policy_sha)

                git("reset", "--hard", release)
                (repo / "packaging/conda/build.sh").write_text("#!/bin/sh\n# replace libmeep\n")
                git("add", ".")
                git("commit", "-qm", "replace package engine")
                attacker = git("rev-parse", "HEAD")
                build_changes = [{
                    "path": "packaging/conda/build.sh",
                    "measured_source": TRANSITION._git_entry(repo, release, "packaging/conda/build.sh"),
                    "release_source": TRANSITION._git_entry(repo, attacker, "packaging/conda/build.sh"),
                }]
                attacker_policy = dict(policy)
                attacker_policy.update(
                    measured_source_commit=release,
                    release_source_commit=attacker,
                    verifier={
                        "path": "scripts/verify-v110-release-source-transition.py",
                        **TRANSITION._git_entry(repo, release, "scripts/verify-v110-release-source-transition.py"),
                    },
                    approved_changes=build_changes,
                )
                attacker_policy.pop("policy_id", None)
                attacker_policy["policy_id"] = TRANSITION._canonical_sha256(attacker_policy)
                attacker_policy_path = root / "attacker-policy.json"
                attacker_policy_path.write_text(json.dumps(attacker_policy, sort_keys=True))
                attacker_policy_path.chmod(0o444)
                attacker_policy_sha = hashlib.sha256(attacker_policy_path.read_bytes()).hexdigest()
                with self.assertRaises(TRANSITION.TransitionError):
                    TRANSITION.derive(repo, attacker, attacker_policy_path, attacker_policy_sha)

    def test_glibc_floor_and_milestone_audits_cannot_be_relaxed(self):
        self.assert_rejected(
            lambda value: value["compatibility_target"].__setitem__(
                "maximum_glibc_floor", "2.38"
            )
        )
        self.assert_rejected(
            lambda value: value["validation"].__setitem__(
                "adversarial_audit_after_every_milestone", False
            )
        )
        self.assert_rejected(
            lambda value: value["validation"].__setitem__(
                "performance_repetitions", 1
            )
        )
        self.assert_rejected(
            lambda value: value["validation"]["user_input_artifacts"][0].__setitem__(
                "release_gate_required", True
            )
        )
        self.assert_rejected(
            lambda value: value["validation"]["user_input_artifacts"][1].__setitem__(
                "release_gate_required", False
            )
        )

    def test_curl_phase_replay_switch_preserves_default_and_fails_closed(self):
        source = (ROOT / "src/gpu_backend.cpp").read_text(encoding="utf-8")
        start = source.index("bool resident_curl_phase_batch::replay_if_ready()")
        end = source.index("\nvoid resident_curl_phase_batch::finish", start)
        implementation = source[start:end]
        self.assertIn(
            'std::getenv("MEEP_GPU_ENABLE_CURL_PHASE_REPLAY")', implementation
        )
        self.assertIn(
            'std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") != nullptr',
            implementation,
        )
        self.assertIn(
            '(explicit_enable && std::strcmp(explicit_enable, "1") != 0)',
            implementation,
        )


class V110ExternalEvidenceContractTest(unittest.TestCase):
    """Checks that require the checksum-bound workspace inputs outside Git."""

    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_repository_contract_documents_bound_files_and_inputs_pass(self):
        result = VALIDATOR.validate()
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["base_commit"], VALIDATOR.EXPECTED_BASE)
        self.assertEqual(result["user_inputs_checked"], 2)
        self.assertRegex(result["contract_sha256"], r"^[0-9a-f]{64}$")

    def test_actual_user_inputs_and_archive_members_are_byte_bound(self):
        records = VALIDATOR.validate_user_inputs(self.contract, ROOT.parent)
        self.assertEqual(
            [item["id"] for item in records],
            ["hor_tm_60deg_saw_v6", "aunp_r4000_repro_20260807"],
        )
        self.assertEqual(len(records[1]["members"]), 4)
        candidate = copy.deepcopy(self.contract)
        candidate["validation"]["user_input_artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaises(VALIDATOR.ContractError):
            VALIDATOR.validate_user_inputs(candidate, ROOT.parent)


if __name__ == "__main__":
    unittest.main()
