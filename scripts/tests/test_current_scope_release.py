#!/usr/bin/env python3
"""Contract tests for the user-revised release aggregator."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_current_scope_release.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_current_scope_release", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RELEASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RELEASE
SPEC.loader.exec_module(RELEASE)

GPU0 = "GPU-00000000-0000-0000-0000-000000009001"
GPU1 = "GPU-00000000-0000-0000-0000-000000009002"


def performance_report() -> dict:
    roles = {
        "cpu-fp32-to-one-gpu": ("cpu8", "gpu1", [4.0, 6.0], [2.0, 3.0], 1.5),
        "cpu-fp32-to-two-gpu": ("cpu8", "gpu2", [4.0, 6.0], [1.5, 2.5], 2.0),
        "one-gpu-to-two-gpu": ("gpu1", "gpu2", [2.0, 3.0], [1.5, 2.5], 1.1),
    }
    timing = {}
    for name, (reference_role, candidate_role, reference, candidate, minimum) in roles.items():
        reference_mean = sum(reference) / 2
        candidate_mean = sum(candidate) / 2
        timing[name] = {
            "reference_role": reference_role,
            "candidate_role": candidate_role,
            "reference_samples_seconds": reference,
            "candidate_samples_seconds": candidate,
            "reference_mean_seconds": reference_mean,
            "candidate_mean_seconds": candidate_mean,
            "mean_speedup": reference_mean / candidate_mean,
            "minimum": minimum,
            "outcome": "PASS",
        }
    diagnostics = {
        role: {
            "sample_count": 2,
            "samples_seconds": samples,
            "arithmetic_mean_seconds": sum(samples) / 2,
            "population_stddev_seconds": 1.0,
            "coefficient_of_variation": 0.2,
            "release_gate": None,
        }
        for role, samples in {
            "performance-cpu-fp32-8r": [4.0, 6.0],
            "performance-cuda-fp32-1g": [2.0, 3.0],
            "performance-cuda-fp32-2g": [1.5, 2.5],
        }.items()
    }
    return {
        "outcome": "PASS",
        "performance_repeats": 2,
        "performance": {
            "fdtd_wall_seconds": timing,
            "workload_end_to_end_seconds": timing,
        },
        "timing_diagnostics": {
            "fdtd_wall_seconds": diagnostics,
            "workload_end_to_end_seconds": diagnostics,
        },
    }


def component_report(
    kind: str, count: int, receipt: str, source: str, git_head: str = "a" * 40
) -> dict:
    contract = {
        "build_receipt": {
            "receipt_id": receipt,
            "source_sha256": source,
            "git_head": git_head,
        },
        "gpu_devices": [GPU0, GPU1],
        "gpu_device": GPU0,
    }
    return {
        "outcome": "PASS",
        "contract": contract,
        "tasks": [
            {"task": {"unit_id": f"{kind}-{index:02d}"}}
            for index in range(1, count + 1)
        ],
    }


def explicit_policy_report(
    receipt: str,
    source: str,
    git_head: str = "a" * 40,
    terminal: dict | None = None,
) -> dict:
    terminal = {"path": "terminal"} if terminal is None else terminal
    directions = RELEASE.explicit_policy_evidence.DIRECTIONS

    def lane(selected: str, recommended: str, nonce: int) -> dict:
        mismatch = selected != recommended
        direction = f"{selected}_selected__{recommended}_recommended"
        event = {
            "event_index": 0,
            "selected_backend": selected,
            "recommended_backend": recommended,
            "direction": direction,
            "reason_class": "hardware_crossover",
            "recommendation_confidence": "high",
            "mismatch_observed": mismatch,
            "attempts": 2 if mismatch else 1,
            "acknowledgements": 1 if mismatch else 0,
            "acknowledgement_reused": False,
            "acknowledgement_scope": "simulation" if mismatch else "",
            "acknowledgement_origin_cohort_digest": (
                f"{nonce + 300:064x}" if mismatch else ""
            ),
            "acknowledgement_allowed": True,
            "acknowledgement_reason": (
                "fixture explicit backend acknowledgement" if mismatch else ""
            ),
            "warning_present": mismatch,
            "final_acknowledgement": mismatch,
            "acknowledgement_scope_digest": f"{nonce + 200:064x}",
            "workload_fingerprint_evaluated": mismatch,
            "workload_fingerprint": f"{nonce:064x}" if mismatch else "",
            "recommendation_policy_version": (
                RELEASE.explicit_policy_evidence.backend_guard.POLICY_VERSION
            ),
        }
        counts = {item: int(item == direction) for item in directions}
        return {
            "source": "fixture",
            "selected_backend": selected,
            "classification_unit": "guard-event",
            "case_level_recommendation": None,
            "event_count": 1,
            "attempt_count": event["attempts"],
            "acknowledgement_count": event["acknowledgements"],
            "mutation_before_acknowledgement": False,
            "events_sha256": f"{nonce + 100:064x}",
            "event_views_sha256": (
                RELEASE.explicit_policy_evidence._canonical_sha256([event])
            ),
            "events": [event],
            "direction_counts": counts,
            "reason_class_counts": {"hardware_crossover": 1},
            "execution_wall_seconds": [1.0],
            "warning_wall_seconds": [0.1 if mismatch else 0.0],
        }

    nonce = 1

    def cases(feature_id: str, paths: tuple[str, ...]) -> list[dict]:
        nonlocal nonce
        records = []
        for index, path in enumerate(paths):
            cpu_recommended = "cpu" if index % 2 == 0 else "cuda"
            cuda_recommended = "cuda" if index % 2 == 0 else "cpu"
            records.append(
                {
                    "case_path": path,
                    "coverage_feature": feature_id,
                    "classification_unit": "guard-event",
                    "case_level_recommendation": None,
                    "lanes": {
                        "cpu": lane("cpu", cpu_recommended, nonce),
                        "cuda": lane("cuda", cuda_recommended, nonce + 1),
                    },
                }
            )
            nonce += 2
        return records

    features = {
        "policy.recommend_cpu": {
            "outcome": "PASS",
            "cases": cases(
                "policy.recommend_cpu",
                RELEASE.explicit_policy_evidence.CPU_RECOMMENDATION_CASES,
            ),
        },
        "policy.recommend_cuda": {
            "outcome": "PASS",
            "cases": cases(
                "policy.recommend_cuda",
                RELEASE.explicit_policy_evidence.CUDA_RECOMMENDATION_CASES,
            ),
        },
        "policy.long_horizon_small": {
            "outcome": "PASS",
            "cases": cases(
                "policy.long_horizon_small",
                RELEASE.explicit_policy_evidence.LONG_HORIZON_CASES,
            ),
        },
    }
    aunp_lanes = {"cpu": [], "cuda": []}
    for selected, count in (("cpu", 2), ("cuda", 4)):
        for index in range(count):
            recommended = selected if index % 2 == 0 else (
                "cuda" if selected == "cpu" else "cpu"
            )
            aunp_lanes[selected].append(lane(selected, recommended, nonce))
            nonce += 1
    features["policy.recommend_cuda"]["aunp"] = {
        "classification_unit": "guard-event",
        "case_level_recommendation": None,
        "performance_task_count": 6,
        "lanes": aunp_lanes,
    }
    report = {
        "outcome": "PASS",
        "receipt_id": receipt,
        "source_sha256": source,
        "git_head": git_head,
        "backend_selection": "user-explicit-only",
        "automatic_backend_selection": False,
        "classification_unit": "guard-event",
        "case_level_recommendations_used": False,
        "features": features,
        "inputs": {
            "paired_complete": terminal,
            "aunp_complete": terminal,
        },
    }
    all_lanes = [
        lane_record
        for feature in features.values()
        for case in feature["cases"]
        for lane_record in case["lanes"].values()
    ] + aunp_lanes["cpu"] + aunp_lanes["cuda"]
    report["aggregate"] = (
        RELEASE.explicit_policy_evidence.aggregate_directions(all_lanes)
    )
    return report


class CurrentScopeReleaseTests(unittest.TestCase):
    def test_explicit_policy_contract_survives_sorted_json_round_trip(self):
        report = explicit_policy_report("receipt", "1" * 64)
        retained = json.loads(json.dumps(report, sort_keys=True))
        self.assertNotEqual(
            tuple(retained["features"]),
            RELEASE.explicit_policy_evidence.EXPECTED_FEATURES,
        )
        projection = (
            RELEASE.explicit_policy_evidence.policy_contract_projection(
                retained
            )
        )
        self.assertEqual(projection["aggregate"], report["aggregate"])

    def test_aunp_performance_is_exactly_two_run_arithmetic_mean(self) -> None:
        value = performance_report()
        result = RELEASE.validate_aunp_performance(value)
        self.assertEqual(result["measured_runs_per_topology"], 2)
        self.assertEqual(result["aggregation"], "arithmetic_mean")
        self.assertEqual(result["dispersion_role"], "diagnostic_only")

        value["performance"]["fdtd_wall_seconds"]["cpu-fp32-to-one-gpu"][
            "reference_samples_seconds"
        ].append(8.0)
        with self.assertRaisesRegex(Exception, "exactly two"):
            RELEASE.validate_aunp_performance(value)

    def test_median_or_conservative_speedup_cannot_enter_release_metric(self) -> None:
        value = performance_report()
        metric = value["performance"]["fdtd_wall_seconds"][
            "cpu-fp32-to-one-gpu"
        ]
        metric["median_speedup"] = metric["mean_speedup"]
        with self.assertRaisesRegex(Exception, "schema differs"):
            RELEASE.validate_aunp_performance(value)

    def test_feature_closure_requires_a_passing_route_for_every_feature(self) -> None:
        features = []
        for index in range(39):
            features.append(
                {
                    "feature_id": f"family.feature_{index}",
                    "family": "family",
                    "status": "planned",
                    "primary_routes": [
                        {
                            "evidence": f"python/example-{index}.py",
                            "routes": [
                                {"kind": "paired", "unit_id": "paired-01"}
                            ],
                        }
                    ],
                }
            )
        features.extend(
            {
                "feature_id": feature_id,
                "family": "distributed_policy",
                "status": "planned",
                "primary_routes": [
                    {
                        "evidence": "python/example-policy.py",
                        "routes": [{"kind": "paired", "unit_id": "paired-01"}],
                    }
                ],
            }
            for feature_id in RELEASE.explicit_policy_evidence.EXPECTED_FEATURES
        )
        policy = explicit_policy_report("receipt", "1" * 64)
        closure = RELEASE.feature_closure(
            {"features": features},
            {"paired": {"paired-01"}},
            set(),
            policy,
        )
        self.assertEqual(len(closure), 42)
        self.assertEqual(
            closure[-1]["evidence"][0]["component"], "explicit_policy"
        )
        features[0]["primary_routes"][0]["routes"][0]["unit_id"] = "missing"
        with self.assertRaisesRegex(Exception, "did not pass"):
            RELEASE.feature_closure(
                {"features": features},
                {"paired": {"paired-01"}},
                set(),
                policy,
            )
        policy["automatic_backend_selection"] = True
        with self.assertRaisesRegex(Exception, "explicit.*policy"):
            RELEASE.feature_closure(
                {"features": features},
                {"paired": {"paired-01"}},
                set(),
                policy,
            )
        policy = explicit_policy_report("receipt", "1" * 64)
        for label, mutate in {
            "classification": lambda value: value.update(
                classification_unit="case"
            ),
            "case-level": lambda value: value.update(
                case_level_recommendations_used=True
            ),
            "aggregate": lambda value: value["aggregate"][
                "direction_counts"
            ].update(cpu_selected__cpu_recommended=0),
            "event": lambda value: value["features"][
                "policy.recommend_cpu"
            ]["cases"][0]["lanes"]["cpu"]["events"][0].update(
                recommended_backend="cuda"
            ),
        }.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(policy)
                mutate(changed)
                with self.assertRaisesRegex(Exception, "explicit-policy"):
                    RELEASE.feature_closure(
                        {"features": features},
                        {"paired": {"paired-01"}},
                        set(),
                        changed,
                    )

    def test_build_closure_requires_identical_receipt_source_and_gpu_identity(self) -> None:
        source = "1" * 64
        receipt = "2" * 64
        aunp = {
            "build": {"receipt_id": receipt, "git_head": "a" * 40},
            "source_snapshot_sha256": source,
            "gpu_devices": [GPU0, GPU1],
            "cpu_physical_cores": 8,
        }
        components = {
            name: component_report(name, count, receipt, source)
            for name, count in RELEASE.EXPECTED_COMPONENT_COUNTS.items()
        }
        closure = RELEASE._build_hardware_closure(aunp, components, "a" * 40)
        self.assertTrue(closure["shared_identical_fp32_build"])
        components["specialized"]["contract"]["gpu_devices"].reverse()
        with self.assertRaisesRegex(Exception, "GPU identity"):
            RELEASE._build_hardware_closure(aunp, components, "a" * 40)
        with self.assertRaisesRegex(Exception, "build or hardware"):
            RELEASE._build_hardware_closure(aunp, components, "b" * 40)

    def test_derive_has_only_aunp_performance_and_79_example_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            repo.mkdir()
            roots = {}
            for name in ("aunp", "paired", "host", "specialized", "explicit_policy"):
                roots[name] = root / name
                roots[name].mkdir()
            (roots["explicit_policy"] / "report.json").write_text("{}\n")
            plan_receipt = root / "plan.json"
            plan_receipt.write_text("{}\n")
            audit_receipt = root / "audit.json"
            audit_receipt.write_text("{}\n")
            deferral = root / "deferral.json"
            deferral.write_text("{}\n")
            source = "1" * 64
            receipt = "2" * 64
            aunp = performance_report()
            aunp.update(
                {
                    "build": {
                        "receipt_id": receipt,
                        "git_head": "a" * 40,
                    },
                    "source_snapshot_sha256": source,
                    "gpu_devices": [GPU0, GPU1],
                    "cpu_physical_cores": 8,
                    "exact_comparisons": [{}, {}],
                }
            )
            components = {
                name: component_report(name, count, receipt, source)
                for name, count in RELEASE.EXPECTED_COMPONENT_COUNTS.items()
            }
            first_unit = components["paired"]["tasks"][0]["task"]["unit_id"]
            features = [
                {
                    "feature_id": f"family.feature_{index}",
                    "family": "family",
                    "status": "planned",
                    "primary_routes": [
                        {
                            "evidence": f"python/example-{index}.py",
                            "routes": [{"kind": "paired", "unit_id": first_unit}],
                        }
                    ],
                }
                for index in range(39)
            ]
            features.extend(
                {
                    "feature_id": feature_id,
                    "family": "distributed_policy",
                    "status": "planned",
                    "primary_routes": [
                        {
                            "evidence": "python/example-policy.py",
                            "routes": [{"kind": "paired", "unit_id": first_unit}],
                        }
                    ],
                }
                for feature_id in RELEASE.explicit_policy_evidence.EXPECTED_FEATURES
            )
            plan = {
                "repository": str(repo),
                "git_commit": "a" * 40,
                "git_status_porcelain": "",
                "plan": {"features": features},
            }
            policy = explicit_policy_report(receipt, source)
            contract = {
                "repo": str(repo),
                "plan_replay": str(plan_receipt),
                "hor_deferral": str(deferral),
                "aunp_output": str(roots["aunp"]),
                "aunp_audit": str(audit_receipt),
                "explicit_policy_output": str(roots["explicit_policy"]),
                "paired_output": str(roots["paired"]),
                "host_output": str(roots["host"]),
                "specialized_output": str(roots["specialized"]),
            }
            with mock.patch.object(
                RELEASE.plan_replay_seal, "verify", return_value=plan
            ), mock.patch.object(
                RELEASE, "validate_hor_deferral", return_value={"outcome": "deferred"}
            ), mock.patch.object(
                RELEASE.aunp_matrix, "verify_hybrid_complete", return_value=aunp
            ), mock.patch.object(
                RELEASE.explicit_policy_evidence,
                "verify",
                return_value=policy,
            ), mock.patch.object(
                RELEASE.paired_matrix, "verify_complete", return_value=components["paired"]
            ), mock.patch.object(
                RELEASE.host_matrix, "verify_complete", return_value=components["host"]
            ), mock.patch.object(
                RELEASE.specialized_matrix,
                "verify_complete",
                return_value=components["specialized"],
            ), mock.patch.object(
                RELEASE, "_audit_summary", return_value={"outcome": "PASS"}
            ), mock.patch.object(
                RELEASE, "_component_terminal", return_value={"path": "terminal"}
            ):
                report = RELEASE.derive(contract)
                policy["inputs"]["paired_complete"] = {"path": "other-terminal"}
                with self.assertRaisesRegex(Exception, "policy build/source/input"):
                    RELEASE.derive(contract)
            self.assertEqual(report["official_meep_examples"]["total_physical_units"], 79)
            self.assertFalse(
                report["official_meep_examples"]["performance_release_gate"]
            )
            self.assertEqual(set(report["aunp"]["performance"]["metrics"]), RELEASE.TIMING_FIELDS)


if __name__ == "__main__":
    unittest.main()
