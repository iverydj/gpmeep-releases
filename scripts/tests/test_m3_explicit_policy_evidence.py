#!/usr/bin/env python3
"""Focused tests for feature-specific explicit-backend M3 evidence."""

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
SOURCE = (
    ROOT
    / "scripts"
    / "python-validation"
    / "m3_explicit_policy_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m3_explicit_policy_evidence", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


REASON = POLICY.paired_matrix.BACKEND_ACKNOWLEDGEMENT_REASON


def guard(
    selected: str, recommended: str, *, reused: bool = False
) -> dict[str, object]:
    mismatch = selected != recommended
    exact = mismatch and not reused
    warning = {
        "schema": POLICY.backend_guard.TELEMETRY_SCHEMA,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "reason_class": "hardware_crossover",
        "recommendation_confidence": "high",
        "rank_local_cells": [600000],
        "run_length": {
            "kind": "until",
            "numeric_duration": 20.0,
            "source_end_time": None,
            "expected_timesteps": 400,
            "dynamic_condition": False,
        },
        "kernel_launch_count": 1600,
        "transfer_bytes": 614400000,
        "monitor_count": 1,
        "monitor_frequency_count": 100,
        "source_count": 1,
        "mpi_gpu_topology": [
            {
                "rank": 0,
                "global_rank": 0,
                "active_size": 1,
                "node_rank": 0,
                "node_size": 1,
                "device_identifier": "fixture-device",
                "device_ordinal": 0,
            }
        ],
        "profiler_or_ablation_id": POLICY.backend_guard.PROFILER_OR_ABLATION_ID,
        "acknowledgement": False,
        "acknowledgement_allowed": True,
        "acknowledgement_reason": "",
        "acknowledgement_reused": reused,
        "acknowledgement_scope": "",
        "acknowledgement_origin_cohort_digest": "",
        "acknowledgement_scope_cache_enabled": True,
        "acknowledgement_scope_digest": "6" * 64,
        "workload_fingerprint_evaluated": exact,
        "workload_fingerprint": "1" * 64 if exact else "",
        "recommendation_policy_version": POLICY.backend_guard.POLICY_VERSION,
        "mpi_topology_digest": "2" * 64,
        "native_state_digest": "3" * 64 if exact else "",
        "process_nonce": "4" * 64 if exact else "",
        "run_nonce": "5" * 64 if exact else "",
        "acknowledgement_digest": "",
    }
    if exact:
        warning["acknowledgement_digest"] = (
            POLICY.backend_guard._acknowledgement_digest(warning)
        )
    final = copy.deepcopy(warning)
    final["acknowledgement"] = mismatch
    final["acknowledgement_reason"] = REASON if mismatch else ""
    final["acknowledgement_scope"] = "simulation" if mismatch else ""
    final["acknowledgement_origin_cohort_digest"] = (
        "7" * 64
        if mismatch and reused
        else final["acknowledgement_digest"] if mismatch else ""
    )
    return {
        "schema": POLICY.backend_guard.SCHEMA,
        "attempts": 2 if exact else 1,
        "acknowledgements": 1 if exact else 0,
        "acknowledgement_reused": reused,
        "mutation_before_acknowledgement": False,
        "warning_wall_seconds": 0.1 if exact else 0.0,
        "execution_wall_seconds": 1.0,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "mismatch_observed": mismatch,
        "warning_telemetry": warning if exact else None,
        "final_telemetry": final,
    }


class ExplicitPolicyEvidenceTests(unittest.TestCase):
    def test_reused_policy_scope_is_projected_without_a_second_warning(self):
        summary = POLICY.summarize_lane(
            [
                guard("cuda", "cpu"),
                guard("cuda", "cpu", reused=True),
            ],
            selected_backend="cuda",
            reason=REASON,
            source="fixture",
        )
        self.assertEqual(2, summary["event_count"])
        self.assertEqual(3, summary["attempt_count"])
        self.assertEqual(1, summary["acknowledgement_count"])
        first, reused = summary["events"]
        self.assertTrue(first["warning_present"])
        self.assertFalse(first["acknowledgement_reused"])
        self.assertTrue(first["workload_fingerprint_evaluated"])
        self.assertFalse(reused["warning_present"])
        self.assertTrue(reused["acknowledgement_reused"])
        self.assertFalse(reused["workload_fingerprint_evaluated"])
        self.assertEqual("", reused["workload_fingerprint"])

    def test_candidate_git_commits_must_match_exactly(self):
        receipt = "a" * 64
        source = "b" * 64
        paired = {
            "contract": {
                "build_receipt": {
                    "receipt_id": receipt,
                    "source_sha256": source,
                    "git_head": "c" * 40,
                }
            }
        }
        aunp = {
            "build": {"receipt_id": receipt, "git_head": "d" * 40},
            "source_snapshot_sha256": source,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            paired_root = root / "paired"
            aunp_root = root / "aunp"
            paired_root.mkdir()
            aunp_root.mkdir()
            with mock.patch.object(
                POLICY.paired_matrix, "verify_complete", return_value=paired
            ), mock.patch.object(
                POLICY.aunp_matrix, "verify_hybrid_complete", return_value=aunp
            ), self.assertRaisesRegex(Exception, "Git identity"):
                POLICY.derive(paired_root, aunp_root)

    def test_both_recommendation_directions_are_exact(self):
        cpu_matching = POLICY.summarize_lane(
            [guard("cpu", "cpu")],
            selected_backend="cpu",
            reason=REASON,
            source="fixture",
        )
        cuda_mismatch = POLICY.summarize_lane(
            [guard("cuda", "cpu")],
            selected_backend="cuda",
            reason=REASON,
            source="fixture",
        )
        cuda_matching = POLICY.summarize_lane(
            [guard("cuda", "cuda")],
            selected_backend="cuda",
            reason=REASON,
            source="fixture",
        )
        cpu_mismatch = POLICY.summarize_lane(
            [guard("cpu", "cuda")],
            selected_backend="cpu",
            reason=REASON,
            source="fixture",
        )
        self.assertEqual(cpu_matching["acknowledgement_count"], 0)
        self.assertEqual(cuda_mismatch["acknowledgement_count"], 1)
        self.assertEqual(cuda_matching["acknowledgement_count"], 0)
        self.assertEqual(cpu_mismatch["acknowledgement_count"], 1)
        self.assertEqual(cpu_mismatch["events"][0]["event_index"], 0)
        self.assertEqual(
            cpu_mismatch["events"][0]["direction"],
            "cpu_selected__cuda_recommended",
        )
        aggregate = POLICY.aggregate_directions(
            [cpu_matching, cuda_mismatch, cuda_matching, cpu_mismatch]
        )
        self.assertTrue(
            aggregate["all_selected_recommended_directions_present"]
        )
        self.assertEqual(aggregate["total_event_count"], 4)

    def test_event_projection_mutations_fail_closed(self):
        lanes = [
            POLICY.summarize_lane(
                [guard(selected, recommended)],
                selected_backend=selected,
                reason=REASON,
                source="fixture",
            )
            for selected, recommended in (
                ("cpu", "cpu"),
                ("cpu", "cuda"),
                ("cuda", "cpu"),
                ("cuda", "cuda"),
            )
        ]
        mutations = {
            "selected": lambda event: event.update(selected_backend="cuda"),
            "recommended": lambda event: event.update(
                recommended_backend="cuda"
            ),
            "direction": lambda event: event.update(direction="wrong"),
            "attempts": lambda event: event.update(attempts=2),
            "acknowledgements": lambda event: event.update(
                acknowledgements=1
            ),
            "warning": lambda event: event.update(warning_present=True),
            "reason": lambda event: event.update(
                acknowledgement_reason="unexpected reason"
            ),
            "policy": lambda event: event.update(
                recommendation_policy_version="wrong"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(lanes)
                mutate(changed[0]["events"][0])
                with self.assertRaisesRegex(Exception, "event|direction"):
                    POLICY.aggregate_directions(changed)

        changed = copy.deepcopy(lanes)
        changed[0]["events"].append(copy.deepcopy(changed[0]["events"][0]))
        changed[0]["event_count"] = 2
        with self.assertRaisesRegex(Exception, "event|direction"):
            POLICY.aggregate_directions(changed)

    def test_environment_reason_cannot_describe_a_mismatch(self):
        lane = POLICY.summarize_lane(
            [guard("cuda", "cpu")],
            selected_backend="cuda",
            reason=REASON,
            source="fixture",
        )
        event = lane["events"][0]
        event["reason_class"] = "environment"
        event["acknowledgement_allowed"] = False
        with self.assertRaisesRegex(Exception, "environment event"):
            POLICY._validate_event_view(
                event, expected_index=0, expected_selected="cuda"
            )

    def test_mixed_event_recommendations_are_classified_independently(self):
        summary = POLICY.summarize_lane(
            [guard("cpu", "cpu"), guard("cpu", "cuda")],
            selected_backend="cpu",
            reason=REASON,
            source="fixture",
        )
        self.assertEqual(summary["case_level_recommendation"], None)
        self.assertEqual(
            summary["direction_counts"]["cpu_selected__cpu_recommended"], 1
        )
        self.assertEqual(
            summary["direction_counts"]["cpu_selected__cuda_recommended"], 1
        )

    def test_attempt_tampering_fails_closed(self):
        tampered = guard("cuda", "cpu")
        tampered["attempts"] = 3
        with self.assertRaisesRegex(Exception, "guard event"):
            POLICY.summarize_lane(
                [tampered],
                selected_backend="cuda",
                reason=REASON,
                source="fixture",
            )

    def test_runner_lane_extracts_site_and_oracle_events(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            stdout = root / "stdout.log"
            stdout.write_text(
                POLICY.oracle.AUDIT_PREFIX
                + json.dumps(
                    {"runs": [{"explicit_backend_guard": guard("cpu", "cpu")}]}
                )
                + "\n",
                encoding="utf-8",
            )
            source, events = POLICY._runner_lane_events(
                root,
                {
                    "backend": "cpu",
                    "stdout_log": "stdout.log",
                    "statistics": {
                        "backend_guard": {
                            "mode": "disabled",
                            "events": [],
                        }
                    },
                },
            )
            self.assertEqual(source, "generic-oracle")
            self.assertEqual(events, [guard("cpu", "cpu")])

            auxiliary = guard("cuda", "cpu")
            observed = guard("cuda", "cpu")
            stdout.write_text(
                POLICY.oracle.AUXILIARY_GUARD_PREFIX
                + json.dumps(
                    {
                        "events": [
                            {"entrypoint": "solve_cw", "guard": auxiliary}
                        ]
                    }
                )
                + "\n"
                + POLICY.oracle.AUDIT_PREFIX
                + json.dumps({"runs": [{"explicit_backend_guard": observed}]})
                + "\n",
                encoding="utf-8",
            )
            source, events = POLICY._runner_lane_events(
                root,
                {
                    "backend": "cuda",
                    "stdout_log": "stdout.log",
                    "statistics": {
                        "backend_guard": {"mode": "disabled", "events": []}
                    },
                },
            )
            self.assertEqual(source, "generic-oracle")
            self.assertEqual(events, [auxiliary, observed])

            source, events = POLICY._runner_lane_events(
                root,
                {
                    "backend": "cuda",
                    "stdout_log": "stdout.log",
                    "statistics": {
                        "backend_guard": {
                            "mode": "reason-bound-policy-scope",
                            "events": [guard("cuda", "cpu")],
                        }
                    },
                },
            )
            self.assertEqual(source, "sitecustomize")
            self.assertEqual(events, [guard("cuda", "cpu")])

    def test_feature_inventory_contains_no_backend_switching_authority(self):
        self.assertEqual(
            POLICY.EXPECTED_FEATURES,
            (
                "policy.recommend_cpu",
                "policy.recommend_cuda",
                "policy.long_horizon_small",
            ),
        )
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"automatic_backend_selection": False', source)
        self.assertIn('"case_level_recommendations_used": False', source)


if __name__ == "__main__":
    unittest.main()
