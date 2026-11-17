"""Exact v1.1 backend-guard telemetry builders for unit tests."""

from __future__ import annotations

import copy
from typing import Any


def telemetry(
    contract: Any,
    selected: str,
    recommended: str,
    *,
    acknowledgement: bool = False,
    reason: str = "",
    ranks: int = 1,
    reused: bool = False,
    acknowledgement_scope: str = "process",
) -> dict[str, Any]:
    mismatch = selected != recommended
    exact = (mismatch and not reused) or (
        ranks > 1 and (not mismatch or acknowledgement_scope == "process")
    )
    result: dict[str, Any] = {
        "schema": contract.TELEMETRY_SCHEMA,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "reason_class": "hardware_crossover",
        "recommendation_confidence": "high",
        "rank_local_cells": [600000] * ranks,
        "run_length": {
            "kind": "until",
            "numeric_duration": 20.0,
            "source_end_time": None,
            "expected_timesteps": 400,
            "dynamic_condition": False,
        },
        "kernel_launch_count": 1600,
        "transfer_bytes": 614400000 * ranks,
        "monitor_count": 1,
        "monitor_frequency_count": 100,
        "source_count": 1,
        "mpi_gpu_topology": [
            {
                "rank": rank,
                "global_rank": rank,
                "active_size": ranks,
                "node_rank": rank,
                "node_size": ranks,
                "device_identifier": f"fixture-device-{rank}",
                "device_ordinal": rank,
            }
            for rank in range(ranks)
        ],
        "profiler_or_ablation_id": contract.PROFILER_OR_ABLATION_ID,
        "acknowledgement": acknowledgement,
        "acknowledgement_allowed": True,
        "acknowledgement_reason": reason if acknowledgement else "",
        "acknowledgement_reused": reused,
        "acknowledgement_scope": (
            acknowledgement_scope if acknowledgement else ""
        ),
        "acknowledgement_origin_cohort_digest": (
            "7" * 64 if acknowledgement and reused else ""
        ),
        "acknowledgement_scope_cache_enabled": True,
        "acknowledgement_scope_digest": "6" * 64,
        "workload_fingerprint_evaluated": exact,
        "workload_fingerprint": "1" * 64 if exact else "",
        "recommendation_policy_version": contract.POLICY_VERSION,
        "mpi_topology_digest": "2" * 64,
        "native_state_digest": "3" * 64 if exact else "",
        "process_nonce": "4" * 64 if exact else "",
        "run_nonce": "5" * 64 if exact else "",
        "acknowledgement_digest": "",
    }
    if exact:
        result["acknowledgement_digest"] = contract._acknowledgement_digest(result)
    if acknowledgement and not reused:
        result["acknowledgement_origin_cohort_digest"] = result[
            "acknowledgement_digest"
        ]
    return result


def evidence(
    contract: Any,
    selected: str,
    recommended: str,
    *,
    reason: str,
    ranks: int = 1,
    execution_wall_seconds: float = 1.0,
) -> dict[str, Any]:
    mismatch = selected != recommended
    warning = telemetry(contract, selected, recommended, ranks=ranks)
    final = copy.deepcopy(warning)
    final["acknowledgement"] = mismatch
    final["acknowledgement_reason"] = reason if mismatch else ""
    final["acknowledgement_scope"] = "process" if mismatch else ""
    final["acknowledgement_origin_cohort_digest"] = (
        final["acknowledgement_digest"] if mismatch else ""
    )
    return {
        "schema": contract.SCHEMA,
        "attempts": 2 if mismatch else 1,
        "acknowledgements": 1 if mismatch else 0,
        "acknowledgement_reused": False,
        "mutation_before_acknowledgement": False,
        "warning_wall_seconds": 0.1 if mismatch else 0.0,
        "execution_wall_seconds": execution_wall_seconds,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "mismatch_observed": mismatch,
        "warning_telemetry": warning if mismatch else None,
        "final_telemetry": final,
    }
