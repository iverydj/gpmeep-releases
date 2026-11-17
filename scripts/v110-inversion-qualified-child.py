#!/usr/bin/env python3
"""Package-owned child for receipt-bound v1.1.0 inversion experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import runpy
import sys
import tempfile
import time
import unittest
import uuid
import weakref
from typing import Any


SCHEMA = "gpmeep-v110-qualified-child-runtime-v10"
OUTPUT_ENV = "GPMEEP_EXPERIMENT_RUNTIME_TERMINAL"
NONCE_ENV = "GPMEEP_EXPERIMENT_NONCE"
POLICY_TELEMETRY_SCHEMA = "gpmeep-v110-backend-recommendation-v3"
POLICY_VERSION = "gpmeep-v110-explicit-backend-guard-v24"
PHASES = (
    "initialization", "fdtd_timestep", "source", "boundary", "dispersive_ade",
    "dft_reduction", "host_device_transfer_and_sync", "mpi", "python_adjoint_mpb", "io",
)
COUNTERS = {
    "curl": ("dispatch", "curl"),
    "update_eh": ("field_updates", "update_eh"),
    "polarization": ("polarizations", "polarization"),
    "source": ("sources", "source"),
    "boundary": ("boundaries", "boundary"),
    "dft": ("dfts", "dft"),
}
# These are the exact workload arguments used by the sealed M4 manifest.
# Accept no caller-supplied workload arguments or replacement entrypoints.
WORKLOAD_ARGUMENTS = {
    "python/examples/phase_in_material.py": ("--validation",),
    "python/examples/material-dispersion.py": ("--validation",),
    "python/examples/refl-quartz.py": (),
    "python/examples/waveguide_crossing.py": ("--validation",),
    "python/examples/adjoint_optimization/multilayer_opt.py": ("--validation",),
    "python/examples/absorber-1d.py": ("--validation",),
    "python/examples/parallel-wvgs-force.py": ("--validation",),
    "python/examples/antenna_pec_ground_plane_1D.py": ("--validation",),
    "python/examples/3rd-harm-1d.py": ("--validation",),
    "python/examples/adjoint_optimization/near2far_3d_validation.py": (),
    "python/examples/edge_emitter_3D.py": ("--validation",),
    "python/examples/dipole_in_vacuum_cyl_off_axis.py": ("--validation",),
    "python/tests/test_adjoint_cyl.py": (),
}
WORKLOAD_SUPPORT = {
    "python/examples/phase_in_material.py": (
        "scripts/python-validation/phase_material_contract.py",
    ),
    "python/examples/absorber-1d.py": (
        "scripts/python-validation/absorber_branch_matrix.py",
    ),
    "python/tests/test_adjoint_cyl.py": (
        "python/tests/utils.py",
        "python/tests/adjoint_cyl_fixed_work_contract.py",
    ),
}
ABLATIONS = {
    "curl-phase-replay": {
        "off": {
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_DISABLE_CURL_PHASE_REPLAY": "1",
        },
        "on": {
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_ENABLE_CURL_PHASE_REPLAY": "1",
        },
    },
    "phase-batching": {
        "off": {
            "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH": "1",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE": "1",
        },
        "on": {
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE": "1",
        },
    },
    "dft-phase-sharing": {
        "off": {"MEEP_GPU_DISABLE_DFT_PHASE_SHARING": "1"},
        "on": {},
    },
    "multi-monitor-dft-batch": {
        "off": {"MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH": "1"},
        "on": {"MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH": "1"},
    },
    "phase-material-mirror-reuse": {
        "off": {"MEEP_GPU_DISABLE_PHASE_MATERIAL_MIRROR_REUSE": "1"},
        "on": {},
    },
    "boundary-span-cache": {
        "off": {"MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE": "1"},
        "on": {},
    },
    "boundary-plan-replay": {
        "off": {"MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY": "1"},
        "on": {},
    },
    "boundary-fast-path": {
        "off": {
            "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY": "1",
            "MEEP_GPU_DISABLE_BOUNDARY_GRAPH": "1",
            "MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO": "1",
            "MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY": "1",
            "MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY": "1",
        },
        "on": {},
    },
    "device-inventory-snapshot": {
        "off": {"MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT": "1"},
        "on": {},
    },
    "backend-ack-scope-cache": {
        "off": {"MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE": "1"},
        "on": {},
    },
    "sparse-lorentzian": {
        "off": {"MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN": "1"},
        "on": {"MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN": "1"},
    },
}


class ChildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChildError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ChildError(f"non-finite JSON constant: {item}")
        ),
    )
    _require(isinstance(value, dict), "installed provenance root differs")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _backend_policy_run_record(mp: Any) -> dict[str, Any]:
    telemetry = mp.gpu.recommendation
    _require(isinstance(telemetry, dict), "backend policy telemetry is absent")
    keys = (
        "schema",
        "selected_backend",
        "recommended_backend",
        "reason_class",
        "recommendation_confidence",
        "acknowledgement",
        "acknowledgement_allowed",
        "acknowledgement_reason",
        "acknowledgement_reused",
        "acknowledgement_scope",
        "acknowledgement_origin_cohort_digest",
        "acknowledgement_scope_cache_enabled",
        "acknowledgement_scope_digest",
        "workload_fingerprint_evaluated",
        "workload_fingerprint",
        "recommendation_policy_version",
        "mpi_topology_digest",
        "acknowledgement_digest",
    )
    _require(
        all(key in telemetry for key in keys),
        "backend policy telemetry fields differ",
    )
    record = {key: telemetry[key] for key in keys}
    _require(
        record["schema"] == POLICY_TELEMETRY_SCHEMA
        and record["recommendation_policy_version"] == POLICY_VERSION
        and record["selected_backend"] in {"cpu", "cuda"}
        and record["recommended_backend"] in {"cpu", "cuda"}
        and record["reason_class"] in {"hardware_crossover", "environment"}
        and record["recommendation_confidence"] in {"medium", "high"}
        and all(
            type(record[name]) is bool
            for name in (
                "acknowledgement",
                "acknowledgement_allowed",
                "acknowledgement_reused",
                "acknowledgement_scope_cache_enabled",
                "workload_fingerprint_evaluated",
            )
        )
        and isinstance(record["acknowledgement_reason"], str)
        and isinstance(record["acknowledgement_scope"], str)
        and isinstance(record["acknowledgement_origin_cohort_digest"], str)
        and isinstance(record["workload_fingerprint"], str)
        and isinstance(record["acknowledgement_digest"], str)
        and _hex64(record["acknowledgement_scope_digest"])
        and _hex64(record["mpi_topology_digest"]),
        "backend policy telemetry values differ",
    )
    if record["acknowledgement"]:
        _require(
            record["acknowledgement_allowed"] is True
            and len(record["acknowledgement_reason"].encode("utf-8")) >= 8
            and record["acknowledgement_scope"] == "process"
            and _hex64(record["acknowledgement_origin_cohort_digest"]),
            "accepted backend policy acknowledgement differs",
        )
        if record["acknowledgement_reused"]:
            _require(
                record["acknowledgement_scope_cache_enabled"] is True
                and record["workload_fingerprint_evaluated"] is False
                and record["workload_fingerprint"] == ""
                and record["acknowledgement_digest"] == "",
                "reused backend policy acknowledgement differs",
            )
        else:
            _require(
                record["workload_fingerprint_evaluated"] is True
                and _hex64(record["workload_fingerprint"])
                and _hex64(record["acknowledgement_digest"])
                and record["acknowledgement_origin_cohort_digest"]
                == record["acknowledgement_digest"],
                "exact backend policy acknowledgement differs",
            )
    else:
        _require(
            record["acknowledgement_reused"] is False
            and record["acknowledgement_reason"] == ""
            and record["acknowledgement_scope"] == ""
            and record["acknowledgement_origin_cohort_digest"] == "",
            "unacknowledged backend policy telemetry differs",
        )
    return record


def _validate_backend_policy_run_sequence(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind every serial cache reuse to an earlier exact cohort receipt."""
    exact_origins: dict[tuple[Any, ...], set[str]] = {}
    for record in records:
        if not record["acknowledgement"]:
            continue
        cohort = (
            record["schema"],
            record["selected_backend"],
            record["recommended_backend"],
            record["reason_class"],
            record["recommendation_confidence"],
            record["recommendation_policy_version"],
            record["acknowledgement_scope"],
            record["acknowledgement_scope_digest"],
            record["mpi_topology_digest"],
            record["acknowledgement_reason"],
        )
        if record["acknowledgement_reused"]:
            _require(
                record["acknowledgement_origin_cohort_digest"]
                in exact_origins.get(cohort, set()),
                "reused backend policy acknowledgement lineage differs",
            )
        else:
            exact_origins.setdefault(cohort, set()).add(
                record["acknowledgement_digest"]
            )
    return records


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(value: Any, label: str) -> pathlib.Path:
    _require(
        isinstance(value, dict)
        and set(value) == {"path", "size_bytes", "sha256"}
        and type(value["size_bytes"]) is int
        and value["size_bytes"] >= 0
        and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64,
        f"{label} provenance record differs",
    )
    lexical = pathlib.Path(value["path"]).absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    path = lexical.resolve(strict=True)
    _require(
        path == lexical
        and path.is_file()
        and path.stat().st_size == value["size_bytes"]
        and _sha256(path) == value["sha256"],
        f"{label} installed bytes differ",
    )
    return path


def _runtime(report_path: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path]:
    report = _load(report_path)
    unsigned = dict(report)
    report_id = unsigned.pop("report_id", None)
    _require(
        report.get("schema_version") == 1
        and report.get("state") == "PASS"
        and isinstance(report_id, str)
        and len(report_id) == 64
        and report_id == _canonical_sha256(unsigned),
        "installed provenance identity differs",
    )
    runtime = report.get("runtime")
    _require(isinstance(runtime, dict), "installed runtime provenance differs")
    python_path = _file_record(runtime.get("python"), "Python")
    module_path = _file_record(runtime.get("module"), "meep module")
    _file_record(runtime.get("extension"), "meep extension")
    _file_record(runtime.get("libmeep"), "libmeep")
    _require(pathlib.Path(sys.executable).resolve(strict=True) == python_path, "child Python differs")
    _require(
        runtime.get("cuda_compiled") is True
        and runtime.get("mpi_enabled") is True
        and runtime.get("single_precision") is True,
        "child runtime capability differs",
    )
    sys.path.insert(0, str(module_path.parent.parent))
    return report, module_path


def _statistics_delta(before: Any, after: Any, path: tuple[str, ...] = ()) -> Any:
    _require(type(before) is type(after), "statistics type changed")
    if isinstance(before, dict):
        _require(set(before) == set(after), "statistics schema changed")
        return {key: _statistics_delta(before[key], after[key], (*path, key)) for key in before}
    _require(
        type(before) is int and type(after) is int and before >= 0 and after >= 0
        and (after >= before or path in RUN_STATE_GAUGES),
        "statistics counter decreased or changed type",
    )
    return after - before


def _counter(delta: dict[str, Any], group: str, key: str) -> int:
    value = delta.get(group, {}).get(key)
    _require(type(value) is int and value >= 0, f"missing native counter {group}.{key}")
    return value


def _physical_gpu_uuid(identifier: str) -> str:
    # Meep exposes cudaUUID_t as 32 hex digits; CUDA_VISIBLE_DEVICES and the
    # retained physical-device census use NVIDIA's GPU-8-4-4-4-12 spelling.
    value = identifier.removeprefix("GPU-").lower()
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ChildError("native GPU identifier is not a physical UUID") from error
    _require(value in {parsed.hex, str(parsed)}, "native GPU UUID spelling differs")
    return "GPU-" + str(parsed)


def _dispatch(delta: dict[str, Any]) -> dict[str, dict[str, int]]:
    result = {backend: {"calls": 0, "points": 0} for backend in ("cpu", "cuda")}
    for group, stem in COUNTERS.values():
        for backend in result:
            result[backend]["calls"] += _counter(delta, group, f"{backend}_{stem}_calls")
            result[backend]["points"] += _counter(delta, group, f"{backend}_{stem}_points")
    return result


def _feature_counters(delta: dict[str, Any]) -> dict[str, Any]:
    phase = delta.get("phase_batch_policy", {})
    names = (
        "phase_curl_replay_checks",
        "phase_curl_replay_hits",
        "phase_curl_replay_unready",
        "phase_curl_replay_generation_misses",
        "phase_curl_replay_mirror_misses",
    )
    replay = {name.removeprefix("phase_curl_replay_"): phase.get(name) for name in names}
    forced = phase.get("phase_curl_forced_batches")
    _require(
        all(type(value) is int and value >= 0 for value in (*replay.values(), forced)),
        "curl phase replay counters differ",
    )
    _require(
        replay["checks"]
        == replay["hits"]
        + replay["unready"]
        + replay["generation_misses"]
        + replay["mirror_misses"],
        "curl phase replay counter conservation differs",
    )
    return {"curl_phase_replay": replay, "phase_curl_forced_batches": forced}


def _device_inventory_probe(mp: Any) -> dict[str, Any]:
    mp.gpu.reset_statistics()
    devices = mp.gpu.devices()
    statistics = mp.gpu.statistics()
    runtime = statistics.get("runtime")
    _require(
        isinstance(devices, list)
        and len(devices) > 0
        and isinstance(runtime, dict),
        "device inventory ablation probe differs",
    )
    result = {
        "device_count": len(devices),
        "inventory_sha256": _canonical_sha256(devices),
        "runtime_availability_probes": runtime.get(
            "runtime_availability_probes"
        ),
        "runtime_device_enumerations": runtime.get(
            "runtime_device_enumerations"
        ),
    }
    _require(
        all(
            type(result[name]) is int and result[name] >= 0
            for name in (
                "device_count",
                "runtime_availability_probes",
                "runtime_device_enumerations",
            )
        )
        and len(result["inventory_sha256"]) == 64
        and result["runtime_availability_probes"] == 1,
        "device inventory ablation counters differ",
    )
    return result


def _apply_ablation(value: str | None) -> dict[str, str]:
    if value is None:
        return {}
    _require(value.count("=") == 1, "ablation setting differs")
    name, state = value.split("=", 1)
    _require(name in ABLATIONS and state in {"off", "on"}, "ablation is not allowlisted")
    controlled = {key for states in ABLATIONS.values() for values in states.values() for key in values}
    for key in controlled:
        os.environ.pop(key, None)
    selected = ABLATIONS[name][state]
    os.environ.update(selected)
    return dict(sorted(selected.items()))


# Embedded identically in the package child and its source-bound verifier.
# Raw state gauges/high-water marks are retained, never summed as work.
RUN_OBSERVATION_SCHEMA = "gpmeep-v110-run-observation-v1"
RUN_STATE_GAUGES = {
    ("resident", "live_resident_device_buffers"),
    ("structured_curl_graph", "structured_curl_graph_live_plans"),
    ("structured_curl_graph", "structured_curl_graph_quarantined_plans"),
}
RUN_TIMING_SINKS = (
    ("connecting_chunks", "Connecting"), ("time_stepping", "Stepping"),
    ("boundaries", "Boundaries"), ("mpi_all_to_all", "MpiAllTime"),
    ("mpi_one_to_one", "MpiOneTime"), ("field_output", "FieldOutput"),
    ("fourier_transform", "FourierTransforming"), ("mpb", "MPBTime"),
    ("get_farfields", "GetFarfieldsTime"), ("other", "Other"),
    ("field_update_b", "FieldUpdateB"), ("field_update_h", "FieldUpdateH"),
    ("field_update_d", "FieldUpdateD"), ("field_update_e", "FieldUpdateE"),
    ("boundary_stepping_b", "BoundarySteppingB"),
    ("boundary_stepping_wh", "BoundarySteppingWH"),
    ("boundary_stepping_ph", "BoundarySteppingPH"),
    ("boundary_stepping_h", "BoundarySteppingH"),
    ("boundary_stepping_d", "BoundarySteppingD"),
    ("boundary_stepping_we", "BoundarySteppingWE"),
    ("boundary_stepping_pe", "BoundarySteppingPE"),
    ("boundary_stepping_e", "BoundarySteppingE"),
)
BOUNDARY_EXECUTION_KEYS = (
    "local_checks", "local_disabled", "local_unready",
    "local_prepare_invalidations", "local_replay_empty", "local_replay_work",
    "local_failures", "local_descriptor_rebuild_attempts", "prepare_calls",
    "prepare_failures", "prepared_arrays", "prepare_host_nanoseconds",
    "plan_empty", "plan_direct_zero", "plan_direct_local", "plan_graph", "plan_staged",
    "span_cache_hits", "span_cache_misses", "span_cache_bypasses",
)
LOCAL_BOUNDARY_OUTCOMES = (
    "local_disabled", "local_unready", "local_prepare_invalidations",
    "local_replay_empty", "local_replay_work", "local_failures",
)


# Exact public schema from the reviewed python/meep.i, bound by source identity.
RUN_STATISTICS_KEYS = {'runtime': ('runtime_availability_probes',
             'runtime_device_enumerations',
             'runtime_device_selections'),
 'dispatch': ('cpu_curl_calls',
              'cpu_curl_points',
              'cuda_curl_calls',
              'cuda_curl_points',
              'host_to_device_bytes',
              'device_to_host_bytes'),
 'material_phasing': ('checks', 'reused', 'disabled', 'ineligible'),
 'resident': ('host_to_device_bytes_avoided',
              'device_to_host_bytes_avoided',
              'device_buffer_allocations',
              'device_buffer_reuses',
              'live_resident_device_buffers'),
 'field_updates': ('cpu_update_eh_calls',
                   'cpu_update_eh_points',
                   'cuda_update_eh_calls',
                   'cuda_update_eh_points'),
 'polarizations': ('cpu_polarization_calls',
                   'cpu_polarization_points',
                   'cuda_polarization_calls',
                   'cuda_polarization_points',
                   'cuda_zero_state_elision_enabled_calls',
                   'cuda_zero_state_elision_enabled_points',
                   'cuda_zero_state_elision_inspectable_calls',
                   'cuda_zero_state_elision_inspectable_points'),
 'sources': ('cpu_source_calls', 'cpu_source_points', 'cuda_source_calls', 'cuda_source_points'),
 'resident_point_reads': ('cuda_read_attempts',
                          'cuda_read_completions',
                          'cuda_read_failures',
                          'scalars_copied',
                          'device_to_host_calls',
                          'device_to_host_bytes',
                          'host_nanoseconds'),
 'boundary_execution': ('local_checks',
                        'local_disabled',
                        'local_unready',
                        'local_prepare_invalidations',
                        'local_replay_empty',
                        'local_replay_work',
                        'local_failures',
                        'local_descriptor_rebuild_attempts',
                        'prepare_calls',
                        'prepare_failures',
                        'prepared_arrays',
                        'prepare_host_nanoseconds',
                        'plan_empty',
                        'plan_direct_zero',
                        'plan_direct_local',
                        'plan_graph',
                        'plan_staged',
                        'span_cache_hits', 'span_cache_misses', 'span_cache_bypasses'),
 'boundaries': ('cpu_boundary_calls',
                'cpu_boundary_points',
                'cuda_boundary_calls',
                'cuda_boundary_points'),
 'dfts': ('cpu_dft_calls', 'cpu_dft_points', 'cuda_dft_calls', 'cuda_dft_points'),
 'dft_batches': ('dft_batch_calls',
                 'dft_submitted_updates',
                 'dft_phase_preparation_launches',
                 'dft_phase_reuses',
                 'dft_update_kernel_launches',
                 'dft_maximum_batch_size',
                 'dft_multi_monitor_automatic_checks',
                 'dft_multi_monitor_automatic_selected',
                 'dft_multi_monitor_automatic_rejected',
                 'dft_multi_monitor_forced_batches',
                 'dft_multi_monitor_batched_updates',
                 'dft_multi_monitor_unbatched_updates',
                 'dft_multi_monitor_plan_uploads',
                 'dft_multi_monitor_plan_reuses',
                 'dft_multi_monitor_metadata_host_to_device_bytes'),
 'dft_reductions': ('cpu_dft_reduction_calls',
                    'cpu_dft_reduction_pairs',
                    'cpu_dft_reduction_terms',
                    'cuda_dft_reduction_calls',
                    'cuda_dft_reduction_pairs',
                    'cuda_dft_reduction_terms',
                    'cuda_dft_reduction_descriptor_uploads',
                    'cuda_dft_reduction_plan_reuses',
                    'cuda_dft_reduction_kernel_launches',
                    'cuda_dft_reduction_result_device_to_host_bytes',
                    'dft_reduction_full_dft_device_to_host_bytes_avoided',
                    'dft_reduction_mpi_allreduce_calls',
                    'dft_reduction_mpi_allreduce_bytes'),
 'dft_materializations': ('cpu_dft_array_materialization_calls',
                          'cpu_dft_array_materialization_points',
                          'cuda_dft_array_materialization_calls',
                          'cuda_dft_array_materialization_points',
                          'host_synthetic_material_array_calls',
                          'host_synthetic_material_array_points',
                          'cpu_dft_output_calls',
                          'cpu_dft_output_points',
                          'cuda_dft_output_calls',
                          'cuda_dft_output_points',
                          'cuda_dft_output_staging_calls',
                          'cuda_dft_output_staging_points',
                          'cuda_dft_output_staging_frequencies',
                          'cuda_dft_output_staging_descriptor_uploads',
                          'cuda_dft_output_staging_plan_reuses',
                          'cuda_dft_output_staging_kernel_launches',
                          'cuda_dft_output_staging_result_device_to_host_bytes',
                          'cuda_dft_output_staging_full_dft_device_to_host_bytes_avoided',
                          'cuda_dft_output_staging_workspace_ceiling_bytes',
                          'cuda_dft_materialization_kernel_launches',
                          'cuda_dft_materialization_result_device_to_host_bytes',
                          'dft_materialization_full_dft_device_to_host_bytes_avoided',
                          'dft_array_mpi_allreduce_calls',
                          'dft_array_mpi_allreduce_bytes'),
 'dft_checkpoints': ('cpu_dft_checkpoint_save_calls',
                     'cpu_dft_checkpoint_save_values',
                     'cuda_dft_checkpoint_save_calls',
                     'cuda_dft_checkpoint_save_values',
                     'cuda_dft_checkpoint_save_device_to_host_bytes',
                     'cuda_dft_checkpoint_save_full_cache_device_to_host_bytes_avoided',
                     'cpu_dft_checkpoint_load_calls',
                     'cpu_dft_checkpoint_load_values',
                     'cuda_dft_checkpoint_load_calls',
                     'cuda_dft_checkpoint_load_values',
                     'cuda_dft_checkpoint_load_host_to_device_bytes'),
 'dft_scales': ('cpu_dft_scale_calls',
                'cpu_dft_scale_values',
                'cuda_dft_scale_calls',
                'cuda_dft_scale_values',
                'cuda_dft_scale_kernel_launches',
                'cuda_dft_scale_host_to_device_bytes'),
 'eigenmode_overlaps': ('cpu_eigenmode_overlap_calls',
                        'cpu_eigenmode_overlap_terms',
                        'cuda_eigenmode_overlap_calls',
                        'cuda_eigenmode_overlap_terms',
                        'cuda_eigenmode_mode_flux_calls',
                        'cuda_eigenmode_mode_mode_calls',
                        'cuda_eigenmode_submitted_pairs',
                        'cuda_eigenmode_descriptor_uploads',
                        'cuda_eigenmode_plan_reuses',
                        'cuda_eigenmode_kernel_launches',
                        'cuda_eigenmode_result_device_to_host_bytes',
                        'eigenmode_full_dft_device_to_host_bytes_avoided',
                        'host_mode_profile_sampling_calls',
                        'host_mode_profile_sampling_points',
                        'eigenmode_zero_rank_channels_skipped',
                        'host_mode_profile_host_to_device_bytes',
                        'eigenmode_mpi_allreduce_calls',
                        'eigenmode_mpi_allreduce_bytes'),
 'ldos': ('cpu_ldos_reduction_calls',
          'cpu_ldos_source_points',
          'cuda_ldos_reduction_calls',
          'cuda_ldos_submitted_profiles',
          'cuda_ldos_source_points',
          'cuda_ldos_descriptor_uploads',
          'cuda_ldos_kernel_launches',
          'cuda_ldos_result_device_to_host_bytes',
          'ldos_full_field_device_to_host_bytes_avoided'),
 'near2far': ('cpu_near2far_transform_calls',
              'cpu_near2far_terms',
              'cuda_near2far_transform_calls',
              'cuda_near2far_terms',
              'cuda_near2far_submitted_chunks',
              'cuda_near2far_source_points',
              'cuda_near2far_output_points',
              'cuda_near2far_frequencies',
              'cuda_near2far_periodic_copies',
              'cuda_near2far_fast_precision_calls',
              'cuda_near2far_mixed_precision_calls',
              'cuda_near2far_cancellation_retries',
              'cuda_near2far_target_tiles',
              'cuda_near2far_frequency_tiles',
              'cuda_near2far_operation_tiles',
              'cuda_near2far_maximum_workspace_bytes',
              'cuda_near2far_descriptor_uploads',
              'cuda_near2far_kernel_launches',
              'cuda_near2far_result_device_to_host_bytes',
              'cuda_near2far_condition_device_to_host_bytes',
              'near2far_dft_device_to_host_bytes_avoided',
              'near2far_mpi_allreduce_calls',
              'near2far_mpi_allreduce_bytes',
              'cpu_near2far_adjoint_calls',
              'cpu_near2far_adjoint_terms',
              'cuda_near2far_adjoint_calls',
              'cuda_near2far_adjoint_terms',
              'cuda_near2far_adjoint_submitted_chunks',
              'cuda_near2far_adjoint_source_points',
              'cuda_near2far_adjoint_far_points',
              'cuda_near2far_adjoint_frequencies',
              'cuda_near2far_adjoint_periodic_copies',
              'cuda_near2far_adjoint_fast_precision_calls',
              'cuda_near2far_adjoint_mixed_precision_calls',
              'cuda_near2far_adjoint_cancellation_retries',
              'cuda_near2far_adjoint_maximum_workspace_bytes',
              'cuda_near2far_adjoint_descriptor_uploads',
              'cuda_near2far_adjoint_kernel_launches',
              'cuda_near2far_adjoint_host_to_device_bytes',
              'cuda_near2far_adjoint_result_device_to_host_bytes',
              'cuda_near2far_adjoint_condition_device_to_host_bytes'),
 'multi_gpu': ('mpi_messages',
               'mpi_scalars',
               'cuda_aware_bytes',
               'pinned_staging_bytes',
               'pinned_device_to_host_bytes',
               'pinned_host_to_device_bytes'),
 'mpi_completion': ('mpi_waitsome_executions', 'mpi_waitall_executions'),
 'boundary_eh_overlap': ('boundary_eh_overlap_checks',
                         'boundary_eh_overlap_eligible',
                         'boundary_eh_overlap_launched_h',
                         'boundary_eh_overlap_launched_e',
                         'boundary_eh_overlap_skipped_disabled',
                         'boundary_eh_overlap_skipped_unsupported_schedule',
                         'boundary_eh_overlap_skipped_no_remote',
                         'boundary_eh_overlap_skipped_cold_topology',
                         'boundary_eh_overlap_rejected'),
 'halo_curl_overlap': ('halo_curl_overlap_checks',
                       'halo_curl_overlap_eligible',
                       'halo_curl_overlap_launches',
                       'halo_curl_overlap_skipped_disabled',
                       'halo_curl_overlap_skipped_unsupported_schedule',
                       'halo_curl_overlap_skipped_no_remote',
                       'halo_curl_overlap_skipped_cold_topology',
                       'halo_curl_overlap_rejected_feature',
                       'halo_curl_overlap_rejected_small',
                       'halo_curl_overlap_full_points',
                       'halo_curl_overlap_interior_points',
                       'halo_curl_overlap_shell_points'),
 'tile_coalescing': ('tile_coalesced_curl_chunk_phases',
                     'tile_coalesced_curl_input_tiles',
                     'tile_coalesced_update_eh_chunk_phases',
                     'tile_coalesced_update_eh_input_tiles'),
 'phase_batch_policy': ('phase_curl_automatic_checks',
                        'phase_curl_automatic_selected',
                        'phase_curl_automatic_rejected',
                        'phase_curl_forced_batches',
                        'phase_curl_batched_operations',
                        'phase_curl_unbatched_operations',
                        'phase_curl_replay_checks',
                        'phase_curl_replay_hits',
                        'phase_curl_replay_unready',
                        'phase_curl_replay_generation_misses',
                        'phase_curl_replay_mirror_misses',
                        'phase_update_eh_automatic_checks',
                        'phase_update_eh_automatic_selected',
                        'phase_update_eh_automatic_rejected',
                        'phase_update_eh_forced_batches',
                        'phase_update_eh_batched_operations',
                        'phase_update_eh_unbatched_operations'),
 'structured_curl_graph': ('structured_curl_graph_collection_checks',
                           'structured_curl_graph_collection_launches',
                           'structured_curl_graph_plan_creations',
                           'structured_curl_graph_plan_rebuilds',
                           'structured_curl_graph_plan_reuses',
                           'structured_curl_graph_plan_destroys',
                           'structured_curl_graph_plan_destroy_failures',
                           'structured_curl_graph_live_plans',
                           'structured_curl_graph_quarantined_plans',
                           'structured_curl_graph_logical_operations',
                           'structured_curl_graph_unsupported_fallbacks',
                           'structured_curl_graph_creation_fallbacks',
                           'structured_curl_graph_launch_failures',
                           'structured_curl_graph_replay_checks',
                           'structured_curl_graph_replay_launches',
                           'structured_curl_graph_replay_unready',
                           'structured_curl_graph_replay_generation_misses',
                           'structured_curl_graph_replay_mirror_misses')}


def _validate_point_read_counters(point, total_d2h):
    _require(point["cuda_read_attempts"] == point["cuda_read_completions"] + point["cuda_read_failures"],
             "point read outcome conservation differs")
    _require(point["cuda_read_completions"] <= point["scalars_copied"] <= 2 * point["cuda_read_attempts"]
             and point["device_to_host_calls"] <= point["scalars_copied"]
             and ((point["device_to_host_calls"] == 0) == (point["scalars_copied"] == 0))
             and point["device_to_host_bytes"] == 4 * point["scalars_copied"]
             and point["device_to_host_bytes"] <= total_d2h,
             "point read transfer accounting differs")
    _require(point["cuda_read_attempts"] != 0 or point["host_nanoseconds"] == 0,
             "point read elapsed time has no attempt")


def _validate_span_cache_counters(boundary):
    # Empty arrays return before range resolution. A failed prepare may have
    # counted its attempted resolution before the successful-array counter.
    # On completed intervals every resolution belongs to a prepared array.
    if boundary["prepare_failures"] == 0:
        _require(sum(boundary[key] for key in (
            "span_cache_hits", "span_cache_misses", "span_cache_bypasses"
        )) <= boundary["prepared_arrays"],
                 "boundary span cache outcomes exceed prepared arrays")


def _validate_statistics_snapshot(snapshot):
    _require(isinstance(snapshot, dict) and set(snapshot) == set(RUN_STATISTICS_KEYS),
             "run statistics group inventory differs")
    for group, names in RUN_STATISTICS_KEYS.items():
        counters = snapshot[group]
        _require(isinstance(counters, dict) and set(counters) == set(names),
                 "run statistics counter inventory differs")
        _require(all(type(value) is int and value >= 0 for value in counters.values()),
                 "run statistics counter type differs")
    phasing = snapshot["material_phasing"]
    _require(phasing["checks"] == sum(phasing[key] for key in ("reused", "disabled", "ineligible")),
             "material phasing decision conservation differs")
    boundary = snapshot["boundary_execution"]
    _validate_span_cache_counters(boundary)
    _require(boundary["local_checks"] == sum(boundary[k] for k in LOCAL_BOUNDARY_OUTCOMES),
             "run boundary decision conservation differs")
    _validate_point_read_counters(snapshot["resident_point_reads"],
                                  snapshot["dispatch"]["device_to_host_bytes"])


def _validate_statistics_progress(before, after):
    _validate_statistics_snapshot(before)
    _validate_statistics_snapshot(after)
    _require(all(after[group][key] >= before[group][key]
                 or (group, key) in RUN_STATE_GAUGES
                 for group, keys in RUN_STATISTICS_KEYS.items() for key in keys),
             "run statistics counter decreased")
    _validate_span_cache_counters({
        key: after["boundary_execution"][key] - before["boundary_execution"][key]
        for key in BOUNDARY_EXECUTION_KEYS
    })
    _validate_point_read_counters(
        {key: after["resident_point_reads"][key] - before["resident_point_reads"][key]
         for key in RUN_STATISTICS_KEYS["resident_point_reads"]},
        after["dispatch"]["device_to_host_bytes"] - before["dispatch"]["device_to_host_bytes"],
    )


def _workload_statistics_dispatch(workload_statistics):
    before, after = workload_statistics["before"], workload_statistics["after"]
    result = {backend: {"calls": 0, "points": 0} for backend in ("cpu", "cuda")}
    for group, stem in (("dispatch", "curl"), ("field_updates", "update_eh"),
                        ("polarizations", "polarization"), ("sources", "source"),
                        ("boundaries", "boundary"), ("dfts", "dft")):
        for backend in result:
            for quantity in ("calls", "points"):
                key = f"{backend}_{stem}_{quantity}"
                result[backend][quantity] += after[group][key] - before[group][key]
    return result


def _validate_run_observations(observations, timing, observed_method, workload_statistics):
    _require(isinstance(observations, list) and len(observations) == timing["run_count"],
             "run observation count differs")
    keys = {"schema", "index", "operation", "wall_seconds", "statistics_before",
            "statistics_after", "field_relation", "field_timing_before", "field_timing_after"}
    _require(isinstance(workload_statistics, dict)
             and set(workload_statistics) == {"before", "after"},
             "workload statistics schema differs")
    _validate_statistics_progress(workload_statistics["before"], workload_statistics["after"])
    cursor = workload_statistics["before"]
    seconds = []
    for index, record in enumerate(observations):
        _require(isinstance(record, dict) and set(record) == keys
                 and record["schema"] == RUN_OBSERVATION_SCHEMA
                 and type(record["index"]) is int and record["index"] == index
                 and record["operation"] == observed_method,
                 "run observation identity differs")
        wall = record["wall_seconds"]
        _require(type(wall) in {int, float} and math.isfinite(wall) and wall > 0,
                 "run observation wall time differs")
        seconds.append(wall)
        before, after = record["statistics_before"], record["statistics_after"]
        _validate_statistics_progress(cursor, before)
        _validate_statistics_progress(before, after)
        cursor = after
        left, right = record["field_timing_before"], record["field_timing_after"]
        for snapshot in (left, right):
            if snapshot is None:
                continue
            _require(isinstance(snapshot, dict)
                     and set(snapshot) == {name for name, _ in RUN_TIMING_SINKS},
                     "run native timing inventory differs")
            for values in snapshot.values():
                _require(isinstance(values, list) and len(values) == 1
                         and type(values[0]) in {int, float}
                         and math.isfinite(values[0]) and values[0] >= 0,
                         "run native timing value differs")
        relation = record["field_relation"]
        _require(isinstance(relation, str), "run fields epoch relation type differs")
        _require(
            (relation == "absent" and left is None and right is None)
            or (relation == "initialized" and left is None and right is not None)
            or (relation == "released" and left is not None and right is None)
            or (relation in {"same", "replaced"} and left is not None and right is not None),
            "run fields epoch relation differs",
        )
        if relation == "same":
            _require(all(right[name][0] >= left[name][0] - 1e-12 for name in left),
                     "run native timing decreased in the same fields epoch")
    _require(math.fsum(seconds) == timing["primary_seconds"],
             "run observation wall sum differs")
    _validate_statistics_progress(cursor, workload_statistics["after"])
    return observations


def _validate_workload_feature_projection(workload_statistics, features):
    before = workload_statistics["before"]["phase_batch_policy"]
    after = workload_statistics["after"]["phase_batch_policy"]
    replay = {name: after["phase_curl_replay_" + name] - before["phase_curl_replay_" + name]
              for name in ("checks", "hits", "unready", "generation_misses", "mirror_misses")}
    forced = after["phase_curl_forced_batches"] - before["phase_curl_forced_batches"]
    _require(features["curl_phase_replay"] == replay
             and features["phase_curl_forced_batches"] == forced,
             "workload statistics and curl feature counters differ")


def _boundary_observation_totals(observations):
    # These 20 counters are explicitly additive. Do not apply this operation to
    # the raw state gauges or maximum-workspace/batch-size high-water marks.
    return {
        name: sum(record["statistics_after"]["boundary_execution"][name]
                  - record["statistics_before"]["boundary_execution"][name]
                  for record in observations)
        for name in BOUNDARY_EXECUTION_KEYS
    }


def _validate_material_phasing_control(setting, backend, observations):
    if setting not in {"phase-material-mirror-reuse=off", "phase-material-mirror-reuse=on"}:
        return None
    _require(backend == "cuda", "material phasing reuse control requires CUDA1")
    totals = {
        name: sum(record["statistics_after"]["material_phasing"][name]
                  - record["statistics_before"]["material_phasing"][name]
                  for record in observations)
        for name in RUN_STATISTICS_KEYS["material_phasing"]
    }
    _require(totals["checks"] > 0, "material phasing control did not execute phasing")
    if setting.endswith("=off"):
        _require(totals["disabled"] == totals["checks"] and totals["reused"] == 0
                 and totals["ineligible"] == 0, "material phasing OFF did not disable reuse")
    else:
        _require(totals["reused"] > 0 and totals["disabled"] == 0,
                 "material phasing ON did not reuse stable mirrors")
    return totals


def _validate_boundary_span_cache_control(setting, backend, observations):
    if setting not in {"boundary-span-cache=off", "boundary-span-cache=on"}:
        return None
    _require(backend == "cuda", "boundary span cache control requires CUDA1")
    totals = _boundary_observation_totals(observations)
    _require(totals["prepare_calls"] > 0 and totals["prepared_arrays"] > 0
             and totals["prepare_failures"] == 0 and totals["local_failures"] == 0,
             "boundary span cache control did not complete eligible work")
    if setting.endswith("=off"):
        _require(totals["span_cache_bypasses"] > 0
                 and totals["span_cache_hits"] == 0 and totals["span_cache_misses"] == 0,
                 "boundary span cache OFF did not bypass the cache")
    else:
        _require(totals["span_cache_hits"] > 0 and totals["span_cache_misses"] > 0
                 and totals["span_cache_bypasses"] == 0,
                 "boundary span cache ON did not reuse validated spans")
    return totals


def _validate_boundary_replay_control(setting, backend, observations):
    if setting not in {"boundary-plan-replay=off", "boundary-plan-replay=on"}:
        return None
    _require(backend == "cuda", "boundary plan replay control requires CUDA1")
    totals = _boundary_observation_totals(observations)
    _require(totals["local_checks"] > 0 and totals["local_failures"] == 0
             and totals["prepare_calls"] > 0 and totals["prepare_failures"] == 0,
             "boundary plan replay control did not complete eligible boundary work")
    if setting.endswith("=off"):
        _require(totals["local_disabled"] > 0
                 and totals["local_replay_work"] == 0 and totals["local_replay_empty"] == 0
                 and totals["local_descriptor_rebuild_attempts"] > 0,
                 "boundary plan replay OFF did not rebuild without replay")
    else:
        _require(totals["local_disabled"] == 0 and totals["local_replay_work"] > 0,
                 "boundary plan replay ON did not execute resident replay work")
    return totals


def _run_field_timing(mp, simulation):
    if simulation.fields is None:
        return None
    timings = simulation.get_timing_data()
    return {name: [float(value) for value in timings[getattr(mp, attribute)]]
            for name, attribute in RUN_TIMING_SINKS}


def _install_scaling(
    mp: Any,
    size: float,
    run_length: float,
    phase_seconds: list[float],
    mismatch_type: type[Exception],
    mismatch_acknowledgements: list[str],
    backend_policy_runs: list[dict[str, Any]],
    *,
    observed_method: str = "run",
    numerical_runs: list[dict[str, Any]] | None = None,
    run_observations: list[dict[str, Any]] | None = None,
) -> Any:
    _require(math.isfinite(size) and size > 0, "sweep size differs")
    _require(math.isfinite(run_length) and run_length > 0, "sweep run length differs")
    original_init = mp.Simulation.__init__
    _require(observed_method in {"run", "run_k_point"}, "observed operation differs")
    original_run = getattr(mp.Simulation, observed_method)
    methods: dict[str, tuple[Any, Any]] = {}

    def pristine(action):
        # The backend guard authenticates the Simulation class. Instrumentation
        # must be absent during both the original operation and its exact retry.
        for name, (_original, observer) in methods.items():
            _require(getattr(mp.Simulation, name) is observer,
                     f"Simulation.{name} observer changed")
        for name, (original, _observer) in methods.items():
            setattr(mp.Simulation, name, original)
        try:
            return action()
        finally:
            changed = [name for name, (original, _) in methods.items()
                       if getattr(mp.Simulation, name) is not original]
            for name, (_original, observer) in methods.items():
                setattr(mp.Simulation, name, observer)
            _require(not changed, "Simulation changed during original dispatch")

    def scaled_init(simulation, *args, **kwargs):
        _require("resolution" in kwargs, "sweep workload resolution is not keyword-bound")
        kwargs["resolution"] = float(kwargs["resolution"]) * size
        return pristine(lambda: original_init(simulation, *args, **kwargs))

    def measured_run(simulation, *args, **kwargs):
        if run_length != 1.0:
            if observed_method == "run_k_point":
                if args and isinstance(args[0], (int, float)):
                    args = (float(args[0]) * run_length, *args[1:])
                elif isinstance(kwargs.get("t"), (int, float)):
                    kwargs["t"] = float(kwargs["t"]) * run_length
                else:
                    raise ChildError("k-point workload has no numeric duration")
            elif "until" in kwargs and isinstance(kwargs["until"], (int, float)):
                kwargs["until"] = float(kwargs["until"]) * run_length
            elif "until_after_sources" in kwargs and isinstance(
                kwargs["until_after_sources"], (int, float)
            ):
                kwargs["until_after_sources"] = float(kwargs["until_after_sources"]) * run_length
            else:
                raise ChildError("sweep workload has no scalable numeric stopping condition")
        def invoke():
            try:
                result = original_run(simulation, *args, **kwargs)
            except mismatch_type as mismatch:
                reason = (
                    "receipt-bound inversion experiment intentionally "
                    "exercises the selected backend"
                )
                mp.gpu.acknowledge_backend_mismatch(
                    mismatch.challenge,
                    reason,
                    scope="process",
                )
                mismatch_acknowledgements.append(reason)
                result = original_run(simulation, *args, **kwargs)
            backend_policy_runs.append(_backend_policy_run_record(mp))
            return result

        # Do not hold the old fields alive across a run which replaces them.
        # These snapshots are outside the unchanged primary wall-time boundary.
        if run_observations is not None:
            fields_before = (None if simulation.fields is None
                             else weakref.ref(simulation.fields))
            statistics_before = mp.gpu.statistics()
            field_timing_before = _run_field_timing(mp, simulation)
        pre_timestep = (0 if numerical_runs is None or simulation.fields is None
                        else int(simulation.fields.t))
        started = time.monotonic_ns()
        try:
            result = pristine(invoke)
        finally:
            phase_seconds.append((time.monotonic_ns() - started) / 1_000_000_000.0)
        if run_observations is not None:
            # Energy/DFT probes below can themselves launch native boundary work.
            # Capture run-only observations before performing any such probes.
            statistics_after = mp.gpu.statistics()
            field_timing_after = _run_field_timing(mp, simulation)
            if fields_before is None:
                relation = "absent" if simulation.fields is None else "initialized"
            elif simulation.fields is None:
                relation = "released"
            else:
                relation = "same" if fields_before() is simulation.fields else "replaced"
            run_observations.append({
                "schema": RUN_OBSERVATION_SCHEMA,
                "index": len(run_observations),
                "operation": observed_method,
                "wall_seconds": phase_seconds[-1],
                "statistics_before": statistics_before,
                "statistics_after": statistics_after,
                "field_relation": relation,
                "field_timing_before": field_timing_before,
                "field_timing_after": field_timing_after,
            })
        if numerical_runs is not None:
            def observe_numerical_run():
                post_timestep = int(simulation.timestep())
                return {
                    "timestep": post_timestep,
                    "timestep_delta": post_timestep - pre_timestep,
                    "meep_time": float(simulation.round_time()),
                    "field_energy": float(simulation.field_energy_in_box(
                        box=simulation.fields.total_volume()
                    )),
                    "dft_norm": float(simulation.fields.dft_norm()),
                }
            numerical_runs.append(pristine(observe_numerical_run))
        return result

    methods["__init__"] = (original_init, scaled_init)
    methods[observed_method] = (original_run, measured_run)
    mp.Simulation.__init__ = scaled_init
    setattr(mp.Simulation, observed_method, measured_run)

    def restore():
        changed = [name for name, (_, observer) in methods.items()
                   if getattr(mp.Simulation, name) is not observer]
        for name, (original, _) in methods.items():
            setattr(mp.Simulation, name, original)
        _require(not changed, "Simulation instrumentation changed before restoration")

    return restore


def workload_contract(workload: pathlib.Path, workload_id: str) -> dict[str, Any]:
    _require(workload_id in WORKLOAD_ARGUMENTS, "workload has no sealed invocation")
    relative = pathlib.PurePosixPath(workload_id)
    repo = workload.parents[len(relative.parts) - 1]
    _require(repo / workload_id == workload, "workload path and ID differ")
    support = []
    for name in WORKLOAD_SUPPORT.get(workload_id, ()):
        path = repo / name
        _require(path.is_file() and not path.is_symlink()
                 and path.resolve(strict=True) == path, "workload support path differs")
        support.append({"path": str(path), "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path)})
    return {
        "arguments": list(WORKLOAD_ARGUMENTS[workload_id]),
        "observed_method": "run_k_point" if workload_id.endswith("/material-dispersion.py") else "run",
        "unittest_discover": workload_id == "python/tests/test_adjoint_cyl.py",
        "support_files": support,
    }


def _phase_helper(contract: dict[str, Any]) -> Any:
    support = contract["support_files"]
    _require(len(support) == 1, "phase workload helper inventory differs")
    path = pathlib.Path(support[0]["path"])
    _require(path.name == "phase_material_contract.py" and _sha256(path) == support[0]["sha256"],
             "phase workload helper identity differs")
    spec = importlib.util.spec_from_file_location("gpmeep_qualified_phase_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_workload(workload: pathlib.Path, contract: dict[str, Any], *,
                  phase_coordinates: tuple[float, float] | None = None) -> dict[str, Any] | None:
    previous_argv = sys.argv
    contents = list(previous_argv)
    previous_path = list(sys.path)
    previous_argv[:] = [str(workload), *contract["arguments"]]
    if phase_coordinates is not None and phase_coordinates != (1.0, 1.0):
        _require(workload.name == "phase_in_material.py", "phase coordinates used for another case")
        size, run_length = phase_coordinates
        previous_argv.extend(["--gpmeep-sweep-size", format(size, ".17g"),
                              "--gpmeep-sweep-run-length", format(run_length, ".17g")])
    sys.path.insert(0, str(workload.parent))
    namespace = None
    try:
        if contract["unittest_discover"]:
            suite = unittest.TestLoader().discover(
                start_dir=str(workload.parent), pattern=workload.name,
                top_level_dir=str(workload.parent),
            )
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
            _require(result.testsRun == 4 and not result.skipped and result.wasSuccessful(),
                     "cylindrical adjoint discovery did not pass all four tests")
        else:
            target = workload
            if workload.name == "absorber-1d.py":
                target = pathlib.Path(contract["support_files"][0]["path"])
                previous_argv[:] = [str(target), str(workload), "--", *contract["arguments"]]
            try:
                namespace = runpy.run_path(str(target), run_name="__main__")
            except SystemExit as error:
                _require(error.code in {None, 0}, "workload exited unsuccessfully")
    finally:
        sys.argv = previous_argv
        previous_argv[:] = contents
        sys.path[:] = previous_path
    return namespace


def _write_terminal(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = path.absolute()
    _require(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "child terminal target differs")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            _require(count > 0, "child terminal write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--runtime-provenance", required=True, type=pathlib.Path)
    parser.add_argument("--workload", required=True, type=pathlib.Path)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--gpmeep-sweep-size", type=float, default=1.0)
    parser.add_argument("--gpmeep-sweep-run-length", type=float, default=1.0)
    parser.add_argument("--gpmeep-ablation")
    args = parser.parse_args(argv)
    output = pathlib.Path(os.environ.get(OUTPUT_ENV, ""))
    nonce = os.environ.get(NONCE_ENV, "")
    _require(output.is_absolute() and len(nonce) == 64, "child launch contract differs")
    ablation_environment = _apply_ablation(args.gpmeep_ablation)
    report, expected_module = _runtime(args.runtime_provenance.resolve(strict=True))
    import meep as mp
    from meep import backend_policy

    _require(pathlib.Path(mp.__file__).resolve(strict=True) == expected_module, "imported meep module differs")
    inventory_probe = None
    if args.gpmeep_ablation in {
        "device-inventory-snapshot=off",
        "device-inventory-snapshot=on",
    }:
        inventory_probe = _device_inventory_probe(mp)
        expected_enumerations = (
            1
            if args.gpmeep_ablation.endswith("=on")
            else 1 + 10 * inventory_probe["device_count"]
        )
        _require(
            inventory_probe["runtime_device_enumerations"]
            == expected_enumerations,
            "device inventory snapshot ablation invariant differs",
        )
    workload = args.workload.absolute()
    _require(
        not workload.is_symlink()
        and workload.resolve(strict=True) == workload
        and workload.is_file(),
        "workload source path differs",
    )
    contract = workload_contract(workload, args.workload_id)
    if args.workload_id in {
        "python/examples/phase_in_material.py", "python/tests/test_adjoint_cyl.py"
    }:
        fixed_work = ("GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK"
                      if workload.name == "phase_in_material.py"
                      else "GPMEEP_ADJOINT_CYL_FIXED_WORK")
        _require(os.environ.get(fixed_work) == "1", "fixed-work workload mode differs")
    phase_seconds: list[float] = []
    mismatch_acknowledgements: list[str] = []
    backend_policy_runs: list[dict[str, Any]] = []
    phase_helper = None
    phase_work = None
    phase_coordinates = None
    phase_numerical_runs: list[dict[str, Any]] = []
    run_observations: list[dict[str, Any]] = []
    if args.workload_id == "python/examples/phase_in_material.py":
        phase_helper = _phase_helper(contract)
        phase_work = phase_helper.work_contract(
            args.gpmeep_sweep_size, args.gpmeep_sweep_run_length, fixed_work=True
        )
        phase_coordinates = (args.gpmeep_sweep_size, args.gpmeep_sweep_run_length)
    restore = _install_scaling(
        mp,
        1.0 if phase_work is not None else args.gpmeep_sweep_size,
        1.0 if phase_work is not None else args.gpmeep_sweep_run_length,
        phase_seconds,
        backend_policy.BackendRecommendationMismatch,
        mismatch_acknowledgements,
        backend_policy_runs,
        observed_method=contract["observed_method"],
        numerical_runs=phase_numerical_runs if phase_work is not None else None,
        run_observations=run_observations,
    )
    mp.gpu.reset_statistics()
    before = mp.gpu.statistics()
    try:
        namespace = _run_workload(workload, contract, phase_coordinates=phase_coordinates)
    finally:
        restore()
    _require(contract == workload_contract(workload, args.workload_id),
             "workload support changed during execution")
    after = mp.gpu.statistics()
    phase_numerical = None
    if phase_work is not None:
        _require(isinstance(namespace, dict), "phase workload did not retain full numerical outputs")
        _require(len(phase_seconds) == phase_work["run_count"], "phase run count differs")
        phase_numerical = phase_helper.write_numerical(
            namespace, output.parent / "phase-numerical.npy", phase_work, phase_numerical_runs
        )
    delta = _statistics_delta(before, after)
    feature_counters = _feature_counters(delta)
    feature_counters["device_inventory_probe"] = inventory_probe
    feature_counters["backend_mismatch_acknowledgements"] = len(
        mismatch_acknowledgements
    )
    if args.gpmeep_ablation == "curl-phase-replay=off":
        _require(
            feature_counters["phase_curl_forced_batches"] > 0
            and all(
                value == 0
                for value in feature_counters["curl_phase_replay"].values()
            ),
            "curl phase replay OFF control did not execute its invariant path",
        )
    elif args.gpmeep_ablation == "curl-phase-replay=on":
        _require(
            feature_counters["phase_curl_forced_batches"] > 0
            and feature_counters["curl_phase_replay"]["checks"] > 0
            and feature_counters["curl_phase_replay"]["hits"] > 0,
            "curl phase replay ON control did not execute replay",
        )
    requested = str(mp.gpu.requested_backend)
    active = str(mp.gpu.active_backend)
    selected = (_physical_gpu_uuid(str(mp.gpu.selected_device_identifier))
                if active == "cuda" else None)
    devices = [selected] if selected else []
    phases = {
        name: {"available": False, "seconds": None} for name in PHASES
    }
    if phase_seconds:
        phases["fdtd_timestep"] = {
            "available": True,
            "seconds": math.fsum(phase_seconds),
        }
    _require(phase_seconds, "sweep workload did not execute a simulation run")
    _require(
        len(backend_policy_runs) == len(phase_seconds),
        "backend policy/run evidence count differs",
    )
    _validate_backend_policy_run_sequence(backend_policy_runs)
    workload_seconds = math.fsum(phase_seconds)
    _require(
        math.isfinite(workload_seconds) and workload_seconds > 0,
        "sweep workload timing differs",
    )
    _validate_run_observations(
        run_observations,
        {"run_count": len(phase_seconds), "primary_seconds": workload_seconds},
        contract["observed_method"],
        {"before": before, "after": after},
    )
    _validate_boundary_replay_control(args.gpmeep_ablation, active, run_observations)
    _validate_boundary_span_cache_control(args.gpmeep_ablation, active, run_observations)
    _validate_material_phasing_control(args.gpmeep_ablation, active, run_observations)
    _validate_workload_feature_projection({"before": before, "after": after}, feature_counters)
    terminal: dict[str, Any] = {
        "schema": SCHEMA,
        "outcome": "PASS",
        "nonce": nonce,
        "runtime_receipt_id": report["report_id"],
        "requested_backend": requested,
        "active_backend": active,
        "selected_physical_gpu_uuids": devices,
        "mpi_ranks": 1,
        "rank_records": [
            {
                "rank": 0,
                "runtime_receipt_id": report["report_id"],
                "requested_backend": requested,
                "active_backend": active,
                "physical_gpu_uuid": selected,
                "dispatch_counters": _dispatch(delta),
            }
        ],
        "phase_profile": phases,
        "timing": {
            "contract": "sum-all-simulation-run-wall-seconds",
            "primary_seconds": workload_seconds,
            "run_count": len(phase_seconds),
        },
        "backend_policy_runs": backend_policy_runs,
        "run_observations": run_observations,
        "workload_statistics": {"before": before, "after": after},
        "feature_counters": feature_counters,
        "phase_numerical": phase_numerical,
        "runtime_files": {
            name: report["runtime"][name]
            for name in ("python", "module", "extension", "libmeep")
        },
        "workload": {
            "id": args.workload_id,
            "path": str(workload),
            "size_bytes": workload.stat().st_size,
            "sha256": _sha256(workload),
            "contract": contract,
        },
        "experiment": {
            "sweep_size": args.gpmeep_sweep_size,
            "sweep_run_length": args.gpmeep_sweep_run_length,
            "ablation": args.gpmeep_ablation,
            "ablation_environment": ablation_environment,
        },
    }
    terminal["terminal_id"] = _canonical_sha256(terminal)
    _write_terminal(output, terminal)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Source hashes must describe the executed helpers and discovered tests.
    # -B alone still reads existing timestamp-valid .pyc files. Use an empty,
    # private cache namespace before importing Meep or workload modules, while
    # keeping this per-process path out of the paired command/environment.
    with tempfile.TemporaryDirectory(prefix="gpmeep-inversion-import-") as cache:
        previous_prefix = sys.pycache_prefix
        previous_dont_write = sys.dont_write_bytecode
        sys.pycache_prefix = cache
        sys.dont_write_bytecode = True
        try:
            return _main(argv)
        finally:
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_dont_write


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ChildError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"v1.1.0 qualified inversion child: FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
