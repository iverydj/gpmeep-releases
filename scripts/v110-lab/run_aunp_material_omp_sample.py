#!/usr/bin/env python3
"""Run one provenance-bound C4 static-material OpenMP AuNP sample.

The sealed AuNP workload, numerical observables, timing boundary, CUDA
dispatch accounting, and atomic evidence publication are inherited from the
reviewed C2 worker.  This adapter changes only the candidate controls and its
resettable causal-counter contract.  Earlier C1/C3 candidates are disabled in
both lanes so an OFF/ON comparison isolates C4 material initialization.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from typing import Any


_BASE_PATH = pathlib.Path(__file__).with_name("run_aunp_zero_state_sample.py")
_BASE_MODULE_NAME = "_gpmeep_c4_zero_state_sample_base"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private C4 sample base module")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)


CANDIDATE_FEATURE = "static-material-openmp-initialization"
SCHEMA = "gpmeep-v110-m1-aunp-material-omp-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-material-omp-sample-complete-v1"
FEATURE_ENABLE = "MEEP_C4_MATERIAL_OMP_ENABLE"
FEATURE_DISABLE = "MEEP_C4_MATERIAL_OMP_DISABLE"
C1_ENABLE = "MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION"
C1_DISABLE = "MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION"
C3_ENABLE = "MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN"
C3_DISABLE = "MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN"
OMP_THREADS = 4
OMP_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(OMP_THREADS),
    "OMP_DYNAMIC": "FALSE",
    "OMP_PROC_BIND": "close",
    "OMP_PLACES": "cores",
}

COUNTERS = (
    "feature_checks",
    "enabled_checks",
    "eligible_checks",
    "disabled_checks",
    "unsafe_material_checks",
    "guard_rejections",
    "stale_epoch_rejections",
    "non_openmp_checks",
    "epsilon_parallel_calls",
    "epsilon_parallel_points",
    "sigma_parallel_calls",
    "sigma_parallel_points",
    "team_observations",
    "team_mismatch_observations",
    "prism_preflight_calls",
)
MAX_GAUGES = ("configured_threads", "observed_threads", "geometry_epoch")
FLAGS = ("prism_preflight_complete", "counter_overflow")


def _expected_feature_environment(mode: str) -> dict[str, str | None]:
    expected = {
        FEATURE_ENABLE: "1" if mode == "enabled" else None,
        FEATURE_DISABLE: None if mode == "enabled" else "1",
        C1_ENABLE: None,
        C1_DISABLE: "1",
        C3_ENABLE: None,
        C3_DISABLE: "1",
    }
    expected.update(OMP_ENVIRONMENT)
    return expected


def _feature_environment(mode: str) -> dict[str, str | None]:
    expected = _expected_feature_environment(mode)
    observed = {name: base.os.environ.get(name) for name in expected}
    if observed != expected:
        raise base.common.WorkloadError(
            f"C4 material OpenMP environment differs: {observed!r}"
        )
    return observed


def _reset_candidate_statistics(mp: Any) -> None:
    mp.reset_material_omp_statistics()


def _candidate_statistics(mp: Any) -> dict[str, Any]:
    statistics = mp.get_material_omp_statistics()
    result = {
        name: int(getattr(statistics, name))
        for name in COUNTERS + MAX_GAUGES + ("reset_generation",)
    }
    result.update({name: bool(getattr(statistics, name)) for name in FLAGS})
    return result


def _candidate_material_manifest(simulation: Any) -> dict[str, Any]:
    structure = getattr(simulation, "structure", None)
    if structure is None or not hasattr(structure, "material_arrays_manifest"):
        raise base.common.WorkloadError("C4 material manifest API is unavailable")
    manifest = json.loads(structure.material_arrays_manifest())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "gpmeep-material-arrays-v1"
        or manifest.get("realnum_bytes") != 4
        or not isinstance(manifest.get("chunks"), list)
    ):
        raise base.common.WorkloadError("C4 material manifest schema differs")
    return manifest


def _counter(group: Any, name: str) -> int:
    if not isinstance(group, dict):
        raise base.common.WorkloadError("C4 statistics group is absent")
    value = group.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise base.common.WorkloadError(f"invalid C4 counter {name}")
    return value


def _flag(group: Any, name: str) -> bool:
    if not isinstance(group, dict) or type(group.get(name)) is not bool:
        raise base.common.WorkloadError(f"invalid C4 flag {name}")
    return bool(group[name])


def _validate_snapshot_schema(snapshot: Any) -> None:
    required = set(COUNTERS + MAX_GAUGES + FLAGS + ("reset_generation",))
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise base.common.WorkloadError("C4 statistics schema differs")
    for name in COUNTERS + MAX_GAUGES + ("reset_generation",):
        _counter(snapshot, name)
    for name in FLAGS:
        _flag(snapshot, name)


def _validate_material_manifest(
    manifest: Any, *, require_arrays: bool
) -> tuple[int, int, int]:
    if type(require_arrays) is not bool:
        raise base.common.WorkloadError(
            "C4 material manifest array requirement is invalid"
        )
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {
            "schema",
            "realnum_bytes",
            "num_chunks",
            "chunks",
        }
        or manifest.get("schema") != "gpmeep-material-arrays-v1"
        or type(manifest.get("realnum_bytes")) is not int
        or manifest.get("realnum_bytes") != 4
        or type(manifest.get("num_chunks")) is not int
        or manifest["num_chunks"] < 1
        or not isinstance(manifest.get("chunks"), list)
        or len(manifest["chunks"]) != manifest["num_chunks"]
    ):
        raise base.common.WorkloadError("C4 material manifest root differs")
    arrays = 0
    bytes_total = 0
    susceptibility_arrays = 0

    def validate_array(
        record: Any,
        expected_elements: int,
        expected_component: int,
        expected_direction: int,
    ) -> bool:
        nonlocal arrays, bytes_total
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "component",
                "direction",
                "trivial",
                "null",
                "elements",
                "bytes",
                "sha256",
            }
            or type(record["component"]) is not int
            or record["component"] != expected_component
            or type(record["direction"]) is not int
            or record["direction"] != expected_direction
            or type(record["trivial"]) is not bool
            or type(record["null"]) is not bool
            or type(record["elements"]) is not int
            or type(record["bytes"]) is not int
        ):
            raise base.common.WorkloadError("C4 material array record differs")
        if record["null"]:
            if record["elements"] or record["bytes"] or record["sha256"] is not None:
                raise base.common.WorkloadError("C4 null material array differs")
            return False
        if (
            record["elements"] != expected_elements
            or record["bytes"] != expected_elements * 4
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise base.common.WorkloadError("C4 material array digest differs")
        arrays += 1
        bytes_total += record["bytes"]
        return True

    for chunk_index, chunk in enumerate(manifest["chunks"]):
        if (
            not isinstance(chunk, dict)
            or set(chunk)
            != {"index", "owned", "ntot", "chi1inv", "susceptibilities"}
            or type(chunk.get("index")) is not int
            or chunk.get("index") != chunk_index
            or type(chunk.get("owned")) is not bool
            or type(chunk.get("ntot")) is not int
            or chunk["ntot"] < 0
            or not isinstance(chunk.get("chi1inv"), list)
            or len(chunk["chi1inv"]) != 20 * 5
            or not isinstance(chunk.get("susceptibilities"), list)
        ):
            raise base.common.WorkloadError("C4 material chunk manifest differs")
        for array_index, record in enumerate(chunk["chi1inv"]):
            component, direction = divmod(array_index, 5)
            validate_array(
                record, chunk["ntot"], component, direction
            )
        expected_order: dict[int, int] = {}
        for susceptibility in chunk["susceptibilities"]:
            if (
                not isinstance(susceptibility, dict)
                or set(susceptibility)
                != {"field_type", "order", "type", "ntot", "sigma"}
                or type(susceptibility["field_type"]) is not int
                or not 0 <= susceptibility["field_type"] < 5
                or type(susceptibility["order"]) is not int
                or susceptibility["order"]
                != expected_order.get(susceptibility["field_type"], 0)
                or not isinstance(susceptibility["type"], str)
                or not susceptibility["type"]
                or type(susceptibility["ntot"]) is not int
                or susceptibility["ntot"] < 0
                or not isinstance(susceptibility["sigma"], list)
                or len(susceptibility["sigma"]) != 20 * 5
            ):
                raise base.common.WorkloadError(
                    "C4 susceptibility manifest differs"
                )
            expected_order[susceptibility["field_type"]] = (
                susceptibility["order"] + 1
            )
            for array_index, record in enumerate(susceptibility["sigma"]):
                component, direction = divmod(array_index, 5)
                if validate_array(
                    record,
                    susceptibility["ntot"],
                    component,
                    direction,
                ):
                    susceptibility_arrays += 1
    if (arrays == 0) != (bytes_total == 0):
        raise base.common.WorkloadError("C4 material manifest totals differ")
    if require_arrays and arrays == 0:
        raise base.common.WorkloadError(
            "C4 structure material manifest has no arrays"
        )
    return arrays, bytes_total, susceptibility_arrays


def _validate_material_rank_distribution(
    rank_records: list[dict[str, Any]],
) -> dict[str, int]:
    if not isinstance(rank_records, list) or not rank_records:
        raise base.common.WorkloadError("C4 rank inventory differs")
    totals = {
        "material_array_count": 0,
        "material_array_bytes": 0,
        "susceptibility_array_count": 0,
    }
    structure_array_count = 0
    structure_susceptibility_count = 0
    for rank_record in rank_records:
        if not isinstance(rank_record, dict):
            raise base.common.WorkloadError("C4 rank record differs")
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != 2:
            raise base.common.WorkloadError("C4 phase inventory differs")
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise base.common.WorkloadError("C4 phase record differs")
            array_count, byte_count, susceptibility_count = (
                _validate_material_manifest(
                    record.get("candidate_material_manifest"),
                    require_arrays=(
                        len(rank_records) == 1 and record_index == 1
                    ),
                )
            )
            totals["material_array_count"] += array_count
            totals["material_array_bytes"] += byte_count
            totals["susceptibility_array_count"] += susceptibility_count
            if record_index == 1:
                structure_array_count += array_count
                structure_susceptibility_count += susceptibility_count
    if structure_array_count == 0:
        raise base.common.WorkloadError(
            "C4 structure material manifest has no arrays across ranks"
        )
    if structure_susceptibility_count == 0:
        raise base.common.WorkloadError(
            "C4 AuNP structure manifest has no susceptibility arrays across ranks"
        )
    return totals


def _validate_feature_counters(
    rank_records: list[dict[str, Any]], mode: str
) -> dict[str, int]:
    totals = {name: 0 for name in COUNTERS}
    totals.update({name: 0 for name in MAX_GAUGES})
    totals.update({name: 0 for name in FLAGS})
    totals.update(_validate_material_rank_distribution(rank_records))
    for rank_record in rank_records:
        records = rank_record.get("records")
        snapshots = []
        for record in records:
            snapshots.extend(
                (
                    record.get("candidate_statistics_before"),
                    record.get("candidate_statistics_after"),
                )
            )
        snapshots.append(rank_record.get("final_candidate_statistics"))
        for snapshot in snapshots:
            _validate_snapshot_schema(snapshot)

        generation = _counter(snapshots[0], "reset_generation")
        if generation == 0 or any(
            _counter(snapshot, "reset_generation") != generation
            for snapshot in snapshots
        ):
            raise base.common.WorkloadError(
                "C4 reset generation changed during one sample"
            )
        for previous, current in zip(snapshots, snapshots[1:]):
            for name in COUNTERS + MAX_GAUGES:
                if _counter(current, name) < _counter(previous, name):
                    raise base.common.WorkloadError(
                        f"C4 statistic decreased: {name}"
                    )
            if _flag(previous, "prism_preflight_complete") and not _flag(
                current, "prism_preflight_complete"
            ):
                raise base.common.WorkloadError(
                    "C4 prism completion flag regressed"
                )

        final = snapshots[-1]
        if _flag(final, "counter_overflow"):
            raise base.common.WorkloadError("C4 counter overflowed")
        if any(
            _counter(final, name)
            for name in (
                "guard_rejections",
                "stale_epoch_rejections",
                "non_openmp_checks",
                "team_mismatch_observations",
            )
        ):
            raise base.common.WorkloadError("C4 fail-closed rejection occurred")

        if mode == "enabled":
            if (
                _counter(final, "feature_checks") == 0
                or _counter(final, "enabled_checks")
                != _counter(final, "feature_checks")
                or _counter(final, "eligible_checks") == 0
                or _counter(final, "disabled_checks")
                or _counter(final, "unsafe_material_checks")
                or _counter(final, "epsilon_parallel_calls") == 0
                or _counter(final, "epsilon_parallel_points") == 0
                or _counter(final, "sigma_parallel_calls") == 0
                or _counter(final, "sigma_parallel_points") == 0
                or _counter(final, "team_observations") == 0
                or _counter(final, "configured_threads") < 2
                or _counter(final, "observed_threads") < 2
                or _counter(final, "prism_preflight_calls") != 1
                or not _flag(final, "prism_preflight_complete")
            ):
                raise base.common.WorkloadError(
                    "enabled C4 lane lacks exact static-material causality"
                )
        elif (
            _counter(final, "feature_checks") == 0
            or _counter(final, "disabled_checks")
            != _counter(final, "feature_checks")
            or _counter(final, "enabled_checks")
            or _counter(final, "eligible_checks")
            or _counter(final, "unsafe_material_checks")
            or any(
                _counter(final, name)
                for name in (
                    "epsilon_parallel_calls",
                    "epsilon_parallel_points",
                    "sigma_parallel_calls",
                    "sigma_parallel_points",
                    "team_observations",
                    "team_mismatch_observations",
                    "configured_threads",
                    "observed_threads",
                )
            )
            or _counter(final, "prism_preflight_calls") != 1
            or not _flag(final, "prism_preflight_complete")
        ):
            raise base.common.WorkloadError(
                "disabled C4 lane reported parallel material work"
            )

        for name in COUNTERS:
            totals[name] += _counter(final, name)
        for name in MAX_GAUGES:
            totals[name] = max(totals[name], _counter(final, name))
        for name in FLAGS:
            totals[name] += int(_flag(final, name))
    return totals


base.SCHEMA = SCHEMA
base.COMPLETE_SCHEMA = COMPLETE_SCHEMA
base.FEATURE_ENABLE = FEATURE_ENABLE
base.FEATURE_DISABLE = FEATURE_DISABLE
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base._feature_environment = _feature_environment
base._expected_feature_environment = _expected_feature_environment
base._reset_candidate_statistics = _reset_candidate_statistics
base._candidate_statistics = _candidate_statistics
base._candidate_material_manifest = _candidate_material_manifest
base._validate_feature_counters = _validate_feature_counters
base.__file__ = str(pathlib.Path(__file__).resolve())

validate_sample = base.validate_sample
parse_args = base.parse_args
main = base.main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.common.WorkloadError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"C4 AuNP sample FAILED: {error}", file=base.sys.stderr, flush=True)
        raise SystemExit(2)
