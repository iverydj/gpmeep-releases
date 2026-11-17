#!/usr/bin/env python3
"""Load and expand the sealed gpmeep v1.1.0 M4 equal-work matrix."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN_RELATIVE = pathlib.Path("packaging/v110-m4-equal-work-plan.json")
EXPECTED_LANES = (
    "upstream_meep_cpu",
    "gpmeep_v110_cpu",
    "gpmeep_v103_cuda_gpu1",
    "gpmeep_v110_cuda_gpu1",
    "gpmeep_v103_cuda_gpu2",
    "gpmeep_v110_cuda_gpu2",
)
GPU2_LANES = {
    "gpmeep_v103_cuda_gpu2",
    "gpmeep_v110_cuda_gpu2",
}
FROZEN_V103 = {
    "package_name": "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",
    "package_sha256": "fd834ca51019ce2f18b26cb1f7252933205e008cdf1b2d9598b54e58ea37ebf3",
    "provenance_report_sha256": "c8598f275ce328d9c02a971061e29fc2864f6707a49f39cda0d3c7901158a0cd",
    "provenance_report_id": "6c5d6e8266a33f8b6b85fa8b5b54c539e5519d74a0ed0cc58eb25b20e0f1cb17",
    "source_commit": "b48c9b678f2316cefa95e48a661f7a877547a000",
    "rebuild_permitted": False,
    "fresh_dedicated_prefix_required": True,
}
PHASE_IN_MATERIAL_CASE = "python/examples/phase_in_material.py"
PHASE_IN_MATERIAL_FIXED_WORK = {
    "environment": "GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK",
    "value": "1",
    "phase_time": 200.0,
    "run_count": 2,
    "final_timestep": 4000,
    "total_timestep_delta": 8000,
    "purpose": "stable timing and one-second external GPU process observability",
}
EXPECTED_CYCLES = (
    (0, "warmup", "forward", "forward"),
    (1, "measured", "forward", "forward"),
    (2, "measured", "reverse", "reverse"),
)
LANE_RUNTIME = {
    "upstream_meep_cpu": ("upstream", "cpu"),
    "gpmeep_v110_cpu": ("v110", "cpu"),
    "gpmeep_v103_cuda_gpu1": ("v103", "cuda"),
    "gpmeep_v110_cuda_gpu1": ("v110", "cuda"),
    "gpmeep_v103_cuda_gpu2": ("v103", "cuda"),
    "gpmeep_v110_cuda_gpu2": ("v110", "cuda"),
}


class M4PlanError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M4PlanError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"M4 plan duplicates key {key!r}")
        result[key] = value
    return result


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(repo: pathlib.Path = ROOT) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    path = (repo / PLAN_RELATIVE).resolve(strict=True)
    try:
        plan = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                M4PlanError(f"M4 plan has non-finite constant {value!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M4PlanError(f"cannot load M4 equal-work plan: {exc}") from exc
    _require(isinstance(plan, dict), "M4 plan root must be an object")
    _require(
        plan.get("schema") == "gpmeep-v110-m4-equal-work-plan-v1"
        and plan.get("release_train") == "1.1.0",
        "M4 plan identity differs",
    )
    _require(
        plan.get("frozen_v103") == FROZEN_V103,
        "M4 frozen-v1.0.3 authority differs",
    )
    contract = json.loads(
        (repo / "packaging/v110-development-contract.json").read_text(
            encoding="utf-8"
        )
    )
    binding = contract.get("performance_inversion_policy", {}).get(
        "equal_work_plan", {}
    )
    _require(
        binding
        == {"path": PLAN_RELATIVE.as_posix(), "sha256": _sha256(path)},
        "M4 plan is not byte-bound by the development contract",
    )
    source = plan.get("source_policy")
    _require(
        source
        == {
            "entrypoint_bytes": "exact-v1.1.0-candidate-commit",
            "same_entrypoint_bytes_for_every_runtime": True,
            "same_arguments_physics_precision_and_stopping_condition": True,
            "precision": "fp32",
            "profiler_free_primary_wall_time": True,
        },
        "M4 equal-work source policy differs",
    )
    lanes = plan.get("lanes")
    _require(
        isinstance(lanes, list)
        and tuple(item.get("id") for item in lanes) == EXPECTED_LANES,
        "M4 lane order differs",
    )
    for lane in lanes:
        lane_id = lane["id"]
        runtime, backend = LANE_RUNTIME[lane_id]
        expected_runtime = {
            "upstream": "upstream_meep",
            "v103": "gpmeep_v103_frozen",
            "v110": "gpmeep_v110_candidate",
        }[runtime]
        _require(
            lane.get("runtime") == expected_runtime
            and lane.get("backend") == backend,
            f"M4 lane runtime differs: {lane_id}",
        )
        if lane_id in GPU2_LANES:
            _require(
                lane.get("topology") == "gpu2"
                and lane.get("only_when_gpu2_relevant") is True,
                f"M4 GPU2 lane differs: {lane_id}",
            )
        else:
            _require(
                set(lane) == {"id", "runtime", "backend", "topology"},
                f"M4 ordinary lane schema differs: {lane_id}",
            )
    cycles = plan.get("cycles")
    _require(
        isinstance(cycles, list)
        and tuple(
            (
                item.get("index"),
                item.get("kind"),
                item.get("case_order"),
                item.get("lane_order"),
            )
            for item in cycles
        )
        == EXPECTED_CYCLES,
        "M4 cycle/reversal contract differs",
    )
    cases = plan.get("cases")
    _require(isinstance(cases, list) and len(cases) == 13, "M4 case count differs")
    paths = [item.get("path") for item in cases]
    _require(
        all(isinstance(path, str) and (repo / path).is_file() for path in paths),
        "M4 case source inventory differs",
    )
    _require(len(set(paths)) == len(paths), "M4 case paths repeat")
    for case in cases:
        expected_keys = {"path", "cpu_ranks", "gpu2", "timing"}
        if case.get("path") == PHASE_IN_MATERIAL_CASE:
            expected_keys.add("fixed_work")
        _require(
            set(case) == expected_keys
            and case["cpu_ranks"] in {1, 8}
            and type(case["gpu2"]) is bool
            and case["timing"] == "sum-all-simulation-run-wall-seconds",
            f"M4 case contract differs: {case.get('path')}",
        )
        if case.get("path") == PHASE_IN_MATERIAL_CASE:
            _require(
                case.get("fixed_work") == PHASE_IN_MATERIAL_FIXED_WORK,
                "M4 phase_in_material fixed-work contract differs",
            )
    tasks = build_tasks(plan)
    counts = plan.get("execution_counts")
    _require(
        counts == {"per_cycle": 60, "all_cycles": 180, "measured": 120}
        and len(tasks) == counts["all_cycles"]
        and sum(task["sample_kind"] == "measured" for task in tasks)
        == counts["measured"],
        "M4 execution census differs",
    )
    reporting = plan.get("reporting")
    _require(
        isinstance(reporting, dict)
        and reporting.get("measured_repetitions") == 2
        and reporting.get("retain_raw_seconds") is True
        and reporting.get("keep_upstream_and_v110_cpu_distinct") is True
        and reporting.get("unavailable_phase_seconds_are_null") is True
        and reporting.get("numerical_oracle_required") is True
        and reporting.get("zero_cpu_fallback_required_for_cuda") is True,
        "M4 reporting contract differs",
    )
    return plan


def _lane_applies(lane_id: str, case: dict[str, Any]) -> bool:
    return lane_id not in GPU2_LANES or case["gpu2"] is True


def topology(lane_id: str, case: dict[str, Any]) -> str:
    if lane_id == "upstream_meep_cpu":
        return "upstream-cpu1" if case["cpu_ranks"] == 1 else "upstream-cpu8"
    if lane_id == "gpmeep_v110_cpu":
        return "cpu1" if case["cpu_ranks"] == 1 else "cpu8"
    if lane_id == "gpmeep_v103_cuda_gpu1":
        return "v103-cuda1"
    if lane_id == "gpmeep_v110_cuda_gpu1":
        return "cuda1"
    if lane_id == "gpmeep_v103_cuda_gpu2":
        return "v103-cuda2"
    if lane_id == "gpmeep_v110_cuda_gpu2":
        return "cuda2"
    raise M4PlanError(f"unknown M4 lane {lane_id!r}")


def build_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = [item["id"] for item in plan["lanes"]]
    cases = list(plan["cases"])
    tasks = []
    for cycle in plan["cycles"]:
        ordered_cases = cases if cycle["case_order"] == "forward" else list(reversed(cases))
        ordered_lanes = lanes if cycle["lane_order"] == "forward" else list(reversed(lanes))
        for case in ordered_cases:
            for lane_id in ordered_lanes:
                if not _lane_applies(lane_id, case):
                    continue
                runtime, backend = LANE_RUNTIME[lane_id]
                task = {
                    "index": len(tasks),
                    "cycle_index": cycle["index"],
                    "sample_kind": cycle["kind"],
                    "case_path": case["path"],
                    "lane_id": lane_id,
                    "runtime": runtime,
                    "backend": backend,
                    "topology": topology(lane_id, case),
                    "timing_contract": case["timing"],
                }
                task["task_id"] = hashlib.sha256(
                    json.dumps(
                        task, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                tasks.append(task)
    return tasks


__all__ = [
    "EXPECTED_CYCLES",
    "EXPECTED_LANES",
    "FROZEN_V103",
    "GPU2_LANES",
    "LANE_RUNTIME",
    "M4PlanError",
    "PHASE_IN_MATERIAL_CASE",
    "PHASE_IN_MATERIAL_FIXED_WORK",
    "build_tasks",
    "load_plan",
    "topology",
]
