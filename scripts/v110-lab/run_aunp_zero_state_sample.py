#!/usr/bin/env python3
"""Run one provenance-bound development AuNP zero-state A/B sample.

This is deliberately separate from the sealed release workload adapter.  It
records the loaded development runtime bytes and makes no release or new
scientific-output claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import resource
import socket
import struct
import subprocess
import sys
import time
from typing import Any


REPO = pathlib.Path(__file__).resolve().parents[2]
USER_WORKLOADS = REPO / "scripts" / "user-workloads"
if str(USER_WORKLOADS) not in sys.path:
    sys.path.insert(0, str(USER_WORKLOADS))

import common  # noqa: E402
import run_aunp_performance_workload as performance  # noqa: E402


SCHEMA = "gpmeep-v110-m1-aunp-zero-state-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-zero-state-sample-complete-v1"
FEATURE_ENABLE = "MEEP_GPU_ENABLE_ZERO_STATE_LORENTZIAN_ELISION"
FEATURE_DISABLE = "MEEP_GPU_DISABLE_ZERO_STATE_LORENTZIAN_ELISION"
FEATURE_COUNTERS = (
    "cuda_zero_state_elision_enabled_calls",
    "cuda_zero_state_elision_enabled_points",
    "cuda_zero_state_elision_inspectable_calls",
    "cuda_zero_state_elision_inspectable_points",
)
CANDIDATE_FEATURE = "zero-state-lorentzian-elision"
LEGACY_CANDIDATE_FIELD_ALLOWED = True
VALID_FEATURE_MODES = ("enabled", "disabled")


def _candidate_feature_matches(record: dict[str, Any]) -> bool:
    default = CANDIDATE_FEATURE if LEGACY_CANDIDATE_FIELD_ALLOWED else None
    return record.get("candidate_feature", default) == CANDIDATE_FEATURE


def _reset_candidate_statistics(mp: Any) -> None:
    """Reset candidate-specific evidence outside the ordinary dispatch API."""


def _candidate_statistics(mp: Any) -> dict[str, Any] | None:
    """Return an optional separately versioned candidate statistics group."""
    return None


def _candidate_material_manifest(simulation: Any) -> dict[str, Any] | None:
    """Return optional candidate-specific raw material-array evidence."""
    del simulation
    return None


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPO,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise common.WorkloadError(
            "development sample Git probe failed: "
            + completed.stderr.strip()
        )
    return completed.stdout


def _require_source_boundary(expected_commit: str) -> None:
    if _git("rev-parse", "HEAD").strip() != expected_commit:
        raise common.WorkloadError("development sample source commit differs")
    if _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        raise common.WorkloadError("development sample source tree is not clean")


def _observable(values: Any, label: str) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype="<f8").reshape(-1)
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise common.WorkloadError(f"AuNP observable {label} is empty or non-finite")
    payload = array.tobytes(order="C")
    return {
        "count": int(array.size),
        "sha256_le_f64": hashlib.sha256(payload).hexdigest(),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sum": float(array.sum(dtype=np.float64)),
    }


def _scalar_observable(value: Any, label: str) -> dict[str, Any]:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise common.WorkloadError(f"AuNP scalar observable {label} is non-finite")
    return {
        "value": scalar,
        "sha256_le_f64": hashlib.sha256(struct.pack("<d", scalar)).hexdigest(),
    }


def _total_field_energy(simulation: Any) -> Any:
    fields = getattr(simulation, "fields", None)
    if fields is None or not hasattr(fields, "total_volume"):
        raise common.WorkloadError("AuNP simulation fields are unavailable")
    return simulation.field_energy_in_box(box=fields.total_volume())


def _expected_feature_environment(mode: str) -> dict[str, str | None]:
    return {
        FEATURE_ENABLE: "1" if mode == "enabled" else None,
        FEATURE_DISABLE: None if mode == "enabled" else "1",
    }


def _feature_environment(mode: str) -> dict[str, str | None]:
    expected = _expected_feature_environment(mode)
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise common.WorkloadError(
            f"zero-state feature environment differs: {observed!r}"
        )
    return observed


def _summary_feature_environment_matches(summary: dict[str, Any]) -> bool:
    mode = summary.get("feature_mode")
    return mode in VALID_FEATURE_MODES and summary.get(
        "feature_environment"
    ) == _expected_feature_environment(mode)


def _counter_value(group: Any, name: str) -> int:
    if not isinstance(group, dict):
        raise common.WorkloadError("polarization statistics group is absent")
    value = group.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise common.WorkloadError(f"invalid polarization counter {name}")
    return value


def _validate_feature_counters(
    rank_records: list[dict[str, Any]], mode: str
) -> dict[str, int]:
    aggregate = {
        "cuda_polarization_calls": 0,
        "cuda_polarization_points": 0,
        **{name: 0 for name in FEATURE_COUNTERS},
    }
    for rank_record in rank_records:
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != 2:
            raise common.WorkloadError("AuNP feature sample phase inventory differs")
        for run_index, record in enumerate(records):
            delta = record.get("statistics_delta")
            if not isinstance(delta, dict):
                raise common.WorkloadError("AuNP feature statistics delta is absent")
            group = delta.get("polarizations")
            calls = _counter_value(group, "cuda_polarization_calls")
            points = _counter_value(group, "cuda_polarization_points")
            feature = {name: _counter_value(group, name) for name in FEATURE_COUNTERS}
            if run_index == 0:
                if calls != 0 or points != 0 or any(feature.values()):
                    raise common.WorkloadError(
                        "reference phase unexpectedly updated polarization"
                    )
            elif calls == 0:
                if points != 0 or any(feature.values()):
                    raise common.WorkloadError(
                        "zero-work AuNP rank reported polarization points or feature work"
                    )
            elif mode == "enabled":
                if (
                    points <= 0
                    or feature[FEATURE_COUNTERS[0]] != calls
                    or feature[FEATURE_COUNTERS[1]] != points
                    or feature[FEATURE_COUNTERS[2]] != calls
                    or feature[FEATURE_COUNTERS[3]] != points
                ):
                    raise common.WorkloadError(
                        "enabled AuNP structure phase lacks exact finite-isotropic "
                        "zero-state causality"
                    )
            elif points <= 0 or any(feature.values()):
                raise common.WorkloadError(
                    "disabled AuNP structure phase reported feature work"
                )
            aggregate["cuda_polarization_calls"] += calls
            aggregate["cuda_polarization_points"] += points
            for name, value in feature.items():
                aggregate[name] += value
    if (
        aggregate["cuda_polarization_calls"] <= 0
        or aggregate["cuda_polarization_points"] <= 0
    ):
        raise common.WorkloadError(
            "AuNP structure phase has no aggregate CUDA polarization work"
        )
    if mode == "enabled" and (
        aggregate[FEATURE_COUNTERS[0]]
            != aggregate["cuda_polarization_calls"]
        or aggregate[FEATURE_COUNTERS[1]]
            != aggregate["cuda_polarization_points"]
        or aggregate[FEATURE_COUNTERS[2]]
            != aggregate["cuda_polarization_calls"]
        or aggregate[FEATURE_COUNTERS[3]]
            != aggregate["cuda_polarization_points"]
    ):
        raise common.WorkloadError(
            "enabled AuNP aggregate feature counters differ from CUDA work"
        )
    if mode == "disabled" and any(
        aggregate[name] for name in FEATURE_COUNTERS
    ):
        raise common.WorkloadError(
            "disabled AuNP aggregate reported feature work"
        )
    return aggregate


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or path.exists():
        raise common.WorkloadError("development sample output must be absent")
    parent = path.parent.resolve(strict=True)
    if path.parent != parent:
        raise common.WorkloadError("development sample output parent is not canonical")
    path.mkdir(mode=0o755)
    return path


def _collective_output(comm: Any, rank: int, requested: pathlib.Path) -> pathlib.Path:
    state = None
    if rank == 0:
        try:
            state = {"output": str(_safe_output(requested))}
        except Exception as exc:
            state = {"error": f"{type(exc).__name__}: {exc}"}
    state = comm.bcast(state, root=0)
    if not isinstance(state, dict) or "error" in state:
        message = state.get("error", "invalid collective output state") \
            if isinstance(state, dict) else "invalid collective output state"
        raise common.WorkloadError(message)
    output = pathlib.Path(state.get("output", ""))
    if output.resolve(strict=True) != output or not output.is_dir():
        raise common.WorkloadError("collective development output is invalid")
    return output


def _memory_snapshot(engine: Any, comm: Any, rank: int, stage: str,
                     required_gib: float) -> dict[str, Any]:
    state = None
    if rank == 0:
        try:
            snapshot = engine.memory_admission_snapshot()
            available = float(snapshot["effective_available_gib"])
            if available < required_gib:
                raise common.WorkloadError(
                    f"insufficient memory for {stage}: {available:.3f} GiB "
                    f"available, {required_gib:.3f} GiB required"
                )
            state = {"snapshot": common.json_safe(snapshot)}
        except Exception as exc:
            state = {"error": f"{type(exc).__name__}: {exc}"}
    state = comm.bcast(state, root=0)
    if "error" in state:
        raise common.WorkloadError(state["error"])
    return state["snapshot"]


def validate_sample(output: pathlib.Path) -> dict[str, Any]:
    output = pathlib.Path(output)
    with common.StableFile(output / "COMPLETE", "C2 sample COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise common.WorkloadError("C2 sample COMPLETE schema differs")
    record = complete.get("summary")
    if not isinstance(record, dict) or set(record) != {
        "path", "size_bytes", "sha256"
    }:
        raise common.WorkloadError("C2 sample summary record differs")
    with common.StableFile(
        output / "summary.json", "C2 sample summary", expected=record
    ) as stable:
        with stable.file_object() as handle:
            summary = json.load(handle)
    if (
        summary.get("schema") != SCHEMA
        or summary.get("state") != "PASS"
        or summary.get("development_only") is not True
        or summary.get("valid_for_release_claim") is not False
        or summary.get("expected_backend") != "cuda"
        or summary.get("feature_mode") not in VALID_FEATURE_MODES
        or not _summary_feature_environment_matches(summary)
        or not _candidate_feature_matches(summary)
        or summary.get("single_precision") is not True
        or summary.get("fixed_window_meep_time")
        != performance.FIXED_WINDOW_MEEP_TIME
        or summary.get("minimum_timesteps_per_phase")
        != performance.MINIMUM_TIMESTEPS_PER_PHASE
    ):
        raise common.WorkloadError("C2 sample summary contract differs")
    records = summary.get("rank_records")
    if not isinstance(records, list) or len(records) != summary.get("mpi_size"):
        raise common.WorkloadError("C2 sample rank inventory differs")
    if [rank.get("rank") for rank in records] != list(range(len(records))):
        raise common.WorkloadError("C2 sample rank ordering differs")
    if summary.get("phase_contract") != performance.PHASE_CONTRACT:
        raise common.WorkloadError("C2 sample phase contract differs")
    performance.validate_phase_contract(records, "cuda", performance.PHASE_CONTRACT)
    if summary.get("feature_counter_totals") != _validate_feature_counters(
        records, summary["feature_mode"]
    ):
        raise common.WorkloadError("C2 sample feature totals were not re-derived")
    for phase_index in range(2):
        observables = [rank["records"][phase_index]["observables"] for rank in records]
        if any(value != observables[0] for value in observables[1:]):
            raise common.WorkloadError("C2 sample observables differ between ranks")
        if len({rank["records"][phase_index]["timestep_delta"] for rank in records}) != 1:
            raise common.WorkloadError("C2 sample timestep counts differ between ranks")
    runtime = records[0].get("runtime_artifacts")
    if any(rank.get("runtime_artifacts") != runtime for rank in records[1:]):
        raise common.WorkloadError("C2 sample ranks loaded different runtime bytes")
    rss_by_rank = [int(rank.get("peak_host_rss_bytes", -1)) for rank in records]
    if any(value <= 0 for value in rss_by_rank):
        raise common.WorkloadError("C2 sample rank RSS is invalid")
    if (
        summary.get("peak_host_rss_bytes_by_rank") != rss_by_rank
        or summary.get("peak_host_rss_bytes_total") != sum(rss_by_rank)
        or summary.get("peak_host_rss_bytes") != max(rss_by_rank)
    ):
        raise common.WorkloadError("C2 sample host RSS was not re-derived")
    phase_wall_seconds = [
        max(float(rank["records"][index]["wall_seconds"]) for rank in records)
        for index in range(2)
    ]
    build_wall_seconds = [
        max(
            float(rank["records"][index]["build_wall_seconds"])
            for rank in records
        )
        for index in range(2)
    ]
    fdtd = sum(phase_wall_seconds)
    initialization = sum(build_wall_seconds)
    end_to_end = max(float(rank["process_wall_seconds"]) for rank in records)
    if (
        summary.get("phase_wall_seconds") != phase_wall_seconds
        or summary.get("build_wall_seconds") != build_wall_seconds
        or not math.isclose(
            float(summary.get("initialization_wall_seconds", -1.0)),
            initialization,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(summary.get("fdtd_wall_seconds", -1.0)),
            fdtd,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(summary.get("workload_end_to_end_seconds", -1.0)),
            end_to_end,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise common.WorkloadError("C2 sample timing was not re-derived")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--feature-mode", required=True, choices=VALID_FEATURE_MODES
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_source_boundary(args.expected_source_commit)
    feature_environment = _feature_environment(args.feature_mode)
    archive = common.verify_file(
        args.archive, common.AUNP_ARCHIVE_SHA256, "AuNP archive"
    )
    import meep as mp
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    output = _collective_output(comm, rank, args.output)
    if size not in (1, 2):
        raise common.WorkloadError("C2 sample requires one or two MPI ranks")
    if not mp.is_single_precision():
        raise common.WorkloadError("C2 sample requires FP32 Meep")
    mp.gpu.set_backend("cuda")
    if str(mp.gpu.active_backend) != "cuda":
        raise common.WorkloadError("C2 sample did not activate CUDA")
    runtime_artifacts = common.runtime_artifact_records(mp)

    preparation = None
    if rank == 0:
        try:
            package_root = common.extract_verified_aunp(archive, output / "input")
            preparation = {"package_root": str(package_root)}
        except Exception as exc:
            preparation = {"error": f"{type(exc).__name__}: {exc}"}
    preparation = comm.bcast(preparation, root=0)
    if "error" in preparation:
        raise common.WorkloadError(preparation["error"])
    comm.Barrier()

    package_root = pathlib.Path(preparation["package_root"])
    engine_path = package_root / "fdtd" / "meep" / "aunp_periodic_fdtd.py"
    sys.path.insert(0, str(engine_path.parent))
    sys.path.insert(0, str(package_root / "fdtd"))
    engine = performance._load_engine(engine_path)
    default_config = engine.serializable_config(engine.AppConfig())
    if default_config != common.AUNP_EXPECTED_DEFAULT_CONFIG:
        raise common.WorkloadError("C2 AuNP defaults differ from sealed input")
    cfg = dataclasses.replace(
        engine.AppConfig(),
        run_mode="simulate",
        output_dir=output / "ephemeral-engine-output",
        required_mpi_ranks=size,
        dft_minimum_run_time=common.AUNP_DFT_MINIMUM_RUN_TIME,
        dft_decay_tolerance=common.AUNP_DFT_DECAY_TOLERANCE,
    )
    cfg.validate()
    engine.validate_runtime_resources(cfg)

    records: list[dict[str, Any]] = []
    reference_flux: dict[str, dict[str, Any]] = {}
    process_started = time.perf_counter()
    previous_cwd = pathlib.Path.cwd()
    mp.gpu.reset_statistics()
    _reset_candidate_statistics(mp)
    try:
        os.chdir(output)
        for run_index, (stage, polarization, structure, monitor_kind) in enumerate(
            performance.PHASE_SPECS
        ):
            required_gib = float(
                cfg.structure_memory_gib if structure else cfg.reference_memory_gib
            )
            admission = _memory_snapshot(
                engine, comm, rank, stage, required_gib
            )
            phase_started = time.perf_counter()
            comm.Barrier()
            build_started = time.perf_counter()
            simulation, monitors = engine.build_simulation(
                cfg,
                polarization,
                include_structure=structure,
                field_monitor_kind=monitor_kind,
            )
            comm.Barrier()
            build_wall_seconds = time.perf_counter() - build_started
            if structure:
                saved = reference_flux.pop(polarization, None)
                if not isinstance(saved, dict) or set(saved) != {
                    "monitor1", "monitor2"
                }:
                    raise common.WorkloadError("C2 reference flux is absent")
                simulation.load_minus_flux_data(monitors.monitor1, saved["monitor1"])
                simulation.load_minus_flux_data(monitors.monitor2, saved["monitor2"])
            before_timestep = int(simulation.fields.t) if simulation.fields else 0
            before_stats = common.json_safe(mp.gpu.statistics())
            before_candidate_stats = common.json_safe(
                _candidate_statistics(mp)
            )
            comm.Barrier()
            fdtd_started = time.perf_counter()
            simulation.run(until=performance.FIXED_WINDOW_MEEP_TIME)
            comm.Barrier()
            wall_seconds = time.perf_counter() - fdtd_started
            after_stats = common.json_safe(mp.gpu.statistics())
            after_candidate_stats = common.json_safe(
                _candidate_statistics(mp)
            )
            delta = common.statistics_delta(before_stats, after_stats)
            observables = {
                "monitor1_flux": _observable(
                    mp.get_fluxes(monitors.monitor1), "monitor1 flux"
                ),
                "monitor2_flux": _observable(
                    mp.get_fluxes(monitors.monitor2), "monitor2 flux"
                ),
                "bottom_flux": _observable(
                    mp.get_fluxes(monitors.bottom_flux), "bottom flux"
                ),
                "field_energy": _scalar_observable(
                    _total_field_energy(simulation), "field energy"
                ),
            }
            record = {
                "rank": rank,
                "run_index": run_index,
                "stage": stage,
                "timestep": int(simulation.timestep()),
                "timestep_delta": int(simulation.timestep()) - before_timestep,
                "meep_time": float(simulation.meep_time()),
                "build_wall_seconds": build_wall_seconds,
                "wall_seconds": wall_seconds,
                "phase_process_seconds_before_reset": time.perf_counter()
                - phase_started,
                "active_backend": str(mp.gpu.active_backend),
                "requested_backend": str(mp.gpu.requested_backend),
                "device_ordinal": int(mp.gpu.selected_device),
                "device_identifier": str(mp.gpu.selected_device_identifier),
                "memory_admission": admission,
                "phase_calls": common.phase_call_totals(delta),
                "phase_counters": common.phase_counter_view(delta),
                "statistics_before": before_stats,
                "statistics_after": after_stats,
                "statistics_delta": delta,
                "candidate_statistics_before": before_candidate_stats,
                "candidate_statistics_after": after_candidate_stats,
                "observables": observables,
            }
            candidate_material_manifest = _candidate_material_manifest(simulation)
            if candidate_material_manifest is not None:
                record["candidate_material_manifest"] = common.json_safe(
                    candidate_material_manifest
                )
            performance._validate_backend_record(record, "cuda")
            if not structure:
                reference_flux[polarization] = {
                    "monitor1": simulation.get_flux_data(monitors.monitor1),
                    "monitor2": simulation.get_flux_data(monitors.monitor2),
                }
            simulation.reset_meep()
            comm.Barrier()
            records.append(record)
    finally:
        os.chdir(previous_cwd)
    comm.Barrier()
    process_wall_seconds = time.perf_counter() - process_started
    peak_host_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "records": records,
        "runtime_artifacts": runtime_artifacts,
        "physical_cores": common.physical_core_affinity(),
        "process_wall_seconds": process_wall_seconds,
        "peak_host_rss_bytes": peak_host_rss_bytes,
        "final_statistics": common.json_safe(mp.gpu.statistics()),
        "final_candidate_statistics": common.json_safe(
            _candidate_statistics(mp)
        ),
    }
    gathered = comm.gather(local, root=0)
    comm.Barrier()

    publication_error = None
    if rank == 0:
        try:
            performance.validate_phase_contract(
                gathered, "cuda", performance.PHASE_CONTRACT
            )
            feature_totals = _validate_feature_counters(
                gathered, args.feature_mode
            )
            for phase_index in range(2):
                expected = gathered[0]["records"][phase_index]["observables"]
                if any(
                    rank_record["records"][phase_index]["observables"] != expected
                    for rank_record in gathered[1:]
                ):
                    raise common.WorkloadError(
                        "C2 observables differ between MPI ranks"
                    )
            phase_wall_seconds = [
                max(
                    float(rank_record["records"][index]["wall_seconds"])
                    for rank_record in gathered
                )
                for index in range(2)
            ]
            build_wall_seconds = [
                max(
                    float(rank_record["records"][index]["build_wall_seconds"])
                    for rank_record in gathered
                )
                for index in range(2)
            ]
            summary = {
                "schema": SCHEMA,
                "state": "PASS",
                "development_only": True,
                "valid_for_release_claim": False,
                "feature_mode": args.feature_mode,
                "candidate_feature": CANDIDATE_FEATURE,
                "feature_environment": feature_environment,
                "expected_backend": "cuda",
                "source_commit": args.expected_source_commit,
                "implementation_commit": args.implementation_commit,
                "archive": common.absolute_file_record(archive, "AuNP archive"),
                "archive_sha256": common.AUNP_ARCHIVE_SHA256,
                "engine": common.file_record(engine_path, output),
                "worker": common.file_record(pathlib.Path(__file__), REPO),
                "single_precision": bool(mp.is_single_precision()),
                "meep_version": str(getattr(mp, "__version__", "unknown")),
                "fixed_window_meep_time": performance.FIXED_WINDOW_MEEP_TIME,
                "minimum_timesteps_per_phase":
                    performance.MINIMUM_TIMESTEPS_PER_PHASE,
                "qualification_profile": common.aunp_qualification_profile(),
                "mpi_size": size,
                "rank_records": gathered,
                "phase_contract": performance.PHASE_CONTRACT,
                "feature_counter_totals": feature_totals,
                "phase_wall_seconds": phase_wall_seconds,
                "build_wall_seconds": build_wall_seconds,
                "initialization_wall_seconds": sum(build_wall_seconds),
                "fdtd_wall_seconds": sum(phase_wall_seconds),
                "workload_end_to_end_seconds": max(
                    float(rank_record["process_wall_seconds"])
                    for rank_record in gathered
                ),
                "peak_host_rss_bytes_by_rank": [
                    int(rank_record["peak_host_rss_bytes"])
                    for rank_record in gathered
                ],
                "peak_host_rss_bytes_total": sum(
                    int(rank_record["peak_host_rss_bytes"])
                    for rank_record in gathered
                ),
                "peak_host_rss_bytes": max(
                    int(rank_record["peak_host_rss_bytes"])
                    for rank_record in gathered
                ),
                "scientific_output_claims": [],
            }
            common.atomic_write_json(output / "summary.json", summary)
            common.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": COMPLETE_SCHEMA,
                    "summary": common.file_record(output / "summary.json", output),
                },
            )
            validate_sample(output)
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = comm.bcast(publication_error, root=0)
    if publication_error:
        raise common.WorkloadError(publication_error)
    comm.Barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (common.WorkloadError, OSError, ValueError) as error:
        print(f"C2 AuNP sample FAILED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
