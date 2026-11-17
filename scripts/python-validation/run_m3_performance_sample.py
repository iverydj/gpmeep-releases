#!/usr/bin/env python3
"""Run and seal one M3 repeated-performance example/topology sample."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
import threading
import time
import uuid
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import gpmeep_provenance as provenance  # noqa: E402
import run_m3_auto_example_case as auto_case  # noqa: E402
import run_m3_mpi_example_case as mpi_case  # noqa: E402
import run_validation as validation  # noqa: E402
import v110_runtime_identity as v110_identity  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    durable_publish_terminal,
    file_record,
)


SCHEMA = "gpmeep-m3-performance-sample-v1"
COMPLETE_SCHEMA = "gpmeep-m3-performance-sample-complete-v1"
TOPOLOGIES = ("cpu8", "cpu1", "cuda1", "cuda2", "auto1")
SAMPLE_KINDS = ("warmup", "measured")
BACKEND_MISMATCH_REASON = "M3 hardware crossover classification run"
ADJOINT_CYL_FIXED_WORK_ENV = "GPMEEP_ADJOINT_CYL_FIXED_WORK"
PHASE_IN_MATERIAL_FIXED_WORK_ENV = "GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK"
PHASE_IN_MATERIAL_CASE = "python/examples/phase_in_material.py"
PHASE_IN_MATERIAL_FIXED_FINAL_TIMESTEP = 4000
PHASE_IN_MATERIAL_FIXED_TOTAL_TIMESTEP_DELTA = 8000
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
INVERSION_CASE_ORDER = (
    PHASE_IN_MATERIAL_CASE,
    "python/examples/material-dispersion.py",
    "python/examples/refl-quartz.py",
    "python/examples/waveguide_crossing.py",
    "python/examples/adjoint_optimization/multilayer_opt.py",
    "python/examples/absorber-1d.py",
    "python/examples/parallel-wvgs-force.py",
    "python/examples/antenna_pec_ground_plane_1D.py",
    "python/examples/3rd-harm-1d.py",
    "python/examples/adjoint_optimization/near2far_3d_validation.py",
    "python/examples/edge_emitter_3D.py",
    "python/examples/dipole_in_vacuum_cyl_off_axis.py",
    "python/tests/test_adjoint_cyl.py",
)
INVERSION_CASES = frozenset(INVERSION_CASE_ORDER)
CASE_TIMING = {
    "python/examples/phase_in_material.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/material-dispersion.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/refl-quartz.py": "sum-all-simulation-run-wall-seconds",
    "python/examples/absorber-1d.py": "sum-all-simulation-run-wall-seconds",
    "python/examples/parallel-wvgs-force.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/antenna_pec_ground_plane_1D.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/3rd-harm-1d.py": "sum-all-simulation-run-wall-seconds",
    "python/examples/waveguide_crossing.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/adjoint_optimization/multilayer_opt.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/adjoint_optimization/near2far_3d_validation.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/dipole_in_vacuum_cyl_off_axis.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/edge_emitter_3D.py": "sum-all-simulation-run-wall-seconds",
    "python/tests/test_adjoint_cyl.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
    "python/examples/metasurface_lens.py": (
        "run-index-18-small-and-19-large-wall-seconds"
    ),
    "python/examples/stochastic_emitter_line.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
}
# These four inventory entries deliberately retain their production validation
# entry points.  The generic oracle wraps them only to observe Simulation.run;
# it does not replace their embedded numerical assertions or mutate their
# timestep/stopping arguments.  The exact run counts follow from the sealed
# validation slices: forward+adjoint plus independent forward differences.
DIRECT_PERFORMANCE_ORACLE_CONTRACTS = {
    "python/examples/waveguide_crossing.py": {
        "manifest_mode": "explicit-example",
        "example_args": ("--validation",),
        "expected_run_count": 20,
        "disable_field_timing_snapshots": True,
    },
    "python/examples/adjoint_optimization/multilayer_opt.py": {
        "manifest_mode": "explicit-example",
        "example_args": ("--validation",),
        "expected_run_count": 3,
    },
    "python/examples/adjoint_optimization/near2far_3d_validation.py": {
        "manifest_mode": "default-example",
        "example_args": (),
        "expected_run_count": 10,
    },
    "python/tests/test_adjoint_cyl.py": {
        "manifest_mode": "default-unittest",
        "example_args": (),
        "expected_run_count": 16,
        "expected_unittest_count": 4,
    },
}
PHASE_NAMES = (
    "initialization",
    "fdtd_timestep",
    "source",
    "boundary",
    "dispersive_ade",
    "dft_reduction",
    "host_device_transfer_and_sync",
    "mpi",
    "python_adjoint_mpb",
    "io",
)
PHASE_TIMING_SINKS = {
    "fdtd_timestep": ("time_stepping",),
    "boundary": ("boundaries",),
    "dft_reduction": ("fourier_transform", "get_farfields"),
    "mpi": ("connecting_chunks", "mpi_all_to_all", "mpi_one_to_one"),
    "python_adjoint_mpb": ("mpb",),
    "io": ("field_output",),
}
REPORT_KEYS = {
    "schema",
    "outcome",
    "case_path",
    "topology",
    "sample_kind",
    "cycle_index",
    "gpu_devices",
    "receipt_id",
    "paths",
    "runtime_snapshot",
    "lane",
    "timing",
    "phase_profile",
    "evidence_code",
}
LANE_KEYS = {
    "spec",
    "requested_backend",
    "run_nonce",
    "command",
    "environment_sha256",
    "process",
    "stdout",
    "stderr",
    "rank_records",
    "tree_manifest",
    "policy_result",
}


def validate_comparison_evidence(
    text: str, comparison: dict[str, Any]
) -> None:
    """Replay only the evidence check required by the comparison mode."""
    mode = comparison.get("mode")
    if mode == "json_metrics":
        validation.extract_json_metrics(text, comparison)
        return
    if mode == "embedded_oracle":
        # Process success, the pinned unittest identities, and every observed
        # Simulation.run record are validated independently by this runner.
        return
    raise WorkloadError(
        f"unsupported M3 performance comparison mode: {mode!r}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--topology", required=True, choices=TOPOLOGIES)
    parser.add_argument("--sample-kind", required=True, choices=SAMPLE_KINDS)
    parser.add_argument("--cycle-index", required=True, type=int)
    parser.add_argument("--python", required=True, type=pathlib.Path)
    parser.add_argument("--build-python", required=True, type=pathlib.Path)
    parser.add_argument("--install-prefix", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--mpiexec", required=True, type=pathlib.Path)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        help="exact v1.1.0 candidate package; selects installed-package mode",
    )
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def validate_sample_identity(kind: str, cycle: int) -> None:
    if kind not in SAMPLE_KINDS or type(cycle) is not int:
        raise WorkloadError("M3 performance sample identity differs")
    if (kind == "warmup" and cycle != 0) or (kind == "measured" and cycle < 1):
        raise WorkloadError("M3 performance sample cycle differs")


def validate_inversion_inventory(repo: pathlib.Path) -> None:
    inventory = validation.load_json(
        repo / "packaging/v110-performance-inversion-inventory.json"
    )
    observations = inventory.get("observations")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("immutable") is not True
        or not isinstance(observations, list)
        or not all(isinstance(item, dict) for item in observations)
    ):
        raise WorkloadError("M3 performance inversion inventory differs")
    workload_order = tuple(item.get("workload_id") for item in observations)
    if workload_order != INVERSION_CASE_ORDER:
        raise WorkloadError("M3 performance inversion inventory order differs")


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(
        mpi_case.normalized_gpu(item) for item in value.split(",") if item
    )
    identities = {mpi_case._identity(item) for item in devices}
    if len(devices) != 2 or len(identities) != 2:
        raise WorkloadError("M3 performance sample requires two distinct GPUs")
    return devices  # type: ignore[return-value]


def select_lane(
    topology: str,
    devices: tuple[str, str],
    case_path: str,
) -> tuple[mpi_case.Lane, str, dict[str, Any]]:
    strict_lanes = dict(
        zip(
            ("cpu8", "cuda1", "cuda2"),
            mpi_case.build_lanes(8, devices),
            strict=True,
        )
    )
    strict_lanes["cpu1"] = mpi_case.Lane("cpu-fp32-1r", "cpu", 1, ())
    if topology in strict_lanes:
        lane = strict_lanes[topology]
        policy = {
            "schema": "gpmeep-m3-forced-backend-policy-v1",
            "classification": "forced-backend-performance-sample",
            "active_backends": None,
        }
        return lane, lane.backend, policy
    if topology != "auto1":
        raise WorkloadError("M3 performance sample topology differs")
    policy = auto_case.policy_for(case_path)
    lane = mpi_case.Lane(
        auto_case.LANE_NAME,
        policy["active_backends"][-1],
        1,
        devices[:1],
    )
    return lane, "auto", policy


def sample_environment(
    lane: mpi_case.Lane,
    requested_backend: str,
    *,
    case_path: str,
    output: pathlib.Path,
    python: pathlib.Path,
    build_python: pathlib.Path,
    runtime_contract: dict[str, Any],
    run_nonce: str,
) -> dict[str, str]:
    environment = mpi_case.clean_environment(
        lane,
        python=python,
        build_python=build_python,
        runtime_contract=runtime_contract,
        run_nonce=run_nonce,
        site_guard_expected=False,
        producer_safe_ambient=mpi_case.expected_performance_producer_safe_ambient(
            output, python
        ),
    )
    if requested_backend == "auto":
        environment.update(
            {
                "MEEP_GPU_BACKEND": "auto",
                "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "auto",
            }
        )
        environment.pop("GPMEEP_VALIDATION_STRICT_CUDA", None)
    if case_path == "python/tests/test_adjoint_cyl.py":
        environment[ADJOINT_CYL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(ADJOINT_CYL_FIXED_WORK_ENV, None)
    if case_path == PHASE_IN_MATERIAL_CASE:
        environment[PHASE_IN_MATERIAL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(PHASE_IN_MATERIAL_FIXED_WORK_ENV, None)
    return environment


def _phase_in_material_fixed_work_command(command: list[str]) -> list[str]:
    replacements = {
        "--expected-final-timestep": (
            "40",
            str(PHASE_IN_MATERIAL_FIXED_FINAL_TIMESTEP),
        ),
        "--expected-total-timestep-delta": (
            "80",
            str(PHASE_IN_MATERIAL_FIXED_TOTAL_TIMESTEP_DELTA),
        ),
    }
    transformed = list(command)
    for option, (expected, replacement) in replacements.items():
        indices = [
            index for index, value in enumerate(transformed) if value == option
        ]
        if len(indices) != 1 or indices[0] + 1 >= len(transformed):
            raise WorkloadError(
                f"phase_in_material performance command lacks exact {option}"
            )
        value_index = indices[0] + 1
        if transformed[value_index] != expected:
            raise WorkloadError(
                f"phase_in_material performance command changed {option}"
            )
        transformed[value_index] = replacement
    return transformed


def performance_oracle_command(
    case: dict[str, Any],
    python: pathlib.Path,
    repo: pathlib.Path,
) -> list[str]:
    command = validation.case_command(case, python, repo)
    oracle = (repo / "scripts/python-validation/run_example_oracle.py").resolve(
        strict=True
    )
    oracle_indices = [
        index
        for index, value in enumerate(command)
        if pathlib.Path(value).name == "run_example_oracle.py"
    ]
    if oracle_indices:
        if (
            len(oracle_indices) != 1
            or pathlib.Path(command[oracle_indices[0]]).resolve(strict=True) != oracle
            or "--acknowledge-backend-mismatch-reason" in command
            or "--upstream-cpu-baseline" in command
        ):
            raise WorkloadError("M3 generic-oracle command differs")
        if case.get("path") == PHASE_IN_MATERIAL_CASE:
            return _phase_in_material_fixed_work_command(command)
        return command

    path = case.get("path")
    try:
        contract = DIRECT_PERFORMANCE_ORACLE_CONTRACTS[path]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(
            f"M3 performance case does not use a sealed oracle route: {path}"
        ) from exc
    mode = contract["manifest_mode"]
    example_args = tuple(contract["example_args"])
    entrypoint = (repo / path).resolve(strict=True)
    if mode == "explicit-example":
        expected_template = ["{python}", f"{{repo}}/{path}", *example_args]
        if case.get("kind") != "example" or case.get("command") != expected_template:
            raise WorkloadError(f"M3 direct example command differs: {path}")
        expected_command = [str(python), str(entrypoint), *example_args]
    elif mode == "default-example":
        if case.get("kind") != "example" or "command" in case or example_args:
            raise WorkloadError(f"M3 default example command differs: {path}")
        expected_command = [str(python), str(entrypoint)]
    elif mode == "default-unittest":
        unittest_contract = case.get("expected_unittest")
        if (
            case.get("kind") != "unittest"
            or "command" in case
            or example_args
            or not isinstance(unittest_contract, dict)
            or unittest_contract.get("count")
            != contract["expected_unittest_count"]
        ):
            raise WorkloadError(f"M3 unittest command differs: {path}")
        expected_command = [
            str(python),
            "-m",
            "unittest",
            "discover",
            "-v",
            "-s",
            str(entrypoint.parent),
            "-p",
            entrypoint.name,
        ]
    else:
        raise WorkloadError(f"M3 direct oracle mode differs: {path}")
    if command != expected_command:
        raise WorkloadError(f"M3 source command differs: {path}")
    oracle_command = [
        str(python),
        str(oracle),
        "--expected-run-count",
        str(contract["expected_run_count"]),
        "--min-field-energy",
        "1e-30",
    ]
    if contract.get("disable_field_timing_snapshots", False):
        oracle_command.append("--disable-field-timing-snapshots")
    if mode == "default-unittest":
        oracle_command.append("--unittest-discover")
    return [*oracle_command, str(entrypoint), "--", *example_args]


def case_command_for_lane(
    case: dict[str, Any],
    python: pathlib.Path,
    repo: pathlib.Path,
    requested_backend: str,
) -> list[str]:
    command = performance_oracle_command(case, python, repo)
    if requested_backend in {"cpu", "cuda"}:
        owner_index = validation.backend_guard_owner_index(command)
        if owner_index is None:
            raise WorkloadError("M3 performance command has no backend guard owner")
        command = [
            *command[: owner_index + 1],
            "--acknowledgement-scope",
            "process",
            *command[owner_index + 1 :],
        ]
        return validation.command_with_backend_acknowledgement(
            command, BACKEND_MISMATCH_REASON
        )
    return command


def derive_timing(
    case_path: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        contract = CASE_TIMING[case_path]
    except KeyError as exc:
        raise WorkloadError(
            f"M3 performance sample case is not sealed: {case_path}"
        ) from exc
    values = [float(record["run_wall_seconds"]) for record in runs]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise WorkloadError("M3 performance sample timing values differ")
    if contract == "sum-all-simulation-run-wall-seconds":
        return {
            "contract": contract,
            "run_count": len(values),
            "run_wall_seconds": values,
            "primary_seconds": sum(values),
        }
    if contract == "run-index-18-small-and-19-large-wall-seconds":
        if len(values) != 20:
            raise WorkloadError("M3 metasurface timing requires exactly 20 runs")
        return {
            "contract": contract,
            "run_count": len(values),
            "run_wall_seconds": values,
            "small_run_index": 18,
            "small_seconds": values[18],
            "large_run_index": 19,
            "large_seconds": values[19],
            "primary_seconds": values[19],
        }
    raise WorkloadError("M3 performance sample timing contract differs")


def derive_phase_profile(
    runs: list[dict[str, Any]], process_seconds: float
) -> dict[str, Any]:
    if (
        isinstance(process_seconds, bool)
        or not isinstance(process_seconds, (int, float))
        or not math.isfinite(float(process_seconds))
        or process_seconds <= 0
    ):
        raise WorkloadError("M3 phase-profile process duration differs")
    run_seconds = [float(record["run_wall_seconds"]) for record in runs]
    if not run_seconds or any(
        not math.isfinite(value) or value <= 0 for value in run_seconds
    ):
        raise WorkloadError("M3 phase-profile run duration differs")
    phases: dict[str, dict[str, Any]] = {}
    outside_observed_runs = max(
        0.0, float(process_seconds) - math.fsum(run_seconds)
    )
    phases["initialization"] = {
        "seconds": None,
        "available": False,
        "basis": (
            "initialization-not-separable-from-import-python-orchestration-"
            "postprocessing-io-and-teardown"
        ),
        "native_sinks": [],
    }
    for phase_name in PHASE_NAMES[1:]:
        sinks = PHASE_TIMING_SINKS.get(phase_name, ())
        values = []
        available = bool(sinks)
        for record in runs:
            timing = record.get("field_timing_delta")
            if not isinstance(timing, dict):
                available = False
                break
            for sink in sinks:
                value = timing.get(sink, {}).get("maximum_seconds")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    available = False
                    break
                values.append(float(value))
            if not available:
                break
        phases[phase_name] = {
            # `0.0` means a measured zero-duration phase.  It must never be
            # used as a placeholder when the frozen runtimes expose no
            # comparable native timer for this phase.
            "seconds": math.fsum(values) if available else None,
            "available": available,
            "basis": (
                "sum-of-per-run-native-rank-maximum-timing-sinks"
                if available
                else "not-separable-with-native-Meep-timing-sinks"
            ),
            "native_sinks": list(sinks),
        }
    if tuple(phases) != PHASE_NAMES:
        raise WorkloadError("M3 phase-profile inventory differs")
    return {
        "schema": "gpmeep-m3-ten-phase-profile-v1",
        "values_are_non_additive": True,
        "run_wall_seconds": math.fsum(run_seconds),
        "process_wall_seconds": float(process_seconds),
        "outside_observed_runs_seconds": outside_observed_runs,
        "outside_observed_runs_basis": (
            "process-wall-minus-sum-Simulation.run-wall"
        ),
        "phases": phases,
    }


def validate_policy(
    runs: list[dict[str, Any]],
    requested_backend: str,
    lane: mpi_case.Lane,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if requested_backend == "auto":
        return auto_case.validate_policy_runs(runs, policy)
    forced = dict(policy)
    forced["active_backends"] = [lane.backend] * len(runs)
    return auto_case.validate_policy_runs(runs, forced)


def repository_source_identity(
    repo: pathlib.Path,
) -> tuple[str, dict[str, Any]]:
    """Return a clean, stable trusted-Git HEAD and compact source snapshot."""
    try:
        head_before = provenance.git_output(
            repo, "rev-parse", "--verify", "HEAD"
        ).strip()
        status_before = provenance.git_output(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--no-renames",
        )
        source = validation.source_snapshot(repo)
        status_after = provenance.git_output(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--no-renames",
        )
        head_after = provenance.git_output(
            repo, "rev-parse", "--verify", "HEAD"
        ).strip()
    except (OSError, RuntimeError, UnicodeError) as error:
        raise WorkloadError(
            f"M3 installed-package source identity is unavailable: {error}"
        ) from error
    if not source.get("available"):
        raise WorkloadError(
            "M3 installed-package source identity is unavailable: "
            + str(source.get("error", "source snapshot failed"))
        )
    if (
        GIT_COMMIT.fullmatch(head_before) is None
        or head_after != head_before
        or status_before != status_after
    ):
        raise WorkloadError(
            "M3 installed-package source identity changed while captured"
        )
    if status_before:
        raise WorkloadError(
            "M3 installed-package source worktree is not clean"
        )
    return head_before, source


def _runtime_paths(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    package = getattr(args, "package", None)
    if package is None:
        return mpi_case._runtime_paths(args)
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    manifest = pathlib.Path(os.path.abspath(args.manifest)).resolve(strict=True)
    if manifest != (repo / "scripts/python-validation/manifest.json").resolve(
        strict=True
    ):
        raise WorkloadError("M3 installed-package manifest is not authoritative")
    source_commit, _source = repository_source_identity(repo)
    try:
        report, installed = v110_identity.validate_report(
            args.build_receipt,
            package,
            args.install_prefix,
            source_commit,
        )
        runtime = v110_identity.runtime_paths(report, installed)
    except v110_identity.V110IdentityError as error:
        raise WorkloadError(str(error)) from error
    supplied = {
        "python": pathlib.Path(os.path.abspath(args.python)).resolve(strict=True),
        "build_python": pathlib.Path(os.path.abspath(args.build_python)).resolve(
            strict=True
        ),
        "install_prefix": pathlib.Path(
            os.path.abspath(args.install_prefix)
        ).resolve(strict=True),
        "build_receipt": pathlib.Path(
            os.path.abspath(args.build_receipt)
        ).resolve(strict=True),
        "mpiexec": pathlib.Path(os.path.abspath(args.mpiexec)).resolve(strict=True),
        "package": pathlib.Path(os.path.abspath(package)).resolve(strict=True),
    }
    if any(supplied[name] != runtime[name] for name in supplied):
        raise WorkloadError("M3 installed-package runtime arguments differ from provenance")
    return {
        "repo": repo,
        "manifest": manifest,
        **supplied,
        "provenance_report": runtime["provenance_report"],
        "verifier": runtime["verifier"],
        "launcher": (SCRIPT_DIR / "m3_mpi_rank_launcher.py").resolve(strict=True),
    }


def _runtime_state(
    paths: dict[str, pathlib.Path],
    *,
    identity_record_paths: set[pathlib.Path] | None = None,
    identity_file_records: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    if "package" not in paths:
        receipt = provenance.verify_build_receipt(
            paths["build_receipt"], paths["repo"]
        )
        if receipt.get("build_kind") != validation.EXPECTED_BUILD_KIND:
            raise WorkloadError("M3 performance sample build kind differs")
        snapshot = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        if not snapshot.get("available"):
            raise WorkloadError(
                "M3 performance sample runtime snapshot failed: "
                + "; ".join(snapshot["problems"])
            )
        if snapshot["runtime_contract"].get("receipt_id") != receipt["receipt_id"]:
            raise WorkloadError("M3 performance sample receipt differs")
        if identity_record_paths is not None:
            identity_record_paths.update(
                path for path in paths.values() if path.is_file()
            )
        return receipt["receipt_id"], snapshot

    source_commit, source_before = repository_source_identity(paths["repo"])
    try:
        report, installed = v110_identity.validate_report(
            paths["provenance_report"],
            paths["package"],
            paths["install_prefix"],
            source_commit,
        )
        owned_records = v110_identity.package_owned_records(report, installed)
        runtime = v110_identity.runtime_paths(report, installed)
        for name in (
            "python",
            "build_python",
            "install_prefix",
            "build_receipt",
            "mpiexec",
            "package",
            "provenance_report",
            "verifier",
        ):
            if paths[name] != runtime[name]:
                raise WorkloadError(
                    f"M3 installed-package live runtime path changed: {name}"
                )
        contract = v110_identity.runtime_contract(report, installed)
        live = v110_identity.live_verify(report, installed)
        snapshot = v110_identity.snapshot(
            paths["repo"],
            report,
            installed,
            contract,
            live,
            captured_at_utc=validation.utc_now(),
        )
        ending_commit, source_after = repository_source_identity(paths["repo"])
    except v110_identity.V110IdentityError as error:
        raise WorkloadError(str(error)) from error
    if (
        ending_commit != source_commit
        or source_after != source_before
        or snapshot.get("source_snapshot") != source_before
    ):
        raise WorkloadError(
            "M3 installed-package source identity changed during runtime capture"
        )
    if identity_record_paths is not None:
        # ``validate_report`` has already checked the full record schema,
        # canonical path, prefix containment, size, and SHA-256 for every
        # installed identity file.  Preserve all of those validated paths for
        # the session, including the conda record and release manifest that do
        # not appear in the compact M3 snapshot.
        identity_record_paths.update(
            path
            for name, path in installed.items()
            if name != "install_prefix"
        )
    if identity_file_records is not None:
        identity_file_records.extend(owned_records)
    return report["report_id"], snapshot


class RuntimeVerificationSession:
    """Reuse one installed-runtime truth inside one controller process.

    Child sample processes still perform their own complete live verification.
    This session only removes redundant parent-side verifier invocations while
    every sample's retained bytes, numerical output, telemetry, and runtime
    snapshot remain independently replayed.  Critical files are held by
    O_NOFOLLOW descriptors, cheap identity checks run on every reuse, and a
    fresh complete verification is mandatory at finalization.
    """

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._path_identity: tuple[tuple[str, str], ...] | None = None
        self._runtime_paths: dict[str, pathlib.Path] | None = None
        self._receipt_id: str | None = None
        self._snapshot: dict[str, Any] | None = None
        self._source_commit: str | None = None
        self._file_anchors: dict[pathlib.Path, dict[str, Any]] = {}
        self._symlink_anchors: dict[pathlib.Path, dict[str, Any]] = {}
        self._directory_anchors: dict[pathlib.Path, tuple[int, ...]] = {}
        self._finalized = False
        self._closed = False

    @staticmethod
    def _paths_identity(
        paths: dict[str, pathlib.Path],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (name, os.path.abspath(os.fspath(path)))
                for name, path in paths.items()
            )
        )

    @staticmethod
    def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    @staticmethod
    def _snapshot_record_paths(value: Any) -> set[pathlib.Path]:
        paths: set[pathlib.Path] = set()

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if (
                    isinstance(item.get("path"), str)
                    and isinstance(item.get("sha256"), str)
                    and type(item.get("size_bytes")) is int
                ):
                    paths.add(pathlib.Path(item["path"]))
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        return paths

    def _open_file_anchor(
        self,
        path: pathlib.Path,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
        expected_mode: int | None = None,
    ) -> None:
        canonical = pathlib.Path(os.path.abspath(os.fspath(path)))
        try:
            resolved = canonical.resolve(strict=True)
        except OSError as error:
            raise WorkloadError(
                f"runtime verification session file is unavailable: {canonical}"
            ) from error
        if resolved != canonical:
            raise WorkloadError(
                "runtime verification session file path contains a symlink: "
                f"{canonical}"
            )
        if canonical in self._file_anchors:
            return
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(canonical, flags)
            descriptor_stat = os.fstat(descriptor)
            named_stat = canonical.lstat()
        except OSError as error:
            if "descriptor" in locals():
                os.close(descriptor)
            raise WorkloadError(
                f"runtime verification session could not anchor {canonical}: "
                f"{error}"
            ) from error
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or stat.S_ISLNK(named_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (named_stat.st_dev, named_stat.st_ino)
        ):
            os.close(descriptor)
            raise WorkloadError(
                f"runtime verification session file is unsafe: {canonical}"
            )
        if (
            expected_sha256 is not None
            or expected_size is not None
            or expected_mode is not None
        ):
            if (
                not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
                or type(expected_size) is not int
                or expected_size < 0
                or type(expected_mode) is not int
                or not 0 <= expected_mode <= 0o7777
                or descriptor_stat.st_size != expected_size
                or stat.S_IMODE(descriptor_stat.st_mode) != expected_mode
            ):
                os.close(descriptor)
                raise WorkloadError(
                    "runtime verification session expected file identity "
                    f"differs: {canonical}"
                )
            digest = hashlib.sha256()
            offset = 0
            try:
                while True:
                    block = os.pread(descriptor, 1024 * 1024, offset)
                    if not block:
                        break
                    digest.update(block)
                    offset += len(block)
                ending_stat = os.fstat(descriptor)
                ending_named_stat = canonical.lstat()
            except OSError as error:
                os.close(descriptor)
                raise WorkloadError(
                    "runtime verification session could not hash anchored "
                    f"file {canonical}: {error}"
                ) from error
            if (
                digest.hexdigest() != expected_sha256
                or self._stat_identity(ending_stat)
                != self._stat_identity(descriptor_stat)
                or stat.S_ISLNK(ending_named_stat.st_mode)
                or (ending_named_stat.st_dev, ending_named_stat.st_ino)
                != (ending_stat.st_dev, ending_stat.st_ino)
            ):
                os.close(descriptor)
                raise WorkloadError(
                    "runtime verification session anchored file bytes differ: "
                    f"{canonical}"
                )
        self._file_anchors[canonical] = {
            "descriptor": descriptor,
            "identity": self._stat_identity(descriptor_stat),
            "sha256": expected_sha256,
            "size_bytes": expected_size,
            "mode": expected_mode,
        }

    def _open_symlink_anchor(
        self, path: pathlib.Path, record: dict[str, Any]
    ) -> None:
        canonical = pathlib.Path(os.path.abspath(os.fspath(path)))
        try:
            metadata = canonical.lstat()
            link_target = os.readlink(canonical)
            content_path = canonical.resolve(strict=True)
        except OSError as error:
            raise WorkloadError(
                "runtime verification session could not anchor symlink "
                f"{canonical}: {error}"
            ) from error
        expected_content = pathlib.Path(record["content_path"])
        if (
            not stat.S_ISLNK(metadata.st_mode)
            or link_target != record["link_target"]
            or content_path != expected_content
            or stat.S_IMODE(metadata.st_mode) != record.get("mode")
        ):
            raise WorkloadError(
                "runtime verification session symlink is unsafe: "
                f"{canonical}"
            )
        self._symlink_anchors[canonical] = {
            "identity": self._stat_identity(metadata),
            "link_target": link_target,
            "content_path": content_path,
        }

    def _capture_anchors(
        self,
        paths: dict[str, pathlib.Path],
        snapshot: dict[str, Any],
        identity_record_paths: set[pathlib.Path],
        identity_file_records: list[dict[str, Any]],
    ) -> None:
        expected_files: dict[pathlib.Path, tuple[str, int, int]] = {}

        def retain_expected(
            path: pathlib.Path, sha256: Any, size_bytes: Any, mode: Any
        ) -> None:
            canonical = pathlib.Path(os.path.abspath(os.fspath(path)))
            if (
                not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                or type(size_bytes) is not int
                or size_bytes < 0
                or type(mode) is not int
                or not 0 <= mode <= 0o7777
            ):
                raise WorkloadError(
                    "runtime verification session package-owned identity differs"
                )
            expected = (sha256, size_bytes, mode)
            if canonical in expected_files and expected_files[canonical] != expected:
                raise WorkloadError(
                    "runtime verification session package-owned identities conflict"
                )
            expected_files[canonical] = expected

        file_paths = {
            pathlib.Path(os.path.abspath(os.fspath(path)))
            for path in paths.values()
            if pathlib.Path(path).is_file()
        }
        file_paths.update(
            pathlib.Path(os.path.abspath(os.fspath(path)))
            for path in self._snapshot_record_paths(snapshot)
        )
        file_paths.update(
            pathlib.Path(os.path.abspath(os.fspath(path)))
            for path in identity_record_paths
        )
        for record in identity_file_records:
            path = pathlib.Path(record["path"])
            if record["path_type"] == "softlink":
                self._open_symlink_anchor(path, record)
                content_path = pathlib.Path(record["content_path"])
                file_paths.add(content_path)
                retain_expected(
                    content_path,
                    record.get("sha256"),
                    record.get("size_bytes"),
                    record.get("content_mode"),
                )
            else:
                file_paths.add(path)
                retain_expected(
                    path,
                    record.get("sha256"),
                    record.get("size_bytes"),
                    record.get("content_mode"),
                )
        for path in sorted(file_paths, key=os.fspath):
            canonical = pathlib.Path(os.path.abspath(os.fspath(path)))
            expected = expected_files.get(canonical)
            self._open_file_anchor(
                path,
                expected_sha256=expected[0] if expected is not None else None,
                expected_size=expected[1] if expected is not None else None,
                expected_mode=expected[2] if expected is not None else None,
            )
        for path in (
            paths["repo"],
            paths["install_prefix"],
            paths["build_python"],
        ):
            canonical = pathlib.Path(os.path.abspath(os.fspath(path)))
            metadata = canonical.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(
                metadata.st_mode
            ):
                raise WorkloadError(
                    "runtime verification session directory is unsafe: "
                    f"{canonical}"
                )
            self._directory_anchors[canonical] = self._stat_identity(metadata)

    def _verify_source(self, repo: pathlib.Path) -> None:
        try:
            commit = provenance.git_output(
                repo, "rev-parse", "--verify", "HEAD"
            ).strip()
            status = provenance.git_output(
                repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--no-renames",
            )
        except (OSError, RuntimeError, UnicodeError) as error:
            raise WorkloadError(
                f"runtime verification session source check failed: {error}"
            ) from error
        if commit != self._source_commit or status:
            raise WorkloadError(
                "runtime verification session source identity changed"
            )

    def _verify_anchors(self, paths: dict[str, pathlib.Path]) -> None:
        if self._owner_pid != os.getpid():
            raise WorkloadError(
                "runtime verification session cannot cross a fork boundary"
            )
        if self._closed or self._finalized:
            raise WorkloadError("runtime verification session is no longer active")
        if self._paths_identity(paths) != self._path_identity:
            raise WorkloadError("runtime verification session paths changed")
        self._verify_source(paths["repo"])
        for path, anchor in self._file_anchors.items():
            descriptor_stat = os.fstat(anchor["descriptor"])
            try:
                named_stat = path.lstat()
            except OSError as error:
                raise WorkloadError(
                    f"runtime verification session file disappeared: {path}"
                ) from error
            if (
                self._stat_identity(descriptor_stat) != anchor["identity"]
                or stat.S_ISLNK(named_stat.st_mode)
                or (named_stat.st_dev, named_stat.st_ino)
                != (descriptor_stat.st_dev, descriptor_stat.st_ino)
            ):
                raise WorkloadError(
                    f"runtime verification session file changed: {path}"
                )
        for path, anchor in self._symlink_anchors.items():
            try:
                metadata = path.lstat()
                link_target = os.readlink(path)
                content_path = path.resolve(strict=True)
            except OSError as error:
                raise WorkloadError(
                    "runtime verification session symlink disappeared: "
                    f"{path}"
                ) from error
            if (
                self._stat_identity(metadata) != anchor["identity"]
                or not stat.S_ISLNK(metadata.st_mode)
                or link_target != anchor["link_target"]
                or content_path != anchor["content_path"]
            ):
                raise WorkloadError(
                    f"runtime verification session symlink changed: {path}"
                )
        for path, identity in self._directory_anchors.items():
            try:
                metadata = path.lstat()
            except OSError as error:
                raise WorkloadError(
                    f"runtime verification session directory disappeared: {path}"
                ) from error
            if (
                self._stat_identity(metadata) != identity
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
            ):
                raise WorkloadError(
                    f"runtime verification session directory changed: {path}"
                )

    def runtime_state(
        self, paths: dict[str, pathlib.Path]
    ) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if self._owner_pid != os.getpid():
                raise WorkloadError(
                    "runtime verification session cannot cross a fork boundary"
                )
            if self._closed or self._finalized:
                raise WorkloadError(
                    "runtime verification session is no longer active"
                )
            identity = self._paths_identity(paths)
            if self._path_identity is None:
                identity_record_paths: set[pathlib.Path] = set()
                identity_file_records: list[dict[str, Any]] = []
                try:
                    receipt_id, snapshot = _runtime_state(
                        paths,
                        identity_record_paths=identity_record_paths,
                        identity_file_records=identity_file_records,
                    )
                    source_commit = provenance.git_output(
                        paths["repo"], "rev-parse", "--verify", "HEAD"
                    ).strip()
                except Exception:
                    # No state or descriptor has been retained yet; a caller
                    # may retry after a transient initial verifier failure.
                    raise
                try:
                    self._capture_anchors(
                        paths,
                        snapshot,
                        identity_record_paths,
                        identity_file_records,
                    )
                except BaseException:
                    self.close(verify=False)
                    raise
                self._path_identity = identity
                self._runtime_paths = dict(paths)
                self._receipt_id = receipt_id
                self._snapshot = copy.deepcopy(snapshot)
                self._source_commit = source_commit
                try:
                    self._verify_anchors(paths)
                except BaseException:
                    self.close(verify=False)
                    raise
            else:
                self._verify_anchors(paths)
            if self._receipt_id is None or self._snapshot is None:
                raise WorkloadError("runtime verification session state is absent")
            return self._receipt_id, copy.deepcopy(self._snapshot)

    def finalize(self) -> None:
        with self._lock:
            if self._path_identity is None or self._runtime_paths is None:
                self.close(verify=False)
                raise WorkloadError("runtime verification session was never used")
            paths = self._runtime_paths
            try:
                self._verify_anchors(paths)
                ending_receipt, ending_snapshot = _runtime_state(paths)
                unchanged, problems = validation.compare_validation_snapshots(
                    self._snapshot, ending_snapshot
                )
                if (
                    ending_receipt != self._receipt_id
                    or not unchanged
                    or ending_snapshot.get("source_snapshot")
                    != self._snapshot.get("source_snapshot")
                ):
                    raise WorkloadError(
                        "runtime verification session final verification "
                        "differed: "
                        + "; ".join(problems)
                    )
                self._finalized = True
            finally:
                self.close(verify=False)

    def abort(self) -> str | None:
        """Best-effort end verification without masking a primary failure."""
        with self._lock:
            if self._closed:
                return None
            if self._path_identity is None:
                self.close(verify=False)
                return None
        try:
            self.finalize()
        except BaseException as error:
            self.close(verify=False)
            return f"{type(error).__name__}: {error}"
        return None

    def close(self, *, verify: bool = False) -> None:
        if verify:
            self.finalize()
            return
        with self._lock:
            if self._closed:
                return
            for anchor in self._file_anchors.values():
                try:
                    os.close(anchor["descriptor"])
                except OSError:
                    # Closing every remaining descriptor is more important
                    # than allowing one teardown error to mask the primary
                    # verification result.
                    pass
            self._file_anchors.clear()
            self._symlink_anchors.clear()
            self._closed = True


def _code_records(paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    result = {
        "sample_runner": absolute_file_record(
            pathlib.Path(__file__).resolve(), "sample runner"
        ),
        "automatic_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_auto_example_case.py", "automatic common"
        ),
        "mpi_example_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_mpi_example_case.py", "MPI example common"
        ),
        "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
        "example_oracle": absolute_file_record(
            SCRIPT_DIR / "run_example_oracle.py", "example oracle"
        ),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "validation runner"
        ),
        "manifest": absolute_file_record(paths["manifest"], "manifest"),
        "adjoint_cyl_fixed_work_contract": absolute_file_record(
            paths["repo"] / "python/tests/adjoint_cyl_fixed_work_contract.py",
            "cylindrical adjoint fixed-work contract",
        ),
        "inversion_inventory": absolute_file_record(
            paths["repo"] / "packaging/v110-performance-inversion-inventory.json",
            "inversion inventory",
        ),
    }
    if "package" in paths:
        result["installed_runtime_identity"] = absolute_file_record(
            SCRIPT_DIR / "v110_runtime_identity.py", "installed runtime identity"
        )
        result["conda_archive_inspector"] = absolute_file_record(
            SCRIPT_DIR.parent / "v110-final-release-receipt.py",
            "Conda archive inspector",
        )
    return result


def _terminal(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "case_path": report["case_path"],
        "topology": report["topology"],
        "sample_kind": report["sample_kind"],
        "cycle_index": report["cycle_index"],
        "receipt_id": report["receipt_id"],
    }


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(output / "PENDING_COMPLETE", _terminal(output, report))
    verify_complete(output, "PENDING_COMPLETE")
    durable_publish_terminal(output / "PENDING_COMPLETE", output / "COMPLETE")


def _validate_process(process: Any) -> None:
    if (
        not isinstance(process, dict)
        or set(process)
        != {
            "exit_code",
            "timeout",
            "output_limit",
            "spawn_error",
            "stdout_size_bytes",
            "stderr_size_bytes",
            "duration_seconds",
        }
        or type(process.get("exit_code")) is not int
        or process.get("exit_code") != 0
        or process.get("timeout") is not False
        or process.get("output_limit") is not None
        or process.get("spawn_error") is not None
        or isinstance(process.get("duration_seconds"), bool)
        or not isinstance(process.get("duration_seconds"), (int, float))
        or not math.isfinite(float(process["duration_seconds"]))
        or process["duration_seconds"] <= 0
        or type(process.get("stdout_size_bytes")) is not int
        or process["stdout_size_bytes"] < 0
        or type(process.get("stderr_size_bytes")) is not int
        or process["stderr_size_bytes"] < 0
    ):
        raise WorkloadError("M3 performance sample process record differs")


def verify_complete(
    output: pathlib.Path,
    terminal_name: str = "COMPLETE",
    *,
    runtime_session: RuntimeVerificationSession | None = None,
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M3 performance sample terminal filename differs")
    terminal = auto_case._load_object(
        output / terminal_name, "M3 performance sample COMPLETE"
    )
    report = auto_case._load_object(
        output / "report.json", "M3 performance sample report"
    )
    if (
        set(terminal)
        != {
            "schema",
            "outcome",
            "report",
            "case_path",
            "topology",
            "sample_kind",
            "cycle_index",
            "receipt_id",
        }
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
        or not isinstance(report, dict)
        or set(report) != REPORT_KEYS
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or terminal != _terminal(output, report)
    ):
        raise WorkloadError("M3 performance sample terminal binding differs")
    validate_sample_identity(report["sample_kind"], report["cycle_index"])
    if report["topology"] not in TOPOLOGIES:
        raise WorkloadError("M3 performance sample retained topology differs")
    paths = {
        name: pathlib.Path(value).resolve(strict=True)
        for name, value in report["paths"].items()
    }
    namespace = argparse.Namespace(
        repo=paths["repo"],
        manifest=paths["manifest"],
        python=paths["python"],
        build_python=paths["build_python"],
        install_prefix=paths["install_prefix"],
        build_receipt=paths["build_receipt"],
        mpiexec=paths["mpiexec"],
        package=paths.get("package"),
    )
    if paths != _runtime_paths(namespace):
        raise WorkloadError("M3 performance sample runtime paths differ")
    if report["evidence_code"] != _code_records(paths):
        raise WorkloadError("M3 performance sample evidence-code identity differs")
    case = mpi_case._case_from_manifest(
        paths["repo"], paths["manifest"], report["case_path"]
    )
    validate_inversion_inventory(paths["repo"])
    if case["path"] not in INVERSION_CASES:
        raise WorkloadError("M3 performance sample retained case differs")
    receipt_id, snapshot = (
        runtime_session.runtime_state(paths)
        if runtime_session is not None
        else _runtime_state(paths)
    )
    unchanged, _problems = validation.compare_validation_snapshots(
        report["runtime_snapshot"], snapshot
    )
    if not snapshot.get("available") or not unchanged:
        raise WorkloadError("M3 performance sample live snapshot differs")
    if (
        receipt_id != report["receipt_id"]
        or snapshot["runtime_contract"].get("receipt_id") != report["receipt_id"]
    ):
        raise WorkloadError("M3 performance sample receipt identity differs")
    devices = normalized_devices(",".join(report["gpu_devices"]))
    if list(devices) != report["gpu_devices"]:
        raise WorkloadError("M3 performance sample GPU inventory differs")
    lane, requested, policy = select_lane(
        report["topology"], devices, case["path"]
    )
    retained = report["lane"]
    if (
        not isinstance(retained, dict)
        or set(retained) != LANE_KEYS
        or retained["spec"] != mpi_case.lane_spec(lane)
        or retained["requested_backend"] != requested
    ):
        raise WorkloadError("M3 performance sample lane schema differs")
    nonce = retained["run_nonce"]
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise WorkloadError("M3 performance sample nonce differs")
    root = output / "lane"
    case_command = case_command_for_lane(
        case, paths["python"], paths["repo"], requested
    )
    command = mpi_case.lane_command(
        lane,
        mpiexec=paths["mpiexec"],
        python=paths["python"],
        launcher=paths["launcher"],
        evidence_root=root,
        case_command=case_command,
    )
    if retained["command"] != command:
        raise WorkloadError("M3 performance sample command differs")
    environment = sample_environment(
        lane,
        requested,
        case_path=case["path"],
        output=output,
        python=paths["python"],
        build_python=paths["build_python"],
        runtime_contract=snapshot["runtime_contract"],
        run_nonce=nonce,
    )
    if retained["environment_sha256"] != validation.environment_sha256(environment):
        raise WorkloadError("M3 performance sample environment differs")
    _validate_process(retained["process"])
    if requested == "auto":
        rank_records, policy_result = auto_case.validate_rank_bundle(
            root,
            case_command,
            snapshot["runtime_contract"],
            nonce,
            devices[0],
            policy,
        )
    else:
        rank_records = mpi_case.validate_rank_bundle(
            lane,
            root,
            case,
            snapshot["runtime_contract"],
            nonce,
            case_command,
        )
        runs = auto_case.audit_runs(root / "stdout.log", requested)
        policy_result = validate_policy(runs, requested, lane, policy)
    if retained["rank_records"] != rank_records:
        raise WorkloadError("M3 performance sample rank replay differs")
    if requested == "auto":
        runs = auto_case.audit_runs(root / "stdout.log", requested)
    if retained["policy_result"] != policy_result:
        raise WorkloadError("M3 performance sample policy replay differs")
    if report["timing"] != derive_timing(case["path"], runs):
        raise WorkloadError("M3 performance sample timing replay differs")
    if report["phase_profile"] != derive_phase_profile(
        runs, retained["process"]["duration_seconds"]
    ):
        raise WorkloadError("M3 performance sample phase replay differs")
    for name in ("stdout", "stderr"):
        if retained[name] != file_record(root / f"{name}.log", root):
            raise WorkloadError(f"M3 performance sample {name} changed")
    if (
        retained["process"]["stdout_size_bytes"]
        != retained["stdout"]["size_bytes"]
        or retained["process"]["stderr_size_bytes"]
        != retained["stderr"]["size_bytes"]
    ):
        raise WorkloadError("M3 performance sample output sizes differ")
    tree_path = output / "tree-manifest.json"
    if (
        retained["tree_manifest"] != file_record(tree_path, output)
        or auto_case._load_object(tree_path, "M3 performance sample tree")
        != mpi_case.derive_tree(root)
    ):
        raise WorkloadError("M3 performance sample raw tree differs")
    validate_comparison_evidence(
        (root / "stdout.log").read_text(encoding="utf-8"), case["comparison"]
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_sample_identity(args.sample_kind, args.cycle_index)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("M3 performance sample timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M3 performance sample output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    case_stable: StableFile | None = None
    try:
        paths = _runtime_paths(args)
        if output == paths["repo"] or paths["repo"] in output.parents:
            raise WorkloadError("M3 performance sample output overlaps source")
        if "package" not in paths and paths[
            "build_python"
        ].parent / "build-provenance.json" != paths["build_receipt"]:
            raise WorkloadError("M3 performance sample build paths differ")
        case = mpi_case._case_from_manifest(
            paths["repo"], paths["manifest"], args.case
        )
        case_stable = StableFile(
            paths["repo"] / case["path"],
            "M3 performance sample selected case",
        )
        validate_inversion_inventory(paths["repo"])
        if case["path"] not in INVERSION_CASES:
            raise WorkloadError("M3 performance sample case is not sealed")
        receipt_id, snapshot = _runtime_state(paths)
        devices = normalized_devices(args.gpu_devices)
        lane, requested, policy = select_lane(args.topology, devices, case["path"])
        root = output / "lane"
        for name in ("statistics", "identity", "work"):
            (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
        nonce = uuid.uuid4().hex
        environment = sample_environment(
            lane,
            requested,
            case_path=case["path"],
            output=output,
            python=paths["python"],
            build_python=paths["build_python"],
            runtime_contract=snapshot["runtime_contract"],
            run_nonce=nonce,
        )
        case_command = case_command_for_lane(
            case, paths["python"], paths["repo"], requested
        )
        command = mpi_case.lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=root,
            case_command=case_command,
        )
        started = time.monotonic()
        process = validation.run_bounded_process(
            command,
            cwd=paths["repo"],
            env=environment,
            timeout_seconds=args.timeout_seconds,
            stdout_file=root / "stdout.log",
            stderr_file=root / "stderr.log",
            stdout_limit=mpi_case.MAX_STDOUT_BYTES,
            stderr_limit=mpi_case.MAX_STDERR_BYTES,
            combined_limit=mpi_case.MAX_STDOUT_BYTES + mpi_case.MAX_STDERR_BYTES,
        )
        process["duration_seconds"] = time.monotonic() - started
        if (
            process["exit_code"] != 0
            or process["timeout"]
            or process["output_limit"] is not None
            or process["spawn_error"] is not None
            or not validation.is_strict_utf8_file(root / "stdout.log")
            or not validation.is_strict_utf8_file(root / "stderr.log")
        ):
            raise WorkloadError("M3 performance sample process failed")
        validate_comparison_evidence(
            (root / "stdout.log").read_text(encoding="utf-8"),
            case["comparison"],
        )
        if requested == "auto":
            rank_records, policy_result = auto_case.validate_rank_bundle(
                root,
                case_command,
                snapshot["runtime_contract"],
                nonce,
                devices[0],
                policy,
            )
        else:
            rank_records = mpi_case.validate_rank_bundle(
                lane,
                root,
                case,
                snapshot["runtime_contract"],
                nonce,
                case_command,
            )
            runs = auto_case.audit_runs(root / "stdout.log", requested)
            policy_result = validate_policy(runs, requested, lane, policy)
        if requested == "auto":
            runs = auto_case.audit_runs(root / "stdout.log", requested)
        timing = derive_timing(case["path"], runs)
        phase_profile = derive_phase_profile(
            runs, process["duration_seconds"]
        )
        atomic_write_json(output / "tree-manifest.json", mpi_case.derive_tree(root))
        lane_record = {
            "spec": mpi_case.lane_spec(lane),
            "requested_backend": requested,
            "run_nonce": nonce,
            "command": command,
            "environment_sha256": validation.environment_sha256(environment),
            "process": process,
            "stdout": file_record(root / "stdout.log", root),
            "stderr": file_record(root / "stderr.log", root),
            "rank_records": rank_records,
            "tree_manifest": file_record(output / "tree-manifest.json", output),
            "policy_result": policy_result,
        }
        ending_receipt_id, ending = _runtime_state(paths)
        unchanged, problems = validation.compare_validation_snapshots(
            snapshot, ending
        )
        if not unchanged or ending_receipt_id != receipt_id:
            raise WorkloadError(
                "M3 performance sample source/build changed: "
                + "; ".join(problems)
            )
        case_stable.verify_unchanged()
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "topology": args.topology,
            "sample_kind": args.sample_kind,
            "cycle_index": args.cycle_index,
            "gpu_devices": list(devices),
            "receipt_id": receipt_id,
            "paths": {name: str(path) for name, path in paths.items()},
            "runtime_snapshot": snapshot,
            "lane": lane_record,
            "timing": timing,
            "phase_profile": phase_profile,
            "evidence_code": _code_records(paths),
        }
        atomic_write_json(output / "report.json", report)
        publish_terminal(output, report)
        return 0
    except Exception as exc:
        atomic_write_json(
            output / "FAILED.json",
            {
                "schema": SCHEMA,
                "outcome": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_unix_seconds": time.time(),
            },
        )
        raise
    finally:
        if case_stable is not None:
            case_stable.close(verify=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 performance sample error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
