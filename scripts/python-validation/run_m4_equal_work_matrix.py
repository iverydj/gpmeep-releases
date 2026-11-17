#!/usr/bin/env python3
"""Run, resume, seal, and semantically replay the 180-task M4 matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import stat
import sys
import tempfile
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
import m4_equal_work_plan as equal_work_plan  # noqa: E402
import m4_equal_work_replay as equal_work_replay  # noqa: E402
import run_m3_performance_sample as v110_sample  # noqa: E402
import run_observed_process as observed  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import run_v110_upstream_performance_sample as upstream_sample  # noqa: E402
import run_v110_v103_performance_sample as v103_sample  # noqa: E402
import run_validation as validation  # noqa: E402
import v110_runtime_identity as v110_identity  # noqa: E402
import verify_telemetry_v2 as telemetry_replay  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    durable_publish_terminal,
    file_record,
)


SCHEMA = "gpmeep-v110-m4-equal-work-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-v110-m4-equal-work-checkpoint-v1"
JOURNAL_SCHEMA = "gpmeep-v110-m4-equal-work-journal-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m4-equal-work-complete-v1"
PREFLIGHT_SCHEMA = "gpmeep-v110-m4-equal-work-preflight-v1"
TASK_COUNT = 180
INITIAL_FREE_BYTES = 40 * 1024**3
PER_TASK_FREE_BYTES = 5 * 1024**3
MINIMUM_FREE_INODES = 100_000
MINIMUM_HOST_AVAILABLE_BYTES = 4 * 1024**3
MINIMUM_CGROUP_FREE_BYTES = 4 * 1024**3
MINIMUM_GPU_FREE_MEMORY_FRACTION = 0.20
MAXIMUM_GPU_TEMPERATURE_CELSIUS = 90.0
MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES = 95_000.0
NVIDIA_OPEN_UVM_HMM_BLOCKED_MAJOR_VERSIONS = frozenset({595})
NVIDIA_VERSION_PATH = pathlib.Path("/proc/driver/nvidia/version")
NVIDIA_UVM_HMM_PATH = pathlib.Path(
    "/sys/module/nvidia_uvm/parameters/uvm_disable_hmm"
)
GENERIC_METRIC_PREFIX = "gpmeep-generic-example-metrics:"
ABSORBER_CASE = "python/examples/absorber-1d.py"
WAVEGUIDE_CROSSING_CASE = "python/examples/waveguide_crossing.py"
NEAR2FAR_3D_CASE = (
    "python/examples/adjoint_optimization/near2far_3d_validation.py"
)
ADJOINT_CYL_CASE = "python/tests/test_adjoint_cyl.py"
GPMEEP_CPU_REFERENCE_CASES = frozenset(
    {WAVEGUIDE_CROSSING_CASE, NEAR2FAR_3D_CASE, ADJOINT_CYL_CASE}
)
WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES = frozenset(
    {
        "optimizer_smoke.epoch0_design[[]*]",
        "optimizer_smoke.final_design[[]*]",
        "optimizer_smoke.*_design_step_l2",
        "optimizer_smoke.final_design_l2",
        "optimizer_smoke.*_gradient_l2_*",
        "optimizer_smoke.*_objective_*",
    }
)
WAVEGUIDE_OPTIMIZER_DESIGN_SIZE = 441
WAVEGUIDE_FIXED_SLICE_RUN_COUNT = 12
WAVEGUIDE_OPTIMIZER_RUN_COUNT = 8
WAVEGUIDE_FIXED_SLICE_TIMESTEP = 20_001
WAVEGUIDE_OPTIMIZER_TIMESTEP = 10_001
WAVEGUIDE_FIXED_WORK_METRICS = frozenset(
    {
        *(
            f"fixed_work.run_timesteps[{index}]"
            for index in range(WAVEGUIDE_FIXED_SLICE_RUN_COUNT)
        ),
        *(
            f"optimizer_smoke.run_timesteps[{index}]"
            for index in range(WAVEGUIDE_OPTIMIZER_RUN_COUNT)
        ),
    }
)
WAVEGUIDE_OPTIMIZER_STRUCTURAL_METRICS = frozenset(
    {
        "optimizer_smoke.beta_initial",
        "optimizer_smoke.beta_final",
        "optimizer_smoke.epoch_count",
        "optimizer_smoke.epoch0_beta",
        "optimizer_smoke.epoch1_beta",
        "optimizer_smoke.epoch0_callback_count",
        "optimizer_smoke.epoch1_callback_count",
        "optimizer_smoke.epoch0_nlopt_evaluations",
        "optimizer_smoke.epoch1_nlopt_evaluations",
        "optimizer_smoke.epoch0_result_code",
        "optimizer_smoke.epoch1_result_code",
        "optimizer_smoke.checkpoint_callback_count",
        "optimizer_smoke.checkpoint_completed_epochs",
        "optimizer_smoke.checkpoint_roundtrip_max_abs_difference",
        "optimizer_smoke.result_roundtrip_max_abs_difference",
    }
)
WAVEGUIDE_OPTIMIZER_CONTINUOUS_SCALAR_METRICS = frozenset(
    {
        "optimizer_smoke.epoch0_design_step_l2",
        "optimizer_smoke.epoch1_design_step_l2",
        "optimizer_smoke.epoch0_gradient_l2_initial",
        "optimizer_smoke.epoch0_gradient_l2_final",
        "optimizer_smoke.epoch1_gradient_l2_initial",
        "optimizer_smoke.epoch1_gradient_l2_final",
        "optimizer_smoke.epoch0_objective_initial",
        "optimizer_smoke.epoch0_objective_final",
        "optimizer_smoke.epoch1_objective_initial",
        "optimizer_smoke.epoch1_objective_final",
        "optimizer_smoke.final_design_l2",
    }
)
WAVEGUIDE_FIXED_PHYSICS_METRICS = frozenset(
    {
        f"{configuration}.{metric}"
        for configuration, metrics in (
            (
                "smoothed_beta10",
                (
                    "adjoint_directional_derivative",
                    "finite_difference_directional_derivative",
                    "gradient_l2",
                    "objective",
                    "objective_perturbed",
                ),
            ),
            (
                "smoothed_beta1000",
                (
                    "adjoint_directional_derivative",
                    "finite_difference_directional_derivative",
                    "gradient_l2",
                    "objective",
                    "objective_perturbed",
                ),
            ),
            (
                "tanh_beta10",
                (
                    "adjoint_directional_derivative",
                    "finite_difference_directional_derivative",
                    "gradient_l2",
                    "objective",
                    "objective_perturbed",
                ),
            ),
            (
                "tanh_beta1000",
                (
                    "adjoint_directional_derivative",
                    "finite_difference_directional_derivative",
                    "gradient_l2",
                    "mapped_design_max_abs_difference",
                    "objective",
                    "objective_perturbed",
                    "saturation_objective_atol",
                ),
            ),
        )
        for metric in metrics
    }
)
WAVEGUIDE_EXPECTED_METRIC_KEYS = frozenset(
    {
        *WAVEGUIDE_FIXED_WORK_METRICS,
        *WAVEGUIDE_OPTIMIZER_STRUCTURAL_METRICS,
        *WAVEGUIDE_OPTIMIZER_CONTINUOUS_SCALAR_METRICS,
        *WAVEGUIDE_FIXED_PHYSICS_METRICS,
        *(
            f"optimizer_smoke.epoch0_design[{index}]"
            for index in range(WAVEGUIDE_OPTIMIZER_DESIGN_SIZE)
        ),
        *(
            f"optimizer_smoke.final_design[{index}]"
            for index in range(WAVEGUIDE_OPTIMIZER_DESIGN_SIZE)
        ),
    }
)
WAVEGUIDE_EXPECTED_METRIC_COUNT = 950
WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT = 893
WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256 = (
    "a156e185a0f5f16bc5f8c9713bae9d6cd700732dc34e8560ac6f4ce6e75d6e6e"
)
if len(WAVEGUIDE_EXPECTED_METRIC_KEYS) != WAVEGUIDE_EXPECTED_METRIC_COUNT:
    raise RuntimeError("internal waveguide metric inventory differs")
ADJOINT_CYL_METRIC_PREFIX = "gpmeep-adjoint-cyl-metrics:"
ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR = 1.0e-4
ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR = 1.0e-5
ADJOINT_CYL_EXPECTED_TIMESTEPS = tuple(
    value for _case in range(4) for value in (48001, 48001, 51100, 51100)
)
ADJOINT_CYL_EXPECTED_MEEP_TIMES = tuple(
    value for _case in range(4) for value in (1200.025, 1200.025, 1277.5, 1277.5)
)
ADJOINT_CYL_EXPECTED_CASES = (
    (0.0, (5.0, 0.0, 20.0)),
    (0.0, (4.0, 0.0, 28.0)),
    (-1.0, (5.0, 0.0, 20.0)),
    (1.2, (5.0, 0.0, 20.0)),
)
PATH_KEYS = {
    "repo", "python", "build_python", "install_prefix", "build_receipt",
    "mpiexec", "upstream_receipt", "v103_install_prefix",
    "v103_provenance_report", "v103_package", "v110_provenance_report",
    "v110_package", "manifest", "v110_runner",
    "upstream_runner", "v103_runner",
}
RUNTIME_INPUT_KEYS = {
    "v110_build_receipt", "upstream_receipt", "v103_provenance_report",
    "v103_package", "v103_installed_identity", "v110_receipt_id",
    "upstream_receipt_id",
    "v110_source_identity", "v110_provenance_report", "v110_package",
    "v110_package_sha256", "v110_candidate_receipt_id",
    "v110_installed_report_id", "v110_installed_runtime",
    "v110_live_verification", "v110_archive_verification",
    "v110_measured_build_mode", "unittest_import_preflight",
}
UNITTEST_PREFLIGHT_SCHEMA = "gpmeep-m4-unittest-import-preflight-set-v1"
UNITTEST_PREFLIGHT_PREFIX = "gpmeep-m4-unittest-preflight:"
UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES = 1024 * 1024
UNITTEST_PREFLIGHT_GUARDED_SIMULATION_ENTRYPOINTS = [
    "init_sim",
    "run",
    "run_k_point",
    "run_k_points",
    "solve_cw",
]
UNITTEST_PREFLIGHT_CASES = {
    "python/tests/test_adjoint_cyl.py": {
        "case_contract_variable": "GPMEEP_ADJOINT_CYL_TEST_CASES",
        "case_contract": [
            [0, [5, 0, 20]],
            [0, [4, 0, 28]],
            [-1, [5, 0, 20]],
            [1.2, [5, 0, 20]],
        ],
        "wrapper_helper": "_check_adjoint_solver_cyl_n2f_fields",
        "wrapper_invocations": [
            {
                "test_id": "test_adjoint_cyl.TestAdjointSolver."
                "test_adjoint_solver_cyl_n2f_fields_0",
                "mode": 0,
                "far_point": [5.0, 0.0, 20.0],
            },
            {
                "test_id": "test_adjoint_cyl.TestAdjointSolver."
                "test_adjoint_solver_cyl_n2f_fields_1",
                "mode": 0,
                "far_point": [4.0, 0.0, 28.0],
            },
            {
                "test_id": "test_adjoint_cyl.TestAdjointSolver."
                "test_adjoint_solver_cyl_n2f_fields_2",
                "mode": -1,
                "far_point": [5.0, 0.0, 20.0],
            },
            {
                "test_id": "test_adjoint_cyl.TestAdjointSolver."
                "test_adjoint_solver_cyl_n2f_fields_3",
                "mode": 1.2,
                "far_point": [5.0, 0.0, 20.0],
            },
        ],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--python", type=pathlib.Path)
    parser.add_argument("--build-python", type=pathlib.Path)
    parser.add_argument("--install-prefix", type=pathlib.Path)
    parser.add_argument("--build-receipt", type=pathlib.Path)
    parser.add_argument("--mpiexec", type=pathlib.Path)
    parser.add_argument("--upstream-receipt", type=pathlib.Path)
    parser.add_argument("--v103-install-prefix", type=pathlib.Path)
    parser.add_argument("--v103-provenance-report", type=pathlib.Path)
    parser.add_argument("--v103-package", type=pathlib.Path)
    parser.add_argument("--v110-provenance-report", type=pathlib.Path)
    parser.add_argument("--v110-package", type=pathlib.Path)
    parser.add_argument("--gpu-devices")
    parser.add_argument("--task-timeout-seconds", type=int, default=7200)
    parser.add_argument("--stdout-limit-mib", type=int, default=64)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="replay every runtime authority and GPU binding without creating output or launching tasks",
    )
    parser.add_argument(
        "--verify-complete",
        action="store_true",
        help="perform a read-only semantic replay of an already complete output",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 60 <= args.task_timeout_seconds <= 48 * 3600:
        raise WorkloadError("M4 task timeout must be in [60s,48h]")
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("M4 stdout limit must be in [1,1024] MiB")
    interval = args.telemetry_interval_seconds
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 1.0 <= interval <= 60.0
    ):
        raise WorkloadError("M4 telemetry interval must be in [1s,60s]")
    if sum(bool(value) for value in (args.resume, args.verify_complete, args.preflight_only)) > 1:
        raise WorkloadError("M4 resume, preflight-only, and verify-complete are mutually exclusive")
    runtime_names = (
        "python", "build_python", "install_prefix", "build_receipt", "mpiexec",
        "upstream_receipt", "v103_install_prefix", "v103_provenance_report",
        "v103_package", "v110_provenance_report", "v110_package", "gpu_devices",
    )
    if not args.verify_complete and any(getattr(args, name) is None for name in runtime_names):
        raise WorkloadError("M4 execution requires every runtime path and GPU UUID")


def normalized_devices(value: str) -> tuple[str, str]:
    return v110_sample.normalized_devices(value)


def task_name(task: dict[str, Any]) -> str:
    return equal_work_replay.task_name(task)


def task_sample_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("samples", task_name(task))


def task_runtime_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task_name(task))


def task_log_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", f"{task_name(task)}.log")


def task_telemetry_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("telemetry", task_name(task))


def task_comparison_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("comparisons", task_name(task) + ".json")


def task_gpu_devices(
    task: dict[str, Any], contract: dict[str, Any]
) -> tuple[str, ...]:
    if task["backend"] != "cuda":
        return ()
    devices = tuple(contract["gpu_devices"])
    return devices if task["topology"].endswith("cuda2") else devices[:1]


def task_command(
    task: dict[str, Any], contract: dict[str, Any], output: pathlib.Path
) -> list[str]:
    paths = contract["paths"]
    common = [
        paths["python"],
        paths[{"upstream": "upstream_runner", "v103": "v103_runner", "v110": "v110_runner"}[task["runtime"]]],
        "--repo",
        paths["repo"],
        "--manifest",
        paths["manifest"],
        "--output",
        str(output.joinpath(*task_sample_relative(task).parts)),
        "--case",
        task["case_path"],
    ]
    tail = [
        "--sample-kind",
        task["sample_kind"],
        "--cycle-index",
        str(task["cycle_index"]),
    ]
    timeout = ["--timeout-seconds", str(contract["limits"]["task_timeout_seconds"])]
    if task["runtime"] == "upstream":
        return [*common, *tail, "--upstream-receipt", paths["upstream_receipt"], *timeout]
    if task["runtime"] == "v103":
        return [
            *common,
            "--topology",
            task["topology"],
            *tail,
            "--install-prefix",
            paths["v103_install_prefix"],
            "--provenance-report",
            paths["v103_provenance_report"],
            "--package",
            paths["v103_package"],
            "--gpu-devices",
            ",".join(contract["gpu_devices"]),
            *timeout,
        ]
    installed = contract["runtime_inputs"]["v110_installed_runtime"]
    return [
        *common,
        "--topology",
        task["topology"],
        *tail,
        "--python",
        installed["python"],
        "--build-python",
        installed["build_python"],
        "--install-prefix",
        installed["install_prefix"],
        "--build-receipt",
        installed["build_receipt"],
        "--mpiexec",
        installed["mpiexec"],
        "--package",
        installed["package"],
        "--gpu-devices",
        ",".join(contract["gpu_devices"]),
        *timeout,
    ]


def capacity_snapshot(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    usage = os.statvfs(path)
    return {
        "path": str(path),
        "free_bytes": usage.f_bavail * usage.f_frsize,
        "total_bytes": usage.f_blocks * usage.f_frsize,
        "free_inodes": usage.f_favail,
        "total_inodes": usage.f_files,
    }


def validate_capacity(record: Any, initial: bool) -> None:
    keys = {"path", "free_bytes", "total_bytes", "free_inodes", "total_inodes"}
    if (
        not isinstance(record, dict)
        or set(record) != keys
        or not isinstance(record["path"], str)
        or not pathlib.Path(record["path"]).is_absolute()
        or any(type(record[name]) is not int or record[name] < 0 for name in keys - {"path"})
        or record["free_bytes"] > record["total_bytes"]
        or record["free_inodes"] > record["total_inodes"]
    ):
        raise WorkloadError("M4 capacity record differs")
    minimum = INITIAL_FREE_BYTES if initial else PER_TASK_FREE_BYTES
    if record["free_bytes"] < minimum or record["free_inodes"] < MINIMUM_FREE_INODES:
        raise WorkloadError("M4 free-space/inode gate failed")


def validate_precreation_capacity(output: pathlib.Path) -> dict[str, Any]:
    """Validate initial capacity without creating the planned output path."""
    output = pathlib.Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M4 fresh output already exists")
    anchor = output.parent
    while not anchor.exists() and not anchor.is_symlink():
        parent = anchor.parent
        if parent == anchor:
            raise WorkloadError("M4 output has no existing capacity anchor")
        anchor = parent
    if (
        anchor.is_symlink()
        or not anchor.is_dir()
        or anchor.resolve(strict=True) != anchor
    ):
        raise WorkloadError("M4 output capacity anchor differs")
    capacity = capacity_snapshot(anchor)
    validate_capacity(capacity, True)
    return capacity


def host_binding() -> dict[str, Any]:
    boot_path = pathlib.Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_path.read_text(encoding="ascii").strip().lower()
    try:
        parsed = uuid.UUID(boot_id)
    except ValueError as exc:
        raise WorkloadError("M4 Linux boot ID differs") from exc
    affinity = sorted(os.sched_getaffinity(0))
    if not affinity:
        raise WorkloadError("M4 controller CPU affinity is empty")
    return {
        "boot_id": str(parsed),
        "logical_cpu_affinity": affinity,
        "physical_core_count": matrix.physical_core_count(),
        "nvidia_uvm_hmm": nvidia_uvm_hmm_binding(),
    }


def nvidia_uvm_hmm_binding(
    version_path: pathlib.Path = NVIDIA_VERSION_PATH,
    hmm_path: pathlib.Path = NVIDIA_UVM_HMM_PATH,
) -> dict[str, Any]:
    """Reject the known NVIDIA 595 open-UVM/HMM process-exit hazard.

    NVIDIA open-gpu-kernel-modules issue #1082 reports CUDA worker teardown
    kernel crashes with HMM enabled. The proposed guard in PR #1110 is absent
    from the 595.71.05 and 595.91.07 Ubuntu sources inspected after the M4
    incident. M4 therefore fails closed before risking another host reboot or
    mixing evidence across boots.
    """

    try:
        version_text = version_path.read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise WorkloadError("M4 cannot verify NVIDIA driver identity") from error
    version_match = re.search(r"\b(\d{3})\.(\d+)\.(\d+)\b", version_text)
    if version_match is None:
        raise WorkloadError("M4 NVIDIA driver version is malformed")
    version = ".".join(version_match.groups())
    major = int(version_match.group(1))
    first_line = version_text.splitlines()[0] if version_text.splitlines() else ""
    open_kernel_module = bool(
        re.match(
            r"^NVRM version: NVIDIA UNIX Open Kernel Module for "
            r"(?:x86_64|aarch64)\s+\d{3}\.\d+\.\d+\b",
            first_line,
        )
    )
    proprietary_kernel_module = bool(
        re.match(
            r"^NVRM version: NVIDIA UNIX (?:x86_64|aarch64) Kernel Module\s+"
            r"\d{3}\.\d+\.\d+\b",
            first_line,
        )
    )
    if open_kernel_module == proprietary_kernel_module:
        raise WorkloadError("M4 NVIDIA kernel-module flavor is unknown")
    known_risk = (
        open_kernel_module
        and major in NVIDIA_OPEN_UVM_HMM_BLOCKED_MAJOR_VERSIONS
    )
    if known_risk:
        try:
            hmm_text = hmm_path.read_text(
                encoding="ascii", errors="strict"
            ).strip()
        except OSError as error:
            raise WorkloadError(
                "M4 cannot verify NVIDIA UVM/HMM process-exit safety"
            ) from error
        if hmm_text not in {"N", "Y", "0", "1"}:
            raise WorkloadError("M4 NVIDIA UVM HMM state is malformed")
        disabled: bool | None = hmm_text in {"Y", "1"}
    else:
        disabled = None
    if known_risk and disabled is not True:
        raise WorkloadError(
            "M4 refuses NVIDIA 595 open-driver with UVM HMM enabled; "
            "apply the documented uvm_disable_hmm=1 workaround and reboot"
        )
    return {
        "schema": "gpmeep-v110-nvidia-uvm-hmm-safety-v1",
        "driver_version": version,
        "driver_module_flavor": "open" if open_kernel_module else "proprietary",
        "hmm_requirement": "must_be_disabled" if known_risk else "not_required",
        "uvm_disable_hmm": disabled,
        "blocked_open_driver_major_versions": sorted(
            NVIDIA_OPEN_UVM_HMM_BLOCKED_MAJOR_VERSIONS
        ),
    }


def validate_host_binding(retained: Any) -> None:
    if retained != host_binding():
        raise WorkloadError("M4 boot or CPU affinity changed")


def validate_repo_ownership(repo: pathlib.Path) -> None:
    if repo.resolve(strict=True) != REPO_DEFAULT.resolve(strict=True):
        raise WorkloadError("M4 repository does not own the executing controller")


def clean_environment(
    task: dict[str, Any], contract: dict[str, Any], runtime: pathlib.Path
) -> dict[str, str]:
    runtime.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "tmp"):
        (runtime / name).mkdir()
    python = pathlib.Path(contract["paths"]["python"])
    selected = task_gpu_devices(task, contract)
    environment = {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "HOME": str(runtime / "home"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "TMPDIR": str(runtime / "tmp"),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(selected),
    }
    if task["backend"] == "cpu":
        environment.update(
            {
                "JAX_PLATFORM_NAME": "cpu",
                "JAX_PLATFORMS": "cpu",
            }
        )
    return environment


def _code_paths() -> dict[str, pathlib.Path]:
    return {
        "controller": pathlib.Path(__file__),
        "plan_loader": SCRIPT_DIR / "m4_equal_work_plan.py",
        "semantic_replay": SCRIPT_DIR / "m4_equal_work_replay.py",
        "v110_runner": SCRIPT_DIR / "run_m3_performance_sample.py",
        "upstream_runner": SCRIPT_DIR / "run_v110_upstream_performance_sample.py",
        "v103_runner": SCRIPT_DIR / "run_v110_v103_performance_sample.py",
        "v103_identity": SCRIPT_DIR / "v103_runtime_identity.py",
        "v110_identity": SCRIPT_DIR / "v110_runtime_identity.py",
        "v103_oracle": SCRIPT_DIR / "run_v103_example_oracle.py",
        "automatic_common": SCRIPT_DIR / "run_m3_auto_example_case.py",
        "mpi_common": SCRIPT_DIR / "run_m3_mpi_example_case.py",
        "rank_launcher": SCRIPT_DIR / "m3_mpi_rank_launcher.py",
        "example_oracle": SCRIPT_DIR / "run_example_oracle.py",
        "unittest_preflight": SCRIPT_DIR / "preflight_unittest_workload.py",
        "unittest_adjoint_cyl_case": REPO_DEFAULT
        / "python/tests/test_adjoint_cyl.py",
        "unittest_adjoint_cyl_contract": REPO_DEFAULT
        / "python/tests/adjoint_cyl_fixed_work_contract.py",
        "validation_runner": SCRIPT_DIR / "run_validation.py",
        "manifest": SCRIPT_DIR / "manifest.json",
        "development_contract": REPO_DEFAULT / "packaging/v110-development-contract.json",
        "equal_work_plan": REPO_DEFAULT / "packaging/v110-m4-equal-work-plan.json",
        "inversion_inventory": REPO_DEFAULT / "packaging/v110-performance-inversion-inventory.json",
        "inversion_validator": REPO_DEFAULT / "scripts/validate-v110-inversion-registry.py",
        "inversion_experiment_producer": REPO_DEFAULT / "scripts/v110-inversion-experiment-sample.py",
        "inversion_qualified_child": REPO_DEFAULT / "scripts/v110-inversion-qualified-child.py",
        "inversion_archive_inspector": REPO_DEFAULT / "scripts/v110-final-release-receipt.py",
        "installed_provenance_verifier": REPO_DEFAULT / "scripts/verify-gpmeep-package.py",
        "installed_provenance_launcher": REPO_DEFAULT / "scripts/gpmeep-verify-provenance-launcher.sh",
        "provenance": SCRIPT_DIR.parent / "gpmeep_provenance.py",
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }


def _code_records(
    repo: pathlib.Path,
) -> tuple[dict[str, StableFile], dict[str, dict[str, Any]]]:
    paths = _code_paths()
    paths["development_contract"] = repo / "packaging/v110-development-contract.json"
    paths["equal_work_plan"] = repo / "packaging/v110-m4-equal-work-plan.json"
    paths["unittest_adjoint_cyl_case"] = repo / "python/tests/test_adjoint_cyl.py"
    paths["unittest_adjoint_cyl_contract"] = (
        repo / "python/tests/adjoint_cyl_fixed_work_contract.py"
    )
    paths["inversion_inventory"] = repo / "packaging/v110-performance-inversion-inventory.json"
    paths["inversion_validator"] = repo / "scripts/validate-v110-inversion-registry.py"
    paths["inversion_experiment_producer"] = repo / "scripts/v110-inversion-experiment-sample.py"
    paths["inversion_qualified_child"] = repo / "scripts/v110-inversion-qualified-child.py"
    paths["inversion_archive_inspector"] = repo / "scripts/v110-final-release-receipt.py"
    paths["installed_provenance_verifier"] = repo / "scripts/verify-gpmeep-package.py"
    paths["installed_provenance_launcher"] = repo / "scripts/gpmeep-verify-provenance-launcher.sh"
    stables = {name: StableFile(path, f"M4 {name}") for name, path in paths.items()}
    return stables, {
        name: matrix.stable_identity_record(stable, repo)
        for name, stable in stables.items()
    }


def _absolute_path(path: pathlib.Path, *, directory: bool, label: str) -> pathlib.Path:
    resolved = pathlib.Path(os.path.abspath(path)).resolve(strict=True)
    metadata = resolved.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or directory != stat.S_ISDIR(metadata.st_mode)
        or (not directory and not stat.S_ISREG(metadata.st_mode))
    ):
        raise WorkloadError(f"M4 {label} path type differs")
    return resolved


def _runtime_paths(args: argparse.Namespace, repo: pathlib.Path) -> dict[str, str]:
    raw = {
        "repo": (repo, True),
        "python": (args.python, False),
        "build_python": (args.build_python, True),
        "install_prefix": (args.install_prefix, True),
        "build_receipt": (args.build_receipt, False),
        "mpiexec": (args.mpiexec, False),
        "upstream_receipt": (args.upstream_receipt, False),
        "v103_install_prefix": (args.v103_install_prefix, True),
        "v103_provenance_report": (args.v103_provenance_report, False),
        "v103_package": (args.v103_package, False),
        "v110_provenance_report": (args.v110_provenance_report, False),
        "v110_package": (args.v110_package, False),
        "manifest": (SCRIPT_DIR / "manifest.json", False),
        "v110_runner": (SCRIPT_DIR / "run_m3_performance_sample.py", False),
        "upstream_runner": (SCRIPT_DIR / "run_v110_upstream_performance_sample.py", False),
        "v103_runner": (SCRIPT_DIR / "run_v110_v103_performance_sample.py", False),
    }
    return {
        name: str(_absolute_path(path, directory=directory, label=name))
        for name, (path, directory) in raw.items()
    }


def _file_input_record(path: pathlib.Path, label: str) -> dict[str, Any]:
    return absolute_file_record(path, label)


def _v110_archive_authority(
    paths: dict[str, str] | dict[str, pathlib.Path],
    report: dict[str, Any],
    runtime: dict[str, pathlib.Path],
) -> dict[str, Any]:
    producer_path = (
        pathlib.Path(paths["repo"]) / "scripts/v110-inversion-experiment-sample.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gpmeep_m4_inversion_archive_preflight", producer_path
    )
    if spec is None or spec.loader is None:
        raise WorkloadError("M4 inversion archive preflight cannot be loaded")
    producer = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = producer
    try:
        spec.loader.exec_module(producer)
        archive, _payload, _semantics = producer._installed_archive_authority(
            repo=pathlib.Path(paths["repo"]),
            package=runtime["package"],
            prefix=runtime["install_prefix"],
            report=report,
            runtime=runtime,
        )
    except Exception as error:
        raise WorkloadError(
            f"M4 inversion archive preflight failed: {error}"
        ) from error
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return archive


def _v110_installed_identity(
    paths: dict[str, str] | dict[str, pathlib.Path], expected_commit: str
) -> tuple[dict[str, Any], dict[str, pathlib.Path], dict[str, Any]]:
    provenance_path = pathlib.Path(paths["v110_provenance_report"])
    raw = validation.load_json(provenance_path)
    try:
        report, installed = v110_identity.validate_report(
            provenance_path,
            pathlib.Path(paths["v110_package"]),
            pathlib.Path(str(raw.get("prefix", ""))),
            expected_commit,
        )
        runtime = v110_identity.runtime_paths(report, installed)
        archive = _v110_archive_authority(paths, report, runtime)
        live = v110_identity.live_verify(report, installed)
    except (
        OSError,
        ValueError,
        WorkloadError,
        v110_identity.V110IdentityError,
    ) as error:
        raise WorkloadError(
            f"M4 v1.1.0 installed package identity differs: {error}"
        ) from error
    runtime_record = {
        name: str(runtime[name])
        for name in (
            "python",
            "build_python",
            "install_prefix",
            "build_receipt",
            "mpiexec",
            "package",
            "provenance_report",
            "verifier",
        )
    }
    return report, runtime, {
        "paths": runtime_record,
        "live": live,
        "archive": archive,
    }


def _v103_installed_identity(
    paths: dict[str, str] | dict[str, pathlib.Path],
) -> dict[str, Any]:
    """Re-prove the frozen runtime and rerun its package-owned verifier."""
    namespace = argparse.Namespace(
        repo=pathlib.Path(paths["repo"]),
        manifest=pathlib.Path(paths["manifest"]),
        install_prefix=pathlib.Path(paths["v103_install_prefix"]),
        provenance_report=pathlib.Path(paths["v103_provenance_report"]),
        package=pathlib.Path(paths["v103_package"]),
    )
    try:
        report, installed, runtime = v103_sample.runtime_paths(namespace)
        live = v103_sample.live_verify(report, installed)
    except (OSError, ValueError, WorkloadError) as error:
        raise WorkloadError(
            f"M4 frozen v1.0.3 installed package identity differs: {error}"
        ) from error
    return {
        "report_id": report["report_id"],
        "package_sha256": report["package_sha256"],
        "source_commit": report["manifest"]["source_commit"],
        "paths": {name: str(path) for name, path in installed.items()},
        "runtime_contract": runtime,
        "live_verification": live,
    }


def _unittest_runtime_specs(
    paths: dict[str, str] | dict[str, pathlib.Path],
    upstream_receipt: dict[str, Any],
    v103_identity: dict[str, Any],
    v110_report: dict[str, Any],
    v110_runtime: dict[str, pathlib.Path],
) -> dict[str, dict[str, Any]]:
    upstream_namespace = argparse.Namespace(
        repo=pathlib.Path(paths["repo"]),
        manifest=pathlib.Path(paths["manifest"]),
        upstream_receipt=pathlib.Path(paths["upstream_receipt"]),
    )
    upstream = upstream_sample.runtime_paths(
        upstream_namespace, upstream_receipt
    )
    v103 = {
        name: pathlib.Path(value)
        for name, value in v103_identity["paths"].items()
    }
    v110_prefix = pathlib.Path(v110_runtime["install_prefix"])
    return {
        "upstream_meep_cpu": {
            "python": upstream["python"],
            "site_packages": upstream["site_packages"],
            "meep_module": upstream["site_packages"] / "meep/__init__.py",
            "library_paths": [
                upstream["libmeep"].parent,
                upstream["exact_lock_prefix"] / "lib",
            ],
            "fontconfig": upstream["fontconfig"],
            "gpmeep": False,
        },
        "gpmeep_v103": {
            "python": v103["python"],
            "site_packages": v103["site_packages"],
            "meep_module": v103["site_packages"] / "meep/__init__.py",
            "library_paths": [v103["install_prefix"] / "lib"],
            "fontconfig": v103["fontconfig"],
            "gpmeep": True,
        },
        "gpmeep_v110": {
            "python": pathlib.Path(v110_runtime["python"]),
            "site_packages": pathlib.Path(v110_runtime["build_python"]),
            "meep_module": pathlib.Path(
                v110_report["runtime"]["module"]["path"]
            ),
            "library_paths": [v110_prefix / "lib"],
            "fontconfig": v110_prefix / "etc/fonts/fonts.conf",
            "gpmeep": True,
        },
    }


def _selected_unittest_contracts(
    repo: pathlib.Path, tasks: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    cases = {
        case["path"]: case
        for case in validation.materialize_cases(
            validation.load_json(repo / "scripts/python-validation/manifest.json"),
            repo,
        )
    }
    selected = {
        task["case_path"]
        for task in tasks
        if cases.get(task["case_path"], {}).get("kind") == "unittest"
    }
    if selected != set(UNITTEST_PREFLIGHT_CASES):
        raise WorkloadError("M4 unittest preflight inventory differs")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(selected):
        manifest_contract = cases[path].get("expected_unittest")
        performance_contract = v110_sample.DIRECT_PERFORMANCE_ORACLE_CONTRACTS.get(
            path
        )
        if (
            not isinstance(manifest_contract, dict)
            or manifest_contract.get("count")
            != len(manifest_contract.get("identities", []))
            or not isinstance(performance_contract, dict)
            or performance_contract.get("manifest_mode") != "default-unittest"
            or performance_contract.get("expected_unittest_count")
            != manifest_contract.get("count")
            or type(performance_contract.get("expected_run_count")) is not int
            or performance_contract["expected_run_count"] <= 0
        ):
            raise WorkloadError("M4 unittest manifest contract differs")
        result[path] = {
            **UNITTEST_PREFLIGHT_CASES[path],
            "test_identities": list(manifest_contract["identities"]),
            "expected_simulation_run_count": performance_contract[
                "expected_run_count"
            ],
        }
    return result


def _run_unittest_import_preflight(
    repo: pathlib.Path,
    path: str,
    contract: dict[str, Any],
    runtime_name: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    helper = repo / "scripts/python-validation/preflight_unittest_workload.py"
    command = [
        str(runtime["python"]),
        "-I",
        "-B",
        str(helper),
        "--case",
        str(repo / path),
        "--meep-site-packages",
        str(runtime["site_packages"]),
        "--expected-meep-module",
        str(runtime["meep_module"]),
        "--expected-simulation-run-count",
        str(contract["expected_simulation_run_count"]),
    ]
    for identity in contract["test_identities"]:
        command.extend(("--expected-test-id", identity))
    command.extend(
        (
            "--case-contract-variable",
            contract["case_contract_variable"],
            "--expected-case-contract",
            json.dumps(
                contract["case_contract"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "--wrapper-helper",
            contract["wrapper_helper"],
        )
    )
    for invocation in contract["wrapper_invocations"]:
        command.extend(
            (
                "--expected-wrapper-invocation",
                json.dumps(
                    invocation,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        )

    with tempfile.TemporaryDirectory(
        prefix=f"gpmeep-m4-unittest-{runtime_name}-"
    ) as temporary:
        root = pathlib.Path(temporary).resolve()
        for name in ("home", "cache", "tmp", "pycache"):
            (root / name).mkdir()
        environment = {
            "HOME": str(root / "home"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TMPDIR": str(root / "tmp"),
            "PATH": f"{pathlib.Path(runtime['python']).parent}:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "FONTCONFIG_FILE": str(runtime["fontconfig"]),
            "LD_LIBRARY_PATH": os.pathsep.join(
                str(item) for item in runtime["library_paths"]
            ),
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "OMPI_MCA_osc": "^ucx",
        }
        if runtime["gpmeep"]:
            environment["MEEP_GPU_BACKEND"] = "cpu"
        # ``-I`` ignores PYTHON* environment variables.  Pass the empty cache
        # prefix on the interpreter command line so ignored, untracked
        # ``__pycache__`` files can never replace the source being attested.
        bounded_command = [
            *command[:3],
            "-X",
            f"pycache_prefix={root / 'pycache'}",
            *command[3:],
        ]
        try:
            process = validation.run_bounded_process(
                bounded_command,
                cwd=repo,
                env=environment,
                timeout_seconds=120,
                stdout_file=root / "stdout.log",
                stderr_file=root / "stderr.log",
                stdout_limit=UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
                stderr_limit=UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
                combined_limit=2 * UNITTEST_PREFLIGHT_MAXIMUM_OUTPUT_BYTES,
            )
            stdout_bytes = (root / "stdout.log").read_bytes()
        except (OSError, RuntimeError) as error:
            raise WorkloadError(
                f"M4 {runtime_name} unittest import preflight failed: {error}"
            ) from error
    if (
        process.get("exit_code") != 0
        or process.get("timeout") is not False
        or process.get("output_limit") is not None
        or process.get("spawn_error") is not None
        or process.get("stdout_size_bytes") != len(stdout_bytes)
        or process.get("stderr_size_bytes") != 0
    ):
        raise WorkloadError(
            f"M4 {runtime_name} unittest import preflight failed"
        )
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise WorkloadError("M4 unittest preflight output is not UTF-8") from error
    payloads = [
        line.removeprefix(UNITTEST_PREFLIGHT_PREFIX)
        for line in stdout.splitlines()
        if line.startswith(UNITTEST_PREFLIGHT_PREFIX)
    ]
    if len(payloads) != 1:
        raise WorkloadError("M4 unittest preflight record count differs")
    try:
        result = json.loads(
            payloads[0],
            object_pairs_hook=validation.unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                WorkloadError(
                    f"M4 unittest preflight contains non-finite value {value!r}"
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise WorkloadError("M4 unittest preflight record is invalid") from error
    expected_keys = {
        "schema",
        "state",
        "case_path",
        "python_executable",
        "meep_module",
        "test_count",
        "test_identities",
        "case_contract",
        "wrapper_invocations",
        "expected_simulation_run_count",
        "simulation_workloads_launched",
        "parameterized_imported",
        "guarded_simulation_entrypoints",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected_keys
        or result.get("schema")
        != "gpmeep-m4-unittest-import-preflight-v1"
        or result.get("state") != "PASS"
        or result.get("case_path") != str((repo / path).resolve(strict=True))
        or result.get("python_executable")
        != str(pathlib.Path(runtime["python"]).resolve(strict=True))
        or result.get("meep_module")
        != str(pathlib.Path(runtime["meep_module"]).resolve(strict=True))
        or result.get("test_count") != len(contract["test_identities"])
        or result.get("test_identities") != contract["test_identities"]
        or result.get("case_contract") != contract["case_contract"]
        or result.get("wrapper_invocations") != contract["wrapper_invocations"]
        or result.get("expected_simulation_run_count")
        != contract["expected_simulation_run_count"]
        or result.get("simulation_workloads_launched") != 0
        or result.get("parameterized_imported") is not False
        or result.get("guarded_simulation_entrypoints")
        != UNITTEST_PREFLIGHT_GUARDED_SIMULATION_ENTRYPOINTS
    ):
        raise WorkloadError("M4 unittest import preflight record differs")
    return result


def _unittest_import_preflight(
    repo: pathlib.Path,
    tasks: list[dict[str, Any]],
    paths: dict[str, str] | dict[str, pathlib.Path],
    upstream_receipt: dict[str, Any],
    v103_identity: dict[str, Any],
    v110_report: dict[str, Any],
    v110_runtime: dict[str, pathlib.Path],
) -> dict[str, Any]:
    contracts = _selected_unittest_contracts(repo, tasks)
    runtimes = _unittest_runtime_specs(
        paths,
        upstream_receipt,
        v103_identity,
        v110_report,
        v110_runtime,
    )
    cases = []
    for path, contract in contracts.items():
        cases.append(
            {
                "case_path": path,
                "runtimes": {
                    name: _run_unittest_import_preflight(
                        repo, path, contract, name, runtime
                    )
                    for name, runtime in runtimes.items()
                },
            }
        )
    return {
        "schema": UNITTEST_PREFLIGHT_SCHEMA,
        "state": "PASS",
        "case_count": len(cases),
        "runtime_count": len(runtimes),
        "simulation_workloads_launched": 0,
        "cases": cases,
    }


def build_contract(
    args: argparse.Namespace,
    repo: pathlib.Path,
    plan: dict[str, Any],
    tasks: list[dict[str, Any]],
    code: dict[str, Any],
    devices: tuple[str, str],
) -> dict[str, Any]:
    current_host_binding = host_binding()
    paths = _runtime_paths(args, repo)
    receipt = provenance.verify_build_receipt(pathlib.Path(paths["build_receipt"]), repo)
    upstream_receipt, upstream_verified = upstream_sample.verify_receipt(
        repo, pathlib.Path(paths["upstream_receipt"])
    )
    if receipt.get("build_kind") != validation.EXPECTED_BUILD_KIND:
        raise WorkloadError("M4 v1.1.0 build kind differs")
    if receipt.get("git_status_porcelain") != []:
        raise WorkloadError("M4 candidate build was not made from a clean checkout")
    if upstream_receipt.get("receipt_id") != upstream_verified.get("receipt_id"):
        raise WorkloadError("M4 upstream receipt identity differs")
    v110_provenance, v110_runtime, installed_identity = _v110_installed_identity(
        paths, receipt["git_head"]
    )
    v103_installed_identity = _v103_installed_identity(paths)
    unittest_import_preflight = _unittest_import_preflight(
        repo,
        tasks,
        paths,
        upstream_receipt,
        v103_installed_identity,
        v110_provenance,
        v110_runtime,
    )
    v110_package_sha256 = v110_provenance["package_sha256"]
    v110_candidate_receipt_id = v110_provenance["v110_candidate_gate"][
        "receipt_id"
    ]
    if pathlib.Path(paths["build_python"]).parent / "build-provenance.json" != pathlib.Path(paths["build_receipt"]):
        raise WorkloadError("M4 v1.1.0 build path relationship differs")
    return {
        "schema": SCHEMA,
        "paths": paths,
        "plan": plan,
        "tasks": tasks,
        "evidence_code": code,
        "runtime_inputs": {
            "v110_build_receipt": _file_input_record(pathlib.Path(paths["build_receipt"]), "M4 v1.1.0 receipt"),
            "upstream_receipt": _file_input_record(pathlib.Path(paths["upstream_receipt"]), "M4 upstream receipt"),
            "v103_provenance_report": _file_input_record(pathlib.Path(paths["v103_provenance_report"]), "M4 v1.0.3 provenance"),
            "v103_package": _file_input_record(pathlib.Path(paths["v103_package"]), "M4 v1.0.3 package"),
            "v103_installed_identity": v103_installed_identity,
            "v110_provenance_report": _file_input_record(pathlib.Path(paths["v110_provenance_report"]), "M4 v1.1.0 provenance"),
            "v110_package": _file_input_record(pathlib.Path(paths["v110_package"]), "M4 v1.1.0 package"),
            "v110_package_sha256": v110_package_sha256,
            "v110_candidate_receipt_id": v110_candidate_receipt_id,
            "v110_receipt_id": receipt["receipt_id"],
            "v110_installed_report_id": v110_provenance["report_id"],
            "v110_installed_runtime": installed_identity["paths"],
            "v110_live_verification": installed_identity["live"],
            "v110_archive_verification": installed_identity["archive"],
            "v110_measured_build_mode": v110_provenance[
                "v110_candidate_gate"
            ]["build_mode"],
            "unittest_import_preflight": unittest_import_preflight,
            "upstream_receipt_id": upstream_verified["receipt_id"],
            "v110_source_identity": {
                "git_head": receipt["git_head"],
                "git_status_porcelain": receipt["git_status_porcelain"],
                "source_snapshot": receipt["source_end"],
            },
        },
        "host_binding": current_host_binding,
        "gpu_devices": list(devices),
        "nvidia_smi": absolute_file_record(pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"),
        "limits": {
            "task_timeout_seconds": args.task_timeout_seconds,
            "stdout_limit_bytes": args.stdout_limit_mib * 1024**2,
            "telemetry_interval_seconds": args.telemetry_interval_seconds,
        },
        "gates": {
            "initial_free_bytes": INITIAL_FREE_BYTES,
            "per_task_free_bytes": PER_TASK_FREE_BYTES,
            "minimum_free_inodes": MINIMUM_FREE_INODES,
            "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
            "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
            "minimum_gpu_free_memory_fraction": MINIMUM_GPU_FREE_MEMORY_FRACTION,
            "maximum_gpu_temperature_celsius": MAXIMUM_GPU_TEMPERATURE_CELSIUS,
            "maximum_host_temperature_millidegrees": MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES,
        },
    }


def validate_contract(contract: Any) -> tuple[pathlib.Path, list[dict[str, Any]], tuple[str, str]]:
    keys = {
        "schema", "paths", "plan", "tasks", "evidence_code", "runtime_inputs",
        "host_binding", "gpu_devices", "nvidia_smi", "limits", "gates",
    }
    if not isinstance(contract, dict) or set(contract) != keys or contract.get("schema") != SCHEMA:
        raise WorkloadError("M4 sealed contract schema differs")
    current_host_binding = host_binding()
    if contract["host_binding"] != current_host_binding:
        raise WorkloadError("M4 boot, CPU affinity, or NVIDIA UVM/HMM state changed")
    paths = contract["paths"]
    if not isinstance(paths, dict) or set(paths) != PATH_KEYS:
        raise WorkloadError("M4 path contract differs")
    resolved = {name: pathlib.Path(value) for name, value in paths.items()}
    if any(not path.is_absolute() or path.resolve(strict=True) != path for path in resolved.values()):
        raise WorkloadError("M4 runtime path is not canonical")
    repo = resolved["repo"]
    validate_repo_ownership(repo)
    plan = equal_work_plan.load_plan(repo)
    tasks = equal_work_plan.build_tasks(plan)
    if contract["plan"] != plan or contract["tasks"] != tasks or len(tasks) != TASK_COUNT:
        raise WorkloadError("M4 plan/tasks were not re-derived")
    expected_code = _code_paths()
    expected_code["development_contract"] = repo / "packaging/v110-development-contract.json"
    expected_code["equal_work_plan"] = repo / "packaging/v110-m4-equal-work-plan.json"
    expected_code["unittest_adjoint_cyl_case"] = repo / "python/tests/test_adjoint_cyl.py"
    expected_code["inversion_inventory"] = repo / "packaging/v110-performance-inversion-inventory.json"
    expected_code["inversion_validator"] = repo / "scripts/validate-v110-inversion-registry.py"
    expected_code["inversion_experiment_producer"] = repo / "scripts/v110-inversion-experiment-sample.py"
    expected_code["inversion_qualified_child"] = repo / "scripts/v110-inversion-qualified-child.py"
    expected_code["inversion_archive_inspector"] = repo / "scripts/v110-final-release-receipt.py"
    expected_code["installed_provenance_verifier"] = repo / "scripts/verify-gpmeep-package.py"
    expected_code["installed_provenance_launcher"] = repo / "scripts/gpmeep-verify-provenance-launcher.sh"
    if set(contract["evidence_code"]) != set(expected_code):
        raise WorkloadError("M4 evidence-code inventory differs")
    for name, path in expected_code.items():
        stable = matrix.stable_record(repo, contract["evidence_code"][name], f"M4 {name}")
        try:
            if stable.path != path.resolve(strict=True):
                raise WorkloadError(f"M4 code path differs: {name}")
        finally:
            stable.close()
    runtime_inputs = contract["runtime_inputs"]
    if not isinstance(runtime_inputs, dict) or set(runtime_inputs) != RUNTIME_INPUT_KEYS:
        raise WorkloadError("M4 runtime input inventory differs")
    record_paths = {
        "v110_build_receipt": "build_receipt",
        "upstream_receipt": "upstream_receipt",
        "v103_provenance_report": "v103_provenance_report",
        "v103_package": "v103_package",
        "v110_provenance_report": "v110_provenance_report",
        "v110_package": "v110_package",
    }
    for record_name, path_name in record_paths.items():
        if runtime_inputs.get(record_name) != _file_input_record(resolved[path_name], f"M4 {record_name}"):
            raise WorkloadError(f"M4 runtime input changed: {record_name}")
    receipt = provenance.verify_build_receipt(resolved["build_receipt"], repo)
    upstream_receipt, upstream_verified = upstream_sample.verify_receipt(
        repo, resolved["upstream_receipt"]
    )
    v110_provenance, v110_runtime, installed_identity = _v110_installed_identity(
        resolved, receipt["git_head"]
    )
    v103_installed_identity = _v103_installed_identity(resolved)
    unittest_import_preflight = _unittest_import_preflight(
        repo,
        tasks,
        resolved,
        upstream_receipt,
        v103_installed_identity,
        v110_provenance,
        v110_runtime,
    )
    v110_package_sha256 = v110_provenance["package_sha256"]
    if (
        runtime_inputs.get("v110_receipt_id") != receipt.get("receipt_id")
        or runtime_inputs.get("upstream_receipt_id") != upstream_verified.get("receipt_id")
        or runtime_inputs.get("v110_source_identity")
        != {
            "git_head": receipt.get("git_head"),
            "git_status_porcelain": receipt.get("git_status_porcelain"),
            "source_snapshot": receipt.get("source_end"),
        }
        or receipt.get("git_status_porcelain") != []
        or runtime_inputs.get("v103_installed_identity")
        != v103_installed_identity
        or runtime_inputs.get("v110_package_sha256") != v110_package_sha256
        or runtime_inputs.get("v110_candidate_receipt_id")
        != v110_provenance.get("v110_candidate_gate", {}).get("receipt_id")
        or runtime_inputs.get("v110_installed_report_id")
        != v110_provenance.get("report_id")
        or runtime_inputs.get("v110_installed_runtime")
        != installed_identity["paths"]
        or runtime_inputs.get("v110_live_verification")
        != installed_identity["live"]
        or runtime_inputs.get("v110_archive_verification")
        != installed_identity["archive"]
        or runtime_inputs.get("v110_measured_build_mode")
        != v110_provenance.get("v110_candidate_gate", {}).get("build_mode")
        or runtime_inputs.get("unittest_import_preflight")
        != unittest_import_preflight
        or v110_provenance.get("state") != "PASS"
        or v110_provenance.get("package_sha256") != v110_package_sha256
        or v110_provenance.get("manifest", {}).get("source_commit")
        != receipt.get("git_head")
    ):
        raise WorkloadError("M4 receipt IDs differ")
    devices = normalized_devices(",".join(contract["gpu_devices"]))
    if list(devices) != contract["gpu_devices"]:
        raise WorkloadError("M4 GPU UUID order differs")
    if contract["nvidia_smi"] != absolute_file_record(pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"):
        raise WorkloadError("M4 nvidia-smi identity differs")
    validate_host_binding(contract["host_binding"])
    expected_gates = {
        "initial_free_bytes": INITIAL_FREE_BYTES,
        "per_task_free_bytes": PER_TASK_FREE_BYTES,
        "minimum_free_inodes": MINIMUM_FREE_INODES,
        "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
        "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
        "minimum_gpu_free_memory_fraction": MINIMUM_GPU_FREE_MEMORY_FRACTION,
        "maximum_gpu_temperature_celsius": MAXIMUM_GPU_TEMPERATURE_CELSIUS,
        "maximum_host_temperature_millidegrees": MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES,
    }
    if contract["gates"] != expected_gates:
        raise WorkloadError("M4 resource gates differ")
    limits = contract["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"task_timeout_seconds", "stdout_limit_bytes", "telemetry_interval_seconds"}
        or not 60 <= limits["task_timeout_seconds"] <= 48 * 3600
        or type(limits["stdout_limit_bytes"]) is not int
        or not 1024**2 <= limits["stdout_limit_bytes"] <= 1024**3
        or limits["stdout_limit_bytes"] % 1024**2
        or not 1.0 <= float(limits["telemetry_interval_seconds"]) <= 60.0
    ):
        raise WorkloadError("M4 sealed limits differ")
    return repo, tasks, devices


def observer_specs(
    task: dict[str, Any], root_pid: int, output: pathlib.Path, contract: dict[str, Any]
) -> list[observed.ObserverSpec]:
    root = output.joinpath(*task_telemetry_relative(task).parts)
    python = contract["paths"]["python"]
    interval = contract["limits"]["telemetry_interval_seconds"]
    label = task_name(task)
    specs = [
        observed.ObserverSpec(
            "host",
            (
                python,
                str(USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py"),
                "--root-pid", str(root_pid), "--output", str(root / "host.jsonl"),
                "--label", label, "--interval-seconds", repr(interval),
            ),
            root / "host-monitor.log",
        )
    ]
    selected = task_gpu_devices(task, contract)
    monitored = selected if selected else tuple(contract["gpu_devices"])
    if monitored:
        selected_flags = tuple(
            value for device in monitored for value in ("--selected-uuid", device)
        )
        specs.append(
            observed.ObserverSpec(
                "gpu",
                (
                    python,
                    str(USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py"),
                    "--root-pid", str(root_pid), "--output", str(root / "gpu.jsonl"),
                    "--label", label, *selected_flags,
                    "--interval-seconds", repr(interval),
                ),
                root / "gpu-monitor.log",
            )
        )
    return specs


def _code_sha256(contract: dict[str, Any], name: str) -> str:
    value = contract.get("evidence_code", {}).get(name, {}).get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadError(f"M4 code digest differs: {name}")
    return value


def derive_telemetry(
    task: dict[str, Any], process: dict[str, Any], output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    root = output.joinpath(*task_telemetry_relative(task).parts)
    gates = contract["gates"]
    selected = task_gpu_devices(task, contract)
    monitored = selected if selected else tuple(contract["gpu_devices"])
    return telemetry_replay.verify_pair(
        host_stream=root / "host.jsonl",
        host_terminal=root / "host.jsonl.COMPLETE.json",
        gpu_stream=root / "gpu.jsonl",
        gpu_terminal=root / "gpu.jsonl.COMPLETE.json",
        expected_root_pid=process["command_pid"],
        expected_label=task_name(task),
        expected_host_monitor_sha256=_code_sha256(contract, "host_telemetry"),
        expected_gpu_uuids=monitored,
        expected_gpu_monitor_sha256=_code_sha256(contract, "gpu_telemetry"),
        expected_nvidia_smi_sha256=contract["nvidia_smi"]["sha256"],
        minimum_host_available_bytes=gates["minimum_host_available_bytes"],
        minimum_cgroup_free_bytes=gates["minimum_cgroup_free_bytes"],
        maximum_host_temperature_millidegrees=gates["maximum_host_temperature_millidegrees"],
        minimum_gpu_free_memory_fraction=gates["minimum_gpu_free_memory_fraction"],
        maximum_gpu_temperature_celsius=gates["maximum_gpu_temperature_celsius"],
        expected_interval_seconds=contract["limits"]["telemetry_interval_seconds"],
        expected_process_started_unix_seconds=process["started_unix_seconds"],
        expected_process_ended_unix_seconds=process["ended_unix_seconds"],
        expected_workload_gpu_idle=task["backend"] == "cpu",
    )


def publish_telemetry(
    task: dict[str, Any], process: dict[str, Any], observers: list[dict[str, Any]],
    output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M4 observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    replay = derive_telemetry(task, process, output, contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, replay)
    return matrix.stable_file_record(path, output, "M4 telemetry replay")


def validate_telemetry(
    task: dict[str, Any], process: dict[str, Any], observers: list[dict[str, Any]],
    retained: Any, output: pathlib.Path, contract: dict[str, Any]
) -> None:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M4 observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    if retained != matrix.stable_file_record(path, output, "M4 telemetry replay"):
        raise WorkloadError("M4 telemetry replay record differs")
    with StableFile(path, "M4 telemetry replay") as stable:
        with stable.file_object() as handle:
            saved = json.load(handle)
    if saved != derive_telemetry(task, process, output, contract):
        raise WorkloadError("M4 telemetry replay was not re-derived")


def _sample_verifier(task: dict[str, Any]):
    return {
        "upstream": upstream_sample.verify_complete,
        "v103": v103_sample.verify_complete,
        "v110": v110_sample.verify_complete,
    }[task["runtime"]]


def derive_sample_replay(
    task: dict[str, Any],
    output: pathlib.Path,
    contract: dict[str, Any],
    *,
    v110_runtime_session: v110_sample.RuntimeVerificationSession | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = output.joinpath(*task_sample_relative(task).parts)
    verifier = _sample_verifier(task)
    report = (
        verifier(root, runtime_session=v110_runtime_session)
        if task["runtime"] == "v110" and v110_runtime_session is not None
        else verifier(root)
    )
    equal_work_replay.validate_sample_report(task, report)
    if task["runtime"] == "v110" and report.get("receipt_id") != contract[
        "runtime_inputs"
    ]["v110_installed_report_id"]:
        raise WorkloadError("M4 v1.1.0 child installed-package identity differs")
    if task["runtime"] == "upstream" and report.get("receipt_id") != contract["runtime_inputs"]["upstream_receipt_id"]:
        raise WorkloadError("M4 upstream child receipt differs")
    selected = list(task_gpu_devices(task, contract))
    if task["runtime"] in {"v103", "v110"}:
        if report.get("gpu_devices") != contract["gpu_devices"]:
            raise WorkloadError("M4 child GPU inventory differs")
    lane = report.get("lane")
    if not isinstance(lane, dict):
        raise WorkloadError("M4 child lane evidence differs")
    if task["backend"] == "cuda":
        if lane.get("requested_backend") != "cuda" or not selected:
            raise WorkloadError("M4 CUDA child did not retain strict CUDA selection")
        rank_records = lane.get("rank_records")
        if not isinstance(rank_records, list) or len(rank_records) not in {1, 2}:
            raise WorkloadError("M4 CUDA rank inventory differs")
        for rank in rank_records:
            if (
                rank.get("backend_contract") != "PASS"
                or not isinstance(rank.get("selected_device_identifier"), str)
            ):
                raise WorkloadError("M4 CUDA child fallback/device evidence differs")
        if {
            v110_sample.mpi_case._identity(rank["selected_device_identifier"])
            for rank in rank_records
        } != {v110_sample.mpi_case._identity(device) for device in selected}:
            raise WorkloadError("M4 CUDA child used an unselected GPU")
    retained = {
        "complete": matrix.stable_file_record(root / "COMPLETE", output, "M4 child COMPLETE"),
        "report": matrix.stable_file_record(root / "report.json", output, "M4 child report"),
        "schema": report["schema"],
        "receipt_id": report["receipt_id"],
        "timing": report["timing"],
    }
    return retained, report


def _read_stdout(output: pathlib.Path, task: dict[str, Any]) -> str:
    path = output.joinpath(*task_sample_relative(task).parts, "lane", "stdout.log")
    with StableFile(path, "M4 child stdout") as stable:
        if stable.initial_stat.st_size > 256 * 1024**2:
            raise WorkloadError("M4 child stdout exceeds replay bound")
        with stable.file_object() as handle:
            return handle.read().decode("utf-8", errors="strict")


def _comparison_contract(case: dict[str, Any]) -> tuple[dict[str, Any], str]:
    comparison = case["comparison"]
    if comparison.get("mode") == "json_metrics":
        return comparison, "manifest-json-metrics"
    if comparison.get("mode") == "embedded_oracle":
        return (
            {
                "mode": "json_metrics",
                "prefix": GENERIC_METRIC_PREFIX,
                "tolerances": {"*": {"rtol": 0.005, "atol": 1e-6}},
            },
            "embedded-oracle-plus-generic-metrics",
        )
    raise WorkloadError("M4 numerical comparison mode differs")


def _diagnostic_copy(
    comparison: dict[str, Any], diagnostic_rules: set[str]
) -> dict[str, Any]:
    copied = dict(comparison)
    tolerances = {
        name: dict(tolerance)
        for name, tolerance in comparison["tolerances"].items()
    }
    missing = diagnostic_rules - set(tolerances)
    if missing:
        raise WorkloadError(
            "M4 diagnostic comparison rules are absent: "
            + ", ".join(sorted(missing))
        )
    for name in diagnostic_rules:
        tolerances[name]["diagnostic_only"] = True
    copied["tolerances"] = tolerances
    return copied


def _near2far_upstream_forward_contract(case: dict[str, Any]) -> dict[str, Any]:
    comparison, _method = _comparison_contract(case)
    return _diagnostic_copy(
        comparison,
        {
            "gradient_l2",
            "gradient_by_frequency_l2*",
            "center_parameter_gradient",
            "derivative_checks*.adjoint_derivative",
            "derivative_checks*.adjoint_derivative_by_frequency*",
            "derivative_checks*.relative_derivative_error",
            "maximum_relative_derivative_error",
        },
    )


def _waveguide_upstream_optimizer_contract(
    case: dict[str, Any],
) -> dict[str, Any]:
    comparison, _method = _comparison_contract(case)
    return _diagnostic_copy(
        comparison, set(WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES)
    )


def _waveguide_metrics(
    text: str, comparison: dict[str, Any]
) -> dict[str, float]:
    metrics = validation.extract_json_metrics(text, comparison)
    keys = frozenset(metrics)
    if keys != WAVEGUIDE_EXPECTED_METRIC_KEYS:
        missing = sorted(WAVEGUIDE_EXPECTED_METRIC_KEYS - keys)
        unexpected = sorted(keys - WAVEGUIDE_EXPECTED_METRIC_KEYS)
        raise WorkloadError(
            "M4 waveguide metric inventory differs: "
            f"count={len(keys)}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    if (
        validation.flattened_metric_keys_sha256(metrics)
        != WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256
    ):
        raise WorkloadError("M4 waveguide metric-key digest differs")
    return metrics


def _waveguide_pairwise_gate(
    result: dict[str, Any], *, diagnostic_optimizer: bool
) -> dict[str, Any]:
    compact = _compact_comparison(result)
    evidence = compact.get("metric_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("count") != WAVEGUIDE_EXPECTED_METRIC_COUNT
        or evidence.get("passed_count") != WAVEGUIDE_EXPECTED_METRIC_COUNT
        or evidence.get("metric_keys_sha256")
        != WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256
    ):
        raise WorkloadError("M4 waveguide pairwise metric census differs")
    if diagnostic_optimizer:
        if (
            evidence.get("diagnostic_only_count")
            != WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
            or type(evidence.get("diagnostic_only_out_of_tolerance_count"))
            is not int
        ):
            raise WorkloadError(
                "M4 waveguide optimizer diagnostic census differs"
            )
    elif (
        "diagnostic_only_count" in evidence
        or "diagnostic_only_out_of_tolerance_count" in evidence
    ):
        raise WorkloadError("M4 waveguide CUDA comparison was weakened")
    _waveguide_diagnostic_summary_gate(
        compact, diagnostic_optimizer=diagnostic_optimizer
    )
    return compact


def _waveguide_diagnostic_summary_gate(
    pairwise: dict[str, Any], *, diagnostic_optimizer: bool
) -> None:
    """Bind the aggregate diagnostic census to its per-rule metadata."""
    evidence = pairwise.get("metric_evidence")
    summary = pairwise.get("difference_summary")
    if not isinstance(evidence, dict) or not isinstance(summary, dict):
        raise WorkloadError("M4 waveguide diagnostic summary differs")
    marked: set[str] = set()
    marked_metric_count = 0
    marked_out_of_tolerance_count = 0
    diagnostic_fields = {
        "diagnostic_only", "diagnostic_only_out_of_tolerance_count"
    }
    for rule, leaf in summary.items():
        if not isinstance(rule, str) or not isinstance(leaf, dict):
            raise WorkloadError("M4 waveguide diagnostic summary differs")
        present = diagnostic_fields & set(leaf)
        if not present:
            continue
        if (
            present != diagnostic_fields
            or leaf.get("diagnostic_only") is not True
            or type(leaf.get("diagnostic_only_out_of_tolerance_count"))
            is not int
            or leaf["diagnostic_only_out_of_tolerance_count"] < 0
            or type(leaf.get("metric_count")) is not int
            or leaf["metric_count"] <= 0
            or leaf["diagnostic_only_out_of_tolerance_count"]
            > leaf["metric_count"]
        ):
            raise WorkloadError("M4 waveguide diagnostic leaf differs")
        marked.add(rule)
        marked_metric_count += leaf["metric_count"]
        marked_out_of_tolerance_count += leaf[
            "diagnostic_only_out_of_tolerance_count"
        ]
    if diagnostic_optimizer:
        if (
            marked != set(WAVEGUIDE_OPTIMIZER_CONTINUOUS_RULES)
            or marked_metric_count
            != WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
            or evidence.get("diagnostic_only_count") != marked_metric_count
            or evidence.get("diagnostic_only_out_of_tolerance_count")
            != marked_out_of_tolerance_count
        ):
            raise WorkloadError("M4 waveguide diagnostic rule scope differs")
    elif marked:
        raise WorkloadError("M4 waveguide CUDA diagnostic rules were hidden")


def _waveguide_optimizer_self_oracle(
    text: str, comparison: dict[str, Any]
) -> dict[str, Any]:
    metrics = _waveguide_metrics(text, comparison)
    expected_exact = {
        **{
            f"fixed_work.run_timesteps[{index}]": float(
                WAVEGUIDE_FIXED_SLICE_TIMESTEP
            )
            for index in range(WAVEGUIDE_FIXED_SLICE_RUN_COUNT)
        },
        **{
            f"optimizer_smoke.run_timesteps[{index}]": float(
                WAVEGUIDE_OPTIMIZER_TIMESTEP
            )
            for index in range(WAVEGUIDE_OPTIMIZER_RUN_COUNT)
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
    }
    if any(metrics.get(name) != value for name, value in expected_exact.items()):
        raise WorkloadError("M4 waveguide optimizer structural oracle failed")

    epoch0_design = _indexed_values(metrics, "optimizer_smoke.epoch0_design")
    final_design = _indexed_values(metrics, "optimizer_smoke.final_design")
    if (
        len(epoch0_design) != WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        or len(final_design) != WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
    ):
        raise WorkloadError("M4 waveguide optimizer design size differs")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (*epoch0_design, *final_design)
    ):
        raise WorkloadError("M4 waveguide optimizer design bounds failed")

    positive_metrics = (
        "optimizer_smoke.epoch0_design_step_l2",
        "optimizer_smoke.epoch1_design_step_l2",
        "optimizer_smoke.epoch0_gradient_l2_initial",
        "optimizer_smoke.epoch0_gradient_l2_final",
        "optimizer_smoke.epoch1_gradient_l2_initial",
        "optimizer_smoke.epoch1_gradient_l2_final",
        "optimizer_smoke.final_design_l2",
    )
    if any(
        not math.isfinite(metrics.get(name, math.nan))
        or metrics[name] <= 1.0e-10
        for name in positive_metrics
    ):
        raise WorkloadError("M4 waveguide optimizer progress oracle failed")
    objective_metrics = (
        "optimizer_smoke.epoch0_objective_initial",
        "optimizer_smoke.epoch0_objective_final",
        "optimizer_smoke.epoch1_objective_initial",
        "optimizer_smoke.epoch1_objective_final",
    )
    if any(
        not math.isfinite(metrics.get(name, math.nan))
        for name in objective_metrics
    ):
        raise WorkloadError("M4 waveguide optimizer objective oracle failed")

    computed_epoch0_step = math.sqrt(
        math.fsum((value - 0.5) ** 2 for value in epoch0_design)
    )
    computed_epoch1_step = math.sqrt(
        math.fsum(
            (final - epoch0) ** 2
            for epoch0, final in zip(
                epoch0_design, final_design, strict=True
            )
        )
    )
    reported_epoch0_step = metrics[
        "optimizer_smoke.epoch0_design_step_l2"
    ]
    reported_epoch1_step = metrics[
        "optimizer_smoke.epoch1_design_step_l2"
    ]
    if (
        computed_epoch0_step <= 1.0e-10
        or computed_epoch1_step <= 1.0e-10
        or not math.isclose(
            computed_epoch0_step,
            reported_epoch0_step,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            computed_epoch1_step,
            reported_epoch1_step,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise WorkloadError("M4 waveguide optimizer design-step oracle failed")

    computed_l2 = math.sqrt(math.fsum(value * value for value in final_design))
    reported_l2 = metrics["optimizer_smoke.final_design_l2"]
    if not math.isclose(
        computed_l2, reported_l2, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise WorkloadError("M4 waveguide optimizer L2 oracle failed")
    return {
        "outcome": "PASS",
        "design_size": len(final_design),
        "design_minimum": min(final_design),
        "design_maximum": max(final_design),
        "computed_epoch0_design_step_l2": computed_epoch0_step,
        "reported_epoch0_design_step_l2": reported_epoch0_step,
        "computed_epoch1_design_step_l2": computed_epoch1_step,
        "reported_epoch1_design_step_l2": reported_epoch1_step,
        "computed_final_design_l2": computed_l2,
        "reported_final_design_l2": reported_l2,
        "structural_metrics": expected_exact,
    }


def _absorber_scale_aware_contract(case: dict[str, Any]) -> dict[str, Any]:
    comparison, _method = _comparison_contract(case)
    copied = dict(comparison)
    tolerances = {
        name: dict(tolerance)
        for name, tolerance in comparison["tolerances"].items()
    }
    original = tolerances.pop("example_results.absorber_trace_fields.*", None)
    if original is None:
        raise WorkloadError("M4 absorber trace tolerance is absent")
    for name in (
        "example_results.absorber_trace_fields.count",
        "example_results.absorber_trace_fields.l2",
        "example_results.absorber_trace_fields.max_abs",
        "example_results.absorber_trace_fields.shape[[]*",
        "example_results.absorber_trace_fields.weighted_imag",
    ):
        tolerances[name] = dict(original)
    for name in (
        "example_results.absorber_trace_fields.real[[]*",
        "example_results.absorber_trace_fields.imag[[]*",
    ):
        tolerances[name] = {**original, "diagnostic_only": True}
    copied["tolerances"] = tolerances
    return copied


def _indexed_values(metrics: dict[str, float], prefix: str) -> list[float]:
    values = []
    index = 0
    while True:
        key = f"{prefix}[{index}]"
        if key not in metrics:
            break
        values.append(metrics[key])
        index += 1
    if not values:
        raise WorkloadError(f"M4 indexed metric {prefix!r} is absent")
    matching = {key for key in metrics if key.startswith(prefix + "[")}
    expected = {f"{prefix}[{item}]" for item in range(len(values))}
    if matching != expected:
        raise WorkloadError(f"M4 indexed metric {prefix!r} is not contiguous")
    return values


def _absorber_scale_gate(
    reference_text: str, candidate_text: str, comparison: dict[str, Any]
) -> dict[str, Any]:
    reference = validation.extract_json_metrics(reference_text, comparison)
    candidate = validation.extract_json_metrics(candidate_text, comparison)
    components = {}
    retained: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for component in ("real", "imag"):
        prefix = f"example_results.absorber_trace_fields.{component}"
        expected = _indexed_values(reference, prefix)
        observed = _indexed_values(candidate, prefix)
        if len(expected) != len(observed):
            raise WorkloadError("M4 absorber trace component length differs")
        errors = [abs(first - second) for first, second in zip(expected, observed, strict=True)]
        maximum_error = max(errors, default=0.0)
        retained[component] = (expected, observed, errors)
        components[component] = {
            "count": len(errors),
            "maximum_absolute_error": maximum_error,
        }
    all_reference = retained["real"][0] + retained["imag"][0]
    all_candidate = retained["real"][1] + retained["imag"][1]
    all_errors = retained["real"][2] + retained["imag"][2]
    peak = max(
        (abs(value) for value in (*all_reference, *all_candidate)), default=0.0
    )
    maximum_error = max(all_errors, default=0.0)
    rms_error = math.sqrt(
        math.fsum(value * value for value in all_errors) / len(all_errors)
    )
    peak_normalized = maximum_error / max(peak, 1.0e-30)
    rms_peak_normalized = rms_error / max(peak, 1.0e-30)
    if (
        peak_normalized > ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR
        or rms_peak_normalized > ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR
    ):
        raise WorkloadError("M4 absorber scale-aware trace comparison failed")
    return {
        "outcome": "PASS",
        "maximum_peak_normalized_error": ABSORBER_TRACE_MAX_PEAK_NORMALIZED_ERROR,
        "maximum_rms_peak_normalized_error": (
            ABSORBER_TRACE_RMS_PEAK_NORMALIZED_ERROR
        ),
        "observed_peak": peak,
        "observed_maximum_absolute_error": maximum_error,
        "observed_peak_normalized_maximum_error": peak_normalized,
        "observed_rms_peak_normalized_error": rms_peak_normalized,
        "components": components,
    }


def _adjoint_cyl_contract(*, upstream_forward_only: bool) -> dict[str, Any]:
    tolerances: dict[str, dict[str, Any]] = {
        "schema_version": {"rtol": 0.0, "atol": 0.0},
        "fixed_work": {"rtol": 0.0, "atol": 0.0},
        "case_count": {"rtol": 0.0, "atol": 0.0},
        "cases*.case_index": {"rtol": 0.0, "atol": 0.0},
        "cases*.m": {"rtol": 0.0, "atol": 0.0},
        "cases*.far_point*": {"rtol": 0.0, "atol": 0.0},
        "cases*.adjoint_objective": {"rtol": 1.0e-5, "atol": 1.0e-8},
        "cases*.forward_objective": {"rtol": 1.0e-5, "atol": 1.0e-8},
        "cases*.perturbed_forward_objective": {"rtol": 1.0e-5, "atol": 1.0e-8},
        "cases*.adjoint_directional_derivative": {"rtol": 0.02, "atol": 2.0e-8},
        "cases*.finite_difference_derivative": {"rtol": 0.08, "atol": 1.0e-9},
        "cases*.relative_derivative_error": {"rtol": 0.10, "atol": 0.02},
    }
    if upstream_forward_only:
        for name in (
            "cases*.adjoint_directional_derivative",
            "cases*.relative_derivative_error",
        ):
            tolerances[name]["diagnostic_only"] = True
    else:
        # Relative errors are independently bounded below.  Equality of two
        # cancellation-sensitive error quotients is not a physical oracle.
        tolerances["cases*.relative_derivative_error"]["diagnostic_only"] = True
    return {
        "mode": "json_metrics",
        "prefix": ADJOINT_CYL_METRIC_PREFIX,
        "tolerances": tolerances,
    }


def _adjoint_cyl_self_oracle(text: str, comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = validation.extract_json_metrics(text, comparison)
    if (
        metrics.get("schema_version") != 1.0
        or metrics.get("fixed_work") != 1.0
        or metrics.get("case_count") != float(len(ADJOINT_CYL_EXPECTED_CASES))
    ):
        raise WorkloadError("M4 cylindrical adjoint fixed-work identity differs")
    bounds = []
    for index, (expected_m, expected_far_point) in enumerate(
        ADJOINT_CYL_EXPECTED_CASES
    ):
        if metrics.get(f"cases[{index}].case_index") != float(index):
            raise WorkloadError("M4 cylindrical adjoint case index differs")
        m = metrics[f"cases[{index}].m"]
        far_point = tuple(
            metrics[f"cases[{index}].far_point[{coordinate}]"]
            for coordinate in range(3)
        )
        if m != expected_m or far_point != expected_far_point:
            raise WorkloadError("M4 cylindrical adjoint case identity differs")
        adjoint_objective = metrics[f"cases[{index}].adjoint_objective"]
        forward_objective = metrics[f"cases[{index}].forward_objective"]
        objective_relative_error = abs(adjoint_objective - forward_objective) / max(
            abs(adjoint_objective), abs(forward_objective), 1.0e-30
        )
        if objective_relative_error > 1.0e-3:
            raise WorkloadError("M4 cylindrical adjoint objective oracle failed")
        adjoint_derivative = metrics[
            f"cases[{index}].adjoint_directional_derivative"
        ]
        finite_difference_derivative = metrics[
            f"cases[{index}].finite_difference_derivative"
        ]
        reported_error = metrics[f"cases[{index}].relative_derivative_error"]
        error = abs(adjoint_derivative - finite_difference_derivative) / max(
            abs(adjoint_derivative), abs(finite_difference_derivative), 1.0e-30
        )
        if (
            not math.isfinite(reported_error)
            or reported_error < 0.0
            or not math.isclose(
                reported_error, error, rel_tol=1.0e-14, abs_tol=1.0e-15
            )
        ):
            raise WorkloadError(
                "M4 cylindrical adjoint reported derivative error is inconsistent"
            )
        limit = 0.6 if m == 0.0 else 0.4
        if error > limit:
            raise WorkloadError("M4 cylindrical adjoint finite-difference oracle failed")
        bounds.append(
            {
                "case_index": index,
                "objective_relative_error": objective_relative_error,
                "objective_limit": 1.0e-3,
                "relative_derivative_error": error,
                "reported_relative_derivative_error": reported_error,
                "derivative_limit": limit,
            }
        )
    return {"outcome": "PASS", "bounds": bounds}


def _adjoint_cyl_fixed_work_gate(text: str) -> dict[str, Any]:
    comparison = {
        "mode": "json_metrics",
        "prefix": GENERIC_METRIC_PREFIX,
        "tolerances": {"*": {"rtol": 0.0, "atol": 0.0, "diagnostic_only": True}},
    }
    metrics = validation.extract_json_metrics(text, comparison)
    raw_timesteps = _indexed_values(metrics, "per_run.timestep")
    raw_deltas = _indexed_values(metrics, "per_run.timestep_delta")
    if any(value != int(value) for value in (*raw_timesteps, *raw_deltas)):
        raise WorkloadError("M4 cylindrical adjoint timestep is not integral")
    timesteps = tuple(int(value) for value in raw_timesteps)
    deltas = tuple(int(value) for value in raw_deltas)
    meep_times = tuple(_indexed_values(metrics, "per_run.meep_time"))
    if (
        metrics.get("run_count") != 16.0
        or timesteps != ADJOINT_CYL_EXPECTED_TIMESTEPS
        or deltas != ADJOINT_CYL_EXPECTED_TIMESTEPS
        or len(meep_times) != len(ADJOINT_CYL_EXPECTED_MEEP_TIMES)
        or any(
            abs(observed - expected) > 0.05
            for observed, expected in zip(
                meep_times, ADJOINT_CYL_EXPECTED_MEEP_TIMES, strict=True
            )
        )
    ):
        raise WorkloadError("M4 cylindrical adjoint workload is not fixed-work")
    return {
        "outcome": "PASS",
        "run_count": len(timesteps),
        "timesteps": list(timesteps),
        "timestep_deltas": list(deltas),
        "meep_times": list(meep_times),
    }


def _compact_comparison(result: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "outcome", "reason", "metric_evidence", "difference_summary",
        "failure_count", "failures_reported", "failures_truncated",
        "failure_records_sha256", "failures", "diagnostic_only_count",
        "diagnostic_only_out_of_tolerance_count",
    }
    compact = {name: result[name] for name in keys if name in result}
    if compact.get("outcome") != "PASS" or compact.get("failure_count") != 0:
        raise WorkloadError("M4 same-cycle numerical comparison failed")
    return compact


def _numerical_comparison(
    case: dict[str, Any], candidate_lane: str,
    reference_text: str, candidate_text: str,
) -> tuple[dict[str, Any], str]:
    path = case["path"]
    if path == ABSORBER_CASE:
        comparison = _absorber_scale_aware_contract(case)
        pairwise = validation.compare_json_metrics(
            reference_text, candidate_text, comparison
        )
        return (
            {
                "outcome": "PASS",
                "failure_count": 0,
                "pairwise": _compact_comparison(pairwise),
                "scale_aware_trace": _absorber_scale_gate(
                    reference_text, candidate_text, comparison
                ),
            },
            "manifest-plus-scale-aware-trace",
        )
    if path == WAVEGUIDE_CROSSING_CASE:
        diagnostic_optimizer = candidate_lane == "gpmeep_v110_cpu"
        if candidate_lane == "gpmeep_v110_cpu":
            comparison = _waveguide_upstream_optimizer_contract(case)
            method = (
                "upstream-fixed-slices-plus-diagnostic-optimizer-trajectory-"
                "and-self-oracles"
            )
        else:
            comparison, _unused = _comparison_contract(case)
            method = "gpmeep-cpu-cross-lane-plus-optimizer-self-oracles"
        result = validation.compare_json_metrics(
            reference_text, candidate_text, comparison
        )
        return (
            {
                "outcome": "PASS",
                "failure_count": 0,
                "pairwise": _waveguide_pairwise_gate(
                    result, diagnostic_optimizer=diagnostic_optimizer
                ),
                "reference_self_oracle": _waveguide_optimizer_self_oracle(
                    reference_text, comparison
                ),
                "candidate_self_oracle": _waveguide_optimizer_self_oracle(
                    candidate_text, comparison
                ),
            },
            method,
        )
    if path == NEAR2FAR_3D_CASE:
        if candidate_lane == "gpmeep_v110_cpu":
            comparison = _near2far_upstream_forward_contract(case)
            method = "upstream-forward-fd-plus-candidate-self-oracle"
        else:
            comparison, _unused = _comparison_contract(case)
            method = "gpmeep-cpu-cross-lane-plus-candidate-self-oracle"
        result = validation.compare_json_metrics(
            reference_text, candidate_text, comparison
        )
        return _compact_comparison(result), method
    if path == ADJOINT_CYL_CASE:
        upstream_forward_only = candidate_lane == "gpmeep_v110_cpu"
        comparison = _adjoint_cyl_contract(
            upstream_forward_only=upstream_forward_only
        )
        result = validation.compare_json_metrics(
            reference_text, candidate_text, comparison
        )
        return (
            {
                "outcome": "PASS",
                "failure_count": 0,
                "pairwise": _compact_comparison(result),
                "reference_self_oracle": _adjoint_cyl_self_oracle(
                    reference_text, comparison
                ),
                "candidate_self_oracle": _adjoint_cyl_self_oracle(
                    candidate_text, comparison
                ),
                "reference_fixed_work": _adjoint_cyl_fixed_work_gate(
                    reference_text
                ),
                "candidate_fixed_work": _adjoint_cyl_fixed_work_gate(
                    candidate_text
                ),
            },
            (
                "upstream-forward-fd-plus-fixed-work-self-oracle"
                if upstream_forward_only
                else "gpmeep-cpu-cross-lane-plus-fixed-work-self-oracle"
            ),
        )
    comparison, method = _comparison_contract(case)
    result = validation.compare_json_metrics(
        reference_text, candidate_text, comparison
    )
    return _compact_comparison(result), method


def derive_numerical_results(
    output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    cases = validation.materialize_cases(
        validation.load_json(pathlib.Path(contract["paths"]["manifest"])),
        pathlib.Path(contract["paths"]["repo"]),
    )
    by_case = {case["path"]: case for case in cases}
    tasks = contract["tasks"]
    by_key = {
        (task["cycle_index"], task["case_path"], task["lane_id"]): task
        for task in tasks
    }
    if len(by_key) != TASK_COUNT:
        raise WorkloadError("M4 numerical task inventory differs")
    rows = []
    for task in tasks:
        if task["lane_id"] == "upstream_meep_cpu":
            continue
        reference_lane = (
            "gpmeep_v110_cpu"
            if (
                task["case_path"] in GPMEEP_CPU_REFERENCE_CASES
                and task["lane_id"] != "gpmeep_v110_cpu"
            )
            else "upstream_meep_cpu"
        )
        reference = by_key.get(
            (task["cycle_index"], task["case_path"], reference_lane)
        )
        if reference is None:
            raise WorkloadError("M4 numerical CPU reference is absent")
        try:
            case = by_case[task["case_path"]]
        except KeyError as exc:
            raise WorkloadError("M4 numerical case is absent") from exc
        result, method = _numerical_comparison(
            case,
            task["lane_id"],
            _read_stdout(output, reference),
            _read_stdout(output, task),
        )
        rows.append(
            {
                "cycle_index": task["cycle_index"],
                "sample_kind": task["sample_kind"],
                "case_path": task["case_path"],
                "reference_lane": reference_lane,
                "candidate_lane": task["lane_id"],
                "method": method,
                "result": result,
            }
        )
    if len(rows) != 141:
        raise WorkloadError("M4 numerical comparison census differs")
    return {
        "schema": "gpmeep-v110-m4-numerical-replay-v1",
        "outcome": "PASS",
        "comparison_count": len(rows),
        "comparisons": rows,
    }


def derive_aggregate(
    output: pathlib.Path, contract: dict[str, Any], completed: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_session = v110_sample.RuntimeVerificationSession()
    try:
        reports = _derive_verified_sample_reports(
            output,
            contract,
            completed,
            v110_runtime_session=runtime_session,
        )
        result = _derive_aggregate_from_reports(output, contract, reports)
        runtime_session.finalize()
        return result
    except BaseException as error:
        runtime_error = runtime_session.abort()
        if runtime_error is not None:
            raise WorkloadError(
                "M4 aggregate derivation failed and runtime finalization "
                f"also failed: primary={type(error).__name__}: {error}; "
                f"runtime={runtime_error}"
            ) from error
        raise


def _derive_verified_sample_reports(
    output: pathlib.Path,
    contract: dict[str, Any],
    completed: list[dict[str, Any]],
    *,
    v110_runtime_session: v110_sample.RuntimeVerificationSession | None = None,
) -> list[dict[str, Any]]:
    if len(completed) != TASK_COUNT:
        raise WorkloadError("M4 completed-task census differs")
    reports = []
    for task, retained in zip(contract["tasks"], completed, strict=True):
        sample_replay, report = derive_sample_replay(
            task,
            output,
            contract,
            v110_runtime_session=v110_runtime_session,
        )
        if retained.get("sample_replay") != sample_replay:
            raise WorkloadError("M4 retained child replay differs")
        reports.append(report)
    return reports


def _derive_aggregate_from_reports(
    output: pathlib.Path,
    contract: dict[str, Any],
    reports: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(reports) != TASK_COUNT or len(contract["tasks"]) != TASK_COUNT:
        raise WorkloadError("M4 verified-report census differs")
    for task, report in zip(contract["tasks"], reports, strict=True):
        equal_work_replay.validate_sample_report(task, report)
    return (
        equal_work_replay.derive_results(contract["plan"], reports),
        derive_numerical_results(output, contract),
    )


def derive_inversion_records(
    output: pathlib.Path, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    completed = _load_checkpoint(output, contract)["completed_tasks"]
    runtime_session = v110_sample.RuntimeVerificationSession()
    try:
        reports = _derive_verified_sample_reports(
            output,
            contract,
            completed,
            v110_runtime_session=runtime_session,
        )
        result = _derive_inversion_records_from_reports(
            output, contract, completed, reports
        )
        runtime_session.finalize()
        return result
    except BaseException as error:
        runtime_error = runtime_session.abort()
        if runtime_error is not None:
            raise WorkloadError(
                "M4 inversion derivation failed and runtime finalization "
                f"also failed: primary={type(error).__name__}: {error}; "
                f"runtime={runtime_error}"
            ) from error
        raise


def _derive_inversion_records_from_reports(
    output: pathlib.Path,
    contract: dict[str, Any],
    completed: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        len(completed) != TASK_COUNT
        or len(reports) != TASK_COUNT
        or len(contract["tasks"]) != TASK_COUNT
    ):
        raise WorkloadError("M4 inversion verified-report census differs")
    evidence: dict[str, dict[str, Any]] = {}
    for task, retained, report in zip(
        contract["tasks"], completed, reports, strict=True
    ):
        equal_work_replay.validate_sample_report(task, report)
        replay = retained.get("sample_replay")
        record = replay.get("report") if isinstance(replay, dict) else None
        if not isinstance(record, dict):
            raise WorkloadError("M4 inversion child report evidence differs")
        evidence[task["task_id"]] = {
            "path": record["path"],
            "sha256": record["sha256"],
        }
    inventory_path = pathlib.Path(contract["paths"]["repo"]) / "packaging/v110-performance-inversion-inventory.json"
    inventory = validation.load_json(inventory_path)
    inventory_record = matrix.stable_file_record(
        inventory_path, pathlib.Path(contract["paths"]["repo"]), "M4 inversion inventory"
    )
    numerical_record = matrix.stable_file_record(
        output / "numerical-replay.json", output, "M4 numerical replay"
    )
    return equal_work_replay.derive_inversion_records(
        contract["plan"],
        reports,
        evidence,
        inventory,
        inventory_record["path"],
        inventory_record["sha256"],
        {"path": numerical_record["path"], "sha256": numerical_record["sha256"]},
    )


def publish_inversion_records(
    output: pathlib.Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = output / "inversion-records"
    root.mkdir(exist_ok=False)
    retained = []
    for record in records:
        path = root / f"{record['record_id']}.json"
        atomic_write_json(path, record)
        retained.append(matrix.stable_file_record(path, output, "M4 inversion record"))
    return retained


def validate_inversion_records(
    output: pathlib.Path,
    retained: Any,
    expected: list[dict[str, Any]],
    repo: pathlib.Path,
) -> None:
    if not isinstance(retained, list) or len(retained) != len(expected):
        raise WorkloadError("M4 inversion-record file inventory differs")
    for file_record_value, record in zip(retained, expected, strict=True):
        expected_path = output / "inversion-records" / f"{record['record_id']}.json"
        if file_record_value != matrix.stable_file_record(expected_path, output, "M4 inversion record"):
            raise WorkloadError("M4 inversion-record file changed")
        saved = matrix.load_stable_json_record(output, file_record_value, "M4 inversion record")
        if saved != record:
            raise WorkloadError("M4 inversion record was not re-derived")
    validate_normative_inversion_registry(output, repo)


def _load_normative_inversion_registry(repo: pathlib.Path) -> Any:
    validator_path = repo / "scripts/validate-v110-inversion-registry.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_m4_normative_inversion_registry", validator_path
    )
    if spec is None or spec.loader is None:
        raise WorkloadError("M4 normative inversion registry validator is unavailable")
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    return validator


def validate_normative_inversion_registry(
    output: pathlib.Path, repo: pathlib.Path
) -> dict[str, Any]:
    validator = _load_normative_inversion_registry(repo)
    try:
        result = validator.validate_registry(
            require_closed=False,
            evidence_root=output,
            records_directory=output / "inversion-records",
        )
    except Exception as exc:
        raise WorkloadError(
            f"M4 normative inversion registry replay failed: {exc}"
        ) from exc
    expected_result = {
        "schema_version": 1,
        "observed_count": 13,
        "registered_count": 13,
        "missing_count": 0,
        "missing_workloads": [],
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            raise WorkloadError(
                f"M4 normative inversion registry result differs: {key}"
            )
    blockers = result.get("blocking_workloads")
    open_workloads = result.get("open_workloads")
    if (
        result.get("state") not in {"PASS", "OPEN"}
        or not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not isinstance(open_workloads, list)
        or open_workloads != sorted(set(open_workloads))
        or not set(blockers).issubset(open_workloads)
        or result.get("blocking_count") != len(blockers)
        or result.get("open_count") != len(open_workloads)
        or result.get("closed_count") + result.get("open_count") != 13
        or (result.get("state") == "PASS") != (not open_workloads)
    ):
        raise WorkloadError("M4 normative inversion closure computation differs")
    return result


def write_checkpoint(
    output: pathlib.Path, contract: dict[str, Any], hardware_before: dict[str, Any],
    capacity_before: dict[str, Any], completed: list[dict[str, Any]]
) -> None:
    atomic_write_json(
        output / "CHECKPOINT.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract": contract,
            "hardware_before": hardware_before,
            "capacity_before": capacity_before,
            "completed_tasks": completed,
        },
    )


def append_event(output: pathlib.Path, journal: dict[str, Any], event: dict[str, Any]) -> None:
    journal["events"].append(event)
    atomic_write_json(output / "JOURNAL.json", journal)


def expected_journal(
    journal: Any, contract: dict[str, Any], completed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if (
        not isinstance(journal, dict)
        or set(journal) != {"schema", "contract", "events"}
        or journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("contract") != contract
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise WorkloadError("M4 journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(float(first.get("unix_seconds", math.nan))):
        raise WorkloadError("M4 journal start event differs")
    expected = [first]
    for record in completed:
        expected.extend(
            [
                {"state": "task-started", "task": record["task"]},
                {"state": "task-process-ended", "task": record["task"], "process": record["process"]},
                {"state": "task-validated", "task": record["task"], "sample_complete": record["sample_replay"]["complete"]},
            ]
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("M4 journal disagrees with checkpoint")
    return expected


def _validate_completed_task_with_report(
    task: dict[str, Any], retained: Any, output: pathlib.Path,
    contract: dict[str, Any], devices: tuple[str, str],
    *,
    v110_runtime_session: v110_sample.RuntimeVerificationSession | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = {
        "task", "process", "observers", "telemetry_replay", "sample_replay",
        "hardware", "capacity_before", "capacity_after",
    }
    if not isinstance(retained, dict) or set(retained) != keys or retained["task"] != task:
        raise WorkloadError("M4 completed-task record/order differs")
    matrix._verify_process_record(
        output, retained["process"], task_log_relative(task).as_posix(), task_name(task)
    )
    if retained["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M4 task command differs")
    validate_telemetry(
        task, retained["process"], retained["observers"], retained["telemetry_replay"],
        output, contract,
    )
    matrix.validate_hardware_evidence(retained["hardware"], list(devices))
    validate_capacity(retained["capacity_before"], False)
    validate_capacity(retained["capacity_after"], False)
    sample_replay, report = derive_sample_replay(
        task,
        output,
        contract,
        v110_runtime_session=v110_runtime_session,
    )
    if retained["sample_replay"] != sample_replay:
        raise WorkloadError("M4 sample replay differs")
    return retained, report


def validate_completed_task(
    task: dict[str, Any], retained: Any, output: pathlib.Path,
    contract: dict[str, Any], devices: tuple[str, str],
    *,
    v110_runtime_session: v110_sample.RuntimeVerificationSession | None = None,
) -> dict[str, Any]:
    validated, _report = _validate_completed_task_with_report(
        task,
        retained,
        output,
        contract,
        devices,
        v110_runtime_session=v110_runtime_session,
    )
    return validated


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        with stable.file_object() as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _load_checkpoint(output: pathlib.Path, contract: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _load_json(output / "CHECKPOINT.json", "M4 checkpoint")
    if (
        set(checkpoint) != {"schema", "contract", "hardware_before", "capacity_before", "completed_tasks"}
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("contract") != contract
        or not isinstance(checkpoint.get("completed_tasks"), list)
    ):
        raise WorkloadError("M4 checkpoint differs")
    return checkpoint


def _terminal_payload(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "journal": file_record(output / "JOURNAL.json", output),
        "checkpoint": report["checkpoint"],
        "performance_replay": report["performance_replay"],
        "numerical_replay": report["numerical_replay"],
        "inversion_records": report["inversion_records"],
    }


def _record_directory_identity(metadata: os.stat_result) -> tuple[Any, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _record_file_identity(metadata: os.stat_result) -> tuple[Any, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_record_from_descriptor_root(
    root: pathlib.Path,
    relative: pathlib.PurePosixPath,
    expected: dict[str, Any],
    label: str,
) -> pathlib.Path:
    """Hash a record while holding every no-follow ancestor descriptor."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_flags = flags | os.O_DIRECTORY
    descriptors: list[int] = []
    links: list[tuple[int, str, int, tuple[Any, ...]]] = []
    try:
        root_descriptor = matrix._open_directory_path_no_symlinks(root)
        descriptors.append(root_descriptor)
        root_identity = _record_directory_identity(os.fstat(root_descriptor))
        current = root_descriptor
        for part in relative.parts[:-1]:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise WorkloadError(f"{label} traverses a symlink or non-directory")
            child = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(child)
            identity = _record_directory_identity(os.fstat(child))
            if identity != _record_directory_identity(before):
                raise WorkloadError(f"{label} ancestor changed while it was opened")
            links.append((current, part, child, identity))
            current = child

        name = relative.parts[-1]
        before = os.stat(name, dir_fd=current, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise WorkloadError(f"{label} is not a regular file")
        file_descriptor = os.open(name, flags, dir_fd=current)
        descriptors.append(file_descriptor)
        opened = os.fstat(file_descriptor)
        if _record_file_identity(opened) != _record_file_identity(before):
            raise WorkloadError(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        while True:
            block = os.read(file_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if (
            _record_file_identity(os.fstat(file_descriptor))
            != _record_file_identity(opened)
            or opened.st_size != expected["size_bytes"]
            or digest.hexdigest() != expected["sha256"]
        ):
            raise WorkloadError(f"{label} file identity differs")
        named_after = os.stat(name, dir_fd=current, follow_symlinks=False)
        if _record_file_identity(named_after) != _record_file_identity(opened):
            raise WorkloadError(f"{label} name changed while it was read")
        for parent, part, child, identity in reversed(links):
            if (
                _record_directory_identity(os.fstat(child)) != identity
                or _record_directory_identity(
                    os.stat(part, dir_fd=parent, follow_symlinks=False)
                )
                != identity
            ):
                raise WorkloadError(f"{label} ancestor changed while it was read")
        named_root = matrix._open_directory_path_no_symlinks(root)
        try:
            if _record_directory_identity(os.fstat(named_root)) != root_identity:
                raise WorkloadError(f"{label} evidence root changed while it was read")
        finally:
            os.close(named_root)
        return root.joinpath(*relative.parts).absolute()
    except OSError as exc:
        raise WorkloadError(f"{label} cannot be read safely: {exc}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _validate_embedded_file_records(
    value: Any,
    evidence_root: pathlib.Path,
    cache: dict[pathlib.Path, tuple[int, str]],
    label: str,
    relative_prefix: pathlib.PurePosixPath = pathlib.PurePosixPath(),
) -> None:
    """Validate records from one structural root without following ancestors."""

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_embedded_file_records(
                item, evidence_root, cache, f"{label}[{index}]", relative_prefix
            )
        return
    if not isinstance(value, dict):
        return
    if set(value) == {"path", "size_bytes", "sha256"}:
        raw = value.get("path")
        if not isinstance(raw, str) or not raw:
            raise WorkloadError(f"{label} file-record path differs")
        candidate = pathlib.Path(raw)
        if candidate.is_absolute():
            absolute = pathlib.PurePosixPath(raw)
            base = pathlib.Path("/")
            relative = pathlib.PurePosixPath(*absolute.parts[1:])
        else:
            relative = pathlib.PurePosixPath(raw)
            if (
                relative.is_absolute()
                or not relative.parts
                or "." in relative.parts
                or ".." in relative.parts
                or "\\" in raw
            ):
                raise WorkloadError(f"{label} file-record path is unsafe")
            base = evidence_root
            relative = relative_prefix / relative
        if (
            not relative.parts
            or "." in relative.parts
            or ".." in relative.parts
            or "\\" in raw
        ):
            raise WorkloadError(f"{label} file-record path is unsafe")
        canonical_record = {
            "path": relative.as_posix(),
            "size_bytes": value.get("size_bytes"),
            "sha256": value.get("sha256"),
        }
        expected = (value.get("size_bytes"), value.get("sha256"))
        if (
            type(expected[0]) is not int
            or expected[0] < 0
            or not isinstance(expected[1], str)
            or len(expected[1]) != 64
        ):
            raise WorkloadError(f"{label} file-record identity differs")
        candidate = _validate_record_from_descriptor_root(
            base, relative, canonical_record, label
        )
        retained = cache.get(candidate)
        if retained is not None:
            if retained != expected:
                raise WorkloadError(f"{label} repeats a file with another identity")
            return
        cache[candidate] = expected
        return
    for name, item in value.items():
        _validate_embedded_file_records(
            item, evidence_root, cache, f"{label}.{name}", relative_prefix
        )


def _bounded_child_context(
    repo: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    """Derive immutable child-runtime inputs once for the bounded replay."""

    paths = {name: pathlib.Path(value) for name, value in contract["paths"].items()}
    cases = validation.materialize_cases(
        validation.load_json(paths["manifest"]), repo
    )
    by_case = {case["path"]: case for case in cases}
    if not {task["case_path"] for task in contract["tasks"]} <= set(by_case):
        raise WorkloadError("M4 authority child case inventory differs")

    upstream_receipt, upstream_verified = upstream_sample.verify_receipt(
        repo, paths["upstream_receipt"]
    )
    upstream_namespace = argparse.Namespace(
        repo=repo,
        manifest=paths["manifest"],
        upstream_receipt=paths["upstream_receipt"],
    )
    upstream_paths = upstream_sample.runtime_paths(
        upstream_namespace, upstream_receipt
    )

    v103_identity = contract["runtime_inputs"]["v103_installed_identity"]
    v103_namespace = argparse.Namespace(
        repo=repo,
        manifest=paths["manifest"],
        install_prefix=paths["v103_install_prefix"],
        provenance_report=paths["v103_provenance_report"],
        package=paths["v103_package"],
    )
    v103_report, v103_paths, v103_contract = v103_sample.runtime_paths(
        v103_namespace
    )
    if (
        {name: str(path) for name, path in v103_paths.items()}
        != v103_identity["paths"]
        or v103_contract != v103_identity["runtime_contract"]
    ):
        raise WorkloadError("M4 authority v1.0.3 child runtime differs")

    v110_runtime_paths = contract["runtime_inputs"]["v110_installed_runtime"]
    v110_provenance, v110_installed = v110_identity.validate_report(
        paths["v110_provenance_report"],
        paths["v110_package"],
        pathlib.Path(v110_runtime_paths["install_prefix"]),
        contract["runtime_inputs"]["v110_source_identity"]["git_head"],
    )
    v110_paths = v110_identity.runtime_paths(v110_provenance, v110_installed)
    v110_contract = v110_identity.runtime_contract(
        v110_provenance, v110_installed
    )
    if {
        name: str(path) for name, path in v110_paths.items()
    } != v110_runtime_paths:
        raise WorkloadError("M4 authority v1.1.0 child runtime differs")
    source_snapshot = validation.source_snapshot(repo)
    if not source_snapshot.get("available"):
        raise WorkloadError("M4 authority source snapshot is unavailable")
    v110_child_paths = {
        "repo": repo,
        "manifest": paths["manifest"],
        "launcher": repo / "scripts/python-validation/m3_mpi_rank_launcher.py",
        **v110_paths,
    }
    return {
        "cases": by_case,
        "source_snapshot": source_snapshot,
        "upstream_receipt": upstream_receipt,
        "upstream_receipt_id": upstream_verified["receipt_id"],
        "upstream_paths": upstream_paths,
        "upstream_receipt_record": absolute_file_record(
            upstream_paths["upstream_receipt"], "upstream receipt"
        ),
        "upstream_code": upstream_sample._code_records(upstream_paths),
        "v103_identity": v103_identity,
        "v103_report": v103_report,
        "v103_paths": v103_paths,
        "v103_contract": v103_contract,
        "v103_provenance": {
            "report": absolute_file_record(
                v103_paths["provenance_report"], "provenance"
            ),
            "package": absolute_file_record(v103_paths["package"], "package"),
            "live_verification": v103_identity["live_verification"],
        },
        "v103_code": {
            case_path: v103_sample._code_records(v103_paths, case_path)
            for case_path in {task["case_path"] for task in contract["tasks"]}
        },
        "v110_provenance": v110_provenance,
        "v110_installed": v110_installed,
        "v110_paths": v110_paths,
        "v110_contract": v110_contract,
        "v110_child_paths": v110_child_paths,
        "v110_provenance_record": {
            name: value
            for name, value in absolute_file_record(
                v110_paths["provenance_report"], "v1.1.0 provenance report"
            ).items()
            if name != "size_bytes"
        },
        "v110_code": v110_sample._code_records(v110_child_paths),
    }


def _validate_nonce(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WorkloadError(f"{label} nonce differs")
    return value


def _validate_child_lane_semantics(
    task: dict[str, Any],
    report: dict[str, Any],
    root: pathlib.Path,
    contract: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Replay child control/process evidence while omitting numerical oracles."""

    runtime = task["runtime"]
    retained = report["lane"]
    module = {"upstream": upstream_sample, "v103": v103_sample, "v110": v110_sample}[runtime]
    if not isinstance(retained, dict) or set(retained) != module.LANE_KEYS:
        raise WorkloadError("M4 authority child lane schema differs")
    case = context["cases"][task["case_path"]]
    lane_root = root / "lane"
    descriptor = matrix._open_directory_path_no_symlinks(lane_root)
    os.close(descriptor)
    nonce = _validate_nonce(retained["run_nonce"], "M4 authority child")

    if runtime == "upstream":
        paths = context["upstream_paths"]
        expected_paths = {name: str(path) for name, path in paths.items()}
        if (
            report["paths"] != expected_paths
            or report["receipt_id"] != context["upstream_receipt_id"]
            or report["receipt"] != context["upstream_receipt_record"]
            or report["evidence_code"] != context["upstream_code"]
        ):
            raise WorkloadError("M4 authority upstream child identity differs")
        topology, lane = upstream_sample.topology_and_lane(case["path"])
        if report["topology"] != topology or retained["spec"] != v110_sample.mpi_case.lane_spec(lane):
            raise WorkloadError("M4 authority upstream child lane differs")
        case_command = upstream_sample.upstream_case_command(
            case, paths["python"], paths["repo"]
        )
        expected_command = v110_sample.mpi_case.lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=lane_root,
            case_command=case_command,
        )
        expected_environment = upstream_sample.clean_environment(
            paths, nonce, output=root, case_path=case["path"]
        )
        rank_records = upstream_sample.validate_rank_bundle(
            lane, lane_root, case, case_command, nonce
        )
        runs = v110_sample.auto_case.audit_runs(
            lane_root / "stdout.log", "cpu", upstream_cpu_baseline=True
        )
    elif runtime == "v103":
        paths = context["v103_paths"]
        identity = context["v103_identity"]
        if (
            report["paths"] != {name: str(path) for name, path in paths.items()}
            or report["receipt_id"] != identity["report_id"]
            or report["runtime_contract"] != context["v103_contract"]
            or report["source_snapshot"] != context["source_snapshot"]
            or report["provenance"] != context["v103_provenance"]
            or report["evidence_code"] != context["v103_code"][case["path"]]
        ):
            raise WorkloadError("M4 authority v1.0.3 child identity differs")
        case_plan = v103_sample._case_plan(contract["plan"], case["path"])
        lane = v103_sample.select_lane(
            report["topology"], tuple(contract["gpu_devices"]), case_plan
        )
        if (
            retained["spec"] != v110_sample.mpi_case.lane_spec(lane)
            or retained["requested_backend"] != "cuda"
        ):
            raise WorkloadError("M4 authority v1.0.3 child lane differs")
        case_command = v103_sample.legacy_case_command(
            case, paths["python"], paths["repo"]
        )
        expected_command = v110_sample.mpi_case.lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=lane_root,
            case_command=case_command,
        )
        expected_environment = v103_sample.clean_environment(
            lane,
            paths,
            context["v103_contract"],
            context["v103_report"],
            nonce,
            output=root,
            case_path=case["path"],
        )
        rank_records = v110_sample.mpi_case.validate_rank_bundle(
            lane,
            lane_root,
            case,
            context["v103_contract"],
            nonce,
            case_command,
        )
        runs = v110_sample.auto_case.audit_runs(
            lane_root / "stdout.log", "cuda", frozen_v103_baseline=True
        )
        process_identity = v103_sample.parse_process_identity(
            lane_root / "stdout.log",
            v103_sample.expected_process_identity(context["v103_report"], paths),
        )
        policy = {
            "schema": "gpmeep-m4-frozen-v103-policy-v1",
            "classification": "forced-frozen-v103-cuda-performance-sample",
            "active_backends": ["cuda"] * len(runs),
        }
        policy_result = v110_sample.validate_policy(runs, "cuda", lane, policy)
        if (
            retained["policy_result"] != policy_result
            or retained["process_identity"] != process_identity
        ):
            raise WorkloadError("M4 authority v1.0.3 dispatch policy differs")
    else:
        paths = context["v110_child_paths"]
        expected_paths = {name: str(path) for name, path in paths.items()}
        snapshot = report["runtime_snapshot"]
        captured = snapshot.get("captured_at_utc") if isinstance(snapshot, dict) else None
        if not isinstance(captured, str) or not captured:
            raise WorkloadError("M4 authority v1.1.0 snapshot timestamp differs")
        provenance_record = context["v110_provenance_record"]
        v110_provenance = context["v110_provenance"]
        expected_snapshot = {
            "captured_at_utc": captured,
            "available": True,
            "source_snapshot": context["source_snapshot"],
            "build_extension": {
                "available": True,
                **v110_provenance["runtime"]["extension"],
            },
            "build_receipt": {
                "available": True,
                **provenance_record,
                "receipt_id": v110_provenance["report_id"],
                "candidate_receipt_id": v110_provenance["v110_candidate_gate"][
                    "receipt_id"
                ],
                "package_sha256": v110_provenance["package_sha256"],
                "live_verification": contract["runtime_inputs"][
                    "v110_live_verification"
                ],
            },
            "runtime_contract": context["v110_contract"],
            "problems": [],
        }
        if (
            report["paths"] != expected_paths
            or report["receipt_id"]
            != contract["runtime_inputs"]["v110_installed_report_id"]
            or snapshot != expected_snapshot
            or report["evidence_code"] != context["v110_code"]
        ):
            raise WorkloadError("M4 authority v1.1.0 child identity differs")
        lane, requested, policy = v110_sample.select_lane(
            report["topology"], tuple(contract["gpu_devices"]), case["path"]
        )
        if (
            retained["spec"] != v110_sample.mpi_case.lane_spec(lane)
            or retained["requested_backend"] != requested
            or requested not in {"cpu", "cuda"}
        ):
            raise WorkloadError("M4 authority v1.1.0 child lane differs")
        case_command = v110_sample.case_command_for_lane(
            case, paths["python"], paths["repo"], requested
        )
        expected_command = v110_sample.mpi_case.lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=lane_root,
            case_command=case_command,
        )
        expected_environment = v110_sample.sample_environment(
            lane,
            requested,
            case_path=case["path"],
            output=root,
            python=paths["python"],
            build_python=paths["build_python"],
            runtime_contract=context["v110_contract"],
            run_nonce=nonce,
        )
        rank_records = v110_sample.mpi_case.validate_rank_bundle(
            lane,
            lane_root,
            case,
            context["v110_contract"],
            nonce,
            case_command,
        )
        runs = v110_sample.auto_case.audit_runs(
            lane_root / "stdout.log", requested
        )
        policy_result = v110_sample.validate_policy(runs, requested, lane, policy)
        if retained["policy_result"] != policy_result:
            raise WorkloadError("M4 authority v1.1.0 dispatch policy differs")

    if retained["command"] != expected_command:
        raise WorkloadError("M4 authority child command differs")
    if retained["environment_sha256"] != validation.environment_sha256(
        expected_environment
    ):
        raise WorkloadError("M4 authority child environment differs")
    v110_sample._validate_process(retained["process"])
    if retained["rank_records"] != rank_records:
        raise WorkloadError("M4 authority child rank records differ")
    if report["timing"] != v110_sample.derive_timing(case["path"], runs):
        raise WorkloadError("M4 authority child timing replay differs")
    if report["phase_profile"] != v110_sample.derive_phase_profile(
        runs, retained["process"]["duration_seconds"]
    ):
        raise WorkloadError("M4 authority child phase replay differs")
    for name in ("stdout", "stderr"):
        if retained[name] != file_record(lane_root / f"{name}.log", lane_root):
            raise WorkloadError(f"M4 authority child {name} differs")
    if (
        retained["process"]["stdout_size_bytes"] != retained["stdout"]["size_bytes"]
        or retained["process"]["stderr_size_bytes"] != retained["stderr"]["size_bytes"]
    ):
        raise WorkloadError("M4 authority child output sizes differ")
    tree_path = root / "tree-manifest.json"
    if (
        retained["tree_manifest"] != file_record(tree_path, root)
        or v110_sample.auto_case._load_object(
            tree_path, "M4 authority child tree manifest"
        )
        != v110_sample.mpi_case.derive_tree(lane_root)
    ):
        raise WorkloadError("M4 authority child tree manifest differs")


def _validate_completed_task_authority(
    task: dict[str, Any],
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
    devices: tuple[str, str],
    cache: dict[pathlib.Path, tuple[int, str]],
    child_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate transitive task bytes without repeating numerical oracles."""

    keys = {
        "task", "process", "observers", "telemetry_replay", "sample_replay",
        "hardware", "capacity_before", "capacity_after",
    }
    if not isinstance(retained, dict) or set(retained) != keys or retained["task"] != task:
        raise WorkloadError("M4 authority completed-task record/order differs")
    matrix._verify_process_record(
        output, retained["process"], task_log_relative(task).as_posix(), task_name(task)
    )
    _validate_embedded_file_records(
        {
            "process": retained["process"],
            "observers": retained["observers"],
            "telemetry_replay": retained["telemetry_replay"],
        },
        output,
        cache,
        "M4 authority outer task evidence",
    )
    if retained["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M4 authority task command differs")
    validate_telemetry(
        task, retained["process"], retained["observers"], retained["telemetry_replay"],
        output, contract,
    )
    matrix.validate_hardware_evidence(retained["hardware"], list(devices))
    validate_capacity(retained["capacity_before"], False)
    validate_capacity(retained["capacity_after"], False)

    sample = retained["sample_replay"]
    if not isinstance(sample, dict) or set(sample) != {
        "complete", "report", "schema", "receipt_id", "timing"
    }:
        raise WorkloadError("M4 authority child replay schema differs")
    root = output.joinpath(*task_sample_relative(task).parts)
    if sample["complete"] != matrix.stable_file_record(
        root / "COMPLETE", output, "M4 authority child COMPLETE"
    ) or sample["report"] != matrix.stable_file_record(
        root / "report.json", output, "M4 authority child report"
    ):
        raise WorkloadError("M4 authority child file identity differs")
    terminal = matrix.load_stable_json_record(
        output, sample["complete"], "M4 authority child COMPLETE"
    )
    report = matrix.load_stable_json_record(
        output, sample["report"], "M4 authority child report"
    )
    module = {
        "upstream": upstream_sample,
        "v103": v103_sample,
        "v110": v110_sample,
    }[task["runtime"]]
    if (
        set(report) != module.REPORT_KEYS
        or report.get("schema") != module.SCHEMA
        or sample["schema"] != report["schema"]
        or sample["receipt_id"] != report.get("receipt_id")
        or terminal != module._terminal(root, report)
    ):
        raise WorkloadError("M4 authority child terminal/report binding differs")
    equal_work_replay.validate_sample_report(task, report, sample["timing"])
    if task["runtime"] == "v110" and report.get("receipt_id") != contract[
        "runtime_inputs"
    ]["v110_installed_report_id"]:
        raise WorkloadError("M4 authority v1.1.0 child identity differs")
    if task["runtime"] == "upstream" and report.get("receipt_id") != contract[
        "runtime_inputs"
    ]["upstream_receipt_id"]:
        raise WorkloadError("M4 authority upstream child identity differs")
    selected = list(task_gpu_devices(task, contract))
    if task["runtime"] in {"v103", "v110"} and report.get("gpu_devices") != list(
        contract["gpu_devices"]
    ):
        raise WorkloadError("M4 authority child GPU inventory differs")
    lane = report["lane"]
    _validate_embedded_file_records(
        {name: value for name, value in report.items() if name != "lane"},
        output,
        cache,
        "M4 authority child report",
        task_sample_relative(task),
    )
    _validate_embedded_file_records(
        {name: value for name, value in lane.items() if name != "tree_manifest"},
        output,
        cache,
        "M4 authority child lane",
        task_sample_relative(task) / "lane",
    )
    _validate_embedded_file_records(
        lane["tree_manifest"],
        output,
        cache,
        "M4 authority child tree manifest",
        task_sample_relative(task),
    )
    if child_context is None:
        child_context = _bounded_child_context(
            pathlib.Path(contract["paths"]["repo"]), contract
        )
    _validate_child_lane_semantics(task, report, root, contract, child_context)
    if task["backend"] == "cuda":
        ranks = lane["rank_records"]
        if (
            lane["requested_backend"] != "cuda"
            or not selected
            or len(ranks) != len(selected)
            or {
                v110_sample.mpi_case._identity(rank["selected_device_identifier"])
                for rank in ranks
            }
            != {v110_sample.mpi_case._identity(device) for device in selected}
        ):
            raise WorkloadError("M4 authority CUDA child used an unselected GPU")
    return report


def _revalidate_cached_reports_authority(
    output: pathlib.Path,
    contract: dict[str, Any],
    completed: list[dict[str, Any]],
    devices: tuple[str, str],
    cached_reports: list[dict[str, Any]],
) -> None:
    """Recheck every transitive child byte before accepting cached results."""

    if (
        len(completed) != TASK_COUNT
        or len(cached_reports) != TASK_COUNT
        or len(contract["tasks"]) != TASK_COUNT
    ):
        raise WorkloadError("M4 cached-report authority census differs")
    repo = pathlib.Path(contract["paths"]["repo"])
    cache: dict[pathlib.Path, tuple[int, str]] = {}
    child_context = _bounded_child_context(repo, contract)
    replayed = [
        _validate_completed_task_authority(
            task,
            retained,
            output,
            contract,
            devices,
            cache,
            child_context,
        )
        for task, retained in zip(
            contract["tasks"], completed, strict=True
        )
    ]
    if replayed != cached_reports:
        raise WorkloadError("M4 cached child reports changed before acceptance")


_COMPACT_COMPARISON_REQUIRED_KEYS = {
    "outcome",
    "reason",
    "metric_evidence",
    "difference_summary",
    "failure_count",
    "failures_reported",
    "failures_truncated",
    "failure_records_sha256",
    "failures",
}
_COMPACT_COMPARISON_OPTIONAL_KEYS = {
    "diagnostic_only_count",
    "diagnostic_only_out_of_tolerance_count",
}
_DIFFERENCE_LEAF_REQUIRED_KEYS = {
    "arithmetic_overflow_count",
    "failure_count",
    "max_absolute_error",
    "max_absolute_error_metric",
    "max_absolute_error_overflow",
    "max_absolute_error_record",
    "max_normalized_error",
    "max_normalized_error_metric",
    "max_normalized_error_overflow",
    "max_normalized_error_record",
    "metric_count",
    "zero_tolerance_failure_count",
}
_DIFFERENCE_LEAF_OPTIONAL_KEYS = {
    "diagnostic_only",
    "diagnostic_only_out_of_tolerance_count",
}


def _validate_failure_free_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkloadError(f"{label} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_failure_free_tree(item, f"{label}[{index}]")
        return
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} contains an unsupported value")
    if "outcome" in value and value["outcome"] != "PASS":
        raise WorkloadError(f"{label} contains a non-PASS outcome")
    if "failure_count" in value and (
        type(value["failure_count"]) is not int or value["failure_count"] != 0
    ):
        raise WorkloadError(f"{label} contains failures")
    if "failures" in value and (
        not isinstance(value["failures"], list) or value["failures"]
    ):
        raise WorkloadError(f"{label} contains failure records")
    if {"count", "passed_count", "failure_count"} <= set(value):
        if (
            type(value["count"]) is not int
            or type(value["passed_count"]) is not int
            or type(value["failure_count"]) is not int
            or value["count"] < 0
            or value["passed_count"] < 0
            or value["count"]
            != value["passed_count"] + value["failure_count"]
        ):
            raise WorkloadError(f"{label} metric census differs")
    for name, item in value.items():
        is_count = name == "count" or name == "run_count" or name.endswith(
            "_count"
        )
        if is_count and isinstance(item, dict):
            if (
                not _DIFFERENCE_LEAF_REQUIRED_KEYS <= set(item)
                or not set(item)
                <= _DIFFERENCE_LEAF_REQUIRED_KEYS
                | _DIFFERENCE_LEAF_OPTIONAL_KEYS
            ):
                raise WorkloadError(f"{label}.{name} difference schema differs")
        elif is_count and (
            type(item) is not int or item < 0
        ):
            raise WorkloadError(f"{label}.{name} count differs")
        if (
            name == "failures_reported"
            and (type(item) is not int or item != 0)
        ) or (
            (name.endswith("failure_count") or name.endswith("overflow_count"))
            and not isinstance(item, dict)
            and (type(item) is not int or item != 0)
        ):
            raise WorkloadError(f"{label}.{name} failure census differs")
        _validate_failure_free_tree(item, f"{label}.{name}")


def _validate_compact_comparison(value: Any, label: str) -> None:
    if (
        not isinstance(value, dict)
        or not _COMPACT_COMPARISON_REQUIRED_KEYS <= set(value)
        or not set(value)
        <= _COMPACT_COMPARISON_REQUIRED_KEYS | _COMPACT_COMPARISON_OPTIONAL_KEYS
        or value.get("outcome") != "PASS"
        or type(value.get("failure_count")) is not int
        or value["failure_count"] != 0
        or type(value.get("failures_reported")) is not int
        or value["failures_reported"] != 0
        or value.get("failures_truncated") is not False
        or value.get("failures") != []
    ):
        raise WorkloadError(f"{label} compact comparison differs")
    diagnostic_count = value.get("diagnostic_only_count", 0)
    diagnostic_out_of_tolerance = value.get(
        "diagnostic_only_out_of_tolerance_count", 0
    )
    if (
        type(diagnostic_count) is not int
        or diagnostic_count < 0
        or type(diagnostic_out_of_tolerance) is not int
        or diagnostic_out_of_tolerance < 0
        or diagnostic_out_of_tolerance > diagnostic_count
    ):
        raise WorkloadError(f"{label} diagnostic count differs")
    _validate_failure_free_tree(value, label)


def _expected_numerical_method(case_path: str, candidate_lane: str) -> str:
    if case_path == ABSORBER_CASE:
        return "manifest-plus-scale-aware-trace"
    if case_path == WAVEGUIDE_CROSSING_CASE:
        return (
            "upstream-fixed-slices-plus-diagnostic-optimizer-trajectory-"
            "and-self-oracles"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-optimizer-self-oracles"
        )
    if case_path == NEAR2FAR_3D_CASE:
        return (
            "upstream-forward-fd-plus-candidate-self-oracle"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-candidate-self-oracle"
        )
    if case_path == ADJOINT_CYL_CASE:
        return (
            "upstream-forward-fd-plus-fixed-work-self-oracle"
            if candidate_lane == "gpmeep_v110_cpu"
            else "gpmeep-cpu-cross-lane-plus-fixed-work-self-oracle"
        )
    return "manifest-json-metrics"


def _validate_waveguide_optimizer_self_oracle(
    value: Any, label: str
) -> None:
    keys = {
        "outcome",
        "design_size",
        "design_minimum",
        "design_maximum",
        "computed_epoch0_design_step_l2",
        "reported_epoch0_design_step_l2",
        "computed_epoch1_design_step_l2",
        "reported_epoch1_design_step_l2",
        "computed_final_design_l2",
        "reported_final_design_l2",
        "structural_metrics",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise WorkloadError(f"{label} waveguide self-oracle schema differs")
    numeric_pairs = (
        (
            "computed_epoch0_design_step_l2",
            "reported_epoch0_design_step_l2",
        ),
        (
            "computed_epoch1_design_step_l2",
            "reported_epoch1_design_step_l2",
        ),
        ("computed_final_design_l2", "reported_final_design_l2"),
    )
    expected_structure = {
        **{
            f"fixed_work.run_timesteps[{index}]": float(
                WAVEGUIDE_FIXED_SLICE_TIMESTEP
            )
            for index in range(WAVEGUIDE_FIXED_SLICE_RUN_COUNT)
        },
        **{
            f"optimizer_smoke.run_timesteps[{index}]": float(
                WAVEGUIDE_OPTIMIZER_TIMESTEP
            )
            for index in range(WAVEGUIDE_OPTIMIZER_RUN_COUNT)
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
    }
    if (
        value.get("outcome") != "PASS"
        or value.get("design_size") != WAVEGUIDE_OPTIMIZER_DESIGN_SIZE
        or type(value.get("design_size")) is not int
        or type(value.get("design_minimum")) not in {int, float}
        or not math.isfinite(float(value["design_minimum"]))
        or type(value.get("design_maximum")) not in {int, float}
        or not math.isfinite(float(value["design_maximum"]))
        or not 0.0
        <= float(value["design_minimum"])
        <= float(value["design_maximum"])
        <= 1.0
        or not all(
            type(value.get(name)) in {int, float}
            and math.isfinite(float(value[name]))
            for pair in numeric_pairs
            for name in pair
        )
        or not all(
            float(value[computed]) > 1.0e-10
            and math.isclose(
                float(value[computed]),
                float(value[reported]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for computed, reported in numeric_pairs
        )
        or value.get("structural_metrics") != expected_structure
    ):
        raise WorkloadError(f"{label} waveguide self-oracle differs")


def _validate_numerical_result(
    case_path: str, candidate_lane: str, value: Any, label: str
) -> None:
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} result is not an object")
    if case_path == ABSORBER_CASE:
        if set(value) != {
            "outcome", "failure_count", "pairwise", "scale_aware_trace"
        }:
            raise WorkloadError(f"{label} absorber result schema differs")
        _validate_compact_comparison(value["pairwise"], f"{label}.pairwise")
    elif case_path == WAVEGUIDE_CROSSING_CASE:
        if set(value) != {
            "outcome",
            "failure_count",
            "pairwise",
            "reference_self_oracle",
            "candidate_self_oracle",
        }:
            raise WorkloadError(f"{label} waveguide result schema differs")
        pairwise = value["pairwise"]
        _validate_compact_comparison(pairwise, f"{label}.pairwise")
        evidence = pairwise.get("metric_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("count") != WAVEGUIDE_EXPECTED_METRIC_COUNT
            or evidence.get("passed_count") != WAVEGUIDE_EXPECTED_METRIC_COUNT
            or evidence.get("metric_keys_sha256")
            != WAVEGUIDE_EXPECTED_METRIC_KEYS_SHA256
        ):
            raise WorkloadError(f"{label} waveguide metric census differs")
        if candidate_lane == "gpmeep_v110_cpu":
            if (
                evidence.get("diagnostic_only_count")
                != WAVEGUIDE_OPTIMIZER_DIAGNOSTIC_METRIC_COUNT
                or type(
                    evidence.get("diagnostic_only_out_of_tolerance_count")
                )
                is not int
            ):
                raise WorkloadError(
                    f"{label} waveguide diagnostic census differs"
                )
        elif (
            "diagnostic_only_count" in evidence
            or "diagnostic_only_out_of_tolerance_count" in evidence
        ):
            raise WorkloadError(f"{label} waveguide CUDA comparison weakened")
        _waveguide_diagnostic_summary_gate(
            pairwise,
            diagnostic_optimizer=candidate_lane == "gpmeep_v110_cpu",
        )
        for field in ("reference_self_oracle", "candidate_self_oracle"):
            _validate_waveguide_optimizer_self_oracle(
                value[field], f"{label}.{field}"
            )
    elif case_path == ADJOINT_CYL_CASE:
        if set(value) != {
            "outcome",
            "failure_count",
            "pairwise",
            "reference_self_oracle",
            "candidate_self_oracle",
            "reference_fixed_work",
            "candidate_fixed_work",
        }:
            raise WorkloadError(f"{label} adjoint result schema differs")
        _validate_compact_comparison(value["pairwise"], f"{label}.pairwise")
    else:
        _validate_compact_comparison(value, label)
    if (
        value.get("outcome") != "PASS"
        or type(value.get("failure_count")) is not int
        or value["failure_count"] != 0
    ):
        raise WorkloadError(f"{label} result outcome differs")
    if case_path != WAVEGUIDE_CROSSING_CASE:
        _validate_failure_free_tree(value, label)


def _validate_bounded_numerical_replay(
    numerical: Any,
    tasks: list[dict[str, Any]],
    repo: pathlib.Path = REPO_DEFAULT,
) -> None:
    if (
        not isinstance(numerical, dict)
        or set(numerical) != {"schema", "outcome", "comparison_count", "comparisons"}
        or numerical.get("schema") != "gpmeep-v110-m4-numerical-replay-v1"
        or numerical.get("outcome") != "PASS"
        or type(numerical.get("comparison_count")) is not int
        or numerical.get("comparison_count") != 141
        or not isinstance(numerical.get("comparisons"), list)
        or len(numerical["comparisons"]) != 141
    ):
        raise WorkloadError("M4 authority numerical replay schema differs")
    numerical_validator = _load_normative_inversion_registry(repo)
    expected = []
    for task in tasks:
        if task["lane_id"] == "upstream_meep_cpu":
            continue
        reference_lane = (
            "gpmeep_v110_cpu"
            if task["case_path"] in GPMEEP_CPU_REFERENCE_CASES
            and task["lane_id"] != "gpmeep_v110_cpu"
            else "upstream_meep_cpu"
        )
        expected.append(
            {
                "cycle_index": task["cycle_index"],
                "sample_kind": task["sample_kind"],
                "case_path": task["case_path"],
                "reference_lane": reference_lane,
                "candidate_lane": task["lane_id"],
            }
        )
    for row, identity in zip(numerical["comparisons"], expected, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {*identity, "method", "result"}
            or any(row.get(name) != value for name, value in identity.items())
            or row.get("method")
            != _expected_numerical_method(
                identity["case_path"], identity["candidate_lane"]
            )
        ):
            raise WorkloadError("M4 authority numerical comparison differs")
        _validate_numerical_result(
            identity["case_path"], identity["candidate_lane"], row["result"],
            f"M4 authority numerical comparison {identity['case_path']}",
        )
        try:
            numerical_validator._verify_m4_numerical_result(
                identity["case_path"],
                identity["candidate_lane"],
                row["method"],
                row["result"],
                f"M4 authority numerical comparison {identity['case_path']}",
            )
        except Exception as exc:
            raise WorkloadError(
                "M4 authority numerical comparison normative replay failed: "
                f"{exc}"
            ) from exc


def verify_experiment_authority(output: pathlib.Path) -> dict[str, Any]:
    """Replay the sealed M4 identity without repeating the expensive oracles.

    Supplemental inversion samples call this path before every launch.  The
    full semantic verifier remains the release authority; this bounded replay
    verifies every terminal-bound aggregate, checkpoint, journal, inversion
    record, and resume archive byte, then checks their cross-links.  The
    experiment producer separately revalidates the live installed package,
    source commit, archive payload, and runtime identities against the returned
    contract.
    """

    lexical = output.absolute()
    if lexical.is_symlink():
        raise WorkloadError("M4 authority output is a symlink")
    output = lexical.resolve(strict=True)
    if output != lexical or not output.is_dir():
        raise WorkloadError("M4 authority output is not a direct directory")
    contradictory = [
        name
        for name in ("PENDING_M4_COMPLETE", "RUNNING.json", "FAILED.json")
        if (output / name).exists() or (output / name).is_symlink()
    ]
    if contradictory:
        raise WorkloadError(
            f"M4 authority has contradictory state markers: {contradictory}"
        )
    terminal = _load_json(output / "M4_COMPLETE", "M4 authority COMPLETE")
    report = matrix.load_stable_json_record(
        output, terminal.get("report"), "M4 authority report"
    )
    journal = matrix.load_stable_json_record(
        output, terminal.get("journal"), "M4 authority journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, terminal.get("checkpoint"), "M4 authority checkpoint"
    )
    report_keys = {
        "schema", "outcome", "contract", "tasks", "performance", "numerical",
        "hardware", "capacity_initial", "capacity_final", "resume_history",
        "checkpoint", "performance_replay", "numerical_replay",
        "inversion_records",
    }
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {"schema", "contract", "hardware_before", "capacity_before", "completed_tasks"}
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("M4 authority report/checkpoint schema differs")
    contract = checkpoint["contract"]
    completed = checkpoint["completed_tasks"]
    repo, tasks, devices = validate_contract(contract)
    if (
        not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report["contract"] != contract
        or report["tasks"] != completed
    ):
        raise WorkloadError("M4 authority task/contract inventory differs")
    if expected_journal(journal, contract, completed) != journal.get("events"):
        raise WorkloadError("M4 authority journal differs from its checkpoint")
    cache: dict[pathlib.Path, tuple[int, str]] = {}
    child_context = _bounded_child_context(repo, contract)
    reports = [
        _validate_completed_task_authority(
            task, retained, output, contract, devices, cache, child_context
        )
        for task, retained in zip(tasks, completed, strict=True)
    ]
    saved_performance = matrix.load_stable_json_record(
        output, report["performance_replay"], "M4 authority performance replay"
    )
    saved_numerical = matrix.load_stable_json_record(
        output, report["numerical_replay"], "M4 authority numerical replay"
    )
    retained_performance = dict(saved_performance)
    inversion_values = retained_performance.pop("inversion_records", None)
    if (
        retained_performance != equal_work_replay.derive_results(contract["plan"], reports)
        or saved_performance != report["performance"]
        or saved_numerical != report["numerical"]
    ):
        raise WorkloadError("M4 authority aggregate verdict differs")
    _validate_bounded_numerical_replay(saved_numerical, tasks, repo)
    if (
        not isinstance(inversion_values, list)
        or len(inversion_values) != 13
        or not isinstance(report["inversion_records"], list)
        or len(report["inversion_records"]) != 13
    ):
        raise WorkloadError("M4 authority inversion census differs")
    inventory_path = repo / "packaging/v110-performance-inversion-inventory.json"
    inventory = validation.load_json(inventory_path)
    inventory_record = matrix.stable_file_record(
        inventory_path, repo, "M4 authority inversion inventory"
    )
    numerical_record = matrix.stable_file_record(
        output / "numerical-replay.json", output, "M4 authority numerical replay"
    )
    sample_evidence = {
        task["task_id"]: {
            "path": retained["sample_replay"]["report"]["path"],
            "sha256": retained["sample_replay"]["report"]["sha256"],
        }
        for task, retained in zip(tasks, completed, strict=True)
    }
    expected_inversions = equal_work_replay.derive_inversion_records(
        contract["plan"], reports, sample_evidence, inventory,
        inventory_record["path"], inventory_record["sha256"],
        {"path": numerical_record["path"], "sha256": numerical_record["sha256"]},
    )
    if inversion_values != expected_inversions:
        raise WorkloadError("M4 authority inversion semantics differ")
    for index, (record, expected) in enumerate(
        zip(report["inversion_records"], expected_inversions, strict=True)
    ):
        saved = matrix.load_stable_json_record(
            output, record, f"M4 authority inversion record {index}"
        )
        if saved != expected:
            raise WorkloadError("M4 authority inversion record differs")
    matrix.validate_hardware_evidence(report["hardware"], list(devices))
    if checkpoint["hardware_before"] != report["hardware"]["before"]:
        raise WorkloadError("M4 authority initial hardware differs")
    if checkpoint["capacity_before"] != report["capacity_initial"]:
        raise WorkloadError("M4 authority initial capacity differs")
    validate_capacity(report["capacity_initial"], True)
    validate_capacity(report["capacity_final"], False)
    if report["resume_history"] != matrix.resume_history_records(output):
        raise WorkloadError("M4 authority resume history differs")
    if terminal != _terminal_payload(output, report):
        raise WorkloadError("M4 authority terminal payload differs")
    return report


def verify_complete(
    output: pathlib.Path, terminal_name: str = "M4_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"M4_COMPLETE", "PENDING_M4_COMPLETE"}:
        raise WorkloadError("M4 terminal filename differs")
    forbidden = {"RUNNING.json", "FAILED.json"}
    forbidden.add(
        "PENDING_M4_COMPLETE"
        if terminal_name == "M4_COMPLETE"
        else "M4_COMPLETE"
    )
    contradictory = sorted(
        name
        for name in forbidden
        if (output / name).exists() or (output / name).is_symlink()
    )
    if contradictory:
        raise WorkloadError(
            f"M4 PASS has contradictory state markers: {contradictory}"
        )
    terminal = _load_json(output / terminal_name, "M4 COMPLETE")
    if terminal.get("schema") != COMPLETE_SCHEMA or terminal.get("outcome") != "PASS":
        raise WorkloadError("M4 COMPLETE is not a PASS")
    report = matrix.load_stable_json_record(output, terminal.get("report"), "M4 report")
    journal = matrix.load_stable_json_record(output, terminal.get("journal"), "M4 journal")
    checkpoint = matrix.load_stable_json_record(output, terminal.get("checkpoint"), "M4 checkpoint")
    report_keys = {
        "schema", "outcome", "contract", "tasks", "performance", "numerical",
        "hardware", "capacity_initial", "capacity_final", "resume_history",
        "checkpoint", "performance_replay", "numerical_replay",
        "inversion_records",
    }
    if not isinstance(report, dict) or set(report) != report_keys or report.get("schema") != SCHEMA or report.get("outcome") != "PASS":
        raise WorkloadError("M4 report is not an exact PASS")
    contract = checkpoint.get("contract") if isinstance(checkpoint, dict) else None
    _repo, tasks, devices = validate_contract(contract)
    completed = checkpoint.get("completed_tasks")
    if (
        report["contract"] != contract
        or not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report["tasks"] != completed
    ):
        raise WorkloadError("M4 completed task inventory differs")
    reports = []
    runtime_session = v110_sample.RuntimeVerificationSession()
    try:
        for task, retained in zip(tasks, completed, strict=True):
            _validated, sample_report = _validate_completed_task_with_report(
                task,
                retained,
                output,
                contract,
                devices,
                v110_runtime_session=runtime_session,
            )
            reports.append(sample_report)
        if expected_journal(journal, contract, completed) != journal["events"]:
            raise WorkloadError("M4 journal has unsealed events")
        performance, numerical = _derive_aggregate_from_reports(
            output, contract, reports
        )
        saved_numerical = matrix.load_stable_json_record(
            output, report["numerical_replay"], "M4 numerical_replay"
        )
        if saved_numerical != numerical:
            raise WorkloadError("M4 numerical_replay was not re-derived")
        inversion_records = _derive_inversion_records_from_reports(
            output, contract, completed, reports
        )
        validate_inversion_records(
            output,
            report["inversion_records"],
            inversion_records,
            pathlib.Path(contract["paths"]["repo"]),
        )
        performance["inversion_records"] = inversion_records
        replay_records = {
            "performance_replay": (
                output / "performance-replay.json",
                performance,
            ),
        }
        for name, (path, derived) in replay_records.items():
            retained_record = report[name]
            saved = matrix.load_stable_json_record(
                output, retained_record, f"M4 {name}"
            )
            if saved != derived:
                raise WorkloadError(f"M4 {name} was not re-derived")
        if report["performance"] != performance or report["numerical"] != numerical:
            raise WorkloadError("M4 report aggregation differs")
        matrix.validate_hardware_evidence(report["hardware"], list(devices))
        if checkpoint.get("hardware_before") != report["hardware"]["before"]:
            raise WorkloadError("M4 initial hardware differs")
        if checkpoint.get("capacity_before") != report["capacity_initial"]:
            raise WorkloadError("M4 initial capacity differs")
        validate_capacity(report["capacity_initial"], True)
        validate_capacity(report["capacity_final"], False)
        if report["resume_history"] != matrix.resume_history_records(output):
            raise WorkloadError("M4 resume history differs")
        _revalidate_cached_reports_authority(
            output, contract, completed, devices, reports
        )
        if terminal != _terminal_payload(output, report):
            raise WorkloadError("M4 COMPLETE payload differs")
        # The full live provenance check is deliberately the final
        # source/runtime operation before accepting this terminal record.
        runtime_session.finalize()
        return report
    except BaseException as error:
        runtime_error = runtime_session.abort()
        if runtime_error is not None:
            raise WorkloadError(
                "M4 terminal verification failed and runtime finalization "
                f"also failed: primary={type(error).__name__}: {error}; "
                f"runtime={runtime_error}"
            ) from error
        raise


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(output / "PENDING_M4_COMPLETE", _terminal_payload(output, report))
    verify_complete(output, "PENDING_M4_COMPLETE")
    durable_publish_terminal(output / "PENDING_M4_COMPLETE", output / "M4_COMPLETE")


def remove_running_marker(output: pathlib.Path, *, required: bool) -> None:
    marker = output / "RUNNING.json"
    try:
        mode = marker.lstat().st_mode
    except FileNotFoundError:
        if required:
            raise WorkloadError("M4 RUNNING marker is absent")
        return
    if stat.S_ISDIR(mode):
        raise WorkloadError("M4 RUNNING marker is a directory")
    marker.unlink()
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_output_isolation(output: pathlib.Path, repo: pathlib.Path) -> None:
    absolute = pathlib.Path(os.path.abspath(output))
    repo = repo.resolve(strict=True)
    if absolute == repo or absolute in repo.parents or repo in absolute.parents:
        raise WorkloadError("M4 output overlaps the immutable source repository")


def validate_controller_dependencies() -> None:
    """Fail before launch when a terminal-evidence dependency is absent."""
    try:
        module = importlib.import_module("jsonschema")
    except ImportError as error:
        raise WorkloadError(
            "M4 controller requires the pinned jsonschema dependency"
        ) from error
    if not hasattr(module, "Draft202012Validator"):
        raise WorkloadError("M4 controller jsonschema capability differs")


def preflight_contract(
    args: argparse.Namespace, repo: pathlib.Path
) -> dict[str, Any]:
    """Prove that the exact final authorities are usable before any task starts."""
    host_binding()
    plan = equal_work_plan.load_plan(repo)
    tasks = equal_work_plan.build_tasks(plan)
    devices = normalized_devices(args.gpu_devices)
    code_stables: dict[str, StableFile] = {}
    input_stables: list[StableFile] = []
    runtime_session: v110_sample.RuntimeVerificationSession | None = None
    try:
        capacity = validate_precreation_capacity(args.output)
        code_stables, code_records = _code_records(repo)
        for path, label in (
            (args.build_receipt, "M4 v1.1.0 receipt"),
            (args.upstream_receipt, "M4 upstream receipt"),
            (args.v103_provenance_report, "M4 v1.0.3 provenance"),
            (args.v103_package, "M4 v1.0.3 package"),
            (args.v110_provenance_report, "M4 v1.1.0 provenance"),
            (args.v110_package, "M4 v1.1.0 package"),
            (pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"),
        ):
            input_stables.append(StableFile(path.resolve(strict=True), label))
        contract = build_contract(args, repo, plan, tasks, code_records, devices)
        validated_repo, validated_tasks, validated_devices = validate_contract(contract)
        hardware = matrix.hardware_snapshot(validated_devices)
        matrix.validate_hardware_evidence(
            {"before": hardware, "after": hardware}, list(validated_devices)
        )
        for stable in [*code_stables.values(), *input_stables]:
            stable.verify_unchanged()
        runtime = contract["runtime_inputs"]
        result = {
            "schema": PREFLIGHT_SCHEMA,
            "state": "PASS",
            "workloads_launched": 0,
            "repo": str(validated_repo),
            "source_commit": runtime["v110_source_identity"]["git_head"],
            "upstream_receipt_id": runtime["upstream_receipt_id"],
            "v110_candidate_receipt_id": runtime["v110_candidate_receipt_id"],
            "v110_package_sha256": runtime["v110_package_sha256"],
            "unittest_import_preflight": runtime[
                "unittest_import_preflight"
            ],
            "task_count": len(validated_tasks),
            "measured_task_count": sum(
                task["sample_kind"] == "measured" for task in validated_tasks
            ),
            "gpu_devices": list(validated_devices),
            "hardware": hardware,
            "capacity": capacity,
        }
        result["preflight_id"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return result
    finally:
        for stable in [*code_stables.values(), *input_stables]:
            stable.close(verify=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    validate_controller_dependencies()
    nvidia_uvm_hmm_binding()
    if args.verify_complete:
        verify_complete(args.output)
        return 0
    repo = args.repo.resolve(strict=True)
    validate_repo_ownership(repo)
    _validate_output_isolation(args.output, repo)
    if args.preflight_only:
        print(json.dumps(preflight_contract(args, repo), sort_keys=True))
        return 0
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("M4 resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "M4_COMPLETE").exists():
            raise WorkloadError("completed M4 output cannot be resumed")
    else:
        # Rehearse every authority and import-only unittest contract before
        # creating even the LOCK/output directory.  The normal controller
        # validates the same sealed records again after acquisition so a
        # runtime mutation between preflight and launch also fails closed.
        preflight_contract(args, repo)
        if args.output.is_symlink():
            raise WorkloadError("M4 output is a symlink")
        output = pathlib.Path(os.path.abspath(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError("M4 output parent contains a symlink")
        output.mkdir(mode=0o700, exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("M4 lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire M4 lock: {exc}") from exc

    code_stables: dict[str, StableFile] = {}
    input_stables: list[StableFile] = []
    try:
        plan = equal_work_plan.load_plan(repo)
        tasks = equal_work_plan.build_tasks(plan)
        devices = normalized_devices(args.gpu_devices)
        code_stables, code_records = _code_records(repo)
        for path, label in (
            (args.build_receipt, "M4 v1.1.0 receipt"),
            (args.upstream_receipt, "M4 upstream receipt"),
            (args.v103_provenance_report, "M4 v1.0.3 provenance"),
            (args.v103_package, "M4 v1.0.3 package"),
            (args.v110_provenance_report, "M4 v1.1.0 provenance"),
            (args.v110_package, "M4 v1.1.0 package"),
            (pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"),
        ):
            input_stables.append(StableFile(path.resolve(strict=True), label))
        contract = build_contract(args, repo, plan, tasks, code_records, devices)
        validate_contract(contract)
        runtime_session = v110_sample.RuntimeVerificationSession()
        current_hardware = matrix.hardware_snapshot(devices)
        current_capacity = capacity_snapshot(output)
        completed: list[dict[str, Any]] = []
        if args.resume:
            checkpoint = _load_checkpoint(output, contract)
            hardware_before = checkpoint["hardware_before"]
            capacity_before = checkpoint["capacity_before"]
            matrix.validate_hardware_evidence(
                {"before": hardware_before, "after": current_hardware}, list(devices)
            )
            validate_capacity(capacity_before, True)
            if len(checkpoint["completed_tasks"]) > TASK_COUNT:
                raise WorkloadError("M4 checkpoint has too many tasks")
            for retained, task in zip(checkpoint["completed_tasks"], tasks, strict=False):
                completed.append(
                    validate_completed_task(
                        task,
                        retained,
                        output,
                        contract,
                        devices,
                        v110_runtime_session=runtime_session,
                    )
                )
            journal = _load_json(output / "JOURNAL.json", "M4 journal")
            prefix = expected_journal(journal, contract, completed)
            partial = [
                pathlib.PurePosixPath("PENDING_M4_COMPLETE"),
                pathlib.PurePosixPath("performance-replay.json"),
                pathlib.PurePosixPath("numerical-replay.json"),
                pathlib.PurePosixPath("inversion-records"),
                pathlib.PurePosixPath("report.json"),
            ]
            if len(completed) < TASK_COUNT:
                task = tasks[len(completed)]
                partial.extend(
                    [task_sample_relative(task), task_runtime_relative(task), task_log_relative(task), task_telemetry_relative(task)]
                )
            matrix.archive_interrupted_attempt(output, partial)
            journal["events"] = prefix
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(output / "RUNNING.json", {"schema": SCHEMA, "resumed_unix_seconds": time.time(), "completed_tasks": len(completed)})
        else:
            hardware_before = current_hardware
            capacity_before = current_capacity
            validate_capacity(capacity_before, True)
            journal = {"schema": JOURNAL_SCHEMA, "contract": contract, "events": [{"state": "controller-started", "unix_seconds": time.time()}]}
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(output / "RUNNING.json", {"schema": SCHEMA, "started_unix_seconds": time.time()})
            write_checkpoint(output, contract, hardware_before, capacity_before, completed)

        for task in tasks[len(completed):]:
            validate_host_binding(contract["host_binding"])
            before_capacity = capacity_snapshot(output)
            validate_capacity(before_capacity, False)
            before_hardware = matrix.hardware_snapshot(devices)
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = clean_environment(task, contract, runtime)
            append_event(output, journal, {"state": "task-started", "task": task})
            process, observer_records = observed.run_bounded_observed(
                task_command(task, contract, output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                contract["limits"]["task_timeout_seconds"] + 600,
                contract["limits"]["stdout_limit_bytes"],
                lambda pid, selected=task: observer_specs(selected, pid, output, contract),
                observer_environment=environment,
                observer_exit_timeout_seconds=contract["limits"]["telemetry_interval_seconds"] * 3 + 30,
                working_directory=repo,
            )
            append_event(output, journal, {"state": "task-process-ended", "task": task, "process": process})
            if process["returncode"] != 0 or process["timed_out"] or process["output_limited"]:
                raise WorkloadError(f"M4 task failed: {task_name(task)}")
            after_hardware = matrix.hardware_snapshot(devices)
            hardware = {"before": before_hardware, "after": after_hardware}
            matrix.validate_hardware_evidence(hardware, list(devices))
            telemetry_record = publish_telemetry(task, process, observer_records, output, contract)
            sample_replay, _sample_report = derive_sample_replay(
                task,
                output,
                contract,
                v110_runtime_session=runtime_session,
            )
            retained = {
                "task": task,
                "process": process,
                "observers": observer_records,
                "telemetry_replay": telemetry_record,
                "sample_replay": sample_replay,
                "hardware": hardware,
                "capacity_before": before_capacity,
                "capacity_after": capacity_snapshot(output),
            }
            completed.append(retained)
            append_event(output, journal, {"state": "task-validated", "task": task, "sample_complete": sample_replay["complete"]})
            write_checkpoint(output, contract, hardware_before, capacity_before, completed)

        reports = _derive_verified_sample_reports(
            output,
            contract,
            completed,
            v110_runtime_session=runtime_session,
        )
        performance, numerical = _derive_aggregate_from_reports(
            output, contract, reports
        )
        atomic_write_json(output / "numerical-replay.json", numerical)
        inversion_records = _derive_inversion_records_from_reports(
            output, contract, completed, reports
        )
        inversion_record_files = publish_inversion_records(output, inversion_records)
        performance["inversion_records"] = inversion_records
        atomic_write_json(output / "performance-replay.json", performance)
        hardware = {"before": hardware_before, "after": matrix.hardware_snapshot(devices)}
        matrix.validate_hardware_evidence(hardware, list(devices))
        validate_host_binding(contract["host_binding"])
        for stable in [*code_stables.values(), *input_stables]:
            stable.verify_unchanged()
        runtime_session.finalize()
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "contract": contract,
            "tasks": completed,
            "performance": performance,
            "numerical": numerical,
            "hardware": hardware,
            "capacity_initial": capacity_before,
            "capacity_final": capacity_snapshot(output),
            "resume_history": matrix.resume_history_records(output),
            "checkpoint": matrix.stable_file_record(output / "CHECKPOINT.json", output, "M4 checkpoint"),
            "performance_replay": matrix.stable_file_record(output / "performance-replay.json", output, "M4 performance replay"),
            "numerical_replay": matrix.stable_file_record(output / "numerical-replay.json", output, "M4 numerical replay"),
            "inversion_records": inversion_record_files,
        }
        atomic_write_json(output / "report.json", report)
        remove_running_marker(output, required=True)
        publish_terminal(output, report)
        return 0
    except Exception as exc:
        runtime_error = (
            runtime_session.abort() if runtime_session is not None else None
        )
        remove_running_marker(output, required=False)
        pending = output / "PENDING_M4_COMPLETE"
        if pending.exists() or pending.is_symlink():
            if pending.is_dir() and not pending.is_symlink():
                raise WorkloadError("M4 pending terminal is a directory") from exc
            pending.unlink()
        failure = {
            "schema": SCHEMA,
            "outcome": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_unix_seconds": time.time(),
            "partial_artifacts": matrix.collect_partial_evidence(output),
        }
        if (output / "JOURNAL.json").is_file():
            failure["journal"] = file_record(output / "JOURNAL.json", output)
        if runtime_error is not None:
            failure["runtime_finalization_error"] = runtime_error
        matrix.publish_failure_record(output, failure)
        if runtime_error is not None:
            raise WorkloadError(
                "M4 controller failed and runtime finalization also failed: "
                f"primary={type(exc).__name__}: {exc}; "
                f"runtime={runtime_error}"
            ) from exc
        raise
    finally:
        if runtime_session is not None:
            runtime_session.close(verify=False)
        for stable in [*code_stables.values(), *input_stables]:
            stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M4 equal-work matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
