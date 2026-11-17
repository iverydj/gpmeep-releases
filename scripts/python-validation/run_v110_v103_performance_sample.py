#!/usr/bin/env python3
"""Run and seal one exact frozen-v1.0.3 CUDA sample for the v1.1 M4 matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import pwd
import subprocess
import sys
import time
import uuid
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_m3_auto_example_case as auto_case  # noqa: E402
import run_m3_mpi_example_case as mpi_case  # noqa: E402
import run_m3_performance_sample as performance  # noqa: E402
import run_validation as validation  # noqa: E402
import v103_runtime_identity as identity  # noqa: E402
from common import (  # noqa: E402
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    durable_publish_terminal,
    file_record,
)


SCHEMA = "gpmeep-v110-frozen-v103-performance-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-frozen-v103-performance-sample-complete-v1"
TOPOLOGIES = ("v103-cuda1", "v103-cuda2")
PLAN_RELATIVE = pathlib.Path("packaging/v110-m4-equal-work-plan.json")
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
    "provenance",
    "source_snapshot",
    "runtime_contract",
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
    "process_identity",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--topology", required=True, choices=TOPOLOGIES)
    parser.add_argument(
        "--sample-kind", required=True, choices=performance.SAMPLE_KINDS
    )
    parser.add_argument("--cycle-index", required=True, type=int)
    parser.add_argument("--install-prefix", required=True, type=pathlib.Path)
    parser.add_argument("--provenance-report", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def _canonical_file(path: pathlib.Path, label: str) -> pathlib.Path:
    provided = pathlib.Path(os.path.abspath(path))
    try:
        metadata = provided.lstat()
    except OSError as exc:
        raise WorkloadError(f"{label} is unavailable: {exc}") from exc
    if provided.is_symlink() or not provided.is_file() or provided.resolve(strict=True) != provided:
        raise WorkloadError(f"{label} must be a canonical non-symlink file")
    if metadata.st_size < 0:
        raise WorkloadError(f"{label} metadata differs")
    return provided


def _load_plan(repo: pathlib.Path) -> dict[str, Any]:
    plan = validation.load_json(repo / PLAN_RELATIVE)
    if (
        plan.get("schema") != "gpmeep-v110-m4-equal-work-plan-v1"
        or plan.get("release_train") != "1.1.0"
        or plan.get("frozen_v103")
        != {
            "package_name": identity.FROZEN_PACKAGE_NAME,
            "package_sha256": identity.FROZEN_PACKAGE_SHA256,
            "provenance_report_sha256": identity.FROZEN_PROVENANCE_REPORT_SHA256,
            "provenance_report_id": identity.FROZEN_PROVENANCE_REPORT_ID,
            "source_commit": identity.FROZEN_SOURCE_COMMIT,
            "rebuild_permitted": False,
            "fresh_dedicated_prefix_required": True,
        }
    ):
        raise WorkloadError("M4 frozen-v1.0.3 equal-work plan differs")
    return plan


def _case_plan(plan: dict[str, Any], case_path: str) -> dict[str, Any]:
    matches = [item for item in plan.get("cases", []) if item.get("path") == case_path]
    if len(matches) != 1:
        raise WorkloadError("M4 frozen-v1.0.3 case is not uniquely planned")
    return matches[0]


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(
        mpi_case.normalized_gpu(item) for item in value.split(",") if item
    )
    if len(devices) != 2 or len({mpi_case._identity(item) for item in devices}) != 2:
        raise WorkloadError("M4 frozen-v1.0.3 runner requires two distinct GPUs")
    return devices  # type: ignore[return-value]


def select_lane(
    topology: str,
    devices: tuple[str, str],
    case_plan: dict[str, Any],
) -> mpi_case.Lane:
    if topology == "v103-cuda1":
        return mpi_case.Lane("v103-cuda-fp32-1r", "cuda", 1, devices[:1])
    if topology == "v103-cuda2" and case_plan.get("gpu2") is True:
        return mpi_case.Lane("v103-cuda-fp32-2r", "cuda", 2, devices)
    raise WorkloadError("M4 frozen-v1.0.3 topology/case pairing differs")


def runtime_paths(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, pathlib.Path], dict[str, Any]]:
    repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
    if not repo.is_dir() or repo.is_symlink():
        raise WorkloadError("M4 source repository path differs")
    manifest = _canonical_file(args.manifest, "M4 manifest")
    if manifest != (repo / "scripts/python-validation/manifest.json").resolve(strict=True):
        raise WorkloadError("M4 manifest is not authoritative")
    try:
        report, installed = identity.validate_report(
            args.provenance_report,
            args.package,
            pathlib.Path(os.path.abspath(args.install_prefix)),
        )
        contract = identity.runtime_contract(report, installed)
    except identity.V103IdentityError as exc:
        raise WorkloadError(str(exc)) from exc
    prefix = installed["install_prefix"]
    python = installed["python_executable"]
    site_packages = installed["meep_module"].parent.parent
    mpiexec = (prefix / "bin/mpirun").resolve(strict=True)
    verifier = (prefix / "bin/gpmeep-verify-provenance").resolve(strict=True)
    if (
        not site_packages.is_dir()
        or not mpiexec.is_file()
        or not verifier.is_file()
        or not os.access(python, os.X_OK)
        or not os.access(mpiexec, os.X_OK)
        or not os.access(verifier, os.X_OK)
    ):
        raise WorkloadError("M4 frozen-v1.0.3 executable layout differs")
    for path in (python, site_packages, mpiexec, verifier):
        try:
            path.relative_to(prefix)
        except ValueError as exc:
            raise WorkloadError("M4 frozen-v1.0.3 runtime escaped its prefix") from exc
    report_path = installed["provenance_report"]
    package = installed["package"]
    if prefix in report_path.parents or prefix in package.parents:
        raise WorkloadError("M4 frozen-v1.0.3 package/provenance must be prefix-external")
    paths = {
        "repo": repo,
        "manifest": manifest,
        "install_prefix": prefix,
        "provenance_report": report_path,
        "package": package,
        "python": python,
        "site_packages": site_packages,
        "mpiexec": mpiexec,
        "verifier": verifier,
        "launcher": (SCRIPT_DIR / "m3_mpi_rank_launcher.py").resolve(strict=True),
        "fontconfig": pathlib.Path(contract["fontconfig_file"]["path"]).resolve(strict=True),
        "libmeep": installed["libmeep"],
    }
    return report, paths, contract


def live_verify(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> dict[str, Any]:
    """Rerun the verifier shipped in the historical package and compare all output."""
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise WorkloadError(f"M4 verifier account home is unavailable: {exc}") from exc
    environment = {
        "HOME": str(account_home),
        "PATH": f"{paths['install_prefix'] / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(paths["install_prefix"] / "lib"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MEEP_GPU_BACKEND": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        # The historical Open MPI build probes osc_ucx while importing meep
        # even though this verifier uses no one-sided MPI windows.  Keep the
        # exclusion local to this closed verifier environment.
        "OMPI_MCA_osc": "^ucx",
    }
    command = [
        str(paths["verifier"]),
        "--prefix",
        str(paths["install_prefix"]),
        "--expected-source-commit",
        identity.FROZEN_SOURCE_COMMIT,
        "--expected-package-sha256",
        identity.FROZEN_PACKAGE_SHA256,
    ]
    try:
        process = subprocess.run(
            command,
            cwd=paths["install_prefix"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise WorkloadError(f"M4 frozen-v1.0.3 live verifier failed: {exc}") from exc
    payloads = []
    for line in process.stdout.splitlines():
        if line.startswith("{"):
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise WorkloadError("M4 v1.0.3 verifier emitted invalid JSON") from exc
    if process.returncode != 0 or process.stderr or payloads != [report]:
        raise WorkloadError("M4 frozen-v1.0.3 live provenance differs")
    return {
        "command": command,
        "report_id": report["report_id"],
        "package_sha256": report["package_sha256"],
        "state": "PASS",
    }


def legacy_case_command(
    case: dict[str, Any], python: pathlib.Path, repo: pathlib.Path
) -> list[str]:
    command = performance.performance_oracle_command(case, python, repo)
    ordinary = (repo / "scripts/python-validation/run_example_oracle.py").resolve(strict=True)
    historical = (repo / "scripts/python-validation/run_v103_example_oracle.py").resolve(strict=True)
    if len(command) < 3 or pathlib.Path(command[1]).resolve(strict=True) != ordinary:
        raise WorkloadError("M4 v1.0.3 case does not use the generic oracle")
    if "--upstream-cpu-baseline" in command or "--acknowledge-backend-mismatch-reason" in command:
        raise WorkloadError("M4 v1.0.3 case contains another runtime mode")
    return [command[0], str(historical), *command[2:]]


def clean_environment(
    lane: mpi_case.Lane,
    paths: dict[str, pathlib.Path],
    contract: dict[str, Any],
    report: dict[str, Any],
    nonce: str,
    *,
    output: pathlib.Path,
    case_path: str,
) -> dict[str, str]:
    environment = mpi_case.clean_environment(
        lane,
        python=paths["python"],
        build_python=paths["site_packages"],
        runtime_contract=contract,
        run_nonce=nonce,
        site_guard_expected=False,
        producer_safe_ambient=mpi_case.expected_performance_producer_safe_ambient(
            output, paths["python"]
        ),
    )
    environment.update(
        {
            identity.ENV_REPORT: str(paths["provenance_report"]),
            identity.ENV_PACKAGE: str(paths["package"]),
            identity.ENV_REPORT_ID: report["report_id"],
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "cuda",
        }
    )
    if case_path == "python/tests/test_adjoint_cyl.py":
        environment[performance.ADJOINT_CYL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(performance.ADJOINT_CYL_FIXED_WORK_ENV, None)
    if case_path == performance.PHASE_IN_MATERIAL_CASE:
        environment[performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV, None)
    return environment


def expected_process_identity(
    report: dict[str, Any], paths: dict[str, pathlib.Path]
) -> dict[str, Any]:
    return {
        "schema": "gpmeep-frozen-v103-process-identity-v1",
        "report_id": report["report_id"],
        "package_sha256": identity.FROZEN_PACKAGE_SHA256,
        "source_commit": identity.FROZEN_SOURCE_COMMIT,
        "python": str(paths["python"]),
        "module": report["runtime"]["module"]["path"],
        "extension": report["runtime"]["extension"]["path"],
        "libmeep": str(paths["libmeep"]),
    }


def parse_process_identity(
    stdout: pathlib.Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    prefix = auto_case.oracle.V103_IDENTITY_PREFIX
    payloads = [
        line[len(prefix) :]
        for line in stdout.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        raise WorkloadError("M4 v1.0.3 process identity count differs")
    try:
        value = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise WorkloadError("M4 v1.0.3 process identity is invalid") from exc
    if value != expected:
        raise WorkloadError("M4 v1.0.3 process identity differs")
    return value


def _code_records(
    paths: dict[str, pathlib.Path], case_path: str
) -> dict[str, dict[str, Any]]:
    return {
        "sample_runner": absolute_file_record(pathlib.Path(__file__).resolve(), "runner"),
        "runtime_identity": absolute_file_record(
            SCRIPT_DIR / "v103_runtime_identity.py", "runtime identity"
        ),
        "historical_oracle": absolute_file_record(
            SCRIPT_DIR / "run_v103_example_oracle.py", "historical oracle"
        ),
        "generic_oracle": absolute_file_record(
            SCRIPT_DIR / "run_example_oracle.py", "generic oracle"
        ),
        "performance_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_performance_sample.py", "performance common"
        ),
        "automatic_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_auto_example_case.py", "automatic common"
        ),
        "mpi_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_mpi_example_case.py", "MPI common"
        ),
        "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "validation runner"
        ),
        "manifest": absolute_file_record(paths["manifest"], "manifest"),
        "adjoint_cyl_fixed_work_contract": absolute_file_record(
            paths["repo"] / "python/tests/adjoint_cyl_fixed_work_contract.py",
            "cylindrical adjoint fixed-work contract",
        ),
        "equal_work_plan": absolute_file_record(
            paths["repo"] / PLAN_RELATIVE, "equal-work plan"
        ),
        "inversion_inventory": absolute_file_record(
            paths["repo"] / "packaging/v110-performance-inversion-inventory.json",
            "inversion inventory",
        ),
        "case_source": absolute_file_record(paths["repo"] / case_path, "case source"),
    }


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


def _namespace_from_report(report: dict[str, Any]) -> argparse.Namespace:
    raw = report.get("paths")
    if not isinstance(raw, dict):
        raise WorkloadError("M4 v1.0.3 retained runtime paths differ")
    return argparse.Namespace(
        repo=pathlib.Path(raw["repo"]),
        manifest=pathlib.Path(raw["manifest"]),
        install_prefix=pathlib.Path(raw["install_prefix"]),
        provenance_report=pathlib.Path(raw["provenance_report"]),
        package=pathlib.Path(raw["package"]),
    )


def verify_complete(
    output: pathlib.Path, terminal_name: str = "COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M4 v1.0.3 terminal filename differs")
    terminal = auto_case._load_object(output / terminal_name, "M4 v1.0.3 COMPLETE")
    report = auto_case._load_object(output / "report.json", "M4 v1.0.3 report")
    if (
        not isinstance(report, dict)
        or set(report) != REPORT_KEYS
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or terminal != _terminal(output, report)
    ):
        raise WorkloadError("M4 v1.0.3 terminal/report binding differs")
    performance.validate_sample_identity(report["sample_kind"], report["cycle_index"])
    namespace = _namespace_from_report(report)
    retained_paths = report["paths"]
    saved_devices = report.get("gpu_devices")
    if not isinstance(saved_devices, list):
        raise WorkloadError("M4 v1.0.3 GPU inventory differs")
    namespace.gpu_devices = ",".join(saved_devices)
    fresh_report, paths, contract = runtime_paths(namespace)
    if retained_paths != {name: str(path) for name, path in paths.items()}:
        raise WorkloadError("M4 v1.0.3 live runtime paths differ")
    if report["provenance"] != {
        "report": absolute_file_record(paths["provenance_report"], "provenance"),
        "package": absolute_file_record(paths["package"], "package"),
        "live_verification": live_verify(fresh_report, paths),
    }:
        raise WorkloadError("M4 v1.0.3 provenance replay differs")
    if report["runtime_contract"] != contract:
        raise WorkloadError("M4 v1.0.3 runtime contract differs")
    source = validation.source_snapshot(paths["repo"])
    if not source.get("available") or report["source_snapshot"] != source:
        raise WorkloadError("M4 v1.0.3 source snapshot differs")
    plan = _load_plan(paths["repo"])
    case = mpi_case._case_from_manifest(paths["repo"], paths["manifest"], report["case_path"])
    performance.validate_inversion_inventory(paths["repo"])
    case_plan = _case_plan(plan, case["path"])
    devices = normalized_devices(namespace.gpu_devices)
    lane = select_lane(report["topology"], devices, case_plan)
    retained = report["lane"]
    if not isinstance(retained, dict) or set(retained) != LANE_KEYS:
        raise WorkloadError("M4 v1.0.3 lane schema differs")
    if retained["spec"] != mpi_case.lane_spec(lane) or retained["requested_backend"] != "cuda":
        raise WorkloadError("M4 v1.0.3 lane identity differs")
    nonce = retained["run_nonce"]
    if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
        raise WorkloadError("M4 v1.0.3 nonce differs")
    root = output / "lane"
    case_command = legacy_case_command(case, paths["python"], paths["repo"])
    command = mpi_case.lane_command(
        lane,
        mpiexec=paths["mpiexec"],
        python=paths["python"],
        launcher=paths["launcher"],
        evidence_root=root,
        case_command=case_command,
    )
    if retained["command"] != command:
        raise WorkloadError("M4 v1.0.3 command differs")
    environment = clean_environment(
        lane, paths, contract, fresh_report, nonce, output=output,
        case_path=case["path"],
    )
    if retained["environment_sha256"] != validation.environment_sha256(environment):
        raise WorkloadError("M4 v1.0.3 environment differs")
    performance._validate_process(retained["process"])
    rank_records = mpi_case.validate_rank_bundle(
        lane, root, case, contract, nonce, case_command
    )
    if retained["rank_records"] != rank_records:
        raise WorkloadError("M4 v1.0.3 rank records differ")
    runs = auto_case.audit_runs(
        root / "stdout.log", "cuda", frozen_v103_baseline=True
    )
    process_identity = parse_process_identity(
        root / "stdout.log", expected_process_identity(fresh_report, paths)
    )
    policy = {
        "schema": "gpmeep-m4-frozen-v103-policy-v1",
        "classification": "forced-frozen-v103-cuda-performance-sample",
        "active_backends": ["cuda"] * len(runs),
    }
    policy_result = performance.validate_policy(runs, "cuda", lane, policy)
    if retained["policy_result"] != policy_result or retained["process_identity"] != process_identity:
        raise WorkloadError("M4 v1.0.3 dispatch policy differs")
    if report["timing"] != performance.derive_timing(case["path"], runs):
        raise WorkloadError("M4 v1.0.3 timing replay differs")
    if report["phase_profile"] != performance.derive_phase_profile(
        runs, retained["process"]["duration_seconds"]
    ):
        raise WorkloadError("M4 v1.0.3 phase replay differs")
    for name in ("stdout", "stderr"):
        if retained[name] != file_record(root / f"{name}.log", root):
            raise WorkloadError(f"M4 v1.0.3 {name} changed")
    tree = output / "tree-manifest.json"
    if retained["tree_manifest"] != file_record(tree, output) or auto_case._load_object(
        tree, "M4 v1.0.3 tree"
    ) != mpi_case.derive_tree(root):
        raise WorkloadError("M4 v1.0.3 raw tree differs")
    performance.validate_comparison_evidence(
        (root / "stdout.log").read_text(encoding="utf-8"), case["comparison"]
    )
    if report["evidence_code"] != _code_records(paths, case["path"]):
        raise WorkloadError("M4 v1.0.3 evidence code differs")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    performance.validate_sample_identity(args.sample_kind, args.cycle_index)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("M4 v1.0.3 timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M4 v1.0.3 output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        report, paths, contract = runtime_paths(args)
        if output == paths["repo"] or paths["repo"] in output.parents or output == paths["install_prefix"] or paths["install_prefix"] in output.parents:
            raise WorkloadError("M4 v1.0.3 output overlaps source/runtime")
        plan = _load_plan(paths["repo"])
        case = mpi_case._case_from_manifest(paths["repo"], paths["manifest"], args.case)
        performance.validate_inversion_inventory(paths["repo"])
        case_plan = _case_plan(plan, case["path"])
        devices = normalized_devices(args.gpu_devices)
        lane = select_lane(args.topology, devices, case_plan)
        source = validation.source_snapshot(paths["repo"])
        if not source.get("available"):
            raise WorkloadError("M4 v1.0.3 source snapshot is unavailable")
        live = live_verify(report, paths)
        root = output / "lane"
        for name in ("statistics", "identity", "work"):
            (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
        nonce = uuid.uuid4().hex
        environment = clean_environment(
            lane, paths, contract, report, nonce, output=output,
            case_path=case["path"],
        )
        case_command = legacy_case_command(case, paths["python"], paths["repo"])
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
            raise WorkloadError("M4 v1.0.3 sample process failed")
        text = (root / "stdout.log").read_text(encoding="utf-8")
        performance.validate_comparison_evidence(text, case["comparison"])
        runs = auto_case.audit_runs(
            root / "stdout.log", "cuda", frozen_v103_baseline=True
        )
        process_identity = parse_process_identity(
            root / "stdout.log", expected_process_identity(report, paths)
        )
        rank_records = mpi_case.validate_rank_bundle(
            lane, root, case, contract, nonce, case_command
        )
        policy = {
            "schema": "gpmeep-m4-frozen-v103-policy-v1",
            "classification": "forced-frozen-v103-cuda-performance-sample",
            "active_backends": ["cuda"] * len(runs),
        }
        policy_result = performance.validate_policy(runs, "cuda", lane, policy)
        timing = performance.derive_timing(case["path"], runs)
        phase_profile = performance.derive_phase_profile(
            runs, process["duration_seconds"]
        )
        atomic_write_json(output / "tree-manifest.json", mpi_case.derive_tree(root))
        ending_report, ending_paths, ending_contract = runtime_paths(args)
        if (
            ending_report != report
            or ending_paths != paths
            or ending_contract != contract
            or live_verify(ending_report, ending_paths) != live
            or validation.source_snapshot(paths["repo"]) != source
        ):
            raise WorkloadError("M4 v1.0.3 source/runtime changed during sample")
        lane_record = {
            "spec": mpi_case.lane_spec(lane),
            "requested_backend": "cuda",
            "run_nonce": nonce,
            "command": command,
            "environment_sha256": validation.environment_sha256(environment),
            "process": process,
            "stdout": file_record(root / "stdout.log", root),
            "stderr": file_record(root / "stderr.log", root),
            "rank_records": rank_records,
            "tree_manifest": file_record(output / "tree-manifest.json", output),
            "policy_result": policy_result,
            "process_identity": process_identity,
        }
        final = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "topology": args.topology,
            "sample_kind": args.sample_kind,
            "cycle_index": args.cycle_index,
            "gpu_devices": list(devices),
            "receipt_id": report["report_id"],
            "paths": {name: str(path) for name, path in paths.items()},
            "provenance": {
                "report": absolute_file_record(paths["provenance_report"], "provenance"),
                "package": absolute_file_record(paths["package"], "package"),
                "live_verification": live,
            },
            "source_snapshot": source,
            "runtime_contract": contract,
            "lane": lane_record,
            "timing": timing,
            "phase_profile": phase_profile,
            "evidence_code": _code_records(paths, case["path"]),
        }
        atomic_write_json(output / "report.json", final)
        publish_terminal(output, final)
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        WorkloadError,
        identity.V103IdentityError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"M4 frozen-v1.0.3 sample error: {error}", file=sys.stderr)
        raise SystemExit(2)
