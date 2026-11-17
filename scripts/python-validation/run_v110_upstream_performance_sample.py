#!/usr/bin/env python3
"""Run and seal one authoritative upstream-Meep CPU M3 timing sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import pwd
import stat
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
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    durable_publish_terminal,
    file_record,
)


RECEIPT_TOOL = SCRIPT_DIR.parent / "upstream-cpu-baseline-receipt.py"
RECEIPT_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_v110_upstream_cpu_baseline_receipt", RECEIPT_TOOL
)
if RECEIPT_SPEC is None or RECEIPT_SPEC.loader is None:
    raise RuntimeError("cannot load upstream baseline receipt verifier")
receipt_tool = importlib.util.module_from_spec(RECEIPT_SPEC)
RECEIPT_SPEC.loader.exec_module(receipt_tool)


SCHEMA = "gpmeep-v110-upstream-performance-sample-v1"
COMPLETE_SCHEMA = "gpmeep-v110-upstream-performance-sample-complete-v1"
DEFAULT_TOPOLOGY = "upstream-cpu8"
DEFAULT_LANE = mpi_case.Lane("upstream-cpu-fp32-8r", "cpu", 8, ())
ZERO_DIMENSIONAL_CASE = "python/examples/material-dispersion.py"
ZERO_DIMENSIONAL_TOPOLOGY = "upstream-cpu1"
NEAR2FAR_ADJOINT_CASE = (
    "python/examples/adjoint_optimization/near2far_3d_validation.py"
)
NEAR2FAR_UPSTREAM_CAPTURE_ARGUMENT = "--gpmeep-upstream-performance-capture"
ZERO_DIMENSIONAL_LANE = mpi_case.Lane(
    "upstream-cpu-fp32-1r", "cpu", 1, ()
)
REPORT_KEYS = {
    "schema",
    "outcome",
    "case_path",
    "topology",
    "sample_kind",
    "cycle_index",
    "receipt_id",
    "paths",
    "receipt",
    "lane",
    "timing",
    "phase_profile",
    "evidence_code",
}
LANE_KEYS = {
    "spec",
    "run_nonce",
    "command",
    "environment_sha256",
    "process",
    "stdout",
    "stderr",
    "rank_records",
    "tree_manifest",
}


def topology_and_lane(case_path: str) -> tuple[str, mpi_case.Lane]:
    """Return the only physically valid upstream CPU topology for a case."""
    if case_path == ZERO_DIMENSIONAL_CASE:
        return ZERO_DIMENSIONAL_TOPOLOGY, ZERO_DIMENSIONAL_LANE
    return DEFAULT_TOPOLOGY, DEFAULT_LANE


def validate_comparison_evidence(
    text: str, comparison: dict[str, Any]
) -> None:
    """Replay the evidence check appropriate to the sealed comparison mode."""
    mode = comparison.get("mode")
    if mode == "json_metrics":
        validation.extract_json_metrics(text, comparison)
        return
    if mode == "embedded_oracle":
        # The unittest process exit status, pinned unittest contract, and
        # generic-oracle run audit are validated independently by the caller.
        return
    raise WorkloadError(
        f"unsupported upstream performance comparison mode: {mode!r}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--sample-kind", required=True, choices=performance.SAMPLE_KINDS
    )
    parser.add_argument("--cycle-index", required=True, type=int)
    parser.add_argument("--upstream-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def _load_receipt(path: pathlib.Path) -> dict[str, Any]:
    with StableFile(path, "upstream CPU baseline receipt") as stable:
        if stable.initial_stat.st_size > mpi_case.MAX_JSON_BYTES:
            raise WorkloadError("upstream CPU baseline receipt exceeds its bound")
        with stable.file_object() as handle:
            try:
                value = json.load(
                    handle,
                    object_pairs_hook=validation.unique_json_object,
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        WorkloadError(
                            f"upstream receipt contains non-finite value {item!r}"
                        )
                    ),
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise WorkloadError(
                    f"upstream CPU baseline receipt is invalid: {exc}"
                ) from exc
    if not isinstance(value, dict):
        raise WorkloadError("upstream CPU baseline receipt root differs")
    return value


def verify_receipt(
    repo: pathlib.Path, receipt_path: pathlib.Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        verified = receipt_tool.verify(
            argparse.Namespace(
                operation="verify", repo=repo, receipt=receipt_path
            )
        )
    except Exception as exc:
        raise WorkloadError(f"upstream CPU baseline receipt failed: {exc}") from exc
    receipt = _load_receipt(receipt_path)
    if (
        verified.get("state") != "PASS"
        or receipt.get("receipt_id") != verified.get("receipt_id")
    ):
        raise WorkloadError("upstream CPU baseline receipt identity differs")
    return receipt, verified


def _canonical(path: pathlib.Path, *, directory: bool, label: str) -> pathlib.Path:
    provided = pathlib.Path(os.path.abspath(path))
    try:
        metadata = provided.lstat()
    except OSError as exc:
        raise WorkloadError(f"upstream runtime path is unavailable: {label}: {exc}")
    resolved = provided.resolve(strict=True)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or resolved != provided
        or directory != stat.S_ISDIR(metadata.st_mode)
        or not directory
        and not stat.S_ISREG(metadata.st_mode)
    ):
        raise WorkloadError(f"upstream runtime path type differs: {label}")
    return resolved


def runtime_paths(args: argparse.Namespace, receipt: dict[str, Any]) -> dict[str, pathlib.Path]:
    repo = _canonical(args.repo, directory=True, label="repo")
    manifest = _canonical(args.manifest, directory=False, label="manifest")
    receipt_path = _canonical(
        args.upstream_receipt, directory=False, label="upstream_receipt"
    )
    contract = validation.load_json(
        repo / receipt_tool.CONTRACT_RELATIVE
    )
    expected_receipt = (
        repo
        / contract["build"]["build_directory"]
        / "upstream-cpu-baseline-receipt.json"
    ).resolve(strict=True)
    if receipt_path != expected_receipt:
        raise WorkloadError("upstream receipt is not the authoritative build receipt")
    python = pathlib.Path(receipt["python"]["path"]).resolve(strict=True)
    mpiexec = pathlib.Path(receipt["mpiexec"]["path"]).resolve(strict=True)
    if not os.access(python, os.X_OK) or not os.access(mpiexec, os.X_OK):
        raise WorkloadError("upstream runtime executable is not executable")
    install_prefix = (
        repo / contract["build"]["install_prefix"]
    ).resolve(strict=True)
    exact_lock_prefix = (
        repo / contract["dependency_environment"]["prefix_path"]
    ).resolve(strict=True)
    extension = pathlib.Path(
        receipt["artifacts"]["python_extension"]["path"]
    ).resolve(strict=True)
    if extension.parent.name != "meep":
        raise WorkloadError("upstream Python extension package path differs")
    site_packages = extension.parent.parent
    paths = {
        "repo": repo,
        "manifest": manifest,
        "upstream_receipt": receipt_path,
        "python": python,
        "mpiexec": mpiexec,
        "launcher": (SCRIPT_DIR / "m3_mpi_rank_launcher.py").resolve(strict=True),
        "install_prefix": install_prefix,
        "exact_lock_prefix": exact_lock_prefix,
        "site_packages": site_packages,
        "libmeep": pathlib.Path(
            receipt["artifacts"]["libmeep"]["path"]
        ).resolve(strict=True),
        "fontconfig": pathlib.Path(
            receipt["qualification"]["upstream_fontconfig_configuration"]["path"]
        ).resolve(strict=True),
    }
    if manifest != (repo / "scripts/python-validation/manifest.json").resolve(strict=True):
        raise WorkloadError("upstream sample manifest is not authoritative")
    if install_prefix not in paths["libmeep"].parents:
        raise WorkloadError("upstream libmeep escaped its install prefix")
    if site_packages.parent.parent.parent != install_prefix:
        raise WorkloadError("upstream Python package escaped its install prefix")
    for name, path in paths.items():
        if name in {
            "repo",
            "install_prefix",
            "exact_lock_prefix",
            "site_packages",
        }:
            if not path.is_dir():
                raise WorkloadError(f"upstream runtime directory differs: {name}")
        elif not path.is_file():
            raise WorkloadError(f"upstream runtime file differs: {name}")
    return paths


def upstream_case_command(
    case: dict[str, Any], python: pathlib.Path, repo: pathlib.Path
) -> list[str]:
    command = performance.performance_oracle_command(case, python, repo)
    oracle = (repo / "scripts/python-validation/run_example_oracle.py").resolve(
        strict=True
    )
    if len(command) < 3 or pathlib.Path(command[1]).resolve(strict=True) != oracle:
        raise WorkloadError("upstream performance case does not use the generic oracle")
    if "--acknowledge-backend-mismatch-reason" in command:
        raise WorkloadError("upstream performance case contains a gpmeep acknowledgement")
    if NEAR2FAR_UPSTREAM_CAPTURE_ARGUMENT in command:
        raise WorkloadError(
            "upstream performance case already contains its capture argument"
        )
    result = [command[0], command[1], "--upstream-cpu-baseline", *command[2:]]
    if case["path"] == NEAR2FAR_ADJOINT_CASE:
        result.append(NEAR2FAR_UPSTREAM_CAPTURE_ARGUMENT)
    return result


def clean_environment(
    paths: dict[str, pathlib.Path], run_nonce: str, *, output: pathlib.Path,
    case_path: str,
) -> dict[str, str]:
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(
            strict=True
        )
    except (KeyError, OSError) as exc:
        raise WorkloadError(f"upstream MPI account home is unavailable: {exc}") from exc
    environment = mpi_case.expected_performance_producer_safe_ambient(
        output, paths["python"]
    )
    environment.update(
        {
            "HOME": str(account_home),
            "PATH": f"{paths['python'].parent}:/usr/bin:/bin",
            "PYTHONPATH": str(paths["site_packages"]),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (str(paths["libmeep"].parent), str(paths["exact_lock_prefix"] / "lib"))
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "FONTCONFIG_FILE": str(paths["fontconfig"]),
            "CUDA_VISIBLE_DEVICES": "",
            "GPMEEP_VALIDATION_EXPECTED_BACKEND": "cpu",
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "cpu",
            "GPMEEP_VALIDATION_RUN_NONCE": run_nonce,
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            # The sealed benchmark uses no MPI one-sided windows.  The local
            # Open MPI osc_ucx component fails during component query, so keep
            # it out of this isolated harness without changing user runtime
            # defaults or point-to-point/collective transports.
            "OMPI_MCA_osc": "^ucx",
        }
    )
    for forbidden in (
        "MEEP_GPU_BACKEND",
        "GPMEEP_VALIDATION_STRICT_CUDA",
        "GPMEEP_VALIDATION_BUILD_RECEIPT_ID",
    ):
        environment.pop(forbidden, None)
    if case_path == "python/tests/test_adjoint_cyl.py":
        environment[performance.ADJOINT_CYL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(performance.ADJOINT_CYL_FIXED_WORK_ENV, None)
    if case_path == performance.PHASE_IN_MATERIAL_CASE:
        environment[performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV] = "1"
    else:
        environment.pop(performance.PHASE_IN_MATERIAL_FIXED_WORK_ENV, None)
    return environment


def validate_rank_bundle(
    lane: mpi_case.Lane,
    root: pathlib.Path,
    case: dict[str, Any],
    case_command: list[str],
    run_nonce: str,
) -> list[dict[str, Any]]:
    identities = sorted((root / "identity").glob("rank-*.json"))
    statistics = sorted((root / "statistics").glob("rank-*.json"))
    expected_names = [f"rank-{rank:05d}.json" for rank in range(lane.ranks)]
    if [path.name for path in identities] != expected_names or statistics:
        raise WorkloadError("upstream CPU rank evidence inventory differs")
    records = []
    pids = set()
    for rank, identity_path in enumerate(identities):
        identity = mpi_case._load_object(identity_path, "upstream rank identity")
        expected_statistics = root / "statistics" / expected_names[rank]
        expected_work = root / "work" / f"rank-{rank:05d}"
        if (
            set(identity)
            != {
                "schema",
                "rank",
                "world_size",
                "local_rank",
                "pid",
                "executable",
                "command",
                "run_nonce",
                "requested_backend",
                "visible_devices",
                "statistics",
                "work",
            }
            or identity.get("schema") != mpi_case.rank_launcher.SCHEMA
            or identity.get("rank") != rank
            or identity.get("world_size") != lane.ranks
            or identity.get("local_rank") != rank
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or identity.get("executable") != case_command[0]
            or identity.get("command") != case_command
            or identity.get("run_nonce") != run_nonce
            or identity.get("requested_backend") != "cpu"
            or identity.get("visible_devices") != ""
            or identity.get("statistics") != str(expected_statistics)
            or identity.get("work") != str(expected_work)
            or not expected_work.is_dir()
        ):
            raise WorkloadError(f"upstream CPU rank {rank} identity differs")
        pids.add(identity["pid"])
        records.append(
            {
                "rank": rank,
                "identity": file_record(identity_path, root),
                "pid": identity["pid"],
                "statistics_absent": True,
            }
        )
    if len(pids) != lane.ranks:
        raise WorkloadError("upstream CPU rank process identities are not distinct")
    if case.get("kind") == "unittest":
        mpi_case.validate_unittest_mpi_output(lane, root, case)
    return records


def _code_records(paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    return {
        "sample_runner": absolute_file_record(pathlib.Path(__file__).resolve(), "runner"),
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
        "example_oracle": absolute_file_record(
            SCRIPT_DIR / "run_example_oracle.py", "example oracle"
        ),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "validation runner"
        ),
        "receipt_tool": absolute_file_record(RECEIPT_TOOL, "receipt tool"),
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


def _terminal(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "case_path": report["case_path"],
        "sample_kind": report["sample_kind"],
        "cycle_index": report["cycle_index"],
        "receipt_id": report["receipt_id"],
    }


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(output / "PENDING_COMPLETE", _terminal(output, report))
    verify_complete(output, "PENDING_COMPLETE")
    durable_publish_terminal(output / "PENDING_COMPLETE", output / "COMPLETE")


def verify_complete(
    output: pathlib.Path, terminal_name: str = "COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("upstream sample terminal filename differs")
    terminal = auto_case._load_object(output / terminal_name, "upstream COMPLETE")
    report = auto_case._load_object(output / "report.json", "upstream report")
    if (
        not isinstance(report, dict)
        or set(report) != REPORT_KEYS
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or terminal != _terminal(output, report)
    ):
        raise WorkloadError("upstream sample terminal binding differs")
    performance.validate_sample_identity(
        report["sample_kind"], report["cycle_index"]
    )
    raw_paths = report.get("paths")
    if not isinstance(raw_paths, dict):
        raise WorkloadError("upstream sample runtime paths differ")
    namespace = argparse.Namespace(
        repo=pathlib.Path(raw_paths["repo"]),
        manifest=pathlib.Path(raw_paths["manifest"]),
        upstream_receipt=pathlib.Path(raw_paths["upstream_receipt"]),
    )
    receipt, verified = verify_receipt(namespace.repo, namespace.upstream_receipt)
    paths = runtime_paths(namespace, receipt)
    if raw_paths != {name: str(path) for name, path in paths.items()}:
        raise WorkloadError("upstream sample retained paths differ")
    if report["receipt_id"] != verified["receipt_id"]:
        raise WorkloadError("upstream sample live receipt differs")
    if report["receipt"] != absolute_file_record(
        paths["upstream_receipt"], "upstream receipt"
    ):
        raise WorkloadError("upstream sample receipt record differs")
    if report["evidence_code"] != _code_records(paths):
        raise WorkloadError("upstream sample code identity differs")
    case = mpi_case._case_from_manifest(
        paths["repo"], paths["manifest"], report["case_path"]
    )
    performance.validate_inversion_inventory(paths["repo"])
    if case["path"] not in performance.INVERSION_CASES:
        raise WorkloadError("upstream sample retained case differs")
    topology, expected_lane = topology_and_lane(case["path"])
    if report.get("topology") != topology:
        raise WorkloadError("upstream sample topology differs")
    case_command = upstream_case_command(case, paths["python"], paths["repo"])
    root = output / "lane"
    lane = report["lane"]
    if (
        not isinstance(lane, dict)
        or set(lane) != LANE_KEYS
        or lane["spec"] != mpi_case.lane_spec(expected_lane)
    ):
        raise WorkloadError("upstream sample lane differs")
    nonce = lane["run_nonce"]
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(item not in "0123456789abcdef" for item in nonce)
    ):
        raise WorkloadError("upstream sample nonce differs")
    command = mpi_case.lane_command(
        expected_lane,
        mpiexec=paths["mpiexec"],
        python=paths["python"],
        launcher=paths["launcher"],
        evidence_root=root,
        case_command=case_command,
    )
    if lane["command"] != command:
        raise WorkloadError("upstream sample command differs")
    environment = clean_environment(
        paths, nonce, output=output, case_path=case["path"]
    )
    if lane["environment_sha256"] != validation.environment_sha256(environment):
        raise WorkloadError("upstream sample environment differs")
    performance._validate_process(lane["process"])
    records = validate_rank_bundle(
        expected_lane, root, case, case_command, nonce
    )
    if lane["rank_records"] != records:
        raise WorkloadError("upstream sample rank replay differs")
    for name in ("stdout", "stderr"):
        if lane[name] != file_record(root / f"{name}.log", root):
            raise WorkloadError(f"upstream sample {name} changed")
    if (
        lane["process"]["stdout_size_bytes"] != lane["stdout"]["size_bytes"]
        or lane["process"]["stderr_size_bytes"] != lane["stderr"]["size_bytes"]
    ):
        raise WorkloadError("upstream sample output sizes differ")
    runs = auto_case.audit_runs(
        root / "stdout.log", "cpu", upstream_cpu_baseline=True
    )
    if report["timing"] != performance.derive_timing(case["path"], runs):
        raise WorkloadError("upstream sample timing replay differs")
    if report["phase_profile"] != performance.derive_phase_profile(
        runs, lane["process"]["duration_seconds"]
    ):
        raise WorkloadError("upstream sample phase replay differs")
    validate_comparison_evidence(
        (root / "stdout.log").read_text(encoding="utf-8"),
        case["comparison"],
    )
    tree_path = output / "tree-manifest.json"
    if (
        lane["tree_manifest"] != file_record(tree_path, output)
        or auto_case._load_object(tree_path, "upstream sample tree")
        != mpi_case.derive_tree(root)
    ):
        raise WorkloadError("upstream sample raw tree differs")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    performance.validate_sample_identity(args.sample_kind, args.cycle_index)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("upstream sample timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("upstream sample output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        repo = pathlib.Path(os.path.abspath(args.repo)).resolve(strict=True)
        receipt_path = pathlib.Path(os.path.abspath(args.upstream_receipt))
        receipt, verified = verify_receipt(repo, receipt_path)
        paths = runtime_paths(args, receipt)
        if output == paths["repo"] or paths["repo"] in output.parents:
            raise WorkloadError("upstream sample output overlaps source")
        case = mpi_case._case_from_manifest(
            paths["repo"], paths["manifest"], args.case
        )
        performance.validate_inversion_inventory(paths["repo"])
        if case["path"] not in performance.INVERSION_CASES:
            raise WorkloadError("upstream performance case is not sealed")
        topology, selected_lane = topology_and_lane(case["path"])
        root = output / "lane"
        for name in ("statistics", "identity", "work"):
            (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
        nonce = uuid.uuid4().hex
        environment = clean_environment(
            paths, nonce, output=output, case_path=case["path"]
        )
        case_command = upstream_case_command(case, paths["python"], paths["repo"])
        command = mpi_case.lane_command(
            selected_lane,
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
            raise WorkloadError("upstream performance sample process failed")
        validate_comparison_evidence(
            (root / "stdout.log").read_text(encoding="utf-8"),
            case["comparison"],
        )
        runs = auto_case.audit_runs(
            root / "stdout.log", "cpu", upstream_cpu_baseline=True
        )
        timing = performance.derive_timing(case["path"], runs)
        phase_profile = performance.derive_phase_profile(
            runs, process["duration_seconds"]
        )
        rank_records = validate_rank_bundle(
            selected_lane, root, case, case_command, nonce
        )
        atomic_write_json(output / "tree-manifest.json", mpi_case.derive_tree(root))
        ending_receipt, ending_verified = verify_receipt(
            paths["repo"], paths["upstream_receipt"]
        )
        if (
            ending_verified != verified
            or ending_receipt.get("receipt_id") != receipt.get("receipt_id")
        ):
            raise WorkloadError("upstream baseline changed during the sample")
        lane_record = {
            "spec": mpi_case.lane_spec(selected_lane),
            "run_nonce": nonce,
            "command": command,
            "environment_sha256": validation.environment_sha256(environment),
            "process": process,
            "stdout": file_record(root / "stdout.log", root),
            "stderr": file_record(root / "stderr.log", root),
            "rank_records": rank_records,
            "tree_manifest": file_record(output / "tree-manifest.json", output),
        }
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "topology": topology,
            "sample_kind": args.sample_kind,
            "cycle_index": args.cycle_index,
            "receipt_id": verified["receipt_id"],
            "paths": {name: str(path) for name, path in paths.items()},
            "receipt": absolute_file_record(
                paths["upstream_receipt"], "upstream receipt"
            ),
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 upstream performance sample error: {error}", file=sys.stderr)
        raise SystemExit(2)
