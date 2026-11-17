#!/usr/bin/env python3
"""Run and replay one measured sample for typed v1.1.0 inversion evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "gpmeep-v110-inversion-experiment-sample-v9"
COMPLETE_SCHEMA = "gpmeep-v110-inversion-experiment-sample-complete-v9"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]{16,}$")
MAX_LOG_BYTES = 64 * 1024**2
PHASES = (
    "initialization", "fdtd_timestep", "source", "boundary", "dispersive_ade",
    "dft_reduction", "host_device_transfer_and_sync", "mpi", "python_adjoint_mpb", "io",
)
DISPATCH_BACKENDS = ("cpu", "cuda")
CHILD_SCHEMA = "gpmeep-v110-qualified-child-runtime-v10"
POLICY_TELEMETRY_SCHEMA = "gpmeep-v110-backend-recommendation-v3"
POLICY_VERSION = "gpmeep-v110-explicit-backend-guard-v24"
POLICY_RUN_KEYS = {
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
}
M4_REPORT_SCHEMA = "gpmeep-v110-m4-equal-work-matrix-v1"
CHILD_TERMINAL_ENV = "GPMEEP_EXPERIMENT_RUNTIME_TERMINAL"
CHILD_NONCE_ENV = "GPMEEP_EXPERIMENT_NONCE"
ROOT = pathlib.Path(__file__).resolve().parents[1]
CHILD_RELATIVE = pathlib.PurePosixPath(
    "share/gpmeep/libexec/v110-inversion-qualified-child.py"
)
ABLATION_ENVIRONMENTS = {
    "curl-phase-replay=off": {
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
        "MEEP_GPU_DISABLE_CURL_PHASE_REPLAY": "1",
    },
    "curl-phase-replay=on": {
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
        "MEEP_GPU_ENABLE_CURL_PHASE_REPLAY": "1",
    },
    "phase-batching=off": {
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL": "1",
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE": "1",
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH": "1",
    },
    "phase-batching=on": {
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE": "1",
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH": "1",
    },
    "dft-phase-sharing=off": {"MEEP_GPU_DISABLE_DFT_PHASE_SHARING": "1"},
    "dft-phase-sharing=on": {},
    "multi-monitor-dft-batch=off": {"MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH": "1"},
    "multi-monitor-dft-batch=on": {"MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH": "1"},
    "phase-material-mirror-reuse=off": {"MEEP_GPU_DISABLE_PHASE_MATERIAL_MIRROR_REUSE": "1"},
    "phase-material-mirror-reuse=on": {},
    "boundary-span-cache=off": {"MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE": "1"},
    "boundary-span-cache=on": {},
    "boundary-plan-replay=off": {"MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY": "1"},
    "boundary-plan-replay=on": {},
    "boundary-fast-path=off": {
        "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY": "1",
        "MEEP_GPU_DISABLE_BOUNDARY_GRAPH": "1",
        "MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO": "1",
        "MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY": "1",
        "MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY": "1",
    },
    "boundary-fast-path=on": {},
    "device-inventory-snapshot=off": {
        "MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT": "1"
    },
    "device-inventory-snapshot=on": {},
    "backend-ack-scope-cache=off": {
        "MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE": "1"
    },
    "backend-ack-scope-cache=on": {},
    "sparse-lorentzian=off": {"MEEP_GPU_DISABLE_SPARSE_LORENTZIAN_PLAN": "1"},
    "sparse-lorentzian=on": {"MEEP_GPU_ENABLE_SPARSE_LORENTZIAN_PLAN": "1"},
}


class ExperimentError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExperimentError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentError(f"invalid JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_file(), f"{label} is unsafe")
    return resolved


def _direct_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical and resolved.is_dir(), f"{label} is unsafe")
    return resolved


def _file_record(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    path = _direct_file(path, "experiment artifact")
    _require(path.is_relative_to(root), "experiment artifact escaped output")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _absolute_record(path: pathlib.Path) -> dict[str, Any]:
    path = _direct_file(path, "experiment executable")
    _require(path.stat().st_mode & stat.S_IXUSR, "experiment executable is not executable")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _absolute_file_record(path: pathlib.Path, label: str) -> dict[str, Any]:
    path = _direct_file(path, label)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _tree_snapshot(root: pathlib.Path) -> list[dict[str, Any]]:
    root = _direct_directory(root, "evidence tree")
    pending = [root]
    root_status = root.lstat()
    records: list[dict[str, Any]] = [
        {
            "path": ".",
            "kind": "directory",
            "device": root_status.st_dev,
            "inode": root_status.st_ino,
            "mode": root_status.st_mode,
            "links": root_status.st_nlink,
            "uid": root_status.st_uid,
            "gid": root_status.st_gid,
            "size_bytes": root_status.st_size,
            "mtime_ns": root_status.st_mtime_ns,
            "ctime_ns": root_status.st_ctime_ns,
        }
    ]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            status = path.lstat()
            relative = path.relative_to(root).as_posix()
            _require(not stat.S_ISLNK(status.st_mode), "evidence tree contains a symlink")
            if stat.S_ISDIR(status.st_mode):
                kind = "directory"
                pending.append(path)
            else:
                _require(
                    stat.S_ISREG(status.st_mode),
                    "evidence tree contains a non-regular entry",
                )
                kind = "file"
            records.append(
                {
                    "path": relative,
                    "kind": kind,
                    "device": status.st_dev,
                    "inode": status.st_ino,
                    "mode": status.st_mode,
                    "links": status.st_nlink,
                    "uid": status.st_uid,
                    "gid": status.st_gid,
                    "size_bytes": status.st_size,
                    "mtime_ns": status.st_mtime_ns,
                    "ctime_ns": status.st_ctime_ns,
                }
            )
    return sorted(records, key=lambda item: item["path"])


def _isolated_m4_replay(
    controller: pathlib.Path,
    output: pathlib.Path,
    runtime_python: pathlib.Path,
    install_prefix: pathlib.Path,
) -> dict[str, Any]:
    controller = _direct_file(controller, "official M4 controller")
    runtime_python = _direct_file(runtime_python, "official runtime Python")
    install_prefix = _direct_directory(install_prefix, "official runtime prefix")
    bootstrap = """
import importlib.util
import json
import pathlib
import sys

controller_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("gpmeep_isolated_m4_replay", controller_path)
if spec is None or spec.loader is None:
    raise RuntimeError("M4 controller cannot be loaded")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps(module.verify_experiment_authority(output_path), sort_keys=True, separators=(",", ":")))
"""
    try:
        with tempfile.TemporaryDirectory(
            prefix="gpmeep-m4-replay-bytecode-"
        ) as bytecode_cache:
            completed = subprocess.run(
                [
                    str(runtime_python),
                    "-I",
                    "-B",
                    "-X",
                    f"pycache_prefix={bytecode_cache}",
                    "-c",
                    bootstrap,
                    str(controller),
                    str(output),
                ],
                cwd="/",
                env={
                    "PATH": f"{install_prefix / 'bin'}:/usr/bin:/bin",
                    "LD_LIBRARY_PATH": str(install_prefix / "lib"),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONNOUSERSITE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise ExperimentError("isolated M4 authority replay timed out") from error
    _require(
        completed.returncode == 0,
        "isolated M4 authority replay failed: "
        + completed.stderr[:2000].decode(errors="replace"),
    )
    _require(
        len(completed.stdout) <= MAX_LOG_BYTES,
        "isolated M4 authority replay output is too large",
    )
    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=_unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExperimentError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentError("isolated M4 authority replay output is invalid") from error
    _require(isinstance(value, dict), "isolated M4 authority replay is not an object")
    return value


def _load_module(name: str, path: pathlib.Path) -> Any:
    path = _direct_file(path, "module source")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        initial = os.fstat(descriptor)
        named = path.lstat()
        source = bytearray()
        offset = 0
        while True:
            block = os.pread(descriptor, 1024 * 1024, offset)
            if not block:
                break
            source.extend(block)
            offset += len(block)
        ending = os.fstat(descriptor)
        ending_named = path.lstat()
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ExperimentError(f"cannot read exact module source: {path}") from error
    os.close(descriptor)
    _require(
        stat.S_ISREG(initial.st_mode)
        and len(source) == initial.st_size
        and initial.st_size <= 4 * 1024 * 1024
        and (initial.st_dev, initial.st_ino) == (named.st_dev, named.st_ino)
        and (
            initial.st_dev,
            initial.st_ino,
            initial.st_mode,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        )
        == (
            ending.st_dev,
            ending.st_ino,
            ending.st_mode,
            ending.st_size,
            ending.st_mtime_ns,
            ending.st_ctime_ns,
        )
        and (ending.st_dev, ending.st_ino)
        == (ending_named.st_dev, ending_named.st_ino)
        and not stat.S_ISLNK(named.st_mode)
        and not stat.S_ISLNK(ending_named.st_mode),
        f"module source changed while opening: {path}",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        code = compile(bytes(source), str(path), "exec")
        exec(code, module.__dict__)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def _git(repo: pathlib.Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    _require(completed.returncode == 0, "experiment source Git authority differs")
    return completed.stdout


def _source_authority(
    repo: pathlib.Path, source_commit: str, workload_id: str
) -> tuple[pathlib.Path, dict[str, Any]]:
    repo = _direct_directory(repo, "experiment source repository")
    _require((repo / ".git").exists(), "experiment source is not a Git worktree")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is not None
        and _git(repo, "rev-parse", "HEAD").decode().strip() == source_commit
        and _git(repo, "status", "--porcelain", "--untracked-files=all") == b"",
        "experiment source commit is not exact and clean",
    )
    relative = pathlib.PurePosixPath(workload_id)
    _require(
        workload_id not in {"", "."}
        and not relative.is_absolute()
        and ".." not in relative.parts,
        "experiment workload ID is invalid",
    )
    workload = _direct_file(repo.joinpath(*relative.parts), "experiment workload")
    _require(
        _git(repo, "ls-files", "--error-unmatch", "--", workload_id).decode().strip()
        == workload_id
        and _git(repo, "show", f"{source_commit}:{workload_id}") == workload.read_bytes(),
        "experiment workload is not the committed source byte stream",
    )
    return workload, {
        "source_commit": source_commit,
        "repo": str(repo),
        "workload": _absolute_file_record(workload, "experiment workload"),
    }


def _owned_child(
    prefix: pathlib.Path,
    report: dict[str, Any],
    source_child: pathlib.Path,
    package_payload: dict[str, Any],
) -> pathlib.Path:
    child = _direct_file(prefix.joinpath(*CHILD_RELATIVE.parts), "qualified child")
    _require(child.read_bytes() == source_child.read_bytes(), "qualified child differs from source")
    conda_record = _load(pathlib.Path(report["conda_record"]["path"]))
    paths = conda_record.get("paths_data", {}).get("paths")
    _require(isinstance(paths, list), "installed package ownership inventory is absent")
    matches = [item for item in paths if isinstance(item, dict) and item.get("_path") == str(CHILD_RELATIVE)]
    _require(len(matches) == 1, "qualified child is not uniquely package-owned")
    item = matches[0]
    expected_sha = item.get("sha256_in_prefix") or item.get("sha256")
    _require(
        item.get("path_type") == "hardlink"
        and expected_sha == _sha256(child)
        and type(item.get("size_in_bytes")) is int
        and item["size_in_bytes"] == child.stat().st_size,
        "qualified child package ownership differs",
    )
    archived = package_payload.get(str(CHILD_RELATIVE))
    _require(
        archived is not None
        and archived.member_type == "regular"
        and archived.sha256 == _sha256(child)
        and archived.size_bytes == child.stat().st_size,
        "qualified child differs from the exact package archive",
    )
    return child


def _installed_archive_authority(
    *,
    repo: pathlib.Path,
    package: pathlib.Path,
    prefix: pathlib.Path,
    report: dict[str, Any],
    runtime: dict[str, pathlib.Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inspector = _load_module(
        "gpmeep_v110_inversion_conda_inspector",
        repo / "scripts/v110-final-release-receipt.py",
    )
    try:
        index, payload, install_semantics = inspector._conda_payload_files(
            package, "M4 measured v1.1.0 package"
        )
    except Exception as error:
        raise ExperimentError(f"exact package is not a genuine Conda archive: {error}") from error
    expected_build = {
        "development": "cuda124_mpi_openmpi_py311_0_development",
        "candidate": "cuda124_mpi_openmpi_py311_0",
    }[report["v110_candidate_gate"]["build_mode"]]
    _require(index.get("build") == expected_build, "Conda archive build mode differs")
    record = _load(pathlib.Path(report["conda_record"]["path"]))
    entries = record.get("paths_data", {}).get("paths")
    _require(isinstance(entries, list), "installed Conda paths_data is absent")
    installed: dict[str, dict[str, Any]] = {}
    for item in entries:
        _require(
            isinstance(item, dict)
            and isinstance(item.get("_path"), str)
            and item["_path"] not in installed,
            "installed Conda ownership entry differs",
        )
        name = item["_path"]
        expected = install_semantics.get(name)
        _require(expected is not None, "installed path is absent from the Conda archive")
        member = payload.get(name)
        _require(member is not None, "installed path is absent from the package payload")
        installed_path = prefix.joinpath(*pathlib.PurePosixPath(name).parts)
        try:
            _require(
                installed_path.parent.resolve(strict=True) == installed_path.parent,
                f"installed path contains a symlink ancestor: {name}",
            )
            installed_metadata = installed_path.lstat()
        except OSError as error:
            raise ExperimentError(f"installed path is unavailable: {name}") from error
        _require(
            stat.S_IMODE(installed_metadata.st_mode) == member.mode,
            f"installed path mode differs from exact archive: {name}",
        )
        if expected["path_type"] == "hardlink":
            _require(
                item.get("path_type") == "hardlink"
                and (item.get("sha256_in_prefix") or item.get("sha256"))
                == expected["sha256"]
                and item.get("size_in_bytes") == expected["size_in_bytes"],
                f"installed Conda ownership bytes differ: {name}",
            )
            _require(
                stat.S_ISREG(installed_metadata.st_mode)
                and not stat.S_ISLNK(installed_metadata.st_mode)
                and installed_metadata.st_size == expected["size_in_bytes"]
                and _sha256(installed_path) == expected["sha256"],
                f"installed bytes differ from exact archive: {name}",
            )
        else:
            _require(
                item.get("path_type") == "softlink"
                and item.get("sha256_in_prefix") == expected["target_sha256"]
                and item.get("size_in_bytes") == expected["target_size_in_bytes"],
                f"installed Conda ownership link bytes differ: {name}",
            )
            _require(
                stat.S_ISLNK(installed_metadata.st_mode),
                f"installed Conda ownership link type differs: {name}",
            )
            try:
                target_text = os.readlink(installed_path)
                normalized_target = inspector._safe_link_target(
                    name,
                    target_text,
                    f"installed softlink {name}",
                    hardlink=False,
                )
                target_path = installed_path.resolve(strict=True)
                target_path.relative_to(prefix)
                target_metadata = target_path.stat()
            except Exception as error:
                raise ExperimentError(
                    f"installed softlink is unsafe: {name}"
                ) from error
            _require(
                normalized_target == expected["link_target"],
                f"installed softlink target differs from exact archive: {name}",
            )
            _require(
                stat.S_ISREG(target_metadata.st_mode)
                and target_metadata.st_size == expected["target_size_in_bytes"]
                and _sha256(target_path) == expected["target_sha256"],
                f"installed softlink target bytes differ from exact archive: {name}",
            )
        installed[name] = item
    ownership = report.get("ownership", {})
    _require(
        set(installed) == set(install_semantics)
        and ownership.get("owned_paths") == len(installed)
        and ownership.get("regular_files", 0)
        + ownership.get("symlinks", 0)
        == len(installed)
        and ownership.get("content_hashes_verified") == len(installed)
        and ownership.get("content_sizes_verified") == len(installed)
        and ownership.get("prefix_rewritten_sizes") == 0,
        "installed and archived Conda ownership inventories differ",
    )
    archive_paths = {
        "module": report["runtime"]["module"],
        "extension": report["runtime"]["extension"],
        "libmeep": report["runtime"]["libmeep"],
        "release_manifest": report["manifest"]["file"],
    }
    for label, retained in archive_paths.items():
        path = pathlib.Path(retained["path"])
        try:
            relative = path.relative_to(prefix).as_posix()
        except ValueError as error:
            raise ExperimentError(f"{label} escaped installed prefix") from error
        member = payload.get(relative)
        _require(
            member is not None
            and member.member_type == "regular"
            and member.sha256 == retained["sha256"]
            and member.size_bytes == retained["size_bytes"],
            f"{label} differs from exact package archive",
        )
    critical = {
        "verifier": (
            "share/gpmeep/libexec/gpmeep-verify-provenance.py",
            "scripts/verify-gpmeep-package.py",
        ),
        "launcher": (
            "bin/gpmeep-verify-provenance",
            "scripts/gpmeep-verify-provenance-launcher.sh",
        ),
    }
    critical_records: dict[str, Any] = {}
    for label, (relative, source_relative) in critical.items():
        member = payload.get(relative)
        installed_path = _direct_file(prefix / relative, f"installed {label}")
        source_path = _direct_file(
            repo / source_relative, f"source {label}"
        )
        _require(
            member is not None
            and member.member_type == "regular"
            and member.data == installed_path.read_bytes(),
            f"installed {label} differs from exact package archive",
        )
        _require(
            member.data == source_path.read_bytes(),
            f"installed {label} differs from the exact source commit",
        )
        critical_records[f"installed_{label}"] = _absolute_file_record(
            installed_path, f"installed {label}"
        )
        critical_records[f"source_{label}"] = _absolute_file_record(
            source_path, f"source {label}"
        )
    return (
        {
            "format": "conda-v2",
            "name": index["name"],
            "version": index["version"],
            "build": index["build"],
            "subdir": index["subdir"],
            "payload_path_count": len(payload),
            "install_semantics_sha256": _canonical_sha256(install_semantics),
            **critical_records,
        },
        payload,
        install_semantics,
    )


def _m4_authority(
    *,
    repo: pathlib.Path,
    m4_output: pathlib.Path,
    source_commit: str,
    package: pathlib.Path,
    provenance: pathlib.Path,
    report: dict[str, Any],
    runtime: dict[str, pathlib.Path],
    live: dict[str, Any],
    archive: dict[str, Any],
) -> dict[str, Any]:
    output = _direct_directory(m4_output, "official M4 output")
    terminal = _direct_file(output / "M4_COMPLETE", "official M4 terminal")
    terminal_record = _absolute_file_record(terminal, "official M4 terminal")
    controller = _direct_file(
        repo / "scripts/python-validation/run_m4_equal_work_matrix.py",
        "official M4 controller",
    )
    tree_before = _tree_snapshot(output)
    try:
        m4_report = _isolated_m4_replay(
            controller, output, runtime["python"], runtime["install_prefix"]
        )
    except Exception as error:
        raise ExperimentError(f"official M4 package authority replay failed: {error}") from error
    _require(
        tree_before == _tree_snapshot(output)
        and terminal_record == _absolute_file_record(terminal, "official M4 terminal"),
        "official M4 evidence tree changed during replay",
    )
    _require(
        isinstance(m4_report, dict)
        and m4_report.get("schema") == M4_REPORT_SCHEMA
        and m4_report.get("outcome") == "PASS",
        "official M4 terminal is not a sealed COMPLETE result",
    )
    contract = m4_report.get("contract")
    _require(isinstance(contract, dict), "official M4 package authority is absent")
    paths = contract.get("paths")
    inputs = contract.get("runtime_inputs")
    _require(isinstance(paths, dict) and isinstance(inputs, dict), "official M4 runtime authority differs")
    expected_runtime = {
        name: str(runtime[name])
        for name in (
            "python", "build_python", "install_prefix", "build_receipt", "mpiexec",
            "package", "provenance_report", "verifier",
        )
    }
    _require(
        pathlib.Path(paths.get("repo", "")) == repo
        and pathlib.Path(paths.get("v110_package", "")) == package
        and pathlib.Path(paths.get("v110_provenance_report", "")) == provenance
        and inputs.get("v110_package") == _absolute_file_record(package, "M4 package")
        and inputs.get("v110_provenance_report")
        == _absolute_file_record(provenance, "M4 provenance")
        and inputs.get("v110_package_sha256") == report["package_sha256"]
        and inputs.get("v110_installed_report_id") == report["report_id"]
        and inputs.get("v110_candidate_receipt_id")
        == report["v110_candidate_gate"]["receipt_id"]
        and inputs.get("v110_installed_runtime") == expected_runtime
        and inputs.get("v110_live_verification") == live
        and inputs.get("v110_archive_verification") == archive
        and inputs.get("v110_source_identity", {}).get("git_head") == source_commit,
        "experiment runtime differs from the final M4 package authority",
    )
    return {
        "output": str(output),
        "terminal": terminal_record,
        "contract_sha256": _canonical_sha256(contract),
        "package_sha256": report["package_sha256"],
        "installed_report_id": report["report_id"],
        "candidate_receipt_id": report["v110_candidate_gate"]["receipt_id"],
        "installed_runtime": expected_runtime,
        "live_verification": live,
        "archive_verification": archive,
    }


def _runtime_authority(
    *,
    repo: pathlib.Path,
    source_commit: str,
    runtime_provenance: pathlib.Path,
    package: pathlib.Path,
    install_prefix: pathlib.Path,
    m4_output: pathlib.Path,
    workload_id: str,
) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path, pathlib.Path]:
    workload, source = _source_authority(repo, source_commit, workload_id)
    identity = _load_module(
        "gpmeep_v110_inversion_runtime_identity",
        repo / "scripts/python-validation/v110_runtime_identity.py",
    )
    try:
        report, installed = identity.validate_report(
            runtime_provenance, package, install_prefix, source_commit
        )
        runtime = identity.runtime_paths(report, installed)
        archive, package_payload, _install_semantics = _installed_archive_authority(
            repo=repo,
            package=runtime["package"],
            prefix=runtime["install_prefix"],
            report=report,
            runtime=runtime,
        )
        live = identity.live_verify(report, installed)
        m4 = _m4_authority(
            repo=repo,
            m4_output=m4_output,
            source_commit=source_commit,
            package=runtime["package"],
            provenance=runtime["provenance_report"],
            report=report,
            runtime=runtime,
            live=live,
            archive=archive,
        )
    except Exception as error:
        raise ExperimentError(f"installed runtime authority differs: {error}") from error
    source_child = _direct_file(
        repo / "scripts/v110-inversion-qualified-child.py", "qualified child source"
    )
    _require(
        _git(repo, "show", f"{source_commit}:scripts/v110-inversion-qualified-child.py")
        == source_child.read_bytes(),
        "qualified child source is not committed",
    )
    child = _owned_child(
        installed["install_prefix"], report, source_child, package_payload
    )
    child_module = _load_module("gpmeep_v110_qualified_workload_contract", source_child)
    workload_contract = child_module.workload_contract(workload, workload_id)
    for support in workload_contract["support_files"]:
        path = _direct_file(pathlib.Path(support["path"]), "workload support")
        relative = path.relative_to(repo).as_posix()
        _require(_git(repo, "show", f"{source_commit}:{relative}") == path.read_bytes(),
                 "workload support is not the exact committed source")
    authority = {
        **source,
        "runtime_receipt_id": report["report_id"],
        "runtime_provenance": _absolute_file_record(runtime_provenance, "runtime provenance"),
        "package": _absolute_file_record(package, "runtime package"),
        "install_prefix": str(installed["install_prefix"]),
        "runtime_python": _absolute_file_record(runtime["python"], "runtime Python"),
        "qualified_child": _absolute_file_record(child, "qualified child"),
        "qualified_child_source": _absolute_file_record(source_child, "qualified child source"),
        "workload_contract": workload_contract,
        "package_archive": archive,
        "live_verification": live,
        "m4_authority": m4,
        "runtime_files": {
            name: report["runtime"][name]
            for name in ("python", "module", "extension", "libmeep")
        },
    }
    return authority, runtime["python"], child, workload


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    _require(destination.parent.is_dir(), "experiment output parent is absent")
    _require(not destination.exists() and not destination.is_symlink(), "refusing to overwrite experiment artifact")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _environment(
    topology: str, gpu_devices: list[str], label: str, install_prefix: pathlib.Path
) -> dict[str, str]:
    selected = "" if topology.startswith("cpu") else ",".join(gpu_devices)
    profiles = {
        "baseline": {},
        "cuda-launch-blocking": {"CUDA_LAUNCH_BLOCKING": "1"},
    }
    _require(label in profiles, "experiment environment profile is not allowlisted")
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ExperimentError("experiment account home is unavailable") from error
    return {
        "PATH": f"{install_prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(install_prefix / "lib"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(account_home),
        "MPLBACKEND": "Agg",
        "GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK": "1",
        "GPMEEP_ADJOINT_CYL_FIXED_WORK": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": selected,
        "MEEP_GPU_BACKEND": "cpu" if topology.startswith("cpu") else "cuda",
        "GPMEEP_EXPERIMENT_ENVIRONMENT": label,
        **profiles[label],
    }


def _phase_profile(values: Any) -> dict[str, Any]:
    _require(
        isinstance(values, dict)
        and set(values) == set(PHASES)
        and all(
            isinstance(value, dict)
            and set(value) == {"available", "seconds"}
            and type(value["available"]) is bool
            and (
                value["seconds"] is None
                if not value["available"]
                else type(value["seconds"]) in {int, float}
                and math.isfinite(value["seconds"])
                and value["seconds"] >= 0
            )
            for value in values.values()
        ),
        "experiment phase profile differs",
    )
    return {
        "phases": {
            name: {
                "available": values[name]["available"],
                "seconds": (
                    float(values[name]["seconds"])
                    if values[name]["available"]
                    else None
                ),
            }
            for name in PHASES
        }
    }


def _backend_policy_runs(
    values: Any, *, run_count: int, expected_backend: str
) -> list[dict[str, Any]]:
    _require(
        isinstance(values, list) and len(values) == run_count,
        "experiment child backend policy/run count differs",
    )
    for record in values:
        _require(
            isinstance(record, dict)
            and set(record) == POLICY_RUN_KEYS
            and record["schema"] == POLICY_TELEMETRY_SCHEMA
            and record["recommendation_policy_version"] == POLICY_VERSION
            and record["selected_backend"] == expected_backend
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
            and HEX64.fullmatch(record["acknowledgement_scope_digest"])
            and HEX64.fullmatch(record["mpi_topology_digest"]),
            "experiment child backend policy telemetry differs",
        )
        if record["acknowledgement"]:
            _require(
                record["acknowledgement_allowed"] is True
                and len(record["acknowledgement_reason"].encode("utf-8")) >= 8
                and record["acknowledgement_scope"] == "process"
                and HEX64.fullmatch(
                    record["acknowledgement_origin_cohort_digest"]
                ),
                "experiment child accepted acknowledgement differs",
            )
            if record["acknowledgement_reused"]:
                _require(
                    record["acknowledgement_scope_cache_enabled"] is True
                    and record["workload_fingerprint_evaluated"] is False
                    and record["workload_fingerprint"] == ""
                    and record["acknowledgement_digest"] == "",
                    "experiment child reused acknowledgement differs",
                )
            else:
                _require(
                    record["workload_fingerprint_evaluated"] is True
                    and HEX64.fullmatch(record["workload_fingerprint"])
                    and HEX64.fullmatch(record["acknowledgement_digest"])
                    and record["acknowledgement_origin_cohort_digest"]
                    == record["acknowledgement_digest"],
                    "experiment child exact acknowledgement differs",
                )
        else:
            _require(
                record["acknowledgement_reused"] is False
                and record["acknowledgement_reason"] == ""
                and record["acknowledgement_scope"] == ""
                and record["acknowledgement_origin_cohort_digest"] == "",
                "experiment child unacknowledged telemetry differs",
            )
    exact_origins: dict[tuple[Any, ...], set[str]] = {}
    for record in values:
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
                "experiment child reused acknowledgement lineage differs",
            )
        else:
            exact_origins.setdefault(cohort, set()).add(
                record["acknowledgement_digest"]
            )
    return values


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


def _child_runtime(
    path: pathlib.Path,
    *,
    nonce: str,
    topology: str,
    runtime_receipt_id: str,
    gpu_devices: list[str],
    authority: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    path = _direct_file(path, "experiment child runtime terminal")
    child = _load(path)
    keys = {
        "schema", "outcome", "nonce", "runtime_receipt_id", "requested_backend",
        "active_backend", "selected_physical_gpu_uuids", "mpi_ranks", "rank_records",
        "phase_profile", "timing", "backend_policy_runs", "feature_counters",
        "runtime_files", "workload", "experiment", "phase_numerical", "terminal_id",
        "run_observations", "workload_statistics",
    }
    _require(set(child) == keys, "experiment child runtime terminal schema differs")
    unsigned = dict(child)
    terminal_id = unsigned.pop("terminal_id", None)
    expected_backend = "cpu" if topology.startswith("cpu") else "cuda"
    expected_ranks = int(topology.removeprefix(expected_backend))
    _require(
        child["schema"] == CHILD_SCHEMA
        and child["outcome"] == "PASS"
        and child["nonce"] == nonce
        and child["runtime_receipt_id"] == runtime_receipt_id
        and child["requested_backend"] == expected_backend
        and child["active_backend"] == expected_backend
        and child["selected_physical_gpu_uuids"] == gpu_devices
        and child["mpi_ranks"] == expected_ranks
        and child["runtime_files"] == authority["runtime_files"]
        and child["workload"]
        == {
            "id": authority["workload"]["path"].removeprefix(
                authority["repo"] + "/"
            ),
            **authority["workload"],
            "contract": authority["workload_contract"],
        }
        and child["experiment"] == experiment
        and terminal_id == _canonical_sha256(unsigned),
        "experiment child runtime identity differs",
    )
    ranks = child["rank_records"]
    _require(
        isinstance(ranks, list)
        and len(ranks) == expected_ranks
        and [record.get("rank") for record in ranks if isinstance(record, dict)]
        == list(range(expected_ranks)),
        "experiment child rank inventory differs",
    )
    observed_devices: list[str] = []
    other_backend = "cuda" if expected_backend == "cpu" else "cpu"
    for record in ranks:
        _require(
            set(record) == {
                "rank", "runtime_receipt_id", "requested_backend", "active_backend",
                "physical_gpu_uuid", "dispatch_counters",
            }
            and record["runtime_receipt_id"] == runtime_receipt_id
            and record["requested_backend"] == expected_backend
            and record["active_backend"] == expected_backend,
            "experiment child rank runtime identity differs",
        )
        expected_device = None
        if expected_backend == "cuda":
            expected_device = gpu_devices[record["rank"] % len(gpu_devices)]
            observed_devices.append(record["physical_gpu_uuid"])
        _require(
            record["physical_gpu_uuid"] == expected_device,
            "experiment child physical GPU UUID differs",
        )
        counters = record["dispatch_counters"]
        _require(
            isinstance(counters, dict) and set(counters) == set(DISPATCH_BACKENDS),
            "experiment child dispatch schema differs",
        )
        for backend in DISPATCH_BACKENDS:
            value = counters[backend]
            _require(
                isinstance(value, dict)
                and set(value) == {"calls", "points"}
                and type(value["calls"]) is int
                and type(value["points"]) is int
                and value["calls"] >= 0
                and value["points"] >= 0
                and ((value["calls"] == 0) == (value["points"] == 0)),
                "experiment child dispatch counters differ",
            )
        _require(
            counters[expected_backend]["calls"] > 0
            and counters[expected_backend]["points"] > 0,
            "experiment child lacks requested-backend work",
        )
        _require(
            counters[other_backend] == {"calls": 0, "points": 0},
            "experiment child used forbidden backend fallback",
        )
    _require(
        sorted(set(observed_devices)) == sorted(gpu_devices),
        "experiment child did not exercise every selected physical GPU",
    )
    child["phase_profile"] = _phase_profile(child["phase_profile"])
    timing = child["timing"]
    _require(
        isinstance(timing, dict)
        and set(timing) == {"contract", "primary_seconds", "run_count"}
        and timing["contract"] == "sum-all-simulation-run-wall-seconds"
        and type(timing["primary_seconds"]) in {int, float}
        and math.isfinite(timing["primary_seconds"])
        and timing["primary_seconds"] > 0
        and type(timing["run_count"]) is int
        and timing["run_count"] > 0
        and child["phase_profile"]["phases"]["fdtd_timestep"]
        == {"available": True, "seconds": timing["primary_seconds"]},
        "experiment child workload timing differs",
    )
    _validate_run_observations(child["run_observations"], timing,
                               authority["workload_contract"]["observed_method"],
                               child["workload_statistics"])
    _require(len(ranks) == 1 and ranks[0]["dispatch_counters"]
             == _workload_statistics_dispatch(child["workload_statistics"]),
             "experiment workload statistics and dispatch differ")
    _validate_boundary_replay_control(experiment["ablation"], expected_backend,
                                      child["run_observations"])
    _validate_boundary_span_cache_control(experiment["ablation"], expected_backend,
                                          child["run_observations"])
    _validate_material_phasing_control(experiment["ablation"], expected_backend,
                                       child["run_observations"])
    child["backend_policy_runs"] = _backend_policy_runs(
        child["backend_policy_runs"],
        run_count=timing["run_count"],
        expected_backend=expected_backend,
    )
    features = child["feature_counters"]
    _require(
        isinstance(features, dict)
        and set(features)
        == {
            "curl_phase_replay",
            "phase_curl_forced_batches",
            "device_inventory_probe",
            "backend_mismatch_acknowledgements",
        }
        and type(features["phase_curl_forced_batches"]) is int
        and features["phase_curl_forced_batches"] >= 0
        and type(features["backend_mismatch_acknowledgements"]) is int
        and features["backend_mismatch_acknowledgements"] >= 0
        and isinstance(features["curl_phase_replay"], dict)
        and set(features["curl_phase_replay"])
        == {"checks", "hits", "unready", "generation_misses", "mirror_misses"}
        and all(
            type(value) is int and value >= 0
            for value in features["curl_phase_replay"].values()
        ),
        "experiment child curl phase replay counters differ",
    )
    _validate_workload_feature_projection(child["workload_statistics"], features)
    replay = features["curl_phase_replay"]
    _require(
        replay["checks"]
        == replay["hits"]
        + replay["unready"]
        + replay["generation_misses"]
        + replay["mirror_misses"],
        "experiment child curl phase replay counter conservation differs",
    )
    inventory_probe = features["device_inventory_probe"]
    if experiment["ablation"] in {
        "device-inventory-snapshot=off",
        "device-inventory-snapshot=on",
    }:
        _require(
            isinstance(inventory_probe, dict)
            and set(inventory_probe)
            == {
                "device_count",
                "inventory_sha256",
                "runtime_availability_probes",
                "runtime_device_enumerations",
            }
            and all(
                type(inventory_probe[name]) is int and inventory_probe[name] >= 0
                for name in (
                    "device_count",
                    "runtime_availability_probes",
                    "runtime_device_enumerations",
                )
            )
            and HEX64.fullmatch(inventory_probe["inventory_sha256"] or "")
            is not None
            and inventory_probe["device_count"] > 0
            and inventory_probe["runtime_availability_probes"] == 1,
            "experiment child device inventory probe differs",
        )
        expected_enumerations = (
            1
            if experiment["ablation"].endswith("=on")
            else 1 + 10 * inventory_probe["device_count"]
        )
        _require(
            inventory_probe["runtime_device_enumerations"]
            == expected_enumerations,
            "experiment child device inventory ablation invariant differs",
        )
    else:
        _require(
            inventory_probe is None,
            "unexpected experiment child device inventory probe",
        )
    if experiment["ablation"] == "curl-phase-replay=off":
        _require(
            features["phase_curl_forced_batches"] > 0
            and all(value == 0 for value in replay.values()),
            "experiment child curl phase replay OFF invariant differs",
        )
    elif experiment["ablation"] == "curl-phase-replay=on":
        _require(
            features["phase_curl_forced_batches"] > 0
            and replay["checks"] > 0
            and replay["hits"] > 0,
            "experiment child curl phase replay ON invariant differs",
        )
    if child["workload"]["id"] == "python/examples/phase_in_material.py":
        support = authority["workload_contract"]["support_files"]
        _require(len(support) == 1, "phase numerical helper inventory differs")
        helper_path = _direct_file(pathlib.Path(support[0]["path"]), "phase numerical helper")
        _require(helper_path.name == "phase_material_contract.py"
                 and _absolute_file_record(helper_path, "phase numerical helper") == support[0],
                 "phase numerical helper binding differs")
        helper = _load_module("gpmeep_inversion_phase_numerical", helper_path)
        try:
            expected_work = helper.work_contract(
                experiment["sweep_size"], experiment["sweep_run_length"], fixed_work=True
            )
            _require(timing["run_count"] == expected_work["run_count"], "phase numerical run count differs")
            artifact = _direct_file(path.parent / "phase-numerical.npy", "phase numerical artifact")
            helper.verify_numerical(child["phase_numerical"], artifact, expected_work)
            _require(child["phase_numerical"]["artifact"] == _file_record(artifact, path.parent),
                     "phase numerical artifact reference differs")
        except (ValueError, TypeError, KeyError, OSError) as error:
            raise ExperimentError(f"phase numerical evidence differs: {error}") from error
    else:
        _require(child["phase_numerical"] is None, "unexpected phase numerical evidence")
    return child


def _experiment_arguments(values: list[str]) -> tuple[list[str], dict[str, Any]]:
    _require(all(isinstance(item, str) for item in values), "experiment arguments differ")
    if not values:
        return [], {
            "sweep_size": 1.0,
            "sweep_run_length": 1.0,
            "ablation": None,
            "ablation_environment": {},
        }
    if len(values) == 2 and values[0] == "--gpmeep-ablation":
        setting = values[1]
        _require(
            re.fullmatch(
                r"(?:curl-phase-replay|phase-batching|dft-phase-sharing|multi-monitor-dft-batch|boundary-fast-path|boundary-plan-replay|boundary-span-cache|phase-material-mirror-reuse|device-inventory-snapshot|backend-ack-scope-cache|sparse-lorentzian)=(?:off|on)",
                setting,
            )
            is not None,
            "experiment ablation is not allowlisted",
        )
        return list(values), {
            "sweep_size": 1.0,
            "sweep_run_length": 1.0,
            "ablation": setting,
            "ablation_environment": ABLATION_ENVIRONMENTS[setting],
        }
    _require(
        len(values) == 4
        and values[0] == "--gpmeep-sweep-size"
        and values[2] == "--gpmeep-sweep-run-length",
        "experiment arguments are not a canonical scaling or ablation request",
    )
    try:
        size = float(values[1])
        run_length = float(values[3])
    except ValueError as error:
        raise ExperimentError("experiment scaling coordinate differs") from error
    _require(
        math.isfinite(size) and size > 0 and math.isfinite(run_length) and run_length > 0,
        "experiment scaling coordinate differs",
    )
    return list(values), {
        "sweep_size": size,
        "sweep_run_length": run_length,
        "ablation": None,
        "ablation_environment": {},
    }


def _validate_identity(
    workload_id: str,
    cycle_index: int,
    topology: str,
    runtime_receipt_id: str,
    gpu_devices: list[str],
    environment_label: str,
) -> None:
    _require(
        isinstance(workload_id, str)
        and workload_id
        and not pathlib.PurePosixPath(workload_id).is_absolute()
        and ".." not in pathlib.PurePosixPath(workload_id).parts,
        "experiment workload ID is invalid",
    )
    _require(cycle_index in {1, 2}, "experiment cycle must be measured cycle 1 or 2")
    _require(topology in {"cpu1", "cuda1"}, "experiment producer supports one process/device")
    _require(HEX64.fullmatch(runtime_receipt_id or "") is not None, "experiment runtime receipt ID differs")
    expected_devices = 0 if topology.startswith("cpu") else int(topology[-1])
    _require(
        len(gpu_devices) == expected_devices
        and len(set(gpu_devices)) == len(gpu_devices)
        and all(GPU_UUID.fullmatch(item) is not None for item in gpu_devices),
        "experiment GPU identity differs",
    )
    _require(
        re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", environment_label or "") is not None,
        "experiment environment label differs",
    )


def create(
    *,
    output: pathlib.Path,
    repo: pathlib.Path,
    source_commit: str,
    runtime_provenance: pathlib.Path,
    package: pathlib.Path,
    install_prefix: pathlib.Path,
    m4_output: pathlib.Path,
    workload_id: str,
    cycle_index: int,
    topology: str,
    gpu_devices: list[str],
    environment_label: str,
    experiment_arguments: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    authority, runtime_python, qualified_child, workload = _runtime_authority(
        repo=repo,
        source_commit=source_commit,
        runtime_provenance=runtime_provenance,
        package=package,
        install_prefix=install_prefix,
        m4_output=m4_output,
        workload_id=workload_id,
    )
    runtime_receipt_id = authority["runtime_receipt_id"]
    _validate_identity(
        workload_id,
        cycle_index,
        topology,
        runtime_receipt_id,
        gpu_devices,
        environment_label,
    )
    _require(
        isinstance(timeout_seconds, int) and 1 <= timeout_seconds <= 48 * 3600,
        "experiment timeout differs",
    )
    experiment_arguments, expected_experiment = _experiment_arguments(
        experiment_arguments
    )
    command = [
        str(runtime_python),
        str(qualified_child),
        "--runtime-provenance",
        str(_direct_file(runtime_provenance, "runtime provenance")),
        "--workload",
        str(workload),
        "--workload-id",
        workload_id,
        *experiment_arguments,
    ]
    executable = runtime_python
    output = output.absolute()
    _require(not output.exists() and not output.is_symlink(), "experiment output already exists")
    output.mkdir(mode=0o700)
    work_directory = output / "workload"
    work_directory.mkdir(mode=0o700)
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    child_terminal_path = output / "child-runtime.json"
    child_nonce = os.urandom(32).hex()
    environment = _environment(
        topology, gpu_devices, environment_label, pathlib.Path(authority["install_prefix"])
    )
    execution_environment = {
        **environment,
        CHILD_TERMINAL_ENV: str(child_terminal_path),
        CHILD_NONCE_ENV: child_nonce,
    }
    executable_record = _absolute_record(executable)
    started = time.monotonic_ns()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=work_directory,
                env=execution_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ExperimentError("experiment command timed out") from error
    ended = time.monotonic_ns()
    process_seconds = (ended - started) / 1_000_000_000.0
    _require(completed.returncode == 0, "experiment command failed")
    _require(
        stdout_path.stat().st_size <= MAX_LOG_BYTES
        and stderr_path.stat().st_size <= MAX_LOG_BYTES,
        "experiment log exceeds its bound",
    )
    _require(
        executable_record == _absolute_record(executable),
        "experiment executable changed during execution",
    )
    child_runtime = _child_runtime(
        child_terminal_path,
        nonce=child_nonce,
        topology=topology,
        runtime_receipt_id=runtime_receipt_id,
        gpu_devices=gpu_devices,
        authority=authority,
        experiment=expected_experiment,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "outcome": "PASS",
        "case_path": workload_id,
        "topology": topology,
        "sample_kind": "measured",
        "cycle_index": cycle_index,
        "receipt_id": runtime_receipt_id,
        "gpu_devices": gpu_devices,
        "runtime_snapshot": {
            "runtime_contract": {"receipt_id": child_runtime["runtime_receipt_id"]},
            "child_runtime_terminal_id": child_runtime["terminal_id"],
            "backend_policy_runs": child_runtime["backend_policy_runs"],
            "run_observations": child_runtime["run_observations"],
            "workload_statistics": child_runtime["workload_statistics"],
            "feature_counters": child_runtime["feature_counters"],
            "phase_numerical": child_runtime["phase_numerical"],
        },
        "lane": {
            "spec": {"devices": gpu_devices},
            "command": command,
            "environment_sha256": _canonical_sha256(environment),
        },
        "timing": {
            **child_runtime["timing"],
            "process_seconds": process_seconds,
        },
        "phase_profile": child_runtime["phase_profile"],
        "environment_label": environment_label,
        "environment": environment,
        "runtime_authority": authority,
        "executable": executable_record,
        "child_nonce": child_nonce,
        "child_runtime": _file_record(child_terminal_path, output),
        "process": {"returncode": completed.returncode, "timed_out": False},
        "stdout": _file_record(stdout_path, output),
        "stderr": _file_record(stderr_path, output),
    }
    report["report_id"] = _canonical_sha256(report)
    _write_new(output / "report.json", report)
    terminal: dict[str, Any] = {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": _file_record(output / "report.json", output),
        "report_id": report["report_id"],
    }
    terminal["terminal_id"] = _canonical_sha256(terminal)
    _write_new(output / "COMPLETE", terminal)
    return verify_complete(output)


def verify_complete(output: pathlib.Path) -> dict[str, Any]:
    output = output.resolve(strict=True)
    _require(output.is_dir(), "experiment output is not a directory")
    report = _load(_direct_file(output / "report.json", "experiment report"))
    terminal = _load(_direct_file(output / "COMPLETE", "experiment terminal"))
    report_keys = {
        "schema", "outcome", "case_path", "topology", "sample_kind", "cycle_index",
        "receipt_id", "gpu_devices", "runtime_snapshot", "lane", "timing",
        "phase_profile", "environment_label", "environment", "executable", "process",
        "runtime_authority", "child_nonce", "child_runtime", "stdout", "stderr", "report_id",
    }
    _require(set(report) == report_keys, "experiment report schema differs")
    unsigned_report = dict(report)
    report_id = unsigned_report.pop("report_id")
    _require(
        report["schema"] == SCHEMA
        and report["outcome"] == "PASS"
        and report["sample_kind"] == "measured"
        and HEX64.fullmatch(report["child_nonce"] or "") is not None
        and report_id == _canonical_sha256(unsigned_report),
        "experiment report identity differs",
    )
    _validate_identity(
        report["case_path"],
        report["cycle_index"],
        report["topology"],
        report["receipt_id"],
        report["gpu_devices"],
        report["environment_label"],
    )
    retained_authority = report["runtime_authority"]
    _require(isinstance(retained_authority, dict), "experiment runtime authority differs")
    authority, runtime_python, qualified_child, workload = _runtime_authority(
        repo=pathlib.Path(retained_authority.get("repo", "")),
        source_commit=retained_authority.get("source_commit", ""),
        runtime_provenance=pathlib.Path(
            retained_authority.get("runtime_provenance", {}).get("path", "")
        ),
        package=pathlib.Path(retained_authority.get("package", {}).get("path", "")),
        install_prefix=pathlib.Path(retained_authority.get("install_prefix", "")),
        m4_output=pathlib.Path(
            retained_authority.get("m4_authority", {}).get("output", "")
        ),
        workload_id=report["case_path"],
    )
    _require(
        authority == retained_authority
        and authority["runtime_receipt_id"] == report["receipt_id"],
        "experiment installed/source authority changed",
    )
    command = report["lane"].get("command") if isinstance(report["lane"], dict) else None
    _require(
        isinstance(command, list)
        and len(command) >= 8
        and command[:8]
        == [
            str(runtime_python),
            str(qualified_child),
            "--runtime-provenance",
            authority["runtime_provenance"]["path"],
            "--workload",
            str(workload),
            "--workload-id",
            report["case_path"],
        ],
        "experiment command is not the fixed qualified child",
    )
    experiment_arguments, expected_experiment = _experiment_arguments(command[8:])
    _require(command[8:] == experiment_arguments, "experiment command arguments differ")
    _require(
        isinstance(report["runtime_snapshot"], dict)
        and set(report["runtime_snapshot"])
        == {
            "runtime_contract",
            "child_runtime_terminal_id",
            "backend_policy_runs",
            "run_observations",
            "workload_statistics",
            "feature_counters",
            "phase_numerical",
        }
        and report["runtime_snapshot"]["runtime_contract"] == {"receipt_id": report["receipt_id"]}
        and set(report["lane"]) == {"spec", "command", "environment_sha256"}
        and report["lane"]["spec"] == {"devices": report["gpu_devices"]}
        and report["lane"]["environment_sha256"] == _canonical_sha256(report["environment"])
        and report["environment"]
        == _environment(
            report["topology"],
            report["gpu_devices"],
            report["environment_label"],
            pathlib.Path(authority["install_prefix"]),
        ),
        "experiment runtime/environment binding differs",
    )
    child_path = output / report["child_runtime"].get("path", "")
    _require(
        isinstance(report["child_runtime"], dict)
        and set(report["child_runtime"]) == {"path", "size_bytes", "sha256"}
        and report["child_runtime"] == _file_record(child_path, output),
        "experiment child runtime artifact differs",
    )
    child = _load(child_path)
    _require(
        child.get("terminal_id") == report["runtime_snapshot"]["child_runtime_terminal_id"]
        and child.get("runtime_receipt_id") == report["receipt_id"]
        and child.get("selected_physical_gpu_uuids") == report["gpu_devices"],
        "experiment child runtime report binding differs",
    )
    child = _child_runtime(
        child_path,
        nonce=report["child_nonce"],
        topology=report["topology"],
        runtime_receipt_id=report["receipt_id"],
        gpu_devices=report["gpu_devices"],
        authority=authority,
        experiment=expected_experiment,
    )
    _require(
        report["executable"] == _absolute_record(runtime_python),
        "experiment executable changed",
    )
    timing = report["timing"]
    _require(
        isinstance(timing, dict)
        and set(timing)
        == {"contract", "primary_seconds", "run_count", "process_seconds"}
        and {
            name: timing[name]
            for name in ("contract", "primary_seconds", "run_count")
        }
        == child["timing"]
        and type(timing["process_seconds"]) in {int, float}
        and math.isfinite(timing["process_seconds"])
        and timing["process_seconds"] >= timing["primary_seconds"],
        "experiment timing differs",
    )
    _require(
        report["runtime_snapshot"]["feature_counters"]
        == child["feature_counters"],
        "experiment child feature-counter binding differs",
    )
    _require(
        report["runtime_snapshot"]["backend_policy_runs"]
        == child["backend_policy_runs"],
        "experiment child backend-policy binding differs",
    )
    _require(report["runtime_snapshot"]["run_observations"] == child["run_observations"],
             "experiment child run-observation binding differs")
    _require(report["runtime_snapshot"]["workload_statistics"] == child["workload_statistics"],
             "experiment child workload-statistics binding differs")
    _require(report["runtime_snapshot"]["phase_numerical"] == child["phase_numerical"],
             "experiment child phase numerical binding differs")
    _require(report["process"] == {"returncode": 0, "timed_out": False}, "experiment process differs")
    for label in ("stdout", "stderr"):
        retained = report[label]
        _require(
            isinstance(retained, dict)
            and set(retained) == {"path", "size_bytes", "sha256"}
            and retained == _file_record(output / retained["path"], output)
            and retained["size_bytes"] <= MAX_LOG_BYTES,
            f"experiment {label} differs",
        )
    phase = report["phase_profile"]
    _require(
        isinstance(phase, dict)
        and phase == child["phase_profile"],
        "experiment phase profile differs",
    )
    unsigned_terminal = dict(terminal)
    terminal_id = unsigned_terminal.pop("terminal_id", None)
    _require(
        set(terminal) == {"schema", "outcome", "report", "report_id", "terminal_id"}
        and terminal["schema"] == COMPLETE_SCHEMA
        and terminal["outcome"] == "PASS"
        and terminal["report"] == _file_record(output / "report.json", output)
        and terminal["report_id"] == report_id
        and terminal_id == _canonical_sha256(unsigned_terminal),
        "experiment terminal binding differs",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--create", action="store_true")
    operation.add_argument("--verify", action="store_true")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--runtime-provenance", type=pathlib.Path)
    parser.add_argument("--package", type=pathlib.Path)
    parser.add_argument("--install-prefix", type=pathlib.Path)
    parser.add_argument("--m4-output", type=pathlib.Path)
    parser.add_argument("--workload-id")
    parser.add_argument("--cycle-index", type=int)
    parser.add_argument("--topology")
    parser.add_argument("--gpu-device", action="append", default=[])
    parser.add_argument("--environment-label", default="baseline")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--gpmeep-sweep-size")
    parser.add_argument("--gpmeep-sweep-run-length")
    parser.add_argument("--gpmeep-ablation")
    args = parser.parse_args(argv)
    try:
        if args.create:
            _require(
                all(
                    (
                        args.repo,
                        args.source_commit,
                        args.runtime_provenance,
                        args.package,
                        args.install_prefix,
                        args.m4_output,
                        args.workload_id,
                        args.cycle_index,
                        args.topology,
                    )
                ),
                "experiment create arguments are incomplete",
            )
            _require(
                (args.gpmeep_sweep_size is None)
                == (args.gpmeep_sweep_run_length is None)
                and not (
                    args.gpmeep_ablation is not None
                    and args.gpmeep_sweep_size is not None
                ),
                "experiment controls are mutually exclusive and complete",
            )
            experiment_arguments = (
                ["--gpmeep-ablation", args.gpmeep_ablation]
                if args.gpmeep_ablation is not None
                else [
                    "--gpmeep-sweep-size",
                    args.gpmeep_sweep_size,
                    "--gpmeep-sweep-run-length",
                    args.gpmeep_sweep_run_length,
                ]
                if args.gpmeep_sweep_size is not None
                else []
            )
            result = create(
                output=args.output,
                repo=args.repo,
                source_commit=args.source_commit,
                runtime_provenance=args.runtime_provenance,
                package=args.package,
                install_prefix=args.install_prefix,
                m4_output=args.m4_output,
                workload_id=args.workload_id,
                cycle_index=args.cycle_index,
                topology=args.topology,
                gpu_devices=args.gpu_device,
                environment_label=args.environment_label,
                experiment_arguments=experiment_arguments,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            _require(
                not any(
                    (
                        args.repo,
                        args.source_commit,
                        args.runtime_provenance,
                        args.package,
                        args.install_prefix,
                        args.m4_output,
                        args.workload_id,
                        args.cycle_index,
                        args.topology,
                        args.gpu_device,
                        args.gpmeep_sweep_size,
                        args.gpmeep_sweep_run_length,
                        args.gpmeep_ablation,
                    )
                ),
                "experiment verify arguments differ",
            )
            result = verify_complete(args.output)
        print(json.dumps({"schema": SCHEMA, "state": "PASS", "report_id": result["report_id"]}, sort_keys=True))
        return 0
    except (OSError, ExperimentError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"v1.1.0 inversion experiment: FAIL: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
