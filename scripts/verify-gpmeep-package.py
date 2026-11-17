#!/usr/bin/env python3
"""Verify installed gpmeep package ownership and runtime provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any

_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))
from gpmeep_release_manifest import (  # noqa: E402
    DISTRIBUTION_VERSION,
    RELEASE_MANIFEST_FIELDS,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    SEALED_M3_ANCHOR,
    ReleaseManifestError,
    validate_cuda_architecture_audit,
    validate_glibc_compatibility_audit,
    validate_release_manifest,
)

SCHEMA_VERSION = 1
CONDA_BUILD_NUMBER = 0
CONDA_BUILD_STRING = "cuda124_mpi_openmpi_py311_0"
CONDA_SUBDIR = "linux-64"
_V110_CANDIDATE_RECEIPT_RELATIVE = "share/gpmeep/v110-candidate-build-receipt.json"


class VerificationError(RuntimeError):
    pass


def _run_v110_validator(contract: pathlib.Path, validator: pathlib.Path, mode: str) -> None:
    if not validator.is_file():
        raise VerificationError("v1.1 development-contract validator is absent")
    result = subprocess.run(
        [sys.executable, str(validator), "--contract", str(contract), mode],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode != 0:
        raise VerificationError("v1.1 development contract preflight failed")


def _validate_v110_development_contract(prefix: pathlib.Path) -> dict[str, Any] | None:
    """Bind v1.1 package verification to the source/installed contract gate."""
    source_root = _SCRIPT_DIRECTORY.parent
    source_contract = source_root / "packaging/v110-development-contract.json"
    # The canonical source verifier lives in ``scripts``.  Its installed copy
    # lives in ``share/gpmeep/libexec``, whose parent intentionally contains a
    # package-relative ``packaging`` directory as well.  Do not mistake that
    # installed payload for a complete source tree: ``--contract-only`` also
    # validates source documentation and release-controller wiring which are
    # deliberately outside the installed-package contract.
    if _SCRIPT_DIRECTORY.name == "scripts":
        if not source_contract.is_file():
            raise VerificationError("v1.1 source development contract is absent")
        _run_v110_validator(
            source_contract,
            _SCRIPT_DIRECTORY / "validate-v110-development-contract.py",
            "--contract-only",
        )
    if DISTRIBUTION_VERSION != "1.1.0":
        return None

    share = prefix / "share/gpmeep"
    packaging = share / "packaging"
    contract = packaging / "v110-development-contract.json"
    equal_work_plan = packaging / "v110-m4-equal-work-plan.json"
    schema = packaging / "v110-performance-inversion-record.schema.json"
    inventory = packaging / "v110-performance-inversion-inventory.json"
    receipt = prefix / _V110_CANDIDATE_RECEIPT_RELATIVE
    validator = share / "libexec/validate-v110-development-contract.py"
    receipt_tool = share / "libexec/v110-release-gate-receipt.py"
    final_receipt_tool = share / "libexec/v110-final-release-receipt.py"
    closure_tool = share / "libexec/close-v110-inversion-registry.py"
    for required in (
        contract,
        equal_work_plan,
        schema,
        inventory,
        receipt,
        validator,
        receipt_tool,
        final_receipt_tool,
        closure_tool,
    ):
        if not required.is_file():
            raise VerificationError(f"installed v1.1 package omits gate file: {required.name}")
    _run_v110_validator(contract, validator, "--installed-package")
    return {
        "contract": contract,
        "equal_work_plan": equal_work_plan,
        "schema": schema,
        "inventory": inventory,
        "receipt": receipt,
        "receipt_tool": receipt_tool,
        "final_receipt_tool": final_receipt_tool,
        "closure_tool": closure_tool,
    }


def _verify_v110_candidate_receipt(
    files: dict[str, Any], *, expected_mode: str
) -> dict[str, Any]:
    arguments = [
        sys.executable,
        "-I",
        str(files["receipt_tool"]),
        "--verify",
        "--receipt",
        str(files["receipt"]),
        "--contract",
        str(files["contract"]),
        "--schema",
        str(files["schema"]),
        "--inventory",
        str(files["inventory"]),
        "--embedded-only",
    ]
    if expected_mode == "candidate":
        arguments.append("--require-candidate")
    result = subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode != 0:
        raise VerificationError("v1.1 candidate-build receipt verification failed")
    try:
        receipt = json.loads(
            files["receipt"].read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("v1.1 candidate-build receipt is unreadable") from error
    if receipt.get("build_mode") != expected_mode:
        raise VerificationError("v1.1 package build mode differs from its candidate receipt")
    return receipt


def _verify_v110_final_receipt(
    files: dict[str, Any],
    *,
    receipt_path: pathlib.Path,
    package_path: pathlib.Path,
    qualification_manifest: pathlib.Path,
    evidence_root: pathlib.Path,
    release_source_root: pathlib.Path,
    candidate_contract_gate: pathlib.Path,
    candidate_registry_gate: pathlib.Path,
    candidate_attestation: pathlib.Path,
    m4_candidate_manifest: pathlib.Path,
    candidate_evidence_root: pathlib.Path,
    source_transition_policy: pathlib.Path,
    expected_source_transition_policy_sha256: str,
    source_transition_verifier: pathlib.Path,
    expected_source_transition_verifier_sha256: str,
    final_audit_authority: pathlib.Path,
    expected_final_audit_authority_sha256: str,
) -> dict[str, Any]:
    source_root = release_source_root.resolve(strict=True)
    if not source_root.is_dir() or source_root.is_symlink():
        raise VerificationError("v1.1 release source root is unsafe")
    source_files = {
        "final_receipt_tool": source_root / "scripts/v110-final-release-receipt.py",
        "receipt_tool": source_root / "scripts/v110-release-gate-receipt.py",
        "contract": source_root / "packaging/v110-development-contract.json",
        "schema": source_root / "packaging/v110-performance-inversion-record.schema.json",
        "inventory": source_root / "packaging/v110-performance-inversion-inventory.json",
    }
    if any(not path.is_file() or path.is_symlink() for path in source_files.values()):
        raise VerificationError("v1.1 release source root omits an exact verifier input")
    arguments = [
        sys.executable,
        "-I",
        str(source_files["final_receipt_tool"]),
        "--verify",
        "--receipt", str(receipt_path),
        "--package", str(package_path),
        "--candidate-receipt", str(files["receipt"]),
        "--candidate-tool", str(source_files["receipt_tool"]),
        "--contract", str(source_files["contract"]),
        "--inversion-schema", str(source_files["schema"]),
        "--inventory", str(source_files["inventory"]),
        "--qualification-manifest", str(qualification_manifest),
        "--evidence-root", str(evidence_root),
        "--candidate-contract-gate", str(candidate_contract_gate),
        "--candidate-registry-gate", str(candidate_registry_gate),
        "--candidate-attestation", str(candidate_attestation),
        "--m4-candidate-manifest", str(m4_candidate_manifest),
        "--candidate-evidence-root", str(candidate_evidence_root),
        "--source-transition-policy", str(source_transition_policy),
        "--expected-source-transition-policy-sha256", expected_source_transition_policy_sha256,
        "--source-transition-verifier", str(source_transition_verifier),
        "--expected-source-transition-verifier-sha256", expected_source_transition_verifier_sha256,
        "--final-audit-authority", str(final_audit_authority),
        "--expected-final-audit-authority-sha256", expected_final_audit_authority_sha256,
    ]
    result = subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode != 0:
        raise VerificationError("v1.1 external final-release receipt verification failed")
    try:
        return json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("v1.1 external final-release receipt is unreadable") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise VerificationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _one_conda_record(prefix: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    matches = sorted(
        (prefix / "conda-meta").glob(f"gpmeep-{DISTRIBUTION_VERSION}-*.json")
    )
    if len(matches) != 1:
        raise VerificationError(
            f"expected one gpmeep {DISTRIBUTION_VERSION} conda record, "
            f"found {len(matches)}"
        )
    if list((prefix / "conda-meta").glob("pymeep-*.json")):
        raise VerificationError("gpmeep cannot share an environment with pymeep")
    try:
        value = json.loads(
            matches[0].read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("gpmeep conda record is unreadable") from error
    if value.get("name") != "gpmeep" or value.get("version") != DISTRIBUTION_VERSION:
        raise VerificationError("gpmeep conda record identity is invalid")
    valid_build = value.get("build") == CONDA_BUILD_STRING
    if DISTRIBUTION_VERSION == "1.1.0":
        valid_build = value.get("build") in {
            "cuda124_mpi_openmpi_py311_0_development",
            "cuda124_mpi_openmpi_py311_0",
        }
    if (
        type(value.get("build_number")) is not int
        or value["build_number"] != CONDA_BUILD_NUMBER
        or not valid_build
        or value.get("subdir") != CONDA_SUBDIR
    ):
        raise VerificationError("gpmeep conda build identity is invalid")
    return matches[0], value


def _owned_paths(record: dict[str, Any]) -> list[str]:
    values = record.get("files")
    if not isinstance(values, list) or not values:
        paths_data = record.get("paths_data")
        if isinstance(paths_data, dict) and isinstance(paths_data.get("paths"), list):
            values = [item.get("_path") for item in paths_data["paths"]]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise VerificationError("gpmeep conda record has no owned-file list")
    if len(values) != len(set(values)):
        raise VerificationError("gpmeep conda record repeats an owned path")
    return sorted(values)


def _validate_package_identity(
    record: dict[str, Any], expected_sha256: str | None
) -> str:
    actual = record.get("sha256")
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise VerificationError("gpmeep conda record has no valid package SHA-256")
    if expected_sha256 is not None and actual != expected_sha256:
        raise VerificationError("gpmeep conda record package SHA-256 differs")
    return actual


def _owned_path_metadata(
    record: dict[str, Any], paths: list[str]
) -> dict[str, dict[str, Any]]:
    paths_data = record.get("paths_data")
    if not isinstance(paths_data, dict) or not isinstance(
        paths_data.get("paths"), list
    ):
        raise VerificationError("gpmeep conda record has no paths_data metadata")
    metadata: dict[str, dict[str, Any]] = {}
    for item in paths_data["paths"]:
        if not isinstance(item, dict):
            raise VerificationError("gpmeep paths_data contains a non-object entry")
        relative = item.get("_path")
        if not isinstance(relative, str) or not relative or relative in metadata:
            raise VerificationError("gpmeep paths_data has an invalid or repeated path")
        metadata[relative] = item
    if set(metadata) != set(paths):
        raise VerificationError("gpmeep files and paths_data inventories differ")
    return metadata


def _audit_owned_paths(
    prefix: pathlib.Path,
    paths: list[str],
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    regular = 0
    symlinks = 0
    verified_hashes = 0
    verified_sizes = 0
    prefix_rewritten_sizes = 0
    for relative in paths:
        candidate = pathlib.PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
            raise VerificationError(f"unsafe package-owned path: {relative}")
        path = prefix.joinpath(*candidate.parts)
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(prefix)
            except ValueError as error:
                raise VerificationError(
                    f"package symlink escapes the environment: {relative}"
                ) from error
            symlinks += 1
        elif path.is_file():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(prefix)
            except ValueError as error:
                raise VerificationError(
                    f"package regular file escapes the environment: {relative}"
                ) from error
            regular += 1
        else:
            raise VerificationError(f"package-owned file is absent: {relative}")
        if metadata is not None:
            item = metadata[relative]
            path_type = item.get("path_type")
            if path_type == "softlink" and not path.is_symlink():
                raise VerificationError(f"expected a package symlink: {relative}")
            if path_type == "hardlink" and path.is_symlink():
                raise VerificationError(f"unexpected package symlink: {relative}")
            if path_type not in {"hardlink", "softlink"}:
                raise VerificationError(
                    f"unsupported package path type {path_type!r}: {relative}"
                )
            source_hash = item.get("sha256")
            expected_hash = item.get("sha256_in_prefix") or source_hash
            if (
                not isinstance(expected_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            ):
                raise VerificationError(f"package path has no valid hash: {relative}")
            if _sha256_file(resolved) != expected_hash:
                raise VerificationError(f"package-owned file hash differs: {relative}")
            expected_size = item.get("size_in_bytes")
            if not isinstance(expected_size, int) or expected_size < 0:
                raise VerificationError(f"package path has no valid size: {relative}")
            if source_hash is not None and source_hash != expected_hash:
                # Conda records the pre-rewrite size even though replacing its
                # fixed prefix placeholder can change a text file's length.
                # sha256_in_prefix remains authoritative for installed bytes.
                prefix_rewritten_sizes += 1
            else:
                if resolved.stat().st_size != expected_size:
                    raise VerificationError(
                        f"package-owned file size differs: {relative}"
                    )
                verified_sizes += 1
            verified_hashes += 1
    return {
        "owned_paths": len(paths),
        "regular_files": regular,
        "symlinks": symlinks,
        "content_hashes_verified": verified_hashes,
        "content_sizes_verified": verified_sizes,
        "prefix_rewritten_sizes": prefix_rewritten_sizes,
    }


def _mapped_libmeep(prefix: pathlib.Path) -> pathlib.Path:
    candidates: set[pathlib.Path] = set()
    for line in (
        pathlib.Path("/proc/self/maps")
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()
    ):
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = pathlib.Path(columns[5])
            if path.name.startswith("libmeep.so") and path.is_file():
                candidates.add(path.resolve(strict=True))
    if len(candidates) != 1:
        raise VerificationError(f"expected one mapped libmeep, found {len(candidates)}")
    path = next(iter(candidates))
    try:
        path.relative_to(prefix)
    except ValueError as error:
        raise VerificationError(
            f"mapped libmeep escaped gpmeep prefix: {path}"
        ) from error
    return path


def _runtime(prefix: pathlib.Path) -> dict[str, Any]:
    if pathlib.Path(sys.prefix).resolve() != prefix:
        raise VerificationError(
            f"verifier must run with {prefix}/bin/python, got {sys.executable}"
        )
    os.environ.setdefault("MEEP_GPU_BACKEND", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import meep as mp
    import meep._meep as native

    if not mp.gpu.compiled or not mp.with_mpi() or not mp.is_single_precision():
        raise VerificationError("installed runtime is not CUDA+MPI FP32 gpmeep")
    module = pathlib.Path(mp.__file__).resolve(strict=True)
    extension = pathlib.Path(native.__file__).resolve(strict=True)
    for name, path in (("module", module), ("extension", extension)):
        try:
            path.relative_to(prefix)
        except ValueError as error:
            raise VerificationError(f"{name} escaped gpmeep prefix: {path}") from error
    return {
        "python": _file_record(pathlib.Path(sys.executable)),
        "module": _file_record(module),
        "extension": _file_record(extension),
        "libmeep": _file_record(_mapped_libmeep(prefix)),
        "meep_version": mp.__version__,
        "cuda_compiled": bool(mp.gpu.compiled),
        "mpi_enabled": bool(mp.with_mpi()),
        "single_precision": bool(mp.is_single_precision()),
    }


def _manifest(prefix: pathlib.Path, expected_commit: str | None) -> dict[str, Any]:
    lexical_path = prefix / "share/gpmeep/release.json"
    try:
        path = lexical_path.resolve(strict=True)
        path.relative_to(prefix)
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise OSError("release manifest is not regular")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("gpmeep release manifest is unreadable") from error
    try:
        validate_release_manifest(value, expected_source_commit=expected_commit)
        validate_cuda_architecture_audit(
            prefix,
            value["cuda_architecture_audit_sha256"],
            value["cuda_real_architectures"],
            value["cuda_virtual_architectures"],
        )
        validate_glibc_compatibility_audit(
            prefix, value["glibc_compatibility_audit_sha256"]
        )
    except ReleaseManifestError as error:
        raise VerificationError(str(error)) from error
    value["file"] = _file_record(path)
    return value


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent:
        raise VerificationError("report parent cannot contain symlinks")
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise VerificationError(f"refusing to overwrite report: {destination}")
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=parent
    )
    temporary = pathlib.Path(temporary_name)
    linked = False
    committed = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        linked = True
        if stat.S_IMODE(destination.stat().st_mode) != 0o600:
            raise VerificationError("report mode differs after publication")
        if destination.read_bytes() != payload:
            raise VerificationError("report bytes differ after publication")
        temporary.unlink()
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        committed = True
    finally:
        temporary.unlink(missing_ok=True)
        if linked and not committed:
            destination.unlink(missing_ok=True)
            try:
                directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify installed gpmeep provenance")
    parser.add_argument("--prefix", type=pathlib.Path, default=pathlib.Path(sys.prefix))
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-package-sha256")
    parser.add_argument("--package-file", type=pathlib.Path)
    parser.add_argument("--final-release-receipt", type=pathlib.Path)
    parser.add_argument("--qualification-manifest", type=pathlib.Path)
    parser.add_argument("--release-evidence-root", type=pathlib.Path)
    parser.add_argument("--release-source-root", type=pathlib.Path)
    parser.add_argument("--candidate-contract-gate", type=pathlib.Path)
    parser.add_argument("--candidate-registry-gate", type=pathlib.Path)
    parser.add_argument("--candidate-attestation", type=pathlib.Path)
    parser.add_argument("--m4-candidate-manifest", type=pathlib.Path)
    parser.add_argument("--candidate-evidence-root", type=pathlib.Path)
    parser.add_argument("--source-transition-policy", type=pathlib.Path)
    parser.add_argument("--expected-source-transition-policy-sha256")
    parser.add_argument("--source-transition-verifier", type=pathlib.Path)
    parser.add_argument("--expected-source-transition-verifier-sha256")
    parser.add_argument("--final-audit-authority", type=pathlib.Path)
    parser.add_argument("--expected-final-audit-authority-sha256")
    parser.add_argument("--report", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prefix = args.prefix.resolve(strict=True)
    if (
        args.expected_source_commit is not None
        and re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is None
    ):
        raise VerificationError("--expected-source-commit must be 40 lowercase hex")
    if (
        args.expected_package_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.expected_package_sha256) is None
    ):
        raise VerificationError("--expected-package-sha256 must be 64 lowercase hex")
    record_path, conda_record = _one_conda_record(prefix)
    v110_gate = _validate_v110_development_contract(prefix)
    v110_receipt = None
    final_release = None
    if v110_gate is not None:
        build_string = conda_record.get("build", "")
        expected_mode = "development" if build_string.endswith("_development") else "candidate"
        v110_receipt = _verify_v110_candidate_receipt(v110_gate, expected_mode=expected_mode)
    package_sha256 = _validate_package_identity(
        conda_record, args.expected_package_sha256
    )
    paths = _owned_paths(conda_record)
    metadata = _owned_path_metadata(conda_record, paths)
    audit = _audit_owned_paths(prefix, paths, metadata)
    required_paths = {
        "bin/gpmeep-self-check",
        "bin/gpmeep-verify-provenance",
        "share/gpmeep/cuda-architecture-inventory.json",
        "share/gpmeep/glibc-compatibility.json",
        "share/gpmeep/libexec/check-cuda-architecture-inventory.py",
        "share/gpmeep/libexec/gpmeep_backend_guard.py",
        "share/gpmeep/libexec/gpmeep_release_manifest.py",
        "share/gpmeep/release.json",
    }
    if DISTRIBUTION_VERSION == "1.1.0":
        required_paths.update(
            {
                "share/gpmeep/libexec/v110-release-gate-receipt.py",
                "share/gpmeep/libexec/v110-final-release-receipt.py",
                "share/gpmeep/libexec/close-v110-inversion-registry.py",
                "share/gpmeep/libexec/validate-v110-development-contract.py",
                "share/gpmeep/libexec/v110-inversion-qualified-child.py",
                "share/gpmeep/packaging/v110-development-contract.json",
                "share/gpmeep/packaging/v110-m4-equal-work-plan.json",
                "share/gpmeep/packaging/v110-performance-inversion-record.schema.json",
                "share/gpmeep/packaging/v110-performance-inversion-inventory.json",
                "share/gpmeep/v110-candidate-build-receipt.json",
            }
        )
    missing = sorted(required_paths - set(paths))
    if missing:
        raise VerificationError(f"gpmeep package omits required paths: {missing}")
    manifest = _manifest(prefix, args.expected_source_commit)
    if v110_receipt is not None and v110_receipt["source_commit"] != manifest["source_commit"]:
        raise VerificationError("v1.1 gate receipt source commit differs from release manifest")
    final_arguments = (
        args.package_file,
        args.final_release_receipt,
        args.qualification_manifest,
        args.release_evidence_root,
        args.release_source_root,
        args.candidate_contract_gate,
        args.candidate_registry_gate,
        args.candidate_attestation,
        args.m4_candidate_manifest,
        args.candidate_evidence_root,
        args.source_transition_policy,
        args.expected_source_transition_policy_sha256,
        args.source_transition_verifier,
        args.expected_source_transition_verifier_sha256,
        args.final_audit_authority,
        args.expected_final_audit_authority_sha256,
    )
    if any(item is not None for item in final_arguments):
        if v110_gate is None or v110_receipt is None or not all(item is not None for item in final_arguments):
            raise VerificationError("final-release verification requires package, receipt, qualification/evidence, release source, and candidate replay inputs")
        if v110_receipt["build_mode"] != "candidate":
            raise VerificationError("a development package cannot carry final release authority")
        final_release = _verify_v110_final_receipt(
            v110_gate,
            receipt_path=args.final_release_receipt,
            package_path=args.package_file,
            qualification_manifest=args.qualification_manifest,
            evidence_root=args.release_evidence_root,
            release_source_root=args.release_source_root,
            candidate_contract_gate=args.candidate_contract_gate,
            candidate_registry_gate=args.candidate_registry_gate,
            candidate_attestation=args.candidate_attestation,
            m4_candidate_manifest=args.m4_candidate_manifest,
            candidate_evidence_root=args.candidate_evidence_root,
            source_transition_policy=args.source_transition_policy,
            expected_source_transition_policy_sha256=args.expected_source_transition_policy_sha256,
            source_transition_verifier=args.source_transition_verifier,
            expected_source_transition_verifier_sha256=args.expected_source_transition_verifier_sha256,
            final_audit_authority=args.final_audit_authority,
            expected_final_audit_authority_sha256=args.expected_final_audit_authority_sha256,
        )
        if (
            final_release.get("state") != "RELEASE_AUTHORIZED"
            or final_release.get("package", {}).get("sha256") != package_sha256
            or final_release.get("candidate_authority", {}).get("receipt_id") != v110_receipt["receipt_id"]
            or final_release.get("candidate_authority", {}).get("source_commit") != manifest["source_commit"]
        ):
            raise VerificationError("external final-release authority is not bound to this installed package")
    report = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "prefix": str(prefix),
        "conda_record": _file_record(record_path),
        "package_url": conda_record.get("url"),
        "package_sha256": package_sha256,
        "ownership": audit,
        "manifest": manifest,
        "runtime": _runtime(prefix),
    }
    if v110_receipt is not None:
        report["v110_candidate_gate"] = v110_receipt
    if final_release is not None:
        report["v110_final_release"] = final_release
    report["report_id"] = hashlib.sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    if args.report is not None:
        _write_new(args.report, report)
    print(
        "gpmeep provenance PASS: "
        f"source={report['manifest']['source_commit']}, report_id={report['report_id']}"
    )
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"gpmeep provenance FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
