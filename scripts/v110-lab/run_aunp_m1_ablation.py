#!/usr/bin/env python3
"""Run and replay the v1.1.0 M1 OFF/C1/C3/C1+C3/C4 ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import subprocess
import sys

import run_aunp_m1_ablation_sample as ablation_sample


_BASE_CONTROLLER_PATH = pathlib.Path(__file__).with_name(
    "run_aunp_zero_state_canary.py"
)
_BASE_MODULE_NAME = "_gpmeep_v110_m1_ablation_controller_base"
_BASE_SPEC = importlib.util.spec_from_file_location(
    _BASE_MODULE_NAME, _BASE_CONTROLLER_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the private M1 ablation controller base")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_MODULE_NAME] = base
_BASE_SPEC.loader.exec_module(base)


MODES = ablation_sample.MODES
CANDIDATE_FEATURE = ablation_sample.CANDIDATE_FEATURE
CONTROLLER_PATH = pathlib.Path(__file__).resolve()
WORKER_PATH = CONTROLLER_PATH.parent / "run_aunp_m1_ablation_sample.py"
METRICS = (
    "initialization_wall_seconds",
    "fdtd_wall_seconds",
    "end_to_end_seconds",
)
VALIDATION_GATE_SCHEMA = "gpmeep-v110-m1-pre-performance-validation-gate-v1"
AUDIT_GATE_SCHEMA = "gpmeep-v110-m1-pre-performance-adversarial-audit-gate-v1"
VALIDATION_GATE_KEY = "m1_validation_gate"
AUDIT_GATE_KEY = "m1_adversarial_audit_gate"
BUILD_RECEIPT_KEY = "m1_build_receipt"
BUILD_IDENTITY_KEY = "m1_build_identity"
QUALIFICATION_FONTCONFIG_KEY = "m1_qualification_fontconfig"

base.sample = ablation_sample
base.SCHEMA = "gpmeep-v110-m1-aunp-ablation-canary-v1"
base.COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-ablation-canary-complete-v1"
base.RUN_SCHEMA = "gpmeep-v110-m1-aunp-ablation-canary-run-v1"
base.CHECKPOINT_SCHEMA = "gpmeep-v110-m1-aunp-ablation-checkpoint-v1"
base.TELEMETRY_SCHEMA = "gpmeep-v110-m1-aunp-ablation-telemetry-v1"
base.RUNTIME_ATTESTATION_SCHEMA = "gpmeep-v110-m1-runtime-attestation-v1"
base.CANDIDATE_FEATURE = CANDIDATE_FEATURE
base.CONTROLLER_PATH = CONTROLLER_PATH
base.WORKER_PATH = WORKER_PATH
base.ATTESTATION_VALIDATION_KEY = "m1_validation"
base.ATTESTATION_AUDIT_KEY = "m1_adversarial_audit"
base.LEGACY_CANDIDATE_FIELD_ALLOWED = False
base.SUPPORT_SOURCE_PATHS = {
    "base_controller": _BASE_CONTROLLER_PATH,
    "base_worker": ablation_sample._BASE_PATH,
    "c3_worker": pathlib.Path(ablation_sample.c3.__file__).resolve(),
    "c4_worker": pathlib.Path(ablation_sample.c4.__file__).resolve(),
}
base.IMPLEMENTATION_ALLOWED_DESCENDANTS = (
    "scripts/v110-lab/",
    "scripts/tests/test_v110_aunp_m1_ablation.py",
)


def _build_identity(receipt: dict[str, object]) -> dict[str, object]:
    source = receipt.get("source_start")
    source_sha256 = source.get("sha256") if isinstance(source, dict) else None
    identity = {
        "receipt_id": receipt.get("receipt_id"),
        "artifact_set_id": receipt.get("artifact_set_id"),
        "build_input_id": receipt.get("build_input_id"),
        "build_kind": receipt.get("build_kind"),
        "git_head": receipt.get("git_head"),
        "source_sha256": source_sha256,
    }
    if (
        set(identity) != {
            "receipt_id", "artifact_set_id", "build_input_id", "build_kind",
            "git_head", "source_sha256",
        }
        or any(
            not isinstance(value, str) or not value
            for value in identity.values()
        )
    ):
        raise base.CanaryError("M1 build receipt identity is invalid")
    return identity


def _load_bound_json(record: dict[str, object], label: str) -> object:
    base._validated_absolute_record(record, label)
    return base._load_stable_json(
        pathlib.Path(str(record["path"])), label, expected=record
    )


def _validate_gate_payloads(
    validation_gate: object,
    audit_gate: object,
    *,
    source_commit: str,
    implementation_commit: str,
    validation_report: dict[str, object],
    audit_report: dict[str, object],
    validation_gate_record: dict[str, object],
    build_receipt: dict[str, object],
    build_identity: dict[str, object],
) -> None:
    expected_validation_keys = {
        "schema", "status", "source_commit", "implementation_commit",
        "validation_report", "build_receipt", "build_identity",
    }
    if (
        not isinstance(validation_gate, dict)
        or set(validation_gate) != expected_validation_keys
        or validation_gate.get("schema") != VALIDATION_GATE_SCHEMA
        or validation_gate.get("status") != "PASS"
        or validation_gate.get("source_commit") != source_commit
        or validation_gate.get("implementation_commit") != implementation_commit
        or not base.common.typed_json_equal(
            validation_gate.get("validation_report"), validation_report
        )
        or not base.common.typed_json_equal(
            validation_gate.get("build_receipt"), build_receipt
        )
        or not base.common.typed_json_equal(
            validation_gate.get("build_identity"), build_identity
        )
    ):
        raise base.CanaryError("M1 validation gate rejected the supplied evidence")

    expected_audit_keys = {
        "schema", "decision", "severity", "source_commit",
        "implementation_commit", "validation_report", "validation_gate",
        "audit_report", "build_receipt", "build_identity",
    }
    severity = audit_gate.get("severity") if isinstance(audit_gate, dict) else None
    severity_valid = (
        isinstance(severity, dict)
        and set(severity) == {"critical", "high", "medium", "low"}
        and all(
            type(severity[name]) is int and severity[name] >= 0
            for name in severity
        )
    )
    if (
        not isinstance(audit_gate, dict)
        or set(audit_gate) != expected_audit_keys
        or audit_gate.get("schema") != AUDIT_GATE_SCHEMA
        or audit_gate.get("decision") != "ACCEPT"
        or not severity_valid
        or any(severity[name] != 0 for name in ("critical", "high", "medium"))
        or audit_gate.get("source_commit") != source_commit
        or audit_gate.get("implementation_commit") != implementation_commit
        or not base.common.typed_json_equal(
            audit_gate.get("validation_report"), validation_report
        )
        or not base.common.typed_json_equal(
            audit_gate.get("validation_gate"), validation_gate_record
        )
        or not base.common.typed_json_equal(
            audit_gate.get("audit_report"), audit_report
        )
        or not base.common.typed_json_equal(
            audit_gate.get("build_receipt"), build_receipt
        )
        or not base.common.typed_json_equal(
            audit_gate.get("build_identity"), build_identity
        )
    ):
        raise base.CanaryError("M1 adversarial audit gate is not an acceptance")


def _verified_receipt_and_identity(
    receipt_path: pathlib.Path,
    runtime: dict[str, dict[str, object]],
    implementation_commit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    scripts_dir = base.REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from gpmeep_provenance import ProvenanceError, verify_build_receipt

    try:
        receipt = verify_build_receipt(
            receipt_path, base.REPO, verify_source=False
        )
        base.common.validate_runtime_against_receipt(
            runtime, receipt, base.REPO
        )
    except (ProvenanceError, base.common.WorkloadError) as exc:
        raise base.CanaryError(f"M1 build receipt verification failed: {exc}") from exc
    identity = _build_identity(receipt)
    if identity["git_head"] != implementation_commit:
        raise base.CanaryError("M1 implementation commit differs from build receipt")
    return receipt, identity


def _qualification_fontconfig_record(
    receipt: dict[str, object],
) -> dict[str, object]:
    configuration_files = receipt.get("configuration_files")
    relative_record = (
        configuration_files.get("qualification_fontconfig")
        if isinstance(configuration_files, dict)
        else None
    )
    if (
        not isinstance(relative_record, dict)
        or set(relative_record) != {"path", "size_bytes", "sha256"}
        or not isinstance(relative_record.get("path"), str)
    ):
        raise base.CanaryError(
            "M1 build receipt has no qualification Fontconfig record"
        )
    relative_path = pathlib.PurePosixPath(relative_record["path"])
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or "." in relative_path.parts
    ):
        raise base.CanaryError(
            "M1 qualification Fontconfig receipt path is unsafe"
        )
    path = base.REPO.joinpath(*relative_path.parts)
    observed_relative = base.common.file_record(path, base.REPO)
    if not base.common.typed_json_equal(relative_record, observed_relative):
        raise base.CanaryError(
            "M1 qualification Fontconfig differs from build receipt"
        )
    text = path.read_text(encoding="utf-8")
    if (
        text.count('<cachedir prefix="xdg">fontconfig</cachedir>') != 1
        or text.count("<cachedir") != 1
        or "var/cache/fontconfig" in text
    ):
        raise base.CanaryError(
            "M1 qualification Fontconfig cache policy is unsafe"
        )
    return base.common.absolute_file_record(
        path, "M1 qualification Fontconfig"
    )


def _attestation_extension_builder(
    args: argparse.Namespace,
    source_commit: str,
    implementation_commit: str,
    runtime: dict[str, dict[str, object]],
    validation_report: dict[str, object],
    audit_report: dict[str, object],
) -> dict[str, object]:
    receipt_path = args.build_receipt.resolve(strict=True)
    receipt, build_identity = _verified_receipt_and_identity(
        receipt_path, runtime, implementation_commit
    )
    build_receipt = base.common.absolute_file_record(
        receipt_path, "M1 build receipt"
    )
    validation_gate_path = args.m1_validation_gate.resolve(strict=True)
    audit_gate_path = args.m1_adversarial_audit_gate.resolve(strict=True)
    validation_gate_record = base.common.absolute_file_record(
        validation_gate_path, "M1 validation gate"
    )
    audit_gate_record = base.common.absolute_file_record(
        audit_gate_path, "M1 adversarial audit gate"
    )
    validation_gate = _load_bound_json(
        validation_gate_record, "M1 validation gate"
    )
    audit_gate = _load_bound_json(
        audit_gate_record, "M1 adversarial audit gate"
    )
    _validate_gate_payloads(
        validation_gate,
        audit_gate,
        source_commit=source_commit,
        implementation_commit=implementation_commit,
        validation_report=validation_report,
        audit_report=audit_report,
        validation_gate_record=validation_gate_record,
        build_receipt=build_receipt,
        build_identity=build_identity,
    )
    return {
        VALIDATION_GATE_KEY: validation_gate_record,
        AUDIT_GATE_KEY: audit_gate_record,
        BUILD_RECEIPT_KEY: build_receipt,
        BUILD_IDENTITY_KEY: build_identity,
        QUALIFICATION_FONTCONFIG_KEY: _qualification_fontconfig_record(
            receipt
        ),
    }


def _attestation_extension_validator(
    attestation: dict[str, object],
    source_commit: str,
    implementation_commit: str,
) -> None:
    validation_report = attestation.get(base.ATTESTATION_VALIDATION_KEY)
    audit_report = attestation.get(base.ATTESTATION_AUDIT_KEY)
    validation_gate_record = attestation.get(VALIDATION_GATE_KEY)
    audit_gate_record = attestation.get(AUDIT_GATE_KEY)
    build_receipt = attestation.get(BUILD_RECEIPT_KEY)
    build_identity = attestation.get(BUILD_IDENTITY_KEY)
    qualification_fontconfig = attestation.get(
        QUALIFICATION_FONTCONFIG_KEY
    )
    for record, label in (
        (validation_report, "M1 validation report"),
        (audit_report, "M1 adversarial audit report"),
        (validation_gate_record, "M1 validation gate"),
        (audit_gate_record, "M1 adversarial audit gate"),
        (build_receipt, "M1 build receipt"),
        (qualification_fontconfig, "M1 qualification Fontconfig"),
    ):
        base._validated_absolute_record(record, label)
    if not isinstance(build_identity, dict):
        raise base.CanaryError("M1 attested build identity is invalid")
    runtime = attestation.get("runtime_artifacts")
    if not isinstance(runtime, dict):
        raise base.CanaryError("M1 attested runtime artifact inventory is invalid")
    receipt, observed_identity = _verified_receipt_and_identity(
        pathlib.Path(str(build_receipt["path"])),
        runtime,
        implementation_commit,
    )
    if not base.common.typed_json_equal(build_identity, observed_identity):
        raise base.CanaryError("M1 attested build identity changed")
    if not base.common.typed_json_equal(
        qualification_fontconfig,
        _qualification_fontconfig_record(receipt),
    ):
        raise base.CanaryError(
            "M1 attested qualification Fontconfig changed"
        )
    validation_gate = _load_bound_json(
        validation_gate_record, "M1 validation gate"
    )
    audit_gate = _load_bound_json(
        audit_gate_record, "M1 adversarial audit gate"
    )
    _validate_gate_payloads(
        validation_gate,
        audit_gate,
        source_commit=source_commit,
        implementation_commit=implementation_commit,
        validation_report=validation_report,
        audit_report=audit_report,
        validation_gate_record=validation_gate_record,
        build_receipt=build_receipt,
        build_identity=build_identity,
    )


def task_plan() -> list[dict[str, object]]:
    orders = (
        (("gpu1", MODES), ("gpu2", tuple(reversed(MODES)))),
        (("gpu1", tuple(reversed(MODES))), ("gpu2", MODES)),
    )
    tasks: list[dict[str, object]] = []
    for repeat, topology_orders in enumerate(orders):
        sequence = 0
        for topology, modes in topology_orders:
            for mode in modes:
                tasks.append(
                    {
                        "repeat": repeat,
                        "sequence": sequence,
                        "topology": topology,
                        "ranks": 1 if topology == "gpu1" else 2,
                        "feature_mode": mode,
                        "task_id": (
                            f"r{repeat}-{sequence}-{topology}-{mode.replace('+', '_')}"
                        ),
                    }
                )
                sequence += 1
    return tasks


def _omp_threads(task: dict[str, object]) -> int:
    mode = task.get("feature_mode")
    if mode not in MODES:
        raise base.CanaryError("M1 task mode is invalid")
    return ablation_sample.c4.OMP_THREADS if mode == "c4" else 1


def _environment(
    task: dict[str, object], python: pathlib.Path, pythonpath: pathlib.Path,
    library_path: str, devices: list[str], home: pathlib.Path,
    runtime_attestation: dict[str, object] | None = None,
) -> dict[str, str]:
    ranks = task.get("ranks")
    mode = task.get("feature_mode")
    if type(ranks) is not int or ranks not in (1, 2) or mode not in MODES:
        raise base.CanaryError("M1 task environment contract is invalid")
    prefix = python.parent.parent
    # The receipt-qualified Conda fonts.conf names a cache inside the sealed
    # package prefix.  Fontconfig can chmod and populate that cache even when
    # the builder made it read-only.  Reuse the receipt-qualified minimal
    # configuration produced by the authoritative builder and keep every
    # mutable cache below the private runtime home instead.
    if not isinstance(runtime_attestation, dict):
        raise base.CanaryError("M1 runtime attestation is unavailable")
    qualification_fontconfig = runtime_attestation.get(
        QUALIFICATION_FONTCONFIG_KEY
    )
    base._validated_absolute_record(
        qualification_fontconfig, "M1 qualification Fontconfig"
    )
    environment = {
        "HOME": str(home),
        "CONDA_PREFIX": str(prefix),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "TMPDIR": str(home),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(home / ".config" / "matplotlib"),
        "FONTCONFIG_FILE": str(qualification_fontconfig["path"]),
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(pythonpath),
        "LD_LIBRARY_PATH": library_path,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(devices[:ranks]),
        "CUDA_CACHE_DISABLE": "1",
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for name, value in ablation_sample._expected_feature_environment(mode).items():
        if value is not None:
            environment[name] = value
    if ranks == 1:
        environment["MEEP_GPU_DEVICE"] = "0"
    else:
        environment["MEEP_GPU_MPI_TRANSPORT"] = "pinned"
        environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
    return environment


def _mode_tasks(tasks, *, repeat, topology):
    selected = {
        task["feature_mode"]: task
        for task in tasks
        if task["repeat"] == repeat and task["topology"] == topology
    }
    if set(selected) != set(MODES):
        raise base.CanaryError("M1 repeat/topology mode inventory differs")
    return selected


def derive_results(tasks: list[dict[str, object]]) -> dict[str, object]:
    plan = task_plan()
    if len(tasks) != len(plan) or any(
        any(task.get(name) != value for name, value in expected.items())
        for task, expected in zip(tasks, plan)
    ):
        raise base.CanaryError("M1 task order or identity differs")
    runtime_artifacts = tasks[0].get("runtime_artifacts")
    if not isinstance(runtime_artifacts, dict) or any(
        task.get("runtime_artifacts") != runtime_artifacts for task in tasks[1:]
    ):
        raise base.CanaryError("M1 tasks loaded different runtime bytes")

    for repeat in range(base.REPEATS):
        for topology in ("gpu1", "gpu2"):
            selected = _mode_tasks(tasks, repeat=repeat, topology=topology)
            signature = selected["off"]["observable_signature"]
            if any(
                selected[mode]["observable_signature"] != signature
                for mode in MODES[1:]
            ):
                raise base.CanaryError(
                    f"M1 {topology} repeat {repeat} observables differ"
                )

    means: dict[str, object] = {}
    spreads: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    for topology in ("gpu1", "gpu2"):
        topology_means = {}
        topology_spreads = {}
        for mode in MODES:
            selected = [
                task
                for task in tasks
                if task["topology"] == topology and task["feature_mode"] == mode
            ]
            topology_means[mode] = {}
            topology_spreads[mode] = {}
            for metric in METRICS:
                values = [float(task[metric]) for task in selected]
                topology_means[mode][metric] = base._mean(values)
                topology_spreads[mode][metric] = max(values) / min(values)
        means[topology] = topology_means
        spreads[topology] = topology_spreads
        comparisons[topology] = {}
        for mode in MODES[1:]:
            comparisons[topology][mode] = {}
            for metric in METRICS:
                baseline = topology_means["off"][metric]
                candidate = topology_means[mode][metric]
                ratio = candidate / baseline
                comparisons[topology][mode][metric] = {
                    "off_mean_seconds": baseline,
                    "candidate_mean_seconds": candidate,
                    "candidate_to_off_ratio": ratio,
                    "improvement_fraction": 1.0 - ratio,
                }

    resource_checks = []
    for repeat in range(base.REPEATS):
        for topology in ("gpu1", "gpu2"):
            selected = _mode_tasks(tasks, repeat=repeat, topology=topology)
            baseline = selected["off"]
            baseline_host = baseline["peak_host_rss_bytes_by_rank"]
            baseline_devices = baseline["telemetry"]["devices"]
            if (
                not isinstance(baseline_host, list)
                or len(baseline_host) != baseline["ranks"]
                or any(value <= 0 for value in baseline_host)
            ):
                raise base.CanaryError("M1 OFF host RSS inventory differs")
            for mode in MODES[1:]:
                candidate = selected[mode]
                candidate_host = candidate["peak_host_rss_bytes_by_rank"]
                candidate_devices = candidate["telemetry"]["devices"]
                if (
                    not isinstance(candidate_host, list)
                    or len(candidate_host) != len(baseline_host)
                    or any(value <= 0 for value in candidate_host)
                    or set(candidate_devices) != set(baseline_devices)
                ):
                    raise base.CanaryError("M1 candidate resource inventory differs")
                devices = {}
                for uuid in sorted(baseline_devices):
                    off_device = baseline_devices[uuid]
                    mode_device = candidate_devices[uuid]
                    off_memory = off_device["peak_attributed_memory_mib"]
                    mode_memory = mode_device["peak_attributed_memory_mib"]
                    if off_memory <= 0 or mode_memory < 0:
                        raise base.CanaryError("M1 attributed GPU memory is invalid")
                    devices[uuid] = {
                        "attributed_peak_memory_ratio": mode_memory / off_memory,
                        "start_temperature_delta_c": abs(
                            mode_device["start_temperature_c"]
                            - off_device["start_temperature_c"]
                        ),
                        "maximum_temperature_c": max(
                            mode_device["maximum_temperature_c"],
                            off_device["maximum_temperature_c"],
                        ),
                        "off_throttle_clean": off_device.get("throttle_clean") is True,
                        "candidate_throttle_clean": (
                            mode_device.get("throttle_clean") is True
                        ),
                    }
                resource_checks.append(
                    {
                        "repeat": repeat,
                        "topology": topology,
                        "mode": mode,
                        "host_peak_rss_ratios_by_rank": [
                            candidate_value / off_value
                            for candidate_value, off_value in zip(
                                candidate_host, baseline_host
                            )
                        ],
                        "host_peak_rss_total_ratio": (
                            candidate["peak_host_rss_bytes_total"]
                            / baseline["peak_host_rss_bytes_total"]
                        ),
                        "devices": devices,
                    }
                )

    spread_pass = all(
        value <= base.MAXIMUM_SAMPLE_SPREAD
        for topology in spreads.values()
        for mode in topology.values()
        for value in mode.values()
    )
    resource_pass = all(
        all(
            ratio <= 1.0 + base.MAXIMUM_MEMORY_INCREASE
            for ratio in check["host_peak_rss_ratios_by_rank"]
        )
        and check["host_peak_rss_total_ratio"]
        <= 1.0 + base.MAXIMUM_MEMORY_INCREASE
        and all(
            device["attributed_peak_memory_ratio"]
            <= 1.0 + base.MAXIMUM_MEMORY_INCREASE
            and device["maximum_temperature_c"] <= base.MAXIMUM_TEMPERATURE_C
            and device["off_throttle_clean"]
            and device["candidate_throttle_clean"]
            for device in check["devices"].values()
        )
        for check in resource_checks
    )
    no_material_regression = all(
        comparisons[topology][mode]["end_to_end_seconds"][
            "candidate_to_off_ratio"
        ]
        <= 1.0 + base.MAXIMUM_BLOCKER_REGRESSION
        for topology in ("gpu1", "gpu2")
        for mode in MODES[1:]
    )
    scaling = {
        mode: means["gpu1"][mode]["fdtd_wall_seconds"]
        / means["gpu2"][mode]["fdtd_wall_seconds"]
        for mode in MODES
    }
    gates = {
        "all_modes_same_binary_and_bit_identical": True,
        "no_candidate_end_to_end_regression_over_3_percent": (
            no_material_regression
        ),
        "two_repeat_spread_at_most_1_20": spread_pass,
        "memory_thermal_and_throttle_contract": resource_pass,
    }
    return {
        "means_seconds": means,
        "sample_max_to_min": spreads,
        "comparisons_to_off": comparisons,
        "gpu2_to_gpu1_fdtd_scaling": scaling,
        "resource_checks": resource_checks,
        "gates": gates,
        "promotion_outcome": (
            "M1_CANDIDATES_ACCEPTED" if all(gates.values()) else "M1_BLOCKED"
        ),
    }


base.task_plan = task_plan
base._omp_threads = _omp_threads
base._environment = _environment
base.derive_results = derive_results
base.ATTESTATION_EXTENSION_BUILDER = _attestation_extension_builder
base.ATTESTATION_EXTENSION_VALIDATOR = _attestation_extension_validator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)

    execute = subparsers.add_parser("run")
    execute.add_argument("--archive", required=True, type=pathlib.Path)
    execute.add_argument("--output", required=True, type=pathlib.Path)
    execute.add_argument("--python", required=True, type=pathlib.Path)
    execute.add_argument("--mpiexec", required=True, type=pathlib.Path)
    execute.add_argument("--pythonpath", required=True, type=pathlib.Path)
    execute.add_argument("--library-path", required=True)
    execute.add_argument("--implementation-commit", required=True)
    execute.add_argument(
        "--runtime-attestation", required=True, type=pathlib.Path
    )
    execute.add_argument("--gpu-devices", default="0,1")
    execute.add_argument("--resume", action="store_true")
    execute.set_defaults(function=base.run)

    replay = subparsers.add_parser("validate")
    replay.add_argument("--output", required=True, type=pathlib.Path)
    replay.set_defaults(function=lambda args: (base.validate(args.output), 0)[1])

    attest = subparsers.add_parser("attest-runtime")
    attest.add_argument("--output", required=True, type=pathlib.Path)
    attest.add_argument("--python", required=True, type=pathlib.Path)
    attest.add_argument("--pythonpath", required=True, type=pathlib.Path)
    attest.add_argument("--library-path", required=True)
    attest.add_argument("--implementation-commit", required=True)
    attest.add_argument("--build-receipt", required=True, type=pathlib.Path)
    attest.add_argument(
        "--m1-validation",
        dest="c1_validation",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--m1-adversarial-audit",
        dest="c1_adversarial_audit",
        required=True,
        type=pathlib.Path,
    )
    attest.add_argument(
        "--m1-validation-gate", required=True, type=pathlib.Path
    )
    attest.add_argument(
        "--m1-adversarial-audit-gate", required=True, type=pathlib.Path
    )
    attest.set_defaults(function=base.attest_runtime)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.CanaryError,
        base.common.WorkloadError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"M1 AuNP ablation FAILED: {error}", file=base.sys.stderr)
        raise SystemExit(1)
