#!/usr/bin/env python3
"""Run and replay the C2 paired GPU1/GPU2 AuNP zero-state canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


REPO = pathlib.Path(__file__).resolve().parents[2]
USER_WORKLOADS = REPO / "scripts" / "user-workloads"
LAB_DIR = pathlib.Path(__file__).resolve().parent
for directory in (USER_WORKLOADS, LAB_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common  # noqa: E402
import run_aunp_zero_state_sample as sample  # noqa: E402


SCHEMA = "gpmeep-v110-m1-aunp-zero-state-canary-v1"
COMPLETE_SCHEMA = "gpmeep-v110-m1-aunp-zero-state-canary-complete-v1"
RUN_SCHEMA = "gpmeep-v110-m1-aunp-zero-state-canary-run-v1"
CHECKPOINT_SCHEMA = "gpmeep-v110-m1-aunp-zero-state-canary-checkpoint-v1"
TELEMETRY_SCHEMA = "gpmeep-v110-m1-aunp-zero-state-telemetry-v1"
RUNTIME_ATTESTATION_SCHEMA = "gpmeep-v110-m1-runtime-attestation-v1"
CANDIDATE_FEATURE = "zero-state-lorentzian-elision"
CONTROLLER_PATH = pathlib.Path(__file__).resolve()
WORKER_PATH = LAB_DIR / "run_aunp_zero_state_sample.py"
ATTESTATION_VALIDATION_KEY = "c1_validation"
ATTESTATION_AUDIT_KEY = "c1_adversarial_audit"
ATTESTATION_EXTENSION_BUILDER = None
ATTESTATION_EXTENSION_VALIDATOR = None
LEGACY_CANDIDATE_FIELD_ALLOWED = True
SUPPORT_SOURCE_PATHS: dict[str, pathlib.Path] = {}
REPEATS = 2
MAXIMUM_SAMPLE_SPREAD = 1.20
MINIMUM_IMPROVEMENT = 0.10
MAXIMUM_BLOCKER_REGRESSION = 0.03
MINIMUM_GPU2_SCALING = 1.10
MAXIMUM_MEMORY_INCREASE = 0.10
MAXIMUM_PAIRED_START_TEMPERATURE_DELTA_C = 5.0
MAXIMUM_TEMPERATURE_C = 80.0
TELEMETRY_INTERVAL_SECONDS = 0.25
MINIMUM_ACTIVE_SM_CLOCK_MHZ = 1000.0
MAXIMUM_POWER_LIMIT_FRACTION = 0.98
IMPLEMENTATION_ALLOWED_DESCENDANTS = (
    "scripts/v110-lab/",
    "scripts/tests/test_v110_aunp_sparse_lorentzian_canary.py",
)


class CanaryError(RuntimeError):
    pass


def _candidate_feature_matches(record: dict[str, Any]) -> bool:
    default = CANDIDATE_FEATURE if LEGACY_CANDIDATE_FIELD_ALLOWED else None
    return record.get("candidate_feature", default) == CANDIDATE_FEATURE


def task_plan() -> list[dict[str, Any]]:
    orders = (
        (("gpu1", "disabled"), ("gpu1", "enabled"),
         ("gpu2", "enabled"), ("gpu2", "disabled")),
        (("gpu1", "enabled"), ("gpu1", "disabled"),
         ("gpu2", "disabled"), ("gpu2", "enabled")),
    )
    result = []
    for repeat, order in enumerate(orders):
        for sequence, (topology, mode) in enumerate(order):
            result.append(
                {
                    "repeat": repeat,
                    "sequence": sequence,
                    "topology": topology,
                    "ranks": 1 if topology == "gpu1" else 2,
                    "feature_mode": mode,
                    "task_id": f"r{repeat}-{sequence}-{topology}-{mode}",
                }
            )
    return result


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
        raise CanaryError("Git probe failed: " + completed.stderr.strip())
    return completed.stdout


def _source_boundary(implementation_commit: str) -> tuple[str, list[str]]:
    head = _git("rev-parse", "HEAD").strip()
    if len(head) != 40:
        raise CanaryError("harness source commit is invalid")
    if _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        raise CanaryError("C2 harness requires a clean source tree")
    ancestor = subprocess.run(
        ["/usr/bin/git", "merge-base", "--is-ancestor",
         implementation_commit, head],
        cwd=REPO,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise CanaryError("implementation commit is not an ancestor of harness HEAD")
    changed = _git(
        "diff", "--name-only", f"{implementation_commit}..{head}"
    ).splitlines()
    disallowed = [
        name for name in changed
        if not _implementation_descendant_allowed(name)
    ]
    if disallowed:
        raise CanaryError(
            "core source changed after implementation boundary: "
            + ", ".join(disallowed)
        )
    return head, changed


def _implementation_descendant_allowed(name: str) -> bool:
    return any(
        name.startswith(allowed) if allowed.endswith("/")
        else name == allowed
        for allowed in IMPLEMENTATION_ALLOWED_DESCENDANTS
    )


def _canonical_id(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_absolute_record(record: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "size_bytes", "sha256"}
        or not isinstance(record.get("path"), str)
        or not pathlib.Path(record["path"]).is_absolute()
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] <= 0
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise CanaryError(f"C2 {label} absolute file record differs")
    path = common.verify_file(
        pathlib.Path(record["path"]), record["sha256"], f"C2 {label}"
    )
    if path.stat().st_size != record["size_bytes"]:
        raise CanaryError(f"C2 {label} size differs")
    return record


def _load_runtime_attestation(
    path: pathlib.Path, source_commit: str, implementation_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = pathlib.Path(path).resolve(strict=True)
    attestation = _load_stable_json(path, "C2 runtime attestation")
    if not isinstance(attestation, dict):
        raise CanaryError("C2 runtime attestation is invalid")
    unsigned = dict(attestation)
    receipt_id = unsigned.pop("attestation_id", None)
    runtime = attestation.get("runtime_artifacts")
    if (
        attestation.get("schema") != RUNTIME_ATTESTATION_SCHEMA
        or attestation.get(
            "candidate_feature",
            CANDIDATE_FEATURE if LEGACY_CANDIDATE_FIELD_ALLOWED else None,
        ) != CANDIDATE_FEATURE
        or attestation.get("source_commit") != source_commit
        or attestation.get("implementation_commit") != implementation_commit
        or receipt_id != _canonical_id(unsigned)
        or not isinstance(runtime, dict)
        or set(runtime) != {"python", "extension", "libmeep"}
    ):
        raise CanaryError("C2 runtime attestation contract differs")
    for name, record in runtime.items():
        _validated_absolute_record(record, f"attested runtime {name}")
    for name in (ATTESTATION_VALIDATION_KEY, ATTESTATION_AUDIT_KEY):
        _validated_absolute_record(
            attestation.get(name), f"runtime attestation {name}"
        )
    if ATTESTATION_EXTENSION_VALIDATOR is not None:
        ATTESTATION_EXTENSION_VALIDATOR(
            attestation, source_commit, implementation_commit
        )
    return attestation, common.absolute_file_record(path, "C2 runtime attestation")


def attest_runtime(args: argparse.Namespace) -> int:
    output = pathlib.Path(os.path.abspath(os.fspath(args.output)))
    if output.exists() or output.is_symlink():
        raise CanaryError("C2 runtime attestation output must be absent")
    if output.parent.resolve(strict=True) != output.parent:
        raise CanaryError("C2 runtime attestation parent is not canonical")
    source_commit, harness_changes = _source_boundary(args.implementation_commit)
    python = args.python.resolve(strict=True)
    pythonpath = args.pythonpath.resolve(strict=True)
    probe = (
        "import json,sys;"
        f"sys.path.insert(0,{str(USER_WORKLOADS)!r});"
        "import common,meep as mp;"
        "print('C2_RUNTIME_JSON='+json.dumps("
        "common.runtime_artifact_records(mp),allow_nan=False,sort_keys=True,"
        "separators=(',',':')),flush=True)"
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        cwd=REPO,
        env={
            "HOME": str(output.parent),
            "PATH": f"{python.parent}:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(pythonpath),
            "LD_LIBRARY_PATH": args.library_path,
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise CanaryError(
            "C2 runtime probe failed: " + completed.stderr.strip()
        )
    try:
        payloads = [
            line.removeprefix("C2_RUNTIME_JSON=")
            for line in completed.stdout.splitlines()
            if line.startswith("C2_RUNTIME_JSON=")
        ]
        if len(payloads) != 1:
            raise json.JSONDecodeError(
                "runtime marker inventory differs", completed.stdout, 0
            )
        runtime = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise CanaryError("C2 runtime probe output is invalid") from exc
    if not isinstance(runtime, dict) or set(runtime) != {
        "python", "extension", "libmeep"
    }:
        raise CanaryError("C2 runtime probe artifact inventory differs")
    for name, record in runtime.items():
        _validated_absolute_record(record, f"runtime probe {name}")
    validation_record = common.absolute_file_record(
        args.c1_validation.resolve(strict=True), "C1 validation report"
    )
    audit_record = common.absolute_file_record(
        args.c1_adversarial_audit.resolve(strict=True), "C1 audit report"
    )
    extension = {}
    if ATTESTATION_EXTENSION_BUILDER is not None:
        extension = ATTESTATION_EXTENSION_BUILDER(
            args,
            source_commit,
            args.implementation_commit,
            runtime,
            validation_record,
            audit_record,
        )
        if not isinstance(extension, dict):
            raise CanaryError("C2 runtime attestation extension is invalid")
    attestation = {
        "schema": RUNTIME_ATTESTATION_SCHEMA,
        "source_commit": source_commit,
        "implementation_commit": args.implementation_commit,
        "implementation_to_harness_changes": harness_changes,
        "candidate_feature": CANDIDATE_FEATURE,
        "runtime_artifacts": runtime,
        ATTESTATION_VALIDATION_KEY: validation_record,
        ATTESTATION_AUDIT_KEY: audit_record,
        "development_only": True,
        "valid_for_release_claim": False,
    }
    if set(attestation) & set(extension):
        raise CanaryError("C2 runtime attestation extension collides")
    attestation.update(extension)
    attestation["attestation_id"] = _canonical_id(attestation)
    common.atomic_write_json(output, attestation)
    _load_runtime_attestation(
        output, source_commit, args.implementation_commit
    )
    print(
        "C2 runtime attestation COMPLETE: " + attestation["attestation_id"],
        flush=True,
    )
    return 0


def _safe_output(path: pathlib.Path, resume: bool) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path.is_symlink():
        raise CanaryError("C2 output must be absolute and non-symlink")
    parent = path.parent.resolve(strict=True)
    if parent != path.parent:
        raise CanaryError("C2 output parent is not canonical")
    if resume:
        if path.resolve(strict=True) != path or not path.is_dir():
            raise CanaryError("C2 resume output is unavailable")
    else:
        if path.exists():
            raise CanaryError("fresh C2 output already exists")
        path.mkdir(mode=0o755)
    return path


def _parse_optional_number(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "[n/a]", "nan"}:
        return None
    number = float(stripped)
    return number if math.isfinite(number) else None


def _parse_active(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "active":
        return True
    if normalized == "not active":
        return False
    raise CanaryError(f"invalid NVIDIA throttle state: {value!r}")


def _parse_clock_event_mask(value: str) -> int:
    normalized = value.strip().lower()
    if not normalized.startswith("0x"):
        raise CanaryError(f"invalid NVIDIA clock-event mask: {value!r}")
    try:
        result = int(normalized, 16)
    except ValueError as exc:
        raise CanaryError(
            f"invalid NVIDIA clock-event mask: {value!r}"
        ) from exc
    if result < 0:
        raise CanaryError(f"invalid NVIDIA clock-event mask: {value!r}")
    return result


def parse_nvidia_smi(payload: str) -> list[dict[str, Any]]:
    records = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 19:
            raise CanaryError("nvidia-smi telemetry column count differs")
        (
            index,
            uuid,
            temperature,
            temperature_limit,
            memory,
            power,
            power_limit,
            utilization,
            clock,
            pstate,
            active_mask,
            software_power_cap,
            software_thermal,
            hardware_thermal,
            gpu_idle,
            applications_clocks_setting,
            hardware_slowdown,
            hardware_power_brake,
            sync_boost,
        ) = columns
        if not index.isdigit() or not uuid:
            raise CanaryError("nvidia-smi telemetry identity is invalid")
        numeric = {
            "temperature_c": _parse_optional_number(temperature),
            "temperature_limit_c": _parse_optional_number(temperature_limit),
            "memory_used_mib": _parse_optional_number(memory),
            "power_w": _parse_optional_number(power),
            "enforced_power_limit_w": _parse_optional_number(power_limit),
            "utilization_percent": _parse_optional_number(utilization),
            "sm_clock_mhz": _parse_optional_number(clock),
            "pstate": pstate,
            "active_clock_event_mask": active_mask,
            "active_clock_event_mask_value": _parse_clock_event_mask(active_mask),
            "software_power_cap_active": _parse_active(software_power_cap),
            "software_thermal_slowdown_active": _parse_active(software_thermal),
            "hardware_thermal_slowdown_active": _parse_active(hardware_thermal),
            "gpu_idle_active": _parse_active(gpu_idle),
            "applications_clocks_setting_active":
                _parse_active(applications_clocks_setting),
            "hardware_slowdown_active": _parse_active(hardware_slowdown),
            "hardware_power_brake_slowdown_active":
                _parse_active(hardware_power_brake),
            "sync_boost_active": _parse_active(sync_boost),
        }
        if numeric["temperature_c"] is None or numeric["memory_used_mib"] is None:
            raise CanaryError("nvidia-smi lacks required temperature or memory")
        records.append({"index": int(index), "uuid": uuid, **numeric})
    if not records:
        raise CanaryError("nvidia-smi returned no telemetry rows")
    return records


def _normalize_gpu_uuid(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise CanaryError(f"invalid GPU UUID: {value!r}")
    return normalized


def summarize_telemetry(
    samples: list[dict[str, Any]], expected_device_count: int
) -> dict[str, Any]:
    if not isinstance(samples, list) or len(samples) < 2:
        raise CanaryError("C2 telemetry has fewer than two samples")
    if expected_device_count not in (1, 2):
        raise CanaryError("C2 telemetry expected device count is invalid")
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    expected_uuids: set[str] | None = None
    previous_elapsed = -1.0
    for snapshot in samples:
        if not isinstance(snapshot, dict):
            raise CanaryError("C2 telemetry snapshot is invalid")
        elapsed = float(snapshot.get("elapsed_seconds", -1.0))
        records = snapshot.get("devices")
        if (
            not math.isfinite(elapsed)
            or elapsed <= previous_elapsed
            or not isinstance(records, list)
            or len(records) != expected_device_count
        ):
            raise CanaryError("C2 telemetry snapshot contract differs")
        previous_elapsed = elapsed
        uuids = [record.get("uuid") for record in records
                 if isinstance(record, dict)]
        if len(uuids) != expected_device_count or len(set(uuids)) != len(uuids):
            raise CanaryError("C2 telemetry snapshot device inventory differs")
        current_uuids = set(uuids)
        if expected_uuids is None:
            expected_uuids = current_uuids
        elif current_uuids != expected_uuids:
            raise CanaryError("C2 telemetry device identity changed")
        for record in records:
            utilization = record.get("utilization_percent")
            if (
                isinstance(utilization, bool)
                or not isinstance(utilization, (int, float))
                or not math.isfinite(float(utilization))
                or not 0.0 <= float(utilization) <= 100.0
            ):
                raise CanaryError("C2 GPU utilization telemetry is invalid")
            by_uuid.setdefault(record["uuid"], []).append(record)
    if len(by_uuid) != expected_device_count:
        raise CanaryError("C2 telemetry device count differs")
    devices = {}
    for uuid, records in sorted(by_uuid.items()):
        baseline_memory = float(records[0]["memory_used_mib"])
        active_records = [
            record for record in records[1:]
            if (record["utilization_percent"] or 0.0) > 0.0
        ]
        if not active_records:
            raise CanaryError(
                f"C2 telemetry observed no active workload sample for {uuid}"
            )
        peak_memory = max(record["memory_used_mib"] for record in records)
        power_fractions = [
            record["power_w"] / record["enforced_power_limit_w"]
            for record in active_records
            if record["power_w"] is not None
            and record["enforced_power_limit_w"] is not None
            and record["enforced_power_limit_w"] > 0
        ]
        power_telemetry_complete = len(power_fractions) == len(active_records)
        active_clocks = [
            record["sm_clock_mhz"] for record in active_records
            if record["sm_clock_mhz"] is not None
        ]
        clock_telemetry_complete = len(active_clocks) == len(active_records)
        active_pstates = sorted({record["pstate"] for record in active_records})
        throttle_clean = (
            clock_telemetry_complete
            and min(active_clocks) >= MINIMUM_ACTIVE_SM_CLOCK_MHZ
            and all(pstate in {"P0", "P1", "P2"} for pstate in active_pstates)
            and not any(
                record["software_power_cap_active"]
                or record["software_thermal_slowdown_active"]
                or record["hardware_thermal_slowdown_active"]
                or record["gpu_idle_active"]
                or record["applications_clocks_setting_active"]
                or record["hardware_slowdown_active"]
                or record["hardware_power_brake_slowdown_active"]
                or record["sync_boost_active"]
                for record in active_records
            )
            and not any(
                record["active_clock_event_mask_value"] != 0
                for record in active_records
            )
            and power_telemetry_complete
            and max(power_fractions) <= MAXIMUM_POWER_LIMIT_FRACTION
            and max(record["temperature_c"] for record in records)
                <= MAXIMUM_TEMPERATURE_C
        )
        devices[uuid] = {
            "index": records[0]["index"],
            "start_temperature_c": records[0]["temperature_c"],
            "end_temperature_c": records[-1]["temperature_c"],
            "maximum_temperature_c": max(
                record["temperature_c"] for record in records
            ),
            "baseline_memory_used_mib": baseline_memory,
            "peak_memory_used_mib": peak_memory,
            "peak_attributed_memory_mib": peak_memory - baseline_memory,
            "maximum_power_w": max(
                (record["power_w"] for record in records
                 if record["power_w"] is not None),
                default=None,
            ),
            "minimum_sm_clock_mhz": min(
                (record["sm_clock_mhz"] for record in records
                 if record["sm_clock_mhz"] is not None),
                default=None,
            ),
            "minimum_active_sm_clock_mhz": min(active_clocks, default=None),
            "maximum_active_power_limit_fraction": max(
                power_fractions, default=None
            ),
            "power_telemetry_complete": power_telemetry_complete,
            "clock_telemetry_complete": clock_telemetry_complete,
            "active_clock_event_mask_values": sorted({
                record["active_clock_event_mask_value"]
                for record in active_records
            }),
            "active_pstates": active_pstates,
            "software_power_cap_observed": any(
                record["software_power_cap_active"] for record in active_records
            ),
            "software_thermal_slowdown_observed": any(
                record["software_thermal_slowdown_active"]
                for record in active_records
            ),
            "hardware_thermal_slowdown_observed": any(
                record["hardware_thermal_slowdown_active"]
                for record in active_records
            ),
            "gpu_idle_observed": any(
                record["gpu_idle_active"] for record in active_records
            ),
            "applications_clocks_setting_observed": any(
                record["applications_clocks_setting_active"]
                for record in active_records
            ),
            "hardware_slowdown_observed": any(
                record["hardware_slowdown_active"] for record in active_records
            ),
            "hardware_power_brake_slowdown_observed": any(
                record["hardware_power_brake_slowdown_active"]
                for record in active_records
            ),
            "sync_boost_observed": any(
                record["sync_boost_active"] for record in active_records
            ),
            "active_sample_count": len(active_records),
            "throttle_clean": throttle_clean,
        }
    return {
        "sample_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
        "sample_count": len(samples),
        "devices": devices,
    }


class NvidiaSampler:
    def __init__(self, devices: list[str]):
        self.devices = devices
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = time.monotonic()

    def _query(self) -> None:
        command = [
            "/usr/bin/nvidia-smi",
            "-i", ",".join(self.devices),
            "--query-gpu=index,uuid,temperature.gpu,temperature.gpu.tlimit,"
            "memory.used,power.draw,enforced.power.limit,utilization.gpu,"
            "clocks.sm,pstate,clocks_event_reasons.active,"
            "clocks_event_reasons.sw_power_cap,"
            "clocks_event_reasons.sw_thermal_slowdown,"
            "clocks_event_reasons.hw_thermal_slowdown,"
            "clocks_event_reasons.gpu_idle,"
            "clocks_event_reasons.applications_clocks_setting,"
            "clocks_event_reasons.hw_slowdown,"
            "clocks_event_reasons.hw_power_brake_slowdown,"
            "clocks_event_reasons.sync_boost",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise CanaryError(
                "nvidia-smi failed: " + completed.stderr.strip()
            )
        self.samples.append(
            {
                "elapsed_seconds": time.monotonic() - self.started,
                "devices": parse_nvidia_smi(completed.stdout),
            }
        )

    def _run(self) -> None:
        while not self.stop_event.wait(TELEMETRY_INTERVAL_SECONDS):
            try:
                self._query()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    def __enter__(self) -> "NvidiaSampler":
        # Synchronous pre-launch sample: subprocess execution cannot begin
        # until an ambient memory/temperature baseline is durably in memory.
        self._query()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            self.errors.append("telemetry thread did not terminate")
        if not self.errors:
            try:
                self._query()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def summary(self) -> dict[str, Any]:
        if self.errors:
            raise CanaryError("; ".join(self.errors))
        return summarize_telemetry(self.samples, len(self.devices))


def _omp_threads(task: dict[str, Any]) -> int:
    del task
    omp_threads = getattr(sample, "OMP_THREADS", 1)
    if type(omp_threads) is not int or omp_threads < 1:
        raise CanaryError("candidate OpenMP thread count is invalid")
    return omp_threads


def _mpi_map_by(task: dict[str, Any]) -> str:
    # Open MPI's `core:PE=N` is contradictory for N > 1 because the selected
    # mapping object contains only one CPU.  Mapping by slots lets Open MPI
    # allocate N adjacent processing elements before binding the rank to
    # those cores, and is equivalent to the former mapping when N == 1.
    return f"slot:PE={_omp_threads(task)}"


def _environment(
    task: dict[str, Any], python: pathlib.Path, pythonpath: pathlib.Path,
    library_path: str, devices: list[str], home: pathlib.Path,
    runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, str]:
    del runtime_attestation
    omp_threads = _omp_threads(task)
    environment = {
        "HOME": str(home),
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(pythonpath),
        "LD_LIBRARY_PATH": library_path,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(devices[: task["ranks"]]),
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
        "OMP_NUM_THREADS": str(omp_threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    if task["ranks"] == 1:
        environment["MEEP_GPU_DEVICE"] = "0"
    else:
        environment["MEEP_GPU_MPI_TRANSPORT"] = "pinned"
        environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
    if task["feature_mode"] == "enabled":
        environment[sample.FEATURE_ENABLE] = "1"
    else:
        environment[sample.FEATURE_DISABLE] = "1"
    return environment


def _summary_signature(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records = summary["rank_records"][0]["records"]
    return [
        {
            "stage": record["stage"],
            "timestep": record["timestep"],
            "timestep_delta": record["timestep_delta"],
            "meep_time": record["meep_time"],
            "observables": record["observables"],
        }
        for record in records
    ]


def _load_stable_json(
    path: pathlib.Path, label: str, expected: dict[str, Any] | None = None
) -> Any:
    with common.StableFile(path, label, expected=expected) as stable:
        with stable.file_object() as handle:
            return json.load(handle)


def _require_file_record(record: Any, expected_path: str, label: str) -> None:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "size_bytes", "sha256"}
        or record.get("path") != expected_path
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise CanaryError(f"C2 {label} file record differs")


def _stable_file_record(
    path: pathlib.Path, root: pathlib.Path, label: str
) -> dict[str, Any]:
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CanaryError(f"C2 {label} escapes evidence root") from exc
    with common.StableFile(path, label) as stable:
        return {
            "path": relative.as_posix(),
            "size_bytes": stable.initial_stat.st_size,
            "sha256": stable.sha256,
        }


def _task_paths(
    output: pathlib.Path, task: dict[str, Any]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    sample_output = output / "samples" / task["task_id"]
    log_path = output / "logs" / f"{task['task_id']}.log"
    telemetry_path = output / "telemetry" / f"{task['task_id']}.json"
    return sample_output, log_path, telemetry_path


def _task_from_evidence(
    *, task: dict[str, Any], output: pathlib.Path, source_commit: str,
    implementation_commit: str, gpu_devices: list[str] | None = None,
    expected_runtime_artifacts: dict[str, Any] | None = None
) -> dict[str, Any]:
    sample_output, log_path, telemetry_path = _task_paths(output, task)
    summary = sample.validate_sample(sample_output)
    raw_telemetry = _load_stable_json(
        telemetry_path, f"C2 task {task['task_id']} telemetry"
    )
    if (
        not isinstance(raw_telemetry, dict)
        or raw_telemetry.get("schema") != TELEMETRY_SCHEMA
        or raw_telemetry.get("task_id") != task["task_id"]
        or raw_telemetry.get("ranks") != task["ranks"]
        or raw_telemetry.get("feature_mode") != task["feature_mode"]
        or not _candidate_feature_matches(raw_telemetry)
        or not isinstance(raw_telemetry.get("selected_gpu_selectors"), list)
        or len(raw_telemetry["selected_gpu_selectors"]) != task["ranks"]
        or (
            gpu_devices is not None
            and raw_telemetry["selected_gpu_selectors"]
                != gpu_devices[:task["ranks"]]
        )
    ):
        raise CanaryError("C2 task telemetry binding differs")
    launcher_wall_seconds = float(
        raw_telemetry.get("launcher_wall_seconds", -1.0)
    )
    if not math.isfinite(launcher_wall_seconds) or launcher_wall_seconds <= 0:
        raise CanaryError("C2 task launcher wall time is invalid")
    telemetry = summarize_telemetry(
        raw_telemetry.get("samples"), task["ranks"]
    )
    if raw_telemetry.get("summary") != telemetry:
        raise CanaryError("C2 task telemetry summary was not re-derived")
    if (
        summary["source_commit"] != source_commit
        or summary["implementation_commit"] != implementation_commit
        or summary["feature_mode"] != task["feature_mode"]
        or not _candidate_feature_matches(summary)
        or summary["mpi_size"] != task["ranks"]
    ):
        raise CanaryError("C2 task summary binding differs")
    identifiers = []
    for rank in summary["rank_records"]:
        phase_identifiers = {
            record["device_identifier"] for record in rank["records"]
        }
        if len(phase_identifiers) != 1:
            raise CanaryError("C2 rank device identity changed between phases")
        identifiers.append(next(iter(phase_identifiers)))
    if len(set(identifiers)) != task["ranks"]:
        raise CanaryError("C2 task did not select distinct devices")
    selected_uuid_set = {_normalize_gpu_uuid(value) for value in identifiers}
    telemetry_uuid_set = {
        _normalize_gpu_uuid(value) for value in telemetry["devices"]
    }
    if selected_uuid_set != telemetry_uuid_set:
        raise CanaryError("C2 worker and telemetry GPU UUIDs differ")
    runtime_artifacts = summary["rank_records"][0]["runtime_artifacts"]
    if (
        expected_runtime_artifacts is not None
        and runtime_artifacts != expected_runtime_artifacts
    ):
        raise CanaryError("C2 task runtime differs from C1 attestation")
    return {
        **task,
        "device_identifiers": identifiers,
        "initialization_wall_seconds": float(
            summary["initialization_wall_seconds"]
        ),
        "fdtd_wall_seconds": float(summary["fdtd_wall_seconds"]),
        "end_to_end_seconds": float(summary["workload_end_to_end_seconds"]),
        "launcher_wall_seconds": launcher_wall_seconds,
        "peak_host_rss_bytes_by_rank": summary["peak_host_rss_bytes_by_rank"],
        "peak_host_rss_bytes_total": int(summary["peak_host_rss_bytes_total"]),
        "peak_host_rss_bytes": int(summary["peak_host_rss_bytes"]),
        "feature_counter_totals": summary["feature_counter_totals"],
        "observable_signature": _summary_signature(summary),
        "runtime_artifacts": runtime_artifacts,
        "telemetry": telemetry,
        "sample_summary": _stable_file_record(
            sample_output / "summary.json", output, "C2 sample summary"
        ),
        "sample_complete": _stable_file_record(
            sample_output / "COMPLETE", output, "C2 sample COMPLETE"
        ),
        "log": _stable_file_record(log_path, output, "C2 task log"),
        "telemetry_file": _stable_file_record(
            telemetry_path, output, "C2 task telemetry"
        ),
    }


def _mean(values: list[float]) -> float:
    if len(values) != REPEATS or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise CanaryError("C2 aggregate requires two finite positive samples")
    return statistics.fmean(values)


def derive_results(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    plan = task_plan()
    if len(tasks) != len(plan) or any(
        any(task.get(name) != value for name, value in expected.items())
        for task, expected in zip(tasks, plan)
    ):
        raise CanaryError("C2 task order or identity differs")

    runtime_artifacts = tasks[0].get("runtime_artifacts")
    if not isinstance(runtime_artifacts, dict) or any(
        task.get("runtime_artifacts") != runtime_artifacts for task in tasks[1:]
    ):
        raise CanaryError("C2 tasks loaded different runtime bytes")

    for repeat in range(REPEATS):
        for topology in ("gpu1", "gpu2"):
            pair = [
                task for task in tasks
                if task["repeat"] == repeat and task["topology"] == topology
            ]
            if {task["feature_mode"] for task in pair} != {"enabled", "disabled"}:
                raise CanaryError("C2 A/B pair inventory differs")
            if pair[0]["observable_signature"] != pair[1]["observable_signature"]:
                raise CanaryError(
                    f"C2 {topology} repeat {repeat} A/B observables are not bit-identical"
                )

    means: dict[str, dict[str, dict[str, float]]] = {}
    spread: dict[str, dict[str, dict[str, float]]] = {}
    for topology in ("gpu1", "gpu2"):
        means[topology] = {}
        spread[topology] = {}
        for mode in ("disabled", "enabled"):
            selected = [
                task for task in tasks
                if task["topology"] == topology and task["feature_mode"] == mode
            ]
            means[topology][mode] = {}
            spread[topology][mode] = {}
            for name in ("fdtd_wall_seconds", "end_to_end_seconds"):
                values = [float(task[name]) for task in selected]
                means[topology][mode][name] = _mean(values)
                spread[topology][mode][name] = max(values) / min(values)

    comparisons: dict[str, Any] = {}
    blocker = False
    for topology in ("gpu1", "gpu2"):
        comparisons[topology] = {}
        for name in ("fdtd_wall_seconds", "end_to_end_seconds"):
            disabled = means[topology]["disabled"][name]
            enabled = means[topology]["enabled"][name]
            ratio = enabled / disabled
            blocker = blocker or ratio > 1.0 + MAXIMUM_BLOCKER_REGRESSION
            comparisons[topology][name] = {
                "disabled_mean_seconds": disabled,
                "enabled_mean_seconds": enabled,
                "enabled_to_disabled_ratio": ratio,
                "improvement_fraction": 1.0 - ratio,
            }

    geomean = {}
    for name in ("fdtd_wall_seconds", "end_to_end_seconds"):
        ratios = [comparisons[topology][name]["enabled_to_disabled_ratio"]
                  for topology in ("gpu1", "gpu2")]
        ratio = math.sqrt(ratios[0] * ratios[1])
        geomean[name] = {
            "enabled_to_disabled_ratio": ratio,
            "improvement_fraction": 1.0 - ratio,
            "minimum_improvement_fraction": MINIMUM_IMPROVEMENT,
        }

    memory_checks = []
    thermal_checks = []
    for repeat in range(REPEATS):
        for topology in ("gpu1", "gpu2"):
            pair = {
                task["feature_mode"]: task for task in tasks
                if task["repeat"] == repeat and task["topology"] == topology
            }
            disabled_host_by_rank = pair["disabled"].get(
                "peak_host_rss_bytes_by_rank"
            )
            enabled_host_by_rank = pair["enabled"].get(
                "peak_host_rss_bytes_by_rank"
            )
            if (
                not isinstance(disabled_host_by_rank, list)
                or not isinstance(enabled_host_by_rank, list)
                or len(disabled_host_by_rank) != pair["disabled"]["ranks"]
                or len(enabled_host_by_rank) != pair["enabled"]["ranks"]
                or len(disabled_host_by_rank) != len(enabled_host_by_rank)
                or any(value <= 0 for value in disabled_host_by_rank)
                or any(value <= 0 for value in enabled_host_by_rank)
            ):
                raise CanaryError("C2 A/B host RSS rank inventory differs")
            host_ratios_by_rank = [
                enabled / disabled
                for enabled, disabled in zip(
                    enabled_host_by_rank, disabled_host_by_rank
                )
            ]
            disabled_host_total = pair["disabled"].get(
                "peak_host_rss_bytes_total"
            )
            enabled_host_total = pair["enabled"].get(
                "peak_host_rss_bytes_total"
            )
            if (
                disabled_host_total != sum(disabled_host_by_rank)
                or enabled_host_total != sum(enabled_host_by_rank)
            ):
                raise CanaryError("C2 A/B host RSS total was not re-derived")
            host_total_ratio = enabled_host_total / disabled_host_total
            device_checks = {}
            disabled_devices = pair["disabled"]["telemetry"]["devices"]
            enabled_devices = pair["enabled"]["telemetry"]["devices"]
            if set(disabled_devices) != set(enabled_devices):
                raise CanaryError("C2 A/B pair telemetry devices differ")
            for uuid in sorted(disabled_devices):
                disabled_device = disabled_devices[uuid]
                enabled_device = enabled_devices[uuid]
                disabled_attributed = disabled_device[
                    "peak_attributed_memory_mib"
                ]
                enabled_attributed = enabled_device[
                    "peak_attributed_memory_mib"
                ]
                if disabled_attributed <= 0 or enabled_attributed < 0:
                    raise CanaryError("C2 attributed GPU memory is invalid")
                device_checks[uuid] = {
                    "attributed_peak_memory_ratio":
                        enabled_attributed / disabled_attributed,
                    "start_temperature_delta_c": abs(
                        enabled_device["start_temperature_c"]
                        - disabled_device["start_temperature_c"]
                    ),
                    "maximum_temperature_c": max(
                        enabled_device["maximum_temperature_c"],
                        disabled_device["maximum_temperature_c"],
                    ),
                    "disabled_throttle_clean":
                        disabled_device.get("throttle_clean") is True,
                    "enabled_throttle_clean":
                        enabled_device.get("throttle_clean") is True,
                }
            memory_checks.append(
                {
                    "repeat": repeat,
                    "topology": topology,
                    "host_peak_rss_ratios_by_rank": host_ratios_by_rank,
                    "host_peak_rss_total_ratio": host_total_ratio,
                    "devices": device_checks,
                }
            )
            thermal_checks.extend(
                {
                    "repeat": repeat,
                    "topology": topology,
                    "uuid": uuid,
                    "start_temperature_delta_c": values[
                        "start_temperature_delta_c"
                    ],
                    "maximum_temperature_c": values["maximum_temperature_c"],
                    "disabled_throttle_clean": values[
                        "disabled_throttle_clean"
                    ],
                    "enabled_throttle_clean": values[
                        "enabled_throttle_clean"
                    ],
                }
                for uuid, values in device_checks.items()
            )

    memory_pass = all(
        all(
            ratio <= 1.0 + MAXIMUM_MEMORY_INCREASE
            for ratio in check["host_peak_rss_ratios_by_rank"]
        )
        and check["host_peak_rss_total_ratio"] <= 1.0 + MAXIMUM_MEMORY_INCREASE
        and all(
            device["attributed_peak_memory_ratio"]
                <= 1.0 + MAXIMUM_MEMORY_INCREASE
            for device in check["devices"].values()
        )
        for check in memory_checks
    )
    thermal_pass = all(
        check["start_temperature_delta_c"]
            <= MAXIMUM_PAIRED_START_TEMPERATURE_DELTA_C
        and check["maximum_temperature_c"] <= MAXIMUM_TEMPERATURE_C
        and check["disabled_throttle_clean"]
        and check["enabled_throttle_clean"]
        for check in thermal_checks
    )
    spread_pass = all(
        value <= MAXIMUM_SAMPLE_SPREAD
        for topology in spread.values()
        for mode in topology.values()
        for value in mode.values()
    )
    scaling = (
        means["gpu1"]["enabled"]["fdtd_wall_seconds"]
        / means["gpu2"]["enabled"]["fdtd_wall_seconds"]
    )
    performance_pass = (
        not blocker
        and all(
            value["improvement_fraction"] >= MINIMUM_IMPROVEMENT
            for value in geomean.values()
        )
        and scaling >= MINIMUM_GPU2_SCALING
        and memory_pass
        and thermal_pass
        and spread_pass
    )
    return {
        "means_seconds": means,
        "sample_max_to_min": spread,
        "comparisons": comparisons,
        "geomean": geomean,
        "enabled_gpu2_to_gpu1_fdtd_scaling": scaling,
        "memory_checks": memory_checks,
        "thermal_checks": thermal_checks,
        "gates": {
            "no_topology_metric_regression_over_3_percent": not blocker,
            "geomean_fdtd_improvement_at_least_10_percent":
                geomean["fdtd_wall_seconds"]["improvement_fraction"]
                >= MINIMUM_IMPROVEMENT,
            "geomean_end_to_end_improvement_at_least_10_percent":
                geomean["end_to_end_seconds"]["improvement_fraction"]
                >= MINIMUM_IMPROVEMENT,
            "enabled_gpu2_scaling_at_least_1_1":
                scaling >= MINIMUM_GPU2_SCALING,
            "host_rank_total_and_each_device_memory_increase_at_most_10_percent":
                memory_pass,
            "paired_thermal_conditions_comparable_and_throttle_clean":
                thermal_pass,
            "two_repeat_spread_at_most_1_20": spread_pass,
        },
        "promotion_outcome":
            "POSITIVE_CANARY" if performance_pass else "NO_PROMOTION",
    }


def _run_task(
    *, task: dict[str, Any], output: pathlib.Path, archive: pathlib.Path,
    python: pathlib.Path, mpiexec: pathlib.Path, pythonpath: pathlib.Path,
    library_path: str, devices: list[str], source_commit: str,
    implementation_commit: str, home: pathlib.Path,
    expected_runtime_artifacts: dict[str, Any],
    runtime_attestation: dict[str, Any],
) -> dict[str, Any]:
    sample_output, log_path, telemetry_path = _task_paths(output, task)
    if sample_output.exists() or log_path.exists() or telemetry_path.exists():
        raise CanaryError("C2 task has pre-existing or partial output")
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(mpiexec), "--bind-to", "core", "--map-by",
        _mpi_map_by(task),
        "--report-bindings", "-n", str(task["ranks"]),
        str(python), "-u", str(WORKER_PATH),
        "--archive", str(archive), "--output", str(sample_output),
        "--feature-mode", task["feature_mode"],
        "--expected-source-commit", source_commit,
        "--implementation-commit", implementation_commit,
    ]
    started = time.perf_counter()
    with log_path.open("xb") as log, NvidiaSampler(
        devices[: task["ranks"]]
    ) as sampler:
        completed = subprocess.run(
            command,
            cwd=output,
            env=_environment(
                task,
                python,
                pythonpath,
                library_path,
                devices,
                home,
                runtime_attestation,
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20 * 60,
        )
        log.flush()
        os.fsync(log.fileno())
    launcher_wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-40:]
        )
        raise CanaryError(
            f"C2 task {task['task_id']} failed with {completed.returncode}:\n{tail}"
        )
    telemetry = {
        "schema": TELEMETRY_SCHEMA,
        "task_id": task["task_id"],
        "ranks": task["ranks"],
        "feature_mode": task["feature_mode"],
        "candidate_feature": CANDIDATE_FEATURE,
        "selected_gpu_selectors": devices[: task["ranks"]],
        "launcher_wall_seconds": launcher_wall_seconds,
        "samples": sampler.samples,
        "summary": sampler.summary(),
    }
    common.atomic_write_json(telemetry_path, telemetry)
    return _task_from_evidence(
        task=task,
        output=output,
        source_commit=source_commit,
        implementation_commit=implementation_commit,
        gpu_devices=devices,
        expected_runtime_artifacts=expected_runtime_artifacts,
    )


def _resume_tasks(
    *, output: pathlib.Path, source_commit: str, implementation_commit: str,
    gpu_devices: list[str] | None = None,
    expected_runtime_artifacts: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    plan = task_plan()
    checkpoint_path = output / "CHECKPOINT.json"
    if checkpoint_path.exists():
        checkpoint = _load_stable_json(checkpoint_path, "C2 checkpoint")
        stored_tasks = (
            checkpoint.get("tasks") if isinstance(checkpoint, dict) else None
        )
        completed_ids = (
            checkpoint.get("completed_task_ids")
            if isinstance(checkpoint, dict) else None
        )
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or not isinstance(stored_tasks, list)
            or not isinstance(completed_ids, list)
            or len(stored_tasks) > len(plan)
            or completed_ids
                != [task["task_id"] for task in plan[:len(stored_tasks)]]
            or completed_ids != [task.get("task_id") for task in stored_tasks]
        ):
            raise CanaryError("C2 checkpoint is not an exact task-plan prefix")
    else:
        stored_tasks = []
    rederived = []
    for expected, stored in zip(plan, stored_tasks):
        task = _task_from_evidence(
            task=expected,
            output=output,
            source_commit=source_commit,
            implementation_commit=implementation_commit,
            gpu_devices=gpu_devices,
            expected_runtime_artifacts=expected_runtime_artifacts,
        )
        if task != stored:
            raise CanaryError("C2 checkpoint task differs from retained evidence")
        rederived.append(task)

    # The worker COMPLETE, controller log, and raw telemetry are all durable
    # before the controller writes CHECKPOINT.  Recover exactly that one
    # crash window; any partial task or more than one uncheckpointed task is
    # inconsistent and remains fail-closed.
    next_index = len(stored_tasks)
    if next_index < len(plan):
        next_task = plan[next_index]
        sample_output, log_path, telemetry_path = _task_paths(output, next_task)
        terminal = (
            (sample_output / "COMPLETE").is_file()
            and log_path.is_file()
            and telemetry_path.is_file()
        )
        any_evidence = (
            sample_output.exists() or log_path.exists() or telemetry_path.exists()
        )
        if terminal:
            if any(
                path.exists()
                for later_task in plan[next_index + 1:]
                for path in _task_paths(output, later_task)
            ):
                raise CanaryError(
                    "C2 evidence extends beyond one recoverable task"
                )
            recovered = _task_from_evidence(
                task=next_task,
                output=output,
                source_commit=source_commit,
                implementation_commit=implementation_commit,
                gpu_devices=gpu_devices,
                expected_runtime_artifacts=expected_runtime_artifacts,
            )
            rederived.append(recovered)
            common.atomic_write_json(checkpoint_path, {
                "schema": CHECKPOINT_SCHEMA,
                "completed_task_ids": [
                    task["task_id"] for task in rederived
                ],
                "tasks": rederived,
            })
            next_index += 1
        elif any_evidence:
            raise CanaryError("C2 next task has nonterminal partial evidence")
    for task in plan[next_index:]:
        if any(path.exists() for path in _task_paths(output, task)):
            raise CanaryError("C2 evidence extends beyond checkpoint prefix")
    return rederived


def validate(output: pathlib.Path) -> dict[str, Any]:
    output = pathlib.Path(output).resolve(strict=True)
    complete = _load_stable_json(output / "COMPLETE", "C2 canary COMPLETE")
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise CanaryError("C2 canary COMPLETE schema differs")
    record = complete.get("report")
    _require_file_record(record, "report.json", "report")
    report = _load_stable_json(
        output / "report.json", "C2 canary report", expected=record
    )
    if (
        report.get("schema") != SCHEMA
        or report.get("state") != "COMPLETE"
        or report.get("development_only") is not True
        or report.get("valid_for_release_claim") is not False
        or report.get("repeats") != REPEATS
        or not _candidate_feature_matches(report)
        or report.get("task_plan") != task_plan()
        or not isinstance(report.get("source_commit"), str)
        or not isinstance(report.get("implementation_commit"), str)
    ):
        raise CanaryError("C2 canary report contract differs")
    for name, expected_path in (
        ("run_contract", "RUN.json"),
        ("checkpoint", "CHECKPOINT.json"),
        ("runtime_attestation", "RUNTIME_ATTESTATION.json"),
        ("controller", CONTROLLER_PATH.relative_to(REPO).as_posix()),
        ("worker", WORKER_PATH.relative_to(REPO).as_posix()),
    ):
        _require_file_record(report.get(name), expected_path, name)
    support_sources = report.get("support_sources", {})
    if (
        not isinstance(support_sources, dict)
        or set(support_sources) != set(SUPPORT_SOURCE_PATHS)
    ):
        raise CanaryError("C2 support-source inventory differs")
    for name, path in sorted(SUPPORT_SOURCE_PATHS.items()):
        _require_file_record(
            support_sources.get(name),
            path.relative_to(REPO).as_posix(),
            f"support source {name}",
        )
    run_contract = _load_stable_json(
        output / "RUN.json", "C2 run contract", expected=report.get("run_contract")
    )
    runtime_attestation, _ = _load_runtime_attestation(
        output / "RUNTIME_ATTESTATION.json",
        report["source_commit"],
        report["implementation_commit"],
    )
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or not _candidate_feature_matches(run_contract)
        or run_contract.get("source_commit") != report["source_commit"]
        or run_contract.get("implementation_commit")
            != report["implementation_commit"]
        or run_contract.get("implementation_to_harness_changes")
            != report.get("implementation_to_harness_changes")
        or run_contract.get("archive") != report.get("archive")
        or run_contract.get("runtime_attestation")
            != report.get("runtime_attestation")
        or run_contract.get("runtime_attestation_id")
            != runtime_attestation.get("attestation_id")
        or run_contract.get("runtime_attestation_id")
            != report.get("runtime_attestation_id")
        or run_contract.get("expected_runtime_artifacts")
            != runtime_attestation.get("runtime_artifacts")
        or run_contract.get("expected_runtime_artifacts")
            != report.get("expected_runtime_artifacts")
        or run_contract.get("task_plan") != task_plan()
        or not isinstance(run_contract.get("gpu_devices"), list)
        or len(run_contract["gpu_devices"]) != 2
        or len(set(run_contract["gpu_devices"])) != 2
        or run_contract.get("development_only") is not True
        or run_contract.get("valid_for_release_claim") is not False
    ):
        raise CanaryError("C2 retained run contract differs")
    stored_tasks = report.get("tasks")
    if not isinstance(stored_tasks, list) or len(stored_tasks) != len(task_plan()):
        raise CanaryError("C2 canary tasks are absent")
    tasks = []
    for planned, stored in zip(task_plan(), stored_tasks):
        rederived = _task_from_evidence(
            task=planned,
            output=output,
            source_commit=report["source_commit"],
            implementation_commit=report["implementation_commit"],
            gpu_devices=run_contract["gpu_devices"],
            expected_runtime_artifacts=run_contract[
                "expected_runtime_artifacts"
            ],
        )
        if stored != rederived:
            raise CanaryError("C2 report task was not re-derived from evidence")
        tasks.append(rederived)
    checkpoint = _load_stable_json(
        output / "CHECKPOINT.json",
        "C2 final checkpoint",
        expected=report.get("checkpoint"),
    )
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("completed_task_ids")
            != [task["task_id"] for task in task_plan()]
        or checkpoint.get("tasks") != tasks
    ):
        raise CanaryError("C2 final checkpoint differs from report evidence")
    for name, path in (
        ("controller", CONTROLLER_PATH),
        ("worker", WORKER_PATH),
    ):
        with common.StableFile(path, f"C2 {name}", expected=report.get(name)):
            pass
    for name, path in sorted(SUPPORT_SOURCE_PATHS.items()):
        with common.StableFile(
            path,
            f"C2 support source {name}",
            expected=support_sources.get(name),
        ):
            pass
    derived = derive_results(tasks)
    if report.get("results") != derived:
        raise CanaryError("C2 canary results were not re-derived")
    return report


def run(args: argparse.Namespace) -> int:
    output = _safe_output(args.output, args.resume)
    implementation_commit = args.implementation_commit
    source_commit, harness_changes = _source_boundary(implementation_commit)
    archive = common.verify_file(
        args.archive, common.AUNP_ARCHIVE_SHA256, "C2 AuNP archive"
    )
    python = args.python.resolve(strict=True)
    mpiexec = args.mpiexec.resolve(strict=True)
    pythonpath = args.pythonpath.resolve(strict=True)
    devices = [value.strip() for value in args.gpu_devices.split(",") if value.strip()]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise CanaryError("C2 requires two distinct GPU selectors")
    home = output / "runtime-home"
    home.mkdir(mode=0o700, exist_ok=True)
    runtime_attestation, runtime_attestation_source = _load_runtime_attestation(
        args.runtime_attestation, source_commit, implementation_commit
    )
    runtime_attestation_path = output / "RUNTIME_ATTESTATION.json"
    if runtime_attestation_path.exists():
        if _load_stable_json(
            runtime_attestation_path, "C2 retained runtime attestation"
        ) != runtime_attestation:
            raise CanaryError("C2 retained runtime attestation differs")
    else:
        common.atomic_write_json(runtime_attestation_path, runtime_attestation)
    runtime_attestation_record = _stable_file_record(
        runtime_attestation_path, output, "C2 retained runtime attestation"
    )
    expected_runtime_artifacts = runtime_attestation["runtime_artifacts"]
    run_contract = {
        "schema": RUN_SCHEMA,
        "candidate_feature": CANDIDATE_FEATURE,
        "source_commit": source_commit,
        "implementation_commit": implementation_commit,
        "implementation_to_harness_changes": harness_changes,
        "archive": common.absolute_file_record(archive, "C2 AuNP archive"),
        "python": common.absolute_file_record(python, "C2 Python"),
        "mpiexec": common.absolute_file_record(mpiexec, "C2 mpiexec"),
        "pythonpath": str(pythonpath),
        "library_path": args.library_path,
        "gpu_devices": devices,
        "runtime_attestation": runtime_attestation_record,
        "runtime_attestation_source": runtime_attestation_source,
        "runtime_attestation_id": runtime_attestation["attestation_id"],
        "expected_runtime_artifacts": expected_runtime_artifacts,
        "task_plan": task_plan(),
        "development_only": True,
        "valid_for_release_claim": False,
    }
    run_path = output / "RUN.json"
    if run_path.exists():
        if _load_stable_json(run_path, "C2 run contract") != run_contract:
            raise CanaryError("C2 resume contract differs")
    else:
        if args.resume:
            raise CanaryError("C2 resume output lacks a run contract")
        common.atomic_write_json(run_path, run_contract)
    if (output / "COMPLETE").exists() or (output / "report.json").exists():
        raise CanaryError("C2 run output is already terminal; use validate")

    tasks = _resume_tasks(
        output=output,
        source_commit=source_commit,
        implementation_commit=implementation_commit,
        gpu_devices=devices,
        expected_runtime_artifacts=expected_runtime_artifacts,
    )
    for task in task_plan()[len(tasks):]:
        print(f"C2 start {task['task_id']}", flush=True)
        result = _run_task(
            task=task,
            output=output,
            archive=archive,
            python=python,
            mpiexec=mpiexec,
            pythonpath=pythonpath,
            library_path=args.library_path,
            devices=devices,
            source_commit=source_commit,
            implementation_commit=implementation_commit,
            home=home,
            expected_runtime_artifacts=expected_runtime_artifacts,
            runtime_attestation=runtime_attestation,
        )
        tasks.append(result)
        common.atomic_write_json(output / "CHECKPOINT.json", {
            "schema": CHECKPOINT_SCHEMA,
            "completed_task_ids": [item["task_id"] for item in tasks],
            "tasks": tasks,
        })
        print(
            f"C2 complete {task['task_id']}: "
            f"fdtd={result['fdtd_wall_seconds']:.6f}s "
            f"e2e={result['end_to_end_seconds']:.6f}s",
            flush=True,
        )
    if _git("rev-parse", "HEAD").strip() != source_commit or _git(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines():
        raise CanaryError("C2 source boundary changed during execution")
    results = derive_results(tasks)
    report = {
        "schema": SCHEMA,
        "state": "COMPLETE",
        "candidate_feature": CANDIDATE_FEATURE,
        "development_only": True,
        "valid_for_release_claim": False,
        "source_commit": source_commit,
        "implementation_commit": implementation_commit,
        "implementation_to_harness_changes": harness_changes,
        "archive": run_contract["archive"],
        "runtime_attestation": runtime_attestation_record,
        "runtime_attestation_id": runtime_attestation["attestation_id"],
        "expected_runtime_artifacts": expected_runtime_artifacts,
        "controller": common.file_record(CONTROLLER_PATH, REPO),
        "worker": common.file_record(WORKER_PATH, REPO),
        "support_sources": {
            name: common.file_record(path, REPO)
            for name, path in sorted(SUPPORT_SOURCE_PATHS.items())
        },
        "run_contract": _stable_file_record(
            output / "RUN.json", output, "C2 run contract"
        ),
        "checkpoint": _stable_file_record(
            output / "CHECKPOINT.json", output, "C2 final checkpoint"
        ),
        "repeats": REPEATS,
        "task_plan": task_plan(),
        "tasks": tasks,
        "results": results,
        "disclosure": (
            "development paired canary only; no release or new scientific-output claim"
        ),
    }
    common.atomic_write_json(output / "report.json", report)
    common.atomic_write_json(output / "COMPLETE", {
        "schema": COMPLETE_SCHEMA,
        "report": common.file_record(output / "report.json", output),
    })
    validate(output)
    print(
        "C2 canary COMPLETE: " + results["promotion_outcome"],
        flush=True,
    )
    return 0


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
    execute.add_argument("--runtime-attestation", required=True, type=pathlib.Path)
    execute.add_argument("--gpu-devices", default="0,1")
    execute.add_argument("--resume", action="store_true")
    execute.set_defaults(function=run)
    replay = subparsers.add_parser("validate")
    replay.add_argument("--output", required=True, type=pathlib.Path)
    replay.set_defaults(function=lambda args: (validate(args.output), 0)[1])
    attest = subparsers.add_parser("attest-runtime")
    attest.add_argument("--output", required=True, type=pathlib.Path)
    attest.add_argument("--python", required=True, type=pathlib.Path)
    attest.add_argument("--pythonpath", required=True, type=pathlib.Path)
    attest.add_argument("--library-path", required=True)
    attest.add_argument("--implementation-commit", required=True)
    attest.add_argument("--c1-validation", required=True, type=pathlib.Path)
    attest.add_argument(
        "--c1-adversarial-audit", required=True, type=pathlib.Path
    )
    attest.set_defaults(function=attest_runtime)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        CanaryError,
        common.WorkloadError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"C2 AuNP canary FAILED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
