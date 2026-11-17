#!/usr/bin/env python3
"""Fail-closed helper for explicit-backend qualification runs.

gpmeep v1.1.0 never changes a caller-selected CPU/CUDA backend.  Release
qualification sometimes exercises both choices even when the policy recommends
only one.  This helper preserves the real warning boundary: the first exact run
must fail before mutation, a reason-bearing acknowledgement is issued for that
challenge, and only the identical run is retried.  Later operations may report
an audited process-local policy-scope reuse without another failed attempt.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import time
from typing import Any, Callable, TypeVar


SCHEMA = "gpmeep-v110-explicit-backend-guard-evidence-v2"
MINIMUM_REASON_BYTES = 8
MAXIMUM_REASON_BYTES = 512
TELEMETRY_SCHEMA = "gpmeep-v110-backend-recommendation-v3"
POLICY_VERSION = "gpmeep-v110-explicit-backend-guard-v24"
PROFILER_OR_ABLATION_ID = "v110-m1-m2-aunp-crossover-calibration"
_ACKNOWLEDGEMENT_BINDING_KEYS = (
    "selected_backend",
    "recommended_backend",
    "workload_fingerprint",
    "recommendation_policy_version",
    "mpi_topology_digest",
    "native_state_digest",
    "process_nonce",
    "run_nonce",
)
_TELEMETRY_KEYS = {
    "schema",
    "selected_backend",
    "recommended_backend",
    "reason_class",
    "recommendation_confidence",
    "rank_local_cells",
    "run_length",
    "kernel_launch_count",
    "transfer_bytes",
    "monitor_count",
    "monitor_frequency_count",
    "source_count",
    "mpi_gpu_topology",
    "profiler_or_ablation_id",
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
    "native_state_digest",
    "process_nonce",
    "run_nonce",
    "acknowledgement_digest",
}


class BackendGuardEvidenceError(RuntimeError):
    """Raised when a qualification run cannot prove the warning/ack contract."""


_T = TypeVar("_T")


def validate_reason(reason: Any) -> str:
    if not isinstance(reason, str):
        raise BackendGuardEvidenceError("backend acknowledgement reason must be text")
    try:
        encoded = reason.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise BackendGuardEvidenceError(
            "backend acknowledgement reason must be valid UTF-8 text"
        ) from exc
    if not MINIMUM_REASON_BYTES <= len(encoded) <= MAXIMUM_REASON_BYTES:
        raise BackendGuardEvidenceError(
            "backend acknowledgement reason must occupy 8 to 512 UTF-8 bytes"
        )
    return reason


def _timestep(simulation: Any) -> int:
    fields = getattr(simulation, "fields", None)
    return int(fields.t) if fields is not None else 0


def _mismatch_type(mp: Any) -> type[BaseException]:
    policy = None
    module_name = None
    name = getattr(mp, "__name__", None)
    if isinstance(name, str) and name:
        module_name = f"{name}.backend_policy"
        try:
            policy = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
    if policy is None:
        policy = getattr(mp, "backend_policy", None)
    mismatch = getattr(policy, "BackendRecommendationMismatch", None)
    if not isinstance(mismatch, type) or not issubclass(mismatch, BaseException):
        raise BackendGuardEvidenceError(
            "gpmeep backend recommendation mismatch type is unavailable"
        )
    return mismatch


def _exact_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite_nonnegative_number_or_none(value: Any) -> bool:
    return value is None or (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _acknowledgement_digest(telemetry: dict[str, Any]) -> str:
    bindings = {key: telemetry[key] for key in _ACKNOWLEDGEMENT_BINDING_KEYS}
    encoded = json.dumps(
        {"schema": "gpmeep-v110-backend-ack-v1", **bindings},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_run_length(value: Any, label: str) -> None:
    expected = {
        "kind",
        "numeric_duration",
        "source_end_time",
        "expected_timesteps",
        "dynamic_condition",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise BackendGuardEvidenceError(f"{label} run-length telemetry differs")
    if value["kind"] not in {"until", "until_after_sources"}:
        raise BackendGuardEvidenceError(f"{label} run-length kind differs")
    if not _finite_nonnegative_number_or_none(value["numeric_duration"]):
        raise BackendGuardEvidenceError(f"{label} run-length duration differs")
    if not _finite_nonnegative_number_or_none(value["source_end_time"]):
        raise BackendGuardEvidenceError(f"{label} source end time differs")
    expected_timesteps = value["expected_timesteps"]
    if expected_timesteps is not None and not _exact_nonnegative_int(
        expected_timesteps
    ):
        raise BackendGuardEvidenceError(f"{label} expected timesteps differ")
    if type(value["dynamic_condition"]) is not bool:
        raise BackendGuardEvidenceError(f"{label} dynamic-condition type differs")
    if value["dynamic_condition"] is (value["numeric_duration"] is not None):
        raise BackendGuardEvidenceError(
            f"{label} dynamic-condition/duration state differs"
        )


def _validate_topology(
    value: Any, rank_local_cells: list[int], label: str
) -> None:
    topology_keys = {
        "rank",
        "global_rank",
        "active_size",
        "node_rank",
        "node_size",
        "device_identifier",
        "device_ordinal",
    }
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(rank_local_cells)
    ):
        raise BackendGuardEvidenceError(f"{label} MPI/GPU topology differs")
    ranks: list[int] = []
    global_ranks: list[int] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != topology_keys:
            raise BackendGuardEvidenceError(f"{label} MPI/GPU topology differs")
        if any(
            not _exact_nonnegative_int(item[key])
            for key in ("rank", "global_rank", "active_size", "node_rank")
        ) or not (type(item["node_size"]) is int and item["node_size"] > 0):
            raise BackendGuardEvidenceError(f"{label} MPI/GPU rank types differ")
        if (
            item["active_size"] != len(value)
            or item["node_rank"] >= item["node_size"]
            or type(item["device_ordinal"]) is not int
            or item["device_ordinal"] < -1
            or not isinstance(item["device_identifier"], str)
            or len(item["device_identifier"].encode("utf-8")) > 256
        ):
            raise BackendGuardEvidenceError(f"{label} MPI/GPU rank values differ")
        ranks.append(item["rank"])
        global_ranks.append(item["global_rank"])
    if sorted(ranks) != list(range(len(value))) or len(set(global_ranks)) != len(value):
        raise BackendGuardEvidenceError(f"{label} MPI/GPU rank inventory differs")


def _telemetry(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _TELEMETRY_KEYS:
        raise BackendGuardEvidenceError(f"{label} telemetry schema differs")
    result = copy.deepcopy(value)
    selected = result["selected_backend"]
    recommended = result["recommended_backend"]
    if (
        result["schema"] != TELEMETRY_SCHEMA
        or selected not in {"cpu", "cuda"}
        or recommended not in {"cpu", "cuda"}
        or result["reason_class"] not in {"hardware_crossover", "environment"}
        or result["recommendation_confidence"] not in {"medium", "high"}
        or result["recommendation_policy_version"] != POLICY_VERSION
        or result["profiler_or_ablation_id"] != PROFILER_OR_ABLATION_ID
    ):
        raise BackendGuardEvidenceError(f"{label} backend telemetry identity differs")
    rank_local_cells = result["rank_local_cells"]
    if (
        not isinstance(rank_local_cells, list)
        or not rank_local_cells
        or any(type(item) is not int or item <= 0 for item in rank_local_cells)
    ):
        raise BackendGuardEvidenceError(f"{label} local-cell telemetry differs")
    _validate_run_length(result["run_length"], label)
    launches = result["kernel_launch_count"]
    if launches is not None and not _exact_nonnegative_int(launches):
        raise BackendGuardEvidenceError(f"{label} kernel launch count differs")
    for key in (
        "transfer_bytes",
        "monitor_count",
        "monitor_frequency_count",
        "source_count",
    ):
        if not _exact_nonnegative_int(result[key]):
            raise BackendGuardEvidenceError(f"{label} {key} differs")
    _validate_topology(result["mpi_gpu_topology"], rank_local_cells, label)
    for key in ("mpi_topology_digest", "acknowledgement_scope_digest"):
        if not _lower_sha256(result[key]):
            raise BackendGuardEvidenceError(f"{label} {key} differs")
    acknowledgement = result["acknowledgement"]
    allowed = result["acknowledgement_allowed"]
    reason = result["acknowledgement_reason"]
    reused = result["acknowledgement_reused"]
    acknowledgement_scope = result["acknowledgement_scope"]
    origin_cohort_digest = result[
        "acknowledgement_origin_cohort_digest"
    ]
    cache_enabled = result["acknowledgement_scope_cache_enabled"]
    fingerprint_evaluated = result["workload_fingerprint_evaluated"]
    if any(
        type(value) is not bool
        for value in (
            acknowledgement,
            allowed,
            reused,
            cache_enabled,
            fingerprint_evaluated,
        )
    ):
        raise BackendGuardEvidenceError(f"{label} acknowledgement types differ")
    exact_fields = (
        "workload_fingerprint",
        "native_state_digest",
        "process_nonce",
        "run_nonce",
        "acknowledgement_digest",
    )
    if fingerprint_evaluated:
        for key in exact_fields:
            if not _lower_sha256(result[key]):
                raise BackendGuardEvidenceError(f"{label} {key} differs")
        if result["acknowledgement_digest"] != _acknowledgement_digest(result):
            raise BackendGuardEvidenceError(
                f"{label} acknowledgement digest differs"
            )
    elif any(result[key] != "" for key in exact_fields):
        raise BackendGuardEvidenceError(
            f"{label} unevaluated workload bindings differ"
        )
    if acknowledgement:
        validate_reason(reason)
        if (
            not allowed
            or selected == recommended
            or acknowledgement_scope not in {"simulation", "process"}
            or not _lower_sha256(origin_cohort_digest)
        ):
            raise BackendGuardEvidenceError(f"{label} acknowledgement state differs")
        if reused and not cache_enabled:
            raise BackendGuardEvidenceError(
                f"{label} acknowledgement reuse state differs"
            )
        if reused and fingerprint_evaluated is not (len(rank_local_cells) > 1):
            raise BackendGuardEvidenceError(
                f"{label} acknowledgement reuse fingerprint mode differs"
            )
        if not reused and origin_cohort_digest != result["acknowledgement_digest"]:
            raise BackendGuardEvidenceError(
                f"{label} exact acknowledgement cohort differs"
            )
    elif reason != "" or acknowledgement_scope != "" or origin_cohort_digest != "":
        raise BackendGuardEvidenceError(f"{label} acknowledgement audit differs")
    elif reused:
        raise BackendGuardEvidenceError(f"{label} acknowledgement reuse differs")
    if selected == recommended:
        mpi_exact_required = len(rank_local_cells) > 1
        if fingerprint_evaluated is not mpi_exact_required:
            raise BackendGuardEvidenceError(
                f"{label} matching recommendation fingerprint mode differs"
            )
    if (
        selected != recommended
        and result["reason_class"] != "environment"
        and not acknowledgement
        and not fingerprint_evaluated
    ):
        raise BackendGuardEvidenceError(
            f"{label} unacknowledged mismatch lacks an exact fingerprint"
        )
    if result["reason_class"] == "environment":
        # An environment recommendation is non-acknowledgeable because the
        # selected CUDA topology may not have a valid device assignment or
        # enough device memory.  The same diagnostic is also a valid final
        # state for an explicitly selected CPU run: CPU remains usable and is
        # already the recommended backend, so there is no mismatch to
        # acknowledge.  Requiring ``selected == "cuda"`` here incorrectly
        # rejected those successful CPU runs after they had finished.
        if recommended != "cpu" or allowed or acknowledgement:
            raise BackendGuardEvidenceError(f"{label} environment state differs")
    elif not allowed:
        raise BackendGuardEvidenceError(f"{label} acknowledgement permission differs")
    return result


def _stable_telemetry(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "acknowledgement",
            "acknowledgement_reason",
            "acknowledgement_scope",
            "acknowledgement_origin_cohort_digest",
        }
    }


def invoke_pristine_class_method(
    owner_class: type,
    method_name: str,
    observer: Callable[..., Any],
    original: Callable[..., _T],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    original_methods: dict[str, Callable[..., Any]] | None = None,
    guarded_methods: dict[str, Callable[..., Any]] | None = None,
) -> _T:
    """Call an instrumented method with the package class fully restored.

    Qualification observes several entry points on one class.  Restoring only
    the method currently being invoked leaves its sibling observers installed
    while package code runs.  That is both an inaccurate package identity and
    a false mutation under backend-policy's transitive class-integrity check.
    When the complete method maps are supplied, every observer is therefore
    authenticated and replaced as one fail-closed group for the duration of
    the pristine call.
    """

    if (original_methods is None) != (guarded_methods is None):
        raise BackendGuardEvidenceError(
            "pristine class dispatch method maps are incomplete"
        )
    if original_methods is None:
        originals = {method_name: original}
        observers = {method_name: observer}
    else:
        if (
            type(original_methods) is not dict
            or type(guarded_methods) is not dict
            or not original_methods
            or set(original_methods) != set(guarded_methods)
            or original_methods.get(method_name) is not original
            or guarded_methods.get(method_name) is not observer
            or any(
                type(name) is not str
                or not name
                or not callable(original_methods[name])
                or not callable(guarded_methods[name])
                for name in original_methods
            )
        ):
            raise BackendGuardEvidenceError(
                "pristine class dispatch method maps differ"
            )
        originals = original_methods
        observers = guarded_methods

    method_bindings = tuple(
        (name, originals[name], observers[name])
        for name in sorted(originals)
    )
    for name, _, guarded_method in method_bindings:
        if getattr(owner_class, name, None) is not guarded_method:
            raise BackendGuardEvidenceError(
                f"{owner_class.__name__}.{name} instrumentation changed "
                "before original dispatch"
            )

    activated: list[tuple[str, Callable[..., Any], Callable[..., Any]]] = []
    try:
        for name, pristine_method, guarded_method in method_bindings:
            setattr(owner_class, name, pristine_method)
            activated.append((name, pristine_method, guarded_method))
        result = original(instance, *args, **kwargs)
    except BaseException as error:
        changed = [
            name
            for name, pristine_method, _ in activated
            if getattr(owner_class, name, None) is not pristine_method
        ]
        for name, _, guarded_method in reversed(activated):
            setattr(owner_class, name, guarded_method)
        if changed:
            raise BackendGuardEvidenceError(
                f"{owner_class.__name__} methods changed while the original "
                "implementation was active: " + ", ".join(changed)
            ) from error
        raise
    changed = [
        name
        for name, pristine_method, _ in activated
        if getattr(owner_class, name, None) is not pristine_method
    ]
    for name, _, guarded_method in reversed(activated):
        setattr(owner_class, name, guarded_method)
    if changed:
        raise BackendGuardEvidenceError(
            f"{owner_class.__name__} methods changed while the original "
            "implementation was active: " + ", ".join(changed)
        )
    return result


def invoke_with_acknowledgement(
    mp: Any,
    simulation: Any,
    operation: Callable[[], _T],
    reason: str | None,
    acknowledgement_scope: str = "simulation",
) -> tuple[_T, dict[str, Any]]:
    """Invoke one exact run and return its result plus guard evidence.

    ``reason=None`` never acknowledges a mismatch.  This is intentional: a
    caller that did not explicitly opt in receives gpmeep's original warning
    exception unchanged.
    """

    if not callable(operation):
        raise BackendGuardEvidenceError("backend-guard operation is not callable")
    if acknowledgement_scope not in {"simulation", "process"}:
        raise BackendGuardEvidenceError(
            "backend acknowledgement scope must be 'simulation' or 'process'"
        )
    if reason is not None:
        reason = validate_reason(reason)
    mismatch_type = _mismatch_type(mp)
    pre_fields = getattr(simulation, "fields", None)
    pre_timestep = _timestep(simulation)
    attempts = 1
    acknowledgements = 0
    warning_wall_seconds = 0.0
    warning_telemetry: dict[str, Any] | None = None
    execution_started_ns = time.monotonic_ns()
    try:
        result = operation()
    except mismatch_type as mismatch:
        warning_ns = time.monotonic_ns()
        if reason is None:
            raise
        if (
            getattr(simulation, "fields", None) is not pre_fields
            or _timestep(simulation) != pre_timestep
        ):
            raise BackendGuardEvidenceError(
                "backend recommendation mismatch mutated simulation state before "
                "acknowledgement"
            ) from mismatch
        warning_telemetry = _telemetry(mismatch.telemetry, "warning")
        challenge = getattr(mismatch, "challenge", None)
        if (
            warning_telemetry["selected_backend"]
            == warning_telemetry["recommended_backend"]
            or warning_telemetry.get("acknowledgement") is not False
            or warning_telemetry.get("acknowledgement_allowed") is not True
            or not isinstance(challenge, str)
            or len(challenge) != 64
            or any(character not in "0123456789abcdef" for character in challenge)
            or warning_telemetry.get("acknowledgement_digest") != challenge
        ):
            raise BackendGuardEvidenceError(
                "backend recommendation mismatch telemetry is inconsistent"
            ) from mismatch
        warning_wall_seconds = (warning_ns - execution_started_ns) / 1_000_000_000.0
        mp.gpu.acknowledge_backend_mismatch(
            challenge, reason, scope=acknowledgement_scope
        )
        acknowledgements = 1
        attempts = 2
        execution_started_ns = time.monotonic_ns()
        result = operation()
    ended_ns = time.monotonic_ns()
    execution_wall_seconds = (ended_ns - execution_started_ns) / 1_000_000_000.0
    if (
        not math.isfinite(warning_wall_seconds)
        or warning_wall_seconds < 0
        or not math.isfinite(execution_wall_seconds)
        or execution_wall_seconds < 0
    ):
        raise BackendGuardEvidenceError("backend-guard wall time is invalid")
    final_telemetry = _telemetry(getattr(mp.gpu, "recommendation", None), "final")
    selected = final_telemetry["selected_backend"]
    recommended = final_telemetry["recommended_backend"]
    mismatch_observed = selected != recommended
    acknowledgement_reused = final_telemetry["acknowledgement_reused"]
    if mismatch_observed:
        if acknowledgements == 1:
            if acknowledgement_reused:
                raise BackendGuardEvidenceError(
                    "new backend acknowledgement was marked as reused"
                )
        elif acknowledgements == 0:
            if not acknowledgement_reused:
                raise BackendGuardEvidenceError(
                    "backend mismatch succeeded without acknowledgement evidence"
                )
        else:
            raise BackendGuardEvidenceError(
                "backend recommendation acknowledgement count differs"
            )
    elif acknowledgements or acknowledgement_reused:
        raise BackendGuardEvidenceError(
            "matching backend contains acknowledgement evidence"
        )
    if acknowledgements:
        assert warning_telemetry is not None and reason is not None
        if (
            final_telemetry.get("acknowledgement") is not True
            or final_telemetry.get("acknowledgement_reason") != reason
            or final_telemetry.get("acknowledgement_scope")
            != acknowledgement_scope
            or final_telemetry.get("acknowledgement_origin_cohort_digest")
            != challenge
            or _stable_telemetry(final_telemetry)
            != _stable_telemetry(warning_telemetry)
        ):
            raise BackendGuardEvidenceError(
                "acknowledged backend recommendation telemetry differs"
            )
    elif mismatch_observed:
        if (
            warning_telemetry is not None
            or final_telemetry.get("acknowledgement") is not True
            or final_telemetry.get("acknowledgement_reused") is not True
            or reason is None
            or final_telemetry.get("acknowledgement_reason") != reason
            or warning_wall_seconds != 0.0
        ):
            raise BackendGuardEvidenceError(
                "reused backend acknowledgement evidence differs"
            )
    elif (
        warning_telemetry is not None
        or final_telemetry.get("acknowledgement") is not False
        or warning_wall_seconds != 0.0
    ):
        raise BackendGuardEvidenceError(
            "matching backend run contains unexpected acknowledgement evidence"
        )
    return result, {
        "schema": SCHEMA,
        "attempts": attempts,
        "acknowledgements": acknowledgements,
        "acknowledgement_reused": acknowledgement_reused,
        "mutation_before_acknowledgement": False,
        "warning_wall_seconds": warning_wall_seconds,
        "execution_wall_seconds": execution_wall_seconds,
        "selected_backend": selected,
        "recommended_backend": recommended,
        "mismatch_observed": mismatch_observed,
        "warning_telemetry": warning_telemetry,
        "final_telemetry": final_telemetry,
    }


def validate_evidence(
    record: Any,
    *,
    expected_backend: str,
    expected_reason: str | None,
) -> dict[str, Any]:
    """Revalidate one serialized guard record at an evidence boundary."""

    expected_keys = {
        "schema",
        "attempts",
        "acknowledgements",
        "acknowledgement_reused",
        "mutation_before_acknowledgement",
        "warning_wall_seconds",
        "execution_wall_seconds",
        "selected_backend",
        "recommended_backend",
        "mismatch_observed",
        "warning_telemetry",
        "final_telemetry",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise BackendGuardEvidenceError("backend-guard evidence schema differs")
    if record["schema"] != SCHEMA or expected_backend not in {"cpu", "cuda"}:
        raise BackendGuardEvidenceError("backend-guard evidence identity differs")
    if record["selected_backend"] != expected_backend:
        raise BackendGuardEvidenceError("backend-guard selected backend differs")
    for name in ("warning_wall_seconds", "execution_wall_seconds"):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BackendGuardEvidenceError("backend-guard wall time type differs")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise BackendGuardEvidenceError("backend-guard wall time differs")
    mismatch = record["recommended_backend"] != expected_backend
    if (
        record["recommended_backend"] not in {"cpu", "cuda"}
        or record["mismatch_observed"] is not mismatch
        or record["mutation_before_acknowledgement"] is not False
    ):
        raise BackendGuardEvidenceError("backend-guard recommendation state differs")
    reused = record["acknowledgement_reused"]
    if type(reused) is not bool or reused and not mismatch:
        raise BackendGuardEvidenceError(
            "backend-guard acknowledgement reuse differs"
        )
    expected_attempts = 2 if mismatch and not reused else 1
    expected_acknowledgements = 1 if mismatch and not reused else 0
    if (
        type(record["attempts"]) is not int
        or type(record["acknowledgements"]) is not int
        or record["attempts"] != expected_attempts
        or record["acknowledgements"] != expected_acknowledgements
    ):
        raise BackendGuardEvidenceError("backend-guard attempt inventory differs")
    final = _telemetry(record["final_telemetry"], "retained final")
    if (
        final.get("selected_backend") != expected_backend
        or final.get("recommended_backend") != record["recommended_backend"]
        or final.get("acknowledgement") is not mismatch
        or final.get("acknowledgement_reused") is not reused
    ):
        raise BackendGuardEvidenceError("backend-guard final telemetry differs")
    if mismatch and not reused:
        reason = validate_reason(expected_reason)
        warning = _telemetry(record["warning_telemetry"], "retained warning")
        if (
            warning.get("selected_backend") != expected_backend
            or warning.get("recommended_backend") != record["recommended_backend"]
            or warning.get("acknowledgement") is not False
            or warning.get("acknowledgement_allowed") is not True
            or not isinstance(warning.get("acknowledgement_digest"), str)
            or len(warning["acknowledgement_digest"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in warning["acknowledgement_digest"]
            )
            or final.get("acknowledgement_reason") != reason
            or _stable_telemetry(final) != _stable_telemetry(warning)
        ):
            raise BackendGuardEvidenceError("backend-guard warning telemetry differs")
    elif mismatch:
        reason = validate_reason(expected_reason)
        if (
            record["warning_telemetry"] is not None
            or record["warning_wall_seconds"] != 0.0
            or final.get("acknowledgement_reason") != reason
        ):
            raise BackendGuardEvidenceError(
                "reused backend guard evidence differs"
            )
    elif (
        record["warning_telemetry"] is not None
        or record["warning_wall_seconds"] != 0.0
        or expected_reason is not None and validate_reason(expected_reason) != expected_reason
    ):
        raise BackendGuardEvidenceError("matching backend guard evidence differs")
    return copy.deepcopy(record)
