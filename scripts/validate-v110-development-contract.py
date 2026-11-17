#!/usr/bin/env python3
"""Fail-closed validator for the gpmeep v1.1.0 development contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tarfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "packaging/v110-development-contract.json"
PLAN = ROOT / "doc/docs/GPMEEP_V1_1_0_DEVELOPMENT_PLAN.md"
FUTURE_TODO = ROOT / "doc/docs/GPMEEP_FUTURE_TODO.md"
README = ROOT / "README.md"
EXPECTED_BASE = "b48c9b678f2316cefa95e48a661f7a877547a000"
EXPECTED_DISTRIBUTION_VERSION = "1.1.0"
EXPECTED_CANONICAL_CONTRACT_SHA256 = (
    "3552ca5a31a125f49fcbb53968a47c3c4dd159bcfc86de8b132477c965ea8c74"
)
EXPECTED_DEPENDABOT_POLICY_SHA256 = (
    "fdc9440a866a6e08995f276fe141f0d2e36e04a7706451c9c8c8ed68f5971357"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_LANES = [
    "upstream_meep_cpu",
    "gpmeep_v110_cpu",
    "gpmeep_v103_cuda_gpu1",
    "gpmeep_v110_cuda_gpu1",
    "gpmeep_v103_cuda_gpu2",
    "gpmeep_v110_cuda_gpu2",
]
EXPECTED_PHASES = [
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
]
EXPECTED_TELEMETRY = [
    "selected_backend",
    "recommended_backend",
    "reason_class",
    "recommendation_confidence",
    "rank_local_cells",
    "run_length",
    "kernel_launch_count",
    "transfer_bytes",
    "monitor_count",
    "mpi_gpu_topology",
    "profiler_or_ablation_id",
    "acknowledgement",
    "acknowledgement_reason",
    "acknowledgement_reused",
    "acknowledgement_scope",
    "acknowledgement_origin_cohort_digest",
    "acknowledgement_scope_cache_enabled",
    "acknowledgement_scope_digest",
    "workload_fingerprint_evaluated",
    "recommendation_policy_version",
]
EXPECTED_ACK_BINDINGS = [
    "selected_backend",
    "recommended_backend",
    "workload_fingerprint",
    "recommendation_policy_version",
    "mpi_topology_digest",
    "native_state_digest",
    "process_nonce",
    "run_nonce",
]
EXPECTED_ACK_SCOPE_BINDINGS = [
    "selected_backend",
    "recommended_backend",
    "reason_class",
    "recommendation_confidence",
    "recommendation_policy_version",
    "mpi_topology_digest",
    "policy_facts",
    "policy_inputs",
]
EXPECTED_ACK_FAILURES = [
    "missing_acknowledgement",
    "empty_or_out_of_range_reason",
    "stale_digest",
    "changed_backend_recommendation_or_workload",
    "changed_mpi_topology",
    "rank_digest_mismatch",
    "replayed_nonce",
]
EXPECTED_CANDIDATES = {
    "c1_guarded_zero_state_lorentzian_drude_elision": "reapply_and_ablate",
    "c2_aunp_canary": "measurement_not_independent_optimization",
    "c3_resident_sparse_lorentzian_plan": "reapply_and_ablate",
    "c4_a2_safe_static_material_initialization": "implement_and_ablate",
    "adaptive_field_phase_batching": "small_domains_only",
    "source_dft_batching_and_cuda_graph": "fail_closed_eligibility",
    "multi_gpu_overlap": "profile_before_change",
    "gpmeep_cpu_regression": "reproduce_on_equal_build_before_change",
}


class ContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_active_distribution_identity() -> None:
    """Require every source of package identity to name one exact v1.1.0."""
    specifications = {
        "packaging/conda/meta.yaml": re.compile(
            r'^\{% set distribution_version = "([^"]+)" %\}$', re.MULTILINE
        ),
        "packaging/conda/release.json.in": re.compile(
            r'^\s*"distribution_version": "([^"]+)",\s*$', re.MULTILINE
        ),
        "scripts/build-conda-package.sh": re.compile(
            r'^DISTRIBUTION_VERSION="([^"]+)"$', re.MULTILINE
        ),
        "scripts/gpmeep_release_manifest.py": re.compile(
            r'^DISTRIBUTION_VERSION = "([^"]+)"$', re.MULTILINE
        ),
    }
    for relative, pattern in specifications.items():
        path = ROOT / relative
        _require(path.is_file(), f"active distribution identity file is absent: {relative}")
        versions = pattern.findall(path.read_text(encoding="utf-8"))
        _require(
            versions == [EXPECTED_DISTRIBUTION_VERSION],
            f"active distribution version differs in {relative}",
        )


def validate_dependabot_policy_bytes(policy: bytes) -> None:
    """Accept only the reviewed canonical policy; do not partially parse YAML."""
    _require(
        hashlib.sha256(policy).hexdigest() == EXPECTED_DEPENDABOT_POLICY_SHA256,
        "release-only Dependabot policy bytes differ",
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def _load_json(raw: bytes | str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _path(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        _require(
            isinstance(value, dict) and key in value,
            f"missing {'.'.join(traversed)}",
        )
        value = value[key]
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == expected, f"{name} key inventory differs")
    return value


def _all_true(value: dict[str, Any], names: set[str], prefix: str) -> None:
    for name in names:
        _require(value.get(name) is True, f"{prefix}.{name} must be true")


def canonical_contract_sha256(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def validate_contract_data(data: dict[str, Any]) -> None:
    _exact_keys(
        data,
        {
            "schema_version",
            "release_train",
            "base_release",
            "scope",
            "backend_policy",
            "performance_inversion_policy",
            "performance_candidates",
            "compatibility_target",
            "development_isolation",
            "validation",
            "gate_integration",
            "release_gates",
        },
        "contract",
    )
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1, "unsupported schema_version")
    _require(data["release_train"] == "1.1.0", "wrong release_train")

    base = _exact_keys(data["base_release"], {"tag", "commit", "frozen"}, "base_release")
    _require(base["tag"] == "v1.0.3", "wrong base tag")
    _require(base["commit"] == EXPECTED_BASE, "wrong base commit")
    _require(base["frozen"] is True, "base release must be frozen")

    scope = _exact_keys(
        data["scope"],
        {
            "python_interface",
            "scheme_interface",
            "fp32_cpu_and_cuda",
            "mpi_multi_gpu",
            "adjoint_inverse_design",
            "broad_fdtd_feature_validation",
        },
        "scope",
    )
    _all_true(
        scope,
        {
            "python_interface",
            "fp32_cpu_and_cuda",
            "mpi_multi_gpu",
            "adjoint_inverse_design",
            "broad_fdtd_feature_validation",
        },
        "scope",
    )
    _require(scope["scheme_interface"] is False, "Scheme must remain out of scope")

    backend = _exact_keys(
        data["backend_policy"],
        {
            "allowed_user_choices",
            "user_choice_is_authoritative",
            "unspecified_backend_is_interpreted_as",
            "unspecified_backend_is_still_recommendation_guarded",
            "automatic_backend_selection",
            "runtime_backend_switching",
            "silent_cpu_fallback",
            "mismatch_first_attempt",
            "mismatch_override_requires_acknowledgement",
            "override_is_audited",
            "acknowledgement",
        },
        "backend_policy",
    )
    _require(backend["allowed_user_choices"] == ["cpu", "cuda"], "choices must be cpu/cuda")
    _require(backend["user_choice_is_authoritative"] is True, "user choice is not authoritative")
    _require(backend["unspecified_backend_is_interpreted_as"] == "cpu", "legacy default changed")
    _require(
        backend["unspecified_backend_is_still_recommendation_guarded"] is True,
        "legacy CPU default must still be guarded",
    )
    for key in ("automatic_backend_selection", "runtime_backend_switching", "silent_cpu_fallback"):
        _require(backend[key] is False, f"backend_policy.{key} must be false")
    _require(backend["mismatch_first_attempt"] == "fail_before_simulation", "mismatch timing changed")
    _require(backend["mismatch_override_requires_acknowledgement"] is True, "ack is optional")
    _require(backend["override_is_audited"] is True, "override audit disabled")

    acknowledgement = _exact_keys(
        backend["acknowledgement"],
        {
            "scope",
            "default_reuse_scope",
            "optional_reuse_scopes",
            "process_scope_requires_explicit_opt_in",
            "required_digest_bindings",
            "reusable_scope_digest_bindings",
            "maximum_simulation_scopes",
            "maximum_process_scopes",
            "reuse_is_process_local",
            "fork_inheritance_is_forbidden",
            "diagnostic_cache_disable_environment",
            "reason_min_utf8_bytes",
            "reason_max_utf8_bytes",
            "all_mpi_ranks_must_agree_on_digest",
            "mpi_reuse_requires_exact_workload_consensus",
            "consume_before_first_simulation_state_mutation",
            "replay_is_forbidden",
            "fail_closed_conditions",
        },
        "backend_policy.acknowledgement",
    )
    _require(
        acknowledgement["scope"]
        == "exact_first_run_then_explicit_simulation_or_process_policy_scope",
        "acknowledgement scope differs",
    )
    _require(
        acknowledgement["default_reuse_scope"] == "simulation"
        and acknowledgement["optional_reuse_scopes"] == ["process"]
        and acknowledgement["process_scope_requires_explicit_opt_in"] is True,
        "acknowledgement scope opt-in differs",
    )
    _require(
        acknowledgement["required_digest_bindings"] == EXPECTED_ACK_BINDINGS,
        "acknowledgement binding inventory differs",
    )
    _require(
        acknowledgement["reusable_scope_digest_bindings"]
        == EXPECTED_ACK_SCOPE_BINDINGS,
        "acknowledgement reusable-scope binding inventory differs",
    )
    _require(
        acknowledgement["maximum_simulation_scopes"] == 32
        and acknowledgement["maximum_process_scopes"] == 32,
        "acknowledgement scope-cache bounds differ",
    )
    _require(
        acknowledgement["diagnostic_cache_disable_environment"]
        == "MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE",
        "acknowledgement diagnostic override differs",
    )
    _require(acknowledgement["reason_min_utf8_bytes"] == 8, "ack reason minimum changed")
    _require(acknowledgement["reason_max_utf8_bytes"] == 512, "ack reason maximum changed")
    _all_true(
        acknowledgement,
        {
            "all_mpi_ranks_must_agree_on_digest",
            "mpi_reuse_requires_exact_workload_consensus",
            "consume_before_first_simulation_state_mutation",
            "replay_is_forbidden",
            "reuse_is_process_local",
            "fork_inheritance_is_forbidden",
        },
        "backend_policy.acknowledgement",
    )
    _require(
        acknowledgement["fail_closed_conditions"] == EXPECTED_ACK_FAILURES,
        "ack failure inventory differs",
    )

    inversion = _exact_keys(
        data["performance_inversion_policy"],
        {
            "comparison_lanes",
            "same_physics_precision_stopping_condition",
            "unavailable_phase_representation",
            "equal_work_plan",
            "required_decomposition",
            "reason_classes",
            "required_telemetry",
            "registry",
        },
        "performance_inversion_policy",
    )
    _require(inversion["comparison_lanes"] == EXPECTED_LANES, "comparison lanes differ")
    _require(inversion["same_physics_precision_stopping_condition"] is True, "unequal work allowed")
    _require(
        inversion["unavailable_phase_representation"]
        == {
            "available": False,
            "measured_seconds": [None, None],
            "mean_seconds": None,
            "reason_code_required": True,
            "basis_per_sample_required": True,
            "native_sinks_per_sample_required": True,
            "cycle_indices": [1, 2],
        },
        "unavailable phase representation differs",
    )
    _require(
        inversion["equal_work_plan"]
        == {
            "path": "packaging/v110-m4-equal-work-plan.json",
            "sha256": "a00267645358bde4f26ca7cca84886e1ec385c1f013c68c363efe560dd5bc183",
        },
        "equal-work plan binding differs",
    )
    _require(inversion["required_decomposition"] == EXPECTED_PHASES, "phase decomposition differs")
    _require(inversion["required_telemetry"] == EXPECTED_TELEMETRY, "telemetry inventory differs")

    classes = _exact_keys(
        inversion["reason_classes"],
        {"hardware_crossover", "engine_overhead", "environment", "unknown"},
        "reason_classes",
    )
    hardware = _exact_keys(
        classes["hardware_crossover"],
        {"warning_may_be_terminal", "requires_size_run_length_sweep_and_profiler_evidence"},
        "reason_classes.hardware_crossover",
    )
    _all_true(hardware, set(hardware), "reason_classes.hardware_crossover")
    engine = _exact_keys(
        classes["engine_overhead"],
        {
            "warning_may_be_terminal",
            "optimization_and_remeasurement_required",
            "representative_high_impact_case_blocks_release",
        },
        "reason_classes.engine_overhead",
    )
    _require(engine["warning_may_be_terminal"] is False, "engine warning cannot be terminal")
    _all_true(
        engine,
        {"optimization_and_remeasurement_required", "representative_high_impact_case_blocks_release"},
        "reason_classes.engine_overhead",
    )
    environment = _exact_keys(
        classes["environment"],
        {"warning_may_be_terminal", "configuration_fix_or_fail_closed_required"},
        "reason_classes.environment",
    )
    _require(environment["warning_may_be_terminal"] is False, "environment warning cannot be terminal")
    _require(environment["configuration_fix_or_fail_closed_required"] is True, "environment fix disabled")
    unknown = _exact_keys(
        classes["unknown"],
        {
            "warning_may_be_terminal",
            "excluded_from_performance_claims",
            "representative_high_impact_case_blocks_release",
        },
        "reason_classes.unknown",
    )
    _require(unknown["warning_may_be_terminal"] is False, "unknown warning cannot be terminal")
    _all_true(
        unknown,
        {"excluded_from_performance_claims", "representative_high_impact_case_blocks_release"},
        "reason_classes.unknown",
    )
    registry = _exact_keys(
        inversion["registry"],
        {
            "schema",
            "schema_sha256",
            "initial_inventory",
            "initial_inventory_sha256",
            "records_directory",
            "release_gate_tool",
            "one_primary_class_per_observed_inversion",
            "all_observed_inversions_must_be_registered",
            "representative_selection_requires_immutable_inventory_and_rationale",
            "evidence_files_require_sha256",
            "release_rejects_open_representative_engine_or_unknown",
            "release_rejects_unregistered_observed_inversion",
        },
        "performance_inversion_policy.registry",
    )
    _require(registry["schema"] == "packaging/v110-performance-inversion-record.schema.json", "registry schema path changed")
    _require(HEX64.fullmatch(registry["schema_sha256"]) is not None, "registry schema hash invalid")
    _require(
        registry["initial_inventory"]
        == "packaging/v110-performance-inversion-inventory.json",
        "registry inventory path changed",
    )
    _require(
        HEX64.fullmatch(registry["initial_inventory_sha256"]) is not None,
        "registry inventory hash invalid",
    )
    _require(
        registry["records_directory"] == "performance-inversions/v1.1.0/records",
        "registry records directory changed",
    )
    _require(
        registry["release_gate_tool"]
        == "scripts/validate-v110-inversion-registry.py",
        "registry gate tool changed",
    )
    _all_true(
        registry,
        set(registry)
        - {
            "schema",
            "schema_sha256",
            "initial_inventory",
            "initial_inventory_sha256",
            "records_directory",
            "release_gate_tool",
        },
        "performance_inversion_policy.registry",
    )

    _require(data["performance_candidates"] == EXPECTED_CANDIDATES, "performance candidates differ")

    compatibility = _exact_keys(
        data["compatibility_target"],
        {"primary_platform", "maximum_glibc_floor", "required_native_test_environments", "non_linux_cuda_is_future_work"},
        "compatibility_target",
    )
    _require(compatibility["primary_platform"] == "linux-x86_64", "primary platform changed")
    _require(compatibility["maximum_glibc_floor"] == "2.35", "glibc floor regressed")
    _require(
        compatibility["required_native_test_environments"] == ["ubuntu-22.04", "one-newer-ubuntu"],
        "native portability matrix differs",
    )
    _require(compatibility["non_linux_cuda_is_future_work"] is True, "non-Linux CUDA scope changed")

    isolation = _exact_keys(
        data["development_isolation"],
        {
            "repository_relative_to_workspace",
            "required_branch",
            "origin_url",
            "independent_git_object_store",
            "git_alternates_forbidden",
            "m0_commit_must_have_frozen_base_as_first_parent",
            "builder_prefix_relative",
            "builder_lock",
            "builder_lock_sha256",
            "test_prefix_relative",
            "test_lock",
            "test_lock_sha256",
            "attestation_tool",
            "attestation_must_be_external_mode_600",
            "source_worktree_must_be_clean",
        },
        "development_isolation",
    )
    _all_true(
        isolation,
        {
            "independent_git_object_store",
            "git_alternates_forbidden",
            "m0_commit_must_have_frozen_base_as_first_parent",
            "attestation_must_be_external_mode_600",
            "source_worktree_must_be_clean",
        },
        "development_isolation",
    )
    _require(
        isolation["repository_relative_to_workspace"] == "gpmeep-v1.1.0-dev",
        "development repository changed",
    )
    _require(isolation["required_branch"] == "dev/v1.1.0", "development branch changed")
    _require(
        isolation["origin_url"] == "git@github.com:iverydj/gpmeep.git",
        "development origin changed",
    )
    _require(
        isolation["builder_prefix_relative"] == ".envs/gpmeep-v1.1-builder",
        "builder prefix changed",
    )
    _require(
        isolation["builder_lock"]
        == "environment/locks/package-builder-linux-64.lock",
        "builder lock path changed",
    )
    _require(
        isolation["test_prefix_relative"] == ".envs/gpmeep-v1.1-test",
        "test prefix changed",
    )
    _require(
        isolation["test_lock"] == "environment/locks/gpmeep-runtime-linux-64.lock",
        "test lock path changed",
    )
    _require(
        isolation["attestation_tool"] == "scripts/attest-v110-m0.py",
        "attestation tool changed",
    )
    for key in ("builder_lock_sha256", "test_lock_sha256"):
        _require(HEX64.fullmatch(isolation[key]) is not None, f"{key} is invalid")

    validation = _exact_keys(
        data["validation"],
        {
            "user_input_artifacts",
            "aunp_contract",
            "performance_repetitions",
            "cpu_baseline_requires_physical_core_topology",
            "deduplicate_examples_by_feature_coverage",
            "adversarial_audit_after_every_milestone",
            "retain_intermediate_timing_numerical_and_runtime_identity",
        },
        "validation",
    )
    _require(isinstance(validation["user_input_artifacts"], list), "user inputs must be a list")
    _require(
        [item.get("id") for item in validation["user_input_artifacts"]]
        == ["hor_tm_60deg_saw_v6", "aunp_r4000_repro_20260807"],
        "user input inventory differs",
    )
    first, second = validation["user_input_artifacts"]
    _exact_keys(
        first,
        {
            "id",
            "workspace_relative_path",
            "type",
            "sha256",
            "release_gate_required",
            "evaluation_status",
            "performance_claim_eligible",
        },
        "user_input.hor_tm",
    )
    _require(
        first
        == {
            "id": "hor_tm_60deg_saw_v6",
            "workspace_relative_path": "meep_gpu/test_example/Hor_TM_60deg_saw_v6.py",
            "type": "python",
            "sha256": "4a14fb7a6770802aeefa116768dd28ea3d92900f36fb06e1f01b9de5706f8bef",
            "release_gate_required": False,
            "evaluation_status": "NOT_EVALUATED_USER_DEFERRED_LONG_RUNTIME",
            "performance_claim_eligible": False,
        },
        "Hor_TM input contract differs",
    )
    _exact_keys(
        second,
        {
            "id",
            "workspace_relative_path",
            "type",
            "sha256",
            "release_gate_required",
            "evaluation_status",
            "performance_claim_eligible",
            "required_members",
        },
        "user_input.aunp",
    )
    _require(
        second["workspace_relative_path"]
        == "meep_gpu/test_example/aunp_r4000_repro_20260807code.tar.gz"
        and second["type"] == "tar.gz"
        and second["sha256"]
        == "448e01dcf433435fc5e00ef40ef0cb260870b002a113f3340eacee84f205ff96",
        "AuNP outer input contract differs",
    )
    _require(
        second["release_gate_required"] is True
        and second["evaluation_status"] == "REQUIRED"
        and second["performance_claim_eligible"] is True,
        "AuNP release status differs",
    )
    _require(
        second["required_members"]
        == {
            "aunp_r4000_repro/REPRODUCE.md": "39439d2cbdd24528952abb83a59ab5eb58e9b11dedad203c0a04912cd4da849b",
            "aunp_r4000_repro/fdtd/sim_config.py": "f514896f51a5368ffe3f79d389425f4a271fc79bd1d8396278b6f8468c85b0a1",
            "aunp_r4000_repro/fdtd/meep/run_production.py": "7ff087b40a6b4e96f2a2c72fdfa0536a666dd49999096fd60816b77a3988146a",
            "aunp_r4000_repro/fdtd/meep/aunp_periodic_fdtd.py": "a56ade815e2689f037da547ca077fc41505c761364c27e58382b13da5f349cec",
        },
        "AuNP archive member contract differs",
    )
    _require(validation["performance_repetitions"] == 2, "performance repetitions must be two")
    _all_true(
        validation,
        {
            "cpu_baseline_requires_physical_core_topology",
            "deduplicate_examples_by_feature_coverage",
            "adversarial_audit_after_every_milestone",
            "retain_intermediate_timing_numerical_and_runtime_identity",
        },
        "validation",
    )
    aunp = _exact_keys(
        validation["aunp_contract"],
        {"artifact_id", "polarization", "minimum_run", "dft_field_decay", "correctness_and_fixed_work_performance_are_separate"},
        "validation.aunp_contract",
    )
    _require(aunp["artifact_id"] == "aunp_r4000_repro_20260807", "AuNP artifact binding changed")
    _require(aunp["polarization"] == "TM", "AuNP must be TM only")
    _require(aunp["minimum_run"] == 20, "AuNP minimum run changed")
    _require(aunp["dft_field_decay"] == 5e-8, "AuNP DFT decay changed")
    _require(aunp["correctness_and_fixed_work_performance_are_separate"] is True, "AuNP timing contracts merged")

    integration = _exact_keys(
        data["gate_integration"],
        {
            "static_validator",
            "m0_attestation_tool",
            "negative_tests",
            "manual_no_email_actions_workflow",
            "package_builder",
            "installed_package_verifier",
            "candidate_build_receipt_tool",
            "final_release_receipt_tool",
            "release_evidence_generator",
            "publication_controller",
            "source_repository_slug",
            "public_repository_slug",
            "public_repository_id",
            "public_repository_node_id",
            "public_artifact_inspector",
            "private_source_authority_tool",
            "public_source_verifier",
            "public_publication_controller",
            "public_prior_tag",
            "public_snapshot_method",
            "development_build_mode_flag",
            "candidate_build_mode_flag",
            "development_package_build_suffix",
            "development_marker_suffix",
            "package_owned_candidate_receipt",
            "package_owned_contract_directory",
            "candidate_requires_external_attestation",
            "candidate_requires_exact_external_attestation_sha256",
            "candidate_requires_closed_inversion_registry",
            "candidate_receipt_never_authorizes_publication",
            "final_release_receipt_is_external",
            "final_release_receipt_binds_exact_package_sha256",
            "publication_requires_final_release_receipt",
            "publication_forbids_package_rebuild_or_mutation",
            "public_target_is_public",
            "public_target_actions_disabled",
            "public_target_immutable_releases",
            "public_forbids_build_or_package_mutation",
            "public_forbids_development_history",
            "public_package_must_be_byte_identical",
            "public_requires_prior_ref_identity",
            "public_requires_independent_publication_audit",
            "public_prior_release_is_source_only",
            "public_each_new_release_requires_signed_immutable_attestation",
        },
        "gate_integration",
    )
    _all_true(
        integration,
        {
            "candidate_requires_external_attestation",
            "candidate_requires_exact_external_attestation_sha256",
            "candidate_requires_closed_inversion_registry",
            "candidate_receipt_never_authorizes_publication",
            "final_release_receipt_is_external",
            "final_release_receipt_binds_exact_package_sha256",
            "publication_requires_final_release_receipt",
            "publication_forbids_package_rebuild_or_mutation",
            "public_target_is_public",
            "public_target_actions_disabled",
            "public_target_immutable_releases",
            "public_forbids_build_or_package_mutation",
            "public_forbids_development_history",
            "public_package_must_be_byte_identical",
            "public_requires_prior_ref_identity",
            "public_requires_independent_publication_audit",
            "public_prior_release_is_source_only",
            "public_each_new_release_requires_signed_immutable_attestation",
        },
        "gate_integration",
    )
    _require(integration["candidate_build_receipt_tool"] == "scripts/v110-release-gate-receipt.py", "candidate receipt tool changed")
    _require(integration["final_release_receipt_tool"] == "scripts/v110-final-release-receipt.py", "final receipt tool changed")
    _require(integration["release_evidence_generator"] == "scripts/v110-release-evidence.py", "release evidence generator changed")
    _require(integration["publication_controller"] == "scripts/publish-v110-release.sh", "publication controller changed")
    _require(integration["source_repository_slug"] == "iverydj/gpmeep", "source repository changed")
    _require(integration["public_repository_slug"] == "iverydj/gpmeep-releases", "public repository changed")
    _require(integration["private_source_authority_tool"] == "scripts/v110-release-only-receipt.py", "private source authority tool changed")
    _require(integration["public_source_verifier"] == "scripts/verify-public-source.py", "public source verifier changed")
    _require(integration["public_publication_controller"] == "scripts/publish-v110-public.py", "public publication controller changed")
    _require(integration["public_prior_tag"] == "v1.0.3", "public prior tag changed")
    _require(type(integration["public_repository_id"]) is int and integration["public_repository_id"] == 1372662716, "public repository ID changed")
    _require(integration["public_repository_node_id"] == "R_kgDOUdErvA", "public repository node ID changed")
    _require(integration["public_artifact_inspector"] == "scripts/verify-public-artifacts.py", "public artifact inspector changed")
    _require(integration["public_snapshot_method"] == "public-parent-preserved-runtime-source-v1", "public snapshot method changed")
    _require(integration["development_build_mode_flag"] == "--development-build", "development build flag changed")
    _require(integration["candidate_build_mode_flag"] == "--candidate-build", "candidate build flag changed")
    _require(integration["development_package_build_suffix"] == "_development", "development package build suffix changed")
    _require(integration["development_marker_suffix"] == ".NON_PUBLISHABLE", "development marker suffix changed")
    _require(integration["package_owned_candidate_receipt"] == "share/gpmeep/v110-candidate-build-receipt.json", "package candidate receipt path changed")
    _require(integration["package_owned_contract_directory"] == "share/gpmeep/packaging", "package contract directory changed")

    gates = _exact_keys(
        data["release_gates"],
        {
            "at_least_one_structural_optimization_has_repeatable_benefit",
            "no_material_regression_on_large_positive_workloads",
            "backend_guard_validated",
            "broad_correctness_validated",
            "multi_gpu_validated_on_real_hardware",
            "packaging_and_provenance_validated",
            "portability_matrix_validated",
            "all_milestone_adversarial_audits_pass",
            "no_unresolved_representative_engine_overhead",
            "no_unclassified_representative_gpu_cpu_inversion",
        },
        "release_gates",
    )
    _all_true(gates, set(gates), "release_gates")

    _require(
        canonical_contract_sha256(data) == EXPECTED_CANONICAL_CONTRACT_SHA256,
        "contract canonical digest differs; update contract, validator, and audit together",
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_user_inputs(data: dict[str, Any], workspace_root: pathlib.Path) -> list[dict[str, Any]]:
    workspace_root = workspace_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for item in _path(data, "validation", "user_input_artifacts"):
        _require(isinstance(item, dict), "user input entry must be an object")
        allowed = {
            "id",
            "workspace_relative_path",
            "type",
            "sha256",
            "release_gate_required",
            "evaluation_status",
            "performance_claim_eligible",
        }
        if item.get("type") == "tar.gz":
            allowed.add("required_members")
        _exact_keys(item, allowed, f"user_input.{item.get('id', 'unknown')}")
        relative = pathlib.PurePosixPath(item["workspace_relative_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe user input path")
        path = (workspace_root / pathlib.Path(*relative.parts)).resolve(strict=True)
        _require(path.is_relative_to(workspace_root) and path.is_file(), "user input escaped workspace")
        observed = _sha256(path)
        _require(observed == item["sha256"], f"user input hash mismatch: {item['id']}")
        members: dict[str, str] = {}
        if item["type"] == "tar.gz":
            required = item["required_members"]
            _require(isinstance(required, dict) and required, "archive member inventory is empty")
            with tarfile.open(path, "r:gz") as archive:
                names = [member.name for member in archive.getmembers()]
                for name, expected in required.items():
                    _require(names.count(name) == 1, f"archive member missing or duplicated: {name}")
                    member = archive.getmember(name)
                    _require(member.isfile(), f"archive member is not a regular file: {name}")
                    extracted = archive.extractfile(member)
                    _require(extracted is not None, f"cannot read archive member: {name}")
                    member_hash = hashlib.sha256(extracted.read()).hexdigest()
                    _require(member_hash == expected, f"archive member hash mismatch: {name}")
                    members[name] = member_hash
        records.append(
            {
                "id": item["id"],
                "path": str(path),
                "sha256": observed,
                "members": members,
                "release_gate_required": item["release_gate_required"],
                "evaluation_status": item["evaluation_status"],
                "performance_claim_eligible": item["performance_claim_eligible"],
            }
        )
    return records


def validate_campaign_sources(
    data: dict[str, Any], workspace_root: pathlib.Path
) -> list[dict[str, Any]]:
    workspace_root = workspace_root.resolve(strict=True)
    registry = _path(data, "performance_inversion_policy", "registry")
    inventory = _load_json((ROOT / registry["initial_inventory"]).read_bytes())
    campaign = _exact_keys(
        inventory["source_campaign"],
        {
            "id",
            "source_commit",
            "summary_evidence",
            "controller_evidence",
            "qualification",
        },
        "source_campaign",
    )
    _require(
        campaign["id"] == "v1.0.1-final-report-386e3c4a",
        "source campaign id differs",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", campaign["source_commit"]) is not None,
        "source campaign commit is invalid",
    )
    _require(
        campaign["qualification"]
        == "diagnostic_seed_only_requires_v110_equal_work_remeasurement",
        "source campaign qualification differs",
    )

    evidence_records: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("summary_evidence", "controller_evidence"):
        evidence = _exact_keys(
            campaign[name],
            {"workspace_relative_path", "sha256"},
            f"source_campaign.{name}",
        )
        _require(HEX64.fullmatch(evidence["sha256"]) is not None, "campaign evidence hash is invalid")
        relative = pathlib.PurePosixPath(evidence["workspace_relative_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "campaign evidence path is unsafe")
        path = (workspace_root / pathlib.Path(*relative.parts)).resolve(strict=True)
        _require(path.is_relative_to(workspace_root) and path.is_file(), "campaign evidence escaped workspace")
        _require(_sha256(path) == evidence["sha256"], f"source campaign evidence hash differs: {name}")
        loaded[name] = _load_json(path.read_bytes())
        evidence_records.append({"id": name, "path": str(path), "sha256": evidence["sha256"]})

    summary = loaded["summary_evidence"]
    controller = loaded["controller_evidence"]
    _require(controller.get("commit") == campaign["source_commit"], "campaign controller commit differs")
    _require(controller.get("outcome") == "PASS", "campaign controller did not pass")
    _require(
        controller.get("performance_evidence", {}).get("valid_for_speed_gate") is False,
        "historical campaign must remain diagnostic-only",
    )
    general = _path(summary, "general_examples")
    _require(
        general.get("controller_report_sha256") == campaign["controller_evidence"]["sha256"],
        "summary does not bind the campaign controller",
    )
    rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in general["generic_rows"]:
        if row.get("cpu_over_gpu", 1.0) < 1.0:
            rows[row["path"]] = ("general_cpu_vs_gpu1", row)
    for row in general["mpi_rows"]:
        if row.get("gpu1_seconds", 0.0) > row.get("cpu8_seconds", float("inf")):
            rows[row["path"]] = ("mpi_cpu8_vs_gpu1_gpu2", row)
    _require(len(rows) == 13, "historical slower-case inventory differs")
    routes = {
        item["case_id"]: item
        for item in controller.get("route_ledger", [])
        if item.get("case_id") in rows
    }
    _require(set(routes) == set(rows), "campaign routes do not cover every slower case")

    for observation in inventory["observations"]:
        workload_id = observation["workload_id"]
        _require(workload_id in rows and workload_id in routes, "observation lacks campaign evidence")
        source_lane, result_row = rows[workload_id]
        _require(observation["source_lane"] == source_lane, "observation source lane differs")
        identity = observation["workload_identity"]
        _require(identity["source_commit"] == campaign["source_commit"], "workload source commit differs")
        _require(identity["entrypoint"] == workload_id, "workload entrypoint differs")
        route = routes[workload_id]
        _require(identity["campaign_route_id"] == route["route_id"], "campaign route id differs")
        _require(identity["campaign_route_sha256"] == _canonical_sha256(route), "campaign route hash differs")
        _require(
            identity["campaign_task_record_sha256"]
            == route["primary_evidence"]["task_record_sha256"],
            "campaign task receipt differs",
        )
        _require(identity["campaign_result_sha256"] == _canonical_sha256(result_row), "campaign result hash differs")
        _require(identity["campaign_topologies"] == route["topologies"], "campaign topology differs")
        try:
            blob = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"{campaign['source_commit']}:{workload_id}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ContractError(f"cannot resolve historical workload bytes: {workload_id}") from error
        _require(hashlib.sha256(blob).hexdigest() == identity["source_file_sha256"], "historical workload byte hash differs")
    return evidence_records


def validate_documents() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    todo = FUTURE_TODO.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for marker in (
        "No automatic backend selection",
        "Hardware/workload crossover",
        "gpmeep engine or design overhead",
        "Environment or configuration defect",
        "Unknown",
        "A warning is not an optimization",
        "Ubuntu 22.04",
        "adversarial audit",
        "one-shot",
        "workload fingerprint",
        "inversion registry",
    ):
        _require(marker in plan, f"development plan is missing: {marker}")
    for marker in ("Linux aarch64", "ROCm/HIP", "macOS", "musl"):
        _require(marker in todo, f"future TODO is missing: {marker}")
    for marker in (
        "Released v1.0.x backend behavior",
        "v1.1.0 target: explicit CPU/CUDA choice",
        "v1.0.x-only automatic policy",
    ):
        _require(marker in readme, f"README backend version boundary is missing: {marker}")


def validate_bound_files(data: dict[str, Any]) -> None:
    registry = _path(data, "performance_inversion_policy", "registry")
    equal_work = _path(data, "performance_inversion_policy", "equal_work_plan")
    equal_work_path = ROOT / equal_work["path"]
    _require(equal_work_path.is_file(), "equal-work plan is absent")
    _require(
        _sha256(equal_work_path) == equal_work["sha256"],
        "equal-work plan hash mismatch",
    )
    schema_path = ROOT / registry["schema"]
    _require(schema_path.is_file(), "inversion schema is absent")
    _require(_sha256(schema_path) == registry["schema_sha256"], "inversion schema hash mismatch")
    schema = _load_json(schema_path.read_text(encoding="utf-8"))
    _require(schema.get("type") == "object", "inversion schema root type changed")
    _require(schema.get("additionalProperties") is False, "inversion schema permits unknown keys")
    primary = _path(schema, "properties", "primary_class", "enum")
    _require(primary == ["hardware_crossover", "engine_overhead", "environment", "unknown"], "inversion primary class enum differs")

    inventory_path = ROOT / registry["initial_inventory"]
    _require(inventory_path.is_file(), "inversion inventory is absent")
    _require(
        _sha256(inventory_path) == registry["initial_inventory_sha256"],
        "inversion inventory hash mismatch",
    )
    inventory = _load_json(inventory_path.read_text(encoding="utf-8"))
    _exact_keys(
        inventory,
        {
            "schema_version",
            "inventory_id",
            "immutable",
            "source_campaign",
            "selection_policy_version",
            "selection_rule",
            "observations",
        },
        "inversion inventory",
    )
    _require(inventory["schema_version"] == 1, "inventory schema changed")
    _require(inventory["immutable"] is True, "inversion inventory must be immutable")
    campaign = _exact_keys(
        inventory["source_campaign"],
        {
            "id",
            "source_commit",
            "summary_evidence",
            "controller_evidence",
            "qualification",
        },
        "inversion inventory source campaign",
    )
    _require(re.fullmatch(r"[0-9a-f]{40}", campaign["source_commit"]) is not None, "campaign commit is invalid")
    for key in ("summary_evidence", "controller_evidence"):
        evidence = _exact_keys(
            campaign[key],
            {"workspace_relative_path", "sha256"},
            f"source campaign {key}",
        )
        _require(HEX64.fullmatch(evidence["sha256"]) is not None, "campaign evidence hash is invalid")
    _require(len(inventory["observations"]) == 13, "inversion inventory count differs")
    workload_ids: list[str] = []
    representatives = 0
    for position, item in enumerate(inventory["observations"]):
        _exact_keys(
            item,
            {
                "workload_id",
                "workload_name_sha256",
                "workload_identity",
                "source_lane",
                "representative",
                "impact",
                "rationale",
            },
            f"inversion inventory observation {position}",
        )
        _require(HEX64.fullmatch(item["workload_name_sha256"]) is not None, "invalid workload-name digest")
        _require(item["impact"] in {"low", "medium", "high"}, "invalid inversion impact")
        _require(isinstance(item["rationale"], str) and len(item["rationale"]) >= 32, "weak representative rationale")
        _require(
            item["workload_name_sha256"]
            == hashlib.sha256(item["workload_id"].encode("utf-8")).hexdigest(),
            "workload-name digest differs",
        )
        identity = _exact_keys(
            item["workload_identity"],
            {
                "source_commit",
                "source_file_sha256",
                "entrypoint",
                "canonical_entrypoint_args",
                "precision",
                "physics_and_stopping_source",
                "campaign_route_id",
                "campaign_route_sha256",
                "campaign_task_record_sha256",
                "campaign_result_sha256",
                "campaign_topologies",
                "case_contract_sha256",
            },
            f"workload identity {position}",
        )
        for key in (
            "source_file_sha256",
            "campaign_route_sha256",
            "campaign_task_record_sha256",
            "campaign_result_sha256",
            "case_contract_sha256",
        ):
            _require(HEX64.fullmatch(identity[key]) is not None, f"invalid workload identity {key}")
        _require(re.fullmatch(r"[0-9a-f]{40}", identity["source_commit"]) is not None, "invalid workload source commit")
        _require(identity["entrypoint"] == item["workload_id"], "workload identity entrypoint differs")
        _require(identity["canonical_entrypoint_args"] == [], "historical entrypoint arguments differ")
        _require(identity["precision"] == "fp32", "historical precision differs")
        _require(identity["physics_and_stopping_source"] == "entrypoint_source_bytes", "physics/stopping binding differs")
        _require(
            identity["case_contract_sha256"]
            == _canonical_sha256({key: value for key, value in identity.items() if key != "case_contract_sha256"}),
            "workload case contract digest differs",
        )
        workload_ids.append(item["workload_id"])
        representatives += item["representative"] is True
    _require(len(set(workload_ids)) == len(workload_ids), "inventory repeats a workload")
    _require(representatives == 8, "representative inversion inventory differs")


def validate_gate_integration(data: dict[str, Any]) -> None:
    validate_active_distribution_identity()
    dependabot_path = ROOT / ".github/dependabot.yml"
    _require(
        dependabot_path.is_file() and not dependabot_path.is_symlink(),
        "Dependabot policy must be a direct regular file",
    )
    validate_dependabot_policy_bytes(dependabot_path.read_bytes())
    integration = data["gate_integration"]
    validator_name = pathlib.Path(integration["static_validator"]).name
    for key in ("manual_no_email_actions_workflow", "package_builder", "installed_package_verifier"):
        path = ROOT / integration[key]
        _require(path.is_file(), f"gate integration file is absent: {integration[key]}")
        text = path.read_text(encoding="utf-8")
        _require(validator_name in text, f"static contract validator is not wired into {integration[key]}")
    registry_gate = ROOT / _path(data, "performance_inversion_policy", "registry", "release_gate_tool")
    _require(registry_gate.is_file(), "inversion registry release gate is absent")
    builder_text = (ROOT / integration["package_builder"]).read_text(encoding="utf-8")
    _require(registry_gate.name in builder_text, "inversion registry release gate is not wired into package builder")
    candidate_tool = ROOT / integration["candidate_build_receipt_tool"]
    final_tool = ROOT / integration["final_release_receipt_tool"]
    evidence_generator = ROOT / integration["release_evidence_generator"]
    publication_controller = ROOT / integration["publication_controller"]
    source_authority_tool = ROOT / integration["private_source_authority_tool"]
    public_source_verifier = ROOT / integration["public_source_verifier"]
    public_inspector = ROOT / integration["public_artifact_inspector"]
    public_controller = ROOT / integration["public_publication_controller"]
    for path, label in (
        (candidate_tool, "candidate-build receipt tool"),
        (final_tool, "final-release receipt tool"),
        (evidence_generator, "release evidence generator"),
        (publication_controller, "private publication controller"),
        (source_authority_tool, "private source authority tool"),
        (public_source_verifier, "public source verifier"),
        (public_inspector, "public artifact inspector"),
        (public_controller, "public publication controller"),
    ):
        _require(path.is_file(), f"{label} is absent")
    verifier_text = (ROOT / integration["installed_package_verifier"]).read_text(encoding="utf-8")
    recipe_text = (ROOT / "packaging/conda/build.sh").read_text(encoding="utf-8")
    meta_text = (ROOT / "packaging/conda/meta.yaml").read_text(encoding="utf-8")
    workflow_text = (ROOT / integration["manual_no_email_actions_workflow"]).read_text(encoding="utf-8")
    negative_test_text = (ROOT / integration["negative_tests"]).read_text(encoding="utf-8")
    for marker in (
        integration["development_build_mode_flag"],
        integration["candidate_build_mode_flag"],
        "--candidate-attestation",
        "--candidate-attestation-sha256",
        "--m4-candidate-manifest",
        "--require-closed",
        integration["development_package_build_suffix"],
        integration["development_marker_suffix"],
        candidate_tool.name,
    ):
        _require(marker in builder_text, f"package builder omits gate marker: {marker}")
    _require(
        integration["package_owned_candidate_receipt"] in verifier_text,
        "installed verifier omits package-owned candidate receipt",
    )
    _require("--require-candidate" in verifier_text, "installed verifier can skip candidate authority")
    for marker in (
        "--final-release-receipt",
        "--qualification-manifest",
        "--release-evidence-root",
        final_tool.name,
    ):
        _require(marker in verifier_text, f"installed verifier omits final-release marker: {marker}")
    for filename in (
        "v110-development-contract.json",
        "v110-m4-equal-work-plan.json",
        "v110-performance-inversion-record.schema.json",
        "v110-performance-inversion-inventory.json",
        "v110-candidate-build-receipt.json",
        candidate_tool.name,
        final_tool.name,
    ):
        _require(filename in recipe_text, f"package recipe omits gate file: {filename}")
    for marker in (
        final_tool.name,
        candidate_tool.name,
        "--final-receipt",
        "--candidate-receipt",
        "--qualification-manifest",
        "--execute",
        "refs/tags/$TAG",
    ):
        _require(
            marker in publication_controller.read_text(encoding="utf-8"),
            f"publication controller omits trust marker: {marker}",
        )
    source_controller_text = publication_controller.read_text(encoding="utf-8")
    _require(
        integration["source_repository_slug"] in source_controller_text,
        "source publication controller is not repository-restricted",
    )
    public_text = public_controller.read_text(encoding="utf-8")
    for marker in (
        source_authority_tool.name, public_source_verifier.name, public_inspector.name,
        integration["public_repository_slug"], str(integration["public_repository_id"]),
        integration["public_repository_node_id"], integration["public_snapshot_method"],
        "--execute", "--resume", "verify_audit", "public_tag_object",
        "verify_source_authority_files", "verify-asset", "immutable-releases",
    ):
        _require(marker in public_text, f"public controller omits trust marker: {marker}")
    _require("verify_source_authority_files" in source_authority_tool.read_text(encoding="utf-8"),
             "private source authority replay is absent")
    _require("GPMEEP_V110_BUILD_MODE" in meta_text, "package metadata does not distinguish build modes")
    for marker in (
        "scripts/create-env.sh v110-test",
        ".envs/gpmeep-v1.1-test/bin/python",
        'version("jsonschema") == "4.26.0"',
        "scripts.tests.test_v110_development_contract.V110DevelopmentContractTest",
    ):
        _require(marker in workflow_text, f"manual workflow omits locked schema runtime marker: {marker}")
    host_python = "/usr/bin/" + "python3"
    _require(host_python not in negative_test_text, "contract tests may escape to host Python")


def validate(
    contract_path: pathlib.Path = DEFAULT_CONTRACT,
    *,
    workspace_root: pathlib.Path | None = None,
    require_user_inputs: bool = True,
    validate_source_files: bool = True,
) -> dict[str, Any]:
    raw = contract_path.read_bytes()
    data = _load_json(raw)
    _require(isinstance(data, dict), "contract root must be an object")
    validate_contract_data(data)
    validate_bound_files(data)
    if validate_source_files:
        validate_documents()
        validate_gate_integration(data)
    inputs: list[dict[str, Any]] = []
    campaign_inputs: list[dict[str, Any]] = []
    if require_user_inputs:
        inputs = validate_user_inputs(data, workspace_root or ROOT.parent)
        campaign_inputs = validate_campaign_sources(data, workspace_root or ROOT.parent)
    try:
        relative = str(contract_path.relative_to(ROOT))
    except ValueError:
        relative = str(contract_path)
    return {
        "schema_version": 2,
        "state": "PASS",
        "release_train": "1.1.0",
        "base_commit": EXPECTED_BASE,
        "contract": relative,
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "contract_canonical_sha256": canonical_contract_sha256(data),
        "user_inputs_checked": len(inputs),
        "campaign_inputs_checked": len(campaign_inputs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workspace-root", type=pathlib.Path, default=ROOT.parent)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="skip external user-input bytes; intended only for source-only CI/build preflight",
    )
    parser.add_argument(
        "--installed-package",
        action="store_true",
        help="validate the canonical contract and bundled schema/inventory without source-tree docs or external inputs",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        _require(not (args.contract_only and args.installed_package), "validation modes are mutually exclusive")
        result = validate(
            args.contract.resolve(),
            workspace_root=args.workspace_root,
            require_user_inputs=not (args.contract_only or args.installed_package),
            validate_source_files=not args.installed_package,
        )
    except (ContractError, OSError, json.JSONDecodeError, tarfile.TarError) as exc:
        if args.json:
            print(json.dumps({"schema_version": 2, "state": "FAIL", "error": str(exc)}, sort_keys=True))
        else:
            print(f"v1.1.0 development contract: FAIL: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("v1.1.0 development contract: PASS")
        print(f"contract_sha256={result['contract_sha256']}")
        print(f"contract_canonical_sha256={result['contract_canonical_sha256']}")
        print(f"user_inputs_checked={result['user_inputs_checked']}")
        print(f"campaign_inputs_checked={result['campaign_inputs_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
