from __future__ import annotations

import copy
import contextlib
import base64
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from unittest import mock

from scripts.tests.test_v110_development_contract import (
    CONTRACT,
    RECEIPT as CANDIDATE,
    REGISTRY,
    VALIDATOR,
    candidate_manifest_fixture,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FINAL = load_script(
    "v110_final_receipt", ROOT / "scripts/v110-final-release-receipt.py"
)
NATIVE = load_script(
    "v110_native_witness", ROOT / "scripts/v110-native-witness.py"
)


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _json_ref(
    root: pathlib.Path, role: str, payload: dict[str, object]
) -> dict[str, object]:
    path = root / f"{role}.json"
    _write_json(path, payload)
    content_id = next(
        (
            payload[key]
            for key in ("receipt_id", "report_id", "content_id", "plan_id", "terminal_id")
            if isinstance(payload.get(key), str)
        ),
        FINAL._canonical_sha256(payload),
    )
    schema = payload.get("schema")
    if schema is None:
        schema = f"schema_version:{payload['schema_version']}"
    return {
        "role": role,
        "bundle_relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": FINAL._sha256(path),
        "schema": schema,
        "content_id": content_id,
    }


def _text_ref(
    root: pathlib.Path, role: str, suffix: str, media_type: str, text: str
) -> dict[str, object]:
    path = root / f"{role}{suffix}"
    path.write_text(text, encoding="utf-8")
    digest = FINAL._sha256(path)
    return {
        "role": role,
        "bundle_relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "schema": media_type,
        "content_id": digest,
    }


def _tar_bytes(
    files: dict[str, bytes], *, softlinks: dict[str, str] | None = None
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755 if name.startswith("bin/") else 0o644
            archive.addfile(info, io.BytesIO(payload))
        for name, target in (softlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            info.mode = 0o777
            archive.addfile(info)
    return stream.getvalue()


def _zstd(payload: bytes) -> bytes:
    result = subprocess.run(
        ["/usr/bin/zstd", "-q", "-c"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def _write_conda_v2(
    path: pathlib.Path,
    candidate_bytes: bytes,
    *,
    build: str = "cuda124_mpi_openmpi_py311_0",
    payload_overrides: dict[str, bytes] | None = None,
    path_metadata_overrides: dict[str, dict[str, object]] | None = None,
    path_metadata_removals: dict[str, set[str]] | None = None,
    softlinks: dict[str, str] | None = None,
) -> None:
    payload_files = {
        FINAL.CANDIDATE_RECEIPT_PATH: candidate_bytes,
        FINAL.INSTALLED_VERIFIER_PATH: b"#!/usr/bin/env python3\n",
        FINAL.INSTALLED_LAUNCHER_PATH: b"#!/bin/bash -p\n",
        FINAL.INSTALLED_SELF_CHECK_PATH: b"#!/usr/bin/env python3\n",
        "lib/libmeep.so.38": b"\x7fELFlibmeep-core\n",
        "lib/python3.11/site-packages/meep/_meep.so": b"\x7fELFpython-extension-core\n",
    }
    payload_files.update(payload_overrides or {})
    if FINAL.GLIBC_AUDIT_PATH not in payload_files:
        elf_files = []
        for name, payload in sorted(payload_files.items()):
            if payload.startswith(b"\x7fELF"):
                elf_files.append(
                    {
                        "path": name,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "required_versions": ["2.17"],
                        "maximum_required": "2.17",
                    }
                )
        payload_files[FINAL.GLIBC_AUDIT_PATH] = json.dumps(
            {
                "schema": "gpmeep-glibc-compatibility-v1",
                "state": "PASS",
                "maximum_allowed": "2.17",
                "overall_maximum_required": "2.17",
                "elf_count": len(elf_files),
                "elf_files": elf_files,
            },
            sort_keys=True,
        ).encode("utf-8")
    softlinks = softlinks or {}
    if set(payload_files) & set(softlinks):
        raise ValueError("fixture path cannot be both a regular file and a softlink")
    owned = sorted(set(payload_files) | set(softlinks))
    path_entries = [
        {
            "_path": item,
            "path_type": "hardlink",
            "sha256": hashlib.sha256(payload_files[item]).hexdigest(),
            "size_in_bytes": len(payload_files[item]),
            **((path_metadata_overrides or {}).get(item, {})),
        }
        for item in sorted(payload_files)
    ]
    for item, target in sorted(softlinks.items()):
        resolved = pathlib.PurePosixPath(item).parent.joinpath(target)
        resolved_text = str(resolved)
        target_payload = payload_files.get(resolved_text)
        metadata: dict[str, object] = {
            "_path": item,
            "path_type": "softlink",
        }
        if target_payload is not None:
            metadata.update(
                sha256=hashlib.sha256(target_payload).hexdigest(),
                size_in_bytes=len(target_payload),
            )
        metadata.update((path_metadata_overrides or {}).get(item, {}))
        for key in (path_metadata_removals or {}).get(item, set()):
            metadata.pop(key, None)
        path_entries.append(metadata)
    path_entries.sort(key=lambda item: str(item["_path"]))
    info_files = {
        "info/index.json": json.dumps(
            {
                "name": "gpmeep",
                "version": "1.1.0",
                "build": build,
                "build_number": 0,
                "subdir": "linux-64",
            },
            sort_keys=True,
        ).encode("utf-8"),
        "info/paths.json": json.dumps(
            {"paths_version": 1, "paths": path_entries},
            sort_keys=True,
        ).encode("utf-8"),
        "info/files": ("\n".join(owned) + "\n").encode("utf-8"),
    }
    stem = path.name.removesuffix(".conda")
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", '{"conda_pkg_format_version":2}')
        archive.writestr(
            f"pkg-{stem}.tar.zst",
            _zstd(_tar_bytes(payload_files, softlinks=softlinks)),
        )
        archive.writestr(f"info-{stem}.tar.zst", _zstd(_tar_bytes(info_files)))


class V110FinalReleaseReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.source_commit = "c" * 40
        contract_gate = self.root / "contract-gate.json"
        registry_gate = self.root / "registry-gate.json"
        attestation_gate = self.root / "attestation-gate.json"
        contract_result = VALIDATOR.validate(require_user_inputs=False)
        registry_result = REGISTRY.validate_registry(
            require_closed=False, evidence_root=None
        )
        registry_result.update(
            state="PASS",
            registered_count=13,
            missing_count=0,
            blocking_count=0,
            closed_count=13,
            open_count=0,
            missing_workloads=[],
            blocking_workloads=[],
            open_workloads=[],
        )
        _write_json(contract_gate, contract_result)
        _write_json(registry_gate, registry_result)
        candidate_manifest = candidate_manifest_fixture(
            self.root, self.source_commit
        )
        closed_records = self.root / "candidate-closed-records"
        closed_records.mkdir()
        closed_record_refs = []
        for index in range(13):
            path = closed_records / f"record-{index:02d}.json"
            path.write_text(json.dumps({"record": index}) + "\n")
            closed_record_refs.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        closed_terminal_payload = {
            "schema": CANDIDATE.CLOSED_REGISTRY_SCHEMA,
            "state": "PASS",
            "records_directory": closed_records.relative_to(self.root).as_posix(),
            "records": closed_record_refs,
            "registry_result": registry_result,
        }
        closed_terminal_payload["terminal_id"] = CANDIDATE._canonical_sha256(
            closed_terminal_payload
        )
        closed_terminal = self.root / "candidate-closed-registry.json"
        _write_json(closed_terminal, closed_terminal_payload)
        self.closed_registry_authority = {
            "file": {
                "bundle_relative_path": closed_terminal.name,
                "size_bytes": closed_terminal.stat().st_size,
                "sha256": hashlib.sha256(closed_terminal.read_bytes()).hexdigest(),
            },
            "terminal": closed_terminal_payload,
        }
        self.closed_records = closed_records
        self.closed_terminal = closed_terminal
        attestation_gate.write_text('{"fixture":"attestation"}\n')
        expected_attestation = hashlib.sha256(attestation_gate.read_bytes()).hexdigest()
        evidence_files = [candidate_manifest, closed_terminal]
        evidence_files.extend(sorted(closed_records.iterdir()))
        evidence_files.extend(
            self.root / item["bundle_relative_path"]
            for item in json.loads(candidate_manifest.read_text())["evidence"].values()
        )
        self.attested = {
            "repository": {
                "m0_commit": "b" * 40,
                "current_head": self.source_commit,
                "clean": True,
            },
            "external_evidence": [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in evidence_files
            ],
        }
        self.contract_result = contract_result
        self.registry_result = registry_result
        self.source_policy = self.root / "independent-source-policy.json"
        self.source_policy.write_text('{"fixture":"operator-authority"}\n')
        self.source_policy_sha256 = hashlib.sha256(
            self.source_policy.read_bytes()
        ).hexdigest()
        self.source_verifier = self.root / "independent-source-verifier.py"
        self.source_verifier.write_bytes(
            (ROOT / "scripts/verify-v110-release-source-transition.py").read_bytes()
        )
        self.source_verifier_sha256 = hashlib.sha256(
            self.source_verifier.read_bytes()
        ).hexdigest()
        self.candidate_inputs = {
            "candidate_contract_gate_path": contract_gate,
            "candidate_registry_gate_path": registry_gate,
            "candidate_attestation_path": attestation_gate,
            "candidate_manifest_path": candidate_manifest,
            "candidate_evidence_root": self.root,
            "source_policy_path": self.source_policy,
            "expected_source_policy_sha256": self.source_policy_sha256,
            "source_verifier_path": self.source_verifier,
            "expected_source_verifier_sha256": self.source_verifier_sha256,
        }
        with self._candidate_producer_patches():
            candidate = CANDIDATE.create(
                mode="candidate",
                source_commit=self.source_commit,
                contract_gate_path=contract_gate,
                registry_gate_path=registry_gate,
                attestation_gate_path=attestation_gate,
                expected_attestation_sha256=expected_attestation,
                candidate_manifest_path=candidate_manifest,
                evidence_root=self.root,
                source_policy_path=self.source_policy,
                expected_source_policy_sha256=self.source_policy_sha256,
                source_verifier_path=self.source_verifier,
                expected_source_verifier_sha256=self.source_verifier_sha256,
                closed_registry_terminal_path=closed_terminal,
                expected_closed_registry_sha256=hashlib.sha256(
                    closed_terminal.read_bytes()
                ).hexdigest(),
                expected_closed_registry_terminal_id=closed_terminal_payload[
                    "terminal_id"
                ],
                closed_registry_records_directory=closed_records,
            )
        self.candidate_receipt = self.root / "candidate-receipt.json"
        CANDIDATE._write_new(self.candidate_receipt, candidate)
        self.candidate_id = candidate["receipt_id"]

        self.package = (
            self.root / "gpmeep-1.1.0-cuda124_mpi_openmpi_py311_0.conda"
        )
        _write_conda_v2(self.package, self.candidate_receipt.read_bytes())
        self.package_sha256 = FINAL._sha256(self.package)
        development_receipt = copy.deepcopy(candidate)
        development_receipt["build_mode"] = "development"
        development_receipt["state"] = "DEVELOPMENT_ONLY"
        development_receipt.pop("receipt_id", None)
        development_receipt["receipt_id"] = FINAL._canonical_sha256(
            development_receipt
        )
        self.development_receipt_id = development_receipt["receipt_id"]
        self.measured_package = self.root / "gpmeep-1.1.0-measured-development.conda"
        _write_conda_v2(
            self.measured_package,
            (json.dumps(development_receipt, sort_keys=True) + "\n").encode(),
            build="cuda124_mpi_openmpi_py311_0_development",
        )
        self.measured_package_sha256 = FINAL._sha256(self.measured_package)
        self.package_identity = {
            "file_name": self.package.name,
            "size_bytes": self.package.stat().st_size,
            "sha256": self.package_sha256,
            "format": "conda-v2",
            "name": "gpmeep",
            "version": "1.1.0",
            "build": "cuda124_mpi_openmpi_py311_0",
            "subdir": "linux-64",
            "embedded_candidate_receipt_id": self.candidate_id,
        }
        installed = []
        self.platform_bindings = {}
        self.native_witness_refs = {}
        self.installed_report_ids = []
        for index, platform in enumerate(
            ("ubuntu-22.04", "ubuntu-24.04")
        ):
            prefix = self.root / f"prefix-{index}"
            (prefix / FINAL.INSTALLED_VERIFIER_PATH).parent.mkdir(parents=True)
            (prefix / FINAL.INSTALLED_LAUNCHER_PATH).parent.mkdir(parents=True, exist_ok=True)
            (prefix / FINAL.INSTALLED_VERIFIER_PATH).write_bytes(b"#!/usr/bin/env python3\n")
            (prefix / FINAL.INSTALLED_LAUNCHER_PATH).write_bytes(b"#!/bin/bash -p\n")
            (prefix / FINAL.INSTALLED_SELF_CHECK_PATH).write_bytes(b"#!/usr/bin/env python3\n")
            (prefix / FINAL.INSTALLED_LAUNCHER_PATH).chmod(0o755)
            payload: dict[str, object] = {
                "schema_version": 1,
                "state": "PASS",
                "prefix": str(prefix),
                "conda_record": {},
                "package_url": self.package.name,
                "package_sha256": self.package_sha256,
                "ownership": {},
                "manifest": {"source_commit": self.source_commit},
                "runtime": {},
                "v110_candidate_gate": {"receipt_id": self.candidate_id},
            }
            payload["report_id"] = FINAL._canonical_sha256(payload)
            self.installed_report_ids.append(payload["report_id"])
            role = f"installed_provenance_{index}"
            report_ref = _json_ref(self.root, role, payload)
            version = platform.removeprefix("ubuntu-")
            os_release_bytes = f'ID=ubuntu\nVERSION_ID="{version}"\n'.encode()
            namespaces = {
                "mnt": f"mnt:[{1000 + index}]",
                "pid": f"pid:[{2000 + index}]",
                "user": f"user:[{3000 + index}]",
            }
            witness: dict[str, object] = {
                "schema": NATIVE.SCHEMA,
                "state": "PASS",
                "platform": platform,
                "prefix": str(prefix.resolve()),
                "source_commit": self.source_commit,
                "package_sha256": self.package_sha256,
                "candidate_receipt_id": self.candidate_id,
                "installed_report_id": payload["report_id"],
                "installed_report_sha256": report_ref["sha256"],
                "image_identity": f"sha256:{index + 1:064x}",
                "capture_source": "/etc/os-release",
                "os_release": {
                    "id": "ubuntu",
                    "version_id": version,
                    "size_bytes": len(os_release_bytes),
                    "sha256": hashlib.sha256(os_release_bytes).hexdigest(),
                },
                "namespaces": namespaces,
                "namespace_id": NATIVE._canonical_sha256(namespaces),
                "kernel": {
                    "sysname": "Linux",
                    "release": f"fixture-{index}",
                    "machine": "x86_64",
                },
                "libc": "glibc 2.35" if index == 0 else "glibc 2.39",
                "cgroup_sha256": f"{index + 2:064x}",
            }
            witness["environment_id"] = NATIVE._canonical_sha256(witness)
            witness["terminal_id"] = NATIVE._canonical_sha256(witness)
            native_ref = _json_ref(self.root, f"native_witness_{index}", witness)
            self.native_witness_refs[platform] = native_ref
            self_check: dict[str, object] = {
                "schema_version": 2,
                "state": "PASS",
                "mode": "gpu2",
                "fixture_prefix": str(prefix.resolve()),
                "fixture_package_sha256": self.package_sha256,
                "fixture_candidate_receipt_id": self.candidate_id,
            }
            self_check["report_id"] = FINAL._canonical_sha256(self_check)
            self_check_ref = _json_ref(
                self.root, f"self_check_{index}", self_check
            )
            environment_id = witness["environment_id"]
            self.platform_bindings[platform] = {
                "environment_id": environment_id,
                "prefix": str(prefix.resolve()),
                "os_release_sha256": witness["os_release"]["sha256"],
                "installed_report_id": payload["report_id"],
                "namespace_id": witness["namespace_id"],
                "image_identity": witness["image_identity"],
                "kernel": witness["kernel"],
                "libc": witness["libc"],
            }
            installed.append(
                {
                    "environment_id": environment_id,
                    "platform": platform,
                    "prefix": str(prefix),
                    "native_witness": native_ref,
                    "self_check": self_check_ref,
                    "package_sha256": self.package_sha256,
                    "source_commit": self.source_commit,
                    "candidate_receipt_id": self.candidate_id,
                    "report": report_ref,
                }
            )

        final_findings = [
            {
                "id": finding,
                "severity": "HIGH",
                "state": "CLOSED",
                "evidence_sha256": hashlib.sha256(finding.encode()).hexdigest(),
            }
            for finding in FINAL.FINAL_AUDIT_FINDINGS
        ]
        final_decision: dict[str, object] = {
            "decision": "APPROVE_RELEASE",
            "approver": "fixture-operator",
            "finding_ids": list(FINAL.FINAL_AUDIT_FINDINGS),
            "package_sha256": self.package_sha256,
        }
        final_decision["decision_id"] = FINAL._canonical_sha256(final_decision)
        final_audit_producer: dict[str, object] = {
            "schema": FINAL.FINAL_AUDIT_PRODUCER_SCHEMA,
            "state": "PASS",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "auditor": "independent-fixture-auditor",
            "scope": {
                "independent": True,
                "exact_source_commit": self.source_commit,
                "exact_package_sha256": self.package_sha256,
                "finding_ids": list(FINAL.FINAL_AUDIT_FINDINGS),
            },
            "findings": final_findings,
            "human_release_decision": final_decision,
        }
        final_audit_producer["terminal_id"] = FINAL._canonical_sha256(
            final_audit_producer
        )
        final_audit_producer_ref = _json_ref(
            self.root, "independent_final_audit", final_audit_producer
        )
        self.final_findings = final_findings
        self.final_decision = final_decision
        self.final_audit_producer_ref = final_audit_producer_ref
        final_audit_terminal = {
            key: value for key, value in final_audit_producer_ref.items() if key != "role"
        }

        qualification_payloads: dict[str, dict[str, object]] = {
            "installed_correctness": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["installed_correctness"],
                "state": "PASS",
                "numerical_oracles_pass": True,
                "m4_report_id": "e" * 64,
                "development_package_receipt_id": self.development_receipt_id,
            },
            "installed_multi_gpu": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["installed_multi_gpu"],
                "state": "PASS",
                "distinct_gpu_uuids": True,
                "cpu_fallback_count": 0,
                "m4_report_id": "e" * 64,
                "development_package_receipt_id": self.development_receipt_id,
            },
            "installed_aunp": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["installed_aunp"],
                "state": "PASS",
                "polarization": "TM",
                "minimum_run": 20,
                "dft_field_decay": 5e-8,
                "installed_prefix": self.platform_bindings["ubuntu-22.04"]["prefix"],
                "package_provenance_report_id": self.installed_report_ids[0],
            },
            "ubuntu_22_04_native": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["ubuntu_22_04_native"],
                "state": "PASS",
                "platform": "ubuntu-22.04",
                "native_install": True,
                **self.platform_bindings["ubuntu-22.04"],
            },
            "newer_ubuntu_native": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["newer_ubuntu_native"],
                "state": "PASS",
                "platform": "ubuntu-24.04",
                "native_install": True,
                **self.platform_bindings["ubuntu-24.04"],
            },
            "glibc_floor": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["glibc_floor"],
                "state": "PASS",
                "overall_maximum_required": "2.17",
                "installed_report_ids": sorted(self.installed_report_ids),
            },
            "final_adversarial_audit": {
                "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["final_adversarial_audit"],
                "state": "PASS",
                "open_critical_high_medium": 0,
                "producer_terminal": final_audit_terminal,
            },
            "closed_inversion_registry": {
                "schema": "gpmeep-v110-closed-inversion-registry-v1",
                "state": "PASS",
                "records_directory": "closed-inversion-records",
                "records": [{} for _ in range(13)],
                "registry_result": {
                    "schema_version": 1,
                    "state": "PASS",
                    "inventory_sha256": "f" * 64,
                    "observed_count": 13,
                    "registered_count": 13,
                    "missing_count": 0,
                    "blocking_count": 0,
                    "closed_count": 13,
                    "open_count": 0,
                    "missing_workloads": [],
                    "blocking_workloads": [],
                    "open_workloads": [],
                },
            },
        }
        closure = qualification_payloads["closed_inversion_registry"]
        closure["terminal_id"] = FINAL._canonical_sha256(closure)
        qualification_refs = {
            role: _json_ref(self.root, role, payload)
            for role, payload in qualification_payloads.items()
        }
        self.final_audit_authority = self.root / final_audit_producer_ref["bundle_relative_path"]
        self.final_audit_authority_sha256 = FINAL._sha256(self.final_audit_authority)
        qualification_refs["final_report_markdown"] = _text_ref(
            self.root,
            "final_report_markdown",
            ".md",
            "text/markdown",
            "# gpmeep v1.1.0 final report\n",
        )
        qualification_refs["final_report_html"] = _text_ref(
            self.root,
            "final_report_html",
            ".html",
            "text/html",
            '<html><body><img data-plot-id="speedup" src="data:image/png;base64,'
            + base64.b64encode(PNG_1X1).decode("ascii")
            + '"></body></html>\n',
        )
        complete_payload: dict[str, object] = {
            "schema": FINAL.M4_COMPLETE_SCHEMA,
            "outcome": "PASS",
        }
        complete_path = self.root / "M4_COMPLETE"
        _write_json(complete_path, complete_payload)
        performance_ref = {
            "role": "three_way_performance",
            "bundle_relative_path": complete_path.name,
            "size_bytes": complete_path.stat().st_size,
            "sha256": FINAL._sha256(complete_path),
            "schema": FINAL.M4_COMPLETE_SCHEMA,
            "content_id": FINAL._canonical_sha256(complete_payload),
        }
        plan = json.loads(
            (ROOT / "packaging/v110-m4-equal-work-plan.json").read_text()
        )
        case_reports = {}
        for case in plan["cases"]:
            ratios = {
                "v103-gpu1_to_v110-gpu1": {
                    "ratio_of_average_seconds": 1.1
                }
            }
            lanes = {"gpmeep_v110_cuda_gpu1": {}}
            if case["gpu2"]:
                lanes["gpmeep_v110_cuda_gpu2"] = {}
                ratios["v103-gpu2_to_v110-gpu2"] = {
                    "ratio_of_average_seconds": 1.1
                }
            case_reports[case["path"]] = {"lanes": lanes, "ratios": ratios}
        self.equal_work_report = {
            "schema": "gpmeep-v110-m4-equal-work-matrix-v1",
            "outcome": "PASS",
            "report_id": "e" * 64,
            "contract": {
                "plan": plan,
                "paths": {"v110_package": str(self.measured_package)},
                "runtime_inputs": {
                    "v103_package": {
                        "sha256": FINAL.FROZEN_V103_PACKAGE_SHA256
                    },
                    "v110_source_identity": {"git_head": self.source_commit},
                    "v110_package_sha256": self.measured_package_sha256,
                    "v110_receipt_id": self.development_receipt_id,
                },
            },
            "performance": {
                "schema": "gpmeep-v110-m4-equal-work-aggregation-v2",
                "outcome": "PASS",
                "task_count": 180,
                "measured_task_count": 120,
                "raw_seconds": [{} for _ in range(180)],
                "cases": case_reports,
            },
            "numerical": {
                "schema": "gpmeep-v110-m4-numerical-replay-v1",
                "outcome": "PASS",
                "comparison_count": 141,
                "comparisons": [{} for _ in range(141)],
            },
        }
        expected_plot = FINAL._expected_speedup_png(self.equal_work_report)
        plot_data_uri = (
            "data:image/png;base64," + base64.b64encode(expected_plot).decode("ascii")
        )
        html_reference = qualification_refs["final_report_html"]
        html_path = self.root / html_reference["bundle_relative_path"]
        html_path.write_text(
            f'<html><body><img data-plot-id="speedup" src="{plot_data_uri}"></body></html>\n',
            encoding="utf-8",
        )
        html_digest = FINAL._sha256(html_path)
        html_reference.update(
            size_bytes=html_path.stat().st_size,
            sha256=html_digest,
            content_id=html_digest,
        )
        plot_width, plot_height = FINAL._decode_png(expected_plot)
        self.qualification = {
            "schema": FINAL.QUALIFICATION_SCHEMA,
            "state": "PASS",
            "source_commit": self.source_commit,
            "candidate_receipt_id": self.candidate_id,
            "package": self.package_identity,
            "installed_provenance": installed,
            "qualification": qualification_refs,
            "report_plots": [
                {
                    "id": "speedup",
                    "mime_type": "image/png",
                    "width_px": plot_width,
                    "height_px": plot_height,
                    "size_bytes": len(expected_plot),
                    "sha256": hashlib.sha256(expected_plot).hexdigest(),
                    "data_uri_sha256": hashlib.sha256(plot_data_uri.encode("ascii")).hexdigest(),
                    "source_data_sha256": FINAL._canonical_sha256(
                        self.equal_work_report["performance"]["cases"]
                    ),
                }
            ],
            "performance": {
                "state": "PASS",
                "report": performance_ref,
                "lanes": list(FINAL.PERFORMANCE_LANES),
                "repetitions": 2,
                "equal_work": True,
                "actual_seconds_retained": True,
                "numerical_oracles_pass": True,
                "no_cpu_fallback": True,
                "no_material_v110_regression": True,
                "structural_optimization_benefit": True,
                "frozen_v103": {
                    "version": "1.0.3",
                    "source_commit": FINAL.FROZEN_V103_COMMIT,
                    "package_sha256": FINAL.FROZEN_V103_PACKAGE_SHA256,
                },
                "v110_package_sha256": self.measured_package_sha256,
            },
        }
        self.qualification_path = self.root / "qualification.json"
        _write_json(self.qualification_path, self.qualification)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _candidate_producer_patches(self):
        stack = mock.patch.object(
            CANDIDATE, "_replay_contract_gate", return_value=self.contract_result
        )
        # Candidate verifier is imported again by the final tool, so final tests
        # patch its narrow replay boundary separately in _create/_verify.
        class Patches:
            def __enter__(inner):
                inner.stack = __import__("contextlib").ExitStack()
                inner.stack.enter_context(stack)
                inner.stack.enter_context(mock.patch.object(CANDIDATE, "_require_isolated_authority"))
                def preflight(**kwargs):
                    manifest = json.loads(pathlib.Path(kwargs["candidate_manifest_path"]).read_text())
                    reference = manifest["evidence"]["m4_source_audit"]
                    return json.loads(
                        pathlib.Path(kwargs["evidence_root"])
                        .joinpath(reference["bundle_relative_path"])
                        .read_text()
                    )
                inner.stack.enter_context(mock.patch.object(CANDIDATE, "_preflight_source_authority", side_effect=preflight))
                inner.stack.enter_context(mock.patch.object(CANDIDATE, "_replay_registry_gate", return_value=self.registry_result))
                inner.stack.enter_context(
                    mock.patch.object(
                        CANDIDATE,
                        "_replay_closed_registry",
                        return_value=(
                            self.closed_registry_authority,
                            self.closed_records,
                        ),
                    )
                )
                inner.stack.enter_context(mock.patch.object(CANDIDATE, "_replay_attestation_gate", return_value=self.attested))
                inner.stack.enter_context(mock.patch.object(CANDIDATE, "_replay_native_evidence", side_effect=lambda role, path, payload, release_source_commit, **kwargs: CANDIDATE._canonical_sha256(payload)))
                return inner

            def __exit__(inner, *args):
                return inner.stack.__exit__(*args)
        return Patches()

    def _final_replay_patches(self):
        candidate = json.loads(self.candidate_receipt.read_text())
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(FINAL, "_require_isolated_release_runtime"))
        stack.enter_context(
            mock.patch.object(
                FINAL,
                "_load_candidate_tool",
                return_value=type(
                    "CandidateTool",
                    (),
                    {"verify": staticmethod(lambda **kwargs: candidate)},
                ),
            )
        )
        stack.enter_context(mock.patch.object(FINAL, "_verify_installed_prefix"))
        stack.enter_context(
            mock.patch.object(
                FINAL,
                "_replay_installed_self_check",
                side_effect=lambda path, **_kwargs: json.loads(path.read_text()),
            )
        )
        stack.enter_context(mock.patch.object(FINAL, "_verify_qualification_payload"))
        stack.enter_context(mock.patch.object(FINAL, "_verify_report_semantics"))
        stack.enter_context(
            mock.patch.object(
                FINAL, "_replay_equal_work", return_value=self.equal_work_report
            )
        )
        stack.enter_context(
            mock.patch.object(
                FINAL,
                "_replay_equal_work_registry",
                return_value={
                    "schema_version": 1,
                    "state": "PASS",
                    "observed_count": 13,
                    "registered_count": 13,
                    "missing_count": 0,
                    "blocking_count": 0,
                    "closed_count": 13,
                    "open_count": 0,
                    "missing_workloads": [],
                    "blocking_workloads": [],
                    "open_workloads": [],
                },
            )
        )
        closure = json.loads(
            (self.root / "closed_inversion_registry.json").read_text()
        )
        stack.enter_context(
            mock.patch.object(
                FINAL,
                "_replay_closed_inversion_registry",
                return_value=closure,
            )
        )
        return stack

    def _create(self) -> dict[str, object]:
        with self._final_replay_patches():
            return FINAL.create(
                package_path=self.package,
                candidate_receipt_path=self.candidate_receipt,
                candidate_tool_path=ROOT / "scripts/v110-release-gate-receipt.py",
                contract_path=CONTRACT,
                inversion_schema_path=ROOT
                / "packaging/v110-performance-inversion-record.schema.json",
                inventory_path=ROOT
                / "packaging/v110-performance-inversion-inventory.json",
                qualification_manifest_path=self.qualification_path,
                evidence_root=self.root,
                final_audit_authority_path=self.final_audit_authority,
                expected_final_audit_authority_sha256=self.final_audit_authority_sha256,
                **self.candidate_inputs,
            )

    def _verify(self, receipt_path: pathlib.Path) -> dict[str, object]:
        with self._final_replay_patches():
            return FINAL.verify(
                receipt_path=receipt_path,
                package_path=self.package,
                candidate_receipt_path=self.candidate_receipt,
                candidate_tool_path=ROOT / "scripts/v110-release-gate-receipt.py",
                contract_path=CONTRACT,
                inversion_schema_path=ROOT
                / "packaging/v110-performance-inversion-record.schema.json",
                inventory_path=ROOT
                / "packaging/v110-performance-inversion-inventory.json",
                qualification_manifest_path=self.qualification_path,
                evidence_root=self.root,
                final_audit_authority_path=self.final_audit_authority,
                expected_final_audit_authority_sha256=self.final_audit_authority_sha256,
                **self.candidate_inputs,
            )

    def test_zstd_decoder_has_fixed_compressed_and_decompressed_bounds(self):
        self.assertEqual(FINAL.MAX_CONDA_COMPRESSED_MEMBER_BYTES, 128 * 1024**2)
        self.assertEqual(FINAL.MAX_CONDA_DECOMPRESSED_MEMBER_BYTES, 512 * 1024**2)

        exact = b"x" * 4096
        with mock.patch.object(
            FINAL, "MAX_CONDA_DECOMPRESSED_MEMBER_BYTES", len(exact)
        ):
            self.assertEqual(FINAL._zstd_decompress(_zstd(exact), "exact"), exact)

        bomb = b"y" * 8192
        with mock.patch.object(FINAL, "MAX_CONDA_DECOMPRESSED_MEMBER_BYTES", 4096):
            with self.assertRaisesRegex(
                FINAL.FinalReleaseError, "zstd output exceeds the 512 MiB limit"
            ):
                FINAL._zstd_decompress(_zstd(bomb), "bomb")

        compressed = _zstd(b"bounded")
        with mock.patch.object(
            FINAL, "MAX_CONDA_COMPRESSED_MEMBER_BYTES", len(compressed)
        ):
            self.assertEqual(
                FINAL._zstd_decompress(compressed, "compressed exact"), b"bounded"
            )
        with mock.patch.object(
            FINAL, "MAX_CONDA_COMPRESSED_MEMBER_BYTES", len(compressed) - 1
        ):
            with self.assertRaisesRegex(
                FINAL.FinalReleaseError,
                "compressed member exceeds the 128 MiB limit",
            ):
                FINAL._zstd_decompress(compressed, "outer")

    def test_zstd_decoder_distinguishes_timeout_and_decoder_failure(self):
        with mock.patch.object(
            FINAL.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["/usr/bin/zstd"], 120),
        ):
            with self.assertRaisesRegex(FINAL.FinalReleaseError, "decoding timed out"):
                FINAL._zstd_decompress(b"input", "timeout")
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "zstd decoder failed"):
            FINAL._zstd_decompress(b"not-zstd", "invalid")

    def test_final_receipt_loader_restores_module_cache_on_success_and_failure(self):
        success = self.root / "loader-success.py"
        failure = self.root / "loader-failure.py"
        success.write_text("value = 17\n", encoding="utf-8")
        failure.write_text("raise RuntimeError('loader failure')\n", encoding="utf-8")

        prior_name = "gpmeep_final_receipt_loader_prior"
        prior = types.ModuleType(prior_name)
        sys.modules[prior_name] = prior
        try:
            loaded = FINAL._load_module(prior_name, success)
            self.assertEqual(loaded.value, 17)
            self.assertIs(sys.modules[prior_name], prior)
            with self.assertRaisesRegex(RuntimeError, "loader failure"):
                FINAL._load_module(prior_name, failure)
            self.assertIs(sys.modules[prior_name], prior)
        finally:
            sys.modules.pop(prior_name, None)

        fresh_name = "gpmeep_final_receipt_loader_fresh"
        sys.modules.pop(fresh_name, None)
        loaded = FINAL._load_module(fresh_name, success)
        self.assertEqual(loaded.value, 17)
        self.assertNotIn(fresh_name, sys.modules)
        with self.assertRaisesRegex(RuntimeError, "loader failure"):
            FINAL._load_module(fresh_name, failure)
        self.assertNotIn(fresh_name, sys.modules)

    def test_final_receipt_is_external_mode_600_and_exactly_replayable(self):
        receipt = self._create()
        self.assertEqual(receipt["state"], "RELEASE_AUTHORIZED")
        self.assertEqual(receipt["package"]["sha256"], self.package_sha256)
        self.assertEqual(
            receipt["authorization"],
            {
                "release_publication": True,
                "git_tag": True,
                "package_mutation": False,
            },
        )
        path = self.root / "final-receipt.json"
        FINAL._write_new(path, receipt)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(self._verify(path), receipt)
        with self.assertRaises(FINAL.FinalReleaseError):
            FINAL._write_new(path, receipt)

    def test_changed_package_bytes_invalidate_final_authority(self):
        receipt = self._create()
        path = self.root / "final-receipt.json"
        _write_json(path, receipt)
        self.package.write_bytes(b"different package bytes\n")
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "not a valid conda-v2 archive"
        ):
            self._verify(path)

    def test_arbitrary_bytes_named_conda_are_rejected(self):
        self.package.write_bytes(b"not a conda archive\n")
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "not a valid conda-v2 archive"
        ):
            self._create()

    def test_installed_verifier_must_match_archive_owned_bytes(self):
        _identity, members, _payload = FINAL._inspect_conda_v2(
            self.package, self.candidate_receipt, self.candidate_id
        )
        entry = self.qualification["installed_provenance"][0]
        prefix = pathlib.Path(entry["prefix"])
        (prefix / FINAL.INSTALLED_VERIFIER_PATH).write_bytes(b"tampered verifier\n")
        report = json.loads(
            (self.root / entry["report"]["bundle_relative_path"]).read_text()
        )
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "verifier bytes differ"
        ):
            FINAL._verify_installed_prefix(
                prefix=prefix,
                retained_report=report,
                archive_members=members,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
            )

    def test_installed_self_check_must_match_archive_owned_bytes(self):
        _identity, members, _payload = FINAL._inspect_conda_v2(
            self.package, self.candidate_receipt, self.candidate_id
        )
        entry = self.qualification["installed_provenance"][0]
        prefix = pathlib.Path(entry["prefix"])
        (prefix / FINAL.INSTALLED_SELF_CHECK_PATH).write_bytes(
            b"tampered self-check\n"
        )
        report = json.loads(
            (self.root / entry["report"]["bundle_relative_path"]).read_text()
        )
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "self-check bytes differ"
        ):
            FINAL._verify_installed_prefix(
                prefix=prefix,
                retained_report=report,
                archive_members=members,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
            )

    def test_development_measurement_may_promote_only_byte_identical_runtime(self):
        _identity, _members, candidate_payload = FINAL._inspect_conda_v2(
            self.package, self.candidate_receipt, self.candidate_id
        )
        result = FINAL._runtime_payload_equivalence(
            self.measured_package, self.package, candidate_payload
        )
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(
            result["allowed_changed_paths"],
            [FINAL.CANDIDATE_RECEIPT_PATH],
        )

        changed = self.root / "gpmeep-1.1.0-changed-engine.conda"
        development_bytes = FINAL._member_bytes(
            FINAL._conda_payload_files(
                self.measured_package, "development fixture"
            )[1][FINAL.CANDIDATE_RECEIPT_PATH],
            "development fixture receipt",
        )
        _write_conda_v2(
            changed,
            development_bytes,
            build="cuda124_mpi_openmpi_py311_0_development",
            payload_overrides={"lib/libmeep.so.38": b"changed-engine\n"},
        )
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "payload transition differs"
        ):
            FINAL._runtime_payload_equivalence(
                changed, self.package, candidate_payload
            )

        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "cannot be its own M4 measurement"
        ):
            FINAL._runtime_payload_equivalence(
                self.package, self.package, candidate_payload
            )

    def test_development_measurement_rejects_install_time_prefix_transform(self):
        _identity, _members, candidate_payload = FINAL._inspect_conda_v2(
            self.package, self.candidate_receipt, self.candidate_id
        )
        transformed = self.root / "gpmeep-1.1.0-transformed-development.conda"
        development_bytes = FINAL._member_bytes(
            FINAL._conda_payload_files(self.measured_package, "development fixture")[1][FINAL.CANDIDATE_RECEIPT_PATH],
            "development fixture receipt",
        )
        _write_conda_v2(
            transformed,
            development_bytes,
            build="cuda124_mpi_openmpi_py311_0_development",
            path_metadata_overrides={
                "lib/libmeep.so.38": {
                    "prefix_placeholder": "/opt/anaconda1anaconda2anaconda3",
                    "file_mode": "binary",
                }
            },
        )
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "install-time payload transform"):
            FINAL._runtime_payload_equivalence(transformed, self.package, candidate_payload)

    def test_conda_builder_softlink_byte_metadata_is_verified(self):
        package = self.root / "gpmeep-1.1.0-softlink-builder-shape.conda"
        _write_conda_v2(
            package,
            self.candidate_receipt.read_bytes(),
            softlinks={"lib/libmeep.so": "libmeep.so.38"},
        )
        _index, payload, semantics = FINAL._conda_payload_files(
            package, "softlink builder-shape fixture"
        )
        target = payload["lib/libmeep.so.38"]
        self.assertEqual(payload["lib/libmeep.so"].member_type, "symlink")
        self.assertEqual(
            semantics["lib/libmeep.so"],
            {
                "path_type": "softlink",
                "link_target": "lib/libmeep.so.38",
                "target_sha256": target.sha256,
                "target_size_in_bytes": target.size_bytes,
                "no_link": False,
            },
        )

    def test_conda_softlink_byte_metadata_tamper_is_rejected(self):
        target = b"\x7fELFlibmeep-core\n"
        tamper_cases = {
            "sha256": {
                "sha256": "0" * 64,
                "size_in_bytes": len(target),
            },
            "size": {
                "sha256": hashlib.sha256(target).hexdigest(),
                "size_in_bytes": len(target) + 1,
            },
            "boolean-size": {
                "sha256": hashlib.sha256(target).hexdigest(),
                "size_in_bytes": True,
            },
        }
        for case, metadata in tamper_cases.items():
            with self.subTest(case=case):
                package = self.root / f"gpmeep-1.1.0-softlink-{case}.conda"
                _write_conda_v2(
                    package,
                    self.candidate_receipt.read_bytes(),
                    softlinks={"lib/libmeep.so": "libmeep.so.38"},
                    path_metadata_overrides={"lib/libmeep.so": metadata},
                )
                with self.assertRaisesRegex(
                    FINAL.FinalReleaseError, "softlink target hash/size differs"
                ):
                    FINAL._conda_payload_files(package, f"softlink {case} fixture")

    def test_conda_softlink_unknown_metadata_or_type_tamper_is_rejected(self):
        cases = {
            "unknown-key": ({"unexpected": "value"}, "ownership entry schema differs"),
            "wrong-type": ({"path_type": "hardlink"}, "hardlink metadata differs"),
        }
        for case, (metadata, expected) in cases.items():
            with self.subTest(case=case):
                package = self.root / f"gpmeep-1.1.0-softlink-{case}.conda"
                _write_conda_v2(
                    package,
                    self.candidate_receipt.read_bytes(),
                    softlinks={"lib/libmeep.so": "libmeep.so.38"},
                    path_metadata_overrides={"lib/libmeep.so": metadata},
                )
                with self.assertRaisesRegex(FINAL.FinalReleaseError, expected):
                    FINAL._conda_payload_files(package, f"softlink {case} fixture")

    def test_conda_softlink_partial_byte_metadata_is_rejected(self):
        for missing in ("sha256", "size_in_bytes"):
            with self.subTest(missing=missing):
                package = self.root / f"gpmeep-1.1.0-softlink-missing-{missing}.conda"
                _write_conda_v2(
                    package,
                    self.candidate_receipt.read_bytes(),
                    softlinks={"lib/libmeep.so": "libmeep.so.38"},
                    path_metadata_removals={"lib/libmeep.so": {missing}},
                )
                with self.assertRaisesRegex(
                    FINAL.FinalReleaseError,
                    "softlink byte metadata schema differs",
                ):
                    FINAL._conda_payload_files(
                        package, f"softlink missing-{missing} fixture"
                    )

    def test_real_development_package_softlinks_when_requested(self):
        package_text = os.environ.get("GPMEEP_V110_REAL_PACKAGE")
        if package_text is None:
            self.skipTest("GPMEEP_V110_REAL_PACKAGE was not supplied")
        package = pathlib.Path(package_text)
        index, payload, semantics = FINAL._conda_payload_files(
            package, "real development package"
        )
        self.assertEqual(
            index["build"], "cuda124_mpi_openmpi_py311_0_development"
        )
        softlinks = {
            name: record
            for name, record in semantics.items()
            if record["path_type"] == "softlink"
        }
        self.assertTrue(softlinks)
        for name, record in softlinks.items():
            self.assertEqual(payload[name].member_type, "symlink")
            self.assertRegex(record["target_sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(record["target_size_in_bytes"], 0)

    def test_tar_member_parser_preserves_safe_links_and_modes(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:") as archive:
            regular = tarfile.TarInfo("lib/libmeep.so.38")
            regular.mode = 0o755
            regular.size = 4
            archive.addfile(regular, io.BytesIO(b"ELF!"))
            link = tarfile.TarInfo("lib/libmeep.so")
            link.type = tarfile.SYMTYPE
            link.mode = 0o777
            link.linkname = "libmeep.so.38"
            archive.addfile(link)
        members = FINAL._tar_files(stream.getvalue(), "safe fixture")
        self.assertEqual(members["lib/libmeep.so.38"].mode, 0o755)
        self.assertEqual(members["lib/libmeep.so"].member_type, "symlink")
        self.assertEqual(
            members["lib/libmeep.so"].link_target, "lib/libmeep.so.38"
        )

    def test_tar_member_parser_rejects_unsafe_or_dangling_links(self):
        for target, expected in (
            ("/etc/passwd", "absolute"),
            ("../escape", "contains '..'"),
            ("./libmeep.so.38", "not canonical"),
            ("absent.so", "dangling"),
            ("libmeep.so", "cycle"),
        ):
            with self.subTest(target=target):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w:") as archive:
                    link = tarfile.TarInfo("lib/libmeep.so")
                    link.type = tarfile.SYMTYPE
                    link.linkname = target
                    archive.addfile(link)
                with self.assertRaisesRegex(FINAL.FinalReleaseError, expected):
                    FINAL._tar_files(stream.getvalue(), "unsafe fixture")

    def test_frozen_v103_and_all_six_lanes_are_mandatory(self):
        self.qualification["performance"]["frozen_v103"][
            "package_sha256"
        ] = "0" * 64
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "v1.0.3 identity"):
            self._create()
        self.qualification["performance"]["frozen_v103"][
            "package_sha256"
        ] = FINAL.FROZEN_V103_PACKAGE_SHA256
        self.qualification["performance"]["lanes"].pop()
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "lane inventory"):
            self._create()

    def test_equal_work_replay_requires_exact_raw_and_numerical_census(self):
        self.equal_work_report["performance"]["raw_seconds"].pop()
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "performance replay census"):
            self._create()

    def test_equal_work_recomputes_material_regression(self):
        first = next(iter(self.equal_work_report["performance"]["cases"].values()))
        first["ratios"]["v103-gpu1_to_v110-gpu1"][
            "ratio_of_average_seconds"
        ] = 0.89
        expected_plot = FINAL._expected_speedup_png(self.equal_work_report)
        data_uri = "data:image/png;base64," + base64.b64encode(expected_plot).decode("ascii")
        reference = self.qualification["qualification"]["final_report_html"]
        html_path = self.root / reference["bundle_relative_path"]
        html_path.write_text(
            f'<html><body><img data-plot-id="speedup" src="{data_uri}"></body></html>\n',
            encoding="utf-8",
        )
        digest = FINAL._sha256(html_path)
        reference.update(size_bytes=html_path.stat().st_size, sha256=digest, content_id=digest)
        width, height = FINAL._decode_png(expected_plot)
        self.qualification["report_plots"] = [{
            "id": "speedup",
            "mime_type": "image/png",
            "width_px": width,
            "height_px": height,
            "size_bytes": len(expected_plot),
            "sha256": hashlib.sha256(expected_plot).hexdigest(),
            "data_uri_sha256": hashlib.sha256(data_uri.encode("ascii")).hexdigest(),
            "source_data_sha256": FINAL._canonical_sha256(
                self.equal_work_report["performance"]["cases"]
            ),
        }]
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "material v1.1.0 regression"):
            self._create()

    def test_native_environments_require_derived_ids_and_concrete_ubuntu_versions(self):
        self.qualification["installed_provenance"][0]["environment_id"] = "claimed-id"
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "was not derived"):
            self._create()

    def test_native_environment_rejects_noncanonical_platform(self):
        self.qualification["installed_provenance"][0]["environment_id"] = (
            self.platform_bindings["ubuntu-22.04"]["environment_id"]
        )
        self.qualification["installed_provenance"][1]["platform"] = (
            "one-newer-ubuntu"
        )
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "platform is invalid"):
            self._create()

    def test_native_glibc_and_final_audit_producers_replay_without_critical_mock(self):
        _metadata, _members, candidate_payload = FINAL._inspect_conda_v2(
            self.package, self.candidate_receipt, self.candidate_id
        )

        native_reference = self.native_witness_refs["ubuntu-22.04"]
        native_producer = {
            key: value for key, value in native_reference.items() if key != "role"
        }
        native: dict[str, object] = {
            "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["ubuntu_22_04_native"],
            "state": "PASS",
            "role": "ubuntu_22_04_native",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "producer_terminal": native_producer,
            "native_install": True,
            "platform": "ubuntu-22.04",
            **self.platform_bindings["ubuntu-22.04"],
        }
        native["report_id"] = FINAL._canonical_sha256(native)
        FINAL._verify_qualification_payload(
            "ubuntu_22_04_native",
            native,
            source_commit=self.source_commit,
            package_sha256=self.package_sha256,
            candidate_receipt_id=self.candidate_id,
            evidence_root=self.root,
            candidate_payload=candidate_payload,
        )
        native["namespace_id"] = "9" * 64
        native.pop("report_id")
        native["report_id"] = FINAL._canonical_sha256(native)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "replayed native witness"):
            FINAL._verify_qualification_payload(
                "ubuntu_22_04_native",
                native,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
                evidence_root=self.root,
                candidate_payload=candidate_payload,
            )

        glibc_payload = FINAL._json_bytes(
            FINAL._member_bytes(
                candidate_payload[FINAL.GLIBC_AUDIT_PATH], "test GLIBC audit"
            ),
            "test GLIBC audit",
        )
        glibc_reference = _json_ref(self.root, "glibc_producer", glibc_payload)
        glibc: dict[str, object] = {
            "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["glibc_floor"],
            "state": "PASS",
            "role": "glibc_floor",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "producer_terminal": {
                key: value for key, value in glibc_reference.items() if key != "role"
            },
            "overall_maximum_required": glibc_payload["overall_maximum_required"],
            "audited_elf_sha256": sorted(
                item["sha256"] for item in glibc_payload["elf_files"]
            ),
            "installed_report_ids": sorted(self.installed_report_ids),
        }
        glibc["report_id"] = FINAL._canonical_sha256(glibc)
        FINAL._verify_qualification_payload(
            "glibc_floor",
            glibc,
            source_commit=self.source_commit,
            package_sha256=self.package_sha256,
            candidate_receipt_id=self.candidate_id,
            evidence_root=self.root,
            candidate_payload=candidate_payload,
        )
        glibc["audited_elf_sha256"][0] = "8" * 64
        glibc.pop("report_id")
        glibc["report_id"] = FINAL._canonical_sha256(glibc)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "GLIBC producer inventory"):
            FINAL._verify_qualification_payload(
                "glibc_floor",
                glibc,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
                evidence_root=self.root,
                candidate_payload=candidate_payload,
            )

        audit: dict[str, object] = {
            "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["final_adversarial_audit"],
            "state": "PASS",
            "role": "final_adversarial_audit",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "producer_terminal": {
                key: value
                for key, value in self.final_audit_producer_ref.items()
                if key != "role"
            },
            "findings": self.final_findings,
            "open_critical_high_medium": 0,
            "human_release_decision": self.final_decision,
        }
        audit["report_id"] = FINAL._canonical_sha256(audit)
        FINAL._verify_qualification_payload(
            "final_adversarial_audit",
            audit,
            source_commit=self.source_commit,
            package_sha256=self.package_sha256,
            candidate_receipt_id=self.candidate_id,
            evidence_root=self.root,
            candidate_payload=candidate_payload,
        )
        audit["findings"][0]["evidence_sha256"] = "7" * 64
        audit.pop("report_id")
        audit["report_id"] = FINAL._canonical_sha256(audit)
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "open finding census|independent producer"
        ):
            FINAL._verify_qualification_payload(
                "final_adversarial_audit",
                audit,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
                evidence_root=self.root,
                candidate_payload=candidate_payload,
            )

    def test_recomputed_receipt_id_cannot_expand_or_change_authority(self):
        receipt = self._create()
        receipt["authorization"]["package_mutation"] = True
        unsigned = dict(receipt)
        unsigned.pop("receipt_id")
        receipt["receipt_id"] = FINAL._canonical_sha256(unsigned)
        path = self.root / "final-receipt.json"
        _write_json(path, receipt)
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "not exactly re-derived"
        ):
            self._verify(path)

    def test_final_html_report_must_embed_images(self):
        reference = self.qualification["qualification"]["final_report_html"]
        html = self.root / reference["bundle_relative_path"]
        html.write_text("<html><body>no embedded plot</body></html>\n")
        digest = FINAL._sha256(html)
        reference.update(
            size_bytes=html.stat().st_size, sha256=digest, content_id=digest
        )
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "embed its images"):
            self._create()

    def test_correct_schema_final_pass_cannot_bypass_raw_producer_replay(self):
        raw_directory = self.root / "forged-m4"
        raw_directory.mkdir()
        raw = raw_directory / "M4_COMPLETE"
        raw_payload = {"schema": FINAL.M4_COMPLETE_SCHEMA, "outcome": "PASS"}
        _write_json(raw, raw_payload)
        producer = {
            "schema": FINAL.M4_COMPLETE_SCHEMA,
            "content_id": FINAL._canonical_sha256(raw_payload),
            "sha256": FINAL._sha256(raw),
            "size_bytes": raw.stat().st_size,
            "bundle_relative_path": str(raw.relative_to(self.root)),
        }
        payload = {
            "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["installed_correctness"],
            "state": "PASS",
            "role": "installed_correctness",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "producer_terminal": producer,
            "numerical_oracles_pass": True,
            "comparison_count": 141,
            "m4_report_id": "0" * 64,
            "development_package_receipt_id": "d" * 64,
        }
        payload["report_id"] = FINAL._canonical_sha256(payload)
        with mock.patch.object(
            FINAL,
            "_replay_equal_work",
            return_value={
                "report_id": "0" * 64,
                "numerical": {"outcome": "FAIL", "comparison_count": 141},
            },
        ):
            with self.assertRaisesRegex(FINAL.FinalReleaseError, "correctness census"):
                FINAL._verify_qualification_payload(
                    "installed_correctness",
                    payload,
                    source_commit=self.source_commit,
                    package_sha256=self.package_sha256,
                    candidate_receipt_id=self.candidate_id,
                    evidence_root=self.root,
                    candidate_payload={},
                )

    def test_final_qualification_rejects_pending_m4_terminal(self):
        raw_directory = self.root / "pending-m4"
        raw_directory.mkdir()
        raw = raw_directory / "PENDING_M4_COMPLETE"
        raw_payload = {"schema": FINAL.M4_COMPLETE_SCHEMA, "outcome": "PASS"}
        _write_json(raw, raw_payload)
        payload = {
            "schema": FINAL.QUALIFICATION_ROLE_SCHEMAS["installed_correctness"],
            "state": "PASS",
            "role": "installed_correctness",
            "source_commit": self.source_commit,
            "package_sha256": self.package_sha256,
            "candidate_receipt_id": self.candidate_id,
            "producer_terminal": {
                "schema": FINAL.M4_COMPLETE_SCHEMA,
                "content_id": FINAL._canonical_sha256(raw_payload),
                "sha256": FINAL._sha256(raw),
                "size_bytes": raw.stat().st_size,
                "bundle_relative_path": str(raw.relative_to(self.root)),
            },
            "numerical_oracles_pass": True,
            "comparison_count": 141,
            "m4_report_id": "0" * 64,
            "development_package_receipt_id": self.development_receipt_id,
        }
        payload["report_id"] = FINAL._canonical_sha256(payload)
        with self.assertRaisesRegex(
            FINAL.FinalReleaseError, "official M4 terminal"
        ):
            FINAL._verify_qualification_payload(
                "installed_correctness",
                payload,
                source_commit=self.source_commit,
                package_sha256=self.package_sha256,
                candidate_receipt_id=self.candidate_id,
                evidence_root=self.root,
                candidate_payload={},
            )

    def test_embedded_plot_manifest_binds_exact_decoded_bytes(self):
        self.qualification["report_plots"][0]["sha256"] = "0" * 64
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "plot bytes"):
            self._create()

    def test_embedded_plot_rejects_one_byte_fake_png(self):
        reference = self.qualification["qualification"]["final_report_html"]
        html = self.root / reference["bundle_relative_path"]
        html.write_text(
            '<html><body><img data-plot-id="speedup" '
            'src="data:image/png;base64,AA=="></body></html>\n',
            encoding="utf-8",
        )
        digest = FINAL._sha256(html)
        reference.update(size_bytes=html.stat().st_size, sha256=digest, content_id=digest)
        self.qualification["report_plots"][0].update(
            size_bytes=1,
            sha256=hashlib.sha256(b"\x00").hexdigest(),
            data_uri_sha256=hashlib.sha256(b"data:image/png;base64,AA==").hexdigest(),
        )
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "PNG magic"):
            self._create()

    def test_embedded_plot_rejects_truncated_rgba_scanline(self):
        def chunk(kind: bytes, data: bytes) -> bytes:
            crc = FINAL.binascii.crc32(kind + data) & 0xFFFFFFFF
            return len(data).to_bytes(4, "big") + kind + data + crc.to_bytes(4, "big")

        ihdr = (
            (1).to_bytes(4, "big")
            + (1).to_bytes(4, "big")
            + bytes((8, 6, 0, 0, 0))
        )
        malformed = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", FINAL.zlib.compress(b"x"))
            + chunk(b"IEND", b"")
        )
        reference = self.qualification["qualification"]["final_report_html"]
        html = self.root / reference["bundle_relative_path"]
        data_uri = "data:image/png;base64," + base64.b64encode(malformed).decode("ascii")
        html.write_text(
            f'<html><body><img data-plot-id="speedup" src="{data_uri}"></body></html>\n',
            encoding="utf-8",
        )
        digest = FINAL._sha256(html)
        reference.update(size_bytes=html.stat().st_size, sha256=digest, content_id=digest)
        self.qualification["report_plots"][0].update(
            size_bytes=len(malformed),
            sha256=hashlib.sha256(malformed).hexdigest(),
            data_uri_sha256=hashlib.sha256(data_uri.encode("ascii")).hexdigest(),
        )
        _write_json(self.qualification_path, self.qualification)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "pixel stream length"):
            self._create()

    def test_png_rejects_late_or_oversized_palette(self):
        def chunk(kind: bytes, data: bytes) -> bytes:
            return FINAL._png_chunk(kind, data)

        truecolor_ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes((8, 2, 0, 0, 0))
        late_palette = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", truecolor_ihdr)
            + chunk(b"IDAT", FINAL.zlib.compress(b"\x00\x00\x00\x00"))
            + chunk(b"PLTE", b"\x00\x00\x00")
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "palette"):
            FINAL._decode_png(late_palette)

        indexed_ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes((1, 3, 0, 0, 0))
        oversized_palette = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", indexed_ihdr)
            + chunk(b"PLTE", b"\x00\x00\x00\x7f\x7f\x7f\xff\xff\xff")
            + chunk(b"IDAT", FINAL.zlib.compress(b"\x00\x00"))
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "bit depth"):
            FINAL._decode_png(oversized_palette)

    def test_report_semantics_require_exact_replayed_seconds_and_tables(self):
        comparisons = [
            {
                "sample_kind": "measured" if index < 94 else "warmup",
                "error": 0.0,
            }
            for index in range(141)
        ]
        report = {
            "performance": {"raw_seconds": [{"seconds": 1.25}], "cases": {"case": {"ratio": 1.1}}},
            "numerical": {"comparison_count": 141, "comparisons": comparisons},
        }
        report_id = FINAL._canonical_sha256(report)
        data = {
            "schema": "gpmeep-v110-final-report-data-v1",
            "m4_report_id": report_id,
            "actual_seconds": report["performance"]["raw_seconds"],
            "cases": report["performance"]["cases"],
            "numerical_comparison_count": 141,
            "measured_cycle_numerical_comparison_count": 94,
            "numerical_comparisons": report["numerical"]["comparisons"],
        }
        row_sets = {
            "actual-seconds": [
                (str(index), value)
                for index, value in enumerate(report["performance"]["raw_seconds"])
            ],
            "means-ratios": [
                (key, report["performance"]["cases"][key])
                for key in sorted(report["performance"]["cases"])
            ],
            "numerical-results": [
                (str(index), value)
                for index, value in enumerate(report["numerical"]["comparisons"])
            ],
        }
        headings = {
            "actual-seconds": "Actual seconds",
            "means-ratios": "Means and ratios",
            "numerical-results": "Numerical results",
        }
        html_tables = []
        markdown_tables = []
        for table_id, rows in row_sets.items():
            html_rows = "".join(
                "<tr><td>" + key + "</td><td><code>"
                + FINAL.html_module.escape(
                    json.dumps(value, sort_keys=True, separators=(",", ":")), quote=True
                )
                + "</code></td></tr>"
                for key, value in rows
            )
            html_tables.append(
                f'<table id="{table_id}"><thead><tr><th>Key</th><th>Canonical value</th>'
                f"</tr></thead><tbody>{html_rows}</tbody></table>"
            )
            markdown_rows = "\n".join(
                f"| {key} | {json.dumps(value, sort_keys=True, separators=(',', ':'))} |"
                for key, value in rows
            )
            markdown_tables.append(
                f"## {headings[table_id]}\n\n| Key | Canonical value |\n|---|---|\n{markdown_rows}"
            )
        comparison_html, comparison_markdown = (
            FINAL._actual_time_comparison_tables(report)
        )
        html = (
            "<html><body>" + comparison_html + "".join(html_tables)
            + '<script id="gpmeep-v110-report-data" type="application/json">'
            + json.dumps(data, sort_keys=True, separators=(",", ":"))
            + "</script></body></html>"
        )
        markdown = (
            f"M4 report ID: `{report_id}`\n"
            + comparison_markdown
            + "\n"
            + "\n".join(markdown_tables)
        )
        FINAL._verify_report_semantics(html, markdown, report)
        visibly_wrong = html.replace("1.25", "999999", 1)
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "visible table"):
            FINAL._verify_report_semantics(visibly_wrong, markdown, report)
        tampered = html.replace('"seconds":1.25', '"seconds":1.0')
        with self.assertRaisesRegex(FINAL.FinalReleaseError, "differ from M4 replay"):
            FINAL._verify_report_semantics(tampered, markdown, report)

    def test_publication_controller_rejects_symlinked_package_before_replay(self):
        package_link = self.root / "package-link.conda"
        package_link.symlink_to(self.package)
        completed = subprocess.run(
            [
                "/bin/bash",
                "-p",
                str(ROOT / "scripts/publish-v110-release.sh"),
                "--repo",
                str(ROOT),
                "--package",
                str(package_link),
                "--final-receipt",
                str(self.qualification_path),
                "--candidate-receipt",
                str(self.candidate_receipt),
                "--qualification-manifest",
                str(self.qualification_path),
                "--evidence-root",
                str(self.root),
                "--candidate-contract-gate",
                str(self.candidate_inputs["candidate_contract_gate_path"]),
                "--candidate-registry-gate",
                str(self.candidate_inputs["candidate_registry_gate_path"]),
                "--candidate-attestation",
                str(self.candidate_inputs["candidate_attestation_path"]),
                "--m4-candidate-manifest",
                str(self.candidate_inputs["candidate_manifest_path"]),
                "--candidate-evidence-root",
                str(self.candidate_inputs["candidate_evidence_root"]),
                "--source-transition-policy",
                str(self.source_policy),
                "--source-transition-policy-sha256",
                self.source_policy_sha256,
                "--source-transition-verifier",
                str(self.source_verifier),
                "--source-transition-verifier-sha256",
                self.source_verifier_sha256,
                "--final-audit-authority",
                str(self.final_audit_authority),
                "--final-audit-authority-sha256",
                self.final_audit_authority_sha256,
                "--asset-directory",
                str(self.root),
                "--repository",
                "iverydj/gpmeep",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unsafe release input", completed.stderr)

    def test_publication_controller_is_draft_first_nonclobbering_and_resumable(self):
        script = (ROOT / "scripts/publish-v110-release.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--resume", script)
        self.assertIn('gh release create "$TAG" --repo "$REPOSITORY" --draft --verify-tag', script)
        self.assertNotIn("--clobber", script)
        self.assertIn("existing draft asset differs and will not be overwritten", script)
        self.assertIn("--verify-asset-directory", script)
        self.assertGreaterEqual(script.count('--expected-manifest-id "$MANIFEST_ID"'), 2)
        self.assertLess(
            script.index('"/repos/$REPOSITORY/immutable-releases"'),
            script.index('/usr/bin/git -C "$REPO" tag -a'),
        )
        self.assertGreaterEqual(script.count('gh release download "$TAG"'), 3)
        publish_offset = script.index('gh release edit "$TAG"')
        first_download_offset = script.index('gh release download "$TAG"')
        self.assertLess(first_download_offset, publish_offset)
        self.assertIn("verify_remote_tag", script[publish_offset:])
        self.assertIn("candidate-receipt-id: $CANDIDATE_RECEIPT_ID", script)
        self.assertIn("final-receipt-id: $FINAL_RECEIPT_ID", script)
        self.assertIn("asset-manifest-id: $MANIFEST_ID", script)
        self.assertIn("local tag identity message differs", script)
        self.assertIn("'.isImmutable'", script)

    def test_measurement_accepts_computed_blocker_but_final_closure_rejects_it(self):
        result = {
            "schema_version": 1,
            "state": "OPEN",
            "observed_count": 13,
            "registered_count": 13,
            "missing_count": 0,
            "blocking_count": 1,
            "closed_count": 12,
            "open_count": 1,
            "missing_workloads": [],
            "blocking_workloads": ["python/examples/waveguide_crossing.py"],
            "open_workloads": ["python/examples/waveguide_crossing.py"],
        }
        validator = type(
            "RegistryValidator",
            (),
            {"validate_registry": staticmethod(lambda **_kwargs: result)},
        )
        with mock.patch.object(FINAL, "_load_module", return_value=validator):
            self.assertEqual(
                FINAL._replay_equal_work_registry(self.root)["state"], "OPEN"
            )
        closure = {
            "schema": "gpmeep-v110-closed-inversion-registry-v1",
            "state": "PASS",
            "registry_result": result,
        }
        producer = type(
            "ClosureProducer",
            (),
            {"verify": staticmethod(lambda *_args: closure)},
        )
        with mock.patch.object(FINAL, "_load_module", return_value=producer):
            with self.assertRaisesRegex(
                FINAL.FinalReleaseError,
                "did not prove closure",
            ):
                FINAL._replay_closed_inversion_registry(
                    self.root / "closed-inversion-registry.json", self.root
                )

    def test_final_receipt_cli_rejects_unreplayed_candidate_fixture(self):
        output = self.root / "cli-final-receipt.json"
        common = [
            "--package",
            str(self.package),
            "--candidate-receipt",
            str(self.candidate_receipt),
            "--candidate-tool",
            str(ROOT / "scripts/v110-release-gate-receipt.py"),
            "--contract",
            str(CONTRACT),
            "--inversion-schema",
            str(ROOT / "packaging/v110-performance-inversion-record.schema.json"),
            "--inventory",
            str(ROOT / "packaging/v110-performance-inversion-inventory.json"),
            "--qualification-manifest",
            str(self.qualification_path),
            "--evidence-root",
            str(self.root),
            "--candidate-contract-gate",
            str(self.candidate_inputs["candidate_contract_gate_path"]),
            "--candidate-registry-gate",
            str(self.candidate_inputs["candidate_registry_gate_path"]),
            "--candidate-attestation",
            str(self.candidate_inputs["candidate_attestation_path"]),
            "--m4-candidate-manifest",
            str(self.candidate_inputs["candidate_manifest_path"]),
            "--candidate-evidence-root",
            str(self.candidate_inputs["candidate_evidence_root"]),
            "--source-transition-policy",
            str(self.source_policy),
            "--expected-source-transition-policy-sha256",
            self.source_policy_sha256,
            "--source-transition-verifier",
            str(self.source_verifier),
            "--expected-source-transition-verifier-sha256",
            self.source_verifier_sha256,
            "--final-audit-authority",
            str(self.final_audit_authority),
            "--expected-final-audit-authority-sha256",
            self.final_audit_authority_sha256,
        ]
        created = subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/v110-final-release-receipt.py"),
                "--create",
                "--output",
                str(output),
                *common,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(created.returncode, 0)
        self.assertIn("candidate receipt replay failed", created.stderr)
        self.assertFalse(output.exists())



class InstalledPrefixReplayEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.prefix = pathlib.Path(self.temporary.name).resolve() / "installed"
        self.report = {
            "state": "PASS",
            "prefix": str(self.prefix),
            "package_sha256": "b" * 64,
            "manifest": {"source_commit": "a" * 40},
            "v110_candidate_gate": {"receipt_id": "c" * 64},
        }
        # This real subprocess rejects missing or inherited HOME and import
        # injection, then supplies the retained report for the normal replay.
        launcher = (
            "#!/usr/bin/python3\n"
            "import json, os, pathlib, pwd, sys\n"
            "assert os.environ['HOME'] == str(pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True))\n"
            "assert 'PYTHONPATH' not in os.environ\n"
            "assert os.environ['PYTHONNOUSERSITE'] == '1'\n"
            "pathlib.Path(sys.argv[sys.argv.index('--report') + 1]).write_text("
            + repr(json.dumps(self.report)) + ")\n"
        ).encode()
        self.members = {
            "installed_verifier": b"fixture verifier\n",
            "installed_launcher": launcher,
            "installed_self_check": b"fixture self-check\n",
        }
        for relative, key in (
            (FINAL.INSTALLED_VERIFIER_PATH, "installed_verifier"),
            (FINAL.INSTALLED_LAUNCHER_PATH, "installed_launcher"),
            (FINAL.INSTALLED_SELF_CHECK_PATH, "installed_self_check"),
        ):
            destination = self.prefix / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.members[key])
            destination.chmod(0o700)

    def replay(self):
        FINAL._verify_installed_prefix(
            prefix=self.prefix,
            retained_report=self.report,
            archive_members=self.members,
            source_commit="a" * 40,
            package_sha256="b" * 64,
            candidate_receipt_id="c" * 64,
        )

    def test_replay_child_gets_account_home_from_empty_or_poisoned_parent(self):
        for environment in ({}, {"HOME": "/nonexistent/injected-home", "PYTHONPATH": "/injected"}):
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.replay()

    def test_missing_account_home_fails_before_launch(self):
        with mock.patch.object(FINAL.pwd, "getpwuid", side_effect=KeyError("missing account")):
            with mock.patch.object(FINAL.subprocess, "run") as run:
                with self.assertRaisesRegex(FINAL.FinalReleaseError, "account home is unavailable"):
                    self.replay()
                run.assert_not_called()

    def test_non_directory_account_home_fails_before_launch(self):
        with mock.patch.object(FINAL.pwd, "getpwuid", return_value=types.SimpleNamespace(
            pw_dir=str(self.prefix / FINAL.INSTALLED_VERIFIER_PATH)
        )):
            with mock.patch.object(FINAL.subprocess, "run") as run:
                with self.assertRaisesRegex(FINAL.FinalReleaseError, "account home is not a directory"):
                    self.replay()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
