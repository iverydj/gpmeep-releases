#!/usr/bin/env python3
"""Adversarial and local-bare integration tests for the release-only mirror."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "scripts/v110-release-only-receipt.py"
PUBLISHER = ROOT / "scripts/publish-v110-release-only.sh"


def load_tool():
    spec = importlib.util.spec_from_file_location("gpmeep_v110_release_only", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MIRROR = load_tool()


def git(*arguments: str, cwd: pathlib.Path | None = None, input_bytes: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "1700000000 +0000",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_DATE": "1700000000 +0000",
    }
    result = subprocess.run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
        timeout=60,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", "replace"))
    return result


def oid(repo: pathlib.Path, revision: str) -> str:
    return git("rev-parse", revision, cwd=repo).stdout.decode("ascii").strip()


class V110ReleaseOnlyMirrorTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.source = self.root / "development"
        self.source.mkdir()
        git("init", "-b", "main", cwd=self.source)

        (self.source / "secret.txt").write_bytes(b"DO-NOT-DISTRIBUTE-SENTINEL\n")
        (self.source / "old.txt").write_bytes(b"old development bytes\n")
        git("add", "--", "secret.txt", "old.txt", cwd=self.source)
        git("commit", "-m", "private ancestor", cwd=self.source)
        self.secret_commit = oid(self.source, "HEAD")
        self.secret_blob = oid(self.source, "HEAD:secret.txt")

        (self.source / "secret.txt").unlink()
        (self.source / "old.txt").unlink()
        executable = self.source / "run-example"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        (self.source / "empty").write_bytes(b"")
        (self.source / "binary.bin").write_bytes(bytes(range(256)))
        (self.source / "unicode-한글.txt").write_text("한글\n", encoding="utf-8")
        (self.source / "target.txt").write_bytes(b"symlink target\n")
        (self.source / "link").symlink_to("target.txt")
        (self.source / ".gitattributes").write_text("* text eol=crlf\n*.bin filter=evil\n", encoding="utf-8")
        dependabot = self.source / ".github/dependabot.yml"
        dependabot.parent.mkdir()
        dependabot.write_bytes((ROOT / ".github/dependabot.yml").read_bytes())
        git("add", "-A", cwd=self.source)
        git("commit", "-m", "authorized release tree", cwd=self.source)
        self.source_commit = oid(self.source, "HEAD")
        self.source_tree = oid(self.source, "HEAD^{tree}")

        self.assets = self.root / "assets"
        self.assets.mkdir(mode=0o700)
        self.package = self.assets / "gpmeep-1.1.0-cuda_test.conda"
        self.package.write_bytes(b"exact package bytes\n")
        self.candidate = self.assets / "candidate.json"
        candidate_payload = {
            "schema_version": "gpmeep-v110-release-gate-v3",
            "state": "CANDIDATE_BUILD_AUTHORIZED",
            "source_commit": self.source_commit,
        }
        candidate_payload["receipt_id"] = MIRROR._canonical_sha256(candidate_payload)
        self.candidate.write_bytes(MIRROR._canonical_bytes(candidate_payload) + b"\n")
        self.final = self.assets / "final.json"
        final_package = {
            "file_name": self.package.name,
            "size_bytes": self.package.stat().st_size,
            "sha256": MIRROR._sha256_file(self.package),
            "format": "conda-v2",
            "name": "gpmeep",
            "version": "1.1.0",
            "build": "cuda_test",
            "subdir": "linux-64",
            "embedded_candidate_receipt_id": candidate_payload["receipt_id"],
        }
        final_payload = {
            "schema": "gpmeep-v110-final-release-v1",
            "state": "RELEASE_AUTHORIZED",
            "candidate_authority": {
                "receipt_id": candidate_payload["receipt_id"],
                "source_commit": self.source_commit,
                "file": {
                    "size_bytes": self.candidate.stat().st_size,
                    "sha256": MIRROR._sha256_file(self.candidate),
                },
            },
            "package": final_package,
            "tag_intent": {"name": "v1.1.0", "target_commit": self.source_commit},
        }
        final_payload["receipt_id"] = MIRROR._canonical_sha256(final_payload)
        self.final.write_bytes(MIRROR._canonical_bytes(final_payload) + b"\n")
        source_records = []
        for role, path in sorted(
            {"candidate_receipt": self.candidate, "final_receipt": self.final, "package": self.package}.items()
        ):
            source_records.append(
                {
                    "role": role,
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": MIRROR._sha256_file(path),
                }
            )
        source_unsigned = {
            "schema": "gpmeep-v110-release-asset-manifest-v1",
            "assets": source_records,
        }
        source_unsigned["manifest_id"] = MIRROR._canonical_sha256(source_unsigned)
        self.source_manifest = self.assets / "source-assets.json"
        self.source_manifest.write_bytes(MIRROR._canonical_bytes(source_unsigned) + b"\n")

        private_tag_message = MIRROR._expected_private_tag_message(
            source_commit=self.source_commit,
            package_sha256=final_package["sha256"],
            candidate_id=candidate_payload["receipt_id"],
            final_id=final_payload["receipt_id"],
            manifest_id=source_unsigned["manifest_id"],
        )
        git("tag", "-a", "v1.1.0", "-m", private_tag_message, cwd=self.source)
        self.source_tag_object = oid(self.source, "refs/tags/v1.1.0")

        self.final_verifier = self.root / "fixture-final-verifier.py"
        self.final_verifier.write_text(
            """#!/usr/bin/env python3
import argparse, hashlib, json, pathlib
p=argparse.ArgumentParser(); p.add_argument('--verify', action='store_true'); p.add_argument('--receipt', required=True); p.add_argument('--package', required=True); p.add_argument('--candidate-receipt', required=True); a=p.parse_args()
candidate=json.loads(pathlib.Path(a.candidate_receipt).read_text()); final=json.loads(pathlib.Path(a.receipt).read_text()); package=pathlib.Path(a.package).read_bytes()
assert a.verify and candidate['state']=='CANDIDATE_BUILD_AUTHORIZED'
assert final['state']=='RELEASE_AUTHORIZED' and final['candidate_authority']['receipt_id']==candidate['receipt_id']
assert final['package']['sha256']==hashlib.sha256(package).hexdigest()
print(json.dumps({'schema':'gpmeep-v110-final-release-v1','state':'PASS','receipt_id':final['receipt_id'],'package_sha256':final['package']['sha256']}, sort_keys=True))
""",
            encoding="utf-8",
        )
        self.final_verifier_sha256 = MIRROR._sha256_file(self.final_verifier)
        self.final_verifier_args: list[str] = []
        verification = MIRROR._run_pinned_final_verifier(
            verifier_path=self.final_verifier,
            expected_verifier_sha256=self.final_verifier_sha256,
            verifier_arguments=self.final_verifier_args,
            final_receipt_path=self.final,
            package_path=self.package,
            candidate_receipt_path=self.candidate,
        )

        def record(path: pathlib.Path, identity_name: str, identity: str) -> dict[str, object]:
            return {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": MIRROR._sha256_file(path),
                identity_name: identity,
            }

        self.source_authority = {
            "schema": MIRROR.SOURCE_AUTHORITY_SCHEMA,
            "repository": {"slug": "iverydj/gpmeep", "id": 101, "node_id": "R_private"},
            "source_tag": {"name": "v1.1.0", "object_id": self.source_tag_object},
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "final_receipt": record(self.final, "receipt_id", final_payload["receipt_id"]),
            "candidate_receipt": record(self.candidate, "receipt_id", candidate_payload["receipt_id"]),
            "package": {
                "name": self.package.name,
                "size_bytes": self.package.stat().st_size,
                "sha256": MIRROR._sha256_file(self.package),
                "source_commit": self.source_commit,
                "conda_metadata": {
                    key: value
                    for key, value in final_package.items()
                    if key not in {"file_name", "size_bytes", "sha256"}
                },
            },
            "source_asset_manifest": record(self.source_manifest, "manifest_id", source_unsigned["manifest_id"]),
            "private_release": {
                "id": 303,
                "node_id": "R_source_release",
                "published": True,
                "prerelease": False,
                "immutable": True,
            },
            "verification": verification,
        }
        self.identity = {
            "name": "gpmeep release mirror",
            "email": "release@gpmeep.invalid",
            "epoch": 1700000100,
            "timezone": "+0000",
        }
        bootstrap_state = self.root / "bootstrap-snapshot"
        bootstrap_state.mkdir(mode=0o700)
        bootstrap_snapshot = MIRROR.prepare_snapshot(
            source_repo=self.source,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            state_root=bootstrap_state,
            identity=self.identity,
            source_tag="v1.1.0",
            source_tag_object=self.source_tag_object,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        (
            self.prior_repo,
            self.prior_assets_root,
            self.prior_authority,
            self.prior_release_metadata_path,
        ) = self.make_prior_release("default-v103-prior")
        self.prior_refs = ["refs/tags/v1.0.3"]
        self.external_audit = self.assets / "independent-mirror-audit.json"
        external_audit_value = {
            "schema": MIRROR.EXTERNAL_AUDIT_SCHEMA,
            "state": "GO",
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "snapshot_id": bootstrap_snapshot["snapshot_id"],
            "snapshot_commit": bootstrap_snapshot["snapshot_commit"],
            "source_tag_object": self.source_tag_object,
            "source_repository": {"slug": "iverydj/gpmeep", "id": 101, "node_id": "R_private"},
            "private_release": {
                "id": 303,
                "node_id": "R_source_release",
                "tag": "v1.1.0",
                "published": True,
                "prerelease": False,
                "immutable": True,
            },
            "target_repository": {
                "slug": "example-organization/gpmeep-releases",
                "id": 202,
                "node_id": "R_release_only",
            },
            "organization_access": {
                "organization": "example-organization",
                "id": 404,
                "node_id": "O_fixture",
                "default_repository_permission": "read",
                "all_members_can_read": True,
            },
            "implementation": {
                "receipt_tool_sha256": MIRROR._sha256_file(TOOL_PATH),
                "publisher_sha256": MIRROR._sha256_file(PUBLISHER),
                "tests_sha256": MIRROR._sha256_file(pathlib.Path(__file__)),
            },
            "github_cli": {
                "version": MIRROR.AUDITED_GITHUB_CLI_VERSION,
                "sha256": MIRROR.AUDITED_GITHUB_CLI_SHA256,
            },
            "source_verifier_sha256": self.final_verifier_sha256,
            "target_history": {
                "never_imported_or_transferred": True,
                "advertised_refs_sha256": MIRROR._canonical_sha256([
                    {"object_id": MIRROR.V103_TARGET_TAG_OBJECT, "ref": "refs/tags/v1.0.3"}
                ]),
                "prior_releases_sha256": MIRROR._canonical_sha256([self.prior_authority]),
            },
            "findings": {"open_critical": 0, "open_high": 0, "open_medium": 0},
            "auditor": {"independent": True, "name": "fixture adversary", "completed_at_utc": "2026-09-04T00:00:00Z"},
        }
        external_audit_value["audit_id"] = MIRROR._canonical_sha256(external_audit_value)
        self.external_audit.write_bytes(MIRROR._canonical_bytes(external_audit_value) + b"\n")
        self.external_audit.chmod(0o600)
        self.external_audit_sha256 = MIRROR._sha256_file(self.external_audit)
        audit_record = MIRROR.external_audit_record(self.external_audit, self.external_audit_sha256)
        self.target_history_attestation = {
            "schema": MIRROR.TARGET_HISTORY_SCHEMA,
            "state": "HISTORY_CONFIDENTIALITY_ATTESTED",
            "mode": "audited-prior-releases",
            "never_imported_or_transferred": True,
            "repository": {
                "slug": "example-organization/gpmeep-releases",
                "id": 202,
                "node_id": "R_release_only",
            },
            "advertised_refs": self.prior_refs,
            "prior_releases": [self.prior_authority],
            "external_audit": {key: audit_record[key] for key in ("name", "size_bytes", "sha256", "audit_id")},
        }
        self.target_history_attestation["attestation_id"] = MIRROR._canonical_sha256(self.target_history_attestation)
        self.target_history_attestation_path = self.root / "target-history-attestation.json"
        self.target_history_attestation_path.write_bytes(
            MIRROR._canonical_bytes(self.target_history_attestation) + b"\n"
        )
        self.target_preflight = {
            "schema": MIRROR.TARGET_PREFLIGHT_SCHEMA,
            "repository": {
                "slug": "example-organization/gpmeep-releases",
                "id": 202,
                "node_id": "R_release_only",
            },
            "visibility": "private",
            "fork": False,
            "template": False,
            "archived": False,
            "disabled": False,
            "actions_enabled": False,
            "immutable_releases_enabled": True,
            "advertised_refs": self.prior_refs,
            "bootstrap": {
                "mode": "audited-prior-releases",
                "attestation_sha256": MIRROR._sha256_file(self.target_history_attestation_path),
                "attestation_id": self.target_history_attestation["attestation_id"],
            },
            "organization_access": {
                "organization": "example-organization",
                "id": 404,
                "node_id": "O_fixture",
                "default_repository_permission": "read",
                "all_members_can_read": True,
            },
            "automation": {
                "actions_enabled": False,
                "dependabot_security_updates_enabled": False,
                "dependabot_version_updates_enabled": False,
                "no_live_default_branch": True,
                "repository_administration_read_authorized": True,
                "source_dependabot_limits": {"github-actions": 0, "pip": 0},
            },
            "external_mirror_audit": audit_record,
        }
        self.source_authority_path = self.root / "source-authority.json"
        self.target_preflight_path = self.root / "target-preflight.json"
        self.source_authority_path.write_bytes(MIRROR._canonical_bytes(self.source_authority) + b"\n")
        self.target_preflight_path.write_bytes(MIRROR._canonical_bytes(self.target_preflight) + b"\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_state(self, name: str = "state") -> tuple[pathlib.Path, dict[str, object], dict[str, object], dict[str, object]]:
        state = self.root / name
        state.mkdir(mode=0o700)
        snapshot = MIRROR.prepare_snapshot(
            source_repo=self.source,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            state_root=state,
            identity=self.identity,
            source_tag="v1.1.0",
            source_tag_object=self.source_tag_object,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        receipt = MIRROR.create_receipt(self.source_authority, snapshot, self.target_preflight)
        local_assets = self.root / f"{name}-assets"
        local_assets.mkdir(mode=0o700)
        for path in (
            self.package,
            self.candidate,
            self.final,
            self.source_manifest,
            self.external_audit,
            self.target_history_attestation_path,
        ):
            shutil.copyfile(path, local_assets / path.name)
            if path == self.external_audit:
                (local_assets / path.name).chmod(0o600)
        receipt_path = local_assets / "gpmeep-v1.1.0-release-only-receipt.json"
        MIRROR._write_new_json(receipt_path, receipt, "test mirror receipt")
        manifest = MIRROR.create_distribution_manifest(
            source_manifest_path=local_assets / self.source_manifest.name,
            asset_directory=local_assets,
            mirror_receipt_path=receipt_path,
            target_history_attestation_path=local_assets / self.target_history_attestation_path.name,
        )
        attestation = MIRROR.seal_tag(
            snapshot_repo=state / "isolated-snapshot.git",
            output_root=state,
            receipt=receipt,
            distribution_manifest=manifest,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        return state, snapshot, receipt, attestation

    def seed_prior_target(self, target: pathlib.Path) -> None:
        if git(
            "--git-dir", str(target), "show-ref", "--verify", "refs/tags/v1.0.3",
            check=False,
        ).returncode != 0:
            git(
                "--git-dir", str(self.prior_repo), "push", str(target),
                "refs/tags/v1.0.3:refs/tags/v1.0.3",
            )

    @staticmethod
    def prior_ref_state() -> bytes:
        return f"{MIRROR.V103_TARGET_TAG_OBJECT} refs/tags/v1.0.3\n".encode()

    def publisher_command(self, state: pathlib.Path, target: pathlib.Path, *extra: str) -> list[str]:
        self.seed_prior_target(target)
        return [
            str(PUBLISHER),
            "--source-repo", str(self.source),
            "--source-authority", str(self.source_authority_path),
            "--target-preflight", str(self.target_preflight_path),
            "--target-history-attestation", str(self.target_history_attestation_path),
            "--external-audit", str(self.external_audit),
            "--external-audit-sha256", self.external_audit_sha256,
            "--asset-directory", str(self.assets),
            "--state-root", str(state),
            "--repository", "example-organization/gpmeep-releases",
            "--identity-name", self.identity["name"],
            "--identity-email", self.identity["email"],
            "--identity-epoch", str(self.identity["epoch"]),
            "--identity-timezone", self.identity["timezone"],
            "--forbid-object", self.secret_commit,
            "--forbid-object", self.secret_blob,
            "--final-verifier", str(self.final_verifier),
            "--final-verifier-sha256", self.final_verifier_sha256,
            "--local-bare-remote", str(target),
            "--prior-assets-root", str(self.prior_assets_root),
            *extra,
        ]

    def run_publisher(
        self,
        command: list[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env=env,
        )
        if check and result.returncode != 0:
            self.fail(
                "stdout:\n"
                + result.stdout.decode("utf-8", "replace")
                + "\nstderr:\n"
                + result.stderr.decode("utf-8", "replace")
            )
        return result

    def run_tool(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            ["/usr/bin/python3", "-I", "-S", str(TOOL_PATH), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        if check and result.returncode != 0:
            self.fail(result.stderr.decode("utf-8", "replace"))
        return result

    @staticmethod
    def write_json(path: pathlib.Path, value: object) -> None:
        path.write_bytes(MIRROR._canonical_bytes(value) + b"\n")
        path.chmod(0o600)

    @staticmethod
    def signed_release_fixture(
        *, repository_id: int, owner_id: int, release_id: int, tag_object: str,
        assets: list[dict[str, object]],
    ) -> dict[str, object]:
        purl = f"pkg:github/{MIRROR.TARGET_REPOSITORY}@v1.0.3"
        subjects = [
            {"name": item["name"], "digest": {"sha256": item["sha256"]}}
            for item in assets
        ]
        subjects.append({"uri": purl, "digest": {"sha1": tag_object}})
        statement = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": subjects,
            "predicateType": "https://in-toto.io/attestation/release/v0.2",
            "predicate": {
                "databaseId": str(release_id), "ownerId": str(owner_id),
                "packageId": str(repository_id), "purl": purl,
                "repository": MIRROR.TARGET_REPOSITORY,
                "repositoryId": str(repository_id), "tag": "v1.0.3",
            },
        }
        payload = MIRROR._canonical_bytes(statement)
        bundle = {
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {
                "timestampVerificationData": {"rfc3161Timestamps": [{
                    "signedTimestamp": base64.b64encode(b"T" * 64).decode(),
                }]},
                "certificate": {"rawBytes": base64.b64encode(b"C" * 128).decode()},
            },
            "dsseEnvelope": {
                "payload": base64.b64encode(payload).decode(),
                "payloadType": "application/vnd.in-toto+json",
                "signatures": [{"sig": base64.b64encode(b"S" * 64).decode()}],
            },
        }
        return {
            "attestation": {"bundle": bundle, "bundle_url": "", "initiator": ""},
            "verificationResult": {
                "mediaType": "application/vnd.dev.sigstore.verificationresult+json;version=0.1",
                "statement": statement,
                "signature": {"certificate": {
                    "certificateIssuer": "Fixture Fulcio issuer",
                    "subjectAlternativeName": "https://dotcom.releases.github.com",
                }},
                "verifiedTimestamps": [{
                    "type": "TimestampAuthority", "uri": "timestamp.githubapp.com",
                    "timestamp": "2026-09-04T00:00:00Z",
                }],
                "verifiedIdentity": {
                    "subjectAlternativeName": {
                        "subjectAlternativeName": "",
                        "regexp": r"^https://dotcom\.releases\.github\.com$",
                    },
                    "issuer": {"issuer": "", "regexp": ".*"},
                },
            },
        }

    def make_prior_release(
        self, name: str = "prior"
    ) -> tuple[pathlib.Path, pathlib.Path, dict[str, object], pathlib.Path]:
        prior = self.root / f"{name}.git"
        git("init", "--bare", "--template=", str(prior))
        alternate = prior / "objects/info/alternates"
        object_directory = git("-C", str(ROOT), "rev-parse", "--path-format=absolute",
                               "--git-path", "objects").stdout.decode().strip()
        alternate.write_text(object_directory + "\n")
        tree = MIRROR.V103_SOURCE_TREE
        self.assertEqual(git("-C", str(ROOT), "cat-file", "-t", tree).stdout.strip(), b"tree")
        commit_payload = (
            f"tree {tree}\n"
            "author Dong-Joon Yi <iver.ydj@gmail.com> 1788408852 +0900\n"
            "committer Dong-Joon Yi <iver.ydj@gmail.com> 1788408852 +0900\n\n"
            "gpmeep v1.0.3 release snapshot\n"
        ).encode()
        root_commit = git(
            "--git-dir", str(prior), "hash-object", "-t", "commit", "-w", "--stdin",
            input_bytes=commit_payload,
        ).stdout.decode().strip()
        tag_payload = (
            f"object {root_commit}\ntype commit\ntag v1.0.3\n"
            "tagger Dong-Joon Yi <iver.ydj@gmail.com> 1788408852 +0900\n\n"
            "gpmeep v1.0.3 — field-feedback compatibility release\n"
        ).encode()
        tag = git(
            "--git-dir", str(prior), "hash-object", "-t", "tag", "-w", "--stdin",
            input_bytes=tag_payload,
        ).stdout.decode().strip()
        self.assertEqual(root_commit, MIRROR.V103_SNAPSHOT_COMMIT)
        self.assertEqual(tag, MIRROR.V103_TARGET_TAG_OBJECT)
        git("--git-dir", str(prior), "update-ref", "refs/tags/v1.0.3", tag, "0" * 40)
        git("--git-dir", str(prior), "repack", "-a", "-d")
        alternate.unlink()

        prior_assets_root = self.root / f"{name}-assets"
        prior_assets = prior_assets_root / "v1.0.3"
        prior_assets_root.mkdir(mode=0o700)
        prior_assets.mkdir(mode=0o700)
        legacy_paths: list[pathlib.Path] = []
        for index in range(MIRROR.V103_RELEASE_ASSET_COUNT):
            path = prior_assets / f"legacy-v103-{index:02d}.bin"
            path.write_bytes(f"legacy exact bytes {index}\n".encode())
            legacy_paths.append(path)
        legacy_records = [
            {
                "role": "v103_legacy_release_asset", "name": path.name,
                "size_bytes": path.stat().st_size, "sha256": MIRROR._sha256_file(path),
            }
            for path in legacy_paths
        ]
        tree_manifest = MIRROR.tree_manifest(prior, tree)
        tag_payload_sha256 = MIRROR._sha256_bytes(
            git("--git-dir", str(prior), "cat-file", "tag", tag).stdout
        )
        commit_payload_sha256 = MIRROR._sha256_bytes(
            git("--git-dir", str(prior), "cat-file", "commit", root_commit).stdout
        )
        legacy_receipt_value = {
            "schema": MIRROR.PRIOR_MIGRATION_RECEIPT_SCHEMA,
            "source_authority": {
                "repository": {
                    "slug": MIRROR.V103_SOURCE_REPOSITORY,
                    "id": MIRROR.V103_SOURCE_REPOSITORY_ID,
                    "node_id": MIRROR.V103_SOURCE_REPOSITORY_NODE_ID,
                },
                "legacy_target_repository": {
                    "slug": MIRROR.TARGET_REPOSITORY,
                    "id": MIRROR.V103_LEGACY_TARGET_REPOSITORY_ID,
                    "node_id": MIRROR.V103_LEGACY_TARGET_REPOSITORY_NODE_ID,
                },
                "source_release": {
                    "id": MIRROR.V103_SOURCE_RELEASE_ID,
                    "node_id": MIRROR.V103_SOURCE_RELEASE_NODE_ID, "tag": "v1.0.3",
                    "draft": False, "prerelease": False, "immutable": False,
                    "asset_count": MIRROR.V103_RELEASE_ASSET_COUNT,
                    "assets_sha256": MIRROR._canonical_sha256(legacy_records),
                },
                "source_commit": MIRROR.V103_SOURCE_COMMIT,
                "source_tree": MIRROR.V103_SOURCE_TREE,
                "source_tag": {
                    "name": "v1.0.3", "object_id": MIRROR.V103_SOURCE_TAG_OBJECT,
                    "target": MIRROR.V103_SOURCE_COMMIT,
                    "payload_sha256": MIRROR.V103_SOURCE_TAG_PAYLOAD_SHA256,
                },
            },
            "snapshot": {
                "snapshot_commit": root_commit,
                "snapshot_tree": tree,
                "parent_count": 0,
                "reachable_commit_count": 1,
                "tree_manifest_sha256": tree_manifest["manifest_sha256"],
                "commit_payload_sha256": commit_payload_sha256,
                "tag_object": tag,
                "tag_target": root_commit,
                "tag_payload_sha256": tag_payload_sha256,
            },
            "target": {
                "repository": {"slug": MIRROR.TARGET_REPOSITORY, "id": 202, "node_id": "R_release_only"},
                "organization": {"login": "example-organization", "id": 404, "node_id": "O_fixture"},
                "tag": "v1.0.3",
                "release": {"id": 103, "node_id": "R_v103", "published": True, "prerelease": False, "immutable": True},
            },
        }
        legacy_receipt_value["publication_id"] = MIRROR._canonical_sha256(legacy_receipt_value)
        legacy_receipt = prior_assets / MIRROR.PRIOR_MIGRATION_RECEIPT_NAME
        self.write_json(legacy_receipt, legacy_receipt_value)
        receipt_record = {
            "role": "migration_receipt", "name": legacy_receipt.name,
            "size_bytes": legacy_receipt.stat().st_size, "sha256": MIRROR._sha256_file(legacy_receipt),
        }
        legacy_manifest_value = {
            "schema": MIRROR.PRIOR_MIGRATION_MANIFEST_SCHEMA,
            "mirror_publication_id": legacy_receipt_value["publication_id"],
            "legacy_asset_count": MIRROR.V103_RELEASE_ASSET_COUNT,
            "assets": sorted([*legacy_records, receipt_record], key=lambda item: item["name"]),
        }
        legacy_manifest_value["manifest_id"] = MIRROR._canonical_sha256(legacy_manifest_value)
        legacy_manifest = prior_assets / MIRROR.PRIOR_MIGRATION_MANIFEST_NAME
        self.write_json(legacy_manifest, legacy_manifest_value)
        signed_asset_records = [
            {"name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
            for item in legacy_manifest_value["assets"]
        ]
        signed_asset_records.append({
            "name": legacy_manifest.name, "size_bytes": legacy_manifest.stat().st_size,
            "sha256": MIRROR._sha256_file(legacy_manifest),
        })
        signed_asset_records.sort(key=lambda item: item["name"])
        live_assets = [
            {
                "id": 500 + index, "node_id": f"RA_{500 + index}",
                "name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"],
            }
            for index, item in enumerate(signed_asset_records)
        ]
        release_metadata = {
            "schema": "gpmeep-v103-release-only-live-release-v1",
            "repository": {"slug": MIRROR.TARGET_REPOSITORY, "id": 202, "node_id": "R_release_only"},
            "organization": {"login": "example-organization", "id": 404, "node_id": "O_fixture"},
            "release": {
                "id": 103, "node_id": "R_v103", "tag": "v1.0.3",
                "title": "gpmeep v1.0.3", "body": "immutable migration",
                "draft": False, "prerelease": False, "immutable": True,
                "assets": live_assets,
            },
        }
        release_metadata["capture_id"] = MIRROR._canonical_sha256(release_metadata)
        release_metadata_path = prior_assets / "release-metadata.json"
        self.write_json(release_metadata_path, release_metadata)
        attestation_path = prior_assets / MIRROR.PRIOR_RELEASE_ATTESTATION_NAME
        self.write_json(
            attestation_path,
            self.signed_release_fixture(
                repository_id=202, owner_id=404, release_id=103,
                tag_object=tag, assets=signed_asset_records,
            ),
        )
        authority = MIRROR._derive_prior_release_authority(
            repo=prior,
            tag="v1.0.3",
            release_directory=prior_assets,
        )
        return prior, prior_assets_root, authority, release_metadata_path

    def make_v103_migration_audit(
        self, name: str = "migration"
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, str, pathlib.Path]:
        legacy_repo, evidence_root, authority, _metadata = self.make_prior_release(name)
        evidence = evidence_root / "v1.0.3"
        receipt = json.loads((evidence / MIRROR.PRIOR_MIGRATION_RECEIPT_NAME).read_text())
        manifest = json.loads((evidence / MIRROR.PRIOR_MIGRATION_MANIFEST_NAME).read_text())
        legacy_assets = self.root / f"{name}-legacy-assets"
        legacy_assets.mkdir(mode=0o700)
        legacy_records = [item for item in manifest["assets"] if item["role"] == "v103_legacy_release_asset"]
        for record in legacy_records:
            shutil.copyfile(evidence / record["name"], legacy_assets / record["name"])
        target = self.root / f"{name}-fresh-target.git"
        git("init", "--bare", "--template=", str(target))
        fake_gh = self.root / f"{name}-gh"
        fake_gh.write_bytes(b"#!/bin/sh\nexit 0\n")
        fake_gh.chmod(0o755)
        audit = {
            "schema": MIRROR.PRIOR_MIGRATION_AUDIT_SCHEMA,
            "state": "GO",
            "source_authority": receipt["source_authority"],
            "snapshot": {**receipt["snapshot"], "reachable_inventory": authority["reachable_inventory"]},
            "target": {
                "repository": receipt["target"]["repository"],
                "organization": receipt["target"]["organization"],
                "never_imported_or_transferred": True,
                "empty_ref_inventory_sha256": MIRROR._canonical_sha256([]),
            },
            "legacy_assets": legacy_records,
            "implementation": {
                "receipt_tool_sha256": MIRROR._sha256_file(TOOL_PATH),
                "migration_publisher_sha256": MIRROR._sha256_file(ROOT / "scripts/publish-v103-release-only-migration.sh"),
                "tests_sha256": MIRROR._sha256_file(pathlib.Path(__file__)),
            },
            "github_cli": {
                "version": MIRROR.AUDITED_GITHUB_CLI_VERSION,
                "sha256": MIRROR._sha256_file(fake_gh),
            },
            "findings": {"open_critical": 0, "open_high": 0, "open_medium": 0},
            "auditor": {"independent": True, "name": "migration fixture adversary", "completed_at_utc": "2026-09-04T00:00:00Z"},
        }
        audit["audit_id"] = MIRROR._canonical_sha256(audit)
        audit_path = self.root / f"{name}-migration-audit.json"
        self.write_json(audit_path, audit)
        return legacy_repo, legacy_assets, target, audit_path, MIRROR._sha256_file(audit_path), fake_gh

    def make_v103_controller_fixture(
        self, name: str = "migration-controller"
    ) -> dict[str, pathlib.Path]:
        legacy_repo, legacy_assets, target, audit_path, _audit_sha, _unused_gh = self.make_v103_migration_audit(name)
        fixture_root = self.root / f"{name}-controller"
        fixture_root.mkdir(mode=0o700)
        uploads = fixture_root / "uploads"
        uploads.mkdir(mode=0o700)
        state_file = fixture_root / "github-state.json"
        self.write_json(state_file, {"release": None, "next_asset_id": 700, "upload_calls": 0})
        fake_gh = fixture_root / "gh"
        fake_gh.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/python3
                import base64, hashlib, json, pathlib, shutil, subprocess, sys
                STATE = pathlib.Path({str(state_file)!r})
                UPLOADS = pathlib.Path({str(uploads)!r})
                SOURCE = pathlib.Path({str(legacy_assets)!r})
                TARGET_REPO = pathlib.Path({str(target)!r})
                TARGET = {MIRROR.TARGET_REPOSITORY!r}
                SOURCE_REPO = {MIRROR.V103_SOURCE_REPOSITORY!r}
                argv = sys.argv[1:]
                def load(): return json.loads(STATE.read_text())
                def save(value):
                    temporary = STATE.with_suffix('.tmp')
                    temporary.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')
                    temporary.chmod(0o600); temporary.replace(STATE)
                def option(name): return argv[argv.index(name) + 1]
                def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
                def source_assets():
                    return [{{'id':100+i,'node_id':'SA_'+str(i),'name':p.name,'size':p.stat().st_size,'digest':'sha256:'+sha(p)}} for i,p in enumerate(sorted(SOURCE.iterdir()))]
                def signed(release):
                    tag = subprocess.check_output(['/usr/bin/git','--git-dir',str(TARGET_REPO),'rev-parse','refs/tags/v1.0.3']).decode().strip()
                    purl = 'pkg:github/' + TARGET + '@v1.0.3'
                    subjects = [{{'name':a['name'],'digest':{{'sha256':a['digest'].split(':',1)[1]}}}} for a in release['assets']]
                    subjects.append({{'uri':purl,'digest':{{'sha1':tag}}}})
                    statement = {{'_type':'https://in-toto.io/Statement/v1','subject':subjects,'predicateType':'https://in-toto.io/attestation/release/v0.2','predicate':{{'databaseId':str(release['id']),'ownerId':'404','packageId':'202','purl':purl,'repository':TARGET,'repositoryId':'202','tag':'v1.0.3'}}}}
                    payload = json.dumps(statement, sort_keys=True, separators=(',', ':')).encode()
                    bundle = {{'mediaType':'application/vnd.dev.sigstore.bundle.v0.3+json','verificationMaterial':{{'timestampVerificationData':{{'rfc3161Timestamps':[{{'signedTimestamp':base64.b64encode(b'T'*64).decode()}}]}},'certificate':{{'rawBytes':base64.b64encode(b'C'*128).decode()}}}},'dsseEnvelope':{{'payload':base64.b64encode(payload).decode(),'payloadType':'application/vnd.in-toto+json','signatures':[{{'sig':base64.b64encode(b'S'*64).decode()}}]}}}}
                    return {{'attestation':{{'bundle':bundle,'bundle_url':'','initiator':''}},'verificationResult':{{'mediaType':'application/vnd.dev.sigstore.verificationresult+json;version=0.1','statement':statement,'signature':{{'certificate':{{'certificateIssuer':'Fixture Fulcio','subjectAlternativeName':'https://dotcom.releases.github.com'}}}},'verifiedTimestamps':[{{'type':'TimestampAuthority','uri':'timestamp.githubapp.com','timestamp':'2026-09-04T00:00:00Z'}}],'verifiedIdentity':{{'subjectAlternativeName':{{'subjectAlternativeName':'','regexp':'^https://dotcom\\\\.releases\\\\.github\\\\.com$'}},'issuer':{{'issuer':'','regexp':'.*'}}}}}}}}
                if len(argv) >= 3 and argv[0] == 'release' and argv[-1] == '--help':
                    print('gh release ' + argv[1] + ' --repo OWNER/REPO'); sys.exit(0)
                state = load()
                if argv and argv[0] == 'api':
                    endpoint = argv[-1]
                    if endpoint.endswith('/automated-security-fixes'):
                        print('HTTP/2 404 Not Found'); sys.exit(1)
                    if endpoint == f'/repos/{{SOURCE_REPO}}':
                        if state.get('release') and state['release'].get('immutable') and state.get('terminal_verify_seen') and state.get('fail_before_terminal_publish_once') and not state.get('failed_before_terminal_publish'):
                            state['failed_before_terminal_publish']=True; save(state); sys.exit(76)
                        print(json.dumps({{'full_name':SOURCE_REPO,'id':{MIRROR.V103_SOURCE_REPOSITORY_ID},'node_id':{MIRROR.V103_SOURCE_REPOSITORY_NODE_ID!r},'private':True,'archived':False,'disabled':False}})); sys.exit(0)
                    if endpoint == f'/repos/{{SOURCE_REPO}}/releases/tags/v1.0.3':
                        print(json.dumps({{'id':{MIRROR.V103_SOURCE_RELEASE_ID},'node_id':{MIRROR.V103_SOURCE_RELEASE_NODE_ID!r},'tag_name':'v1.0.3','draft':False,'prerelease':False,'immutable':False,'assets':source_assets()}})); sys.exit(0)
                    if endpoint == f'/repos/{{SOURCE_REPO}}/git/ref/tags/v1.0.3':
                        print(json.dumps({{'object':{{'type':'tag','sha':{MIRROR.V103_SOURCE_TAG_OBJECT!r}}}}})); sys.exit(0)
                    if endpoint == f'/repos/{{SOURCE_REPO}}/git/tags/{MIRROR.V103_SOURCE_TAG_OBJECT}':
                        print(json.dumps({{'object':{{'type':'commit','sha':{MIRROR.V103_SOURCE_COMMIT!r}}}}})); sys.exit(0)
                    if endpoint == f'/repos/{{SOURCE_REPO}}/git/commits/{MIRROR.V103_SOURCE_COMMIT}':
                        print(json.dumps({{'tree':{{'sha':{MIRROR.V103_SOURCE_TREE!r}}}}})); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}':
                        print(json.dumps({{'full_name':TARGET,'id':202,'node_id':'R_release_only','private':True,'visibility':'private','fork':False,'is_template':False,'archived':False,'disabled':False,'permissions':{{'admin':True}}}})); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/actions/permissions': print('{{"enabled":false}}'); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/immutable-releases': print('{{"enabled":true}}'); sys.exit(0)
                    if endpoint == '/orgs/example-organization':
                        print(json.dumps({{'login':'example-organization','id':404,'node_id':'O_fixture','default_repository_permission':'read'}})); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/releases?per_page=100':
                        releases=([state['release']] if state['release'] else [])+state.get('extra_releases',[])
                        print(json.dumps([releases])); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/releases/tags/v1.0.3':
                        if not state['release']: sys.exit(1)
                        print(json.dumps(state['release'])); sys.exit(0)
                    print('unknown api '+endpoint, file=sys.stderr); sys.exit(40)
                if len(argv) >= 2 and argv[0] == 'release':
                    command = argv[1]
                    if command == 'create':
                        state['release']={{'id':603,'node_id':'RR_v103','tag_name':'v1.0.3','name':option('--title'),'body':option('--notes'),'draft':True,'prerelease':False,'immutable':False,'assets':[]}}; save(state); sys.exit(0)
                    if command == 'upload':
                        path=pathlib.Path(argv[3]); destination=UPLOADS/path.name; shutil.copyfile(path,destination)
                        state['upload_calls']+=1; state['release']['assets'].append({{'id':state['next_asset_id'],'node_id':'RA_'+str(state['next_asset_id']),'name':path.name,'size':path.stat().st_size,'digest':'sha256:'+sha(path)}}); state['next_asset_id']+=1; save(state)
                        fail=state.get('fail_upload_at')
                        if fail==state['upload_calls'] and not state.get('upload_failed'): state['upload_failed']=True; save(state); sys.exit(72)
                        failures=state.get('fail_upload_calls',[]); failed=state.get('failed_upload_calls',[])
                        if state['upload_calls'] in failures and state['upload_calls'] not in failed:
                            failed.append(state['upload_calls']); state['failed_upload_calls']=failed; save(state); sys.exit(72)
                        sys.exit(0)
                    if command == 'edit':
                        state['release']['draft']=False; state['release']['immutable']=True; save(state)
                        if state.get('fail_after_publish_once') and not state.get('failed_after_publish'):
                            state['failed_after_publish']=True; save(state); sys.exit(73)
                        sys.exit(0)
                    if command == 'download':
                        directory=pathlib.Path(option('--dir')); directory.mkdir(parents=True,exist_ok=True); repository=option('--repo')
                        paths=sorted(SOURCE.iterdir()) if repository==SOURCE_REPO else [UPLOADS/a['name'] for a in state['release']['assets']]
                        for index,path in enumerate(paths,1):
                            shutil.copyfile(path,directory/path.name)
                            if repository==TARGET and state.get('fail_download_after')==index and not state.get('failed_download'):
                                state['failed_download']=True; save(state); sys.exit(74)
                        sys.exit(0)
                    if command == 'verify-asset':
                        path=pathlib.Path(argv[3]); expected=UPLOADS/path.name
                        state['verify_asset_calls']=state.get('verify_asset_calls',0)+1; save(state)
                        if state.get('fail_verify_asset_at')==state['verify_asset_calls'] and not state.get('failed_verify_asset'):
                            state['failed_verify_asset']=True; save(state); sys.exit(75)
                        sys.exit(0 if expected.is_file() and sha(path)==sha(expected) else 75)
                    if command == 'verify':
                        state['terminal_verify_seen']=True; save(state)
                        print(json.dumps(signed(state['release']),sort_keys=True,separators=(',',':'))); sys.exit(0)
                print('unsupported '+repr(argv), file=sys.stderr); sys.exit(41)
                """
            )
        )
        fake_gh.chmod(0o755)
        fake_digest = MIRROR._sha256_file(fake_gh)

        instrumented = fixture_root / "instrumented"
        instrumented.mkdir(mode=0o700)
        (instrumented / "tests").mkdir(mode=0o700)
        tool_copy = instrumented / TOOL_PATH.name
        tool_copy.write_text(TOOL_PATH.read_text().replace(MIRROR.AUDITED_GITHUB_CLI_SHA256, fake_digest))
        tool_copy.chmod(0o755)
        controller_copy = instrumented / "publish-v103-release-only-migration.sh"
        controller_copy.write_text(
            (ROOT / "scripts/publish-v103-release-only-migration.sh").read_text()
            .replace(MIRROR.AUDITED_GITHUB_CLI_SHA256, fake_digest)
            .replace("git@github.com:${REPOSITORY}.git", str(target))
        )
        controller_copy.chmod(0o755)
        shutil.copyfile(pathlib.Path(__file__), instrumented / "tests/test_v110_release_only_mirror.py")

        audit = json.loads(audit_path.read_text())
        audit["implementation"] = {
            "receipt_tool_sha256": MIRROR._sha256_file(tool_copy),
            "migration_publisher_sha256": MIRROR._sha256_file(controller_copy),
            "tests_sha256": MIRROR._sha256_file(instrumented / "tests/test_v110_release_only_mirror.py"),
        }
        audit["github_cli"]["sha256"] = fake_digest
        audit.pop("audit_id")
        audit["audit_id"] = MIRROR._canonical_sha256(audit)
        self.write_json(audit_path, audit)
        return {
            "legacy_repo": legacy_repo, "legacy_assets": legacy_assets,
            "target": target, "audit": audit_path, "gh": fake_gh,
            "controller": controller_copy, "state": state_file,
        }

    def make_github_stub(self, name: str = "github-stub") -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
        root = self.root / name
        root.mkdir(mode=0o700)
        state_path = root / "state.json"
        control_path = root / "control.json"
        upload_root = root / "uploads"
        upload_root.mkdir(mode=0o700)
        private_source = root / "private-source-assets"
        private_source.mkdir(mode=0o700)
        for source_asset in (self.package, self.candidate, self.final, self.source_manifest):
            shutil.copyfile(source_asset, private_source / source_asset.name)
            (private_source / source_asset.name).chmod(0o600)
        prior_evidence = self.prior_assets_root / "v1.0.3"
        prior_capture = json.loads(self.prior_release_metadata_path.read_text())
        prior_live = prior_capture["release"]
        prior_release = {
            "id": prior_live["id"],
            "node_id": prior_live["node_id"],
            "tag_name": prior_live["tag"],
            "name": prior_live["title"],
            "body": prior_live["body"],
            "draft": prior_live["draft"],
            "prerelease": prior_live["prerelease"],
            "immutable": prior_live["immutable"],
            "assets": [
                {
                    "id": item["id"], "node_id": item["node_id"],
                    "name": item["name"], "size": item["size_bytes"],
                    "digest": "sha256:" + item["sha256"],
                }
                for item in prior_live["assets"]
            ],
        }
        self.write_json(
            state_path,
            {"release": None, "prior_release": prior_release, "next_asset_id": 500, "upload_calls": 0},
        )
        self.write_json(control_path, {})
        stub = root / "gh"
        stub.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/python3
                import base64, hashlib, json, pathlib, shutil, subprocess, sys

                STATE = pathlib.Path({str(state_path)!r})
                CONTROL = pathlib.Path({str(control_path)!r})
                UPLOADS = pathlib.Path({str(upload_root)!r})
                SOURCE = pathlib.Path({str(private_source)!r})
                PRIOR = pathlib.Path({str(prior_evidence)!r})
                TARGET = {MIRROR.TARGET_REPOSITORY!r}
                PRIVATE = 'iverydj/gpmeep'
                argv = sys.argv[1:]

                def load(path):
                    return json.loads(path.read_text())

                def save(path, value):
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')
                    temporary.chmod(0o600)
                    temporary.replace(path)

                def option(name):
                    index = argv.index(name)
                    return argv[index + 1]

                def digest(path):
                    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()

                def release_attestation():
                    target_path = control.get('target_path')
                    if not target_path:
                        sys.exit(80)
                    tag_result = subprocess.run(
                        ['/usr/bin/git', '--git-dir', target_path, 'rev-parse', 'refs/tags/v1.1.0'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    if tag_result.returncode != 0:
                        sys.exit(81)
                    tag_object = tag_result.stdout.decode().strip()
                    purl = 'pkg:github/' + TARGET + '@v1.1.0'
                    subjects = []
                    for asset in state['release']['assets']:
                        subjects.append({{'name': asset['name'], 'digest': {{'sha256': asset['digest'].split(':', 1)[1]}}}})
                    subjects.append({{'uri': purl, 'digest': {{'sha1': tag_object}}}})
                    statement = {{
                        '_type': 'https://in-toto.io/Statement/v1',
                        'subject': subjects,
                        'predicateType': 'https://in-toto.io/attestation/release/v0.2',
                        'predicate': {{
                            'databaseId': str(state['release']['id']),
                            'ownerId': '404', 'packageId': '202', 'purl': purl,
                            'repository': TARGET, 'repositoryId': '202', 'tag': 'v1.1.0',
                        }},
                    }}
                    attestation_race = control.get('attestation_race')
                    if attestation_race == 'tag_object':
                        subjects[-1]['digest']['sha1'] = 'f' * 40
                    if attestation_race == 'asset_digest':
                        subjects[0]['digest']['sha256'] = 'f' * 64
                    if attestation_race == 'release_id':
                        statement['predicate']['databaseId'] = '999'
                    payload = json.dumps(statement, sort_keys=True, separators=(',', ':')).encode()
                    bundle = {{
                        'mediaType': 'application/vnd.dev.sigstore.bundle.v0.3+json',
                        'verificationMaterial': {{
                            'timestampVerificationData': {{'rfc3161Timestamps': [{{
                                'signedTimestamp': base64.b64encode(b'T' * 64).decode(),
                            }}]}},
                            'certificate': {{'rawBytes': base64.b64encode(b'C' * 128).decode()}},
                        }},
                        'dsseEnvelope': {{
                            'payload': base64.b64encode(payload).decode(),
                            'payloadType': 'application/vnd.in-toto+json',
                            'signatures': [{{'sig': base64.b64encode(b'S' * 64).decode()}}],
                        }},
                    }}
                    verified_statement = json.loads(json.dumps(statement))
                    if attestation_race == 'payload_mismatch':
                        verified_statement['predicate']['tag'] = 'v9.9.9'
                    verified_san = 'https://dotcom.releases.github.com'
                    if attestation_race == 'certificate_san':
                        verified_san = 'https://attacker.invalid'
                    return {{
                        'attestation': {{'bundle': bundle, 'bundle_url': '', 'initiator': ''}},
                        'verificationResult': {{
                            'mediaType': 'application/vnd.dev.sigstore.verificationresult+json;version=0.1',
                            'statement': verified_statement,
                            'signature': {{'certificate': {{
                                'certificateIssuer': 'Fixture Fulcio issuer',
                                'subjectAlternativeName': verified_san,
                            }}}},
                            'verifiedTimestamps': [{{
                                'type': 'TimestampAuthority', 'uri': 'timestamp.githubapp.com',
                                'timestamp': '2026-09-04T00:00:00Z',
                            }}],
                            'verifiedIdentity': {{
                                'subjectAlternativeName': {{
                                    'subjectAlternativeName': '',
                                    'regexp': '^https://dotcom\\\\.releases\\\\.github\\\\.com$',
                                }},
                                'issuer': {{'issuer': '', 'regexp': '.*'}},
                            }},
                        }},
                    }}

                def inject_ref(stage):
                    if control.get('inject_ref_stage') != stage or control.get('ref_injected'):
                        return
                    target_path = control.get('target_path')
                    if not target_path:
                        return
                    resolved = subprocess.run(
                        ['/usr/bin/git', '--git-dir', target_path, 'rev-parse', 'refs/tags/v1.1.0^{{}}'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    if resolved.returncode != 0:
                        return
                    result = subprocess.run(
                        ['/usr/bin/git', '--git-dir', target_path, '-c', 'core.hooksPath=/dev/null',
                         'update-ref', 'refs/heads/attacker', resolved.stdout.decode().strip()],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    if result.returncode != 0:
                        return
                    control['ref_injected'] = True; save(CONTROL, control)

                state = load(STATE)
                control = load(CONTROL)
                if len(argv) >= 3 and argv[0] == 'release' and argv[-1] == '--help':
                    command = argv[1]
                    if control.get('missing_capability') == command:
                        print('unsupported', file=sys.stderr)
                        sys.exit(2)
                    print(f'gh release {{command}} [options] --repo OWNER/REPO')
                    sys.exit(0)

                if argv and argv[0] == 'api':
                    endpoint = argv[-1]
                    if endpoint.endswith('/automated-security-fixes'):
                        print('HTTP/2 404 Not Found')
                        sys.exit(1)
                    if endpoint == f'/repos/{{TARGET}}':
                        repository = {{
                            'full_name': TARGET, 'id': 202, 'node_id': 'R_release_only',
                            'private': True, 'visibility': 'private', 'fork': False,
                            'is_template': False, 'archived': False, 'disabled': False,
                            'permissions': {{'admin': True, 'maintain': True, 'push': True, 'triage': True, 'pull': True}},
                        }}
                        if state.get('release', {{}}) and state['release'].get('immutable') and control.get('postpublish_race') == 'repo_archived' and not control.get('postpublish_triggered'):
                            repository['archived'] = True
                            control['postpublish_triggered'] = True
                            save(CONTROL, control)
                        if state.get('terminal_release_view_calls') == 1 and control.get('terminal_live_race') == 'repository_archived':
                            repository['archived'] = True
                        if state.get('release') is None:
                            inject_ref('after_tag')
                        print(json.dumps(repository))
                        sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/actions/permissions':
                        print(json.dumps({{'enabled': False}})); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/immutable-releases':
                        print(json.dumps({{'enabled': True}})); sys.exit(0)
                    if endpoint == '/orgs/example-organization':
                        print(json.dumps({{'login': 'example-organization', 'id': 404, 'node_id': 'O_fixture', 'default_repository_permission': 'read'}})); sys.exit(0)
                    if endpoint == f'/repos/{{PRIVATE}}':
                        print(json.dumps({{'full_name': PRIVATE, 'id': 101, 'node_id': 'R_private', 'private': True, 'archived': False, 'disabled': False}})); sys.exit(0)
                    if endpoint == f'/repos/{{PRIVATE}}/releases/tags/v1.1.0':
                        prerelease = control.get('postpublish_race') == 'source_prerelease' and state.get('release', {{}}) and state['release'].get('immutable')
                        print(json.dumps({{
                            'id': 303, 'node_id': 'R_source_release', 'tag_name': 'v1.1.0',
                            'draft': False, 'prerelease': bool(prerelease), 'immutable': True,
                            'assets': [{{'name': path.name}} for path in sorted(SOURCE.iterdir()) if path.is_file()],
                        }})); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/releases?per_page=100':
                        release = state.get('release')
                        state['release_view_calls'] = state.get('release_view_calls', 0) + 1
                        if release and release.get('immutable'):
                            state['immutable_release_views'] = state.get('immutable_release_views', 0) + 1
                            race = control.get('postpublish_release_race')
                            if race and state['immutable_release_views'] == 1:
                                if race == 'id': release['id'] += 1
                                if race == 'node': release['node_id'] = 'RR_swapped'
                                if race == 'tag': release['tag_name'] = 'v9.9.9'
                                if race == 'immutable_false': release['immutable'] = False
                            final_race = control.get('final_release_race')
                            if final_race and state['immutable_release_views'] >= 2 and not control.get('final_release_triggered'):
                                if final_race == 'id': release['id'] += 1
                                if final_race == 'node': release['node_id'] = 'RR_final_swapped'
                                control['final_release_triggered'] = True; save(CONTROL, control)
                        if release and release.get('draft') and len(release.get('assets', [])) == 8 and not control.get('prepublish_triggered'):
                            race = control.get('prepublish_race')
                            if race == 'prerelease': release['prerelease'] = True
                            if race == 'title': release['name'] = 'attacker title'
                            if race == 'extra':
                                release['assets'].append({{'id': 9999, 'node_id': 'RA_extra', 'name': 'attacker-extra', 'size': 1, 'digest': 'sha256:' + '0' * 64}})
                            if race:
                                control['prepublish_triggered'] = True
                                state['release'] = release
                                save(STATE, state); save(CONTROL, control)
                            inject_ref('prepublish')
                        state['release'] = release; save(STATE, state)
                        releases = [state['prior_release']] + ([release] if release else []) + state.get('extra_releases', [])
                        print(json.dumps([releases])); sys.exit(0)
                    if endpoint == f'/repos/{{TARGET}}/releases/tags/v1.0.3':
                        prior_release = json.loads(json.dumps(state['prior_release']))
                        if control.get('malformed_prior_release'):
                            prior_release['assets'][0].pop('digest', None)
                        print(json.dumps(prior_release)); sys.exit(0)
                    print('unknown api endpoint: ' + endpoint, file=sys.stderr); sys.exit(40)

                if len(argv) >= 2 and argv[0] == 'release':
                    command = argv[1]
                    if command == 'view':
                        release = state.get('release')
                        if release is None:
                            sys.exit(83)
                        state['terminal_release_view_calls'] = state.get('terminal_release_view_calls', 0) + 1
                        assets = []
                        for asset in release['assets']:
                            assets.append({{
                                'url': 'https://github.com/' + TARGET + '/releases/download/v1.1.0/' + asset['name'],
                                'apiUrl': 'https://api.github.com/repos/' + TARGET + '/releases/assets/' + str(asset['id']),
                                'id': asset['node_id'], 'name': asset['name'], 'label': '',
                                'size': asset['size'], 'digest': asset['digest'], 'state': 'uploaded',
                                'createdAt': '2026-09-04T00:00:00Z', 'updatedAt': '2026-09-04T00:00:00Z',
                                'downloadCount': 0, 'contentType': 'application/octet-stream',
                            }})
                        projection = {{
                            'apiUrl': 'https://api.github.com/repos/' + TARGET + '/releases/' + str(release['id']),
                            'assets': assets, 'body': release['body'], 'databaseId': release['id'],
                            'id': release['node_id'], 'isDraft': release['draft'],
                            'isImmutable': release['immutable'], 'isPrerelease': release['prerelease'],
                            'name': release['name'], 'tagName': release['tag_name'],
                        }}
                        if state['terminal_release_view_calls'] == 2 and control.get('terminal_live_race') == 'release_title':
                            projection['name'] = 'attacker changed title'
                        save(STATE, state)
                        print(json.dumps(projection))
                        sys.exit(0)
                    if command == 'create':
                        release = {{
                            'id': 601, 'node_id': 'RR_v110', 'tag_name': 'v1.1.0',
                            'name': option('--title'), 'body': option('--notes'), 'draft': True,
                            'prerelease': False, 'immutable': False, 'assets': [],
                        }}
                        state['release'] = release; save(STATE, state)
                        inject_ref('after_draft')
                        if control.get('fail_after_create') and not control.get('create_triggered'):
                            control['create_triggered'] = True; save(CONTROL, control); sys.exit(71)
                        sys.exit(0)
                    if command == 'upload':
                        source = pathlib.Path(argv[3])
                        destination = UPLOADS / source.name
                        shutil.copyfile(source, destination); destination.chmod(0o600)
                        state['upload_calls'] += 1
                        asset = {{'id': state['next_asset_id'], 'node_id': 'RA_' + str(state['next_asset_id']), 'name': source.name, 'size': source.stat().st_size, 'digest': digest(source)}}
                        state['next_asset_id'] += 1
                        state['release']['assets'].append(asset); save(STATE, state)
                        failed_uploads = control.setdefault('upload_failures_triggered', [])
                        if state['upload_calls'] in control.get('fail_upload_numbers', []) and state['upload_calls'] not in failed_uploads:
                            failed_uploads.append(state['upload_calls']); save(CONTROL, control); sys.exit(72)
                        sys.exit(0)
                    if command == 'download':
                        repository = option('--repo'); tag = argv[2]; directory = pathlib.Path(option('--dir')); directory.mkdir(parents=True, exist_ok=True)
                        pattern = option('--pattern') if '--pattern' in argv else None
                        if repository == PRIVATE:
                            state['private_download_calls'] = state.get('private_download_calls', 0) + 1
                            mutate_at = control.get('mutate_private_on_download_number')
                            if mutate_at == state['private_download_calls'] and not control.get('private_asset_mutated'):
                                selected = sorted(path for path in SOURCE.iterdir() if path.is_file())[0]
                                selected.write_bytes(selected.read_bytes() + b'attacker')
                                control['private_asset_mutated'] = True; save(CONTROL, control)
                            paths = [path for path in SOURCE.iterdir() if path.is_file()]
                        elif tag == 'v1.0.3':
                            paths = [path for path in PRIOR.iterdir() if path.name not in ('release-metadata.json', 'release-attestation.json')]
                        else:
                            paths = [UPLOADS / item['name'] for item in state['release']['assets']]
                        for source in paths:
                            if pattern is None or source.name == pattern:
                                shutil.copyfile(source, directory / source.name)
                        save(STATE, state); sys.exit(0)
                    if command == 'edit':
                        if control.get('fail_before_publish_once') and not control.get('before_publish_triggered'):
                            control['before_publish_triggered'] = True; save(CONTROL, control); sys.exit(76)
                        state['release']['draft'] = False
                        state['release']['immutable'] = not control.get('publish_without_immutable', False)
                        save(STATE, state); inject_ref('postpublish'); sys.exit(0)
                    if command == 'verify':
                        if argv[2] == 'v1.0.3':
                            if control.get('fail_prior_verify'):
                                sys.exit(86)
                            print((PRIOR / 'release-attestation.json').read_text(), end='')
                            sys.exit(0)
                        if control.get('fail_release_verify_once') and not control.get('release_verify_triggered'):
                            control['release_verify_triggered'] = True; save(CONTROL, control); sys.exit(77)
                        if not state.get('release', {{}}).get('immutable'):
                            sys.exit(73)
                        if '--format' in argv:
                            if option('--format') != 'json':
                                sys.exit(82)
                            print(json.dumps(release_attestation(), sort_keys=True, separators=(',', ':')))
                        sys.exit(0)
                    if command == 'verify-asset':
                        source = pathlib.Path(argv[3])
                        if not source.is_absolute() or not source.is_file(): sys.exit(74)
                        expected = UPLOADS / source.name
                        if not expected.is_file() or digest(source) != digest(expected): sys.exit(75)
                        state['verify_asset_calls'] = state.get('verify_asset_calls', 0) + 1
                        failed = control.setdefault('verify_asset_failures_triggered', [])
                        if state['verify_asset_calls'] in control.get('fail_verify_asset_numbers', []) and state['verify_asset_calls'] not in failed:
                            failed.append(state['verify_asset_calls']); save(STATE, state); save(CONTROL, control); sys.exit(78)
                        failed_names = control.setdefault('verify_asset_names_triggered', [])
                        if source.name in control.get('fail_verify_asset_names', []) and source.name not in failed_names:
                            failed_names.append(source.name); save(STATE, state); save(CONTROL, control); sys.exit(79)
                        log = state.setdefault('verified_paths', [])
                        log.append(str(source)); save(STATE, state); sys.exit(0)
                print('unsupported gh invocation: ' + repr(argv), file=sys.stderr)
                sys.exit(41)
                """
            ),
            encoding="utf-8",
        )
        stub.chmod(0o755)
        return stub, state_path, control_path, upload_root

    def publisher_github_stub_command(
        self,
        state: pathlib.Path,
        target: pathlib.Path,
        gh_stub: pathlib.Path,
        *extra: str,
        asset_directory: pathlib.Path | None = None,
    ) -> list[str]:
        command = self.publisher_command(state, target)
        local_index = command.index("--local-bare-remote")
        del command[local_index:local_index + 2]
        control_path = gh_stub.parent / "control.json"
        control = json.loads(control_path.read_text())
        control["target_path"] = str(target)
        self.write_json(control_path, control)
        command[0] = str(self.make_instrumented_publisher(target, gh_stub))
        command.extend(
            [
                "--gh-binary", str(gh_stub),
                "--gh-sha256", MIRROR._sha256_file(gh_stub),
                *extra,
            ]
        )
        if asset_directory is not None:
            self.replace_option(command, "--asset-directory", asset_directory)
            self.replace_option(command, "--external-audit", asset_directory / self.external_audit.name)
        return command

    def make_instrumented_publisher(self, target: pathlib.Path, gh_stub: pathlib.Path) -> pathlib.Path:
        harness = self.root / f"instrumented-controller-{target.name}"
        publisher = harness / PUBLISHER.name
        if publisher.exists():
            return publisher
        harness.mkdir(mode=0o700)
        terminal_tool = harness / "test-only-terminal-tool.py"
        tool_text = TOOL_PATH.read_text(encoding="utf-8")
        tool_text = tool_text.replace(
            'PUBLICATION_TERMINAL_SCHEMA = "gpmeep-v110-release-only-publication-terminal-v2"',
            'PUBLICATION_TERMINAL_SCHEMA = "gpmeep-v110-test-only-controller-terminal-v1"',
        ).replace(
            '"ORGANIZATION_RELEASE_PUBLISHED_IMMUTABLE"',
            '"TEST_ONLY_STUB_FLOW_COMPLETED"',
        ).replace(
            'AUDITED_RECEIPT_TOOL_PATH = pathlib.Path(__file__)',
            f'AUDITED_RECEIPT_TOOL_PATH = pathlib.Path({str(TOOL_PATH)!r})',
        ).replace(
            'APPROVED_GITHUB_CLI_SHA256 = AUDITED_GITHUB_CLI_SHA256',
            f'APPROVED_GITHUB_CLI_SHA256 = {MIRROR._sha256_file(gh_stub)!r}',
        )
        self.assertNotIn("ORGANIZATION_RELEASE_PUBLISHED_IMMUTABLE", tool_text)
        terminal_tool.write_text(tool_text, encoding="utf-8")
        terminal_tool.chmod(0o700)

        publisher_text = PUBLISHER.read_text(encoding="utf-8")
        replacements = {
            'PUBLISHER_FILE=$(canonical_file "$0")': f'PUBLISHER_FILE=$(canonical_file "{PUBLISHER}")',
            'AUDITED_TOOL=$(canonical_file "$SCRIPT_DIRECTORY/v110-release-only-receipt.py")': f'AUDITED_TOOL=$(canonical_file "{TOOL_PATH}")',
            'TOOL=$AUDITED_TOOL': f'TOOL=$(canonical_file "{terminal_tool}")',
            'MIRROR_TEST=$(canonical_file "$SCRIPT_DIRECTORY/tests/test_v110_release_only_mirror.py")': f'MIRROR_TEST=$(canonical_file "{pathlib.Path(__file__).resolve()}")',
            'ACTIVE_REMOTE="git@github.com:${REPOSITORY}.git"': f'ACTIVE_REMOTE="{target}"',
            'REMOTE_PROTOCOL_ALLOW=never': 'REMOTE_PROTOCOL_ALLOW=always',
            'PRIVATE_REMOTE_URL="git@github.com:${PRIVATE_REPOSITORY}.git"': f'PRIVATE_REMOTE_URL="{self.source}"',
            'APPROVED_GH_SHA256=$AUDITED_GH_SHA256': f'APPROVED_GH_SHA256={MIRROR._sha256_file(gh_stub)}',
        }
        for old, new in replacements.items():
            self.assertIn(old, publisher_text)
            publisher_text = publisher_text.replace(old, new)
        self.assertNotIn("--test-github-bare-remote", publisher_text)
        publisher.write_text(publisher_text, encoding="utf-8")
        publisher.chmod(0o700)
        return publisher

    @staticmethod
    def github_stub_environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({"GH_TOKEN": "stub-token"})
        return environment

    def fresh_release_assets(self, name: str) -> pathlib.Path:
        assets = self.root / name
        assets.mkdir(mode=0o700)
        for source in (self.package, self.candidate, self.final, self.source_manifest, self.external_audit):
            shutil.copyfile(source, assets / source.name)
            (assets / source.name).chmod(0o600)
        return assets

    def write_empty_external_audit(self, path: pathlib.Path) -> tuple[pathlib.Path, str]:
        value = json.loads(self.external_audit.read_text())
        value.pop("audit_id")
        value["target_history"]["advertised_refs_sha256"] = MIRROR._canonical_sha256([])
        value["target_history"]["prior_releases_sha256"] = MIRROR._canonical_sha256([])
        value["audit_id"] = MIRROR._canonical_sha256(value)
        self.write_json(path, value)
        path.chmod(0o600)
        return path, MIRROR._sha256_file(path)

    @staticmethod
    def replace_option(command: list[str], option: str, value: pathlib.Path | str) -> None:
        command[command.index(option) + 1] = str(value)

    def verify_source_authority(self) -> dict[str, object]:
        return MIRROR.verify_source_authority_files(
            self.source_authority,
            self.assets,
            source_repo=self.source,
            final_verifier_path=self.final_verifier,
            final_verifier_sha256=self.final_verifier_sha256,
            final_verifier_arguments=self.final_verifier_args,
            trusted_source_tag_object=self.source_tag_object,
        )

    def test_parentless_identical_tree_is_deterministic_and_secret_is_absent(self) -> None:
        first_state, first_snapshot, first_receipt, first_tag = self.make_state("first")
        second_state, second_snapshot, second_receipt, second_tag = self.make_state("second")
        self.assertEqual(first_snapshot["snapshot_commit"], second_snapshot["snapshot_commit"])
        self.assertEqual(first_snapshot["snapshot_id"], second_snapshot["snapshot_id"])
        self.assertEqual(first_receipt, second_receipt)
        self.assertEqual(first_tag["tag_object"], second_tag["tag_object"])
        self.assertEqual(first_tag["attestation_id"], second_tag["attestation_id"])
        publisher = first_state / "isolated-publisher.git"
        self.assertEqual(oid(publisher, "refs/tags/v1.1.0^{}^{tree}"), self.source_tree)
        self.assertEqual(git("rev-list", "--count", "--all", cwd=publisher).stdout.strip(), b"1")
        self.assertNotEqual(git("cat-file", "-e", self.secret_commit, cwd=publisher, check=False).returncode, 0)
        self.assertNotEqual(git("cat-file", "-e", self.secret_blob, cwd=publisher, check=False).returncode, 0)
        commit_payload = git("cat-file", "commit", first_snapshot["snapshot_commit"], cwd=publisher).stdout
        self.assertNotIn(b"\nparent ", b"\n" + commit_payload.split(b"\n\n", 1)[0] + b"\n")
        self.assertEqual(first_snapshot["tree_manifest"], MIRROR.tree_manifest(publisher, self.source_tree))
        self.assertEqual(stat.S_IMODE(first_state.stat().st_mode), 0o700)
        for path in first_state.rglob("*"):
            self.assertFalse(path.is_symlink())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)

    def test_local_bare_dry_run_has_zero_remote_mutation_then_exact_resume(self) -> None:
        target = self.root / "target.git"
        git("init", "--bare", "--template=", str(target))
        self.seed_prior_target(target)
        state = self.root / "publisher-state"
        state.mkdir(mode=0o700)
        before = git("show-ref", cwd=target, check=False).stdout
        before_objects = sorted(path.relative_to(target).as_posix() for path in (target / "objects").rglob("*") if path.is_file())
        # A crash before terminal directory publication leaves an ignored,
        # non-authoritative attempt.  Exact retry must not delete or trust it.
        abandoned = state / ".snapshot-attempt.abandoned"
        abandoned.mkdir(mode=0o700)
        (abandoned / "partial").write_bytes(b"incomplete")
        hostile_bin = self.root / "hostile-bin"
        hostile_bin.mkdir()
        hostile_marker = self.root / "hostile-path-was-used"
        for command_name in ("git", "gh", "python3", "jq"):
            command = hostile_bin / command_name
            command.write_text(f"#!/bin/sh\n/usr/bin/touch '{hostile_marker}'\nexit 99\n")
            command.chmod(0o755)
        bash_env = self.root / "hostile-bash-env"
        bash_env.write_text(f"/usr/bin/touch '{hostile_marker}'\n")
        hostile_environment = os.environ.copy()
        hostile_environment.update(
            {
                "PATH": str(hostile_bin),
                "BASH_ENV": str(bash_env),
                "ENV": str(bash_env),
                "PYTHONPATH": str(self.source),
                "GIT_CONFIG_GLOBAL": str(self.root / "attacker.gitconfig"),
            }
        )
        dry = self.run_publisher(self.publisher_command(state, target), env=hostile_environment)
        self.assertIn(b"preflight PASS", dry.stdout)
        self.assertFalse(hostile_marker.exists())
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, before)
        self.assertEqual(
            sorted(path.relative_to(target).as_posix() for path in (target / "objects").rglob("*") if path.is_file()),
            before_objects,
        )
        self.assertEqual((abandoned / "partial").read_bytes(), b"incomplete")
        execute = self.run_publisher(self.publisher_command(state, target, "--execute", "--resume"))
        self.assertIn(b"tag publication PASS", execute.stdout)
        target_tag = oid(target, "refs/tags/v1.1.0")
        expected_tag = json.loads((state / "sealed-state/tag-attestation.json").read_text())["tag_object"]
        self.assertEqual(target_tag, expected_tag)
        self.assertEqual(oid(target, "refs/tags/v1.1.0^{}^{tree}"), self.source_tree)
        self.assertEqual(git("rev-list", "--all", "--count", cwd=target).stdout.strip(), b"2")
        self.assertNotEqual(git("cat-file", "-e", self.secret_commit, cwd=target, check=False).returncode, 0)
        self.assertNotEqual(git("cat-file", "-e", self.secret_blob, cwd=target, check=False).returncode, 0)
        clone = self.root / "all-refs.git"
        git("clone", "--mirror", str(target), str(clone))
        self.assertNotEqual(git("cat-file", "-e", self.secret_blob, cwd=clone, check=False).returncode, 0)
        resumed = self.run_publisher(self.publisher_command(state, target, "--execute", "--resume"))
        self.assertIn(b"tag publication PASS", resumed.stdout)
        normal_again = self.run_publisher(self.publisher_command(state, target, "--execute"), check=False)
        self.assertNotEqual(normal_again.returncode, 0)
        self.assertIn(b"use --execute --resume", normal_again.stderr)

    def test_production_controller_stub_publishes_and_replays_durable_terminal(self) -> None:
        target = self.root / "stub-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "stub-controller-state"
        state.mkdir(mode=0o700)
        assets = self.fresh_release_assets("stub-release-assets")
        gh_stub, stub_state_path, _control_path, _uploads = self.make_github_stub("successful-stub")
        environment = self.github_stub_environment()
        command = self.publisher_github_stub_command(
            state, target, gh_stub, asset_directory=assets
        )
        dry = self.run_publisher(command, env=environment)
        self.assertIn(b"preflight PASS", dry.stdout)
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertIsNone(json.loads(stub_state_path.read_text())["release"])
        execute = self.run_publisher(
            self.publisher_github_stub_command(
                state, target, gh_stub, "--execute", "--resume", asset_directory=assets
            ),
            env=environment,
        )
        self.assertIn(b"release-only mirror publication PASS", execute.stdout)
        terminal_path = state / "publication-terminal.json"
        terminal = json.loads(terminal_path.read_text())
        terminal_unsigned = dict(terminal)
        terminal_id = terminal_unsigned.pop("terminal_id")
        self.assertEqual(terminal["schema"], "gpmeep-v110-test-only-controller-terminal-v1")
        self.assertEqual(terminal["state"], "TEST_ONLY_STUB_FLOW_COMPLETED")
        self.assertEqual(terminal_id, MIRROR._canonical_sha256(terminal_unsigned))
        self.assertEqual(stat.S_IMODE(terminal_path.stat().st_mode), 0o600)
        self.assertEqual(terminal["tag"]["object_id"], oid(target, "refs/tags/v1.1.0"))
        self.assertEqual(len(terminal["assets"]), 8)
        stub_state = json.loads(stub_state_path.read_text())
        self.assertFalse(stub_state["release"]["draft"])
        self.assertFalse(stub_state["release"]["prerelease"])
        self.assertTrue(stub_state["release"]["immutable"])
        self.assertGreaterEqual(stub_state["private_download_calls"], 5)
        self.assertEqual(len(stub_state["verified_paths"]), 8)
        self.assertTrue(all(pathlib.Path(path).is_absolute() for path in stub_state["verified_paths"]))
        resumed = self.run_publisher(
            self.publisher_github_stub_command(
                state, target, gh_stub, "--execute", "--resume", asset_directory=assets
            ),
            env=environment,
        )
        self.assertIn(terminal["terminal_id"].encode(), resumed.stdout)
        self.assertEqual(json.loads(terminal_path.read_text()), terminal)

    def test_v110_controller_rejects_any_unrelated_existing_release(self) -> None:
        target = self.root / "unrelated-release-target.git"
        git("init", "--bare", "--template=", str(target))
        state_root = self.root / "unrelated-release-state"
        state_root.mkdir(mode=0o700)
        assets = self.fresh_release_assets("unrelated-release-assets")
        gh_stub, stub_state_path, _control_path, _uploads = self.make_github_stub(
            "unrelated-release-stub"
        )
        remote_state = json.loads(stub_state_path.read_text())
        remote_state["extra_releases"] = [{
            "id": 602,
            "node_id": "RR_unrelated",
            "tag_name": "v0.0.0",
            "name": "unrelated",
            "body": "must not be ignored",
            "draft": True,
            "prerelease": False,
            "immutable": False,
            "assets": [],
        }]
        self.write_json(stub_state_path, remote_state)
        result = self.run_publisher(
            self.publisher_github_stub_command(
                state_root, target, gh_stub, "--execute", asset_directory=assets
            ),
            check=False,
            env=self.github_stub_environment(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"target release state cannot be classified", result.stderr)
        after = json.loads(stub_state_path.read_text())
        self.assertIsNone(after["release"])
        self.assertEqual(after["upload_calls"], 0)
        self.assertEqual(after["extra_releases"], remote_state["extra_releases"])
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertFalse((state_root / "publication-terminal.json").exists())

    def test_v110_controller_and_receipt_reject_valid_empty_bootstrap(self) -> None:
        target = self.root / "empty-bootstrap-target.git"
        git("init", "--bare", "--template=", str(target))
        state_root = self.root / "empty-bootstrap-state"
        state_root.mkdir(mode=0o700)
        assets = self.fresh_release_assets("empty-bootstrap-assets")

        empty_audit = json.loads(self.external_audit.read_text())
        empty_audit.pop("audit_id")
        empty_audit["target_history"]["advertised_refs_sha256"] = MIRROR._canonical_sha256([])
        empty_audit["target_history"]["prior_releases_sha256"] = MIRROR._canonical_sha256([])
        empty_audit["audit_id"] = MIRROR._canonical_sha256(empty_audit)
        empty_audit_path = assets / self.external_audit.name
        self.write_json(empty_audit_path, empty_audit)
        empty_audit_path.chmod(0o600)
        empty_audit_record = MIRROR.external_audit_record(
            empty_audit_path, MIRROR._sha256_file(empty_audit_path)
        )
        empty_history = {
            "schema": MIRROR.TARGET_HISTORY_SCHEMA,
            "state": "HISTORY_CONFIDENTIALITY_ATTESTED",
            "mode": "empty-new-repository",
            "never_imported_or_transferred": True,
            "repository": self.target_history_attestation["repository"],
            "advertised_refs": [],
            "prior_releases": [],
            "external_audit": {
                key: empty_audit_record[key]
                for key in ("name", "size_bytes", "sha256", "audit_id")
            },
        }
        empty_history["attestation_id"] = MIRROR._canonical_sha256(empty_history)
        empty_history_path = self.root / "empty-bootstrap-history.json"
        self.write_json(empty_history_path, empty_history)
        empty_preflight = json.loads(json.dumps(self.target_preflight))
        empty_preflight["advertised_refs"] = []
        empty_preflight["bootstrap"] = {
            "mode": "empty-new-repository",
            "attestation_sha256": MIRROR._sha256_file(empty_history_path),
            "attestation_id": empty_history["attestation_id"],
        }
        empty_preflight["external_mirror_audit"] = empty_audit_record
        empty_preflight_path = self.root / "empty-bootstrap-preflight.json"
        self.write_json(empty_preflight_path, empty_preflight)
        self.assertEqual(
            MIRROR.validate_target_history_attestation(empty_history)["mode"],
            "empty-new-repository",
        )
        self.assertEqual(
            MIRROR.validate_target_preflight(empty_preflight)["bootstrap"]["mode"],
            "empty-new-repository",
        )
        snapshot_state = self.root / "empty-bootstrap-snapshot"
        snapshot_state.mkdir(mode=0o700)
        snapshot = MIRROR.prepare_snapshot(
            source_repo=self.source,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            state_root=snapshot_state,
            identity=self.identity,
            source_tag="v1.1.0",
            source_tag_object=self.source_tag_object,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "audited v1.0.3 prior tag"):
            MIRROR.create_receipt(self.source_authority, snapshot, empty_preflight)

        gh_stub, stub_state_path, _control_path, _uploads = self.make_github_stub(
            "empty-bootstrap-stub"
        )
        command = self.publisher_github_stub_command(
            state_root, target, gh_stub, "--execute", asset_directory=assets
        )
        self.replace_option(command, "--target-preflight", empty_preflight_path)
        self.replace_option(command, "--target-history-attestation", empty_history_path)
        self.replace_option(command, "--external-audit", empty_audit_path)
        command[command.index("--external-audit-sha256") + 1] = MIRROR._sha256_file(empty_audit_path)
        result = self.run_publisher(
            command, check=False, env=self.github_stub_environment()
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"requires the independently audited v1.0.3 prior release", result.stderr)
        after = json.loads(stub_state_path.read_text())
        self.assertIsNone(after["release"])
        self.assertEqual(after["upload_calls"], 0)
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertFalse((state_root / "run-intent.json").exists())

    def test_v110_controller_rejects_bad_live_prior_capture_before_mutation(self) -> None:
        scenarios = {
            "malformed-release": {"malformed_prior_release": True},
            "missing-signed-attestation": {"fail_prior_verify": True},
        }
        for name, controls in scenarios.items():
            with self.subTest(name=name):
                target = self.root / f"bad-prior-{name}-target.git"
                git("init", "--bare", "--template=", str(target))
                state_root = self.root / f"bad-prior-{name}-state"
                state_root.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"bad-prior-{name}-assets")
                gh_stub, stub_state_path, control_path, _uploads = self.make_github_stub(
                    f"bad-prior-{name}-stub"
                )
                self.write_json(control_path, controls)
                result = self.run_publisher(
                    self.publisher_github_stub_command(
                        state_root, target, gh_stub, "--execute", asset_directory=assets
                    ),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                after = json.loads(stub_state_path.read_text())
                self.assertIsNone(after["release"])
                self.assertEqual(after["upload_calls"], 0)
                self.assertEqual(
                    git("show-ref", cwd=target, check=False).stdout,
                    self.prior_ref_state(),
                )
                self.assertFalse((state_root / "publication-terminal.json").exists())

    def test_production_stub_resumes_after_draft_and_every_asset_upload_interruption(self) -> None:
        target = self.root / "interrupted-stub-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "interrupted-stub-state"
        state.mkdir(mode=0o700)
        assets = self.fresh_release_assets("interrupted-stub-assets")
        gh_stub, stub_state_path, control_path, _uploads = self.make_github_stub("interrupted-stub")
        self.write_json(
            control_path,
            {"fail_after_create": True, "fail_upload_numbers": list(range(1, 9))},
        )
        environment = self.github_stub_environment()
        first = self.run_publisher(
            self.publisher_github_stub_command(
                state, target, gh_stub, "--execute", asset_directory=assets
            ),
            check=False,
            env=environment,
        )
        self.assertNotEqual(first.returncode, 0)
        interrupted = json.loads(stub_state_path.read_text())
        self.assertTrue(interrupted["release"]["draft"])
        self.assertEqual(interrupted["upload_calls"], 0)
        self.assertEqual(git("show-ref", "--verify", "refs/tags/v1.1.0", cwd=target).returncode, 0)

        resume_command = lambda: self.publisher_github_stub_command(
            state, target, gh_stub, "--execute", "--resume", asset_directory=assets
        )
        for upload_number in range(1, 9):
            attempt = self.run_publisher(resume_command(), check=False, env=environment)
            self.assertNotEqual(attempt.returncode, 0, f"upload {upload_number} did not interrupt")
            interrupted = json.loads(stub_state_path.read_text())
            self.assertEqual(interrupted["upload_calls"], upload_number)
            self.assertTrue(interrupted["release"]["draft"])
            self.assertFalse(interrupted["release"]["immutable"])
            self.assertFalse((state / "publication-terminal.json").exists())

        completed = self.run_publisher(resume_command(), env=environment)
        self.assertIn(b"publication PASS", completed.stdout)
        final = json.loads(stub_state_path.read_text())
        self.assertEqual(final["upload_calls"], 8)
        self.assertFalse(final["release"]["draft"])
        self.assertTrue(final["release"]["immutable"])
        self.assertTrue((state / "publication-terminal.json").is_file())

    def test_instrumented_controller_rejects_private_asset_race_before_push(self) -> None:
        target = self.root / "private-asset-race-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "private-asset-race-state"
        state.mkdir(mode=0o700)
        assets = self.fresh_release_assets("private-asset-race-assets")
        gh_stub, _stub_state, control, _uploads = self.make_github_stub("private-asset-race-stub")
        self.write_json(control, {"mutate_private_on_download_number": 2})
        result = self.run_publisher(
            self.publisher_github_stub_command(state, target, gh_stub, "--execute", asset_directory=assets),
            check=False,
            env=self.github_stub_environment(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"private source authority changed before tag push", result.stderr)
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertFalse((state / "publication-terminal.json").exists())

    def test_instrumented_controller_exactly_resumes_publish_and_signature_failures(self) -> None:
        scenarios = {
            "before-publish": {"fail_before_publish_once": True},
            "after-immutable": {"fail_release_verify_once": True},
        }
        for name, controls in scenarios.items():
            with self.subTest(name=name):
                target = self.root / f"{name}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"{name}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"{name}-assets")
                gh_stub, stub_state, control, _uploads = self.make_github_stub(f"{name}-stub")
                self.write_json(control, controls)
                command = lambda: self.publisher_github_stub_command(
                    state, target, gh_stub, "--execute", "--resume", asset_directory=assets
                )
                first = self.run_publisher(command(), check=False, env=self.github_stub_environment())
                self.assertNotEqual(first.returncode, 0)
                self.assertFalse((state / "publication-terminal.json").exists())
                completed = self.run_publisher(command(), env=self.github_stub_environment())
                self.assertIn(b"publication PASS", completed.stdout)
                self.assertTrue((state / "publication-terminal.json").is_file())
                self.assertTrue(json.loads(stub_state.read_text())["release"]["immutable"])

        target = self.root / "asset-signature-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "asset-signature-state"
        state.mkdir(mode=0o700)
        assets = self.fresh_release_assets("asset-signature-assets")
        gh_stub, stub_state, control, _uploads = self.make_github_stub("asset-signature-stub")
        expected_names = sorted(
            [
                self.package.name,
                self.candidate.name,
                self.final.name,
                self.source_manifest.name,
                self.external_audit.name,
                "gpmeep-v1.1.0-release-only-receipt.json",
                "gpmeep-v1.1.0-target-history-attestation.json",
                MIRROR.DISTRIBUTION_MANIFEST_NAME,
            ]
        )
        self.write_json(control, {"fail_verify_asset_names": expected_names})
        command = lambda: self.publisher_github_stub_command(
            state, target, gh_stub, "--execute", "--resume", asset_directory=assets
        )
        for name in expected_names:
            attempt = self.run_publisher(command(), check=False, env=self.github_stub_environment())
            self.assertNotEqual(attempt.returncode, 0, name)
            self.assertFalse((state / "publication-terminal.json").exists())
        completed = self.run_publisher(command(), env=self.github_stub_environment())
        self.assertIn(b"publication PASS", completed.stdout)
        self.assertEqual(
            sorted(json.loads(control.read_text())["verify_asset_names_triggered"]), expected_names
        )
        self.assertTrue(json.loads(stub_state.read_text())["release"]["immutable"])

    def test_instrumented_controller_rejects_ref_and_release_identity_races(self) -> None:
        controls = [
            (f"ref-{stage}", {"inject_ref_stage": stage})
            for stage in ("after_tag", "after_draft", "prepublish", "postpublish")
        ]
        controls.extend(
            (f"published-{race}", {"postpublish_release_race": race})
            for race in ("id", "node", "tag", "immutable_false")
        )
        controls.extend(
            (f"final-{race}", {"final_release_race": race}) for race in ("id", "node")
        )
        for name, control_value in controls:
            with self.subTest(name=name):
                target = self.root / f"{name}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"{name}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"{name}-assets")
                gh_stub, _stub_state, control, _uploads = self.make_github_stub(f"{name}-stub")
                self.write_json(control, {**control_value, "target_path": str(target)})
                result = self.run_publisher(
                    self.publisher_github_stub_command(state, target, gh_stub, "--execute", asset_directory=assets),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((state / "publication-terminal.json").exists())

    def test_instrumented_controller_rejects_terminal_live_metadata_races(self) -> None:
        for race in ("repository_archived", "release_title"):
            with self.subTest(race=race):
                target = self.root / f"terminal-{race}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"terminal-{race}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"terminal-{race}-assets")
                gh_stub, _stub_state, control, _uploads = self.make_github_stub(
                    f"terminal-{race}-stub"
                )
                self.write_json(control, {"terminal_live_race": race})
                result = self.run_publisher(
                    self.publisher_github_stub_command(
                        state, target, gh_stub, "--execute", asset_directory=assets
                    ),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((state / "publication-terminal.json").exists())

    def test_instrumented_controller_rejects_signed_attestation_substitutions(self) -> None:
        for race in ("tag_object", "asset_digest", "release_id", "payload_mismatch", "certificate_san"):
            with self.subTest(race=race):
                target = self.root / f"attestation-{race}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"attestation-{race}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"attestation-{race}-assets")
                gh_stub, _stub_state, control, _uploads = self.make_github_stub(
                    f"attestation-{race}-stub"
                )
                self.write_json(control, {"attestation_race": race})
                result = self.run_publisher(
                    self.publisher_github_stub_command(
                        state, target, gh_stub, "--execute", asset_directory=assets
                    ),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((state / "publication-terminal.json").exists())

    def test_production_stub_rejects_missing_capability_before_any_mutation(self) -> None:
        target = self.root / "incapable-stub-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "incapable-stub-state"
        state.mkdir(mode=0o700)
        assets = self.fresh_release_assets("incapable-stub-assets")
        gh_stub, stub_state_path, control_path, _uploads = self.make_github_stub("incapable-stub")
        self.write_json(control_path, {"missing_capability": "verify-asset"})
        result = self.run_publisher(
            self.publisher_github_stub_command(
                state, target, gh_stub, "--execute", asset_directory=assets
            ),
            check=False,
            env=self.github_stub_environment(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"lacks required release capabilities", result.stderr)
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertIsNone(json.loads(stub_state_path.read_text())["release"])
        self.assertFalse((state / "run-intent.json").exists())

    def test_production_stub_rejects_prepublish_metadata_and_asset_races(self) -> None:
        for race in ("prerelease", "title", "extra"):
            with self.subTest(race=race):
                target = self.root / f"prepublish-{race}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"prepublish-{race}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"prepublish-{race}-assets")
                gh_stub, stub_state_path, control_path, _uploads = self.make_github_stub(
                    f"prepublish-{race}-stub"
                )
                self.write_json(control_path, {"prepublish_race": race})
                result = self.run_publisher(
                    self.publisher_github_stub_command(
                        state, target, gh_stub, "--execute", asset_directory=assets
                    ),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"prepublication checkpoint", result.stderr)
                release = json.loads(stub_state_path.read_text())["release"]
                self.assertTrue(release["draft"])
                self.assertFalse(release["immutable"])
                self.assertFalse((state / "publication-terminal.json").exists())

    def test_production_stub_rejects_postpublish_target_and_source_races_without_terminal(self) -> None:
        for race in ("repo_archived", "source_prerelease"):
            with self.subTest(race=race):
                target = self.root / f"postpublish-{race}-target.git"
                git("init", "--bare", "--template=", str(target))
                state = self.root / f"postpublish-{race}-state"
                state.mkdir(mode=0o700)
                assets = self.fresh_release_assets(f"postpublish-{race}-assets")
                gh_stub, stub_state_path, control_path, _uploads = self.make_github_stub(
                    f"postpublish-{race}-stub"
                )
                self.write_json(control_path, {"postpublish_race": race})
                result = self.run_publisher(
                    self.publisher_github_stub_command(
                        state, target, gh_stub, "--execute", asset_directory=assets
                    ),
                    check=False,
                    env=self.github_stub_environment(),
                )
                self.assertNotEqual(result.returncode, 0)
                release = json.loads(stub_state_path.read_text())["release"]
                self.assertFalse(release["draft"])
                self.assertTrue(release["immutable"])
                self.assertFalse((state / "publication-terminal.json").exists())

    def test_resume_rejects_different_tag_and_never_rewrites_it(self) -> None:
        target = self.root / "different-target.git"
        git("init", "--bare", "--template=", str(target))
        other_blob = git("--git-dir", str(target), "hash-object", "-w", "--stdin", input_bytes=b"different\n").stdout.decode().strip()
        tree_line = f"100644 blob {other_blob}\tdifferent\n".encode()
        other_tree = git("--git-dir", str(target), "mktree", input_bytes=tree_line).stdout.decode().strip()
        other_commit = git("--git-dir", str(target), "commit-tree", other_tree, input_bytes=b"different root\n").stdout.decode().strip()
        tag_payload = (
            f"object {other_commit}\ntype commit\ntag v1.1.0\ntagger Fixture <fixture@example.invalid> 1700000000 +0000\n\ndifferent\n"
        ).encode()
        other_tag = git("--git-dir", str(target), "mktag", input_bytes=tag_payload).stdout.decode().strip()
        git("--git-dir", str(target), "update-ref", "refs/tags/v1.1.0", other_tag, "0" * 40)
        state = self.root / "different-state"
        state.mkdir(mode=0o700)
        result = self.run_publisher(self.publisher_command(state, target, "--execute", "--resume"), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"full ref union differs", result.stderr)
        self.assertEqual(oid(target, "refs/tags/v1.1.0"), other_tag)

    def test_prior_target_with_hidden_object_is_rejected_without_mutation(self) -> None:
        target = self.root / "contaminated-empty-target.git"
        git("init", "--bare", "--template=", str(target))
        hidden = git(
            "--git-dir", str(target), "hash-object", "-w", "--stdin",
            input_bytes=b"unadvertised prior development object\n",
        ).stdout.decode().strip()
        state = self.root / "contaminated-state"
        state.mkdir(mode=0o700)
        result = self.run_publisher(self.publisher_command(state, target, "--execute", "--resume"), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"prior release repository has unreachable or corrupt objects", result.stderr)
        self.assertEqual(git("show-ref", cwd=target, check=False).stdout, self.prior_ref_state())
        self.assertEqual(git("cat-file", "-e", hidden, cwd=target, check=False).returncode, 0)

    def test_unexpected_target_ref_is_rejected_without_rewrite(self) -> None:
        target = self.root / "unexpected-ref-target.git"
        git("init", "--bare", "--template=", str(target))
        blob = git("--git-dir", str(target), "hash-object", "-w", "--stdin", input_bytes=b"unexpected\n").stdout.decode().strip()
        tree = git("--git-dir", str(target), "mktree", input_bytes=f"100644 blob {blob}\tfile\n".encode()).stdout.decode().strip()
        commit = git("--git-dir", str(target), "commit-tree", tree, input_bytes=b"unexpected\n").stdout.decode().strip()
        git("--git-dir", str(target), "update-ref", "refs/heads/main", commit, "0" * 40)
        state = self.root / "unexpected-ref-state"
        state.mkdir(mode=0o700)
        result = self.run_publisher(self.publisher_command(state, target, "--execute", "--resume"), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(git("show-ref", "--verify", "refs/heads/main", cwd=target, check=False).returncode, 0)
        self.assertNotEqual(git("show-ref", "--verify", "refs/tags/v1.1.0", cwd=target, check=False).returncode, 0)

    def test_tree_policy_rejects_lfs_and_submodule(self) -> None:
        bare = self.root / "policy.git"
        git("init", "--bare", str(bare))
        lfs_blob = git(
            "--git-dir", str(bare), "hash-object", "-w", "--stdin",
            input_bytes=b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1\n",
        ).stdout.decode().strip()
        lfs_tree = git(
            "--git-dir", str(bare), "mktree", input_bytes=f"100644 blob {lfs_blob}\tpointer\n".encode()
        ).stdout.decode().strip()
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "LFS"):
            MIRROR.tree_manifest(bare, lfs_tree)
        submodule_tree = git(
            "--git-dir", str(bare), "mktree", input_bytes=f"160000 commit {self.source_commit}\tsubmodule\n".encode()
        ).stdout.decode().strip()
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "submodules"):
            MIRROR.tree_manifest(bare, submodule_tree)

    def test_isolation_rejects_alternate_promisor_shallow_graft_replace_and_extra_ref(self) -> None:
        state, snapshot, _receipt, tag = self.make_state("tamper-base")
        original = state / "isolated-publisher.git"
        mutations = {
            "alternate": lambda repo: (repo / "objects/info/alternates").write_text(str(self.source / ".git/objects") + "\n"),
            "http-alternate": lambda repo: (repo / "objects/info/http-alternates").write_text("https://attacker.invalid/objects\n"),
            "promisor": lambda repo: git("--git-dir", str(repo), "config", "remote.bad.promisor", "true"),
            "url-instead-of": lambda repo: git("--git-dir", str(repo), "config", "url.ssh://attacker.invalid/.insteadOf", "git@github.com:"),
            "include": lambda repo: git("--git-dir", str(repo), "config", "include.path", str(self.root / "evil.gitconfig")),
            "ssh-command": lambda repo: git("--git-dir", str(repo), "config", "core.sshCommand", "false"),
            "protocol": lambda repo: git("--git-dir", str(repo), "config", "protocol.file.allow", "always"),
            "remote": lambda repo: git("--git-dir", str(repo), "config", "remote.evil.url", str(self.source)),
            "shallow": lambda repo: (repo / "shallow").write_text(snapshot["snapshot_commit"] + "\n"),
            "graft": lambda repo: ((repo / "info").mkdir(exist_ok=True), (repo / "info/grafts").write_text(snapshot["snapshot_commit"] + "\n")),
            "replace": lambda repo: git("--git-dir", str(repo), "update-ref", f"refs/replace/{snapshot['snapshot_commit']}", snapshot["snapshot_commit"]),
            "extra ref": lambda repo: git("--git-dir", str(repo), "update-ref", "refs/heads/unexpected", snapshot["snapshot_commit"]),
            "hook": lambda repo: ((repo / "hooks").mkdir(exist_ok=True), (repo / "hooks/pre-push").write_text("#!/bin/sh\nexit 0\n")),
            "index": lambda repo: (repo / "index").write_bytes(b"attacker index"),
            "fetch-head": lambda repo: (repo / "FETCH_HEAD").write_text(snapshot["snapshot_commit"] + "\n"),
            "merge-head": lambda repo: (repo / "MERGE_HEAD").write_text(snapshot["snapshot_commit"] + "\n"),
            "bisect-log": lambda repo: (repo / "BISECT_LOG").write_text("attacker state\n"),
            "rebase-apply": lambda repo: (repo / "rebase-apply").mkdir(),
            "rebase-merge": lambda repo: (repo / "rebase-merge").mkdir(),
            "sequencer": lambda repo: (repo / "sequencer").mkdir(),
            "nested symlink": lambda repo: (
                (repo / "info").mkdir(exist_ok=True),
                (repo / "info/evil-link").symlink_to("/etc/passwd"),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                copied = self.root / ("tampered-" + label.replace(" ", "-"))
                shutil.copytree(original, copied)
                mutate(copied)
                with self.assertRaises(MIRROR.ReleaseOnlyError):
                    MIRROR.verify_isolated_repository(
                        copied,
                        snapshot_commit=snapshot["snapshot_commit"],
                        source_tree=self.source_tree,
                        source_commit=self.source_commit,
                        tag_name="v1.1.0",
                        tag_object=tag["tag_object"],
                        forbidden_objects=[self.secret_blob],
                    )

        detached = self.root / "tampered-detached-head"
        shutil.copytree(original, detached)
        attacker_commit = git(
            "--git-dir", str(detached), "commit-tree", self.source_tree,
            input_bytes=b"attacker hidden HEAD\n",
        ).stdout.decode().strip()
        git("--git-dir", str(detached), "update-ref", "--no-deref", "HEAD", attacker_commit)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "HEAD differs"):
            MIRROR.verify_isolated_repository(
                detached,
                snapshot_commit=snapshot["snapshot_commit"],
                source_tree=self.source_tree,
                source_commit=self.source_commit,
                tag_name="v1.1.0",
                tag_object=tag["tag_object"],
            )

    def test_source_tag_change_between_reads_is_rejected(self) -> None:
        state = self.root / "tag-race"
        state.mkdir(mode=0o700)
        real = MIRROR._tag_state
        with mock.patch.object(MIRROR, "_tag_state", side_effect=[real(self.source, "v1.1.0"), ("f" * 40, self.source_commit)]):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "changed during"):
                MIRROR.prepare_snapshot(
                    source_repo=self.source,
                    source_commit=self.source_commit,
                    source_tree=self.source_tree,
                    state_root=state,
                    identity=self.identity,
                    source_tag="v1.1.0",
                    source_tag_object=self.source_tag_object,
                    forbidden_objects=[self.secret_blob],
                )

    def test_receipts_are_strict_fail_closed_and_exclusive(self) -> None:
        state, snapshot, receipt, _tag = self.make_state("strict")
        self.assertEqual(
            self.verify_source_authority(),
            self.source_authority,
        )
        replayed = MIRROR.replay_receipt(
            receipt,
            source_authority=self.source_authority,
            snapshot=snapshot,
            target_preflight=self.target_preflight,
            snapshot_repo=state / "isolated-snapshot.git",
        )
        self.assertEqual(replayed, receipt)
        changed = json.loads(json.dumps(receipt))
        changed["unexpected"] = True
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "schema differs"):
            MIRROR.validate_receipt(changed)
        unsafe = json.loads(json.dumps(self.target_preflight))
        unsafe["actions_enabled"] = True
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "Actions"):
            MIRROR.validate_target_preflight(unsafe)
        output = self.root / "exclusive.json"
        MIRROR._write_new_json(output, {"value": 1}, "exclusive test")
        before = output.read_bytes()
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "overwrite"):
            MIRROR._write_new_json(output, {"value": 2}, "exclusive test")
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        unsafe_parent = self.root / "unsafe-output-parent"
        unsafe_parent.mkdir(mode=0o755)
        unsafe_parent.chmod(0o755)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "parent mode must be 0700"):
            MIRROR._write_new_json(unsafe_parent / "evidence.json", {"value": 1}, "unsafe output")

    def test_source_authority_replay_rejects_mutated_private_asset(self) -> None:
        self.verify_source_authority()
        before = self.candidate.read_bytes()
        self.candidate.write_bytes(before + b"mutation")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "bytes differ"):
            self.verify_source_authority()

    def test_pinned_final_verifier_rejects_attacker_invalid_candidate(self) -> None:
        invalid = self.root / "invalid-candidate.json"
        payload = json.loads(self.candidate.read_text())
        payload["state"] = "ATTACKER_AUTHORIZED"
        unsigned = dict(payload)
        unsigned.pop("receipt_id")
        payload["receipt_id"] = MIRROR._canonical_sha256(unsigned)
        invalid.write_bytes(MIRROR._canonical_bytes(payload) + b"\n")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "rejected the authority chain"):
            MIRROR._run_pinned_final_verifier(
                verifier_path=self.final_verifier,
                expected_verifier_sha256=self.final_verifier_sha256,
                verifier_arguments=[],
                final_receipt_path=self.final,
                package_path=self.package,
                candidate_receipt_path=invalid,
            )

    def test_private_source_tag_message_must_exactly_replay_source_publisher(self) -> None:
        git("tag", "-d", "v1.1.0", cwd=self.source)
        git("tag", "-a", "v1.1.0", "-m", "same commit, attacker message", cwd=self.source)
        changed = json.loads(json.dumps(self.source_authority))
        changed["source_tag"]["object_id"] = oid(self.source, "refs/tags/v1.1.0")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "out-of-band trusted object"):
            MIRROR.verify_source_authority_files(
                changed,
                self.assets,
                source_repo=self.source,
                final_verifier_path=self.final_verifier,
                final_verifier_sha256=self.final_verifier_sha256,
                final_verifier_arguments=[],
                trusted_source_tag_object=self.source_tag_object,
            )
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "message differs"):
            MIRROR.verify_source_authority_files(
                changed,
                self.assets,
                source_repo=self.source,
                final_verifier_path=self.final_verifier,
                final_verifier_sha256=self.final_verifier_sha256,
                final_verifier_arguments=[],
                trusted_source_tag_object=changed["source_tag"]["object_id"],
            )

    def test_target_settings_matrix_is_fail_closed(self) -> None:
        mutations = {
            "owner/name": ("repository", "slug", "other/gpmeep"),
            "public": (None, "visibility", "public"),
            "fork": (None, "fork", True),
            "template": (None, "template", True),
            "archived": (None, "archived", True),
            "disabled": (None, "disabled", True),
            "Actions": (None, "actions_enabled", True),
            "immutable": (None, "immutable_releases_enabled", False),
            "org permission": ("organization_access", "default_repository_permission", "none"),
            "org all-members": ("organization_access", "all_members_can_read", False),
            "Dependabot security": ("automation", "dependabot_security_updates_enabled", True),
            "Dependabot version": ("automation", "dependabot_version_updates_enabled", True),
            "live default branch": ("automation", "no_live_default_branch", False),
            "Administration read": ("automation", "repository_administration_read_authorized", False),
        }
        for label, (parent, key, value) in mutations.items():
            with self.subTest(label=label):
                changed = json.loads(json.dumps(self.target_preflight))
                if parent is None:
                    changed[key] = value
                else:
                    changed[parent][key] = value
                with self.assertRaises(MIRROR.ReleaseOnlyError):
                    MIRROR.validate_target_preflight(changed)

    def test_dependabot_policy_rejects_duplicate_or_nonzero_limit(self) -> None:
        policy = self.source / ".github/dependabot.yml"
        original = policy.read_text(encoding="utf-8")
        attacks = {
            "nonzero duplicate": "    open-pull-requests-limit: 1\n",
            "same-value duplicate": "    open-pull-requests-limit: 0\n",
            "quoted duplicate": '    "open-pull-requests-limit": 7\n',
            "duplicate top-level": "updates: []\n",
            "flow semantic key": "extra: {open-pull-requests-limit: 7}\n",
            "unicode escaped semantic key": (
                '  - "package-\\u0065cosystem": npm\n'
                '    directory: /\n'
                '    schedule: {interval: daily}\n'
            ),
            "explicit semantic key": (
                "  - ? package-ecosystem\n"
                "    : npm\n"
                "    directory: /\n"
            ),
            "tagged semantic key": (
                "  - !!map {package-ecosystem: npm, directory: /}\n"
            ),
        }
        for label, suffix in attacks.items():
            with self.subTest(label=label):
                policy.write_text(original + suffix, encoding="utf-8")
                git("add", ".github/dependabot.yml", cwd=self.source)
                git("commit", "-m", f"malformed dependabot policy: {label}", cwd=self.source)
                tree = oid(self.source, "HEAD^{tree}")
                with self.assertRaises(MIRROR.ReleaseOnlyError):
                    MIRROR.verify_dependabot_suppression(self.source / ".git", tree)

    def test_distribution_manifest_is_semantically_bound_to_receipt(self) -> None:
        _state, _snapshot, receipt, _tag = self.make_state("distribution-cross-binding")
        assets = self.root / "distribution-cross-binding-assets"
        manifest = MIRROR.create_distribution_manifest(
            source_manifest_path=assets / self.source_manifest.name,
            asset_directory=assets,
            mirror_receipt_path=assets / "gpmeep-v1.1.0-release-only-receipt.json",
            target_history_attestation_path=assets / self.target_history_attestation_path.name,
        )
        changed = json.loads(json.dumps(manifest))
        changed["source_manifest_id"] = "f" * 64
        unsigned = dict(changed)
        unsigned.pop("manifest_id")
        changed["manifest_id"] = MIRROR._canonical_sha256(unsigned)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "source manifest identity"):
            MIRROR.cross_validate_distribution_metadata(changed, receipt)
        with self.assertRaises(MIRROR.ReleaseOnlyError):
            MIRROR.tag_message(receipt, changed)
        changed_history = json.loads(json.dumps(manifest))
        history_record = next(
            record for record in changed_history["assets"]
            if record["role"] == "target_history_attestation"
        )
        history_record["sha256"] = "e" * 64
        changed_history.pop("manifest_id")
        changed_history["manifest_id"] = MIRROR._canonical_sha256(changed_history)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "target history bytes"):
            MIRROR.cross_validate_distribution_metadata(changed_history, receipt)
        self.write_json(assets / MIRROR.DISTRIBUTION_MANIFEST_NAME, manifest)
        MIRROR.cross_validate_distribution(manifest, receipt, assets)
        (assets / "attacker-extra.txt").write_text("unlisted\n")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "directory census"):
            MIRROR.cross_validate_distribution(manifest, receipt, assets)

    def test_external_audit_rejects_changed_implementation_bytes(self) -> None:
        audit = json.loads(self.external_audit.read_text())
        audit["implementation"]["receipt_tool_sha256"] = "f" * 64
        unsigned = dict(audit)
        unsigned.pop("audit_id")
        audit["audit_id"] = MIRROR._canonical_sha256(unsigned)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "implementation bytes changed"):
            MIRROR.verify_audited_implementation(audit)
        changed_cli = json.loads(self.external_audit.read_text())
        changed_cli["github_cli"]["sha256"] = "f" * 64
        changed_cli.pop("audit_id")
        changed_cli["audit_id"] = MIRROR._canonical_sha256(changed_cli)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "GitHub CLI authority"):
            MIRROR.validate_external_mirror_audit(changed_cli)
        self.external_audit.chmod(0o666)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "mode must be 0600"):
            MIRROR.external_audit_record(self.external_audit, self.external_audit_sha256)

    def private_live_record(self, name):
        path = self.assets / name
        return {"name": name, "size": path.stat().st_size,
                "digest": "sha256:" + MIRROR._sha256_file(path), "state": "uploaded"}

    def test_source_authority_accepts_full_private_release_manifest_only(self) -> None:
        manifest = json.loads(self.source_manifest.read_text())
        for role, name in (("qualification_manifest", "qualification.json"),
                           ("complete_evidence", "private-evidence.tar.gz")):
            path = self.assets / name
            path.write_bytes(b"private fixture evidence\n")
            manifest["assets"].append({"role": role, "name": name,
                                       "size_bytes": path.stat().st_size,
                                       "sha256": MIRROR._sha256_file(path)})
        manifest["assets"].sort(key=lambda item: item["role"])
        manifest.pop("manifest_id")
        manifest["manifest_id"] = MIRROR._canonical_sha256(manifest)
        self.write_json(self.source_manifest, manifest)
        message = MIRROR._expected_private_tag_message(
            source_commit=self.source_commit, package_sha256=self.source_authority["package"]["sha256"],
            candidate_id=self.source_authority["candidate_receipt"]["receipt_id"],
            final_id=self.source_authority["final_receipt"]["receipt_id"], manifest_id=manifest["manifest_id"])
        git("tag", "-d", "v1.1.0", cwd=self.source)
        git("tag", "-a", "v1.1.0", "-m", message, cwd=self.source)
        tag_object = oid(self.source, "refs/tags/v1.1.0")
        repository_path = self.root / "full-source-repository.json"
        release_path = self.root / "full-source-release.json"
        self.write_json(repository_path, {"full_name": "iverydj/gpmeep", "id": 101,
                                           "node_id": "R_private", "private": True})
        release = {"id": 303, "node_id": "R_source_release", "tag_name": "v1.1.0",
                   "draft": False, "prerelease": False, "immutable": True,
                   "assets": [self.private_live_record(item["name"]) for item in manifest["assets"]]
                             + [self.private_live_record(self.source_manifest.name)]}
        self.write_json(release_path, release)
        def create():
            return MIRROR.create_source_authority(
                source_repo=self.source, asset_directory=self.assets,
                final_receipt_name=self.final.name, candidate_receipt_name=self.candidate.name,
                package_name=self.package.name, source_manifest_name=self.source_manifest.name,
                private_repository_metadata_path=repository_path, private_release_metadata_path=release_path,
                final_verifier_path=self.final_verifier, final_verifier_sha256=self.final_verifier_sha256,
                final_verifier_arguments=[], trusted_source_tag_object=tag_object)
        authority = create()
        self.assertEqual(authority["source_asset_manifest"]["manifest_id"], manifest["manifest_id"])
        for index in range(len(release["assets"])):
            for key, invalid in (("size", 1), ("digest", "sha256:" + "0" * 64), ("state", "starter")):
                original = release["assets"][index][key]
                release["assets"][index][key] = invalid
                self.write_json(release_path, release)
                with self.subTest(index=index, key=key), self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "live asset bytes or state"):
                    create()
                release["assets"][index][key] = original
        release["assets"].append({"name": "unreviewed.txt"})
        self.write_json(release_path, release)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "live asset inventory"):
            create()
        release["assets"].pop()
        self.write_json(release_path, release)
        (self.assets / "private-evidence.tar.gz").write_bytes(b"changed evidence\n")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "source asset bytes differ"):
            create()

    def test_creator_clis_and_run_intent_replay_exact_live_captures(self) -> None:
        private_repository_metadata = self.root / "private-repository.json"
        private_release_metadata = self.root / "private-release.json"
        self.write_json(
            private_repository_metadata,
            {
                "full_name": "iverydj/gpmeep",
                "id": 101,
                "node_id": "R_private",
                "private": True,
            },
        )
        self.write_json(
            private_release_metadata,
            {
                "id": 303,
                "node_id": "R_source_release",
                "tag_name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "assets": [
                    self.private_live_record(name)
                    for name in sorted(
                        [self.final.name, self.candidate.name, self.package.name, self.source_manifest.name]
                    )
                ],
            },
        )
        generated_source = self.root / "generated-source-authority.json"
        self.run_tool(
            "create-source-authority",
            "--source-repo", str(self.source),
            "--final-verifier", str(self.final_verifier),
            "--final-verifier-sha256", self.final_verifier_sha256,
            "--asset-directory", str(self.assets),
            "--final-receipt", self.final.name,
            "--candidate-receipt", self.candidate.name,
            "--package", self.package.name,
            "--source-manifest", self.source_manifest.name,
            "--private-repository-metadata", str(private_repository_metadata),
            "--private-release-metadata", str(private_release_metadata),
            "--trusted-source-tag-object", self.source_tag_object,
            "--output", str(generated_source),
        )
        self.assertEqual(json.loads(generated_source.read_text()), self.source_authority)
        self.run_tool(
            "verify-source-authority",
            "--source-repo", str(self.source),
            "--final-verifier", str(self.final_verifier),
            "--final-verifier-sha256", self.final_verifier_sha256,
            "--source-authority", str(generated_source),
            "--asset-directory", str(self.assets),
            "--trusted-source-tag-object", self.source_tag_object,
        )
        prerelease_metadata = json.loads(private_release_metadata.read_text())
        prerelease_metadata["prerelease"] = True
        self.write_json(private_release_metadata, prerelease_metadata)
        prerelease = self.run_tool(
            "create-source-authority",
            "--source-repo", str(self.source),
            "--final-verifier", str(self.final_verifier),
            "--final-verifier-sha256", self.final_verifier_sha256,
            "--asset-directory", str(self.assets),
            "--final-receipt", self.final.name,
            "--candidate-receipt", self.candidate.name,
            "--package", self.package.name,
            "--source-manifest", self.source_manifest.name,
            "--private-repository-metadata", str(private_repository_metadata),
            "--private-release-metadata", str(private_release_metadata),
            "--trusted-source-tag-object", self.source_tag_object,
            "--output", str(self.root / "must-not-create-prerelease-authority.json"),
            check=False,
        )
        self.assertNotEqual(prerelease.returncode, 0)
        self.assertIn(b"private source release live metadata differs", prerelease.stderr)

        empty_target = self.root / "creator-target.git"
        git("init", "--bare", "--template=", str(empty_target))
        advertised_refs = self.root / "creator-target-refs.tsv"
        advertised_refs.write_bytes(b"")
        repository_metadata = self.root / "target-repository.json"
        actions_permissions = self.root / "target-actions.json"
        immutable_releases = self.root / "target-immutable.json"
        organization_metadata = self.root / "target-organization.json"
        self.write_json(
            repository_metadata,
            {
                "full_name": MIRROR.TARGET_REPOSITORY,
                "id": 202,
                "node_id": "R_release_only",
                "private": True,
                "visibility": "private",
                "fork": False,
                "is_template": False,
                "archived": False,
                "disabled": False,
                "permissions": {"admin": True, "maintain": True, "push": True, "triage": True, "pull": True},
            },
        )
        self.write_json(actions_permissions, {"enabled": False})
        self.write_json(immutable_releases, {"enabled": True})
        self.write_json(
            organization_metadata,
            {
                "login": "example-organization",
                "id": 404,
                "node_id": "O_fixture",
                "default_repository_permission": "read",
            },
        )
        creator_audit, creator_audit_sha256 = self.write_empty_external_audit(
            self.root / "creator-empty-target-audit.json"
        )
        generated_history = self.root / "generated-target-history.json"
        self.run_tool(
            "create-target-history",
            "--repo", str(empty_target),
            "--repository-metadata", str(repository_metadata),
            "--mode", "empty-new-repository",
            "--advertised-refs", str(advertised_refs),
            "--external-audit", str(creator_audit),
            "--external-audit-sha256", creator_audit_sha256,
            "--output", str(generated_history),
        )
        generated_history_value = MIRROR.validate_target_history_attestation(
            json.loads(generated_history.read_text())
        )
        self.assertEqual(generated_history_value["mode"], "empty-new-repository")
        self.assertEqual(generated_history_value["prior_releases"], [])
        generated_preflight = self.root / "generated-target-preflight.json"
        self.run_tool(
            "create-target-preflight",
            "--repository-metadata", str(repository_metadata),
            "--actions-permissions", str(actions_permissions),
            "--immutable-releases", str(immutable_releases),
            "--organization-metadata", str(organization_metadata),
            "--dependabot-security-updates-status", "disabled",
            "--advertised-refs", str(advertised_refs),
            "--history-attestation", str(generated_history),
            "--external-audit", str(creator_audit),
            "--external-audit-sha256", creator_audit_sha256,
            "--output", str(generated_preflight),
        )
        generated_preflight_value = MIRROR.validate_target_preflight(
            json.loads(generated_preflight.read_text())
        )
        self.assertEqual(generated_preflight_value["bootstrap"]["mode"], "empty-new-repository")
        nonadmin_metadata = json.loads(repository_metadata.read_text())
        nonadmin_metadata["permissions"]["admin"] = False
        nonadmin_path = self.root / "target-repository-nonadmin.json"
        self.write_json(nonadmin_path, nonadmin_metadata)
        nonadmin = self.run_tool(
            "create-target-preflight",
            "--repository-metadata", str(nonadmin_path),
            "--actions-permissions", str(actions_permissions),
            "--immutable-releases", str(immutable_releases),
            "--organization-metadata", str(organization_metadata),
            "--dependabot-security-updates-status", "disabled",
            "--advertised-refs", str(advertised_refs),
            "--history-attestation", str(generated_history),
            "--external-audit", str(creator_audit),
            "--external-audit-sha256", creator_audit_sha256,
            "--output", str(self.root / "must-not-create-preflight.json"),
            check=False,
        )
        self.assertNotEqual(nonadmin.returncode, 0)
        self.assertIn(b"Administration(read) authority", nonadmin.stderr)
        for unsafe_field in ("archived", "disabled"):
            unsafe_metadata = json.loads(repository_metadata.read_text())
            unsafe_metadata[unsafe_field] = True
            unsafe_path = self.root / f"target-repository-{unsafe_field}.json"
            self.write_json(unsafe_path, unsafe_metadata)
            unsafe = self.run_tool(
                "create-target-preflight",
                "--repository-metadata", str(unsafe_path),
                "--actions-permissions", str(actions_permissions),
                "--immutable-releases", str(immutable_releases),
                "--organization-metadata", str(organization_metadata),
                "--dependabot-security-updates-status", "disabled",
                "--advertised-refs", str(advertised_refs),
                "--history-attestation", str(generated_history),
                "--external-audit", str(creator_audit),
                "--external-audit-sha256", creator_audit_sha256,
                "--output", str(self.root / f"must-not-create-{unsafe_field}-preflight.json"),
                check=False,
            )
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertIn(b"target live repository metadata differs", unsafe.stderr)

        state_root = self.root / "intent-state"
        state_root.mkdir(mode=0o700)
        intent = self.root / "run-intent.json"
        common = [
            "--source-repo", str(self.source),
            "--source-authority", str(generated_source),
            "--target-preflight", str(generated_preflight),
            "--target-history-attestation", str(generated_history),
            "--external-audit", str(creator_audit),
            "--external-audit-sha256", creator_audit_sha256,
            "--asset-directory", str(self.assets),
            "--state-root", str(state_root),
            "--repository", MIRROR.TARGET_REPOSITORY,
            "--identity-name", str(self.identity["name"]),
            "--identity-email", str(self.identity["email"]),
            "--identity-epoch", str(self.identity["epoch"]),
            "--identity-timezone", str(self.identity["timezone"]),
            "--forbid-object", self.secret_commit,
            "--forbid-object", self.secret_blob,
            "--local-bare-remote", str(empty_target),
        ]
        self.run_tool("create-run-intent", "--output", str(intent), *common)
        self.run_tool("verify-run-intent", "--intent", str(intent), *common)
        changed = list(common)
        changed[changed.index(str(self.identity["epoch"]))] = str(int(self.identity["epoch"]) + 1)
        failed = self.run_tool("verify-run-intent", "--intent", str(intent), *changed, check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn(b"run intent differs", failed.stderr)

        other_target = self.root / "other-target.git"
        git("init", "--bare", "--template=", str(other_target))
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "run intent differs"):
            MIRROR.replay_run_intent(
                json.loads(intent.read_text()),
                source_repo=self.source,
                source_authority_path=generated_source,
                target_preflight_path=generated_preflight,
                target_history_attestation_path=generated_history,
                external_audit_path=creator_audit,
                asset_directory=self.assets,
                state_root=state_root,
                repository=MIRROR.TARGET_REPOSITORY,
                identity=self.identity,
                forbidden_objects=[self.secret_commit, self.secret_blob],
                local_bare_remote=other_target,
                gh_binary=None,
                gh_sha256=None,
                gh_capabilities_sha256=None,
                trusted_external_audit_sha256=creator_audit_sha256,
            )

    def test_creator_inputs_cannot_change_during_capture(self) -> None:
        target = self.root / "toctou-target.git"
        git("init", "--bare", "--template=", str(target))
        state = self.root / "toctou-state"
        state.mkdir(mode=0o700)
        real_intent_file = MIRROR._intent_file
        calls = 0

        def racing_intent_file(path: pathlib.Path, label: str):
            nonlocal calls
            value = real_intent_file(path, label)
            calls += 1
            if calls == 1:
                self.target_preflight_path.write_bytes(self.target_preflight_path.read_bytes())
                self.target_preflight_path.chmod(0o600)
            return value

        with mock.patch.object(MIRROR, "_intent_file", side_effect=racing_intent_file):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                MIRROR.create_run_intent(
                    source_repo=self.source,
                    source_authority_path=self.source_authority_path,
                    target_preflight_path=self.target_preflight_path,
                    target_history_attestation_path=self.target_history_attestation_path,
                    external_audit_path=self.external_audit,
                    asset_directory=self.assets,
                    state_root=state,
                    repository=MIRROR.TARGET_REPOSITORY,
                    identity=self.identity,
                    forbidden_objects=[self.secret_commit, self.secret_blob],
                    local_bare_remote=target,
                    gh_binary=None,
                    gh_sha256=None,
                    gh_capabilities_sha256=None,
                    trusted_external_audit_sha256=self.external_audit_sha256,
                )

        def create_local_intent() -> dict[str, object]:
            return MIRROR.create_run_intent(
                source_repo=self.source,
                source_authority_path=self.source_authority_path,
                target_preflight_path=self.target_preflight_path,
                target_history_attestation_path=self.target_history_attestation_path,
                external_audit_path=self.external_audit,
                asset_directory=self.assets,
                state_root=state,
                repository=MIRROR.TARGET_REPOSITORY,
                identity=self.identity,
                forbidden_objects=[self.secret_commit, self.secret_blob],
                local_bare_remote=target,
                gh_binary=None,
                gh_sha256=None,
                gh_capabilities_sha256=None,
                trusted_external_audit_sha256=self.external_audit_sha256,
            )

        intent_file_calls = 0

        def racing_asset_tree(path: pathlib.Path, label: str):
            nonlocal intent_file_calls
            value = real_intent_file(path, label)
            intent_file_calls += 1
            if intent_file_calls == 1:
                self.package.write_bytes(self.package.read_bytes())
            return value

        with mock.patch.object(MIRROR, "_intent_file", side_effect=racing_asset_tree):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                create_local_intent()

        original_source_mode = stat.S_IMODE(self.source.stat().st_mode)
        intent_file_calls = 0

        def racing_directory_identity(path: pathlib.Path, label: str):
            nonlocal intent_file_calls
            value = real_intent_file(path, label)
            intent_file_calls += 1
            if intent_file_calls == 1:
                self.source.chmod(0o700 if original_source_mode != 0o700 else 0o750)
            return value

        try:
            with mock.patch.object(MIRROR, "_intent_file", side_effect=racing_directory_identity):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    create_local_intent()
        finally:
            self.source.chmod(original_source_mode)

        real_tree_snapshot = MIRROR._regular_tree_snapshot
        tree_snapshot_calls = 0

        def racing_source_refs(path: pathlib.Path, label: str):
            nonlocal tree_snapshot_calls
            value = real_tree_snapshot(path, label)
            tree_snapshot_calls += 1
            if tree_snapshot_calls == 2:
                git("update-ref", "refs/heads/toctou-source", self.source_commit, cwd=self.source)
            return value

        try:
            with mock.patch.object(MIRROR, "_regular_tree_snapshot", side_effect=racing_source_refs):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    create_local_intent()
        finally:
            git("update-ref", "-d", "refs/heads/toctou-source", self.source_commit, cwd=self.source)

        real_source_git = MIRROR._source_git
        source_git_calls = 0

        def racing_remote_refs(repo: pathlib.Path, arguments: list[str]):
            nonlocal source_git_calls
            value = real_source_git(repo, arguments)
            source_git_calls += 1
            if source_git_calls == 2:
                remote_blob = git(
                    "--git-dir", str(target), "hash-object", "-w", "--stdin", input_bytes=b"intent race\n"
                ).stdout.decode("ascii").strip()
                git("--git-dir", str(target), "update-ref", "refs/tags/v9.9.8", remote_blob, "0" * 40)
            return value

        try:
            with mock.patch.object(MIRROR, "_source_git", side_effect=racing_remote_refs):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    create_local_intent()
        finally:
            git("--git-dir", str(target), "update-ref", "-d", "refs/tags/v9.9.8", check=False)

        history_target = self.root / "toctou-history-target.git"
        git("init", "--bare", "--template=", str(history_target))
        repository_metadata = self.root / "toctou-repository.json"
        refs = self.root / "toctou-refs.tsv"
        self.write_json(
            repository_metadata,
            {
                "full_name": MIRROR.TARGET_REPOSITORY,
                "id": 202,
                "node_id": "R_release_only",
                "private": True,
                "fork": False,
                "is_template": False,
            },
        )
        refs.write_bytes(b"")
        refs.chmod(0o600)
        empty_history_audit, empty_history_audit_sha256 = self.write_empty_external_audit(
            self.root / "toctou-empty-history-audit.json"
        )
        real_load = MIRROR._load_json
        changed = False

        def racing_load(path: pathlib.Path, label: str):
            nonlocal changed
            value = real_load(path, label)
            if not changed and pathlib.Path(path) == repository_metadata:
                changed = True
                refs.write_bytes(refs.read_bytes())
                refs.chmod(0o600)
            return value

        with mock.patch.object(MIRROR, "_load_json", side_effect=racing_load):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                MIRROR.create_target_history_attestation(
                    repo=history_target,
                    repository_metadata_path=repository_metadata,
                    mode="empty-new-repository",
                    advertised_refs_path=refs,
                    prior_assets_root=None,
                    external_audit_path=empty_history_audit,
                    trusted_external_audit_sha256=empty_history_audit_sha256,
                )

        for raced_path in (repository_metadata, empty_history_audit):
            changed = False

            def racing_history_file(path: pathlib.Path, label: str):
                nonlocal changed
                value = real_load(path, label)
                if not changed and pathlib.Path(path) == raced_path:
                    changed = True
                    raced_path.write_bytes(raced_path.read_bytes())
                    raced_path.chmod(0o600)
                return value

            with mock.patch.object(MIRROR, "_load_json", side_effect=racing_history_file):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    MIRROR.create_target_history_attestation(
                        repo=history_target,
                        repository_metadata_path=repository_metadata,
                        mode="empty-new-repository",
                        advertised_refs_path=refs,
                        prior_assets_root=None,
                        external_audit_path=empty_history_audit,
                        trusted_external_audit_sha256=empty_history_audit_sha256,
                    )

        history_snapshot_calls = 0
        real_stable_snapshot = MIRROR._stable_file_snapshot

        def racing_live_history_refs(paths: list[pathlib.Path], label: str):
            nonlocal history_snapshot_calls
            value = real_stable_snapshot(paths, label)
            history_snapshot_calls += 1
            if history_snapshot_calls == 2:
                remote_blob = git(
                    "--git-dir", str(history_target), "hash-object", "-w", "--stdin", input_bytes=b"history race\n"
                ).stdout.decode("ascii").strip()
                git("--git-dir", str(history_target), "update-ref", "refs/tags/v9.9.9", remote_blob, "0" * 40)
            return value

        try:
            with mock.patch.object(MIRROR, "_stable_file_snapshot", side_effect=racing_live_history_refs):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    MIRROR.create_target_history_attestation(
                        repo=history_target,
                        repository_metadata_path=repository_metadata,
                        mode="empty-new-repository",
                        advertised_refs_path=refs,
                        prior_assets_root=None,
                        external_audit_path=empty_history_audit,
                        trusted_external_audit_sha256=empty_history_audit_sha256,
                    )
        finally:
            git("--git-dir", str(history_target), "update-ref", "-d", "refs/tags/v9.9.9", check=False)

        private_repository = self.root / "toctou-private-repository.json"
        private_release = self.root / "toctou-private-release.json"
        self.write_json(
            private_repository,
            {
                "full_name": "iverydj/gpmeep",
                "id": 101,
                "node_id": "R_private",
                "private": True,
            },
        )
        self.write_json(
            private_release,
            {
                "id": 303,
                "node_id": "R_source_release",
                "tag_name": "v1.1.0",
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "assets": [
                    self.private_live_record(name)
                    for name in sorted(
                        [self.final.name, self.candidate.name, self.package.name, self.source_manifest.name]
                    )
                ],
            },
        )
        changed = False

        def racing_source_load(path: pathlib.Path, label: str):
            nonlocal changed
            value = real_load(path, label)
            if not changed and pathlib.Path(path) == private_repository:
                changed = True
                private_release.write_bytes(private_release.read_bytes())
                private_release.chmod(0o600)
            return value

        with mock.patch.object(MIRROR, "_load_json", side_effect=racing_source_load):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                MIRROR.create_source_authority(
                    source_repo=self.source,
                    asset_directory=self.assets,
                    final_receipt_name=self.final.name,
                    candidate_receipt_name=self.candidate.name,
                    package_name=self.package.name,
                    source_manifest_name=self.source_manifest.name,
                    private_repository_metadata_path=private_repository,
                    private_release_metadata_path=private_release,
                    final_verifier_path=self.final_verifier,
                    final_verifier_sha256=self.final_verifier_sha256,
                    final_verifier_arguments=[],
                    trusted_source_tag_object=self.source_tag_object,
                )

        source_snapshot_calls = 0

        def racing_live_source_refs(paths: list[pathlib.Path], label: str):
            nonlocal source_snapshot_calls
            value = real_stable_snapshot(paths, label)
            source_snapshot_calls += 1
            if source_snapshot_calls == 2:
                git("update-ref", "refs/heads/toctou-authority", self.source_commit, cwd=self.source)
            return value

        try:
            with mock.patch.object(MIRROR, "_stable_file_snapshot", side_effect=racing_live_source_refs):
                with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                    MIRROR.create_source_authority(
                        source_repo=self.source,
                        asset_directory=self.assets,
                        final_receipt_name=self.final.name,
                        candidate_receipt_name=self.candidate.name,
                        package_name=self.package.name,
                        source_manifest_name=self.source_manifest.name,
                        private_repository_metadata_path=private_repository,
                        private_release_metadata_path=private_release,
                        final_verifier_path=self.final_verifier,
                        final_verifier_sha256=self.final_verifier_sha256,
                        final_verifier_arguments=[],
                        trusted_source_tag_object=self.source_tag_object,
                    )
        finally:
            git("update-ref", "-d", "refs/heads/toctou-authority", self.source_commit, cwd=self.source)

        preflight_repository = self.root / "toctou-preflight-repository.json"
        preflight_actions = self.root / "toctou-preflight-actions.json"
        preflight_immutable = self.root / "toctou-preflight-immutable.json"
        preflight_organization = self.root / "toctou-preflight-organization.json"
        self.write_json(
            preflight_repository,
            {
                "full_name": MIRROR.TARGET_REPOSITORY,
                "id": 202,
                "node_id": "R_release_only",
                "private": True,
                "visibility": "private",
                "fork": False,
                "is_template": False,
                "archived": False,
                "disabled": False,
                "permissions": {"admin": True},
            },
        )
        self.write_json(preflight_actions, {"enabled": False})
        self.write_json(preflight_immutable, {"enabled": True})
        preflight_refs = self.root / "toctou-preflight-refs.tsv"
        preflight_refs.write_text(
            f"{MIRROR.V103_TARGET_TAG_OBJECT}\trefs/tags/v1.0.3\n",
            encoding="ascii",
        )
        self.write_json(
            preflight_organization,
            {
                "login": "example-organization",
                "id": 404,
                "node_id": "O_fixture",
                "default_repository_permission": "read",
            },
        )
        changed = False

        def racing_preflight_load(path: pathlib.Path, label: str):
            nonlocal changed
            value = real_load(path, label)
            if not changed and pathlib.Path(path) == preflight_repository:
                changed = True
                preflight_actions.write_bytes(preflight_actions.read_bytes())
                preflight_actions.chmod(0o600)
            return value

        with mock.patch.object(MIRROR, "_load_json", side_effect=racing_preflight_load):
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "inputs changed during creation"):
                MIRROR.create_target_preflight(
                    repository_metadata_path=preflight_repository,
                    actions_permissions_path=preflight_actions,
                    immutable_releases_path=preflight_immutable,
                    organization_metadata_path=preflight_organization,
                    dependabot_security_updates_status="disabled",
                    advertised_refs_path=preflight_refs,
                    history_attestation_path=self.target_history_attestation_path,
                    external_audit_path=self.external_audit,
                    trusted_external_audit_sha256=self.external_audit_sha256,
                )

    def test_published_union_accepts_exact_disconnected_prior_and_current_tags(self) -> None:
        prior, prior_assets_root, authority, _release_metadata = self.make_prior_release("combined")
        baseline_records = [
            {"object_id": authority["tag_object"], "ref": "refs/tags/v1.0.3"}
        ]
        state = self.root / "combined-state"
        state.mkdir(mode=0o700)
        snapshot = MIRROR.prepare_snapshot(
            source_repo=self.source,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            state_root=state,
            identity=self.identity,
            source_tag="v1.1.0",
            source_tag_object=self.source_tag_object,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        assets = self.root / "combined-current-assets"
        assets.mkdir(mode=0o700)
        for path in (self.package, self.candidate, self.final, self.source_manifest):
            shutil.copyfile(path, assets / path.name)
        audit_path = assets / self.external_audit.name
        audit = json.loads(self.external_audit.read_text())
        audit["target_history"]["advertised_refs_sha256"] = MIRROR._canonical_sha256(baseline_records)
        audit["target_history"]["prior_releases_sha256"] = MIRROR._canonical_sha256([authority])
        audit.pop("audit_id")
        audit["audit_id"] = MIRROR._canonical_sha256(audit)
        self.write_json(audit_path, audit)
        audit_sha256 = MIRROR._sha256_file(audit_path)
        audit_record = MIRROR.external_audit_record(audit_path, audit_sha256)
        history = {
            "schema": MIRROR.TARGET_HISTORY_SCHEMA,
            "state": "HISTORY_CONFIDENTIALITY_ATTESTED",
            "mode": "audited-prior-releases",
            "never_imported_or_transferred": True,
            "repository": {"slug": MIRROR.TARGET_REPOSITORY, "id": 202, "node_id": "R_release_only"},
            "advertised_refs": ["refs/tags/v1.0.3"],
            "prior_releases": [authority],
            "external_audit": {
                key: audit_record[key] for key in ("name", "size_bytes", "sha256", "audit_id")
            },
        }
        history["attestation_id"] = MIRROR._canonical_sha256(history)
        history_path = assets / "combined-history.json"
        self.write_json(history_path, history)
        preflight = json.loads(json.dumps(self.target_preflight))
        preflight["advertised_refs"] = ["refs/tags/v1.0.3"]
        preflight["bootstrap"] = {
            "mode": "audited-prior-releases",
            "attestation_sha256": MIRROR._sha256_file(history_path),
            "attestation_id": history["attestation_id"],
        }
        preflight["external_mirror_audit"] = audit_record
        preflight = MIRROR.validate_target_preflight(preflight)
        receipt = MIRROR.create_receipt(self.source_authority, snapshot, preflight)
        receipt_path = assets / "gpmeep-v1.1.0-release-only-receipt.json"
        MIRROR._write_new_json(receipt_path, receipt, "combined mirror receipt")
        manifest = MIRROR.create_distribution_manifest(
            source_manifest_path=assets / self.source_manifest.name,
            asset_directory=assets,
            mirror_receipt_path=receipt_path,
            target_history_attestation_path=history_path,
        )
        self.write_json(assets / MIRROR.DISTRIBUTION_MANIFEST_NAME, manifest)
        tag = MIRROR.seal_tag(
            snapshot_repo=state / "isolated-snapshot.git",
            output_root=state,
            receipt=receipt,
            distribution_manifest=manifest,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        git(
            "--git-dir", str(state / "isolated-publisher.git"),
            "push", str(prior), "refs/tags/v1.1.0:refs/tags/v1.1.0",
        )
        result = MIRROR.verify_published_repository(
            repo=prior,
            receipt=receipt,
            distribution_manifest=manifest,
            asset_directory=assets,
            target_history_attestation=history,
            prior_assets_root=prior_assets_root,
            trusted_external_audit_sha256=audit_sha256,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        self.assertEqual(result["state"], "PUBLISHED_RELEASE_ONLY_VERIFIED")
        self.assertEqual(result["refs"], ["refs/tags/v1.0.3", "refs/tags/v1.1.0"])
        self.assertEqual(result["tag_object"], tag["tag_object"])
        self.assertEqual(set(result["roots"]), {authority["snapshot_commit"], snapshot["snapshot_commit"]})

        # Reconstruct the full current-release proof using only files present
        # in a clean organization-release download directory.
        clean_download = self.root / "clean-release-download"
        clean_download.mkdir(mode=0o700)
        for record in manifest["assets"]:
            shutil.copyfile(assets / record["name"], clean_download / record["name"])
            (clean_download / record["name"]).chmod(0o600)
        distribution_path = clean_download / "gpmeep-v1.1.0-release-only-assets.json"
        self.write_json(distribution_path, manifest)
        records = {record["role"]: record for record in manifest["assets"]}
        clean_result = MIRROR.verify_published_repository(
            repo=prior,
            receipt=json.loads((clean_download / records["mirror_receipt"]["name"]).read_text()),
            distribution_manifest=json.loads(distribution_path.read_text()),
            asset_directory=clean_download,
            target_history_attestation=json.loads(
                (clean_download / records["target_history_attestation"]["name"]).read_text()
            ),
            prior_assets_root=prior_assets_root,
            trusted_external_audit_sha256=audit_sha256,
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        self.assertEqual(clean_result, result)

    def test_prior_release_audit_rejects_main_child_and_accepts_exact_immutable_tag_release(self) -> None:
        prior, prior_assets_root, authority, release_metadata_path = self.make_prior_release()
        root_commit = authority["snapshot_commit"]
        tree = authority["snapshot_tree"]
        prior_audit_path = self.root / "prior-independent-audit.json"
        prior_audit = json.loads(self.external_audit.read_text())
        prior_audit["target_history"]["advertised_refs_sha256"] = MIRROR._canonical_sha256(
            [{"object_id": authority["tag_object"], "ref": "refs/tags/v1.0.3"}]
        )
        prior_audit["target_history"]["prior_releases_sha256"] = MIRROR._canonical_sha256([authority])
        prior_audit.pop("audit_id")
        prior_audit["audit_id"] = MIRROR._canonical_sha256(prior_audit)
        self.write_json(prior_audit_path, prior_audit)
        prior_audit_sha256 = MIRROR._sha256_file(prior_audit_path)
        history = {
            "schema": MIRROR.TARGET_HISTORY_SCHEMA,
            "state": "HISTORY_CONFIDENTIALITY_ATTESTED",
            "mode": "audited-prior-releases",
            "never_imported_or_transferred": True,
            "repository": {"slug": MIRROR.TARGET_REPOSITORY, "id": 202, "node_id": "R_release_only"},
            "advertised_refs": ["refs/tags/v1.0.3"],
            "prior_releases": [authority],
            "external_audit": {key: MIRROR.external_audit_record(prior_audit_path, prior_audit_sha256)[key] for key in ("name", "size_bytes", "sha256", "audit_id")},
        }
        history["attestation_id"] = MIRROR._canonical_sha256(history)
        child = git(
            "--git-dir", str(prior), "commit-tree", tree, "-p", root_commit,
            input_bytes=b"dependabot-only main child\n",
        ).stdout.decode().strip()
        git("--git-dir", str(prior), "update-ref", "refs/heads/main", child, "0" * 40)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "advertised ref inventory"):
            MIRROR.audit_prior_release_repository(
                prior,
                history_attestation=history,
                prior_assets_root=prior_assets_root,
                external_audit_path=prior_audit_path,
                trusted_external_audit_sha256=prior_audit_sha256,
            )

        git("--git-dir", str(prior), "update-ref", "-d", "refs/heads/main", child)
        git("--git-dir", str(prior), "prune", "--expire=now")
        audit = MIRROR.audit_prior_release_repository(
            prior,
            history_attestation=history,
            prior_assets_root=prior_assets_root,
            external_audit_path=prior_audit_path,
            trusted_external_audit_sha256=prior_audit_sha256,
        )
        self.assertEqual(audit["state"], "PRIOR_RELEASE_TOPOLOGY_PASS")
        self.assertEqual(audit["reachable_commits"], [root_commit])

        # A stale/tampered REST capture must not remain trusted merely because
        # the tag and release assets themselves are unchanged.
        release_metadata = json.loads(release_metadata_path.read_text())
        for field, unsafe_value in (("immutable", False), ("prerelease", True)):
            changed_metadata = json.loads(json.dumps(release_metadata))
            changed_metadata["release"][field] = unsafe_value
            changed_metadata.pop("capture_id")
            changed_metadata["capture_id"] = MIRROR._canonical_sha256(changed_metadata)
            self.write_json(release_metadata_path, changed_metadata)
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "not final and immutable"):
                MIRROR.audit_prior_release_repository(
                    prior,
                    history_attestation=history,
                    prior_assets_root=prior_assets_root,
                    external_audit_path=prior_audit_path,
                    trusted_external_audit_sha256=prior_audit_sha256,
                )
        self.write_json(release_metadata_path, release_metadata)

    def test_prior_release_evidence_rejects_extra_files_legacy_receipts_and_signed_substitution(self) -> None:
        prior, prior_assets_root, authority, _metadata_path = self.make_prior_release("prior-negative")
        evidence = prior_assets_root / "v1.0.3"

        extra = evidence / "attacker-extra.bin"
        extra.write_bytes(b"ignored by suffix-based scanners\n")
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "evidence directory census"):
            MIRROR._derive_prior_release_authority(
                repo=prior, tag="v1.0.3", release_directory=evidence,
            )
        extra.unlink()

        receipt_path = evidence / MIRROR.PRIOR_MIGRATION_RECEIPT_NAME
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        receipt["schema"] = "gpmeep-v103-release-only-legacy-migration-v1"
        receipt.pop("publication_id")
        receipt["publication_id"] = MIRROR._canonical_sha256(receipt)
        self.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "receipt schema"):
            MIRROR._derive_prior_release_authority(
                repo=prior, tag="v1.0.3", release_directory=evidence,
            )
        receipt_path.write_bytes(receipt_bytes)
        receipt_path.chmod(0o600)

        attestation_path = evidence / MIRROR.PRIOR_RELEASE_ATTESTATION_NAME
        signed = json.loads(attestation_path.read_text())
        signed["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = "f" * 64
        signed["attestation"]["bundle"]["dsseEnvelope"]["payload"] = base64.b64encode(
            MIRROR._canonical_bytes(signed["verificationResult"]["statement"])
        ).decode()
        self.write_json(attestation_path, signed)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "asset digest differs"):
            MIRROR._derive_prior_release_authority(
                repo=prior, tag="v1.0.3", release_directory=evidence,
            )

        # The trusted independent audit must bind the entire prior authority,
        # not merely its ref name/OID pair.
        changed = json.loads(json.dumps(authority))
        changed["release"]["title"] = "attacker title"
        self.assertNotEqual(MIRROR._canonical_sha256([changed]), MIRROR._canonical_sha256([authority]))

    def test_v103_migration_audit_and_run_intent_replay_are_fail_closed(self) -> None:
        legacy_repo, legacy_assets, target, audit_path, audit_sha256, fake_gh = self.make_v103_migration_audit()
        fake_digest = MIRROR._sha256_file(fake_gh)
        with (
            mock.patch.object(MIRROR, "AUDITED_GITHUB_CLI_SHA256", fake_digest),
            mock.patch.object(MIRROR, "APPROVED_GITHUB_CLI_SHA256", fake_digest),
        ):
            audit = MIRROR.verify_v103_migration_audit_files(
                json.loads(audit_path.read_text()), audit_path=audit_path,
                trusted_sha256=audit_sha256, legacy_repo=legacy_repo,
                legacy_assets=legacy_assets,
            )
            intent = MIRROR.create_v103_migration_run_intent(
                audit_path=audit_path, trusted_audit_sha256=audit_sha256,
                legacy_repo=legacy_repo, legacy_assets=legacy_assets,
                target_repo=target, gh_binary=fake_gh,
            )
            self.assertEqual(intent["audit"], {"sha256": audit_sha256, "audit_id": audit["audit_id"]})

            # Exact resume rebuilds the authority from the real OOB audit even
            # after the target has acquired the approved migration tag.
            git(
                "--git-dir", str(legacy_repo), "push", str(target),
                "refs/tags/v1.0.3:refs/tags/v1.0.3",
            )
            replay = MIRROR.create_v103_migration_run_intent(
                audit_path=audit_path, trusted_audit_sha256=audit_sha256,
                legacy_repo=legacy_repo, legacy_assets=legacy_assets,
                target_repo=target, gh_binary=fake_gh, require_empty_target=False,
            )
            self.assertEqual(replay, intent)

            forged = json.loads(json.dumps(intent))
            forged["audit"]["audit_id"] = "f" * 64
            forged.pop("intent_id")
            forged["intent_id"] = MIRROR._canonical_sha256(forged)
            self.assertNotEqual(forged, replay)

            extra = legacy_assets / "attacker-extra.bin"
            extra.write_bytes(b"extra\n")
            with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "asset directory census"):
                MIRROR.verify_v103_migration_audit_files(
                    json.loads(audit_path.read_text()), audit_path=audit_path,
                    trusted_sha256=audit_sha256, legacy_repo=legacy_repo,
                    legacy_assets=legacy_assets,
                )

    def test_v103_migration_controller_has_no_force_delete_and_orders_signed_terminal(self) -> None:
        text = (ROOT / "scripts/publish-v103-release-only-migration.sh").read_text()
        self.assertNotRegex(text, r"\bgit\b[^\n]*(?:--force|-f\b|push[^\n]*\+refs/)")
        self.assertNotRegex(text, r"\bgh_call\s+(?:release\s+delete|repo\s+delete|api[^\n]*-X\s+DELETE)")
        self.assertNotIn("--clobber", text)
        tag_push = text.index('push --porcelain "git@github.com:${REPOSITORY}.git"')
        draft = text.index('release create "$TAG"')
        upload = text.index('release upload "$TAG"')
        publish = text.index('release edit "$TAG"')
        verify_asset = text.index('release verify-asset "$TAG"')
        terminal = text.index('create-v103-migration-terminal')
        self.assertLess(tag_push, draft)
        self.assertLess(draft, upload)
        self.assertLess(upload, publish)
        self.assertLess(publish, verify_asset)
        self.assertLess(verify_asset, terminal)
        self.assertGreaterEqual(text.count("verify_source_live"), 7)
        self.assertGreaterEqual(text.count("verify_target_controls"), 7)

    def test_v103_migration_controller_rejects_any_unrelated_existing_release(self) -> None:
        fixture = self.make_v103_controller_fixture("migration-unrelated-release")
        state_root = self.root / "migration-unrelated-release-state"
        state_root.mkdir(mode=0o700)
        state = json.loads(fixture["state"].read_text())
        state["extra_releases"] = [{
            "id": 602,
            "node_id": "RR_unrelated",
            "tag_name": "v0.0.0",
            "name": "unrelated",
            "body": "must not be ignored",
            "draft": True,
            "prerelease": False,
            "immutable": False,
            "assets": [],
        }]
        self.write_json(fixture["state"], state)
        command = [
            str(fixture["controller"]), "--audit", str(fixture["audit"]),
            "--audit-sha256", MIRROR._sha256_file(fixture["audit"]),
            "--legacy-repo", str(fixture["legacy_repo"]),
            "--legacy-assets", str(fixture["legacy_assets"]),
            "--state-root", str(state_root), "--gh-binary", str(fixture["gh"]),
            "--gh-sha256", MIRROR._sha256_file(fixture["gh"]),
        ]
        environment = {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "GH_TOKEN": "fixture-token",
        }
        rejected = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, timeout=180,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(b"requires zero existing releases", rejected.stderr)
        after = json.loads(fixture["state"].read_text())
        self.assertIsNone(after["release"])
        self.assertEqual(after["upload_calls"], 0)
        self.assertEqual(after["extra_releases"], state["extra_releases"])
        self.assertFalse(git("--git-dir", str(fixture["target"]), "show-ref", check=False).stdout)
        self.assertFalse((state_root / "v103-migration-run-intent.json").exists())

    def test_v103_migration_controller_dry_run_upload_crash_resume_and_terminal(self) -> None:
        fixture = self.make_v103_controller_fixture()
        state_root = self.root / "migration-controller-state"
        state_root.mkdir(mode=0o700)
        base = [
            str(fixture["controller"]), "--audit", str(fixture["audit"]),
            "--audit-sha256", MIRROR._sha256_file(fixture["audit"]),
            "--legacy-repo", str(fixture["legacy_repo"]),
            "--legacy-assets", str(fixture["legacy_assets"]),
            "--state-root", str(state_root), "--gh-binary", str(fixture["gh"]),
            "--gh-sha256", MIRROR._sha256_file(fixture["gh"]),
        ]
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "GH_TOKEN": "fixture-token"}
        dry = subprocess.run(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, timeout=180)
        self.assertEqual(dry.returncode, 0, dry.stderr.decode())
        self.assertFalse(json.loads(fixture["state"].read_text())["release"])
        self.assertFalse(git("--git-dir", str(fixture["target"]), "show-ref", check=False).stdout)

        state = json.loads(fixture["state"].read_text())
        state["fail_upload_calls"] = list(range(1, MIRROR.V103_MIGRATED_RELEASE_ASSET_COUNT + 1))
        self.write_json(fixture["state"], state)
        for checkpoint in range(1, MIRROR.V103_MIGRATED_RELEASE_ASSET_COUNT + 1):
            with self.subTest(upload_checkpoint=checkpoint):
                interrupted = subprocess.run(
                    [*base, "--execute", "--resume"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=environment, timeout=240,
                )
                self.assertNotEqual(interrupted.returncode, 0)
                after_failure = json.loads(fixture["state"].read_text())
                self.assertTrue(after_failure["release"]["draft"])
                self.assertEqual(len(after_failure["release"]["assets"]), checkpoint, interrupted.stderr.decode())
                self.assertEqual(after_failure["failed_upload_calls"], list(range(1, checkpoint + 1)))
                self.assertFalse((state_root / "v103-migration-terminal.json").exists())

        resumed = subprocess.run(
            [*base, "--execute", "--resume"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, timeout=300,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
        final = json.loads(fixture["state"].read_text())["release"]
        self.assertFalse(final["draft"])
        self.assertTrue(final["immutable"])
        self.assertEqual(len(final["assets"]), MIRROR.V103_MIGRATED_RELEASE_ASSET_COUNT)
        terminal = json.loads((state_root / "v103-migration-terminal.json").read_text())
        self.assertEqual(terminal["state"], "IMMUTABLE_MIGRATION_VERIFIED")
        self.assertEqual(terminal["prior_release_authority"]["tag_object"], MIRROR.V103_TARGET_TAG_OBJECT)
        self.assertIsNone(terminal["incident_resolution"])

    def test_v103_migration_controller_postpublish_crash_resume_matrix(self) -> None:
        cases = (
            ("postpublish", {"fail_after_publish_once": True}, "failed_after_publish"),
            ("download", {"fail_download_after": 7}, "failed_download"),
            ("verify-asset", {"fail_verify_asset_at": 7}, "failed_verify_asset"),
            ("preterminal", {"fail_before_terminal_publish_once": True}, "failed_before_terminal_publish"),
        )
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "GH_TOKEN": "fixture-token"}
        for label, injected, observed in cases:
            with self.subTest(crash_checkpoint=label):
                fixture = self.make_v103_controller_fixture(f"migration-{label}")
                state_root = self.root / f"migration-{label}-state"
                state_root.mkdir(mode=0o700)
                base = [
                    str(fixture["controller"]), "--audit", str(fixture["audit"]),
                    "--audit-sha256", MIRROR._sha256_file(fixture["audit"]),
                    "--legacy-repo", str(fixture["legacy_repo"]),
                    "--legacy-assets", str(fixture["legacy_assets"]),
                    "--state-root", str(state_root), "--gh-binary", str(fixture["gh"]),
                    "--gh-sha256", MIRROR._sha256_file(fixture["gh"]),
                ]
                state = json.loads(fixture["state"].read_text())
                state.update(injected)
                self.write_json(fixture["state"], state)
                interrupted = subprocess.run(
                    [*base, "--execute"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=environment, timeout=300,
                )
                self.assertNotEqual(interrupted.returncode, 0)
                failed_state = json.loads(fixture["state"].read_text())
                self.assertTrue(failed_state[observed], interrupted.stderr.decode())
                self.assertTrue(failed_state["release"]["immutable"])
                self.assertFalse((state_root / "v103-migration-terminal.json").exists())
                incident_path = state_root / "v103-migration-incident.json"
                incident_bytes = incident_path.read_bytes()
                incident = MIRROR.validate_v103_migration_incident(json.loads(incident_bytes))
                self.assertFalse(list(state_root.glob(".v103-download.*")))
                self.assertFalse(list(state_root.glob(".v103-terminal.*")))

                resumed = subprocess.run(
                    [*base, "--execute", "--resume"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=environment, timeout=300,
                )
                self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
                self.assertEqual(incident_path.read_bytes(), incident_bytes)
                terminal = MIRROR.validate_v103_migration_terminal(
                    json.loads((state_root / "v103-migration-terminal.json").read_text())
                )
                resolution = terminal["incident_resolution"]
                self.assertEqual(resolution["sha256"], hashlib.sha256(incident_bytes).hexdigest())
                self.assertEqual(resolution["incident_id"], incident["incident_id"])
                self.assertEqual(resolution["resolved_by"], "SIGNED_IMMUTABLE_MIGRATION_TERMINAL")
                self.assertFalse(list(state_root.glob(".v103-download.*")))
                self.assertFalse(list(state_root.glob(".v103-terminal.*")))

    def test_tag_attestation_replays_exact_message_and_manifest(self) -> None:
        state, _snapshot, receipt, attestation = self.make_state("tag-replay")
        local_assets = self.root / "tag-replay-assets"
        manifest = MIRROR.create_distribution_manifest(
            source_manifest_path=local_assets / self.source_manifest.name,
            asset_directory=local_assets,
            mirror_receipt_path=local_assets / "gpmeep-v1.1.0-release-only-receipt.json",
            target_history_attestation_path=local_assets / self.target_history_attestation_path.name,
        )
        expected_message = (
            "gpmeep v1.1.0 release-only mirror\n"
            f"authoritative-source-commit: {self.source_commit}\n"
            f"authoritative-source-tree: {self.source_tree}\n"
            f"snapshot-commit: {receipt['snapshot']['snapshot_commit']}\n"
            f"package-sha256: {self.source_authority['package']['sha256']}\n"
            f"candidate-receipt-id: {self.source_authority['candidate_receipt']['receipt_id']}\n"
            f"final-receipt-id: {self.source_authority['final_receipt']['receipt_id']}\n"
            f"mirror-publication-id: {receipt['publication_id']}\n"
            f"independent-mirror-audit-id: {self.target_preflight['external_mirror_audit']['audit_id']}\n"
            f"distribution-asset-manifest-id: {manifest['manifest_id']}"
        )
        message = MIRROR.tag_message(receipt, manifest)
        self.assertEqual(message, expected_message)
        self.assertEqual(message.count("distribution-asset-manifest-id:"), 1)
        replayed = MIRROR.replay_tag_attestation(
            receipt=receipt,
            distribution_manifest=manifest,
            attestation=attestation,
            publisher_repo=state / "isolated-publisher.git",
            forbidden_objects=[self.secret_commit, self.secret_blob],
        )
        self.assertEqual(replayed, attestation)
        changed = json.loads(json.dumps(manifest))
        changed["manifest_id"] = "f" * 64
        with self.assertRaises(MIRROR.ReleaseOnlyError):
            MIRROR.replay_tag_attestation(
                receipt=receipt,
                distribution_manifest=changed,
                attestation=attestation,
                publisher_repo=state / "isolated-publisher.git",
            )

    def test_production_publication_terminal_rejects_fabricated_metadata_and_fake_gh(self) -> None:
        state, _snapshot, receipt, tag_attestation = self.make_state("terminal-authority")
        assets = self.root / "terminal-authority-assets"
        manifest_path = assets / MIRROR.DISTRIBUTION_MANIFEST_NAME
        receipt_path = assets / "gpmeep-v1.1.0-release-only-receipt.json"
        manifest = MIRROR.create_distribution_manifest(
            source_manifest_path=assets / self.source_manifest.name,
            asset_directory=assets,
            mirror_receipt_path=receipt_path,
            target_history_attestation_path=assets / self.target_history_attestation_path.name,
        )
        self.write_json(manifest_path, manifest)
        fake_gh = self.root / "terminal-gh"
        fake_gh.write_bytes(b"#!/bin/sh\nprintf '%s\\n' '{\"attestation\":{},\"verificationResult\":{}}'\nexit 0\n")
        fake_gh.chmod(0o700)
        fake_remote = self.root / "terminal-fake-remote.git"
        git("init", "--bare", "--template=", str(fake_remote))
        run_intent = MIRROR.create_run_intent(
            source_repo=self.source,
            source_authority_path=self.source_authority_path,
            target_preflight_path=self.target_preflight_path,
            target_history_attestation_path=self.target_history_attestation_path,
            external_audit_path=self.external_audit,
            asset_directory=assets,
            state_root=state,
            repository=MIRROR.TARGET_REPOSITORY,
            identity=self.identity,
            forbidden_objects=[self.secret_commit, self.secret_blob],
            local_bare_remote=fake_remote,
            gh_binary=None,
            gh_sha256=None,
            gh_capabilities_sha256=None,
            trusted_external_audit_sha256=self.external_audit_sha256,
        )
        run_intent["transport"] = {
            "mode": "github",
            "remote_url": f"git@github.com:{MIRROR.TARGET_REPOSITORY}.git",
            "gh_binary": {
                "path": str(fake_gh),
                "size_bytes": fake_gh.stat().st_size,
                "sha256": MIRROR.AUDITED_GITHUB_CLI_SHA256,
            },
            "capabilities_sha256": "a" * 64,
        }
        run_intent.pop("intent_id")
        run_intent["intent_id"] = MIRROR._canonical_sha256(run_intent)
        MIRROR.validate_run_intent(run_intent)
        keywords = {
            "receipt": receipt,
            "distribution_manifest": manifest,
            "tag_attestation": tag_attestation,
            "run_intent": run_intent,
            "asset_directory": assets,
            "tag_object": tag_attestation["tag_object"],
            "gh_binary": fake_gh,
            "gh_capabilities_sha256": "a" * 64,
            "trusted_external_audit_sha256": self.external_audit_sha256,
        }
        with mock.patch.dict(os.environ, {"GH_TOKEN": "stub-token"}), self.assertRaisesRegex(
            MIRROR.ReleaseOnlyError, "audited approved binary"
        ):
            MIRROR.create_publication_terminal(**keywords)

    def test_forged_self_rehashed_external_audit_fails_trusted_digest(self) -> None:
        forged = json.loads(self.external_audit.read_text())
        forged["auditor"]["name"] = "self-appointed attacker"
        forged.pop("audit_id")
        forged["audit_id"] = MIRROR._canonical_sha256(forged)
        path = self.root / "forged-independent-audit.json"
        self.write_json(path, forged)
        with self.assertRaisesRegex(MIRROR.ReleaseOnlyError, "out-of-band trusted digest"):
            MIRROR.external_audit_record(path, self.external_audit_sha256)

    def test_rehashed_audit_identity_substitutions_fail_semantic_authority(self) -> None:
        state, snapshot, _receipt, _tag = self.make_state("audit-identity-substitution")
        del state
        mutations = {
            "source repository": lambda audit: audit["source_repository"].__setitem__("id", 999),
            "private release": lambda audit: audit["private_release"].__setitem__("node_id", "R_attacker"),
            "source tag": lambda audit: audit.__setitem__("source_tag_object", "f" * 40),
            "target repository": lambda audit: audit["target_repository"].__setitem__("node_id", "R_target_attacker"),
            "organization": lambda audit: audit["organization_access"].__setitem__("id", 999),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                assets = self.root / f"forged-{name.replace(' ', '-')}-assets"
                shutil.copytree(self.assets, assets)
                forged = json.loads(self.external_audit.read_text())
                mutate(forged)
                forged.pop("audit_id")
                forged["audit_id"] = MIRROR._canonical_sha256(forged)
                forged_path = assets / self.external_audit.name
                self.write_json(forged_path, forged)
                forged_sha256 = MIRROR._sha256_file(forged_path)
                target = json.loads(json.dumps(self.target_preflight))
                target["external_mirror_audit"] = MIRROR.external_audit_record(
                    forged_path, forged_sha256
                )
                with self.assertRaises(MIRROR.ReleaseOnlyError):
                    MIRROR.verify_external_mirror_audit(
                        target,
                        asset_directory=assets,
                        source_authority=self.source_authority,
                        snapshot=snapshot,
                        trusted_sha256=forged_sha256,
                    )

    def test_publisher_has_no_force_clobber_delete_or_build_path(self) -> None:
        text = PUBLISHER.read_text(encoding="utf-8")
        forbidden = ("--force", "--force-with-lease", "--clobber", "push --mirror", "push --all", "release delete", "api --method DELETE", "build-conda", "conda-build")
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)
        self.assertIn('"$TAG_REF:$TAG_REF"', text)
        self.assertNotIn("--remote-url", text)
        self.assertNotIn("--test-github-bare-remote", text)
        self.assertNotIn("GPMEEP_RELEASE_ONLY_TESTING", text)
        self.assertNotRegex(text, r"(?m)(?<![A-Za-z0-9_])gh (?:api|release)\b")
        self.assertNotIn("command -v gh", text)
        self.assertIn("--gh-binary", text)
        self.assertIn("--gh-sha256", text)
        self.assertIn("APPROVED_GH_SHA256=$AUDITED_GH_SHA256", text)
        self.assertIn("/usr/bin/ssh -F /dev/null -o BatchMode=yes -o ClearAllForwardings=yes", text)
        self.assertGreaterEqual(text.count("verify_audited_implementation"), 4)
        self.assertGreaterEqual(text.count("verify-published"), 2)
        self.assertIn('signed_asset_path=$(canonical_file "$DOWNLOAD_DIRECTORY/$signed_asset")', text)
        self.assertIn('release verify-asset v1.1.0 "$signed_asset_path"', text)
        self.assertNotIn('release verify-asset v1.1.0 "$signed_asset"', text)
        prepublication_download = text.index(
            'gh_call release download v1.1.0 --repo "$REPOSITORY" --dir "$DOWNLOAD_DIRECTORY"'
        )
        verify_asset = text.index('release verify-asset v1.1.0 "$signed_asset_path"')
        postpublication_download = text.index(
            'gh_call release download v1.1.0 --repo "$REPOSITORY" --dir "$POST_DIRECTORY"'
        )
        self.assertLess(prepublication_download, verify_asset)
        self.assertLess(verify_asset, postpublication_download)
        self.assertIn("X-GitHub-Api-Version: 2026-03-10", text)
        self.assertIn(".permissions.admin", text)
        self.assertEqual(text.count('MIRROR_RECEIPT=$(canonical_file "$MIRROR_RECEIPT")'), 1)
        tool_text = TOOL_PATH.read_text(encoding="utf-8")
        self.assertNotIn('terminal.add_argument("--repository-metadata"', tool_text)
        self.assertNotIn('terminal.add_argument("--release-metadata"', tool_text)
        self.assertIn('"release", "verify", DEFAULT_TAG, "--repo", TARGET_REPOSITORY,', tool_text)
        self.assertIn('"release", "view", DEFAULT_TAG, "--repo", TARGET_REPOSITORY,', tool_text)
        self.assertEqual(text.count('/usr/bin/rm -rf -- "$TEMPORARY_ROOT"'), 1)
        self.assertNotIn("TEMPORARY_DIRECTORIES", text)
        gate = text.index('if [[ "$EXECUTE" != true ]]')
        push = text.index(" push --porcelain ")
        self.assertLess(gate, push)

    def test_verify_asset_uses_existing_downloaded_file_with_empty_cwd(self) -> None:
        empty_cwd = self.root / "empty-cwd"
        empty_cwd.mkdir()
        download = self.root / "verified-download"
        download.mkdir()
        asset = download / "gpmeep-1.1.0-cuda_test.conda"
        asset.write_bytes(b"downloaded exact bytes\n")
        stub = self.root / "gh-stub"
        marker = self.root / "gh-stub-argument"
        stub.write_text(
            "#!/bin/sh\n"
            "test \"$1\" = release && test \"$2\" = verify-asset && test \"$3\" = v1.1.0 || exit 90\n"
            "test -f \"$4\" || exit 91\n"
            f"/usr/bin/printf '%s' \"$4\" >'{marker}'\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        result = subprocess.run(
            [str(stub), "release", "verify-asset", "v1.1.0", str(asset), "--repo", MIRROR.TARGET_REPOSITORY],
            cwd=empty_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(marker.read_text(), str(asset))
        self.assertFalse((empty_cwd / asset.name).exists())

    def test_receipt_tool_has_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"), filename=str(TOOL_PATH))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            literal_keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(
                len(literal_keys),
                len(set(literal_keys)),
                f"duplicate literal dictionary key near line {node.lineno}",
            )


if __name__ == "__main__":
    unittest.main()
