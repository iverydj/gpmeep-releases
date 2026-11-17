#!/usr/bin/env python3
"""Publish a reviewed 1.1.0 snapshot to the existing public release repository.

The public snapshot has public ancestry and preserves original binary source
authority. Private evidence is verified locally and is never an upload source.
No operation rebuilds, edits, deletes or replaces a package, tag or release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ARTIFACTS = module("gpmeep_public_inspection", "verify-public-artifacts.py")
SOURCE = ARTIFACTS.SOURCE
AUTHORITY = module("gpmeep_private_source_authority", "v110-release-only-receipt.py")
require = SOURCE.require
canonical = ARTIFACTS.canonical
digest = SOURCE.digest
SLUG = "iverydj/gpmeep-releases"
SOURCE_SLUG = "iverydj/gpmeep"
TARGET_ID = 1372662716
TARGET_NODE = "R_kgDOUdErvA"
TAG = "v1.1.0"
SNAPSHOT_METHOD = "public-parent-preserved-runtime-source-v1"
REMOTE = "git@github.com:iverydj/gpmeep-releases.git"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
IMPLEMENTATION = ("publish-v110-public.py", "verify-public-source.py",
                  "verify-public-artifacts.py", "v110-release-only-receipt.py",
                  "v110-final-release-receipt.py", "verify-v110-final-runtime.py")


def exact(value, fields, label):
    require(isinstance(value, dict) and set(value) == set(fields), f"{label} fields differ")
    return value


def file_bytes(path, expected):
    require(isinstance(expected, str) and HEX64.fullmatch(expected), "invalid file SHA-256 pin")
    path = ARTIFACTS.regular(Path(path))
    data = path.read_bytes()
    require(digest(data) == expected, f"pinned file changed: {path.name}")
    return data


def file_record(value):
    exact(value, {"path", "sha256"}, "pinned file")
    return file_bytes(value["path"], value["sha256"])


def directory(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            "directory must be absolute without symlink components")
    return path


def implementation():
    return {name: digest(Path(__file__).with_name(name).read_bytes()) for name in IMPLEMENTATION}


def verify_public_git_config(repo):
    """Forbid repository-local transport, import and object-store overrides."""
    git = SOURCE.git
    require(git(repo, "rev-parse", "--absolute-git-dir").decode().strip() == str(repo / ".git") and
            git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip() == str(repo / ".git"),
            "public staging must have an independent Git directory")
    allowed = {"core.repositoryformatversion": "0", "core.filemode": "true",
               "core.bare": "false", "core.logallrefupdates": "true",
               "user.name": None, "user.email": None}
    config = {}
    for record in git(repo, "config", "--local", "--no-includes", "--null", "--list").split(b"\0"):
        if not record:
            continue
        key, separator, value = record.decode().partition("\n")
        require(separator and key in allowed and key not in config, "unsafe or duplicate public Git configuration")
        require(allowed[key] is None or value == allowed[key], "unsafe public Git configuration value")
        config[key] = value
    require(config.get("core.repositoryformatversion") == "0" and config.get("core.bare") == "false",
            "public Git repository format differs")
    return digest(canonical(config))


verify_final_runtime = module("gpmeep_final_runtime", "verify-v110-final-runtime.py").verify_final_runtime


def verify_plan(plan):
    exact(plan, {"schema", "source_repo", "source_assets", "source_authority",
                 "final_verifier", "trusted_private_tag_object", "public_repo",
                 "public_commit", "public_tag_object", "prior_refs", "privacy_policy",
                 "assets", "notes", "github_cli", "release_title"}, "publication plan")
    require(plan["schema"] == "gpmeep-v110-publication-plan-v1", "plan schema differs")
    require(plan["release_title"] == "gpmeep v1.1.0", "release title differs")
    private_repo = directory(plan["source_repo"])
    public_repo = directory(plan["public_repo"])
    private_assets = directory(plan["source_assets"])
    authority = SOURCE.read_json(file_record(plan["source_authority"]))
    AUTHORITY.validate_source_authority(authority)
    verifier = exact(plan["final_verifier"], {"path", "sha256", "arguments", "python"}, "final verifier")
    file_bytes(verifier["path"], verifier["sha256"])
    runtime = verify_final_runtime(private_repo, authority["source_commit"], verifier["python"])
    public_git_config = verify_public_git_config(public_repo)
    require(isinstance(verifier["arguments"], list) and
            all(isinstance(arg, str) for arg in verifier["arguments"]), "invalid final verifier arguments")
    authority = AUTHORITY.verify_source_authority_files(
        authority, private_assets, source_repo=private_repo,
        final_verifier_path=Path(verifier["path"]), final_verifier_sha256=verifier["sha256"],
        final_verifier_arguments=verifier["arguments"],
        trusted_source_tag_object=plan["trusted_private_tag_object"])
    require(authority["repository"]["slug"] == SOURCE_SLUG, "private source repository differs")
    prior = exact(plan["prior_refs"], {"refs/heads/main", "refs/tags/v1.0.3"}, "prior public refs")
    for oid in (*prior.values(), plan["public_commit"], plan["public_tag_object"]):
        require(isinstance(oid, str) and SOURCE.HEX40.fullmatch(oid), "invalid public object pin")
    git = SOURCE.git
    parents = git(public_repo, "rev-list", "--parents", "-n", "1", plan["public_commit"]).decode().split()
    require(parents == [plan["public_commit"], prior["refs/heads/main"]],
            "snapshot must extend exactly the reviewed public main commit")
    require(git(public_repo, "rev-parse", f"refs/tags/{TAG}").decode().strip() == plan["public_tag_object"],
            "public tag object differs")
    require(git(public_repo, "cat-file", "-t", plan["public_tag_object"]).strip() == b"tag",
            "public tag must be annotated")
    require(git(public_repo, "rev-parse", f"refs/tags/{TAG}^{{commit}}").decode().strip() == plan["public_commit"],
            "public tag target differs")
    tag_bytes = git(public_repo, "cat-file", "tag", plan["public_tag_object"])
    expected_message = (f"gpmeep {TAG}\nbinary-source-commit: {authority['source_commit']}\n"
                        f"package-sha256: {authority['package']['sha256']}\n"
                        f"final-receipt-id: {authority['final_receipt']['receipt_id']}\n")
    require(tag_bytes.partition(b"\n\n")[2] == expected_message.encode(), "public tag authority message differs")
    policy = SOURCE.read_json(file_record(plan["privacy_policy"]))
    # Apply the same schema and nonempty-inventory check as the inspection CLI.
    require(policy == ARTIFACTS.policy_read(Path(plan["privacy_policy"]["path"])), "privacy policy changed")
    source = SOURCE.verify(private_repo, authority["source_commit"], public_repo, plan["public_commit"], **policy)
    metadata = SOURCE.read_json(git(public_repo, "show", f"{plan['public_commit']}:PUBLIC_RELEASE.json"))
    require(metadata.get("binary_source_commit") == authority["source_commit"] and
            metadata.get("package_sha256") == authority["package"]["sha256"] and
            metadata.get("version") == "1.1.0", "public metadata substitutes binary authority")
    require(isinstance(plan["assets"], list) and plan["assets"], "public asset list is empty")
    allowed = {"package": ".conda", "report_markdown": ".md", "report_html": ".html",
               "checksums": ".sha256", "public_metadata": ".json"}
    assets = {}
    names = set()
    for item in plan["assets"]:
        exact(item, {"role", "path", "sha256", "size_bytes"}, "public asset")
        role = item["role"]
        require(role in allowed and role not in assets, "public asset role is forbidden or duplicate")
        data = file_bytes(item["path"], item["sha256"])
        path = Path(item["path"])
        require(path.is_absolute() and SAFE_NAME.fullmatch(path.name) and path.name not in names,
                "invalid or duplicate public asset name")
        require(path.suffix == allowed[role] and type(item["size_bytes"]) is int and
                len(data) == item["size_bytes"] > 0, "public asset size or format differs")
        # Selecting a private receipt/log directory as public content is forbidden.
        if role != "package":
            require(not path.is_relative_to(private_assets), "public report came from private asset directory")
        assets[role] = item
        names.add(path.name)
    require({"package", "report_markdown", "report_html", "checksums", "public_metadata"} == set(assets),
            "public asset roles are incomplete")
    package = authority["package"]
    require(assets["package"]["sha256"] == package["sha256"] and
            assets["package"]["size_bytes"] == package["size_bytes"] and
            Path(assets["package"]["path"]).name == package["name"], "public package differs from qualified bytes")
    require(SOURCE.read_json(file_bytes(assets["public_metadata"]["path"], assets["public_metadata"]["sha256"])) == metadata,
            "public metadata asset differs from snapshot")
    checksums = "".join(f"{item['sha256']}  {Path(item['path']).name}\n"
                        for role, item in sorted(assets.items(), key=lambda pair: Path(pair[1]["path"]).name)
                        if role != "checksums")
    require(file_bytes(assets["checksums"]["path"], assets["checksums"]["sha256"]) == checksums.encode(),
            "public checksums do not bind exactly the other public assets")
    notes = file_record(plan["notes"])
    ARTIFACTS.scan("release-notes.md", notes, policy)
    require(notes.strip() and len(notes) <= 125000, "release notes are empty or too large")
    for text in (authority["source_commit"], package["sha256"]):
        require(text.encode() in notes, "release notes omit original binary identity")
    file_record(plan["github_cli"])
    inspection = ARTIFACTS.verify(policy, package=Path(assets["package"]["path"]),
        assets=[Path(item["path"]) for role, item in assets.items() if role != "package"],
        public_repo=public_repo, private_repo=private_repo)
    expected_local_refs = dict(prior)
    expected_local_refs.update({"refs/heads/main": plan["public_commit"],
                                f"refs/tags/{TAG}": plan["public_tag_object"]})
    require(inspection["history"]["refs"] == expected_local_refs,
            "staging must contain only reviewed main, prior tag and candidate tag")
    report = {"schema": "gpmeep-v110-publication-review-v1", "state": "READY_FOR_INDEPENDENT_AUDIT",
              "publication_authorized": False, "plan_id": digest(canonical(plan)),
              "snapshot_method": SNAPSHOT_METHOD,
              "final_verifier_runtime": runtime, "public_git_config_sha256": public_git_config,
              "source_authority_sha256": plan["source_authority"]["sha256"],
              "source_report_id": source["report_sha256"], "artifact_report_id": inspection["report_id"],
              "implementation": implementation(), "target": {"slug": SLUG, "id": TARGET_ID, "node_id": TARGET_NODE}}
    report["report_id"] = digest(canonical(report))
    return report, authority, assets, notes.decode("utf-8")


def verify_audit(audit, report, plan_sha256):
    exact(audit, {"schema", "state", "plan_sha256", "review_report_id", "implementation",
                  "findings", "checks", "reviewer", "audit_id"}, "independent publication audit")
    unsigned = dict(audit)
    audit_id = unsigned.pop("audit_id")
    require(audit_id == digest(canonical(unsigned)), "independent audit content identity differs")
    require(audit["schema"] == "gpmeep-v110-publication-audit-v1" and audit["state"] == "GO",
            "independent publication audit is not GO")
    require(audit["plan_sha256"] == plan_sha256 and audit["review_report_id"] == report["report_id"] and
            audit["implementation"] == report["implementation"], "independent audit is stale")
    require(audit["findings"] == {"critical": 0, "high": 0, "medium": 0}, "publication findings remain open")
    require(all(type(value) is int for value in audit["findings"].values()), "invalid finding counts")
    require(audit["checks"] == {"documentation_and_media_reviewed": True,
                                "remote_history_and_cached_objects_reviewed": True,
                                "exact_package_authority_replayed": True,
                                "publication_controller_adversarially_reviewed": True},
            "independent publication checks incomplete")
    require(all(value is True for value in audit["checks"].values()), "invalid independent check results")
    require(isinstance(audit["reviewer"], str) and audit["reviewer"].strip(), "audit reviewer is absent")
    return audit


class GitHub:
    def __init__(self, cli):
        self.cli = cli

    def run(self, args, *, data=None, missing=False):
        file_record(self.cli)
        # Preserve credentials only; caller GH_HOST/REPO/debug/config overrides
        # cannot redirect requests or disclose authentication in command output.
        env = {"PATH": "/usr/bin:/bin", "HOME": os.path.expanduser("~"), "LANG": "C.UTF-8",
               "GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1"}
        for key in ("GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"):
            if key in os.environ:
                env[key] = os.environ[key]
        result = subprocess.run([self.cli["path"], *args], input=data, capture_output=True,
                                env=env, timeout=1800)
        if missing and result.returncode and b"HTTP 404" in result.stderr:
            return None
        require(result.returncode == 0, f"GitHub operation failed: {args[0]}")
        return result.stdout

    def api(self, endpoint, *, method="GET", body=None, missing=False):
        args = ["api", "--hostname", "github.com", endpoint, "--method", method]
        data = None
        if body is not None:
            args += ["--input", "-"]
            data = canonical(body)
        raw = self.run(args, data=data, missing=missing)
        return None if raw is None else (SOURCE.read_json(raw) if raw.strip() else {})

    def target(self):
        metadata = self.api(f"repos/{SLUG}")
        require(metadata.get("full_name") == SLUG and metadata.get("id") == TARGET_ID and
                metadata.get("node_id") == TARGET_NODE and metadata.get("private") is False and
                metadata.get("archived") is False and metadata.get("default_branch") == "main",
                "live public repository identity differs")
        return metadata

    def refs(self):
        result = {}
        # GitHub REST matching-refs also exposes unexpected heads/tags. Inspect
        # all advertised Git refs independently so PR refs cannot be ignored.
        raw = subprocess.run(["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "ls-remote", "--refs", REMOTE],
                             capture_output=True, timeout=120, cwd="/",
                             env={"PATH": "/usr/bin:/bin", "HOME": os.path.expanduser("~"),
                                  "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
                                  "GIT_TERMINAL_PROMPT": "0"})
        require(raw.returncode == 0, "live public ref inventory failed")
        for line in raw.stdout.decode().splitlines():
            oid, ref = line.split()
            require(ref not in result, "duplicate remote ref")
            result[ref] = oid
        return result


def release_assets(release, expected, *, allow_subset):
    records = release.get("assets")
    require(isinstance(records, list), "remote release asset inventory is absent")
    names = set()
    for record in records:
        name = record.get("name")
        require(name in expected and name not in names, "unreviewed or duplicate remote asset")
        names.add(name)
        local = expected[name]
        require(record.get("state") == "uploaded" and record.get("size") == local["size_bytes"] and
                record.get("digest") == "sha256:" + local["sha256"], "remote asset digest or size differs")
    require(allow_subset or names == set(expected), "remote release omits reviewed assets")
    return names


def publish(plan, authority, assets, notes, *, execute, resume):
    gh = GitHub(plan["github_cli"])
    public_git_config = verify_public_git_config(Path(plan["public_repo"]))
    gh.target()
    source_meta = gh.api(f"repos/{SOURCE_SLUG}")
    require(source_meta.get("private") is True and
            {key: source_meta.get(api_key) for key, api_key in (("slug", "full_name"), ("id", "id"), ("node_id", "node_id"))}
            == authority["repository"], "live private source identity differs")
    private_release = gh.api(f"repos/{SOURCE_SLUG}/releases/tags/{TAG}")
    require(private_release.get("id") == authority["private_release"]["id"] and
            private_release.get("node_id") == authority["private_release"]["node_id"] and
            private_release.get("tag_name") == TAG and
            private_release.get("draft") is False and private_release.get("prerelease") is False and
            private_release.get("immutable") is True, "private source release authority differs")
    gh.run(["release", "verify", TAG, "--repo", SOURCE_SLUG, "--format", "json"])
    source_root = directory(plan["source_assets"])
    manifest_record = authority["source_asset_manifest"]
    manifest_path = source_root / manifest_record["name"]
    file_bytes(manifest_path, manifest_record["sha256"])
    private_records = AUTHORITY.verify_private_live_assets(
        private_release, SOURCE.read_json(manifest_path.read_bytes()), manifest_path)
    for name, record in sorted(private_records.items()):
        # The complete evidence archive can exceed public inspection limits.
        # Stream its digest without reading the private bundle into memory.
        path = AUTHORITY._regular_file(source_root / name, "private asset")
        require(path.stat().st_size == record["size_bytes"] and
                AUTHORITY._sha256_file(path) == record["sha256"], "private asset bytes changed")
        gh.run(["release", "verify-asset", TAG, str(path), "--repo", SOURCE_SLUG, "--format", "json"])
    source_tag = gh.api(f"repos/{SOURCE_SLUG}/git/ref/tags/{TAG}")
    require(source_tag.get("object", {}).get("sha") == plan["trusted_private_tag_object"],
            "live private source tag differs")
    prior = plan["prior_refs"]
    final_refs = {**prior, "refs/heads/main": plan["public_commit"], f"refs/tags/{TAG}": plan["public_tag_object"]}
    refs = gh.refs()
    require(refs == prior or (resume and refs == final_refs), "public remote refs changed or are unreviewed")
    previous_release = gh.api(f"repos/{SLUG}/releases/tags/v1.0.3")
    require(previous_release.get("draft") is False and previous_release.get("prerelease") is False and
            previous_release.get("assets") == [], "prior source-only release changed")
    current = gh.api(f"repos/{SLUG}/releases/tags/{TAG}", missing=True)
    require(current is None or resume, "release already exists; explicit resume required")
    expected = {Path(item["path"]).name: item for item in assets.values()}

    def check_release(release, *, subset):
        require(release.get("tag_name") == TAG and release.get("name") == plan["release_title"] and
                release.get("body") == notes and release.get("prerelease") is False,
                "remote release metadata differs")
        return release_assets(release, expected, allow_subset=subset)

    if current is not None:
        check_release(current, subset=current.get("draft") is True)
        require(current.get("draft") is True or current.get("immutable") is True,
                "existing published release is mutable")
    if not execute:
        return {"state": "LIVE_PREFLIGHT_PASS", "mutations": False}
    immutable = gh.api(f"repos/{SLUG}/immutable-releases")
    if immutable.get("enabled") is not True:
        gh.api(f"repos/{SLUG}/immutable-releases", method="PUT")
    require(gh.api(f"repos/{SLUG}/immutable-releases").get("enabled") is True,
            "immutable releases are not enabled")
    permissions = gh.api(f"repos/{SLUG}/actions/permissions")
    if permissions.get("enabled") is not False:
        gh.api(f"repos/{SLUG}/actions/permissions", method="PUT", body={"enabled": False})
    require(gh.api(f"repos/{SLUG}/actions/permissions").get("enabled") is False,
            "public target Actions must be disabled")
    gh.target()
    require(gh.refs() == refs, "remote refs changed before publication")
    require(verify_public_git_config(Path(plan["public_repo"])) == public_git_config,
            "public Git configuration changed before push")
    if refs == prior:
        # Both refs are an atomic fast-forward/addition; never force or import
        # development refs. Public-history verification ran before this push.
        SOURCE.git(Path(plan["public_repo"]), "push", "--atomic", REMOTE,
                   f"{plan['public_commit']}:refs/heads/main", f"{plan['public_tag_object']}:refs/tags/{TAG}")
    require(gh.refs() == final_refs, "public refs failed exact post-push verification")
    if current is None:
        current = gh.api(f"repos/{SLUG}/releases", method="POST", body={
            "tag_name": TAG, "target_commitish": plan["public_commit"],
            "name": plan["release_title"], "body": notes, "draft": True, "prerelease": False})
    if current.get("draft") is True:
        uploaded = check_release(current, subset=True)
        for name, item in sorted(expected.items()):
            file_bytes(item["path"], item["sha256"])
            if name not in uploaded:
                # Upload a verified private copy. A user editing an original
                # report after review cannot substitute it during gh's read.
                data = file_bytes(item["path"], item["sha256"])
                with tempfile.TemporaryDirectory(prefix="gpmeep-public-upload-") as staging:
                    staged = Path(staging) / name
                    staged.write_bytes(data)
                    staged.chmod(0o400)
                    gh.run(["release", "upload", TAG, str(staged), "--repo", SLUG])
        current = gh.api(f"repos/{SLUG}/releases/tags/{TAG}")
        check_release(current, subset=False)
        for item in assets.values():
            file_bytes(item["path"], item["sha256"])
        require(gh.refs() == final_refs and gh.api(f"repos/{SLUG}/immutable-releases").get("enabled") is True,
                "publication state changed before finalizing draft")
        current = gh.api(f"repos/{SLUG}/releases/{current['id']}", method="PATCH",
                         body={"draft": False, "make_latest": "true"})
    require(current.get("immutable") is True and current.get("draft") is False,
            "final public release is not immutable")
    check_release(current, subset=False)
    gh.run(["release", "verify", TAG, "--repo", SLUG, "--format", "json"])
    for item in assets.values():
        file_bytes(item["path"], item["sha256"])
        gh.run(["release", "verify-asset", TAG, item["path"], "--repo", SLUG, "--format", "json"])
    require(gh.refs() == final_refs, "final public refs differ")
    return {"state": "PUBLISHED_AND_VERIFIED", "repository": SLUG, "release_id": current["id"],
            "public_commit": plan["public_commit"], "binary_source_commit": authority["source_commit"],
            "package_sha256": authority["package"]["sha256"], "immutable": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.resume or args.execute, "resume requires execute")
        require(not (args.offline and args.execute), "offline mode cannot publish")
        require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
        plan = SOURCE.read_json(file_bytes(args.plan, args.plan_sha256))
        report, authority, assets, notes = verify_plan(plan)
        if args.execute:
            require(args.audit is not None and args.audit_sha256 is not None,
                    "execution requires independently pinned publication audit")
            audit = SOURCE.read_json(file_bytes(args.audit, args.audit_sha256))
            verify_audit(audit, report, args.plan_sha256)
        if not args.offline:
            result = publish(plan, authority, assets, notes, execute=args.execute, resume=args.resume)
            report = {"review": report, "result": result}
            if args.execute:
                report["independent_audit"] = {"audit_id": audit["audit_id"],
                                               "sha256": args.audit_sha256}
        # Never overwrite earlier evidence, even on a resumed publication.
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
                staged = Path(stream.name)
                stream.write(json.dumps(report, sort_keys=True, indent=2).encode() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, args.output)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        terminal = report["result"] if "result" in report else {
            "state": report["state"], "report_id": report["report_id"]}
        print(json.dumps(terminal))
        return 0
    except (SOURCE.PublicSourceError, AUTHORITY.ReleaseOnlyError, OSError, ValueError,
            subprocess.SubprocessError) as exc:
        parser.exit(1, f"public publication stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
