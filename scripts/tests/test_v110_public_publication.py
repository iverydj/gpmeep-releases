"""Exercise reviewed public publication without network or real releases."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "public_publish", Path(__file__).resolve().parents[1] / "publish-v110-public.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class PublicPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.repo("private")
        self.public = self.repo("public")
        self.private_assets = self.root / "private-assets"
        self.private_assets.mkdir()
        self.assets = self.root / "public-assets"
        self.assets.mkdir()
        (self.private / "src.cpp").write_text("void step() {}\n")
        (self.private / "README.md").write_text("private-machine\n")
        self.source_commit = self.commit(self.private)
        (self.public / "README.md").write_text("Previous public release\n")
        self.prior_commit = self.commit(self.public)
        self.git(self.public, "tag", "-a", "v1.0.3", "-m", "public prior release")
        self.prior_tag = self.git(self.public, "rev-parse", "refs/tags/v1.0.3")
        package = self.private_assets / "gpmeep-1.1.0-cuda_test.conda"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("metadata.json", '{"conda_pkg_format_version":2}')
            archive.writestr("info-fixture.tar.zst", self.tar("info/index.json", b"{}"))
            archive.writestr("pkg-fixture.tar.zst", self.tar("lib/solver.so", b"\x7fELF\x00public"))
        self.package_sha = TOOL.digest(package.read_bytes())
        self.authority = {
            "source_commit": self.source_commit,
            "package": {"name": package.name, "size_bytes": package.stat().st_size, "sha256": self.package_sha},
            "final_receipt": {"receipt_id": "e" * 64},
            "repository": {"slug": TOOL.SOURCE_SLUG, "id": 1, "node_id": "SOURCE"},
            "private_release": {"id": 2, "node_id": "PRIVATE_RELEASE"}}
        manifest = self.private_assets / "source-assets.json"
        manifest_data = {"schema": "gpmeep-v110-release-asset-manifest-v1", "assets": [
            {"role": "package", "name": package.name, "size_bytes": package.stat().st_size,
             "sha256": self.package_sha}]}
        manifest_data["manifest_id"] = TOOL.digest(TOOL.canonical(manifest_data))
        manifest.write_bytes(TOOL.canonical(manifest_data))
        self.authority["source_asset_manifest"] = {"name": manifest.name,
            "size_bytes": manifest.stat().st_size, "sha256": TOOL.digest(manifest.read_bytes())}
        self.metadata = {"version": "1.1.0", "binary_source_commit": self.source_commit,
                         "package_sha256": self.package_sha}
        (self.public / "src.cpp").write_bytes((self.private / "src.cpp").read_bytes())
        (self.public / "README.md").write_text("Public release\n")
        (self.public / "PUBLIC_RELEASE.json").write_bytes(TOOL.canonical(self.metadata))
        public_commit = self.commit(self.public)
        message = (f"gpmeep v1.1.0\nbinary-source-commit: {self.source_commit}\n"
                   f"package-sha256: {self.package_sha}\nfinal-receipt-id: {'e' * 64}")
        self.git(self.public, "tag", "-a", "v1.1.0", "-m", message)
        self.authority_file = self.root / "authority.json"
        self.authority_file.write_bytes(TOOL.canonical(self.authority))
        policy = self.root / "privacy.json"
        policy.write_bytes(TOOL.canonical({"forbidden_literals": ["private-machine"], "fixture_replacements": {}}))
        final_verifier = self.root / "final.py"
        final_verifier.write_text("# pinned verifier fixture\n")
        gh = self.root / "gh"
        gh.write_text("# pinned GitHub CLI fixture\n")
        notes = self.root / "notes.md"
        notes.write_text(f"gpmeep v1.1.0\n{self.source_commit}\n{self.package_sha}\n")
        roles = [self.asset("package", package)]
        for role, name, data in (("report_markdown", "report.md", b"Reviewed report\n"),
                                  ("report_html", "report.html", b"<p>Reviewed report</p>\n"),
                                  ("public_metadata", "PUBLIC_RELEASE.json", TOOL.canonical(self.metadata))):
            path = self.assets / name
            path.write_bytes(data)
            roles.append(self.asset(role, path))
        checksum = self.assets / "SHA256SUMS.sha256"
        checksum.write_text("".join(f"{item['sha256']}  {Path(item['path']).name}\n"
                                   for item in sorted(roles, key=lambda row: Path(row["path"]).name)))
        roles.append(self.asset("checksums", checksum))
        self.plan = {"schema": "gpmeep-v110-publication-plan-v1", "source_repo": str(self.private),
                     "source_assets": str(self.private_assets), "source_authority": self.pin(self.authority_file),
                     "final_verifier": {**self.pin(final_verifier), "arguments": [], "python": self.pin(Path(sys.executable).resolve())},
                     "trusted_private_tag_object": "f" * 40, "public_repo": str(self.public),
                     "public_commit": public_commit,
                     "public_tag_object": self.git(self.public, "rev-parse", "refs/tags/v1.1.0"),
                     "prior_refs": {"refs/heads/main": self.prior_commit, "refs/tags/v1.0.3": self.prior_tag},
                     "privacy_policy": self.pin(policy), "assets": roles, "notes": self.pin(notes),
                     "github_cli": self.pin(gh), "release_title": "gpmeep v1.1.0"}
        patch = mock.patch.object(TOOL.AUTHORITY, "verify_source_authority_files", return_value=self.authority)
        self.authority_replay = patch.start()
        self.addCleanup(patch.stop)
        for target, name, result in ((TOOL.AUTHORITY, "validate_source_authority", self.authority),
                                     (TOOL, "verify_final_runtime", {"fixture": True})):
            patch = mock.patch.object(target, name, return_value=result)
            patch.start()
            self.addCleanup(patch.stop)

    def tar(self, name, data):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        return subprocess.run(["/usr/bin/zstd", "-q", "-c"], input=stream.getvalue(),
                              capture_output=True, check=True).stdout

    def git(self, repo, *args):
        return subprocess.run(["/usr/bin/git", "-C", str(repo), *args],
                              capture_output=True, check=True).stdout.decode().strip()

    def repo(self, name):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, "init", "-q", "-b", "main")
        self.git(repo, "config", "user.name", "Test")
        self.git(repo, "config", "user.email", "test@example.invalid")
        return repo

    def commit(self, repo):
        self.git(repo, "add", "--all")
        self.git(repo, "commit", "-qm", "fixture")
        return self.git(repo, "rev-parse", "HEAD")

    def pin(self, path):
        return {"path": str(path), "sha256": TOOL.digest(path.read_bytes())}

    def asset(self, role, path):
        return {"role": role, **self.pin(path), "size_bytes": path.stat().st_size}

    def review(self):
        return TOOL.verify_plan(self.plan)

    def test_offline_review_replays_authority_and_real_git_archive_checks(self):
        report, authority, _, _ = self.review()
        self.authority_replay.assert_called_once()
        self.assertFalse(report["publication_authorized"])
        self.assertEqual(authority, self.authority)

    def test_cli_offline_result_is_exclusive_and_machine_readable(self):
        plan_path = self.root / "plan.json"
        plan_path.write_bytes(TOOL.canonical(self.plan))
        output = self.root / "review.json"
        argv = ["publisher", "--plan", str(plan_path), "--plan-sha256", self.pin(plan_path)["sha256"],
                "--offline", "--output", str(output)]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(TOOL.main(), 0)
        self.assertEqual(json.loads(stdout.getvalue())["state"], "READY_FOR_INDEPENDENT_AUDIT")
        before = output.read_bytes()
        with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            TOOL.main()
        self.assertEqual(output.read_bytes(), before)

    def test_cli_execute_without_pinned_audit_never_calls_publisher(self):
        plan_path = self.root / "plan.json"
        plan_path.write_bytes(TOOL.canonical(self.plan))
        argv = ["publisher", "--plan", str(plan_path), "--plan-sha256", self.pin(plan_path)["sha256"],
                "--execute", "--output", str(self.root / "result.json")]
        with mock.patch("sys.argv", argv), mock.patch.object(TOOL, "publish") as publish, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            TOOL.main()
        publish.assert_not_called()

    def test_original_authority_failure_is_not_replaced_by_public_checks(self):
        self.authority_replay.side_effect = TOOL.AUTHORITY.ReleaseOnlyError("final authority failed")
        with self.assertRaisesRegex(ValueError, "authority failed"):
            self.review()

    def test_changed_asset_and_unreviewed_private_asset_are_rejected(self):
        item = self.plan["assets"][1]
        Path(item["path"]).write_text("changed after review")
        with self.assertRaisesRegex(ValueError, "pinned file changed"):
            self.review()
        self.plan["assets"].append({**item, "role": "complete_evidence"})
        item.update(self.pin(Path(item["path"])), size_bytes=Path(item["path"]).stat().st_size)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.review()

    def test_extra_staging_ref_is_rejected(self):
        self.git(self.public, "branch", "unreviewed")
        with self.assertRaisesRegex(ValueError, "only reviewed"):
            self.review()

    def test_incorrect_public_parent_is_rejected(self):
        self.plan["prior_refs"]["refs/heads/main"] = self.source_commit
        with self.assertRaisesRegex(ValueError, "exactly the reviewed public"):
            self.review()

    def test_audit_must_bind_exact_plan_report_and_implementation(self):
        report = self.review()[0]
        audit = {"schema": "gpmeep-v110-publication-audit-v1", "state": "GO",
                 "plan_sha256": "a" * 64, "review_report_id": report["report_id"],
                 "implementation": report["implementation"], "findings": {"critical": 0, "high": 0, "medium": 0},
                 "checks": {"documentation_and_media_reviewed": True,
                            "remote_history_and_cached_objects_reviewed": True,
                            "exact_package_authority_replayed": True,
                            "publication_controller_adversarially_reviewed": True}, "reviewer": "independent fixture"}
        audit["audit_id"] = TOOL.digest(TOOL.canonical(audit))
        TOOL.verify_audit(audit, report, "a" * 64)
        for field, value in (("state", "NO_GO"), ("plan_sha256", "b" * 64),
                             ("implementation", {}), ("review_report_id", "c" * 64)):
            bad = copy.deepcopy(audit)
            bad.pop("audit_id")
            bad[field] = value
            bad["audit_id"] = TOOL.digest(TOOL.canonical(bad))
            with self.subTest(field=field), self.assertRaises(ValueError):
                TOOL.verify_audit(bad, report, "a" * 64)

    def fake_github(self):
        case = self

        class Fake:
            def __init__(self):
                self.remote_refs = dict(case.plan["prior_refs"])
                self.current = None
                self.writes = []
                self.immutable = False
                self.actions = True
                self.commands = []
                self.fail_upload = False
                self.private_assets = [{"name": path.name, "size": path.stat().st_size,
                    "digest": "sha256:" + TOOL.digest(path.read_bytes()), "state": "uploaded"}
                    for path in case.private_assets.iterdir()]

            def target(self):
                return {"id": TOOL.TARGET_ID}

            def refs(self):
                return dict(self.remote_refs)

            def run(self, args, **kwargs):
                self.commands.append(args)
                if args[:2] == ["release", "upload"]:
                    if self.fail_upload:
                        raise ValueError("upload interrupted")
                    path = Path(args[3])
                    self.current["assets"].append({"name": path.name, "state": "uploaded",
                        "size": path.stat().st_size, "digest": "sha256:" + TOOL.digest(path.read_bytes())})
                return b"{}"

            def api(self, endpoint, *, method="GET", body=None, missing=False):
                if method != "GET":
                    self.writes.append((endpoint, method, body))
                if endpoint == f"repos/{TOOL.SOURCE_SLUG}":
                    return {"private": True, "full_name": TOOL.SOURCE_SLUG, "id": 1, "node_id": "SOURCE"}
                if endpoint == f"repos/{TOOL.SOURCE_SLUG}/releases/tags/v1.1.0":
                    return {"id": 2, "node_id": "PRIVATE_RELEASE", "tag_name": TOOL.TAG, "draft": False, "prerelease": False, "immutable": True, "assets": self.private_assets}
                if endpoint == f"repos/{TOOL.SOURCE_SLUG}/git/ref/tags/v1.1.0":
                    return {"object": {"sha": case.plan["trusted_private_tag_object"]}}
                if endpoint.endswith("/releases/tags/v1.0.3"):
                    return {"draft": False, "prerelease": False, "assets": []}
                if endpoint.endswith("/releases/tags/v1.1.0"):
                    return copy.deepcopy(self.current)
                if endpoint.endswith("/immutable-releases"):
                    if method == "PUT":
                        self.immutable = True
                    return {"enabled": self.immutable}
                if endpoint.endswith("/actions/permissions"):
                    if method == "PUT":
                        self.actions = body["enabled"]
                    return {"enabled": self.actions}
                if endpoint.endswith("/releases") and method == "POST":
                    self.current = {**body, "id": 100, "assets": [], "immutable": False}
                    return copy.deepcopy(self.current)
                if endpoint.endswith("/releases/100") and method == "PATCH":
                    self.current.update(draft=False, immutable=True)
                    return copy.deepcopy(self.current)
                raise AssertionError(endpoint)

        return Fake()

    def publish(self, fake, execute=True, resume=False):
        _, authority, assets, notes = self.review()
        real_git = TOOL.SOURCE.git
        def push(repo, *args):
            if args[0] != "push":
                return real_git(repo, *args)
            self.assertEqual(args[:2], ("push", "--atomic"))
            self.assertNotIn("--force", args)
            self.assertIn(self.plan["public_tag_object"] + ":refs/tags/v1.1.0", args)
            fake.remote_refs.update({"refs/heads/main": self.plan["public_commit"],
                                      "refs/tags/v1.1.0": self.plan["public_tag_object"]})
            return b""
        with mock.patch.object(TOOL, "GitHub", return_value=fake), mock.patch.object(TOOL.SOURCE, "git", side_effect=push):
            return TOOL.publish(self.plan, authority, assets, notes, execute=execute, resume=resume)

    def test_live_dry_run_has_no_mutations(self):
        fake = self.fake_github()
        self.assertEqual(self.publish(fake, execute=False)["state"], "LIVE_PREFLIGHT_PASS")
        self.assertEqual(fake.writes, [])
        self.assertIsNone(fake.current)

    def test_atomic_publication_uploads_only_reviewed_bytes_and_verifies_all(self):
        fake = self.fake_github()
        self.assertEqual(self.publish(fake)["state"], "PUBLISHED_AND_VERIFIED")
        self.assertEqual(len(fake.current["assets"]), 5)
        self.assertTrue(fake.current["immutable"])
        self.assertFalse(fake.actions)
        verified = [cmd for cmd in fake.commands if cmd[:2] == ["release", "verify-asset"]]
        self.assertEqual(len(verified), 7)  # private package + manifest and all public assets

    def test_existing_publication_requires_resume_and_is_not_reuploaded(self):
        fake = self.fake_github()
        self.publish(fake)
        writes = len(fake.writes)
        commands = len(fake.commands)
        with self.assertRaisesRegex(ValueError, "resume|unreviewed"):
            self.publish(fake)
        self.assertEqual(self.publish(fake, resume=True)["state"], "PUBLISHED_AND_VERIFIED")
        self.assertEqual(len(fake.writes), writes)
        self.assertFalse(any(cmd[:2] == ["release", "upload"] for cmd in fake.commands[commands:]))

    def test_interrupted_upload_can_resume_without_deleting_or_replacing(self):
        fake = self.fake_github()
        fake.fail_upload = True
        with self.assertRaisesRegex(ValueError, "interrupted"):
            self.publish(fake)
        self.assertTrue(fake.current["draft"])
        fake.fail_upload = False
        self.assertEqual(self.publish(fake, resume=True)["state"], "PUBLISHED_AND_VERIFIED")
        self.assertEqual(sum(method == "POST" for _, method, _ in fake.writes), 1)

    def test_remote_ref_drift_stops_before_writes(self):
        fake = self.fake_github()
        fake.remote_refs["refs/pull/1/head"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "remote refs"):
            self.publish(fake)
        self.assertEqual(fake.writes, [])

    def test_local_transport_configuration_is_rejected_before_review(self):
        for key, value in (("url.file:///tmp/redirect.insteadOf", TOOL.REMOTE),
                           ("core.sshCommand", "false"), ("include.path", "/tmp/config"),
                           ("remote.origin.url", TOOL.REMOTE)):
            with self.subTest(key=key):
                self.git(self.public, "config", key, value)
                with self.assertRaisesRegex(ValueError, "unsafe.*configuration"):
                    self.review()
                self.git(self.public, "config", "--unset", key)

    def test_private_remote_asset_byte_or_state_drift_stops_before_writes(self):
        for key, value in (("digest", "sha256:" + "0" * 64), ("size", 1), ("state", "starter")):
            fake = self.fake_github()
            fake.private_assets[-1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "live asset bytes or state"):
                self.publish(fake)
            self.assertEqual(fake.writes, [])
        fake = self.fake_github()
        fake.private_assets.pop()
        with self.assertRaisesRegex(ValueError, "live asset inventory"):
            self.publish(fake)
        self.assertEqual(fake.writes, [])

    def test_remote_asset_mismatch_stops_resume(self):
        fake = self.fake_github()
        self.publish(fake)
        fake.current["assets"][0]["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            self.publish(fake, resume=True)


if __name__ == "__main__":
    unittest.main()
