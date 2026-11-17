"""Real archive/history adversarial checks for the public privacy inspector."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import struct
import tarfile
import tempfile
import unittest
from unittest import mock
import urllib.request
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "public_artifacts", Path(__file__).resolve().parents[1] / "verify-public-artifacts.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
POLICY = {"forbidden_literals": ["private-machine"], "fixture_replacements": {}}


class PublicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def scan(self, name, data):
        return TOOL.scan(name, data, POLICY)

    def tar(self, names, *, owner="builder"):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, data in names:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.uname = owner
                archive.addfile(info, io.BytesIO(data))
        return subprocess.run(["/usr/bin/zstd", "-q", "-c"], input=buffer.getvalue(),
                              capture_output=True, check=True).stdout

    def package(self, info=None, payload=None, owner="builder"):
        path = self.root / "candidate.conda"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", '{"conda_pkg_format_version":2}')
            archive.writestr("info-test.tar.zst", self.tar(
                info or [("info/index.json", b'{}')], owner=owner))
            archive.writestr("pkg-test.tar.zst", self.tar(
                payload or [("lib/solver.so", b'\x7fELF\x00public')]))
        return path

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

    def commit(self, repo, name, data):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        self.git(repo, "add", "--all")
        self.git(repo, "commit", "-qm", f"update {name}")
        return self.git(repo, "rev-parse", "HEAD")

    def test_plain_public_artifact_has_no_release_authority(self):
        path = self.root / "summary.md"
        path.write_text("Numerical validation passed.\n")
        result = TOOL.verify(POLICY, assets=[path])
        self.assertEqual(result["state"], "LITERALS_CHECKED")
        self.assertFalse(result["publication_authorized"])

    def test_secret_in_filename_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "private literal"):
            self.scan("private-machine.md", b"safe")

    def test_json_escaped_literal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "private literal"):
            self.scan("report.json", b'{"host":"\\u0070rivate-machine"}')

    def test_escaped_base64_characters_preserve_media_identity(self):
        payload = b"public-bytes"
        encoded = base64.b64encode(payload)
        plain = b"data:image/png;base64," + encoded
        escaped_json = b'{"image":"' + plain.replace(b"V", b"\\u0056") + b'"}'
        escaped_html = b'<img src="' + plain.replace(b"V", b"&#86;") + b'">'
        for name, data in (("report.json", escaped_json), ("report.html", escaped_html)):
            with self.subTest(name=name):
                media = self.scan(name, data)["embedded_media"]
                self.assertEqual(len(media), 1)
                self.assertEqual(media[0]["sha256"], TOOL.digest(payload))

    def test_html_and_url_escapes_are_rejected(self):
        for data in (b"&#112;rivate-machine", b"%70rivate-machine"):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "private literal"):
                self.scan("report.html", data)

    def test_embedded_data_url_is_inspected_and_inventoried(self):
        bad = b'<img src="data:image/png;base64,' + base64.b64encode(b"private-machine") + b'">'
        with self.assertRaisesRegex(ValueError, "private literal"):
            self.scan("report.html", bad)
        good = b'<img src="data:image/png;base64,' + base64.b64encode(b"public picture bytes") + b'">'
        result = self.scan("report.html", good)
        self.assertEqual(result["embedded_media"][0]["mime_type"], "image/png")

    def test_decoded_json_and_html_data_urls_are_inspected(self):
        encoded = base64.b64encode(b"private-machine").decode()
        values = [("report.json", ('{"image":"\\u0064ata:image/png;base64,' + encoded + '"}').encode()),
                  ("report.html", ('<img src="data:image/png;base64,&#' + str(ord(encoded[0])) + ';' + encoded[1:] + '">').encode())]
        for name, data in values:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "private literal"):
                self.scan(name, data)

    def test_data_url_optional_type_and_parameters_are_inspected(self):
        for header in ("data:text/plain;charset=utf-8;base64,", "data:;base64,"):
            for payload in (b"public-media", b"private-machine"):
                url = header + base64.b64encode(payload).decode()
                # Python's local data handler confirms these are valid URLs;
                # no HTTP request is made.
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.read(), payload)
                html = f'<img src="{url}">'.encode()
                with self.subTest(header=header, payload=payload):
                    if payload == b"private-machine":
                        with self.assertRaisesRegex(ValueError, "private literal"):
                            self.scan("report.html", html)
                    else:
                        media = self.scan("report.html", html)["embedded_media"]
                        self.assertEqual(media[0]["sha256"], TOOL.digest(payload))

    def test_zip_aba_replacement_cannot_mix_hash_and_member_bytes(self):
        path = self.root / "candidate.conda"
        def write(payload):
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("metadata.json", '{"conda_pkg_format_version":2}')
                archive.writestr("info-test.tar.zst", self.tar([("info/index.json", payload)]))
                archive.writestr("pkg-test.tar.zst", self.tar([("lib/solver.so", b"public")]))
            return path.read_bytes()
        good = write(b"{}")
        bad = write(b'{"host":"\\u0070rivate-machine"}')
        original_zip = zipfile.ZipFile
        class Replacement:
            def __init__(self, target):
                path.write_bytes(good)
                self.archive = original_zip(target)
            def __enter__(self):
                return self.archive
            def __exit__(self, *args):
                self.archive.close()
                path.write_bytes(bad)
        with mock.patch.object(TOOL.zipfile, "ZipFile", side_effect=Replacement), \
             self.assertRaisesRegex(ValueError, "private literal"):
            TOOL.inspect_conda(path, POLICY)
        self.assertEqual(path.read_bytes(), bad)

    def test_ref_aba_replacement_cannot_hide_private_parent(self):
        private = self.repo("private")
        self.commit(private, "private.txt", "private source")
        public = self.repo("public")
        self.commit(public, "README.md", "private-machine")
        tip = self.commit(public, "README.md", "safe public content")
        tree = self.git(public, "rev-parse", "HEAD^{tree}")
        clean = self.git(public, "commit-tree", tree, "-m", "parentless clean commit")
        real_git = TOOL.SOURCE.git
        captures = 0
        def interleave(repo, *args):
            nonlocal captures
            if repo == public and args[0] == "for-each-ref":
                captures += 1
                if captures == 1:
                    captured = real_git(repo, *args)
                    self.git(public, "update-ref", "refs/heads/main", clean)
                    return captured
                self.git(public, "update-ref", "refs/heads/main", tip)
            return real_git(repo, *args)
        try:
            with mock.patch.object(TOOL.SOURCE, "git", side_effect=interleave), \
                 self.assertRaisesRegex(ValueError, "private literal"):
                TOOL.inspect_history(public, private, POLICY)
        finally:
            self.git(public, "update-ref", "refs/heads/main", tip)

    def test_credentials_and_unknown_hardware_ids_are_rejected(self):
        for value in (b"xoxb-" + b"1" * 30, b"ghp_" + b"A" * 36,
                      b"github_pat_" + b"A" * 40,
                      ("GPU-" + "12345678-1234-1234-1234-123456789abc").encode()):
            with self.subTest(value=value[:5]), self.assertRaises(ValueError):
                self.scan("lib/solver.so", b"\x7fELF\0" + value)

    def test_synthetic_fixture_and_generic_path_remain_reviewable(self):
        result = self.scan("test.py", b"GPU-00000000-0000-0000-0000-000000000001 /scratch/failed-runs")
        self.assertEqual(result["generic_private_path_count"], 1)

    def test_all_package_members_are_checked_without_mutation(self):
        path = self.package()
        before = path.read_bytes()
        report = TOOL.inspect_conda(path, POLICY)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual({r["name"] for r in report["members"]}, {"info/index.json", "lib/solver.so"})

    def test_metadata_payload_and_tar_owner_leaks_are_rejected(self):
        cases = [dict(info=[("info/index.json", b'{"host":"private-machine"}')]),
                 dict(payload=[("lib/solver.so", b"\x00private-machine\x00")]),
                 dict(owner="private-machine")]
        for case in cases:
            with self.subTest(case=list(case)), self.assertRaisesRegex(ValueError, "private literal"):
                TOOL.inspect_conda(self.package(**case), POLICY)

    def test_rendered_recipe_is_rejected_even_without_known_literal(self):
        path = self.package(info=[("info/recipe/meta.yaml", b"source: unknown-build-path")])
        with self.assertRaisesRegex(ValueError, "recipe"):
            TOOL.inspect_conda(path, POLICY)

    def test_zip_metadata_and_zstd_skippable_bytes_are_scanned(self):
        for kind in ("archive_comment", "member_comment", "extra", "skippable"):
            for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                with self.subTest(kind=kind, compression=compression):
                    path = self.root / "extra.conda"
                    info = zipfile.ZipInfo("metadata.json")
                    info.compress_type = compression
                    if kind == "member_comment":
                        info.comment = b"private-machine"
                    if kind == "extra":
                        info.extra = struct.pack("<HH", 0xCAFE, 15) + b"private-machine"
                    payload = self.tar([("lib/solver.so", b"ELF public")])
                    if kind == "skippable":
                        payload = struct.pack("<II", 0x184D2A50, 15) + b"private-machine" + payload
                    with zipfile.ZipFile(path, "w", compression=compression) as archive:
                        if kind == "archive_comment":
                            archive.comment = b"private-machine"
                        archive.writestr(info, '{"conda_pkg_format_version":2}')
                        archive.writestr("info-extra.tar.zst", self.tar([("info/index.json", b"{}")]))
                        archive.writestr("pkg-extra.tar.zst", payload)
                    with self.assertRaisesRegex(ValueError, "private literal"):
                        TOOL.inspect_conda(path, POLICY)

    def test_package_traversal_and_wrong_namespace_are_rejected(self):
        for name in ("../escape", "info/unexpected.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                TOOL.inspect_conda(self.package(payload=[(name, b"{}")]), POLICY)

    def test_symlink_asset_is_rejected(self):
        actual = self.root / "actual.md"
        actual.write_text("public")
        link = self.root / "link.md"
        link.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, "symlink"):
            TOOL.verify(POLICY, assets=[link])

    def test_history_checks_deleted_content_and_tag_messages(self):
        private = self.repo("private")
        self.commit(private, "src.cpp", "private implementation")
        public = self.repo("public")
        self.commit(public, "README.md", "private-machine")
        self.commit(public, "README.md", "clean now")
        with self.assertRaisesRegex(ValueError, "private literal"):
            TOOL.inspect_history(public, private, POLICY)
        clean = self.repo("clean")
        self.commit(clean, "README.md", "public")
        self.git(clean, "tag", "-a", "v1.0.3", "-m", "private-machine")
        with self.assertRaisesRegex(ValueError, "private literal"):
            TOOL.inspect_history(clean, private, POLICY)

    def test_private_ancestry_is_rejected_even_with_clean_tip(self):
        private = self.repo("private")
        self.commit(private, "src.cpp", "runtime bytes")
        public = self.root / "public"
        subprocess.run(["/usr/bin/git", "clone", "-q", str(private), str(public)], check=True)
        with self.assertRaisesRegex(ValueError, "private development commits"):
            TOOL.inspect_history(public, private, POLICY)

    def test_ref_names_and_direct_blob_targets_are_inspected(self):
        private = self.repo("private")
        self.commit(private, "src.cpp", "private source")
        public = self.repo("public")
        tip = self.commit(public, "README.md", "public")
        self.git(public, "tag", "private-machine", tip)
        with self.assertRaisesRegex(ValueError, "private literal"):
            TOOL.inspect_history(public, private, POLICY)
        self.git(public, "tag", "-d", "private-machine")
        blob = subprocess.run(["/usr/bin/git", "-C", str(public), "hash-object", "-w", "--stdin"],
                              input=b"private-machine", capture_output=True, check=True).stdout.decode().strip()
        self.git(public, "tag", "orphan-blob", blob)
        with self.assertRaisesRegex(ValueError, "private literal"):
            TOOL.inspect_history(public, private, POLICY)

    def test_shallow_common_directory_cannot_hide_private_parent(self):
        private = self.repo("private")
        self.commit(private, "src.cpp", "private source")
        public = self.repo("public")
        self.commit(public, "README.md", "private-machine")
        tip = self.commit(public, "README.md", "public")
        (public / ".git/shallow").write_text(tip + "\n")
        linked = self.root / "linked"
        self.git(public, "worktree", "add", "-qb", "review", str(linked))
        with self.assertRaisesRegex(ValueError, "shallow"):
            TOOL.inspect_history(linked, private, POLICY)

    def test_clean_history_passes_but_has_no_remote_privacy_claim(self):
        private = self.repo("private")
        self.commit(private, "src.cpp", "runtime bytes")
        public = self.repo("public")
        self.commit(public, "README.md", "public")
        result = TOOL.verify(POLICY, public_repo=public, private_repo=private)
        self.assertEqual(len(result["history"]["commits"]), 1)
        self.assertIn("remote refs and cached commits", result["remaining_review"])
        self.assertFalse(result["publication_authorized"])


if __name__ == "__main__":
    unittest.main()
