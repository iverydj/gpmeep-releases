"""Exercise public-source verification against real Git object stores."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "public_source", Path(__file__).resolve().parents[1] / "verify-public-source.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
OLD = "GPU-" + "12345678-1234-1234-1234-123456789abc"
NEW = "GPU-00000000-0000-0000-0000-000000000001"


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.public = self.root / "public"
        for repo in (self.source, self.public):
            repo.mkdir()
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "Test")
            self.git(repo, "config", "user.email", "test@example.invalid")
        self.write(self.source, "src/step.cpp", b"void step() {}\n")
        self.write(self.source, "scripts/build.sh", b"#!/bin/sh\nexit 0\n", 0o755)
        self.write(self.source, "python/api.py", b"precision = 'single'\n")
        self.write(self.source, "LICENSE", b"License terms\n")
        self.write(self.source, "README.md", b"Private host example: old-host\n")
        self.write(self.source, "artifacts/raw.log", b"old-host private evidence\n")
        self.write(self.source, "scripts/tests/test_device.py", f"DEVICE = {OLD!r}\n".encode())
        self.notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["print(1)\n"],
             "execution_count": 1, "outputs": [{"output_type": "stream", "text": ["old-host"]}]}]}
        self.write(self.source, "python/example.ipynb", json.dumps(self.notebook).encode())
        self.source_commit = self.commit(self.source)
        for path in self.source.rglob("*"):
            rel = path.relative_to(self.source)
            if path.is_file() and ".git" not in rel.parts and rel.parts[0] != "artifacts":
                dest = self.public / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        self.write(self.public, "README.md", b"Public documentation\n")
        self.write(self.public, "scripts/tests/test_device.py", f"DEVICE = {NEW!r}\n".encode())
        public_notebook = json.loads(json.dumps(self.notebook))
        public_notebook["cells"][0].update(outputs=[], execution_count=None)
        self.write(self.public, "python/example.ipynb", json.dumps(public_notebook).encode())
        self.write(self.public, "PUBLIC_RELEASE.json", json.dumps({"binary_source_commit": self.source_commit}).encode())

    def git(self, repo, *args):
        return subprocess.run(["/usr/bin/git", "-C", str(repo), *args],
                              capture_output=True, check=True).stdout.decode().strip()

    def write(self, repo, name, data, mode=0o644):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)

    def commit(self, repo):
        self.git(repo, "add", "--all")
        self.git(repo, "commit", "-qm", "fixture", "--allow-empty")
        return self.git(repo, "rev-parse", "HEAD")

    def verify(self, **overrides):
        options = dict(source_repo=self.source, source_commit=self.source_commit,
                       public_repo=self.public, public_commit=self.commit(self.public),
                       forbidden_literals=["old-host"], fixture_replacements={OLD: NEW})
        options.update(overrides)
        return TOOL.verify(**options)

    def test_permitted_edits_preserve_solver_and_never_authorize_release(self):
        report = self.verify()
        self.assertEqual(report["state"], "CONTENT_VERIFIED")
        self.assertFalse(report["publication_authorized"])
        records = {row["path"]: row for row in report["files"]}
        self.assertEqual(records["src/step.cpp"]["classification"], "identical")
        self.assertEqual(records["python/example.ipynb"]["classification"], "notebook_outputs_only")
        self.assertEqual(records["scripts/tests/test_device.py"]["classification"], "synthetic_fixture_uuids")
        self.assertEqual(report["omitted_paths"], ["artifacts/raw.log"])

    def test_changed_solver_is_rejected(self):
        self.write(self.public, "src/step.cpp", b"void step() { abort(); }\n")
        with self.assertRaisesRegex(TOOL.PublicSourceError, "runtime/build"):
            self.verify()

    def test_missing_runtime_or_license_is_rejected(self):
        (self.public / "LICENSE").unlink()
        with self.assertRaisesRegex(TOOL.PublicSourceError, "omits protected"):
            self.verify()

    def test_private_artifact_must_be_omitted(self):
        self.write(self.public, "artifacts/empty.log", b"")
        with self.assertRaisesRegex(TOOL.PublicSourceError, "private evidence"):
            self.verify()

    def test_new_code_is_rejected(self):
        self.write(self.public, "python/startup.py", b"print('changed')\n")
        with self.assertRaisesRegex(TOOL.PublicSourceError, "unverified runtime"):
            self.verify()

    def test_executable_documentation_is_rejected(self):
        (self.public / "README.md").chmod(0o755)
        with self.assertRaisesRegex(TOOL.PublicSourceError, "mode changed"):
            self.verify()

    def test_notebook_source_is_not_an_output(self):
        obj = json.loads((self.public / "python/example.ipynb").read_text())
        obj["cells"][0]["source"] = ["print(2)\n"]
        self.write(self.public, "python/example.ipynb", json.dumps(obj).encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "notebook source"):
            self.verify()

    def test_notebook_metadata_changes_are_not_silently_accepted(self):
        obj = json.loads((self.public / "python/example.ipynb").read_text())
        obj["metadata"] = {"execution": "different"}
        self.write(self.public, "python/example.ipynb", json.dumps(obj).encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "notebook source or metadata"):
            self.verify()

    def test_uuid_redaction_does_not_permit_fixture_logic_changes(self):
        self.write(self.public, "scripts/tests/test_device.py", f"DEVICE = {NEW!r}\nassert False\n".encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "fixture logic changed"):
            self.verify()

    def test_uuid_mapping_must_not_collapse_distinct_devices(self):
        other = "GPU-" + "87654321-1234-1234-1234-123456789abc"
        with self.assertRaisesRegex(TOOL.PublicSourceError, "distinct device"):
            self.verify(fixture_replacements={OLD: NEW, other: NEW})

    def test_encoding_cookie_cannot_change_execution_or_inject_code(self):
        for prefix in (b"# coding: latin-1\n", b'# coding: unicode_escape\n#\\u000araise RuntimeError("injected")\n'):
            with self.subTest(prefix=prefix):
                original = f'DEVICE = {OLD!r}\nLABEL = "é"\n'.encode()
                self.write(self.source, "scripts/tests/test_device.py", original)
                self.source_commit = self.commit(self.source)
                self.write(self.public, "PUBLIC_RELEASE.json", json.dumps({"binary_source_commit": self.source_commit}).encode())
                self.write(self.public, "scripts/tests/test_device.py", prefix + original.replace(OLD.encode(), NEW.encode()))
                with self.assertRaisesRegex(TOOL.PublicSourceError, "fixture logic"):
                    self.verify()

    def test_escaped_existing_identity_is_not_available_as_replacement(self):
        escaped = NEW.replace("GPU-0", r"GPU-\u0030", 1)
        self.write(self.source, "scripts/tests/test_device.py", f'DEVICE = {OLD!r}\nOTHER = "{escaped}"\n'.encode())
        self.source_commit = self.commit(self.source)
        self.write(self.public, "PUBLIC_RELEASE.json", json.dumps({"binary_source_commit": self.source_commit}).encode())
        self.write(self.public, "scripts/tests/test_device.py", f'DEVICE = {NEW!r}\nOTHER = {NEW!r}\n'.encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "collides"):
            self.verify()

    def test_existing_fixture_identity_cannot_be_a_redaction_destination(self):
        self.write(self.source, "scripts/tests/test_device.py", f"DEVICE = {OLD!r}\nOTHER = {NEW!r}\n".encode())
        self.source_commit = self.commit(self.source)
        self.write(self.public, "scripts/tests/test_device.py", f"DEVICE = {NEW!r}\nOTHER = {NEW!r}\n".encode())
        self.write(self.public, "PUBLIC_RELEASE.json", json.dumps({"binary_source_commit": self.source_commit}).encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "collides"):
            self.verify()

    def test_original_build_identity_cannot_be_relabelled(self):
        self.write(self.public, "PUBLIC_RELEASE.json", json.dumps({"binary_source_commit": "f" * 40}).encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "original binary source"):
            self.verify()

    def test_private_identifier_in_a_filename_is_rejected(self):
        self.write(self.public, "doc/old-host.md", b"otherwise harmless\n")
        with self.assertRaisesRegex(TOOL.PublicSourceError, "private literal"):
            self.verify()

    def test_credential_pattern_is_rejected(self):
        self.write(self.public, "README.md", b"token=" + b"xoxb-" + b"0" * 16)
        with self.assertRaisesRegex(TOOL.PublicSourceError, "credential pattern"):
            self.verify()

    def test_json_escaped_private_text_is_rejected(self):
        obj = json.loads((self.public / "python/example.ipynb").read_text())
        obj["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["old-host"]}]
        escaped = json.dumps(obj).replace("old-host", "\\u006fld-host")
        self.write(self.public, "python/example.ipynb", escaped.encode())
        with self.assertRaisesRegex(TOOL.PublicSourceError, "private literal"):
            self.verify()

    def test_new_documentation_symlink_is_rejected(self):
        (self.public / "guide.md").symlink_to("README.md")
        with self.assertRaisesRegex(TOOL.PublicSourceError, "new metadata is executable or a symlink"):
            self.verify()

    def test_mutable_ref_is_rejected(self):
        with self.assertRaisesRegex(TOOL.PublicSourceError, "immutable commit"):
            self.verify(source_commit="HEAD")

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(TOOL.PublicSourceError, "duplicate JSON"):
            TOOL.read_json(b'{"key": 1, "key": 2}')

    def test_worktree_mutation_does_not_change_named_object_evidence(self):
        report = self.verify()
        commit = report["public_commit"]
        self.write(self.public, "src/step.cpp", b"uncommitted mutation\n")
        replay = TOOL.verify(self.source, self.source_commit, self.public, commit,
                             forbidden_literals=["old-host"], fixture_replacements={OLD: NEW})
        self.assertEqual(report, replay)


if __name__ == "__main__":
    unittest.main()
