from __future__ import annotations

import hashlib
import pathlib
import subprocess
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
STAGER = SCRIPTS / "stage-conda-lock-archives.py"
CUDA_BUILDER = SCRIPTS / "build-meep-cuda-mpi-python.sh"
CPU_BUILDER = SCRIPTS / "build-meep-cpu-mpi-python-fp64.sh"
CONTROL_RUNNER = SCRIPTS / "gpmeep-control-python.py"


class CondaLockArchiveStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.cache = self.root / "pkgs"
        self.cache.mkdir()
        self.payload = b"exact conda package archive\n"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.name = "fixture-1.0-0.conda"
        self.url = f"https://example.invalid/linux-64/{self.name}"
        self.lock = self.root / "environment.lock"
        self.lock.write_text(
            f"@EXPLICIT\n{self.url}#{self.digest}\n", encoding="utf-8"
        )
        (self.cache / self.name).write_bytes(self.payload)

    @property
    def destination(self) -> pathlib.Path:
        return self.cache / "https/example.invalid/linux-64" / self.name

    def invoke(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "/usr/bin/python3",
                "-S",
                "-P",
                "-B",
                str(STAGER),
                "--lock",
                str(self.lock),
                "--package-cache",
                str(self.cache),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    def test_stages_verified_archive_and_is_idempotent(self) -> None:
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("1 staged, 1 total", first.stdout)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o444)
        second = self.invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("0 staged, 1 total", second.stdout)

    def test_missing_and_wrong_root_archive_fail_closed(self) -> None:
        (self.cache / self.name).unlink()
        missing = self.invoke()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("is absent", missing.stderr)
        (self.cache / self.name).write_bytes(b"wrong bytes\n")
        wrong = self.invoke()
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("SHA-256 mismatch", wrong.stderr)

    def test_source_and_destination_symlinks_fail_closed(self) -> None:
        source = self.cache / self.name
        real = self.cache / "real.conda"
        source.rename(real)
        source.symlink_to(real)
        linked_source = self.invoke()
        self.assertNotEqual(linked_source.returncode, 0)
        self.assertIn("not a canonical regular file", linked_source.stderr)

        source.unlink()
        real.rename(source)
        self.destination.parent.mkdir(parents=True)
        self.destination.symlink_to(source)
        linked_destination = self.invoke()
        self.assertNotEqual(linked_destination.returncode, 0)
        self.assertIn("not a canonical regular file", linked_destination.stderr)

    def test_lock_hash_and_url_are_strict(self) -> None:
        self.lock.write_text(f"@EXPLICIT\n{self.url}#ABC\n", encoding="utf-8")
        bad_hash = self.invoke()
        self.assertNotEqual(bad_hash.returncode, 0)
        self.assertIn("lowercase SHA-256", bad_hash.stderr)
        self.lock.write_text(
            f"@EXPLICIT\nhttps://example.invalid/../{self.name}#{self.digest}\n",
            encoding="utf-8",
        )
        bad_url = self.invoke()
        self.assertNotEqual(bad_url.returncode, 0)
        self.assertIn("unmappable URL", bad_url.stderr)

    def test_authoritative_builders_invoke_checked_in_stager(self) -> None:
        cuda_source = CUDA_BUILDER.read_text(encoding="utf-8")
        cpu_source = CPU_BUILDER.read_text(encoding="utf-8")
        control_source = CONTROL_RUNNER.read_text(encoding="utf-8")
        for source in (cuda_source, cpu_source):
            self.assertIn('"${SCRIPT_DIR}/stage-conda-lock-archives.py"', source)
            self.assertIn('--lock "${LOCK}" --package-cache', source)
        self.assertIn('"stage-conda-lock-archives.py"', control_source)


if __name__ == "__main__":
    unittest.main()
