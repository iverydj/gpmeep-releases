"""Exercise the real candidate shell runtime and its locked scientific imports.

Run with the contracted v110-test interpreter. The shell fixture substitutes
only environment creation (no prefix may be modified by a test); the production
runtime verifier still checks the actual committed lock and Micromamba list.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-conda-package.sh"
RUNTIME_PATH = ROOT / "scripts/verify-v110-final-runtime.py"
SPEC = importlib.util.spec_from_file_location("candidate_gate_runtime_test", RUNTIME_PATH)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def shell_function(name: str) -> str:
    text = BUILDER.read_text()
    start = text.index(name + "() {\n")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


class CandidateGateRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python = Path(sys.executable).resolve(strict=True)
        cls.runtime_repo = cls.python.parents[3]
        cls.contract = json.loads(
            (cls.runtime_repo / "packaging/v110-development-contract.json").read_bytes()
        )
        isolation = cls.contract["development_isolation"]
        if cls.python != cls.runtime_repo / isolation["test_prefix_relative"] / "bin/python3.11":
            raise RuntimeError("run these tests with the contracted v110-test Python")
        cls.commit = subprocess.check_output(
            ["/usr/bin/git", "--no-optional-locks", "-C", str(cls.runtime_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        cls.python_pin = {
            "path": str(cls.python),
            "sha256": hashlib.sha256(cls.python.read_bytes()).hexdigest(),
        }
        cls.lock = subprocess.check_output(
            ["/usr/bin/git", "--no-optional-locks", "-C", str(cls.runtime_repo),
             "show", f"{cls.commit}:{isolation['test_lock']}"],
        )

    def test_candidate_shell_selects_verified_runtime_for_real_aunp_imports(self):
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            scripts = temporary / "scripts"
            scripts.mkdir()
            # Retain the real verifier. Only create-env is replaced so this test
            # cannot install, recreate, or alter the existing interpreter.
            for filename in ("verify-v110-final-runtime.py", "verify-public-source.py"):
                (scripts / filename).write_bytes((ROOT / "scripts" / filename).read_bytes())
            creation_log = temporary / "creation.json"
            (scripts / "create-env.sh").write_text(
                '#!/bin/bash\nset -euo pipefail\n'
                '[[ "$#" == 1 && "$1" == v110-test ]]\n'
                'printf \'["v110-test"]\\n\' > ' + shlex.quote(str(creation_log)) + '\n'
            )
            payload = temporary / "scientific_gate.py"
            (temporary / "absent").mkdir()
            payload.write_text('''import importlib.util, json, pathlib, sys
import numpy, h5py
from scipy import optimize
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("actual_candidate_evidence", root / "scripts/v110-candidate-evidence.py")
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
try:
    candidate._replay_raw("aunp_source_candidate", pathlib.Path(sys.argv[2]) / "COMPLETE")
except Exception as error:
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        raise
    if "COMPLETE" not in str(error):
        raise
else:
    raise AssertionError("missing AuNP evidence must fail after real producer imports")
print(json.dumps({"python": sys.executable, "analysis_imports": ["numpy", "scipy", "h5py"],
                  "missing_raw_rejected": True}))
''')
            builder_prefix = self.runtime_repo / ".envs/gpmeep-v1.1-builder"
            harness = temporary / "gate.sh"
            assignments = {
                "REPO_ROOT": str(self.runtime_repo), "SCRIPT_DIR": str(scripts),
                "PREFIX": str(builder_prefix), "BUILD_MODE": "candidate",
                "SOURCE_COMMIT": self.commit,
                "V110_RUNTIME_VERIFIER": str(scripts / "verify-v110-final-runtime.py"),
            }
            harness.write_text(
                "#!/bin/bash -p\nset -euo pipefail\n"
                + "\n".join(key + "=" + shlex.quote(value) for key, value in assignments.items()) + "\n"
                + shell_function("canonical_direct_file") + "\n"
                + shell_function("prepare_v110_gate_runtime") + "\n"
                + 'prepare_v110_gate_runtime\n'
                + '[[ "$PREFIX" == ' + shlex.quote(str(builder_prefix)) + ' ]]\n'
                + '"$V110_GATE_PYTHON" -I -B ' + shlex.quote(str(payload)) + ' '
                + shlex.quote(str(ROOT)) + ' ' + shlex.quote(str(temporary / "absent")) + "\n"
            )
            environment = {"PATH": "/usr/bin:/bin", "HOME": str(temporary),
                           "TMPDIR": str(temporary), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(["/bin/bash", "-p", str(harness)], env=environment,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["python"], str(self.python))
            self.assertEqual(value["analysis_imports"], ["numpy", "scipy", "h5py"])
            self.assertTrue(value["missing_raw_rejected"])
            self.assertEqual(json.loads(creation_log.read_text()), ["v110-test"])

    def test_development_uses_builder_without_analysis_environment_creation(self):
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            prefix = temporary / "builder"
            (prefix / "bin").mkdir(parents=True)
            python = prefix / "bin/python3.11"
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o700)
            (temporary / "packaging").mkdir()
            (temporary / "packaging/v110-development-contract.json").write_text("{}")
            script = "set -euo pipefail\n" + "\n".join(
                key + "=" + shlex.quote(value) for key, value in {
                    "REPO_ROOT": str(temporary), "PREFIX": str(prefix),
                    "SCRIPT_DIR": str(temporary / "absent"), "BUILD_MODE": "development",
                }.items()
            ) + "\n" + shell_function("prepare_v110_gate_runtime") + '\nprepare_v110_gate_runtime\nprintf "%s" "$V110_GATE_PYTHON"\n'
            result = subprocess.run(["/bin/bash", "-p", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, str(python))

    def test_runtime_rejects_unpinned_python_bytes(self):
        with self.assertRaisesRegex(ValueError, "runtime file SHA-256 differs"):
            RUNTIME.verify_final_runtime(self.runtime_repo, self.commit,
                                         dict(self.python_pin, sha256="0" * 64))

    def test_runtime_rejects_another_interpreter_path_before_package_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            other = Path(raw) / "python3.11"
            other.write_bytes(b"a different interpreter")
            pin = {"path": str(other), "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "pinned contracted test Python"):
                RUNTIME.verify_final_runtime(self.runtime_repo, self.commit, pin)

    def test_runtime_rejects_missing_or_extra_explicit_packages(self):
        urls = [line for line in self.lock.splitlines() if b"://" in line]
        numpy = next(line for line in urls if b"/numpy-" in line)
        variants = {
            "missing_numpy": b"\n".join(line for line in urls if line != numpy),
            "extra_package": b"\n".join(urls + [b"https://invalid.example/extra-1-0.conda#" + b"0" * 64]),
        }
        real_run = subprocess.run
        for label, inventory in variants.items():
            def run(command, *args, **kwargs):
                if command[0] == str(self.runtime_repo / ".tools/micromamba"):
                    return subprocess.CompletedProcess(command, 0, stdout=inventory, stderr=b"")
                return real_run(command, *args, **kwargs)
            with self.subTest(label=label), mock.patch.object(RUNTIME.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "exact committed lock"):
                    RUNTIME.verify_final_runtime(self.runtime_repo, self.commit, self.python_pin)

    def test_committed_analysis_lock_declares_numeric_dependencies(self):
        isolation = self.contract["development_isolation"]
        self.assertEqual(hashlib.sha256(self.lock).hexdigest(), isolation["test_lock_sha256"])
        names = {line.rsplit(b"/", 1)[1].split(b"#")[0].rsplit(b"-", 2)[0]
                 for line in self.lock.splitlines() if b"://" in line}
        self.assertTrue({b"numpy", b"scipy", b"h5py"} <= names)
        script = BUILDER.read_text()
        selection = script.index("\nprepare_v110_gate_runtime\n")
        self.assertGreater(selection, script.index('/bin/bash -p "${SCRIPT_DIR}/create-env.sh" "${ENVIRONMENT_PROFILE}"'))
        self.assertLess(selection, script.index('CLOSURE_RESULT="$("${V110_GATE_PYTHON}"'))


if __name__ == "__main__":
    unittest.main()
