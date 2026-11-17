#!/usr/bin/env python3
"""Tests for the simulation-free M4 unittest import preflight."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import py_compile
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/python-validation/preflight_unittest_workload.py"
CASE = ROOT / "python/tests/test_adjoint_cyl.py"
PREFIX = "gpmeep-m4-unittest-preflight:"
IDENTITIES = [
    "test_adjoint_cyl.TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0",
    "test_adjoint_cyl.TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_1",
    "test_adjoint_cyl.TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_2",
    "test_adjoint_cyl.TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_3",
]
CASE_CONTRACT = [
    [0, [5, 0, 20]],
    [0, [4, 0, 28]],
    [-1, [5, 0, 20]],
    [1.2, [5, 0, 20]],
]
INVOCATIONS = [
    {"test_id": IDENTITIES[0], "mode": 0, "far_point": [5.0, 0.0, 20.0]},
    {"test_id": IDENTITIES[1], "mode": 0, "far_point": [4.0, 0.0, 28.0]},
    {"test_id": IDENTITIES[2], "mode": -1, "far_point": [5.0, 0.0, 20.0]},
    {"test_id": IDENTITIES[3], "mode": 1.2, "far_point": [5.0, 0.0, 20.0]},
]


def fake_meep(site_packages: pathlib.Path) -> pathlib.Path:
    package = site_packages / "meep"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text(
        "CYLINDRICAL = object()\n"
        "class Vector3:\n"
        "    def __init__(self, x=0, y=0, z=0):\n"
        "        self.x, self.y, self.z = x, y, z\n"
        "class Medium:\n"
        "    def __init__(self, **kwargs): pass\n"
        "class PML:\n"
        "    def __init__(self, **kwargs): pass\n"
        "class Simulation:\n"
        "    def init_sim(self, *args, **kwargs): pass\n"
        "    def run(self, *args, **kwargs): pass\n"
        "    def run_k_point(self, *args, **kwargs): pass\n"
        "    def run_k_points(self, *args, **kwargs): pass\n"
        "    def solve_cw(self, *args, **kwargs): pass\n",
        encoding="utf-8",
    )
    (package / "adjoint.py").write_text("\n", encoding="utf-8")
    return module


def command(
    site_packages: pathlib.Path,
    module: pathlib.Path,
    case: pathlib.Path,
    pycache_prefix: pathlib.Path,
) -> list[str]:
    result = [
        sys.executable,
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        str(HELPER),
        "--case",
        str(case),
        "--meep-site-packages",
        str(site_packages),
        "--expected-meep-module",
        str(module),
        "--expected-simulation-run-count",
        "16",
    ]
    for identity in IDENTITIES:
        result.extend(("--expected-test-id", identity))
    result.extend(
        (
            "--case-contract-variable",
            "GPMEEP_ADJOINT_CYL_TEST_CASES",
            "--expected-case-contract",
            json.dumps(CASE_CONTRACT, separators=(",", ":")),
            "--wrapper-helper",
            "_check_adjoint_solver_cyl_n2f_fields",
        )
    )
    for invocation in INVOCATIONS:
        result.extend(
            (
                "--expected-wrapper-invocation",
                json.dumps(invocation, sort_keys=True, separators=(",", ":")),
            )
        )
    return result


def synthetic_case(*, setup_entrypoint: str | None = None) -> str:
    setup = ""
    if setup_entrypoint is not None:
        setup = (
            "    def setUp(self):\n"
            f"        getattr(mp.Simulation(), {setup_entrypoint!r})()\n"
        )
    methods = "".join(
        f"    def test_adjoint_solver_cyl_n2f_fields_{index}(self):\n"
        f"        self._check_case({index})\n"
        for index in range(4)
    )
    return (
        "import unittest\n"
        "import meep as mp\n"
        "GPMEEP_ADJOINT_CYL_TEST_CASES = "
        "((0,(5,0,20)),(0,(4,0,28)),(-1,(5,0,20)),(1.2,(5,0,20)))\n"
        "class TestAdjointSolver(unittest.TestCase):\n"
        + setup
        + "    def _check_adjoint_solver_cyl_n2f_fields(self, m, far_x): pass\n"
        "    def _check_case(self, index):\n"
        "        m, point = GPMEEP_ADJOINT_CYL_TEST_CASES[index]\n"
        "        self._check_adjoint_solver_cyl_n2f_fields("
        "m, [mp.Vector3(*point)])\n"
        + methods
    )


class M4UnittestPreflightTests(unittest.TestCase):
    def test_actual_adjoint_case_preserves_identity_parameters_and_no_dependency(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            site_packages = root / "site-packages"
            pycache_prefix = root / "pycache"
            pycache_prefix.mkdir()
            module = fake_meep(site_packages)
            process = subprocess.run(
                command(site_packages, module, CASE, pycache_prefix),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        records = [
            json.loads(line.removeprefix(PREFIX))
            for line in process.stdout.splitlines()
            if line.startswith(PREFIX)
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["test_identities"], IDENTITIES)
        self.assertEqual(record["case_contract"], CASE_CONTRACT)
        self.assertEqual(record["wrapper_invocations"], INVOCATIONS)
        self.assertEqual(record["expected_simulation_run_count"], 16)
        self.assertEqual(record["simulation_workloads_launched"], 0)
        self.assertFalse(record["parameterized_imported"])

    def test_missing_import_fails_before_any_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            site_packages = root / "site-packages"
            pycache_prefix = root / "pycache"
            pycache_prefix.mkdir()
            module = fake_meep(site_packages)
            case = root / "test_adjoint_cyl.py"
            case.write_text(
                "import dependency_that_must_not_exist_for_m4\n",
                encoding="utf-8",
            )
            process = subprocess.run(
                command(site_packages, module, case, pycache_prefix),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(process.returncode, 2)
        self.assertNotIn(PREFIX, process.stdout)
        self.assertIn("unittest identities differ", process.stderr)

    def test_correct_identities_reject_each_simulation_entrypoint(self) -> None:
        for entrypoint in ("init_sim", "run_k_point"):
            with self.subTest(entrypoint=entrypoint):
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary).resolve()
                    site_packages = root / "site-packages"
                    pycache_prefix = root / "pycache"
                    pycache_prefix.mkdir()
                    module = fake_meep(site_packages)
                    case = root / "test_adjoint_cyl.py"
                    case.write_text(
                        synthetic_case(setup_entrypoint=entrypoint),
                        encoding="utf-8",
                    )
                    process = subprocess.run(
                        command(site_packages, module, case, pycache_prefix),
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                self.assertEqual(process.returncode, 2)
                self.assertNotIn(PREFIX, process.stdout)
                self.assertIn("attempted a simulation workload", process.stderr)

    def test_explicit_empty_pycache_prefix_ignores_untracked_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            site_packages = root / "site-packages"
            pycache_prefix = root / "empty-pycache"
            pycache_prefix.mkdir()
            module = fake_meep(site_packages)
            case = root / "test_adjoint_cyl.py"
            marker = root / "malicious-pyc-executed"
            marker_statement = f"open({str(marker)!r}, 'w').write('bad')\n"
            benign_statement = "#" + " " * (len(marker_statement) - 2) + "\n"
            fixture = synthetic_case()
            malicious = marker_statement + fixture
            benign = benign_statement + fixture
            self.assertEqual(len(malicious.encode()), len(benign.encode()))
            case.write_text(malicious, encoding="utf-8")
            source_stat = case.stat()
            bytecode = pathlib.Path(importlib.util.cache_from_source(str(case)))
            bytecode.parent.mkdir()
            py_compile.compile(str(case), cfile=str(bytecode), doraise=True)
            case.write_text(benign, encoding="utf-8")
            os.utime(
                case,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            protected_command = command(
                site_packages, module, case, pycache_prefix
            )
            option_index = protected_command.index("-X")
            unsafe_command = (
                protected_command[:option_index]
                + protected_command[option_index + 2 :]
            )
            unsafe = subprocess.run(
                unsafe_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 0, unsafe.stderr)
            self.assertTrue(marker.exists())
            marker.unlink()
            process = subprocess.run(
                protected_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertFalse(marker.exists())
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn(PREFIX, process.stdout)


if __name__ == "__main__":
    unittest.main()
