"""Tests for the isolated v1.1.0 upstream FP32 CPU baseline receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "upstream-cpu-baseline-receipt.py"
)
BUILDER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "build-upstream-meep-cpu-mpi-python-fp32.sh"
)
REPOSITORY_ROOT = MODULE_PATH.parents[1]
SPEC = importlib.util.spec_from_file_location("v110_upstream_cpu_receipt_test", MODULE_PATH)
assert SPEC and SPEC.loader
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpstreamCpuBaselineReceiptTest(unittest.TestCase):
    def test_builder_keeps_micromamba_process_locks_inside_repo(self) -> None:
        builder = BUILDER_PATH.read_text(encoding="utf-8")
        self.assertIn('MAMBA_CACHE="${MAMBA_ROOT}/cache"', builder)
        self.assertIn('XDG_CACHE_HOME="${MAMBA_CACHE}" PATH=', builder)
        self.assertIn('--env "XDG_CACHE_HOME=${MAMBA_CACHE}"', builder)

    def test_builder_gates_and_retains_the_exact_dependency_package_set(self) -> None:
        builder = BUILDER_PATH.read_text(encoding="utf-8")
        self.assertIn("MAMBA_SHA256=", builder)
        self.assertGreaterEqual(builder.count("verify_micromamba_binary"), 4)
        self.assertIn('"${RECEIPT_TOOL}" verify-environment', builder)
        self.assertIn("--prefix \"${PREFIX}\" --explicit --sha256", builder)
        self.assertIn(
            'cmp -s "${PREFIX_PACKAGE_SET}" "${LOCK_PACKAGE_SET}"', builder
        )
        self.assertIn(
            'cmp -s "${PREFIX_PACKAGE_SET}" "${POST_BUILD_PACKAGE_SET}"',
            builder,
        )
        self.assertIn(
            '"${LOCK_SHA256}" != "${CONTRACT_LOCK_SHA256}"', builder
        )
        self.assertIn("upstream-dependency-package-set.log", builder)

    def test_builder_preserves_cache_isolation_across_clean_environments(self) -> None:
        builder = BUILDER_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            builder.count('XDG_CACHE_HOME="${QUALIFICATION_CACHE}"'), 3
        )
        self.assertEqual(
            builder.count('MPLCONFIGDIR="${QUALIFICATION_CACHE}/matplotlib"'),
            3,
        )
        self.assertEqual(
            builder.count('FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}"'), 3
        )
        self.assertNotIn("etc/fonts/fonts.conf</include>", builder)
        self.assertIn(
            "<cachedir>%s/fontconfig</cachedir>",
            builder,
        )

    def test_builder_bootstrap_parses_public_source_archive_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / "wrong.zip"
            archive.write_bytes(b"wrong archive\n")
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-p",
                    str(BUILDER_PATH),
                    "--source-archive",
                    str(archive),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=30,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("source archive SHA-256 differs", result.stderr)
        self.assertNotIn("usage:", result.stderr)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.temporary.name).resolve()
        (self.repo / "packaging").mkdir()
        (self.repo / "environment/locks").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        self.archive = self.repo / "upstream.zip"
        self.archive.write_bytes(b"fixed upstream archive fixture\n")
        self.lock = self.repo / "environment/locks/cuda-mpi-linux-64.lock"
        self.lock.write_text(
            "# Generated with Micromamba 2.8.1 from fixture.yml\n"
            "# platform: linux-64\n"
            "@EXPLICIT\n"
            "https://example.invalid/linux-64/ca-certificates-1-0.conda#"
            + "a" * 64
            + "\n"
            "https://example.invalid/noarch/b-1-0.conda#"
            + "b" * 64
            + "\n",
            encoding="utf-8",
        )
        self.builder = self.repo / "scripts/build-upstream.sh"
        self.builder.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.builder.chmod(0o755)
        self.prefix = self.repo / ".envs/meep-gpu-cuda-mpi"
        (self.prefix / "bin").mkdir(parents=True)
        self.python = self.prefix / "bin/python3.11"
        self.mpiexec = self.prefix / "bin/mpirun"
        self.python.write_bytes(b"fixture python\n")
        self.mpiexec.write_bytes(b"fixture mpiexec\n")
        (self.prefix / "bin/python").symlink_to("python3.11")
        (self.prefix / "bin/mpiexec").symlink_to("mpirun")
        (self.repo / ".micromamba/cache").mkdir(parents=True)
        self.package_state = self.prefix / "fixture-package-state"
        self.package_entries = RECEIPT._dependency_package_entries(self.lock)
        self.extra_package = (
            "https://example.invalid/linux-64/c-1-0.conda#" + "c" * 64
        )
        tools = self.repo / ".tools"
        tools.mkdir()
        self.micromamba = tools / "micromamba"
        def package_record(entry: str, name: str) -> dict[str, str]:
            url, digest = entry.rsplit("#", 1)
            return {
                "build_string": "0",
                "dist_name": f"{name}-1-0",
                "name": name,
                "sha256": digest,
                "url": url,
                "version": "1",
            }

        first = package_record(self.package_entries[0], "ca-certificates")
        second = package_record(self.package_entries[1], "b")
        extra = package_record(self.extra_package, "c")
        inventory_payloads = {
            "normal": json.dumps([first, second], separators=(",", ":")),
            "missing": json.dumps([first], separators=(",", ":")),
            "extra": json.dumps([first, second, extra], separators=(",", ":")),
            "replaced": json.dumps([first, extra], separators=(",", ":")),
            "wrong_name": json.dumps(
                [{**first, "name": "wrong-a"}, second], separators=(",", ":")
            ),
            "repartitioned_identity": json.dumps(
                [
                    {
                        **first,
                        "build_string": "1-0",
                        "name": "ca",
                        "version": "certificates",
                    },
                    second,
                ],
                separators=(",", ":"),
            ),
        }
        self.micromamba.write_text(
            "#!/bin/sh\n"
            "state=normal\n"
            f"if [ -r {shlex.quote(str(self.package_state))} ]; then\n"
            f"  IFS= read -r state < {shlex.quote(str(self.package_state))}\n"
            "fi\n"
            'case "$state" in\n'
            "  missing)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['missing'])}\n"
            "    ;;\n"
            "  extra)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['extra'])}\n"
            "    ;;\n"
            "  replaced)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['replaced'])}\n"
            "    ;;\n"
            "  wrong_name)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['wrong_name'])}\n"
            "    ;;\n"
            "  repartitioned_identity)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['repartitioned_identity'])}\n"
            "    ;;\n"
            "  *)\n"
            f"    printf '%s\\n' {shlex.quote(inventory_payloads['normal'])}\n"
            "    ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        self.micromamba.chmod(0o755)
        self.original_micromamba_sha256 = RECEIPT.PINNED_MICROMAMBA_SHA256
        RECEIPT.PINNED_MICROMAMBA_SHA256 = sha256(self.micromamba)
        install = self.repo / "install/upstream"
        python_package = install / "lib/python3.11/site-packages/meep"
        (python_package / "mpb").mkdir(parents=True)
        self.artifacts = {
            "libmeep": install / "lib/libmeep.so.30.0.0",
            "python_extension": python_package / "_meep.so.38.0.0",
            "mpb_extension": python_package / "mpb/_mpb.so.38.0.0",
            "libpympb": install / "lib/libpympb.so.30.0.0",
        }
        for name, path in self.artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name + "\n").encode("ascii"))
        build = self.repo / "build/upstream"
        self.qualification = {
            "python_mpi_two_rank_smoke": (
                build / "qualification-logs/python-mpi-two-rank-smoke.log"
            ),
            "python_single_process_import": (
                build / "qualification-logs/python-single-process-import.log"
            ),
            "upstream_binary_partition_utils": (
                build / "qualification-logs/upstream-binary-partition-utils.log"
            ),
            "upstream_build_configuration": build / "source/meep-master/Makefile",
            "upstream_dependency_package_set": (
                build / "qualification-logs/upstream-dependency-package-set.log"
            ),
            "upstream_fontconfig_configuration": (
                build / "qualification-fontconfig.conf"
            ),
        }
        for name, path in self.qualification.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                "CXXFLAGS = -Wall -O2 -fopenmp\n"
                if name == "upstream_build_configuration"
                else (
                    (
                        '<?xml version="1.0"?>\n'
                        '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                        "<fontconfig>\n"
                        "  <dir>/usr/share/fonts</dir>\n"
                        f"  <dir>{self.prefix}/share/fonts</dir>\n"
                        f"  <dir>{self.prefix}/fonts</dir>\n"
                        f'  <include ignore_missing="yes">{self.prefix}'
                        "/etc/fonts/conf.d</include>\n"
                        f"  <cachedir>{build}/qualification-cache/"
                        "fontconfig</cachedir>\n"
                        "</fontconfig>\n"
                    )
                    if name == "upstream_fontconfig_configuration"
                    else name + ":PASS\n"
                )
            )
            path.write_bytes(payload.encode("ascii"))
        contract = {
            "schema_version": 1,
            "baseline_id": "fixture-upstream-cpu",
            "source": {
                "archive_sha256": sha256(self.archive),
                "archive_top_level_directory": "meep-master",
                "upstream_commit": "1" * 40,
            },
            "dependency_environment": {
                "lock_path": "environment/locks/cuda-mpi-linux-64.lock",
                "lock_sha256": sha256(self.lock),
                "prefix_path": ".envs/meep-gpu-cuda-mpi",
            },
            "build": {
                "configure_arguments": ["--enable-single", "--with-mpi", "--with-python"],
                "build_directory": "build/upstream",
                "install_prefix": "install/upstream",
                "builder": "scripts/build-upstream.sh",
                "required_cxx_optimization_flag": "-O2",
                "qualification": {
                    "python_mpi_two_rank_smoke": (
                        "qualification-logs/python-mpi-two-rank-smoke.log"
                    ),
                    "python_single_process_import": (
                        "qualification-logs/python-single-process-import.log"
                    ),
                    "upstream_binary_partition_utils": (
                        "qualification-logs/upstream-binary-partition-utils.log"
                    ),
                    "upstream_build_configuration": "source/meep-master/Makefile",
                    "upstream_dependency_package_set": (
                        "qualification-logs/upstream-dependency-package-set.log"
                    ),
                    "upstream_fontconfig_configuration": (
                        "qualification-fontconfig.conf"
                    ),
                },
            },
            "measurement_role": "fixture",
            "gpu_backend_forbidden": True,
            "scheme_forbidden": True,
        }
        (self.repo / RECEIPT.CONTRACT_RELATIVE).write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.qualification["upstream_dependency_package_set"].write_text(
            RECEIPT._expected_dependency_package_set(self.repo, contract),
            encoding="utf-8",
        )
        (self.repo / ".gitignore").write_text(
            ".envs/\n.micromamba/\n.tools/\n", encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "fixture@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Fixture"], check=True
        )
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-q", "-m", "fixture"], check=True
        )

    def tearDown(self) -> None:
        RECEIPT.PINNED_MICROMAMBA_SHA256 = self.original_micromamba_sha256
        self.temporary.cleanup()

    def create_args(self, output: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            operation="create",
            repo=self.repo,
            archive=self.archive,
            python=self.prefix / "bin/python",
            mpiexec=self.prefix / "bin/mpiexec",
            artifact=[f"{name}={path}" for name, path in self.artifacts.items()],
            qualification=[
                f"{name}={path}" for name, path in self.qualification.items()
            ],
            output=output,
        )

    def test_create_and_verify_bind_every_artifact(self) -> None:
        output = self.repo / ".git/fixture-receipt.json"
        value = RECEIPT.create(self.create_args(output))
        RECEIPT._write(output, value)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        verified = RECEIPT.verify(
            argparse.Namespace(operation="verify", repo=self.repo, receipt=output)
        )
        self.assertEqual(verified["state"], "PASS")
        self.assertEqual(verified["receipt_id"], value["receipt_id"])
        self.assertEqual(
            value["micromamba"]["sha256"], RECEIPT.PINNED_MICROMAMBA_SHA256
        )
        forged = json.loads(output.read_text(encoding="utf-8"))
        forged["artifacts"]["libmeep"]["path"] = str(self.artifacts["python_extension"])
        forged_without_id = dict(forged)
        forged_without_id.pop("receipt_id")
        forged["receipt_id"] = RECEIPT._canonical_digest(forged_without_id)
        output.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "artifact path differs"):
            RECEIPT.verify(
                argparse.Namespace(operation="verify", repo=self.repo, receipt=output)
            )

    def test_receipt_must_be_resealed_after_the_final_source_commit(self) -> None:
        output = self.repo / ".git/upstream-before-final-commit.json"
        value = RECEIPT.create(self.create_args(output))
        RECEIPT._write(output, value)
        (self.repo / "README.md").write_text("final release trust change\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "final source boundary"],
            check=True,
        )
        with self.assertRaisesRegex(
            RECEIPT.ReceiptError, "repository source identity differs"
        ):
            RECEIPT.verify(
                argparse.Namespace(operation="verify", repo=self.repo, receipt=output)
            )

    def test_artifact_inventory_is_exact(self) -> None:
        values = [f"{name}={path}" for name, path in self.artifacts.items()]
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "inventory differs"):
            RECEIPT._artifact_map(values[:-1])
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "duplicate artifact"):
            RECEIPT._artifact_map([*values, values[0]])

        invalid_extension = self.artifacts["python_extension"].with_name(
            "_meep.so.untrusted"
        )
        invalid_extension.write_bytes(b"invalid extension fixture\n")
        invalid_artifacts = dict(self.artifacts)
        invalid_artifacts["python_extension"] = invalid_extension
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "artifact path differs"):
            RECEIPT._validate_artifact_paths(
                self.repo, RECEIPT._contract(self.repo)[1], invalid_artifacts
            )

    def test_qualification_inventory_and_content_are_enforced(self) -> None:
        values = [f"{name}={path}" for name, path in self.qualification.items()]
        contract = RECEIPT._contract(self.repo)[1]
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "inventory differs"):
            RECEIPT._qualification_map(values[:-1], contract)
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "duplicate qualification"):
            RECEIPT._qualification_map([*values, values[0]], contract)

        configuration = self.qualification["upstream_build_configuration"]
        configuration.write_text("CXXFLAGS = -Wall -O0\n", encoding="utf-8")
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "optimization flag is absent"):
            RECEIPT.create(self.create_args(self.repo / ".git/rejected-receipt.json"))

    def test_fontconfig_qualification_is_exact(self) -> None:
        self.qualification["upstream_fontconfig_configuration"].write_text(
            "<fontconfig/>\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "Fontconfig.*differs"):
            RECEIPT._validate_fontconfig_configuration(
                self.repo,
                RECEIPT._contract(self.repo)[1],
                self.qualification,
            )

    def test_dependency_package_set_qualification_is_exact(self) -> None:
        package_set = self.qualification["upstream_dependency_package_set"]
        package_set.write_text(
            package_set.read_text(encoding="utf-8").replace(
                "package_count=2", "package_count=3"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            RECEIPT.ReceiptError, "dependency package set differs"
        ):
            RECEIPT.create(self.create_args(self.repo / ".git/rejected-receipt.json"))

    def test_verify_rejects_a_contract_whose_lock_digest_is_stale(self) -> None:
        output = self.repo / ".git/fixture-receipt.json"
        value = RECEIPT.create(self.create_args(output))
        RECEIPT._write(output, value)
        contract_path = self.repo / RECEIPT.CONTRACT_RELATIVE
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["dependency_environment"]["lock_sha256"] = "0" * 64
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        forged = json.loads(output.read_text(encoding="utf-8"))
        forged["contract"] = RECEIPT._file_record(contract_path)
        forged_without_id = dict(forged)
        forged_without_id.pop("receipt_id")
        forged["receipt_id"] = RECEIPT._canonical_digest(forged_without_id)
        output.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "lock digest differs"):
            RECEIPT.verify(
                argparse.Namespace(operation="verify", repo=self.repo, receipt=output)
            )

    def test_live_prefix_is_rechecked_for_extra_missing_and_replaced_packages(self) -> None:
        for index, state in enumerate(("extra", "missing", "replaced")):
            with self.subTest(state=state):
                self.package_state.unlink(missing_ok=True)
                output = self.repo / f".git/live-prefix-{index}.json"
                value = RECEIPT.create(self.create_args(output))
                RECEIPT._write(output, value)
                self.package_state.write_text(state + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RECEIPT.ReceiptError, "live dependency prefix package set differs"
                ):
                    RECEIPT.verify(
                        argparse.Namespace(
                            operation="verify", repo=self.repo, receipt=output
                        )
                    )
        self.package_state.unlink(missing_ok=True)

    def test_micromamba_binary_is_pinned_and_rechecked(self) -> None:
        output = self.repo / ".git/micromamba-replacement.json"
        value = RECEIPT.create(self.create_args(output))
        RECEIPT._write(output, value)
        with self.micromamba.open("a", encoding="utf-8") as stream:
            stream.write("# replaced after receipt\n")
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "Micromamba.*digest differs"):
            RECEIPT.verify(
                argparse.Namespace(operation="verify", repo=self.repo, receipt=output)
            )

    def test_live_package_name_is_bound_to_its_distribution_identity(self) -> None:
        self.package_state.write_text("wrong_name\n", encoding="utf-8")
        with self.assertRaisesRegex(
            RECEIPT.ReceiptError, "distribution identity differs"
        ):
            RECEIPT.verify_environment(
                argparse.Namespace(operation="verify-environment", repo=self.repo)
            )

    def test_live_package_identity_cannot_be_repartitioned_at_hyphens(self) -> None:
        self.package_state.write_text("repartitioned_identity\n", encoding="utf-8")
        with self.assertRaisesRegex(RECEIPT.ReceiptError, "URL identity differs"):
            RECEIPT.verify_environment(
                argparse.Namespace(operation="verify-environment", repo=self.repo)
            )

    def test_verify_environment_replays_live_inventory(self) -> None:
        result = RECEIPT.verify_environment(
            argparse.Namespace(operation="verify-environment", repo=self.repo)
        )
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["package_count"], 2)
        self.assertEqual(
            result["micromamba"]["sha256"], RECEIPT.PINNED_MICROMAMBA_SHA256
        )

    def test_capture_environment_writes_the_canonical_name_url_sha_set(self) -> None:
        output = self.repo / ".git/dependency-package-set.log"
        result = RECEIPT.capture_environment(
            argparse.Namespace(
                operation="capture-environment", repo=self.repo, output=output
            )
        )
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        payload = output.read_text(encoding="utf-8")
        self.assertIn(
            f"package=ca-certificates\t{self.package_entries[0]}\n", payload
        )
        self.assertIn(f"package=b\t{self.package_entries[1]}\n", payload)

    def test_receipt_metadata_schema_is_exact(self) -> None:
        for index, mutation in enumerate(("unknown", "commit", "timestamp")):
            with self.subTest(mutation=mutation):
                output = self.repo / f".git/metadata-{index}.json"
                value = RECEIPT.create(self.create_args(output))
                RECEIPT._write(output, value)
                forged = json.loads(output.read_text(encoding="utf-8"))
                if mutation == "unknown":
                    forged["unknown_claim"] = "unbound"
                    expected = "field inventory differs"
                elif mutation == "commit":
                    forged["upstream_commit"] = "f" * 40
                    expected = "upstream commit differs"
                else:
                    forged["created_at_utc"] = "not-a-timestamp"
                    expected = "timestamp is invalid"
                forged_without_id = dict(forged)
                forged_without_id.pop("receipt_id")
                forged["receipt_id"] = RECEIPT._canonical_digest(forged_without_id)
                output.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaisesRegex(RECEIPT.ReceiptError, expected):
                    RECEIPT.verify(
                        argparse.Namespace(
                            operation="verify", repo=self.repo, receipt=output
                        )
                    )

    def test_explicit_lock_grammar_is_fail_closed(self) -> None:
        valid = self.lock.read_text(encoding="utf-8")
        variants = {
            "incomplete_generator_header": valid.replace(
                "# Generated with Micromamba 2.8.1 from fixture.yml",
                "# Generated with Micromamba ",
            ),
            "missing_directive": valid.replace("@EXPLICIT\n", ""),
            "duplicate_directive": valid.replace(
                "@EXPLICIT\n", "@EXPLICIT\n@EXPLICIT\n"
            ),
            "garbage_payload": valid + "MALFORMED-UNPARSED-LINE\n",
            "comment_after_directive": valid.replace(
                "@EXPLICIT\n", "@EXPLICIT\n# hidden payload\n"
            ),
            "duplicate_url": valid + self.package_entries[0] + "\n",
            "duplicate_url_new_hash": valid
            + self.package_entries[0].rsplit("#", 1)[0]
            + "#"
            + "d" * 64
            + "\n",
            "missing_terminal_newline": valid.rstrip("\n"),
        }
        for name, payload in variants.items():
            with self.subTest(name=name):
                self.lock.write_text(payload, encoding="utf-8")
                with self.assertRaises(RECEIPT.ReceiptError):
                    RECEIPT._dependency_package_entries(self.lock)
        self.lock.write_text(valid, encoding="utf-8")

    def test_repository_contract_matches_the_checked_in_dependency_lock(self) -> None:
        contract = json.loads(
            (REPOSITORY_ROOT / RECEIPT.CONTRACT_RELATIVE).read_text(encoding="utf-8")
        )
        lock = REPOSITORY_ROOT / contract["dependency_environment"]["lock_path"]
        self.assertEqual(
            contract["dependency_environment"]["lock_sha256"], sha256(lock)
        )


if __name__ == "__main__":
    unittest.main()
