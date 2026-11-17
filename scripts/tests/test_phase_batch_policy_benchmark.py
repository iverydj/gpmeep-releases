from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import pathlib
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_phase_batch_policy_benchmark_test",
    SCRIPTS / "benchmark-phase-batch-policy.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def marker(prefix: str, value: dict) -> str:
    return prefix + json.dumps(value, separators=(",", ":")) + "\n"


def policy(condition: str, pixels: int, ranks: int) -> dict[str, int]:
    value = {key: 0 for key in BENCHMARK.POLICY_KEYS}
    for prefix in ("curl", "update_eh"):
        mode = (
            "automatic"
            if prefix == "update_eh" or condition == "automatic"
            else condition.removeprefix("curl-")
        )
        if mode == "automatic":
            if prefix == "curl":
                value[f"{prefix}_automatic_checks"] = 10 * ranks
                if pixels == 32:
                    value[f"{prefix}_automatic_selected"] = 10 * ranks
                    value[f"{prefix}_batched_operations"] = 80 * ranks
                else:
                    value[f"{prefix}_automatic_rejected"] = 10 * ranks
                    value[f"{prefix}_unbatched_operations"] = 80 * ranks
                continue
            value[f"{prefix}_automatic_checks"] = 10 * ranks
            if pixels == 256 and ranks == 1:
                value[f"{prefix}_automatic_rejected"] = 10
                value[f"{prefix}_unbatched_operations"] = 80
            else:
                value[f"{prefix}_automatic_selected"] = 10 * ranks
                value[f"{prefix}_batched_operations"] = 80 * ranks
        elif mode == "forced":
            value[f"{prefix}_forced_batches"] = 10 * ranks
            value[f"{prefix}_batched_operations"] = 80 * ranks
    return value


def workload_parameters(pixels: int) -> dict[str, int]:
    matches = [item for item in BENCHMARK.workloads(1) if item["pixels"] == pixels]
    if len(matches) != 1:
        raise AssertionError(f"missing unique workload for {pixels}")
    return matches[0]


def worker_record(pixels: int, ranks: int, seconds: float = 1.0) -> dict:
    probes = [[0.1 + index * 0.001, 0.01] for index in range(18)]
    probe_l2 = math.sqrt(
        sum(pair[0] * pair[0] + pair[1] * pair[1] for pair in probes)
    )
    phases = {
        name: {
            "cpu_calls": 0,
            "cuda_calls": 10,
            "expectation": "cuda_required",
        }
        for name in BENCHMARK.MGB.REQUIRED_CUDA_PHASES
    }
    phases["polarization"] = {
        "cpu_calls": 0,
        "cuda_calls": 0,
        "expectation": "not_applicable",
    }
    cells = pixels**3
    workload = workload_parameters(pixels)
    steps = workload["steps"]
    mpi_scalars = 100 if ranks == 2 else 0
    return {
        "schema_version": 4,
        "mpi_ranks": ranks,
        "pixels": pixels,
        "cells": cells,
        "warmup_steps": workload["warmup_steps"],
        "steps": steps,
        "loop_tile_base_db": 128,
        "bfast": False,
        "source_enabled": True,
        "overlap_material": False,
        "seconds": seconds,
        "mcells_per_second": cells * steps / seconds / 1e6,
        "requested_transport": "pinned",
        "selected_transport": "pinned" if ranks == 2 else "none",
        "completion_policy": "waitsome",
        "initial_condition": "affine-v1",
        "cut_probe_values": probes,
        "cut_probe_l2": probe_l2,
        "ez_l2": 2.0,
        "ez_weighted_checksum": 3.0,
        "energy": 4.0,
        "dft_norm": 5.0,
        "remote_cut_scalars": 100 if ranks == 2 else 0,
        "remote_cut_l2": 1.0 if ranks == 2 else 0.0,
        "cpu_calls": 0,
        "cuda_calls": 50,
        "phase_calls": phases,
        "h2d_bytes": 400 if ranks == 2 else 0,
        "d2h_bytes": 400 if ranks == 2 else 0,
        "mpi_messages": 4 if ranks == 2 else 0,
        "mpi_scalars": mpi_scalars,
        "cuda_aware_bytes": 0,
        "pinned_bytes": mpi_scalars * 4,
    }


def worker_output(condition: str, pixels: int, ranks: int) -> str:
    record = worker_record(pixels, ranks)
    devices = [
        {
            "rank": rank,
            "ordinal": rank,
            "uuid": f"gpu-{rank}",
            "name": "Synthetic GPU",
            "compute_major": 8,
            "compute_minor": 6,
            "compatible": True,
        }
        for rank in range(ranks)
    ]
    return "".join(
        [
            *(marker(BENCHMARK.BASE.DEVICE_PREFIX, item) for item in devices),
            *(
                marker(
                    BENCHMARK.PROCESS_PREFIX,
                    {"rank": rank, "pid": 1000 + rank},
                )
                for rank in range(ranks)
            ),
            marker(
                BENCHMARK.BASE.SOURCE_PROFILE_PREFIX,
                {"profile": "dual-electric-v1"},
            ),
            marker(
                BENCHMARK.BASE.INITIAL_CONDITION_PREFIX,
                {"profile": "affine-v1"},
            ),
            marker(
                BENCHMARK.BASE.INITIALIZATION_TIMING_PREFIX,
                {"applications_per_rank": 1, "max_seconds": 0.1},
            ),
            marker(BENCHMARK.POLICY_PREFIX, policy(condition, pixels, ranks)),
            marker(BENCHMARK.BASE.BENCHMARK_PREFIX, record),
        ]
    )


def configuration(pixels: int = 128, ranks: int = 1, repeats: int = 3) -> dict:
    workload = workload_parameters(pixels)
    return {
        "pixels": pixels,
        "ranks": ranks,
        "warmup_steps": workload["warmup_steps"],
        "steps": workload["steps"],
        "repeats": repeats,
    }


def samples_for_summary(
    pixels: int, ranks: int, times: dict[str, list[float]]
) -> list[dict]:
    result = []
    devices = [
        {
            "rank": rank,
            "ordinal": rank,
            "uuid": f"gpu-{rank}",
            "name": "Synthetic GPU",
            "compute_major": 8,
            "compute_minor": 6,
            "compatible": True,
        }
        for rank in range(ranks)
    ]
    repeats = len(next(iter(times.values())))
    for repeat in range(1, repeats + 1):
        for order_index, condition in enumerate(BENCHMARK.CONDITIONS, 1):
            result.append(
                {
                    "identity": {
                        "repeat": repeat,
                        "order_index": order_index,
                        "condition": condition,
                    },
                    "devices": copy.deepcopy(devices),
                    "result": worker_record(
                        pixels, ranks, times[condition][repeat - 1]
                    ),
                }
            )
    return result


class EnvironmentTests(unittest.TestCase):
    def test_automatic_removes_all_inherited_curl_eh_controls(self) -> None:
        inherited = {
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL": "1",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH": "1",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH": "1",
            "KEEP": "yes",
        }
        result = BENCHMARK.condition_environment(
            inherited, "automatic", configuration()
        )
        self.assertEqual(result["KEEP"], "yes")
        self.assertFalse(any("PHASE_BATCHED_CURL" in key for key in result))
        self.assertFalse(any("PHASE_BATCHED_UPDATE_EH" in key for key in result))

    def test_curl_controls_are_isolated_from_automatic_eh(self) -> None:
        disabled = BENCHMARK.condition_environment(
            {}, "curl-disabled", configuration()
        )
        forced = BENCHMARK.condition_environment({}, "curl-forced", configuration())
        self.assertEqual(disabled, {
            "MEEP_GPU_MULTI_PIXELS": "128",
            "MEEP_GPU_MULTI_WARMUP_STEPS": "12",
            "MEEP_GPU_MULTI_STEPS": "80",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL": "1",
        })
        self.assertEqual(forced, {
            "MEEP_GPU_MULTI_PIXELS": "128",
            "MEEP_GPU_MULTI_WARMUP_STEPS": "12",
            "MEEP_GPU_MULTI_STEPS": "80",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL": "1",
        })


class StrictParsingTests(unittest.TestCase):
    def test_all_conditions_validate_for_one_and_two_gpus(self) -> None:
        for ranks in (1, 2):
            for pixels in (32, 128, 256):
                for condition in BENCHMARK.CONDITIONS:
                    record, counters, devices, processes = BENCHMARK.validate_sample(
                        condition,
                        configuration(pixels, ranks),
                        worker_output(condition, pixels, ranks),
                    )
                    self.assertEqual(record["cpu_calls"], 0)
                    self.assertEqual(len(devices), ranks)
                    self.assertEqual(len(processes), ranks)
                    self.assertEqual(set(counters), BENCHMARK.POLICY_KEYS)

    def test_policy_schema_duplicate_and_negative_values_fail_closed(self) -> None:
        output = worker_output("automatic", 128, 1)
        bad = output.replace(
            '"update_eh_automatic_checks":10',
            '"update_eh_automatic_checks":-1',
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid exact schema"):
            BENCHMARK.validate_sample("automatic", configuration(), bad)
        duplicate = output + marker(
            BENCHMARK.POLICY_PREFIX, policy("automatic", 128, 1)
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            BENCHMARK.validate_sample("automatic", configuration(), duplicate)

    def test_production_default_large_grid_curl_must_reject_consistently(self) -> None:
        base = policy("automatic", 128, 1)
        invalid = []
        for field, value in (
            ("curl_automatic_checks", 0),
            ("curl_automatic_selected", 1),
            ("curl_automatic_rejected", 0),
            ("curl_forced_batches", 1),
            ("curl_batched_operations", 1),
            ("curl_unbatched_operations", 0),
        ):
            mutated = dict(base)
            mutated[field] = value
            invalid.append((field, mutated))
        for field, counters in invalid:
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "automatic curl|large-grid curl"
            ):
                BENCHMARK.validate_policy("automatic", 128, 1, counters)

    def test_automatic_eh_accounting_fails_closed(self) -> None:
        invalid = []
        rejected_without_unbatched = policy("automatic", 128, 1)
        rejected_without_unbatched["update_eh_automatic_selected"] = 5
        rejected_without_unbatched["update_eh_automatic_rejected"] = 5
        invalid.append(rejected_without_unbatched)

        selected_without_batched = policy("automatic", 128, 1)
        selected_without_batched["update_eh_batched_operations"] = 0
        invalid.append(selected_without_batched)

        batched_without_selected = policy("automatic", 128, 1)
        batched_without_selected["update_eh_automatic_selected"] = 0
        batched_without_selected["update_eh_automatic_rejected"] = 10
        invalid.append(batched_without_selected)

        for counters in invalid:
            with self.subTest(counters=counters):
                with self.assertRaisesRegex(
                    RuntimeError, "accounting|unbatched operations"
                ):
                    BENCHMARK.validate_policy(
                        "automatic", 256, 2, counters
                    )

    def test_wrong_policy_mode_and_unknown_marker_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disabled curl"):
            BENCHMARK.validate_sample(
                "curl-disabled",
                configuration(),
                worker_output("curl-forced", 128, 1),
            )
        with self.assertRaisesRegex(RuntimeError, "unknown or mixed-version"):
            BENCHMARK.validate_sample(
                "automatic",
                configuration(),
                worker_output("automatic", 128, 1) + "gpmeep-future-v9:{}\n",
            )

    def test_device_aliasing_and_cpu_dispatch_are_rejected(self) -> None:
        output = worker_output("automatic", 128, 2)
        with self.assertRaisesRegex(RuntimeError, "one-rank-per-GPU"):
            BENCHMARK.validate_sample(
                "automatic", configuration(128, 2), output.replace("gpu-1", "gpu-0")
            )
        cpu_output = output.replace('"cpu_calls":0', '"cpu_calls":1', 1)
        with self.assertRaisesRegex(Exception, "CPU|cpu_calls"):
            BENCHMARK.validate_sample(
                "automatic", configuration(128, 2), cpu_output
            )


class SummaryTests(unittest.TestCase):
    gate_names = {
        "maximum_median_paired_automatic_over_curl_disabled_time_ratio",
        "maximum_median_paired_automatic_over_curl_forced_time_ratio",
    }

    def test_both_no_regression_gates_apply_to_every_grid(self) -> None:
        times = {
            "automatic": [1.00, 1.00, 1.00],
            "curl-disabled": [1.00, 1.00, 1.00],
            "curl-forced": [1.00, 1.00, 1.00],
        }
        for pixels in (32, 128, 256):
            with self.subTest(pixels=pixels):
                summary = BENCHMARK.summarize_configuration(
                    configuration(pixels, 1),
                    samples_for_summary(pixels, 1, times),
                )
                self.assertTrue(summary["pass"])
                self.assertEqual(len(summary["gates"]), 2)
                self.assertEqual(
                    {gate["name"] for gate in summary["gates"]},
                    self.gate_names,
                )
                self.assertTrue(all(gate["pass"] for gate in summary["gates"]))

    def test_each_comparator_rejects_regression_at_every_grid(self) -> None:
        cases = {
            "curl-disabled": {
                "automatic": [1.03, 1.03, 1.03],
                "curl-disabled": [1.00, 1.00, 1.00],
                "curl-forced": [1.20, 1.20, 1.20],
            },
            "curl-forced": {
                "automatic": [1.03, 1.03, 1.03],
                "curl-disabled": [1.20, 1.20, 1.20],
                "curl-forced": [1.00, 1.00, 1.00],
            },
        }
        for pixels in (32, 128, 256):
            for comparator, times in cases.items():
                with self.subTest(pixels=pixels, comparator=comparator):
                    summary = BENCHMARK.summarize_configuration(
                        configuration(pixels, 1),
                        samples_for_summary(pixels, 1, times),
                    )
                    gates = {gate["name"]: gate for gate in summary["gates"]}
                    failed_name = (
                        "maximum_median_paired_automatic_over_"
                        f"{comparator.replace('-', '_')}_time_ratio"
                    )
                    self.assertFalse(summary["pass"])
                    self.assertEqual(set(gates), self.gate_names)
                    self.assertFalse(gates[failed_name]["pass"])
                    self.assertTrue(
                        all(
                            gate["pass"]
                            for name, gate in gates.items()
                            if name != failed_name
                        )
                    )

    def test_no_regression_threshold_boundary_is_inclusive(self) -> None:
        for automatic, expected in ((1.02, True), (1.0201, False)):
            with self.subTest(automatic=automatic):
                times = {
                    "automatic": [automatic] * 3,
                    "curl-disabled": [1.00] * 3,
                    "curl-forced": [1.00] * 3,
                }
                summary = BENCHMARK.summarize_configuration(
                    configuration(128, 1),
                    samples_for_summary(128, 1, times),
                )
                self.assertEqual(summary["pass"], expected)
                self.assertTrue(
                    all(gate["pass"] == expected for gate in summary["gates"])
                )

    def test_changed_observable_fails_closed(self) -> None:
        times = {
            "automatic": [1.00, 1.00, 1.00],
            "curl-disabled": [1.00, 1.00, 1.00],
            "curl-forced": [1.00, 1.00, 1.00],
        }
        samples = samples_for_summary(256, 1, times)
        samples[1]["result"]["energy"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "physical observable"):
            BENCHMARK.summarize_configuration(configuration(256, 1), samples)

    def test_markdown_reports_every_configuration(self) -> None:
        times = {
            "automatic": [1.0, 1.0, 1.0],
            "curl-disabled": [1.4, 1.4, 1.4],
            "curl-forced": [1.0, 1.0, 1.0],
        }
        item = BENCHMARK.summarize_configuration(
            configuration(128, 2), samples_for_summary(128, 2, times)
        )
        markdown = BENCHMARK.render_markdown(
            {
                "pass": True,
                "sample_count": 9,
                "receipt_id": "a" * 64,
                "configurations": [item],
            }
        )
        self.assertIn("128³", markdown)
        self.assertIn("1.4000×", markdown)


class ExecutionSealingTests(unittest.TestCase):
    def make_execution_args(self, root: pathlib.Path) -> types.SimpleNamespace:
        archive = root / "archive"
        archive.mkdir()
        executable = archive / "worker"
        library = archive / "libmeep.so"
        mpiexec = archive / "mpiexec"
        mca = archive / "mca.conf"
        executable.write_bytes(b"worker")
        library.write_bytes(b"library")
        mpiexec.write_bytes(b"mpiexec")
        mca.write_bytes(b"mca")
        executable.chmod(0o555)
        library.chmod(0o444)
        mpiexec.chmod(0o555)
        mca.chmod(0o444)
        return types.SimpleNamespace(
            result_dir=root,
            execution_executable=executable,
            execution_library=library,
            execution_mpiexec=mpiexec,
            execution_mca=mca,
            execution_snapshot_executable=executable,
            execution_snapshot_library=library,
            execution_snapshot_mpiexec=mpiexec,
            execution_snapshot_mca=mca,
            timeout=30,
        )

    def test_command_executes_only_archived_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_execution_args(pathlib.Path(temporary))
            command = BENCHMARK.sample_command(
                args, {"PATH": "/usr/bin"}, configuration(128, 2)
            )
            self.assertIn(str(args.execution_mpiexec), command)
            self.assertIn(str(args.execution_executable), command)
            self.assertIn(f"LD_PRELOAD={args.execution_library}", command)

    def test_same_bytes_at_a_replaced_inode_fail_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self.make_execution_args(root)
            before = BENCHMARK.execution_identity(args)
            old = args.execution_executable.with_suffix(".old")
            args.execution_executable.rename(old)
            args.execution_executable.write_bytes(b"worker")
            args.execution_executable.chmod(0o555)
            after = BENCHMARK.execution_identity(args)
            self.assertEqual(
                before["executable"]["sha256"], after["executable"]["sha256"]
            )
            self.assertNotEqual(
                before["executable"]["inode"], after["executable"]["inode"]
            )
            self.assertNotEqual(before, after)

    def test_persistent_handle_keeps_original_image_after_path_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self.make_execution_args(root)
            descriptor, binding = BENCHMARK.create_execution_handle(
                args.execution_snapshot_executable, root, "executable"
            )
            try:
                old = args.execution_snapshot_executable.with_suffix(".old")
                args.execution_snapshot_executable.rename(old)
                args.execution_snapshot_executable.write_bytes(b"forged")
                args.execution_snapshot_executable.chmod(0o555)
                with self.assertRaisesRegex(RuntimeError, "execution image changed"):
                    BENCHMARK.verify_execution_handle(binding, descriptor)
                self.assertEqual(
                    pathlib.Path(binding["execution_path"]).read_bytes(), b"worker"
                )
                self.assertEqual(args.execution_snapshot_executable.read_bytes(), b"forged")
            finally:
                __import__("os").close(descriptor)


class ProcessMonitorTests(unittest.TestCase):
    def test_expected_workers_on_both_gpus_pass(self) -> None:
        output = "\n".join(
            [
                "0 1000 gpu-mpi-per",
                "1 1001 gpu-mpi-per",
                "0 - -",
            ]
        )
        record = BENCHMARK.parse_process_monitor(
            output, True, -15, {1000: 0, 1001: 1}
        )
        self.assertTrue(record["pass"])

    def test_current_driver_row_without_worker_name_passes(self) -> None:
        output = "\n".join(
            [
                "# gpu pid type sm mem enc dec jpg ofa fb ccpm command",
                "0 1000 C 2 0 - - - - 324 0 3",
                "1 1001 C 1 0 - - - - 262 0",
            ]
        )
        record = BENCHMARK.parse_process_monitor(
            output, True, 0, {1000: 0, 1001: 1}
        )
        self.assertTrue(record["pass"])
        self.assertEqual(record["unrelated_process_rows"], [])

    def test_missing_expected_worker_fails_closed(self) -> None:
        record = BENCHMARK.parse_process_monitor(
            "0 1000 C 1 0 - - - - 100 0 3",
            True,
            -15,
            {1000: 0, 1001: 1},
        )
        self.assertFalse(record["pass"])
        self.assertEqual(record["unobserved_expected_worker_pids"], [1001])

    def test_expected_worker_on_wrong_gpu_fails_closed(self) -> None:
        output = "\n".join(
            [
                "1 1000 C 1 0 - - - - 100 0 3",
                "1 1001 C 1 0 - - - - 100 0 3",
            ]
        )
        record = BENCHMARK.parse_process_monitor(
            output, True, -15, {1000: 0, 1001: 1}
        )
        self.assertFalse(record["pass"])
        self.assertEqual(len(record["unrelated_process_rows"]), 1)

    def test_malformed_non_header_row_fails_closed(self) -> None:
        output = "\n".join(
            [
                "0 1000 C 1 0 - - - - 100 0 3",
                "1 1001 C 1 0 - - - - 100 0",
                "malformed-row",
            ]
        )
        record = BENCHMARK.parse_process_monitor(
            output, True, -15, {1000: 0, 1001: 1}
        )
        self.assertFalse(record["pass"])
        self.assertEqual(record["unrelated_process_rows"], ["malformed-row"])

    def test_monitor_lifecycle_must_be_alive_with_accepted_exit(self) -> None:
        output = "\n".join(
            [
                "0 1000 C 1 0 - - - - 100 0 3",
                "1 1001 C 1 0 - - - - 100 0",
            ]
        )
        self.assertFalse(
            BENCHMARK.parse_process_monitor(
                output, True, 7, {1000: 0, 1001: 1}
            )["pass"]
        )
        self.assertFalse(
            BENCHMARK.parse_process_monitor(
                output, False, 0, {1000: 0, 1001: 1}
            )["pass"]
        )

    def test_unrelated_same_name_pid_fails_closed(self) -> None:
        output = "\n".join(
            [
                "0 1000 gpu-mpi-per",
                "1 1001 gpu-mpi-per",
                "0 9999 gpu-mpi-per",
            ]
        )
        record = BENCHMARK.parse_process_monitor(
            output, True, -15, {1000: 0, 1001: 1}
        )
        self.assertFalse(record["pass"])
        self.assertEqual(record["unrelated_process_rows"], ["0 9999 gpu-mpi-per"])


class TerminalEvidenceTests(unittest.TestCase):
    def seed(self, root: pathlib.Path) -> None:
        (root / "summary.json").write_text("{}\n", encoding="utf-8")
        (root / "report.md").write_text("report\n", encoding="utf-8")
        (root / "protocol.json").write_text("{}\n", encoding="utf-8")

    def test_complete_is_last_and_binds_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.seed(root)
            BENCHMARK.finalize_success(root, "a" * 64)
            self.assertTrue((root / "COMPLETE.json").is_file())
            self.assertFalse((root / "FAILED.json").exists())
            complete = json.loads((root / "COMPLETE.json").read_text())
            manifest = json.loads((root / BENCHMARK.MANIFEST_NAME).read_text())
            self.assertEqual(
                complete["manifest"]["sha256"],
                BENCHMARK.BASE.sha256_file(root / BENCHMARK.MANIFEST_NAME),
            )
            paths = {item["path"] for item in manifest["files"]}
            self.assertNotIn(BENCHMARK.MANIFEST_NAME, paths)
            self.assertTrue(paths.isdisjoint(BENCHMARK.TERMINAL_NAMES))
            BENCHMARK.finalize_failure(root, {"state": "FAILED_EXECUTION"})
            self.assertFalse((root / "FAILED.json").exists())

    def test_manifest_failure_cannot_leave_complete_and_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.seed(root)
            with mock.patch.object(
                BENCHMARK,
                "write_evidence_manifest",
                side_effect=OSError("synthetic manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic"):
                    BENCHMARK.finalize_success(root, "a" * 64)
            self.assertFalse((root / "COMPLETE.json").exists())
            BENCHMARK.finalize_failure(
                root,
                {"schema_version": 1, "state": "FAILED_EXECUTION"},
            )
            self.assertTrue((root / "FAILED.json").is_file())
            self.assertFalse((root / "COMPLETE.json").exists())


class SourceTests(unittest.TestCase):
    def test_runner_has_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(
            (SCRIPTS / "benchmark-phase-batch-policy.py").read_text(
                encoding="utf-8"
            )
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            literal_keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(len(literal_keys), len(set(literal_keys)))

    def test_matrix_is_exact_and_balanced(self) -> None:
        matrix = BENCHMARK.configurations(6)
        self.assertEqual(
            {(item["pixels"], item["ranks"]) for item in matrix},
            {(32, 1), (32, 2), (128, 1), (128, 2), (256, 1), (256, 2)},
        )
        small = [item for item in matrix if item["pixels"] == 32]
        self.assertTrue(
            all(item["warmup_steps"] == 120 and item["steps"] == 800 for item in small)
        )
        self.assertTrue(all(item["repeats"] == 6 for item in matrix))
        self.assertEqual(
            set().union(*(set(order) for order in BENCHMARK.CONDITION_ORDERS)),
            set(BENCHMARK.CONDITIONS),
        )
        positions = {
            condition: [0, 0, 0] for condition in BENCHMARK.CONDITIONS
        }
        for repeat in range(6):
            for position, condition in enumerate(
                BENCHMARK.CONDITION_ORDERS[
                    repeat % len(BENCHMARK.CONDITION_ORDERS)
                ]
            ):
                positions[condition][position] += 1
        self.assertEqual(set(map(tuple, positions.values())), {(2, 2, 2)})

    def test_local_modules_are_loaded_from_and_bound_to_exact_bytes(self) -> None:
        self.assertEqual(
            set(BENCHMARK.LOADED_MODULE_SOURCES),
            {
                "gpmeep_benchmark_evidence.py",
                "gpmeep_qualification_contract.py",
                "gpmeep_provenance.py",
                "multi_gpu_benchmark.py",
                "benchmark-dft-phase-sharing.py",
            },
        )
        for path, source, digest in BENCHMARK.LOADED_MODULE_SOURCES.values():
            self.assertEqual(path.read_bytes(), source)
            self.assertEqual(
                __import__("hashlib").sha256(source).hexdigest(), digest
            )


if __name__ == "__main__":
    unittest.main()
