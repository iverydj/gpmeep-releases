from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_dft_multi_monitor_benchmark",
    SCRIPTS / "benchmark-dft-multi-monitor.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

RUNTIME_UUIDS = ("a" * 32, "b" * 32)
NVIDIA_SMI_UUIDS = (
    "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
)


def prefixed(prefix: str, value: dict) -> str:
    return prefix + json.dumps(value, separators=(",", ":")) + "\n"


def mutate_marker(
    output: str,
    prefix: str,
    index: int,
    *,
    mutation: dict | None = None,
    drop: bool = False,
) -> str:
    result: list[str] = []
    matched = 0
    for line in output.splitlines(keepends=True):
        if line.startswith(prefix):
            if matched == index:
                matched += 1
                if drop:
                    continue
                record = json.loads(line[len(prefix):])
                if mutation:
                    record.update(mutation)
                line = prefixed(prefix, record)
            else:
                matched += 1
        result.append(line)
    if matched <= index:
        raise AssertionError(f"marker index {index} is absent for {prefix!r}")
    return "".join(result)


def topology(ranks: int = 2) -> dict:
    return {
        "name": f"{ranks}gpu",
        "ranks": ranks,
        "visible_devices": "0" if ranks == 1 else "0,1",
    }


def workload() -> dict:
    return dict(BENCHMARK.WORKLOADS[0])


def frequency_heavy_workload() -> dict:
    return dict(BENCHMARK.WORKLOADS[1])


def sample_output(
    condition: str,
    *,
    ranks: int = 2,
    identity: dict | None = None,
    dft_mutation: dict | None = None,
    multi_mutation: dict | None = None,
    benchmark_mutation: dict | None = None,
) -> str:
    selected_transport = "none" if ranks == 1 else "cuda-aware"
    requested_transport = "auto" if ranks == 1 else "cuda-aware"
    record = {
        "schema_version": 4,
        "mpi_ranks": ranks,
        "pixels": 32,
        "cells": 32**3,
        "warmup_steps": 40,
        "steps": 2400,
        "loop_tile_base_db": 128,
        "bfast": False,
        "source_enabled": True,
        "overlap_material": False,
        "seconds": 2.0,
        "mcells_per_second": 32**3 * 2400 / 2.0 / 1.0e6,
        "requested_transport": requested_transport,
        "selected_transport": selected_transport,
        "completion_policy": "waitsome",
        "initial_condition": "trigonometric-v1",
        "cut_probe_values": [[0.2, 0.1]] * 18,
        "cut_probe_l2": 1.0,
        "ez_l2": 2.0,
        "ez_weighted_checksum": 3.0,
        "energy": 4.0,
        "dft_norm": 5.0,
        "remote_cut_scalars": 0 if ranks == 1 else 24,
        "remote_cut_l2": 0.0 if ranks == 1 else 0.5,
        "cpu_calls": 0,
        "cuda_calls": 100,
        "phase_calls": {
            name: {
                "cpu_calls": 0,
                "cuda_calls": 20,
                "expectation": "cuda_required",
            }
            for name in ("curl", "update_eh", "source", "boundary", "dft")
        },
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "mpi_messages": 0 if ranks == 1 else 4,
        "mpi_scalars": 0 if ranks == 1 else 32,
        "cuda_aware_bytes": 0 if ranks == 1 else 128,
        "pinned_bytes": 0,
    }
    record["phase_calls"]["polarization"] = {
        "cpu_calls": 0,
        "cuda_calls": 0,
        "expectation": "not_applicable",
    }
    if benchmark_mutation:
        record.update(benchmark_mutation)
    dft = {
        "batch_calls": 2400,
        "submitted_updates": 153600,
        "phase_preparation_launches": 20,
        "phase_reuses": 153580,
        "update_kernel_launches": 2400 if condition == "forced" else 153600,
        "maximum_batch_size": 64,
    }
    if dft_mutation:
        dft.update(dft_mutation)
    if condition == "forced":
        multi = {
            "automatic_checks": 0,
            "automatic_selected": 0,
            "automatic_rejected": 0,
            "forced_batches": 2400,
            "batched_updates": 153600,
            "unbatched_updates": 0,
            "plan_uploads": 0,
            "plan_reuses": 2400,
            "metadata_host_to_device_bytes": 0,
        }
        warmup_plan = {
            "metadata_host_to_device_bytes": 512,
            "plan_reuses": 39,
            "plan_uploads": 1,
        }
    elif condition == "automatic":
        multi = {
            "automatic_checks": 2400,
            "automatic_selected": 0,
            "automatic_rejected": 2400,
            "forced_batches": 0,
            "batched_updates": 0,
            "unbatched_updates": 153600,
            "plan_uploads": 0,
            "plan_reuses": 0,
            "metadata_host_to_device_bytes": 0,
        }
        warmup_plan = {
            "metadata_host_to_device_bytes": 0,
            "plan_reuses": 0,
            "plan_uploads": 0,
        }
    else:
        multi = {
            "automatic_checks": 0,
            "automatic_selected": 0,
            "automatic_rejected": 0,
            "forced_batches": 0,
            "batched_updates": 0,
            "unbatched_updates": 153600,
            "plan_uploads": 0,
            "plan_reuses": 0,
            "metadata_host_to_device_bytes": 0,
        }
        warmup_plan = {
            "metadata_host_to_device_bytes": 0,
            "plan_reuses": 0,
            "plan_uploads": 0,
        }
    if multi_mutation:
        multi.update(multi_mutation)
    identity = identity or {"components": 2, "frequencies": 2, "monitors": 64}
    devices = [
        {
            "rank": rank,
            "ordinal": rank,
            "uuid": RUNTIME_UUIDS[rank],
            "name": f"GPU {rank}",
            "compute_major": 8,
            "compute_minor": 6,
            "compatible": True,
        }
        for rank in range(ranks)
    ]
    processes = [
        {"rank": rank, "pid": 1000 + rank} for rank in range(ranks)
    ]
    eager = {
        key: (0 if ranks == 1 else 4) for key in BENCHMARK.EAGER_FIELDS
    }
    pingpong = {
        key: (0 if ranks == 1 else 2) for key in BENCHMARK.PINGPONG_FIELDS
    }
    graph = {
        "creations": 0,
        "launches": 0 if ranks == 1 else 2,
    }
    return "".join(
        [
            *(prefixed(BENCHMARK.BASE.DEVICE_PREFIX, item) for item in devices),
            *(prefixed(BENCHMARK.BASE.PROCESS_PREFIX, item) for item in processes),
            prefixed(
                BENCHMARK.BASE.SOURCE_PROFILE_PREFIX,
                {"profile": "single-ez-v1"},
            ),
            prefixed(
                BENCHMARK.BASE.INITIAL_CONDITION_PREFIX,
                {"profile": "trigonometric-v1"},
            ),
            prefixed(
                BENCHMARK.BASE.INITIALIZATION_TIMING_PREFIX,
                {"applications_per_rank": 1, "max_seconds": 0.25},
            ),
            prefixed(BENCHMARK.DFT_WORKLOAD_PREFIX, identity),
            prefixed(BENCHMARK.BASE.EAGER_PREFIX, eager),
            prefixed(BENCHMARK.BASE.PINGPONG_PREFIX, pingpong),
            prefixed(BENCHMARK.BASE.GRAPH_PREFIX, graph),
            prefixed(BENCHMARK.BASE.STRUCTURED_CURL_GRAPH_PREFIX, {}),
            prefixed(BENCHMARK.DFT_WARMUP_PLAN_PREFIX, warmup_plan),
            prefixed(BENCHMARK.BASE.DFT_PREFIX, dft),
            prefixed(BENCHMARK.BASE.DFT_MULTI_PREFIX, multi),
            prefixed(BENCHMARK.BASE.BENCHMARK_PREFIX, record),
        ]
    )


class ValidationTests(unittest.TestCase):
    def test_all_conditions_and_topologies_are_accepted(self) -> None:
        for ranks in (1, 2):
            for condition in ("forced", "disabled", "automatic"):
                parsed = BENCHMARK.validate_sample(
                    condition,
                    topology(ranks),
                    workload(),
                    sample_output(condition, ranks=ranks),
                )
                self.assertEqual(parsed["identity"]["monitors"], 64)

    def test_unknown_marker_is_rejected(self) -> None:
        output = sample_output("forced") + (
            "rank 0: gpmeep-dft-workload-v1:{} "
            "gpmeep-ghost-v1:{}\n"
        )
        with self.assertRaisesRegex(RuntimeError, "unknown gpmeep"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_duplicate_required_communication_marker_is_rejected(self) -> None:
        output = sample_output("forced") + prefixed(
            BENCHMARK.BASE.GRAPH_PREFIX, {"creations": 2, "launches": 2}
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one boundary graph"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_exact_multi_schema_is_required(self) -> None:
        output = sample_output(
            "forced", multi_mutation={"unexpected": 0}
        )
        with self.assertRaisesRegex(RuntimeError, "exact integer schema"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_forced_plan_reuse_is_required(self) -> None:
        output = sample_output(
            "forced",
            multi_mutation={
                "plan_uploads": 2400,
                "plan_reuses": 0,
                "metadata_host_to_device_bytes": 512,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "reuse a fused"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_automatic_selection_is_rejected_until_qualified(self) -> None:
        output = sample_output(
            "automatic",
            dft_mutation={"update_kernel_launches": 153537},
            multi_mutation={
                "automatic_selected": 1,
                "automatic_rejected": 2399,
                "batched_updates": 64,
                "unbatched_updates": 153536,
                "plan_uploads": 1,
                "metadata_host_to_device_bytes": 512,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "fail closed"):
            BENCHMARK.validate_sample(
                "automatic", topology(), workload(), output
            )

    def test_workload_identity_is_exact(self) -> None:
        output = sample_output(
            "forced",
            identity={"components": 2, "frequencies": 3, "monitors": 64},
        )
        with self.assertRaisesRegex(RuntimeError, "differs from the protocol"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_condition_environments_are_mutually_exclusive(self) -> None:
        forced = BENCHMARK.condition_environment("forced")
        disabled = BENCHMARK.condition_environment("disabled")
        automatic = BENCHMARK.condition_environment("automatic")
        self.assertEqual(
            forced,
            {
                "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH": "1",
                "MEEP_GPU_EXPECT_MULTI_MONITOR_DFT_BATCH": "1",
            },
        )
        self.assertEqual(
            disabled,
            {
                "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH": "1",
                "MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH": "1",
            },
        )
        self.assertEqual(
            automatic,
            {"MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH": "1"},
        )

    def test_pair_summary_uses_paired_kernel_times(self) -> None:
        pairs = []
        for index in range(8):
            forced = 1.0 + 0.01 * index
            disabled = 1.5 + 0.01 * index
            pairs.append(
                {
                    "forced": {
                        "parsed": {"benchmark": {"seconds": forced}}
                    },
                    "disabled": {
                        "parsed": {"benchmark": {"seconds": disabled}}
                    },
                }
            )
        result = BENCHMARK.summarize_pairs(
            topology(1), workload(), pairs
        )
        self.assertTrue(result["pass"])
        self.assertGreater(result["median_paired_speedup_percent"], 10.0)
        self.assertIn(
            "t90_upper_forced_over_disabled_time_ratio", result
        )
        self.assertNotIn("t90_upper_time_ratio", result)

    def test_frequency_heavy_lane_is_a_long_non_regression_guard(self) -> None:
        fixed = frequency_heavy_workload()
        self.assertEqual(BENCHMARK.PROTOCOL_ID,
                         "m27-dft-multi-monitor-authoritative-v2")
        self.assertEqual(fixed["steps"], 2400)
        self.assertGreaterEqual(fixed["minimum_measured_seconds"], 2.0)
        self.assertIsNone(fixed["minimum_median_speedup_percent"])
        self.assertEqual(fixed["maximum_t90_upper_time_ratio"], 1.02)

        pairs = [
            {
                "forced": {"parsed": {"benchmark": {"seconds": forced}}},
                "disabled": {
                    "parsed": {"benchmark": {"seconds": disabled}}
                },
            }
            for forced, disabled in (
                (4.01, 4.00),
                (4.00, 4.01),
                (4.02, 4.01),
                (4.01, 4.02),
                (4.00, 4.00),
                (4.02, 4.02),
            )
        ]
        result = BENCHMARK.summarize_pairs(topology(1), fixed, pairs)
        gate_names = {gate["name"] for gate in result["gates"]}
        self.assertNotIn("minimum_median_paired_speedup_percent", gate_names)
        self.assertEqual(
            gate_names,
            {"maximum_t90_upper_forced_over_disabled_ratio"},
        )
        self.assertTrue(result["pass"])

    def test_exact_benchmark_schema_is_required(self) -> None:
        output = sample_output(
            "forced", benchmark_mutation={"unexpected": 0}
        )
        with self.assertRaisesRegex(RuntimeError, "fixed workload"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_cpu_phase_fallback_is_rejected(self) -> None:
        phase_calls = {
            name: {
                "cpu_calls": 0,
                "cuda_calls": 20,
                "expectation": "cuda_required",
            }
            for name in ("curl", "update_eh", "source", "boundary", "dft")
        }
        phase_calls["curl"] = {
            "cpu_calls": 1,
            "cuda_calls": 19,
            "expectation": "cuda_required",
        }
        phase_calls["polarization"] = {
            "cpu_calls": 0,
            "cuda_calls": 0,
            "expectation": "not_applicable",
        }
        output = sample_output(
            "forced",
            benchmark_mutation={"cpu_calls": 1, "cuda_calls": 99,
                                "phase_calls": phase_calls},
        )
        with self.assertRaisesRegex(RuntimeError, "fixed workload"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_cut_probe_shape_is_exact(self) -> None:
        output = sample_output(
            "forced", benchmark_mutation={"cut_probe_values": [[1.0, 0.0]]}
        )
        with self.assertRaisesRegex(RuntimeError, "cut-probe"):
            BENCHMARK.validate_sample(
                "forced", topology(), workload(), output
            )

    def test_expected_pid_gpu_map_rejects_reuse(self) -> None:
        parsed = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        sample = {"parsed": parsed}
        with self.assertRaisesRegex(RuntimeError, "PID was reused"):
            BENCHMARK.expected_pid_gpu_map(
                [sample, sample],
                {NVIDIA_SMI_UUIDS[0]: 0, NVIDIA_SMI_UUIDS[1]: 1},
            )

    def test_pid_gpu_map_uses_physical_uuid_mapping(self) -> None:
        parsed = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        mapping = BENCHMARK.expected_pid_gpu_map(
            [{"parsed": parsed}],
            {NVIDIA_SMI_UUIDS[0]: 1, NVIDIA_SMI_UUIDS[1]: 0},
        )
        self.assertEqual(mapping, {1000: 1, 1001: 0})

    def test_cuda_uuid_normalization_matches_runtime_and_nvidia_smi(self) -> None:
        self.assertEqual(
            BENCHMARK.canonical_cuda_uuid(RUNTIME_UUIDS[0]),
            RUNTIME_UUIDS[0],
        )
        self.assertEqual(
            BENCHMARK.canonical_cuda_uuid(NVIDIA_SMI_UUIDS[0]),
            RUNTIME_UUIDS[0],
        )
        self.assertEqual(
            BENCHMARK.canonical_cuda_uuid(NVIDIA_SMI_UUIDS[1].upper()),
            RUNTIME_UUIDS[1],
        )

    def test_cuda_uuid_normalization_rejects_malformed_and_colliding_maps(self) -> None:
        for malformed in (
            None,
            "",
            "gpu-0",
            "MIG-" + "a" * 32,
            "0" * 32,
            "GPU-00000000-0000-0000-0000-000000000000",
        ):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                RuntimeError, "UUID"
            ):
                BENCHMARK.canonical_cuda_uuid(malformed)
        with self.assertRaisesRegex(RuntimeError, "not one-to-one"):
            BENCHMARK.canonical_gpu_index_map(
                {RUNTIME_UUIDS[0]: 0, NVIDIA_SMI_UUIDS[0]: 1}
            )

    def test_cuda_uuid_map_rejects_invalid_or_nonunique_indices(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "empty"):
            BENCHMARK.canonical_gpu_index_map({})
        for mapping in (
            {RUNTIME_UUIDS[0]: -1},
            {RUNTIME_UUIDS[0]: True},
            {RUNTIME_UUIDS[0]: 0, RUNTIME_UUIDS[1]: 0},
        ):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(
                RuntimeError, "not one-to-one"
            ):
                BENCHMARK.canonical_gpu_index_map(mapping)

    def test_pid_gpu_map_rejects_worker_uuid_absent_from_inventory(self) -> None:
        parsed = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        with self.assertRaisesRegex(RuntimeError, "absent from nvidia-smi"):
            BENCHMARK.expected_pid_gpu_map(
                [{"parsed": parsed}], {NVIDIA_SMI_UUIDS[0]: 0}
            )

    def test_pid_gpu_map_boundary_rejects_incomplete_evidence(self) -> None:
        parsed = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        missing_process = json.loads(json.dumps(parsed))
        missing_process["processes"].pop()
        duplicate_ranks = json.loads(json.dumps(parsed))
        duplicate_ranks["devices"][1]["rank"] = 0
        duplicate_ranks["processes"][1]["rank"] = 0
        boolean_pid = json.loads(json.dumps(parsed))
        boolean_pid["processes"][0]["pid"] = True
        empty_device_name = json.loads(json.dumps(parsed))
        empty_device_name["devices"][0]["name"] = ""
        string_compute_major = json.loads(json.dumps(parsed))
        string_compute_major["devices"][0]["compute_major"] = "8"
        boolean_compute_minor = json.loads(json.dumps(parsed))
        boolean_compute_minor["devices"][0]["compute_minor"] = True
        incompatible_device = json.loads(json.dumps(parsed))
        incompatible_device["devices"][0]["compatible"] = False
        invalid_sets = (
            [],
            [{"parsed": None}],
            [{"parsed": missing_process}],
            [{"parsed": duplicate_ranks}],
            [{"parsed": boolean_pid}],
            [{"parsed": empty_device_name}],
            [{"parsed": string_compute_major}],
            [{"parsed": boolean_compute_minor}],
            [{"parsed": incompatible_device}],
        )
        for samples in invalid_sets:
            with self.subTest(samples=samples), self.assertRaises(RuntimeError):
                BENCHMARK.expected_pid_gpu_map(
                    samples,
                    {NVIDIA_SMI_UUIDS[0]: 0, NVIDIA_SMI_UUIDS[1]: 1},
                )

    def test_validate_sample_rejects_device_and_process_rank_gaps(self) -> None:
        base = sample_output("forced")
        invalid_outputs = (
            (
                "missing-device",
                mutate_marker(base, BENCHMARK.BASE.DEVICE_PREFIX, 1, drop=True),
                "CUDA device records",
            ),
            (
                "duplicate-device-rank",
                mutate_marker(
                    base,
                    BENCHMARK.BASE.DEVICE_PREFIX,
                    1,
                    mutation={"rank": 0},
                ),
                "CUDA topology",
            ),
            (
                "missing-process",
                mutate_marker(base, BENCHMARK.BASE.PROCESS_PREFIX, 1, drop=True),
                "process IDs",
            ),
            (
                "duplicate-process-rank",
                mutate_marker(
                    base,
                    BENCHMARK.BASE.PROCESS_PREFIX,
                    1,
                    mutation={"rank": 0},
                ),
                "process IDs",
            ),
            (
                "duplicate-process-pid",
                mutate_marker(
                    base,
                    BENCHMARK.BASE.PROCESS_PREFIX,
                    1,
                    mutation={"pid": 1000},
                ),
                "process IDs",
            ),
        )
        for label, output, message in invalid_outputs:
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError, message
            ):
                BENCHMARK.validate_sample(
                    "forced", topology(), workload(), output
                )

    def test_pid_gpu_map_accumulates_unique_pids_across_samples(self) -> None:
        first = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        second_output = mutate_marker(
            sample_output("disabled"),
            BENCHMARK.BASE.PROCESS_PREFIX,
            0,
            mutation={"pid": 2000},
        )
        second_output = mutate_marker(
            second_output,
            BENCHMARK.BASE.PROCESS_PREFIX,
            1,
            mutation={"pid": 2001},
        )
        second = BENCHMARK.validate_sample(
            "disabled", topology(), workload(), second_output
        )
        self.assertEqual(
            BENCHMARK.expected_pid_gpu_map(
                [{"parsed": first}, {"parsed": second}],
                {NVIDIA_SMI_UUIDS[0]: 0, NVIDIA_SMI_UUIDS[1]: 1},
            ),
            {1000: 0, 1001: 1, 2000: 0, 2001: 1},
        )

    def test_pid_gpu_map_rejects_same_gpu_under_two_uuid_formats(self) -> None:
        parsed = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )
        parsed["devices"] = [
            {**parsed["devices"][0], "uuid": RUNTIME_UUIDS[0]},
            {**parsed["devices"][1], "uuid": NVIDIA_SMI_UUIDS[0]},
        ]
        with self.assertRaisesRegex(RuntimeError, "not one-to-one"):
            BENCHMARK.expected_pid_gpu_map(
                [{"parsed": parsed}],
                {NVIDIA_SMI_UUIDS[0]: 0, NVIDIA_SMI_UUIDS[1]: 1},
            )

    def test_device_identity_change_is_rejected(self) -> None:
        first = BENCHMARK.validate_sample(
            "forced", topology(), workload(), sample_output("forced")
        )["devices"]
        second = [dict(record) for record in first]
        second[1]["uuid"] = "c" * 32
        catalog: dict[int, dict] = {}
        BENCHMARK.update_device_catalog(catalog, first)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            BENCHMARK.update_device_catalog(catalog, second)

    def test_process_monitor_binds_exact_worker_pids(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "# gpu pid type sm mem enc dec command\n"
                "0 1000 C 1 1 0 0 gpu-mpi-per\n"
                "1 1001 C 1 1 0 0 gpu-mpi-per\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertTrue(record["pass"])
        self.assertEqual(
            record["expected_pid_to_gpu"], {"1000": 0, "1001": 1}
        )

    def test_process_monitor_accepts_current_driver_rows_without_worker_name(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n"
                "0 1000 C 2 0 - - - - 324 0 3\n"
                "1 1001 C 1 0 - - - - 262 0\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertTrue(record["pass"])
        self.assertEqual(record["unrelated_process_rows"], [])

    def test_process_monitor_rejects_unrelated_pid_without_worker_name(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "0 1000 C 1 0 - - - - 100 0 3\n"
                "1 1001 C 1 0 - - - - 100 0\n"
                "0 9999 C 1 0 - - - - 100 0\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertFalse(record["pass"])
        self.assertEqual(len(record["unrelated_process_rows"]), 1)

    def test_process_monitor_rejects_empty_expected_pid_map(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text("0 - - - - - - - - - -\n", encoding="utf-8")
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {}
            )
        self.assertFalse(record["pass"])

    def test_process_monitor_rejects_missing_and_wrong_gpu_workers(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        cases = (
            "0 1000 C 1 0 - - - - 100 0\n",
            "1 1000 C 1 0 - - - - 100 0\n"
            "1 1001 C 1 0 - - - - 100 0\n",
        )
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "pmon.log"
                path.write_text(text, encoding="utf-8")
                stream = path.open("ab")
                record = BENCHMARK.stop_process_monitor(
                    FakeProcess(), stream, path, {1000: 0, 1001: 1}
                )
            self.assertFalse(record["pass"])

    def test_process_monitor_rejects_malformed_row(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "0 1000 C 1 0 - - - - 100 0\n"
                "1 1001 C 1 0 - - - - 100 0\n"
                "malformed-row\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertFalse(record["pass"])
        self.assertEqual(record["unrelated_process_rows"], ["malformed-row"])

    def test_process_monitor_rejects_dead_monitor(self) -> None:
        class FakeProcess:
            returncode = 0

            def poll(self) -> int:
                return self.returncode

            def terminate(self) -> None:
                raise AssertionError("dead monitor must not be terminated")

            def wait(self, timeout: int) -> int:
                return self.returncode

            def kill(self) -> None:
                raise AssertionError("dead monitor must not be killed")

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "0 1000 C 1 0 - - - - 100 0\n"
                "1 1001 C 1 0 - - - - 100 0\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertFalse(record["pass"])

    def test_process_monitor_rejects_bad_exit_after_termination(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 7

            def wait(self, timeout: int) -> int:
                assert self.returncode is not None
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "0 1000 C 1 0 - - - - 100 0\n"
                "1 1001 C 1 0 - - - - 100 0\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertFalse(record["pass"])

    def test_process_monitor_rejects_timeout_kill(self) -> None:
        class FakeProcess:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                pass

            def wait(self, timeout: int) -> int:
                if self.returncode is None:
                    raise BENCHMARK.subprocess.TimeoutExpired("pmon", timeout)
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pmon.log"
            path.write_text(
                "0 1000 C 1 0 - - - - 100 0\n"
                "1 1001 C 1 0 - - - - 100 0\n",
                encoding="utf-8",
            )
            stream = path.open("ab")
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), stream, path, {1000: 0, 1001: 1}
            )
        self.assertFalse(record["pass"])

    def test_success_terminal_binds_manifest_and_excludes_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            (root / "report.md").write_text("report\n", encoding="utf-8")
            BENCHMARK.finalize_success(root, "a" * 64)
            complete = json.loads(
                (root / "COMPLETE.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / BENCHMARK.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            names = {item["path"] for item in manifest["files"]}
            self.assertEqual(complete["state"], "COMPLETE")
            self.assertNotIn("COMPLETE.json", names)
            self.assertNotIn("FAILED.json", names)
            BENCHMARK.finalize_failure(root, {"state": "FAILED"})
            self.assertFalse((root / "FAILED.json").exists())

    def test_execution_handle_detects_in_place_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            image = root / "image"
            image.write_bytes(b"original")
            descriptor, binding = BENCHMARK.create_execution_handle(
                image, root, "image"
            )
            try:
                image.write_bytes(b"mutated!")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    BENCHMARK.verify_execution_handle(binding, descriptor)
            finally:
                import os
                os.close(descriptor)

    def test_observable_mutation_is_visible_to_canonical_comparison(self) -> None:
        baseline = BENCHMARK.validate_sample(
            "disabled", topology(), workload(), sample_output("disabled")
        )["observables"]
        mutated_output = sample_output("disabled").replace(
            '"dft_norm":5.0', '"dft_norm":5.25'
        )
        mutated = BENCHMARK.validate_sample(
            "disabled", topology(), workload(), mutated_output
        )["observables"]
        self.assertNotEqual(baseline, mutated)


if __name__ == "__main__":
    unittest.main()
