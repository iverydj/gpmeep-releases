#!/usr/bin/env python3
"""Focused tests for the M3 multi-topology Python example unit."""

from __future__ import annotations

import importlib.util
import copy
import json
import pathlib
import pwd
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_mpi_example_case.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_mpi_example_case", SOURCE)
assert SPEC is not None and SPEC.loader is not None
CASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CASE
SPEC.loader.exec_module(CASE)


GPU0 = "GPU-00000000-0000-0000-0000-000000009001"
GPU1 = "GPU-00000000-0000-0000-0000-000000009002"


class M3MpiExampleCaseTests(unittest.TestCase):
    def test_fixed_topology_inventory(self) -> None:
        lanes = CASE.build_lanes(8, (GPU0, GPU1))
        self.assertEqual([lane.name for lane in lanes], list(CASE.LANE_ORDER))
        self.assertEqual([lane.ranks for lane in lanes], [8, 1, 2])
        self.assertEqual(lanes[0].devices, ())
        self.assertEqual(lanes[2].devices, (GPU0, GPU1))
        with self.assertRaisesRegex(Exception, "exactly 8"):
            CASE.build_lanes(4, (GPU0, GPU1))
        with self.assertRaisesRegex(Exception, "distinct"):
            CASE.build_lanes(8, (GPU0, GPU0.lower()))

    def test_lane_command_is_explicit(self) -> None:
        lane = CASE.Lane("cuda-fp32-2g", "cuda", 2, (GPU0, GPU1))
        command = CASE.lane_command(
            lane,
            mpiexec=pathlib.Path("/opt/bin/mpirun"),
            python=pathlib.Path("/opt/bin/python3.11"),
            launcher=pathlib.Path("/repo/launcher.py"),
            evidence_root=pathlib.Path("/evidence/lane"),
            case_command=["/opt/bin/python3.11", "/repo/example.py", "--validation"],
        )
        self.assertEqual(command[1:7], ["--bind-to", "core", "--map-by", "core:PE=1", "--report-bindings", "-n"])
        self.assertEqual(command[7], "2")
        self.assertEqual(command[-3:], ["/opt/bin/python3.11", "/repo/example.py", "--validation"])

    def test_clean_environment_retains_real_home_for_mpiexec_startup(self) -> None:
        lane = CASE.Lane("cpu-fp32-8r", "cpu", 8, ())
        with tempfile.TemporaryDirectory() as raw:
            parent = pathlib.Path(raw)
            environment = CASE.clean_environment(
                lane,
                python=pathlib.Path("/opt/bin/python3.11"),
                build_python=parent / "python",
                runtime_contract={
                    "fontconfig_file": {"path": str(parent / "fonts.conf")},
                    "receipt_id": "receipt",
                    "libmeep": {"path": str(parent / "lib/libmeep.so")},
                    "installed_environment": str(parent / "environment"),
                },
                run_nonce="nonce",
                site_guard_expected=True,
                producer_safe_ambient={
                    "PATH": "/producer/bin",
                    "TMPDIR": str(parent / "producer-tmp"),
                    "LC_ALL": "C",
                    "LANG": "C",
                },
            )
        self.assertEqual(
            environment["HOME"],
            str(pathlib.Path(pwd.getpwuid(CASE.os.getuid()).pw_dir).resolve()),
        )
        self.assertEqual(environment["MEEP_GPU_BACKEND"], "cpu")
        self.assertEqual(environment["OMPI_MCA_osc"], "^ucx")
        self.assertEqual(environment["TMPDIR"], str(parent / "producer-tmp"))
        self.assertEqual(
            environment[
                "GPMEEP_VALIDATION_BACKEND_ACKNOWLEDGEMENT_REASON"
            ],
            CASE.BACKEND_ACKNOWLEDGEMENT_REASON,
        )

    def test_producer_safe_ambient_is_explicit_and_fail_closed(self) -> None:
        ambient = {
            "PATH": "/producer/bin",
            "TMPDIR": "/producer/tmp",
            "LC_ALL": "C",
            "LANG": "C",
        }
        self.assertEqual(CASE.validate_producer_safe_ambient(ambient), ambient)
        self.assertIsNot(CASE.validate_producer_safe_ambient(ambient), ambient)
        with self.assertRaisesRegex(Exception, "safe ambient"):
            CASE.validate_producer_safe_ambient({"CUDA_VISIBLE_DEVICES": "0"})
        with self.assertRaisesRegex(Exception, "safe ambient"):
            CASE.validate_producer_safe_ambient({"LANG": 1})
        with self.assertRaisesRegex(Exception, "safe ambient"):
            CASE.validate_producer_safe_ambient({"LANG": "C\x00hostile"})
        with self.assertRaisesRegex(Exception, "safe ambient"):
            CASE.validate_producer_safe_ambient({**ambient, "TERM": "xterm"})
        missing = dict(ambient)
        missing.pop("TMPDIR")
        with self.assertRaisesRegex(Exception, "safe ambient"):
            CASE.validate_producer_safe_ambient(missing)

    def test_producer_safe_ambient_record_is_path_bound_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            task = "17-python-example"
            output = root / "raw" / task
            runtime_tmp = root / "runtime" / task / "tmp"
            output.mkdir(parents=True)
            runtime_tmp.mkdir(parents=True)
            python = root / "environment" / "bin" / "python3.11"
            values = {
                "PATH": f"{python.parent}:/usr/bin:/bin",
                "TMPDIR": str(runtime_tmp),
                "LC_ALL": "C",
                "LANG": "C",
            }
            record = CASE.producer_safe_ambient_record(values, output, python)
            self.assertEqual(
                CASE.validate_producer_safe_ambient_record(
                    record, output, python
                ),
                values,
            )

            for label, mutate in {
                "record hash": lambda item: item.update(sha256="0" * 64),
                "path": lambda item: item["values"].update(PATH="/wrong"),
                "tmp": lambda item: item["values"].update(TMPDIR="/wrong"),
                "locale": lambda item: item["values"].update(LANG="en_US"),
                "extra": lambda item: item["values"].update(TERM="xterm"),
            }.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(record)
                    mutate(changed)
                    if label != "record hash":
                        changed["sha256"] = CASE.validation.environment_sha256(
                            changed["values"]
                        )
                    with self.assertRaises(Exception):
                        CASE.validate_producer_safe_ambient_record(
                            changed, output, python
                        )

    def test_producer_safe_ambient_rejects_symlinked_runtime_layout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            task = "17-python-example"
            output = root / "raw" / task
            output.mkdir(parents=True)
            target = root / "actual-tmp"
            target.mkdir()
            runtime_parent = root / "runtime" / task
            runtime_parent.mkdir(parents=True)
            (runtime_parent / "tmp").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Exception, "symbolic link"):
                CASE.expected_producer_safe_ambient(
                    output, root / "environment/bin/python3.11"
                )

    def test_performance_safe_ambient_is_path_derived_and_ambient_free(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            task = "001-performance-case"
            output = root / "samples" / task
            runtime_tmp = root / "runtime" / task / "tmp"
            output.mkdir(parents=True)
            runtime_tmp.mkdir(parents=True)
            python = pathlib.Path(sys.executable).resolve(strict=True)
            expected = {
                "PATH": f"{python.parent}:/usr/bin:/bin",
                "TMPDIR": str(runtime_tmp),
                "LC_ALL": "C",
                "LANG": "C",
            }
            hostile = {
                "PATH": "/hostile",
                "TMPDIR": "/hostile/tmp",
                "LANG": "xx_YY",
                "LC_ALL": "xx_YY",
                "LC_CTYPE": "xx_YY",
                "TERM": "hostile",
                "TZ": "Etc/Hostile",
            }
            with mock.patch.dict(CASE.os.environ, hostile, clear=True):
                first = CASE.expected_performance_producer_safe_ambient(
                    output, python
                )
            with mock.patch.dict(CASE.os.environ, {}, clear=True):
                second = CASE.expected_performance_producer_safe_ambient(
                    output, python
                )
            self.assertEqual(first, expected)
            self.assertEqual(second, expected)
            self.assertEqual(
                CASE.validation.environment_sha256(first),
                CASE.validation.environment_sha256(second),
            )

    def test_performance_safe_ambient_rejects_uncontrolled_layouts(self) -> None:
        python = pathlib.Path(sys.executable).resolve(strict=True)
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            wrong = root / "raw" / "task"
            (root / "runtime" / "task" / "tmp").mkdir(parents=True)
            wrong.mkdir(parents=True)
            with self.assertRaisesRegex(Exception, "controlled layout"):
                CASE.expected_performance_producer_safe_ambient(wrong, python)

            missing = root / "samples" / "missing-runtime"
            missing.mkdir(parents=True)
            with self.assertRaisesRegex(Exception, "directory layout"):
                CASE.expected_performance_producer_safe_ambient(missing, python)

            target = root / "actual-tmp"
            target.mkdir()
            linked = root / "samples" / "linked-runtime"
            linked.mkdir()
            runtime_parent = root / "runtime" / "linked-runtime"
            runtime_parent.mkdir(parents=True)
            (runtime_parent / "tmp").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Exception, "symbolic link"):
                CASE.expected_performance_producer_safe_ambient(linked, python)

            relative = pathlib.Path("samples/task")
            with self.assertRaisesRegex(Exception, "must be absolute"):
                CASE.expected_performance_producer_safe_ambient(relative, python)

    def test_manifest_selection_accepts_pinned_zero_skip_mpi_unittest(self) -> None:
        case = CASE._case_from_manifest(
            ROOT,
            ROOT / "scripts/python-validation/manifest.json",
            "python/tests/test_adjoint_cyl.py",
        )
        self.assertEqual(case["kind"], "unittest")
        self.assertEqual(case["comparison"]["mode"], "embedded_oracle")
        self.assertEqual(case["allowed_unittest_skips"], 0)
        self.assertEqual(
            case["mpi_aggregate_cuda_call_counters"],
            ["near2far.cuda_near2far_adjoint_calls"],
        )
        self.assertEqual(
            case["mpi_required_per_rank_call_counters"],
            ["near2far.near2far_mpi_allreduce_calls"],
        )

    @staticmethod
    def _mpi_counter_stats(
        *, cuda_adjoint: int, cpu_adjoint: int, allreduces: int
    ) -> dict[str, object]:
        return {
            "statistics": {
                "near2far": {
                    "cuda_near2far_adjoint_calls": cuda_adjoint,
                    "cpu_near2far_adjoint_calls": cpu_adjoint,
                    "near2far_mpi_allreduce_calls": allreduces,
                }
            }
        }

    def test_monitor_local_cuda_activity_is_aggregate_with_rank_collectives(self) -> None:
        case = {
            "required_cuda_call_counters": [
                "near2far.cuda_near2far_adjoint_calls"
            ],
            "mpi_aggregate_cuda_call_counters": [
                "near2far.cuda_near2far_adjoint_calls"
            ],
            "mpi_required_per_rank_call_counters": [
                "near2far.near2far_mpi_allreduce_calls"
            ],
        }
        cuda_lane = CASE.Lane(
            "cuda-fp32-2g", "cuda", 2, (GPU0, GPU1)
        )
        cuda_stats = [
            self._mpi_counter_stats(
                cuda_adjoint=4, cpu_adjoint=0, allreduces=12
            ),
            self._mpi_counter_stats(
                cuda_adjoint=0, cpu_adjoint=0, allreduces=12
            ),
        ]
        evidence = CASE.validate_mpi_counter_contract(
            cuda_lane, case, cuda_stats
        )
        self.assertEqual(
            evidence["aggregate"][
                "near2far.cuda_near2far_adjoint_calls"
            ]["values"],
            [4, 0],
        )
        self.assertEqual(
            evidence["per_rank"][
                "near2far.near2far_mpi_allreduce_calls"
            ],
            [12, 12],
        )

        missing_aggregate = json.loads(json.dumps(cuda_stats))
        missing_aggregate[0]["statistics"]["near2far"][
            "cuda_near2far_adjoint_calls"
        ] = 0
        with self.assertRaisesRegex(
            Exception, "required aggregate CUDA activity"
        ):
            CASE.validate_mpi_counter_contract(
                cuda_lane, case, missing_aggregate
            )

        missing_participant = json.loads(json.dumps(cuda_stats))
        missing_participant[1]["statistics"]["near2far"][
            "near2far_mpi_allreduce_calls"
        ] = 0
        with self.assertRaisesRegex(Exception, "did not execute on ranks.*1"):
            CASE.validate_mpi_counter_contract(
                cuda_lane, case, missing_participant
            )

        cpu_fallback = json.loads(json.dumps(cuda_stats))
        cpu_fallback[1]["statistics"]["near2far"][
            "cpu_near2far_adjoint_calls"
        ] = 1
        with self.assertRaisesRegex(Exception, "recorded CPU fallback"):
            CASE.validate_mpi_counter_contract(
                cuda_lane, case, cpu_fallback
            )

        cpu_lane = CASE.Lane("cpu-fp32-8r", "cpu", 8, ())
        cpu_stats = [
            self._mpi_counter_stats(
                cuda_adjoint=0, cpu_adjoint=4, allreduces=12
            )
            for _ in range(cpu_lane.ranks)
        ]
        evidence = CASE.validate_mpi_counter_contract(
            cpu_lane, case, cpu_stats
        )
        self.assertEqual(
            evidence["aggregate"][
                "near2far.cuda_near2far_adjoint_calls"
            ]["cpu_sum"],
            32,
        )

        missing_cpu_reference = json.loads(json.dumps(cpu_stats))
        for stats in missing_cpu_reference:
            stats["statistics"]["near2far"][
                "cpu_near2far_adjoint_calls"
            ] = 0
        with self.assertRaisesRegex(
            Exception, "CPU reference did not execute paired aggregate"
        ):
            CASE.validate_mpi_counter_contract(
                cpu_lane, case, missing_cpu_reference
            )

    def _rank_root(self, parent: pathlib.Path) -> pathlib.Path:
        root = parent / "lane"
        for name in ("identity", "statistics", "work"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    def _write_rank(
        self,
        root: pathlib.Path,
        rank: int,
        lane: object,
        command: list[str],
        selected: str,
    ) -> None:
        name = f"rank-{rank:05d}.json"
        pid = 7000 + rank
        identity = {
            "schema": CASE.rank_launcher.SCHEMA,
            "rank": rank,
            "world_size": lane.ranks,
            "local_rank": rank,
            "pid": pid,
            "executable": command[0],
            "command": command,
            "run_nonce": "nonce",
            "requested_backend": lane.backend,
            "visible_devices": ",".join(lane.devices),
            "statistics": str(root / "statistics" / name),
            "work": str(root / "work" / f"rank-{rank:05d}"),
        }
        (root / "work" / f"rank-{rank:05d}").mkdir()
        (root / "identity" / name).write_text(json.dumps(identity))
        (root / "statistics" / name).write_text(
            json.dumps({"pid": pid, "selected_device_identifier": selected})
        )

    def test_rank_bundle_binds_both_physical_devices(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._rank_root(pathlib.Path(raw))
            lane = CASE.Lane("cuda-fp32-2g", "cuda", 2, (GPU0, GPU1))
            command = ["/opt/python3.11", "/repo/example.py"]
            self._write_rank(root, 0, lane, command, GPU0)
            self._write_rank(root, 1, lane, command, GPU1)
            with mock.patch.object(
                CASE.validation, "verify_backend_contract", return_value=(True, [])
            ):
                records = CASE.validate_rank_bundle(
                    lane, root, {"path": "example.py"}, {}, "nonce", command
                )
            self.assertEqual([record["rank"] for record in records], [0, 1])
            self.assertEqual(
                {record["selected_device_identifier"] for record in records},
                {GPU0, GPU1},
            )

            (root / "statistics" / "rank-00001.json").write_text(
                json.dumps({"pid": 7001, "selected_device_identifier": GPU0})
            )
            with mock.patch.object(
                CASE.validation, "verify_backend_contract", return_value=(True, [])
            ):
                with self.assertRaisesRegex(Exception, "distinct GPU"):
                    CASE.validate_rank_bundle(
                        lane, root, {"path": "example.py"}, {}, "nonce", command
                    )

    def test_tree_manifest_is_complete_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw) / "lane"
            (root / "a").mkdir(parents=True)
            (root / "a" / "value.bin").write_bytes(b"abc")
            tree = CASE.derive_tree(root)
            self.assertEqual(tree["file_count"], 1)
            self.assertEqual(tree["total_bytes"], 3)
            self.assertEqual(tree["files"][0]["path"], "a/value.bin")
            (root / "link").symlink_to(root / "a" / "value.bin")
            with self.assertRaisesRegex(Exception, "symlink"):
                CASE.derive_tree(root)

    def test_full_metric_comparisons_use_manifest_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            for name, value in zip(CASE.LANE_ORDER, (1.0, 1.001, 0.999)):
                lane = output / "lanes" / name
                lane.mkdir(parents=True)
                (lane / "stdout.log").write_text(f"metric:{json.dumps({'value': value})}\n")
            result = CASE.derive_comparisons(
                output,
                {
                    "comparison": {
                        "mode": "json_metrics",
                        "prefix": "metric:",
                        "tolerances": {"value": {"rtol": 0.01, "atol": 0.0}},
                    }
                },
            )
            self.assertEqual(set(result), {"cpu8-vs-gpu1", "cpu8-vs-gpu2", "gpu1-vs-gpu2"})
            self.assertTrue(all(value["outcome"] == "PASS" for value in result.values()))

    def test_embedded_unittest_requires_one_exact_suite_per_rank(self) -> None:
        case = CASE._case_from_manifest(
            ROOT,
            ROOT / "scripts/python-validation/manifest.json",
            "python/tests/test_adjoint_cyl.py",
        )
        lane = CASE.Lane("cuda-fp32-2g", "cuda", 2, (GPU0, GPU1))
        expected = case["expected_unittest"]["identities"]
        one_copy = "\n".join(
            [
                *[
                    f"test_method ({identity}) ... ok"
                    for identity in expected
                ],
                "----------------------------------------------------------------------",
                f"Ran {len(expected)} tests in 1.000s",
                "",
                "OK",
            ]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "stdout.log").write_text("", encoding="utf-8")
            (root / "stderr.log").write_text(
                one_copy + "\n" + one_copy + "\n", encoding="utf-8"
            )
            result = CASE.validate_unittest_mpi_output(lane, root, case)
            self.assertEqual(result["output_copies"], 2)
            self.assertEqual(result["outcome"], "PASS")

            (root / "stderr.log").write_text(one_copy + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "one exact zero-skip"):
                CASE.validate_unittest_mpi_output(lane, root, case)

    def test_embedded_unittest_accepts_rank_interleaved_verbose_lines(self) -> None:
        case = CASE._case_from_manifest(
            ROOT,
            ROOT / "scripts/python-validation/manifest.json",
            "python/tests/test_adjoint_cyl.py",
        )
        lane = CASE.Lane("cuda-fp32-2g", "cuda", 2, (GPU0, GPU1))
        expected = case["expected_unittest"]["identities"]
        interleaved = "\n".join(
            [
                *[
                    " ".join(
                        f"test_method ({identity}) ..." for _ in range(lane.ranks)
                    )
                    + " ok\nok"
                    for identity in expected
                ],
                *[
                    "\n".join(
                        [
                            "-" * 70,
                            f"Ran {len(expected)} tests in 1.000s",
                            "",
                            "OK",
                        ]
                    )
                    for _ in range(lane.ranks)
                ],
            ]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "stdout.log").write_text("", encoding="utf-8")
            (root / "stderr.log").write_text(interleaved, encoding="utf-8")
            result = CASE.validate_unittest_mpi_output(lane, root, case)
            self.assertEqual(result["output_copies"], lane.ranks)
            self.assertEqual(result["outcome"], "PASS")

    def test_embedded_oracle_comparisons_cover_all_topology_pairs(self) -> None:
        result = CASE.derive_comparisons(
            pathlib.Path("/unused"),
            {"comparison": {"mode": "embedded_oracle"}},
        )
        self.assertEqual(
            set(result), {"cpu8-vs-gpu1", "cpu8-vs-gpu2", "gpu1-vs-gpu2"}
        )
        self.assertTrue(
            all(
                value["outcome"] == "PASS" and value["failure_count"] == 0
                for value in result.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
