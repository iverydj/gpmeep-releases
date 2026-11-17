#!/usr/bin/env python3
"""Short CUDA children must have real, identity-bound observer rows before exit."""
from __future__ import annotations

import atexit
import copy
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOOK_PATH = ROOT / "scripts/python-validation/sitecustomize.py"
spec = importlib.util.spec_from_file_location("observation_test_hook", HOOK_PATH)
HOOK = importlib.util.module_from_spec(spec)
old_finders = list(sys.meta_path)
spec.loader.exec_module(HOOK)
sys.meta_path[:] = old_finders
atexit.unregister(HOOK._capture)

paired_spec = importlib.util.spec_from_file_location(
    "observation_test_paired", ROOT / "scripts/python-validation/run_m3_paired_matrix.py"
)
PAIRED = importlib.util.module_from_spec(paired_spec)
sys.modules[paired_spec.name] = PAIRED
paired_spec.loader.exec_module(PAIRED)
GPU = "GPU-11111111-1111-1111-1111-111111111111"
ENV = "GPMEEP_VALIDATION_GPU_OBSERVATION"


class ObservationBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.task = PAIRED.Task(9, "materials-sources-08", "test", "case.py", "test", "test", 9, None)
        self.directory = self.root / "telemetry" / PAIRED.task_name(self.task)
        self.directory.mkdir(parents=True)
        raw = pathlib.Path("/proc/self/stat").read_text()
        self.ticks = int(raw[raw.rfind(")") + 2:].split()[19])
        self.config = {
            "root_pid": os.getpid(), "root_start_time_ticks": self.ticks,
            "label": PAIRED.task_name(self.task), "gpu_uuid": GPU,
            "host_stream": str(self.directory / "host.jsonl"),
            "gpu_stream": str(self.directory / "gpu.jsonl"), "timeout_seconds": 33.0,
        }
        shared = {"root_pid": os.getpid(), "root_start_time_ticks": self.ticks,
                  "label": self.config["label"], "root_alive": True,
                  "sequence": 4, "captured_unix_seconds": time.time()}
        self.rows = {
            "host": dict(shared, schema="gpmeep-host-process-telemetry-sample-v2",
                         processes=[{"pid": os.getpid(), "start_time_ticks": self.ticks, "state": "R"}]),
            "gpu": dict(shared, schema="gpmeep-gpu-telemetry-sample-v3", query_error=None,
                        selected_gpu_uuids=[GPU], compute_applications=[{"pid": os.getpid(), "gpu_uuid": GPU}]),
        }
        self.result = {"expected_backend": "cuda", "active_backend": "cuda",
                       "strict_cuda_marker": True, "selected_device_identifier": "1" * 32,
                       "pid": os.getpid(), "run_nonce": "a" * 32}
        self.contract = {"gpu_device": GPU, "limits": {"telemetry_interval_seconds": 1.0}}
        self.write_rows()

    def write_rows(self):
        for kind, row in self.rows.items():
            pathlib.Path(self.config[kind + "_stream"]).write_text(json.dumps(row) + "\n")

    def observe(self):
        with mock.patch.object(HOOK, "_GPU_OBSERVATION_AT_START", json.dumps(self.config)):
            HOOK._await_gpu_observation(self.result)

    def verify(self):
        run = {"statistics": self.result, "run_nonce": self.result["run_nonce"],
               "environment_contract": {"environment": {ENV: json.dumps(self.config)}}}
        PAIRED.validate_child_observation(self.task, run, self.root, self.contract)

    def assert_times_out(self):
        with mock.patch.object(HOOK.time, "monotonic", side_effect=[0, 34]):
            with self.assertRaises(TimeoutError):
                self.observe()
        self.assertNotIn("gpu_observation", self.result)

    def test_real_positive_rows_are_bound_and_replayed(self):
        before = {kind: pathlib.Path(self.config[kind + "_stream"]).read_bytes() for kind in self.rows}
        self.observe()
        self.verify()
        self.assertEqual(self.result["gpu_observation"]["state"], "OBSERVED")
        for kind in self.rows:
            self.assertEqual(before[kind], pathlib.Path(self.config[kind + "_stream"]).read_bytes())

    def test_no_opt_in_does_not_read_or_wait(self):
        with mock.patch.object(HOOK, "_GPU_OBSERVATION_AT_START", None), mock.patch.object(HOOK, "_observation_tail", side_effect=AssertionError):
            HOOK._await_gpu_observation(self.result)
        self.assertNotIn("gpu_observation", self.result)

    def test_late_environment_change_cannot_enable_or_replace_barrier(self):
        with mock.patch.object(HOOK, "_GPU_OBSERVATION_AT_START", None), mock.patch.dict(os.environ, {ENV: json.dumps(self.config)}):
            HOOK._await_gpu_observation(self.result)
        self.assertNotIn("gpu_observation", self.result)
        with mock.patch.object(HOOK, "_GPU_OBSERVATION_AT_START", json.dumps(self.config)), mock.patch.dict(os.environ, {ENV: "hostile late value"}):
            HOOK._await_gpu_observation(self.result)
        self.verify()

    def test_declared_backend_release_still_accepts_real_child_observation(self):
        self.result.update(active_backend="cpu", selected_device_identifier="")
        self.observe()
        self.verify()

    def test_other_pid_or_reused_pid_cannot_release_child(self):
        for kind, key in (("gpu", "pid"), ("host", "start_time_ticks")):
            with self.subTest(kind=kind):
                saved = copy.deepcopy(self.rows)
                field = "compute_applications" if kind == "gpu" else "processes"
                self.rows[kind][field][0][key] += 1
                self.write_rows()
                self.assert_times_out()
                self.rows = saved

    def test_other_gpu_missing_apps_and_query_error_cannot_release(self):
        for change in ({"compute_applications": []}, {"query_error": "failed"}, {"selected_gpu_uuids": [GPU.replace("1", "2")]}):
            with self.subTest(change=change):
                original = self.rows["gpu"]
                self.rows["gpu"] = dict(original, **change)
                self.write_rows()
                self.assert_times_out()
                self.rows["gpu"] = original

    def test_stale_child_time_is_not_accepted(self):
        self.rows["gpu"]["captured_unix_seconds"] = HOOK._STARTED - 1
        self.write_rows()
        self.assert_times_out()

    def test_root_identity_label_or_terminal_mismatch_fails(self):
        for change in ({"root_start_time_ticks": self.ticks + 1}, {"label": "old-attempt"}, {"root_alive": False}):
            with self.subTest(change=change):
                original = self.rows["gpu"]
                self.rows["gpu"] = dict(original, **change)
                self.write_rows()
                with self.assertRaises(ValueError):
                    self.observe()
                self.rows["gpu"] = original

    def test_partial_append_is_not_evidence_until_newline(self):
        path = pathlib.Path(self.config["gpu_stream"])
        path.write_text(json.dumps(self.rows["gpu"]))
        def complete(_delay):
            with path.open("a") as stream:
                stream.write("\n")
        with mock.patch.object(HOOK.time, "sleep", side_effect=complete):
            self.observe()
        self.verify()

    def test_symlink_and_fifo_rejected_without_blocking(self):
        path = pathlib.Path(self.config["gpu_stream"])
        path.unlink()
        path.symlink_to(self.config["host_stream"])
        with self.assertRaises(OSError):
            self.observe()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(ValueError):
            self.observe()

    def test_changed_final_row_and_reference_cannot_replay(self):
        self.observe()
        path = pathlib.Path(self.config["gpu_stream"])
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"pid": ', b'"pid": 1'))
        with self.assertRaises(PAIRED.WorkloadError):
            self.verify()
        path.write_bytes(original)
        self.result["gpu_observation"]["samples"]["gpu"]["offset"] = 1
        with self.assertRaises(PAIRED.WorkloadError):
            self.verify()

    def test_timeout_propagates_as_capture_error(self):
        destination = self.root / "statistics.json"
        with mock.patch.dict(os.environ, {"GPMEEP_VALIDATION_STATS_FILE": str(destination)}, clear=True), mock.patch.object(HOOK, "_await_gpu_observation", side_effect=TimeoutError("not observed")):
            HOOK._capture(_native_snapshot=lambda: None)
        stats = json.loads(destination.read_text())
        self.assertEqual(stats["capture_status"], "capture_error")
        self.assertIn("not observed", stats["capture_error"])

    def test_replaced_or_truncated_stream_fails_while_waiting(self):
        path = pathlib.Path(self.config["gpu_stream"])
        self.rows["gpu"]["compute_applications"] = []
        self.write_rows()
        for replace in (False, True):
            with self.subTest(replace=replace):
                self.write_rows()
                def mutate(_delay):
                    if replace:
                        path.rename(path.with_suffix(".old"))
                    path.write_bytes(b"")
                with mock.patch.object(HOOK.time, "sleep", side_effect=mutate):
                    with self.assertRaisesRegex(ValueError, "replaced or truncated"):
                        self.observe()


if __name__ == "__main__":
    unittest.main()
