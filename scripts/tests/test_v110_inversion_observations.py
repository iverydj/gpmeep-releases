"""Pure contract tests and a simulated capture-order test; no physical claims."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import types
import textwrap
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHILD = load("inversion_observation_child", "v110-inversion-qualified-child.py")
PRODUCER = load("inversion_observation_producer", "v110-inversion-experiment-sample.py")
MODULES = (CHILD, PRODUCER)


def empty_statistics():
    return {group: dict.fromkeys(names, 0) for group, names in CHILD.RUN_STATISTICS_KEYS.items()}


def observation(state="on"):
    before = empty_statistics()
    for group, key in CHILD.RUN_STATE_GAUGES:
        before[group][key] = 10
    before["dft_batches"]["dft_maximum_batch_size"] = 5
    after = copy.deepcopy(before)
    counters = after["boundary_execution"]
    counters.update(local_checks=6, prepare_calls=6, prepared_arrays=60,
                    prepare_host_nanoseconds=1000, plan_direct_local=6)
    if state == "on":
        counters.update(local_unready=1, local_replay_work=5, local_descriptor_rebuild_attempts=1)
    else:
        counters.update(local_disabled=6, local_descriptor_rebuild_attempts=6)
    for group, key in CHILD.RUN_STATE_GAUGES:
        after[group][key] = 5
    return {
        "schema": CHILD.RUN_OBSERVATION_SCHEMA, "index": 0, "operation": "run",
        "wall_seconds": 1.25, "statistics_before": before, "statistics_after": after,
        "field_relation": "initialized", "field_timing_before": None,
        "field_timing_after": {name: [1.0] for name, _ in CHILD.RUN_TIMING_SINKS},
    }


class TestInversionObservations(unittest.TestCase):
    def test_full_schema_matches_native_python_binding(self):
        binding = (Path(__file__).parents[2] / "python/meep.i").read_text()
        start = binding.index("        def statistics(self):")
        end = binding.index("    gpu = _GpuController()", start)
        method = ast.parse(textwrap.dedent(binding[start:end])).body[0]
        returned = next(node.value for node in method.body if isinstance(node, ast.Return))
        actual = {}
        for key, value in zip(returned.keys, returned.values):
            values = value.args[0] if isinstance(value, ast.Call) else value.generators[0].iter
            actual[ast.literal_eval(key)] = tuple(ast.literal_eval(values))
        for module in MODULES:
            self.assertEqual(actual, module.RUN_STATISTICS_KEYS)

    def test_curl_feature_summary_must_match_raw_workload_counters(self):
        raw = {"before": empty_statistics(), "after": empty_statistics()}
        features = {"curl_phase_replay": dict.fromkeys(
            ("checks", "hits", "unready", "generation_misses", "mirror_misses"), 0
        ), "phase_curl_forced_batches": 0}
        for module in MODULES:
            module._validate_workload_feature_projection(raw, features)
            for key in ("phase_curl_replay_checks", "phase_curl_forced_batches"):
                forged = copy.deepcopy(raw)
                forged["after"]["phase_batch_policy"][key] = 1
                with self.assertRaises((RuntimeError, ValueError)):
                    module._validate_workload_feature_projection(forged, features)

    def validate(self, module, record):
        return module._validate_run_observations(
            [record], {"run_count": 1, "primary_seconds": 1.25}, "run",
            {"before": record["statistics_before"], "after": record["statistics_after"]},
        )

    def reject(self, record):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                with self.assertRaises((RuntimeError, ValueError)):
                    self.validate(module, record)

    def test_both_validators_accept_gauge_decrease_and_both_real_arms(self):
        for module in MODULES:
            for state in ("off", "on"):
                value = observation(state)
                self.validate(module, value)
                totals = module._validate_boundary_replay_control(
                    "boundary-plan-replay=" + state, "cuda", [value]
                )
                self.assertEqual(totals["local_checks"], 6)

    def test_maximum_is_retained_and_must_not_decrease(self):
        value = observation()
        value["statistics_after"]["dft_batches"]["dft_maximum_batch_size"] = 4
        self.reject(value)

    def test_unknown_live_prefix_is_not_a_gauge(self):
        value = observation()
        value["statistics_before"]["resident"]["live_fake_counter"] = 3
        value["statistics_after"]["resident"]["live_fake_counter"] = 2
        self.reject(value)

    def test_missing_native_group_and_key_are_rejected(self):
        for group in (True, False):
            value = observation()
            for side in ("statistics_before", "statistics_after"):
                if group:
                    del value[side]["boundary_execution"]
                else:
                    del value[side]["boundary_execution"]["plan_graph"]
            self.reject(value)

    def test_missing_nonboundary_group_and_key_are_rejected(self):
        for remove_group in (True, False):
            value = observation()
            for side in ("statistics_before", "statistics_after"):
                if remove_group:
                    del value[side]["dispatch"]
                else:
                    del value[side]["dispatch"]["cuda_curl_calls"]
            self.reject(value)

    def test_point_read_outcomes_and_transfer_units_are_checked(self):
        value = observation()
        point = value["statistics_after"]["resident_point_reads"]
        point.update(cuda_read_attempts=2, cuda_read_completions=1,
                     cuda_read_failures=1, scalars_copied=3,
                     device_to_host_calls=3, device_to_host_bytes=12, host_nanoseconds=100)
        value["statistics_after"]["dispatch"]["device_to_host_bytes"] = 12
        for module in MODULES:
            self.validate(module, value)
        for key, invalid in (("cuda_read_attempts", 3), ("device_to_host_bytes", 8),
                             ("device_to_host_calls", 4), ("scalars_copied", 0)):
            forged = copy.deepcopy(value)
            forged["statistics_after"]["resident_point_reads"][key] = invalid
            self.reject(forged)
        forged = copy.deepcopy(value)
        forged["statistics_after"]["dispatch"]["device_to_host_bytes"] = 8
        self.reject(forged)

    def test_point_interval_cannot_reuse_earlier_total_transfer_budget(self):
        value = observation()
        for side, count in (("statistics_before", 1), ("statistics_after", 2)):
            value[side]["resident_point_reads"].update(
                cuda_read_attempts=count, cuda_read_completions=count,
                scalars_copied=count, device_to_host_calls=count,
                device_to_host_bytes=4 * count, host_nanoseconds=count,
            )
            value[side]["dispatch"]["device_to_host_bytes"] = 100
        self.reject(value)

    def test_point_interval_cannot_reuse_earlier_attempt_capacity(self):
        for extra_scalar in (True, False):
            value = observation()
            before = value["statistics_before"]["resident_point_reads"]
            before.update(cuda_read_attempts=1, cuda_read_completions=1,
                          scalars_copied=1, device_to_host_calls=1,
                          device_to_host_bytes=4, host_nanoseconds=100)
            after = copy.deepcopy(before)
            after["host_nanoseconds"] += 1
            if extra_scalar:
                after.update(scalars_copied=2, device_to_host_calls=2, device_to_host_bytes=8)
            value["statistics_after"]["resident_point_reads"] = after
            value["statistics_before"]["dispatch"]["device_to_host_bytes"] = 4
            value["statistics_after"]["dispatch"]["device_to_host_bytes"] = after["device_to_host_bytes"]
            for module in MODULES:
                module._validate_statistics_snapshot(value["statistics_before"])
                module._validate_statistics_snapshot(value["statistics_after"])
            self.reject(value)

    def test_repeated_or_overlapping_counter_intervals_are_rejected(self):
        first = observation()
        second = observation()
        second["index"] = 1
        for module in MODULES:
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_run_observations(
                    [first, second], {"run_count": 2, "primary_seconds": 2.5}, "run",
                    {"before": first["statistics_before"], "after": second["statistics_after"]},
                )

    def test_run_cannot_extend_beyond_global_counter_bounds(self):
        value = observation()
        for which in ("before", "after"):
            global_stats = {"before": copy.deepcopy(value["statistics_before"]),
                            "after": copy.deepcopy(value["statistics_after"])}
            if which == "after":
                global_stats["after"]["boundary_execution"]["prepare_calls"] = 0
            else:
                global_stats["before"]["boundary_execution"]["prepare_calls"] = 1
            for module in MODULES:
                with self.assertRaises((RuntimeError, ValueError)):
                    module._validate_run_observations(
                        [value], {"run_count": 1, "primary_seconds": 1.25}, "run", global_stats
                    )

    def test_between_run_work_is_retained_without_forcing_equality(self):
        first = observation()
        second = copy.deepcopy(first)
        second["index"] = 1
        second["statistics_before"] = copy.deepcopy(first["statistics_after"])
        second["statistics_before"]["dispatch"]["host_to_device_bytes"] += 100
        second["statistics_after"] = copy.deepcopy(second["statistics_before"])
        second["statistics_after"]["dispatch"]["host_to_device_bytes"] += 20
        global_stats = {"before": first["statistics_before"],
                        "after": copy.deepcopy(second["statistics_after"])}
        global_stats["after"]["dispatch"]["host_to_device_bytes"] += 300
        for module in MODULES:
            module._validate_run_observations(
                [first, second], {"run_count": 2, "primary_seconds": 2.5}, "run", global_stats
            )

    def test_counter_types_and_negative_values_are_rejected(self):
        for invalid in (True, 1.0, -1):
            value = observation()
            value["statistics_after"]["boundary_execution"]["plan_graph"] = invalid
            self.reject(value)

    def test_boundary_decision_conservation_is_checked(self):
        value = observation()
        value["statistics_after"]["boundary_execution"]["local_checks"] += 1
        self.reject(value)

    def test_same_epoch_timer_reset_is_rejected(self):
        value = observation()
        value["field_relation"] = "same"
        value["field_timing_before"] = {name: [2.0] for name, _ in CHILD.RUN_TIMING_SINKS}
        self.reject(value)

    def test_replacement_retains_raw_timer_epochs_without_subtraction(self):
        value = observation()
        value["field_relation"] = "replaced"
        value["field_timing_before"] = {name: [2.0] for name, _ in CHILD.RUN_TIMING_SINKS}
        for module in MODULES:
            self.validate(module, value)
            self.assertEqual(value["field_timing_before"]["boundaries"], [2.0])

    def test_epoch_presence_and_timing_rank_inventory_are_checked(self):
        value = observation()
        value["field_relation"] = "same"
        self.reject(value)
        value = observation()
        value["field_timing_after"]["boundaries"] = [1.0, 1.0]
        self.reject(value)

    def test_wall_sum_count_and_identity_are_checked(self):
        for key, invalid in (("wall_seconds", 1.5), ("index", True),
                             ("operation", "run_k_point"), ("schema", "old")):
            value = observation()
            value[key] = invalid
            self.reject(value)
        for module in MODULES:
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_run_observations(
                    [], {"run_count": 1, "primary_seconds": 1.25}, "run",
                    {"before": empty_statistics(), "after": empty_statistics()},
                )

    def test_wrong_arm_and_cpu_and_empty_replay_are_rejected(self):
        for module in MODULES:
            for state, actual, backend in (("on", "off", "cuda"), ("off", "on", "cuda"),
                                           ("on", "on", "cpu")):
                with self.assertRaises((RuntimeError, ValueError)):
                    module._validate_boundary_replay_control(
                        "boundary-plan-replay=" + state, backend, [observation(actual)]
                    )
            value = observation()
            counters = value["statistics_after"]["boundary_execution"]
            counters["local_replay_empty"] = counters.pop("local_replay_work")
            counters["local_replay_work"] = 0
            self.validate(module, value)
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_boundary_replay_control("boundary-plan-replay=on", "cuda", [value])

    def test_material_phasing_controls_require_matching_native_decisions(self):
        for module in MODULES:
            for state in ("off", "on"):
                value = observation()
                value["statistics_after"]["material_phasing"].update(
                    checks=5, reused=4 if state == "on" else 0,
                    disabled=5 if state == "off" else 0, ineligible=1 if state == "on" else 0)
                self.validate(module, value)
                totals = module._validate_material_phasing_control(
                    "phase-material-mirror-reuse=" + state, "cuda", [value])
                self.assertEqual(totals["checks"], 5)
                for backend, arm in (("cpu", state), ("cuda", "off" if state == "on" else "on")):
                    with self.assertRaises((RuntimeError, ValueError)):
                        module._validate_material_phasing_control(
                            "phase-material-mirror-reuse=" + arm, backend, [value])
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_material_phasing_control(
                    "phase-material-mirror-reuse=on", "cuda", [observation()])
            value = observation()
            value["statistics_after"]["material_phasing"].update(checks=5, ineligible=5)
            self.validate(module, value)
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_material_phasing_control(
                    "phase-material-mirror-reuse=on", "cuda", [value])

    def test_material_phasing_census_rejects_unaccounted_decisions(self):
        for key in ("checks", "reused", "disabled", "ineligible"):
            value = observation()
            value["statistics_after"]["material_phasing"][key] = 1
            self.reject(value)

    def test_material_phasing_environment_is_exact_and_clears_inherited_disable(self):
        hostile = {key: "0" for states in CHILD.ABLATIONS.values()
                   for values in states.values() for key in values}
        with patch.dict(os.environ, hostile):
            for state in ("off", "on"):
                setting = "phase-material-mirror-reuse=" + state
                expected = ({"MEEP_GPU_DISABLE_PHASE_MATERIAL_MIRROR_REUSE": "1"}
                            if state == "off" else {})
                self.assertEqual(CHILD._apply_ablation(setting), expected)
                _, parsed = PRODUCER._experiment_arguments(["--gpmeep-ablation", setting])
                self.assertEqual(parsed["ablation_environment"], expected)
                self.assertEqual({key: os.environ[key] for key in hostile if key in os.environ}, expected)

    def test_span_cache_control_requires_measured_matching_arms(self):
        for module in MODULES:
            for state in ("off", "on"):
                value = observation()
                counters = value["statistics_after"]["boundary_execution"]
                counters.update(span_cache_hits=50 if state == "on" else 0,
                                span_cache_misses=10 if state == "on" else 0,
                                span_cache_bypasses=60 if state == "off" else 0)
                self.validate(module, value)
                module._validate_boundary_span_cache_control(
                    "boundary-span-cache=" + state, "cuda", [value])
                for backend, arm in (("cpu", state), ("cuda", "off" if state == "on" else "on")):
                    with self.assertRaises((RuntimeError, ValueError)):
                        module._validate_boundary_span_cache_control(
                            "boundary-span-cache=" + arm, backend, [value])
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_boundary_span_cache_control(
                    "boundary-span-cache=on", "cuda", [observation()])

    def test_span_cache_outcomes_cannot_exceed_prepared_arrays(self):
        for key in ("span_cache_hits", "span_cache_misses", "span_cache_bypasses"):
            value = observation()
            value["statistics_after"]["boundary_execution"][key] = 6000000
            self.reject(value)
        # Valid cumulative totals can still conceal an impossible interval.
        for module in MODULES:
            before, after = empty_statistics(), empty_statistics()
            before["boundary_execution"].update(prepared_arrays=100)
            after["boundary_execution"].update(prepared_arrays=101, span_cache_hits=2)
            module._validate_statistics_snapshot(after)
            with self.assertRaises((RuntimeError, ValueError)):
                module._validate_statistics_progress(before, after)

    def test_span_cache_exact_controls_clear_inherited_switches(self):
        hostile = {key: "0" for states in CHILD.ABLATIONS.values()
                   for values in states.values() for key in values}
        with patch.dict(os.environ, hostile):
            for state in ("off", "on"):
                setting = "boundary-span-cache=" + state
                expected = ({"MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE": "1"}
                            if state == "off" else {})
                self.assertEqual(CHILD._apply_ablation(setting), expected)
                _, parsed = PRODUCER._experiment_arguments(["--gpmeep-ablation", setting])
                self.assertEqual(parsed["ablation_environment"], expected)
                self.assertEqual({key: os.environ[key] for key in hostile if key in os.environ}, expected)

    def test_exact_controls_clear_inherited_switches(self):
        hostile = {key: "0" for states in CHILD.ABLATIONS.values()
                   for values in states.values() for key in values}
        with patch.dict(os.environ, hostile):
            for state in ("off", "on"):
                setting = "boundary-plan-replay=" + state
                expected = ({"MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY": "1"}
                            if state == "off" else {})
                self.assertEqual(CHILD._apply_ablation(setting), expected)
                self.assertEqual(PRODUCER.ABLATION_ENVIRONMENTS[setting], expected)
                _, parsed = PRODUCER._experiment_arguments(["--gpmeep-ablation", setting])
                self.assertEqual(parsed["ablation_environment"], expected)
                self.assertEqual({key: os.environ[key] for key in hostile if key in os.environ}, expected)

    def test_bad_control_arguments_are_rejected(self):
        for setting in ("boundary-plan-replay=0", "boundary-plan-replay=on=x", "boundary-plan=on"):
            with self.assertRaises(CHILD.ChildError):
                CHILD._apply_ablation(setting)
            with self.assertRaises(PRODUCER.ExperimentError):
                PRODUCER._experiment_arguments(["--gpmeep-ablation", setting])
        with self.assertRaises(PRODUCER.ExperimentError):
            PRODUCER._experiment_arguments(["--gpmeep-ablation", "boundary-plan-replay=on", "--extra"])

    def test_capture_precedes_energy_probe_and_preserves_primary_timer(self):
        counters = empty_statistics()
        events = []

        class Fields:
            t = 2
            def total_volume(self):
                return None
            def dft_norm(self):
                events.append("dft_probe")
                return 0.0

        class Simulation:
            def __init__(self, *, resolution):
                self.fields = None
            def run(self, *, until):
                events.append("run")
                self.fields = Fields()
                counters["boundary_execution"].update(local_checks=1, local_replay_work=1,
                                                      prepare_calls=1)
            def get_timing_data(self):
                events.append("timing_snapshot")
                return {index: [1.0] for index, _ in enumerate(CHILD.RUN_TIMING_SINKS)}
            def timestep(self):
                return self.fields.t
            def round_time(self):
                return 0.1
            def field_energy_in_box(self, *, box):
                events.append("energy_probe")
                counters["boundary_execution"]["local_checks"] += 100
                counters["boundary_execution"]["local_replay_work"] += 100
                return 1.0

        def statistics():
            events.append("statistics_snapshot")
            return copy.deepcopy(counters)

        mp = types.SimpleNamespace(Simulation=Simulation,
                                   gpu=types.SimpleNamespace(statistics=statistics))
        for index, (_, attribute) in enumerate(CHILD.RUN_TIMING_SINKS):
            setattr(mp, attribute, index)
        seconds, numerical, observed = [], [], []
        ticks = iter((1000, 2000))
        def clock():
            tick = next(ticks)
            events.append("clock_start" if tick == 1000 else "clock_end")
            return tick
        restore = CHILD._install_scaling(mp, 1.0, 1.0, seconds, RuntimeError, [], [],
                                         numerical_runs=numerical, run_observations=observed)
        try:
            simulation = mp.Simulation(resolution=10)
            with patch.object(CHILD, "_backend_policy_run_record", return_value={}), \
                 patch.object(CHILD.time, "monotonic_ns", side_effect=clock):
                simulation.run(until=0.1)
        finally:
            restore()
        self.assertEqual(seconds, [0.000001])
        CHILD._validate_run_observations(
            observed, {"run_count": 1, "primary_seconds": seconds[0]}, "run",
            {"before": empty_statistics(), "after": counters},
        )
        self.assertEqual(CHILD._boundary_observation_totals(observed)["local_replay_work"], 1)
        self.assertEqual(counters["boundary_execution"]["local_replay_work"], 101)
        self.assertEqual(events, ["statistics_snapshot", "clock_start", "run", "clock_end", "statistics_snapshot",
                                  "timing_snapshot", "energy_probe", "dft_probe"])


if __name__ == "__main__":
    unittest.main()
