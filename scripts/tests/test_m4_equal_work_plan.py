#!/usr/bin/env python3
"""Tests for the 180-sample v1.1 M4 equal-work expansion."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/m4_equal_work_plan.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m4_equal_work_plan_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


class M4EqualWorkPlanTests(unittest.TestCase):
    def test_exact_census_and_reverse_cycle(self) -> None:
        plan = PLAN.load_plan(ROOT)
        self.assertEqual(plan["frozen_v103"], PLAN.FROZEN_V103)
        phase = next(
            case
            for case in plan["cases"]
            if case["path"] == PLAN.PHASE_IN_MATERIAL_CASE
        )
        self.assertEqual(phase["fixed_work"], PLAN.PHASE_IN_MATERIAL_FIXED_WORK)
        tasks = PLAN.build_tasks(plan)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(sum(task["sample_kind"] == "measured" for task in tasks), 120)
        by_cycle = {
            cycle: [task for task in tasks if task["cycle_index"] == cycle]
            for cycle in range(3)
        }
        self.assertEqual([len(by_cycle[index]) for index in range(3)], [60, 60, 60])
        forward_pairs = [
            (task["case_path"], task["lane_id"]) for task in by_cycle[1]
        ]
        reverse_pairs = [
            (task["case_path"], task["lane_id"]) for task in by_cycle[2]
        ]
        self.assertEqual(reverse_pairs, list(reversed(forward_pairs)))

    def test_cpu1_and_gpu2_scopes_are_exact(self) -> None:
        tasks = PLAN.build_tasks(PLAN.load_plan(ROOT))
        cpu1 = {
            task["case_path"]
            for task in tasks
            if task["topology"] in {"cpu1", "upstream-cpu1"}
        }
        self.assertEqual(cpu1, {"python/examples/material-dispersion.py"})
        gpu2 = {
            task["case_path"]
            for task in tasks
            if task["lane_id"] in PLAN.GPU2_LANES
        }
        self.assertEqual(
            gpu2,
            {
                "python/examples/adjoint_optimization/near2far_3d_validation.py",
                "python/examples/edge_emitter_3D.py",
                "python/examples/dipole_in_vacuum_cyl_off_axis.py",
                "python/tests/test_adjoint_cyl.py",
            },
        )

    def test_every_task_id_is_unique_and_lane_runtime_is_explicit(self) -> None:
        tasks = PLAN.build_tasks(PLAN.load_plan(ROOT))
        self.assertEqual(len({task["task_id"] for task in tasks}), len(tasks))
        for task in tasks:
            self.assertEqual(
                (task["runtime"], task["backend"]),
                PLAN.LANE_RUNTIME[task["lane_id"]],
            )


if __name__ == "__main__":
    unittest.main()
