import contextlib
import json
import os
import unittest

import numpy as np

import meep as mp


@contextlib.contextmanager
def material_omp_control(enabled):
    saved = {
        "MEEP_C4_MATERIAL_OMP_ENABLE": os.environ.get(
            "MEEP_C4_MATERIAL_OMP_ENABLE"
        ),
        "MEEP_C4_MATERIAL_OMP_DISABLE": os.environ.get(
            "MEEP_C4_MATERIAL_OMP_DISABLE"
        ),
    }
    try:
        os.environ.pop("MEEP_C4_MATERIAL_OMP_ENABLE", None)
        os.environ.pop("MEEP_C4_MATERIAL_OMP_DISABLE", None)
        if enabled is not None:
            os.environ[
                "MEEP_C4_MATERIAL_OMP_ENABLE"
                if enabled
                else "MEEP_C4_MATERIAL_OMP_DISABLE"
            ] = "1"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestMaterialOpenMP(unittest.TestCase):
    def setUp(self):
        mp.gpu.set_backend("cpu")

    def tearDown(self):
        mp.gpu.set_backend("cpu")

    @staticmethod
    def _run_static_prism(enabled):
        susceptibility = mp.LorentzianSusceptibility(
            frequency=0.31, gamma=0.08, sigma=0.25
        )
        medium = mp.Medium(
            epsilon=2.4, E_susceptibilities=[susceptibility]
        )
        vertices = [
            mp.Vector3(-0.45, -0.35),
            mp.Vector3(0.50, -0.30),
            mp.Vector3(0.10, 0.55),
        ]
        with material_omp_control(enabled):
            mp.reset_material_omp_statistics()
            simulation = mp.Simulation(
                cell_size=mp.Vector3(2.4, 2.4),
                resolution=24,
                geometry=[
                    mp.Prism(vertices, height=mp.inf, material=medium)
                ],
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.28, fwidth=0.10),
                        component=mp.Ez,
                        center=mp.Vector3(-0.6, 0),
                    )
                ],
            )
            simulation.run(until=0.10)
            epsilon = simulation.get_array(component=mp.Dielectric).copy()
            field = simulation.get_array(component=mp.Ez).copy()
            material_manifest = json.loads(
                simulation.structure.material_arrays_manifest()
            )
            stats = mp.get_material_omp_statistics()
            evidence = {
                name: getattr(stats, name)
                for name in (
                    "feature_checks",
                    "enabled_checks",
                    "eligible_checks",
                    "disabled_checks",
                    "unsafe_material_checks",
                    "guard_rejections",
                    "stale_epoch_rejections",
                    "non_openmp_checks",
                    "epsilon_parallel_calls",
                    "epsilon_parallel_points",
                    "sigma_parallel_calls",
                    "sigma_parallel_points",
                    "team_observations",
                    "team_mismatch_observations",
                    "configured_threads",
                    "observed_threads",
                    "prism_preflight_calls",
                    "prism_preflight_complete",
                    "counter_overflow",
                )
            }
            simulation.reset_meep()
            return epsilon, field, material_manifest, evidence

    def test_static_prism_off_on_bitwise_and_causal(self):
        epsilon_off, field_off, manifest_off, stats_off = self._run_static_prism(
            False
        )
        epsilon_on, field_on, manifest_on, stats_on = self._run_static_prism(True)

        self.assertTrue(np.array_equal(epsilon_off, epsilon_on))
        self.assertTrue(np.array_equal(field_off, field_on))
        self.assertEqual(manifest_off, manifest_on)
        self.assertEqual(manifest_on["schema"], "gpmeep-material-arrays-v1")
        self.assertEqual(manifest_on["realnum_bytes"], 4)
        self.assertTrue(
            any(
                not array["null"] and array["sha256"]
                for chunk in manifest_on["chunks"]
                for array in chunk["chi1inv"]
            )
        )
        self.assertTrue(
            any(
                not array["null"] and array["sha256"]
                for chunk in manifest_on["chunks"]
                for susceptibility in chunk["susceptibilities"]
                for array in susceptibility["sigma"]
            )
        )
        self.assertEqual(stats_off["epsilon_parallel_calls"], 0)
        self.assertEqual(stats_off["sigma_parallel_calls"], 0)
        self.assertGreater(stats_off["disabled_checks"], 0)
        self.assertGreater(stats_on["enabled_checks"], 0)
        self.assertGreater(stats_on["eligible_checks"], 0)
        self.assertGreater(stats_on["epsilon_parallel_calls"], 0)
        self.assertGreater(stats_on["epsilon_parallel_points"], 0)
        self.assertGreater(stats_on["sigma_parallel_calls"], 0)
        self.assertGreater(stats_on["sigma_parallel_points"], 0)
        self.assertGreater(stats_on["team_observations"], 0)
        self.assertEqual(stats_on["team_mismatch_observations"], 0)
        self.assertGreaterEqual(stats_on["configured_threads"], 2)
        self.assertGreaterEqual(stats_on["observed_threads"], 2)
        self.assertTrue(stats_on["prism_preflight_complete"])
        self.assertEqual(stats_on["guard_rejections"], 0)
        self.assertEqual(stats_on["stale_epoch_rejections"], 0)
        self.assertEqual(stats_on["unsafe_material_checks"], 0)
        self.assertEqual(stats_on["non_openmp_checks"], 0)
        self.assertFalse(stats_on["counter_overflow"])

    def test_user_material_fails_closed_to_serial(self):
        def material(point):
            del point
            return mp.Medium(epsilon=1.8)

        with material_omp_control(True):
            mp.reset_material_omp_statistics()
            simulation = mp.Simulation(
                cell_size=mp.Vector3(1.5, 1.5),
                resolution=12,
                material_function=material,
            )
            simulation.init_sim()
            stats = mp.get_material_omp_statistics()
            self.assertGreater(stats.unsafe_material_checks, 0)
            self.assertEqual(stats.epsilon_parallel_calls, 0)
            self.assertEqual(stats.sigma_parallel_calls, 0)
            self.assertEqual(stats.guard_rejections, 0)
            self.assertEqual(stats.stale_epoch_rejections, 0)
            simulation.reset_meep()

    def test_unset_control_preserves_legacy_serial_path(self):
        with material_omp_control(None):
            mp.reset_material_omp_statistics()
            simulation = mp.Simulation(
                cell_size=mp.Vector3(1.5, 1.5),
                resolution=12,
                default_material=mp.Medium(epsilon=1.8),
            )
            simulation.init_sim()
            stats = mp.get_material_omp_statistics()
            self.assertGreater(stats.disabled_checks, 0)
            self.assertEqual(stats.enabled_checks, 0)
            self.assertEqual(stats.epsilon_parallel_calls, 0)
            self.assertEqual(stats.sigma_parallel_calls, 0)
            self.assertEqual(stats.guard_rejections, 0)
            self.assertEqual(stats.stale_epoch_rejections, 0)
            simulation.reset_meep()

    def test_malformed_or_conflicting_controls_fail_closed(self):
        control_names = (
            "MEEP_C4_MATERIAL_OMP_ENABLE",
            "MEEP_C4_MATERIAL_OMP_DISABLE",
        )
        saved = {name: os.environ.get(name) for name in control_names}
        cases = (("bad", None), (None, ""), ("1", "1"))
        try:
            for enable, disable in cases:
                with self.subTest(enable=enable, disable=disable):
                    for name in control_names:
                        os.environ.pop(name, None)
                    if enable is not None:
                        os.environ[control_names[0]] = enable
                    if disable is not None:
                        os.environ[control_names[1]] = disable
                    simulation = mp.Simulation(
                        cell_size=mp.Vector3(1.0, 1.0), resolution=8
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "mutually exclusive exact '=1' values"
                    ):
                        simulation.init_sim()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
