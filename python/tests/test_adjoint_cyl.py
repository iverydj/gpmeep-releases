import os

import meep as mp

try:
    import meep.adjoint as mpa
except:
    import adjoint as mpa

import unittest
from enum import Enum

import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from adjoint_cyl_fixed_work_contract import fixed_work_metrics_payload
from utils import ApproxComparisonTestCase

rng = np.random.RandomState(2)
resolution = 20
dimensions = mp.CYLINDRICAL
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

sr = 6
sz = 6
cell_size = mp.Vector3(sr, 0, sz)
dpml = 1.0
boundary_layers = [mp.PML(thickness=dpml)]

design_region_resolution = int(2 * resolution)
design_r = 5
design_z = 2
Nr, Nz = (
    int(design_r * design_region_resolution) + 1,
    int(design_z * design_region_resolution) + 1,
)

fcen = 1 / 1.55
width = 0.2
fwidth = width * fcen

## random design region
p = 0.5 * rng.rand(Nr * Nz)
## random epsilon perturbation for design region
deps = 1e-5
dp = deps * rng.rand(Nr * Nz)

# This tuple is consumed both by the four explicit unittest wrappers below and
# by the M4 import-only preflight.  Keep it limited to JSON-like scalar data so
# the preflight can prove the exact numerical cases without running a
# simulation or depending on a third-party test decorator.
GPMEEP_ADJOINT_CYL_TEST_CASES = (
    (0, (5, 0, 20)),
    (0, (4, 0, 28)),
    (-1, (5, 0, 20)),
    (1.2, (5, 0, 20)),
)

GPMEEP_ADJOINT_CYL_FIXED_WORK_ENV = "GPMEEP_ADJOINT_CYL_FIXED_WORK"
GPMEEP_ADJOINT_CYL_METRICS_PREFIX = "gpmeep-adjoint-cyl-metrics:"


def _fixed_work_requested():
    value = os.environ.get(GPMEEP_ADJOINT_CYL_FIXED_WORK_ENV)
    if value not in (None, "1"):
        raise RuntimeError(
            f"{GPMEEP_ADJOINT_CYL_FIXED_WORK_ENV} must be absent or exactly '1'"
        )
    return value == "1"


GPMEEP_ADJOINT_CYL_FIXED_WORK = _fixed_work_requested()


def _fixed_work_metrics_payload(rows, *, is_master):
    return fixed_work_metrics_payload(
        rows,
        is_master=is_master,
        expected_case_count=len(GPMEEP_ADJOINT_CYL_TEST_CASES),
        prefix=GPMEEP_ADJOINT_CYL_METRICS_PREFIX,
    )


def get_source(m):
    source_center = mp.Vector3(
        design_r / 2,
        0,
        -(sz / 2 - dpml + design_z / 2) / 2,
    )
    # exclude r=0 whenever |m| > 1
    source_size = mp.Vector3(
        design_r if abs(m) <= 1 else 0.5 * design_r,
        0,
        0,
    )
    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
    source = [
        mp.Source(
            src,
            component=mp.Er,
            center=source_center,
            size=source_size,
        )
    ]

    return source


def forward_simulation(design_params, m, far_x):
    matgrid = mp.MaterialGrid(
        mp.Vector3(Nr, 0, Nz),
        SiO2,
        Si,
        weights=design_params.reshape(Nr, 1, Nz),
        do_averaging=True,
    )

    geometry = [
        mp.Block(
            center=mp.Vector3(design_r / 2, 0, 0),
            size=mp.Vector3(design_r, 0, design_z),
            material=matgrid,
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=get_source(m),
        geometry=geometry,
        dimensions=dimensions,
        m=m,
    )

    frequencies = [fcen]

    mode = sim.add_near2far(
        frequencies,
        mp.Near2FarRegion(
            center=mp.Vector3(design_r / 2, 0, (sz / 2 - dpml + design_z / 2) / 2),
            size=mp.Vector3(design_r, 0, 0),
            weight=+1,
        ),
    )

    sim.run(until_after_sources=1200)
    Er = sim.get_farfield(mode, far_x[0])
    sim.reset_meep()

    return abs(Er[0]) ** 2


def adjoint_solver(design_params, m, far_x):

    design_variables = mp.MaterialGrid(
        mp.Vector3(Nr, 0, Nz), SiO2, Si, do_averaging=True
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(design_r / 2, 0, 0),
            size=mp.Vector3(design_r, 0, design_z),
        ),
    )
    geometry = [
        mp.Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variables,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=get_source(m),
        resolution=resolution,
        dimensions=dimensions,
        m=m,
    )

    NearRegions = [
        mp.Near2FarRegion(
            center=mp.Vector3(design_r / 2, 0, (sz / 2 - dpml + design_z / 2) / 2),
            size=mp.Vector3(design_r, 0, 0),
            weight=+1,
        )
    ]
    FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)
    ob_list = [FarFields]

    def J(alpha):
        return npa.abs(alpha[0, 0, 0]) ** 2

    fixed_work = (
        {
            "decay_by": 0.0,
            "minimum_run_time": 1200,
            "maximum_run_time": 1200,
        }
        if GPMEEP_ADJOINT_CYL_FIXED_WORK
        else {"maximum_run_time": 1200}
    )
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
        **fixed_work,
    )

    f, dJ_du = opt([design_params])
    sim.reset_meep()

    return f, dJ_du


class TestAdjointSolver(ApproxComparisonTestCase):
    fixed_work_metrics = []

    @classmethod
    def setUpClass(cls):
        cls.fixed_work_metrics = []

    @classmethod
    def tearDownClass(cls):
        if not GPMEEP_ADJOINT_CYL_FIXED_WORK:
            return
        payload = _fixed_work_metrics_payload(
            cls.fixed_work_metrics,
            is_master=not hasattr(mp, "am_master") or mp.am_master(),
        )
        if payload is not None:
            print(payload)

    def _check_adjoint_solver_cyl_n2f_fields(self, m, far_x):
        print("*** TESTING CYLINDRICAL Near2Far ADJOINT FEATURES ***")
        print(f"Current test: m={m}, far_x={far_x}")
        adjsol_obj, adjsol_grad = adjoint_solver(p, m, far_x)

        ## compute unperturbed S12
        S12_unperturbed = forward_simulation(p, m, far_x)

        ## compare objective results
        print(
            f"|Er|^2 -- adjoint solver: {adjsol_obj}, traditional simulation: {S12_unperturbed}"
        )

        self.assertClose(adjsol_obj, S12_unperturbed, epsilon=1e-3)

        ## compute perturbed S12
        S12_perturbed = forward_simulation(p + dp, m, far_x)

        ## compare gradients
        if adjsol_grad.ndim < 2:
            adjsol_grad = np.expand_dims(adjsol_grad, axis=1)
        adj_scale = (dp[None, :] @ adjsol_grad).flatten()
        fd_grad = S12_perturbed - S12_unperturbed
        print(f"Directional derivative -- adjoint solver: {adj_scale}, FD: {fd_grad}")
        if GPMEEP_ADJOINT_CYL_FIXED_WORK:
            case_index = len(self.fixed_work_metrics)
            expected_m, expected_far_point = GPMEEP_ADJOINT_CYL_TEST_CASES[
                case_index
            ]
            observed_far_point = [
                float(far_x[0].x),
                float(far_x[0].y),
                float(far_x[0].z),
            ]
            if float(m) != float(expected_m) or observed_far_point != [
                float(value) for value in expected_far_point
            ]:
                raise RuntimeError("fixed-work cylindrical adjoint case order differs")
            adjoint_directional_derivative = float(np.asarray(adj_scale).reshape(-1)[0])
            finite_difference_derivative = float(fd_grad)
            derivative_scale = max(
                abs(adjoint_directional_derivative),
                abs(finite_difference_derivative),
                1.0e-30,
            )
            self.fixed_work_metrics.append(
                {
                    "case_index": case_index,
                    "m": float(m),
                    "far_point": observed_far_point,
                    "adjoint_objective": float(np.asarray(adjsol_obj).reshape(-1)[0]),
                    "forward_objective": float(S12_unperturbed),
                    "perturbed_forward_objective": float(S12_perturbed),
                    "adjoint_directional_derivative": adjoint_directional_derivative,
                    "finite_difference_derivative": finite_difference_derivative,
                    "relative_derivative_error": abs(
                        adjoint_directional_derivative
                        - finite_difference_derivative
                    )
                    / derivative_scale,
                }
            )
        tol = 0.6 if m == 0 else 0.4
        self.assertClose(adj_scale, fd_grad, epsilon=tol)

    def _check_case(self, case_index):
        m, far_point = GPMEEP_ADJOINT_CYL_TEST_CASES[case_index]
        self._check_adjoint_solver_cyl_n2f_fields(
            m, [mp.Vector3(*far_point)]
        )

    # Keep the four stable unittest identities historically generated by
    # parameterized.expand while making this numerical validation runnable
    # from an installed gpmeep environment.  ``parameterized`` is a developer
    # test-runner dependency, not a Meep runtime dependency, and neither the
    # frozen v1.0.3 package nor a normal v1.1.0 installation provides it.
    def test_adjoint_solver_cyl_n2f_fields_0(self):
        self._check_case(0)

    def test_adjoint_solver_cyl_n2f_fields_1(self):
        self._check_case(1)

    def test_adjoint_solver_cyl_n2f_fields_2(self):
        self._check_case(2)

    def test_adjoint_solver_cyl_n2f_fields_3(self):
        self._check_case(3)


if __name__ == "__main__":
    unittest.main()
