"""test_adjoint_utils.py

Check various components of the adjoint solver codebase, like
filters, which may not need explicit gradient computation
(i.e. forward and adjoint runs).

"""
import gc
import meep as mp
from meep import backend_policy

try:
    import meep.adjoint as mpa
except:
    import adjoint as mpa

import unittest
import weakref
from enum import Enum

import numpy as np
import parameterized
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from utils import ApproxComparisonTestCase

_TOL = 1e-6 if mp.is_single_precision() else 1e-14

## ensure reproducible results
rng = np.random.RandomState(9861548)


class TestAdjointUtils(ApproxComparisonTestCase):
    def test_filtered_source_sparse_response_error_is_finite(self):
        """Zero requested bins are valid and must not poison diagnostics."""
        from meep.adjoint.filter_source import FilteredSource

        source = FilteredSource(
            center_frequency=0.85,
            frequencies=np.array([0.75, 0.95]),
            frequency_response=np.array(
                [[1.0 + 0.25j, 0.0], [0.0, -0.4 + 0.1j]]
            ),
            dt=0.04,
        )
        self.assertTrue(np.isfinite(source.err))
        self.assertGreaterEqual(source.err, 0.0)

    def test_filtered_source_callbacks_do_not_retain_owner(self):
        """Native callbacks retain calculation state, not the source owner."""
        from meep.adjoint.filter_source import FilteredSource

        source = FilteredSource(
            center_frequency=0.85,
            frequencies=np.array([0.75, 0.95]),
            frequency_response=np.array(
                [[1.0 + 0.25j, 0.0], [0.0, -0.4 + 0.1j]]
            ),
            dt=0.04,
        )
        sample_time = 0.1
        main_callback = source.src_func
        basis_callback = source.bf[0]
        expected_main = main_callback(sample_time)
        expected_basis = basis_callback(sample_time)
        np.testing.assert_allclose(expected_main, source(sample_time))
        source_ref = weakref.ref(source)
        basis_source_refs = [weakref.ref(item) for item in source.time_src_bf]

        del source
        for _ in range(3):
            gc.collect()

        self.assertIsNone(source_ref())
        self.assertTrue(all(item() is None for item in basis_source_refs))
        np.testing.assert_allclose(main_callback(sample_time), expected_main)
        np.testing.assert_allclose(basis_callback(sample_time), expected_basis)

    def test_filtered_source_policy_binds_semantic_state(self):
        """Late source registration omits only its derived native handle."""
        from meep.adjoint import filter_source
        from meep.adjoint.filter_source import FilteredSource

        source = FilteredSource(
            center_frequency=0.85,
            frequencies=np.array([0.75, 0.95]),
            frequency_response=np.array(
                [[1.0 + 0.25j, 0.0], [0.0, -0.4 + 0.1j]]
            ),
            dt=0.04,
        )
        self.assertIn(type(source), backend_policy._TRUSTED_MEEP_CLASSES)
        first = backend_policy._canonical(
            source, backend_policy._CanonicalTraversal()
        )
        instance_items = dict(
            first["value"]["callable_state"]["dict"]["value"]["mapping"]
        )
        self.assertNotIn("swigobj", instance_items)
        sample_time = 0.1
        np.testing.assert_allclose(source.src_func(sample_time), source(sample_time))
        source.nodes = source.nodes.copy()
        source.nodes.flat[0] += 0.125 - 0.05j
        np.testing.assert_allclose(source.src_func(sample_time), source(sample_time))
        changed = backend_policy._canonical(
            source, backend_policy._CanonicalTraversal()
        )
        self.assertNotEqual(first, changed)
        original_np = filter_source.np
        try:
            filter_source.np = object()
            self.assertFalse(
                backend_policy._is_trusted_meep_class(type(source))
            )
        finally:
            filter_source.np = original_np
        self.assertTrue(backend_policy._is_trusted_meep_class(type(source)))
        original_sqrt = filter_source.np.sqrt
        try:
            filter_source.np.sqrt = lambda value: value
            self.assertFalse(
                backend_policy._is_trusted_meep_class(type(source))
            )
        finally:
            filter_source.np.sqrt = original_sqrt
        self.assertTrue(backend_policy._is_trusted_meep_class(type(source)))
        methods = {
            name: member
            for name, member in FilteredSource.__dict__.items()
            if callable(member)
        }
        registration = compile(
            "_backend_policy._register_trusted_meep_class(FilteredSource)",
            filter_source.__file__,
            "exec",
        )
        for base in (object, FilteredSource.__bases__[0]):
            fabricated = type("FilteredSource", (base,), methods)
            fabricated.__module__ = filter_source.__name__
            fabricated.__qualname__ = "FilteredSource"
            filter_source.FilteredSource = fabricated
            try:
                with self.assertRaisesRegex(
                    backend_policy.BackendRecommendationError,
                    "definition differs from package source",
                ):
                    exec(registration, filter_source.__dict__)
            finally:
                filter_source.FilteredSource = FilteredSource
        original_class_name = FilteredSource.__name__
        try:
            FilteredSource.__name__ = "WrongName"
            self.assertFalse(
                backend_policy._is_trusted_meep_class(FilteredSource)
            )
        finally:
            FilteredSource.__name__ = original_class_name
        rect = FilteredSource.rect
        original_rect_name = rect.__name__
        original_rect_doc = rect.__doc__
        try:
            rect.__name__ = "wrong_rect"
            rect.__doc__ = "mutated package method"
            self.assertFalse(
                backend_policy._is_trusted_meep_class(FilteredSource)
            )
        finally:
            rect.__name__ = original_rect_name
            rect.__doc__ = original_rect_doc
        class_cell = FilteredSource.__init__.__closure__[0]
        original_class_cell = class_cell.cell_contents
        try:
            class_cell.cell_contents = FilteredSource.__bases__[0]
            self.assertFalse(
                backend_policy._is_trusted_meep_class(FilteredSource)
            )
        finally:
            class_cell.cell_contents = original_class_cell
        self.assertTrue(
            backend_policy._is_trusted_meep_class(FilteredSource)
        )

    @parameterized.parameterized.expand(
        [
            ("1.0_1.0_20_conic", 1.0, 1.0, 20, 0.24, mpa.conic_filter),
            ("1.0_1.0_23_conic", 1.0, 1.0, 23, 0.24, mpa.conic_filter),
            ("0.887_1.56_conic", 0.887, 1.56, 20, 0.24, mpa.conic_filter),
            ("0.887_1.56_conic", 0.887, 1.56, 31, 0.24, mpa.conic_filter),
            ("0.887_1.56_gaussian", 0.887, 1.56, 20, 0.24, mpa.gaussian_filter),
            ("0.887_1.56_cylindrical", 0.887, 1.56, 20, 0.24, mpa.cylindrical_filter),
        ]
    )
    def test_filter_offset(self, test_name, Lx, Ly, resolution, radius, filter_func):
        """ensure that the filters are indeed zero-phase"""
        print("Testing ", test_name)
        Nx, Ny = int(round(resolution * Lx)) + 1, int(round(resolution * Ly)) + 1
        x = np.random.rand(Nx, Ny)
        x = x + np.fliplr(x)
        x = x + np.flipud(x)
        y = filter_func(x, radius, Lx, Ly, resolution)
        self.assertClose(y, np.fliplr(y), epsilon=_TOL)
        self.assertClose(y, np.flipud(y), epsilon=_TOL)


if __name__ == "__main__":
    unittest.main()
