"""FilteredSource carrier metadata regression; source objects only, no solver."""
import unittest

import numpy as np


class TestFilteredSourceCarrier(unittest.TestCase):
    def test_filtered_source_basis_carriers_preserve_frequency_order(self):
        """Each native basis advertises its carrier; the combined source keeps its center."""
        from meep.adjoint.filter_source import FilteredSource

        for frequencies in ([1 / 0.95, 1 / 1.05], [1 / 1.05, 1 / 0.95]):
            with self.subTest(frequencies=frequencies):
                carriers = np.array(frequencies)
                center = float(np.mean(carriers))
                source = FilteredSource(
                    center_frequency=center,
                    frequencies=carriers,
                    frequency_response=np.array([1.0 + 0.25j, -0.4 + 0.1j]),
                    dt=0.04,
                )
                self.assertEqual(len(source.bf), len(carriers))
                self.assertEqual(len(source.time_src_bf), len(carriers))
                np.testing.assert_array_equal(source.center_frequencies, carriers)
                self.assertEqual(source.center_frequency, center)
                self.assertEqual(source.swigobj.frequency().real, center)
                for carrier, callback, basis in zip(
                    carriers, source.bf, source.time_src_bf
                ):
                    self.assertEqual(basis.center_frequency, carrier)
                    self.assertEqual(basis.swigobj.frequency().real, carrier)
                    self.assertEqual(basis.fwidth, source.fwidth)
                    self.assertEqual(basis.end_time, source.T)
                    self.assertFalse(basis.is_integrated)
                    for t in (0.0, source.dt, source.T / 3, source.T, source.T + source.dt):
                        self.assertEqual(basis.src_func(t), callback(t))
                        self.assertEqual(basis.swigobj.current(t, source.dt), callback(t))



if __name__ == "__main__":
    unittest.main()
