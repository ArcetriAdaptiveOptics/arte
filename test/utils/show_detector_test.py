#!/usr/bin/env python
import doctest
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.show_detector import ShowDetector


class ShowDetectorTest(unittest.TestCase):

    def setUp(self):
        self.array = np.arange(100, dtype=float).reshape(10, 10)
        self.sd = ShowDetector(self.array)

    def tearDown(self):
        plt.close('all')

    def _clim(self, ax):
        return ax.images[0].get_clim()

    def test_docstring(self):
        import arte.utils.show_detector as show_detector_module
        doctest.testmod(show_detector_module, raise_on_error=True)

    def test_default(self):
        ax = self.sd.show(show=False)
        self.assertEqual(self._clim(ax), (0, 99))
        self.assertEqual(ax.get_xlabel(), 'column [pix]')

    def test_cut_wings(self):
        ax = self.sd.show(cut_top_wing=10, cut_bottom_wing=10, show=False)
        np.testing.assert_allclose(self._clim(ax),
                                   np.percentile(self.array, [10, 90]))

    def test_cut_top_wing_clipped(self):
        ax = self.sd.show(cut_top_wing=-5, cut_bottom_wing=-5, show=False)
        self.assertEqual(self._clim(ax), (0, 99))
        _ = self.sd.show(cut_top_wing=200, cut_bottom_wing=10, show=False)

    def test_vmax_and_cut_bottom_wing(self):
        ax = self.sd.show(vmax=50, cut_bottom_wing=10, show=False)
        np.testing.assert_allclose(self._clim(ax),
                                   (np.percentile(self.array, 10), 50))

    def test_cut_bottom2mode(self):
        array = np.ones((10, 10))
        array[0, :5] = 3
        array[1, 0] = 0
        ax = ShowDetector(array).show(cut_bottom2mode=True, show=False)
        self.assertEqual(self._clim(ax), (1, 3))

    def test_log(self):
        ax = self.sd.show(log=True, show=False)
        self.assertEqual(self._clim(ax), (1, 99))

    def test_xy_and_options(self):
        x = np.linspace(-1, 1, 10)
        ax = self.sd.show(x=x, y=x, origin='upper', cells=(2, 2, 'r'),
                          units='ADU', title='t', show=False)
        self.assertEqual(ax.get_title(), 't')
        self.assertEqual(ax.get_xlabel(), 'x')
        self.assertEqual(len(ax.lines), 2)

    def test_no_colorbar(self):
        ax = self.sd.imshow(colorbar=False, show=False)
        self.assertEqual(len(ax.get_figure().axes), 1)

    def test_profiles(self):
        ax = self.sd.plot_xyc_profile(show=False, pix_scale=0.5)
        self.assertEqual(len(ax.lines), 2)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), self.array[5])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), self.array[:, 5])


if __name__ == "__main__":
    unittest.main()
