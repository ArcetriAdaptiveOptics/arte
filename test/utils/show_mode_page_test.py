#!/usr/bin/env python
import doctest
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.show_mode_page import show_mode_page


class ShowModePageTest(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_docstring(self):
        import arte.utils.show_mode_page as show_mode_page_module
        doctest.testmod(show_mode_page_module, raise_on_error=True)

    def test_last_page_has_no_stale_modes(self):
        modes = np.random.rand(4, 4, 6)
        axs = show_mode_page(modes, 0, 2, nrc=(1, 4), saturation=1, show=False)
        self.assertEqual(len(axs), 2)
        page1 = axs[1].images[0].get_array().filled(np.nan)
        self.assertTrue(np.all(np.isnan(page1[:, 8:])))
        np.testing.assert_allclose(page1[:, :4], np.flipud(modes[:, :, 4]))

    def test_saturation(self):
        modes = np.random.randn(4, 4, 4)
        axs = show_mode_page(modes, nrc=(2, 2), saturation=0.5, show=False)
        data = axs[0].images[0].get_array().filled(np.nan)
        self.assertAlmostEqual(np.nanmax(np.abs(data)), 0.5 * np.abs(modes).max())

    def test_pupil(self):
        pupil = np.zeros((4, 4), dtype=bool)
        pupil[1:3, 1:3] = True
        modes = np.random.rand(pupil.sum(), 3)
        axs = show_mode_page(modes, pupil=pupil, nrc=(1, 3), saturation=1, show=False)
        data = axs[0].images[0].get_array().filled(np.nan)
        self.assertEqual(np.sum(~np.isnan(data)), 12)

    def test_pages_beyond_modes(self):
        axs = show_mode_page(np.random.rand(4, 4, 3), 0, 5, nrc=(1, 2), show=False)
        self.assertEqual(len(axs), 2)

    def test_wrong_shape(self):
        with self.assertRaises(ValueError):
            show_mode_page(np.random.rand(16, 3), show=False)


if __name__ == "__main__":
    unittest.main()
