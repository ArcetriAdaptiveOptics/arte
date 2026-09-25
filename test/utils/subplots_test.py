#!/usr/bin/env python
import doctest
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.subplots import subplots


class SubplotsTest(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_docstring(self):
        import arte.utils.subplots as subplots_module
        doctest.testmod(subplots_module, raise_on_error=True)

    def test_size(self):
        default = np.array(plt.rcParams['figure.figsize'])
        fig, axs = subplots(2, 3, scale=2)
        np.testing.assert_allclose(fig.get_size_inches(), default * [3 * 2, 2 * 2])
        self.assertEqual(axs.shape, (2, 3))

    def test_single_axis(self):
        _, ax = subplots()
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main()
