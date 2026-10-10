#!/usr/bin/env python
import doctest
import unittest

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.gcfa import gcfa


class GcfaTest(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_docstring(self):
        import arte.utils.gcfa as gcfa_module
        doctest.testmod(gcfa_module, raise_on_error=True)

    def test_no_axis_clears_current_figure(self):
        fig0 = plt.figure()
        fig0.add_subplot(2, 1, 1)
        fig0.add_subplot(2, 1, 2)
        fig, ax = gcfa()
        self.assertIs(fig, fig0)
        self.assertEqual(len(fig.axes), 1)

    def test_axis_is_used(self):
        fig0, ax0 = plt.subplots()
        fig, ax = gcfa(ax0)
        self.assertIs(fig, fig0)
        self.assertIs(ax, ax0)

    def test_overplot(self):
        _, ax0 = plt.subplots()
        ax0.plot([0, 1])
        _, ax = gcfa(ax0, overplot=True)
        self.assertEqual(len(ax.lines), 1)
        _, ax = gcfa(ax0, overplot=False)
        self.assertEqual(len(ax.lines), 0)

    def test_colorbar_is_removed(self):
        fig0, ax0 = plt.subplots()
        im = ax0.imshow([[0, 1], [2, 3]])
        fig0.colorbar(im, ax=ax0)
        self.assertEqual(len(fig0.axes), 2)
        gcfa(ax0)
        self.assertEqual(len(fig0.axes), 1)


if __name__ == "__main__":
    unittest.main()
