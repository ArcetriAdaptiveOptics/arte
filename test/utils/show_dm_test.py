#!/usr/bin/env python
import doctest
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.show_dm import ShowDM


class ShowDMTest(unittest.TestCase):

    def setUp(self):
        self.coord = np.random.rand(10, 2)
        self.dm = ShowDM(self.coord, diam=(0.5, 2.0), opt_diam=1.5)

    def tearDown(self):
        plt.close('all')

    def test_docstring(self):
        import arte.utils.show_dm as show_dm_module
        doctest.testmod(show_dm_module, raise_on_error=True)

    def test_diameters(self):
        self.assertEqual(self.dm.din(), 0.5)
        self.assertEqual(self.dm.dout(), 2.0)
        self.assertIsNone(self.dm.opt_din())
        self.assertEqual(self.dm.opt_dout(), 1.5)
        dm = ShowDM(self.coord)
        self.assertIsNone(dm.din())
        self.assertIsNone(dm.dout())

    def test_z_coordinate(self):
        np.testing.assert_array_equal(self.dm.z, np.zeros(10))
        coord3 = np.random.rand(10, 3)
        np.testing.assert_array_equal(ShowDM(coord3).z, coord3[:, 2])

    def test_display(self):
        fig, ax = self.dm.display(np.arange(10.), title='t', plot_opt=True, plot_num=True)
        self.assertEqual(ax.get_title(), 't')
        self.assertEqual(len(ax.texts), 10)
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(len(fig.axes), 2)

    def test_act_list(self):
        _, ax = self.dm.display(np.arange(3.), act_list=[1, 4, 7], plot_num=True, colorbar=False)
        self.assertEqual([t.get_text() for t in ax.texts], ['1', '4', '7'])

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            self.dm.display(np.arange(3.))
        with self.assertRaises(ValueError):
            self.dm.display(np.arange(3.), act_list=[1, 2])


if __name__ == "__main__":
    unittest.main()
