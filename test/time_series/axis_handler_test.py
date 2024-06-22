#!/usr/bin/env python
import unittest
import numpy as np

from arte.time_series.axis_handler import AxisHandler


class AxisHandlerTest(unittest.TestCase):

    def test_axes_1d_same_order(self):
        handler = AxisHandler(axes=('time', 'data'))
        data = np.zeros((5, 4))
        assert handler.transpose(data, axes=('time', 'data')).shape == (5, 4)

    def test_axes_1d_transposed(self):
        handler = AxisHandler(axes=('time', 'data'))
        data = np.zeros((5, 4))
        assert handler.transpose(data, axes=('data', 'time')).shape == (4, 5)

    def test_axes_not_set_in_get_data(self):
        handler = AxisHandler(axes=('time', 'data'))
        data = np.zeros((5, 4))
        assert handler.transpose(data, None).shape == (5, 4)

    def test_axes_wrong(self):
        handler = AxisHandler(axes=('time', 'data'))
        data = np.zeros((5, 4))
        with self.assertRaises(ValueError):
            _ = handler.transpose(data, axes=('pippo', 'pluto'))

    def test_axes_2d_transposed(self):
        handler = AxisHandler(axes=('time', 'rows', 'cols'))
        data = np.zeros((2, 3, 4))
        assert handler.transpose(data, axes=('time', 'cols', 'rows')).shape == (2, 4, 3)


if __name__ == "__main__":
    unittest.main()
