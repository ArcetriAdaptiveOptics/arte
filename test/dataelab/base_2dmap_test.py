# -*- coding: utf-8 -*-

import unittest
import numpy as np
import astropy.units as u

from arte.dataelab.base_2dmap import Base2dMap


class Base2dMapTest(unittest.TestCase):

    def setUp(self):
        self.map2d = np.array([[0, 1, 0],
                               [1, 1, 1]])
        self.m = Base2dMap(self.map2d)

    def test_nvalid(self):
        self.assertEqual(self.m.nvalid, 4)

    def test_shape(self):
        self.assertEqual(self.m.shape, (2, 3))

    def test_as_mask(self):
        # Masked-array convention: True where the element is NOT valid
        np.testing.assert_array_equal(self.m.as_mask(), self.map2d == 0)

    def test_from_idx1d(self):
        m = Base2dMap.from_idx1d((2, 3), [1, 3, 4, 5])
        np.testing.assert_array_equal(m.get_data(), self.map2d.astype(bool))

    def test_remap_image_2d_input(self):
        data = np.arange(8).reshape(2, 4)
        frame = self.m.remap_image(data)
        self.assertEqual(frame.shape, (2, 2, 3))
        # Valid elements are filled in row-major (np.nonzero) order
        np.testing.assert_array_equal(frame[0], [[0, 0, 0], [1, 2, 3]])
        np.testing.assert_array_equal(frame[1], [[0, 4, 0], [5, 6, 7]])

    def test_remap_image_1d_input(self):
        frame = self.m.remap_image(np.arange(4))
        self.assertEqual(frame.shape, (1, 2, 3))

    def test_remap_image_keeps_dtype(self):
        frame = self.m.remap_image(np.arange(4, dtype=np.float32))
        self.assertEqual(frame.dtype, np.float32)

    def test_remap_image_wrong_size(self):
        with self.assertRaises(ValueError):
            self.m.remap_image(np.arange(5))

    def test_map_has_no_unit_by_default(self):
        m = Base2dMap(self.map2d)
        self.assertFalse(isinstance(m.get_data(), u.Quantity))


if __name__ == "__main__":
    unittest.main()
