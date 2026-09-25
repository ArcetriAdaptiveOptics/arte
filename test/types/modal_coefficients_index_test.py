#!/usr/bin/env python
import pickle
import unittest
import numpy as np
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.zernike_coefficients import ZernikeCoefficients


class GetModeIndexTest(unittest.TestCase):

    def setUp(self):
        # Z2, Z3, Z4
        self.z = ZernikeCoefficients(np.array([10., 20., 30.]))

    def test_valid_indexes(self):
        self.assertEqual(self.z.getZ(2), 10)
        self.assertEqual(self.z.getZ(np.int64(3)), 20)
        np.testing.assert_array_equal(self.z.getZ([2, 3, 4]), [10, 20, 30])
        np.testing.assert_array_equal(self.z.getZ(np.array([4, 2])), [30, 10])
        np.testing.assert_array_equal(self.z.getZ((2, 3)), [10, 20])
        np.testing.assert_array_equal(
            self.z.getZ([[2, 3], [4, 2]]), [[10, 20], [30, 10]])

    def test_piston_raises(self):
        self.assertRaises(IndexError, self.z.getZ, 1)
        self.assertRaises(IndexError, self.z.getZ, [1])

    def test_index_below_first_mode_raises(self):
        for j in [0, -1, -2, [4, 1], [2, 0]]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, self.z.getZ, j)

    def test_index_above_range_raises(self):
        self.assertRaises(IndexError, self.z.getZ, 5)
        self.assertRaises(IndexError, self.z.getZ, [2, 5])

    def test_booleans_raise(self):
        for j in [True, False, np.True_, [True, False, True],
                  np.array([True, False, True])]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, self.z.getZ, j)

    def test_negative_index_with_first_mode_zero_raises(self):
        m = ModalCoefficients(np.array([10., 20., 30.]))
        self.assertEqual(m.getM(0), 10)
        self.assertRaises(IndexError, m.getM, -1)

    def test_first_mode_changed_at_runtime_is_honoured(self):
        self.z.FIRST_MODE = 1
        self.assertEqual(self.z.getZ(1), 10.)
        np.testing.assert_array_equal(self.z.getZ([1, 3]), [10, 30])
        self.assertRaises(IndexError, self.z.getZ, 0)
        self.assertRaises(IndexError, self.z.getZ, 4)

    def test_generic_first_mode(self):
        m = ModalCoefficients(np.array([10., 20., 30.]), first_mode=5)
        self.assertEqual(m.getM(5), 10)
        self.assertRaises(IndexError, m.getM, 4)
        self.assertRaises(IndexError, m.getM, 8)

    def test_non_integer_indexes_still_raise(self):
        for j in [2.0, [2.0], 2.5, []]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, self.z.getZ, j)

    def test_subclass_overriding_getM(self):
        class Legacy(ZernikeCoefficients):
            def getM(self, modeIndexes):
                return 'legacy'
        self.assertEqual(Legacy(np.zeros(3)).getZ(1), 'legacy')

    def test_pickle_roundtrip(self):
        z2 = pickle.loads(pickle.dumps(self.z))
        self.assertEqual(z2, self.z)
        self.assertEqual(z2.getZ(3), 20.)
        self.assertRaises(IndexError, z2.getZ, 1)


if __name__ == "__main__":
    unittest.main()
