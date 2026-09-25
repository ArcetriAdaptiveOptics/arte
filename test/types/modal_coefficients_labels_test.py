#!/usr/bin/env python
import pickle
import unittest
import numpy as np
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.zernike_coefficients import ZernikeCoefficients


class FirstZernikeModeAliasTest(unittest.TestCase):

    def test_class_default_is_two(self):
        self.assertEqual(ZernikeCoefficients.FIRST_ZERNIKE_MODE, 2)
        z = ZernikeCoefficients(np.array([10., 20., 30.]))
        self.assertEqual(z.FIRST_MODE, 2)
        self.assertEqual(z.FIRST_ZERNIKE_MODE, 2)

    def test_runtime_change_on_instance_is_effective(self):
        z = ZernikeCoefficients(np.array([10., 20., 30.]))
        z.FIRST_ZERNIKE_MODE = 1
        self.assertEqual(z.FIRST_MODE, 1)
        np.testing.assert_array_equal(z.zernikeIndexes(), [1, 2, 3])
        self.assertEqual(z.getZ(1), 10.)
        self.assertEqual(list(z.toDictionary().keys()), [1, 2, 3])
        self.assertRaises(IndexError, z.getZ, 4)
        # the class default is not affected
        self.assertEqual(ZernikeCoefficients.FIRST_ZERNIKE_MODE, 2)
        self.assertEqual(ZernikeCoefficients(np.zeros(2)).FIRST_MODE, 2)

    def test_first_mode_and_alias_are_the_same(self):
        z = ZernikeCoefficients(np.array([10., 20., 30.]))
        z.FIRST_MODE = 5
        self.assertEqual(z.FIRST_ZERNIKE_MODE, 5)

    def test_constructor_first_mode(self):
        z = ZernikeCoefficients(np.array([10., 20.]), first_mode=1)
        self.assertEqual(z.getZ(1), 10.)
        z = ZernikeCoefficients.fromNumpyArray([10., 20.], first_mode=4)
        np.testing.assert_array_equal(z.zernikeIndexes(), [4, 5])

    def test_subclass_class_level_override(self):
        class ZWithPiston(ZernikeCoefficients):
            FIRST_ZERNIKE_MODE = 1
        self.assertEqual(ZWithPiston.FIRST_ZERNIKE_MODE, 1)
        z = ZWithPiston(np.array([10., 20., 30.]))
        self.assertEqual(z.getZ(1), 10.)
        z.FIRST_ZERNIKE_MODE = 3
        self.assertEqual(z.getZ(3), 10.)

    def test_pickle_keeps_runtime_first_mode(self):
        z = ZernikeCoefficients(np.array([10., 20., 30.]))
        z.FIRST_ZERNIKE_MODE = 1
        z2 = pickle.loads(pickle.dumps(z))
        self.assertEqual(z2.getZ(1), 10.)
        self.assertEqual(z2, z)


class ArithmeticKeepsLabelsTest(unittest.TestCase):

    def test_unary_and_scalar_ops_keep_first_mode(self):
        for z in [ZernikeCoefficients(np.array([1., 2., 3.]), first_mode=1),
                  ModalCoefficients(np.array([1., 2., 3.]), first_mode=5)]:
            for res in [z + 1, 1 + z, z - 1, 1 - z, z * 2, 2 * z,
                        z / 2, 2 / z, -z, +z]:
                with self.subTest(cls=type(z).__name__, res=res):
                    self.assertEqual(res.FIRST_MODE, z.FIRST_MODE)
                    self.assertIs(type(res), type(z))

    def test_same_first_mode_sum_is_unchanged(self):
        # historical behaviour: shorter added to the head of the longer,
        # dtype of the longer one
        z1 = ZernikeCoefficients(np.arange(10, dtype=np.float32))
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32))
        res = z1 + z2
        self.assertEqual(res.FIRST_MODE, 2)
        self.assertEqual(res.toNumpyArray().dtype, np.float32)
        want = np.arange(21, dtype=np.float32)
        want[:10] *= 2
        np.testing.assert_array_equal(res.toNumpyArray(), want)

    def test_sum_aligns_on_mode_index(self):
        # Z1..Z3 + Z2..Z5
        a = ZernikeCoefficients(np.array([1., 2., 3.]), first_mode=1)
        b = ZernikeCoefficients(np.array([10., 20., 30., 40.]))
        for res in [a + b, b + a]:
            self.assertEqual(res.FIRST_MODE, 1)
            np.testing.assert_array_equal(res.toNumpyArray(),
                                          [1., 12., 23., 30., 40.])
            self.assertEqual(res.getZ(4), 30.)

    def test_sum_with_gap(self):
        a = ModalCoefficients(np.array([1., 2.]), first_mode=0)
        b = ModalCoefficients(np.array([10.]), first_mode=4)
        res = a + b
        np.testing.assert_array_equal(res.modeIndexes(), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(res.toNumpyArray(), [1, 2, 0, 0, 10])

    def test_inplace_ops_align(self):
        a = ZernikeCoefficients(np.array([1., 2., 3.]), first_mode=1)
        b = ZernikeCoefficients(np.array([10., 20.]))
        a += b
        self.assertEqual(a.FIRST_MODE, 1)
        np.testing.assert_array_equal(a.toNumpyArray(), [1., 12., 23.])
        a -= b
        np.testing.assert_array_equal(a.toNumpyArray(), [1., 2., 3.])

    def test_sub_aligns(self):
        a = ZernikeCoefficients(np.array([5., 1., 1.]), first_mode=1)
        b = ZernikeCoefficients(np.array([1., 1.]))
        np.testing.assert_array_equal((a - b).toNumpyArray(), [5., 0., 0.])

    def test_equality_takes_labels_into_account(self):
        a = ZernikeCoefficients(np.array([1., 2.]))
        b = ZernikeCoefficients(np.array([1., 2.]), first_mode=1)
        self.assertNotEqual(a, b)
        self.assertEqual(a, ZernikeCoefficients(np.array([1., 2.])))


if __name__ == "__main__":
    unittest.main()
