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

    def test_default_labels(self):
        np.testing.assert_array_equal(self.z.zernikeIndexes(), [2, 3, 4])
        self.assertEqual(self.z.FIRST_MODE, 2)
        m = ModalCoefficients(np.array([1., 2.]))
        np.testing.assert_array_equal(m.modeIndexes(), [0, 1])
        m = ModalCoefficients(np.array([1., 2.]), first_mode=5)
        np.testing.assert_array_equal(m.modeIndexes(), [5, 6])

    def test_valid_indexes(self):
        self.assertEqual(self.z.getZ(2), 10)
        self.assertEqual(self.z.getZ(np.int64(3)), 20)
        self.assertEqual(np.ndim(self.z.getZ(2)), 0)
        np.testing.assert_array_equal(self.z.getZ([2, 3, 4]), [10, 20, 30])
        np.testing.assert_array_equal(self.z.getZ([4]), [30])
        np.testing.assert_array_equal(self.z.getZ(np.array([4, 2])), [30, 10])
        np.testing.assert_array_equal(self.z.getZ((2, 3)), [10, 20])
        np.testing.assert_array_equal(
            self.z.getZ([[2, 3], [4, 2]]), [[10, 20], [30, 10]])

    def test_missing_indexes_raise(self):
        for j in [1, [1], 0, -1, -2, 5, [4, 1], [2, 5]]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, self.z.getZ, j)

    def test_non_integer_indexes_raise(self):
        for j in [True, False, np.True_, [True, False, True],
                  np.array([True, False, True]), 2.0, [2.0], 2.5, []]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, self.z.getZ, j)

    def test_sparse_labels(self):
        z = ZernikeCoefficients(np.array([10., 20., 30.]), mode_indexes=[2, 3, 11])
        np.testing.assert_array_equal(z.zernikeIndexes(), [2, 3, 11])
        self.assertEqual(z.getZ(11), 30)
        np.testing.assert_array_equal(z.getZ([11, 2]), [30, 10])
        self.assertEqual(list(z.toDictionary().keys()), [2, 3, 11])
        for j in [4, 5, 10, 12, 1]:
            with self.subTest(j=j):
                self.assertRaises(IndexError, z.getZ, j)

    def test_unsorted_labels(self):
        m = ModalCoefficients(np.array([10., 20., 30.]), mode_indexes=[11, 2, 3])
        np.testing.assert_array_equal(m.modeIndexes(), [11, 2, 3])
        self.assertEqual(m.FIRST_MODE, 11)
        np.testing.assert_array_equal(m.getM([2, 3, 11]), [20, 30, 10])

    def test_piston_labels(self):
        z = ZernikeCoefficients(np.array([5., 1., 2.]), mode_indexes=range(1, 4))
        self.assertEqual(z.getZ(1), 5)
        self.assertEqual(z.FIRST_MODE, 1)

    def test_invalid_mode_indexes(self):
        for idx in [[2, 2, 3], [2, 3], [2, 3, 4, 5], [2., 3., 4.], [[2, 3, 4]]]:
            with self.subTest(idx=idx):
                self.assertRaises(ValueError, ModalCoefficients,
                                  np.zeros(3), mode_indexes=idx)

    def test_mode_indexes_are_read_only(self):
        self.assertRaises(ValueError, self.z.modeIndexes().__setitem__, 0, 7)

    def test_empty(self):
        m = ModalCoefficients(np.array([]), first_mode=3)
        self.assertEqual(m.numberOfModes(), 0)
        self.assertEqual(m.FIRST_MODE, 3)
        self.assertEqual(len(m.modeIndexes()), 0)


class ReadOnlyFirstModeTest(unittest.TestCase):

    def test_first_mode_cannot_be_set(self):
        z = ZernikeCoefficients(np.array([10., 20.]))
        with self.assertRaises(AttributeError):
            z.FIRST_MODE = 1
        m = ModalCoefficients(np.array([10., 20.]))
        with self.assertRaises(AttributeError):
            m.FIRST_MODE = 1

    def test_first_zernike_mode_cannot_be_set_on_instance(self):
        z = ZernikeCoefficients(np.array([10., 20.]))
        with self.assertRaises(AttributeError):
            z.FIRST_ZERNIKE_MODE = 1
        np.testing.assert_array_equal(z.zernikeIndexes(), [2, 3])

    def test_first_zernike_mode(self):
        self.assertEqual(ZernikeCoefficients.FIRST_ZERNIKE_MODE, 2)
        z = ZernikeCoefficients(np.array([10., 20.]), mode_indexes=[4, 5])
        self.assertEqual(z.FIRST_ZERNIKE_MODE, 4)


class ArithmeticTest(unittest.TestCase):

    def test_same_labels(self):
        a = ZernikeCoefficients(np.array([1., 2., 3.]))
        b = ZernikeCoefficients(np.array([10., 20., 30.]))
        res = a + b
        np.testing.assert_array_equal(res.zernikeIndexes(), [2, 3, 4])
        np.testing.assert_array_equal(res.toNumpyArray(), [11., 22., 33.])

    def test_different_length_same_first_mode_is_unchanged(self):
        # historical behaviour: shorter added to the head of the longer
        z1 = ZernikeCoefficients(np.arange(10, dtype=np.float32))
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32))
        want = np.arange(21, dtype=np.float32)
        want[:10] *= 2
        for res in [z1 + z2, z2 + z1]:
            np.testing.assert_array_equal(res.toNumpyArray(), want)
            np.testing.assert_array_equal(res.zernikeIndexes(), np.arange(2, 23))
            self.assertEqual(res.toNumpyArray().dtype, np.float32)

    def test_sum_on_union_of_labels(self):
        # issue #57
        a = ModalCoefficients(np.array([1., 2.]), first_mode=2)
        b = ModalCoefficients(np.array([10., 20.]), first_mode=5)
        res = a + b
        np.testing.assert_array_equal(res.modeIndexes(), [2, 3, 5, 6])
        np.testing.assert_array_equal(res.toNumpyArray(), [1., 2., 10., 20.])
        a = ZernikeCoefficients(np.array([1., 2., 3.]), mode_indexes=[1, 2, 3])
        b = ZernikeCoefficients(np.array([10., 20., 30., 40.]))
        for res in [a + b, b + a]:
            np.testing.assert_array_equal(res.zernikeIndexes(), [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(res.toNumpyArray(), [1., 12., 23., 30., 40.])

    def test_sum_unsorted_labels(self):
        a = ModalCoefficients(np.array([1., 2.]), mode_indexes=[7, 3])
        b = ModalCoefficients(np.array([10., 20.]), mode_indexes=[3, 5])
        res = a + b
        np.testing.assert_array_equal(res.modeIndexes(), [7, 3, 5])
        np.testing.assert_array_equal(res.toNumpyArray(), [1., 12., 20.])
        np.testing.assert_array_equal(res.getM([3, 5, 7]), [12., 20., 1.])

    def test_sum_different_dtypes(self):
        # issue #57, comment
        res = ZernikeCoefficients(np.zeros(1)) + ZernikeCoefficients(np.arange(10))
        np.testing.assert_array_equal(res.toNumpyArray(), np.arange(10.))
        res = ZernikeCoefficients(np.arange(3)) + ZernikeCoefficients(np.full(3, 0.5))
        np.testing.assert_array_equal(res.toNumpyArray(), [0.5, 1.5, 2.5])

    def test_subtraction_and_in_place(self):
        a = ZernikeCoefficients(np.array([5., 1., 1.]), mode_indexes=[1, 2, 3])
        b = ZernikeCoefficients(np.array([1., 1.]))
        np.testing.assert_array_equal((a - b).toNumpyArray(), [5., 0., 0.])
        np.testing.assert_array_equal((b - a).toNumpyArray(), [-5., 0., 0.])
        a += b
        np.testing.assert_array_equal(a.zernikeIndexes(), [1, 2, 3])
        np.testing.assert_array_equal(a.toNumpyArray(), [5., 2., 2.])
        a -= b
        np.testing.assert_array_equal(a.toNumpyArray(), [5., 1., 1.])
        c = ZernikeCoefficients(np.array([1.]), mode_indexes=[9])
        c += b
        np.testing.assert_array_equal(c.zernikeIndexes(), [2, 3, 9])
        np.testing.assert_array_equal(c.toNumpyArray(), [1., 1., 1.])

    def test_scalar_and_unary_ops_keep_labels_and_type(self):
        for z in [ZernikeCoefficients(np.array([1., 2., 4.]), mode_indexes=[1, 5, 9]),
                  ModalCoefficients(np.array([1., 2., 4.]), mode_indexes=[7, 3, 5])]:
            for res in [z + 1, 1 + z, z - 1, 1 - z, z * 2, 2 * z,
                        z / 2, 2 / z, -z, +z]:
                with self.subTest(cls=type(z).__name__, res=res):
                    np.testing.assert_array_equal(res.modeIndexes(), z.modeIndexes())
                    self.assertIs(type(res), type(z))

    def test_sum_of_list(self):
        zs = [ZernikeCoefficients(np.ones(3)), ZernikeCoefficients(np.ones(2))]
        np.testing.assert_array_equal(sum(zs).toNumpyArray(), [2., 2., 1.])

    def test_equality_takes_labels_into_account(self):
        a = ZernikeCoefficients(np.array([1., 2.]))
        self.assertEqual(a, ZernikeCoefficients(np.array([1., 2.])))
        self.assertNotEqual(a, ZernikeCoefficients(np.array([1., 2.]), mode_indexes=[1, 2]))


class PickleTest(unittest.TestCase):

    def test_roundtrip(self):
        z = ZernikeCoefficients(np.array([10., 20., 30.]), mode_indexes=[2, 3, 11])
        z2 = pickle.loads(pickle.dumps(z))
        self.assertEqual(z2, z)
        self.assertEqual(z2.getZ(11), 30.)

    def test_state_of_previous_versions(self):
        # Objects pickled before mode_indexes existed only have FIRST_MODE
        for cls, first_mode in [(ZernikeCoefficients, 2), (ModalCoefficients, 5)]:
            with self.subTest(cls=cls.__name__):
                obj = cls.__new__(cls)
                obj.__setstate__({'_coefficients': np.array([10., 20.]),
                                  '_counter': 3, 'FIRST_MODE': first_mode})
                np.testing.assert_array_equal(
                    obj.modeIndexes(), [first_mode, first_mode + 1])
                self.assertEqual(obj.getM(first_mode + 1), 20.)
                self.assertEqual(obj.FIRST_MODE, first_mode)
                self.assertEqual(obj.counter(), 3)


if __name__ == "__main__":
    unittest.main()
