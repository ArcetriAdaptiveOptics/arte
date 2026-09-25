#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
from arte.utils.karhunen_loeve_decomposer import KarhunenLoeveModalDecomposer
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.zernike_coefficients import ZernikeCoefficients


class ZernikeDecomposerStartModeTest(unittest.TestCase):

    def setUp(self):
        self.radius = 32
        self.mask = CircularMask((2 * self.radius, 2 * self.radius), self.radius)
        self.zg = ZernikeGenerator(self.mask)
        # 5 Z1 + 1 Z2 + 2 Z3 + 3 Z4 + 0.5 Z7
        self.true = {1: 5., 2: 1., 3: 2., 4: 3., 7: 0.5}
        self.wf = sum(c * self.zg.getZernike(j) for j, c in self.true.items())
        # partial pupil: on it the Zernike modes are not zero-mean
        partial = self.mask.mask().copy()
        partial[:, :self.radius // 2] = True
        self.partial_mask = BaseMask(partial)

    def _true(self, j):
        return self.true.get(j, 0.)

    def test_default_ignores_piston(self):
        md = ZernikeModalDecomposer(8)
        zc = md.measureZernikeCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(2, 10))
        self.assertRaises(IndexError, zc.getZ, 1)
        for j in zc.zernikeIndexes():
            self.assertAlmostEqual(zc.getZ(j), self._true(j), places=6)

    def test_start_mode_1_measures_piston_with_correct_labels(self):
        md = ZernikeModalDecomposer(9)
        zc = md.measureModalCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask, start_mode=1)
        self.assertIsInstance(zc, ZernikeCoefficients)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(1, 10))
        self.assertEqual(md.getLastRank(), 9)
        for j in zc.zernikeIndexes():
            self.assertAlmostEqual(zc.getZ(j), self._true(j), places=6)

    def test_start_mode_1_on_partial_mask_matches_default_fit(self):
        # Frisch-Waugh-Lovell: fitting piston explicitly is equivalent to
        # removing the mean from data and modes, for all the other modes
        wf = Wavefront(self.wf)
        zc_def = ZernikeModalDecomposer(8).measureZernikeCoefficientsFromWavefront(
            wf, self.mask, self.partial_mask)
        zc_pis = ZernikeModalDecomposer(9).measureModalCoefficientsFromWavefront(
            wf, self.mask, self.partial_mask, start_mode=1)
        np.testing.assert_allclose(zc_pis.getZ(list(range(2, 10))),
                                   zc_def.toNumpyArray(), atol=1e-9)
        self.assertAlmostEqual(zc_pis.getZ(1), 5., places=6)

    def test_start_mode_above_default(self):
        # only modes of the fitted basis (plus piston, always removed here)
        wf = 5. * self.zg.getZernike(1) + 3. * self.zg.getZernike(4) + \
            0.5 * self.zg.getZernike(7)
        md = ZernikeModalDecomposer(4)
        zc = md.measureModalCoefficientsFromWavefront(
            Wavefront(wf), self.mask, self.mask, start_mode=4)
        np.testing.assert_array_equal(zc.zernikeIndexes(), [4, 5, 6, 7])
        self.assertRaises(IndexError, zc.getZ, 3)
        self.assertAlmostEqual(zc.getZ(4), 3., places=6)
        self.assertAlmostEqual(zc.getZ(7), 0.5, places=6)

    def test_default_labels_after_start_mode_call(self):
        md = ZernikeModalDecomposer(4)
        md.measureModalCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask, start_mode=4)
        zc = md.measureZernikeCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask)
        np.testing.assert_array_equal(zc.zernikeIndexes(), [2, 3, 4, 5])

    def test_round_trip_with_piston(self):
        md = ZernikeModalDecomposer(9)
        zc = md.measureModalCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask, start_mode=1)
        wf_rec = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
        np.testing.assert_allclose(wf_rec.toNumpyArray().compressed(),
                                   np.ma.masked_array(self.wf, self.mask.mask()).compressed(),
                                   atol=1e-9)

    def test_recompose_follows_coefficient_labels(self):
        md = ZernikeModalDecomposer(3)
        want = self.zg.getZernike(1) * 7 + self.zg.getZernike(2)
        zc = ZernikeCoefficients.fromNumpyArray([7., 1.])
        zc.FIRST_ZERNIKE_MODE = 1
        wf = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
        np.testing.assert_allclose(wf.toNumpyArray().compressed(),
                                   np.ma.masked_array(want, self.mask.mask()).compressed(),
                                   atol=1e-9)

    def test_recompose_default_labels_unchanged(self):
        md = ZernikeModalDecomposer(3)
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])  # Z4
        wf = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
        z4 = np.ma.masked_array(self.zg.getZernike(4), self.mask.mask()).compressed()
        np.testing.assert_allclose(wf.toNumpyArray().compressed(),
                                   z4 - z4.mean(), atol=1e-9)

    def test_recompose_generic_modal_coefficients_uses_default(self):
        md = ZernikeModalDecomposer(3)
        mc = ModalCoefficients(np.array([0., 0., 1.]))
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])
        np.testing.assert_array_equal(
            md.recomposeWavefrontFromModalCoefficients(mc, self.mask).toNumpyArray(),
            md.recomposeWavefrontFromModalCoefficients(zc, self.mask).toNumpyArray())

    def test_recompose_inconsistent_start_mode_raises(self):
        md = ZernikeModalDecomposer(3)
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])
        self.assertRaises(ValueError, md.recomposeWavefrontFromModalCoefficients,
                          zc, self.mask, start_mode=1)


class KLDecomposerLabelsTest(unittest.TestCase):

    def test_kl_labels_start_from_zero(self):
        mask = CircularMask((32, 32), 16)
        wf = Wavefront(ZernikeGenerator(mask).getZernike(4))
        mc = KarhunenLoeveModalDecomposer(5).measureModalCoefficientsFromWavefront(
            wf, mask, mask)
        np.testing.assert_array_equal(mc.modeIndexes(), np.arange(5))


if __name__ == "__main__":
    unittest.main()
