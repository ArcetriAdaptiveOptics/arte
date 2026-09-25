#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
from arte.utils.karhunen_loeve_decomposer import KarhunenLoeveModalDecomposer
from arte.utils.radial_basis_decomposer import RadialBasisModalDecomposer
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.zernike_coefficients import ZernikeCoefficients


class ZernikeDecomposerModeIndexesTest(unittest.TestCase):

    def setUp(self):
        self.radius = 32
        self.mask = CircularMask((2 * self.radius, 2 * self.radius), self.radius)
        self.zg = ZernikeGenerator(self.mask)
        # 5 Z1 + 1 Z2 + 2 Z3 + 3 Z4 + 0.5 Z7 + 0.25 Z11
        self.true = {1: 5., 2: 1., 3: 2., 4: 3., 7: 0.5, 11: 0.25}
        self.wf = sum(c * self.zg.getZernike(j) for j, c in self.true.items())
        # partial pupil: on it the Zernike modes are not zero-mean
        partial = self.mask.mask().copy()
        partial[:, :self.radius // 2] = True
        self.partial_mask = BaseMask(partial)

    def _measure(self, md, **kwargs):
        return md.measureModalCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask, **kwargs)

    def _assert_true_values(self, zc):
        for j in zc.zernikeIndexes():
            self.assertAlmostEqual(zc.getZ(j), self.true.get(j, 0.), places=6,
                                   msg='Z%d' % j)

    def test_default_labels(self):
        zc = ZernikeModalDecomposer(10).measureZernikeCoefficientsFromWavefront(
            Wavefront(self.wf), self.mask, self.mask)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(2, 12))
        self.assertRaises(IndexError, zc.getZ, 1)
        self._assert_true_values(zc)

    def test_sparse_mode_indexes(self):
        md = ZernikeModalDecomposer(10)
        wf = self.zg.getZernike(2) + 2 * self.zg.getZernike(3) + \
            3 * self.zg.getZernike(11)
        for idx in [[2, 3, 11], [11, 2, 3]]:
            with self.subTest(idx=idx):
                zc = md.measureZernikeCoefficientsFromWavefront(
                    Wavefront(wf), self.mask, mode_indexes=idx)
                self.assertIsInstance(zc, ZernikeCoefficients)
                np.testing.assert_array_equal(zc.zernikeIndexes(), idx)
                np.testing.assert_allclose(zc.getZ([2, 3, 11]), [1, 2, 3], atol=1e-9)
                self.assertRaises(IndexError, zc.getZ, 4)

    def test_mode_indexes_with_piston_measure_piston(self):
        zc = self._measure(ZernikeModalDecomposer(11), mode_indexes=range(1, 12))
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(1, 12))
        self._assert_true_values(zc)

    def test_start_mode_labels(self):
        md = ZernikeModalDecomposer(11)
        zc = self._measure(md, start_mode=1)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(1, 12))
        self._assert_true_values(zc)
        # without Z2 and Z3, not in the fitted basis
        self.wf = self.wf - self.zg.getZernike(2) - 2 * self.zg.getZernike(3)
        del self.true[2], self.true[3]
        zc = self._measure(ZernikeModalDecomposer(8), start_mode=4)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(4, 12))
        self._assert_true_values(zc)

    def test_piston_fit_on_partial_pupil_matches_default_fit(self):
        # Frisch-Waugh-Lovell: fitting piston explicitly is equivalent to
        # removing the mean from data and modes, for all the other modes
        wf = Wavefront(self.wf)
        md = ZernikeModalDecomposer(10)
        zc_def = md.measureZernikeCoefficientsFromWavefront(
            wf, self.mask, self.partial_mask)
        zc_pis = md.measureZernikeCoefficientsFromWavefront(
            wf, self.mask, self.partial_mask, mode_indexes=range(1, 12))
        np.testing.assert_allclose(zc_pis.getZ(np.arange(2, 12)),
                                   zc_def.toNumpyArray(), atol=1e-9)
        self.assertAlmostEqual(zc_pis.getZ(1), 5., places=6)

    def test_inconsistent_arguments_raise(self):
        md = ZernikeModalDecomposer(3)
        self.assertRaises(ValueError, self._measure, md,
                          mode_indexes=[2, 3], start_mode=2)
        self.assertRaises(ValueError, self._measure, md,
                          mode_indexes=[2, 3], nModes=3)
        self.assertRaises(ValueError, self._measure, md, mode_indexes=[2, 2])

    def test_equivalent_selections_share_the_cache(self):
        md = ZernikeModalDecomposer(4)
        # functools.cache is shared by all the instances: count the misses
        cache_info = md.cachedSyntheticReconstructorFromWavefront.cache_info
        misses = cache_info().misses
        results = [self._measure(md), self._measure(md, start_mode=2),
                   self._measure(md, mode_indexes=[2, 3, 4, 5]),
                   self._measure(md, nModes=4, mode_indexes=range(2, 6))]
        for zc in results[1:]:
            self.assertEqual(zc, results[0])
        self.assertEqual(cache_info().misses - misses, 1)

    def test_slopes_mode_indexes(self):
        idx = [2, 3, 7]
        coeffs = [1., -2., 0.5]
        dx = sum(c * self.zg.getDerivativeX(j) for j, c in zip(idx, coeffs))
        dy = sum(c * self.zg.getDerivativeY(j) for j, c in zip(idx, coeffs))
        slopes = Slopes.from_2dmaps(dx, dy)
        md = ZernikeModalDecomposer(10)
        zc = md.measureZernikeCoefficientsFromSlopes(
            slopes, self.mask, mode_indexes=idx)
        np.testing.assert_array_equal(zc.zernikeIndexes(), idx)
        np.testing.assert_allclose(zc.getZ(idx), coeffs, atol=1e-9)
        zc = md.measureZernikeCoefficientsFromSlopes(slopes, self.mask)
        np.testing.assert_array_equal(zc.zernikeIndexes(), np.arange(2, 12))
        np.testing.assert_allclose(zc.getZ(idx), coeffs, atol=1e-9)

    def _compressed(self, a):
        return np.ma.masked_array(a, self.mask.mask()).compressed()

    def test_round_trip(self):
        md = ZernikeModalDecomposer(11)
        for kwargs in [dict(mode_indexes=range(1, 12)),
                       dict(mode_indexes=[11, 1, 2, 3, 4, 7])]:
            with self.subTest(**kwargs):
                zc = self._measure(md, **kwargs)
                wf = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
                np.testing.assert_allclose(wf.toNumpyArray().compressed(),
                                           self._compressed(self.wf), atol=1e-9)

    def test_recompose_follows_labels(self):
        md = ZernikeModalDecomposer(3)
        zc = ZernikeCoefficients.fromNumpyArray([7., 1.], mode_indexes=[1, 11])
        wf = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
        want = self._compressed(7 * self.zg.getZernike(1) + self.zg.getZernike(11))
        np.testing.assert_allclose(wf.toNumpyArray().compressed(), want, atol=1e-9)

    def test_recompose_default_labels_unchanged(self):
        md = ZernikeModalDecomposer(3)
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])  # Z4
        wf = md.recomposeWavefrontFromModalCoefficients(zc, self.mask)
        z4 = self._compressed(self.zg.getZernike(4))
        np.testing.assert_allclose(wf.toNumpyArray().compressed(),
                                   z4 - z4.mean(), atol=1e-9)

    def test_recompose_generic_modal_coefficients_as_z2(self):
        md = ZernikeModalDecomposer(3)
        mc = ModalCoefficients(np.array([0., 0., 1.]))
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])
        with self.assertWarns(DeprecationWarning):
            wf = md.recomposeWavefrontFromModalCoefficients(mc, self.mask)
        np.testing.assert_array_equal(
            wf.toNumpyArray(),
            md.recomposeWavefrontFromModalCoefficients(zc, self.mask).toNumpyArray())

    def test_recompose_inconsistent_start_mode_raises(self):
        md = ZernikeModalDecomposer(3)
        zc = ZernikeCoefficients.fromNumpyArray([0., 0., 1.])
        self.assertRaises(ValueError, md.recomposeWavefrontFromModalCoefficients,
                          zc, self.mask, start_mode=1)
        md.recomposeWavefrontFromModalCoefficients(zc, self.mask, start_mode=2)


class OtherDecomposersTest(unittest.TestCase):

    def setUp(self):
        self.mask = CircularMask((32, 32), 16)
        self.wf = Wavefront(ZernikeGenerator(self.mask).getZernike(4))

    def test_default_first_mode_matches_generator(self):
        coords = [(10, 10), (16, 16), (20, 12)]
        for md in [ZernikeModalDecomposer(3), KarhunenLoeveModalDecomposer(3),
                   RadialBasisModalDecomposer(coords)]:
            with self.subTest(md=type(md).__name__):
                gen = md._generator(3, self.mask, self.mask, rbfFunction='TPS_RBF')
                self.assertEqual(md.DEFAULT_FIRST_MODE, gen.first_mode())

    def test_kl_labels(self):
        md = KarhunenLoeveModalDecomposer(5)
        mc = md.measureModalCoefficientsFromWavefront(self.wf, self.mask, self.mask)
        np.testing.assert_array_equal(mc.modeIndexes(), np.arange(5))
        sub = md.measureModalCoefficientsFromWavefront(
            self.wf, self.mask, self.mask, mode_indexes=[4, 0, 2])
        np.testing.assert_array_equal(sub.modeIndexes(), [4, 0, 2])
        wf = md.recomposeWavefrontFromModalCoefficients(sub, self.mask)
        self.assertEqual(wf.toNumpyArray().shape, (32, 32))

    def test_rbf_labels(self):
        coords = [(10, 10), (16, 16), (20, 12), (12, 20)]
        md = RadialBasisModalDecomposer(coords)
        mc = md.measureModalCoefficientsFromWavefront(
            self.wf, self.mask, self.mask, mode_indexes=[3, 1], rbfFunction='TPS_RBF')
        np.testing.assert_array_equal(mc.modeIndexes(), [3, 1])


if __name__ == "__main__":
    unittest.main()
