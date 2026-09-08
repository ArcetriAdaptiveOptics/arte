#!/usr/bin/env python
import unittest
import numpy as np

from arte.utils.decorator import override
from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel
from arte.wfs.pyramid.four_facet_pyramid_kernel import FourFacetPyramidKernel
from arte.wfs.pyramid.effective_modulation import (
    no_modulation_weighting_function_hat,
)


class _SingleMaskPyramidKernelForTest(PyramidKernel):
    """ Minimal concrete PyramidKernel, only used to exercise the base
    class methods in isolation from FourFacetPyramidKernel. """

    def __init__(self, grid_size, pupil_radius_in_pixels):
        self._grid_size = grid_size
        self._pupil_radius_in_pixels = pupil_radius_in_pixels

    @override
    def _focal_plane_masks(self):
        c = np.arange(self._grid_size) - self._grid_size // 2
        x, y = np.meshgrid(c, c)
        return [np.exp(2j * np.pi * 5 * x / self._grid_size)]

    @override
    def _pupil_mask(self):
        c = np.arange(self._grid_size) - self._grid_size // 2
        x, y = np.meshgrid(c, c)
        return (x ** 2 + y ** 2 <=
               self._pupil_radius_in_pixels ** 2).astype(float)


class PyramidKernelTest(unittest.TestCase):

    def setUp(self):
        self._grid_size = 64
        self._pupil_radius = 12
        self._kernel = _SingleMaskPyramidKernelForTest(
            self._grid_size, self._pupil_radius)
        self._omega_hat = no_modulation_weighting_function_hat(
            self._grid_size)

    def test_impulse_response_is_real_and_finite(self):
        ir = self._kernel.impulse_response(self._omega_hat)
        self.assertEqual(ir.shape, (self._grid_size, self._grid_size))
        self.assertTrue(np.all(np.isfinite(ir)))
        self.assertTrue(np.iscomplexobj(ir) is False)
        self.assertGreater(np.max(np.abs(ir)), 0)

    def test_transfer_function_is_fft_of_impulse_response(self):
        ir = self._kernel.impulse_response(self._omega_hat)
        tf_direct = self._kernel.transfer_function(self._omega_hat)
        tf_from_ir = self._kernel.transfer_function_from_impulse_response(ir)
        np.testing.assert_allclose(tf_direct, tf_from_ir)

    def test_sensitivity_map_is_nonnegative(self):
        s = self._kernel.sensitivity_map(self._omega_hat)
        self.assertTrue(np.all(s >= -1e-10))

    def test_optical_gain_of_identical_impulse_responses_is_one(self):
        ir = self._kernel.impulse_response(self._omega_hat)
        rng = np.random.default_rng(0)
        mode = rng.standard_normal((self._grid_size, self._grid_size))
        gain = self._kernel.optical_gain(mode, ir, ir)
        self.assertAlmostEqual(gain, 1.0, places=8)

    def test_optical_gain_scales_linearly_with_a_scaled_response(self):
        ir_reference = self._kernel.impulse_response(self._omega_hat)
        scale = 0.37
        ir_current = scale * ir_reference
        rng = np.random.default_rng(1)
        mode = rng.standard_normal((self._grid_size, self._grid_size))
        gain = self._kernel.optical_gain(mode, ir_reference, ir_current)
        self.assertAlmostEqual(gain, scale, places=8)


class FourFacetPyramidKernelTest(unittest.TestCase):

    def setUp(self):
        self._grid_size = 128
        self._pupil_radius = 16
        self._facet_separation = 40  # integer, grid-commensurate
        self._pyramid = FourFacetPyramidKernel(
            grid_size=self._grid_size,
            pupil_radius_in_pixels=self._pupil_radius,
            facet_separation_in_pixels=self._facet_separation)
        self._omega_hat = no_modulation_weighting_function_hat(
            self._grid_size)

    def test_pupil_mask_area_matches_disk_area(self):
        pupil = self._pyramid._pupil_mask()
        expected_area = np.pi * self._pupil_radius ** 2
        self.assertAlmostEqual(np.sum(pupil), expected_area, delta=0.05 *
                               expected_area)

    def test_focal_plane_masks_are_four_pure_phase_ramps(self):
        masks = self._pyramid._focal_plane_masks()
        self.assertEqual(len(masks), 4)
        for m in masks:
            self.assertEqual(m.shape, (self._grid_size, self._grid_size))
            np.testing.assert_allclose(np.abs(m), 1.0)

    def test_facet_masks_are_shifted_to_the_expected_quadrant(self):
        # With an integer, grid-commensurate facet_separation_in_pixels,
        # the Fourier transform of each phase-ramp mask is (up to
        # spectral leakage) a peak exactly at the expected offset from
        # the grid center: this checks the sign/order convention of
        # _FACET_DIRECTIONS against Fig. 6 of Fauvarque et al. 2019.
        masks = self._pyramid._focal_plane_masks()
        expected_offsets = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        half = self._facet_separation / np.sqrt(2)
        center = self._grid_size // 2
        for mask, (dx, dy) in zip(masks, expected_offsets):
            spectrum = np.abs(PyramidKernel._fft2c(mask))
            peak_row, peak_col = np.unravel_index(np.argmax(spectrum),
                                                   spectrum.shape)
            self.assertAlmostEqual(peak_col - center, dx * half, delta=1.0)
            self.assertAlmostEqual(peak_row - center, dy * half, delta=1.0)

    def test_slope_maps_match_manual_combination_of_the_four_channels(self):
        ir = [self._pyramid.impulse_response(self._omega_hat, i)
             for i in range(4)]
        expected_x = (ir[1] + ir[3]) - (ir[0] + ir[2])
        expected_y = (ir[0] + ir[1]) - (ir[3] + ir[2])
        np.testing.assert_allclose(
            self._pyramid.slope_x_impulse_response(self._omega_hat),
            expected_x)
        np.testing.assert_allclose(
            self._pyramid.slope_y_impulse_response(self._omega_hat),
            expected_y)


if __name__ == "__main__":
    unittest.main()
