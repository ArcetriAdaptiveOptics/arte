#!/usr/bin/env python
import unittest
import numpy as np

from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel
from arte.wfs.pyramid.effective_modulation import (
    no_modulation_weighting_function_hat,
    ring_modulation_weighting_function_hat,
    effective_modulation_from_structure_function,
    effective_modulation_from_frame,
)


class EffectiveModulationTest(unittest.TestCase):

    def setUp(self):
        self._grid_size = 64
        c = np.arange(self._grid_size) - self._grid_size // 2
        x, y = np.meshgrid(c, c)
        self._pupil_mask = (x ** 2 + y ** 2 <= 12 ** 2).astype(float)

    def test_no_modulation_weighting_function_is_all_ones(self):
        w_hat = no_modulation_weighting_function_hat(self._grid_size)
        np.testing.assert_array_equal(
            w_hat, np.ones((self._grid_size, self._grid_size)))

    def test_ring_modulation_weighting_function_peaks_at_one(self):
        w_hat = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=8)
        center = self._grid_size // 2
        self.assertAlmostEqual(w_hat[center, center], 1.0)

    def test_zero_radius_ring_is_equivalent_to_no_modulation(self):
        w_hat_ring = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=0)
        w_hat_none = no_modulation_weighting_function_hat(self._grid_size)
        np.testing.assert_allclose(w_hat_ring, w_hat_none)

    def test_effective_modulation_default_is_finite_pupil_case(self):
        w_hat = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=8)
        omega_hat = effective_modulation_from_structure_function(
            w_hat, self._pupil_mask)
        np.testing.assert_array_equal(omega_hat, w_hat * self._pupil_mask)

    def test_dynamic_residual_reduces_the_effective_modulation_amplitude(
           self):
        w_hat = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=8)
        structure_function = np.full((self._grid_size, self._grid_size),
                                     2.0)
        omega_hat_no_residual = effective_modulation_from_structure_function(
            w_hat, self._pupil_mask)
        omega_hat_with_residual = (
            effective_modulation_from_structure_function(
                w_hat, self._pupil_mask,
                structure_function=structure_function))
        self.assertTrue(np.all(
            np.abs(omega_hat_with_residual) <=
            np.abs(omega_hat_no_residual) + 1e-12))
        # exp(-1) attenuation for structure_function=2 everywhere
        # the pupil is nonzero.
        ratio = np.divide(
            np.abs(omega_hat_with_residual), np.abs(omega_hat_no_residual),
            out=np.zeros_like(omega_hat_no_residual, dtype=float),
            where=self._pupil_mask != 0)
        np.testing.assert_allclose(
            ratio[self._pupil_mask != 0], np.exp(-1.0))

    def test_static_residual_term_is_phase_only(self):
        w_hat = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=8)
        gradient_term = np.full((self._grid_size, self._grid_size), 0.7)
        omega_hat_no_static = effective_modulation_from_structure_function(
            w_hat, self._pupil_mask)
        omega_hat_with_static = (
            effective_modulation_from_structure_function(
                w_hat, self._pupil_mask,
                static_phase_gradient_term=gradient_term))
        np.testing.assert_allclose(
            np.abs(omega_hat_with_static), np.abs(omega_hat_no_static))

    def test_effective_modulation_from_frame_matches_its_definition(self):
        w_hat = ring_modulation_weighting_function_hat(
            self._grid_size, radius_in_pixels=8)
        rng = np.random.default_rng(0)
        frame = rng.random((self._grid_size, self._grid_size))
        omega_hat = effective_modulation_from_frame(frame, w_hat)
        expected = PyramidKernel._fft2c(frame) * w_hat
        np.testing.assert_allclose(omega_hat, expected)


if __name__ == "__main__":
    unittest.main()
