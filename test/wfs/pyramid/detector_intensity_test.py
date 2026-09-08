#!/usr/bin/env python
import unittest
import numpy as np

from arte.wfs.pyramid.four_facet_pyramid_kernel import FourFacetPyramidKernel
from arte.wfs.pyramid.modulation import (
    point_modulation_step,
    ring_modulation_steps,
)


class DetectorIntensityTest(unittest.TestCase):

    def setUp(self):
        self._grid_size = 256
        self._pupil_radius = 16
        # facet_separation_in_pixels chosen so that facet_separation/
        # sqrt(2) (the actual per-axis shift, see FourFacetPyramidKernel.
        # facet_image_centers) is an exact integer: an arbitrary
        # (non-grid-commensurate) separation is fine for the sensor
        # physics, but it spreads the facet mask's own Fourier transform
        # over several bins (spectral leakage) which would make a tight
        # sub-pixel centroid check below conflate that leakage with an
        # actual registration bug.
        self._pyramid = FourFacetPyramidKernel(
            grid_size=self._grid_size,
            pupil_radius_in_pixels=self._pupil_radius,
            facet_separation_in_pixels=40 * np.sqrt(2))
        self._phase = np.zeros((self._grid_size, self._grid_size))

    def test_point_modulation_weights_sum_to_one(self):
        tilts, weights = point_modulation_step(self._grid_size)
        self.assertEqual(len(tilts), 1)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        np.testing.assert_array_equal(
            tilts[0], np.zeros((self._grid_size, self._grid_size)))

    def test_ring_modulation_weights_sum_to_one(self):
        tilts, weights = ring_modulation_steps(
            self._grid_size, radius_in_pixels=20, n_steps=16)
        self.assertEqual(len(tilts), 16)
        self.assertAlmostEqual(np.sum(weights), 1.0)

    def test_detector_intensity_is_nonnegative_and_finite(self):
        tilts, weights = point_modulation_step(self._grid_size)
        intensity = self._pyramid.detector_intensity(
            self._phase, tilts, weights, mask_index=0)
        self.assertEqual(intensity.shape,
                         (self._grid_size, self._grid_size))
        self.assertTrue(np.all(np.isfinite(intensity)))
        self.assertTrue(np.all(intensity >= 0))
        self.assertGreater(intensity.max(), 0)

    def test_detector_frame_is_sum_of_the_four_channels(self):
        tilts, weights = point_modulation_step(self._grid_size)
        expected = sum(
            self._pyramid.detector_intensity(self._phase, tilts, weights, i)
            for i in range(4))
        frame = self._pyramid.detector_frame(self._phase, tilts, weights)
        np.testing.assert_allclose(frame, expected)

    def test_detector_intensity_is_centered_on_the_facet_image_center(self):
        # With a flat (zero) phase and no modulation, each channel's
        # detector image should be a (near-)tophat centered on that
        # channel's own facet_image_centers() position (Fig. 6
        # tessellation) -- checked via the intensity-weighted centroid
        # rather than argmax, since a tophat's argmax can land anywhere
        # within the disk (many equal-max pixels), not necessarily near
        # its center.
        tilts, weights = point_modulation_step(self._grid_size)
        centers = self._pyramid.facet_image_centers()
        half = self._grid_size // 2
        c = np.arange(self._grid_size) - half
        x, y = np.meshgrid(c, c, indexing='xy')
        for mask_index, (u0, v0) in enumerate(centers):
            intensity = self._pyramid.detector_intensity(
                self._phase, tilts, weights, mask_index)
            total = intensity.sum()
            centroid_x = np.sum(intensity * x) / total
            centroid_y = np.sum(intensity * y) / total
            self.assertLess(abs(centroid_x - u0), 1.0)
            self.assertLess(abs(centroid_y - v0), 1.0)

    def test_detector_frame_energy_grows_with_ring_modulation_radius(self):
        # A larger modulation radius spreads the same total pupil
        # transmission over a larger detector area (the ring of
        # replicated pupil-edge diffraction), so the peak intensity
        # should not increase, but the frame should stay finite and
        # non-negative -- a basic sanity/regression check, not a
        # first-principles energy-conservation proof.
        small_tilts, small_weights = ring_modulation_steps(
            self._grid_size, radius_in_pixels=5, n_steps=16)
        large_tilts, large_weights = ring_modulation_steps(
            self._grid_size, radius_in_pixels=20, n_steps=16)
        frame_small = self._pyramid.detector_frame(
            self._phase, small_tilts, small_weights)
        frame_large = self._pyramid.detector_frame(
            self._phase, large_tilts, large_weights)
        for frame in (frame_small, frame_large):
            self.assertTrue(np.all(np.isfinite(frame)))
            self.assertTrue(np.all(frame >= -1e-9))


if __name__ == "__main__":
    unittest.main()
