#!/usr/bin/env python
import unittest
import numpy as np

from arte.wfs.pyramid.quadrant_pyramid_kernel import QuadrantPyramidKernel
from arte.wfs.pyramid.four_facet_pyramid_kernel import FourFacetPyramidKernel
from arte.wfs.pyramid.modulation import point_modulation_step


class QuadrantPyramidKernelTest(unittest.TestCase):

    def setUp(self):
        self._grid_size = 256
        self._pupil_radius = 16
        self._facet_separation = 40 * np.sqrt(2)  # integer per-axis shift
        self._pyramid = QuadrantPyramidKernel(
            grid_size=self._grid_size,
            pupil_radius_in_pixels=self._pupil_radius,
            facet_separation_in_pixels=self._facet_separation)
        self._tilts, self._weights = point_modulation_step(self._grid_size)
        self._zero_phase = np.zeros((self._grid_size, self._grid_size))

    def test_focal_plane_mask_is_a_single_finite_pure_phase_mask(self):
        masks = self._pyramid._focal_plane_masks()
        self.assertEqual(len(masks), 1)
        np.testing.assert_allclose(np.abs(masks[0]), 1.0)

    def test_detector_frame_is_nonnegative_and_finite(self):
        frame = self._pyramid.detector_frame(
            self._zero_phase, self._tilts, self._weights)
        self.assertTrue(np.all(np.isfinite(frame)))
        self.assertTrue(np.all(frame >= 0))
        self.assertGreater(frame.max(), 0)

    def test_detector_frame_is_centered_on_facet_image_centers(self):
        # Same intensity-weighted-centroid check as
        # FourFacetPyramidKernelTest: even with a single finite mask,
        # the 4 lobes should still be centered close to the same
        # facet_image_centers() (the mask's *slope* per quadrant, hence
        # where each lobe peaks, is unchanged -- only its finite extent
        # is new).
        half = self._grid_size // 2
        c = np.arange(self._grid_size) - half
        x, y = np.meshgrid(c, c, indexing='xy')
        frame = self._pyramid.detector_frame(
            self._zero_phase, self._tilts, self._weights)
        # Split the grid into 4 quadrants around the center to isolate
        # each lobe's own centroid (they do not overlap for a
        # well-separated geometry).
        for u0, v0 in self._pyramid.facet_image_centers():
            quadrant = ((np.sign(x) == np.sign(u0)) &
                       (np.sign(y) == np.sign(v0)))
            sub = frame * quadrant
            total = sub.sum()
            cx = np.sum(sub * x) / total
            cy = np.sum(sub * y) / total
            self.assertLess(abs(cx - u0), 2.0)
            self.assertLess(abs(cy - v0), 2.0)

    def test_finite_mask_diffracts_into_the_gap_unlike_unbounded_masks(
           self):
        # The whole point of this class: unlike FourFacetPyramidKernel
        # (4 independent unbounded masks -> exact Dirac-delta transfer
        # function -> zero energy outside the 4 shifted pupil copies,
        # see run_analytic_flux_ratio_mod0.py's finding), a real finite
        # mask has genuine Fourier structure and must diffract some
        # light into the gap between the lobes.
        half = self._grid_size // 2
        c = np.arange(self._grid_size) - half
        x, y = np.meshgrid(c, c, indexing='xy')
        r = self._pupil_radius
        disk = np.zeros((self._grid_size, self._grid_size), dtype=bool)
        for u0, v0 in self._pyramid.facet_image_centers():
            disk |= (x - u0) ** 2 + (y - v0) ** 2 <= r ** 2
        # A generous "gap" region: inside the square spanned by the 4
        # centers, outside every disk.
        shift = self._facet_separation / np.sqrt(2)
        gap = (np.abs(x) <= shift) & (np.abs(y) <= shift) & ~disk

        finite_frame = self._pyramid.detector_frame(
            self._zero_phase, self._tilts, self._weights)
        finite_gap_energy = finite_frame[gap].sum()
        finite_disk_energy = finite_frame[disk].sum()
        self.assertGreater(finite_gap_energy / finite_disk_energy, 1e-6)

        unbounded_pyramid = FourFacetPyramidKernel(
            grid_size=self._grid_size,
            pupil_radius_in_pixels=self._pupil_radius,
            facet_separation_in_pixels=self._facet_separation)
        unbounded_frame = unbounded_pyramid.detector_frame(
            self._zero_phase, self._tilts, self._weights)
        unbounded_gap_energy = unbounded_frame[gap].sum()
        unbounded_disk_energy = unbounded_frame[disk].sum()

        self.assertGreater(
            finite_gap_energy / finite_disk_energy,
            10 * unbounded_gap_energy / unbounded_disk_energy)


if __name__ == "__main__":
    unittest.main()
