import numpy as np

from arte.types.mask import CircularMask
from arte.utils.decorator import override
from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel


class FourFacetPyramidKernel(PyramidKernel):
    """ Convolutional-kernel model of the classical 4-faces pyramid WFS,
    implementing Appendix B of Fauvarque et al. 2019 (see
    `arte.wfs.pyramid.abstract_pyramid_kernel.PyramidKernel` for the
    references and the shared grid convention).

    Each of the 4 pyramid facets is modelled, as in the paper, as a
    separate Fourier WFS whose mask is a pure phase ramp (a tilted flat
    window / prism) covering the whole focal plane -- not as a single
    mask split in 4 quadrants. The 4 facets are numbered as in Fig. 6 of
    the paper::

        1 | 2
        --+--
        3 | 4

    with facet 1 = top-left, 2 = top-right, 3 = bottom-left,
    4 = bottom-right, matching `mask_index` 0, 1, 2, 3 respectively.

    Parameters
    ----------
    grid_size: int
        Size (in pixels) of the square grid used for every array
        (pupil, masks, impulse responses, ...). Must leave enough room
        around the pupil for the 4 replicated pupil images to not
        overlap, i.e. it must be significantly larger than
        `2 * pupil_radius_in_pixels`.

    pupil_radius_in_pixels: float
        Radius of the (circular, unobstructed) entrance pupil, in
        pixels of the grid.

    facet_separation_in_pixels: float, default=None
        Distance, in pixels of the grid, between the centers of two
        horizontally or vertically adjacent replicated pupil images
        produced by the pyramid (i.e. the apex separation power of the
        pyramid). By default `3 * pupil_radius_in_pixels`, i.e. the 4
        pupil images just fit on the grid without overlapping.

    Example
    -------
    >>> pyr = FourFacetPyramidKernel(grid_size=128,
    ...                              pupil_radius_in_pixels=16)
    >>> ir1 = pyr.impulse_response(effective_modulation_hat, mask_index=0)
    """

    def __init__(self,
                grid_size,
                pupil_radius_in_pixels,
                facet_separation_in_pixels=None):
        self._grid_size = grid_size
        self._pupil_radius_in_pixels = pupil_radius_in_pixels
        if facet_separation_in_pixels is None:
            facet_separation_in_pixels = 3 * pupil_radius_in_pixels
        self._facet_separation_in_pixels = facet_separation_in_pixels

    # Facet directions in (x, y) grid units, matching Fig. 6 of
    # Fauvarque et al. 2019: 1=top-left, 2=top-right, 3=bottom-left,
    # 4=bottom-right.
    _FACET_DIRECTIONS = ((-1, 1), (1, 1), (-1, -1), (1, -1))

    def facet_image_centers(self):
        """ (x, y) pixel offset from the grid center of each of the 4
        reimaged pupil images on the detector, in the same `mask_index`
        order as `_focal_plane_masks`/`impulse_response` (0=top-left,
        1=top-right, 2=bottom-left, 3=bottom-right, Fig. 6 numbering).
        Direct consequence of each facet mask being a pure phase ramp
        (Fourier shift theorem): useful to build masks of the detector
        plane (e.g. which pixels belong to which reimaged pupil, or to
        the diffraction gaps between them) matching this class's own
        `detector_frame`.

        Returns
        -------
        centers: list of tuple(float, float), length 4
        """
        shift = self._facet_separation_in_pixels / np.sqrt(2)
        return [(dx * shift, dy * shift) for dx, dy in self._FACET_DIRECTIONS]

    @override
    def _focal_plane_masks(self):
        return [
            np.exp(1j * self._phase_ramp(self._grid_size, u0, v0))
            for u0, v0 in self.facet_image_centers()
        ]

    @override
    def _pupil_mask(self):
        mask = CircularMask((self._grid_size, self._grid_size),
                            maskRadius=self._pupil_radius_in_pixels)
        return mask.asTransmissionValue()

    def slope_x_impulse_response(self, effective_modulation_hat):
        """ Impulse response of the "Sx" slopes map, eq. (68) of
        Fauvarque et al. 2019::

            IR_x = (IR_2 + IR_4) - (IR_1 + IR_3)

        Parameters
        ----------
        effective_modulation_hat: numpy.ndarray(complex) [grid_size,
            grid_size]

        Returns
        -------
        ir_x: numpy.ndarray(float) [grid_size, grid_size]
        """
        ir1, ir2, ir3, ir4 = (
            self.impulse_response(effective_modulation_hat, i)
            for i in range(4))
        return (ir2 + ir4) - (ir1 + ir3)

    def slope_y_impulse_response(self, effective_modulation_hat):
        """ Impulse response of the "Sy" slopes map, eq. (69) of
        Fauvarque et al. 2019::

            IR_y = (IR_1 + IR_2) - (IR_4 + IR_3)

        Parameters
        ----------
        effective_modulation_hat: numpy.ndarray(complex) [grid_size,
            grid_size]

        Returns
        -------
        ir_y: numpy.ndarray(float) [grid_size, grid_size]
        """
        ir1, ir2, ir3, ir4 = (
            self.impulse_response(effective_modulation_hat, i)
            for i in range(4))
        return (ir1 + ir2) - (ir4 + ir3)

    def detector_frame(self, phase, modulation_tilts, modulation_weights):
        """ Combined detector frame: the sum of the 4 facets'
        `PyramidKernel.detector_intensity`, i.e. what a real pyramid WFS
        camera actually records in one image -- the 4 reimaged pupil
        images (at `facet_image_centers`) plus whatever light the exact,
        non-perturbative propagation (eq. 8) sends into the gaps between
        them. Exact/non-perturbative like `detector_intensity`: no
        small-phase assumption on `phase`.

        Parameters
        ----------
        phase: numpy.ndarray(float) [grid_size, grid_size]

        modulation_tilts: sequence of numpy.ndarray(float) [grid_size,
            grid_size]

        modulation_weights: sequence of float

        Returns
        -------
        frame: numpy.ndarray(float) [grid_size, grid_size]
        """
        return sum(
            self.detector_intensity(phase, modulation_tilts,
                                    modulation_weights, i)
            for i in range(4))
