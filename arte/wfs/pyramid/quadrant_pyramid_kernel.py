import numpy as np

from arte.types.mask import CircularMask
from arte.utils.decorator import override
from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel


class QuadrantPyramidKernel(PyramidKernel):
    """ Model of the classical 4-faces pyramid WFS using the REAL,
    single, finite-extent combined mask -- the physical pyramid glass
    shape -- instead of the 4 independent, unbounded phase-ramp masks of
    `arte.wfs.pyramid.four_facet_pyramid_kernel.FourFacetPyramidKernel`
    (Appendix B of Fauvarque et al. 2019).

    The mask is the classic pyramid phase function, proportional to
    `|x| + |y|` (a real 4-sided pyramid glass has, by construction,
    optical thickness growing linearly with distance from the apex,
    with a slope set independently in each of the 4 quadrants by the
    sign of x/y) -- the same construction used by SPECULA's own
    `specula.processing_objects.modulated_pyramid.ModulatedPyramid.
    get_pyr_tlt` (there built quadrant-by-quadrant; here written directly
    as the equivalent, simpler `|x|+|y|` closed form, verified to
    reproduce the same 4 facet deflection directions as
    `FourFacetPyramidKernel._FACET_DIRECTIONS`).

    Why this class exists (see the project's own analysis first): with
    `FourFacetPyramidKernel`'s unbounded masks, `PyramidKernel.
    detector_intensity`/`detector_frame` cannot produce ANY light in the
    diffraction gaps between the 4 reimaged pupil images -- an unbounded
    pure phase ramp has a Fourier transform that is an exact Dirac delta,
    and convolving a unit-modulus field with an exact delta is a pure
    shift, which cannot alter its intensity distribution. A real pyramid
    apex/edge is finite (even a geometrically perfect one, with none of
    SPECULA's own `pyr_edge_def_ld`/`pyr_tip_def_ld`-type manufacturing
    defects, which are 0 throughout this project's configs), and a sharp
    finite edge always diffracts (basic Fraunhofer optics, the same
    reason a perfect Foucault knife-edge still produces diffraction
    fringes) -- this class is what is needed to see that.

    This is a pure implementation choice, not a new piece of theory:
    Fauvarque et al. 2019's general Kernel formalism (eq. 7-8, used
    unmodified by `PyramidKernel.detector_intensity`) makes no assumption
    on the mask's support. Only the *convolutional*, closed-form
    slopes-map machinery specific to Appendix B (`impulse_response`,
    `optical_gain`, and everything built on them) genuinely requires the
    unbounded/well-separated-images idealization -- it is NOT
    implemented on top of this class (there is only one, single, mask
    channel here, not 4 independent ones to combine into `IR_x`/`IR_y`).
    Only `detector_intensity`/`detector_frame` (the exact, eq. (8),
    non-perturbative machinery already generic in `PyramidKernel`) apply
    here unchanged.

    Parameters
    ----------
    grid_size: int
        See `FourFacetPyramidKernel`.

    pupil_radius_in_pixels: float
        See `FourFacetPyramidKernel`.

    facet_separation_in_pixels: float, default=None
        Same meaning/units as `FourFacetPyramidKernel`'s parameter of the
        same name (the grid-center-to-facet-image-center distance): this
        class reproduces the exact same 4 facet_image_centers, only the
        mask connecting them is now finite instead of 4 independent
        unbounded ones.
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

    # Same numbering/directions as FourFacetPyramidKernel, so that the
    # two classes' facet_image_centers()/detector_frame() are directly
    # comparable.
    _FACET_DIRECTIONS = ((-1, 1), (1, 1), (-1, -1), (1, -1))

    def facet_image_centers(self):
        """ Same as `FourFacetPyramidKernel.facet_image_centers`: (x, y)
        pixel offset from the grid center of each of the 4 reimaged
        pupil images -- still meaningful here (it is where the 4 lobes
        of the single finite mask's diffraction pattern peak), even
        though there is only one mask/channel (`mask_index` is always 0
        for this class). """
        shift = self._facet_separation_in_pixels / np.sqrt(2)
        return [(dx * shift, dy * shift) for dx, dy in self._FACET_DIRECTIONS]

    @override
    def _focal_plane_masks(self):
        c = np.arange(self._grid_size) - self._grid_size // 2
        x, y = np.meshgrid(c, c, indexing='xy')
        shift = self._facet_separation_in_pixels / np.sqrt(2)
        ramp = 2 * np.pi * shift * (np.abs(x) + np.abs(y)) / self._grid_size
        return [np.exp(1j * ramp)]

    @override
    def _pupil_mask(self):
        mask = CircularMask((self._grid_size, self._grid_size),
                            maskRadius=self._pupil_radius_in_pixels)
        return mask.asTransmissionValue()

    def detector_frame(self, phase, modulation_tilts, modulation_weights):
        """ Combined detector frame. Unlike `FourFacetPyramidKernel.
        detector_frame` (which sums 4 separate channels), this is just
        `PyramidKernel.detector_intensity` on the single combined mask
        (`mask_index=0`): the 4 reimaged pupil images and the light
        diffracted into the gaps between them are already all part of
        the one propagation, not summed from independent channels.
        """
        return self.detector_intensity(phase, modulation_tilts,
                                       modulation_weights, mask_index=0)
