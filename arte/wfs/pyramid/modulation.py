""" Discretized tip/tilt modulation tilt maps, for
`arte.wfs.pyramid.abstract_pyramid_kernel.PyramidKernel.detector_intensity`
(the exact, eq. (8)-of-Fauvarque-et-al.-2019 propagation) -- not to be
confused with `arte.wfs.pyramid.effective_modulation`, which builds the
*linear*-model's effective modulation weighting function `omega_hat` and
needs only the modulation's Fourier transform `w_hat`, never the actual
per-step tilt maps.

`detector_intensity`/`detector_frame` are non-linear in the phase (no
Taylor expansion), so the modulation integral of eq. (8) cannot be
summarized by `w_hat` alone -- it must be evaluated as an explicit
(weighted) sum over a finite set of tilts sampling one modulation cycle,
which is what this module builds.
"""
import numpy as np

from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel


def ring_modulation_steps(grid_size, radius_in_pixels, n_steps=32):
    """ Discretized circular ("ring") tip/tilt modulation: `n_steps`
    tilt maps evenly spaced in angle around a circle of the given
    radius, each with the uniform weight `1/n_steps` (a discrete version
    of the continuous ring weighting function whose Fourier transform is
    used in `arte.wfs.pyramid.effective_modulation.
    ring_modulation_weighting_function_hat` -- `radius_in_pixels` has the
    same meaning/units in both places, and `n_steps` should be large
    enough to resolve the ring smoothly, e.g. several times
    `2*pi*radius_in_pixels / grid_size`).

    Parameters
    ----------
    grid_size: int

    radius_in_pixels: float
        Modulation radius, in pixels of the pupil-conjugate/detector
        grid (the same convention as `PyramidKernel.detector_intensity`'s
        `phase` and `arte.wfs.pyramid.four_facet_pyramid_kernel.
        FourFacetPyramidKernel`'s `facet_separation_in_pixels`).

    n_steps: int, default=32

    Returns
    -------
    tilts: list of numpy.ndarray(float) [grid_size, grid_size], length \
        `n_steps`
    weights: numpy.ndarray(float) [n_steps]
        Uniform, summing to 1 (eq. 6).
    """
    angles = 2 * np.pi * np.arange(n_steps) / n_steps
    tilts = [
        PyramidKernel._phase_ramp(grid_size,
                                  radius_in_pixels * np.cos(a),
                                  radius_in_pixels * np.sin(a))
        for a in angles
    ]
    weights = np.full(n_steps, 1.0 / n_steps)
    return tilts, weights


def point_modulation_step(grid_size):
    """ Non-modulated case: a single, centered (zero) tilt with weight 1
    -- the discretized equivalent of the Dirac modulation weighting
    function of eq. (20) of Fauvarque et al. 2019 ("Without modulation").

    Parameters
    ----------
    grid_size: int

    Returns
    -------
    tilts: list of one numpy.ndarray(float) [grid_size, grid_size]
    weights: numpy.ndarray(float), shape (1,)
        `[1.0]`.
    """
    return [np.zeros((grid_size, grid_size))], np.array([1.0])
