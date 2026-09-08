""" Builders for the "effective modulation weighting function" `omega_hat`
consumed by `arte.wfs.pyramid.abstract_pyramid_kernel.PyramidKernel`.

`omega_hat` is injected into `PyramidKernel` methods as a plain array
instead of being part of the sensor class hierarchy, because it varies
along an axis that is independent of the sensor geometry (see the
`PyramidKernel` docstring). This module collects the two ways to build it
discussed in the source papers:

- from a nominal modulation plus an assumed statistical model of the
  residual phase (structure function, and/or a low-order static
  aberration term) -- eqs. (41)-(46) of Fauvarque et al. 2019. This is
  the time-averaged, telemetry-only route (no additional hardware),
  giving an `omega_hat` representative of many frames at a given
  residual-phase level.

- from an actual focal-plane frame (measured by a dedicated "Gain
  Scheduling Camera", or -- in a simulation, where the true residual
  phase is known -- computed directly from it) -- eqs. (10)-(13) of
  Chambouleyron et al. 2021. This is the single-frame, ground-truth
  route, useful to validate the statistical route above against exact
  per-frame values.

All functions here share the grid convention documented in
`PyramidKernel`: `w_hat` (the Fourier transform of the nominal
modulation weighting function) and the returned `omega_hat` are arrays
in the pupil/detector domain.
"""
import numpy as np
from scipy.special import j0

from arte.wfs.pyramid.abstract_pyramid_kernel import PyramidKernel


def _centered_radial_frequency(grid_size):
    """ Radial spatial frequency coordinate, in cycles per pixel,
    of the pupil/detector grid -- the variable dual to a distance
    expressed in pixels of the focal-plane grid. """
    c = np.arange(grid_size) - grid_size // 2
    x, y = np.meshgrid(c, c, indexing='xy')
    return np.sqrt(x ** 2 + y ** 2) / grid_size


def no_modulation_weighting_function_hat(grid_size):
    """ `w_hat` for a non-modulated pyramid WFS: the modulation
    weighting function is a centered Dirac delta, whose Fourier
    transform is the identity function -- eq. (20) of Fauvarque et al.
    2019, "Without modulation" case.

    Parameters
    ----------
    grid_size: int

    Returns
    -------
    w_hat: numpy.ndarray(float) [grid_size, grid_size]
        An array of ones.
    """
    return np.ones((grid_size, grid_size))


def ring_modulation_weighting_function_hat(grid_size, radius_in_pixels):
    """ `w_hat` for a circular ("ring") tip/tilt modulation of the given
    radius: `w_hat(rho) = J0(2 pi * radius_in_pixels * rho)`, the first
    Bessel function of the first kind, as discussed around eq. (27) of
    Fauvarque et al. 2019.

    Parameters
    ----------
    grid_size: int

    radius_in_pixels: float
        Modulation radius, in pixels of the focal-plane grid.

    Returns
    -------
    w_hat: numpy.ndarray(float) [grid_size, grid_size]
    """
    rho = _centered_radial_frequency(grid_size)
    return j0(2 * np.pi * radius_in_pixels * rho)


def effective_modulation_from_structure_function(w_hat,
                                                  pupil_mask,
                                                  structure_function=None,
                                                  static_phase_gradient_term=None):
    """ Time-averaged effective modulation weighting function built from
    a nominal modulation, the entrance pupil, and (optionally) residual
    phase statistics -- eqs. (41)-(46) of Fauvarque et al. 2019::

        omega_hat = w_hat * I_P * exp(-D/2) * exp(i * D_phi_s)

    Passing `structure_function=None` and `static_phase_gradient_term=
    None` (the default) gives the "finite pupil, no residual" case,
    eq. (43): `omega_hat = w_hat * I_P`.

    Parameters
    ----------
    w_hat: numpy.ndarray [grid_size, grid_size]
        Fourier transform of the nominal modulation weighting function,
        e.g. from `no_modulation_weighting_function_hat` or
        `ring_modulation_weighting_function_hat`.

    pupil_mask: numpy.ndarray [grid_size, grid_size]
        Entrance pupil indicator function `I_P`, same grid as `w_hat`
        (e.g. the array returned by a `PyramidKernel._pupil_mask`
        subclass implementation).

    structure_function: numpy.ndarray(float) [grid_size, grid_size], \
        optional
        Structure function `D` of the dynamic residual phase (assumed
        Gaussian, zero-mean and stationary), eq. (33)-(35). Typically
        estimated from real closed-loop telemetry rather than an
        idealized turbulence model, to stay consistent with the
        residual statistics actually seen by the sensor.

    static_phase_gradient_term: numpy.ndarray(float) [grid_size, \
        grid_size], optional
        The `D(phi_s)` differential term of a static residual phase
        (e.g. NCPA), eq. (36)-(39). Only accurate for low-order static
        aberrations (the "low order approximation" of sec. 5.2 of the
        paper) -- validate this assumption before trusting it for a
        given `phi_s`.

    Returns
    -------
    omega_hat: numpy.ndarray(complex) [grid_size, grid_size]
    """
    omega_hat = w_hat * pupil_mask
    if structure_function is not None:
        omega_hat = omega_hat * np.exp(-0.5 * structure_function)
    if static_phase_gradient_term is not None:
        omega_hat = omega_hat * np.exp(1j * static_phase_gradient_term)
    return omega_hat


def effective_modulation_from_frame(focal_plane_intensity, w_hat):
    """ Single-frame effective modulation weighting function built from
    an actual focal-plane image -- eqs. (10)-(13) of Chambouleyron et
    al. 2021::

        omega_hat = FT(focal_plane_intensity) * w_hat

    `focal_plane_intensity` (`PSF_phi` in the paper, `Omega_phi` once
    convolved with the modulation) is the image recorded by a Gain
    Scheduling Camera in Chambouleyron et al. 2021's practical setup; in
    a simulation where the true residual phase is known exactly, it can
    instead be computed directly from it (the corresponding
    modulated/aberrated PSF), giving a ground-truth `omega_hat` with no
    hardware involved, to validate
    `effective_modulation_from_structure_function` against.

    Parameters
    ----------
    focal_plane_intensity: numpy.ndarray(float) [grid_size, grid_size]
        Focal-plane intensity image, sampled on the focal-plane grid
        (same grid as the masks returned by
        `PyramidKernel._focal_plane_masks`).

    w_hat: numpy.ndarray [grid_size, grid_size]
        Fourier transform of the nominal modulation weighting function.

    Returns
    -------
    omega_hat: numpy.ndarray(complex) [grid_size, grid_size]
    """
    return PyramidKernel._fft2c(focal_plane_intensity) * w_hat
