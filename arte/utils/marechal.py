import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity


def wavefront_rms_2_strehl_ratio(wavefrontRms, wavelength):
    """
    Compute Strehl ratio from wavefront aberration according to
    Marechal approximation

    Parameters
    ----------
    wavefrontRms : `astropy.units.quantity.Quantity` or float
        Wavefront aberration rms; if float units must be in meter

    wavelength : `astropy.units.quantity.Quantity` or float
        Wavelength to compute Strehl Ratio; if float units must be in meter


    Returns
    -------
    results : float
        Strehl ratio
    """
    if not isinstance(wavefrontRms, Quantity):
        wavefrontRms = wavefrontRms * u.m
    if not isinstance(wavelength, Quantity):
        wavelength = wavelength * u.m

    return np.exp(-(wavefrontRms / wavelength * 2 * np.pi) ** 2).value


def strehl_ratio_2_wavefront_rms(strehlRatio, wavelength):
    """
    Compute wavefront aberration from Strehl ratio according to
    Marechal approximation

    Parameters
    ----------
    strehlRatio : float
        Strehl ratio (0.0-1.0)

    wavelength : `astropy.units.quantity.Quantity` or float
        Wavelength to compute Strehl Ratio; if float units must be in meter


    Returns
    -------
    wavefrontRms : `astropy.units.quantity.Quantity`
        Wavefront aberration rms

    """
    if not isinstance(wavelength, Quantity):
        wavelength = wavelength * u.m

    return wavelength / (2 * np.pi) * np.sqrt(-1 * np.log(strehlRatio))


def scale_strehl_ratio(strehlRatio, fromWavelength, toWavelength):
    """
    Scale Strehl ratio from one wavelength to another one according to
    Marechal approximation

    Parameters
    ----------
    strehlRatio : float
        Strehl ratio (0.0-1.0)

    fromWavelength : `astropy.units.quantity.Quantity` or float
        Wavelength Strehl Ratio is given at; if float units must be in meter

    toWavelength : `astropy.units.quantity.Quantity` or float
        Wavelength to compute Strehl Ratio at; if float units must be in meter


    Returns
    -------
    strehlRatio : float
        Strehl Ratio computed at toWavelength
    """

    if isinstance(fromWavelength, Quantity):
        fromWavelength = fromWavelength.to(u.m).value
    if isinstance(toWavelength, Quantity):
        toWavelength = toWavelength.to(u.m).value

    return strehlRatio ** (fromWavelength / toWavelength) ** 2
