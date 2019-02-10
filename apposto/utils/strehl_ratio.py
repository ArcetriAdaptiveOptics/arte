import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity


def wavefrontRms2StrehlRatio(wavefrontRms, wavelength):
    """
    Compute Strehl ratio from wavefront aberration according to
    Marechal approximation

    Parameters
    ----------
    wavefrontRms : `~astropy.units.Quantity` or float
        Wavefront aberration rms; if float units must be in meter

    wavelength : `~astropy.units.Quantity` or float
        Wavelength to compute Strehl Ratio; if float units must be in meter


    Returns
    -------
    results : float
        Strehl ratio
    """
    if not isinstance(wavefrontRms, Quantity):
        wavefrontRms= wavefrontRms * u.m
    if not isinstance(wavelength, Quantity):
        wavelength= wavelength * u.m

    return np.exp(-(wavefrontRms/wavelength*2*np.pi)**2)


def strehlRatio2WavefrontRms(strehlRatio, wavelength):
    """
    Compute wavefront aberration from Strehl ratio according to
    Marechal approximation

    Parameters
    ----------
    strehlRatio : float
        Strehl ratio (0.0-1.0)

    wavelength : `~astropy.units.Quantity` or float
        Wavelength to compute Strehl Ratio; if float units must be in meter


    Returns
    -------
    wavefrontRms : `~astropy.units.Quantity`
        Wavefront aberration rms

    """
    if not isinstance(wavelength, Quantity):
        wavelength= wavelength * u.m

    return np.sqrt(- wavelength**2 / (4*np.pi**2) * np.log(strehlRatio))


def scaleStrehlRatio(strehlRatio, fromWavelength, toWavelength):
    """
    Scale Strehl ratio from one wavelength to another one according to
    Marechal approximation

    Parameters
    ----------
    strehlRatio : float
        Strehl ratio (0.0-1.0)

    fromWavelength : `~astropy.units.Quantity` or float
        Wavelength Strehl Ratio is given at; if float units must be in meter

    toWavelength : `~astropy.units.Quantity` or float
        Wavelength to compute Strehl Ratio at; if float units must be in meter


    Returns
    -------
    strehlRatio : float
        Strehl Ratio computed at toWavelength
    """

    wfRms= strehlRatio2WavefrontRms(strehlRatio, fromWavelength)
    return wavefrontRms2StrehlRatio(wfRms, toWavelength)
