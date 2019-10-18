'''
Created on Oct 13, 2019

@author: lbusoni
'''
import matplotlib.pyplot as plt
import synphot
from astropy import units as u
from synphot import SourceSpectrum
from synphot.models import Box1D
from synphot.spectrum import SpectralElement
from synphot.observation import Observation
from apposto.phot.spectral_types import PickelsLibrary
from apposto.phot import spectral_types


def getNormalizedSpectrum(spectralType, magnitude, filter_name):
    """
    spec_data = getNormalizedSpectrum(spectral_type, magnitude, filter_name)

    Returns a structure containing the synthetic spectrum of the star having the spectral type and magnitude 
    in the specified input filter. Magnitude is in VEGAMAG-F(lambda) system.
    Spectra are from PICKLES, PASP, 110, 863 (1998)
    Absolute flux spectra, no effect of athmospheric and instrument transmission

    INPUT:
    spectral_type:   scalar string. spectral type and luminosity class (e.g. G2V or M4III)
    magnitude:       scalar float. magnitude in the filter_name filter
    filter_name:     scalar string. Name of the filter (see get_filter(/GET) for a filter database)
    filter_family:   scalar string. Name of the filter family (see get_filter(/GET) for a filter database)

    RETURN:
    SourceSpectrum object defining the spectrum

    EXAMPLES:
    return the photon flux outside the earth atmosphere of a A0V, G2V and M0V star with magnitude 8.0 in R filter (Bessel90 standard)
    data_G2V = GET_SPECTRUM("G2V", 8.0, "R", "UBVRI-Bessell90", /PHOT)
    data_M0V = GET_SPECTRUM("M0V", 8.0, "R", "UBVRI-Bessell90", /PHOT)
    data_A0V = GET_SPECTRUM("A0V", 8.0, "R", "UBVRI-Bessell90", /PHOT)
    p0 = plot(data_G2V.wavelength, data_G2V.flux, XTITLE=data_G2V.lambda_unit, YTITLE=data_G2V.flux_unit,NAME="G2V")
    p1 = plot(data_M0V.wavelength, data_M0V.flux, COLOR="red", OVERPLOT=p0, NAME="M0V")
    p2 = plot(data_A0V.wavelength, data_A0V.flux, COLOR="blue", OVERPLOT=p0, NAME="A0V")
    p3 = legend(TARGET=[p0,p1,p2])
"""
    # read the sourcespectrum
    spectrum= SourceSpectrum.from_file(
        PickelsLibrary.filename(spectralType))

    bandpass= SpectralElement.from_filter(filter_name)

    spectrum_norm= spectrum.normalize(
        magnitude*synphot.units.VEGAMAG,
        bandpass,
        vegaspec=SourceSpectrum.from_vega())

    return spectrum_norm


# get_vega() downloads this one
synphot.specio.read_remote_spec(
    'http://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_008.fits')

# G5V of UVKLIB subset of Pickles library
# see http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas.html
synphot.specio.read_remote_spec(
    'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_27.fits')

# read the sourcespectrum
spG5V= SourceSpectrum.from_file(
    'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_27.fits')
spG2V= SourceSpectrum.from_file(
    'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_26.fits')

filtR= SpectralElement.from_filter('johnson_r')
spG2V_19r= spG2V.normalize(
    19*synphot.units.VEGAMAG, filtR,
    vegaspec=SourceSpectrum.from_vega())

bp= SpectralElement(Box1D, x_0=700*u.nm, width=600*u.nm)
obs= Observation(spG2V_19r, bp)
obs.countrate(area=50*u.m**2)


bp220= SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)
bpCRed= SpectralElement(Box1D, x_0=1650*u.nm, width=300*u.nm)*0.8

Observation(spG2V_19r, bp220).countrate(area=50*u.m**2)
Observation(spG2V_19r, bpCRed).countrate(area=50*u.m**2)


spM0V_8R= getNormalizedSpectrum('M0V', 8.0, 'johnson_r')


uPhotonSecM2Micron=u.photon/(u.s * u.m**2 * u.micron)

spG2V_8R= getNormalizedSpectrum(spectral_types.G2V, 8.0, 'johnson_r')
plt.plot(spG2V_8R.waveset, spG2V_8R(spG2V_8R.waveset).to(uPhotonSecM2Micron))

# compare with Armando's
spG2V_19R= getNormalizedSpectrum(spectral_types.G2V, 19, 'johnson_r')
bp= SpectralElement(Box1D, x_0=700*u.nm, width=600*u.nm)
obs= Observation(spG2V_19R, bp)
obs.countrate(area=50*u.m**2)


# zeropoint in filtro r in erg/s/cm2/A
Observation(getNormalizedSpectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).effstim('flam')
# zeropoint in ph/s/m2
Observation(getNormalizedSpectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).countrate(area=1*u.m**2)


class Photometry(object):

    def __init__(self, spectral_type, magnitude, filter, bandpass_filter):
        pass
    
    def countrate(self, area):
        pass
