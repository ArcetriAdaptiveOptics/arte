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
from arte.photometry.spectral_types import PickelsLibrary
from arte.photometry import spectral_types
from arte.photometry.filters import Filters
from arte.photometry.normalized_star_spectrum import get_normalized_star_spectrum






def misc():
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

    spM0V_8R= get_normalized_star_spectrum('M0V', 8.0, 'johnson_r')


    uPhotonSecM2Micron=u.photon/(u.s * u.m**2 * u.micron)

    spG2V_8R= get_normalized_star_spectrum("G2V", 8.0, 'johnson_r')
    plt.plot(spG2V_8R.waveset, spG2V_8R(spG2V_8R.waveset).to(uPhotonSecM2Micron))

    # compare with Armando's
    spG2V_19R= get_normalized_star_spectrum("G2V", 19, 'johnson_r')
    bp= SpectralElement(Box1D, x_0=700*u.nm, width=600*u.nm)
    obs= Observation(spG2V_19R, bp)
    obs.countrate(area=50*u.m**2)


    # zeropoint in filtro r in erg/s/cm2/A
    Observation(get_normalized_star_spectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).effstim('flam')
    # zeropoint in ph/s/m2
    Observation(get_normalized_star_spectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).countrate(area=1*u.m**2)


def zeropoints():
    Observation(get_normalized_star_spectrum('A0V', 0, 'johnson_r'), SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)).countrate(area=1*u.m**2)
    Observation(get_normalized_star_spectrum('G2V', 0, 'johnson_r'), SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)).countrate(area=1*u.m**2)
    Observation(get_normalized_star_spectrum('K3V', 0, 'johnson_r'), SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)).countrate(area=1*u.m**2)
    Observation(get_normalized_star_spectrum('M0V', 0, 'johnson_r'), SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)).countrate(area=1*u.m**2)
    Observation(get_normalized_star_spectrum('M4V', 0, 'johnson_r'), SpectralElement(Box1D, x_0=800*u.nm, width=400*u.nm)).countrate(area=1*u.m**2)


class Photometry(object):

    def __init__(self, spectral_type, magnitude, filter, bandpass_filter):
        pass

    def countrate(self, area):
        pass


