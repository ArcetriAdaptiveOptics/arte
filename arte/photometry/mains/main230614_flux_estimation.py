import numpy as np
import astropy.units as u
from arte.photometry.filters import Filters
from arte.photometry.normalized_star_spectrum import get_normalized_star_spectrum
from arte.photometry.eso_sky_calc import EsoSkyCalc
from synphot.models import Empirical1D
from synphot.spectrum import SpectralElement
from arte.photometry.transmissive_elements import TransmissiveElement, Bandpass
from synphot.observation import Observation
from arte.photometry.morfeo_transmissive_systems import MorfeoLowOrderChannelTransmissiveSystem_002, \
    MorfeoReferenceChannelTransmissiveSystem_002
import warnings


def main230711_estimate_expected_flux_at_R_WFS():
    '''
    Estimate the expected flux at the R WFS considering the throughput curve
    reported in 'E-MAO-SA0-INA-TNO-005 WFS Flux budget'.
    '''
    filt_LO_name = Filters.BESSEL_H
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_LO_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    warnings.filterwarnings('ignore')
    r_wfs = MorfeoReferenceChannelTransmissiveSystem_002() 
    
    filt = sky_se * r_wfs.transmittance
    
    obs = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    
    area_subap = 1 * u.m ** 2
    counts = obs.countrate(area=area_subap)
    exp_time = 1 * u.s
    print('Expected flux at R WFS in ph/s/m2 (x1e8): %s'
          % ((counts * exp_time).decompose() / 1e8))


def main230711_estimate_expected_flux_at_LO_WFS():
    '''
    Estimate the expected flux at the LO WFS considering the throughput curve
    reported in 'E-MAO-SA0-INA-TNO-005 WFS Flux budget'.
    '''
    filt_LO_name = Filters.BESSEL_H
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_LO_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    warnings.filterwarnings('ignore')
    lo_wfs = MorfeoLowOrderChannelTransmissiveSystem_002() 
    
    filt = sky_se * lo_wfs.transmittance
    
    obs = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    
    area_subap = 1 * u.m ** 2
    counts = obs.countrate(area=area_subap)
    exp_time = 1 * u.s
    print('Expected flux at LO WFS in ph/s/m2 (x1e8): %s'
          % ((counts * exp_time).decompose() / 1e8))


def main230614_check_expected_flux_at_LO_with_SA0():
    '''
    In 'E-MAO-SA0-INA-DER-001_02 AO Design and analysis report' the throughput
    considered for the telescope + MORFEO system (up to LO WFS included) is
    constant: 0.327 in H band that is defined from 1485 to 1815 nm.
    '''
    filt_LO_name = Filters.BESSEL_H
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_LO_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    lo_ch_constant_transmittance = TransmissiveElement(
        transmittance=Bandpass.top_hat(
            peak_wl=1650 * u.nm, delta_wl=165 * u.nm, high_ampl=0.327,
            low_ampl=0.),
        reflectance=Bandpass.zero())
    
    filt = sky_se * lo_ch_constant_transmittance.transmittance
    
    obs = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    
    area_subap = 1 * u.m ** 2
    counts = obs.countrate(area=area_subap)
    exp_time = 1 * u.s
    print('Expected flux at LO WFS in ph/s/m2 (x1e8): %s'
          % ((counts * exp_time).decompose() / 1e8))
    print('SA0 expected flux at LO WFS in ph/s/m2 (x1e8): 9.41')
