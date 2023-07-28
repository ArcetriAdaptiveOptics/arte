import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from arte.photometry.filters import Filters
from arte.photometry.normalized_star_spectrum import get_normalized_star_spectrum
from arte.photometry.eso_sky_calc import EsoSkyCalc
from synphot.models import Empirical1D
from synphot.spectrum import SpectralElement
from arte.photometry.transmissive_elements import TransmissiveElement, Bandpass
from synphot.observation import Observation
from arte.photometry.morfeo_transmissive_systems import MorfeoLgsChannelTransmissiveSystem_005, \
    MorfeoReferenceChannelTransmissiveSystem_004, \
    MorfeoLowOrderChannelTransmissiveSystem_003
import warnings


def main230725_estimate_expected_flux_at_LO_WFS_J_cut_included():
    '''
    Estimate the expected flux for a K5V star with m=0 (Vega system) at the LO
    WFS considering the throughput curve reported in
    'E-MAO-SA0-INA-TNO-005 WFS Flux budget'.
    The cut of J band is here considered.
    Values for Vega star are reported for comparison.
    '''
    filt_LO_name = Filters.BESSEL_H
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_LO_name)
    f_K5V = get_normalized_star_spectrum(spectral_type='K5V', magnitude=0,
                                          filter_name=filt_LO_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    warnings.filterwarnings('ignore')
    lo_wfs = MorfeoLowOrderChannelTransmissiveSystem_003() 
    
    wv_min = 1.490 * u.um
    wv_max = 1.780 * u.um
    delta = (wv_max - wv_min) / 2
    cut = Bandpass.top_hat(wv_min + delta, delta, 1., 0.)
    
    filt = sky_se * lo_wfs.transmittance * cut
    
    obs_vega = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    obs_K5V = Observation(spec=f_K5V, band=filt, force='taper',
                          binset=f_vega.waveset)
    
    area_subap = 1 * u.m ** 2
    exp_time = 1 * u.s
    counts_vega = obs_vega.countrate(area=area_subap)
    print('\nExpected flux for Vega star at LO WFS in e-/s/m2 (x1e8): %s'
          % ((counts_vega * exp_time).decompose() / 1e8))
    counts_K5V = obs_K5V.countrate(area=area_subap)
    print('\nExpected flux for K5V star (m=0 in Vega system) at LO WFS in e-/s/m2 (x1e8): %s'
          % ((counts_K5V * exp_time).decompose() / 1e8))
    return f_vega, f_K5V, obs_vega, obs_K5V


def main230711_estimate_expected_flux_at_R_WFS(star_spectral_type='K5V'):
    '''
    Estimate the expected flux at the R WFS considering the throughput curve
    reported in 'E-MAO-SA0-INA-TNO-005 WFS Flux budget'.
    '''
    filt_R_name = Filters.ESO_ETC_R
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_R_name)
    f_source = get_normalized_star_spectrum(spectral_type=star_spectral_type,
                                         magnitude=0,
                                         filter_name=filt_R_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    warnings.filterwarnings('ignore')
    r_wfs = MorfeoReferenceChannelTransmissiveSystem_004() 
    
    filt = sky_se * r_wfs.transmittance
    
    obs_vega = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    obs_source = Observation(spec=f_source, band=filt, force='taper',
                          binset=f_source.waveset)
    
    area_subap = 1 * u.m ** 2
    exp_time = 1 * u.s
    counts_vega = obs_vega.countrate(area=area_subap)
    print('\nExpected flux for Vega star at R WFS in e-/s/m2 (x1e8): %s'
          % ((counts_vega * exp_time).decompose() / 1e8))
    counts_source = obs_source.countrate(area=area_subap)
    print('\nExpected flux for %s star (m=0 in Vega system) at R WFS in e-/s/m2 (x1e8): %s'
          % (star_spectral_type, (counts_source * exp_time).decompose() / 1e8))
    return f_vega, f_source, obs_vega, obs_source


def main230711_estimate_expected_flux_at_LO_WFS(star_spectral_type='K5V'):
    '''
    Estimate the expected flux for a K5V star with m=0 (Vega system) at the LO
    WFS considering the throughput curve reported in
    'E-MAO-SA0-INA-TNO-005 WFS Flux budget'.
    Values for Vega star are reported for comparison.
    '''
    filt_LO_name = Filters.ESO_ETC_H
    
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    f_vega = get_normalized_star_spectrum(spectral_type='vega', magnitude=0,
                                          filter_name=filt_LO_name)
    f_source = get_normalized_star_spectrum(spectral_type=star_spectral_type,
                                            magnitude=0,
                                            filter_name=filt_LO_name)
    
    sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
    sky_se = SpectralElement(Empirical1D, points=sky.lam,
                             lookup_table=sky.trans)
    
    warnings.filterwarnings('ignore')
    lo_wfs = MorfeoLowOrderChannelTransmissiveSystem_003() 
    
    filt = sky_se * lo_wfs.transmittance
    
    obs_vega = Observation(spec=f_vega, band=filt, force='taper',
                      binset=f_vega.waveset)
    obs_source = Observation(spec=f_source, band=filt, force='taper',
                             binset=f_source.waveset)
    
    area_subap = 1 * u.m ** 2
    exp_time = 1 * u.s
    counts_vega = obs_vega.countrate(area=area_subap)
    print('\nExpected flux for Vega star at LO WFS in e-/s/m2 (x1e8): %s'
          % ((counts_vega * exp_time).decompose() / 1e8))
    counts_source = obs_source.countrate(area=area_subap)
    print('\nExpected flux for %s star (m=0 in Vega system) at LO WFS in e-/s/m2 (x1e8): %s'
          % (star_spectral_type, (counts_source * exp_time).decompose() / 1e8))
    return f_vega, f_source, obs_vega, obs_source


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
    print('Expected flux at LO WFS in e-/s/m2 (x1e8): %s'
          % ((counts * exp_time).decompose() / 1e8))
    print('SA0 expected flux at LO WFS in e-/s/m2 (x1e8): 9.41')
    

def main230717_estimate_expected_flux_at_LGS_WFS():
    flux_at_M1 = 4.4e6 * u.ph / u.m ** 2 / u.s
    D_M1 = 38.542 * u.m
    N_subap = 68
    T = 2 * u.ms
    lgs_ch = MorfeoLgsChannelTransmissiveSystem_005()
    waveset = lgs_ch.transmittance.waveset
    lgs_transmittance = lgs_ch.transmittance(waveset)
    flux_at_lgs = flux_at_M1 * lgs_transmittance * T.to(u.s) * (
        D_M1 / N_subap) ** 2
    print('\nThroughput w/o losses at 0.589 μm: %s' % lgs_transmittance.max())
    print('\nFlux at 0.589 μm: %s e-/subap/frame' % flux_at_lgs.max())

    sony_acc_angle_loss = 0.92
    lenslet_manu_loss = 0.8
    throughput = lgs_transmittance.max() * sony_acc_angle_loss * lenslet_manu_loss
    print('\nThroughput w/ losses: %s' % throughput)
    flux_final = flux_at_M1 * throughput * T.to(u.s) * (D_M1 / N_subap) ** 2
    print('\nFinal flux: %s' % flux_final)
