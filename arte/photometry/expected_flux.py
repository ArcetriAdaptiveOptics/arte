import numpy as np
from astropy import units as u
from arte.photometry.filters import Filters
from arte.photometry.normalized_star_spectrum import get_normalized_star_spectrum
from arte.photometry.eso_sky_calc import EsoSkyCalc
from synphot.models import Empirical1D
from synphot.spectrum import SpectralElement
from synphot.observation import Observation
import warnings


def _remove_empty_bins(waveset, f_source, filt):
    """
    Remove bins from waveset that don't contain any samples from source or filter.
    
    This prevents ZeroDivisionError in synphot's calcbinflux when bins are empty.
    
    Parameters
    ----------
    waveset : astropy.units.Quantity
        Input wavelength set
    f_source : synphot.spectrum.SourceSpectrum
        Source spectrum
    filt : synphot.spectrum.SpectralElement
        Filter/bandpass
        
    Returns
    -------
    waveset : astropy.units.Quantity
        Sanitized wavelength set with empty bins removed
    """
    from synphot import binning, utils
    
    ws = u.Quantity(waveset).to(u.Angstrom)
    
    # Calculate bin edges
    bin_edges = binning.calculate_bin_edges(ws.value)
    
    # Merge all wavelengths from bins, source, and filter
    spwave = utils.merge_wavelengths(bin_edges.value, ws.value)
    if f_source.waveset is not None:
        spwave = utils.merge_wavelengths(spwave, f_source.waveset.to(u.Angstrom).value)
    if filt.waveset is not None:
        spwave = utils.merge_wavelengths(spwave, filt.waveset.to(u.Angstrom).value)
    
    # Find which bins contain at least one sample
    idx = np.searchsorted(spwave, bin_edges)
    non_empty = idx[1:] > idx[:-1]
    
    # Keep only non-empty bins
    waveset_clean = ws[non_empty]
    
    if len(waveset_clean) < 2:
        raise ValueError(
            f"Too few non-empty bins in waveset. Original bins: {len(ws)}, "
            f"Non-empty bins: {len(waveset_clean)}")
    
    n_removed = len(ws) - len(waveset_clean)
    if n_removed > 0:
        warnings.warn(
            f"Removed {n_removed} empty bins from waveset "
            f"({n_removed/len(ws)*100:.1f}%)")
    
    return waveset_clean


def expected_flux(
        transmissive_system,
        star_spectral_type='K5V',
        magnitude=0,
        filter_name=Filters.ESO_ETC_R,
        waveset=None,
        return_details=False,
        zenith_angle=30 * u.deg,
        include_sky=True):
    '''
    Estimate the expected flux for the given transmissive system
    from a source of a given spectral type .
    
    Parameters
    ----------
    transmissive_system: arte.photometry.transmissive_elements.TransmissiveSystem or TransmissiveElement
        The optical system through which to calculate the flux
    star_spectral_type: str
        Spectral type of the star (default is 'K5V')
    magnitude: float
        Magnitude of the source in the Vega system (default is 0)
    filter_name: str
        Filter name for magnitude normalization (default is ESO_ETC_R)
    waveset: astropy.units.Quantity, optional
        Wavelength set where to compute the expected flux.
        If None, uses the native waveset of the source spectrum (default is None)
    return_details: bool
        If True, also return f_source and obs_source (default is False)
    zenith_angle: astropy.units.Quantity
        Zenith angle for airmass calculation (default is 30 degrees)
    include_sky: bool
        If True, include sky transmission in the calculation (default is True)
    
    Returns
    -------
    counts_source : astropy.units.Quantity
        Expected flux in e-/s/m^2
    f_source : synphot.spectrum.SourceSpectrum (optional)
        Normalized star spectrum of the source.
        Returned if return_details is True
    obs_source : synphot.observation.Observation (optional)
        Observation object for the source filtered by the transmissive system.
        Returned if return_details is True
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> from arte.photometry.transmissive_elements_catalogs import FiltersCatalog
    >>> filt = FiltersCatalog.bessel_H()
    >>> flux = expected_flux(filt)
    
    K5V star with magnitude 8 in R filter through Bessel H:
    
    >>> flux = expected_flux(filt, star_spectral_type='K5V', magnitude=8, 
    ...                      filter_name=Filters.ESO_ETC_R)
    
    Custom wavelength set from 1500 to 1700 nm and zenith angle:
    
    >>> wv = np.arange(15000, 17000, 50) * u.Angstrom
    >>> flux = expected_flux(filt, waveset=wv, zenith_angle=45*u.deg)
    
    Get detailed output and exclude sky contribution:
    
    >>> flux, f_src, obs = expected_flux(filt, return_details=True, include_sky=False)
    
    Compare flux in K band for 2 different glasses:
    
    >>> wvset = np.arange(2000, 2200, 10) * u.nm
    >>> ideal = expected_flux(TransmissiveElement.ideal(), waveset=wvset)
    >>> suprasil = expected_flux(GlassesCatalog.suprasil3002_10mm_internal_001(), waveset=wvset)
    >>> npsk53a = expected_flux(GlassesCatalog.schott_NPSK53A_10_mm_internal_001(), waveset=wvset)
    '''
    
    f_source = get_normalized_star_spectrum(spectral_type=star_spectral_type,
                                            magnitude=magnitude,
                                            filter_name=filter_name)

    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    
    if include_sky:
        sky = EsoSkyCalc(airmass=airmass, incl_moon='N')
        sky_se = SpectralElement(Empirical1D, points=sky.lam,
                                 lookup_table=sky.trans)
        filt = sky_se * transmissive_system.transmittance
    else:
        filt = transmissive_system.transmittance

    warnings.filterwarnings('ignore')

    # Generate initial binset if not provided
    if waveset is None:
        # Let synphot determine the binset
        if filt.waveset is not None:
            initial_binset = filt.waveset
        elif f_source.waveset is not None:
            initial_binset = f_source.waveset
        else:
            raise ValueError("Both source and filter have undefined waveset")
        
        # Remove empty bins from the generated binset
        waveset = _remove_empty_bins(initial_binset, f_source, filt)
    else:
        # Remove empty bins from user-provided waveset
        waveset = _remove_empty_bins(waveset, f_source, filt)

    obs_source = Observation(spec=f_source, band=filt,
                             binset=waveset, force='taper')

    area_subap = 1 * u.m ** 2
    exp_time = 1 * u.s
    
    counts_source = obs_source.countrate(area=area_subap, binned=False, wavelengths=waveset)
    counts_source = (counts_source * exp_time).decompose()  # e-/s/m^2

    print('Expected flux for %s star (mag=%s in %s filter) in e-/s/m2 (x1e8): %s'
          % (star_spectral_type, magnitude, filter_name, counts_source / 1e8))

    if return_details:
        return counts_source, f_source, obs_source
    return counts_source
