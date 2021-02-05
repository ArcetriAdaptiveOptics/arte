
import synphot
from synphot import SourceSpectrum
from arte.photometry.spectral_types import PickelsLibrary
from arte.photometry.filters import Filters


def get_normalized_star_spectrum(spectral_type, magnitude, filter_name):
    """
    spec_data = get_normalized_star_spectrum(spectral_type, magnitude, filter_name)

    Returns a structure containing the synthetic spectrum of the star having the spectral type and magnitude 
    in the specified input filter. Magnitude is in VEGAMAG-F(lambda) system.
    Spectra are from PICKLES, PASP, 110, 863 (1998)
    Absolute flux spectra, no effect of atmospheric and instrument transmission


    Parameters
    ----------
        r0AtZenith: float
                    overall r0 at zenith [m]

        spectral_type:  string.
                        spectral type and luminosity class (e.g. G2V or M4III) or 'vega'
        magnitude:  float.
                    magnitude in the filter_name filter
        filter_name: string.
                     Name of the filter. See Filters.get() for the list of available filters

    Returns
    -------
        spectrum: synphot.SourceSpectrum object defining the spectrum

    Examples
    --------
    Plot the spectrum of a vega, A0V, G2V stars of mag=8 defined on JohnsonR filter

    >>> sp= get_normalized_star_spectrum('vega', 8, Filters.JOHNSON_R)
    >>> spA0V= get_normalized_star_spectrum('A0V', 8, Filters.JOHNSON_R)
    >>> spG2V= get_normalized_star_spectrum('G2V', 8, Filters.JOHNSON_R)
    >>> plt.plot(sp.waveset, sp(sp.waveset), label='Vega')
    >>> plt.plot(spA0V.waveset, spA0V(spA0V.waveset), label='A0V')
    >>> plt.plot(spG2V.waveset, spG2V(spG2V.waveset), label='G2V')
    >>> plt.grid(True)
    >>> plt.xlabel('nm')
    >>> plt.ylabel('FLAM')
    >>> plt.xlim(0, 10000)
    >>> plt.legend()
"""
    # read the sourcespectrum
    if spectral_type == 'vega':
        spectrum = SourceSpectrum.from_vega()
    else:
        spectrum = SourceSpectrum.from_file(
            PickelsLibrary.filename(spectral_type))

    bandpass = Filters.get(filter_name)

    spectrum_norm = spectrum.normalize(
        magnitude * synphot.units.VEGAMAG,
        bandpass,
        vegaspec=SourceSpectrum.from_vega())

    return spectrum_norm
