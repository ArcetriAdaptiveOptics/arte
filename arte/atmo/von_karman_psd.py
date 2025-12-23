'''
@author: giuliacarla
'''

import numpy as np
from scipy.special import jv
import astropy.units as u


class VonKarmanPsd():
    '''
    This class computes the spatial Power Spectral Density (PSD) of turbulent
    phase assuming the Von Karman spectrum.
    The PSD is obtained from the following expression:

    .. math::

        PSD(f,h) = c * r_0(h)^{-5/3} * ( f^2+ \\frac{1}{L_0^2})^{-11/6}

        c = ( \\frac{24}{5} \Gamma(6/5) )^{5/6}  \\frac{\Gamma(11/6)^2}{2 \pi^{11/3} } = 0.0228955


    Parameters
    ----------
    fried_param: float
        Fried parameter characterizing the atmospheric turbulence [m].
    outer_scale: float
        Outer scale of the atmospheric turbulence [m].

    Example
    -------
    Compute the total variance of Kolmogorov over a 10m telescope and 
    compare variance [rad2] with Noll('76): :math:`\Delta_1= 1.029 (D/r_0)^{5/3}`

    >>> R = 5
    >>> r0 = 0.1
    >>> L0 = np.inf
    >>> psd = von_karman_psd.VonKarmanPsd(r0, L0)
    >>> freqs = np.logspace(-8, 4, 1000)
    >>> bess = scipy.special.jv(1, 2*np.pi*R*freqs)
    >>> psdPistonRem = psd.spatial_psd(freqs) * (1 - (bess/(np.pi*R*freqs))**2)
    >>> varInRad2 = np.trapezoid(psdPistonRem*2*np.pi*freqs, freqs)
    >>> varInRad2Noll = 1.029*(2*R/r0)**(5./3)
    >>> print("%g %g" % (varInRad2, varInRad2Noll))
    2214.36 2216.91

    or use shortcut function von_karman_psd.rms()
    '''
    NUM_CONST = 0.02289558710855519

    def __init__(self, fried_param, outer_scale):
        self._r0 = fried_param
        self._L0 = outer_scale
        self._checkInput()

    def _checkInput(self):
        if np.isscalar(self._r0):
            assert np.isscalar(self._L0), 'L0 must be scalar like r0'
        else:
            assert len(self._r0) == len(self._L0), \
                'len of L0 and r0 differ: %d vs %d' % (
                    len(self._L0), len(self._r0))

    def _computeVonKarmanPsd(self, freqs):
        if type(self._r0) == np.ndarray:
            self._psd = np.array([
                self.NUM_CONST * self._r0[i] ** (-5. / 3) *
                (freqs ** 2 + 1 / self._L0[i] ** 2) ** (-11. / 6)
                for i in range(self._r0.shape[0])])
        else:
            self._psd = self.NUM_CONST * self._r0 ** (-5. / 3) * \
                (freqs ** 2 + 1 / self._L0 ** 2) ** (-11. / 6)

    def spatial_psd(self, freqs):
        '''
        Spatial Power Spectral Density of Von Karman turbulence

        Parameters
        ----------
        freqs: :class:`~numpy:numpy.ndarray`
            Spatial frequencies vector[m^-1].

        Returns
        -------
        psd: :class:`~numpy:numpy.ndarray`
            power spectral density computed at the specified frequencies
        '''
        self._computeVonKarmanPsd(freqs)
        return self._psd

    def plot_von_karman_psd_vs_frequency(self, freqs, idx=None):
        import matplotlib.pyplot as plt
        psd = self.spatial_psd(freqs)
        if idx is None:
            plt.loglog(freqs, psd)
        else:
            plt.loglog(freqs, psd[idx])
        plt.xlabel('Frequency [m$^{-1}$]')
        plt.ylabel('PSD [rad$^{2}$]')


def rms(diameter: u.m,
        wavelength: u.nm,
        fried_param: u.m,
        outer_scale: u.m,
        freqs=None):
    '''
    Von Karman wavefront rms value over a circular aperture

    Parameters
    ----------
    diameter: `~astropy.units.quantity.Quantity` equivalent to meter
        Aperture diameter

    wavelength: `~astropy.units.quantity.Quantity` equivalent to nanometer
        wavelength

    fried_param: `~astropy.units.quantity.Quantity` equivalent to meter
        Fried parameter r0 defined at the specified wavelength

    outer_scale: `~astropy.units.quantity.Quantity` equivalent to meter
        Outer scale L0. Use np.inf for Kolmogorov spectrum

    Other Parameters
    ----------
    freqs:  array of `~astropy.units.quantity.Quantity` equivalent to 1/meter
        spatial frequencies array. Default logspace(-8, 4, 1000) m^-1

    Returns
    -------
    rms: `~astropy.units.quantity.Quantity` equivalent to nm
        wavefront rms for the specified von Karman turbulence
    '''
    R = 0.5 * diameter.to(u.m).value
    wl = wavelength.to(u.nm).value
    r0 = fried_param.to(u.m).value
    L0 = outer_scale.to(u.m).value
    psd = VonKarmanPsd(r0, L0)
    if freqs is None:
        freqs = np.logspace(-8, 4, 1000) / u.m
    freqs = freqs.to(1 / u.m).value
    bess = jv(1, 2 * np.pi * R * freqs)
    psdTotal = psd.spatial_psd(freqs)
    psdPistonRem = psdTotal * (1 - (bess / (np.pi * R * freqs)) ** 2)
    varInRad2 = np.trapezoid(psdPistonRem * 2 * np.pi * freqs, freqs)
    return np.sqrt(varInRad2) * wl / 2 / np.pi * u.nm

