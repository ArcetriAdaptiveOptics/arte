
import numpy as np
import astropy.units as u

from arte.utils.help import ThisClassCanHelp, add_to_help

class MagEstimator(ThisClassCanHelp):
    '''
    Estimates magnitude from detector counts
   
    From https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html#mags

Band	Filter curve	Photometric zeropoint in ergs/s/cm2/A
(flux for zero mag in Vega system)
U	U filter curve	U0 = 4.18023e-09
B	B filter curve	B0 = 6.60085e-09
V	V filter curve	V0 = 3.60994e-09
R	R filter curve	R0 = 2.28665e-09
I	I filter curve	I0 = 1.22603e-09
Z	Z filter curve	Z0 = 7.76068e-10
Y	Y filter curve	Y0 = 5.973e-10
J	J filter curve	J0 = 3.12e-10
H	H filter curve	H0 = 1.14e-10
K	K filter curve	K0 = 3.94e-11
L	L filter curve	L0 = 4.83e-12
M	M filter curve	M0 = 2.04e-12
N	N filter curve	N0 = 1.23e-13
Q	Q filter curve	Q0 = 6.8e-15

    Written by: A. Puglisi, Feb. 2020  alfio.puglisi@inaf.it
    '''

    def __init__(self, total_adus,
                       detector_gain,
                       detector_adu_e_ratio,
                       detector_nsubaps,
                       detector_freq,
                       wfs_transmission,
                       detector_bandname,
                       detector_bandwidth,
                       detector_qe,
                       telescope):

        self._total_adus = total_adus
        self._detector_gain = detector_gain
        self._detector_adu_e_ratio = detector_adu_e_ratio
        self._detector_nsubaps = detector_nsubaps
        self._detector_freq = detector_freq
        self._wfs_transmission = wfs_transmission
        self._bandname = detector_bandname
        self._bandwidth = detector_bandwidth
        self._detector_qe = detector_qe
        self._telescope = telescope

    @property
    @add_to_help
    def bandname(self):
        ''' Name of the band used for magnitude estimation '''
        return self._bandname

    @add_to_help
    def flux_zero(self):
        '''
        Zero point in photons/sec

        Calculates zero points using references fluxes from
        https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html#mags
        '''
        if self._bandname == 'R':
            ergs, wl = 2.28665e-9*u.erg/u.s/u.cm**2/u.angstrom, 650e-9*u.m
        elif self._bandname == 'I':
            ergs, wl = 1.22603e-9*u.erg/u.s/u.cm**2/u.angstrom, 900e-9*u.m
        elif self._bandname == 'V':
            ergs, wl = 3.60994e-9*u.erg/u.s/u.cm**2/u.angstrom, 532e-9*u.m
        else:
            raise Exception('Unsupported band '+self._bandname)

        photons = self._ergs_to_photons(ergs, wl)

        return (photons * self._telescope.area() * self._bandwidth).to(u.ph/u.s)
    
    def _ergs_to_photons(self, ergs, wl):
        '''
        Converts from ergs to number of photons at a given wavelength

        Uses the ergs/cm^2/s/A to photons/cm^2/s/A constants from:
        https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
        http://www.stsci.edu/~strolger/docs/UNITS.txt
        '''
        return ergs * 5.0341125e7 * wl.to(u.angstrom).value * (u.ph/u.erg)

    @add_to_help
    def photons_per_subap_per_frame(self):
        ''' Photons/subap/frame detected by sensor'''
        gain = self._detector_gain
        nsubaps = self._detector_nsubaps
        adu_e_ratio = self._detector_adu_e_ratio
        qe = self._detector_qe

        return self._total_adus * adu_e_ratio / gain / nsubaps / qe

    @add_to_help
    def photons_per_second(self):
        ''' Photons/sec detected by sensor'''
        ph_subap_frame = self.photons_per_subap_per_frame()

        freq = self._detector_freq.to('1/s')
        nsubaps = self._detector_nsubaps
        transmission = self._wfs_transmission

        return ph_subap_frame * freq * nsubaps / transmission

    @add_to_help
    def mag(self):
        ''' Estimated magnitude '''
        flux = self.photons_per_second()
        zero_flux = self.flux_zero()

        return -2.5*np.log10(flux / zero_flux)

