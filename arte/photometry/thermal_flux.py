import numpy as np
import astropy.units as u
from synphot.spectrum import SourceSpectrum
from synphot.models import BlackBody1D

    
class ThermalFluxOnWFS():
    '''
    Compute the thermal flux measured as background on a WFS detector due to the
    emissivity of the elements in the optical path of the system.
    Assume all the elements emit as a black body with the same temperature. 
    Return the thermal flux in [e-/s/µm/px].
    
    Parameters
    ----------
    temperature_in_K: `astropy.units.quantity.Quantity`
        Temperature of the black body emission in [K].
    emissivity: `synphot.spectrum.SpectralElement`
        Emissivity of the system up to the detector.
    lenslet_focal_length: `astropy.units.quantity.Quantity`
        Focal length of the WFS lenslet array in [mm].
    lenslet_size: `astropy.units.quantity.Quantity`
        Size of the WFS lenslet array in [mm]
    pixel_size: `astropy.units.quantity.Quantity`
        Pixel size of the detector in [µm/pix]
    qe: `arte.photometry.transmissive_elements.TransmissiveElement`
        Quantum efficiency of the detector.
    '''
    
    def __init__(self, temperature_in_K, emissivity, lenslet_focal_length,
                 lenslet_size, pixel_size, qe):
        self._T = temperature_in_K
        self._emissivity = emissivity
        self._la_focal_length = lenslet_focal_length
        self._la_size = lenslet_size
        self._px_size = pixel_size
        self._qe = qe 
        self.waveset = emissivity.waveset
        
    def _compute_blackbody_spectrum(self):
        self.bb_spectrum = SourceSpectrum(BlackBody1D, temperature=self._T)
    
    def _compute_solid_angle(self):
        self.solid_angle = np.pi / 4 * self._la_size ** 2 / (
            self._la_focal_length ** 2 + (self._la_size / 2) ** 2) * u.sr
    
    def thermal_flux(self):
        '''
        Returns
        -------
        thermal_flux: `astropy.units.quantity.Quantity`
            Thermal flux on the WFS detector in [e-/s/pix/µm]
        '''
        self._compute_blackbody_spectrum()
        self._compute_solid_angle()
        # TODO: ricontrollare le units
        th_flux = (self.bb_spectrum * self._emissivity * self._qe.transmittance
                   )(self.waveset) * self.solid_angle.value * (
                       self._px_size.to(u.cm)) ** 2
        return th_flux
