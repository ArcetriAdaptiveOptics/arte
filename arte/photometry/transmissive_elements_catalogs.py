import os
import astropy.units as u
from arte.photometry.transmissive_elements import Bandpass, TransmissiveElement
from arte.utils.package_data import dataRootDir
from synphot.spectrum import SpectralElement


class RestoreTransmissiveElements(object):

    @classmethod
    def transmissive_elements_folder(cls):
        rootDir = dataRootDir()
        dirname = os.path.join(rootDir, 'photometry', 'optical_elements')
        return dirname

    @classmethod
    def restore_transmissive_elements_from_fits(cls, foldername):
        filename = os.path.join(foldername, '0.fits')
        return TransmissiveElement.from_fits(filename)

    @classmethod
    def restore_transmittance_from_dat(cls, foldername, wavelength_unit):
        filename = os.path.join(foldername, 't.dat')
        return SpectralElement.from_file(filename, wave_unit=wavelength_unit)

    @classmethod
    def restore_reflectance_from_dat(cls, foldername, wavelength_unit):
        filename = os.path.join(foldername, 'r.dat')
        return SpectralElement.from_file(filename, wave_unit=wavelength_unit)
    

class EltTransmissiveElementsCatalog():
    
    @classmethod
    def _EltFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'elt',
            foldername)
    
    @classmethod
    def ag_mirror_elt(cls):
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('ag_mirror_elt'))
        
    @classmethod
    def al_mirror_elt(cls):
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('al_mirror_elt'))
        
    @classmethod
    def spider(cls):
        a = Bandpass.one() * 0.043
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te


class MorfeoTransmissiveElementsCatalog():
    
    @classmethod
    def _MorfeoFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'morfeo',
            foldername)
        
    @classmethod
    def lgs_dichroic(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_dichroic'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('lgs_dichroic'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        # te.add_spectral_points([200*u.nm, 7*u.um], transmittance=[0,0], reflectance=[1,1])
        return te
    
    @classmethod
    def visir_dichroic(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('visir_dichroic'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('visir_dichroic'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def infrasil_1mm(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def infrasil_1mm_B_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_B_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def infrasil_1mm_C_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_C_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def lgs_lens(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_lens'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def lgs_wfs(cls):
        '''
        Spiegazione ... ricevuta via email da tizio nella data
        versione dei dati
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_wfs'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def CaF2_lens_C_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('caf2_lens_C_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def adc_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('adc_coated'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('adc_coated'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def collimator_doublet_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def notch_filter(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('notch_filter'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def ref_custom_filter(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ref_custom_filter'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def sapphire_window(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('sapphire_window'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def ccd220_qe(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ccd_220_qe'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_qe(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_qe'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def c_red_one_filters(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_filters'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
