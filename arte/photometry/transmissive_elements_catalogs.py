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
    def ag_mirror_elt_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        ''' 
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('ag_mirror_elt_001'))
        
    @classmethod
    def al_mirror_elt_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('al_mirror_elt_001'))
        
    @classmethod
    def spider_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
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
    def lgs_dichroic_001(cls):
        '''
        LGS dichroic.
        MATERION-like (average) coating.
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls". 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_dichroic_001'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('lgs_dichroic_001'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        # te.add_spectral_points([200*u.nm, 7*u.um], transmittance=[0,0], reflectance=[1,1])
        return te
    
    @classmethod
    def lgs_dichroic_002(cls):
        '''
        LGS dichroic: special coating/suprasil 3002 (80mm)/AR coating (@589 nm)
        
        Data from Demetrio? 
        '''
        pass
    
    @classmethod
    def visir_dichroic_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('visir_dichroic_001'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('visir_dichroic_001'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def schmidt_plate_001(cls):
        '''
        Schmidt plate: Infrasil window (Thorlabs).
        Thickness: 1 mm.
        
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def schmidt_plate_002(cls):
        '''
        Simplified version of Schmidt plate to extract the LGS WFS throughput.
        Based on E-MAO-SF0-INA-DER-001_02, an overall transmittance of 0.95 at 
        589 nm is considered for 2 surfaces with broadband (0.6-2.4 um) AR 
        coating + Suprasil 3002 substrate (thickness = 85 mm).       
        '''
        t = Bandpass.top_hat(589 * u.nm, 1 * u.nm, 0.95, 0)
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te

    @classmethod
    def infrasil_1mm_B_coated_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_B_coated_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def infrasil_1mm_C_coated_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_C_coated_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def lgso_lens_001(cls):
        '''
        Lens in the LGS Objective (LGSO).
        Narrowband (589 nm) AR coating (CHECK).
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgso_lens_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def lgso_fm_001(cls):
        '''
        Folding mirror in the LGS Objective (LGSO).
        First approximation: reflectance curve is simply a peak of 0.99 at
            589 nm.
        Peak value is taken from Section 3.2.15 of
            "E-MAO-SF0-INA-DER-001_02 MAORY  System Optical Design and Analysis Report.pdf".
        
        '''
        r = Bandpass.top_hat(589 * u.nm, 1 * u.nm, 0.99, 0)
        t = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te
    
    @classmethod
    def lgs_wfs_001(cls):
        '''
        LGS WFS transmission excluding camera QE, detector windows and notch
        filter.
        
        Data from Patrick Rabou (received by email on 23-01-2023).
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_wfs_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def CaF2_lens_C_coated_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('caf2_lens_C_coated_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def adc_coated_001(cls):
        '''
        Coated ADC: 2x S-NPH1/N-LAK22/S-NPH1
        Thickness of S-NPH1 (a): 4 mm.
        Thickness of N-LAK22: 8 mm.
        Thickness of S-NPH1 (b): 4 mm.
        
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('adc_coated_001'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('adc_coated_001'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def collimator_doublet_coated_001(cls):
        '''
        Coated collimator doublet: N-SF15/N-BAK1.
        Thickness of N-SF15: 3 mm.
        Thickness of N-BAK1: 5 mm.
        
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls". 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated_001'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated_001'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def collimator_doublet_coated_002(cls):
        '''
        Collimator doublet N-SF5/N-BK7.
        Thickness of N-SF5 = 2.5 mm.
        Thickness of N-BK7 = 6 mm.
        
        Data from ?MB?
         
        '''
        pass
    
    @classmethod
    def notch_filter_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('notch_filter_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def ref_custom_filter_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ref_custom_filter_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def sapphire_window_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('sapphire_window_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def ccd220_qe_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ccd_220_qe_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_qe_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_qe_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def c_red_one_filters_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_filters_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def c_blue_qe_001(cls):
        '''
        C-BLUE camera of First Light Imaging with SONY IMX425 detector.
        First approximation: QE curve is simply a peak of 0.75 at 589 nm.
        QE value is taken from Section 4.5 of
            "E-MAO-PL0-IPA-ANR-013_01 MAORY LGS WFS Analysis Report.pdf" and 
            includes camera and detector windows.
        '''
        t = Bandpass.top_hat(589 * u.nm, 1 * u.nm, 0.75, 0)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

