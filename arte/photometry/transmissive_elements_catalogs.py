import os
import astropy.units as u
from arte.photometry.transmissive_elements import Bandpass, TransmissiveElement, \
    TransmissiveSystem, Direction
from arte.utils.package_data import dataRootDir
from synphot.spectrum import SpectralElement
from arte.photometry.transmittance_calculator import interface_glass_to_glass, \
    internal_transmittance_calculator
from synphot.models import Empirical1D


class RestoreTransmissiveElements(object):

    @classmethod
    def transmissive_elements_folder(cls):
        rootDir = dataRootDir()
        dirname = os.path.join(rootDir, 'photometry', 'transmissive_elements')
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

    @classmethod
    def restore_refractive_index_from_dat(cls, foldername, wavelength_unit):
        filename = os.path.join(foldername, 'n.dat')
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
    def ag_mirror_elt_002(cls):
        '''
        Measured data (Advanced Silver Coating development contract by ESO
        with Fraunhofer IOF).
        
        Received from Demetrio by email on 16/03/2023 (filename:
        "ProtectedSilver_ESO_Measured_1.0".
        ''' 
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._EltFolder('ag_mirror_elt_002'), u.nm)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te

    @classmethod
    def ag_mirror_elt_003(cls):
        '''
        The single silver surface reflectivity has been extracted by telescope
        model throughput (fresh clean ELT) done by R. Holzloehner.
        
        Received from Demetrio by email on 16/03/2023 (filename:
        "ProtectedSilver_ESO_ExtrapolatedModel_1.0").
        
        Same data as version 001.
        ''' 
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._EltFolder('ag_mirror_elt_003'), u.nm)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te
        
    @classmethod
    def al_mirror_elt_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('al_mirror_elt_001'))

    @classmethod
    def al_mirror_elt_002(cls):
        '''
        Data from Demetrio: "OxidizedAluminium_ESO_Model_1.0".
        Received by email on 16/03/2023.
        '''
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._EltFolder('al_mirror_elt_002'), u.nm)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te
        
    @classmethod
    def spider_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        a = Bandpass.one() * 0.043
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te


class CoatingsTransmissiveElementsCatalog():
    
    @classmethod
    def _CoatingsFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'coatings',
            foldername)
        
    @classmethod
    def materion_average_001(cls):
        '''
        MATERION-like coating.
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls". 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatingsFolder('materion_average_001'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._CoatingsFolder('materion_average_001'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def materion_average_002(cls):
        '''
        MATERION-like coating.
        Data (average) from Andrea Bianco.
        
        Received from Demetrio by email on 16/03/2023 (filename:
        "DichroicFilter_Design1_ABI_reflectance_1.0",
        "DichroicFilter_Design1_ABI_transmittance_1.0").
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatingsFolder('materion_average_002'), u.nm)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._CoatingsFolder('materion_average_002'), u.nm)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def ar_coating_589nm_001(cls):
        '''
        Narrowband (589 nm) AR coating. This is a simplified version, i.e.
        a peak at 589 nm.
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatingsFolder('ar_589nm_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def ar_coating_broadband_001(cls):
        '''
        Broadband AR coating for CPM.
        Data from Demetrio, received by email on 12/04/2023.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatingsFolder('ar_broadband_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def ar_coating_swir_001(cls):
        '''
        SWIR AR coating.
        Data from Edmund Optics Website.
        '''
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._CoatingsFolder('ar_swir_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te

    @classmethod
    def ar_coating_nir_i_001(cls):
        '''
        NIR I AR coating.
        Data from Edmund Optics Website.
        '''
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._CoatingsFolder('ar_nir_i_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te

    @classmethod
    def ar_coating_amus_001(cls):
        '''
        AR coating for LO WFS lenslet array. This is a simplified version based
        on Amus saying that the average transmission over 1500-1800 nm range is
        98%.
        NOTE: 98% is assumed over the whole range defined in Bandpass.
        '''
        t = Bandpass.one() * 0.98
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def lzh_coating_for_visir_dichroic_001(cls):
        '''
        Coating for VISIR dichroic from LZH. Data extrapolated from LZH plot.
        No information from ~1000 to 1450 nm.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatingsFolder('lzh_visir_dichroic_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te


class DetectorsTransmissiveElementsCatalog():
    
    @classmethod
    def _DetectorsFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'detectors',
            foldername)
        
    @classmethod
    def ccd220_qe_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        Standard silicon data.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('ccd_220_qe_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def ccd220_qe_002(cls):
        '''
        Data from Teledyne e2v website.
        Deep depleted silicon data.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('ccd_220_qe_002'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def ccd220_qe_003(cls):
        '''
        Data from ESO doc: 'ESO-287869_2 ALICE Camera Technical Requirements
        Specifications.pdf'.
        Deep depleted silicon data.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('ccd_220_qe_003'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_qe_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('c_red_one_qe_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def c_red_one_H_filter_001(cls):
        '''
        C-RED One cold filter (H). 
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('c_red_one_H_filter_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_K_filter_001(cls):
        '''
        C-RED One cold filter (K). 
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('c_red_one_K_filter_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_filters_001(cls):
        '''
        Data from Cedric's spreadsheet "background_calc_maory_v12.xls".
        2H + 2K filters are considered.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._DetectorsFolder('c_red_one_filters_001'), u.um)
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
        t = Bandpass.top_hat(589 * u.nm, 1 * u.nm, 0.7, 0)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    

class CoatedGlassesTransmissiveElementsCatalog():
    
    @classmethod
    def _CoatedGlassesFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'coated_glasses',
            foldername)
        
    @classmethod
    def infrasil_1mm_B_coated_001(cls):
        '''
        Infrasil window B-coated.
        Thickness: 1 mm.
        
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatedGlassesFolder('infrasil_1mm_B_coated_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def infrasil_1mm_C_coated_001(cls):
        '''
        Infrasil window C-coated.
        Thickness: 1 mm.
        
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._CoatedGlassesFolder('infrasil_1mm_C_coated_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
        

class GlassesTransmissiveElementsCatalog():
    
    @classmethod
    def _GlassesFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'glasses',
            foldername)
        
    @classmethod
    def cdgm_HQK3L_7mm_internal_001(cls):
        '''
        CDGM H-QK3L substrate of 7 mm thickness.
        Transmittance is internal.
        Data from RefractiveIndex website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('cdgm_HQK3L_7mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te    

    @classmethod
    def ohara_SFTM16_3mm_internal_001(cls):
        '''
        Ohara S-FTM16 substrate of 3 mm thickness.
        Transmittance is internal.
        Data from RefractiveIndex website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SFTM16_3mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te    
        
    @classmethod
    def infrasil_1mm_001(cls):
        '''
        Infrasil window.
        Thickness: 1 mm.
        Transmittance is external.
        
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('infrasil_1mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def suprasil3002_10mm_001(cls):
        '''
        Suprasil 3002 substrate of 10 mm thickness.
        Transmittance is external.
        Data from Heraeus website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_10mm_001'), u.um)
        # TODO: Assuming reflectance equal to zero is wrong in case of 
        # external transmittance data. Fix it.
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def suprasil3001_3mm_internal_001(cls):
        '''
        Suprasil 3001 substrate of 3 mm thickness.
        Transmittance is internal.
        Data extrapolated from 10 mm curves.
        '''
        supra10mm = cls.suprasil3002_10mm_001()
        wv = supra10mm.waveset
        t1 = supra10mm.transmittance(wv)
        l1 = 10
        l2 = 3
        t2 = internal_transmittance_calculator(l1, l2, t1)
        t = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=t2)
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te
        
    @classmethod
    def suprasil3002_10mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 10 mm thickness.
        Transmittance is internal.
        Data from Heraeus website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_10mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def suprasil3002_40mm_001(cls):
        '''
        Suprasil 3002 substrate of 40 mm thickness.
        Transmittance is external.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_40mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_40mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 40 mm thickness.
        Transmittance is internal.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_40mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def suprasil3002_60mm_001(cls):
        '''
        Suprasil 3002 substrate of 60 mm thickness.
        Transmittance is external.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_60mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_60mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 60 mm thickness.
        Transmittance is internal.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_60mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_70mm_001(cls):
        '''
        Suprasil 3002 substrate of 70 mm thickness.
        Transmittance is external.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_70mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_70mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 70 mm thickness.
        Transmittance is internal.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_70mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def suprasil3002_80mm_001(cls):
        '''
        Suprasil 3002 substrate of 80 mm thickness.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_80mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_80mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 80 mm thickness.
        Transmittance is internal.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_80mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_85mm_001(cls):
        '''
        Suprasil 3002 substrate of 85 mm thickness.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_85mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_85mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 85 mm thickness.
        Transmittance is internal.
        Data has been extrapolated from 10 mm Suprasil 3002 data that has been
        collected from Heraeus website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_85mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
        
    @classmethod
    def suprasil3002_108mm_001(cls):
        '''
        Suprasil 3002 substrate of 108 mm thickness.
        Transmittance is external.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_108mm_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def suprasil3002_108mm_internal_001(cls):
        '''
        Suprasil 3002 substrate of 108 mm thickness.
        Transmittance is internal.
        Data from Heraeus website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('suprasil3002_108mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def ohara_quartz_SK1300_10mm_internal_001(cls):
        '''
        Ohara quartz SK-1300 substrate of 10 mm thickness.
        Transmittance is internal.
        Data from Ohara website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_quartz_SK1300_10mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def ohara_quartz_SK1300_85mm_internal_001(cls):
        '''
        Ohara quartz SK-1300 substrate of 85 mm thickness.
        Transmittance is internal.
        Data derived from transmittance data for 10 mm thickness. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_quartz_SK1300_85mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def ohara_SFPL51_10mm_internal_001(cls):
        '''
        Ohara SFPL-51 substrate of 10 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SFPL51_10mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def ohara_PBL35Y_10mm_internal_001(cls):
        '''
        Ohara PBL-35Y substrate of 10 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_PBL35Y_10mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def ohara_PBL35Y_3mm_internal_001(cls):
        '''
        Ohara PBL-35Y substrate of 3 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_PBL35Y_3mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def schott_NSF2_9dot8_mm_internal_001(cls):
        '''
        Schott N-SF2 substrate of 9.8 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('schott_NSF2_9.8mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def schott_NPSK53A_10_mm_internal_001(cls):
        '''
        Schott N-PSK53A substrate of 10 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('schott_NPSK53A_10mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def interface_ohara_SFPL51_to_ohara_PBL35Y_001(cls):
        '''
        Interface between Ohara SFPL-51 and Ohara PBL-35Y.
        Data from RefractiveInfo website. 
        '''
        n_sfpl51 = RestoreTransmissiveElements.restore_refractive_index_from_dat(
            cls._GlassesFolder('ohara_SFPL51_10mm_internal_001'), u.um)
        n_pbl35y = RestoreTransmissiveElements.restore_refractive_index_from_dat(
            cls._GlassesFolder('ohara_PBL35Y_10mm_internal_001'), u.um)
        wv = n_sfpl51.waveset
        r = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=interface_glass_to_glass(n_sfpl51(wv), n_pbl35y(wv)))
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te
    
    @classmethod
    def interface_schott_NSF2_to_schott_NPSK53A_001(cls):
        '''
        Interface between Schott N-SF2 and Schott N-PSK53A.
        Data from RefractiveInfo website. 
        '''
        n_sf2 = RestoreTransmissiveElements.restore_refractive_index_from_dat(
            cls._GlassesFolder('schott_NSF2_9.8mm_internal_001'), u.um)
        n_npsk53a = RestoreTransmissiveElements.restore_refractive_index_from_dat(
            cls._GlassesFolder('schott_NPSK53A_10mm_internal_001'), u.um)
        wv = n_sf2.waveset
        r = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=interface_glass_to_glass(n_sf2(wv), n_npsk53a(wv)))
        a = Bandpass.zero()
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
        LGS dichroic. This is a simplified version: one surface only with 
        MATERION-like average coating.
        '''
        te = CoatingsTransmissiveElementsCatalog.materion_average_001()
        return te
    
    @classmethod
    def lgs_dichroic_002(cls):
        '''
        LGS dichroic. The element is composed by:
            - 1 surface with MATERION-like coating
            - 85 mm of Suprasil 3002 substrate
            - 1 surface with AR coating (589 nm)
        '''
        materion_coating = CoatingsTransmissiveElementsCatalog.materion_average_001()
        substrate = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
        ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        
        lgs_dichroic = TransmissiveSystem()
        lgs_dichroic.add(materion_coating, Direction.TRANSMISSION)
        lgs_dichroic.add(substrate, Direction.TRANSMISSION)
        lgs_dichroic.add(ar_coating, Direction.TRANSMISSION)
        return lgs_dichroic.as_transmissive_element()
    
    @classmethod
    def lgs_dichroic_003(cls):
        '''
        LGS dichroic. The element is composed by:
            - 1 surface with MATERION-like coating
            - 85 mm of Suprasil 3002 substrate
            - 1 surface with AR coating (589 nm)
            
        Difference wrt version 002: materion-like coating is from A. Bianco.
        '''
        materion_coating = CoatingsTransmissiveElementsCatalog.materion_average_002()
        substrate = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
        ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        
        lgs_dichroic = TransmissiveSystem()
        lgs_dichroic.add(materion_coating, Direction.TRANSMISSION)
        lgs_dichroic.add(substrate, Direction.TRANSMISSION)
        lgs_dichroic.add(ar_coating, Direction.TRANSMISSION)
        return lgs_dichroic.as_transmissive_element()

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
    def visir_dichroic_002(cls):
        '''
        VISIR dichroic. The design is from "E-MAO-PN0-INA-DER-001 MAORY LOR WFS
        Module Design Report":
            - Coating from LZH
            - 3 mm of fused silica substrate
            - AR coating (we assume here the same coating as for the LO lenslet)
        '''
        lzh_coating = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
        substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
        ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
        
        visir_dich = TransmissiveSystem()
        visir_dich.add(lzh_coating, Direction.TRANSMISSION)
        visir_dich.add(substrate, Direction.TRANSMISSION)
        visir_dich.add(ar_coating, Direction.TRANSMISSION)
        return visir_dich.as_transmissive_element()

    @classmethod
    def schmidt_plate_001(cls):
        '''
        Schmidt plate: we assume an Infrasil window of thickness = 1 mm.
        '''
        te = GlassesTransmissiveElementsCatalog.infrasil_1mm_001()
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
    def schmidt_plate_003(cls):
        '''
        Correcting plate (CPM). The element is composed by:
            - 1 surface with broadband (0.6-2.4 um) AR coating
            - 85 mm of Ohara quartz SK-1300 substrate
            - 1 surface with broadband (0.6-2.4 um) AR coating        
        '''
        ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_broadband_001()
        substrate = GlassesTransmissiveElementsCatalog.ohara_quartz_SK1300_85mm_internal_001()
        
        cpm = TransmissiveSystem()
        cpm.add(ar_coating, Direction.TRANSMISSION)
        cpm.add(substrate, Direction.TRANSMISSION)
        cpm.add(ar_coating, Direction.TRANSMISSION)
        return cpm.as_transmissive_element()
    
    @classmethod
    def lgso_lens_001(cls):
        '''
        Lens in the LGS Objective (LGSO). This is a simplified version: one
        surface with narrowband (589 nm) AR coating.
        '''
        te = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        return te
    
    @classmethod
    def lgso_lens_002(cls):
        '''
        Lens in the LGS Objective (LGSO). This is a simplified version: two
        surfaces, both with narrowband (589 nm) AR coating.
        '''
        sup1 = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        sup2 = sup1
        lgso_l = TransmissiveSystem()
        lgso_l.add(sup1, Direction.TRANSMISSION)
        lgso_l.add(sup2, Direction.TRANSMISSION)
        return lgso_l.as_transmissive_element()
    
    @classmethod
    def lgso_lens1_001(cls):
        '''
        First lens in the LGS Objective (LGSO), composed by (ref.
        E-MAO-SF0-INA-DER-001_02):
            - 1 surface with narrowband AR coating (589 nm)
            - Fused silica substrate of thickness = 108 mm (we assume here
                a Suprasil3002 slab)
            - 1 surface with narrowband AR coating (589 nm)
        '''
        sup1 = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        sup2 = GlassesTransmissiveElementsCatalog.suprasil3002_108mm_internal_001()
        sup3 = sup1
        lgso_l = TransmissiveSystem()
        lgso_l.add(sup1, Direction.TRANSMISSION)
        lgso_l.add(sup2, Direction.TRANSMISSION)
        lgso_l.add(sup3, Direction.TRANSMISSION)
        return lgso_l.as_transmissive_element()
    
    @classmethod
    def lgso_lens2_001(cls):
        '''
        Second lens in the LGS Objective (LGSO), composed by (ref.
        E-MAO-SF0-INA-DER-001_02):
            - 1 surface with narrowband AR coating (589 nm)
            - Fused silica substrate of thickness = 70 mm (we assume here
                a Suprasil3002 slab)
            - 1 surface with narrowband AR coating (589 nm)
        '''
        sup1 = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        sup2 = GlassesTransmissiveElementsCatalog.suprasil3002_70mm_internal_001()
        sup3 = sup1
        lgso_l = TransmissiveSystem()
        lgso_l.add(sup1, Direction.TRANSMISSION)
        lgso_l.add(sup2, Direction.TRANSMISSION)
        lgso_l.add(sup3, Direction.TRANSMISSION)
        return lgso_l.as_transmissive_element()
    
    @classmethod
    def lgso_lens3_001(cls):
        '''
        Third lens in the LGS Objective (LGSO), composed by (ref.
        E-MAO-SF0-INA-DER-001_02):
            - 1 surface with narrowband AR coating (589 nm)
            - Fused silica substrate of thickness = 40 mm (we assume here
                a Suprasil3002 slab)
            - 1 surface with narrowband AR coating (589 nm)
        '''
        sup1 = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        sup2 = GlassesTransmissiveElementsCatalog.suprasil3002_40mm_internal_001()
        sup3 = sup1
        lgso_l = TransmissiveSystem()
        lgso_l.add(sup1, Direction.TRANSMISSION)
        lgso_l.add(sup2, Direction.TRANSMISSION)
        lgso_l.add(sup3, Direction.TRANSMISSION)
        return lgso_l.as_transmissive_element()
    
    @classmethod
    def lgso_lens4_001(cls):
        '''
        Fourth lens in the LGS Objective (LGSO), composed by (ref.
        E-MAO-SF0-INA-DER-001_02):
            - 1 surface with narrowband AR coating (589 nm)
            - Fused silica substrate of thickness = 60 mm (we assume here
                a Suprasil3002 slab)
            - 1 surface with narrowband AR coating (589 nm)
        '''
        sup1 = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
        sup2 = GlassesTransmissiveElementsCatalog.suprasil3002_60mm_internal_001()
        sup3 = sup1
        lgso_l = TransmissiveSystem()
        lgso_l.add(sup1, Direction.TRANSMISSION)
        lgso_l.add(sup2, Direction.TRANSMISSION)
        lgso_l.add(sup3, Direction.TRANSMISSION)
        return lgso_l.as_transmissive_element()
    
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
    def lowfs_adc_001(cls):
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
    def lowfs_adc_002(cls):
        '''
        ADC in the LO WFS path. The design is taken from
        E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1 and it
        includes a first prism composed of the following surfaces:
            - SWIR AR coating
            - Schott N-SF2 9.8 mm substrate
            - Schott N-PSK53A 10 mm substrate
            - SWIR AR coating
        and a second prism that is identical and symmetric wrt the first.
        '''
        swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
        substrate1 = GlassesTransmissiveElementsCatalog.schott_NSF2_9dot8_mm_internal_001()
        substrate2 = GlassesTransmissiveElementsCatalog.schott_NPSK53A_10_mm_internal_001()
        
        adc = TransmissiveSystem()
        adc.add(swir_coating, Direction.TRANSMISSION)
        adc.add(substrate1, Direction.TRANSMISSION)
        adc.add(substrate2, Direction.TRANSMISSION)
        adc.add(swir_coating, Direction.TRANSMISSION)
        
        adc.add(swir_coating, Direction.TRANSMISSION)
        adc.add(substrate2, Direction.TRANSMISSION)
        adc.add(substrate1, Direction.TRANSMISSION)
        adc.add(swir_coating, Direction.TRANSMISSION)
        return adc.as_transmissive_element()
    
    @classmethod
    def lowfs_lenslet_001(cls):
        '''
        Lenslet array in the LO WFS path. The design is taken from Aµs and it is
        as follows:
            - AR coating
            - Suprasil 3001 3 mm substrate
            - AR coating
        '''
        ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
        substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
        
        la = TransmissiveSystem()
        la.add(ar_coating, Direction.TRANSMISSION)
        la.add(substrate, Direction.TRANSMISSION)
        la.add(ar_coating, Direction.TRANSMISSION)
        return la.as_transmissive_element()
   
    @classmethod
    def lowfs_collimator_doublet_001(cls):
        '''
        Collimator doublet in the LO WFS path. The design is taken from
        E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1 and it is
        as follows:
            - SWIR AR coating
            - SFPL-51 10 mm substrate
            - PBL-35Y 3 mm substrate
            - SWIR AR coating
        '''
        swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
        substrate1 = GlassesTransmissiveElementsCatalog.ohara_SFPL51_10mm_internal_001()
        substrate2 = GlassesTransmissiveElementsCatalog.ohara_PBL35Y_3mm_internal_001()
        
        lowfs_collimator = TransmissiveSystem()
        lowfs_collimator.add(swir_coating, Direction.TRANSMISSION)
        lowfs_collimator.add(substrate1, Direction.TRANSMISSION)
        lowfs_collimator.add(substrate2, Direction.TRANSMISSION)
        lowfs_collimator.add(swir_coating, Direction.TRANSMISSION)
        return lowfs_collimator.as_transmissive_element()
    
    @classmethod
    def refwfs_collimator_doublet_001(cls):
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
    def refwfs_collimator_doublet_002(cls):
        '''
        Collimator doublet in the R WFS path. The design is taken from 
        E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1 and it is
        as follows:
            - NIR I AR coating
            - Ohara S-FTM16 3 mm substrate
            - CDGM H-QK3L 7 mm substrate
            - NIR I AR coating
         
        '''
        nir_i_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
        substrate1 = GlassesTransmissiveElementsCatalog.ohara_SFTM16_3mm_internal_001()
        substrate2 = GlassesTransmissiveElementsCatalog.cdgm_HQK3L_7mm_internal_001()
        
        rwfs_collimator = TransmissiveSystem()
        rwfs_collimator.add(nir_i_coating, Direction.TRANSMISSION)
        rwfs_collimator.add(substrate1, Direction.TRANSMISSION)
        rwfs_collimator.add(substrate2, Direction.TRANSMISSION)
        rwfs_collimator.add(nir_i_coating, Direction.TRANSMISSION)
        return rwfs_collimator.as_transmissive_element()
    
    @classmethod
    def rwfs_lenslet_001(cls):
        '''
        Lenslet array in the R WFS path. The design is taken from 
        E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1 and it is
        as follows:
            - NIR I AR coating
            - Fused silica 2.15 mm substrate (we use 3 mm of Suprasil3001)
            - NIR I AR coating
        '''
        nir_i_ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
        substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
        
        la = TransmissiveSystem()
        la.add(nir_i_ar_coating, Direction.TRANSMISSION)
        la.add(substrate, Direction.TRANSMISSION)
        la.add(nir_i_ar_coating, Direction.TRANSMISSION)
        return la.as_transmissive_element()
    
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
    def alice_entrance_window_001(cls):
        '''
        Entrance window of ALICE camera. This is a simplified version based on 
        ESO-322090: "the transmission of the window with anti-reflection coating
        applied on both surfaces shall be >97% for an angle of incidence between
        0° and 20°, over wavelength range specified in [REQ-EW-010]."
        
        NOTE: 97% is assumed over the whole range defined in Bandpass.
        '''
        t = Bandpass.one() * 0.97
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
