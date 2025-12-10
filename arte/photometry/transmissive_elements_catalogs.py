import os
import astropy.units as u
from arte.photometry.transmissive_elements import Bandpass, TransmissiveElement, set_element_id_from_method
from arte.utils.package_data import dataRootDir
from synphot.spectrum import SpectralElement
from arte.photometry.transmittance_calculator import interface_glass_to_glass, \
    internal_transmittance_calculator, internal_transmittance_from_external_one
from synphot.models import Empirical1D
from arte.photometry.filters import Filters
from pathlib import Path

class RestoreTransmissiveElements(object):

    @classmethod
    def transmissive_elements_folder(cls):
        rootDir = str(dataRootDir())
        dirname = Path(rootDir) / 'photometry' / 'transmissive_elements'
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


class CoatingsCatalog():
    
    @classmethod
    def _CoatingsFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'coatings',
            foldername)
        
    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
    def ar_coating_589nm_002(cls):
        '''
        Narrowband (589 nm) simplified AR coating.
        Top-hat width 20nm, peak 0.995
        '''
        t = Bandpass.top_hat(589 * u.nm, 10 * u.nm, 0.995, 0)
        a = Bandpass.zero()
        te = TransmissiveElement(absorptance=a, transmittance=t)
        return te

    @classmethod
    @set_element_id_from_method
    def ar_coating_589nm_003(cls):
        '''
        Narrowband (589 nm) simplified AR coating.
        Top-hat width 40nm, peak 0.995
        '''
        t = Bandpass.top_hat(589 * u.nm, 20 * u.nm, 0.995, 0)
        a = Bandpass.zero()
        te = TransmissiveElement(absorptance=a, transmittance=t)
        return te



    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
    def ar_coating_RI_band_flat(cls):
        '''
        AR coating assumed flat in 0.6-1.0 um with R=1%.
        '''
        t = Bandpass.top_hat_ramped(0.55*u.um, 0.6*u.um, 1.0*u.um, 1.05*u.um, 0., 0.990)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    @set_element_id_from_method
    def ar_coating_RI_band_flat_003(cls):
        '''
        AR coating assumed flat in 0.4-1.1 um with R=1%.
        '''
        t = Bandpass.top_hat_ramped(0.35*u.um, 0.4*u.um, 1.1*u.um, 1.15*u.um, 0., 0.990)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    @set_element_id_from_method
    def thorlabs_ab_broadband_ar_001(cls):
        '''
        Thorlabs AB broadband AR coating 
        '''
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._CoatingsFolder('thorlabs_ab_broadband_ar_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te


class CoatedGlassesCatalog():
    
    @classmethod
    def _CoatedGlassesFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'coated_glasses',
            foldername)
        
    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
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
        

class GlassesCatalog():
    
    @classmethod
    def _GlassesFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'glasses',
            foldername)


    @classmethod
    @set_element_id_from_method
    def sapphire_2mm_internal_001(cls):
        sap_ext_10mm = GlassesCatalog.sapphire_10mm_001()
        sap_ext_5mm = GlassesCatalog.sapphire_5mm_001()
        wv = sap_ext_10mm.waveset
        t_ext_1 = sap_ext_10mm.transmittance(wv)
        t_ext_2 = sap_ext_5mm.transmittance(wv)
        l1 = 10
        l2 = 5
        l3 = 2
        t_int_2 = internal_transmittance_from_external_one(
            t_ext_1, t_ext_2, l1, l2)
        t_int_3 = internal_transmittance_calculator(l2, l3, t_int_2)
        t = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=t_int_3)
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te 
    

    @classmethod
    @set_element_id_from_method
    def sapphire_5mm_001(cls):
        '''
        Sapphire substrate of 5 mm thickness.
        Transmittance is total.
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('sapphire_5mm_001'), u.um
        )
        # Not really true, but we don't use a in this case.
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te


    @classmethod
    @set_element_id_from_method
    def sapphire_10mm_001(cls):
        '''
        Sapphire substrate of 10 mm thickness.
        Transmittance is total.
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('sapphire_10mm_001'), u.nm
        )
        # Not really true, but we don't use a in this case.
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te


    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
    def infrasil_1mm_001(cls):
        '''
        Infrasil window.
        Thickness: 1 mm.
        Transmittance is external.
        
        Data from Thorlabs website.
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('infrasil_1mm_001'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    @set_element_id_from_method
    def suprasil3001_3mm_internal_001(cls):
        '''
        Suprasil 3001 substrate of 3 mm thickness.
        Transmittance is internal.
        Data extrapolated from 10 mm curves.
        '''
        supra10mm = cls.suprasil3002_10mm_internal_001()
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
    def ohara_SBSM2_10_7mm_internal_001(cls):
        '''
        Ohara S-BSM2 substrate of 10.7mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SBSM2_10.7mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
   
    @classmethod
    @set_element_id_from_method
    def ohara_STIM27_3mm_internal_001(cls):
        '''
        Ohara S-TIM27 substrate of 3 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_STIM27_3mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    @set_element_id_from_method
    def ohara_STIM27_3_5mm_internal_001(cls):
        '''
        Ohara S-TIM27 substrate of 3.5 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_STIM27_3.5mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    @set_element_id_from_method
    def ohara_STIM28_3mm_internal_001(cls):
        '''
        Ohara S-TIM28 substrate of 3 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_STIM28_3mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    
    @classmethod
    @set_element_id_from_method
    def ohara_SFPM2_6mm_internal_001(cls):
        '''
        Ohara S-FPM2 substrate of 6 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SFPM2_6mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    @set_element_id_from_method
    def ohara_SFPM2_7mm_internal_001(cls):
        '''
        Ohara S-FPM2 substrate of 7 mm thickness.
        Transmittance is internal.
        Data extrapolated from 6 mm curves.
        '''
        orig=cls.ohara_SFPM2_6mm_internal_001()
        wv = orig.waveset
        t2 = internal_transmittance_calculator(6, 7, orig.transmittance(wv))
        t = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=t2)
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te

        
        
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SFPM2_6mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te



    @classmethod
    @set_element_id_from_method
    def ohara_SLAH96_10_5mm_internal_001(cls):
        '''
        Ohara S-LAH96 substrate of 10.5 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SLAH96_10.5mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te


    @classmethod
    @set_element_id_from_method
    def ohara_SNBH56_5_8mm_internal_001(cls):
        '''
        Ohara S-NBH56 substrate of 5.8 mm thickness.
        Transmittance is internal.
        Data extrapolated from 9.9 mm curves.
        '''
        orig = cls.ohara_SNBH56_9_9mm_internal_001()
        wv = orig.waveset
        t2 = internal_transmittance_calculator(9.9, 5.8, orig.transmittance(wv))
        t = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=t2)
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, transmittance=t)
        return te


    @classmethod
    @set_element_id_from_method
    def ohara_SNBH56_9_9mm_internal_001(cls):
        '''
        Ohara S-NBH56 substrate of 9.9 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('ohara_SNBH56_9.9mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
    def schott_NSF6_4_mm_internal_001(cls):
        '''
        Schott N-SF6 substrate of 4 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('schott_NSF6_4mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    @set_element_id_from_method
    def schott_NBK7_5_mm_internal_001(cls):
        '''
        Schott N-BK7 substrate of 5 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('schott_NBK7_5mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    @set_element_id_from_method
    def schott_NLAK22_6_5_mm_internal_001(cls):
        '''
        Schott N-LAK22 substrate of 6.5 mm thickness.
        Transmittance is internal.
        Data from RefractiveInfo website. 
        '''
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._GlassesFolder('schott_NLAK22_6.5mm_internal_001'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    @set_element_id_from_method
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
    @set_element_id_from_method
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
    @set_element_id_from_method
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


class FiltersCatalog():
    
    @classmethod
    @set_element_id_from_method
    def bessel_H(cls):
        t = Filters.get(Filters.BESSELL_H)
        # a = Bandpass.zero()
        # te = TransmissiveElement(transmittance=t, absorptance=a)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
