from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog, \
    CoatedGlassesTransmissiveElementsCatalog
from arte.photometry.transmissive_elements import TransmissiveSystem, Direction


def EltTransmissiveSystem():
    # spider = EltTransmissiveElementsCatalog.spider_001()
    m1 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m2 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m3 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m4 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m5 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    ts = TransmissiveSystem()
    # TODO: remove spider in transmission
    # ts.add(spider, Direction.TRANSMISSION)
    ts.add(m1, Direction.REFLECTION)
    ts.add(m2, Direction.REFLECTION)
    ts.add(m3, Direction.REFLECTION)
    ts.add(m4, Direction.REFLECTION)
    ts.add(m5, Direction.REFLECTION)
    return ts


def MorfeoLgsChannelTransmissiveSystem_001():
    '''
    Configuration from Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_001()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_001()
    lgs_lens = MorfeoTransmissiveElementsCatalog.lgso_lens_001()
    fm = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    notch_filter = MorfeoTransmissiveElementsCatalog.notch_filter_001()
    lenslets = MorfeoTransmissiveElementsCatalog.lgso_lens_001() 
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window_001()
    ccd220 = DetectorsTransmissiveElementsCatalog.ccd220_qe_001()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(notch_filter, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lenslets, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(ccd220, Direction.TRANSMISSION)
    
    return ts


def MorfeoLgsChannelTransmissiveSystem_002():
    '''
    Same configuration as Cedric's spreadsheet "background_calc_maory_v12.xls"
    up to notch_filter (included).
    Data from Patrick Rabou are considered for lgs_wfs.
    C-BLUE camera is considered.
      
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_002()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_002()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_lens = MorfeoTransmissiveElementsCatalog.lgso_lens_002()
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs_001()
    c_blue = DetectorsTransmissiveElementsCatalog.c_blue_qe_001()
    
    ts = EltTransmissiveSystem()   
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_wfs, Direction.TRANSMISSION)
    ts.add(c_blue, Direction.TRANSMISSION)
    
    return ts


def MorfeoLgsChannelTransmissiveSystem_003():
    '''
    LGS lenses have been updated with respect to version 002: the fused silica
    substrate is now considered between the AR coatings. The substrate
    thickness for each lens is taken from 'E-MAO-SF0-INA-DER-001_02'.
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_002()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_002()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs_001()
    c_blue = DetectorsTransmissiveElementsCatalog.c_blue_qe_001()
    
    ts = EltTransmissiveSystem()   
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.TRANSMISSION)
    ts.add(lgs_l1, Direction.TRANSMISSION)
    ts.add(lgs_l2, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_l3, Direction.TRANSMISSION)
    ts.add(lgs_l4, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_wfs, Direction.TRANSMISSION)
    ts.add(c_blue, Direction.TRANSMISSION)
    
    return ts


def MorfeoLowOrderChannelTransmissiveSystem_001():
    '''
    Configuration from Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''  
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_001()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_001()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic_001()
    caf2_lens = MorfeoTransmissiveElementsCatalog.CaF2_lens_C_coated_001()
    adc = MorfeoTransmissiveElementsCatalog.adc_coated_001()
    fused_silica_lenslets = CoatedGlassesTransmissiveElementsCatalog.infrasil_1mm_C_coated_001()
    fused_silica_window = MorfeoTransmissiveElementsCatalog.schmidt_plate_001()
    c_red1_filters = DetectorsTransmissiveElementsCatalog.c_red_one_filters_001()
    c_red1 = DetectorsTransmissiveElementsCatalog.c_red_one_qe_001()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.REFLECTION)
    ts.add(caf2_lens, Direction.TRANSMISSION)
    ts.add(adc, Direction.TRANSMISSION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(fused_silica_lenslets, Direction.TRANSMISSION)
    ts.add(fused_silica_window, Direction.TRANSMISSION)
    ts.add(c_red1_filters, Direction.TRANSMISSION)
    ts.add(c_red1, Direction.TRANSMISSION)
    
    return ts


def MorfeoReferenceChannelTransmissiveSystem_001():
    '''
    Configuration from Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_001()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_001()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic_001()
    collimator = MorfeoTransmissiveElementsCatalog.collimator_doublet_coated_001()
    custom_filter = MorfeoTransmissiveElementsCatalog.ref_custom_filter_001()  
    fused_silica_window = MorfeoTransmissiveElementsCatalog.schmidt_plate_001()  
    fused_silica_lenslets = CoatedGlassesTransmissiveElementsCatalog.infrasil_1mm_B_coated_001()
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window_001()
    ccd220 = DetectorsTransmissiveElementsCatalog.ccd220_qe_001()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.TRANSMISSION)
    ts.add(collimator, Direction.TRANSMISSION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(custom_filter, Direction.TRANSMISSION)
    ts.add(fused_silica_window, Direction.TRANSMISSION)
    ts.add(fused_silica_lenslets, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(ccd220, Direction.TRANSMISSION)
    
    return ts
