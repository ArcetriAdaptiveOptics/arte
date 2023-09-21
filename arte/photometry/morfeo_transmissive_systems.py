from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog, \
    CoatedGlassesTransmissiveElementsCatalog, \
    CoatingsTransmissiveElementsCatalog
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
    notch_filter = MorfeoTransmissiveElementsCatalog.lgs_wfs_notch_filter_001()
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


def MorfeoLgsChannelTransmissiveSystem_004():
    '''
    The Schmidt Plate has been updated with respect to version 002.
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
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


def MorfeoLgsChannelTransmissiveSystem_005():
    '''
    - Protected silver coating data has been updated with ESO shared
      measurements made by Fraunhofer (see E-MAO-SF0-INA-ANR-001 OFDR MORFEO
      Analysis Report.pdf, sec. 3.14.2).   
    - The LGS dichroic has been updated considering the FDR design. In particular,
      the substrate thickness is 80 mm and the coating here considered is the
      "exp min" profile from LMA measurements.
    - The laser-line coating for the LGSO-FMs has now a peak value of 0.995
        instead of the old 0.990.
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_002()
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


def MorfeoLgsChannelTransmissiveSystem_006():
    '''
    - Data on FLI C-BLUE QE have been updated, considering FLI 2023 measurements
    on C-BLUE having a camera window of AR-coated BK7.    
    - Notch filter was missing in the previous versions and it is here included.
    '''
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_002()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs_001()
    notch_filter = MorfeoTransmissiveElementsCatalog.lgs_wfs_notch_filter_003()
    c_blue = DetectorsTransmissiveElementsCatalog.c_blue_qe_003()
    
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
    ts.add(notch_filter, Direction.TRANSMISSION)
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
    adc = MorfeoTransmissiveElementsCatalog.lowfs_adc_001()
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


def MorfeoLowOrderChannelTransmissiveSystem_002():
    '''
    Configuration from E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1
    (design for FDR).
    - Protected silver coating data has been updated with ESO shared measurements
      made by Fraunhofer (see E-MAO-SF0-INA-ANR-001 OFDR MORFEO Analysis Report.pdf,
      sec. 3.14.2).
    '''  
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.materion_average_002()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pickoff_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    fold_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator1 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator2 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pup_steer_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002() 
    visir_dichroic = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
    doublet_collimator = MorfeoTransmissiveElementsCatalog.lowfs_collimator_doublet_001()
    adc = MorfeoTransmissiveElementsCatalog.lowfs_adc_002()
    lenslet_array = MorfeoTransmissiveElementsCatalog.lowfs_lenslet_001()
    # TODO: data for sapphire window
    # sapphire_window = 
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
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)
    ts.add(pickoff_mirror, Direction.REFLECTION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(focus_compensator1, Direction.REFLECTION)
    ts.add(focus_compensator2, Direction.REFLECTION)
    ts.add(pup_steer_mirror, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.REFLECTION)
    ts.add(doublet_collimator, Direction.TRANSMISSION)
    ts.add(adc, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(lenslet_array, Direction.TRANSMISSION)
    # ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(c_red1_filters, Direction.TRANSMISSION)
    ts.add(c_red1, Direction.TRANSMISSION)
    
    return ts


def MorfeoLowOrderChannelTransmissiveSystem_003():
    '''
    Configuration from E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1
    (design for FDR).
    
    The coating of LGS dichroic has been changed with respect to version 002 of
    MorfeoLowOrderChannelTransmissiveSystem. In particular, the "exp min"
    curve from LMA measurements is here considered, assuming an angle of
    incidence of 11.3 deg.    
    '''  
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pickoff_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    fold_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator1 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator2 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pup_steer_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002() 
    visir_dichroic = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
    doublet_collimator = MorfeoTransmissiveElementsCatalog.lowfs_collimator_doublet_001()
    adc = MorfeoTransmissiveElementsCatalog.lowfs_adc_002()
    lenslet_array = MorfeoTransmissiveElementsCatalog.lowfs_lenslet_001()
    # TODO: data for sapphire window
    # sapphire_window = 
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
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)
    ts.add(pickoff_mirror, Direction.REFLECTION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(focus_compensator1, Direction.REFLECTION)
    ts.add(focus_compensator2, Direction.REFLECTION)
    ts.add(pup_steer_mirror, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.REFLECTION)
    ts.add(doublet_collimator, Direction.TRANSMISSION)
    ts.add(adc, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(lenslet_array, Direction.TRANSMISSION)
    # ts.add(sapphire_window, Direction.TRANSMISSION)
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
    collimator = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_001()
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


def MorfeoReferenceChannelTransmissiveSystem_002():
    '''
    Configuration from E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1.
    '''    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_001()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.materion_average_002()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    pickoff_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    fold_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    focus_compensator1 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    focus_compensator2 = EltTransmissiveElementsCatalog.ag_mirror_elt_001()
    pup_steer_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_001() 
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic_002()
    collimator = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_002()
    alice_entrance_window = MorfeoTransmissiveElementsCatalog.alice_entrance_window_001()
    lenslet_array = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    # TODO: stessi dati della sapphire window di FREDA
    # ccd220_sapphire_window = 
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_002()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)
    ts.add(pickoff_mirror, Direction.REFLECTION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(focus_compensator1, Direction.REFLECTION)
    ts.add(focus_compensator2, Direction.REFLECTION)
    ts.add(pup_steer_mirror, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(collimator, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(alice_entrance_window, Direction.TRANSMISSION)
    ts.add(lenslet_array, Direction.TRANSMISSION)
    # ts.add(ccd220_window, Direction.TRANSMISSION)
    ts.add(ccd220_qe, Direction.TRANSMISSION)
    
    return ts


def MorfeoReferenceChannelTransmissiveSystem_003():
    '''
    Configuration from E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1
    (design for FDR).
    - Protected silver coating data has been updated with ESO shared measurements
      made by Fraunhofer (see E-MAO-SF0-INA-ANR-001 OFDR MORFEO Analysis Report.pdf,
      sec. 3.14.2).
    - For what concerns the CCD220 QE, we consider here the minimum QE required
      in ESO-287869_2 ALICE Camera Technical Requirements Specifications
      ([REQ-ALI-039]).
    '''    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.materion_average_002()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pickoff_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    fold_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator1 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator2 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pup_steer_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002() 
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic_002()
    collimator = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_002()
    alice_entrance_window = MorfeoTransmissiveElementsCatalog.alice_entrance_window_001()
    lenslet_array = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    # TODO: stessi dati della sapphire window di FREDA
    # ccd220_sapphire_window = 
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_003()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)
    ts.add(pickoff_mirror, Direction.REFLECTION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(focus_compensator1, Direction.REFLECTION)
    ts.add(focus_compensator2, Direction.REFLECTION)
    ts.add(pup_steer_mirror, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(collimator, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(alice_entrance_window, Direction.TRANSMISSION)
    ts.add(lenslet_array, Direction.TRANSMISSION)
    # ts.add(ccd220_window, Direction.TRANSMISSION)
    ts.add(ccd220_qe, Direction.TRANSMISSION)
    
    return ts


def MorfeoReferenceChannelTransmissiveSystem_004():
    '''
    Configuration from E-MAO-PN0-INA-ANR-001 MAORY LOR WFS Module Analysis Report_3D1
    (design for FDR).
    The coating of LGS dichroic has been changed with respect to version 003 of
    MorfeoReferenceChannelTransmissiveSystem. In particular, the "exp min"
    curve from LMA measurements is here considered, assuming an angle of
    incidence of 11.3 deg.
    '''    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pickoff_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    fold_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator1 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    focus_compensator2 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    pup_steer_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002() 
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic_002()
    collimator = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_002()
    alice_entrance_window = MorfeoTransmissiveElementsCatalog.alice_entrance_window_001()
    lenslet_array = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    # TODO: stessi dati della sapphire window di FREDA
    # ccd220_sapphire_window = 
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_003()
    
    ts = EltTransmissiveSystem()
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)
    ts.add(pickoff_mirror, Direction.REFLECTION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(focus_compensator1, Direction.REFLECTION)
    ts.add(focus_compensator2, Direction.REFLECTION)
    ts.add(pup_steer_mirror, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(collimator, Direction.TRANSMISSION)
    ts.add(fold_mirror, Direction.REFLECTION)
    ts.add(alice_entrance_window, Direction.TRANSMISSION)
    ts.add(lenslet_array, Direction.TRANSMISSION)
    # ts.add(ccd220_window, Direction.TRANSMISSION)
    ts.add(ccd220_qe, Direction.TRANSMISSION)
    
    return ts


def MorfeoMainPathOptics_001():
    '''
    The correcting plate is with Ohara SK-1300 substrate and the LGS dichroic
    coating is the "env_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_env_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)    
    return ts


def MorfeoMainPathOptics_002():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "env_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_env_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)    
    return ts


def MorfeoMainPathOptics_003():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "exp_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)    
    return ts


def MorfeoMainPathOptics_004():
    '''
    The correcting plate is with Ohara SK-1300 substrate and the LGS dichroic
    coating is the "exp_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)    
    return ts


def MorfeoMainPathOptics_005():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "exp_min". DMs are assumed silver-coated.     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    lgs_dichroic = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    m11 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m12 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11, Direction.REFLECTION)
    ts.add(m12, Direction.REFLECTION)    
    return ts


def MorfeoLGSO_001():
    '''
    The correcting plate is with Ohara SK-1300 substrate and the LGS dichroic
    coating is the "env_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_004()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
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
    return ts


def MorfeoLGSO_002():
    '''
    The correcting plate is with Ohara SK-1300 substrate and the LGS dichroic
    coating is the "exp_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
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
    return ts


def MorfeoLGSO_003():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "exp_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
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
    return ts


def MorfeoLGSO_004():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "env_min".     
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_004()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
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
    return ts


def MorfeoLGSO_005():
    '''
    The correcting plate is with Suprasil 3002 substrate and the LGS dichroic
    coating is the "exp_min". DMs are assumed silver-coated.  
    '''
    correcting_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m9 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    m10 = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    fm = MorfeoTransmissiveElementsCatalog.lgso_fm_001()
    lgs_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    lgs_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    lgs_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    lgs_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    
    ts = TransmissiveSystem()   
    ts.add(correcting_plate, Direction.TRANSMISSION)
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
    return ts
