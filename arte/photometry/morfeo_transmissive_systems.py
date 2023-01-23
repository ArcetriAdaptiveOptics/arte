from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog,\
    MorfeoTransmissiveElementsCatalog
from arte.photometry.transmissive_elements import TransmissiveSystem, Direction


def EltTransmissiveSystem():
    spider = EltTransmissiveElementsCatalog.spider()
    m1 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m2 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m3 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m4 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m5 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    ts = TransmissiveSystem()
    ts.add(spider, Direction.TRANSMISSION)
    ts.add(m1, Direction.REFLECTION)
    ts.add(m2, Direction.REFLECTION)
    ts.add(m3, Direction.REFLECTION)
    ts.add(m4, Direction.REFLECTION)
    ts.add(m5, Direction.REFLECTION)
    return ts


def MorfeoLgsChannelTransmissiveSystem_001():
    '''
    From Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''
    ts = EltTransmissiveSystem()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    lgs_lens = MorfeoTransmissiveElementsCatalog.lgs_lens()
    fm = EltTransmissiveElementsCatalog.ag_mirror_elt()
    notch_filter = MorfeoTransmissiveElementsCatalog.notch_filter()
    lenslets = MorfeoTransmissiveElementsCatalog.lgs_lens() 
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window()
    ccd220 = MorfeoTransmissiveElementsCatalog.ccd220_qe()
    
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
    Data on LGS WFS are from Patrick Rabiou.
    '''
    ts = EltTransmissiveSystem()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    fm = EltTransmissiveElementsCatalog.ag_mirror_elt()
    lgs_lens = MorfeoTransmissiveElementsCatalog.lgs_lens()
    notch_filter = MorfeoTransmissiveElementsCatalog.notch_filter()
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs()
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window()
    ccd220 = MorfeoTransmissiveElementsCatalog.ccd220_qe()
    
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
    ts.add(lgs_wfs, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    #TODO: remove ccd220 and insert c-blue camera data
    ts.add(ccd220, Direction.TRANSMISSION)
    
    return ts

def MorfeoLowOrderChannelTransmissiveSystem_001():
    '''
    From Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''
    ts = EltTransmissiveSystem()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic()
    caf2_lens = MorfeoTransmissiveElementsCatalog.CaF2_lens_C_coated()
    adc = MorfeoTransmissiveElementsCatalog.adc_coated()
    fused_silica_lenslets = MorfeoTransmissiveElementsCatalog.infrasil_1mm_C_coated()
    fused_silica_window = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    c_red1_filters = MorfeoTransmissiveElementsCatalog.c_red_one_filters()
    c_red1 = MorfeoTransmissiveElementsCatalog.c_red_one_qe()
    
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
    From Cedric's spreadsheet: "background_calc_maory_v12.xls"
    '''
    ts = EltTransmissiveSystem()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic()
    collimator = MorfeoTransmissiveElementsCatalog.collimator_doublet_coated()
    custom_filter = MorfeoTransmissiveElementsCatalog.ref_custom_filter()  
    fused_silica_window = MorfeoTransmissiveElementsCatalog.infrasil_1mm()   
    fused_silica_lenslets = MorfeoTransmissiveElementsCatalog.infrasil_1mm_B_coated()
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window()
    ccd220 = MorfeoTransmissiveElementsCatalog.ccd220_qe()
    
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