import astropy.units as u
from arte.photometry.morfeo_transmissive_systems import MorfeoLowOrderChannelTransmissiveSystem_003
from arte.photometry.transmissive_elements_catalogs import DetectorsTransmissiveElementsCatalog
from arte.photometry.thermal_flux import ThermalFluxOnWFS
from morfeo.utils.constants import MORFEO


def thermal_flux_on_LO_WFS():
    T = 278.15 * u.K
    lo_wfs_transmissive_system = MorfeoLowOrderChannelTransmissiveSystem_003()
    # remove CRED1 QE from the path
    lo_wfs_transmissive_system.remove(-1)
    # remove cold filters from the path
    lo_wfs_transmissive_system.remove(-1)
    cred1_qe = DetectorsTransmissiveElementsCatalog.c_red_one_qe_001()    
    
    th_fl = ThermalFluxOnWFS(
        temperature_in_K=T,
        emissivity=lo_wfs_transmissive_system.emissivity,
        lenslet_focal_length=MORFEO.lo_wfs_lenslet_focal_length,
        lenslet_size=MORFEO.lo_wfs_pupil_diameter,
        pixel_size=MORFEO.lo_wfs_px_size,
        qe=cred1_qe)
    
    return th_fl 
