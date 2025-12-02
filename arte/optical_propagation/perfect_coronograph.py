import numpy as np
from arte.utils.decorator import override
from arte.optical_propagation.abstract_coronograph import Coronograph    

class PerfectCoronograph(Coronograph):
    """ Class to simulate the perfect coronograph 

    Perfect coronograph formula is from equation (1) in:

    Cavarroc, C., Boccaletti, A., Baudoz, P., Fusco, T., and Rouan, D.,
    “Fundamental limitations on Earth-like planet detection with extremely large telescopes”,
    Astronomy and Astrophysics, vol. 447, no. 1, EDP, pp. 397-403, 2006. doi:10.1051/0004-6361:20053916.
    """

    def __init__(self):
        pass

    @override
    def _get_pupil_mask(self, field):
        field_amp = np.abs(field)
        phase = np.angle(field)[field_amp>1e-12]
        phase_var = np.sum((phase-np.mean(phase))**2)/len(phase)
        pupil_mask = field_amp * (np.sqrt(np.exp(-phase_var))\
                                  -np.exp(1j*np.angle(field),dtype=np.complex128)) \
                            * np.exp(-1j*np.angle(field),dtype=np.complex128)
        return pupil_mask
    
    @override
    def _get_focal_plane_mask(self, field):
        return np.ones(field.shape,dtype=bool)
