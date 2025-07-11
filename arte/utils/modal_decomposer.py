from warnings import warn
from arte.utils.zernike_decomposer import ZernikeModalDecomposer


class ModalDecomposer(ZernikeModalDecomposer):
    '''
    Backward compatibility with old ModalDecomposer class
    '''
    def __init__(self, n_modes=None, n_zernike_modes=None):
        if n_zernike_modes is not None:
            n_modes = n_zernike_modes
        if n_modes is None and n_zernike_modes is None:
            raise ValueError("either n_modes or n_zernike_modes must be specified")
        super().__init__(n_modes)
        warn(
            f"{self.__class__.__name__} will be deprecated or change use. Use ZernikeModalDecomposer instead",
            DeprecationWarning,
            stacklevel=2,
        )

