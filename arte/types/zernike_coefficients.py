import numpy as np
from arte.types.modal_coefficients import ModalCoefficients


class _DefaultFirstZernikeMode():
    '''
    FIRST_ZERNIKE_MODE descriptor.

    On the class it is the default index of the first Zernike mode (2,
    i.e. tip: piston is not part of the default decomposition). On an
    instance it reads as FIRST_MODE and cannot be set, since mode
    indexes are fixed at construction time.
    '''

    def __init__(self, default):
        self.default = default

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.default
        return obj.FIRST_MODE

    def __set__(self, obj, value):
        raise AttributeError(
            '%s: mode indexes are fixed at construction time; build a new '
            'object with mode_indexes= instead (e.g. mode_indexes=range(1, n+1) '
            'to include piston)' % obj.__class__.__name__)


class ZernikeCoefficients(ModalCoefficients):
    '''
    Coefficients of a Zernike decomposition, labelled by Noll index.

    By default the coefficients are labelled Z2, Z3, ... (piston is not
    part of the default decomposition). Use mode_indexes to give any
    other set of Noll indexes, e.g. mode_indexes=[1, 2, 3] to include
    piston or mode_indexes=[2, 3, 11].
    '''
    FIRST_ZERNIKE_MODE = _DefaultFirstZernikeMode(2)

    def __init__(self, coefficients, counter=0, mode_indexes=None):
        super().__init__(coefficients, counter,
                         first_mode=type(self).FIRST_ZERNIKE_MODE,
                         mode_indexes=mode_indexes)

    def zernikeIndexes(self):
        return self.modeIndexes()

    def getZ(self, zernikeIndexes):
        '''
        Return the coefficient(s) of the given Zernike (Noll) index(es).

        Raises IndexError for indexes not in zernikeIndexes(): with the
        default labels getZ(1) (piston) raises. See ModalCoefficients.getM.
        '''
        return self.getM(zernikeIndexes)

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0, mode_indexes=None):
        return ZernikeCoefficients(np.array(coefficientsAsNumpyArray), counter,
                                   mode_indexes=mode_indexes)

    def _canOperateWith(self, other):
        return isinstance(other, ZernikeCoefficients)
