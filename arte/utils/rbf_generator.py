import numpy as np
from arte.types.mask import CircularMask

TPS_RBF='tps_rbf'
GAUSS_RBF='gauss_rbf'
INV_QUADRATIC='inv_quadratic'
MULTIQUADRIC='multiquadric'
INV_MULTIQUADRIC='inv_multiquadric'



class RBFGenerator(object):

    #documentation for this class
    '''
    This class generates a set of radial basis functions (RBF) from a given set of coordinates.
    The radial basis functions are generated using a given RBF function. The RBF functions are
    defined as follows:
    - TPS_RBF: r * log(r)
    - GAUSS_RBF: exp(-(eps*r)**2)
    - INV_QUADRATIC: 1/(1+(eps*r)**2)
    - MULTIQUADRIC: sqrt(1+(eps*r)**2)
    - INV_MULTIQUADRIC: 1/sqrt(1+(eps*r)**2)

    The RBF functions are computed over a boolean mask that defines the pupil. The mask can be
    a CircularMask object or a scalar value representing the radius of the pupil. In the latter
    case, the mask is generated internally.

    Parameters
    ----------
    pupil: real or `~arte.types.mask.CircularMask`
        If a scalar value, the argument is used as pupil diameter in pixels.
        If a `~arte.types.mask.CircularMask`, the argument is used as mask
        representing the unit disk.
    coords: list of tuples
        List of tuples containing the coordinates of the RBF functions.
    rbfFunction: str
        String representing the RBF function to use. The available functions are:
        - TPS_RBF
        - GAUSS_RBF
        - INV_QUADRATIC
        - MULTIQUADRIC
        - INV_MULTIQUADRIC
    eps: real
        Value of the epsilon parameter for the RBF function.

    Examples
    --------
    Create a set of radial basis functions using a set of coordinates and the TPS_RBF function
    >>> from arte.utils.rbf_generator import RBFGenerator
    >>> from arte.types.mask import CircularMask
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> coords = [(10,10), (20,20), (30,30)]
    >>> mask = CircularMask((128,128), 64, (64,64))
    >>> rbf = RBFGenerator(mask, coords, 'TPS_RBF')
    >>> rbf.generate()
    >>> rbfCube = rbf.getRBFCube()
    >>> plt.imshow(rbfCube[:,:,0])
    '''

    def __init__(self, pupil, coords, rbfFunction="TPS_RBF", eps=1.0):
        
        if isinstance(pupil, CircularMask):
            self._radius = pupil.radius()
            self._shape = pupil.shape()
            self._center = pupil.center()
            self._boolean_mask = pupil.mask()
        else:
            self._radius = pupil / 2
            sz = np.ceil(pupil)
            self._shape = (sz, sz)
            self._center = np.ones(2) * (sz / 2)
            cm = CircularMask(
                self._shape, maskCenter=self._center, maskRadius=self._radius)
            self._boolean_mask = cm.mask()

        self._fcnDict = {'TPS_RBF': 'tps_rbf',
                         'GAUSS_RBF': 'gauss_rbf',
                         'INV_QUADRATIC': 'inv_quadratic',
                         'MULTIQUADRIC': 'multiquadric',
                         'INV_MULTIQUADRIC': 'inv_multiquadric'}
        self._dictCoords = {i: coord for i, coord in enumerate(coords)}
        self._rbfFunction = self._fcnDict[rbfFunction]
        self._eps = eps
        self._nmodes = 0
        self._rbfBase = None
        self._dictCache = {}

    def generate(self):
        self._nmodes = len(self._dictCoords) 
        nPxY = self._boolean_mask.shape[0]
        nPxX = self._boolean_mask.shape[1]

        for i in range(self._nmodes):
            y0,x0 = self._dictCoords[i]
            cc = np.expand_dims((x0,y0), axis=(1, 2))
            Y, X = (np.mgrid[0.5: nPxY + 0.5: 1,
                            0.5: nPxX + 0.5: 1] - cc) / self._radius
            r = np.sqrt(X ** 2 + Y ** 2)
            if i not in self._dictCache:
                #print("add mode %d" % i)
                if self._rbfFunction == 'tps_rbf':
                    cdata = self.tps_rbf(r)
                elif self._rbfFunction == 'gauss_rbf':
                    cdata = self.gauss_rbf(r, self._eps)
                elif self._rbfFunction == 'inv_quadratic':
                    cdata = self.inv_quadratic(r, self._eps)
                elif self._rbfFunction == 'multiquadric':
                    cdata = self.multiquadric(r, self._eps)
                elif self._rbfFunction == 'inv_multiquadric':
                    cdata = self.inv_multiquadric(r, self._eps)
                else:
                    raise ValueError("Invalid RBF function %s" % self._rbfFunction)
            self._dictCache[i] = np.ma.masked_array(data=cdata, mask=self._boolean_mask)
        self._rbfBase = np.rollaxis(np.ma.masked_array([self.getRBF(n) 
                                    for n in range(self._nmodes)]),0,3)

    def getRBF(self, index):
        return self._dictCache[index]

    def getRBFDict(self, index):
        return {i: self._rbfBase[:, :, i] for i in index}

    def getRBFCube(self):
        return self._rbfBase
    
    def tps_rbf(self, r):
        return r * np.log(r**r)

    def gauss_rbf(self, r, eps):
        return np.exp(-(eps*r)**2)

    def inv_quadratic(self, r,eps):
        return 1/(1+(eps*r)**2)

    def multiquadric(self, r,eps):
        return np.sqrt(1+(eps*r)**2)

    def inv_multiquadric(self, r,eps):
        return 1/np.sqrt(1+(eps*r)**2)

    def getModesDict(self, indexVector):
        return self.getRBFDict(indexVector)

    def first_mode(self):
        return 0

