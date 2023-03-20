import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator


class NoisePropagationZernikeGradientWFS():

    def __init__(self,
                 nSubaps=3,
                 modesV=np.arange(2, 7),
                 rcond=0.1):
        self._nSubaps= nSubaps
        self._zg= ZernikeGenerator(nSubaps)
        self._modesV= modesV
        self._rcond= rcond
        self._initialize()
        
    def _initialize(self):
        self._phi = None
        self._D = None 
        self._u = None 
        self._s = None 
        self._vh = None
        self._sinv = None  
        self._R = None 
        

    def _compute_interaction_matrix(self):
        self._phi= np.dstack([
            self._zg.getZernike(m).filled(0) for m in self._modesV])
        self._D= np.vstack(
            [np.hstack(
                (self._zg.getDerivativeX(m).compressed(),
                 self._zg.getDerivativeY(m).compressed()))
             for m in self._modesV]).T

    def _compute_reconstructor(self):
        self._u, self._s, self._vh= np.linalg.svd(
            self.interaction_matrix, full_matrices=False)
        self._sinv= 1/ self._s
        self._sinv[np.argwhere(self._s< self._rcond * self._s.max())]=0
        self._R=np.matmul(self._vh.T,
                          np.matmul(np.diag(self._sinv), self._u.T))

    @property
    def nSubaps(self):
        return self._nSubaps


    @property
    def nSlopes(self):
        return self.interaction_matrix.shape[0]


    @property
    def modesVector(self):
        return self._modesV


    @property
    def nModes(self):
        return self.interaction_matrix.shape[1]


    @property
    def phaseCube(self):
        return self._phi


    @property
    def reconstructor(self):
        if self._R is None:
            self._compute_reconstructor()
        return self._R


    @property
    def interaction_matrix(self):
        if self._D is None:
            self._compute_interaction_matrix()
        return self._D


    @property
    def s(self):
        if self._s is None:
            self._compute_reconstructor()
        return self._s


    @property
    def sInv(self):
        if self._sinv is None:
            self._compute_reconstructor()
        return self._sinv


    @property
    def u(self):
        if self._u is None:
            self._compute_reconstructor()
        return self._u


    @property
    def v(self):
        if self._vh is None:
            self._compute_reconstructor()
        return self._vh.T


    @property
    def noise_covariance_matrix(self):
        return np.dot(self.reconstructor, self.reconstructor.T)


    @property
    def noise_total_variance(self):
        return np.trace(self.noise_covariance_matrix)


    def noiseSimulation(self, sigma=1, nSamples=1000):
        return np.var(np.dot(
            self.reconstructor, sigma * np.random.randn(self.nSlopes, nSamples)), axis=1)


    def _noise_rg(self, radial_order):
        return -2.05 * np.log10(radial_order + 1) - 0.53


    def _noise_rg2(self, radial_order):
        return -2.0 * np.log10(radial_order + 1) - 0.76



    # def slopesMapForMode(self, nMode):
    #     return self.interaction_matrix[:, nMode].filled(0).reshape(
    #         self.nSubaps* 2, self.nSubaps)
    #
    #
    # def leftSingularVectorMapForMode(self, nMode):
    #     return self.u[:, nMode].reshape(
    #         self.nSubaps* 2, self.nSubaps)
