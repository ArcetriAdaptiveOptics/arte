import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator


__version__= "$Id: $"


class NoisePropagation():

    def __init__(self,
                 nSubaps=3,
                 modesV=np.arange(1, 7),
                 rcond=0.001):
        self._nSubaps= nSubaps
        self._zg= ZernikeGenerator(nSubaps)
        self._modesV= modesV
        self._rcond= rcond
        self._computeForModes()


    def _computeForModes(self):
        self._phi= np.dstack([
            self._zg.getZernike(m).filled(0) for m in self._modesV])
        self._D= np.vstack(
            [np.hstack(
                (self._zg.getDerivativeX(m).flatten(),
                 self._zg.getDerivativeY(m).flatten()))
             for m in self._modesV]).T
        self._u, self._s, self._vh= np.linalg.svd(
            self._D, full_matrices=False)
        self._sinv= 1/ self._s
        self._sinv[np.argwhere(self._s< self._rcond * self._s.max())]=0
        self._R=np.matmul(self._vh.T,
                          np.matmul(np.diag(self._sinv), self._u.T))


    @property
    def nSubaps(self):
        return self._nSubaps


    @property
    def nSlopes(self):
        return self._D.shape[0]


    @property
    def modesVector(self):
        return self._modesV


    @property
    def nModes(self):
        return self._D.shape[1]


    @property
    def phaseCube(self):
        return self._phi


    @property
    def R(self):
        return self._R


    @property
    def D(self):
        return self._D


    @property
    def s(self):
        return self._s


    @property
    def sInv(self):
        return self._sinv


    @property
    def u(self):
        return self._u


    @property
    def v(self):
        return self._vh.T


    @property
    def noisePropagationMatrix(self):
        return np.dot(self.R, self.R.T)


    @property
    def sigma(self):
        return np.trace(self.noisePropagationMatrix)


    def noiseSimulation(self, sigma=1, nSamples=1000):
        return np.var(np.dot(
            self.R, sigma * np.random.randn(self.nSlopes, nSamples)), axis=1)


    def slopesMapForMode(self, nMode):
        return self.D[:, nMode].filled(0).reshape(
            self.nSubaps* 2, self.nSubaps)


    def leftSingularVectorMapForMode(self, nMode):
        return self.u[:, nMode].filled(0).reshape(
            self.nSubaps* 2, self.nSubaps)
