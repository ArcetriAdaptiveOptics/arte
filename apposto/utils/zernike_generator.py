import numpy as np
from scipy.special.basic import factorial


class ZernikeGenerator(object):

    def __init__(self, nPixelOnDiameter):
        self._nPixel = nPixelOnDiameter
        self._rhoMap, self._thetaMap = self._polar_array(self._nPixel)
        self._dx = None
        self._dy = None
        self._dictCache = {}
        self._dictDxCache = {}
        self._dictDyCache = {}

    def getRadius(self):
        return self._nPixel / 2.

    def _derivativeCoeffX(self, index):
        if (self._dx is None) or (self._dx.shape[0] < index):
            self._dx = self._computeDerivativeCoeffX(index)
        return self._dx[0:index, 0:index]

    def _derivativeCoeffY(self, index):
        if (self._dy is None) or (self._dy.shape[0] < index):
            self._dy = self._computeDerivativeCoeffY(index)
        return self._dy[0:index, 0:index]

    @staticmethod
    def degree(index):
        n = int(0.5 * (np.sqrt(8 * index - 7) - 3)) + 1
        cn = n * (n + 1) / 2 + 1
        if n % 2 == 0:
            m = int(index - cn + 1) // 2 * 2
        else:
            m = int(index - cn) // 2 * 2 + 1
        radialDegree = n
        azimuthalFrequency = m
        return radialDegree, azimuthalFrequency

    def _rnm(self, radialDegree, azimuthalFrequency, rhoArray):
        n = radialDegree
        m = azimuthalFrequency
        rho = rhoArray
        if (n - m) % 2 != 0:
            raise Exception("n-m must be even. Got %d-%d" % (n, m))
        if abs(m) > n:
            raise Exception("The following must be true |m|<=n. Got %d, %d" % 
                            (n, m))
        mask = np.where(rho <= 1, False, True)

        if(n == 0 and m == 0):
            return np.ma.masked_array(data=np.ones(rho.shape), mask=mask)
        rho = np.where(rho < 0, 0, rho)
        Rnm = np.zeros(rho.shape)
        S = (n - abs(m)) // 2
        for s in range(0, S + 1):
            CR = pow(-1, s) * factorial(n - s) / \
                (factorial(s) * factorial(-s + (n + abs(m)) / 2) * 
                 factorial(-s + (n - abs(m)) / 2))
            p = CR * pow(rho, n - 2 * s)
            Rnm = Rnm + p
        return np.ma.masked_array(data=Rnm, mask=mask)

    def _polar(self, index, rhoArray, thetaArray):
        n, m = ZernikeGenerator.degree(index)
        rho = rhoArray
        theta = thetaArray

        Rnm = self._rnm(n, m, rho)
        NC = np.sqrt(2 * (n + 1))
        if m == 0:
            return np.sqrt(0.5) * NC * Rnm
        if index % 2 == 0:
            return NC * Rnm * np.cos(m * theta)
        else:
            return NC * Rnm * np.sin(m * theta)

    def _polar_array(self, nPixel):
        X, Y = np.mgrid[-1 + 1. / nPixel: 1 - 1. / nPixel: nPixel * 1j,
                        -1 + 1. / nPixel: 1 - 1. / nPixel: nPixel * 1j]
        r = np.sqrt(X ** 2 + Y ** 2)
        th = np.arccos(np.transpose(X * 1. / r))
        th = np.where(th < 2. * np.pi, th, 0)
        th = np.where(X < 0, 2. * np.pi - th, th)
        return r, th

    def getZernikeDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getZernike(index)
        return ret

    def getZernike(self, index):
        if index not in list(self._dictCache.keys()):
            self._dictCache[index] = self._polar(index, self._rhoMap,
                                                 self._thetaMap)
        return self._dictCache[index]

    def __getitem__(self, index):
        return self.getZernike(index)

    def _computeDerivativeCoeffX(self, index):
        jmax = index
        G_mat = np.zeros((jmax, jmax))
        for i in range(1, jmax + 1):
            for j in range(1, jmax + 1):
                ni, mi = ZernikeGenerator.degree(i)
                nj, mj = ZernikeGenerator.degree(j)
                if (
                    (
                        (
                            (
                                mi != 0 and mj != 0
                            ) and (
                                (
                                    _isEven(i) and _isEven(j)
                                ) or (
                                    _isOdd(i) and _isOdd(j)
                                )
                            )
                        ) or (
                            (
                                (mi == 0) and _isEven(j)
                            ) or (
                                (mj == 0) and _isEven(i)
                            )
                        )
                    ) and (
                        (mj == mi + 1) or (mj == mi - 1)
                    ) and (
                        j < i
                    )
                ):
                    G_mat[i - 1, j - 1] = np.sqrt((ni + 1) * (nj + 1))
                    if ((mi == 0) or (mj == 0)):
                        G_mat[i - 1, j - 1] *= np.sqrt(2)
        return G_mat

    def _computeDerivativeCoeffY(self, index):
        jmax = index
        G_mat = np.zeros((jmax, jmax))
        for i in range(1, jmax + 1):
            for j in range(1, jmax + 1):
                ni, mi = ZernikeGenerator.degree(i)
                nj, mj = ZernikeGenerator.degree(j)
                if (
                    (
                        (
                            (
                                mi != 0 and mj != 0
                            ) and (
                                (
                                    _isOdd(i) and _isEven(j)
                                ) or (
                                    _isEven(i) and _isOdd(j)
                                )
                            )
                        ) or (
                            (
                                (mi == 0) and _isOdd(j)
                            ) or (
                                (mj == 0) and _isOdd(i)
                            )
                        )
                    ) and (
                        (mj == mi + 1) or (mj == mi - 1)
                    ) and (
                        j < i
                    )
                ):
                    G_mat[i - 1, j - 1] = np.sqrt((ni + 1) * (nj + 1))
                    if ((mi == 0) or (mj == 0)):
                        G_mat[i - 1, j - 1] *= np.sqrt(2)
                    if (
                        (
                            (
                                (mj == mi + 1) and _isOdd(i)
                            ) or (
                                (mj == mi - 1) and _isEven(i)
                            )
                        ) and (
                            mi != 0
                        )
                    ):
                        G_mat[i - 1, j - 1] *= -1

        return G_mat

    def getDerivativeXDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeX(index)
        return ret

    def getDerivativeYDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeY(index)
        return ret

    def getDerivativeX(self, index):
        if index not in list(self._dictDxCache.keys()):
            self._dictDxCache[index] = self._computeDerivativeX(index)
        return self._dictDxCache[index]

    def _computeDerivativeX(self, index):
        coeffX = self._derivativeCoeffX(index)
        dx = self.getZernike(1) * 0.
        for i in range(1, index):
            dx += coeffX[index - 1, i - 1] * self.getZernike(i)
        return dx

    def getDerivativeY(self, index):
        if index not in list(self._dictDyCache.keys()):
            self._dictDyCache[index] = self._computeDerivativeY(index)
        return self._dictDyCache[index]

    def _computeDerivativeY(self, index):
        coeffY = self._derivativeCoeffY(index)
        dy = self.getZernike(1) * 0.
        for i in range(1, index):
            dy += coeffY[index - 1, i - 1] * self.getZernike(i)
        return dy

    @staticmethod
    def radialOrder(j):
        return np.ceil(0.5 * (np.sqrt(8 * j + 1) - 3))


def _isOdd(num):
    return num % 2 != 0


def _isEven(num):
    return num % 2 == 0
