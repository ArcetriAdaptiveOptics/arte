import numpy as np

"""
From Negro84

NB: Negro84 doesn't use the same ordering of Zernike of Noll76. This
class reorder according to Noll76

BUG: something wrong in computed values z2,z3 when z7 is not null in the full
pupil
"""


class ZernikeProjectionOnSubaperture(object):

    def __init__(self,
                 pupilRadiusInMeter,
                 subapsRadiusInMeter,
                 subapOffAxisRadiusInMeter,
                 subapOffAxisAzimuthInDegrees):
        self._R = pupilRadiusInMeter
        self._r = subapsRadiusInMeter
        self._h = subapOffAxisRadiusInMeter
        self._a = subapOffAxisAzimuthInDegrees
        self._roR = self._r / self._R
        self._hoR = self._h / self._R
        self._S = self._computeMatrix()

    def computeZernikeDecomponsitionOnSubap(self, zernikeCoeffFrom2To11):
        return np.dot(zernikeCoeffFrom2To11, self._S)

    def getProjectionMatrix(self):
        return self._S

    def _computeMatrix(self):
        roR = self._r / self._R
        hoR = self._h / self._R
        ca = np.cos(self._a * np.pi / 180)
        sa = np.sin(self._a * np.pi / 180)
        c2a = np.cos(2 * self._a * np.pi / 180)
        s2a = np.sin(2 * self._a * np.pi / 180)

        z2 = np.array([roR, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        z3 = np.array([0, roR, 0, 0, 0, 0, 0, 0, 0, 0])
        z4 = np.array([2 * np.sqrt(3) * hoR * roR * ca,
                       2 * np.sqrt(3) * hoR * roR * sa,
                       roR**2, 0, 0, 0, 0, 0, 0, 0])
        z6 = np.array([np.sqrt(6) * hoR * roR * ca,
                       -np.sqrt(6) * hoR * roR * sa,
                       0, 0, roR**2, 0, 0, 0, 0, 0])
        z5 = np.array([np.sqrt(6) * hoR * roR * sa,
                       np.sqrt(6) * hoR * roR * ca,
                       0, roR**2, 0, 0, 0, 0, 0, 0])
        z8 = np.array([3 * np.sqrt(2) * roR * (hoR**2 * (1 + 2 * ca**2) + 2. / 3 * roR**2 - 2. / 3),
                       3 * np.sqrt(2) * hoR**2 * roR * s2a,
                       2 * np.sqrt(6) * hoR * roR**2 * ca,
                       2 * np.sqrt(3) * hoR * roR**2 * sa,
                       2 * np.sqrt(3) * hoR * roR**2 * ca,
                       0, roR**3, 0, 0, 0])
        z7 = np.array([3 * np.sqrt(2) * hoR**2 * roR * s2a,
                       3 * np.sqrt(2) * roR * (hoR**2 *
                                               (1 + 2 * sa**2) + 2. / 3 * roR**2 - 2. / 3),
                       2 * np.sqrt(6) * hoR * roR**2 * sa,
                       2 * np.sqrt(3) * hoR * roR**2 * ca,
                       -2 * np.sqrt(3) * hoR * roR**2 * sa,
                       roR**3, 0, 0, 0, 0])
        z10 = np.array([3 * np.sqrt(2) * hoR**2 * roR * c2a,
                        -3 * np.sqrt(2) * hoR**2 * roR * s2a,
                        0,
                        -2 * np.sqrt(3) * hoR * roR**2 * sa,
                        2 * np.sqrt(3) * hoR * roR**2 * ca,
                        0, 0, 0, roR**3, 0])
        z9 = np.array([3 * np.sqrt(2) * hoR**2 * roR * s2a,
                       3 * np.sqrt(2) * hoR**2 * roR * c2a,
                       0,
                       2 * np.sqrt(3) * hoR * roR**2 * ca,
                       2 * np.sqrt(3) * hoR * roR**2 * sa,
                       0, 0, roR**3, 0, 0])
        z11 = np.array([12 * np.sqrt(5) * hoR * roR * (hoR**2 + 2. / 3 * roR**2 - 0.5) * ca,
                        12 * np.sqrt(5) * hoR * roR * (hoR**2 +
                                                       2. / 3 * roR**2 - 0.5) * sa,
                        (4 * hoR**2 + roR**2 - 1) * np.sqrt(15) * roR ** 2,
                        2 * np.sqrt(30) * hoR**2 * roR**2 * s2a,
                        2 * np.sqrt(30) * hoR**2 * roR**2 * c2a,
                        2 * np.sqrt(10) * hoR * roR**3 * sa,
                        2 * np.sqrt(10) * hoR * roR**3 * ca,
                        0, 0, roR**4])
        return np.vstack((z2, z3, z4, z5, z6, z7, z8, z9, z10, z11))
