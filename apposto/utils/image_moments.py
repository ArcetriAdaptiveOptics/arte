import numpy as np
from scipy.special import binom

__version__ = "$Id: image_moments.py 43 2016-01-05 11:22:52Z lbusoni $"


class ImageMoments():

    def __init__(self, image, shiftPixels=0.5):
        self._image= image
        self._shiftPixels= shiftPixels
        nrows, ncols = self._image.shape
        self.y, self.x = np.mgrid[:nrows, :ncols] + self._shiftPixels
        self._m00= self._image.sum()


    def rawMoment(self, iord, jord):
        data = self._image * self.x**iord * self.y**jord
        return data.sum()


    def centralNormalizedMoment(self, iord, jord):
        return self.centralMoment(iord, jord) / self._m00


    # TODO: (lb) 20160104 make it faster if needed, it is easy!
    def centralMoment(self, iord, jord):
        if iord == 0 and jord == 0:
            return self._m00
        if (iord == 0 and jord == 1) or (iord == 1 and jord == 0):
            return 0.
        moment=0.
        x_bar= self.xBar()
        y_bar= self.yBar()
        for m in np.arange(iord+1):
            for n in np.arange(jord+1):
                moment += binom(iord, m) * binom(jord, n) * \
                    (-x_bar)**(iord-m) * (-y_bar)**(jord-n) * \
                    self.rawMoment(m, n)
        return moment


    def xBar(self):
        return self.rawMoment(1, 0) / self.rawMoment(0, 0)


    def yBar(self):
        return self.rawMoment(0, 1) / self.rawMoment(0, 0)


    def centroid(self):
        return np.array([self.xBar(), self.yBar()])


    def covarianceMatrix(self):
        u11= self.centralNormalizedMoment(1, 1)
        u20= self.centralNormalizedMoment(2, 0)
        u02= self.centralNormalizedMoment(0, 2)
        cov = np.array([[u20, u11], [u11, u02]])
        return cov


    def eigenvalues(self):
        u11= self.centralNormalizedMoment(1, 1)
        u20= self.centralNormalizedMoment(2, 0)
        u02= self.centralNormalizedMoment(0, 2)
        aa= 0.5* (u20 + u02)
        bb= 0.5* np.sqrt(4* u11** 2 + (u20- u02)** 2)
        return np.array([aa+ bb, aa- bb])


    def eccentricity(self):
        eigen= self.eigenvalues()
        return np.sqrt(1- eigen[1]/ eigen[0])


    def orientation(self):
        u11= self.centralNormalizedMoment(1, 1)
        u20= self.centralNormalizedMoment(2, 0)
        u02= self.centralNormalizedMoment(0, 2)
        return 0.5 * np.arctan2(2* u11, u20- u02)


    def semiAxes(self):
        return 2* np.sqrt(self.eigenvalues())


    def principalMomentsOfInertia(self):
        return np.array([self.centralNormalizedMoment(2, 0),
                         self.centralNormalizedMoment(0, 2)])


    def effectiveRadius(self):
        return np.sqrt(np.product(self.semiAxes()))
