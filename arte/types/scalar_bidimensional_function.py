import numpy as np
from arte.types.domainxy import DomainXY
from arte.utils.radial_profile import computeRadialProfile
from scipy import interpolate


__version__= "$Id: $"


class ScalarBidimensionalFunction(object):
    '''
    Represents a scalar function in an XY plane
    '''
    def __init__(self, values_array, domain=None):

        if not domain:
            domain = DomainXY.from_shape(values_array.shape, 1)

        self._values = values_array
        self._domain = domain
        self._check_passed_arrays()
        
        self.xcoord = self._domain.xcoord
        self.ycoord = self._domain.ycoord
        self.xmap = self._domain.xmap
        self.ymap = self._domain.ymap
        self.extent = self._domain.extent
        self.origin = self._domain.origin

    def _check_passed_arrays(self):
        self._check_shapes()

    def _check_shapes(self):
        assert self.values.shape == self._domain.shape

    @property
    def values(self):
        return self._values

    @property
    def domain(self):
        return self._domain

    @property
    def shape(self):
        return self._values.shape

    def interpolate_in_xy(self, x, y):
        return self._my_interp(x, y, span=3)

    def get_radial_profile(self):
        radial_profile, radial_distance_in_px = computeRadialProfile(
            self._values, self._domain.origin[0], self._domain.origin[1])
        return radial_profile, radial_distance_in_px * self._domain.step[0]

    def plot_radial_profile(self):
        import matplotlib.pyplot as plt
        y, x= self.get_radial_profile()
        plt.plot(x, y)
        plt.show()

    def _my_interp(self, x, y, span=3):
        xs, ys = map(np.array, (x, y))
        z = np.zeros(xs.shape)
        for i, (x, y) in enumerate(zip(xs, ys)):
            # get the indices of the nearest x,y
            box = self._domain.get_boundingbox_slice(x,y,span=span)

            print(i,x,y,self._domain.xmap, self._domain.ymap, box)
            # make slices of X,Y,Z that are only a few items wide
            nX = self._domain.xmap[box]
            nY = self._domain.ymap[box]
            nZ = self.values[box]

            if np.iscomplexobj(nZ):
                z[i]= self._interp_complex(nX, nY, nZ, x, y)
            else:
                z[i]= self._interp_real(nX, nY, nZ, x, y)
        return z

    def _interp_real(self, nX, nY, nZ, x, y):
        intp = interpolate.interp2d(nX, nY, nZ)
        return intp(x, y)[0]

    def _interp_complex(self, nX, nY, nZ, x, y):
        intpr = interpolate.interp2d(nX, nY, nZ.real)
        intpi = interpolate.interp2d(nX, nY, nZ.imag)
        return intpr(x, y)[0] + 1j * intpi(x, y)[0]

    def get_roi(self, xmin, xmax, ymin, ymax):

        box = self._domain.get_crop_slice(xmin, xmax, ymin, ymax)
        nX = self._domain.xmap[box]
        nY = self._domain.ymap[box]
        nZ = self.values[box]
        return ScalarBidimensionalFunction(nZ, domain=DomainXY.from_xy_maps(nX, nY))
