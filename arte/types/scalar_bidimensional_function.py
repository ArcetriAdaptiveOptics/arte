import numpy as np
from arte.utils.radial_profile import computeRadialProfile
from scipy import interpolate


__version__= "$Id: $"

   
class DomainXY():
    '''Holds information about a 2d domain'''
    
    def __init__(self, xcoord_vector, ycoord_vector):
        self._xcoord= xcoord_vector
        self._ycoord= ycoord_vector       
        assert len(self._xcoord.shape) == 1
        assert len(self._ycoord.shape) == 1

    @staticmethod
    def from_vectors(xcoord_vector, ycoord_vector):
        return DomainXY(xcoord_vector, ycoord_vector)

    @staticmethod
    def from_shape(shape, pixel_size=1):

        tot_size_x = shape[0] * pixel_size
        tot_size_y = shape[1] * pixel_size

        x = np.linspace(-(tot_size_x - pixel_size) / 2,
                         (tot_size_x - pixel_size) / 2,
                         shape[0])
        y = np.linspace(-(tot_size_y - pixel_size) / 2,
                         (tot_size_y - pixel_size) / 2,
                         shape[1])
        return DomainXY(x,y)

    @staticmethod
    def from_maps(xmap, ymap):
        xcoord_vector = xmap[0,:]
        ycoord_vector = ymap[:,0]
        return DomainXY(xcoord_vector, ycoord_vector)
    
    @property
    def shape(self):
        return (self.xcoord.shape[0], self.ycoord.shape[0])

    @property
    def xcoord(self):
        return self._xcoord
    
    @property
    def ycoord(self):
        return self._ycoord

    @property
    def xmap(self):
        return np.tile(self.xcoord, (self.ycoord.size,1)).T

    @property
    def ymap(self):
        return np.tile(self.ycoord, (self.xcoord.size,1))

    @property
    def radial_map(self):
        return np.hypot(self.ymap, self.ymap)

    @property
    def step(self):
        return (self._compute_step_of_uniformly_spaced_vector(self.xcoord),
                self._compute_step_of_uniformly_spaced_vector(self.ycoord))

    @property
    def origin(self):
        x = self._interpolate_for_value(self.xcoord, 0.0)
        y = self._interpolate_for_value(self.ycoord, 0.0)
        return (x,y)

    @property
    def extent(self):
        return [self.xcoord.min(), self.xcoord.max(),
                self.ycoord.min(), self.ycoord.max()]
        
    def boundingbox(self, x, y, spn=1):
        xi = np.argmin(np.abs(self.xcoord - x))
        yi = np.argmin(np.abs(self.ycoord - y))
        xlo = max(xi - spn, 0)
        ylo = max(yi - spn, 0)
        xhi = min(xi + spn, self.xcoord.size)
        yhi = min(yi + spn, self.ycoord.size) 
        return np.s_[xlo:xhi, ylo:yhi]

    def crop(self, xmin, xmax, ymin, ymax):
        xmin = np.argmin(np.abs(self.xcoord - xmin))
        xmax = np.argmin(np.abs(self.xcoord - xmax)) + 1
        ymin = np.argmin(np.abs(self.ycoord - ymin))
        ymax = np.argmin(np.abs(self.ycoord - ymax)) + 1
        xlo = max(xmin, 0)
        ylo = max(ymin, 0)
        xhi = min(xmax+ 1, self.xcoord.size)
        yhi = min(ymax+ 1, self.ycoord.size)
        return np.s_[xlo:xhi, ylo:yhi]

    @staticmethod
    def _interpolate_for_value(arr, value):
        return np.interp(value, arr, np.arange(arr.shape[0]),
                         left=np.nan, right=np.nan)

    @staticmethod
    def _compute_step_of_uniformly_spaced_vector(vector):
        delta= vector[1:] - vector[:-1]
        assert np.isclose(0, delta.ptp(), atol=1e-6 * delta[0])
        return delta[0]


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

    def interpolate_in_xy(self, x, y):
        return self._my_interp(x, y, spn=3)

    def get_radial_profile(self):
        radial_profile, radial_distance_in_px = computeRadialProfile(
            self._values, self._domain.origin[0], self._domain.origin[1])
        return radial_profile, radial_distance_in_px * self._domain.step[0]

    def plot_radial_profile(self):
        import matplotlib.pyplot as plt
        y, x= self.get_radial_profile()
        plt.plot(x, y)
        plt.show()

    def _my_interp(self, x, y, spn=3):
        xs, ys = map(np.array, (x, y))
        z = np.zeros(xs.shape)
        for i, (x, y) in enumerate(zip(xs, ys)):
            # get the indices of the nearest x,y
            box = self._domain.boundingbox(x,y,spn=spn)

            print(i,x,y,box)
            # make slices of X,Y,Z that are only a few items wide
            nX = self._domain.xmap[box]
            nY = self._domain.ymap[box]
            nZ = self.values[box]
            
            print(nX)
            print(nY)
            print(nZ)
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

        box = self._domain.crop(xmin, xmax, ymin, ymax)
        nX = self._domain.ymap[box]
        nY = self._domain.ymap[box]
        nZ = self.values[box]
        return ScalarBidimensionalFunction(nZ, domain=DomainXY.from_maps(nX, nY))
