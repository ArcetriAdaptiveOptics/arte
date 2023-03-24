import numpy as np
from arte.types.domainxy import DomainXY
from arte.utils.help import add_help
from arte.utils.radial_profile import computeRadialProfile
from scipy import interpolate


@add_help
class ScalarBidimensionalFunction(object):
    '''
    Represents a scalar function in an XY plane

    The function is initialized with a 2d value and one of the following:
        - a domain over which it is defined
        - two X and Y maps with the domain coordinates
        - (nothing passed) a default domain with the same shape as the value,
          with centered origin and unitary step

    Parameters
    ----------
    values_array: numpy.ndarray
       two-dimensional array with the function value
    xmap: numpy.ndarray, optional
       two-dimensional array with the X coordinate at which each function
       value is sampled.
    ymap: numpy.ndarray, optional
       two-dimensional array with the Y coordinate at which each function
       value is sampled.
    domain: DomainXY instance, optional
       the domain over which the function is sampled.

    Raises
    ------
    ValueError
       if the input parameters do not satisfy the requirements, for example:
           - xmap is passed, but not ymap
           - both xmap/ymap and a domain has been passed
           - values_array is not a 2d array
           - shapes of values and xmap/ymap or domain don't match
    '''

    def __init__(self, values_array, xmap=None, ymap=None, domain=None):

        if len(values_array.shape) != 2:
            raise ValueError('values_array must be 2d')

        if (xmap is not None) or (ymap is not None):
            if (ymap is None) or (xmap is None):
                raise ValueError('xmap and ymap must be specified together')
            if domain is not None:
                raise ValueError('both domain and x/y maps specified')

            domain = DomainXY.from_xy_maps(xmap, ymap)
        else:
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
        if not self.values.shape == self._domain.shape:
            raise ValueError(
                "values shape (%s) don't match domain shape (%s)" %
                (str(self.values.shape), str(self._domain.shape)))

    @property
    def values(self):
        '''2d values array'''
        return self._values

    @property
    def domain(self):
        '''DomainXY instance'''
        return self._domain

    @property
    def shape(self):
        '''(y,x) function shape'''
        return self._values.shape

    def interpolate_in_xy(self, x, y, span=3):
        '''
        Interpolate the function at points x,y

        Returns an array of z[i] corresponding to the values interpolated
        at points x[i], y[i]

        The interpolation algorithm is linear and uses a box of 3x3 points
        centered on the closest point to the interpolation coordinate


        Parameters
        ----------
        x: numpy.ndarray or tuple with size N
            1D array of the x coordinates of the sampling points

        y: numpy.ndarray or tuple with size N
            1D array of the y coordinates of the sampling points

        Returns
        -------
        z: numpy.ndarray
            array of shape (N,) of the sampled values. z[i] = interp(f, x[i], y[i]) 

        '''
        return self._my_interp(x, y, span=span)

    def get_radial_profile(self):
        '''Get the radial profile around the domain origin.

        Assumes that the domain sampling is the same in the X and Y
        directions
        '''
        radial_profile, radial_distance_in_px = computeRadialProfile(
            self._values, self._domain.origin[0], self._domain.origin[1])
        return radial_profile, radial_distance_in_px * self._domain.step[0]

    def plot_radial_profile(self):
        import matplotlib.pyplot as plt
        y, x = self.get_radial_profile()
        plt.plot(x, y)
        plt.show()

    def _my_interp(self, x, y, span=3):
        xs, ys = map(np.array, (np.atleast_1d(x), np.atleast_1d(y)))
        z = np.zeros(xs.shape)
        for i, (x, y) in enumerate(zip(xs, ys)):
            # get the indices of the nearest x,y
            box = self._domain.get_boundingbox_slice(x, y, span=span)

            # make slices of X,Y,Z that are only a few items wide
            nX = self._domain.xmap[box]
            nY = self._domain.ymap[box]
            nZ = self.values[box]

            if np.iscomplexobj(nZ):
                z[i] = self._interp_complex(nX, nY, nZ, x, y)
            else:
                z[i] = self._interp_real(nX, nY, nZ, x, y)
        return z

    def _interp_real(self, nX, nY, nZ, x, y):
        # beware of https://github.com/scipy/scipy/issues/10268 !!!
        # use 1D (nX[0] and nY[:,0]) instead of 2D nX and nY
        intp = interpolate.interp2d(nX[0], nY[:, 0], nZ, kind='linear')
        return intp(x, y)[0]

    def _interp_complex(self, nX, nY, nZ, x, y):
        intpr = interpolate.interp2d(nX[0], nY[:, 0], nZ.real, kind='linear')
        intpi = interpolate.interp2d(nX[0], nY[:, 0], nZ.imag, kind='linear')
        return intpr(x, y)[0] + 1j * intpi(x, y)[0]

    def get_roi(self, xmin, xmax, ymin, ymax):

        box = self._domain.get_crop_slice(xmin, xmax, ymin, ymax)
        cropped_values = self.values[box]
        cropped_domain = self._domain[box]
        return ScalarBidimensionalFunction(cropped_values,
                                           domain=cropped_domain)
