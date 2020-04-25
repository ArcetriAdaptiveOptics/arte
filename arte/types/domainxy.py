# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u

from arte.utils.help import add_help
from arte.math.make_xy import make_xy
from arte.utils.astropy_utils import get_the_unit_if_it_has_one, \
                                     match_and_remove_units


def _accept_one_or_two_elements(x, name=''):

    errmsg = '%s must be either a scalar or a 2-elements sequence' % name

    # Special case for astropy Quantities, who have __len__
    # but throw exceptions if they are scalars.
    if isinstance(x, u.Quantity) and x.isscalar:
        x = (x, x)

    # Make sure it's a sequence, but not a string, which also has a __len__
    if not hasattr(x, '__len__'):
        x = (x, x)
    elif isinstance(x, str):
        raise TypeError(errmsg)

    if len(x) != 2:
        raise TypeError(errmsg)

    return x


@add_help
class DomainXY():
    '''Holds information about a 2d domain

    For users of the old IDL make_xy, this is the same in class form.
    Several initializers are provided to build a domain from different
    data.

    Domains support astropy units and try to play nicely with them. For
    example, the `shift()` method will shift a domain using the correct unit
    if both the domain and the shift parameter has one. If not, everything
    is converted to unitless, shifted, and then the unit is applied again.
    Shapes and indexes in Python are strictly integer data type and cannot
    use a unit.

    Domains can be compared with == and !=, but no ordering is defined
    between them.

    In addition to the access methods, domain data can be 
    Since the domain sampling is regular, the class internally only stores two
    linear vectors for the X and Y sampling. Everything else is dynamically
    calculated on the fly from these vectors.
    '''

    def __init__(self, xcoord_vector, ycoord_vector):
        self._xcoord = xcoord_vector
        self._ycoord = ycoord_vector
        assert len(self._xcoord.shape) == 1
        assert len(self._ycoord.shape) == 1

    @staticmethod
    def from_xy_vectors(x_vector, y_vector):
        '''Build a domain as a cartesian product of two coordinate vectors'''
        return DomainXY(x_vector, y_vector)

    @staticmethod
    def from_extent(xmin, xmax, ymin, ymax, npoints):
        '''Build a domain from a bounding box'''

        npoints = _accept_one_or_two_elements(npoints, 'npoints')

        x = np.linspace(xmin, xmax, npoints[0])
        y = np.linspace(ymin, ymax, npoints[1])
        return DomainXY(x, y)

    @staticmethod
    def from_shape(shape, pixel_size=1):
        '''Build a domain from a shape and a pixel size'''

        pixel_size = _accept_one_or_two_elements(pixel_size, 'pixel_size')

        tot_size = (shape[0] * pixel_size[0], shape[1] * pixel_size[0])

        # a shape is (rows, cols), so equivalent to (y,x)
        y = np.linspace(-(tot_size[0] - pixel_size[0]) / 2,
                        (tot_size[0] - pixel_size[0]) / 2,
                        shape[0])

        x = np.linspace(-(tot_size[1] - pixel_size[1]) / 2,
                        (tot_size[1] - pixel_size[1]) / 2,
                        shape[1])

        return DomainXY.from_xy_vectors(x, y)

    @staticmethod
    def from_makexy(*args, **kwargs):
        '''Same arguments as make_xy'''
        x, y = make_xy(*args, **kwargs)
        return DomainXY.from_xy_maps(x, y)

    @staticmethod
    def from_linspace(*args, **kwargs):
        '''Cartesian product of two identical np.linspace(). Same arguments'''
        v = np.linspace(*args, **kwargs)
        return DomainXY.from_xy_vectors(v, v)

    @staticmethod
    def from_xy_maps(xmap, ymap):
        '''Build a domain from two 2d maps (like the ones from make_xy)'''
        xcoord_vector = xmap[0, :]
        ycoord_vector = ymap[:, 0]
        return DomainXY(xcoord_vector, ycoord_vector)

    @property
    def shape(self):
        '''(y,x) domain shape'''
        return (self.ycoord.shape[0], self.xcoord.shape[0])

    @property
    def xcoord(self):
        '''X coordinates vector'''
        return self._xcoord

    @property
    def ycoord(self):
        '''Y coordinates vector'''
        return self._ycoord

    @property
    def xmap(self):
        '''X coordinates 2d map'''
        return np.tile(self.xcoord, (self.ycoord.size, 1))

    @property
    def ymap(self):
        '''Y coordinates 2d map'''
        return np.tile(self.ycoord, (self.xcoord.size, 1)).T

    @property
    def radial_map(self):
        '''(r, angle [radians]) 2d map'''
        return (np.hypot(self.ymap, self.xmap),
                np.arctan(self.ymap, self.xmap) * u.rad)

    @property
    def step(self):
        '''(dx,dy) = pixel size'''
        return (self._compute_step_of_uniformly_spaced_vector(self.xcoord),
                self._compute_step_of_uniformly_spaced_vector(self.ycoord))

    def __eq__(self, other):

        xcoord, other_xcoord, _ = match_and_remove_units(self.xcoord,
                                                         other.xcoord)
        ycoord, other_ycoord, _ = match_and_remove_units(self.ycoord,
                                                         other.ycoord)

        return np.allclose(xcoord, other_xcoord) and \
            np.allclose(ycoord, other_ycoord)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def origin(self):
        '''(x,y) = location of 0,0 coordinate, interpolated'''
        x = self._interpolate_for_value(self.xcoord, 0.0)
        y = self._interpolate_for_value(self.ycoord, 0.0)
        return (x, y)

    @property
    def extent(self):
        ''' [xmin, xmax, ymin, ymax] = minimum and maximum coordinates'''
        return [self.xcoord.min(), self.xcoord.max(),
                self.ycoord.min(), self.ycoord.max()]

    @property
    def unit(self):
        '''(x,y) = units used on X and Y, or 1 otherwise'''
        return (get_the_unit_if_it_has_one(self.xcoord),
                get_the_unit_if_it_has_one(self.ycoord))

    def shift(self, dx, dy):
        '''Shift the domain in place'''

        dx, xcoord, xunit = match_and_remove_units(self.xcoord, dx)
        dy, ycoord, yunit = match_and_remove_units(self.ycoord, dy)

        self._xcoord = (xcoord + dx) * xunit
        self._ycoord = (ycoord + dy) * yunit

    def shifted(self, dx, dy):
        '''Returns a new shifted domain'''

        new = self[:]
        new.shift(dx, dy)
        return new

    def get_boundingbox_slice(self, x, y, span=1):
        '''Slice that includes (x,y) with "span" pixels around'''

        _, x, _ = match_and_remove_units(self.xcoord, x)
        _, y, _ = match_and_remove_units(self.ycoord, y)

        xi = np.argmin(np.abs(self.xcoord - x))
        yi = np.argmin(np.abs(self.ycoord - y))
        xlo = max(xi - span, 0)
        ylo = max(yi - span, 0)
        xhi = min(xi + span, self.xcoord.size)
        yhi = min(yi + span, self.ycoord.size)
        return np.s_[ylo:yhi, xlo:xhi]

    def _indices(self, xmin, xmax, ymin, ymax, boundary_check=True):
        '''Returns the outer indices that correspond to a bounding box'''

        # Index calculations are unitless, but first, make sure that
        # all units are the same. If no units have been assigned,
        # all these statements are no-ops.

        xcoord, xmin, xmax, _ = match_and_remove_units(self.xcoord, xmin, xmax)
        ycoord, ymin, ymax, _ = match_and_remove_units(self.ycoord, ymin, ymax)

        xlo = np.argmin(np.abs(xcoord - xmin))
        xhi = np.argmin(np.abs(xcoord - xmax)) + 1
        ylo = np.argmin(np.abs(ycoord - ymin))
        yhi = np.argmin(np.abs(ycoord - ymax)) + 1
        if boundary_check:
            xlo = max(xlo, 0)
            ylo = max(ylo, 0)
            xhi = min(xhi + 1, xcoord.size)
            yhi = min(yhi + 1, ycoord.size)

        return xlo, xhi, ylo, yhi

    def get_crop_slice(self, xmin, xmax, ymin, ymax):
        '''Slice to crop the domain'''
        xlo, xhi, ylo, yhi = self._indices(xmin, xmax, ymin, ymax)

        # Slices are (row,col), that is (y,x)
        return np.s_[ylo:yhi, xlo:xhi]

    def __getitem__(self, idx):
        '''Returns another DomainXY slicing this one'''
        # A single xx slice means (xx, :)
        if type(idx) == type(np.s_[1:2]):
            idx = (idx, np.s_[:])

        xmap = self.xmap[idx]
        ymap = self.ymap[idx]
        return DomainXY.from_xy_maps(xmap, ymap)

    def cropped(self, xmin, xmax, ymin, ymax):
        '''Returns a new cropped DomainXY object'''
        xlo, xhi, ylo, yhi = self._indices(xmin, xmax, ymin, ymax)
        xc = self.xcoord[xlo:xhi]
        yc = self.xcoord[ylo:yhi]
        return DomainXY.from_xy_vectors(xc, yc)

    def boundingbox(self, x, y, span=1):
        '''Returns a new domain around the specified point'''
        slice_ = self.get_boundingbox_slice(x, y, span)
        return self[slice_]

    @staticmethod
    def _interpolate_for_value(arr, value):

        # np.interp removes the unit, so let's remember it

        unit = get_the_unit_if_it_has_one(arr)
        vv = np.interp(value, arr, np.arange(arr.shape[0]),
                       left=np.nan, right=np.nan)
        return vv * unit

    @staticmethod
    def _compute_step_of_uniformly_spaced_vector(vector):
        delta = vector[1:] - vector[:-1]

        # np.isclose() does not like quantities
        d0, p2v = delta[0], delta.ptp()
        if isinstance(p2v, u.Quantity):
            d0, p2v = d0.value, p2v.value
        assert np.isclose(0, p2v, atol=1e-6 * d0)
        return delta[0]
