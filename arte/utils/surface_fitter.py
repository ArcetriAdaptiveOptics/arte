'''
Authors
  - M Xompero: written in 2021
'''
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils import shape_fitter as sf
import numpy as np
from skimage import measure
fac = np.math.factorial


class SurfaceFitter(ZernikeGenerator):
    ''' The class will provide a surface fitter on a arbitrary coordinates
    defined surface. Currently is working with Zernike base or polynomial base.
    More bases implementation (e.g. TPS) are still to be implemented


       Parameters
    ----------
    zz : masked ndarray
        Input image

    Returns
    -------
    SurfaceFitterObject


    Examples
    --------
    see surface_fitter_test.py for usage

    '''

    def __init__(self, zz, **kwargs):
        self._data = zz.data
        self._mask = ~zz.mask
        self._umat = None
        self._coeffs = None
        self._method = None
        self._currentbase = None
        self._mode_list = None
        self._coords = self.fit_circle()

    def coeffs(self):
        return self._coeffs

    def umat(self):
        return self._umat

    def fit_circle(self, method='cog', display=False):
        ''' The fit_circle method will estimate a circle shape parameters
        (x_center,y_center,radius) using the provided mask
        Automatically called for Zernike base surface fitting


           Parameters
        ----------
        method: 'cog' (default)
                estimation using center of gravity of pixels
                'correlation'
                estimation using correlation minimization

        Returns
        -------
        (x_center,y_center,radius) tuple
        '''
        if method == 'cog':
            img = np.asarray(self._mask.copy(), dtype=int)
            regions = measure.regionprops(img)
            bubble = regions[0]
            y0, x0 = bubble.centroid
            r = bubble.major_axis_length / 2.
            return [x0, y0, r]
        elif method == 'correlation':
            obj = sf.ShapeFitter(self._mask)
            obj.fit()
            return obj.params()
        else:
            raise Exception("No other circle fitting implemented yet")
            return None

    def set_base(self, mode_list, base='zernike', ordering='noll'):
        ''' Set surface fitting base


        Parameters
        ----------
        mode_list: array
                   mode list index to fit (NB zernike modes has index >=1)
        base: base to set up.
             'zernike' Noll style is the default. Others not implemented yet
             'poly'    polynomial base

        Returns
        -------
        umat: array Npix x Nmodes
              modal base sampled on self._x,self._y coordinates
        '''

        if base == 'zernike':
            self._umat = self._get_zernike_base(self._x, self._y,
                                                mode_list, ordering=ordering)
        elif base == 'tps':
            pass
        elif base == 'poly':
            self._umat = self._get_poly_base(self._x, self._y, mode_list)
        else:
            raise Exception('Not implemented yet')
        self._mode_list = mode_list
        self._currentbase = base

    def _get_zernike_base(self, x, y, mode_list, ordering='noll'):

        if min(mode_list) == 0:
            raise OSError("Zernike index must be greater or equal to 1")
        # rh = np.sqrt(x ** 2 + y ** 2)
        # th = np.arccos(np.transpose(x * 1. / rh))
        # th = np.where(th < 2. * np.pi, th, 0)
        # th = np.where(x < 0, 2. * np.pi - th, th)
        self._rhoMap = np.sqrt(x ** 2 + y ** 2)
        self._thetaMap = np.arctan2(x, y)

        # if ordering == 'noll':
        #     m, n = self._l2mn_noll(j)
        #     cnorm = np.sqrt(n + 1) if m == 0 else np.sqrt(2.0 * (n + 1))
        # elif ordering == 'ansi':  # da rivedere ordine e normalizzazione
        #     m, n = self._l2mn_ansi(j)
        #     cnorm = 1
        n2 = mode_list.size
        m = x.size
        u = np.zeros([m, n2])
        # for k, j in enumerate([0, 1, 2, 4]):
        for j in range(n2):
            u[:, j] = self._polar(mode_list[j], self._rhoMap, self._thetaMap)
        return u

    def _get_poly_base(self, x, y, mode_list):

        poly_ord = np.ceil(0.5 * (np.sqrt(1 + 8 * mode_list) - 3))
        y_pow = mode_list - (poly_ord * (poly_ord + 1) / 2 + 1)
        x_pow = poly_ord - y_pow

        n2 = mode_list.size
        m = x.size
        u = np.zeros([m, n2])
        # for k, j in enumerate([0, 1, 2, 4]):
        for j in range(n2):
            u[:, j] = x**x_pow[j] * y**y_pow[j]
        return u

    def _l2mn_ansi(self, j):
        n = 0
        while (j > n):
            n += 1
            j -= n

        m = -n + 2 * j

        return m, n

    def _l2mn_noll(self, j):
        """
        Find the [n,m] list giving the radial order n and azimuthal order
        of the Zernike polynomial of Noll index j.

        Parameters:
            j (int): The Noll index for Zernike polynomials

        Returns:
            list: n, m values
        """
        n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
        p = (j - (n * (n + 1)) / 2.)
        k = n % 2
        m = int((p + k) / 2.) * 2 - k

        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s

        return [m, n]

    def fit(self, zlist, xx=None, yy=None, base='zernike', ordering='noll'):
        ''' Performs the surface fitting 


        Parameters
        ----------
        mode_list: array
                   mode list index to fit (NB zernike modes has index >=1)
             base: base to set up.
                   'zernike' Noll style is the default. Others not implemented yet
                   'poly'    polynomial base
             [xx:] x coordinates of the masked array  zz.compressed() provided in 
                   object instance.
             [yy:] y coordinates of the masked array  zz.compressed() provided in 
                   object instance.

        Returns
        -------
        coeffs: array 
                modal coefficients 
        '''

        if (xx is None) or (yy is None):
            y0 = self._coords[0]
            x0 = self._coords[1]
            r = self._coords[2]

            ss = np.shape(self._mask)
            x = (np.arange(ss[0]).astype(float) - x0) / r
            xx = np.transpose(np.tile(x, [ss[1], 1]))
            y = (np.arange(ss[1]).astype(float) - y0) / r
            yy = np.tile(y, [ss[0], 1])

        self._x = xx[self._mask]
        self._y = yy[self._mask]
        if (self._umat is None) or (self._currentbase != base):
            self.set_base(zlist, base=base, ordering=ordering)
        zz = self._data[self._mask]
        B = np.transpose(zz.copy())
        coeffs = (np.linalg.lstsq(self._umat, B, rcond=-1))[0]
        self._coeffs = coeffs
        return
