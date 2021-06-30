from skimage import feature
from skimage import measure, draw
from scipy import optimize
from scipy.optimize import LinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from numpy import float32


class ShapeFitter(object):
    ''' The class will provide a shape fitter on a binary mask. Currently is
    only possible to fit a circular shape by using RANSAC or correlation merit
    functions OR to fit an anular shape by using correlation merit only.
    '''

    def __init__(self, binary_mask):
        self._mask = binary_mask.copy()
        self._params = None
        self._shape_fitted = None
        self._method = None
        self._success = None

    def __repr__(self):
        return "mask shape %s, parameters %f, shapefitted %s, method %s, status %s" % (
            self._mask.shape(), self._params, self._shapefitted, self._method, self._success)

    def parameters(self):
        '''Return parameters estimated form '''
        return self._params

    def mask(self):
        '''Return current mask of ShapeFitter object'''
        return self._mask

    # def fit(self, shape2fit='circle', method='', usecanny=True, **keywords):
    #
    #     if shape2fit == 'circle':
    #         self._shape2fit = measure.CircleModel
    #     else:
    #         raise Exception('Other shape but circle not implemented yet')
    #
    #     if method == '':
    #         self._params = self._fit_correlation(shape2fit=shape2fit, **keywords)
    #     elif method == 'RANSAC':
    #         self._params = self._fit_ransac(**keywords)
    #     else:
    #         raise ValueError('Wrong fitting method specified')

    def fit_circle_ransac(self,
                          apply_canny=True,
                          sigma=3,
                          display=False,
                          **keywords):
        '''Perform a circle fitting on the current mask using RANSAC algorithm

        Parameters
        ----------
            apply_canny: bool, default=True
                apply Canny edge detection before performing the fit.
            sigma: float, default=10
                if apply_canny is True, you can decide the Canny kernel size.
            display: bool, default=False
                it shows the result of the fit.
        '''
        self._shape_fitted = 'circle'
        self._method = 'ransac'
        img = np.asarray(self._mask.copy(), dtype=float)
        img[img > 0] = 128

        edge = img.copy()
        if apply_canny:
            edge = feature.canny(img, sigma)

        coords = np.column_stack(np.nonzero(edge))

        model, inliers = measure.ransac(
            coords, measure.CircleModel,
            keywords.pop('min_samples', 10), residual_threshold=0.01,
            max_trials=1000)
        cx, cy, r = model.params

        if display is True:
            print(r"Cx-Cy {:.2f}-{:.2f}, R {:.2f}".format(cx, cy, r))
            rr, cc = draw.disk((model.params[0], model.params[1]),
                               model.params[2],
                               shape=img.shape)
            img[rr, cc] += 512
            # plt.figure()
            self._dispnobl(img)

        self._params = model.params
        self._success = model.estimate(coords)

    def fit_circle_correlation(self,
                               method='Nelder-Mead',
                               display=False,
                               **keywords):
        '''Perform a circle fitting on the current mask using minimization 
        algorithm  with correlation merit functions.

        Tested with following minimizations methods: 'Nelder-Mead'. Relative 
        precision of 1% reached on synthetic images without noise.

        Parameters
        ----------
            method: string, default='Nelder-Mead'
                from scipy.optimize.minimize.
            display: bool, default=False
                SLOWLY shows the progress of the fit.
            **keywords: dict, optional
                passed to scipy.optimize.minimize
        '''

        self._method = 'correlation ' + method

        img = np.asarray(self._mask.copy(), dtype=int)
        regions = measure.regionprops(img)
        bubble = regions[0]

        x0, y0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        if display:
            fign = plt.figure()

        self._shape_fitted = 'circle'
        self._initial_guess = (x0, y0, r)

        def _cost_disk(params):
            x0, y0, r = params
            coords = draw.disk((x0, y0), r, shape=img.shape)
            template = np.zeros_like(img)
            template[coords] = 1
            if display:
                self._dispnobl(template + img, fign)
            return -np.sum((template > 0) & (img > 0))

        res = optimize.minimize(_cost_disk, self._initial_guess,
                                method=method, **keywords)
        self._params = res.x
        self._success = res.success
        if res.success is False or (method != 'COBYLA' and res.nit == 0):
            raise Exception("Fit circle didn't converge %s" % res)

    def fit_annular_correlation(self,
                                method='Nelder-Mead',
                                display=False,
                                **keywords):
        '''Perform a annular circle fitting on the current mask using 
        minimization algorithm  with correlation merit functions.

        Tested with following minimizations methods: 'Nelder-Mead'. Relative 
        precision of 1% reached on synthetic images without noise.

        Parameters
        ----------
            method: string, default='Nelder-Mead'
                from scipy.optimize.minimize. Choose among 'Nelder-Mead'
            display: bool, default=False
                SLOWLY shows the progress of the fit.
            **keywords: dict, optional
                passed to scipy.optimize.minimize
        '''

        self._method = 'correlation ' + method
        img = np.asarray(self._mask.copy(), dtype=int)
        regions = measure.regionprops(img)
        bubble = regions[0]

        x0, y0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        inr = r / 2
        if display:
            fign = plt.figure()

        self._shape_fitted = 'annulus'
        self._initial_guess = (x0, y0, r, inr)

        def _cost_annular_disk(params):
            x0, y0, r, inr = params
            coords = draw.disk((x0, y0), r, shape=img.shape)
            template = np.zeros_like(img)
            template[coords] = 1

            coords2 = draw.disk((x0, y0), inr, shape=img.shape)
            template2 = np.zeros_like(img)
            template2[coords2] = 1
            template -= template2

            if display:
                self._dispnobl(template + img, fign)

            merit_fcn = np.sum((template - img)**2)

            return np.sqrt(merit_fcn)

        linear_constraint = LinearConstraint(
            np.identity(4, float32), np.zeros(4),
            np.zeros(4) + np.max(img.shape))
        res = optimize.minimize(_cost_annular_disk, self._initial_guess,
                                method=method, constraints=linear_constraint,
                                **keywords)
        self._params = res.x
        self._success = res.success
        if res.success is False or (method != 'COBYLA' and res.nit == 0):
            raise Exception("Fit circle with hole didn't converge %s" % res)

    def _dispnobl(self, img, fign=None, **kwargs):

        if fign is not None:
            plt.figure(fign.number)
        plt.clf()
        plt.imshow(img, aspect='auto', **kwargs)
        plt.colorbar()
        plt.draw()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.ioff()
