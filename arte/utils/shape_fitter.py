from skimage import feature
from skimage import measure, draw
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


class ShapeFitter(object):
    ''' The class will provide a shape fitter on a binary mask. Currently is
    only possible to fit a circular shape by using RANSAC or correlation merit
    functions
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

    def fit_circle_ransac(self, **keywords):
        '''Perform a circle fitting on the current mask using RANSAC algorithm

        Parameters
        ----------
            applyCanny: bool
                apply Canny edge detection before perform the fit. Default is True.
            sigma: if applyCanny is True, you can decide the Canny kernel size. Default is 3.
            display: it shows the result of the fit.
        '''
        self._shape_fitted = 'circle'
        self._method = 'ransac'
        img = np.asarray(self._mask.copy(), dtype=float)
        img[img > 0] = 128

        edge = img.copy()
        if keywords.pop('applyCanny', True):
            edge = feature.canny(img, keywords.pop('sigma', 3))

        coords = np.column_stack(np.nonzero(edge))

        model, inliers = measure.ransac(
            coords, measure.CircleModel,
            keywords.pop('min_samples', 10), residual_threshold=0.1,
            max_trials=1000)
        cx, cy, r = model.params

        if keywords.pop('display', False):
            print(r"Cx-Cy {:.2f}-{:.2f}, R {:.2f}".format(cx, cy, r))
            rr, cc = draw.disk((model.params[0], model.params[1]), model.params[2],
                               shape=img.shape)
            img[rr, cc] += 512
            # plt.figure()
            self._dispnobl(img)

        self._params = model.params
        self._success = model.estimate(coords)
        return

    def fit_circle_correlation(self, display=False, **keywords):
        '''Perform a circle fitting on the current mask using minimization algorithm with simple correlation merit functions.
        Tested with following minimizations methods: 'Nelder-Mead', 'Powell', 'CG', 'BFGS','L-BFGS-B',
              'TNC','COBYLA','SLSQP','trust-constr'

        Parameters
        ----------
            display: SLOWLY shows the progress of the fit.
            **keywords  (in particular 'method') inherited from
            scipy.optimize.minimize

        Note
        ----------
            for UNKNOW TO ME reason currently the methods 'CG', 'BFGS','L-BFGS-B
            ','TNC','SLSQP','trust-constr' give SWAPPED x and y coordinates
            center and a poor precision on radius of <3pix against <0.01 of the
            others methods.
        '''

        self._shape_fitted = 'circle'
        method = keywords.pop('method', 'Nelder-Mead')
        self._method = 'correlation ' + method
        img = np.asarray(self._mask.copy(), dtype=int)
        regions = measure.regionprops(img)
        bubble = regions[0]

        y0, x0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        showfit = keywords.pop('display', False)
        if showfit:
            fign = plt.figure()

        def _cost_disk(params):
            x0, y0, r = params
            coords = draw.disk((x0, y0), r, shape=img.shape)
            template = np.zeros_like(img)
            template[coords] = 1
            if showfit:
                self._dispnobl(template + img, fign)
            return -np.sum(template == img)

        res = optimize.minimize(_cost_disk, (x0, y0, r),
                                method=method, **keywords)
        print(res.x)
        self._params = res.x
        self._success = res.success
        # self._params = optimize.fmin(_cost_disk, (x0, y0, r))
        # self._success = -1
        return

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
