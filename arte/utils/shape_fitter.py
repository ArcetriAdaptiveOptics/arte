from skimage import measure, feature, io, color, draw
from skimage import io, color, measure, draw, img_as_bool
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from pickle import NONE, TRUE

class ShapeFitter(object):
    ''' The class will provide a shape fitter on a binary mask. Currently is 
    only possible to fit a circular shape 
    
    
    pupilfit
    Use RANSAC algoritm on the full mask or on the edge (using Canny as edge detector) to estimate the best skimage.measure.CircleModel approximation
    
    
    pupilfit2
    Use fmin to minimize overlapping between the generated and the template pupil
    
    '''    
    def __init__(self, binary_mask):
        self._mask = binary_mask.copy()
        self._params=None
        self._shape2fit=None
        
    def params(self):
        return self._params

    def fit(self, shape2fit='circle', method='', usecanny=True, **keywords):
        
        if shape2fit=='circle':
            self._shape2fit = measure.CircleModel
        else:
            raise Exception('Other shape but circle not implemented yet')
        
        if method=='':
            self._params = self._fit_correlation(shape2fit=shape2fit, **keywords)
        elif method=='RANSAC':
            self._params = self._fit_ransac(**keywords)
        else:
            raise ValueError('Wrong fitting method specified')
        

    def _fit_ransac(self, **keywords):
        
        img = np.asarray(self._mask.copy(), dtype=float)
        img[img > 0] = 128
        
        edge = img.copy()
        if keywords.pop('usecanny', True):
            edge = feature.canny(img, keywords.pop('sigma',3))

        
        coords = np.column_stack(np.nonzero(edge))
    
        model, inliers = measure.ransac(coords, self._shape2fit,
                                    keywords.pop('min_samples',10), residual_threshold=0.1,
                                    max_trials=1000)
        cx, cy, r = model.params
    
        if keywords.pop('display',False):
            print(r"Cx-Cy {:.2f}-{:.2f}, R {:.2f}".format(cx,cy,r))
            rr, cc = draw.disk((model.params[0], model.params[1]), model.params[2],
                            shape=img.shape)
            img[rr, cc] += 512
            #plt.figure()
            self._dispnobl(img)
            
    
        return model.params
    
    
    def _fit_correlation(self, display=False, **keywords):
    
        img = np.asarray(self._mask.copy(), dtype=int)
    #  img[img > 0] = 128
        regions = measure.regionprops(img)
        bubble = regions[0]
    
        y0, x0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        showfit= keywords.pop('display',False)
        if showfit:
            fign = plt.figure()
    
        if keywords['shape2fit']=='circle': 
            
            def _cost_disk(params):
                x0, y0, r = params
                coords = draw.disk((y0, x0), r, shape=img.shape)
                template = np.zeros_like(img)
                template[coords] = 1
                if showfit:
                    self._dispnobl(template+img, fign)
                return -np.sum(template == img)
        
            params = optimize.fmin(_cost_disk, (x0, y0, r))
            
        else:
            raise Exception('Other shape but circle not implemented yet')
        
       
        return params
        
        
    def _dispnobl(self,img, fign=None, **kwargs):
    
        if fign is not None:
            plt.figure(fign.number)
        plt.clf()
        plt.imshow(img,aspect='auto',**kwargs)
        plt.colorbar()
        plt.draw()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.ioff()