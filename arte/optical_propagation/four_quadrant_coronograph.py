import numpy as np

from arte.types.mask import CircularMask
from arte.utils.decorator import override
from arte.optical_propagation.abstract_coronograph import Coronograph    


class FourQuadrantCoronograph(Coronograph):
    """ Class to simulate the 4-quadrant coronograph.
    The 4-quadrant coronograph applies a phase shift of
    pi to two opposite quadrants in the focal plane.

    The 4-quadrant coronograph is described in:
    "The four-quadrant phase-mask coronagraph: white light laboratory results with an achromatic device"
    D.  Mawet, P.  Riaud, J.  Baudrand, P.  Baudoz, A.  Boccaletti, O.  Dupuis, D.  Rouan
    A&A 448 (2) 801-808 (2006), DOI: 10.1051/0004-6361:20054158

    Parameters
    ----------
    referenceLambdaInM: float
        The wavelength at which the coronograph is defined [m].
        This is used to determine the focal plane mask size.
    
    outPupilStopInFractionOfPupil: float (optional)
        Outer radius of the pupil plane stop, in ratio of 
        pupil size (1 = full pupil, 0.3 = 30% pupil).
        By default 1, meaning that the pupil is not masked.

    inPupilStopInFractionOfPupil: float (optional)
        Inner radius of the pupil plane stop, in ratio of 
        pupil size (1 = full pupil, 0.3 = 30% pupil).
        By default 0, meaning that the pupil is not masked.

    Example
    -------
    fq = FourQuadrantCoronograph(referenceLambdaInM=800e-9,
                                 outPupilStopInFractionOfPupil=0.95)
    psf = fq.get_coronographic_psf(input_field, oversampling=4, lambdaInM=1000e-9)
    """

    def __init__(self,
                referenceLambdaInM:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    @override
    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = np.logical_and(outStop.asTransmissionValue(),inStop.mask())
        return pupil_mask
    
    @override
    def _get_focal_plane_mask(self, field):
        nx,ny = field.shape
        cx,cy = nx//2,ny//2
        X,Y = np.mgrid[0:nx,0:ny]
        fmask = np.ones(field.shape)*np.pi
        top_left = np.logical_and(X>=cx, Y<cy)
        bottom_right = np.logical_and(X<cx, Y>=cy)
        fmask[top_left] = 0
        fmask[bottom_right] = 0
        if self.lambdaInM is not None:
            fmask *= self._refLambdaInM/self.lambdaInM
        focal_mask = np.exp(1j*fmask)
        return focal_mask
