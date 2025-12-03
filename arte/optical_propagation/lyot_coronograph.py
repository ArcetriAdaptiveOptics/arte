import numpy as np

from arte.types.mask import CircularMask
from arte.utils.decorator import override
from arte.optical_propagation.abstract_coronograph import Coronograph    


class LyotCoronograph(Coronograph):
    """ Class to simulate the Lyot coronograph 
    
    Parameters
    ----------
    referenceLambdaInM: float
        The wavelength at which the coronograph is defined [m].
        This is used to determine the focal plane mask size.

    inFocalStopInLambdaOverD: float
        Inner radius of the focal plane stop, in lambda/D units.

    outFocalStopInLambdaOverD: float (optional)
        Outer radius of the focal plane stop, in lambda/D units.
        By default None, meaning that higher spatial frequencies
        of the PSF are not filtered by a focal plane mask.
    
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
    lyot = LyotCoronograph(referenceLambdaInM=800e-9,
                           nFocalStopInLambdaOverD=2.0
                           outPupilStopInFractionOfPupil=0.95)
    psf = lyot.get_coronographic_psf(input_field, oversampling=4, lambdaInM=1000e-9) 
    """

    def __init__(self,
                referenceLambdaInM:float,
                inFocalStopInLambdaOverD:float,
                outFocalStopInLambdaOverD:float=None,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._iwaInLambdaOverD = inFocalStopInLambdaOverD
        self._owaInLambdaOverD = outFocalStopInLambdaOverD
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
        lambdaInM2Px = self.oversampling
        if self.lambdaInM is not None:
            lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
        iwa = CircularMask(field.shape, maskRadius=self._iwaInLambdaOverD*lambdaInM2Px)
        if self._owaInLambdaOverD is not None:
            owa = CircularMask(field.shape, maskRadius=self._owaInLambdaOverD*lambdaInM2Px)
            focal_mask = np.logical_and(iwa.mask(),owa.asTransmissionValue())
        else:
            focal_mask = iwa.mask()
        return focal_mask
    

class KnifeEdgeCoronograph(Coronograph):
    """ Class to simulate the Lyot coronograph 
    with a knife-edge focal plane mask.

    Parameters
    ----------
    referenceLambdaInM: float
        The wavelength at which the coronograph is defined [m].
        This is used to determine the focal plane mask size.
    
    iwaFocalStopInLambdaOverD: float
        Inner radius of the focal plane stop, in lambda/D units.
    
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
    knife_edge = KnifeEdgeCoronograph(referenceLambdaInM=800e-9,
                                      iwaFocalStopInLambdaOverD=2.0,
                                      outPupilStopInFractionOfPupil=0.95)
    psf = knife_edge.get_coronographic_psf(input_field, oversampling=4, lambdaInM=1000e-9)
    """

    def __init__(self,
                referenceLambdaInM:float,
                iwaFocalStopInLambdaOverD:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._edgeInLambdaOverD = iwaFocalStopInLambdaOverD
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
        lambdaInM2Px = self.oversampling
        if self.lambdaInM is not None:
            lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
        _,X = np.mgrid[0:field.shape[0],0:field.shape[1]]
        edge = (field.shape[1]//2+self._edgeInLambdaOverD*lambdaInM2Px)
        focal_mask = np.ones(field.shape)
        focal_mask[X<=edge] = 0
        return focal_mask

