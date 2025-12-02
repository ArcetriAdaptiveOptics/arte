import numpy as np

from arte.types.mask import CircularMask
from arte.utils.decorator import override
from arte.optical_propagation.abstract_coronograph import Coronograph    


class VortexCoronograph(Coronograph):
    """ Class to simulate the vortex coronograph.
    The vortex coronograph applies an azimuthal phase ramp
    in the focal plane. An optional inner vortex phase ramp
    can be added to mitigate chromaticity effects close to the optical axis,
    resulting in a double vortex coronograph.

    The reference for the double vortex coronograph is:

    Desai, Niyati & Mawet, Dimitri & Bertrou-Cantou, Arielle & Kraus, 
    Matthias & Deparnay, Arnaud & Serabyn, E. & Ruane, Garreth & Redmond, Susan. (2024). 
    "Prototype development of broadband scalar vortex coronagraphs with phase dimples for exoplanet imaging". 
    73. 10.1117/12.3020702. 

    Parameters
    ----------
    referenceLambdaInM: float
        The wavelength at which the coronograph is defined [m].
        This is used to determine the focal plane mask size.

    charge: int
        The number of cycles [0,2*pi) of the vortex phase ramp.
    
    outPupilStopInFractionOfPupil: float (optional)
        Outer radius of the pupil plane stop, in ratio of 
        pupil size (1 = full pupil, 0.3 = 30% pupil).
        By default 1, meaning that the pupil is not masked.

    inPupilStopInFractionOfPupil: float (optional)
        Inner radius of the pupil plane stop, in ratio of 
        pupil size (1 = full pupil, 0.3 = 30% pupil).
        By default 0, meaning that the pupil is not masked.

    addInVortex: bool (optional)
        Whether to add an inner vortex phase ramp to mitigate chromaticity
        effects close to the optical axis. Default is False.
    
    inVortexRadInLambdaOverD: float (optional)
        Inner radius of the inner vortex phase ramp, in lambda/D units.
        Default is 0.62 lambda/D.

    inVortexCharge: int (optional)
        Charge of the inner vortex phase ramp. Default is equal to the outer vortex charge.

    inVortexShift: float (optional)
        Phase shift applied to the inner vortex phase ramp, in radians.
        Default is pi radians.

    Example
    -------
    vortex = VortexcCoronograph(referenceLambdaInM=800e-9,
                                    charge=6,
                                    outPupilStopInFractionOfPupil=0.95)
    psf = vortex.get_coronographic_psf(input_field, oversampling=4, lambdaInM=1000e-9)

    double_vortex = VortexcCoronograph(referenceLambdaInM=800e-9,
                                    charge=4,
                                    outPupilStopInFractionOfPupil=0.95,
                                    addInVortex=True)
    dv_psf = double_vortex.get_coronographic_psf(input_field, oversampling=4, lambdaInM=1000e-9)
    """

    def __init__(self,
                referenceLambdaInM:float,
                charge:int,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0,
                addInVortex:bool=False,
                inVortexRadInLambdaOverD:float=None,
                inVortexCharge:int=None,
                inVortexShift:float=None):
        self._refLambdaInM = referenceLambdaInM
        self._charge = charge
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil
        self._inVortex = addInVortex
        if addInVortex:
            self._innerRadInLambdaOverD = 0.62 if inVortexRadInLambdaOverD is None else inVortexRadInLambdaOverD
            self._innerCharge = charge if inVortexCharge is None else inVortexCharge
            self._innerShift = np.pi if inVortexShift is None else inVortexShift

    @override
    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = np.logical_and(outStop.asTransmissionValue(),inStop.mask())
        return pupil_mask
    
    @override
    def _get_focal_plane_mask(self, field):
        nx, ny = field.shape
        cx, cy = nx // 2, ny // 2
        X, Y = np.mgrid[0:nx, 0:ny]   
        theta = np.arctan2((X - cx), (Y - cy))
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        vortex = self._charge * theta
        if self._inVortex is True:
            rho = np.sqrt((X-cx)**2+(Y-cy)**2)
            lambdaInM2Px = self.oversampling
            if self.lambdaInM is not None:
                lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
            inTheta = np.arctan2((X - cx), (Y - cy))
            inTheta = (inTheta + 2 * np.pi) % (2 * np.pi)
            inVortex = self._innerCharge * inTheta + self._innerShift
            inRho = self._innerRadInLambdaOverD * lambdaInM2Px
            vortex[rho<=inRho] = inVortex[rho<=inRho]
        if self.lambdaInM is not None:
            vortex *= self._refLambdaInM/self.lambdaInM
        focal_mask = np.exp(1j*vortex)
        return focal_mask

