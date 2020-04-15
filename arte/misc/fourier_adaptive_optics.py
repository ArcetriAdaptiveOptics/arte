import numpy as np
from arte.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform as bfft
from arte.types.scalar_bidimensional_function import \
    ScalarBidimensionalFunction as S2DF
from arte.utils.coordinates import xCoordinatesMap



def logShow(image):
    from matplotlib import colors, cm, pyplot as plt

    norm = colors.LogNorm(vmin=image.max() / 1e6, vmax=image.max(), clip='True')
    plt.imshow(image + 1e-16, cmap=cm.gray, norm=norm, origin="lower")


class FourierAdaptiveOptics(object):

    RAD2ARCSEC= (3600 * 180) / np.pi
    ARCSEC2RAD= np.pi / (180 * 3600)

    def __init__(self,
                 pupilDiameterInMeters=8.0,
                 wavelength=1e-6,
                 focalPlaneFieldOfViewInArcsec=1.0,
                 resolutionFactor=2):
        self._pupilDiameterInMeters= pupilDiameterInMeters
        self._wavelength= wavelength
        self._focalPlaneFieldOfViewInArcsec= focalPlaneFieldOfViewInArcsec
        self._resolutionFactor= resolutionFactor

        self._nyquistFocalPlanePixelSizeInArcsec= 0.5* \
            self._wavelength/ self._pupilDiameterInMeters * self.RAD2ARCSEC

        self._focalPlanePixelSizeInArcsec= \
            self._nyquistFocalPlanePixelSizeInArcsec / self._resolutionFactor

        self._computePixelSizes(self._focalPlanePixelSizeInArcsec,
                                self._focalPlaneFieldOfViewInArcsec,
                                self._wavelength)

        self._pupilDiameterInPixels= self._noFocalPlanePixels
        self._phaseMapInMeters= None

        self._focalPlaneCoordinatesInArcsec= \
            self._createFocalPlaneCoordinatesInArcSec()
        self._focalPlaneAngularFreqCoords= \
            self._createFocalPlaneAngularFrequencyCoordinatesInInverseRadians()
        self._pupilPlaneCoordinatesInMeters= \
            self._createPupilPlaneCoordinatesInMeters()
        self._pupilPlaneSpatialFrequencyCoordinatesInInverseMeters= \
            self._createPupilPlaneSpatialFrequencyCoordiantesInInverseMeters()

        self._resetAll()

        self._mask= None
        self._focalLength= 125.
        self._createCircularMask()
        self._createPupilFunction()

        self.setPhaseMapInMeters(self._createFlatPhaseMap())


    def _computePixelSizes(self,
                           focalPlanePixelSizeInArcsec,
                           focalPlaneFieldOfViewInArcsec,
                           wavelenghtInMeters):
        self._noFocalPlanePixels= int(
            focalPlaneFieldOfViewInArcsec / focalPlanePixelSizeInArcsec)

        self._focalPlaneAngularFrequencyPixelSizeInInverseRadians= 1. / (
            self._noFocalPlanePixels * focalPlanePixelSizeInArcsec *
            self.ARCSEC2RAD)

        self._pupilPlaneSpatialFrequencyPizelSizeInInverseMeters= (
            focalPlanePixelSizeInArcsec * self.ARCSEC2RAD /
            wavelenghtInMeters)

        self._pupilPlanePixelSizeInMeters= 1. / (
            self._noFocalPlanePixels *
            self._pupilPlaneSpatialFrequencyPizelSizeInInverseMeters)



    def _resetAll(self):
        self._field= None
        self._pupilFunction= None
        self._amplitudeTransferFunction= None
        self._psf= None
        self._otf= None
        self._stf= None


    def pupilDiameterInMeters(self):
        return self._pupilDiameterInMeters


    def wavelengthInMeters(self):
        return self._wavelength


    def resolutionFactor(self):
        return self._resolutionFactor


    def focalPlaneFieldOfViewInArcsec(self):
        return self._focalPlaneFieldOfViewInArcsec


    def focalPlaneSizeInPixels(self):
        return self._noFocalPlanePixels


    def focalPlanePixelSizeInArcsec(self):
        return self._focalPlanePixelSizeInArcsec


    def focalPlaneAngularFrequencyPixelSizeInInverseRadians(self):
        return self._focalPlaneAngularFrequencyPixelSizeInInverseRadians


    def pupilPlanePixelSizeInMeters(self):
        return self._pupilPlanePixelSizeInMeters


    def focalPlaneAngularFrequencyCoordinatesInInverseRadians(self):
        return self._focalPlaneAngularFreqCoords

    def focalPlaneCoordinatesInArcsec(self):
        return self._focalPlaneCoordinatesInArcsec

    def pupilPlaneCoordinatesInMeters(self):
        return self._pupilPlaneCoordinatesInMeters

    def pupilPlaneSpatialFrequencyPizelSizeInInverseMeters(self):
        return self._pupilPlaneSpatialFrequencyPizelSizeInInverseMeters



    def field(self):
        if self._field is None:
            self._computeField()
        return self._field


    def psf(self):
        if self._psf is None:
            self._createPsf()
        return self._psf


    def otf(self):
        if self._otf is None:
            self._createOtf()
        return self._otf


    def stf(self):
        if self._stf is None:
            self._createStructureFunction()
        return self._stf


    def pupilFunction(self):
        if self._pupilFunction is None:
            self._createPupilFunction()
        return self._pupilFunction


    def amplitudeTransferFunction(self):
        if self._amplitudeTransferFunction is None:
            self._createAmplitudeTransferFunction()
        return self._amplitudeTransferFunction


    def focalPlaneCoordsInArcsec(self):
        return self._focalPlaneCoordinatesInArcsec / 4.848e-6


    def setPhaseMapInMeters(self, phaseMapInMeters):
        self._phaseMapInMeters= phaseMapInMeters
        self._resetAll()


    def _createCircularMask(self):
        from arte.types.mask import CircularMask
        radiusInPx= 0.5 * self._pupilDiameterInMeters / \
            self.pupilPlanePixelSizeInMeters()
        centerInPx= [self._pupilDiameterInPixels / 2,
                     self._pupilDiameterInPixels / 2]
        self._mask= CircularMask(
            (self._pupilDiameterInPixels, self._pupilDiameterInPixels),
            radiusInPx, centerInPx)


    def _createPupilFunction(self):
        xCoord= xCoordinatesMap(
            self._pupilDiameterInPixels,
            self.pupilPlanePixelSizeInMeters())
        self._pupilFunction= S2DF(
            self._mask.asTransmissionValue(), xCoord, xCoord.T)


    def _createFlatPhaseMap(self):
        nPx= self._pupilDiameterInPixels
        pupPxSize= self.pupilPlanePixelSizeInMeters()
        xCoord= xCoordinatesMap(nPx, pupPxSize)
        phase= S2DF(np.ones((nPx, nPx)), xCoord, xCoord.T)
        return phase


    def _computeField(self):
        phaseInRadians= self._phaseMapInMeters.values() / \
            self._wavelength * 2 * np.pi
        amplitude= np.ones_like(phaseInRadians)
        field= amplitude * np.exp(phaseInRadians * 1j)
        self._field= S2DF(
            field* self._mask.asTransmissionValue(),
            self._phaseMapInMeters.xCoord(),
            self._phaseMapInMeters.yCoord())


    def _extendFieldMap(self, fm, howManyTimes):
        fmExt= np.zeros(np.array(fm.shape) * howManyTimes,
                        dtype=np.complex128)
        fmExt[0:fm.shape[0], 0:fm.shape[1]]= fm
        return fmExt


    def _createAmplitudeTransferFunction(self):
        pupF= self.pupilFunction()
        rescaleCoordFact= 1 / (self.wavelengthInMeters() * self._focalLength)
        self._amplitudeTransferFunction= S2DF(
            pupF.values(),
            pupF.xCoord() * rescaleCoordFact,
            pupF.yCoord() * rescaleCoordFact)


    def _createOtf(self):
        ac= self._autoCorrelate(self.amplitudeTransferFunction())
        self._otf= S2DF(ac.values() / ac.values().max(),
                        ac.xCoord(),
                        ac.yCoord())


    def _autoCorrelate(self, scalar2dfunct):
        functFT= bfft.direct(scalar2dfunct)
        aa=S2DF(np.abs(functFT.values()**2),
                functFT.xCoord(),
                functFT.yCoord())
        return bfft.inverse(aa)


    def _createPsf(self):
        psf= bfft.inverse(self.otf())
        rescaleCoordFact= 1 / self._focalLength
        self._psf= S2DF(psf.values(),
                        psf.xCoord() * rescaleCoordFact,
                        psf.yCoord() * rescaleCoordFact)


    def _createStructureFunction(self):
        extFieldMap= self._extendFieldMap(self.field(), self._resolutionFactor)
        ac= self._autoCorrelate(extFieldMap)
        cc= (np.asarray(ac.shape) / 2).astype(np.int)
        self._stf= 2* (ac[cc[0], cc[1]] - ac)


    def _createFocalPlaneCoordinatesInArcSec(self):
        return bfft.distancesXMap(
            self.focalPlaneSizeInPixels(),
            self.focalPlanePixelSizeInArcsec())


    def _createFocalPlaneAngularFrequencyCoordinatesInInverseRadians(self):
        return bfft.frequenciesXMap(
            self.focalPlaneSizeInPixels(),
            self.focalPlanePixelSizeInArcsec() * self.ARCSEC2RAD)


    def _createPupilPlaneCoordinatesInMeters(self):
        return bfft.distancesXMap(
            self.focalPlaneSizeInPixels(),
            self.pupilPlanePixelSizeInMeters())


    def _createPupilPlaneSpatialFrequencyCoordiantesInInverseMeters(self):
        return bfft.frequenciesXMap(
            self.focalPlaneSizeInPixels(),
            self.pupilPlanePixelSizeInMeters())



class TurbulentPhase(object):

    def __init__(self):
        pass

    def correlationFunction(self):
        pass


    def structureFunction(self):
        pass


    def vonKarmanPowerSpectralDensity(self, r0, L0, frequency):
        return 0.0228* (1./ r0)**(5./ 3) * (frequency**2 + 1./ L0)**(-11./ 6)


    def kolmogorovStructureFunction(self, r0, ro):
        return 6.88* (ro/ r0)**(5./ 3)


    def dist(self, npx):
        pass


