'''
Created on Oct 15, 2018

@author: lbusoni
'''
import numpy as np
from astropy import units as u
from ccdproc.image_collection import ImageFileCollection
import ccdproc
from astropy.visualization import imshow_norm, SqrtStretch
from astropy.visualization.interval import PercentileInterval
from astropy.io import fits
from photutils import DAOStarFinder
from arte.types.scalar_bidimensional_function import \
    ScalarBidimensionalFunction
from arte.utils.coordinates import xCoordinatesMap
from arte.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform
import matplotlib.pyplot as plt
from astropy.visualization.stretch import SquaredStretch


class LuciImageCleaner(object):


    def main(self):
        luciImageDir= "/Volumes/GoogleDrive/Il mio Drive/"+ \
            "adopt/ARGOS/LuciImages/argosraid/201610/data/20161019"
        coll= ImageFileCollection(luciImageDir)
        collDark= ImageFileCollection(
            location=luciImageDir,
            filenames=list(coll.files_filtered(frametyp='dark')))
        collFlat= ImageFileCollection(
            location=luciImageDir,
            filenames=list(coll.files_filtered(frametyp='flat')))
        collSky= ImageFileCollection(
            location=luciImageDir,
            filenames=list(coll.files_filtered(frametyp='sky')))
        collScience= ImageFileCollection(
            location=luciImageDir,
            filenames=list(coll.files_filtered(frametyp='science')))

        darkCcds= [data for data in collDark.ccds()]
        flatCcds= [data for data in collFlat.ccds()]


        combiner= ccdproc.Combiner(darkCcds)
        masterDark= combiner.median_combine()

        flatCombiner= ccdproc.Combiner(flatCcds)
        flat= ccdproc.subtract_dark(
            flatCombiner.median_combine(), masterDark,
            data_exposure=42*u.second,
            dark_exposure=42*u.second)



class ErisLensletArrayCharacterization(object):

    def __init__(self,
                 fname,
                 pixelSizeInMicron,
                 pitchGuessInMicron=None,
                 darkFileName=None,
                 name='a Measure'):
        self._fname= fname
        self._pxSize= pixelSizeInMicron
        ima=fits.getdata(fname)
        if len(ima.shape) == 3:
            ima= ima[0]
        if darkFileName:
            dark= fits.getdata(darkFileName)
            self._image= ima - dark
        else:
            self._image= ima
        self._name= name
        self._positions= None
        self._fft= None
        self._gaussianLineFit= None
        self._cutFFTAlongY= False
        self._noiseFactor=5
        self._pitchGuessInMicron= pitchGuessInMicron


    def _reset(self):
        self._fft=None
        self._gaussianLineFit= None


    def useFFTCutAlongYAxis(self, trueOrFalse):
        self._cutFFTAlongY= trueOrFalse
        self._reset()


    def setNoiseFactor(self, noiseFactor):
        self._noiseFactor= noiseFactor
        self._reset()


    def _computeSpotPosition(self):
        daofind= DAOStarFinder(fwhm=1., threshold=500.)
        sources= daofind(self._image)
        self._positions = np.array((sources['xcentroid'],
                                    sources['ycentroid'])).T


    def showImage(self):
        plt.clf()
        sy, sx= self.image().shape
        xmax= 0.5 * sx - 1./ sx
        ymax= 0.5 * sy - 1./ sy
        extent= (-xmax, xmax, -ymax, ymax)
        imshow_norm(self.image(),
                    origin='lower',
                    interval=PercentileInterval(99.98),
                    stretch=SqrtStretch(),
                    extent=extent)
        plt.title(self._name)
        plt.colorbar()


    def showIntensityMap(self):
        plt.clf()
        iMap= self._computeIntensityMap()
        imshow_norm(iMap,
                    origin='lower',
                    stretch=SquaredStretch())
        plt.title(self._name)
        plt.colorbar()


    def showFFT(self):
        plt.clf()
        imshow_norm(np.abs(self.fft().values()),
                    interval=PercentileInterval(99.98),
                    stretch=SqrtStretch(),
                    extent=self.fft().extent(),
                    origin='lower')
        plt.title(self._name)


    def _fftCut(self):
        cc= (0.5* np.array(self.image().shape)).astype(int)
        if self._cutFFTAlongY:
            return (self.fft().yCoord()[cc[1]:, cc[0]],
                    np.abs(self.fft().values())[cc[1]:, cc[0]])
        else:
            return (self.fft().xCoord()[cc[1], cc[0]:],
                    np.abs(self.fft().values())[cc[1], cc[0]:])



    def showFFTCut(self):
        x, y= self._fftCut()
        lines= self._findFFTLines()
        # Plot the data with the best-fit model
        plt.figure()
        plt.plot(x, y, label='FFT')
        plt.plot(x, self.gaussianLineFit()(x), label='Gaussian fit')
        plt.xlabel('Position [1/px]')
        plt.xlim(0, lines[3]['line_center'].value)
        plt.legend()


    def image(self):
        return self._image


    def spotPositions(self):
        if self._positions is None:
            self._computeSpotPosition()
        return self._positions


    def _distanceFrom(self, xC, yC):
        offset= self.spotPositions() - np.array([xC, yC])
        return np.linalg.norm(offset, axis=1)


    def _image2DF(self):
        image2DF= ScalarBidimensionalFunction(
            self.image(),
            xCoordinatesMap(self.image().shape[1], 1.0),
            xCoordinatesMap(self.image().shape[0], 1.0).T)
        return image2DF


    def fft(self):
        if not self._fft:
            im2D= self._image2DF()
            self._fft= BidimensionalFourierTransform.direct(im2D)
        return self._fft



    def _findFFTLines(self):
        from specutils import Spectrum1D, SpectralRegion
        from specutils.fitting import find_lines_threshold
        from specutils.manipulation import noise_region_uncertainty

        cutX, cutY= self._fftCut()
        spectrum= Spectrum1D(flux=cutY*u.ct,
                             spectral_axis=cutX/u.pix)
        noise_region= SpectralRegion(0.4/u.pix, 0.5/u.pix)
        spectrum= noise_region_uncertainty(spectrum, noise_region)
        lines= find_lines_threshold(spectrum,
                                    noise_factor=self._noiseFactor)
        return lines


    def fitFFT(self):
        from astropy.modeling import models, fitting

        x, y= self._fftCut()
        if self._pitchGuessInMicron is None:
            firstLine= self._findFFTLines()[1]
            x_0= firstLine[0].value
            amplitude= y[firstLine[2]]
        else:
            x_0= 1. / (self._pitchGuessInMicron / self._pxSize)
            amplitude= y[np.abs(x - x_0).argmin()]
        width= 0.1* x_0

        # Fit the data using a Gaussian
        g_init = models.Gaussian1D(amplitude=amplitude,
                                   mean=x_0,
                                   stddev=width)
        self._fitter = fitting.LevMarLSQFitter()
        self._gaussianLineFit = self._fitter(g_init, x, y)



    def fitFFT2D(self, x_0, y_0):
        from astropy.modeling import models, fitting

        width= 0.1* np.max(np.abs([x_0, y_0]))
        boxsz= 3* width
        fftCut= self.fft().cutOutFromROI(x_0 - boxsz, x_0 + boxsz,
                                         y_0 - boxsz, y_0 + boxsz)

        x= fftCut.xCoord()
        y= fftCut.yCoord()
        z= np.abs(fftCut.values())

        amplitude= np.abs(fftCut.interpolateInXY([x_0], [y_0]))

        # Fit the data using a Gaussian
        g_init = models.Gaussian2D(amplitude=amplitude,
                                   x_mean=x_0,
                                   y_mean=y_0,
                                   x_stddev=width,
                                   y_stddev=width,
                                   theta=0)
        self._fitter = fitting.LevMarLSQFitter()
        self._gaussian2DFit = self._fitter(g_init, x, y, z)
        distInUm= self._pxSize / np.linalg.norm((
            self._gaussian2DFit.x_mean, self._gaussian2DFit.y_mean))
        print("%s %g %g %g" % (self._name, x_0, y_0, distInUm))
        return distInUm


    def _guessPositionOfFirstFFTMaximum(self):
        firstLine= self._findFFTLines()[1]
        x_0= firstLine[0].value
        return x_0


    def gaussianLineFit(self):
        if not self._gaussianLineFit:
            self.fitFFT()
        return self._gaussianLineFit


    def getAverageSpotSpacingInMicron(self):
        self._computeSpotSpacingInMicronFromFFT2D()
        return self._spotSpacingInMicron.mean()



    def _computeSpotSpacingInMicronFromFFT2D(self):
        d_0= 1. / (self._pitchGuessInMicron / self._pxSize)
        spcN= self.fitFFT2D(0, d_0)
        spcS= self.fitFFT2D(0, -d_0)
        spcE= self.fitFFT2D(d_0, 0)
        spcW= self.fitFFT2D(-d_0, 0)
        self._spotSpacingInMicron= np.array([spcN, spcE, spcS, spcW])


    def _computeSpotSpacingInMicron(self):
        self.useFFTCutAlongYAxis(False)
        spcX= 1. / self.gaussianLineFit().mean * self._pxSize
        self.useFFTCutAlongYAxis(True)
        spcY= 1. / self.gaussianLineFit().mean * self._pxSize
        self._spotSpacingInMicron= np.array([spcX, spcY])



    def _computeIntensityMap(self):
        fractionOfPitch= 0.4
        ima= self.image()
        hp= fractionOfPitch * self._pitchGuessInMicron / self._pxSize
        mappa= np.zeros(ima.shape)
        pos= self.spotPositions()
        for p in pos:
            roi= np.array((p[1] - hp,
                           p[1] + hp,
                           p[0] - hp,
                           p[0] + hp)
                          ).astype(int)
            mappa[roi[0]:roi[1], roi[2]:roi[3]]= \
                np.sum(ima[roi[0]:roi[1], roi[2]:roi[3]])
        return mappa


def mainEris181213():
    import os
    import glob
    fnames= glob.glob(
        '/Users/lbusoni/Downloads/181213_PitchTest/PitchTest_[LH]O*.fits')
    basedir= os.path.dirname(fnames[0])
    elabdir= os.path.join(basedir, 'elab')
    os.makedirs(elabdir, exist_ok=True)
    darkFileName= os.path.join(basedir, 'PitchTest_dark.fits')



    plt.figure(figsize=(10, 8))
    for fname in fnames:
        measureName= os.path.basename(fname)[10:-5]
        cameraPixelSizeInMicron= 7.4
        if measureName[:2] == 'HO':
            pitchGuessInMicron= 144.
        elif measureName[:2] == 'LO':
            pitchGuessInMicron= 1440.
        else:
            raise Exception(
                'Unknown LA type for file %s. Is it LO or HO?' % fname)
        erisLA= ErisLensletArrayCharacterization(
            fname,
            cameraPixelSizeInMicron,
            pitchGuessInMicron=pitchGuessInMicron,
            darkFileName=darkFileName,
            name=measureName)
        erisLA.showImage()
        plt.title('%s - pitch: %g um' % (
            measureName, erisLA.getAverageSpotSpacingInMicron()))
        plt.savefig(os.path.join(elabdir,
                                 'Image_%s.png' % measureName))
        erisLA.showIntensityMap()
        plt.savefig(os.path.join(elabdir,
                                 'Intensity_%s.png' % measureName))
