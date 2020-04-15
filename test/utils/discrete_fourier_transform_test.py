#!/usr/bin/env python
import unittest
import numpy as np
from apposto.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform as bfft

__version__ = "$Id:$"


class BidimensionalFourierTransformTest(unittest.TestCase):


    def setUp(self):
        pass


    def testDirectTransformConstantMapEvenSize(self):
        sz= 100
        sz2= int(sz/ 2)
        constantMap= np.ones((sz, sz)) * 3.3
        res= bfft.directTransform(constantMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertTrue(
            np.all(np.argwhere(res > 1e-10)[0] == (sz2, sz2)))
        self._checkParseval(constantMap, res)


    def testDirectTransformConstantMapOddSize(self):
        sz= 101
        sz2= int(sz/ 2)
        constantMap= np.ones((sz, sz))
        res= bfft.directTransform(constantMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertTrue(
            np.all(np.argwhere(res > 1e-10)[0] == (sz2, sz2)))
        self._checkParseval(constantMap, res)


    def testInverseTransformDeltaMapEvenSize(self):
        sz= 100
        sz2= int(sz/ 2)
        deltaMap= np.zeros((sz, sz))
        deltaMap[sz2, sz2]= 1.0
        res= bfft.inverseTransform(deltaMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertEqual(0, res.ptp())
        self._checkParseval(deltaMap, res)


    def _makeSinusMap(self, sizeInPoints, pixelSize,
                      amplitude, periodInLengthUnits, phase):
        sizeInLengthUnits= sizeInPoints * pixelSize
        xCoords= np.linspace(0, sizeInLengthUnits, sizeInPoints)
        sinVect= amplitude *np.sin(
            2 * np.pi * xCoords / periodInLengthUnits + phase)
        return np.tile(sinVect, (sizeInPoints, 1))


    def testDirectTransformSinusX(self):
        sizeInPoints= 500
        pixelSize= 0.2
        periodInLengthUnits= 4.0
        amplitude= 13.4
        phase= 0.8
        spatialMap= self._makeSinusMap(
            sizeInPoints, pixelSize, amplitude, periodInLengthUnits, phase)
        spectralMap= bfft.directTransform(spatialMap)
        freqX= bfft.frequenciesXMap(sizeInPoints, pixelSize)
        freqY= bfft.frequenciesYMap(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints), spectralMap.shape)
        self.assertEqual(
            1.0 / periodInLengthUnits,
            np.abs(freqX.flatten()[np.argmax(np.abs(spectralMap))]))
        self.assertEqual(
            0.0,
            np.abs(freqY.flatten()[np.argmax(np.abs(spectralMap))]))
        self._checkParseval(spatialMap, spectralMap)



#     def testInverseTransformSinusX(self):
#         sizeInPoints= 500
#         pixelSize= 0.2
#         amplitude= 2.0 / sizeInPoints
# 
#         spectralMap= np.zeros((sizeInPoints, sizeInPoints))
#         posCoord= [int(sizeInPoints / 2), int(sizeInPoints / 2) + 10]
#         negCoord= [int(sizeInPoints / 2), int(sizeInPoints / 2) - 10]
#         spectralMap[posCoord[0], posCoord[1]]= 1.0
#         spectralMap[negCoord[0], negCoord[1]]= 1.0
#         spatialMap= bfft.inverseTransform(spectralMap, 1.0)
#         freqX= bfft.frequenciesXMap(sizeInPoints, pixelSize)
#         freqSinusPos= freqX[posCoord[0], posCoord[1]]
#         freqSinusNeg= freqX[negCoord[0], negCoord[1]]
#         expectedSpatialMapPos= self._makeSinusMap(
#             sizeInPoints, pixelSize, amplitude, 1./ freqSinusPos, 0)
#         expectedSpatialMapNeg= self._makeSinusMap(
#             sizeInPoints, pixelSize, amplitude, 1./ freqSinusNeg, np.pi)
#         expectedSpatialMap= expectedSpatialMapPos + expectedSpatialMapNeg
# 
#         self.assertEqual((sizeInPoints, sizeInPoints), spectralMap.shape)
#         self._checkParseval(spatialMap, spectralMap)
#         self._checkParseval(spatialMap, expectedSpatialMap)



    def _checkParseval(self, spatialMap, spectralMap):
        self.assertAlmostEqual(np.linalg.norm(spatialMap),
                               np.linalg.norm(spectralMap))


    def testFrequenciesMapEven(self):
        sz= 100
        pixelSize= 1.0
        freqs= bfft.frequenciesNormMap(sz, pixelSize)
        self._checkFrequenciesNormMap(freqs, sz, pixelSize)


    def testFrequenciesMapOdd(self):
        sz= 101
        pixelSize= 1.0
        freqs= bfft.frequenciesNormMap(sz, pixelSize)
        self._checkFrequenciesNormMap(freqs, sz, pixelSize)


    def _isEven(self, num):
        if num % 2 == 0:
            return True
        else:
            return False


    def _checkFrequenciesNormMap(self,
                                 frequenciesMap,
                                 sizeInPoints,
                                 pixelSize):
        sz2= int(sizeInPoints/ 2)
        mostPosFrequency= bfft.mostPositiveFrequency(sizeInPoints, pixelSize)
        mostNegFrequency= bfft.mostNegativeFrequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertEqual(0, frequenciesMap[sz2, sz2])
        self.assertAlmostEqual(-mostNegFrequency* np.sqrt(2),
                               frequenciesMap[0, 0])
        self.assertAlmostEqual(-mostNegFrequency, frequenciesMap[0, sz2])
        self.assertAlmostEqual(-mostNegFrequency, frequenciesMap[sz2, 0])
        self.assertAlmostEqual(mostPosFrequency, frequenciesMap[sz2, -1])
        self.assertAlmostEqual(mostPosFrequency, frequenciesMap[-1, sz2])
        wantedDelta= 1. / (sizeInPoints * pixelSize)
        self.assertEqual(wantedDelta, frequenciesMap[sz2, sz2 + 1])


    def testFrequenciesMapEvenWithPixelSize(self):
        sizeInPoints= 100
        pxSize= 0.1
        freqs= bfft.frequenciesNormMap(sizeInPoints, pxSize)
        self._checkFrequenciesNormMap(freqs, sizeInPoints, pxSize)


    def testFrequenciesXMap(self):
        sizeInPoints= 100
        pixelSize= 1.0
        frequenciesMap= bfft.frequenciesXMap(sizeInPoints, pixelSize)

        sz2= int(sizeInPoints/ 2)
        mostPosFrequency= bfft.mostPositiveFrequency(sizeInPoints, pixelSize)
        mostNegFrequency= bfft.mostNegativeFrequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertEqual(0, frequenciesMap[sz2, sz2])
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[:, 0]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[:, -1]))


    def testFrequenciesXMapOdd(self):
        sizeInPoints= 11
        pixelSize= 0.12
        frequenciesMap= bfft.frequenciesXMap(sizeInPoints, pixelSize)

        sz2= int(sizeInPoints/ 2)
        mostPosFrequency= bfft.mostPositiveFrequency(sizeInPoints, pixelSize)
        mostNegFrequency= bfft.mostNegativeFrequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertTrue(
            np.all(0 == frequenciesMap[:, sz2]))
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[:, 0]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[:, -1]))


    def testFrequenciesYMap(self):
        sizeInPoints= 1022
        pixelSize= 0.001
        frequenciesMap= bfft.frequenciesYMap(sizeInPoints, pixelSize)

        sz2= int(sizeInPoints/ 2)
        mostPosFrequency= bfft.mostPositiveFrequency(sizeInPoints, pixelSize)
        mostNegFrequency= bfft.mostNegativeFrequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertTrue(
            np.all(0 == frequenciesMap[sz2, :]))
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[0, :]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[-1, :]))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()