#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform as bfft
from arte.types.scalar_bidimensional_function import ScalarBidimensionalFunction
from arte.types.domainxy import DomainXY


class BidimensionalFourierTransformTest(unittest.TestCase):

    def setUp(self):
        pass

    def testDirectTransformConstantMapEvenSize(self):
        sz = 100
        sz2 = int(sz / 2)
        constantMap = np.ones((sz, sz)) * 3.3
        res = bfft.direct_transform(constantMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertTrue(
            np.all(np.argwhere(res > 1e-10)[0] == (sz2, sz2)))
        self._checkParseval(constantMap, res)

    def testDirectTransformConstantMapOddSize(self):
        sz = 101
        sz2 = int(sz / 2)
        constantMap = np.ones((sz, sz))
        res = bfft.direct_transform(constantMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertTrue(
            np.all(np.argwhere(res > 1e-10)[0] == (sz2, sz2)))
        self._checkParseval(constantMap, res)

    def testInverseTransformDeltaMapEvenSize(self):
        sz = 100
        sz2 = int(sz / 2)
        deltaMap = np.zeros((sz, sz))
        deltaMap[sz2, sz2] = 1.0
        res = bfft.inverse_transform(deltaMap)
        self.assertEqual((sz, sz), res.shape)
        self.assertEqual(0, np.ptp(res))
        self._checkParseval(deltaMap, res)

    def _makeSinusMap(self, sizeInPoints, pixelSize,
                      amplitude, periodInLengthUnits, phase):
        sizeInLengthUnits = sizeInPoints * pixelSize
        xCoords = np.linspace(0, sizeInLengthUnits, sizeInPoints)
        sinVect = amplitude * np.sin(
            2 * np.pi * xCoords / periodInLengthUnits + phase)
        return np.tile(sinVect, (sizeInPoints, 1))

    def testDirectTransformSinusX(self):
        sizeInPoints = 500
        pixelSize = 0.2
        periodInLengthUnits = 4.0
        amplitude = 13.4
        phase = 0.8
        spatialMap = self._makeSinusMap(
            sizeInPoints, pixelSize, amplitude, periodInLengthUnits, phase)
        spectralMap = bfft.direct_transform(spatialMap)
        freqX = bfft.frequencies_x_map(sizeInPoints, pixelSize)
        freqY = bfft.frequencies_y_map(sizeInPoints, pixelSize)
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
#         spatialMap= bfft.inverse_transform(spectralMap, 1.0)
#         freqX= bfft.frequencies_x_map(sizeInPoints, pixelSize)
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
        sz = 100
        pixelSize = 1.0
        freqs = bfft.frequencies_norm_map(sz, pixelSize)
        self._checkFrequenciesNormMap(freqs, sz, pixelSize)

    def testFrequenciesMapOdd(self):
        sz = 101
        pixelSize = 1.0
        freqs = bfft.frequencies_norm_map(sz, pixelSize)
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
        sz2 = int(sizeInPoints / 2)
        mostPosFrequency = bfft.most_positive_frequency(sizeInPoints, pixelSize)
        mostNegFrequency = bfft.most_negative_frequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertEqual(0, frequenciesMap[sz2, sz2])
        self.assertAlmostEqual(-mostNegFrequency * np.sqrt(2),
                               frequenciesMap[0, 0])
        self.assertAlmostEqual(-mostNegFrequency, frequenciesMap[0, sz2])
        self.assertAlmostEqual(-mostNegFrequency, frequenciesMap[sz2, 0])
        self.assertAlmostEqual(mostPosFrequency, frequenciesMap[sz2, -1])
        self.assertAlmostEqual(mostPosFrequency, frequenciesMap[-1, sz2])
        wantedDelta = 1. / (sizeInPoints * pixelSize)
        self.assertEqual(wantedDelta, frequenciesMap[sz2, sz2 + 1])

    def testFrequenciesMapEvenWithPixelSize(self):
        sizeInPoints = 100
        pxSize = 0.1
        freqs = bfft.frequencies_norm_map(sizeInPoints, pxSize)
        self._checkFrequenciesNormMap(freqs, sizeInPoints, pxSize)

    def testFrequenciesXMap(self):
        sizeInPoints = 100
        pixelSize = 1.0
        frequenciesMap = bfft.frequencies_x_map(sizeInPoints, pixelSize)

        sz2 = int(sizeInPoints / 2)
        mostPosFrequency = bfft.most_positive_frequency(sizeInPoints, pixelSize)
        mostNegFrequency = bfft.most_negative_frequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertEqual(0, frequenciesMap[sz2, sz2])
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[:, 0]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[:, -1]))

    def testFrequenciesXMapOdd(self):
        sizeInPoints = 11
        pixelSize = 0.12
        frequenciesMap = bfft.frequencies_x_map(sizeInPoints, pixelSize)

        sz2 = int(sizeInPoints / 2)
        mostPosFrequency = bfft.most_positive_frequency(sizeInPoints, pixelSize)
        mostNegFrequency = bfft.most_negative_frequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertTrue(
            np.all(0 == frequenciesMap[:, sz2]))
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[:, 0]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[:, -1]))

    def testFrequenciesYMap(self):
        sizeInPoints = 1022
        pixelSize = 0.001
        frequenciesMap = bfft.frequencies_y_map(sizeInPoints, pixelSize)

        sz2 = int(sizeInPoints / 2)
        mostPosFrequency = bfft.most_positive_frequency(sizeInPoints, pixelSize)
        mostNegFrequency = bfft.most_negative_frequency(sizeInPoints, pixelSize)
        self.assertEqual((sizeInPoints, sizeInPoints),
                         frequenciesMap.shape)
        self.assertTrue(
            np.all(0 == frequenciesMap[sz2, :]))
        self.assertTrue(
            np.all(mostNegFrequency == frequenciesMap[0, :]))
        self.assertTrue(
            np.all(mostPosFrequency == frequenciesMap[-1, :]))

    def test_direct_sinus_x(self):
        sizeInPoints = 500
        pixelSize = 0.2
        periodInLengthUnits = 4.0
        amplitude = 13.4
        phase = 0.8
        spatialMap = self._makeSinusMap(
            sizeInPoints, pixelSize, amplitude, periodInLengthUnits, phase)
        xyDomain = DomainXY.from_shape((sizeInPoints, sizeInPoints),
                                       pixelSize)
        xyFunct = ScalarBidimensionalFunction(spatialMap, domain=xyDomain)
        fftFunct = bfft.direct(xyFunct)
        spectralMap = fftFunct.values
        freqX = bfft.frequencies_x_map(sizeInPoints, pixelSize)
        freqY = bfft.frequencies_y_map(sizeInPoints, pixelSize)

        self.assertEqual((sizeInPoints, sizeInPoints), spectralMap.shape)
        self.assertEqual(
            1.0 / periodInLengthUnits,
            np.abs(freqX.flatten()[np.argmax(np.abs(spectralMap))]))
        self.assertEqual(
            0.0,
            np.abs(freqY.flatten()[np.argmax(np.abs(spectralMap))]))
        self._checkParseval(spatialMap, spectralMap)

    def testInverseDeltaMapEvenSize(self):
        sz = 100
        sz2 = int(sz / 2)
        deltaMap = np.zeros((sz, sz))
        deltaMap[sz2, sz2] = 1.0
        xyDomain = DomainXY.from_shape((sz, sz), 1)
        xyFunct = ScalarBidimensionalFunction(deltaMap,
                                              domain=xyDomain)
        fftFunct = bfft.reverse(xyFunct)
        spectralMap = fftFunct.values
        self.assertEqual((sz, sz), spectralMap.shape)
        self.assertEqual(0, np.ptp(spectralMap))
        self._checkParseval(deltaMap, spectralMap)

    def test_rectangular_domain(self):

        szx, szy = (20, 10)
        stepx, stepy = (0.1, 0.4)
        ampl = 1.0
        xy = DomainXY.from_shape((szy, szx), (stepy, stepx))
        constant_map = ampl * np.ones(xy.shape)
        spatial_funct = ScalarBidimensionalFunction(constant_map, domain=xy)
        spectr = bfft.direct(spatial_funct)
        freq_step_x, freq_step_y = spectr.domain.step

        self.assertAlmostEqual(
            0, spectr.xmap[szy // 2, szx // 2])
        self.assertAlmostEqual(
            0, spectr.ymap[szy // 2, szx // 2])
        self.assertAlmostEqual(
            np.sqrt(ampl * szx * szy),
            spectr.values[szy // 2, szx // 2])

        self.assertAlmostEqual(-0.5 / stepx, spectr.xcoord[0])
        self.assertAlmostEqual(1 / (szx * stepx), freq_step_x)

        self.assertAlmostEqual(-0.5 / stepy, spectr.ycoord[0])
        self.assertAlmostEqual(1 / (szy * stepy), freq_step_y)

    def test_inverse_of_direct_return_original(self):
        sz = 4
        spatial_step = 1.0
        ampl = 1.0
        xy = DomainXY.from_shape((sz, sz), spatial_step)
        constant_map = ampl * np.ones(xy.shape)
        original = ScalarBidimensionalFunction(constant_map, domain=xy)
        spectr = bfft.direct(original)
        inverse_spectr = bfft.reverse(spectr)
        self.assertTrue(np.allclose(
            inverse_spectr.values, original.values))

    def test_shifted_domain_should_not_affect_spectrum(self):
        sz = 4
        spatial_step = 2.54
        ampl = 42.0
        xy = DomainXY.from_shape((sz, sz), spatial_step)
        xy.shift(3, 2)
        values = ampl * np.ones(xy.shape)
        spatial_funct = ScalarBidimensionalFunction(
            values, domain=xy)
        spectr_shifted = bfft.direct(spatial_funct)

        xy = DomainXY.from_shape((sz, sz), spatial_step)
        values = ampl * np.ones(xy.shape)
        spatial_funct = ScalarBidimensionalFunction(
            values, domain=xy)
        spectr = bfft.direct(spatial_funct)

        self.assertTrue(np.allclose(
            spectr_shifted.values, spectr.values))

    def test_with_units(self):
        from astropy import units as u
        szx, szy = (20, 10)
        stepx, stepy = (0.1 * u.m, 0.4 * u.kg)
        ampl = 1.0 * u.V
        xy = DomainXY.from_shape((szy, szx), (stepy, stepx))
        map_in_V = ampl * np.ones(xy.shape)
        spatial_funct = ScalarBidimensionalFunction(map_in_V, domain=xy)
        spectr = bfft.direct(spatial_funct)
        self.assertTrue(spectr.xmap.unit.is_equivalent((1 / u.m).unit))
        self.assertTrue(spectr.ymap.unit.is_equivalent((1 / u.kg).unit))
        self.assertTrue(spectr.xcoord.unit.is_equivalent((1 / u.m).unit))
        self.assertTrue(spectr.ycoord.unit.is_equivalent((1 / u.kg).unit))
        self.assertTrue(spectr.values.unit.is_equivalent((u.V)))

    def test_inverse_units(self):
        from astropy import units as u
        szx, szy = (20, 10)
        stepx, stepy = (0.1 * u.m, 0.4 * u.kg)
        ampl = 1.0 * u.V
        xy = DomainXY.from_shape((szy, szx), (stepy, stepx))
        map_in_V = ampl * np.ones(xy.shape)
        spatial_funct = ScalarBidimensionalFunction(map_in_V, domain=xy)
        spectr = bfft.reverse(spatial_funct)
        self.assertTrue(spectr.xmap.unit.is_equivalent((1 / u.m).unit))
        self.assertTrue(spectr.ymap.unit.is_equivalent((1 / u.kg).unit))
        self.assertTrue(spectr.xcoord.unit.is_equivalent((1 / u.m).unit))
        self.assertTrue(spectr.ycoord.unit.is_equivalent((1 / u.kg).unit))
        self.assertTrue(spectr.values.unit.is_equivalent((u.V)))


    def test_parseval(self):    
        x = np.random.rand(321, 456)
        f = bfft.direct_transform(x)
        want = np.sum(np.abs(x)**2)
        got = np.sum(np.abs(f)**2)
        np.testing.assert_allclose(got, want)

        xy = DomainXY.from_shape(x.shape, 0.123)
        spatial_funct = ScalarBidimensionalFunction(x, domain=xy)
        spectr = bfft.reverse(spatial_funct)

        want = np.sum(np.abs(spatial_funct.values)**2)
        got = np.sum(np.abs(spectr.values)**2)
        np.testing.assert_allclose(got, want)
        
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
