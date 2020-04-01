'''
Created on 26 mar 2020

@author: giuliacarla
'''

import numpy as np
from apposto.utils.modal_decomposer import ModalDecomposer
from apposto.utils.zernike_generator import ZernikeGenerator
from apposto.types.wavefront import Wavefront
from apposto.types.mask import CircularMask


class MCAOInteractionMatrix():
    """
    Compute MCAO interaction matrix.

    Parameters
    ----------
    footprints: list
        List of `~apposto.utils.footprint_geometry.FootprintGeometry`
        objects.
        Each element contains the footprint geometry related to
        each of the considered atmospheric layers.

    DMs: list
        List of `~apposto.types.aperture.CircularOpticalAperture` objects.
        The objects represent the deformable mirrors conjugated at the
        considered atmospheric layers.

    n_sensed_modes: int
        Number of Zernike modes measured from the guide stars.

    idx_corrected_modes: list
        List of `numpy.ndarray`. Each element of the list must be an array
        containing the indices of the modes to be corrected on each layer.

    meters_to_pixels: int
        It is the meter-pixel conversion factor, whose value generally depends
        on the scale of the atmospheric turbulence (r0).
        Default is 20, that is obtained if we decide to sample a turbulence
        with r0=10cm using 1 pixel every 5cm.
    """

    def __init__(self, footprints, DMs, n_sensed_modes, idx_corrected_modes,
                 meters_to_pixels=20):
        self._fp = footprints
        self._dm = DMs
        self._n = n_sensed_modes
        self._ml = idx_corrected_modes
        self._metersToPixels = meters_to_pixels
        self._checkInput()
        self._numberOfLayers = len(footprints)
        self._numberOfNgs = len(footprints[0].getNgsFootprint())
        self._numberOfLgs = len(footprints[0].getLgsFootprint())

    def getInteractionMatrixOffTarget(self):
        im_off = self._computeNgsInteractionMatrix()
        return im_off

    def getInteractionMatrixOnTarget(self):
        im_on = self._computeTargetInteractionMatrix()
        return im_on

    def _checkInput(self):
        assert len(self._fp) == len(self._dm) == len(self._ml), \
            'Len of footprints, DMs and idx_corrected_modes must be the same'
        assert len(set(
            np.array([len(fp.getNgsFootprint()) for fp in self._fp]))) == 1, \
            'Number of NGS must be the same on each layer'
        assert len(set(
            np.array([len(fp.getLgsFootprint()) for fp in self._fp]))) == 1, \
            'Number of LGS must be the same on each layer'

    def _fromMetersToPixels(self, inMeters):
        return inMeters * self._metersToPixels

    def _computeTargetInteractionMatrixOneLayer(self, n_layer):
        dm_radius = self._dm[n_layer].getApertureRadius().value
        dm_diameter_pixel = 2 * round(self._fromMetersToPixels(dm_radius))

        shape_frame = (dm_diameter_pixel, dm_diameter_pixel)
        fp = self._fp[n_layer]
        fp_target = fp.getTargetFootprint()[0]
        fp_center_pixel = (round(self._fromMetersToPixels(fp_target.y) +
                                 shape_frame[1] / 2),
                           round(self._fromMetersToPixels(fp_target.x) +
                                 shape_frame[0] / 2))
        fp_radius_pixel = round(self._fromMetersToPixels(fp_target.r))
        fp_mask = CircularMask(shape_frame, fp_radius_pixel,
                               fp_center_pixel)
        n_sense = self._n
        m_correct = self._ml[n_layer]

        zg = ZernikeGenerator(dm_diameter_pixel)
        md = ModalDecomposer(n_zernike_modes=self._n)
        int_mat = np.zeros((n_sense, m_correct.size))

        for i in range(m_correct.size):
            wf = Wavefront(zg.getZernike(m_correct[i]))
            a = md.measureZernikeCoefficientsFromWavefront(
                wf, fp_mask).toNumpyArray()
            int_mat[:, i] = a

        return int_mat

    def _computeTargetInteractionMatrix(self):
        im_list = []
        for n in range(self._numberOfLayers):
            im_target = self._computeTargetInteractionMatrixOneLayer(n)
            im_list.append(im_target)
        return np.hstack(im_list)

    def _computeNgsInteractionMatrixOneLayerOneStar(self, n_layer, n_ngs):
        dm_radius = self._dm[n_layer].getApertureRadius().value
        dm_diameter_pixel = 2 * round(self._fromMetersToPixels(dm_radius))

        shape_frame = (dm_diameter_pixel, dm_diameter_pixel)
        fp = self._fp[n_layer]
        fp_ngs = fp.getNgsFootprint()[n_ngs]
        fp_center_pixel = (round(self._fromMetersToPixels(fp_ngs.y) +
                                 shape_frame[1] / 2),
                           round(self._fromMetersToPixels(fp_ngs.x) +
                                 shape_frame[0] / 2))
        fp_radius_pixel = round(self._fromMetersToPixels(fp_ngs.r))
        fp_mask = CircularMask(shape_frame, fp_radius_pixel,
                               fp_center_pixel)
        n_sense = self._n
        m_correct = self._ml[n_layer]

        zg = ZernikeGenerator(dm_diameter_pixel)
        md = ModalDecomposer(n_zernike_modes=self._n)
        int_mat = np.zeros((n_sense, m_correct.size))

        for i in range(m_correct.size):
            wf = Wavefront(zg.getZernike(m_correct[i]))
            a = md.measureZernikeCoefficientsFromWavefront(
                wf, fp_mask).toNumpyArray()
            int_mat[:, i] = a

        return int_mat

    def _computeNgsInteractionMatrixOneLayer(self, n_layer):
        im_list = []
        for ngs in range(self._numberOfNgs):
            im_ngs = self._computeNgsInteractionMatrixOneLayerOneStar(
                n_layer, ngs)
            im_list.append(im_ngs)
        return np.vstack(im_list)

    def _computeNgsInteractionMatrix(self):
        im_list = []
        for n in range(self._numberOfLayers):
            im_ngs = self._computeNgsInteractionMatrixOneLayer(n)
            im_list.append(im_ngs)
        return np.hstack(im_list)

#     def _computeInteractionMatrixOneLayerOneLgs(self, n_layer, n_lgs):
#         dm_radius = self._dm[n_layer].getApertureRadius().value
#         dm_diameter_pixel = 2 * round(self._fromMetersToPixels(dm_radius))
#
#         shape_frame = (dm_diameter_pixel, dm_diameter_pixel)
#         fp = self._fp[n_layer]
#         fp_lgs = fp.getLgsFootprint()[n_lgs]
#         fp_center_pixel = (round(self._fromMetersToPixels(fp_lgs.y) +
#                                  shape_frame[1] / 2),
#                            round(self._fromMetersToPixels(fp_lgs.x) +
#                                  shape_frame[0] / 2))
#         fp_radius_pixel = round(self._fromMetersToPixels(fp_lgs.r))
#         fp_mask = CircularMask(shape_frame, fp_radius_pixel,
#                                fp_center_pixel)
#         n_sense = self._n
#         m_correct = self._ml[n_layer]
#
#         zg = ZernikeGenerator(dm_diameter_pixel)
#         md = ModalDecomposer(n_zernike_modes=self._n)
#         int_mat = np.zeros((n_sense, m_correct.size))
#
#         for i in range(m_correct.size):
#             wf = Wavefront(zg.getZernike(m_correct[i]))
#             a = md.measureZernikeCoefficientsFromWavefront(
#                 wf, fp_mask).toNumpyArray()
#             int_mat[:, i] = a
#
#         return int_mat


#     def _computeLgsInteractionMatrixOneLayer(self, n_layer):
#         pass
