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
    """

    def __init__(self, footprints, DMs, n_sensed_modes, idx_corrected_modes,
                 metersToPixels):
        self._fp = footprints
        self._dm = DMs
        self._n = n_sensed_modes
        self._ml = idx_corrected_modes
        # Separate function "checkInput"?
        assert len(self._fp) == len(self._dm) == len(self._ml), \
            'Length of footprints, DMs and idx_corrected_modes must be the same.'
        self._numberOfLayers = len(footprints)
        # tira fuori dal footprintgeometry
        self._numberOfNgs = len(footprints[0]._ngsL)
        self._numberOfLgs = len(footprints[0]._lgsL)
        self._numberOfGuideStars = self._numberOfNgs + self._numberOfLgs
        self._metersToPixels = metersToPixels
# Deve diventare una conversione specifica per i vari layer

    def _fromMetersToPixels(self, inMeters):
        return inMeters * self._metersToPixels

    def _computeInteractionMatrixOneLayerOneNGS(self, n_layer, n_ngs):
        dm_radius = self._dm[n_layer].getApertureRadius().value
        dm_diameter_pixel = 2 * round(self.fromMetersToPixels(dm_radius))

        shape_frame = (dm_diameter_pixel, dm_diameter_pixel)
        fp = self._fp[n_layer]
        fp_ngs = fp.getNgsFootprint()[n_ngs]
        fp_center_pixel = (round(self.fromMetersToPixels(fp_ngs.y) +
                                 shape_frame[1] / 2),
                           round(self.fromMetersToPixels(fp_ngs.x) +
                                 shape_frame[0] / 2))
        fp_radius_pixel = round(self.fromMetersToPixels(fp_ngs.r))
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
