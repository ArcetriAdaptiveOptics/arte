'''
Created on 13 mar 2020

@author: giuliacarla
'''

import numpy as np


class LaplaceTransferFunction():
    """
    Return Open Loop Transfer Function(OLTF), Rejection Transfer Function (RTF)
    and Noise Transfer Function (NTF) of a basic control system made of a sensor with
    integration time T and readout time Tr, an integral control with gain G and 
    delay Tc, and sample-and-hold commands to an ideal actuator.
    The TFs are computed using Laplace transform.

    Parameters
    ----------
    temporal_freqs: `~numpy.ndarray`
        Array of temporal frequencies used to compute OLTF, RTF and NTF.

    gain: int or float
        System's gain set by the user. If needed, the system's optimal gain
        can be computed (see module 'set_optimal_gain').

    t_integration: int or float
        Time interval over which the system's sensor integrates the input
        signal.
        It determines the system's loop frequency (f_loop = 1 / t_integration).

    t_readout: int or float
        Sensor's readout time. Default is 0.

    t_control: int or float
        Delay introduced by the digital computation of the signal,
        the filtering and the final analogic conversion of the signal.
        Default is 0.
    """

# TODO: restituire dei tipi 'Transfer Function'?

    def __init__(self, temporal_freqs, t_integration,
                 t_readout=0, t_control=0, gain=1):
        self.temporal_freqs = temporal_freqs
        self.T = t_integration
        self.Tr = t_readout
        self.Tc = t_control
        self.gain = gain
        self.Tdelay = t_readout + t_control
        self._s = 1j * 2 * np.pi * temporal_freqs
        self._chooseIntegrator = False
        self._chooseSampleAndHold = False

    def set_integrator(self, true_false):
        self._chooseIntegrator = true_false

    def set_sample_and_hold(self, true_false):
        self._chooseSampleAndHold = true_false

    def set_optimal_gain(self):
        """
        Find the system's optimal gain based on OLTF's amplitude value
        at the frequency where the phase is equal to 135 degrees.
        """
        amp = self.get_amplitude(self.OLTF())
        amp135 = amp[self.idx_omega_135]
        rounding = 0.001
        if amp135 <= 1 + rounding:
            self.gain = self.gain / amp135
        else:
            raise Exception('Input gain is too high. The system is unstable.')

    def get_bandwidth_at_optimal_gain(self):
        return self.temporal_freqs[self.idx_omega_135]

    def rejection_bandwidth(self):
        self._findRejectionBandOmegaIdx()
        return self.temporal_freqs[self.idx_omega_rejection]

    def _findRejectionBandOmegaIdx(self):
        amp = self.get_amplitude(self.RTF())
        idxMax = np.argwhere(amp == amp.max())[0][0]
        ampToMax = amp[:idxMax]
        diff = abs(ampToMax - 1)
        self.idx_omega_rejection = np.argwhere(diff == diff.min())[0][0]

    def _findOmega135Idx(self):
        phi = self.get_phase(self.OLTF(), unit='deg')
        diff = abs(phi - (-135))
        return np.argwhere(diff == diff.min())[0][0]

    @property
    def idx_omega_135(self):
        return self._findOmega135Idx()

    def OLTF(self):
        """
        Get Open Loop Transfer Function.

        Return
        ------
        oltf: `~numpy.ndarray`
        """

        S = self._computeSensorTransferFunction()
        F = self._computeFilterTransferFunction()
        H = self._computeSampleAndHoldTransferFunction()
        A = self._computeActuatorTransferFunction()
        oltf = S * F * H * A
        self._checkFrequencySampling(oltf)
        return oltf

    def RTF(self):
        """
        Get Rejection Transfer Function.

        Return
        ------
        rtf: `~numpy.ndarray`
        """
        return 1. / (1 + self.OLTF())

    def NTF(self):
        """
        Get Noise Transfer Function

        Return
        ------
        ntf: `~numpy.ndarray`
        """
        return self.OLTF() / (1 + self.OLTF())

    def get_amplitude(self, transfer_function):
        """
        Get transfer function's amplitude.

        Parameters
        ----------
        transfer_function: `~numpy.ndarray`

        Return
        ------
        amplitude: `~numpy.ndarray`
            Transfer function's module.
        """

        amplitude = abs(transfer_function)
        return amplitude

    def get_phase(self, transfer_function, unit='deg'):
        """
        Get transfer function's phase.

        Parameters
        ----------
        transfer_function: `~numpy.ndarray`
        unit: str
            Phase's unit. Default is 'radians'.

        Return
        ------
        phi: `~numpy.ndarray`
            Transfer function's phase.

        unit: str
            Phase's unit. It can be 'radians' or 'deg'. Default is 'deg'.
        """
        phi = np.unwrap(2 * np.angle(transfer_function)) / 2
        if unit == 'radians':
            return phi
        elif unit == 'deg':
            phi = np.rad2deg(phi)
            return phi

    def _checkFrequencySampling(self, transfer_function):
        import warnings

        phiInRad = self.get_phase(transfer_function)
        dPhi = np.diff(phiInRad)
        jump_up = np.where(dPhi > 0)
        if jump_up[0].shape[0] != 0:
            if np.diff(jump_up).min() == 1:
                warnings.warn(
                    'You may have to increase the frequency sample rate.')

    def _computeSensorTransferFunction(self):
        return (1 - np.exp(-self._s * self.T)) / (
            self._s * self.T) * np.exp(-self._s * self.Tr)

    def _computeFilterTransferFunction(self):
        if self._chooseIntegrator is False:
            F = self.gain
        elif self._chooseIntegrator is True:
            F = self.gain / self._s
        return F

    def _computeSampleAndHoldTransferFunction(self):
        if self._chooseSampleAndHold is False:
            H = 1
        elif self._chooseSampleAndHold is True:
            H = (1 - np.exp(-self._s * self.T)) / self._s * np.exp(
                -self._s * self.Tc)
        return H

    def _computeActuatorTransferFunction(self):
        # Per ora solo attuatore perfetto.
        return 1


class ZetaTransferFunction():
    """
    Get OLTF, RTF and NTF of a control system using Zeta transform.

    Parameters
    ----------
    f_loop:

    n_iterations:

    gain:

    delay:

    """

    def __init__(self, f_loop, n_iterations, gain, delay):
        self.loop_frequency = f_loop
        self.n_iter = n_iterations
        self.gain = gain
        self.delay = delay
        self.set_temporal_frequency_array()

    def set_temporal_frequency_array(self, n_points=None):
        f_min = self.loop_frequency / self.n_iter
        f_nyquist = self.loop_frequency / 2
        if n_points is None:
            n_points = int(self.n_iter / 2)
        self.temporal_freqs = np.linspace(f_min, f_nyquist, n_points)
        self.z = np.exp(
            1j * 2 * np.pi * self.temporal_freqs / self.loop_frequency)

    def get_amplitude(self, tf):
        return abs(tf)

    def get_phase(self, tf, unit='radians'):
        phi = np.unwrap(2 * np.angle(tf)) / 2
        if unit == 'radians':
            return phi
        elif unit == 'deg':
            return np.rad2deg(phi)

    def RTF(self):
        rtf = (1 - self.z ** (-1)) / (
            1 - self.z ** (-1) + self.gain * self.z ** (-self.delay))
        return rtf

    def NTF(self):
        ntf = -(self.gain * self.z ** (-self.delay)) / (
            1 - self.z ** (-1) + self.gain * self.z ** (-self.delay))
        return ntf


class IdealTransferFunction():

    def __init__(self, rtf=0, ntf=-1):
        self.rtf = rtf
        self.ntf = ntf

    def RTF(self):
        return self.rtf

    def NTF(self):
        return self.ntf
