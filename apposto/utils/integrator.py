'''
Created on 13 mar 2020

@author: giuliacarla
'''

import numpy as np


class SimpleIntegrator():
    """
    """

    def __init__(self):
        self.delay = None
        self.gain = None

    def setDelay(self, d):
        self.delay = d

    def setGain(self, g):
        self.gain = g

    def setTemporalFrequencies(self, f):
        self.z = np.exp(2j * np.pi * f)

    def getRejectionTransferFunction(self):
        self._computeRTF()
        return self._rtf

    def getNoiseTransferFunction(self):
        self._computeNTF()
        return self._ntf

    def _computeRTF(self):
        self._rtf = (1 - self.z**(-1)) / (
            1 - self.z**(-1) + self.gain * self.z**(-self.delay))

    def _computeNTF(self):
        self._ntf = - (self.gain * self.z**(-self.delay)) / (
            1 - self.z**(-1) + self.gain * self.z**(-self.delay))


class IdealIntegrator():
    """
    """

    def __init__(self, rtf=0., ntf=-1.):
        self._rtf = rtf
        self._ntf = ntf

    def getRejectionTransferFunction(self):
        return self._rtf

    def getNoiseTransferFunction(self):
        return self._ntf
