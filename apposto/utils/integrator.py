'''
Created on 13 mar 2020

@author: giuliacarla
'''


class Integrator():

    def __init__(self):
        self.delay = None
        self.gain = None
        self.rtf = None
        self.ntf = None

    def setDelay(self, d):
        self.delay = d

    def setGain(self, g):
        self.gain = g

    def setRejectionTransferFunction(self, rtf):
        self._rtf = rtf

    def setNoiseTransferFunction(self, ntf):
        self._ntf = ntf

    def getRejectionTransferFunction(self, temporal_freqs=None):
        if self._rtf is None:
            self._computeRTF()
        return self._rtf

    def getNoiseTransferFunction(self, temporal_freqs=None):
        if self._ntf is None:
            self._computeNTF()
        return self._ntf

    def _computeRTF(self):
        self._rtf
        pass

    def _computeNTF(self):
        self._ntf
        pass
