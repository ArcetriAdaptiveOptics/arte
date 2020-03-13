'''
Created on 13 mar 2020

@author: giuliacarla
'''


class Integrator():

    def __init__(self, delay, gain):
        self.delay = delay
        self.gain = gain

    def _computeRTF(self):
        pass

    def _computeNTF(self):
        pass

    def getRejectionTransferFunction(self):
        pass

    def getNoiseTransferFunction(self):
        pass
