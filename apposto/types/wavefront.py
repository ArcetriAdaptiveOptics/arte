import numpy as np


class Wavefront(object):

    def __init__(self, wf, counter=0):
        self._wf = wf
        self._counter = counter

    def toNumpyArray(self):
        return self._wf

    @staticmethod
    def fromNumpyArray(wfAsNumpyArray, counter=0):
        # assert isinstance(wfAsNumpyArray, np.ma.masked_array)
        return Wavefront(wfAsNumpyArray, counter)

    def counter(self):
        return self._counter

    def setCounter(self, counter):
        self._counter = counter

    def __eq__(self, o):
        if self._counter != o._counter:
            return False
        if not np.array_equal(self._wf, o._wf):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

    def std(self):
        return self._wf.std()

