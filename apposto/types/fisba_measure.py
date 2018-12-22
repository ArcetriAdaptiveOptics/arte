import numpy as np
import struct



class FisbaMeasure(object):

    TYPE_SURFACE_DEVIATION= 1
    TYPE_WAVE_ABERRATION= 2
    TYPE_RAW_PHASE_DATA= 3
    TYPE_INTENSITY= 4


    def __init__(self, filename):
        self._filename= filename
        self._dataMap, self._dataType, self._comment= self._convert(
            self._filename)


    def _convert(self, filename):
        with open(filename, 'rb') as binary_file:
            data= bytearray(binary_file.read())
        dataId, nRow, nCol, maskValue, commentLen=struct.unpack(
            "<ciiii", data[0:17])
        dataType=ord(dataId)
        comment=struct.unpack(
            "<%ds" % commentLen,
            data[17:17 + commentLen])[0]
        dataArray=np.array(struct.unpack(
            "<" + "i"* nRow * nCol,
            data[17 + commentLen:17 + commentLen + nRow * nCol * 4])
        ).reshape((nRow, nCol))
        mask= dataArray == maskValue
        return np.ma.array(dataArray, mask=mask), dataType, comment


    def dataType(self):
        return self._dataType

    def map(self):
        return self._dataMap

    def comment(self):
        return self._comment
