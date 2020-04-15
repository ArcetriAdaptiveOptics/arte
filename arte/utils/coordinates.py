import numpy as np


__version__= "$Id: $"



def xCoordinatesMap(sizeInPoints, pixelSize):
    sizeInMeters= sizeInPoints * pixelSize
    x=np.linspace(-(sizeInMeters - pixelSize) / 2,
                  (sizeInMeters - pixelSize) / 2,
                  sizeInPoints)
    return np.tile(x, (sizeInPoints, 1))




