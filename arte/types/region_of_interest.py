import numpy as np


class RegionOfInterest(object):

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin= int(xmin)
        self.xmax= int(xmax)
        self.ymin= int(ymin)
        self.ymax= int(ymax)


    def toNumpyArray(self):
        return np.array([self.xmin, self.xmax, self.ymin, self.xmax])


    def __repr__(self):
        return "[%d:%d, %d:%d]" % (self.xmin, self.xmax, self.ymin, self.xmax)


    def cutOut(self, frame):
        return frame[self.ymin: self.ymax, self.xmin: self.xmax]
