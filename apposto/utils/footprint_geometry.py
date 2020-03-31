import numpy as np
from matplotlib.patches import Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


class FootprintGeometry():

    def __init__(self):
        self._patches = []
        self._xlim = 0
        self._zenithAngleInDeg = 0
        self._lgs = []
        self._ngs = []
        self._targets = []
        self._instrFoVInArcsec = None
        self._layerAltitudeInMeter = 10000
        self._telescopeRadiusInMeter = 4.1
        self._drawMetapupil = True

    def setInstrumentFoV(self, instrFoVInArcsec):
        self._instrFoVInArcsec = instrFoVInArcsec

    def set_zenith_angle(self, zenithAngleInDeg):
        self._zenithAngleInDeg = zenithAngleInDeg

    def setLayerAltitude(self, altitudeInMeter):
        self._layerAltitudeInMeter = altitudeInMeter

    def setTelescopeRadiusInMeter(self, radiusInMeter):
        self._telescopeRadiusInMeter = radiusInMeter

    def addLgs(self,
               launcherRadialDistanceInMeter,
               launcherAzInDeg,
               thetaSkyInArcsec,
               AzSkyInDeg):
        self._lgs.append(
            {'lRo': launcherRadialDistanceInMeter,
             'lAz': launcherAzInDeg,
             'skyTheta': thetaSkyInArcsec,
             'skyAz': AzSkyInDeg})

    def addNgs(self, thetaSkyInArcsec, AzSkyInDeg):
        self._ngs.append(
            {'lRo': 0,
             'lAz': 0,
             'skyTheta': thetaSkyInArcsec,
             'skyAz': AzSkyInDeg})

    def addTarget(self, thetaSkyInArcsec, AzSkyInDeg):
        self._targets.append(
            {'lRo': 0,
             'lAz': 0,
             'skyTheta': thetaSkyInArcsec,
             'skyAz': AzSkyInDeg})

    def compute(self):
        self._lgsL = []
        if self._lgs:
            for l in self._lgs:
                ll = self._polToRect(l['lRo'], l['lAz'])
                cc = self._centerOffset(l['skyTheta'], l['skyAz'],
                                        self._layerAltitudeInMeter)
                ra = self._lgsRadius()
                self._lgsL.append(FootprintXYRadius(
                    ll[0] + cc[0], ll[1] + cc[1], ra))

        self._ngsL = []
        if self._ngs:
            for l in self._ngs:
                cc = self._centerOffset(l['skyTheta'], l['skyAz'],
                                        self._layerAltitudeInMeter)
                ra = self._telescopeRadiusInMeter
                self._ngsL.append(FootprintXYRadius(cc[0], cc[1], ra))

        self._targetsL = []
        for l in self._targets:
            cc = self._centerOffset(l['skyTheta'], l['skyAz'],
                                    self._layerAltitudeInMeter)
            ra = self._telescopeRadiusInMeter
            self._targetsL.append(FootprintXYRadius(cc[0], cc[1], ra))

        self._sciL = []
        if self._instrFoVInArcsec:
            for sciFovCornerDeg in [45, 135, 225, 315]:
                cc = self._centerOffset(
                    self._instrFoVInArcsec / 2 * np.sqrt(2),
                    sciFovCornerDeg,
                    self._layerAltitudeInMeter)
                ra = self._telescopeRadiusInMeter
                self._sciL.append(
                    np.array([cc[0], cc[1], ra, sciFovCornerDeg]))

        self._metapupilL = []
        if self._lgs:
            raLgs = np.max([np.linalg.norm((l.x, l.y)) + l.r
                            for l in self._lgsL])
        if self._ngs:
            raNgs = np.max([np.linalg.norm((l.x, l.y)) + l.r
                            for l in self._ngsL])

        raTargets = np.max([np.linalg.norm((l.x, l.y)) + l.r
                            for l in self._targetsL])
        if not self._lgs:
            ra = np.max([raNgs, raTargets])
        elif not self._ngs:
            ra = np.max([raLgs, raTargets])
        else:
            ra = np.max([raLgs, raNgs, raTargets])

        self._metapupilL.append(FootprintXYRadius(0, 0, ra))

        self._computePatches()

    def getNgsFootprint(self):
        return self._ngsL

    def getLgsFootprint(self):
        return self._lgsL

    def getTargetFootprint(self):
        return self._targetsL

    def getMetapupilFootprint(self):
        return self._metapupilL

    def _computePatches(self):
        self._patches = []
        self._xlim = 0
        for l in self._lgsL:
            self._addAnnularFootprint(
                l.x, l.y, l.r, 0. * l.r, color='y', alpha=0.4)
        for l in self._ngsL:
            self._addAnnularFootprint(
                l.x, l.y, l.r, 0.99 * l.r, color='r', alpha=0.1)
        for l in self._targetsL:
            self._addAnnularFootprint(
                l.x, l.y, l.r, 0.99 * l.r, color='b', alpha=0.5)
        for l in self._sciL:
            self._addAnnularFootprint(
                l[0], l[1], l[2], 0.99 * l[2],
                theta1=l[3] - 10, theta2=l[3] + 10, color='b')
        if self._drawMetapupil:
            for l in self._metapupilL:
                self._addAnnularFootprint(
                    l.x, l.y, l.r, 0.99 * l.r, color='k')

    def scienceFieldRadius(self, rTel, fovInArcsec, hInMeter):
        return rTel + fovInArcsec / 2 * 4.848e-6 * hInMeter

    def _polToRect(self, ro, azInDeg):
        azInRad = azInDeg * np.pi / 180
        return ro * np.array(
            [np.cos(azInRad), np.sin(azInRad)])

    def _centerOffset(self, thetaInArcsec, azInDeg, hInMeter):
        return self._polToRect(
            thetaInArcsec * 4.848e-6 * hInMeter, azInDeg)

    def _lgsDistance(self):
        return 90000 / np.cos(self._zenithAngleInDeg * np.pi / 180)

    def _lgsRadius(self):
        return self._telescopeRadiusInMeter * (
            1 - self._layerAltitudeInMeter / self._lgsDistance())

    def _addAnnularFootprint(
            self, centerX, centerY,
            radiusOut, radiusIn=0,
            theta1=0, theta2=360,
            color='b', alpha=1):
        center = np.array([centerX, centerY])
        self._patches.append(
            Wedge(center, radiusOut, theta1, theta2,
                  width=(radiusOut - radiusIn), color=color, alpha=alpha))
        self._xlim = np.maximum(self._xlim,
                                np.max(np.abs(center) + radiusOut))

    def _addRectangle(
            self, centerX, centerY,
            sideX, sideY,
            color='b', alpha=0.1):
        bl = np.array([int(centerX - sideX // 2), int(centerY - sideY // 2)])
        tl = np.array([int(centerX - sideX // 2), int(centerY + sideY // 2)])
        br = np.array([int(centerX + sideX // 2), int(centerY - sideY // 2)])
        tr = np.array([int(centerX + sideX // 2), int(centerY + sideY // 2)])
        vertexes = np.vstack((bl, tl, tr, br))
        self._patches.append(
            Polygon(vertexes, closed=True, color=color, alpha=alpha))
        self._xlim = np.maximum(
            self._xlim,
            np.linalg.norm(np.array([centerX, centerY])) +
            0.5 * np.linalg.norm(np.array([sideX, sideY])))

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-self._xlim, self._xlim)
        ax.set_ylim(-self._xlim, self._xlim)
        p = PatchCollection(self._patches, match_original=True)
        ax.add_collection(p)
        ax.set_title('Footprints @ %g km, zenith= %g°' % (
            self._layerAltitudeInMeter / 1000, self._zenithAngleInDeg))
        if self._instrFoVInArcsec:
            plt.text(0.5, 0.05,
                     'Science FoV %dx%d"' % (
                         self._instrFoVInArcsec,
                         self._instrFoVInArcsec),
                     color='b',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
        if self._lgs:
            plt.text(0.15, 0.05,
                     'LGS @ %d"' % self._lgs[0]['skyTheta'],
                     color='y',
                     horizontalalignment='center',
                     verticalalignment='center', transform=ax.transAxes)
        if self._ngs:
            plt.text(0.85, 0.05,
                     'NGS @ %d"' % self._ngs[0]['skyTheta'],
                     color='r',
                     horizontalalignment='center',
                     verticalalignment='center', transform=ax.transAxes)
        plt.show()

    def report(self):
        if self._lgs:
            for l in self._lgsL:
                print("LGS (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))
        if self._ngs:
            for l in self._ngsL:
                print("NGS (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))
        for l in self._targetsL:
            print("Targets (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))
        for l in self._sciL:
            print("Science (x,y,r): %f, %f - %f" % (l[0], l[1], l[2]))
        for l in self._metapupilL:
            print("Metapupil (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))


def mainFootprintGeometry(h=12000, lgsTh=15, lgsN=4, ngsTh=60, ngsN=3,
                          sciFov=20,
                          targets=[[0, 0], [60 * 1.414, 45]],
                          rTel=4.1, zenith_angle=0):
    fg = FootprintGeometry()
    fg.setTelescopeRadiusInMeter(rTel)
    fg.set_zenith_angle(zenith_angle)
    fg.setInstrumentFoV(sciFov)
    fg.setLayerAltitude(h)
    if lgsN > 0:
        for azAng in np.arange(0, 360, 360 / lgsN):
            fg.addLgs(0, 0, lgsTh, azAng)
    if ngsN > 0:
        for azAng in np.arange(0, 360, 360 / ngsN):
            fg.addNgs(ngsTh, azAng)
    for t in targets:
        fg.addTarget(t[0], t[1])
    fg.compute()
    fg.plot()
    fg.report()
    return fg


class FootprintXYRadius():

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
