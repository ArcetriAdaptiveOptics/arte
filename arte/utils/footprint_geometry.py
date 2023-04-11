import numpy as np
from matplotlib.patches import Wedge, Polygon, RegularPolygon
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
        self._dms = []
        self._instrFoVInArcsec = None
        self._layerAltitudeInMeter = 10000
        self._telescopeRadiusInMeter = 4.1
        self._drawMetapupil = True

    def setInstrumentFoV(self, instrFoVInArcsec, instrRotInDeg=0):
        self._instrFoVInArcsec = instrFoVInArcsec
        self._instrRotInDeg = instrRotInDeg

    def set_zenith_angle(self, zenithAngleInDeg):
        self._zenithAngleInDeg = zenithAngleInDeg

    def setLayerAltitude(self, altitudeInMeter):
        self._layerAltitudeInMeter = altitudeInMeter

    def setTelescopeRadiusInMeter(self, radiusInMeter):
        self._telescopeRadiusInMeter = radiusInMeter

    def addLgs(self,
               thetaSkyInArcsec,
               AzSkyInDeg):
        self._lgs.append(
            {'lRo': 0,
             'lAz': 0,
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

    def addDm(self, pitchOnLayer, nActs, size, rotationAngleInDeg):
        self._dms.append(
            {'lRo': 0,
             'lAz': 0,
             'pitch': pitchOnLayer,
             'nActs': nActs,
             'size': size,
             'skyRot': rotationAngleInDeg})

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
            for sciFovCornerDeg in np.array([45, 135, 225, 315]
                                            ) + self._instrRotInDeg:
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
        else:
            raLgs = 0
        if self._ngs:
            raNgs = np.max([np.linalg.norm((l.x, l.y)) + l.r
                            for l in self._ngsL])
        else:
            raNgs = 0
        if self._targetsL:
            raTargets = np.max([np.linalg.norm((l.x, l.y)) + l.r
                                for l in self._targetsL])
        else:
            raTargets = 0
        ra = np.max([raLgs, raNgs, raTargets])

        self._metapupilL.append(FootprintXYRadius(0, 0, ra))

        self._dmsL = []
        if self._dms:
            for l in self._dms:
                self._dmsL.append(l)
#                 ll = self._polToRect(l['lRo'], l['lAz'])
#                 cc = self._centerOffset(l['skyTheta'], l['skyAz'],
#                                         self._layerAltitudeInMeter)
#                 ra = self._lgsRadius()
#                 self._dmsL.append(FootprintXYRadius(
#                     ll[0] + cc[0], ll[1] + cc[1], ra))

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
                l.x, l.y, l.r, 0.99 * l.r, color='y', alpha=0.5)
            self._addAnnularFootprint(
                l.x, l.y, l.r, 0.00 * l.r, color='y', alpha=0.1)
        for l in self._ngsL:
            self._addAnnularFootprint(
                l.x, l.y, l.r, 0.99 * l.r, color='r', alpha=0.5)
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
        for l in self._dmsL:
            self._addDmFootprint(l, color='k', alpha=0.3)

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

    def _addDmFootprint(self, l, color, alpha):
        rotAngleRad = l['skyRot'] * np.pi / 180
        for i in np.arange(-l['nActs'], l['nActs'] + 1):
            x = i * l['pitch']
            for j in np.arange(-l['nActs'], l['nActs'] + 1):
                y = j * l['pitch']
                xR = x * np.cos(rotAngleRad) - y * np.sin(rotAngleRad)
                yR = x * np.sin(rotAngleRad) + y * np.cos(rotAngleRad)
                self._addCrossFootprint(
                    xR, yR, l['size'], rotAngleRad, color, alpha)

    def _addCrossFootprint(
            self, centerX, centerY, size,
            thetaAngle, color, alpha):
        numVertices = 5
        center = np.array([centerX, centerY])
        self._patches.append(
            RegularPolygon(center, numVertices, size, thetaAngle,
                           color=color, alpha=alpha))
        # self._xlim = np.maximum(self._xlim,
        #                        np.max(np.abs(center) + size))

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
        self._computePatches()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-self._xlim, self._xlim)
        ax.set_ylim(-self._xlim, self._xlim)
        p = PatchCollection(self._patches, match_original=True)
        ax.add_collection(p)
        ax.set_title('Footprints @ %g km, zenith= %7.3f°' % (
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
        if self._targets:
            for l in self._targetsL:
                print("Targets (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))
        for l in self._sciL:
            print("Science (x,y,r): %f, %f - %f" % (l[0], l[1], l[2]))
        for l in self._metapupilL:
            print("Metapupil (x,y,r): %f, %f - %f" % (l.x, l.y, l.r))


def mainFootprintGeometry(h=12000, lgsTh=15, lgsN=4, ngsTh=60, ngsN=3,
                          sciFov=20,
                          targets=[[0, 0], [60 * 1.414, 45]],
                          rTel=4.1, zenith_angle=0,
                          dm_pitch=1.0, dm_acts=10,
                          dm_act_size=0.1, dm_rot_angle=0):
    fg = FootprintGeometry()
    fg.setTelescopeRadiusInMeter(rTel)
    fg.set_zenith_angle(zenith_angle)
    fg.setInstrumentFoV(sciFov)
    fg.setLayerAltitude(h)
    if lgsN > 0:
        for azAng in np.arange(0, 360, 360 / lgsN):
            fg.addLgs(lgsTh, azAng)
    if ngsN > 0:
        for azAng in np.arange(0, 360, 360 / ngsN):
            fg.addNgs(ngsTh, azAng)
    for t in targets:
        fg.addTarget(t[0], t[1])
    fg.addDm(dm_pitch, dm_acts, dm_act_size, dm_rot_angle)
    fg.compute()
    fg.plot()
    fg.report()
    return fg


class FootprintXYRadius():

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
