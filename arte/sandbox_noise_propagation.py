import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.noise_propagation import NoisePropagation

class TestNoisePropagation(object):

    def one(self):
        import matplotlib.pyplot as plt
        np1_2= NoisePropagation(1, np.arange(1, 3))
        np1_3= NoisePropagation(1, np.arange(1, 4))
        np1_4= NoisePropagation(1, np.arange(1, 5))
        np1_5= NoisePropagation(1, np.arange(1, 6))
        plt.plot(np.diag(np1_5.noisePropagationMatrix))
        plt.plot(np.diag(np1_4.noisePropagationMatrix))
        plt.plot(np.diag(np1_3.noisePropagationMatrix))
        plt.plot(np.diag(np1_2.noisePropagationMatrix))
        plt.show()



    def two(self, sigmas=4):
        import matplotlib.pyplot as plt
        np2_2= NoisePropagation(2, np.arange(1, 3))
        np2_3= NoisePropagation(2, np.arange(1, 4))
        np2_4= NoisePropagation(2, np.arange(1, 5))
        np2_5= NoisePropagation(2, np.arange(1, 6))
        np2_6= NoisePropagation(2, np.arange(1, 7))
        np2_7= NoisePropagation(2, np.arange(1, 8))
        np2_8= NoisePropagation(2, np.arange(1, 9))
        np2_9= NoisePropagation(2, np.arange(1, 10))
        np2_10= NoisePropagation(2, np.arange(1, 11))
        np2_20= NoisePropagation(2, np.arange(1, 21))
        plt.plot(np.diag(np2_20.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_10.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_9.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_8.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_7.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_6.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_5.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_4.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_3.noisePropagationMatrix) * sigmas)
        plt.plot(np.diag(np2_2.noisePropagationMatrix) * sigmas)
        plt.show()


    def noiseRG(self, radialOrder):
        return -2.05 * np.log10(radialOrder + 1) - 0.53


    def noiseRG2(self, radialOrder):
        return -2.0 * np.log10(radialOrder + 1) - 0.76


    def plotRigautAndGendronNoisePropagation(self, noiseProp):
        import matplotlib.pyplot as plt
        from arte.utils import zernike_generator as zg

        modesV= noiseProp.modesVector
        pi= np.diag(noiseProp.noisePropagationMatrix)
        n= zg.ZernikeGenerator.radialOrder(modesV)
        plt.plot(np.log10(n + 1), np.log10(pi / pi[0]))
        plt.plot(np.log10(n + 1), self.noiseRG(n))
        plt.plot(np.log10(n + 1), self.noiseRG2(n))



    def _makeMovie(
            self, imageCube, outputFile=None, fps=10,
            extent=[-1, 1, -1, 1], title='Title'):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib import cm

        fig= plt.figure()
        ax= plt.gca()
        vmin= imageCube.min()
        vmax= imageCube.max()
        im = plt.imshow(imageCube[0], vmin=vmin, vmax=vmax,
                        animated=True, extent=extent,
                        cmap=cm.seismic)
        txt= ax.text(0.02, 0.95, 'frame: 0', transform=ax.transAxes)
        cb= plt.colorbar()

        def updatefig(idx):
            im.set_array(imageCube[idx])
            txt.set_text('frame: %d' % idx)
            return im, txt

        ani = animation.FuncAnimation(
            fig, updatefig,
            np.arange(1, len(imageCube)), interval=50, blit=True)
        if outputFile:
            ani.save(outputFile, fps=fps)
        plt.show()


    def makeMovieOfPhase(self, nModes, outputFile='phase.mp4'):
        zg=ZernikeGenerator(200)
        imas= np.swapaxes(
            np.dstack([
                zg.getZernike(m).filled(0) for m in np.arange(1, nModes)]
            ), 2, 0)
        self._makeMovie(imas, outputFile=outputFile, fps=10)


    def makeMovieOfIntmat(self, noiseProp, outputFile='intmat.mp4'):
        imas= np.swapaxes(
            np.dstack([
                noiseProp.slopesMapForMode(m)
                for m in np.arange(1, noiseProp.nModes)]
            ), 2, 0)
        self._makeMovie(imas, outputFile=outputFile,
                        extent=[-2, 2, -1, 1], fps=10)
        return imas


    def makeMovieOfLeftSingularValues(self,
                                      noiseProp,
                                      outputFile='lsv.mp4'):
        imas= np.swapaxes(
            np.dstack([
                noiseProp.leftSingularVectorMapForMode(m)
                for m in np.arange(1, noiseProp.nModes)]
            ), 2, 0)
        self._makeMovie(imas, outputFile=outputFile,
                        extent=[-2, 2, -1, 1], fps=10)
        return imas


    def makeMovieOfRightSingularValues(self,
                                       noiseProp,
                                       outputFile='rsv.mp4'):
        zg=ZernikeGenerator(200)

        def getRightSingularMode(mIdx):
            wf= 0.* zg.getZernike(1)
            for (m, c) in zip(noiseProp.modesVector,
                              noiseProp.v[mIdx, :]):
                wf+= c* zg.getZernike(m)
            return wf

        imas= np.swapaxes(
            np.dstack([
                getRightSingularMode(mIdx)
                for mIdx in np.arange(len(noiseProp.modesVector))]
            ), 2, 0)
        self._makeMovie(imas, outputFile=outputFile,
                        fps=10)
        return imas
