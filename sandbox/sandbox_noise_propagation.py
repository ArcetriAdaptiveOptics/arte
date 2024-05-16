import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.noise_propagation import NoisePropagationZernikeGradientWFS

class TestNoisePropagation(object):

    def one(self):
        import matplotlib.pyplot as plt
        np1_3= NoisePropagationZernikeGradientWFS(1, np.arange(2, 3))
        np1_4= NoisePropagationZernikeGradientWFS(1, np.arange(2, 4))
        np1_5= NoisePropagationZernikeGradientWFS(1, np.arange(2, 5))
        np1_6= NoisePropagationZernikeGradientWFS(1, np.arange(2, 6))
        plt.plot(np.diag(np1_6.noise_covariance_matrix))
        plt.plot(np.diag(np1_5.noise_covariance_matrix))
        plt.plot(np.diag(np1_4.noise_covariance_matrix))
        plt.plot(np.diag(np1_3.noise_covariance_matrix))
        plt.show()
        print("1 subap 1 modes : %g" % np1_3.noise_total_variance)
        print("1 subap 2 modes : %g" % np1_4.noise_total_variance)
        print("1 subap 3 modes : %g" % np1_5.noise_total_variance)
        print("1 subap 4 modes : %g" % np1_6.noise_total_variance)



    def two(self, sigmas=4):
        import matplotlib.pyplot as plt
        np2_2= NoisePropagationZernikeGradientWFS(2, np.arange(2, 3))
        np2_3= NoisePropagationZernikeGradientWFS(2, np.arange(2, 4))
        np2_4= NoisePropagationZernikeGradientWFS(2, np.arange(2, 5))
        np2_5= NoisePropagationZernikeGradientWFS(2, np.arange(2, 6))
        np2_6= NoisePropagationZernikeGradientWFS(2, np.arange(2, 7))
        np2_7= NoisePropagationZernikeGradientWFS(2, np.arange(2, 8))
        np2_8= NoisePropagationZernikeGradientWFS(2, np.arange(2, 9))
        np2_9= NoisePropagationZernikeGradientWFS(2, np.arange(2, 10))
        np2_10= NoisePropagationZernikeGradientWFS(2, np.arange(2, 11))
        np2_20= NoisePropagationZernikeGradientWFS(2, np.arange(2, 21))
        plt.plot(np.diag(np2_20.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_10.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_9.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_8.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_7.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_6.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_5.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_4.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_3.noise_covariance_matrix) * sigmas)
        plt.plot(np.diag(np2_2.noise_covariance_matrix) * sigmas)
        plt.show()
        print("2 subap 1 modes : %g" % np2_2.noise_total_variance)
        print("2 subap 2 modes : %g" % np2_3.noise_total_variance)
        print("2 subap 3 modes : %g" % np2_4.noise_total_variance)
        print("2 subap 4 modes : %g" % np2_5.noise_total_variance)
        print("2 subap 5 modes : %g" % np2_6.noise_total_variance)
        print("2 subap 6 modes : %g" % np2_7.noise_total_variance)
        print("2 subap 7 modes : %g" % np2_8.noise_total_variance)
        print("2 subap 8 modes : %g" % np2_9.noise_total_variance)
        print("2 subap 9 modes : %g" % np2_10.noise_total_variance)
        print("2 subap 19 modes : %g" % np2_20.noise_total_variance)


    def total_variance(self, n_subaps):
        for i in np.arange(3, 2*n_subaps**2):
            modesV = np.arange(2, i)
            noip = NoisePropagationZernikeGradientWFS(n_subaps, modesV)
            print("%d %d: %g" % (
                n_subaps, len(modesV), noip.noise_total_variance))

    def noiseRG(self, radial_order):
        return -2.05 * np.log10(radial_order + 1) - 0.53


    def noiseRG2(self, radial_order):
        return -2.0 * np.log10(radial_order + 1) - 0.76


    def plotRigautAndGendronNoisePropagation(self, noiseProp):
        import matplotlib.pyplot as plt
        from arte.utils import zernike_generator as zg

        modesV= noiseProp.modesVector
        pi= np.diag(noiseProp.noise_covariance_matrix)
        n= zg.ZernikeGenerator.radial_order(modesV)
        plt.plot(np.log10(n + 1), np.log10(pi / pi[0]))
        plt.plot(np.log10(n + 1), self.noiseRG(n), '.-')
        plt.plot(np.log10(n + 1), self.noiseRG2(n), '.-')



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
