import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt


class Coronograph(object):
    """ Abstract class to simulate coronographic PSFs """
    
    @abstractmethod
    def _get_pupil_mask(self, field):
        """ Override this method with the function
        returning the coronograph pupil-plane mask """

    @abstractmethod
    def _get_focal_plane_mask(self, field):
        """ Override this method with the function
        returning the coronograph focal-plane mask """

    def _get_apodizer(self):
        """ Override this method if the coronograph
        has an apodizer mask """
        return 1.0

    def get_coronographic_psf(self, input_field, oversampling:int, lambdaInM:float=None):
        """
        Function to obtain the post-coronographic PSF.

        Parameters
        ----------
        input_field: numpy.ndarray(complex) [N,N]
            Array of complex values representing the pupil-plane
            electric field upstream the coronograph.
            - Real part: electric field amplitude
            - Imaginary part: electric field phase in RADIANS

        oversampling: int
            Integer for the number of pixels to consider for each lambda/D.

        lambdaInM: float (None)
            Wavelength in meters at which the PSF is evaluated.
            This must be the same as the wavelength at which the 
            input_field phase was computed.
            Default is None and the coronograph uses it reference
            wavelength: i.e. we assume that the coronograph was
            optimized for the wavelength at which the input field 
            is given.

        Returns
        -------
        psf: numpy.ndarray(float) [N*oversampling,N*oversampling]
            The array representing the (padded) PSF.
            Each pixel in the array corresponds to oversampling lambda/D
        """
        self.oversampling = oversampling
        self.lambdaInM = lambdaInM
        pad_width = int(max(input_field.shape)*(self.oversampling-1))//2
        self._apodizer = self._get_apodizer()
        apodized_field = input_field * self._apodizer
        padded_field = np.pad(apodized_field,pad_width=pad_width,mode='constant',constant_values=0.0)
        self._focal_field = np.fft.fftshift(np.fft.fft2(padded_field))
        self._focal_mask = self._get_focal_plane_mask(self._focal_field)
        prop_focal_field = self._focal_field * self._focal_mask
        pupil_field = np.fft.ifft2(np.fft.ifftshift(prop_focal_field))
        self._pupil_mask = self._get_pupil_mask(padded_field)
        coro_field = pupil_field * self._pupil_mask
        self._focal_coro_field = np.fft.fftshift(np.fft.fft2(coro_field))
        psf = abs(self._focal_coro_field)**2
        return psf
    
    def show_coronograph_prop(self, maxLogPsf=None):
        """ Simple function to show the EF propagation through the coronograph """
        phase = np.angle(self._focal_mask)
        fcmap = 'RdBu'
        phase += 2*np.pi * (phase < 0.0)
        if np.max(phase) <= 1e-12:
            fcmap = 'grey'
            phase = self._focal_mask.copy()
        plt.figure(figsize=(22,4))
        plt.subplot(1,4,1)
        plt.imshow(phase,cmap=fcmap,origin='lower')
        plt.title('Focal plane mask')
        plt.colorbar()
        plt.subplot(1,4,2)
        self.showZoomedPSF(np.abs(self._focal_field)**2,
                           1/self.oversampling,title='PSF at focal mask',
                           maxLogVal=maxLogPsf)
        plt.subplot(1,4,3)
        plt.imshow(np.abs(self._pupil_mask),cmap='grey',origin='lower')
        plt.title('Pupil stop')
        plt.subplot(1,4,4)
        self.showZoomedPSF(np.abs(self._focal_coro_field)**2,
                           1/self.oversampling,title='Coronographic PSF',
                           maxLogVal=maxLogPsf)

    @staticmethod
    def showZoomedPSF(image, pixelSize, maxLogVal = None, title='',
                   xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$', zlabel=''):
        imageHalfSizeInPoints= image.shape[0]/2
        roi= [int(imageHalfSizeInPoints*0.8), int(imageHalfSizeInPoints*1.2)]
        imageZoomedLog= np.log(image[roi[0]: roi[1], roi[0]:roi[1]])
        if maxLogVal is None:
            maxLogVal = np.max(imageZoomedLog)
        imageZoomedLog -= maxLogVal
        sz=imageZoomedLog.shape
        plt.imshow(imageZoomedLog, 
                extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                    -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
                    cmap='twilight',vmin=-24)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar= plt.colorbar()
        cbar.ax.set_ylabel(zlabel)