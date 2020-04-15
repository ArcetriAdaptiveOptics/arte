from astropy import units as u
from arte.utils.math import round_up_to_even


class ShackHartmannLensletParameters(object):
    def __init__(self,
                 telescope_diameter,
                 number_of_subapertures,
                 on_sky_field_of_view,
                 pupil_size_on_lenslet,
                 pixelscale):
        self._dTele= telescope_diameter
        self._nSub= number_of_subapertures
        self._skyFov= on_sky_field_of_view
        self._dPupilLensletArray= pupil_size_on_lenslet
        self._pixelscale= pixelscale


    @staticmethod
    def _round_to_even_n_pixels_per_subaps(ccdsize, n_subaps):
        return round_up_to_even(ccdsize/n_subaps)-2


    @staticmethod
    def fromSensorSize(telescope_diameter,
                       number_of_subapertures,
                       ccd_number_of_pixel,
                       ccd_pixel_size,
                       pixelscale):
        n_px_per_subaps= ShackHartmannLensletParameters.\
            _round_to_even_n_pixels_per_subaps(
                ccd_number_of_pixel, number_of_subapertures)
        pupil_size_on_lenslet= \
            n_px_per_subaps*number_of_subapertures*ccd_pixel_size
        on_sky_field_of_view= n_px_per_subaps*pixelscale
        return ShackHartmannLensletParameters(
            telescope_diameter,
            number_of_subapertures,
            on_sky_field_of_view,
            pupil_size_on_lenslet,
            pixelscale)

    def lenslet_focal_length(self):
        return self._dPupilLensletArray**2 / (
            self._nSub * self._dTele * self._skyFov.to(
                "", equivalencies=u.dimensionless_angles()))

    def pixelsize(self):
        return self.lenslet_diameter() / self.n_pixel_per_subaperture()

    def n_pixel_per_subaperture(self):
        return self._skyFov.to("", equivalencies=u.dimensionless_angles()
                               ) / self._pixelscale.to(
            "", equivalencies=u.dimensionless_angles())

    def lenslet_diameter(self):
        return self._dPupilLensletArray / self._nSub

    def lenslet_F_number(self):
        return self.lenslet_focal_length()/self.lenslet_diameter()

    def pupil_size_on_lenslet_array(self):
        return self._dPupilLensletArray

    def print_info(self):
        print("pupil size on lenslet array {0:0.04f}".format(
              self.pupil_size_on_lenslet_array().to(u.mm)))
        print("lenslet focal length {0:0.04f}".format(
            self.lenslet_focal_length().to(u.mm)))
        print("pixelsize %s" % self.pixelsize().to(u.micron))
        print("n of pixel per subap {0:0.04g}".format(
            self.n_pixel_per_subaperture()))
        print("lenslet diameter {0:0.04g}".format(
            self.lenslet_diameter().to(u.mm)))
        print("lenslet F# {0:0.04g}".format(
            self.lenslet_F_number().to('')))
        print("FoV {0:0.04g}".format(self._skyFov.to(u.arcsec)))
