from astropy import units
from arte.utils.constants import Constants
from scipy.special import gamma
from arte.utils.zernike_generator import ZernikeGenerator
import numpy as np


class Seeing(object):

    def __init__(self, seeingAt500nm):
        self._seeingAt500nm = seeingAt500nm
        self._r0At500nm = 0.98 * 500 * units.nm / (
            seeingAt500nm * Constants.ARCSEC2RAD / units.arcsec)

    def to_r0(self, wavelength=500 * units.nm):
        return self._r0At500nm * (wavelength / (500 * units.nm)) ** (6. / 5)

def getKolmogorovCovarianceMatrix(j1, j2):
    n1, m1 = ZernikeGenerator.degree(j1)
    n2, m2 = ZernikeGenerator.degree(j2)

    # Check if m1 is not equal to m2 or if the sum of j1 and j2 is odd and m1 is not zero
    if m1 != m2 or ((not is_even(j1 + j2)) and m1 != 0):
        return 0.0

    # Calculate npn and nmn
    npn = (n1 + n2) // 2 - 1
    nmn = abs(n1 - n2) // 2

    # Calculate the result using the constant from Roddier 1990, divided by 2*pi to convert rad to waves
    result = 0.057494899 * (-1)**((n1 + n2 - 2 * m1) // 2) * np.sqrt((n1 + 1) * (n2 + 1))
    result *= gamma(1.0 / 6.0) / (gamma(17.0 / 6.0))**2 / gamma(29.0 / 6.0)
    c1 = 1.0
    c2 = 1.0

    # Calculate c1 and c2
    if npn > 0:
        for i in range(npn):
            c1 *= (1.0 / 6.0 + i) / (29.0 / 6.0 + i)
    if nmn > 0:
        for i in range(nmn):
            c2 *= (-11.0 / 6.0 + i) / (17.0 / 6.0 + i)

    return result * c1 * c2 * (-1)**nmn
    
def getFullKolmogorovCovarianceMatrix(nModes):
    covar = np.zeros((nModes, nModes))
    for i in range(2, nModes + 2):
        for j in range(2, nModes + 2):
            covar[i - 2, j - 2] = getKolmogorovCovarianceMatrix(i, j)
    return covar

# Helper function to check if a number is even
def is_even(num):
    return num % 2 == 0

