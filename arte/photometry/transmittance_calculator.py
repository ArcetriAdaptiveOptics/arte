import numpy as np


def interface_glass_to_glass(n1, n2):
    '''
    Compute reflectance of a glass, considering the refractive index of the
    medium the light is coming from (n1) and the refractive index of the glass
    (n2)
    '''
    return (n1 - n2) ** 2 / (n1 + n2) ** 2


def external_transmittance_calculator(l1, l2, t1, a):
    '''
    Use external transmittance data of a glass with thickness l2 to compute
    external transmittance of a same glass but with different thickness l1.
    The computation is based on the equation for the external transmittance:
    
    T = (1 - R)**2 * exp(-a * l)
    
    where R is the reflectance, a is the attenuation coefficient and l is the
    thickness of the glass. If we consider two different values of thickness, 
    thus of transmittance, we can compute the unknown transmittance T2 as:
    
    T2 = T1 *  exp(-a * (l2 - l1))

    '''
    t2 = t1 * np.exp(-a * (l2 - l1))
    return t2


def internal_transmittance_calculator(l1, l2, t1):
    '''
    Use internal transmittance data t1 of a glass with thickness l1 to compute 
    internal transmittance t2 of a same glass but with different thickness l2.
    The transmittance is computed with the following equation:
    
        t2 = t1**(l2/l1)
    '''
    t2 = t1 ** (l2 / l1)
    return t2


def attenuation_coefficient_calculator(l1, l2, t1, t2):
    '''
    Compute attenuation coefficient of a glass from external transmittance data
    of two different values of thickness.
    The computation is based on the equation for the external transmittance
    of a glass:
    
    T = (1 - R)**2 * exp(-a * l)
    
    where R is the reflectance, a is the attenuation coefficient and l is the
    thickness of the glass. If we consider two different values of thickness, 
    thus of transmittance, we can compute the ratio between the transmittances:
    
    T1 / T2 = exp(-a * (l1 - l2))
    
    and from this equation we can derive the attenuation coefficient as:
    
    a = (lnT2 - lnT1) / (l1 - l2)
    '''
    a = (np.log(t2) - np.log(t1)) / (l1 - l2)
    return a


def internal_transmittance_from_external_one(t_ext1, t_ext2, l1, l2):
    '''
    Compute the internal transmittance of a substrate with thickness = l2, knowing 
    the external transmittances for both l1 and l2.

    Considering that the ratio between external transmittances and internal ones is the same:

    T_ext(l1) / T_ext(l2) = ((1 - R)**2 * exp(-a * l1)) / ((1 - R)**2 * exp(-a * l2))
                          = exp(-a * l1)) / exp(-a * l2))
                          = T_int(l1) / T_int(l2)

    and considering the relationship between internal transmittances:

    T_int(l1) = T_int(l2)**(l1/l2)

    we can compute:

    T_ext(l1) / T_ext(l2) = T_int(l2)**((l1 - l2) / l2)

    thus:

    T_int(l2) = (T_ext(l1) / T_ext(l2))**(l2 / (l1 - l2))  
    '''

    t_int2 = (t_ext1 / t_ext2) ** (l2 / (l1 - l2))
    return t_int2


