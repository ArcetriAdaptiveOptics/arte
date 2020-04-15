'''
Created on Oct 16, 2018

@author: lbusoni
'''
import numpy as np


def quadraticSum(arrayOfErrorsWithSign):
    ''' quadraticSum(arrayOfErrorsWithSign)
    Sum in quadrature of errors, considering sign

    5 = quadraticSum([3, 4])
    8 = quadraticSum(10, -6])

    '''
    total= 0.
    for err in arrayOfErrorsWithSign:
        if err < 0:
            total -= err**2
        else:
            total += err**2
    return np.sqrt(total)
 