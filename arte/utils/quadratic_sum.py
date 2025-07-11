'''
Created on Oct 16, 2018

@author: lbusoni
'''
import numpy as np


def quadraticSum(arrayOfErrorsWithSign):
    ''' quadraticSum(arrayOfErrorsWithSign)
    Sum in quadrature of errors, considering sign

    5 = quadraticSum([3, 4])
    8 = quadraticSum([10, -6])

    Inputs can be numpy arrays
    '''
    total = 0.
    for err in arrayOfErrorsWithSign:
        total += np.sign(err) * err ** 2
    return np.sign(total) * np.sqrt(np.abs(total))
