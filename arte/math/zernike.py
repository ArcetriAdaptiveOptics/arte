# -*- coding: utf-8 -*-
#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# avaz      2019-09-20  Translated from Arcetri's oaa_lib
# apuglisi  2020-04-10  Added Zernike class
#
#########################################################

import numpy as np
from scipy.special import jacobi


class Zernike():
    '''
    Holds a geometrical Zernike definition.

    After being initialized or resized with a (N,m,n) shape, the *get()*
    method will return a 3d numpy array with the first N zernikes
    (in Noll ordering, first element is piston) with *m*x*n* sampling.
    Data is not actually calculated until the first invocation of the
    *get()* method, which might take a while if the array is big.
    '''
    
    def __init__(self, shape, dtype=np.float32):
        self._data = np.zeros(shape, dtype=dtype)
        self._valid = False

    def resize(self, new_shape):
        '''
        After resizing, Zernike polynomials
        will be recalculated.
        '''
        if new_shape != self._data.shape:
            self._data.resize(new_shape)
            self._valid = False

    def get(self):
        if not self._valid:
            self._recalc()
        return self._data

    def __getitem__(self, idx):
        '''
        Implements self[idx] to return a single
        Zernike polynomials. Following the reference data model:
        https://docs.python.org/3/reference/datamodel.html#object.__getitem__
        TypeError is raised if idx is not an integer number, and
        IndexError is raised if idx is out of bounds.
        '''
        if not isinstance(idx, int):
            raise TypeError

        if idx < 0 or idx >= self._data.shape[0]:
            raise IndexError

        if not self._valid:
            self._recalc()

        return self._data[idx]

    def _recalc(self):
        
        nZern, m, n = self._data.shape
        grid = np.mgrid[0:m,0:n]
        for z in range(nZern):
            self._data[z,:,:] = zern(z+1, (grid[0]+0.5-m/2)/(m/2), (grid[1]+0.5-n/2)/(n/2))
            

def zern(j, x, y, polar=False):
    '''
 ; NAME:
;       ZERN
;
; PURPOSE:
;       The ZERN function returns the value of J-th Zernike polynomial
;       in the points of coordinates X,Y. The output is of the same type
;               as the X and Y vectors. The function convert X,Y
;               coordinates in polar coordinates and call ZERN_JPOLAR function.
;
; CALLING SEQUENCE:
;       Result = ZERN(J, X, Y [,polar=False])
;
; INPUTS:
;       J:      scalar of type integer or long. Index of the polynomial,
;                               J >= 1.
;       X:      n-element vector of type float or double. X coordinates.
;       Y:              n-element vector of type float or double. Y coordinates.
;       polar:  if True, x and y are expressed in polar coordinates.
;
; MODIFICATION HISTORY:
;       Written by:    A. Riccardi; March, 1995.
;       Modified by: A. Puglisi, Oct 2019
                  - translated to Python
                  - added "polar" flag
    '''
    if polar:
        return zern_jpolar(j, x, y)
    else:
        return zern_jpolar(j, np.sqrt(x*x+y*y), np.arctan2(y,x))

   


def zern_noll(j):
    """
    Given a Noll index j, returns n and m
    """
    n = np.int(0.5 * (np.sqrt(8*(j-1)+1)-1))
    cn = n*(n+1)/2+1
    m=(j-cn)
    if n%2==0:
        if m%2!=0:
            m+=1
    else:
        if m%2==0:
            m+=1
    if j%2!=0:
        m*=-1
    m=np.int(m)
    return n,m

def zern_degree(j):
    """
    Given an index j, returns n and m.
    Not sure if correct but this is just abs(Noll)
    """
    n,m = zern_noll(j)
    m=np.abs(m)
    return n,m

def zern_jradial(n,m,r):
    """
    from original idl:
    PURPOSE:
    ;       ZERN_JRADIAL returns the value of the radial portion R(N,M,Rho)
    ;           of the Zernike polynomial of radial degree N and azimuthal
    ;           frequency M in a vector R of radial points.
    ;           The radial Zernike polynomial is defined in the range (0,1)
    ;           and is computed as:
    ;
    ;                   R(N,M,Rho) = Rho^M * P_[(N-M)/2](0,M,2*Rho^2-1)
    ;
    ;           where P_k(a,b,x) is the Jacobi Polynomial of degree k (see
    ;           JACOBI_POL)
    ; INPUTS:
    ;       N:      scalar of type integer or long. N >= 0.
    ;       M:      scalar of type integer or long. 0 <= M <= N and M-N even.
    ;       Rho:    n_elements vector of type float or double.
    """

    k=(n-m)/2.0

    if type(r) is not np.ndarray:
        r = np.array(r)
    zr = r**m * jacobi(k, 0, m)(2*r**2-1)
    return zr

def zern_jpolar(j,r,theta):
    """

    ; INPUTS:
    ;       J:      index of the polynomial, integer J >= 1
    ;       R:      point to evaluate (polar coord.)
    ;       Theta:
    ;
    ; OUTPUTS:
    ;       ZERN_POLAR returns the value of j-th Zernike polynomial
    ;       in the point of polar coordinates r, theta.
    ;       If r>1 then return 0. On error return 0.
    """
    if j<1:
        print('zern_jpolar must have j>=1.')
        return 0
    #calculate n,m from j
    n,m = zern_degree(j)
    result = np.sqrt(n+1.0+r*0.0)*zern_jradial(n,m,r)

    if m==0:
        return result
    elif j%2==0:
        return np.sqrt(2)*result*np.cos(m*theta)
    else:
        return np.sqrt(2)*result*np.sin(m*theta) #??r[0]*0??


def get_u_zern(x,y,idx_list):
    n2=len(idx_list)
    m=len(x)

    u=np.ones([n2,m])
    rho = np.sqrt(x**2+y**2)
    theta = np.arctan(y,x)
    for i in range(n2-1):
        u[i,:] = zern_jpolar(idx_list[i], rho, theta)
    return u

def get_u_poly(x,y,idx_list):
    poly_ord = np.ceil(0.5*(np.sqrt(1+8*idx_list)-3))
    y_pow = idx_list - (poly_ord * (poly_ord+1)/2+1)
    x_pow = poly_ord - y_pow
    n2 = len(idx_list)
    m = len(x)

    u = np.ones([n2,m])
    for i in range(n2-1):
        u[i,:] = x**x_pow[i]*y**y_pow[i] #?
    return u

def surf_fit(x,y,z,idx_list,fn_type='zern'):
    if fn_type == 'zern':
        if [n>1.0 for n in [x**2 + y**2]]:
            print("Warning: the domain should be limited to a circle of unit radius")
        print("Beginning zernike fit.")
        get_u_zern(x,y,)

