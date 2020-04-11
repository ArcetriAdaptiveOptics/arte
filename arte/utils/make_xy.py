
import numpy as np


def make_xy( sampling, ratio, dtype=None, polar=False, vector=False,
             zero_sampled=False, quarter=False, fft=False):
    '''
 
 NAME: 
       MAKE_XY 
 
 PURPOSE: 
       This procedure generates zero-centered domains in
       cartesian plane or axis, tipically for pupil sampling
       and FFT usage.
 
 CATEGORY: 
       Optics. 
 
 CALLING SEQUENCE: 
 
 
       x,y = make_xy( sampling, ratio)
       r, theta = make_xy( sampling, ratio, polar=True)
 
 INPUTS: 

       sampling:   integer scalar. Number of sampling points per dimension of
                   the domain. Sampling>=2.

       ratio:  floating scalar. Extension of sampled domain:
                   -Ratio <= x [,y] <= +Ratio

       polar:  if True, return domain sampling in polar coordinates.


 


 Notes:
 
       Output type (np.float32, np.float64, etc) will correspond
       to the highest precision of the input arguments sampling and ratio.




       POLAR:  set it to have domain sampling in polar coordinates.

       VECTOR: if set, 1-dimensional domain is sampled. Y is not needed in this
               case. If Y is passed it is left unchanged. If POLAR is set, the
              setting of the VECTOR keyword is ignored.

       ZERO_SAMPLED:   if set, origin of the domain is sampled. This
                       keyword is usefull to force the zero to be sampled
                       when sampling is even. When sampling is odd, the
                       zero is always sampled.

      QUARTER:    if set, only 1st quadrant is returned with (X>=0 AND Y>=0).
                 The array returned has:
                       if Sampling is even: Sampling/2 X Sampling/2 elements
                       if Sampling is odd:  (Sampling+1)/2 X (Sampling+1)/2 elements
                                          
       
       FFT:    if set, order the output values for FFT purposes.
                 For example:
                Sampling=4, Ratio=1 vector -3/4,-1/4,+1/4,+3/4 is returned as
                 +1/4,+3/4,-3/4,-1/4.
                 Sampling=4, Ratio=1, /ZERO_SAMPLING vector -1,-1/2,0,+1/2 is
                 returned as 0,1/2,-1,-1/2
                 Sampling=5, Ratio=1 vector -4/5,-2/5,0,+2/5,+4/5 is returned as
                 0,+2/5,+4/5,-4/5,-2/5 
 
 OUTPUTS: 
       X (or R):     floating vector or squared matrix. Returns X values of sampled
                     points. Radial R values if POLAR is set.
 
 OPTIONAL OUTPUTS: 
       Y (or Theta): floating squared matrix. Returns Y values of sampled points.
                     Azimuthal angle Theta values (in radians) if POLAR is set. If VECTOR
                     is set and Y is specified as input, a column vector of Y values is
                     returned.
 HOW THE DOMAIN IS SAMPLED:

   The concept is the following: considering an array of Sample x Sample pixels, the
   bottom-left corner of the bottom-left pixel has coordinates (-Ratio,-Ratio) and the
   top-right cornet of the to-left pixel has coordinates (+Ratio,+Ratio). The procedure
   returns the coordinates of the centers of the pixels. When Sample is even and 
   ZERO_SAMPLE is set, the coordinates of the bottom-left corners of the pixels are returned.
   
           -Sampling is even and ZERO_SAMPLING is not set-
           the edge of the domain is not sampled and the sampling is
          symmetrical respect to origin.

               Ex: Sampling=4, Ratio=1.
                   -1   -0.5    0    0.5    1    Domain (Ex. X axis)
                    |     |     |     |     |
                       *     *     *     *       Sampling points
                     -0.75 -0.25  0.25  0.75     Returned vector


          -Sampling is even and ZERO_SAMPLING is set-
           the lower edge is sampled and the sampling is not symmetrical
           respect to the origin.

              Ex: Sampling=4, Ratio=1.
                   -1   -0.5    0    0.5    1    Domain (Ex. X axis)
                    |     |     |     |     |
                    *     *     *     *          Sampling points
                   -1   -0.5    0    0.5         Returned vector



           -Sampling is odd (ZERO_SAMPLING is ignored)-
           the zero is always sampled.

               Ex: Sampling=5, Ratio=1.
                   -1   -3/5  -1/5   1/5   3/5    1    Domain (Ex. X axis)
                    |     |     |     |     |     |
                       *     *     *     *     *       Sampling points
                     -4/5  -2/5    0    2/5   4/5      Returned vector   

           If FFT keyword is set, output values are ordered for
          FFT purposes:

           Ex: 2-dimensional domain: N = Sampling
           X or Y(0:N/2-1, 0:N/2-1)   1st quadrant (including origin
                                      if ZERO_SAMPLEDis set)
           X or Y(N/2:N, 0:N/2-1)     2nd quadrant
           X or Y(N/2:N, N/2:N)       3rd quadrant
           X or Y(0:N/2-1, N/2:N)     2nd quadrant
 
 EXAMPLE: 
      Compute the squared absolute value of FFT of function
       (x+2*y)*Pupil(r) and display the result.
       Pupil(r)=1. if r=sqrt(x^2+Y^2)<=1., 0. otherwise.

       MAKE_XY, 256, 1., X, Y
       Pupil = X*X+Y*Y LE 1.
       TV_SCL, ABS(FFT((X+2*Y)*Pupil))^2

 MODIFICATION HISTORY: 
       Written by:     A. Riccardi; April, 1995. 
       01 March 2014   A. Riccardi: extended to odd Sampling. Rewritten to simplify the code.
                                    Ratio can be any positive number. No longer constrained
                                    to be larger than Sampling.
       08 September 2019 A. Puglisi: ported to Python, added dtype parameter
    ''' 

    if sampling <= 1:
        raise Exception('make_xy: sampling must be larger than 1')
    even_sampling = ((sampling % 2) == 0)

    if quarter:
       if even_sampling:
          size = sampling//2
          if zero_sampled:
              x0 = 0.0
          else:
              x0 = -0.5
       else:
          size = (sampling+1)//2
          x0 = 0.0
    else:
       size = sampling
       x0 = (sampling-1)/2.0
       if even_sampling and zero_sampled:
           x0 += 0.5

    x = (np.arange(size, dtype=dtype)-x0) / (sampling/2.0) * ratio
        
    if fft and not quarter:
        if even_sampling:
           x = np.roll(x, -sampling/2)
        else:
           x = np.roll(x, -(sampling-1)/2)

    y = None
    if polar or (not vector):
        x = np.tile(x, (size,1))
        y = np.transpose(x).copy()
        if polar:
           x,y = xy_to_polar(x,y)

    return x,y


def xy_to_polar(x,y):

    r=np.sqrt(x*x+y*y)
    y=np.arctan2(y,x)
    return r, y





