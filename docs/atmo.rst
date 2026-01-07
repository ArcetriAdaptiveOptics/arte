Atmospheric turbulence module (`arte.atmo`)
=================================================

Introduction
------------

`arte.atmo` provides several functions to perform computation related to 
the atmospheric turbulence and its optical effects. 

The main class to represent the turbulence is :func:`~arte.atmo.cn2_profile`.

Random phase screen can be generated with is :func:`~arte.atmo.phase_screen_generator` 

Kolmogorov and Von Karman spectra are available :func:`~arte.atmo.von_karman_psd` 

Covariances and cross-power-spectral-densities of Von Karman turbulence 
are computed in :func:`~arte.atmo.von_karman_covariance_calculator`   


API Reference
-------------

For detailed API documentation of all submodules, see:

.. toctree::
   :maxdepth: 1

   atmo_submodules
