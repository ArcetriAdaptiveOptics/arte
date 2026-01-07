Utilities
=========

Overview
--------

The **utils** module provides a collection of general-purpose utilities used throughout
the arte library:

**Mathematical Tools:**

- **discrete_fourier_transform**: FFT operations and Fourier analysis
- **modal_decomposer**: Decomposition of 2D fields into orthogonal modes
- **radial_profile**: Extract radial profiles from 2D arrays
- **rebin**: Array rebinning and downsampling
- **zernike_generator**: Generate Zernike polynomials
- **zernike_projection_on_subaperture**: Project Zernike modes on subapertures
- **math**: General mathematical utilities
- **image_moments**: Compute image moments and centroids
- **generalized_fitting_error**: Fitting error calculations for adaptive optics
- **marechal**: Strehl ratio and Maréchal approximation
- **noise_propagation**: Propagate uncertainties through calculations
- **quadratic_sum**: Quadratic sum utilities for error propagation

**Data Structures:**

- **circular_buffer**: Fixed-size circular buffer for streaming data
- **shared_array**: Shared memory arrays for multiprocessing

**I/O and Logging:**

- **logger**: Logging utilities for simulations and analysis
- **capture_output**: Capture and redirect stdout/stderr
- **tabular_report**: Create formatted tabular reports

**Package Management:**

- **package_data**: Access package data files
- **locate**: Locate files and resources
- **help**: Enhanced help system and documentation access

**Analysis Tools:**

- **compareIDL**: Compare results with IDL implementations
- **footprint_geometry**: Geometric footprint calculations
- **paste**: Array pasting utilities
- **unit_checker**: Check physical unit consistency

**Utilities:**

- **decorator**: Useful decorators for caching, timing, etc.
- **executor**: Execute external programs
- **iterators**: Custom iterators
- **timestamp**: Time stamping utilities for data tracking
- **constants**: Physical and mathematical constants

API Reference
-------------

For detailed API documentation of all submodules, see:

.. toctree::
   :maxdepth: 1

   utils_submodules
