Data Types
==========

Overview
--------

The **types** module defines specialized data types used throughout the arte library.
These classes provide structured representations of optical and adaptive optics concepts:

**Geometric Types:**

- **Aperture**: Telescope pupil definitions (circular, annular, segmented)
- **Mask**: Binary masks for pupil and image plane operations
- **DomainXY**: 2D coordinate systems and spatial domains
- **RegionOfInterest**: Define subregions for analysis

**Wavefront Representations:**

- **Wavefront**: Phase maps with physical units and metadata
- **ZernikeCoefficients**: Modal decomposition in Zernike basis
- **Slopes**: Wavefront sensor measurements (local gradients)
- **FisbaMeasure**: FISBA interferometer measurements

**Control and Calibration:**

- **GuideSource**: Natural or laser guide star definitions
- **ScalarBidimensionalFunction**: Generic 2D scalar fields

These types ensure consistency, proper unit handling, and clear interfaces throughout
the adaptive optics simulation and analysis pipeline.

API Reference
-------------

For detailed API documentation of all submodules, see:

.. toctree::
   :maxdepth: 1

   types_submodules
