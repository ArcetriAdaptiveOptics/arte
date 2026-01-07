Time Series Analysis
====================

Overview
--------

The **time_series** module provides classes and functions for analyzing temporal data,
particularly useful for adaptive optics telemetry and signal processing:

- **TimeSeries**: Container class for temporal data with time stamps, units, and metadata
- **MultiTimeSeries**: Handle multiple synchronized time series
- **Indexer**: Index and access time series data efficiently
- **PSD Computer**: Power spectral density analysis and temporal frequency analysis

These tools support:

- Data storage with proper time stamps and physical units
- Resampling and interpolation
- Statistical analysis (mean, variance, temporal structure)
- Frequency domain analysis
- Data merging and synchronization

API Reference
-------------

For detailed API documentation of all submodules, see:

.. toctree::
   :maxdepth: 1

   time_series_submodules
