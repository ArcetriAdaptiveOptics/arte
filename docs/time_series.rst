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

Fluent API
----------

The TimeSeries class provides a modern **chainable fluent API** for data analysis:

**Chainable Properties:**

- **Ensemble reductions**: ``ensemble_rms``, ``ensemble_mean``, ``ensemble_std``, ``ensemble_median``, ``ensemble_ptp``
- **Time reductions**: ``time_mean``, ``time_std``, ``time_rms``, ``time_median``, ``time_ptp``
- **Value extraction**: ``.value`` property (like pandas)
- **Filtering**: ``filter(**kwargs)`` and ``with_times(times)``

**Example:**

.. code-block:: python

    # Chainable operations
    mean_rms = ts.filter(modes=[2,3,4]).ensemble_rms.time_mean.value
    
    # Bidirectional chaining
    long_exposure = ts.time_mean.ensemble_rms.value

See the :doc:`tutorial notebook <notebook/time_series/time_series_examples>` for detailed examples.

**Legacy API:**

Old methods (``ensemble_rms()``, ``time_average()``, etc.) have been renamed with ``get_`` prefix
and marked deprecated. Use the new property-based API for new code.

API Reference
-------------

For detailed API documentation of all submodules, see:

.. toctree::
   :maxdepth: 1

   time_series_submodules
