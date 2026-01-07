"""Data elaboration framework for adaptive optics laboratory data.

This module provides a framework for analyzing time-series data from adaptive
optics systems. It implements a pattern-based architecture with three main
components working together:

1. **FileWalker**: Manages directory structure and file paths
2. **DataLoader**: Implements lazy loading of data from files
3. **TimeSeries**: Wraps time-series data with metadata and analysis methods
4. **Analyzer**: Coordinates data loading and provides analysis interface

The framework is designed for datasets organized by "tags" (unique identifiers,
typically timestamps) where each tag contains:

- Snapshot of system state (slow-varying parameters)
- Time series of fast-varying data (camera frames, DM commands, slopes)
- Calibration data (reconstructor matrices, reference frames)

Key Features
------------

- **Lazy loading**: Data loaded only when accessed
- **Disk caching**: Computed results cached to disk for performance
- **Unit handling**: Automatic unit conversion using astropy.units
- **Flexible indexing**: Select data subsets (e.g., x/y slopes, quadrants)
- **Batch analysis**: Analyze multiple tags together via AnalyzerSet

Basic Usage
-----------

1. Derive from AbstractFileNameWalker to define your file structure
2. Create TimeSeries subclasses for your data types
3. Derive from BaseAnalyzer to create your main analyzer
4. Use Analyzer.get(tag) to load and analyze data

See Also
--------
arte.dataelab.base_analyzer : Main analyzer base class
arte.dataelab.base_timeseries : Time series base class
arte.dataelab.base_file_walker : File walker base class
"""