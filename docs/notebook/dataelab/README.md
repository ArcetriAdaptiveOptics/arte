# dataelab Notebook Examples

This directory contains Jupyter notebook tutorials demonstrating the arte.dataelab framework.

## Available Notebooks

### dataelab_example.ipynb

Complete tutorial showing:
- Generation of synthetic time series data (modal coefficients)
- Saving data in FITS format
- Creating custom FileWalker and Analyzer classes
- Loading and analyzing data with BaseTimeSeries
- Computing time and ensemble statistics
- Power spectral density analysis
- Time filtering and data visualization

This notebook provides a minimal but complete example suitable for understanding the dataelab workflow without requiring large real-world datasets.

## Running the Notebooks

These notebooks are part of the arte documentation and can be viewed in the compiled documentation at `docs/_build/html/notebook/dataelab/`.

To run them interactively:

```bash
cd docs/notebook/dataelab
jupyter notebook dataelab_example.ipynb
```

Make sure arte is installed in your Python environment.
