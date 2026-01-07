# Arte Documentation

This directory contains the Sphinx documentation for the arte package.

## Building the documentation

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install --file requirements.txt
```

### Building HTML documentation

```bash
make html
```

The generated documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Cleaning build artifacts

```bash
make clean
```

### Other formats

Sphinx supports multiple output formats:

- `make html` - HTML pages (default)
- `make latexpdf` - PDF via LaTeX (requires LaTeX installation)
- `make epub` - EPUB e-book format
- `make help` - Show all available formats

## ReadTheDocs

The documentation is automatically built and published on ReadTheDocs when changes are pushed to the repository. The configuration is in `.readthedocs.yml` in the repository root.

## Documentation structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `*.rst` - Module documentation files
- `notebook/` - Jupyter notebook tutorials
- `_static/` - Static files (images, CSS, etc.)
- `_build/` - Generated documentation (not in git)

## Writing documentation

All docstrings should follow the NumPy documentation style:
https://numpydoc.readthedocs.io/en/latest/format.html

Example:

```python
def example_function(param1, param2):
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
        
    Returns
    -------
    return_type
        Description of return value
        
    Examples
    --------
    >>> example_function(1, 2)
    3
    """
    return param1 + param2
```

## Common issues

### Missing module errors

If you get import errors when building documentation, make sure arte is installed:

```bash
pip install -e ..
```

### SyntaxWarnings in docstrings

Make sure LaTeX expressions in docstrings use raw strings:

```python
r"""
LaTeX expression: \Gamma
"""
```
