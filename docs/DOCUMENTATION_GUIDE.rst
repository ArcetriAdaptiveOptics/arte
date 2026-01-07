Documentation Best Practices
============================

This document provides guidelines for writing good documentation in the arte package.

General Principles
------------------

1. **All documentation must be in English** - code, comments, docstrings, and documentation files
2. **Use proper spelling** - run a spell checker before committing
3. **Be clear and concise** - avoid unnecessary jargon
4. **Include examples** - working code examples help users understand usage
5. **Keep documentation up-to-date** - update docs when changing code

Docstring Style
---------------

Arte uses the NumPy docstring format. This is enforced by numpydoc and checked during documentation builds.

Section Names
^^^^^^^^^^^^^

Use **exactly** these section names (case-sensitive):

* ``Parameters`` - not "Arguments" or "Inputs"
* ``Returns`` - not "Return" 
* ``Raises`` - not "Exceptions"
* ``Yields`` - for generators
* ``Examples`` - not "Example"
* ``See Also`` - related functions/classes
* ``Notes`` - additional information
* ``References`` - citations and external links
* ``Warnings`` - important warnings for users

Common Mistakes to Avoid
^^^^^^^^^^^^^^^^^^^^^^^^

**Wrong:**

.. code-block:: python

    def bad_example(x, y):
        """
        Does something.
        
        Arguments:      # Wrong: should be "Parameters"
        ---------
        x: int          # Wrong: missing space after colon
            Input value
        
        Return:         # Wrong: should be "Returns"
        ------
        result
        
        Example:        # Wrong: should be "Examples"
        -------
        >>> bad_example(1, 2)
        """
        pass

**Correct:**

.. code-block:: python

    def good_example(x, y):
        """
        Compute the sum of two numbers.
        
        Parameters
        ----------
        x : int
            First input value
        y : int
            Second input value
        
        Returns
        -------
        int
            Sum of x and y
        
        Examples
        --------
        >>> good_example(1, 2)
        3
        """
        return x + y

LaTeX in Docstrings
-------------------

When including mathematical formulas, use raw strings to avoid escape sequence warnings:

**Wrong:**

.. code-block:: python

    def formula():
        """
        The formula is \Gamma(x)  # SyntaxWarning: invalid escape sequence
        """
        pass

**Correct:**

.. code-block:: python

    def formula():
        r"""
        The formula is \Gamma(x)
        
        .. math::
            \int_0^\infty x^2 dx
        """
        pass

Type Hints
----------

Use type hints in function signatures when possible. They will be automatically
included in the documentation:

.. code-block:: python

    from typing import Optional, Union
    import numpy as np
    
    def process_data(
        data: np.ndarray,
        threshold: float = 0.5,
        mode: Optional[str] = None
    ) -> np.ndarray:
        """
        Process input data array.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        threshold : float, optional
            Threshold value (default: 0.5)
        mode : str, optional
            Processing mode. If None, uses default mode.
        
        Returns
        -------
        np.ndarray
            Processed data
        """
        pass

Class Documentation
-------------------

Classes should have a docstring describing the class, plus docstrings for
``__init__`` and all public methods:

.. code-block:: python

    class DataProcessor:
        """
        Process and analyze data from experiments.
        
        This class provides methods for loading, processing, and analyzing
        experimental data. It handles various data formats and provides
        utilities for common operations.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        verbose : bool, optional
            Enable verbose output (default: False)
        
        Attributes
        ----------
        data : np.ndarray
            Currently loaded data
        metadata : dict
            Metadata for current data
        
        Examples
        --------
        >>> processor = DataProcessor({'mode': 'fast'})
        >>> processor.load_data('file.npy')
        >>> result = processor.process()
        """
        
        def __init__(self, config, verbose=False):
            self.config = config
            self.verbose = verbose
            self.data = None
            self.metadata = {}
        
        def load_data(self, filename):
            """
            Load data from file.
            
            Parameters
            ----------
            filename : str
                Path to data file
            
            Returns
            -------
            bool
                True if loading was successful
            """
            pass

Module Documentation
--------------------

Each module should have a docstring at the top describing its purpose:

.. code-block:: python

    """
    Atmospheric turbulence modeling
    
    This module provides tools for modeling atmospheric turbulence in
    adaptive optics systems. It includes functions for computing
    von Karman power spectral densities, generating phase screens,
    and analyzing turbulence profiles.
    
    Main classes:
    
    * VonKarmanPsd - von Karman PSD calculator
    * PhaseScreenGenerator - turbulent phase screen generation
    * Cn2Profile - atmospheric Cn2 profile representation
    
    Examples
    --------
    >>> from arte.atmo import VonKarmanPsd
    >>> psd = VonKarmanPsd(r0=0.15, L0=25)
    >>> spectrum = psd.spatial_psd(freq_vector)
    """
    
    import numpy as np
    # ... rest of module

Adding New Modules to Documentation
------------------------------------

When adding a new module, create a corresponding ``.rst`` file in ``docs/``:

1. Create ``docs/mymodule.rst``
2. Add it to the toctree in ``docs/index.rst``
3. Use this template:

.. code-block:: rst

    My Module (`arte.mymodule`)
    ===========================
    
    Introduction
    ------------
    
    Brief description of what this module does.
    
    Submodules
    ----------
    
    submodule1 module
    -----------------
    
    .. automodule:: arte.mymodule.submodule1
       :members:
       :undoc-members:
       :show-inheritance:

Testing Documentation
---------------------

Always test documentation builds locally before pushing:

.. code-block:: bash

    cd docs
    make clean
    make html

Fix any errors or warnings that appear. View the result in your browser:

.. code-block:: bash

    open _build/html/index.html

Jupyter Notebooks as Tutorials
-------------------------------

Tutorials should be Jupyter notebooks placed in ``docs/notebook/``.

They will be automatically rendered in the documentation via nbsphinx.

Requirements for tutorial notebooks:

* Clear title and introduction
* Executable cells (should run without errors)
* Narrative text explaining what's happening
* Clean output (run all cells before committing)
* Minimal dependencies

Add notebooks to ``docs/tutorial.rst`` to include them in the documentation.
