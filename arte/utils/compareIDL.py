# -*- coding: utf-8 -*-
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-08  Created
#
#########################################################
from __future__ import print_function

import os.path
import tempfile
import subprocess
import numpy as np
from scipy.io import readsav
from collections.abc import Iterable


def compareIDL(idlscript, pyscript, vars_to_compare,
               precision=1e-5, verbose=0, tmpfile=None):
    '''
    Compare IDL and Python routines results, to aid in porting.

    This function will run an IDL batch script (containing statements
    that will be executed as if they were typed on the IDL command prompt)
    and a Python script. After both scripts have been run, the variables
    listed in vars_to_compare are extracted from both sessions and
    compared numerically to the specified precision limit.

    Parameters
    ----------
    idlscript: string or list of strings
        IDL statements. If in a single string,
        they must be separated by newlines
    pyscript: string or list of strings
        Python statements. If in a single string,
        they must be separated by newlines
    vars_to_compare: list of strings
        variable names to compare
    precision: float, optional
        relative numerical precision
    verbose: int, optional
        verbose level: if greater than zero, will print on stdout the list
        of variables for which the comparison fails; If greater than one,
        will also print the variable values.
    tmpfile: string, optional
        filename for the temporary file used to save IDL variables.
        If None, a default filename in the system's temporary
        directory will be used.

    Returns
    -------
    bool
        True if the comparison is within the precision limit, False otherwise
    '''

    if tmpfile is None:
        tmpfile = os.path.join(tempfile.gettempdir(), 'idl_compare.sav')

    savecmd = ','.join(['\nSAVE',
                        *vars_to_compare,
                        'FILENAME="%s"\n' % tmpfile])

    if isinstance(idlscript, Iterable) and not isinstance(idlscript, str):
        idlscript = '\n'.join(idlscript)

    if isinstance(pyscript, Iterable) and not isinstance(pyscript, str):
        pyscript = '\n'.join(pyscript)

    p = subprocess.Popen('idl', stdin=subprocess.PIPE, shell=True)
    p.communicate((idlscript + savecmd).encode())
    idldata = readsav(tmpfile)

    exec(pyscript)

    good = []
    for varname in vars_to_compare:
        idlvar = idldata[varname]
        pyvar = locals()[varname]

        if pyvar is None:
            print('%s is None, cannot compare' % varname)
            return False

        goodvar = np.all(np.isclose(idlvar, pyvar))
        if verbose and not goodvar:
            print('%s differs' % varname)
            if verbose > 1:
                print('IDL value   : ', idlvar)
                print('Python value: ', pyvar)
        good.append(goodvar)

    return all(good)

# ___oOo___
