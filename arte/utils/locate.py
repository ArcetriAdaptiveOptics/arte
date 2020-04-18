# -*- coding: utf-8 -*-
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2012-11-05  Created
#
#########################################################
import os, fnmatch

def locate(pattern, rootdir=None):
    '''Generator similar to Unix's *locate* utility
    
    Locates all files matching a pattern, inside and below the root directory.
    If no root directory is given, the current directory is used instead.
    
    Parameters
    ----------
    pattern: string
        the filename pattern to match. Unix wildcards are allowed
    rootdir: string, optional
        the root directory where the search is started. If not set,
        the current directory is used.
        
    Yields
    ------
    string
        the next filename matching *pattern*

    '''
    if rootdir is None:
        rootdir = os.curdir
        
    for path, dirs, files in os.walk(rootdir):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)

def locate_first(pattern, rootdir=None):
    '''Locate the first filename matching *pattern*
    
    Locate the first file a matching, inside and below the root directory.
    If no root directory is given, the current directory is used instead.

    Parameters
    ----------
    pattern: string
        the filename pattern to match. Unix wildcards are allowed
    rootdir: string, optional
        the root directory where the search is started. If not set,
        the current directory is used.
        
    Returns
    -------
    string
        the first filename matching *pattern*
    '''
    loc = locate(pattern, rootdir)
    try:
        return next(loc)
    except StopIteration:
        return None
    
# ___oOo___

