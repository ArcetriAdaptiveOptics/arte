
import sys
from io import StringIO
from contextlib import contextmanager

@contextmanager
def capture_output():
    '''
    Use this context manager to capture stdout and stderr
    from a Python code section. Example:

    with capture_output() as (out, err):
        routines_that_prints_lots()

    out.getvalue() will be the stdout
    err.getvalue() will be the stderr
    '''
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

