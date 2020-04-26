
import sys
from io import StringIO
from contextlib import contextmanager


@contextmanager
def capture_output():
    '''
    Context manager to capture stdout and stderr.

    Captures stddout and stderr from a Python code section,
    using two StringIO objects.
    Example::

      with capture_output() as (out, err):
          routine_that_prints_lots()

    *out.getvalue()* will return as string with whatever
    was printed on stdout. *err.getvalue()* will return
    the same for stderr.
    Nothing will appear on screen.

    '''
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
