# -*- coding: utf-8 -*-
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-12-27  Created
#
#########################################################

from collections import OrderedDict


class TabularReport():
    '''
    Report with column-based format.

    Prints a new line of values each time it is triggered.

    Columns are defined at initialization with the *column_names*, and they
    can be set using the [] operator. Each column will occupy `column_width`
    characters (default 10 characters). Columns will be printed in the order
    they were added.

    Each column has a formatter function, which can set using the
    .fmt dictionary. The formatter function must take a single parameter,
    the value to be displayed, and return a string. By default, all columns
    are formatted with str().

    It is possible to add a column on the fly assigning
    to it using the [] operator. The new column will be assigned the
    default str() formatter.

    Every time the report() function is called, an internal counter is
    incremented and, if at least `decimation` counts have passed since
    the last time, the values are printed on stdout. By default, `decimation`
    is 1 and all lines are printed. A header with the column names is printed
    before the first line and after each `repeat_hdr` lines have been
    printed (default 100). If `add_iter` is True, the counter is printed
    as the first column.

    Attributes
    ----------
    decimation : int
        how many calls to report() are needed before a line is printed
    hdr_decimation: int
        how many lines are printed before the header is printed again
    add_iter: bool
        if True, the iteration number is added as the first column
    column_width: int
        number of characters assigned to each column
    iteration_hdr: str
        string used as column name for the iteration number
    ['column_name']
        Assign values to column using the [] operator.
    fmt['column_name']
        Assign special formatting functions using the fmt dictionary.

    Examples
    --------
    >>> from arte.utils.tabular_report import TabularReport
    >>> r = TabularReport(['pippo','pluto'])
    >>> for i in range(2):
    ...    r['pippo'] = i*2
    ...    r['pluto'] = i+10
    ...    r.report()
    >>> r['paperino'] = 'a'
    >>> r.fmt['paperino'] = ord
    >>> r.print_header()
    >>> r.report()

    Result::

     iteration  pippo      pluto
     ---------------------------------
     1          0          10
     2          2          11

     iteration  pippo      pluto      paperino
     --------------------------------------------
     3          2          11         97
    '''

    def __init__(self, column_names, decimation: int = 1,
                                     hdr_decimation: int = 100,
                                     add_iter: bool = True,
                                     column_width: int = 10,
                                     iteration_hdr='iteration'):

        self.values = OrderedDict.fromkeys(column_names)
        self.fmt = OrderedDict.fromkeys(column_names, str)

        self.decimation = decimation
        self.hdr_decimation = hdr_decimation
        self.add_iter = add_iter
        self.column_width = column_width
        self.iteration_hdr = iteration_hdr

        self.fmt[iteration_hdr] = str

        self.counter = 0
        self.hdr_counter = 0

    def __setitem__(self, key, value):
        if key not in self.fmt:
            self.fmt[key] = str
        self.values[key] = value

    def report(self):
        '''
        Increment the iteration counter and if, conditions match,
        print a header and/or a line of values.
        '''
        self.counter += 1
        if self.counter % self.decimation != 0:
            return

        if self.hdr_counter % self.hdr_decimation == 0:
            self.print_header()

        self.hdr_counter += 1
        self.print_line()

    def print_header(self):
        '''Unconditionally print a line with the header.'''

        cols = self.values.keys()
        if self.add_iter:
            cols = ['iteration'] + list(cols)

        fmt = '%%-%ds ' % self.column_width
        hdr = ''.join([fmt % col for col in cols])

        print()
        print(hdr)
        print('-' * len(hdr))

    def print_line(self):
        '''Unconditionally print a line with all the current values'''

        cols = self.values.keys()
        formatted_values = [self.fmt[col](self.values[col]) for col in cols]

        if self.add_iter:
            formatted_values = [str(self.counter)] + formatted_values

        fmt = '%%-%ds ' % self.column_width
        line = ''.join([fmt % value for value in formatted_values])

        print(line)

# ___oOo___
