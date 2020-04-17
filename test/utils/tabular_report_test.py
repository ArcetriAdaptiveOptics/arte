# -*- coding: utf-8 -*-

import doctest
import unittest
from arte.utils.tabular_report import TabularReport
from arte.utils.capture_output import capture_output

def strip_all(lst):
    return [x.strip() for x in lst]

class TabularReportTest(unittest.TestCase):

    def testDocstring(self):
        '''
        doctest's automated tests only check one line at a time,
        while we want the entire output, so we make our own
        '''
        # Run the example
        docstring = doctest.script_from_examples(TabularReport.__doc__ )
        with capture_output() as (out, err):
            exec(docstring)

        # Add '#' like doctest.script_from_examples() does.
        # The reference lines are the last x lines of the docstring
       
        out = ['#  '+x for x in out.getvalue().splitlines()]
        ref = docstring.splitlines()[-len(out):]
        
        assert( strip_all(out) == strip_all(ref))


if __name__ == "__main__":
    unittest.main()
