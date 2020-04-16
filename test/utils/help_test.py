#!/usr/bin/env python
import unittest
from arte.utils.help import ThisClassCanHelp, add_to_help
from arte.utils.capture_output import capture_output


class HelpTest(unittest.TestCase):

    def setUp(self):
        class MyClass(ThisClassCanHelp):
            """This is my class"""

            class InnerClass(ThisClassCanHelp):
                """An inner class"""
                @add_to_help
                def a_method(self):
                    """This is a method"""
                    pass
            b = InnerClass()
            @add_to_help
            def a_method(self):
                """This is a method"""
                pass
            @add_to_help
            def another_method(self):
                """This is another method"""
                pass

        self.a = MyClass()

    def testFullHelp(self):
        with capture_output() as (out, err):
            self.a.help()
        assert('This is my class' in out.getvalue())
        assert('An inner class' in out.getvalue())
        assert('This is a method' in out.getvalue())
        assert('This is another method' in out.getvalue())

    def testSearch(self):
        with capture_output() as (out, err):
            self.a.help('other')
        assert('This is my class' not in out.getvalue())
        assert('An inner class' not in out.getvalue())
        assert('This is a method' not in out.getvalue())
        assert('This is another method' in out.getvalue())


if __name__ == "__main__":
    unittest.main()
