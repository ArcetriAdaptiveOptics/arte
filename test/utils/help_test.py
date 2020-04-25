# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
from arte.utils.help import add_help, modify_help, hide_from_help
from arte.utils.capture_output import capture_output


@add_help
class AnotherClass():
    """An inner class"""

    def a_method(self):
        """This is a method"""
        pass

@add_help
class MyClass():
    """This is my class"""
    b = AnotherClass()

    def a_method(self):
        """This is a method"""
        pass

    def another_method(self):
        """This is another method"""
        pass

    @modify_help(call='foo(bar)')
    def method_with_modified_call(self):
        """This method has a modified call"""
        pass

    @modify_help(arg_str='foo')
    def method_with_modified_args(self, myparameter):
        """This method has modified args"""
        pass

    @modify_help(doc_str='foo')
    def method_with_replaced_docstr(self):
        """This method has a modified docstring"""
        pass

    @hide_from_help
    def hidden_method(self0):
        '''an hidden method'''
        pass

@add_help(help_function='show_help')
class CustomClass():
    """This is my class"""
    b = AnotherClass()

    def a_method(self):
        """This is a method"""
        pass
        
@add_help(classmethod=True)
class StaticClass():
    '''This is a static class'''

    @classmethod
    def a_method(self):
        '''This is a method'''
        pass

class HelpTest(unittest.TestCase):

    def setUp(self):
        self.a = MyClass()
        self.b = CustomClass()

    def test_full_help(self):
        with capture_output() as (out, err):
            self.a.help()
        assert('This is my class' in out.getvalue())
        assert('An inner class' in out.getvalue())
        assert('This is a method' in out.getvalue())
        assert('This is another method' in out.getvalue())

    def test_search(self):
        with capture_output() as (out, err):
            self.a.help('other')
        assert('This is my class' not in out.getvalue())
        assert('An inner class' not in out.getvalue())
        assert('This is a method' not in out.getvalue())
        assert('This is another method' in out.getvalue())

    def test_modified_call(self):
        with capture_output() as (out, err):
            self.a.help('modified call')    
            
        assert 'foo(bar)' in out.getvalue()
        assert 'method_with_modified_call' not in out.getvalue()
        
    def test_modified_args(self):
        with capture_output() as (out, err):
            self.a.help('modified_args')   
            
        assert '(foo)' in out.getvalue()
        assert 'myparameter' not in out.getvalue()
 
    def test_replaced_docstr(self):
        with capture_output() as (out, err):
            self.a.help('replaced_docstr')   
            
        assert 'foo' in out.getvalue()
        assert 'This is another method' not in out.getvalue()   
        
    def test_custom_help_func(self):

        with self.assertRaises(AttributeError):
            self.b.help()   
        
        self.b.show_help()
    
    def test_static_class(self):
        with capture_output() as (out, err):
            StaticClass.help()
        assert 'This is a static class' in out.getvalue()
        assert 'This is a method' in out.getvalue()

    def test_hidden(self):
        with capture_output() as (out, err):
            self.a.help('hidden')   
            
        assert 'hidden method' not in out.getvalue()   
        

if __name__ == "__main__":
    unittest.main()
