'''
Provides tools to build an interactive, searchable help based on docstrings.

Any class to which :class:`ThisClassCanHelp` is added as a base class
gets a :meth:`help` method that provides an interactive and searchable help
based on the method docstrings. All the methods decorated with
:func:`add_to_help` are added, together with all such methods in all the class members.

Example::

  from arte.utils.help import ThisClassCanHelp, add_to_help

  class InnerClass(ThisClassCanHelp):
      """An inner class"""

      @add_to_help
      def a_method(self):
          """This is a method"""
          pass

  class MyClass(ThisClassCanHelp):
      """This is my class"""
      b = InnerClass()

      @add_to_help
      def a_method(self):
          """This is a method"""
          pass

      @add_to_help
      def another_method(self):
          """This is another method"""
          pass

Interactive session::

  >>> a  = MyClass()
  >>> a.help()
  ---------
                      This is my class
  .a_method()         This is a method
  .another_method()   This is another method
  ---------
  .b              An inner class
  .b.a_method()   This is a method

  >>> a.help('other')
  ---------
  .another_method()   This is another method

  >>> a.help('inner')
  ---------
  .b              An inner class

'''
from arte.utils.not_available import NotAvailable

def _has_hlp(obj):
    return hasattr(obj, '_arte_hlp') and not \
           isinstance(getattr(obj, '_arte_hlp'), NotAvailable)

def _is_hlp_class(name, obj):
    return isinstance(obj, ThisClassCanHelp) and \
           name[0] != '_'


class ThisClassCanHelp():
    '''
    Add this class as a base class to get a help() method.
    '''
    def override_help(self, method_name, help_str):
        '''
        Overrides the help defined statically for method_name
        using help_str
        '''
        # A try/except is way more efficient than hasattr,
        # but NotAvailable swallows all exceptions
        if not hasattr(self, '_arte_hlp_overrides') or \
           isinstance(self._arte_hlp_overrides, NotAvailable):
            self._arte_hlp_overrides = dict()

        self._arte_hlp_overrides[method_name] = help_str

    def _check_overrides(self, method_name):
        '''
        If the override is not found returns an empty string or,
        for the class that CanBeIncomplete, a NotAvailable instance,
        that evaluates to False.
        '''
        try:
            return self._arte_hlp_overrides[method_name]
        except Exception:
            return ''

    def help(self, search='', prefix=''):
        '''
        Prints on stdout a list of methods that match the *search* substring
        or all of them if *search* is left to the default value of an empty
        string, together with a one-line help taken from the first line
        of their docstring, if any.

        The *prefix* argument is prepended to the method name and is used
        for recursive help of every class member.
        '''
        methods = {k:getattr(self,k) for k in dir(self) if callable(getattr(self, k))}
        members = {k:getattr(self,k) for k in dir(self) if not callable(getattr(self, k))}

        hlp_methods = {k:v for k,v in methods.items() if _has_hlp(v)}
        hlp_members = {k:v for k,v in members.items() if _is_hlp_class(k,v)}

        hlp = {prefix: _format_docstring(self)}

        for method in hlp_methods.values():
            name, docstring = method._arte_hlp
            help_string = self._check_overrides(name) or docstring
            hlp[prefix+'.'+name] = help_string

        maxlen = max(map(len, hlp.keys()))
        fmt = '%%-%ds%%s' % (maxlen+3)

        lines = []
        for k in sorted(hlp.keys()):
            line = fmt % (k, hlp[k])
            if search in line:
                lines.append(line)

        if len(lines)>0:
            print('---------')
            for line in lines:
                print(line)

        for name, obj in sorted(hlp_members.items()):
            obj.help(search=search, prefix='%s.%s' % (prefix, name))



class ThisStaticClassCanHelp(ThisClassCanHelp):
    '''Static classes (that only define classmethods)
       should use this version
    '''
    @classmethod
    def help(cls, search='', prefix=''):
        ThisClassCanHelp.help(cls(), search, prefix)

def _format_docstring(obj, default=None):

    hlp = obj.__doc__ or default or 'No docstring defined'
    hlp = hlp.strip().splitlines()[0]
    return hlp

def _wrap_with(f, call=None, arg_str=None, doc_str=None):

    if call:
        name = call
    else:
        name = f.__name__
        if arg_str:
            name += '('+arg_str+')'
        else:
            name += '()'

    hlp = _format_docstring(f)
    f._arte_hlp = (name, hlp)
    
def add_to_help(call=None, arg_str=None, doc_str=None):
    '''
    Decorator to add a method to the help system.

    With no argument, it will take the first docstring line
    and generate a default help string, but other options
    are available::

      @add_to_help
      def mymethod1(self, ....)
          """This method is very cool"""

      @add_to_help(call='mymethod2(foo)')
      def mymethod2(self, ....)
          """Also this one"""

      @add_to_help(arg_str='idx1, idx2')
      def mymethod3(self, ....)
          """Now you see it"""

      @add_to_help(doc_str='Surprise!')
      def mymethod4(self, ....)
          """And now you don't"""

    Resulting help::

      .mymethod1()           : This method is very cool
      .mymethod2(foo)        : Also this one
      .mymethod3(idx1, idx2) : Now you see it
      .mymethod4()           : Surprise!
    '''
    if callable(call):
        # No arguments, 'call' is actually our method.
        _wrap_with(call)
        return call
    else:
        def wrap(f):
            _wrap_with(f, call, arg_str, doc_str)
            return f
        return wrap

