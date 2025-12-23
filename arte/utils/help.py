##########################################################
#
# who       when        what
# --------  ----------  ----------------------------------
# apuglisi  2020-04-23  Reimplemented as a class decorator
# apuglisi  2019-09-28  Created
#
##########################################################
'''
Provides tools to build an interactive, searchable help based on docstrings.

Any class to decorated with @add_help
gets a :meth:`help` method that provides an interactive and searchable help
based on the method docstrings. All public methods (not starting with "_")
are added, together with all such methods in all the members that are
classes decorated with @add_help. Derived classes inherit the help system
without a need to use the @add_help decorator.

Help is built dynamically when invoked, so if a new member is added
to the class at runtime, it will appear in the help too.

The help method name can be customized giving the `help_function` parameter
to the decorator.

If the `classmethod` parameter is True, the help function is created
as a classmethod instead of an ordinary method. This is useful
for classes that only define classmethods and are not normally instanced.

Example::

  from arte.utils.help import add_help

  @add_help
  class InnerClass():
      """An inner class"""

      def a_method(self):
          """This is a method"""
          pass

  @add_help
  class MyClass():
      """This is my class"""
      b = InnerClass()

      def a_method(self):
          """This is a method"""
          pass

      def another_method(self):
          """This is another method"""
          pass

  @add_help(help_function='show_help')
  class CustomClass():
      """This a custom class"""

      def a_method(self):
          """This is a method"""
          pass

Interactive session::

  >>> a  = MyClass()
  >>> a.help()
  ---------
  MyClass                    This is my class
  MyClass.a_method()         This is a method
  MyClass.another_method()   This is another method
  ---------
  MyClass.b              An inner class
  MyClass.b.a_method()   This is a method

  >>> a.help('other')
  ---------
  MyClass.another_method()   This is another method

  >>> a.help('inner')
  ---------
  MyClass.b              An inner class

  >>> b = CustomClass()
  >>> b.show_help()
  ---------
  CustomClass               This a custom class
  CustomClass.a_method()    This is a method

'''
import inspect
from functools import partial
from functools import cached_property

HIDDEN_HELP = '__arte_help'
HIDDEN_NAME = '__arte_name'
HIDDEN_ARGS = '__arte_args'
HIDDEN_HIDE = '__arte_hide'


def _is_public_method(name):
    return name[0] != '_'


def _is_hlp_class(obj):
    return hasattr(obj, HIDDEN_HELP)


def _is_hidden(m):
    return (hasattr(m, HIDDEN_HIDE)) and (getattr(m, HIDDEN_HIDE) is True)


def add_help(cls=None, *, help_function='help', classmethod=False):
    '''
    Decorator to add interactive help to a class

    Parameters
    ----------
    help_function: str, optional
        Name of the method that will be added to the class. Defaults to "help"
    classmethod: bool, optional
        If True, the help method will be added as a classmethod. Default False

    Returns
    -------
    class
        The decorated class type
    '''
    # Trick to allow a decorator without parenthesis
    if cls is not None:
        return add_help()(cls)

    def help(self, search='', prefix=''):
        '''
        Interactive help

        Prints on stdout a list of methods that match the *search* substring
        or all of them if *search* is left to the default value of an empty
        string, together with a one-line help taken from the first line
        of their docstring, if any.

        The *prefix* argument is prepended to the method name and is used
        for recursive help of every class member.
        '''
        properties = ({k: getattr(self.__class__, k)
                      for k in dir(self.__class__)
                      if isinstance(getattr(self.__class__, k), (property, cached_property))})

        items = [k for k in dir(self) if k not in properties]

        items_with_values = [(k, getattr(self, k)) for k in items]
        methods = {k: v for k, v in items_with_values if callable(v)}
        members = {k: v for k, v in items_with_values if not callable(v)}
        methods.update(properties)

        methods = {k: v for k, v in methods.items() if _is_public_method(k) \
                                                      and not _is_hidden(v)}
        members = {k: v for k, v in members.items() if _is_hlp_class(v)}

        if prefix == '':
            prefix = self.__class__.__name__

        hlp = {prefix: _format_docstring(self)}

        for name, method in methods.items():
            if name and method:  # Skip NAs
                display_name = _format_name(method, default=name)
                if name in properties:
                    pars = ''
                else:
                    pars = _format_pars(method)
                helpstr = _format_docstring(method)
                hlp[prefix + '.' + display_name + pars] = helpstr

        maxlen = max(map(len, hlp.keys()))
        fmt = '%%-%ds%%s' % (maxlen + 3)

        lines = []
        for k in sorted(hlp.keys()):
            line = fmt % (k, hlp[k])
            if search in line:
                lines.append(line)

        if len(lines) > 0:
            print('---------')
            for line in lines:
                print(line)

        for name, obj in sorted(members.items()):
            if name[0] != '_':
                obj.help(search=search, prefix='%s.%s' % (prefix, name))

    def decorate(cls):
        setattr(cls, HIDDEN_HELP, False)  # Set attr but do not define a string
        if classmethod:
            # Expose a class-level help that calls instance help without binding issues
            def _class_help(search='', prefix='', __cls=cls):
                return help(__cls(), search=search, prefix=prefix)
            _class_help.__name__ = help.__name__
            # Hide from recursive listing and assign as staticmethod to avoid FutureWarning
            setattr(cls, help_function, staticmethod(hide_from_help(_class_help)))
        else:
            setattr(cls, help_function, hide_from_help(help))
        return cls
    return decorate


def modify_help(call=None, arg_str=None, doc_str=None):
    '''
    Decorator to modify the automatic help for a method.

    Without this decorator, the method signature for help
    is just "method()". Using this decorator, other
    signatures are possible::

      @modify_help(call='mymethod1(foo)')
      def mymethod1(self, ....)
          """This method is very cool"""

      @modify_help(arg_str='idx1, idx2')
      def mymethod2(self, ....)
          """Now you see it"""

      @modify_help(doc_str='Surprise!')
      def mymethod3(self, ....)
          """And now you don't"""

    Resulting help::

      .mymethod1(foo)        : This method is very cool
      .mymethod2(idx1, idx2) : Now you see it
      .mymethod3()           : Surprise!

    .. Note::
        if the method is a @staticmethod, this decorator
        should be inserted *after* the staticmethod one.
    '''
    def wrap(f):
        _wrap_with(f, call, arg_str, doc_str)
        return f
    return wrap


def hide_from_help(f):
    '''Decorator to hide a method from the interactive help'''
    setattr(f, HIDDEN_HIDE, True)
    return f


def _format_docstring(obj, default=None):

    hlp = getattr(obj, HIDDEN_HELP, False)
    docstr = (obj.__doc__ or '').strip()
    hlp = hlp or docstr or default or 'No docstring defined'
    hlp = hlp.strip().splitlines()[0]
    return hlp


def _format_name(obj, default=None):

    if hasattr(obj, '__name__'):
        myname = obj.__name__
    else:
        myname = default

    name = getattr(obj, HIDDEN_NAME, False)
    name = name or myname
    name = name.strip().splitlines()[0]
    return name


def _format_pars(method):
    if isinstance(method, property):
        return ''

    args = getattr(method, HIDDEN_ARGS, None)
    if args is not None:
        return args

    sig = inspect.signature(method)
    return '(' + ','.join(sig.parameters.keys()) + ')'


def _wrap_with(f, call=None, arg_str=None, doc_str=None):

    if call:
        name = call
        args = ''
    else:
        name = f.__name__
        if arg_str:
            args = '(%s)' % arg_str
        else:
            args = None

    hlp = doc_str or _format_docstring(f)
    setattr(f, HIDDEN_NAME, name)
    setattr(f, HIDDEN_HELP, hlp)
    setattr(f, HIDDEN_ARGS, args)

# ___oOo___
