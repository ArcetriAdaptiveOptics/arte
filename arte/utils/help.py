
from arte.utils.not_available import NotAvailable

def _has_hlp(obj):
    return hasattr(obj, '_arte_hlp') and not \
           isinstance(getattr(obj, '_arte_hlp'), NotAvailable)

def _is_hlp_class(name, obj):
    return isinstance(obj, ThisClassCanHelp) and \
           name[0] != '_'

class ThisClassCanHelp(object):

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

    def check_overrides(self, method_name):
        '''
        Returns an empty string or, for the class that CanBeIncomplete,
        a NotAvailable instance, that evaluates to False.
        '''
        try:
            return self._arte_hlp_overrides[method_name]
        except:
            return ''

    def help(self, search='', prefix=''):

        methods = {k:getattr(self,k) for k in dir(self) if callable(getattr(self, k))}
        members = {k:getattr(self,k) for k in dir(self) if not callable(getattr(self, k))}

        hlp_methods = {k:v for k,v in methods.items() if _has_hlp(v)}
        hlp_members = {k:v for k,v in members.items() if _is_hlp_class(k,v)}

        hlp = {prefix: _format_docstring(self)}

        for method in hlp_methods.values():
            name, docstring = method._arte_hlp
            help_string = self.check_overrides(name) or docstring
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


# Static classes (ones that only define classmethods)
# should use this version

class ThisStaticClassCanHelp(ThisClassCanHelp):

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
    Decorator to add a method to the help system. Can be used
    in several ways:

    With no argument, it will take the first docstring line
    and generate a default help string.

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

             
               

