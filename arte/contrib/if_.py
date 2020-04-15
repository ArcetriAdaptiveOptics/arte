
import logging

def if_(condition, warning=None):
    '''
    Decorator to turn a function into a NO-OP
    if the condition is not met.
    Source: https://stackoverflow.com/questions/17946024/deactivate-function-with-decorator

    Example:
    @if_(global_enable)
    def do_something():
        ...

    '''
    def noop_decorator(func):
        return func  # pass through

    def neutered_function(func):
        def neutered(*args, **kw):
            if warning:
                logging.warn(warning)
            return None
        return neutered

    return noop_decorator if condition else neutered_function

