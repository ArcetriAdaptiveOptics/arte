# -*- coding: utf-8 -*-

import astropy.units as u

def make_sure_its_a(unit, v, name=''):
    '''
    If `v` has no unit, apply `unit`. If it has one, check that
    it can be converted to `unit`, and return the converted value.
    Otherwise, raise TypeError.
    '''
    if not isinstance(v, u.Quantity):
        return v * unit
    
    try:
        normalized = v.to(unit)
    except u.UnitConversionError:
        errmsg = 'Variable {} should be {}, got {} instead'.format(
                  name, str(unit), str(v.unit))
        raise TypeError(errmsg)
    return normalized

def unit_check(f):
    '''
    This decorator will check that all the decorated function's arguments
    have the correct astropy units, as defined in the default values of the
    decorated function. If the function does not define a unit for a parameter,
    the corresponding argument is not modified, whether it has a unit or not.
    '''
    # Keep as much as possible out of the wrapper, so it only executes
    # once at function definition time.
    import inspect
    sig = inspect.signature(f)
    pars_to_check = [p for p in sig.parameters.values()
                     if isinstance(p.default, u.Quantity)]

    def wrapper(*args, **kwargs):

        # Reconstruct the decorated function's arguments
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        # Check all arguments that have an astropy unit in their defaults
        for p in pars_to_check:
            bound.arguments[p.name] = make_sure_its_a(p.default.unit,
                                                      bound.arguments[p.name],
                                                      p.name)    
        return f(*bound.args, **bound.kwargs)
    return wrapper
