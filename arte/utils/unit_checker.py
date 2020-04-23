# -*- coding: utf-8 -*-

import astropy.units as u

def make_sure_its_a(unit, v, name=''):
    '''
    Make sure that `v` has the astropy unit `unit`.
    
    If `v` does not have any unit, apply `unit` and return the combined value.
    If it has one, check that it can be converted to `unit`,
    and return the converted value. Otherwise, raise astropy.units.UnitsError.
    
    Parameters
    ----------
    unit: astropy unit
        the wanted unit
    v:
        the value under test
    name: string
        description of `v`. Will be used in error messages
        in case the conversion fails.

    Returns
    -------
    astropy Quantity
        the original value converted to `unit`.
      
    Raises
    ------
    astropy.units.UnitsError
        if the conversion to `unit` fails.

    '''
    if not isinstance(v, u.Quantity):
        return v * unit
    
    try:
        normalized = v.to(unit)
    except u.UnitConversionError:
        errmsg = 'Symbol {} should be {}, got {} instead'.format(
                  name, str(unit), str(v.unit))
        raise u.UnitsError(errmsg)
    return normalized

def unit_check(f):
    '''
    Decorator to add type checking of astropy units to a function.
    
    This decorator will ensure that, each time the decorated function is called,
    all the arguments have the correct astropy units,
    as defined in the default values of the
    decorated function, calling `make_sure_its_a()` for each of them.

    If the function does not define a unit for a parameter,
    the corresponding argument is not modified, whether it has a unit or not.

    If any of the checks fails, raises `TypeError` when the decorated function
    is called.
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
        ret_value = f(*bound.args, **bound.kwargs)

        if sig.return_annotation is not inspect.Signature.empty:
            return make_sure_its_a(sig.return_annotation, ret_value,
                                   'Return value from function '+ f.__name__)
        else:
            return ret_value

    return wrapper
