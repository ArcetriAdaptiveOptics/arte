# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from functools import wraps


def assert_unit_is_equivalent(var, ref):
    '''Make sure that `var` has a unit compatible with `ref`'''

    if isinstance(var, u.Quantity):
        if isinstance(ref, u.Quantity):
            assert var.unit.is_equivalent(ref.unit)
        else:
            raise AssertionError('Variable has units while it should not')
    else:
        if isinstance(ref, u.Quantity):
            raise AssertionError('Variable has no unit, while it should be '
                                 'equivalent to %s' % (ref.unit))


def separate_value_and_unit(var):
    '''Returns the argument value and unit, if `var` has one.

    If not, `var` is returned unchanged and unit is set to 1.

    Parameters
    ----------
    var: any type
        the variable to be tested

    Returns
    -------
    value, unit
        tuple with the value and the astropy unit of `var`
        or 1 if `var` does not have a unit.

    Examples
    --------
    >>> a = 42 * u.m
    >>> separate_value_and_unit(a)
    (42.0, Unit("m"))

    >>> b = 42
    >>> separate_value_and_unit(b)
    (42, 1)
    '''
    if isinstance(var, u.Quantity):
        return var.value, var.unit
    else:
        return var, 1


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


def match_and_remove_unit(var, var_to_match):
    '''
    Make sure that `var` has the same unit as `var_to_match`, if any,
    and then remove the unit. If `var_to_match` has no unit, the value of
    `var` is returned, without its unit if it has one.

    Parameters
    ----------
    var: any type
        the variable to be tested
    var_to_match: any type
        the variable whose unit must be matched

    Returns
    -------
    value
         the value of `var` (without unit attached) after the unit has been
         changed to be the same as `var_to_match`

    Raises
    ------
    u.UnitsError
         if `var` cannot be converted to `var_to_match` using
         the standard astropy method `var`.to()

    Examples
    --------
    >>> a = 42*u.km
    >>> b = 42*u.m
    >>> match_and_remove_unit(a,b)
    42000.0

    >>> a = 42
    >>> c = 3.1415
    >>> match_and_remove_unit(a,c)
    42

    '''
    if isinstance(var, u.Quantity) and isinstance(var_to_match, u.Quantity):
        # Catch the errors here to have nicer tracebacks, otherwise
        # we end up deep into astropy libraries.
        try:
            return var.to(var_to_match.unit).value
        except Exception as e:
            raise u.UnitsError(str(e))

    elif isinstance(var, u.Quantity):
        return var.value
    else:
        return var


def match_and_remove_units(*args):
    '''
    Take a list of variables, make sure that all have the same unit as the
    first one, and return a list of unitless values plus the common unit.

    Order is significant! The first variable's unit is applied to all others.
    If the first one does not have a unit, all units are removed.

    Parameters
    ----------
    *args: an arbitrary number of parameters
         variables whose units must be matched

    Returns
    -------
    list
        list of values after unit conversion and removal, plus the unit.
        If the first parameter had no unit, the unit is set to 1.

    Raises
    ------
    u.UnitsError
         if any of the parameters cannot be converted using
         the standard astropy method `var`.to()

    Examples
    --------
    >>> a = 42*u.m
    >>> b = 42*u.cm
    >>> c = 42*u.km
    >>> match_and_remove_units(a,b,c)
    [42.0, 0.42, 42000.0, Unit("m")]

    >>> a = 42
    >>> b = 42*u.cm
    >>> c = 42*u.km
    >>> match_and_remove_units(a,b,c)
    [42, 42.0, 42.0, 1]
    '''
    if len(args) == 0:
        return [1]
    else:
        _, unit = separate_value_and_unit(args[0])

    newvars = [match_and_remove_unit(x, args[0]) for x in args[0:]]
    return newvars + [unit]


def assert_array_almost_equal_w_units(a, b):
    '''Clone of np.testing.asset_array_almost_equal for u.Quantities'''
    a_, b_, _ = match_and_remove_units(a, b)
    np.testing.assert_array_almost_equal(a_, b_)


def unit_check(f):
    '''
    Decorator to add type checking of astropy units to a function.

    This decorator will ensure that, each time the decorated function is
    called, all the arguments have the correct astropy units,
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

    @wraps(f)
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
                                   'Return value from function ' + f.__name__)
        else:
            return ret_value

    return wrapper
