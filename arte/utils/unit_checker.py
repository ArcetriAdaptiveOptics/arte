# -*- coding: utf-8 -*-

import inspect
from functools import wraps

import numpy as np
import astropy.units as u

from arte.utils.not_available import NotAvailable


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
    > a = 42 * u.m
    > separate_value_and_unit(a)
    (42.0, Unit("m"))

    > b = 42
    > separate_value_and_unit(b)
    (42, 1)
    '''
    if isinstance(var, u.Quantity):
        return var.value, var.unit
    else:
        return var, 1


def make_sure_its_a(unit, v, name='', copy=True):
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
    # Special case for masked arrays, who wraps their data into a "data" attribute
    if isinstance(v, np.ma.MaskedArray):
        return np.ma.MaskedArray(
            make_sure_its_a(unit, v.data, name=name, copy=copy),
            mask=v.mask)

    if isinstance(v, NotAvailable):
        return v

    if not isinstance(v, u.Quantity):
        return u.Quantity(v, unit=unit, copy=copy)

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
