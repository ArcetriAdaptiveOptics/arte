# -*- coding: utf-8 -*-

import astropy.units as u


def get_the_unit_if_it_has_one(var):
    '''Returns the argument unit, or 1 if it does not have any.

    Parameters
    ----------
    var: any type
        the variable to be tested

    Returns
    -------
    unit
        the astropy unit of `var` or 1 if `var` does not have a unit.

    Examples
    --------
    >>> a = 42
    >>> get_the_unit_if_it_has_one(a)
    1

    >>> b = 42*u.m
    >>> get_the_unit_if_it_has_one(b)
    Unit("m")
    '''
    if isinstance(var, u.Quantity):
        return var.unit
    else:
        return 1


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
        unit = get_the_unit_if_it_has_one(args[0])

    newvars = [match_and_remove_unit(x, args[0]) for x in args[0:]]

    return newvars + [unit]
