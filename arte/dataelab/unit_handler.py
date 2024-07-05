
import astropy.units as u
from arte.utils.unit_checker import make_sure_its_a, separate_value_and_unit


class UnitHandler():
    def __init__(self, wanted_unit):
        self._wanted_unit = wanted_unit
        self._actual_unit = None  # Discovered later
        self._actual_unit_name = None
        self._force = False

    def actual_unit(self):
        '''Data unit as an astropy unit, or None'''
        return self._actual_unit

    def actual_unit_name(self):
        '''Data unit string, or None if not set'''
        return self._actual_unit_name

    def set_force(self, force_flag):
        '''Set unit coercion flag'''
        self._force = force_flag

    def apply_unit(self, data):
        '''Appy to *data* our wanted unit, if any, converting if necessary

        If the force flag is not set, unit will be converted, throwing an
        exception is conversion if impossible.
        If the force flag is set, the old unit is stripped and the new one set.

        If not wanted unit has been set, data is unchanged, including its unit, if any.
        '''
        if self._wanted_unit is not None:
            if self._force:
                value, _ = separate_value_and_unit(data)
                newdata = value * self._wanted_unit
            else:
                newdata =  make_sure_its_a(self._wanted_unit, data, copy=False)
        else:
            newdata = data

        # Cache unit if applied
        if isinstance(newdata, u.Quantity):
            self._actual_unit = newdata.unit
            self._actual_unit_name = newdata.unit.to_string() or 'dimensionless'
        else:
            self._actual_unit = None
            self._actual_unit_name = None

        return newdata
