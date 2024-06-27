
import numpy as np
import astropy.units as u

from matplotlib import pyplot
from matplotlib.axes import Axes

from arte.utils.not_available import NotAvailable
from arte.utils.unit_checker import make_sure_its_a, separate_value_and_unit


def modalplot(residual_modes_vector, pol_modes_vector, unit=None,
            overplot=False, plot_to=None,
            title=None, xlabel='Mode index', ylabel='wavefront rms', add_unit_to_ylabel=True,
            res_label='residual modes', res_color='red',
            pol_label='POL modes', pol_color='black'):
    '''
    Example to apply an optical gain of 2:

    modalplot(self.residual_modes.time_std() * 2, 
              self.pol_modes.time_std() * 2,
              )
    '''

    if plot_to is not None:
        plt = plot_to
    else:
        plt = pyplot

    res_data = residual_modes_vector
    pol_data = pol_modes_vector

    if not overplot:
        plt.cla()
        plt.clf()
        if title is None:
            raise ValueError('Title must be set when not overplotting')

    # Make sure that the two units match if at all possible,
    # taking care of leaving NotAvailable values untouched.
    res_valid = not isinstance(res_data, NotAvailable)
    pol_valid = not isinstance(pol_data, NotAvailable)
    res_quantity = isinstance(res_data, u.Quantity)
    pol_quantity = isinstance(pol_data, u.Quantity)
    if unit is not None:
        res_data = make_sure_its_a(unit, res_data)
        pol_data = make_sure_its_a(unit, pol_data)
    elif res_valid and pol_valid and (res_quantity or pol_quantity):
        pol_data = make_sure_its_a(res_data.unit, pol_data)

    res_value, res_unit = separate_value_and_unit(res_data)
    pol_value, pol_unit = separate_value_and_unit(pol_data)

    # Select what to display on the Y axis.
    plot_unit = ''
    if res_quantity:
        plot_unit = res_unit
    elif pol_quantity:
        plot_unit = pol_unit
    if not isinstance(plot_unit, u.UnitBase):
        plot_unit = 'a.u.'
    
    if len(res_data) > 0:
        plt.plot(np.arange(len(res_data)), res_value,
                    color=res_color, label=res_label)
    if len(pol_data) > 0:
        plt.plot(np.arange(len(pol_data)), pol_value,
                    color=pol_color, label=pol_label)

    maxlen = max(len(res_data), len(pol_data))

    if add_unit_to_ylabel:
        ystr = f'{ylabel} [{plot_unit}]'
    else:
        ystr = ylabel

    if not overplot:
        if isinstance(plt, Axes):
            plt.set(xlabel=xlabel, ylabel=ystr)
            if title is not None:
                plt.set(title=title)
        else:
            plt.loglog()
            plt.xscale('symlog')    # Allow 0 index in log plots
#            plt.yscale('symlog')
            plt.xlim([0, maxlen])

            plt.xlabel(xlabel)
            plt.ylabel(ystr)
            if title is not None:
                plt.title(title)
            if not (res_label is None and pol_label is None):
                plt.legend()
    return plt
