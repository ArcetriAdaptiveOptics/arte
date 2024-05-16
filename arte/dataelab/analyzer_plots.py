
import numpy as np
import astropy.units as u

from matplotlib import pyplot
from matplotlib.axes import Axes

from arte.utils.not_available import NotAvailable
from arte.utils.unit_checker import make_sure_its_a, separate_value_and_unit


def modalplot(residual_modes_vector, pol_modes_vector, unit=u.nm,
            overplot=False, plot_to=None,
            title=None, xlabel='Mode index', ylabel='wavefront rms', add_unit_to_ylabel=True,
            cl_label='closed loop', cl_color='red',
            ol_label='open loop', ol_color='black'):
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

    cl_std = residual_modes_vector
    ol_std = pol_modes_vector

    if not overplot:
        plt.cla()
        plt.clf()
        if title is None:
            raise ValueError('Title must be set when not overplotting')

    maxlen = 0
    if not isinstance(cl_std, NotAvailable):
        if unit:
            cl_std = make_sure_its_a(unit, cl_std)
        value, _ = separate_value_and_unit(cl_std)
        plt.plot(np.arange(len(cl_std)), value,
                    color=ol_color, label=cl_label)
        maxlen = max(maxlen, len(cl_std))

    if not isinstance(ol_std, NotAvailable):
        if unit:
            ol_std = make_sure_its_a(unit, ol_std)
        value, _ = separate_value_and_unit(ol_std)
        plt.plot(np.arange(len(ol_std)), value,
                    color=cl_color, label=ol_label)
        maxlen = max(maxlen, len(ol_std))

    if add_unit_to_ylabel and unit:
        ystr = f'{ylabel} [{unit.name}]'
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
            plt.yscale('symlog')
            plt.xlim([0, maxlen])

            plt.xlabel(xlabel)
            plt.ylabel(ystr)
            if title is not None:
                plt.title(title)
            if not (cl_label is None and ol_label is None):
                plt.legend()
    return plt
