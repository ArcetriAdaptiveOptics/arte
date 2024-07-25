
import numpy as np
import astropy.units as u

from matplotlib import pyplot
from matplotlib.axes import Axes

from arte.utils.not_available import NotAvailable
from arte.utils.unit_checker import make_sure_its_a, separate_value_and_unit


def setup_plot(plot_to=None, overplot=False, title=None,
               xlabel='', ylabel='', logx=False, logy=False):
    '''Convenience function for plot setup
    
    Can work both on the main pyplot module or, using the plot_to argument,
    on a particular plot axis
        
    Returns the same input as plot_to, or a new plt module instance.
    '''
    from matplotlib.axes import Axes

    if plot_to is None:
        import matplotlib.pyplot as plt
    else:
        plt = plot_to

    if not overplot:
        plt.cla()
        plt.clf()

    if logx and logy:
        plt.loglog()
    elif logx:
        plt.semilogx()
    elif logy:
        plt.semilogy()

    if isinstance(plt, Axes):
        plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
    else:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return plt


def modalplot(residual_modes_vector, pol_modes_vector, unit=None,
            overplot=False, plot_to=None,
            title=None, xlabel='Mode index', ylabel='wavefront rms',
            res_label='residual modes', res_color='red',
            pol_label='POL modes', pol_color='black'):
    '''
    Example to apply an optical gain of 2:

    modalplot(self.residual_modes.time_std() * 2, 
              self.pol_modes.time_std() * 2,
              )
    '''
    res_data = residual_modes_vector
    pol_data = pol_modes_vector

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

    ylabel = f'{ylabel} [{plot_unit}]'

    plt = setup_plot(plot_to=plot_to, overplot=overplot, title=title,
                    xlabel=xlabel, ylabel=ylabel, logx=True, logy=True)

    if len(res_data) > 0:
        plt.plot(np.arange(len(res_data)), res_value,
                    color=res_color, label=res_label)
    if len(pol_data) > 0:
        plt.plot(np.arange(len(pol_data)), pol_value,
                    color=pol_color, label=pol_label)

    maxlen = max(len(res_data), len(pol_data))

    if not overplot:
        plt.xscale('symlog')    # Allow 0 index in log plots
#            plt.yscale('symlog')
        plt.xlim([0, maxlen])

        if not (res_label is None and pol_label is None):
            plt.legend()
    return plt
