
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
    """Plot modal coefficients comparison for adaptive optics analysis.
    
    Creates a plot comparing residual modal coefficients against POL (or other
    reference) modes. Typically used to visualize loop performance by plotting
    closed-loop residuals vs. open-loop measurements.
    
    Parameters
    ----------
    residual_modes_vector : array_like or astropy.units.Quantity
        Residual mode coefficients (typically RMS or std dev per mode)
    pol_modes_vector : array_like or astropy.units.Quantity
        Reference mode coefficients (typically POL modes or open-loop)
    unit : astropy.units.Unit, optional
        Target unit for both vectors. If specified, both inputs will be
        converted to this unit
    overplot : bool, optional
        If True, add to existing plot. If False, clear plot first
        (default: False)
    plot_to : matplotlib.pyplot or matplotlib.axes.Axes, optional
        Matplotlib object to plot to. If None, uses pyplot (default: None)
    title : str, optional
        Plot title. Required when overplot=False
    xlabel : str, optional
        X-axis label (default: 'Mode index')
    ylabel : str, optional
        Y-axis label (default: 'wavefront rms')
    add_unit_to_ylabel : bool, optional
        If True, append unit to ylabel (default: True)
    res_label : str, optional
        Legend label for residual modes (default: 'residual modes')
    res_color : str, optional
        Color for residual modes plot (default: 'red')
    pol_label : str, optional
        Legend label for POL modes (default: 'POL modes')
    pol_color : str, optional
        Color for POL modes plot (default: 'black')
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot
    
    Examples
    --------
    >>> # Basic usage
    >>> modalplot(residuals, pol_modes, title='AO Performance')
    
    >>> # Apply optical gain and specify units
    >>> modalplot(
    ...     analyzer.residual_modes.time_std() * 2,
    ...     analyzer.pol_modes.time_std() * 2,
    ...     unit=u.nm,
    ...     title='Performance with 2x optical gain'
    ... )
    
    >>> # Overplot multiple datasets
    >>> modalplot(res1, pol1, title='Comparison')
    >>> modalplot(res2, pol2, overplot=True,
    ...          res_label='residual modes 2', res_color='blue')
    
    Notes
    -----
    - Automatically handles astropy units and NotAvailable values
    - When overplotting, title parameter is ignored
    - Legend is automatically generated from labels
    """

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
