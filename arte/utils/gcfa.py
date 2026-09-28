
def gcfa(axis=None, overplot=False):
    '''
    Get the current figure and axis or create a new one if not existing.
    If axis is passed, it is used to get the figure and axis.
    If overplot is True, the axis is not cleared before plotting.

    Side effect: if axis is not passed, the current figure is cleared.
    If the current axes is unknown, but you want to avoid to clear it,
    call: gcfa(plt.gca(), overplot=True)

    Parameters
    ----------
    axis: `~matplotlib.axes.Axes`, optional, default=None
        The axis to be used to get the figure and axis
    overplot: bool, optional, default=False
        If True, the axis is not cleared before plotting.
        Used only if axis is passed.

    Returns
    -------
    fig: `~matplotlib.figure.Figure`
        The figure object
    ax: `~matplotlib.axes.Axes`
        The axis object to be used for plotting

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from arte.utils.gcfa import gcfa
    >>> fig, ax = gcfa()                        # clear current figure, new axis
    >>> _ = ax.plot([0, 1], [0, 1])
    >>> fig, ax = gcfa(ax, overplot=True)       # overplot on the same axis
    >>> _ = ax.plot([0, 1], [1, 0])
    >>> len(ax.lines)
    2
    >>> fig, ax = gcfa(ax)                      # clear the axis
    >>> len(ax.lines)
    0

    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gcf.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes
    '''
    import matplotlib.pyplot as plt

    if axis is None:
        fig = plt.gcf()  # get current figure or create a new one if not existing
        fig.clf()        # reset the axes of the figure if already existing
        ax = fig.gca()   # create new axes
    else:
        fig = axis.get_figure()
        ax = axis

        if not overplot:
            # if no overplot is requested, clean the axis and
            # remove color tables that were referring to the
            # images contained in the axis
            for im in ax.images:
                try:
                    # try if there is a list of colorbars
                    for cb in im.colorbar:
                        cb.ax.remove()
                except TypeError:
                    # no or single colorbar
                    if im.colorbar is not None:
                        im.colorbar.ax.remove()
            ax.cla()

    return fig, ax
