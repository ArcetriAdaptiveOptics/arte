from __future__ import annotations


def subplots(n_row: int = 1, n_col: int = 1, scale: float | int = 1.0):
    '''
    Wrapper to matplotlib.pyplot.subplots adding the possibility to scale the
    size of the generated figure with respect to the standard one.

    The figure size is the matplotlib default size multiplied by n_col
    (width), n_row (height) and scale, so that each subplot keeps the
    default size times scale.

    fig, axs = subplots(n_row=1, n_col=1, scale=1.0)

    Parameters
    ----------
    n_row: int, optional, default=1
        Number of rows of subplots
    n_col: int, optional, default=1
        Number of columns of subplots
    scale: float or int, optional, default=1.0
        Scale factor for the size of each subplot

    Returns
    -------
    fig: `~matplotlib.figure.Figure`
        The figure object
    axs: `~matplotlib.axes.Axes` or `~numpy.ndarray` of `~matplotlib.axes.Axes`
        The axes objects (a single Axes if n_row == n_col == 1)

    Example
    -------
    >>> from arte.utils.subplots import subplots
    >>> fig, axs = subplots(2, 3, scale=0.5)  # 2x3 subplots, each half the default size
    >>> axs.shape
    (2, 3)

    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    '''
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(n_row, n_col)
    fs = fig.get_size_inches()
    fig.set_size_inches([fs[0] * n_col * scale, fs[1] * n_row * scale])
    return fig, axs
