import numpy as np
from arte.utils.gcfa import gcfa


class ShowDetector(object):
    '''
    Display an image array (e.g. a detector frame) with the possibility to
    saturate the top and bottom wings of the colorbar, and to plot
    row/column profiles.

    Parameters
    ----------
    array: `~numpy.ndarray`
        The 2D image array to be displayed

    See Also
    --------
    arte.utils.show_array.show_array : simpler pyplot.matshow() wrapper with
        symmetric colorbar saturation, used by arte.dataelab.

    Example
    -------
    >>> import numpy as np
    >>> from arte.utils.show_detector import ShowDetector
    >>> array = np.random.rand(100, 100)
    >>> sd = ShowDetector(array)
    >>> ax = sd.show(cut_top_wing=1, cut_bottom_wing=1, show=False)

    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    '''

    def __init__(self, array):
        self._array = np.asarray(array)
        self._vmin = np.min(self._array)
        self._vmax = np.max(self._array)

    def show(self, x=None, y=None, cut_top_wing=0, cut_bottom_wing=0, cut_bottom2mode=False,
             title=None, xlabel=None, ylabel=None, units=None,
             log=False, vmin=None, vmax=None, origin='lower',
             xy0=(0, 0), plot_to=None, overplot=False, show=True, cells=None, colorbar=True, **kwargs):
        '''
        Display the image with a colorbar.

        Parameters
        ----------
        x: `~numpy.ndarray`, optional, default=None
            The x-axis values (pixel centers). If None, the x-axis is set to column index
        y: `~numpy.ndarray`, optional, default=None
            The y-axis values (pixel centers). If None, the y-axis is set to row index
        cut_top_wing: float, optional, default=0
            Top value of the colorbar is saturated for array values above the
            (100 - cut_top_wing) percentile. The value is clipped to [0, 100].
            Ignored if vmax is passed.
        cut_bottom_wing: float, optional, default=0
            Bottom value of the colorbar is saturated for array values below the
            cut_bottom_wing percentile. The value is clipped to [0, 100 - cut_top_wing].
            Ignored if vmin is passed or cut_bottom2mode is True.
        cut_bottom2mode: bool, optional, default=False
            If True, the bottom value of the colorbar is set to the most frequent
            (mode) value of the array, overriding cut_bottom_wing.
            Ignored if vmin is passed.
        title: str, optional, default=None
            The title of the plot
        xlabel: str, optional, default=None
            The x-axis label. If None, "column [pix]" or "x" is used
        ylabel: str, optional, default=None
            The y-axis label. If None, "row [pix]" or "y" is used
        units: str, optional, default=None
            The units of the image, shown as colorbar label
        log: bool, optional, default=False
            If True, the image is displayed in log scale. Values <= 0 are
            displayed as the minimum positive value.
        vmin: float, optional, default=None
            The minimum value of the colorbar
        vmax: float, optional, default=None
            The maximum value of the colorbar
        origin: str, optional, default='lower'
            The origin of the image, 'lower' or 'upper'
        xy0: tuple, optional, default=(0, 0)
            Not used, kept for backward compatibility
        plot_to: `~matplotlib.axes.Axes`, optional, default=None
            The axis to be used to plot the image. If None, the current figure
            is cleared and used.
        overplot: bool, optional, default=False
            If True, plot_to axis is not cleared before plotting
        show: bool, optional, default=True
            If True, plt.show() is called
        cells: tuple, optional, default=None
            (n_cells_y, n_cells_x[, color]): draw a grid dividing the image
            in n_cells_y x n_cells_x cells. Default color is 'k'.
        colorbar: bool, optional, default=True
            If True, the colorbar is displayed
        **kwargs: optional
            Additional keyword arguments to be passed to imshow

        Returns
        -------
        ax: `~matplotlib.axes.Axes`
            The axis object used to plot the image
        '''
        import matplotlib.pyplot as plt

        do_min_extend = False
        do_max_extend = False

        array = self._array.copy()

        cut_top_wing = float(np.clip(cut_top_wing, 0, 100))
        cut_bottom_wing = float(np.clip(cut_bottom_wing, 0, 100 - cut_top_wing))

        if vmax is None:
            if cut_top_wing > 0:
                vmax = np.percentile(self._array, 100 - cut_top_wing)
                do_max_extend = True
            else:
                vmax = self._vmax
        else:
            do_max_extend = vmax < self._vmax

        if vmin is None:
            if cut_bottom2mode:
                from scipy import stats
                vmin = np.ravel(stats.mode(self._array, axis=None).mode)[0]
                do_min_extend = True
            elif cut_bottom_wing > 0:
                vmin = np.percentile(self._array, cut_bottom_wing)
                do_min_extend = True
            else:
                vmin = self._vmin
        else:
            do_min_extend = vmin > self._vmin

        if log:
            from matplotlib.colors import LogNorm
            if vmin <= 0.0:
                vmin = np.min(self._array[self._array > 0.0])
                do_min_extend = True
            array[self._array <= 0.0] = vmin
            norm = LogNorm(vmin, vmax)
        else:
            norm = None

        ss = array.shape
        if x is None:
            x0 = 0 - 0.5
            x1 = ss[1] - 0.5
            if xlabel is None:
                xlabel = "column [pix]"
            if ylabel is None:
                ylabel = "row [pix]"
        else:
            dx = x[1] - x[0]
            x0 = x.min() - dx / 2
            x1 = x.max() + dx / 2
            if xlabel is None:
                xlabel = "x"
            if ylabel is None:
                ylabel = "y"

        if y is None:
            y0 = 0 - 0.5
            y1 = ss[0] - 0.5
        else:
            dy = y[1] - y[0]
            y0 = y.min() - dy / 2
            y1 = y.max() + dy / 2

        if origin == 'upper':
            extent = (x0, x1, y1, y0)
        else:
            extent = (x0, x1, y0, y1)

        fig, ax = gcfa(plot_to, overplot)
        if norm is None:
            imgplt = ax.imshow(array, vmin=vmin, vmax=vmax, origin=origin,
                               extent=extent, **kwargs)
        else:
            imgplt = ax.imshow(array, norm=norm, origin=origin,
                               extent=extent, **kwargs)

        if cells is not None:
            color = 'k' if (len(cells) < 3) else cells[2]
            for i in range(1, cells[0]):
                ax.plot(np.array([0, ss[1]]) - 0.5, np.zeros(2) + ss[0] / cells[0] * i - 0.5, color)
            for i in range(1, cells[1]):
                ax.plot(np.zeros(2) + ss[1] / cells[1] * i - 0.5, np.array([0, ss[0]]) - 0.5, color)

        extend = {(False, False): 'neither',
                  (True, False): 'min',
                  (False, True): 'max',
                  (True, True): 'both'}[(bool(do_min_extend), bool(do_max_extend))]

        if colorbar:
            clb = fig.colorbar(imgplt, ax=ax, extend=extend)
            if units is not None:
                clb.set_label(units, rotation=270)

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show:
            plt.show()

        return ax

    def imshow(self, *args, **kwargs):
        '''Same as show'''
        return self.show(*args, **kwargs)

    def plot_x_profile(self, x, y, plot_to=None, overplot=False, show=True, pix_scale=1., label=''):
        '''
        Plot the profile along row y, with abscissa centered on column x
        and scaled by pix_scale. Returns the axis used for plotting.
        '''
        import matplotlib.pyplot as plt
        _, ax = gcfa(plot_to, overplot)
        xx = (np.arange(self._array.shape[1]) - x) * pix_scale
        ax.plot(xx, self._array[y], label=label + 'x-profile')
        if show:
            plt.show()
        return ax

    def plot_y_profile(self, x, y, plot_to=None, overplot=False, show=True, pix_scale=1., label=''):
        '''
        Plot the profile along column x, with abscissa centered on row y
        and scaled by pix_scale. Returns the axis used for plotting.
        '''
        import matplotlib.pyplot as plt
        _, ax = gcfa(plot_to, overplot)
        yy = (np.arange(self._array.shape[0]) - y) * pix_scale
        ax.plot(yy, self._array[:, x], label=label + 'y-profile')
        if show:
            plt.show()
        return ax

    def plot_xy_profile(self, x, y, plot_to=None, overplot=False, show=True, pix_scale=1., label=''):
        '''
        Plot both row and column profiles through pixel (x, y) on the same axis.
        Returns the axis used for plotting.
        '''
        _, ax = gcfa(plot_to, overplot)
        self.plot_x_profile(x, y, plot_to=ax, overplot=True, show=False, pix_scale=pix_scale, label=label)
        return self.plot_y_profile(x, y, plot_to=ax, overplot=True, show=show, pix_scale=pix_scale, label=label)

    def plot_xyc_profile(self, plot_to=None, overplot=False, show=True, pix_scale=1., label=''):
        '''
        Plot both row and column profiles through the central pixel.
        Returns the axis used for plotting.
        '''
        y, x = np.array(self._array.shape) // 2
        return self.plot_xy_profile(x, y, plot_to=plot_to, overplot=overplot, show=show,
                                    pix_scale=pix_scale, label=label)
