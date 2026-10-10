import numpy as np
from arte.utils.gcfa import gcfa


class ShowDM():
    '''
    Display deformable mirror actuator values as colored dots in the
    actuator positions.

    Parameters
    ----------
    act_coord: `~numpy.ndarray`
        (n_act, 2) or (n_act, 3) array of actuator (x, y[, z]) coordinates
    diam: float or sequence of 2 floats, optional, default=None
        DM diameter, or (inner, outer) diameters. If not None, it is
        plotted as a circle (circles) centered on (0, 0)
    opt_diam: float or sequence of 2 floats, optional, default=None
        Optical (pupil) diameter, or (inner, outer) diameters, plotted
        as circle(s) when display() is called with plot_opt=True
    cmap: str or `~matplotlib.colors.Colormap`, optional, default=None
        Colormap used to display the actuator values

    Attributes
    ----------
    nact: int
        Number of actuators
    x, y, z: `~numpy.ndarray`
        Actuator coordinates. z is zero if act_coord is (n_act, 2)

    Example
    -------
    >>> import numpy as np
    >>> from arte.utils.show_dm import ShowDM
    >>> act_coord = np.array([[0, 0], [1, 1], [2, 2]])
    >>> dm = ShowDM(act_coord, diam=6)
    >>> fig, ax = dm.display(np.array([1, 2, 3]), plot_num=True)
    '''

    def __init__(self, act_coord, diam=None, opt_diam=None, cmap=None):
        act_coord = np.asarray(act_coord)
        self.nact = act_coord.shape[0]
        self.x = act_coord[:, 0]
        self.y = act_coord[:, 1]
        if act_coord.shape[1] > 2:
            self.z = act_coord[:, 2]
        else:
            self.z = np.zeros(self.nact)
        self.diam = diam
        self.opt_diam = opt_diam

        self._theta = np.linspace(0, 2 * np.pi, 361)
        self._cmap = cmap
        self._psize = 17

    @staticmethod
    def _outer(diam):
        if diam is None:
            return None
        if np.isscalar(diam):
            return diam
        return diam[1]

    @staticmethod
    def _inner(diam):
        if diam is None or np.isscalar(diam):
            return None
        return diam[0]

    def dout(self):
        '''Outer DM diameter, or None'''
        return self._outer(self.diam)

    def din(self):
        '''Inner DM diameter, or None'''
        return self._inner(self.diam)

    def opt_dout(self):
        '''Outer optical diameter, or None'''
        return self._outer(self.opt_diam)

    def opt_din(self):
        '''Inner optical diameter, or None'''
        return self._inner(self.opt_diam)

    def _plot_circle(self, ax, diam, *args):
        ax.plot(diam / 2 * np.cos(self._theta), diam / 2 * np.sin(self._theta), *args)

    def display(self, val, act_list=None, title=None, plot_to=None, overplot=False,
                plot_opt=False, plot_num=False, plot_diam=True,
                scale=1.0, colorbar=True):
        '''
        Display actuator values.

        Parameters
        ----------
        val: sequence of float
            Values to display. If act_list is None it must have nact elements,
            otherwise the same number of elements as act_list
        act_list: sequence of int, optional, default=None
            Indexes, in the range [0, nact), of the subset of actuators to display
        title: str, optional, default=None
            Title of the plot
        plot_to: `~matplotlib.axes.Axes`, optional, default=None
            The axis to be used. If None, the current figure is cleared and used
        overplot: bool, optional, default=False
            If True, plot_to axis is not cleared before plotting
        plot_opt: bool, optional, default=False
            If True, plot the optical diameter(s)
        plot_num: bool, optional, default=False
            If True, write the actuator index on each actuator
        plot_diam: bool, optional, default=True
            If True, plot the DM diameter(s)
        scale: float, optional, default=1.0
            Scale factor of the dot size
        colorbar: bool, optional, default=True
            If True, the colorbar is displayed

        Returns
        -------
        fig: `~matplotlib.figure.Figure`
            The figure object
        ax: `~matplotlib.axes.Axes`
            The axis object
        '''
        if act_list is None:
            if len(val) != self.nact:
                raise ValueError('val must have %d (nact) elements' % self.nact)
            idx = np.arange(self.nact)
        else:
            if len(val) != len(act_list):
                raise ValueError('val and act_list must have the same size')
            idx = np.asarray(act_list)

        fig, ax = gcfa(plot_to, overplot)
        sc = ax.scatter(self.x[idx], self.y[idx], s=self._psize * scale**2, c=val, cmap=self._cmap)
        if plot_num:
            for i in idx:
                ax.text(self.x[i], self.y[i], str(i), ha="center", va="center", color="r")
        if plot_diam:
            if self.dout() is not None:
                self._plot_circle(ax, self.dout(), 'k')
            if self.din() is not None:
                self._plot_circle(ax, self.din())
        if plot_opt:
            if self.opt_din() is not None:
                self._plot_circle(ax, self.opt_din())
            if self.opt_dout() is not None:
                self._plot_circle(ax, self.opt_dout())
        if title is not None:
            ax.set_title(title)
        ax.set_aspect('equal')
        if colorbar:
            fig.colorbar(sc, ax=ax)
        return fig, ax
