import numpy as np
from astropy import units as u

from arte.time_series.time_series import TimeSeries
from arte.utils.help import modify_help
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import ConstantDataLoader, DataLoader, guess_data_loader
from arte.dataelab.unit_handler import UnitHandler


class BaseTimeSeries(TimeSeries):
    '''
    Generic time series.

    Parameters
    ----------
    data_loader: instance of DataLoader or derived class, or numpy array
         time series data
    time_vector: instance of DataLoader or derived class, or numpy_array, or None
         time vector data. If None, a default counter from 0 to N-1 samples will be assigned
    astropy_unit: astropy unit or None
         if possible, astropy unit to use with the data.
    data_label: string or None
         human-readable label for plot (e.g.: "Surface modal coefficients" )

    '''
    def __init__(self, loader_or_data, time_vector=None, astropy_unit=None, data_label=None):

        if isinstance(loader_or_data, np.ndarray):
            data_loader = ConstantDataLoader(loader_or_data)
        elif isinstance(loader_or_data, str):
            data_loader = guess_data_loader(loader_or_data)
        elif isinstance(loader_or_data, DataLoader):
            data_loader = loader_or_data
        else:
            raise ValueError('loader_or_data must be a Loader class, a numpy array, or a filename')

        if isinstance(time_vector, np.ndarray):
            time_vector_loader = ConstantDataLoader(time_vector)
        elif isinstance(time_vector, str):
            time_vector_loader = guess_data_loader(time_vector)
        elif isinstance(time_vector, DataLoader):
            time_vector_loader = time_vector
        elif time_vector is None:
            time_vector_loader = None
        else:
            raise ValueError('time_vector must be a Loader class, a numpy array, a filename, or None')

        try:
            super().__init__()

            # Also test that the data file is there, when possible
            _ = data_loader.assert_exists()
            if time_vector_loader is not None:
                _ = time_vector_loader.assert_exists()

        except AssertionError:
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = data_loader
        self._time_vector_loader = time_vector_loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    def _get_not_indexed_data(self):
        '''Reimplementation for lazy loading and astropy units'''
        return self._unit_handler.apply_unit(self._data_loader.load())

    def _get_time_vector(self):
        '''Reimplementation for lazy loading'''
        if self._time_vector_loader is not None:
            return self._time_vector_loader.load()
        else:
            return super()._get_time_vector()

    def get_index_of(self, *args, **kwargs):
        '''Return a selection index

        Parameters
        ----------
        *args: tuple
           user-defined arguments for data selection
        **kwargs: dict, optional
           extra user-defined arguments for data selection

        Returns
        -------
        index: None, integer, list of integers, slice objects, or a tuple of the previous possibilities.
           indexes to select a data subset. If None, all data is selected.
           An integer, a list of integers or a slice will select those rows.
           If the data is multidimensional, a tuple can be returned where each element
           will select data across a single dimension.
           For a detailed explanation, see the "Advanced indexing" topic at
           https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        '''
        pass

    def data_label(self):
        return self._data_label

    def data_unit(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name()

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()

    # Override to provide custom displays
    def _get_display_cube(self, data_to_display):
        '''Generate a 3d cube for display'''
        return np.atleast_3d(data_to_display)

    @modify_help(arg_str='[series_idx], [times=[from, to]]')
    def get_display(self, *args, times=None, **kwargs):
        '''Display cube for the specified time interval'''
        data_to_display = self.get_data(*args, times=times, **kwargs)
        display_data = self._get_display_cube(data_to_display)
        if isinstance(display_data, u.Quantity):
            return display_data.value
        else:
            return display_data
        
    @modify_help(call='plot_hist([series_idx], from_freq=xx, to_freq=xx, )')
    def plot_hist(self, *args, from_t=None, to_t=None,
                  overplot=None, plot_to=None,
                  label=None,  **kwargs):
        '''Plot histogram. TODO: does not work, rewrite'''
        hist = self.get_data(*args, **kwargs)
        t = self._get_time_vector()
        if from_t is None: from_t=t.min()
        if to_t is None: to_t=t.max()
        ul = t <= to_t
        dl = t >= from_t
        lim = ul & dl
        t = t[lim]
        hist = hist[lim]

        from matplotlib.axes import Axes
        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to
        if not overplot:
            plt.cla()
            plt.clf()

        lines = plt.plot(t, hist)
        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = 'Time history'
        xlabel = 'time [s]'
        ylabel = '['+units+']'
        if self.data_label() is not None:
            title = title + " of " + self.data_label()
            ylabel = self.data_label() + " " + ylabel

        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt

    @modify_help(call='plot_spectra([series_idx], from_freq=xx, to_freq=xx)')
    def plot_spectra(self, *args, from_freq=None, to_freq=None,
                     segment_factor=None,
                     overplot=False,
                     label=None, plot_to=None,
                     lineary=False, linearx=False,
                     **kwargs):
        '''Plot PSD'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cut_frequency()

        # linearx=True forces lineary to True
        if linearx: lineary=True
        if lineary and not linearx:
            p_shape = power.shape
            if len(p_shape)== 1:
                power *= freq
            else:
                power *= np.repeat(freq.reshape(p_shape[0],1), p_shape[1], axis=1)

        from matplotlib.axes import Axes

        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to

        if not overplot:
            plt.cla()
            plt.clf()
        lines = plt.plot(freq[1:], power[1:])

        if lineary and not linearx:
            plt.semilogx()
        elif linearx and not lineary:
            plt.semilogy()
        elif not (linearx and lineary):
            plt.loglog()

        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = 'PSD'
        if self.data_label() is not None:
            title = title + " of " + self.data_label()
        xlabel = 'frequency [Hz]'
        if lineary and not linearx:
            ylabel = 'frequency*PSD ['+units+'^2]'
        else:
            ylabel = 'PSD ['+units+'^2]/Hz'
        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt

    @modify_help(call='plot_cumulative_spectra([series_idx], from_freq=xx, to_freq=xx)')
    def plot_cumulative_spectra(self, *args, from_freq=None, to_freq=None,
                                segment_factor=None,
                                label=None, plot_to=None,
                                overplot=False, plot_rms=False, lineary=False,
                                **kwargs):
        '''Plot cumulative PSD'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cut_frequency()
        freq_bin = freq[1]-freq[0] # frequency bin

        from matplotlib.axes import Axes

        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to

        if not overplot:
            plt.cla()
            plt.clf()

        # cumulated PSD
        cumpsd = np.cumsum(power, 0) * freq_bin
        if plot_rms:
            # cumulated RMS
            cumpsd = np.sqrt(cumpsd)
            label_str = "RMS"
            label_pow = ""
        else:
            # cumulated PSD
            label_str = "Variance"
            label_pow = "^2"

        lines = plt.plot(freq[1:], cumpsd[1:])
        if lineary:
            plt.semilogx()
        else:
            plt.loglog()

        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = "Cumulated " + label_str
        if self.data_unit() is not None:
            title = title + " of " + self.data_label()
        xlabel = 'frequency [Hz]'
        ylabel = 'Cumulated '+label_str+' ['+units+label_pow+']'
        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt



# __oOo__
