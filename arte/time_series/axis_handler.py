'''
Axis handler class
'''

class AxisHandler():

    def __init__(self, axes=None):
        '''
        Axes transpositions handlers

        Parameters
        ----------
        axes: sequence of str, optional
            Initial axis order. If None, no axis transpositions will be possible.
        '''
        if axes is None:
            self._axes = ()
        elif isinstance(axes, str):
            self._axes = (axes,)
        else:
            self._axes = tuple(axes)

    def transpose(self, data, axes=None):
        '''
        Parameters
        ----------
        data: ndarray
            data to transpose
        axes: sequence of str, optional
            axis order to apply. If None, no transposition is done.
        '''
        if axes is not None:
            if type(axes) is str:
                axes = [axes]
            # Use an explicit for loop in order to capture an erroneus axis name
            idx = []
            try:
                for ax in axes:
                    idx.append(self._axes.index(ax))
            except ValueError:
                raise ValueError(f'Axis {ax} not found')
            data = data.transpose(idx)
        return data

    def axes(self):
        return self._axes
