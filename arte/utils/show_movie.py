import logging

import numpy as np
from arte.utils.subplots import subplots


def show_movie(data_cube, show=True, save_mp4=None, interval=10, blit=True,
               repeat=True, repeat_delay=0, offset=None, scale=1.0, **kwargs):
    '''
    Display a movie of the data cube. It requires an interactive backend, with IPython,
    for instance, use magic command: %matplotlib notebook or %matplotlib widget.
    See also:
    https://stackoverflow.com/questions/25333732/matplotlib-animation-not-working-in-ipython-notebook-blank-plot

    All frames share the same color scale (min and max of the whole cube).

    Parameters
    ----------
    data_cube : ndarray or list or tuple
        A 3D ndarray cube [n_frames, dim1, dim2] or a list or tuple of n_frames
        2D ndarray frames [dim1, dim2] with the same size.
    show : bool, optional
        Display the movie. The default is True.
    save_mp4 : str, optional
        Save the movie in a mp4 file with the given filename (e.g. "movie.mp4").
        It requires ffmpeg installed. The default is None.
    interval : int, optional
        The interval between frames in milliseconds. The default is 10.
    blit : bool, optional
        The blit parameter of `~matplotlib.animation.ArtistAnimation`. The default is True.
    repeat : bool, optional
        Repeat the movie. The default is True.
    repeat_delay : int, optional
        Repeat delay in milliseconds. The default is 0.
    offset : ndarray, optional
        A 2D ndarray [dim1, dim2] to subtract from each frame. The default is None.
    scale : float, optional
        Scale the figure size, see `arte.utils.subplots.subplots`. The default is 1.0.
    **kwargs : optional
        Additional arguments to imshow.

    Returns
    -------
    ani : matplotlib.animation.ArtistAnimation
        The animation object. It must be kept referenced, otherwise
        the animation is garbage collected and stops.

    See Also
    --------
    arte.utils.displays.movie : blocking display loop based on plt.pause(),
        suitable for scripts; interval is in seconds.
    arte.utils.displays.savegif : save frames as an animated GIF.

    Example
    -------
    >>> import numpy as np
    >>> from arte.utils.show_movie import show_movie
    >>> data_cube = np.random.rand(10, 32, 32)
    >>> ani = show_movie(data_cube, show=False, interval=50, cmap='gray')
    '''
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    try:
        # [:] to be compatible with AnalyzerStub obj
        data_cube = np.array(data_cube[:])
    except (TypeError, ValueError, IndexError):
        data_cube = None
    if data_cube is None or data_cube.ndim != 3:
        raise TypeError("The input is not a ndarray cube or a list or a tuple of ndarray frames with the same size.")
    n_frames, dim1, dim2 = data_cube.shape

    if offset is not None:
        offset = np.asarray(offset)
        if offset.ndim != 2:
            raise TypeError("offset is not a 2D ndarray")
        if offset.shape != (dim1, dim2):
            raise TypeError("offset must have the same size as frames in data_cube")
        data_cube = data_cube - offset[np.newaxis, :, :]

    fig, ax = subplots(scale=scale)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    vmin = data_cube.min()
    vmax = data_cube.max()
    ims = []
    for i in range(n_frames):
        im = ax.imshow(data_cube[i], animated=True, vmin=vmin, vmax=vmax, **kwargs)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=blit,
                                    repeat=repeat, repeat_delay=repeat_delay)

    if save_mp4 is not None:
        logger = logging.getLogger(__name__)
        logger.info("Building movie file %s ...", save_mp4)
        ani.save(save_mp4)
        logger.info("... movie file %s saved.", save_mp4)

    if show:
        plt.show()
    return ani
