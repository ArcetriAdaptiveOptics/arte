import numpy as np

from arte.utils.subplots import subplots


def show_mode_page(modal_shapes, from_page=0, n_pages=1, pupil=None, nrc=(8, 10),
                   saturation=0.5, scale=3.5, show=True):
    '''
    Display modal shapes tiled in pages of nr x nc modes, one figure per page.

    Parameters
    ----------
    modal_shapes: `~numpy.ndarray`
        If pupil is None: (y_size, x_size, n_modes) array of 2D modal shapes.
        If pupil is not None: (n_pup, n_modes) array with the values of the
        modes on the n_pup = pupil.sum() valid pixels.
    from_page: int, optional, default=0
        First page to show
    n_pages: int, optional, default=1
        Number of pages to show
    pupil: `~numpy.ndarray` of bool, optional, default=None
        (y_size, x_size) pupil mask, True inside the pupil
    nrc: tuple of 2 int, optional, default=(8, 10)
        Number of rows and columns of modes in each page
    saturation: float, optional, default=0.5
        Values are clipped to +/- saturation times the maximum absolute
        value of the page
    scale: float, optional, default=3.5
        Figure scale factor, see `arte.utils.subplots.subplots`
    show: bool, optional, default=True
        If True, plt.show() is called for each page

    Returns
    -------
    axs: list of `~matplotlib.axes.Axes`
        The axes of the displayed pages

    Example
    -------
    >>> import numpy as np
    >>> from arte.utils.show_mode_page import show_mode_page
    >>> modal_shapes = np.random.rand(16, 16, 100)
    >>> axs = show_mode_page(modal_shapes, 0, 2, nrc=(8, 10), show=False)
    >>> len(axs)
    2
    '''
    import matplotlib.pyplot as plt

    nr, nc = nrc
    if pupil is None:
        mshape = modal_shapes.shape
        if len(mshape) != 3:
            raise ValueError('If pupil is not passed, modal_shapes must be y_size x x_size x n_modes.')
        sr, sc, nm = mshape
    else:
        sr, sc = pupil.shape
        nm = modal_shapes.shape[-1]

    axs = []
    for page in range(from_page, from_page + n_pages):
        first_mode = nr * nc * page
        if first_mode >= nm:
            break
        aa = np.full((sr * nr, sc * nc), np.nan)
        for i in range(nr):
            for j in range(nc):
                idx = j + nc * i + first_mode
                if idx < nm:
                    if pupil is None:
                        vv = modal_shapes[:, :, idx]
                    else:
                        vv = np.full((sr, sc), np.nan)
                        vv[pupil] = modal_shapes[:, idx]
                    aa[i * sr:(i + 1) * sr, j * sc:(j + 1) * sc] = np.flipud(vv)

        aa_p = np.nanmax(np.abs(aa)) * saturation
        aa = np.clip(aa, -aa_p, aa_p)
        _, ax = subplots(scale=scale)
        ax.imshow(aa)
        ax.set_title('Page#{:.0f} - Modes from #{:.0f} to #{:.0f}'.format(
            page, first_mode, min(first_mode + nr * nc, nm) - 1))
        axs.append(ax)
        if show:
            plt.show()
    return axs
