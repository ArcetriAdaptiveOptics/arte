
"""
Display functions
"""

import numpy as np


def movie(frames, interval=0.1, *args, **kwargs):
    '''
    Parameters
    ----------
    frames: ndarray
        3d array [time, rows, cols] with the data to display
    interval: float. optional
        delay between frames in seconds, default=0.1 seconds
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    for i, frame in enumerate(frames):
        if i == 0:
            img = ax.imshow(frame, *args, **kwargs)
        else:
            img.set_data(frame)
        fig.canvas.draw_idle()
        plt.pause(interval)


def savegif(frames, filename, interval=0.1, loop=0):
    '''
    Parameters
    ----------
    frames: ndarray
        3d array [time, rows, cols] with the data to save
    filename: string, pathlib.Path object or file object
        filename or path or fileobject where the animated GIF is saved
    interval: float
        delay between frames in seconds
    loop: int
        number of times the GIF should loop. Default 0, which means to loop forever.
    '''
    from PIL import Image
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(filename, format='gif', save_all=True, append_images=imgs[1:], duration=interval, loop=loop)


def _grouper(iterable, n):
    # From itertools recipe: grouper('ignore'):
    # grouper('ABCDEFG', 3) --> ABC DEF
    return zip(*[iter(iterable)] * n)


def tile(frames, rowlength=10):
    '''
    Parameters
    ----------
    frames: ndarray
        3d array [time, rows, cols] with the data to display
    rowlength: int, optional
        number of frames to arrange in a row
    '''
    rows = [np.hstack(x) for x in _grouper(frames, rowlength)]
    return np.vstack(rows)


# ___oOo___
