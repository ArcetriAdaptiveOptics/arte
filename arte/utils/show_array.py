import numpy as np


def show_array(array, cut_wings=0, title=None, xlabel='column', ylabel='row', units=None, **kwargs):
    '''
    pyplot.matshow() wrapper to add a colorbar and title/labels

    cut_wings=x means that colorbar is saturated for array values below x percentile
    and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
    0 are forced to 0, values above 50 are set to 50.
    '''
    import matplotlib.pyplot as plt

    if cut_wings <= 0:
        imgplt = plt.matshow(array, **kwargs)
        clb = plt.colorbar()
    else:
        if cut_wings >= 50:
            tmp_cut_wings = 50
        else:
            tmp_cut_wings = cut_wings
        
        vmin = np.percentile(array, tmp_cut_wings)
        vmax = np.percentile(array, 100-tmp_cut_wings)
        imgplt = plt.matshow(array, vmin=vmin, vmax=vmax, **kwargs)
        clb = plt.colorbar(extend='both')
    
    if units is not None:
        clb.ax.set_title(units)
    
    if title is not None:
        imgplt.axes.set_title(title)
    imgplt.axes.set_xlabel(xlabel)
    imgplt.axes.set_ylabel(ylabel)
    plt.show()
    
    return imgplt
 
