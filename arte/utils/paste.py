

def _paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos+w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w-max(pos+w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    '''
    Paste `block` array into `wall` in location `loc`.
    
    `loc` must be a tuple (even if block.ndim=1)
    Going outside of the edges of `wall` is properly managed.
    `wall` is modified in place.
    
    Parameters
    ----------
    wall: `~numpy.array` of ndim=N
        Array in which `block` is pasted.
    block: `~numpy.array` of ndim=N
        Array to be pasted into `wall`
    loc: tuple, shape (N,)
        location where to paste `block`.

    Returns
    -------
    None

    Example
    -------
    Insert array [1,2,3,4] into an empty array of size 10 at location 8

    >>> b = np.zeros([10])
    >>> a = np.arange(1, 5)
    >>> b
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    >>> a
    array([1, 2, 3, 4])
    >>> paste(b, a, (8,))
    >>> b
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.])

    References
    ----------
    https://stackoverflow.com/questions/7115437/how-to-embed-a-small-numpy-array-into-a-predefined-block-of-a-large-numpy-arra
    '''

    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(_paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]
