
def dataRootDir():
    import pkg_resources

    dataroot = pkg_resources.resource_filename(
        'arte',
        'data')
    return dataroot
