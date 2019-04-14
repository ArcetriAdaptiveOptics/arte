
def dataRootDir():
    import pkg_resources

    dataroot = pkg_resources.resource_filename(
        'apposto',
        'data')
    return dataroot
