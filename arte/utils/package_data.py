

def _data_root_dir_legacy():
    import pkg_resources

    dataroot = pkg_resources.resource_filename(
        'arte',
        'data')
    return dataroot

def dataRootDir():
    try:
        import importlib.resources
        dataroot = importlib.resources.files('arte') / 'data'
    except AttributeError:
        dataroot = _data_root_dir_legacy()
    return dataroot

