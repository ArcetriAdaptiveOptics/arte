

def dataRootDir():
    import importlib.resources
    dataroot = importlib.resources.files('arte') / 'data'
    return dataroot
