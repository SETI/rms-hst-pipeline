import FileArchive

def getFullArchive():
    return FileArchive.FileArchive('/Volumes/PDART-3TB')

def getMiniArchive():
    return FileArchive.FileArchive('/Users/spaceman/Desktop/Archive')

def getAnyArchive():
    try:
        return getFullArchive()
    except:
        return getMiniArchive()
