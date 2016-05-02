import os.path

import OldFileArchive


def getFullArchive():
    return OldFileArchive.OldFileArchive('/Volumes/PDART-3TB')


def getMiniArchive():
    return OldFileArchive.OldFileArchive('/Users/spaceman/Desktop/Archive')


def getAnyArchive():
    try:
        return getFullArchive()
    except:
        return getMiniArchive()


# Runs a given unary function on all filepaths in the archive.
def forAllFiles(func):
    a = getAnyArchive()
    for inst, prop, vis, f in a.walkFiles():
        file = os.path.join(a.visitFilepath(inst, prop, vis), f)
        func(file)
