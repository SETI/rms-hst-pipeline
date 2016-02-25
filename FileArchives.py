import os.path

import FileArchive


def get_full_archive():
    """
    Returns the complete FileArchive located on the 3TB external drive
    connected to the nightly-build machine.
    """
    return FileArchive.FileArchive('/Volumes/PDART-3TB')


def get_mini_archive():
    """
    Returns the small test FileArchive located on my development machine.
    """
    return FileArchive.FileArchive('/Users/spaceman/Desktop/Archive')


def get_any_archive():
    """
    Return the complete FileArchive if running on the nightly-build
    machine; otherwise return the small test FileArchive on the
    development machine.
    """
    try:
        return get_full_archive()
    except:
        return get_mini_archive()
