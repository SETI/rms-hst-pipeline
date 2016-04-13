import pdart.pds4.Archive


def get_full_archive():
    """
    Returns the complete Archive located on the 3TB external drive
    connected to the nightly-build machine.
    """
    return pdart.pds4.Archive.Archive('/Volumes/PDART-3TB')


def get_mini_archive():
    """
    Returns the small test Archive located on my development machine.
    """
    return pdart.pds4.Archive.Archive('/Users/spaceman/Desktop/Archive')


def get_any_archive():
    """
    Return the complete Archive if running on the nightly-build
    machine; otherwise return the small test Archive on the
    development machine.
    """
    try:
        return get_full_archive()
    except:
        return get_mini_archive()
