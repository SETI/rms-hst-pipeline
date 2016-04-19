import pdart.pds4.Archive


def get_full_archive():
    """
    Return the complete :class:`Archive` located on the multi-terabyte
    external drive connected to the nightly-build machine.
    """
    return pdart.pds4.Archive.Archive('/Volumes/PDART-3TB')


def get_mini_archive():
    """
    Return the small test :class:`Archive` located on the development
    machine.
    """
    return pdart.pds4.Archive.Archive('/Users/spaceman/Desktop/Archive')


def get_any_archive():
    """
    Return the complete :class:`Archive` if running on the
    nightly-build machine; otherwise return the small test
    :class:`Archive` on the development machine.
    """
    try:
        return get_full_archive()
    except:
        return get_mini_archive()
