"""Functionality to find some standard archives."""
import pdart.pds4.Archive


def get_full_archive_dir():
    # type: () -> unicode
    """
    Return the filepath to the complete :class:`~pdart.pds4.Archive`
    on the multi-terabyte external drive connected to the
    nightly-build machine.
    """
    return '/Volumes/PDART-5TB Part One'


def get_full_archive():
    # type: () -> pdart.pds4.Archive.Archive
    """
    Return the complete :class:`~pdart.pds4.Archive` located on the
    multi-terabyte external drive connected to the nightly-build
    machine.
    """
    return pdart.pds4.Archive.Archive(get_full_archive_dir())


def get_mini_archive_dir():
    # type: () -> unicode
    """
    Return the filepath to the small test :class:`~pdart.pds4.Archive`
    located on the development machine.
    """
    return '/Users/spaceman/Desktop/Archive'


def get_mini_archive():
    # type: () -> pdart.pds4.Archive.Archive
    """
    Return the small test :class:`~pdart.pds4.Archive` located on the
    development machine.
    """
    return pdart.pds4.Archive.Archive(get_mini_archive_dir())


def get_any_archive():
    # type: () -> pdart.pds4.Archive.Archive
    """
    Return the complete :class:`~pdart.pds4.Archive` if running on the
    nightly-build machine; otherwise return the small test
    :class:`~pdart.pds4.Archive` on the development machine.
    """
    try:
        return get_full_archive()
    except:
        return get_mini_archive()


def get_any_archive_dir():
    # type: () -> unicode
    """
    Return the root directory of the archive returned by
    :func:`get_any_archive`.
    """
    return get_any_archive().root
