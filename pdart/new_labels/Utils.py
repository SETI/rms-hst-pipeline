"""
Utility functions.
"""
from fs.path import dirname, join

from pdart.pds4.LIDVID import LIDVID


def lidvid_to_lid(lidvid):
    # type: (str) -> str
    """Return the LID of the LIDVID."""
    return str(LIDVID(lidvid).lid())


def lidvid_to_vid(lidvid):
    # type: (str) -> str
    """Return the VID of the LIDVID."""
    return str(LIDVID(lidvid).vid())


def golden_filepath(basename):
    # type: (unicode) -> unicode
    """Return the path to a golden file with the given basename."""
    return join(dirname(__file__), basename)


def golden_file_contents(basename):
    # type: (unicode) -> unicode
    """
    Return the contents as a Unicode string of the golden file with
    the given basename.
    """
    with open(golden_filepath(basename)) as f:
        return f.read()


def path_to_testfile(basename):
    # type: (unicode) -> unicode
    """Return the path to files needed for testing."""
    return join(dirname(__file__), 'testfiles', basename)
