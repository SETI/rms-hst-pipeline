"""
Utility functions.
"""
from typing import TYPE_CHECKING
import unittest
from fs.path import dirname, join
import io
from os.path import isfile

from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Union


def lidvid_to_lid(lidvid):
    # type: (str) -> str
    """Return the LID of the LIDVID."""
    return str(LIDVID(lidvid).lid())


def lidvid_to_vid(lidvid):
    # type: (str) -> str
    """Return the VID of the LIDVID."""
    return str(LIDVID(lidvid).vid())


def _golden_filepath(basename):
    # type: (unicode) -> unicode
    """Return the path to a golden file with the given basename."""
    return join(dirname(__file__), basename)


def _golden_file_contents(basename, is_inventory):
    # type: (unicode, bool) -> unicode
    """
    Return the contents as a Unicode string of the golden file with
    the given basename.
    """
    if is_inventory:
        with open(_golden_filepath(basename), 'rb') as f:
            return f.read()
    else:
        with io.open(_golden_filepath(basename)) as f:
            return f.read()

def assert_golden_file_equal(testcase, basename,
                             calculated_contents, is_inventory=False):
    # type: (unittest.TestCase, unicode, Union[str, unicode], bool) -> None
    """
    Finds the golden file and compares its contents with the given string.
    Raises an exception via the unittest.TestCase argument if they are
    unequal.  If the file does not yet exist, it writes the given string to
    the file.  Inventories must be written with CR/NL line-termination, so
    a parameter for that is provided.
    """

    filepath = _golden_filepath(basename)
    if isfile(filepath):
        testcase.assertEqual(_golden_file_contents(filepath, is_inventory),
                             calculated_contents)
    else:
        if is_inventory:
            assert calculated_contents[-2:] == '\r\n', 'inventory input is wrong'
            with open(filepath, 'wb') as f:
                f.write(calculated_contents)
            with open(filepath, 'rb') as f:
                roundtrip = f.read()
                assert roundtrip[-2:] == '\r\n', 'inventory roundtripped is wrong'
        else:
            with io.open(filepath, 'w') as f:
                f.write(unicode(calculated_contents))
        testcase.fail('Golden file %r did not exist but it was created.' %
                      basename)

def path_to_testfile(basename):
    # type: (unicode) -> unicode
    """Return the path to files needed for testing."""
    return join(dirname(__file__), 'testfiles', basename)
