"""
Utility functions.
"""
import unittest
from fs.path import dirname, join
import io
from os.path import isfile

from pdart.pds4.LIDVID import LIDVID

from typing import Union


def lidvid_to_lid(lidvid: str) -> str:
    """Return the LID of the LIDVID."""
    return str(LIDVID(lidvid).lid())


def lidvid_to_vid(lidvid: str) -> str:
    """Return the VID of the LIDVID."""
    return str(LIDVID(lidvid).vid())


def _golden_filepath(basename: str) -> str:
    """Return the path to a golden file with the given basename."""
    return join(dirname(__file__), basename)


def _golden_file_contents(basename: str) -> bytes:
    # Leave untyped due to problems with open()/io.open().
    """
    Return the contents as a Unicode string of the golden file with
    the given basename.
    """
    with open(_golden_filepath(basename), "rb") as f:
        return f.read()


def assert_golden_file_equal(
    testcase: unittest.TestCase, basename: str, calculated_contents: bytes
) -> None:
    # Leave untyped due to problems with open()/io.open().
    """
    Finds the golden file and compares its contents with the given string.
    Raises an exception via the unittest.TestCase argument if they are
    unequal.  If the file does not yet exist, it writes the given string to
    the file.  Inventories must be written with CR/NL line-termination, so
    a parameter for that is provided.
    """

    filepath = _golden_filepath(basename)
    if isfile(filepath):
        contents = _golden_file_contents(filepath)
        testcase.assertEqual(contents, calculated_contents)
    else:
        with open(filepath, "wb") as f:
            f.write(calculated_contents)
        testcase.fail(f"Golden file {basename!r} did not exist but it was created.")


def path_to_testfile(basename: str) -> str:
    """Return the path to files needed for testing."""
    return join(dirname(__file__), "testfiles", basename)
