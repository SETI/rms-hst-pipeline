"""
Utility functions.
"""
from datetime import date
from typing import Any, Dict, List

import unittest
from os.path import isfile

from fs.path import dirname, join

from pdart.db.bundle_db import BundleDB
from pdart.pds4.lidvid import LIDVID
from pdart.xml.Templates import NodeBuilder

"""The suffixes considered raw data, in order of preference."""
MOD_DATE_FOR_TESTESING: str = "2016-04-20"


def get_current_date() -> str:
    return date.today().strftime("%Y-%m-%d")


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
        testcase.assertEqual(contents, calculated_contents, f"golden file {filepath!r}")
    else:
        with open(filepath, "wb") as f:
            f.write(calculated_contents)
        testcase.fail(f"Golden file {basename!r} did not exist but it was created.")


def path_to_testfile(basename: str) -> str:
    """Return the path to files needed for testing."""
    return join(dirname(__file__), "testfiles", basename)


def wavelength_from_range(min_range: float, max_range: float) -> List[str]:
    """
    Takes in a range, and return a list of relevant wavelength range names.
    Here is the wavelength table that we are interested at:
    from (microns)  to  (microns)     name           PDS4 range
        0.01    -           0.400     Ultraviolet    (10 and 400 nm)
        0.390   -           0.700     Visible        (390 and 700 nm)
        0.65    -           5.0       Near Infrared  (0.65 and 5.0 micrometers)
        0.75    -         300         Infrared       (0.75 and 300 micrometers)
       30       -         300         Far Infrared   (30 and 300 micrometers)
    """
    WAVELENGTH_NAMES = [
        "Ultraviolet",
        "Visible",
        "Near Infrared",
        "Infrared",
        "Far Infrared",
    ]

    if max_range < min_range:
        raise ValueError("Invalid range passed in")
    min_idx, max_idx = 0, 0

    if min_range < 0.01 or (min_range >= 0.01 and min_range <= 0.4):
        min_idx = 0
    elif min_range >= 0.39 and min_range <= 0.7:
        min_idx = 1
    elif min_range >= 0.65 and min_range <= 5:
        min_idx = 2
    elif min_range >= 0.75 and min_range <= 300:
        min_idx = 3
    elif min_range >= 30 and min_range <= 300:
        min_idx = 4
    elif min_range > 300:
        return []

    if max_range > 300 or (max_range >= 30 and max_range <= 300):
        max_idx = 4
    elif max_range >= 0.75 and max_range <= 300:
        max_idx = 3
    elif max_range >= 0.65 and max_range <= 5:
        max_idx = 2
    elif max_range >= 0.39 and max_range <= 0.7:
        max_idx = 1
    elif max_range >= 0.01 and max_range <= 0.4:
        max_idx = 0
    elif max_range < 0.01:
        return []

    return WAVELENGTH_NAMES[min_idx : max_idx + 1]


def date_time_to_date(date_time: str) -> str:
    """
    Take in a date_time string and return a date string, for exampel:
    "2005-01-19T15:41:05Z" to "2005-01-19"
    """
    try:
        idx = date_time.index("T")
    except:
        raise ValueError("Failed to convert from date_time to date")

    return date_time[:idx]
