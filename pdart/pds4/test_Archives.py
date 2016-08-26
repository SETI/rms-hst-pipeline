from pdart.pds4.Archives import *


def test_get_mini_archive_dir():
    """Just a smoketest to force the parsing of pdart.pds4.Archives."""
    assert get_mini_archive_dir() == '/Users/spaceman/Desktop/Archive'
