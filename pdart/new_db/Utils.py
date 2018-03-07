from os.path import dirname, join


def path_to_testfile(basename):
    # type: (unicode) -> unicode
    """Return the path to files needed for testing."""
    return join(dirname(__file__), 'testfiles', basename)
