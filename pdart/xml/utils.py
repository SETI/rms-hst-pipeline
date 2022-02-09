from os.path import dirname, join


def path_to_testfile(basename: str) -> str:
    """Return the path to files needed for testing."""
    return join(dirname(__file__), "testfiles", basename)
