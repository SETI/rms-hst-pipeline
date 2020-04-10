from hashlib import md5
from os.path import dirname, join


def path_to_testfile(basename):
    # type: (unicode) -> unicode
    """Return the path to files needed for testing."""
    return join(dirname(__file__), 'testfiles', basename)


def file_md5(filepath):
    # type: (unicode) -> str
    """Find the hexadecimal digest of a file in the filesystem."""
    CHUNK = 4096
    hasher = md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def string_md5(string_to_hash):
    # type: (str) -> str
    """Find the hexadecimal digest of a string."""
    hasher = md5()
    hasher.update(string_to_hash)
    return hasher.hexdigest()


def unicode_md5(ustring_to_hash):
    # type: (unicode) -> str
    """Find the hexadecimal digest of a string."""
    hasher = md5()
    hasher.update(ustring_to_hash.encode('utf-8'))
    return hasher.hexdigest()
