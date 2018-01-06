from fs.path import dirname, join

from pdart.pds4.LIDVID import LIDVID


def lidvid_to_lid(lidvid):
    # type: (str) -> str
    return str(LIDVID(lidvid).lid())


def lidvid_to_vid(lidvid):
    # type: (str) -> str
    return str(LIDVID(lidvid).vid())


def golden_filepath(basename):
    # type: (unicode) -> unicode
    return join(dirname(__file__), basename)


def golden_file_contents(basename):
    # type: (unicode) -> unicode
    with open(golden_filepath(basename)) as f:
        return f.read()
