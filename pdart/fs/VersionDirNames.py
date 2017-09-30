import re

from pdart.pds4.VID import VID

_DIR_NAME_PATTERN = re.compile('^v\\$(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*))?$')


def vid_to_dir_name(vid):
    # type: (VID) -> unicode
    return 'v$%s' % str(vid)


def dir_name_to_vid(dir_name):
    # type: (unicode) -> VID
    assert is_dir_name(dir_name), '%s is not a dir_name' % dir_name
    return VID(dir_name[2:])


def is_dir_name(dir_name):
    # type: (unicode) -> bool
    return bool(_DIR_NAME_PATTERN.match(str(dir_name)))
