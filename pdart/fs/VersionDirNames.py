import re

from pdart.pds4.VID import VID
import pdart.fs.DirUtils

# _DIR_NAME_PATTERN = re.compile('^v\\$(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*))?$')


def vid_to_dir_name(vid):
    # type: (VID) -> unicode
    return pdart.fs.DirUtils._vid_to_dir_part(vid)
    # return 'v$%s' % str(vid)


def version_id_to_dir_name(version_id):
    # type: (unicode) -> unicode
    return 'v$%s' % version_id


def dir_name_to_vid(dir_name):
    # type: (unicode) -> VID
    return pdart.fs.DirUtils._dir_part_to_vid(dir_name)
    # assert is_dir_name(dir_name), '%s is not a dir_name' % dir_name
    # return VID(dir_name[2:])


def is_dir_name(dir_name):
    # type: (unicode) -> bool
    return pdart.fs.DirUtils._is_dir_part(dir_name)
    # return bool(_DIR_NAME_PATTERN.match(str(dir_name)))
