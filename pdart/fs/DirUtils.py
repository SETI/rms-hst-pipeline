"""
Utility functions to convert from LIDs and LIDVIDs to their canonical
paths in a filesystem.
"""
import re

from fs.path import iteratepath, join

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

_DIR_PART_PATTERN = re.compile('^v\\$(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*))?$')


def _lid_to_parts(lid):
    """
    Extract the parts (bundle, collection, product) from the LID and
    return as a list.
    """
    # type: (LID) -> List[unicode]
    return filter(lambda (x): x is not None,
                  [lid.bundle_id, lid.collection_id, lid.product_id])


def lid_to_dir(lid):
    """
    Convert a LID to a directory path.
    """
    # type: (LID) -> unicode
    dir_parts = _lid_to_parts(lid)
    dir_parts.insert(0, u'/')
    return apply(join, dir_parts)


def lidvid_to_dir(lidvid):
    """
    Convert a LIDVID to a directory path.
    """
    # type: (LIDVID) -> unicode
    dir = lid_to_dir(lidvid.lid())
    vid_bit = _vid_to_dir_part(lidvid.vid())
    return join(dir, vid_bit)


def dir_to_lid(dir):
    """
    Convert a directory path to a LID.  Raise on errors.
    """
    # type: (unicode) -> LID
    parts = iteratepath(dir)
    return LID.create_from_parts(parts)


def dir_to_lidvid(dir):
    """
    Convert a directory path to a LIDVID.  Raise on errors.
    """

    # type: (unicode) -> LIDVID
    parts = iteratepath(dir)
    lid_parts = parts[0:-1]
    vid_part = parts[-1]
    lid = LID.create_from_parts(lid_parts)
    vid = dir_part_to_vid(vid_part)
    return LIDVID.create_from_lid_and_vid(lid, vid)


def _vid_to_dir_part(vid):
    """
    Convert a VID to a directory name.
    """
    # type: (VID) -> unicode
    return 'v$%s' % str(vid)


def dir_part_to_vid(dir_part):
    """
    Convert a directory name to a VID.  Raise on errors.
    """
    # type: (unicode) -> VID
    assert _is_dir_part(dir_part), '%s is not a dir_part' % dir_part
    return VID(dir_part[2:])


def _is_dir_part(dir_part):
    """
    Return True if the directory name is of the right format to correspond
    to a VID.
    """
    # type: (unicode) -> bool
    return bool(_DIR_PART_PATTERN.match(str(dir_part)))
