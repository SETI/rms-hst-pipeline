"""
Utility functions to convert from LIDs and LIDVIDs to their canonical
paths in a filesystem.
"""
import re
from typing import List, Optional

from fs.path import iteratepath, join

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

_DIR_PART_PATTERN = re.compile("^v\\$(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*))?$")


def _lid_to_parts(lid: LID) -> List[str]:
    """
    Extract the parts (bundle, collection, product) from the LID and
    return as a list.
    """
    res = [lid.bundle_id]
    if lid.collection_id:
        res.append(lid.collection_id)
    if lid.product_id:
        res.append(lid.product_id)
    return [str(id) for id in res]


def lid_to_dir(lid: LID) -> str:
    """
    Convert a LID to a directory path.
    """
    dir_parts = _lid_to_parts(lid)
    dir_parts.insert(0, "/")
    return join(*dir_parts)


def lidvid_to_dir(lidvid: LIDVID) -> str:
    """
    Convert a LIDVID to a directory path.
    """
    dir = lid_to_dir(lidvid.lid())
    vid_bit = _vid_to_dir_part(lidvid.vid())
    return join(dir, vid_bit)


def dir_to_lid(dir: str) -> LID:
    """
    Convert a directory path to a LID.  Raise on errors.
    """
    parts = [str(part) for part in iteratepath(dir)]
    return LID.create_from_parts(parts)


def dir_to_lidvid(dir: str) -> LIDVID:
    """
    Convert a directory path to a LIDVID.  Raise on errors.
    """

    parts = [str(part) for part in iteratepath(dir)]
    lid_parts = parts[0:-1]
    vid_part = parts[-1]
    lid = LID.create_from_parts(lid_parts)
    vid = dir_part_to_vid(vid_part)
    return LIDVID.create_from_lid_and_vid(lid, vid)


def _vid_to_dir_part(vid: VID) -> str:
    """
    Convert a VID to a directory name.
    """
    return f"v${vid}"


def dir_part_to_vid(dir_part: str) -> VID:
    """
    Convert a directory name to a VID.  Raise on errors.
    """
    assert _is_dir_part(dir_part), f"{dir_part} is not a dir_part"
    return VID(str(dir_part[2:]))


def _is_dir_part(dir_part: str) -> bool:
    """
    Return True if the directory name is of the right format to correspond
    to a VID.
    """
    return bool(_DIR_PART_PATTERN.match(str(dir_part)))
