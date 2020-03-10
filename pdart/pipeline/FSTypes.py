import re
from fs.path import parts
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fs.base import FS

EMPTY_FS_TYPE = "empty"
SINGLE_VERSIONED_FS_TYPE = "single-versioned"
MULTIVERSIONED_FS_TYPE = "multiversioned"
UNKNOWN_FS_TYPE = "unknown"


def is_version_dir(dirpath):
    PAT = "^v\\$[0-9]+\\.[0-9]+$"
    last_part = parts(dirpath)[-1]
    res = re.search(PAT, last_part)
    return res is not None


def categorize_filesystem(fs):
    # type: (FS) -> str
    top_level_listing = fs.listdir(u"/")
    if not top_level_listing:
        return EMPTY_FS_TYPE
    elif any(name[-1] == "$" for name in top_level_listing):
        return SINGLE_VERSIONED_FS_TYPE
    elif any(is_version_dir(dir) for dir in fs.walk.dirs()):
        return MULTIVERSIONED_FS_TYPE
    else:
        return UNKNOWN_FS_TYPE
