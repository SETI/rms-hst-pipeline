import re

from fs.base import FS
from fs.path import parts

EMPTY_FS_TYPE: str = "empty"
SINGLE_VERSIONED_FS_TYPE: str = "single-versioned"
MULTIVERSIONED_FS_TYPE: str = "multiversioned"
UNKNOWN_FS_TYPE: str = "unknown"


def is_version_dir(dirpath: str) -> bool:
    PAT = r"^v\$[0-9]+\.[0-9]+$"
    last_part = parts(dirpath)[-1]
    res = re.search(PAT, last_part)
    return res is not None


def categorize_filesystem(fs: FS) -> str:
    top_level_listing = fs.listdir("/")
    if not top_level_listing:
        return EMPTY_FS_TYPE
    elif any(name[-1] == "$" for name in top_level_listing):
        return SINGLE_VERSIONED_FS_TYPE
    elif any(is_version_dir(dir) for dir in fs.walk.dirs()):
        return MULTIVERSIONED_FS_TYPE
    else:
        return UNKNOWN_FS_TYPE
