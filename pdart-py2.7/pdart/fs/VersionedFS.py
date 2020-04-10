"""
Utility functions for a versioned filesystem.  Currently, only a
function to get directory contents in a multiversion filesystem,
sorted into "real" files and directories and "bookkeeping" ones.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple
    from fs.base import FS
    from fs.info import Info

    _INFOS = List[Info]

ROOT = u"/"  # type: unicode

SUBDIR_VERSIONS_FILENAME = u"subdir$versions.txt"  # type: unicode

_VERSION_DIR_PREFIX = u"v$"  # type: unicode


def scan_vfs_dir(fs, dir, namespaces=None):
    # type: (FS, unicode, Tuple) -> Tuple[_INFOS, _INFOS, _INFOS, _INFOS]
    """
    Returns a 4-tuple of (ordinary-file infos, ordinary-directory
    infos, subdir-versions-file infos, version-directory infos).  This
    lets us separate "real" files and dirs from the "bookkeeping"
    ones.
    """
    infos = list(fs.scandir(dir, namespaces=namespaces))
    file_infos = [info for info in infos if info.is_file]
    dir_infos = [info for info in infos if info.is_dir]

    ordinary_file_infos = [
        info for info in file_infos if info.name != SUBDIR_VERSIONS_FILENAME
    ]
    subdir_versions_file_infos = [
        info for info in file_infos if info.name == SUBDIR_VERSIONS_FILENAME
    ]
    ordinary_dir_infos = [
        info for info in dir_infos if info.name[0:2] != _VERSION_DIR_PREFIX
    ]
    version_dir_infos = [
        info for info in dir_infos if info.name[0:2] == _VERSION_DIR_PREFIX
    ]
    return (
        ordinary_file_infos,
        ordinary_dir_infos,
        subdir_versions_file_infos,
        version_dir_infos,
    )
