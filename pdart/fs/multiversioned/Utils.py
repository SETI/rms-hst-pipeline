from typing import Iterator

from fs.base import FS
from fs.path import abspath, iteratepath
from fs.subfs import SubFS
import fs.walk

from pdart.pds4.LID import LID


def component_directories(fs: FS, dirpath: str) -> Iterator[str]:
    """
    Returns the relative dirpaths for directories that are contained
    within the directory for a PDS4 component.  Directories in PDS4
    subcomponents will have "$" in their paths (in the subcomponent
    directory name).
    """
    return (
        abspath(dirpath)
        for dirpath in SubFS(fs, dirpath).walk.dirs()
        if "$" not in dirpath
    )


def component_files(fs: FS, dirpath: str) -> Iterator[str]:
    """
    Returns the relative filepaths for files that are contained within
    the directory for a PDS4 component.  Files in PDS4 subcomponents
    will have "$" in their paths (in the subcomponent directory name).
    """
    return (
        abspath(filepath)
        for filepath in SubFS(fs, dirpath).walk.files()
        if "$" not in filepath
    )


def dirpath_to_lid(dirpath: str) -> LID:
    """
    Find the LID corresponding to a directory in a single-versioned
    filesystem.
    """
    parts = [part[:-1] for part in iteratepath(dirpath) if part.endswith("$")]
    return LID.create_from_parts(parts)


def lid_to_dirpath(lid: LID) -> str:
    """
    Find the directory corresponding to a LID in a single-versioned
    filesystem.
    """
    return fs.path.join(*[part + "$" for part in lid.parts()])
