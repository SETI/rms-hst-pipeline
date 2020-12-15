from typing import Iterator

from fs.base import FS
from fs.path import abspath
from fs.subfs import SubFS
import fs.walk


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
