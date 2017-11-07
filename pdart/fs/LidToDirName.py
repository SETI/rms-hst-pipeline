from fs.path import join

from pdart.pds4.LID import LID
import pdart.fs.DirUtils


def lid_to_dir_name(lid):
    # type: (LID) -> unicode
    return pdart.fs.DirUtils.lid_to_dir(lid)
