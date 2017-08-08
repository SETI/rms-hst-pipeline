import os.path

from fs.errors import ResourceNotFound
from fs.mode import Mode
from pdart.fs.NarrowWrapFS import NarrowWrapFS
from pdart.fs.Utils import parent_dir

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fs.base import FS
    from pdart.fs.DeletionPredicate import DeletionPredicate


class FSWithDeletions(NarrowWrapFS):
    """
    A wrapper around a filesystem and a 'DeletionPredicate' that tells
    which files on the filesystem should be considered deleted.  This
    allows us to "delete" files from a read-only system.  Intended to
    be used as a base class.
    """
    def __init__(self, fs, del_pred):
        # type: (FS, DeletionPredicate) -> None
        NarrowWrapFS.__init__(self, fs)
        assert del_pred, 'deletion_predicate required'
        self.deletion_predicate = del_pred

    def getinfo(self, path, namespaces=None):
        self.check()
        if self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return self.delegate_fs().getinfo(path, namespaces=namespaces)

    def listdir(self, path):
        if self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return [child
                    for child in self.delegate_fs().listdir(path)
                    if not self.deletion_predicate.is_deleted(
                        os.path.join(path, child))]

    def makedir(self, path, permissions=None, recreate=False):
        parent = parent_dir(path)
        if self.deletion_predicate.is_deleted(parent):
            raise ResourceNotFound(parent)
        else:
            return self.delegate_fs().makedir(path,
                                              permissions=permissions,
                                              recreate=recreate)

    def openbin(self, path, mode=u'r', buffering=-1, **options):
        mode_obj = Mode(mode)
        if mode_obj.reading and self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)

        if mode_obj.writing:
            parent = parent_dir(path)
            if self.deletion_predicate.is_deleted(parent):
                raise ResourceNotFound(path)

        return self.delegate_fs().openbin(path, mode,
                                          buffering=buffering,
                                          **options)

    def remove(self, path):
        self.check()
        if self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return self.delegate_fs().remove(path)

    def removedir(self, path):
        self.check()
        if self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return self.delegate_fs().removedir(path)

    def setinfo(self, path, info):
        self.check()
        if self.deletion_predicate.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            self.delegate_fs().setinfo(path, info)
