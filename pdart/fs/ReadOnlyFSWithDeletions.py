"""
A (mostly) read-only filesystem from which you can "delete" files.
"""
from fs.errors import (
    DirectoryExists,
    DirectoryNotEmpty,
    FileExpected,
    ResourceNotFound,
    ResourceReadOnly,
)
from fs.mode import Mode
from fs.path import dirname, join
from typing import TYPE_CHECKING

from pdart.fs.DeletionSet import DeletionSet
from pdart.fs.NarrowWrapFS import NarrowWrapFS

if TYPE_CHECKING:
    from fs.base import FS


class ReadOnlyFSWithDeletions(NarrowWrapFS):
    """
    A wrapper around a filesystem and a 'DeletionSet' that tells
    which files on the filesystem should be considered deleted.  This
    allows us to "delete" files from a read-only system.  Intended to
    be used as a base class.
    """

    def __init__(self, base_fs, deletion_set=DeletionSet()):
        # type: (FS, DeletionSet) -> None
        NarrowWrapFS.__init__(self, base_fs)
        self._deletion_set = deletion_set

    def __str__(self):
        return "ReadOnlyFSWithDeletions(%s, %s)" % (self._wrap_fs, self._deletion_set)

    def getinfo(self, path, namespaces=None):
        self.check()
        if self._deletion_set.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return self.delegate_fs().getinfo(path, namespaces=namespaces)

    def listdir(self, path):
        if self._deletion_set.is_deleted(path):
            raise ResourceNotFound(path)
        else:
            return [
                child
                for child in self.delegate_fs().listdir(path)
                if not self._deletion_set.is_deleted(join(path, child))
            ]

    def makedir(self, path, permissions=None, recreate=False):
        parent = dirname(path)
        if self._deletion_set.is_deleted(parent) or not self.delegate_fs().exists(
            parent
        ):
            raise ResourceNotFound(parent)
        elif self.isdir(path):
            raise DirectoryExists(path)
        else:
            raise ResourceReadOnly(parent)

    def openbin(self, path, mode=u"r", buffering=-1, **options):
        mode_obj = Mode(mode)
        if mode_obj.reading and self._deletion_set.is_deleted(path):
            raise ResourceNotFound(path)

        if mode_obj.writing:
            parent = dirname(path)
            if self.isdir(parent):
                raise ResourceReadOnly(parent)
            else:
                raise ResourceNotFound(parent)

        return self.delegate_fs().openbin(path, mode, buffering=buffering, **options)

    def remove(self, path):
        self.check()
        if self._deletion_set.is_deleted(path) or not self.delegate_fs().exists(path):
            raise ResourceNotFound(path)
        elif self.isdir(path):
            raise FileExpected(path)
        else:
            self._deletion_set.delete(path)

    def removedir(self, path):
        self.check()
        if self._deletion_set.is_deleted(path) or not self.delegate_fs().exists(path):
            raise ResourceNotFound(path)
        elif self.isfile(path):
            raise FileExpected(path)
        elif self.listdir(path):
            raise DirectoryNotEmpty(path)
        else:
            self._deletion_set.delete(path)

    def setinfo(self, path, info):
        self.check()
        if self._deletion_set.is_deleted(path) or not self.delegate_fs().exists(path):
            raise ResourceNotFound(path)
        else:
            raise ResourceReadOnly(path)

    def getsyspath(self, path):
        self.check()
        return self.delegate_fs().getsyspath(path)
