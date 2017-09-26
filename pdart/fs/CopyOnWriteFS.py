import fs.errors
from fs.base import FS
from fs.copy import copy_file
from fs.mode import Mode
from fs.path import dirname
from fs.tempfs import TempFS
from typing import TYPE_CHECKING

from pdart.fs.FSWithDeletions import FSWithDeletions
from pdart.fs.DeletionSet import DeletionSet

if TYPE_CHECKING:
    from typing import Set


class Delta(object):
    """
    The changes between two filesystems.  First remove the filepaths given
    in subtractions, then add the files found in the additions filesystem.
    """

    def __init__(self, subtractions, additions):
        """Create a Delta from the subtractions and additions."""
        # type: (Set[unicode], FS) -> None
        self._subtractions = subtractions
        self._additions = additions

    def subtractions(self):
        """Returns a set of the removed filepaths."""
        # type: () -> Set[unicode]
        return self._subtractions

    def additions(self):
        """Returns a filesystem containing the additions."""
        # type: () -> FS
        return self._additions


class CopyOnWriteFS(FS):
    """
    Wraps a read-only filesystem and a read/write filesystem.  Any new
    data goes into the read-only system.
    """

    def __init__(self, readonly_fs, delta_fs=None):
        # type: (FS, FS) -> None
        FS.__init__(self)
        self._deletion_predicate = DeletionSet()
        self._readonly_fs = FSWithDeletions(readonly_fs,
                                            self._deletion_predicate)
        if not delta_fs:
            delta_fs = TempFS()
        self._delta_fs = delta_fs

    def delta(self):
        return Delta(self._deletion_predicate.as_set(), self._delta_fs)

    def normalize(self):
        """Remove all duplicated files."""
        # type: () -> None

        # walk all the files in the delta.  If they are the same as the
        # files in the R/O filesystem, delete them and make sure they
        # aren't marked as deleted.

        assert False, 'unimplemented'

    def getmeta(self, namespace='standard'):
        # TODO Not quite right to use just the delta's meta.  What
        # about the readonly's?
        return self._delta_fs.getmeta(namespace=namespace)

    def _ensure_path_is_writable(self, path):
        # type: (unicode) -> None
        assert path
        if self._readonly_fs.exists(path) and \
                not self._delta_fs.exists(path):
            p = dirname(path)
            self._delta_fs.makedirs(p)
            copy_file(self._readonly_fs, path, self._delta_fs, path)
            self._deletion_predicate.delete(path)

    def getinfo(self, path, namespaces=None):
        self.check()
        try:
            return self._delta_fs.getinfo(path, namespaces=namespaces)
        except fs.errors.ResourceNotFound:
            return self._readonly_fs.getinfo(path, namespaces=namespaces)

    def listdir(self, path):
        self.check()
        if not self.exists(path):
            raise fs.errors.ResourceNotFound(path)
        if self._delta_fs.exists(path):
            res = self._delta_fs.listdir(path)
        else:
            res = []
        if self._readonly_fs.exists(path):
            nondups = [p for p in self._readonly_fs.listdir(path)
                       if p not in res]
            res.extend(nondups)
        return res

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        if self.exists(path) and not recreate:
            raise fs.errors.DirectoryExists(path)
        else:
            # TODO Is this right?  The SubFS is physically on the
            # delta filesystem, but conceptually on self.
            return self._delta_fs.makedir(path,
                                          permissions=permissions,
                                          recreate=recreate)

    def openbin(self, path, mode=u'r', buffering=-1, **options):
        self.check()
        mode_obj = Mode(mode)
        if mode_obj.writing:
            self._ensure_path_is_writable(path)
            return self._delta_fs.openbin(path,
                                          mode=mode,
                                          buffering=buffering,
                                          **options)
        if mode_obj.reading:
            if self._delta_fs.exists(path):
                return self._delta_fs.openbin(path,
                                              mode=mode,
                                              buffering=buffering,
                                              **options)
            else:
                return self._readonly_fs.openbin(path,
                                                 mode=mode,
                                                 buffering=buffering,
                                                 **options)
        assert False, 'openbin (neither writing nor reading)'

    def remove(self, path):
        self.check()
        # TODO Am I handling all complications?
        if self._delta_fs.exists(path):
            self._delta_fs.remove(path)
        else:
            self._readonly_fs.remove(path)

    def removedir(self, path):
        self.check()
        # TODO Am I handling all complications?
        if self._delta_fs.exists(path):
            self._delta_fs.removedir(path)
        else:
            self._readonly_fs.removedir(path)

    def setinfo(self, path, info):
        self.check()
        self._ensure_path_is_writable(path)
        self._delta_fs.setinfo(path, info)
