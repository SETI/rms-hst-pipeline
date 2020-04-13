"""
A copy-on-write filesystem.  It allows us to separate the original
data from the new data.
"""
import fs.errors
from fs.base import FS
from fs.copy import copy_file
from fs.mode import Mode
from fs.path import dirname, recursepath
from fs.tempfs import TempFS
from typing import TYPE_CHECKING

from pdart.fs.DeletionSet import DeletionSet
from pdart.fs.ISingleVersionBundleFS import ISingleVersionBundleFS
from pdart.fs.ReadOnlyFSWithDeletions import ReadOnlyFSWithDeletions
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import Set
    from pdart.fs.OldVersionView import OldVersionView
    from pdart.pds4.LID import LID
    from pdart.pds4.LIDVID import LIDVID


class FSDelta(object):
    """
    The changes between two filesystems.  First remove the filepaths given
    in deletions, then add the files found in the additions filesystem.
    """

    def __init__(self, deletions, additions):
        # type: (Set[unicode], FS) -> None
        """Create a Delta from the subtractions and additions."""
        self._deletions = deletions
        self._additions = additions

    def deletions(self):
        # type: () -> Set[unicode]
        """Returns a set of the removed filepaths."""
        return self._deletions

    def additions(self):
        # type: () -> FS
        """Returns a filesystem containing the additions."""
        return self._additions

    def directories(self):
        # type: () -> Set[unicode]
        """Returns a set of the directories with changes."""
        deletion_dirs = [
            d for f in list(self._deletions) for d in recursepath(dirname(f))
        ]
        addition_dirs = list(self._additions.walk.dirs())
        result = set(deletion_dirs + addition_dirs)
        if result:
            result.add(u"/")
        return result


class CopyOnWriteFS(FS):
    """
    Wraps a read-only filesystem and a read/write filesystem.  Any new
    data goes into the read/write system.
    """

    def __init__(self, base_fs, delta_fs=None):
        # type: (FS, FS) -> None
        FS.__init__(self)
        self._deletion_set = DeletionSet()
        self._readonly_fs = ReadOnlyFSWithDeletions(base_fs, self._deletion_set)
        if not delta_fs:
            delta_fs = TempFS()
        self._delta_fs = delta_fs

    def delta(self):
        # type: () -> FSDelta
        """Extract an FSDelta from the filesystem."""
        return FSDelta(self._deletion_set.as_set(), self._delta_fs)

    def normalize(self):
        # type: () -> None
        """
        Normalize the filesystem by removing redundant changes: remove
        changes that result only in duplicating a file, and remove
        empty directories from the read/write filesystem.
        """
        self._remove_duplicates()
        self._remove_empty_dirs()

    def _remove_empty_dirs(self):
        # type: () -> None
        """
        Remove empty directories from the read/write filesystem.
        """
        delta_fs = self._delta_fs
        for dir_path in delta_fs.walk.dirs(search="depth"):
            if not delta_fs.listdir(dir_path):
                delta_fs.removedir(dir_path)

    def _remove_duplicates(self):
        # type: () -> None
        """
        Remove all unnecessarily duplicated files: files whose
        contents are the same in both the read-only and the read/write
        filesystems.
        """

        # walk all the files in the delta.  If they are the same as the
        # files in the R/O filesystem, delete them and make sure they
        # aren't marked as deleted.
        def files_equal(filepath):
            readonly_del = self._readonly_fs.delegate_fs()
            if readonly_del.exists(filepath):
                bytes_eq = self._delta_fs.getbytes(filepath) == readonly_del.getbytes(
                    filepath
                )
                return bytes_eq
            else:
                return False

        redundant_files = [
            filepath
            for filepath in self._delta_fs.walk.files()
            if files_equal(filepath)
        ]
        for filepath in redundant_files:
            self._deletion_set.undelete(filepath)
            self._delta_fs.remove(filepath)

    def getmeta(self, namespace="standard"):
        """
        Gets the metadata from the filesystem.
        """
        # TODO Not quite right to use just the delta's meta.  What
        # about the readonly's?
        return self._delta_fs.getmeta(namespace=namespace)

    def _ensure_path_is_writable(self, path):
        # type: (unicode) -> None
        assert path
        parent = dirname(path)

        if self._readonly_fs.exists(parent) and not self._delta_fs.exists(parent):
            self._delta_fs.makedirs(parent, recreate=True)

        if self._readonly_fs.exists(path) and not self._delta_fs.exists(path):
            copy_file(self._readonly_fs, path, self._delta_fs, path)
            self._deletion_set.delete(path)
            self._deletion_set.delete(path)

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
            nondups = [p for p in self._readonly_fs.listdir(path) if p not in res]
            res.extend(nondups)
        return res

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        if self.exists(path) and not recreate:
            raise fs.errors.DirectoryExists(path)
        else:
            # TODO Is this right?  The SubFS is physically on the
            # delta filesystem, but conceptually on self.
            parent_dir = dirname(path)
            if self.exists(parent_dir) and not self._delta_fs.exists(parent_dir):
                self._delta_fs.makedirs(parent_dir)
            return self._delta_fs.makedir(
                path, permissions=permissions, recreate=recreate
            )

    def openbin(self, path, mode=u"r", buffering=-1, **options):
        self.check()
        mode_obj = Mode(mode)
        if mode_obj.writing:
            self._ensure_path_is_writable(path)
            return self._delta_fs.openbin(
                path, mode=mode, buffering=buffering, **options
            )
        if mode_obj.reading:
            if self._delta_fs.exists(path):
                return self._delta_fs.openbin(
                    path, mode=mode, buffering=buffering, **options
                )
            else:
                return self._readonly_fs.openbin(
                    path, mode=mode, buffering=buffering, **options
                )
        assert False, "openbin (neither writing nor reading)"

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

    def getsyspath(self, path):
        self.check()
        if self._delta_fs.exists(path):
            return self._delta_fs.getsyspath(path)
        elif self._readonly_fs.exists(path):
            return self._readonly_fs.getsyspath(path)
        else:
            return self._delta_fs.getsyspath(path)


class CopyOnWriteVersionView(CopyOnWriteFS, ISingleVersionBundleFS):
    """
    A CopyOnWriteFS that wraps a OldVersionView, so is itself a OldVersionView too.
    """

    def __init__(self, version_view, delta_fs=None):
        # type: (OldVersionView, FS) -> None
        CopyOnWriteFS.__init__(self, version_view, delta_fs)
        self._version_view = version_view

    def bundle_lidvid(self):
        # type: () -> LIDVID
        """Return the LIDVID for the bundle the filesystem holds."""
        return self._version_view.bundle_lidvid()

    def lid_to_vid(self, lid):
        # type: (LID) -> VID
        """
        Find the VID for the bundle, collection, or product that has
        the given LID.
        """
        try:
            return self._version_view.lid_to_vid(lid)
        except KeyError:
            return VID("1.0")