"""Reformulation of pyfilesystems's functionality to be based on
FSEntrys instead of the filesystem as a whole.  The FSEntryFS class
wraps the FSEntry for the root of the filesystem; the rest of the
functionality automatically derives from that."""
from abc import *
from fs.base import FS
from fs.info import Info
from fs.permissions import Permissions
import io
import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, AnyStr, Dict, IO, List


class FSEntry(object):
    """Represents either a file or a directory.  Directories know their
    content."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def is_file(self):
        # type: () -> bool
        pass

    def is_directory(self):
        # type: () -> bool
        return not self.is_file()

    @abstractmethod
    def directory_contents(self):
        # type: () -> Dict[unicode, FSEntry]
        pass

    @abstractmethod
    def get_information(self):
        # type: () -> Info
        pass

    # Writable filesystem methods

    @abstractmethod
    def open(self, child_filename, mode=u'r', buffering=-1, **options):
        # type: (unicode, unicode, int, **Any) -> IO[Any]
        pass

    @abstractmethod
    def set_information(self, info):
        # type: (Info) -> None
        pass

    @abstractmethod
    def make_directory(self, child_filename, permissions=None, recreate=False):
        # type: (unicode, Permissions, bool) -> None
        pass

    @abstractmethod
    def remove(self, child_filename):
        # type: (unicode) -> None
        pass

    @abstractmethod
    def remove_directory(self, child_dirname):
        # type: (unicode) -> None
        pass


class FSEntryFS(FS):
    def __init__(self, root_fs_entry):
        # type: (FSEntry) -> None
        self.root_fs_entry = root_fs_entry

    def _path_to_fs_entry(self, path):
        # type: (unicode) -> FSEntry
        if path == '/':
            return self.root_fs_entry
        else:
            dirname, basename = os.path.split(path)
            parent_fs_entry = self._path_to_fs_entry(dirname)
            assert parent_fs_entry.is_directory()
            if basename:
                return parent_fs_entry.directory_contents()[basename]
            else:
                return parent_fs_entry

    def listdir(self, path):
        # type: (unicode) -> List[unicode]
        fs_entry = self._path_to_fs_entry(path)
        if fs_entry.is_directory():
            return fs_entry.directory_contents().keys()
        else:
            assert False

    def getinfo(self, path, namespaces=None):
        # type: (unicode, List[AnyStr]) -> Info
        return self._path_to_fs_entry(path).get_information()

    # Writable filesystem methods

    def openbin(self, path, mode=u'r', buffering=-1, **options):
        # type: (unicode, unicode, int, **Any) -> IO[Any]
        dirname, basename = os.path.split(path)
        parent_fs_entry = self._path_to_fs_entry(dirname)
        assert parent_fs_entry.is_directory()
        parent_fs_entry.open(basename)

    def makedir(self, path, permissions=None, recreate=False):
        # type: (unicode, Permissions, bool) -> None
        dirname, basename = os.path.split(path)
        parent_fs_entry = self._path_to_fs_entry(dirname)
        assert parent_fs_entry.is_directory()
        parent_fs_entry.make_directory(basename, permissions, recreate)

    def remove(self, path):
        # type: (unicode) -> None
        dirname, basename = os.path.split(path)
        parent_fs_entry = self._path_to_fs_entry(dirname)
        assert parent_fs_entry.is_directory()
        parent_fs_entry.remove(basename)

    def removedir(self, path):
        # type: (unicode) -> None
        dirname, basename = os.path.split(path)
        parent_fs_entry = self._path_to_fs_entry(dirname)
        assert parent_fs_entry.is_directory()
        parent_fs_entry.remove_directory(basename)

    def setinfo(self, path, info):
        # type: (unicode, Info) -> None
        self._path_to_fs_entry(path).set_information(info)
