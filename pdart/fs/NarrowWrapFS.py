"""
A filesystem that delegates essential methods to a wrapped filesystem.
"""
import abc

from fs.base import FS


class NarrowWrapFS(FS):
    """
    A wrapper around another filesystem, intended to be used as a base
    class.  It's called narrow because it funnels all operations
    through the essential methods on the base FS, simplifying
    reasoning abut its correctness at the cost of possibly losing
    efficiency.  If that doesn't work for you, fs.wrapfs.WrapFS might
    be a better choice.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, wrap_fs):
        # type: (FS) -> None
        FS.__init__(self)
        assert wrap_fs, "filesystem to wrap required"
        self._wrap_fs = wrap_fs

    def delegate_fs(self):
        # type: () -> FS
        return self._wrap_fs

    def getmeta(self, namespace="standard"):
        return self.delegate_fs().getmeta(namespace=namespace)

    @abc.abstractmethod
    def getinfo(self, path, namespaces=None):
        pass

    @abc.abstractmethod
    def listdir(self, path):
        pass

    @abc.abstractmethod
    def makedir(self, path, permissions=None, recreate=False):
        pass

    @abc.abstractmethod
    def openbin(self, path, mode="r", buffering=-1, **options):
        pass

    @abc.abstractmethod
    def remove(self, path):
        pass

    @abc.abstractmethod
    def removedir(self, path):
        pass

    @abc.abstractmethod
    def setinfo(self, path, info):
        pass
