"""
A view on a filesystem to make it read-only.
"""
from fs.errors import ResourceReadOnly

from pdart.fs.NarrowWrapFS import *


class ReadOnlyView(NarrowWrapFS):
    def __init__(self, wrap_fs):
        # type: (FS) -> None
        NarrowWrapFS.__init__(self, wrap_fs)

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        raise ResourceReadOnly(path)

    def remove(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def removedir(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def setinfo(self, path, info):
        self.check()
        raise ResourceReadOnly(path)
