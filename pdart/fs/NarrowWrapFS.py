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
    def __init__(self, wrap_fs):
        FS.__init__(self)
        self._wrap_fs = wrap_fs

    def delegate_fs(self):
        return self._wrap_fs

    def getmeta(self, namespace='standard'):
        return self.delegate_fs().getmeta(namespace=namespace)

    def getinfo(self, path, namespaces=None):
        self.check()
        return self.delegate_fs().getinfo(path, namespaces=namespaces)

    def listdir(self, path):
        self.check()
        return self.delegate_fs().listdir(path)

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        return self.delegate_fs().makedir(path, permissions=permissions,
                                          recreate=recreate)

    def openbin(self, path, mode="r", buffering=-1, **options):
        self.check()
        return self.delegate_fs().openbin(path, mode=mode,
                                          buffering=buffering, **options)

    def remove(self, path):
        self.check()
        self.delegate_fs().remove(path)

    def removedir(self, path):
        self.check()
        self.delegate_fs().removedir(path)

    def setinfo(self, path, info):
        self.check()
        self.delegate_fs().setinfo(path, info)
