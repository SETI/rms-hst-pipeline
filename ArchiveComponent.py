import abc
import os
import os.path

class ArchiveComponent:
    __metaclass__ = abc.ABCMeta

    def __init__(self, arch, lid):
        assert arch
        self.archive = arch
        assert lid
        self.lid = lid

    def __eq__(self, other):
        return self.archive == other.archive and self.lid == other.lid

    def __str__(self):
        return str(self.lid)

    @abc.abstractmethod
    def directoryFilepath(self):
        pass

    def files(self):
        dir = self.directoryFilepath()
        for f in os.listdir(dir):
            if f[0] != '.':
                file = os.path.join(dir, f)
                if (os.path.isfile(file)):
                    yield file

