import ArchiveFile
import Info


class ProductFileInfo(Info.Info):
    def __init__(self, fileAreaNode, file):
        assert isinstance(file, ArchiveFile.ArchiveFile)
        self.file = file

    def fileName(self):
        return self.file.basename
