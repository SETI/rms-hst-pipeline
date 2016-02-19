import ArchiveFile
import Info


class ProductFileInfo(Info.Info):
    def __init__(self, file):
        assert isinstance(file, ArchiveFile.ArchiveFile)
        self.file = file
