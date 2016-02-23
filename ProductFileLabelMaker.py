import abc

import ArchiveFile
import XmlUtils


class ProductFileLabelMaker(XmlUtils.XmlUtils):
    def __init__(self, document, root, archiveFile):
        assert document
        assert root
        self.root = root
        assert isinstance(archiveFile, ArchiveFile.ArchiveFile)
        self.archiveFile = archiveFile

        XmlUtils.XmlUtils.__init__(self, document)
        self.createDefaultXml()

    def createDefaultXml(self):
        fileAreaObservational = self.createChild(self.root,
                                                 'File_Area_Observational')

        file = self.createChild(fileAreaObservational, 'File')
        fileName = self.createChild(file, 'file_name')
        self.setText(fileName, self.archiveFile.basename)
        self.createFileDataXml(fileAreaObservational)

    @abc.abstractmethod
    def createFileDataXml(self, fileAreaObservational):
        pass
