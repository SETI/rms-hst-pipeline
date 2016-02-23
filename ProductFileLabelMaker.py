import abc

import ArchiveFile
import XmlUtils


class ProductFileLabelMaker(XmlUtils.XmlUtils):
    """
    An abstract class to build the part of a PDS4 product label
    corresponding to a single file within the product.  Despite the
    name, this is not a subclass of LabelMaker; rather it provides
    functionality to the ProductLabelMaker.
    """

    def __init__(self, document, root, archiveFile):
        """
        Create the XML corresponding to a single file within the
        product, given the XML document, the root node to which the
        new XML will be added, and the file in the product for which
        XML is to be created.
        """
        assert document
        assert root
        self.root = root
        assert isinstance(archiveFile, ArchiveFile.ArchiveFile)
        self.archiveFile = archiveFile

        XmlUtils.XmlUtils.__init__(self, document)
        self.createDefaultXml()

    def createDefaultXml(self):
        """Create the XML nodes for the product file."""
        fileAreaObservational = self.createChild(self.root,
                                                 'File_Area_Observational')

        file = self.createChild(fileAreaObservational, 'File')
        fileName = self.createChild(file, 'file_name')
        self.setText(fileName, self.archiveFile.basename)
        self.createFileDataXml(fileAreaObservational)

    @abc.abstractmethod
    def createFileDataXml(self, fileAreaObservational):
        """Create the XML nodes describing the product file's data."""
        pass
