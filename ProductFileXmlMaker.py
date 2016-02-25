import abc

import ArchiveFile
import XmlUtils


class ProductFileXmlMaker(XmlUtils.XmlUtils):
    """
    An abstract class to build the part of a PDS4 product label
    corresponding to a single file within the product.  This class
    provides functionality to the ProductLabelMaker.
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

        super(ProductFileXmlMaker, self).__init__(document)
        self.create_default_xml()

    def create_default_xml(self):
        """Create the XML nodes for the product file."""

        # At XPath '/Product_Observational/File_Area_Observational'
        file_area_observational = self.create_child(self.root,
                                                    'File_Area_Observational')

        # At XPath '/Product_Observational/File_Area_Observational/File'
        file = self.create_child(file_area_observational, 'File')
        fileName = self.create_child(file, 'file_name')
        self.set_text(fileName, self.archiveFile.basename)

        # At XPath '/Product_Observational/File_Area_Observational'
        self.createFileDataXml(file_area_observational)

    @abc.abstractmethod
    def createFileDataXml(self, file_area_observational):
        """Create the XML nodes describing the product file's data."""
        pass
