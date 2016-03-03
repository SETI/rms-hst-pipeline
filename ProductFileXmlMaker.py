import abc

import ArchiveFile
import XmlMaker
import XmlUtils


class ProductFileXmlMaker(XmlMaker.XmlMaker):
    def __init__(self, document, archive_file):
        assert isinstance(archive_file, ArchiveFile.ArchiveFile)
        self.archive_file = archive_file
        super(ProductFileXmlMaker, self).__init__(document)

    def create_xml(self, parent):
        assert parent

        # At XPath '/Product_Observational/File_Area_Observational'
        file_area_observational = self.create_child(parent,
                                                    'File_Area_Observational')

        # At XPath '/Product_Observational/File_Area_Observational/File'
        file = self.create_child(file_area_observational, 'File')
        file_name = self.create_child(file, 'file_name')
        self.set_text(file_name, self.archive_file.basename)

        # At XPath '/Product_Observational/File_Area_Observational'
        self.create_file_data_xml(file_area_observational)

    @abc.abstractmethod
    def create_file_data_xml(self, file_area_observational):
        """Create the XML nodes describing the product file's data."""
        pass
