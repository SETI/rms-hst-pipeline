import abc

import pdart.pds4.File
import XmlMaker


class ProductFileXmlMaker(XmlMaker.XmlMaker):
    def __init__(self, document, archive_file):
        assert isinstance(archive_file, pdart.pds4.File.File), \
            'type(%s) = %s' % (archive_file, type(archive_file))
        self.archive_file = archive_file
        self.result = None
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