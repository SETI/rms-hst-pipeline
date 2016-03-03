import XmlMaker


class ProductFilesXmlMaker(XmlMaker.XmlMaker):
    def __init__(self, file_xml_maker_factory, document, product):
        assert product
        self.product = product
        self.file_xml_maker_factory = file_xml_maker_factory
        self.result = None
        super(ProductFilesXmlMaker, self).__init__(document)

    def create_xml(self, parent):
        self.result = [self.process_file(parent, archive_file)
                       for archive_file in self.product.files()]

    def process_file(self, parent, archive_file):
        file_xml_maker = self.file_xml_maker_factory(self.document,
                                                     archive_file)
        file_xml_maker.create_xml(parent)
        return file_xml_maker.targname
