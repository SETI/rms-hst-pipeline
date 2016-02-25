import ProductFileXmlMaker


class DummyProductFileXmlMaker(ProductFileXmlMaker.ProductFileXmlMaker):
    """
    A dummy class that builds the part of a PDS4 product label
    corresponding to a single file within the product.  It incorrectly
    declares that the data consists of a FITS header, then an array of
    a single byte.  This class is meant to be used as a placeholder.
    """

    def __init__(self, document, root, archiveFile):
        super(DummyProductFileXmlMaker, self).__init__(document,
                                                       root,
                                                       archiveFile)

    def createFileDataXml(self, file_area_observational):
        # The following block should be true for all FITS files.

        # At XPath '/Product_Observational/File_Area_Observational/Header'
        header = self.create_child(file_area_observational, 'Header')
        localIdentifier, offset, objectLength, \
            parsingStandardId, description = \
            self.create_children(header,
                                 ["local_identifier", "offset", "object_length",
                                  "parsing_standard_id", "description"])

        self.set_text(localIdentifier, 'header')
        offset.setAttribute('unit', 'byte')
        self.set_text(offset, '0')
        self.set_text(objectLength, '2880')  # TODO wrong; some multiple of this
        objectLength.setAttribute('unit', 'byte')
        self.set_text(parsingStandardId, 'FITS 3.0')
        self.set_text(description, 'Global FITS Header')

        # At XPath '/Product_Observational/File_Area_Observational/Array'
        array = self.create_child(file_area_observational, 'Array')
        offset, axes, axisIndexOrder, elementArray, axisArray = \
            self.create_children(array, ['offset', 'axes', 'axis_index_order',
                                         'Element_Array', 'Axis_Array'])

        offset.setAttribute('unit', 'byte')
        self.set_text(offset, '0')
        self.set_text(axes, '1')
        self.set_text(axisIndexOrder, 'Last Index Fastest')  # TODO Abstract?

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Element_Array'
        dataType = self.create_child(elementArray, 'data_type')
        self.set_text(dataType, 'UnsignedByte')

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Axis_Array'
        axisName, elements, sequenceNumber = \
            self.create_children(axisArray, ['axis_name', 'elements',
                                             'sequence_number'])

        self.set_text(axisName, 'Axis Joe')  # TODO Wrong
        self.set_text(elements, '1')  # TODO Wrong
        self.set_text(sequenceNumber, '1')  # TODO Wrong
