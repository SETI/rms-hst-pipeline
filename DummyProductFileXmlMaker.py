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

    def createFileDataXml(self, fileAreaObservational):
        # The following block should be true for all FITS files.

        # At XPath '/Product_Observational/File_Area_Observational/Header'
        header = self.createChild(fileAreaObservational, 'Header')
        localIdentifier, offset, objectLength, \
            parsingStandardId, description = \
            self.createChildren(header,
                                ["local_identifier", "offset", "object_length",
                                 "parsing_standard_id", "description"])

        self.setText(localIdentifier, 'header')
        offset.setAttribute('unit', 'byte')
        self.setText(offset, '0')
        self.setText(objectLength, '2880')  # TODO wrong; some multiple of this
        objectLength.setAttribute('unit', 'byte')
        self.setText(parsingStandardId, 'FITS 3.0')
        self.setText(description, 'Global FITS Header')

        # At XPath '/Product_Observational/File_Area_Observational/Array'
        array = self.createChild(fileAreaObservational, 'Array')
        offset, axes, axisIndexOrder, elementArray, axisArray = \
            self.createChildren(array, ['offset', 'axes', 'axis_index_order',
                                        'Element_Array', 'Axis_Array'])

        offset.setAttribute('unit', 'byte')
        self.setText(offset, '0')
        self.setText(axes, '1')
        self.setText(axisIndexOrder, 'Last Index Fastest')  # TODO Abstract?

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Element_Array'
        dataType = self.createChild(elementArray, 'data_type')
        self.setText(dataType, 'UnsignedByte')

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Axis_Array'
        axisName, elements, sequenceNumber = \
            self.createChildren(axisArray, ['axis_name', 'elements',
                                            'sequence_number'])

        self.setText(axisName, 'Axis Joe')  # TODO Wrong
        self.setText(elements, '1')  # TODO Wrong
        self.setText(sequenceNumber, '1')  # TODO Wrong
