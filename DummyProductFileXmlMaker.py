import ProductFileXmlMaker


class DummyProductFileXmlMaker(ProductFileXmlMaker.ProductFileXmlMaker):
    """
    A dummy class that builds the part of a PDS4 product label
    corresponding to a single file within the product.  It incorrectly
    declares that the data consists of a FITS header, then an array of
    a single byte.  This class is meant to be used as a placeholder.
    """

    def __init__(self, document, root, archive_file):
        super(DummyProductFileXmlMaker, self).__init__(document,
                                                       root,
                                                       archive_file)

    def create_file_data_xml(self, file_area_observational):
        # The following block should be true for all FITS files.

        # At XPath '/Product_Observational/File_Area_Observational/Header'
        header = self.create_child(file_area_observational, 'Header')
        local_identifier, offset, object_length, \
            parsing_standard_id, description = \
            self.create_children(header,
                                 ["local_identifier", "offset",
                                  "object_length", "parsing_standard_id",
                                  "description"])

        self.set_text(local_identifier, 'header')
        offset.setAttribute('unit', 'byte')
        self.set_text(offset, '0')

        # TODO wrong; really, some multiple of this
        self.set_text(object_length, '2880')

        object_length.setAttribute('unit', 'byte')
        self.set_text(parsing_standard_id, 'FITS 3.0')
        self.set_text(description, 'Global FITS Header')

        # At XPath '/Product_Observational/File_Area_Observational/Array'
        array = self.create_child(file_area_observational, 'Array')
        offset, axes, axis_index_order, element_array, axis_array = \
            self.create_children(array, ['offset', 'axes', 'axis_index_order',
                                         'Element_Array', 'Axis_Array'])

        offset.setAttribute('unit', 'byte')
        self.set_text(offset, '0')
        self.set_text(axes, '1')
        self.set_text(axis_index_order, 'Last Index Fastest')  # TODO Abstract?

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Element_Array'
        data_type = self.create_child(element_array, 'data_type')
        self.set_text(data_type, 'UnsignedByte')

        # At XPath
        # '/Product_Observational/File_Area_Observational/Array/Axis_Array'
        axis_name, elements, sequence_number = \
            self.create_children(axis_array, ['axis_name', 'elements',
                                              'sequence_number'])

        self.set_text(axis_name, 'Axis Joe')  # TODO Wrong
        self.set_text(elements, '1')  # TODO Wrong
        self.set_text(sequence_number, '1')  # TODO Wrong

# was_converted
