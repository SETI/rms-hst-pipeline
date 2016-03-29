import os

import pyfits

import FileArchives
import LabelMaker
import LID
import Product
import XmlMaker


class FitsProductFileXmlMakerD(XmlMaker.XmlMaker):
    """Version of FitsProductFileXmlMaker run on distillation"""
    def __init__(self, document, file_structure):
        super(FitsProductFileXmlMakerD, self).__init__(document)
        assert file_structure
        self.file_structure = file_structure

    def create_xml(self, parent):
        # Take parent == File_Area_Observational with File node
        # already built.  You only have to do Headers and Array_2Ds.
        assert parent

        for d in file_structure:
            header = self.xml.create_child(parent, 'Header')
            local_identifier, offset, object_length, \
                parsing_standard_id, description = \
                self.xml.create_children(header,
                                         ['local_identifier', 'offset',
                                          'object_length',
                                          'parsing_standard_id',
                                          'description'])
            offset.setAttribute('unit', 'byte')
            object_length.setAttribute('unit', 'byte')

            self.xml.set_text(local_identifier, d['local_identifier'])
            self.xml.set_text(offset, d['header_offset'])
            self.xml.set_text(object_length, d['header_size'])
            self.xml.set_text(parsing_standard_id, 'FITS 3.0')
            self.xml.set_text(description, 'Global FITS Header')

            if d['Axis_Array']:
                array_2d_image = self.xml.create_child(parent,
                                                       'Array_2D_Image')
                offset, axes, axis_index_order, element_array = \
                    self.xml.create_children(array_2d_image,
                                             ['offset', 'axes',
                                              'axis_index_order',
                                              'Element_Array'])
                offset.setAttribute('unit', 'byte')
                self.xml.set_text(offset, d['data_offset'])
                self.xml.set_text(axes, d['axes'])
                self.xml.set_text(axis_index_order,
                                  'Last Index Fastest')  # TODO Check this

                data_type = self.xml.create_child(element_array, 'data_type')
                self.xml.set_text(data_type, d['data_type'])

                if 'scaling_factor' in d:
                    scaling_factor = self.xml.create_child(element_array,
                                                           'scaling_factor')
                    self.xml.set_text(scaling_factor, d['scaling_factor'])

                if 'value_offset' in d:
                    value_offset = self.xml.create_child(element_array,
                                                         'value_offset')
                    self.xml.set_text(value_offset, d['value_offset'])

                for i, axis in enumerate(d['Axis_Array']):
                    array_2d_image

                    # TODO Do I really get multiple Axis_Arrays or
                    # should there just be one?
                    axis_array = self.xml.create_child(array_2d_image,
                                                       'Axis_Array')
                    axis_name, elements, sequence_number = \
                        self.xml.create_children(axis_array,
                                                 ['axis_name', 'elements',
                                                  'sequence_number'])
                    self.xml.set_text(axis_name, axis['axis_name'])
                    self.xml.set_text(elements, axis['elements'])
                    # TODO Double-check semantics of 'sequence_number'
                    self.xml.set_text(sequence_number, axis['sequence_number'])
