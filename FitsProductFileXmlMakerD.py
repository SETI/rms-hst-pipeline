import os

import pyfits

import FileArchives
import LabelMaker
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

        for d in self.file_structure:
            header = self.create_child(parent, 'Header')
            local_identifier, offset, object_length, \
                parsing_standard_id, description = \
                self.create_children(header,
                                     ['local_identifier', 'offset',
                                      'object_length',
                                      'parsing_standard_id',
                                      'description'])
            offset.setAttribute('unit', 'byte')
            object_length.setAttribute('unit', 'byte')

            self.set_text(local_identifier, d['local_identifier'])
            self.set_text(offset, str(d['header_offset']))
            self.set_text(object_length, str(d['header_size']))
            self.set_text(parsing_standard_id, 'FITS 3.0')
            self.set_text(description, 'Global FITS Header')

            if d['Axis_Array']:
                array_2d_image = self.create_child(parent,
                                                   'Array_2D_Image')
                offset, axes, axis_index_order, element_array = \
                    self.create_children(array_2d_image,
                                         ['offset', 'axes',
                                          'axis_index_order',
                                          'Element_Array'])
                offset.setAttribute('unit', 'byte')
                self.set_text(offset, str(d['data_offset']))
                self.set_text(axes, str(d['axes']))
                self.set_text(axis_index_order,
                              'Last Index Fastest')  # TODO Check this

                data_type = self.create_child(element_array, 'data_type')
                self.set_text(data_type, d['data_type'])

                if 'scaling_factor' in d:
                    scaling_factor = self.create_child(element_array,
                                                       'scaling_factor')
                    self.set_text(scaling_factor, str(d['scaling_factor']))

                if 'value_offset' in d:
                    value_offset = self.create_child(element_array,
                                                     'value_offset')
                    self.set_text(value_offset, str(d['value_offset']))

                for i, axis in enumerate(d['Axis_Array']):
                    array_2d_image

                    # TODO Do I really get multiple Axis_Arrays or
                    # should there just be one?
                    axis_array = self.create_child(array_2d_image,
                                                   'Axis_Array')
                    axis_name, elements, sequence_number = \
                        self.create_children(axis_array,
                                             ['axis_name', 'elements',
                                              'sequence_number'])
                    self.set_text(axis_name, axis['axis_name'])
                    self.set_text(elements, str(axis['elements']))
                    # TODO Double-check semantics of 'sequence_number'
                    self.set_text(sequence_number,
                                  str(axis['sequence_number']))
