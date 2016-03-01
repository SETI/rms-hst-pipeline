import os

import pyfits

import FileArchives
import LabelMaker
import LID
import Product
import ProductFileXmlMaker
import ProductLabelMaker
import XmlUtils


BITPIX_TABLE = {
    # TODO Verify these
    8: 'UnsignedByte',
    16: 'SignedMSB2',
    32: 'SignedMSB4',
    64: 'SignedMSB8',
    -32: 'IEEE754MSBSingle',
    -62: 'IEEE754MSBDouble'
    }


AXIS_NAME_TABLE = {
    1: 'Line',
    2: 'Sample'
    }


def data_type_from_bitpix(n):
    return BITPIX_TABLE[n]


class FitsPass(object):
    def __init__(self):
        pass

    def do_fits(self, fits, before):
        pass

    def do_hdu(self, hdu, n, before):
        pass

    def do_header(self, n):
        pass

    def do_data(self, n):
        pass


def runFitsPass(filepath, fits_pass):
    fits = pyfits.open(filepath)
    try:
        fits_pass.do_fits(fits, True)
        for n, hdu in enumerate(fits):
            fits_pass.do_hdu(hdu, n, True)
            fits_pass.do_header(n)
            if hdu.fileinfo()['datSpan']:
                fits_pass.do_data(n)
            fits_pass.do_hdu(hdu, n, False)
        fits_pass.do_fits(fits, False)
    finally:
        fits.close()


class XmlFitsPass(FitsPass):
    def __init__(self, xml, file_area_observational):
        super(XmlFitsPass, self).__init__()
        self.xml = xml
        self.file_area_observational = file_area_observational
        self.targname = None

    def do_hdu(self, hdu, n, before):
        if before:
            self.hdu = hdu
            try:
                self.targname = hdu.header['targname']
            except KeyError:
                pass
        else:
            self.hdu = None

    def do_header(self, n):
        info = self.hdu.fileinfo()
        header = self.xml.create_child(self.file_area_observational,
                                       'Header')

        local_identifier, offset, object_length, \
            parsing_standard_id, description = \
            self.xml.create_children(header,
                                     ['local_identifier', 'offset',
                                      'object_length', 'parsing_standard_id',
                                      'description'])

        self.xml.set_text(local_identifier, 'hdu_%d' % n)
        offset.setAttribute('unit', 'byte')
        self.xml.set_text(offset, str(info['hdrLoc']))

        hdr_size = info['datLoc'] - info['hdrLoc']
        assert hdr_size % 2880 == 0
        self.xml.set_text(object_length, str(hdr_size))

        object_length.setAttribute('unit', 'byte')
        self.xml.set_text(parsing_standard_id, 'FITS 3.0')
        self.xml.set_text(description, 'Global FITS Header')

    def do_data(self, n):
        info = self.hdu.fileinfo()
        naxis = self.hdu.header['NAXIS']

        # TODO I need to be smarter here.  For now, assuming it's a
        # image.

        array_2d_image = self.xml.create_child(self.file_area_observational,
                                               'Array_2D_Image')
        offset, axes, axis_index_order, element_array = \
            self.xml.create_children(array_2d_image, ['offset', 'axes',
                                                      'axis_index_order',
                                                      'Element_Array'])
        offset.setAttribute('unit', 'byte')
        self.xml.set_text(offset, str(info['datLoc']))
        self.xml.set_text(axes, str(naxis))
        self.xml.set_text(axis_index_order, 'Last Index Fastest')  # TODO Check

        data_type = self.xml.create_child(element_array, 'data_type')
        self.xml.set_text(data_type,
                          data_type_from_bitpix(self.hdu.header['BITPIX']))

        # TODO unit would go here

        try:
            bscale = self.hdu.header['BSCALE']
            scaling_factor = self.xml.create_child(element_array,
                                                   'scaling_factor')
            self.xml.set_text(scaling_factor, str(bscale))
        except KeyError:
            pass

        try:
            bzero = self.hdu.header['BZERO']
            value_offset = self.xml.create_child(element_array,
                                                 'value_offset')
            self.xml.set_text(value_offset, str(bzero))
        except KeyError:
            pass

        for i in range(1, naxis + 1):
            axis_array = self.xml.create_child(array_2d_image, 'Axis_Array')
            axis_name, elements, sequence_number = \
                self.xml.create_children(axis_array, ['axis_name', 'elements',
                                                      'sequence_number'])
            self.xml.set_text(axis_name, AXIS_NAME_TABLE[i])
            self.xml.set_text(elements, str(self.hdu.header['NAXIS%s' % i]))
            # TODO Double-check semantics of 'sequence_number'
            self.xml.set_text(sequence_number, str(i))

        # TODO Add text to all of these


class FitsProductFileXmlMaker(ProductFileXmlMaker.ProductFileXmlMaker):
    """
    TBD
    """

    def __init__(self, document, root, archive_file):
        self.targname = None
        super(FitsProductFileXmlMaker, self).__init__(document,
                                                      root,
                                                      archive_file)

    def create_file_data_xml(self, file_area_observational):
        xfp = XmlFitsPass(self, file_area_observational)
        runFitsPass(self.archive_file.full_filepath(), xfp)
        self.targname = xfp.targname


def _createLabel():
    def label_checks(filepath):
        return LabelMaker.xml_schema_check(filepath) and \
            LabelMaker.schematron_check(filepath)

    # product_lid = LID.LID('urn:nasa:pds:hst_09059:data_acs_raw:visit_01')
    product_lid = LID.LID('urn:nasa:pds:hst_10534:data_wfpc2_c0m:visit_01')
    archive = FileArchives.get_any_archive()
    product = Product.Product(archive, product_lid)
    product_lm = ProductLabelMaker.ProductLabelMaker(product)
    product_filepath = '/tmp/product.xml'
    product_lm.create_default_xml_file(product_filepath)
    assert label_checks(product_filepath)
    os.system('open /tmp/product.xml')


if __name__ == '__main__':
    _createLabel()
