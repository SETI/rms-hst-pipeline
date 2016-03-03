import sys
import tempfile

import ArchiveFile
import DummyProductFileXmlMaker
import FitsProductFileXmlMaker
import FileArchives
import InstrumentXmlMaker
import LabelMaker
import LID
import Product
import ProductInfo
import Targets
import XmlSchema


class ProductLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, product):
        super(ProductLabelMaker,
              self).__init__(product, ProductInfo.ProductInfo(product))

    def default_xml_name(self):
        assert False, 'ProductLabelMaker.default_xml_name unimplemented'

    def create_default_xml(self):
        product = self.component

        # At XPath '/'
        self.add_processing_instruction('xml-model',
                                        self.info.xml_model_pds_attributes())

        # At XPath '/Product_Observational'
        root = self.create_child(self.document, 'Product_Observational')

        PDS = self.info.pds_namespace_url()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)
        root.setAttribute('xmlns:xsi', self.info.xsi_namespace_url())
        root.setAttribute('xsi:schemaLocation', self.info.pds4_schema_url())

        identification_area, observation_area = \
            self.create_children(root, ['Identification_Area',
                                        'Observation_Area'])

        # At XPath '/Product_Observational/Identification_Area'
        logical_identifier, version_id, title, \
            information_model_version, product_class = \
            self.create_children(identification_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class'])

        self.set_text(logical_identifier, str(product.lid))
        self.set_text(version_id, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(product_class, 'Product_Observational')

        # At XPath '/Product_Observational/Observation_Area'
        time_coordinates, investigation_area = \
            self.create_children(observation_area, ['Time_Coordinates',
                                                    'Investigation_Area'])

        # At XPath '/Product_Observational/Observation_Area/Time_Coordinates'
        start_date_time, stop_date_time = \
            self.create_children(time_coordinates, ['start_date_time',
                                                    'stop_date_time'])
        self.set_text(start_date_time, self.info.start_date_time())
        self.set_text(stop_date_time, self.info.stop_date_time())

        # At XPath '/Product_Observational/Observation_Area/Investigation_Area'
        name, type, internal_reference = \
            self.create_children(investigation_area, ['name', 'type',
                                                      'Internal_Reference'])
        self.set_text(name, self.info.investigation_area_name())
        self.set_text(type, self.info.investigation_area_type())

        # At XPath
        # '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference'
        reference_type = self.create_child(internal_reference,
                                           'reference_type')
        self.set_text(reference_type, self.info.internal_reference_type())

        # At XPath '/Product_Observational/Observation_Area'
        instrument = self.component.collection().instrument()
        if instrument == 'acs':
            instrumentMaker = InstrumentXmlMaker.AcsXmlMaker(self.document)
        elif instrument == 'wfc3':
            instrumentMaker = InstrumentXmlMaker.Wfc3XmlMaker(self.document)
        elif instrument == 'wfpc2':
            instrumentMaker = InstrumentXmlMaker.Wfpc2XmlMaker(self.document)
        else:
            raise Exception('Unknown instrument %s' % instrument)
        instrumentMaker.create_xml(observation_area)

        # At XPath '/Product_Observational'
        def runFPFXM(archiveFile):
            try:
                fpfxm = FitsProductFileXmlMaker.FitsProductFileXmlMaker(
                    self.document, archiveFile)
                fpfxm.create_xml(root)
                return fpfxm.targname
            except IOError:
                return None

        targnames = set([runFPFXM(archiveFile)
                         for archiveFile in product.files()])
        if len(targnames) == 1:
            # Consistent
            targname = targnames.pop()
            if targname:
                self.create_target_identification_xml(targname,
                                                      observation_area)
        elif len(targnames) == 0:
            # TODO How to handle?
            pass
        else:
            # Inconsistent
            print 'Targnames == %s' % targnames

    def create_target_identification_xml(self,
                                         targname,
                                         observation_area):
        # Move this back up
        target_identification = self.create_child(observation_area,
                                                  'Target_Identification')
        nameType = Targets.targname_to_target(targname)
        if nameType:
            name, type, internal_reference = \
                self.create_children(target_identification,
                                     ['name', 'type', 'Internal_Reference'])
            self.set_text(name, nameType[0])
            self.set_text(type, nameType[1])
            lid_reference, reference_type = \
                self.create_children(internal_reference,
                                     ['lid_reference', 'reference_type'])
            self.set_text(lid_reference,
                          'urn:nasa:pds:context:target:%s.%s' %
                          (nameType[1].lower(), nameType[0].lower()))
            self.set_text(reference_type, 'data_to_target')
        else:
            name, type = \
                self.create_children(target_identification, ['name', 'type'])
            self.set_text(name, self.info.unknown_target_identification_name())
            self.set_text(type,
                          self.info.unknown_target_identification_type())


def test_synthesis():
    a = FileArchives.get_any_archive()
    fp = '/tmp/foo.xml'
    for b in a.bundles():
        for c in b.collections():
            for p in c.products():
                print p
                lm = ProductLabelMaker(p)
                lm.write_xml_to_file(fp)
                if not (LabelMaker.xml_schema_check(fp) and
                        LabelMaker.schematron_check(fp)):
                    print '%s did not validate; aborting' % p
                    sys.exit(1)


def make_and_test_product_label(lid):
    if isinstance(lid, str):
        lid = LID.LID(lid)
    assert isinstance(lid, LID.LID)
    a = FileArchives.get_any_archive()
    p = Product.Product(a, lid)
    lm = ProductLabelMaker(p)
    fp = '/tmp/product.xml'
    lm.write_xml_to_file(fp)
    failures = XmlSchema.xml_schema_failures(fp) or \
        XmlSchema.schematron_failures(fp)
    if failures:
        print 'Label for %s at %r did not validate.' % (p, fp)
        print failures
        sys.exit(1)


if __name__ == '__main__':
    # test_synthesis()
    make_and_test_product_label(
        'urn:nasa:pds:hst_05167:data_wfpc2_cmh:visit_04')
