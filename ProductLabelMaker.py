import sys
import tempfile

import ArchiveFile
import DummyProductFileXmlMaker
import FileArchives
import LabelMaker
import ProductInfo


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

        identificationArea, observationArea = \
            self.create_children(root, ['Identification_Area',
                                        'Observation_Area'])

        # At XPath '/Product_Observational/Identification_Area'
        logicalIdentifier, versionId, title, \
            information_model_version, productClass = \
            self.create_children(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class'])

        self.set_text(logicalIdentifier, str(product.lid))
        self.set_text(versionId, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(productClass, 'Product_Observational')

        # At XPath '/Product_Observational/Observation_Area'
        timeCoordinates, investigationArea, \
            observingSystem, targetIdentification = \
            self.create_children(observationArea, ['Time_Coordinates',
                                                   'Investigation_Area',
                                                   'Observing_System',
                                                   'Target_Identification'])

        # At XPath '/Product_Observational/Observation_Area/Time_Coordinates'
        start_date_time, stop_date_time = \
            self.create_children(timeCoordinates, ['start_date_time',
                                                   'stop_date_time'])
        self.set_text(start_date_time, self.info.start_date_time())
        self.set_text(stop_date_time, self.info.stop_date_time())

        # At XPath '/Product_Observational/Observation_Area/Investigation_Area'
        name, type, internalReference = \
            self.create_children(investigationArea, ['name', 'type',
                                                     'Internal_Reference'])
        self.set_text(name, self.info.investigation_area_name())
        self.set_text(type, self.info.investigation_area_type())

        # At XPath
        # '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference'
        referenceType = self.create_child(internalReference, 'reference_type')
        self.set_text(referenceType, self.info.internal_reference_type())

        # At XPath '/Product_Observational/Observation_Area/Observing_System'
        self.createObservingSystemXml(observingSystem)

        # At XPath
        # '/Product_Observational/Observation_Area/Target_Identification'
        name, type = self.create_children(targetIdentification,
                                          ['name', 'type'])
        self.set_text(name, self.info.target_identification_name())
        self.set_text(type, self.info.target_identification_type())

        # At XPath '/Product_Observational'
        for archiveFile in product.files():
            DummyProductFileXmlMaker.DummyProductFileXmlMaker(
                self.document, root, archiveFile)

    def createObservingSystemXml(self, observingSystem):
        # At XPath
        # '/Product_Observational/Observation_Area/Observing_System/'
        instrument = self.component.collection().instrument()
        if instrument == 'acs':
            name, observingSystemComponentHST, observingSystemComponentACS = \
                self.create_children(observingSystem,
                                     ['name',
                                      'Observing_System_Component',
                                      'Observing_System_Component'])
            self.set_text(name,
                          'Hubble Space Telescope Advanced Camera for Surveys')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[0]'
            name, type, internalReference = \
                self.create_children(observingSystemComponentHST,
                                     ['name', 'type', 'Internal_Reference'])
            self.set_text(name, 'Hubble Space Telescope')
            self.set_text(type, 'Spacecraft')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[0]/Internal_Reference'
            lidReference, referenceType = \
                self.create_children(internalReference,
                                     ['lid_reference', 'reference_type'])
            self.set_text(lidReference,
                          'urn:nasa:pds:context:investigation:mission.hst')
            self.set_text(referenceType, 'is_instrument_host')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[1]'
            name, type, internalReference = \
                self.create_children(observingSystemComponentACS,
                                     ['name', 'type', 'Internal_Reference'])
            self.set_text(name, 'Advanced Camera for Surveys')
            self.set_text(type, 'Instrument')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[1]/Internal_Reference'
            lidReference, referenceType = \
                self.create_children(internalReference,
                                     ['lid_reference', 'reference_type'])
            self.set_text(lidReference,
                          'urn:nasa:pds:context:investigation:mission.hst_acs')
            self.set_text(referenceType, 'is_instrument')

        else:  # default path

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System'
            observingSystemComponent = \
                self.create_child(observingSystem, 'Observing_System_Component')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component'
            name, type = self.create_children(observingSystemComponent,
                                              ['name', 'type'])
            self.set_text(name, self.info.observing_system_component_name())
            self.set_text(type, self.info.observing_system_component_type())


def test_synthesis():
    a = FileArchives.getAnyArchive()
    fp = '/tmp/foo.xml'
    for b in a.bundles():
        for c in b.collections():
            for p in c.products():
                print p
                lm = ProductLabelMaker(p)
                lm.create_default_xml_file(fp)
                if not (LabelMaker.xml_schema_check(fp) and
                        LabelMaker.schematron_check(fp)):
                    print '%s did not validate; aborting' % p
                    sys.exit(1)


if __name__ == '__main__':
    test_synthesis()
