import sys
import tempfile

import ArchiveFile
import DummyProductFileLabelMaker
import FileArchives
import LabelMaker
import ProductInfo


class ProductLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, product):
        super(ProductLabelMaker,
              self).__init__(product, ProductInfo.ProductInfo(product))

    def defaultXmlName(self):
        assert False, 'ProductLabelMaker.defaultXmlName unimplemented'

    def createDefaultXml(self):
        product = self.component

        # At XPath '/Product_Observational'
        root = self.createChild(self.document, 'Product_Observational')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, observationArea = \
            self.createChildren(root, ['Identification_Area',
                                       'Observation_Area'])

        # At XPath '/Product_Observational/Identification_Area'
        logicalIdentifier, versionId, title, \
            informationModelVersion, productClass = \
            self.createChildren(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class'])

        self.setText(logicalIdentifier, str(product.lid))
        self.setText(versionId, self.info.versionID())
        self.setText(title, self.info.title())
        self.setText(informationModelVersion,
                     self.info.informationModelVersion())
        self.setText(productClass, 'Product_Observational')

        # At XPath '/Product_Observational/Observation_Area'
        timeCoordinates, investigationArea, \
            observingSystem, targetIdentification = \
            self.createChildren(observationArea, ['Time_Coordinates',
                                                  'Investigation_Area',
                                                  'Observing_System',
                                                  'Target_Identification'])

        # At XPath '/Product_Observational/Observation_Area/Time_Coordinates'
        startDateTime, stopDateTime = \
            self.createChildren(timeCoordinates, ['start_date_time',
                                                  'stop_date_time'])
        self.setText(startDateTime, self.info.startDateTime())
        self.setText(stopDateTime, self.info.stopDateTime())

        # At XPath '/Product_Observational/Observation_Area/Investigation_Area'
        name, type, internalReference = \
            self.createChildren(investigationArea, ['name', 'type',
                                                    'Internal_Reference'])
        self.setText(name, self.info.investigationAreaName())
        self.setText(type, self.info.investigationAreaType())

        # At XPath
        # '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference'
        referenceType = self.createChild(internalReference, 'reference_type')
        self.setText(referenceType, self.info.internalReferenceType())

        # At XPath '/Product_Observational/Observation_Area/Observing_System'
        self.createObservingSystemXml(observingSystem)

        # At XPath
        # '/Product_Observational/Observation_Area/Target_Identification'
        name, type = self.createChildren(targetIdentification,
                                         ['name', 'type'])
        self.setText(name, self.info.targetIdentificationName())
        self.setText(type, self.info.targetIdentificationType())

        # At XPath '/Product_Observational'
        for archiveFile in product.files():
            DummyProductFileLabelMaker.DummyProductFileLabelMaker(
                self.document, root, archiveFile)

    def createObservingSystemXml(self, observingSystem):
        # At XPath
        # '/Product_Observational/Observation_Area/Observing_System/'
        instrument = self.component.collection().instrument()
        if instrument == 'acs':
            name, observingSystemComponentHST, observingSystemComponentACS = \
                self.createChildren(observingSystem,
                                    ['name',
                                     'Observing_System_Component',
                                     'Observing_System_Component'])
            self.setText(name, 'Hubble Space Telescope Advanced Camera for Surveys')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[0]'
            name, type, internalReference = \
                self.createChildren(observingSystemComponentHST,
                                    ['name', 'type', 'Internal_Reference'])
            self.setText(name, 'Hubble Space Telescope')
            self.setText(type, 'Spacecraft')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[0]/Internal_Reference'
            lidReference, referenceType = \
                self.createChildren(internalReference,
                                    ['lid_reference', 'reference_type'])
            self.setText(lidReference,
                         'urn:nasa:pds:context:investigation:mission.hst')
            self.setText(referenceType, 'is_instrument_host')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[1]'
            name, type, internalReference = \
                self.createChildren(observingSystemComponentACS,
                                    ['name', 'type', 'Internal_Reference'])
            self.setText(name, 'Advanced Camera for Surveys')
            self.setText(type, 'Instrument')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component[1]/Internal_Reference'
            lidReference, referenceType = \
                self.createChildren(internalReference,
                                    ['lid_reference', 'reference_type'])
            self.setText(lidReference,
                         'urn:nasa:pds:context:investigation:mission.hst_acs')
            self.setText(referenceType, 'is_instrument')

        else:  # default path

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System'
            observingSystemComponent = \
                self.createChild(observingSystem, 'Observing_System_Component')

            # At XPath
            # '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component'
            name, type = self.createChildren(observingSystemComponent,
                                             ['name', 'type'])
            self.setText(name, self.info.observingSystemComponentName())
            self.setText(type, self.info.observingSystemComponentType())


def testSynthesis():
    a = FileArchives.getAnyArchive()
    fp = '/tmp/foo.xml'
    for b in a.bundles():
        for c in b.collections():
            for p in c.products():
                print p
                lm = ProductLabelMaker(p)
                lm.createDefaultXmlFile(fp)
                if not (LabelMaker.xmlSchemaCheck(fp) and
                        LabelMaker.schematronCheck(fp)):
                    print '%s did not validate; aborting' % p
                    sys.exit(1)

if __name__ == '__main__':
    testSynthesis()
