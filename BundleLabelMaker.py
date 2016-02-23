import Bundle
import BundleInfo
import FileArchives
import LabelMaker


class BundleLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, bundle):
        assert isinstance(bundle, Bundle.Bundle)
        LabelMaker.LabelMaker.__init__(self, bundle,
                                       BundleInfo.BundleInfo(bundle))

    def defaultXmlName(self):
        return 'bundle.xml'

    def createDefaultXml(self):
        bundle = self.component
        root = self.createChild(self.document, 'Product_Bundle')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, bundle_ = \
            self.createChildren(root, ['Identification_Area', 'Bundle'])

        logicalIdentifier, versionId, title, informationModelVersion, \
            productClass, citationInformation = \
            self.createChildren(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.setText(logicalIdentifier, str(bundle.lid))
        self.setText(versionId, self.info.versionID())
        self.setText(title, self.info.title())
        self.setText(informationModelVersion,
                     self.info.informationModelVersion())
        self.setText(productClass, 'Product_Bundle')

        publicationYear = self.createChild(citationInformation,
                                           'publication_year')
        self.setText(publicationYear,
                     self.info.citationInformationPublicationYear())

        description = self.createChild(citationInformation, 'description')
        self.setText(description, self.info.citationInformationDescription())

        bundleType = self.createChild(bundle_, 'bundle_type')
        self.setText(bundleType, 'Archive')

        for collection in bundle.collections():
            bundleMemberEntry = self.createChild(root, 'Bundle_Member_Entry')
            lidReference, memberStatus, referenceType = \
                self.createChildren(bundleMemberEntry, [
                    'lid_reference',
                    'member_status',
                    'reference_type'])
            self.setText(lidReference, str(collection.lid))
            self.setText(memberStatus, 'Primary')
            self.setText(referenceType, 'bundle_has_data_collection')


def testSynthesis():
    # Create sample bundle.xml files for the non-hst_00000 bundles and
    # test them against the XML schema.
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            lm = BundleLabelMaker(b)
            lm.createDefaultXmlFile('bundle.xml')
            if LabelMaker.xmlSchemaCheck('bundle.xml'):
                print ('Yay: bundle.xml for %s ' +
                       'conforms to the XML schema.') % str(b)
            else:
                print ('Boo: bundle.xml for %s ' +
                       'does not conform to the XML schema.') % str(b)
                return
            if LabelMaker.schematronCheck('bundle.xml'):
                print ('Yay: bundle.xml for %s ' +
                       'conforms to the Schematron schema.') % str(b)
            else:
                print ('Boo: bundle.xml for %s ' +
                       'does not conform to the Schematron schema.') % str(b)
                return

if __name__ == '__main__':
    testSynthesis()
