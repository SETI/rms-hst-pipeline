import Bundle
import BundleInfo
import FileArchives
import LabelMaker


class BundleLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, bundle):
        assert isinstance(bundle, Bundle.Bundle)
        super(BundleLabelMaker, self).__init__(bundle,
                                               BundleInfo.BundleInfo(bundle))

    def default_xml_name(self):
        return 'bundle.xml'

    def create_default_xml(self):
        bundle = self.component
        root = self.create_child(self.document, 'Product_Bundle')

        # At XPath '/Product_Bundle'
        PDS = self.info.pds_namespace_url()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, bundle_ = \
            self.create_children(root, ['Identification_Area', 'Bundle'])

        # At XPath '/Product_Bundle/Identification_Area'
        logicalIdentifier, versionId, title, information_model_version, \
            productClass, citationInformation = \
            self.create_children(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.set_text(logicalIdentifier, str(bundle.lid))
        self.set_text(versionId, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(productClass, 'Product_Bundle')

        # At XPath '/Product_Bundle/Identification_Area/Citation_Information'
        publicationYear, description = \
            self.create_children(citationInformation,
                                 ['publication_year', 'description'])

        self.set_text(publicationYear,
                      self.info.citation_information_publication_year())
        self.set_text(description, self.info.citation_information_description())

        # At XPath '/Product_Bundle/Bundle'
        bundleType = self.create_child(bundle_, 'bundle_type')
        self.set_text(bundleType, 'Archive')

        for collection in bundle.collections():
            # At XPath '/Product_Bundle/Bundle_Member_Entry'
            bundleMemberEntry = self.create_child(root, 'Bundle_Member_Entry')
            lidReference, memberStatus, referenceType = \
                self.create_children(bundleMemberEntry, [
                    'lid_reference',
                    'member_status',
                    'reference_type'])
            self.set_text(lidReference, str(collection.lid))
            self.set_text(memberStatus, 'Primary')
            self.set_text(referenceType, 'bundle_has_data_collection')


def test_synthesis():
    # Create sample bundle.xml files for the non-hst_00000 bundles and
    # test them against the XML schema.
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposal_id() != 0:
            lm = BundleLabelMaker(b)
            lm.create_default_xml_file('bundle.xml')
            if LabelMaker.xml_schema_check('bundle.xml'):
                print ('Yay: bundle.xml for %s ' +
                       'conforms to the XML schema.') % str(b)
            else:
                print ('Boo: bundle.xml for %s ' +
                       'does not conform to the XML schema.') % str(b)
                return
            if LabelMaker.schematron_check('bundle.xml'):
                print ('Yay: bundle.xml for %s ' +
                       'conforms to the Schematron schema.') % str(b)
            else:
                print ('Boo: bundle.xml for %s ' +
                       'does not conform to the Schematron schema.') % str(b)
                return

if __name__ == '__main__':
    test_synthesis()
