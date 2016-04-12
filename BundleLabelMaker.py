import pdart.pds4.Bundle
import BundleInfo
import FileArchives
import LabelMaker


class BundleLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, bundle):
        assert isinstance(bundle, pdart.pds4.Bundle.Bundle)
        super(BundleLabelMaker, self).__init__(bundle,
                                               BundleInfo.BundleInfo(bundle))

    def default_xml_name(self):
        return 'bundle.xml'

    def create_xml(self):
        bundle = self.component

        # At XPath '/'
        self.add_processing_instruction('xml-model',
                                        self.info.xml_model_pds_attributes())

        root = self.create_child(self.document, 'Product_Bundle')

        # At XPath '/Product_Bundle'
        PDS = self.info.pds_namespace_url()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identification_area, bundle_ = \
            self.create_children(root, ['Identification_Area', 'Bundle'])

        # At XPath '/Product_Bundle/Identification_Area'
        logical_identifier, version_id, title, information_model_version, \
            product_class, citation_information = \
            self.create_children(identification_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.set_text(logical_identifier, str(bundle.lid))
        self.set_text(version_id, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(product_class, 'Product_Bundle')

        # At XPath '/Product_Bundle/Identification_Area/Citation_Information'
        publication_year, description = \
            self.create_children(citation_information,
                                 ['publication_year', 'description'])

        self.set_text(publication_year,
                      self.info.citation_information_publication_year())
        self.set_text(description,
                      self.info.citation_information_description())

        # At XPath '/Product_Bundle/Bundle'
        bundle_type = self.create_child(bundle_, 'bundle_type')
        self.set_text(bundle_type, 'Archive')

        for collection in bundle.collections():
            # At XPath '/Product_Bundle/Bundle_Member_Entry'
            bundle_member_entry = self.create_child(root,
                                                    'Bundle_Member_Entry')
            lid_reference, member_status, reference_type = \
                self.create_children(bundle_member_entry, [
                    'lid_reference',
                    'member_status',
                    'reference_type'])
            self.set_text(lid_reference, str(collection.lid))
            self.set_text(member_status, 'Primary')
            self.set_text(reference_type, 'bundle_has_data_collection')


def test_synthesis():
    # Create sample bundle.xml files for the non-hst_00000 bundles and
    # test them against the XML schema.
    a = FileArchives.get_any_archive()
    for b in a.bundles():
        if b.proposal_id() != 0:
            lm = BundleLabelMaker(b)
            lm.write_xml_to_file('bundle.xml')
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
