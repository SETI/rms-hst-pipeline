import os
import xml.dom

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
        root = self._createChild(self.document, 'Product_Bundle')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        id_area, b = \
            self._createChildren(root, ['Identification_Area', 'Bundle'])

        log_id, vers_id, title, info_ver, prod_cls, cit_info = \
            self._createChildren(id_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self._setText(log_id, str(bundle.lid))
        self._setText(vers_id, self.info.versionID())
        self._setText(title, self.info.title())
        self._setText(info_ver, self.info.informationModelVersion())
        self._setText(prod_cls, 'Product_Bundle')

        pub_yr = self._createChild(cit_info, 'publication_year')
        self._setText(pub_yr, self.info.citationInformationPublicationYear())
        desc = self._createChild(cit_info, 'description')
        self._setText(desc, self.info.citationInformationDescription())

        b_ty = self._createChild(b, 'bundle_type')
        self._setText(b_ty, 'Archive')

        for collection in bundle.collections():
            mem_entry = self._createChild(root, 'Bundle_Member_Entry')
            lid, stat, ref_ty = self._createChildren(mem_entry, [
                    'lid_reference',
                    'member_status',
                    'reference_type'])
            self._setText(lid, str(collection.lid))
            self._setText(stat, 'Primary')
            self._setText(ref_ty, 'bundle_has_data_collection')


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

testSynthesis()
