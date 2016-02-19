import io
import os
import xml.dom

import CollectionInfo
import FileArchives
import LabelMaker


class CollectionLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, collection):
        LabelMaker.LabelMaker.__init__(
            self, collection, CollectionInfo.CollectionInfo(collection))
        self.inventoryDocument = None
        self.createDefaultCollectionInventory()

    def defaultXmlName(self):
        collection = self.component
        return 'collection_%s.xml' % collection.suffix()

    def defaultInventoryName(self):
        collection = self.component
        return 'collection_%s_inventory.tab' % collection.suffix()

    def createDefaultCollectionInventory(self):
        collection = self.component
        lines = [u'P,%s\n' % str(p.lid) for p in collection.products()]
        # Line endings are native here (i.e., possibly wrong,
        # depending on the platform), but we make sure to write using
        # io.open() with newline='\r\n'
        self.inventoryDocument = ''.join(lines)

    def createDefaultInventoryFile(self, invFilepath=None):
        if invFilepath is None:
            invName = self.defaultInventoryName()
            invFilepath = os.path.join(self.component.directoryFilepath(),
                                       invName)
        # Line endings in the inventoryDocument are native (i.e.,
        # possibly wrong, depending on the platform), so we must write
        # using io.open() with newline='\r\n'
        with io.open(invFilepath, 'w', newline='\r\n') as f:
            f.write(self.inventoryDocument)

    def createDefaultXml(self):
        collection = self.component
        root = self._createChild(self.document, 'Product_Collection')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        id_area, coll, fa_inv = \
            self._createChildren(root, ['Identification_Area',
                                        'Collection',
                                        'File_Area_Inventory'])

        log_id, vers_id, title, info_ver, prod_cls, cit_info = \
            self._createChildren(id_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self._setText(log_id, str(collection.lid))
        self._setText(vers_id, self.info.versionID())
        self._setText(title, self.info.title())
        self._setText(info_ver, self.info.informationModelVersion())
        self._setText(prod_cls, 'Product_Collection')

        pub_yr = self._createChild(cit_info, 'publication_year')
        self._setText(pub_yr, self.info.citationInformationPublicationYear())
        desc = self._createChild(cit_info, 'description')
        self._setText(desc, self.info.citationInformationDescription())

        self._setText(self._createChild(coll, 'collection_type'), 'Data')

        f, inv = self._createChildren(fa_inv, ['File', 'Inventory'])
        fn = self._createChild(f, 'file_name')
        self._setText(fn, self.defaultInventoryName())

        off, pars, records, r_delim, f_delim, rec_delim, ref_ty = \
            self._createChildren(inv, [
                'offset', 'parsing_standard_id',
                'records', 'record_delimiter', 'field_delimiter',
                'Record_Delimited', 'reference_type'])

        self._setText(off, '0')
        off.setAttribute('unit', 'byte')
        self._setText(pars, 'PDS DSV 1')

        productCount = 0
        for p in collection.products():
            productCount += 1
        self._setText(records, str(productCount))

        self._setText(r_delim, 'Carriage-Return Line-Feed')
        self._setText(f_delim, 'Comma')

        fs, gs, f1, f2 = self._createChildren(rec_delim, [
                'fields', 'groups', 'Field_Delimited', 'Field_Delimited'])
        self._setText(fs, '2')
        self._setText(gs, '0')

        nm, fn, dt, mfl = \
            self._createChildren(f1, ['name', 'field_number',
                                      'data_type', 'maximum_field_length'])
        self._setText(nm, 'Member_Status')
        self._setText(fn, '1')
        self._setText(dt, 'ASCII_String')
        mfl.setAttribute('unit', 'byte')
        self._setText(mfl, '1')

        nm, fn, dt = \
            self._createChildren(f2, ['name', 'field_number', 'data_type'])

        self._setText(nm, 'LIDVID_LID')
        self._setText(fn, '2')
        self._setText(dt, 'ASCII_LIDVID_LID')

        self._setText(ref_ty, 'inventory_has_member_product')


def testSynthesis():
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            for c in b.collections():
                lm = CollectionLabelMaker(c)
                lm.createDefaultXmlFile('collection.xml')
                lm.createDefaultInventoryFile(
                    'collection_suffix_inventory.tab')
                if LabelMaker.xmlSchemaCheck('collection.xml'):
                    print ('Yay: collection.xml for %s ' +
                           'conforms to the XML schema.') % str(b)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the XML schema.') % str(b)
                    return
                if LabelMaker.schematronCheck('collection.xml'):
                    print ('Yay: collection.xml for %s ' +
                           'conforms to the Schematron schema.') % str(b)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the Schematron schema.') % \
                           str(b)

testSynthesis()
