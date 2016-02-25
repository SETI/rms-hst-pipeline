import io
import os

import CollectionInfo
import FileArchives
import LabelMaker


class CollectionLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, collection):
        super(CollectionLabelMaker, self).__init__(
            collection, CollectionInfo.CollectionInfo(collection))
        self.inventoryDocument = None
        self.createDefaultCollectionInventory()

    def default_xml_name(self):
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
            invFilepath = os.path.join(self.component.directory_filepath(),
                                       invName)
        # Line endings in the inventoryDocument are native (i.e.,
        # possibly wrong, depending on the platform), so we must write
        # using io.open() with newline='\r\n'
        with io.open(invFilepath, 'w', newline='\r\n') as f:
            f.write(self.inventoryDocument)

    def create_default_xml(self):
        collection = self.component
        # At XPath '/Product_Collection'
        root = self.create_child(self.document, 'Product_Collection')

        PDS = self.info.pds_namespace_url()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, collection_, fileAreaInventory = \
            self.create_children(root, ['Identification_Area',
                                        'Collection',
                                        'File_Area_Inventory'])

        # At XPath '/Product_Collection/Identification_Area'
        logicalIdentifier, versionId, title, information_model_version, \
            productClass, citationInformation = \
            self.create_children(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.set_text(logicalIdentifier, str(collection.lid))
        self.set_text(versionId, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(productClass, 'Product_Collection')

        # At XPath
        # '/Product_Collection/Identification_Area/Citation_Information'
        publicationYear, description = \
            self.create_children(citationInformation,
                                 ['publication_year', 'description'])

        self.set_text(publicationYear,
                      self.info.citation_information_publication_year())

        self.set_text(description, self.info.citation_information_description())

        # At XPath '/Product_Collection/Collection'
        self.set_text(self.create_child(collection_, 'collection_type'), 'Data')

        # At XPath '/Product_Collection/File_Area_Inventory'
        file, inventory = self.create_children(fileAreaInventory,
                                               ['File', 'Inventory'])
        # At XPath '/Product_Collection/File_Area_Inventory/File'
        fileName = self.create_child(file, 'file_name')
        self.set_text(fileName, self.defaultInventoryName())

        # At XPath '/Product_Collection/File_Area_Inventory/Inventory'
        offset, parsingStandardId, records, recordDelimiter, \
            fieldDelimiter, recordDelimited, referenceType = \
            self.create_children(inventory, [
                'offset', 'parsing_standard_id',
                'records', 'record_delimiter', 'field_delimiter',
                'Record_Delimited', 'reference_type'])

        self.set_text(offset, '0')
        offset.setAttribute('unit', 'byte')
        self.set_text(parsingStandardId, 'PDS DSV 1')

        productCount = 0
        for p in collection.products():
            productCount += 1
        self.set_text(records, str(productCount))

        self.set_text(recordDelimiter, 'Carriage-Return Line-Feed')
        self.set_text(fieldDelimiter, 'Comma')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited'
        fields, groups, fieldDelimited1, fieldDelimited2 = \
            self.create_children(recordDelimited, [
                'fields', 'groups', 'Field_Delimited', 'Field_Delimited'])
        self.set_text(fields, '2')
        self.set_text(groups, '0')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited/Field_Delimited[1]'
        name, fieldNumber, dataType, maximumFieldLength = \
            self.create_children(fieldDelimited1,
                                 ['name', 'field_number',
                                  'data_type', 'maximum_field_length'])
        self.set_text(name, 'Member_Status')
        self.set_text(fieldNumber, '1')
        self.set_text(dataType, 'ASCII_String')
        maximumFieldLength.setAttribute('unit', 'byte')
        self.set_text(maximumFieldLength, '1')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited/Field_Delimited[2]'
        name, fieldNumber, dataType = \
            self.create_children(fieldDelimited2,
                                 ['name', 'field_number', 'data_type'])

        self.set_text(name, 'LIDVID_LID')
        self.set_text(fieldNumber, '2')
        self.set_text(dataType, 'ASCII_LIDVID_LID')

        self.set_text(referenceType, 'inventory_has_member_product')


def test_synthesis():
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposal_id() != 0:
            for c in b.collections():
                lm = CollectionLabelMaker(c)
                lm.create_default_xml_file('collection.xml')
                lm.createDefaultInventoryFile(
                    'collection_suffix_inventory.tab')
                if LabelMaker.xml_schema_check('collection.xml'):
                    print ('Yay: collection.xml for %s ' +
                           'conforms to the XML schema.') % str(c)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the XML schema.') % str(c)
                    return
                if LabelMaker.schematron_check('collection.xml'):
                    print ('Yay: collection.xml for %s ' +
                           'conforms to the Schematron schema.') % str(c)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the Schematron schema.') % \
                           str(c)

if __name__ == '__main__':
    test_synthesis()
