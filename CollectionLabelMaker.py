import io
import os

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
        root = self.createChild(self.document, 'Product_Collection')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, collection_, fileAreaInventory = \
            self.createChildren(root, ['Identification_Area',
                                       'Collection',
                                       'File_Area_Inventory'])

        logicalIdentifier, versionId, title, informationModelVersion, \
            productClass, citationInformation = \
            self.createChildren(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.setText(logicalIdentifier, str(collection.lid))
        self.setText(versionId, self.info.versionID())
        self.setText(title, self.info.title())
        self.setText(informationModelVersion,
                     self.info.informationModelVersion())
        self.setText(productClass, 'Product_Collection')

        publicationYear = self.createChild(citationInformation,
                                           'publication_year')
        self.setText(publicationYear,
                     self.info.citationInformationPublicationYear())
        description = self.createChild(citationInformation, 'description')
        self.setText(description, self.info.citationInformationDescription())

        self.setText(self.createChild(collection_, 'collection_type'), 'Data')

        file, inventory = self.createChildren(fileAreaInventory,
                                              ['File', 'Inventory'])
        fileName = self.createChild(file, 'file_name')
        self.setText(fileName, self.defaultInventoryName())

        offset, parsingStandardId, records, recordDelimiter, \
            fieldDelimiter, recordDelimited, referenceType = \
            self.createChildren(inventory, [
                'offset', 'parsing_standard_id',
                'records', 'record_delimiter', 'field_delimiter',
                'Record_Delimited', 'reference_type'])

        self.setText(offset, '0')
        offset.setAttribute('unit', 'byte')
        self.setText(parsingStandardId, 'PDS DSV 1')

        productCount = 0
        for p in collection.products():
            productCount += 1
        self.setText(records, str(productCount))

        self.setText(recordDelimiter, 'Carriage-Return Line-Feed')
        self.setText(fieldDelimiter, 'Comma')

        fields, groups, fieldDelimited1, fieldDelimited2 = \
            self.createChildren(recordDelimited, [
                'fields', 'groups', 'Field_Delimited', 'Field_Delimited'])
        self.setText(fields, '2')
        self.setText(groups, '0')

        name, fieldNumber, dataType, maximumFieldLength = \
            self.createChildren(fieldDelimited1,
                                ['name', 'field_number',
                                 'data_type', 'maximum_field_length'])
        self.setText(name, 'Member_Status')
        self.setText(fieldNumber, '1')
        self.setText(dataType, 'ASCII_String')
        maximumFieldLength.setAttribute('unit', 'byte')
        self.setText(maximumFieldLength, '1')

        name, fieldNumber, dataType = \
            self.createChildren(fieldDelimited2,
                                ['name', 'field_number', 'data_type'])

        self.setText(name, 'LIDVID_LID')
        self.setText(fieldNumber, '2')
        self.setText(dataType, 'ASCII_LIDVID_LID')

        self.setText(referenceType, 'inventory_has_member_product')


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
                           'conforms to the XML schema.') % str(c)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the XML schema.') % str(c)
                    return
                if LabelMaker.schematronCheck('collection.xml'):
                    print ('Yay: collection.xml for %s ' +
                           'conforms to the Schematron schema.') % str(c)
                else:
                    print ('Boo: collection.xml for %s ' +
                           'does not conform to the Schematron schema.') % \
                           str(c)

testSynthesis()
