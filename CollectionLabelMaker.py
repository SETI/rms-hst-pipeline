import io
import os

import CollectionInfo
import FileArchives
import LabelMaker


class CollectionLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, collection):
        super(CollectionLabelMaker, self).__init__(
            collection, CollectionInfo.CollectionInfo(collection))
        self.inventory_document = None
        self.create_default_collection_inventory()

    def default_xml_name(self):
        collection = self.component
        return 'collection_%s.xml' % collection.suffix()

    def default_inventory_name(self):
        collection = self.component
        return 'collection_%s_inventory.tab' % collection.suffix()

    def create_default_collection_inventory(self):
        collection = self.component
        lines = [u'P,%s\n' % str(p.lid) for p in collection.products()]
        # Line endings are native here (i.e., possibly wrong,
        # depending on the platform), but we make sure to write using
        # io.open() with newline='\r\n'
        self.inventory_document = ''.join(lines)

    def create_default_inventory_file(self, inv_filepath=None):
        if inv_filepath is None:
            inv_name = self.default_inventory_name()
            inv_filepath = os.path.join(self.component.directory_filepath(),
                                        inv_name)
        # Line endings in the inventory_document are native (i.e.,
        # possibly wrong, depending on the platform), so we must write
        # using io.open() with newline='\r\n'
        with io.open(inv_filepath, 'w', newline='\r\n') as f:
            f.write(self.inventory_document)

    def create_xml(self):
        collection = self.component

        # At XPath '/'
        self.add_processing_instruction('xml-model',
                                        self.info.xml_model_pds_attributes())

        # At XPath '/Product_Collection'
        root = self.create_child(self.document, 'Product_Collection')

        PDS = self.info.pds_namespace_url()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identification_area, collection_, file_area_inventory = \
            self.create_children(root, ['Identification_Area',
                                        'Collection',
                                        'File_Area_Inventory'])

        # At XPath '/Product_Collection/Identification_Area'
        logical_identifier, version_id, title, information_model_version, \
            product_class, citation_information = \
            self.create_children(identification_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class',
                'Citation_Information'])

        self.set_text(logical_identifier, str(collection.lid))
        self.set_text(version_id, self.info.version_id())
        self.set_text(title, self.info.title())
        self.set_text(information_model_version,
                      self.info.information_model_version())
        self.set_text(product_class, 'Product_Collection')

        # At XPath
        # '/Product_Collection/Identification_Area/Citation_Information'
        publication_year, description = \
            self.create_children(citation_information,
                                 ['publication_year', 'description'])

        self.set_text(publication_year,
                      self.info.citation_information_publication_year())

        self.set_text(description,
                      self.info.citation_information_description())

        # At XPath '/Product_Collection/Collection'
        self.set_text(self.create_child(collection_, 'collection_type'),
                      'Data')

        # At XPath '/Product_Collection/File_Area_Inventory'
        file, inventory = self.create_children(file_area_inventory,
                                               ['File', 'Inventory'])
        # At XPath '/Product_Collection/File_Area_Inventory/File'
        file_name = self.create_child(file, 'file_name')
        self.set_text(file_name, self.default_inventory_name())

        # At XPath '/Product_Collection/File_Area_Inventory/Inventory'
        offset, parsing_standard_id, records, record_delimiter, \
            field_delimiter, record_delimited, reference_type = \
            self.create_children(inventory, [
                'offset', 'parsing_standard_id',
                'records', 'record_delimiter', 'field_delimiter',
                'Record_Delimited', 'reference_type'])

        self.set_text(offset, '0')
        offset.setAttribute('unit', 'byte')
        self.set_text(parsing_standard_id, 'PDS DSV 1')

        product_count = 0
        for p in collection.products():
            product_count += 1
        self.set_text(records, str(product_count))

        self.set_text(record_delimiter, 'Carriage-Return Line-Feed')
        self.set_text(field_delimiter, 'Comma')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited'
        fields, groups, field_delimited1, field_delimited2 = \
            self.create_children(record_delimited, [
                'fields', 'groups', 'Field_Delimited', 'Field_Delimited'])
        self.set_text(fields, '2')
        self.set_text(groups, '0')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited/Field_Delimited[1]'
        name, field_number, date_type, maximum_field_length = \
            self.create_children(field_delimited1,
                                 ['name', 'field_number',
                                  'data_type', 'maximum_field_length'])
        self.set_text(name, 'Member_Status')
        self.set_text(field_number, '1')
        self.set_text(date_type, 'ASCII_String')
        maximum_field_length.setAttribute('unit', 'byte')
        self.set_text(maximum_field_length, '1')

        # At XPath
        # '/Product_Collection/File_Area_Inventory/Inventory/Record_Delimited/Field_Delimited[2]'
        name, field_number, date_type = \
            self.create_children(field_delimited2,
                                 ['name', 'field_number', 'data_type'])

        self.set_text(name, 'LIDVID_LID')
        self.set_text(field_number, '2')
        self.set_text(date_type, 'ASCII_LIDVID_LID')

        self.set_text(reference_type, 'inventory_has_member_product')


def test_synthesis():
    a = FileArchives.get_any_archive()
    for b in a.bundles():
        if b.proposal_id() != 0:
            for c in b.collections():
                lm = CollectionLabelMaker(c)
                lm.write_xml_to_file('collection.xml')
                lm.create_default_inventory_file(
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
