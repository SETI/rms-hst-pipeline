from contextlib import closing
import io
import sys

from pdart.pds4.Collection import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Collection xmlns="http://pds.nasa.gov/pds4/pds/v1"
                    xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier><NODE name="lid"/></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <NODE name="suffix"/> \
images obtained from HST Observing Program \
<NODE name="proposal_id"/>.</title>
    <information_model_version>1.6.0.0</information_model_version>
    <product_class>Product_Collection</product_class>

    <NODE name="Citation_Information" />
  </Identification_Area>
  <Collection>
    <collection_type>Data</collection_type>
  </Collection>
  <File_Area_Inventory>
    <File>
      <file_name><NODE name="inventory_name" /></file_name>
    </File>
    <Inventory>
      <offset unit="byte">0</offset>
      <parsing_standard_id>PDS DSV 1</parsing_standard_id>
      <records>1</records>
      <record_delimiter>Carriage-Return Line-Feed</record_delimiter>
      <field_delimiter>Comma</field_delimiter>
      <Record_Delimited>
        <fields>2</fields>
        <groups>0</groups>
        <Field_Delimited>
          <name>Member Status</name>
          <field_number>1</field_number>
          <data_type>ASCII_String</data_type>
          <maximum_field_length unit="byte">1</maximum_field_length>
        </Field_Delimited>
        <Field_Delimited>
          <name>LIDVID_LID</name>
          <field_number>2</field_number>
          <data_type>ASCII_LIDVID_LID</data_type>
          <maximum_field_length unit="byte">255</maximum_field_length>
        </Field_Delimited>
      </Record_Delimited>
      <reference_type>inventory_has_member_product</reference_type>
    </Inventory>
  </File_Area_Inventory>
</Product_Collection>""")

placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year>2000</publication_year>
<description>### placeholder for \
citation_information_description ###</description>
</Citation_Information>""")({})


class CollectionLabelReduction(Reduction):
    def __init__(self, verify=False):
        Reduction.__init__(self)
        self.verify = verify

    """
    Reduction of a :class:`Collection` to its PDS4 label as a string.
    """
    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        suffix = collection.suffix()
        proposal_id = collection.bundle().proposal_id()
        inventory_name = collection.inventory_name()

        dict = {'lid': interpret_text(str(lid)),
                'suffix': interpret_text(suffix.upper()),
                'proposal_id': interpret_text(str(proposal_id)),
                'Citation_Information': placeholder_citation_information,
                'inventory_name': interpret_text(inventory_name)
                }
        label = make_label(dict).toxml()
        label_fp = Collection(archive, lid).label_filepath()

        inventory_filepath = collection.inventory_filepath()
        with io.open(inventory_filepath, 'w', newline='') as f:
            f.write(make_collection_inventory(collection))

        with open(label_fp, 'w') as f:
            f.write(label)

        if self.verify:
            verify_label_or_throw(label)

        return label


def make_collection_label(collection, verify):
    """
    Create the label text for this :class:`Collection`.  If verify is
    True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    return DefaultReductionRunner().run_collection(
        CollectionLabelReduction(verify), collection)


def make_collection_inventory(collection):
    lines = [u'P,%s\r\n' % str(product.lid)
             for product in collection.products()]
    return ''.join(lines)


def make_db_collection_inventory(conn, collection_lid):
    with closing(conn.cursor()) as cursor:
        lines = [u'P,%s\r\n' % str(product)
                 for (product,) in cursor.execute(
                'SELECT product FROM products WHERE collection=?',
                (collection_lid,))]
    return ''.join(lines)


def make_collection_label_and_inventory(collection):
    """
    Create the label and inventory for a collection and write to the disk.
    """
    inventory_filepath = collection.inventory_filepath()
    with io.open(inventory_filepath, 'w', newline='') as f:
        f.write(make_collection_inventory(collection))

    label_filepath = collection.label_filepath()
    with io.open(label_filepath, 'w') as f:
        f.write(make_collection_label(collection, True))


def make_db_collection_label_and_inventory(conn, lid, verify):
    """
    Create the label and inventory for the collection having this
    :class:'LID' using the database collection.  Write them to disk.
    If verify is True, verify the label against its XML and Schematron
    schemas.  Raise an exception if either fails.
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            """SELECT label_filepath, bundle, suffix,
               inventory_name, inventory_filepath
               FROM collections WHERE collection=?""", (lid,))
        (label_fp, bundle, suffix, inventory_name, inventory_filepath) = \
            cursor.fetchone()

        cursor.execute(
            'SELECT proposal_id FROM bundles where bundle=?', (bundle,))
        (proposal_id,) = cursor.fetchone()

    label = make_label({
            'lid': lid,
            'suffix': suffix,
            'proposal_id': str(proposal_id),
            'Citation_Information': placeholder_citation_information,
            'inventory_name': inventory_name
    }).toxml()

    with open(label_fp, 'w') as f:
        f.write(label)

    if verify:
        verify_label_or_throw(label)

    with io.open(inventory_filepath, 'w', newline='') as f:
        f.write(make_db_collection_inventory(conn, lid))

    print 'collection label and inventory for', lid
