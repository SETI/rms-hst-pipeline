import os.path

from pdart.pds4.LID import *
from pdart.pds4.Collection import *
from pdart.pds4.Product import *
from pdart.reductions.CompositeReduction import *
from pdart.xml.Templates import *


make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Browse
   xmlns="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.xsd">
  <Identification_Area>
    <logical_identifier><NODE name="browse_lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This product contains a browse image of a <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>1.5.0.0</information_model_version>
    <product_class>Product_Browse</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>0.1</version_id>
        <description>PDS4 version-in-development of the product</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Reference_List>
    <Internal_Reference>
      <lid_reference><NODE name="data_lid" /></lid_reference>
      <reference_type>browse_to_data</reference_type>
    </Internal_Reference>
  </Reference_List>
  <File_Area_Browse>
    <File>
      <file_name><NODE name="browse_file_name" /></file_name>
    </File>
    <Encoded_Image>
      <offset unit="byte">0</offset>
      <object_length unit="byte"><NODE name="object_length" /></object_length>
      <encoding_standard_id>JPEG</encoding_standard_id>
    </Encoded_Image>
  </File_Area_Browse>
</Product_Browse>""")


def make_browse_lid(lid):
    coll_parts = lid.collection_id.split('_')
    assert coll_parts[0] == 'data'
    coll_parts[0] = 'browse'
    collection_id = '_'.join(coll_parts)

    lid_parts = lid.lid.split(':')
    lid_parts[4] = collection_id
    return LID(':'.join(lid_parts))


class BrowseProductLabelReduction(Reduction):
    """
    Run on "real" product, but produce a label for the browse product.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        # FIXME This test is duplicated in BrowseProductImage.  Can't
        # I move it up into the CompositeReduction that composes these
        # two?
        if collection.prefix() == 'data' and collection.suffix() == 'raw':
            get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # None
        product = Product(archive, lid)
        collection = product.collection()
        bundle = collection.bundle()

        proposal_id = bundle.proposal_id()
        suffix = collection.suffix()
        browse_lid = make_browse_lid(lid)
        browse_product = Product(archive, browse_lid)
        browse_image_file = list(browse_product.files())[0]
        object_length = os.path.getsize(browse_image_file.full_filepath())

        browse_file_name = lid.product_id + '.jpg'

        label = make_label({
                'proposal_id': str(proposal_id),
                'suffix': suffix,
                'browse_lid': str(browse_lid),
                'data_lid': str(lid),
                'browse_file_name': browse_file_name,
                'object_length': str(object_length)
                }).toxml()

        label_fp = browse_product.label_filepath()

        with open(label_fp, 'w') as f:
            f.write(label)
