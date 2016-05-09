import os.path

from pdart.pds4.Product import *
from pdart.pds4labels.FileContentsLabelReduction import *
from pdart.pds4labels.ObservingSystem import *
from pdart.pds4labels.TargetIdentificationLabelReduction import *
from pdart.pds4labels.TimeCoordinatesLabelReduction import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Observational
   xmlns="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.xsd">
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>1.5.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>0.1</version_id>
        <description>PDS4 version-in-development of the product</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Observation_Area>
    <NODE name="Time_Coordinates" />
    <Investigation_Area>
      <name><NODE name="Investigation_Area_name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><NODE name="investigation_lidvid" />\
</lidvid_reference>
        <reference_type>data_to_investigation</reference_type>
      </Internal_Reference>
    </Investigation_Area>
    <NODE name="Observing_System" />
    <NODE name="Target_Identification" />
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name><NODE name="file_name" /></file_name>
      <FRAGMENT name="file_contents" />
    </File>
  </File_Area_Observational>
</Product_Observational>""")


def mk_Investigation_Area_name(proposal_id):
    return 'HST observing program %d' % proposal_id


def mk_Investigation_Area_lidvid(proposal_id):
    return 'urn:nasa:pds:context:investigation:investigation.hst_%05d::1.0' % \
        proposal_id


class ProductLabelReduction(CompositeReduction):
    def __init__(self):
        CompositeReduction.__init__(self,
                                    [FileContentsLabelReduction(),
                                     TargetIdentificationLabelReduction(),
                                     TimeCoordinatesLabelReduction()])

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        product = Product(archive, lid)
        file_name = os.path.basename(product.absolute_filepath())

        instrument = product.collection().instrument()
        suffix = product.collection().suffix()

        proposal_id = product.bundle().proposal_id()
        investigation_area_name = mk_Investigation_Area_name(proposal_id)
        investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)

        reduced_fits_file = get_reduced_fits_files()[0]
        (file_contents,
         target_identification,
         time_coordinates) = reduced_fits_file

        return make_label({
                'lid': str(lid),
                'suffix': suffix.upper(),
                'proposal_id': str(proposal_id),
                'Investigation_Area_name': investigation_area_name,
                'investigation_lidvid': investigation_area_lidvid,
                'Observing_System': observing_system(instrument),
                'file_name': file_name,
                'Time_Coordinates': time_coordinates,
                'Target_Identification': target_identification,
                'file_contents': file_contents
                }).toxml()


def make_product_label(product, verify):
    """
    Create the label text for this :class:`Product`.  If verify is
    True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    label = DefaultReductionRunner().run_product(ProductLabelReduction(),
                                                 product)
    if verify:
        failures = xml_schema_failures(None, label) and \
            schematron_failures(None, label)
    else:
        failures = None
    if failures is None:
        return label
    else:
        raise Exception('Validation errors: ' + failures)
