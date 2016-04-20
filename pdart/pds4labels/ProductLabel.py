import os.path

from pdart.pds4.Product import *
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
    <logical_identifier><PARAM name="lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <PARAM name="suffix" /> \
image obtained the HST Observing Program <PARAM name="project_id" />\
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
    <PARAM name="Time_Coordinates" />
    <Investigation_Area>
      <name><PARAM name="name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><PARAM name="investigation_lidvid" /></lidvid_reference>
        <reference_type>data_to_investigation</reference_type>
      </Internal_Reference>
    </Investigation_Area>
    <PARAM name="Observing_System" />
    <PARAM name="Target_Identification" />
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name><PARAM name="file_name" /></file_name>
      <PARAM name="file_contents" />
    </File>
  </File_Area_Observational>
</Product_Observational>""")


class ProductLabelReduction(Reduction):
    """
    Reduction of a :class:`Product` to its PDS4 label as a string.
    """
    def reduce_product(self, archive, lid, get_reduced_products):
        product = Product(archive, lid)

        dict = {'lid': interpret_text(str(lid)),
                'suffix': interpret_text('***suffix***'),
                'project_id': interpret_text('***project_id***'),
                'Time_Coordinates': interpret_text('***Time_Coordinates***'),
                'name': interpret_text('***name***'),
                'investigation_lidvid': interpret_text('***investigation_lidvid***'),
                'Observing_System': interpret_text('***Observing_System***'),
                'Target_Identification': interpret_text('***Target_Identification***'),
                'file_name': interpret_text(
                os.path.basename(product.absolute_filepath())),
                'file_contents': interpret_text('***file_contents***'),
                }
        return make_label(dict).toxml()


def make_product_label(product, verify):
    """
    Create the label text for this :class:`Product`.  If verify is
    True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    label = ReductionRunner().run_product(ProductLabelReduction(),
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
