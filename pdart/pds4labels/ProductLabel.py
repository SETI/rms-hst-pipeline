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
image obtained the HST Observing Program <PARAM name="proposal_id" />\
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
      <name><PARAM name="Investigation_Area_name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><PARAM name="investigation_lidvid" />\
        </lidvid_reference>
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


# PLACEHOLDER
time_coordinates = interpret_template("""<Time_Coordinates>
      <start_date_time>2000-01-02Z</start_date_time>
      <stop_date_time>2000-01-02Z</stop_date_time>
    </Time_Coordinates>""")

# PLACEHOLDER
observing_system = interpret_template("""<Observing_System>
      <name>Hubble Space Telescope Advanced Camera for Surveys</name>
      <Observing_System_Component>
        <name>Hubble Space Telescope</name>
        <type>Spacecraft</type>
        <Internal_Reference>
          <lid_reference>urn:nasa:pds:context:instrument_host:spacecraft.hst</lid_reference>
          <reference_type>is_instrument_host</reference_type>
        </Internal_Reference>
      </Observing_System_Component>
      <Observing_System_Component>
        <name>Advanced Camera for Surveys</name>
        <type>Instrument</type>
        <Internal_Reference>
          <lid_reference>urn:nasa:pds:context:instrument:insthost.acs.hst</lid_reference>
          <reference_type>is_instrument</reference_type>
        </Internal_Reference>
      </Observing_System_Component>
    </Observing_System>""")

# PLACEHOLDER
target_identification = interpret_template("""<Target_Identification>
      <name>Magrathea</name>
      <type>Planet</type>
      <description>Home of Slartibartfast (placeholder)</description>
      <Internal_Reference>
        <lid_reference>urn:nasa:pds:context:target:planet.magrathea</lid_reference>
        <reference_type>data_to_target</reference_type>
      </Internal_Reference>
    </Target_Identification>""")


def mk_Investigation_Area_name(proposal_id):
    return 'HST observing program %d' % proposal_id


def mk_Investigation_Area_lidvid(proposal_id):
    return 'urn:nasa:pds:context:investigation:investigation.hst_%05d::1.0' % \
        proposal_id


hdu_contents = interpret_template("""<Header>
<local_identifier></local_identifier>
<offset unit="byte"><PARAM name="offset"/></offset>
<object_length unit="byte"><PARAM name="object_length"/></object_length>
<parsing_standard_id>FITS 3.0</parsing_standard_id>
<description>Global FITS Header</description>
</Header>""")


class ProductLabelReduction(Reduction):
    """
    Reduction of a :class:`Product` to its PDS4 label as a string.
    """
    def reduce_product(self, archive, lid, get_reduced_fits_files):
        file_contents = get_reduced_fits_files()
        # file_contents :: [Doc -> [Node]]

        def file_contents_(doc):
            res = []
            for fc in file_contents:
                res.extend(fc(doc))
            return res
        # file_contents_ :: Doc -> [Node]
        assert is_doc_to_list_of_nodes_function(file_contents_)

        product = Product(archive, lid)
        suffix = product.collection().suffix()
        proposal_id = product.bundle().proposal_id()
        file_name = os.path.basename(product.absolute_filepath())

        dict = {'lid': interpret_text(str(lid)),
                'suffix': interpret_text(suffix.upper()),
                'proposal_id': interpret_text(str(proposal_id)),
                'Time_Coordinates': time_coordinates({}),
                'Investigation_Area_name':
                    interpret_text(mk_Investigation_Area_name(proposal_id)),
                'investigation_lidvid':
                    interpret_text(mk_Investigation_Area_lidvid(proposal_id)),
                'Observing_System': observing_system({}),
                'Target_Identification': target_identification({}),
                'file_name': interpret_text(file_name),
                'file_contents': file_contents_
                }
        return make_label(dict).toxml()

    def reduce_fits_file(self, file, get_reduced_hdus):
        reduced_hdus = get_reduced_hdus()
        assert is_list_of_doc_to_node_functions(reduced_hdus)
        res = combine_multiple_nodes(reduced_hdus)
        assert is_doc_to_list_of_nodes_function(res)
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        offset = str('123')
        object_length = str('456')
        res = hdu_contents({'offset': offset,
                            'object_length': object_length})
        assert is_doc_to_node_function(res)
        return res


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
