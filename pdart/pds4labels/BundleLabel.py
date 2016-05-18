import os.path

from pdart.pds4.Bundle import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1"
                xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier><NODE name="lid"/></logical_identifier>
    <version_id>0.1</version_id>
    <title>This bundle contains images obtained from HST Observing Program
<NODE name="proposal_id"/>.</title>
    <information_model_version>1.5.0.0</information_model_version>
    <product_class>Product_Bundle</product_class>
    <NODE name="Citation_Information" />
  </Identification_Area>
  <Bundle>
    <bundle_type>Archive</bundle_type>
  </Bundle>
  <FRAGMENT name="Bundle_Member_Entries"/>
</Product_Bundle>""")

make_bundle_entry_member = interpret_template(
    """<Bundle_Member_Entry>
    <lid_reference><NODE name="lid"/></lid_reference>
    <member_status>Primary</member_status>
    <reference_type>bundle_has_data_collection</reference_type>
</Bundle_Member_Entry>"""
)

placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year>2000</publication_year>
<description>### placeholder for \
citation_information_description ###</description>
</Citation_Information>""")({})


class BundleLabelReduction(Reduction):
    """
    Reduction of a :class:`Bundle` to its PDS4 label as a string.
    """
    def reduce_bundle(self, archive, lid, get_reduced_collections):
        reduced_collections = get_reduced_collections()
        dict = {'lid': interpret_text(str(lid)),
                'proposal_id': interpret_text(str(Bundle(archive,
                                                         lid).proposal_id())),
                'Citation_Information': placeholder_citation_information,
                'Bundle_Member_Entries':
                combine_nodes_into_fragment(reduced_collections)
                }
        label = make_label(dict).toxml()
        label_fp = Bundle(archive, lid).label_filepath()
        with open(label_fp, 'w') as f:
            f.write(label)
        return label

    def reduce_collection(self, archive, lid, get_reduced_products):
        dict = {'lid': interpret_text(str(lid))}
        return make_bundle_entry_member(dict)


def make_bundle_label(bundle, verify):
    """
    Create the label text for this :class:`Bundle`.  If verify is
    True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    label = DefaultReductionRunner().run_bundle(BundleLabelReduction(),
                                                bundle)
    if verify:
        failures = xml_schema_failures(None, label) and \
            schematron_failures(None, label)
    else:
        failures = None
    if failures is None:
        return label
    else:
        raise Exception('Validation errors: ' + failures)
