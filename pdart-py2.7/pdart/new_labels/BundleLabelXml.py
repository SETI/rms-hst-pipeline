"""Templates to create a label for a bundle."""
from typing import TYPE_CHECKING

from pdart.new_labels.Namespaces import BUNDLE_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION, PDS4_SHORT_VERSION
from pdart.xml.Templates import interpret_document_template, interpret_template

if TYPE_CHECKING:
    from pdart.xml.Templates import DocTemplate, NodeBuilderTemplate

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
%s
<Product_Bundle %s>
  <Identification_Area>
    <logical_identifier><NODE name="bundle_lid"/></logical_identifier>
    <version_id><NODE name="bundle_vid"/></version_id>
    <title>This bundle contains images obtained from HST Observing Program
<NODE name="proposal_id"/>.</title>
    <information_model_version>%s</information_model_version>
    <product_class>Product_Bundle</product_class>
    <NODE name="Citation_Information" />
  </Identification_Area>
  <Bundle>
    <bundle_type>Archive</bundle_type>
  </Bundle>
  <FRAGMENT name="Bundle_Member_Entries"/>
</Product_Bundle>""" % (PDS4_XML_MODEL, BUNDLE_NAMESPACES,
                        INFORMATION_MODEL_VERSION))  # type: DocTemplate
"""
An interpreted document template to create a bundle label.
"""

make_bundle_entry_member = interpret_template(
    """<Bundle_Member_Entry>
    <lidvid_reference><NODE name="collection_lidvid"/></lidvid_reference>
    <member_status>Primary</member_status>
    <reference_type>bundle_has_data_collection</reference_type>
</Bundle_Member_Entry>"""
)  # type: NodeBuilderTemplate
"""
An interpreted fragment template to create a ``<Bundle_Member_Entry
/>`` XML element.
"""
