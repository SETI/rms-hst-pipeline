"""A document template to create a label for a document product."""

from pdart.labels.Namespaces import (
    DOCUMENT_PRODUCT_NAMESPACES,
    HST_XML_MODEL,
    PDS4_XML_MODEL,
)
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION, PDS4_SHORT_VERSION
from pdart.xml.Templates import (
    combine_fragments_into_fragment,
    combine_nodes_into_fragment,
    interpret_document_template,
    interpret_template,
)

from typing import Any, Callable, Dict, List, Tuple
from pdart.citations import Citation_Information
from xml.dom.minidom import Document
from pdart.db.BundleDB import BundleDB
from pdart.xml.Templates import (
    DocTemplate,
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
)

make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
{HST_XML_MODEL}
<Product_Document {DOCUMENT_PRODUCT_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="product_lid" /></logical_identifier>
    <version_id><NODE name="product_vid" /></version_id>
    <title><NODE name="title" /></title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Document</product_class>
    <NODE name="Citation_Information" />
  </Identification_Area>
  <Reference_List>
    <Internal_Reference>
      <lidvid_reference><NODE name="bundle_lidvid" /></lidvid_reference>
      <reference_type>document_to_investigation</reference_type>
    </Internal_Reference>
  </Reference_List>
  <Document>
    <publication_date><NODE name="publication_date" /></publication_date>
    <NODE name="Document_Edition" />
  </Document>
</Product_Document>"""
)


# ----------------
# making <Citation_Information>
# ----------------
def _make_proposal_description(
    proposal_id: int, proposal_title: str, pi_name: str, publication_year: str
) -> str:
    """
    Return a string the describes the proposal.
     """
    return (
        "This document provides a summary of the observation "
        f"plan for HST proposal {proposal_id}, {proposal_title}, "
        f"PI {pi_name}, {publication_year}."
    )


_citation_information_template: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>"""
)


def make_doc_citation_information(info: Citation_Information) -> NodeBuilder:
    return _citation_information_template(
        {
            "author_list": info.author_list,
            "publication_year": info.publication_year,
            "description": info.description,
        }
    )


def make_doc_citation_information2(
    bundle_db: BundleDB, bundle_lid: str, proposal_id: int
) -> NodeBuilder:
    """
    Create a ``<Citation_Information />`` element for the proposal ID.
    """
    proposal_info = bundle_db.get_proposal_info(bundle_lid)
    return _citation_information_template(
        {
            "author_list": proposal_info.author_list,
            "publication_year": proposal_info.publication_year,
            "description": _make_proposal_description(
                proposal_id,
                proposal_info.proposal_title,
                proposal_info.pi_name,
                proposal_info.proposal_year,
            ),
        }
    )


# ----------------
# making <Document_Edition>
# ----------------

_make_file: NodeBuilderTemplate = interpret_template(
    '<file_name><NODE name="file_name" />\
</file_name>'
)

_make_document_standard_id: NodeBuilderTemplate = interpret_template(
    '<document_standard_id>\
<NODE name="document_standard_id" />\
</document_standard_id>'
)


def _make_document_file_entry(file_name: str, document_standard_id: str) -> FragBuilder:
    return combine_nodes_into_fragment(
        [
            _make_file({"file_name": file_name}),
            _make_document_standard_id({"document_standard_id": document_standard_id}),
        ]
    )


_make_document_edition: NodeBuilderTemplate = interpret_template(
    """<Document_Edition>
            <edition_name><NODE name="edition_name" /></edition_name>
            <language><NODE name="language" /></language>
            <files><NODE name="files" /></files>
            <Document_File>
            <FRAGMENT name="document_file_entries" />
            </Document_File>
        </Document_Edition>"""
)


def make_document_edition(
    edition_name: str, file_stds: List[Tuple[str, str]]
) -> NodeBuilder:

    fragments: List[FragBuilder] = [
        _make_document_file_entry(file_name, document_standard_id)
        for (file_name, document_standard_id) in file_stds
    ]

    return _make_document_edition(
        {
            "edition_name": edition_name,
            "language": "English",
            "files": len(file_stds),
            "document_file_entries": combine_fragments_into_fragment(fragments),
        }
    )
