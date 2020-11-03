"""A document template to create a label for a document product."""

from typing import List, Tuple
import os.path

from pdart.citations import Citation_Information
from pdart.db.BundleDB import BundleDB
from pdart.labels.Namespaces import (
    DOCUMENT_PRODUCT_NAMESPACES,
    HST_XML_MODEL,
    PDS4_XML_MODEL,
)
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import (
    DocTemplate,
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
    combine_fragments_into_fragment,
    combine_nodes_into_fragment,
    interpret_document_template,
    interpret_template,
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
      <lidvid_reference><NODE name="investigation_lidvid" /></lidvid_reference>
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

_make_document_file_node: NodeBuilderTemplate = interpret_template(
    """<Document_File>
<file_name><NODE name="file_name" /></file_name>
<document_standard_id><NODE name="document_standard_id" /></document_standard_id>
</Document_File>"""
)


def _make_document_file(file_name: str, document_standard_id: str) -> FragBuilder:
    return _make_document_file_node(
        {"file_name": file_name, "document_standard_id": document_standard_id}
    )


_make_document_edition: NodeBuilderTemplate = interpret_template(
    """<Document_Edition>
            <edition_name><NODE name="edition_name" /></edition_name>
            <language><NODE name="language" /></language>
            <files><NODE name="files" /></files>
            <FRAGMENT name="document_files" />
        </Document_Edition>"""
)


def _get_document_standard_id(file_basename: str) -> str:
    (_, ext) = os.path.splitext(file_basename)
    return {
        ".apt": "UTF-8 Text",  # it's XML
        ".pdf": "PDF",
        ".pro": "7-Bit ASCII Text",
        ".prop": "7-Bit ASCII Text",
        ".txt": "7-Bit ASCII Text",
    }[ext]


def make_document_edition(edition_name: str, file_basenames: List[str]) -> NodeBuilder:

    nodes: List[NodeBuilder] = [
        _make_document_file(file_basename, _get_document_standard_id(file_basename))
        for file_basename in file_basenames
    ]

    return _make_document_edition(
        {
            "edition_name": edition_name,
            "language": "English",
            "files": len(file_basenames),
            "document_files": combine_nodes_into_fragment(nodes),
        }
    )
