"""A document template to create a label for a document product."""

from typing import TYPE_CHECKING
from pdart.new_labels.Namespaces import DOCUMENT_PRODUCT_NAMESPACES
from pdart.new_labels.Placeholders import known_placeholder, placeholder_year
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION, PDS4_SHORT_VERSION
from pdart.xml.Templates import combine_fragments_into_fragment, \
    combine_nodes_into_fragment, interpret_document_template, \
    interpret_template

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Tuple
    from xml.dom.minidom import Document
    from pdart.xml.Templates import DocTemplate, FragBuilder, NodeBuilder, \
        NodeBuilderTemplate

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
    <?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
    <Product_Document %s>
    <Identification_Area>
    <logical_identifier><NODE name="product_lid" /></logical_identifier>
    <version_id><NODE name="product_vid" /></version_id>
    <title><NODE name="title" /></title>
    <information_model_version>%s</information_model_version>
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
    </Product_Document>""" %
    (PDS4_SHORT_VERSION, DOCUMENT_PRODUCT_NAMESPACES,
     INFORMATION_MODEL_VERSION))  # type: DocTemplate


# ----------------
# making <Citation_Information>
# ----------------
def _make_proposal_description(bundle_id, proposal_id):
    # type: (unicode, int) -> unicode
    """
    Return a placeholder string the describes the proposal.
    """
    proposal_title = _make_placeholder_proposal_title(bundle_id)
    pi = _make_placeholder_pi_name(bundle_id)
    yr = _make_placeholder_proposal_year(bundle_id)

    return 'This document provides a summary of the observation ' + \
           'plan for HST proposal %d, %s, PI %s, %s.' % \
           (proposal_id, proposal_title, pi, yr)


_citation_information_template = interpret_template("""<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>""")


# type: NodeBuilderTemplate


def make_citation_information(lid, proposal_id):
    # type: (unicode, int) -> NodeBuilder
    """
    Create a ``<Citation_Information />`` element for the proposal ID.
    """
    return _citation_information_template({
        'author_list': _make_placeholder_author_list(lid),
        'publication_year': _make_placeholder_publication_year(lid),
        'description': _make_proposal_description(
            lid,
            proposal_id)
    })


# ----------------
# making <Document_Edition>
# ----------------

_make_file = interpret_template('<file_name><NODE name="file_name" />\
</file_name>')
# type: NodeBuilderTemplate

_make_document_standard_id = interpret_template('<document_standard_id>\
<NODE name="document_standard_id" />\
</document_standard_id>')


# type: NodeBuilderTemplate


def _make_document_file_entry(file_name, document_standard_id):
    # type: (str, str) -> FragBuilder
    return combine_nodes_into_fragment([
        _make_file({'file_name': file_name}),
        _make_document_standard_id({
            'document_standard_id': document_standard_id})
    ])


_make_document_edition = interpret_template(
    """<Document_Edition>
            <edition_name><NODE name="edition_name" /></edition_name>
            <language><NODE name="language" /></language>
            <files><NODE name="files" /></files>
            <Document_File>
            <FRAGMENT name="document_file_entries" />
            </Document_File>
        </Document_Edition>""")


# type: NodeBuilderTemplate


def make_document_edition(edition_name, file_stds):
    # type: (str, List[Tuple[str, str]]) -> NodeBuilder

    fragments = [_make_document_file_entry(file_name, document_standard_id)
                 for (file_name, document_standard_id) in file_stds]
    # type: List[FragBuilder]

    return _make_document_edition({
        'edition_name': edition_name,
        'language': 'English',
        'files': len(file_stds),
        'document_file_entries': combine_fragments_into_fragment(fragments)
    })


def _make_placeholder_author_list(bundle_id):
    return known_placeholder(bundle_id, 'doc product author_list')


def _make_placeholder_proposal_title(bundle_id):
    return known_placeholder(bundle_id, 'doc product proposal_title')


def _make_placeholder_pi_name(bundle_id):
    return known_placeholder(bundle_id, 'doc product pi_name')


def _make_placeholder_publication_year(bundle_id):
    return placeholder_year(bundle_id, 'doc product publication year')


def _make_placeholder_proposal_year(bundle_id):
    return placeholder_year(bundle_id, 'doc product proposal year')
