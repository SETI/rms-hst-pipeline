"""A document template to create a label for a document product."""
from pdart.xml.Pds4Version import *
from pdart.xml.Templates import *

from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple
if TYPE_CHECKING:
    from xml.dom.minidom import Document


make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
    <?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
    <Product_Document
        xmlns="http://pds.nasa.gov/pds4/pds/v1"
        xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
        xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                            http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd">
    <Identification_Area>
    <logical_identifier><NODE name="product_lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title><NODE name="title" /></title>
    <information_model_version>%s</information_model_version>
    <product_class>Product_Document</product_class>
    <NODE name="Citation_Information" />
    </Identification_Area>
    <Reference_List>
        <Internal_Reference>
            <lid_reference><NODE name="bundle_lid" /></lid_reference>
            <reference_type>document_to_investigation</reference_type>
        </Internal_Reference>
    </Reference_List>
    <Document>
        <publication_date><NODE name="publication_date" /></publication_date>
        <NODE name="Document_Edition" />
    </Document>
    </Product_Document>""" %
    (PDS4_SHORT_VERSION, PDS4_SHORT_VERSION, PDS4_LONG_VERSION))
# type: DocTemplate


# ----------------
# making <Citation_Information>
# ----------------
def make_proposal_description(proposal_id):
    # type: (int) -> unicode
    proposal_title = '{{proposal_title}}'  # TODO
    pi = '{{pi_name}}'  # TODO
    yr = '{{proposal_year}}'  # TODO

    return 'This document provides a summary of the observation ' + \
        'plan for HST proposal %d, %s, PI %s, %s.' % \
        (proposal_id, proposal_title, pi, yr)


make_citation_information = interpret_template("""<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>""")
# type: NodeBuilderTemplate


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
