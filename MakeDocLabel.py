"""
**SCRIPT:** Build a document product label.
"""
from contextlib import closing
from datetime import date
import sqlite3
from xml.dom.minidom import Document

from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Bundle import Bundle
from pdart.pds4.LID import LID
from pdart.pds4labels.DBCalls import *
from pdart.rules.Combinators import *
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

from typing import Any, Dict, Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    import xml.dom.minidom

    _UADict = Dict[str, Any]
    NodeBuilder = Callable[[xml.dom.minidom.Document], xml.dom.minidom.Text]

_make_file = interpret_template('<file_name><NODE name="file_name" />\
</file_name>')
# type: Callable[[_UADict], NodeBuilder]

_make_document_standard_id = interpret_template('<document_standard_id>\
<NODE name="document_standard_id" />\
</document_standard_id>')
# type: Callable[[_UADict], NodeBuilder]


def _make_document_file_entry(file_name, document_standard_id):
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
# type: Callable[[_UADict], NodeBuilder]


def make_document_edition(edition_name, file_stds):
    # type: (str, List[Tuple[str, str]]) -> NodeBuilder
    nodes = [_make_document_file_entry(file_name, document_standard_id)
             for (file_name, document_standard_id) in file_stds]
    # type: List[NodeBuilder]

    return _make_document_edition({
            'edition_name': edition_name,
            'language': 'English',
            'files': len(file_stds),
            'document_file_entries': combine_nodes_into_fragment(nodes)})


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
    <Citation_Information>
        <author_list><NODE name="author_list" /></author_list>
        <publication_year><NODE name="publication_year" /></publication_year>
        <description><NODE name="description" /></description>
    </Citation_Information>
    </Identification_Area>
    <Reference_List>
        <Internal_Reference>
            <lid_reference><NODE name="bundle_lid" /></lid_reference>
            <reference_type>document_to_investigation</reference_type>
        </Internal_Reference>
    </Reference_List>
    <Document>
        <publication_date><NODE name="publication_date" /></publication_date>
        <Document_Edition>
            <edition_name><NODE name="edition_name" /></edition_name>
            <language><NODE name="language" /></language>
            <files>1</files>
            <Document_File>
                <FRAGMENT name="document_file_entry" />
            </Document_File>
        </Document_Edition>
    </Document>
    </Product_Document>""" %
    (PDS4_SHORT_VERSION, PDS4_SHORT_VERSION, PDS4_LONG_VERSION))
# type: Callable[[Dict[str, Any]], Document]


def make_proposal_description(proposal_id):
    # type: (int) -> unicode
    proposal_title = '{{proposal_title}}'  # TODO
    pi = '{{pi_name}}'  # TODOnnnn
    yr = '{{proposal_year}}'  # TODO

    return 'This document provides a summary of the observation ' + \
        'plan for HST proposal %d, %s, PI %s, %s.' % \
        (proposal_id, proposal_title, pi, yr)


if __name__ == '__main__':
    def run(label):
        # type: (str) -> None
        failures = xml_schema_failures(None, label)
        if failures is not None:
            print label
            raise Exception('XML schema validation errors: ' + failures)
        failures = schematron_failures(None, label)
        if failures is not None:
            print label
            raise Exception('Schematron validation errors: ' + failures)

    arch = get_any_archive()
    bundle = Bundle(arch, LID('urn:nasa:pds:hst_14334'))

    proposal_id = bundle.proposal_id()
    description = make_proposal_description(proposal_id)

    title = 'Summary of the observation plan for HST proposal %d' % proposal_id

    label = make_label({
            'bundle_lid': bundle.lid.lid,
            'product_lid': bundle.lid.lid + ':document:phase2',
            'title': title,
            'description': description,
            'publication_year': 2000,  # TODO
            'publication_date': date.today().isoformat(),
            'author_list': '{{author_list}}',  # TODO
            'edition_name': '0.0',  # TODO
            'language': 'English',
            'document_file_entry': _make_document_file_entry(
                'phase2.txt',
                '7-Bit ASCII Text')
            })
    pretty_label = pretty_print(label.toxml())
    print pretty_label

    raise_verbosely(lambda: run(pretty_label))
