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
from pdart.pds4labels.DocumentProductLabelXml import *
from pdart.rules.Combinators import *
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *
from pdart.xml.Templates import _DOC

from typing import Any, Dict, Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    import xml.dom.minidom

    _UADict = Dict[str, Any]
    NodeBuilder = Callable[[xml.dom.minidom.Document], xml.dom.minidom.Text]

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

    title = 'Summary of the observation plan for HST proposal %d' % proposal_id

    label = make_label({
            'bundle_lid': bundle.lid.lid,
            'product_lid': bundle.lid.lid + ':document:phase2',
            'title': title,
            'publication_date': date.today().isoformat(),
            'Citation_Information': make_citation_information({
                    'author_list': '{{author_list}}',  # TODO
                    'publication_year': 2000,  # TODO
                    'description': make_proposal_description(
                        bundle.proposal_id())
                    }),
            'Document_Edition': make_document_edition(
                '0.0',
                [('phase2.txt', '7-Bit ASCII Text')])
            })
    pretty_label = pretty_print(label.toxml())
    print pretty_label

    raise_verbosely(lambda: run(pretty_label))
