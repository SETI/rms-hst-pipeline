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
from pdart.rules.Combinators import *
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

from typing import Any, Dict, Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    import xml.dom.minidom

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
    <logical_identifier><NODE name="lid" /></logical_identifier>
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
            <lid_reference>urn:nasa:pds:hst_05167</lid_reference>
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
                <file_name>phase2.txt</file_name>
                <document_standard_id>7-Bit ASCII Text</document_standard_id>
            </Document_File>
        </Document_Edition>
    </Document>
    </Product_Document>""" %
    (PDS4_SHORT_VERSION, PDS4_SHORT_VERSION, PDS4_LONG_VERSION))
# type: Callable[[Dict[str, Any]], Document]


def get_all_good_bundle_products(cursor, bundle):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute("""SELECT product FROM products
                                  WHERE collection IN
                                  (SELECT collection from collections
                                   WHERE bundle=?) /*EXCEPT
                                  SELECT product FROM bad_fits_files*/""",
                               (bundle,)))


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

    # TODO I need to look at the FITS files for more data.
    # Unfortunately, all the FITS files for this bundle are bad.  So I
    # need to try another one.

    arch = get_any_archive()
    bundle = Bundle(arch, LID('urn:nasa:pds:hst_05167'))
    database_fp = os.path.join(bundle.absolute_filepath(),
                               DATABASE_NAME)
    with closing(sqlite3.connect(database_fp)) as conn:
        with closing(conn.cursor()) as cursor:
            for p in get_all_good_bundle_products(cursor, bundle.lid.lid):
                print p

    # /Users/spaceman/Desktop/Archive/hst_05167/document/phase2.txt
    proposal = 5167

    title = 'Jovian Auroral Ly-Alpha Profile:Cycle 4'
    pi = 'L. Trafton'
    yr = 1995
    description = 'This document provides a summary of the observation ' + \
        'plan for HST proposal %d, %s, PI %s, %d.' % \
        (proposal, title, pi, yr)

    title = 'Summary of the observation plan for HST proposal %d' % proposal
    today = date.today().isoformat()

    label = make_label({
            'lid': 'urn:nasa:pds:hst_05167:document:phase2',
            'title': title,
            'description': description,
            'publication_year': 2000,  # TODO
            'publication_date': today,
            'author_list': 'TODO: Galilei, Galileo',
            'edition_name': 'TODO: use the product version',
            'language': 'English'
            })
    pretty_label = pretty_print(label.toxml())
    print pretty_label

    raise_verbosely(lambda: run(pretty_label))
