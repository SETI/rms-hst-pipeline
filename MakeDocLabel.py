"""
**SCRIPT:** Build a document product label.
"""
from datetime import date
import os.path
import sqlite3

from pdart.db.DatabaseName import DATABASE_NAME
from pdart.db.TableSchemas import *
from pdart.pds4.Archives import *
from pdart.pds4.Bundle import Bundle
from pdart.pds4.LID import LID
from pdart.pds4labels.DocumentProductLabelDB import *
from pdart.rules.Combinators import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *


def ensure_directory(dir):
    # type: (unicode) -> None
    """Make the directory if it doesn't already exist."""
    try:
        os.mkdir(dir)
    except OSError:
        pass
    assert os.path.isdir(dir), dir


if __name__ == '__main__':
    arch = get_any_archive()
    bundle = Bundle(arch, LID('urn:nasa:pds:hst_14334'))
    proposal_id = bundle.proposal_id()
    product_id = bundle.lid.lid + ':document:phase2'
    bundle_fp = bundle.absolute_filepath()
    document_fp = os.path.join(bundle_fp, 'document')
    ensure_directory(document_fp)
    label_fp = os.path.join(document_fp, 'phase2.xml')

    db_filepath = os.path.join(bundle_fp, DATABASE_NAME)
    with closing(sqlite3.connect(db_filepath)) as conn:
        # We (re-)create the table and populate it with the one needed
        # record.
        conn.execute('DROP TABLE IF EXISTS document_products')
        conn.execute(DOCUMENT_PRODUCTS_SCHEMA)
        conn.execute(DOCUMENT_PRODUCTS_SQL, (product_id,
                                             label_fp,
                                             proposal_id))

        title = 'Summary of the observation plan for HST proposal %d' % \
            proposal_id

        lid = bundle.lid.lid
        prop_id = bundle.proposal_id()

        verify = True
        print make_db_document_product_label(conn, lid, verify)
