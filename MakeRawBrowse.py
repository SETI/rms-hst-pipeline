"""
SCRIPT: Run through the archive and generate a browse collection for
each RAW collection, writing them to disk including the collection
inventory and verified label.  If it fails at any point, print the
combined exception as XML to stdout.
"""
from contextlib import closing
import sqlite3

from pdart.db.CreateDatabase import *
from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BrowseProductImage import *
from pdart.pds4labels.BrowseProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *

import pdart.add_pds_tools
import picmaker


class MakeRawBrowseReduction(CompositeReduction):
    """
    When run on an archive, create browse collections for each RAW
    collection.
    """
    def __init__(self):
        CompositeReduction.__init__(self,
                                    [BrowseProductImageReduction(),
                                     BrowseProductLabelReduction()])


def get_conn():
    return sqlite3.connect('/Users/spaceman/Desktop/Archive/archive.spike.db')


if __name__ == '__main__':
    USE_DATABASE = True
    CREATE_DATABASE = False

    archive = get_any_archive()

    if USE_DATABASE:
        with closing(get_conn()) as conn:
            if CREATE_DATABASE:
                create_database(conn, archive)
            make_db_browse_product_images(conn, archive)
            make_db_browse_product_labels(conn, archive)
    else:
        reduction = CompositeReduction([LogCollectionsReduction(),
                                        MakeRawBrowseReduction()])
        raise_verbosely(lambda: run_reduction(reduction, archive))
