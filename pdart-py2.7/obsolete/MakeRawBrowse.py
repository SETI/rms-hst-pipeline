"""
**SCRIPT:** Run through the archive and generate a browse collection
for each raw (RAW, FLT, C0F, C0M) collection, writing them to disk
including the collection inventory and verified label.  If it fails at
any point, print the combined exception as XML to stdout.
"""
from contextlib import closing
import os.path
import sqlite3

from pdart.db.CreateDatabase import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BrowseProductImage import *
from pdart.pds4labels.BrowseProductLabel import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *
from pdart.rules.Combinators import *

import pdart.add_pds_tools
import picmaker


class _MakeRawBrowseReduction(CompositeReduction):
    """
    When run on an archive, create browse collections for each RAW
    collection.
    """
    def __init__(self):
        CompositeReduction.__init__(self,
                                    [BrowseProductImageReduction(),
                                     BrowseProductLabelReduction()])


def _get_conn():
    # type: () -> sqlite3.Connection
    return open_archive_database(get_any_archive())


def _check_raw_browse(archive):
    # type: (Archive) -> None
    for b in archive.bundles():
        collections = list(b.collections())
        for c in collections:
            if c.prefix == 'data' and c.suffix() in RAW_SUFFIXES:
                assert c.browse_collection() in collections


if __name__ == '__main__':
    USE_DATABASE = True
    CREATE_DATABASE = False

    archive = get_any_archive()

    if USE_DATABASE:
        with closing(_get_conn()) as conn:
            if CREATE_DATABASE:
                ArchiveDatabaseCreator(conn, archive).create()
            make_db_browse_product_images(archive, conn)
            make_db_browse_product_labels(archive, conn)
    else:
        reduction = CompositeReduction([LogCollectionsReduction(),
                                        _MakeRawBrowseReduction()])
        raise_verbosely(lambda: run_reduction(reduction, archive))

    _check_raw_browse(archive)