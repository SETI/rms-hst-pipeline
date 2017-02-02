"""
**SCRIPT:** Run through the archive and delete all browse collections.
"""

from contextlib import closing
import sqlite3
import shutil

from pdart.pds4.Archives import *
from pdart.pds4.Bundle import Bundle
from pdart.pds4.Collection import *
from pdart.pds4labels.DBCalls import *
from pdart.reductions.Reduction import *
from pdart.rules.Combinators import *


class DeleteRawBrowseReduction(Reduction):
    """
    Summarizes the archive into ``None``; as a side-effect, deletes
    all browse collections.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()
        # remove from the database
        bundle = Bundle(archive, lid)
        with closing(open_bundle_database(bundle)) as conn:
            with closing(conn.cursor()) as cursor:
                delete_browse_products_and_collections(cursor)

    def reduce_collection(self, archive, lid, get_reduced_products):
        if 'browse_' == lid.collection_id[0:7]:
            collection = Collection(archive, lid)
            shutil.rmtree(collection.absolute_filepath())


def _check_deletion(archive):
    # type: (Archive) -> None

    # assert they're not in the filesystem
    for collection in archive.collections():
        assert collection.prefix() is not "browse"

    # assert they're not in the database
    for bundle in archive.bundles():
        with closing(open_bundle_database(bundle)) as conn:
            with closing(conn.cursor()) as cursor:
                colls = list(get_all_browse_collections(cursor))
                assert not colls, 'Browse collections %s not deleted' % colls
                prods = list(get_all_browse_products(cursor))
                assert not prods, 'Browse products %s not deleted' % prods


if __name__ == '__main__':
    archive = get_any_archive()
    reduction = DeleteRawBrowseReduction()
    raise_verbosely(lambda: run_reduction(reduction, archive))
    _check_deletion(archive)
