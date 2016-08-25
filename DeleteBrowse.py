"""
**SCRIPT:** Run through the archive and delete all browse collections.
"""

import shutil

from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4.Collection import *
from pdart.reductions.Reduction import *


class DeleteRawBrowseReduction(Reduction):
    """
    Summarizes the archive into ``None``; as a side-effect, deletes
    all browse collections.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        if 'browse_' == lid.collection_id[0:7]:
            collection = Collection(archive, lid)
            shutil.rmtree(collection.absolute_filepath())

if __name__ == '__main__':
    archive = get_any_archive()
    reduction = DeleteRawBrowseReduction()
    raise_verbosely(lambda: run_reduction(reduction, archive))
