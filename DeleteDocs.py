"""
**SCRIPT:** Run through the archive and delete all documentation
collections.
"""

import os.path
import shutil

from pdart.pds4.Archives import *
from pdart.pds4.Bundle import *
from pdart.reductions.Reduction import *
from pdart.rules.Combinators import *


class _DeleteDocCollectionsReduction(Reduction):
    """
    Summarizes the archive into ``None``; as a side-effect, deletes
    all document collections.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        bundle = Bundle(archive, lid)
        document_dir = os.path.join(bundle.absolute_filepath(), 'document')
        if os.path.isdir(document_dir):
            print 'Deleting', document_dir
            shutil.rmtree(document_dir)

if __name__ == '__main__':
    archive = get_any_archive()
    reduction = _DeleteDocCollectionsReduction()
    raise_verbosely(lambda: run_reduction(reduction, archive))
