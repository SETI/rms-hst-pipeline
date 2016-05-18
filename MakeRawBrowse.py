"""
SCRIPT: Run through the archive and generate a browse collection for
each RAW collection, writing them to disk including the collection
inventory and verified label.  If it fails at any point, print the
combined exception as XML to stdout.  Optionally delete the
collections afterwards.  (Does not currently create product labels.)
"""

import os
import os.path
import shutil

from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4.Collection import *
from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *
from pdart.reductions.Reduction import *

import pdart.add_pds_tools
import picmaker


def data_lid_to_browse_lid(lid):
    parts = lid.lid.split(':')
    assert parts[4][0:5] == 'data_'
    parts[4] = 'browse_' + parts[4][5:]
    return LID(':'.join(parts))


class MakeRawBrowseReduction(Reduction):
    def __init__(self):
        self.current_collection = None

    """
    When run on an archive, create browse collections for each RAW
    collection.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        if 'browse_' == lid.collection_id[0:7]:
            return
        collection = Collection(archive, lid)
        if collection.suffix() == 'raw':
            new_lid = data_lid_to_browse_lid(lid)
            browse_collection = Collection(archive, new_lid)
            try:
                os.mkdir(browse_collection.absolute_filepath())
            except OSError:
                pass
            assert os.path.isdir(browse_collection.absolute_filepath())
            self.browse_collection_directory = \
                browse_collection.absolute_filepath()
            get_reduced_products()
            self.browse_collection_directory = None
            make_collection_label_and_inventory(browse_collection)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        product = Product(archive, lid)
        get_reduced_fits_files()
        new_lid = data_lid_to_browse_lid(lid)
        browse_product = Product(archive, new_lid)
        # make_product_label(browse_product, True)

    def reduce_fits_file(self, file, get_reduced_hdus):
        """
        Create a JPG file from the FITS file and put it into the
        parallel browse_ directory.
        """
        basename = os.path.basename(file.full_filepath())
        basename = os.path.splitext(basename)[0] + '.jpg'
        visit = HstFilename(basename).visit()
        target_dir = os.path.join(self.browse_collection_directory,
                                  ('visit_%s' % visit))
        res = picmaker.ImagesToPics([file.full_filepath()],
                                    target_dir,
                                    filter="None",
                                    percentiles=(1, 99))


if __name__ == '__main__':
    archive = get_any_archive()
    reduction = CompositeReduction([LogCollectionsReduction(),
                                    MakeRawBrowseReduction()])
    raise_verbosely(lambda: run_reduction(reduction, archive))
