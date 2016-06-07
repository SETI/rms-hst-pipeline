import os
import os.path

from pdart.pds4.Collection import *
from pdart.pds4.Product import *
from pdart.pds4.HstFilename import *
from pdart.reductions.Reduction import *

import pdart.add_pds_tools
import picmaker

def ensure_directory(dir):
    """Make the directory if it doesn't already exist."""
    try:
        os.mkdir(dir)
    except OSError:
        pass
    assert os.path.isdir(dir), dir


def browse_collection_directory(collection):
    dir = collection.browse_collection().absolute_filepath()
    ensure_directory(dir)
    return dir


def is_raw_data_collection(collection):
    return collection.prefix() == 'data' and collection.suffix() == 'raw'


class BrowseProductImageReduction(Reduction):
    """
    Run on "real" product, but create the browse collection's
    directory and write a browse image into the corresponding visit
    directory.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        if is_raw_data_collection(collection):
            get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        collection = Product(archive, lid).collection()
        if is_raw_data_collection(collection):
            get_reduced_fits_files()

    def reduce_fits_file(self, file, get_reduced_hdus):
        # None
        collection = file.component.collection()
        if is_raw_data_collection(collection):
            try:
                basename = os.path.basename(file.full_filepath())
                basename = os.path.splitext(basename)[0] + '.jpg'
                visit = HstFilename(basename).visit()
                target_dir = os.path.join(
                    browse_collection_directory(collection),
                    ('visit_%s' % visit))

                ensure_directory(target_dir)
                picmaker.ImagesToPics([file.full_filepath()],
                                      target_dir,
                                      filter="None",
                                      percentiles=(1, 99))
            except:
                print 'Exception in', file
                raise
