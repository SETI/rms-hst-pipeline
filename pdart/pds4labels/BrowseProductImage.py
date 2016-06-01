import os
import os.path

from pdart.pds4.Collection import *
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


class BrowseProductImageReduction(Reduction):
    """
    Run on "real" product, but create the browse collection's
    directory and write a browse image into the corresponding visit
    directory.
    """
    def __init__(self):
        self.browse_collection_directory = None

    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        if collection.prefix() == 'data' and collection.suffix() == 'raw':
            browse_collection = collection.browse_collection()
            self.browse_collection_directory = \
                browse_collection.absolute_filepath()
            ensure_directory(self.browse_collection_directory)

            get_reduced_products()
            self.browse_collection_directory = None

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files()

    def reduce_fits_file(self, file, get_reduced_hdus):
        # None
        try:
            basename = os.path.basename(file.full_filepath())
            basename = os.path.splitext(basename)[0] + '.jpg'
            visit = HstFilename(basename).visit()
            target_dir = os.path.join(self.browse_collection_directory,
                                      ('visit_%s' % visit))

            ensure_directory(target_dir)
            picmaker.ImagesToPics([file.full_filepath()],
                                  target_dir,
                                  filter="None",
                                  percentiles=(1, 99))
        except:
            print 'Exception in', file
            raise
