"""
Functionality to build a raw browse product image using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
from os import mkdir
from os.path import isdir

from fs.path import basename, join, splitext

from pdart.pds4.Collection import *
from pdart.pds4.Product import *
from pdart.pds4.HstFilename import *
from pdart.reductions.Reduction import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
import pdart.add_pds_tools
import picmaker

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import AnyStr


def ensure_directory(dir):
    # type: (AnyStr) -> None
    """Make the directory if it doesn't already exist."""
    try:
        mkdir(dir)
    except OSError:
        pass
    assert isdir(dir), dir


def _browse_collection_directory(collection):
    # type: (Collection) -> AnyStr
    """
    Given a :class:`~pdart.pds4.Collection` object, return the
    absolute filepath of its browse collection, creating it if
    necessary.
    """
    dir = collection.browse_collection().absolute_filepath()
    ensure_directory(dir)
    return dir


def is_raw_data_collection(collection):
    # type: (Collection) -> bool
    """
    Return True iff the :class:`~pdart.pds4.Collection` is a raw data
    collection.
    """
    return collection.prefix() == 'data' \
        and collection.suffix() in RAW_SUFFIXES


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
                basename = basename(file.full_filepath())
                basename = splitext(basename)[0] + '.jpg'
                visit = HstFilename(basename).visit()
                target_dir = join(
                    _browse_collection_directory(collection),
                    ('visit_%s' % visit))

                ensure_directory(target_dir)
                picmaker.ImagesToPics([str(file.full_filepath())],
                                      str(target_dir),
                                      filter="None",
                                      percentiles=(1, 99))
            except Exception:
                print 'Exception in', file
                raise
