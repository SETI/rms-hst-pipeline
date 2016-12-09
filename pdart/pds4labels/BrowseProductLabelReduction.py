"""
Functionality to build a RAW browse product label using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
import os.path

from pdart.pds4.LID import *
from pdart.pds4.Collection import *
from pdart.pds4.Product import *
from pdart.pds4labels.BrowseProductLabelXml import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.reductions.CompositeReduction import *
from pdart.xml.Pretty import *


def _is_raw_data_collection(collection):
    # type: (Collection) -> bool
    return collection.prefix() == 'data' \
        and collection.suffix() in RAW_SUFFIXES


class BrowseProductLabelReduction(Reduction):
    """
    Run on "real" product, but produce a label for the browse product.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        if _is_raw_data_collection(collection):
            get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # None
        product = Product(archive, lid)
        collection = product.collection()
        if _is_raw_data_collection(collection):
            suffix = collection.suffix()

            bundle = collection.bundle()
            proposal_id = bundle.proposal_id()

            browse_product = product.browse_product()
            browse_image_file = list(browse_product.files())[0]
            object_length = os.path.getsize(browse_image_file.full_filepath())

            browse_file_name = lid.product_id + '.jpg'

            label = make_label({
                    'proposal_id': str(proposal_id),
                    'suffix': suffix,
                    'browse_lid': str(browse_product.lid),
                    'data_lid': str(lid),
                    'browse_file_name': browse_file_name,
                    'object_length': str(object_length)
                    }).toxml()

            label_fp = browse_product.label_filepath()

            with open(label_fp, 'w') as f:
                f.write(label)
