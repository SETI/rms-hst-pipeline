from typing import TYPE_CHECKING

import fs.path

from pdart.new_db.BundleDB import BundleDB
from pdart.new_db.SqlAlchTables import Bundle, Collection, Product
from pdart.new_labels.CollectionLabel import get_collection_label_name
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Callable, Tuple
    _LTD = Callable[[LIDVID], unicode]


def _make_bundle_pair(bundle, lidvid_to_dirpath):
    # type: (Bundle, _LTD) -> Tuple[str, unicode]
    lidvid = str(bundle.lidvid)
    dir = lidvid_to_dirpath(LIDVID(lidvid))
    filepath = 'bundle.xml'
    return (lidvid,
            fs.path.join(dir, filepath))


def _make_collection_pair(bundle_db, collection, lidvid_to_dirpath):
    # type: (BundleDB, Collection, _LTD) -> Tuple[str, unicode]
    lidvid = str(collection.lidvid)
    dir = lidvid_to_dirpath(LIDVID(lidvid))
    filepath = get_collection_label_name(bundle_db, lidvid)
    return (str(collection.lidvid), fs.path.join(dir, filepath))


def _make_product_pair(product, lidvid_to_dirpath):
    # type: (Product, _LTD) -> Tuple[str, unicode]
    lidvid = str(product.lidvid)
    dir = lidvid_to_dirpath(LIDVID(lidvid))
    filepath = '%s.xml' % (LIDVID(lidvid).lid().product_id)
    return (lidvid, fs.path.join(dir, filepath))


def make_transfer_manifest(bundle_db, lidvid_to_dirpath):
    # type: (BundleDB, _LTD) -> str
    bundle = bundle_db.get_bundle()
    pairs = [_make_bundle_pair(bundle, lidvid_to_dirpath)]
    for collection in bundle_db.get_bundle_collections(str(bundle.lidvid)):
        pairs.append(_make_collection_pair(bundle_db,
                                           collection,
                                           lidvid_to_dirpath))
        for product \
                in bundle_db.get_collection_products(str(collection.lidvid)):
            pairs.append(_make_product_pair(product, lidvid_to_dirpath))

    sorted_pairs = sorted(pairs)
    max_width = max(len(lidvid) for (lidvid, _filepath) in sorted_pairs)
    return ''.join(['%-*s %s\n' % (max_width, lidvid, str(filepath))
                    for (lidvid, filepath) in sorted_pairs])
