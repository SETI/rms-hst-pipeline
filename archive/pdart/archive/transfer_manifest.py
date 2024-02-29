from typing import Callable, Tuple

import fs.path

from pdart.db.bundle_db import BundleDB
from pdart.db.sql_alch_tables import Bundle, Collection, Product
from pdart.labels.collection_label import get_collection_label_name
from pdart.pds4.lidvid import LIDVID

_LTD = Callable[[LIDVID], str]


def _make_bundle_pair(bundle: Bundle, lidvid_to_dirpath: _LTD) -> Tuple[str, str]:
    lidvid = str(bundle.lidvid)
    dir = fs.path.relpath(lidvid_to_dirpath(LIDVID(lidvid)))
    filepath = "bundle.xml"
    return (lidvid, fs.path.join(dir, filepath))


def _make_collection_pair(
    bundle_db: BundleDB, collection: Collection, lidvid_to_dirpath: _LTD
) -> Tuple[str, str]:
    lidvid = str(collection.lidvid)
    dir = fs.path.relpath(lidvid_to_dirpath(LIDVID(lidvid)))
    filepath = get_collection_label_name(bundle_db, lidvid)
    return (str(collection.lidvid), fs.path.join(dir, filepath))


def _make_product_pair(product: Product, lidvid_to_dirpath: _LTD) -> Tuple[str, str]:
    lidvid = str(product.lidvid)
    dir = fs.path.relpath(lidvid_to_dirpath(LIDVID(lidvid)))
    product_id = LIDVID(lidvid).lid().product_id
    filepath = f"{product_id}.xml"
    return (lidvid, fs.path.join(dir, filepath))


def make_transfer_manifest(
    bundle_db: BundleDB, bundle_lidvid: str, lidvid_to_dirpath: _LTD
) -> str:
    bundle = bundle_db.get_bundle(bundle_lidvid)
    pairs = [_make_bundle_pair(bundle, lidvid_to_dirpath)]
    for collection in bundle_db.get_bundle_collections(str(bundle.lidvid)):
        pairs.append(_make_collection_pair(bundle_db, collection, lidvid_to_dirpath))
        for product in bundle_db.get_collection_products(str(collection.lidvid)):
            pairs.append(_make_product_pair(product, lidvid_to_dirpath))

    sorted_pairs = sorted(pairs)
    max_width = max(len(lidvid) for (lidvid, _filepath) in sorted_pairs)
    return "".join(
        [
            # TODO rewrite this in f-string notation
            "%-*s %s\n" % (max_width, lidvid, str(filepath))
            for (lidvid, filepath) in sorted_pairs
        ]
    )
