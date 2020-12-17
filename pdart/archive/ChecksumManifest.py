import logging
from typing import Callable, List, Optional, Tuple

import fs.path

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    BundleLabel,
    CollectionInventory,
    CollectionLabel,
    File,
    ProductLabel,
)
from pdart.pds4.LIDVID import LIDVID

_LOGGER = logging.getLogger(__name__)

_LTD = Callable[[LIDVID], str]


def plain_lidvid_to_dirpath(lidvid: LIDVID) -> str:
    lid = lidvid.lid()
    parts = lid.parts()[1:]
    return fs.path.join("/", *parts)


def _visit_of(dirname_segment: str) -> Optional[str]:
    if len(dirname_segment) == 9:
        return "visit_" + dirname_segment[4:6].lower()
    else:
        return None


def plain_lidvid_to_visits_dirpath(lidvid: LIDVID) -> str:
    # This is only run on directories, and always relative to /hst_NNNNN.
    lid = lidvid.lid()
    parts = lid.parts()[1:]
    if len(parts) == 2:
        visit = _visit_of(parts[1])
        if visit is not None:
            parts[1] = visit
    return fs.path.join("/", *parts)


def make_checksum_manifest(
    bundle_db: BundleDB, bundle_lidvid: str, lidvid_to_dirpath: _LTD
) -> str:
    files: List[File] = []

    bundle = bundle_db.get_bundle(bundle_lidvid)
    bundle_lidvid = str(bundle.lidvid)
    label_pairs = [
        make_bundle_label_pair(
            bundle_db.get_bundle_label(bundle_lidvid), lidvid_to_dirpath
        )
    ]

    for collection in bundle_db.get_bundle_collections(bundle_lidvid):
        collection_lidvid = str(collection.lidvid)
        label_pairs.append(
            make_collection_label_pair(
                bundle_db.get_collection_label(collection_lidvid), lidvid_to_dirpath
            )
        )
        label_pairs.append(
            make_collection_inventory_pair(
                bundle_db.get_collection_inventory(collection_lidvid),
                lidvid_to_dirpath,
            )
        )
        for product in bundle_db.get_collection_products(collection_lidvid):
            product_lidvid = str(product.lidvid)
            label_pairs.append(
                make_product_label_pair(
                    bundle_db.get_product_label(product_lidvid), lidvid_to_dirpath
                )
            )
            files.extend(bundle_db.get_product_files(product_lidvid))

    file_pairs = [make_checksum_pair(file, lidvid_to_dirpath) for file in files]

    sorted_pairs = sorted(file_pairs + label_pairs)
    return "".join(f"{hash}  {path}\n" for (path, hash) in sorted_pairs)


def make_bundle_label_pair(
    bundle_label: BundleLabel, lidvid_to_dirpath: _LTD
) -> Tuple[str, str]:
    def label_to_filepath(bundle_label: BundleLabel) -> str:
        dir = lidvid_to_dirpath(LIDVID(bundle_label.bundle_lidvid))
        return fs.path.relpath(fs.path.join(dir, bundle_label.basename))

    return (label_to_filepath(bundle_label), bundle_label.md5_hash)


def make_collection_label_pair(
    collection_label: CollectionLabel, lidvid_to_dirpath: _LTD
) -> Tuple[str, str]:
    def label_to_filepath(collection_label: CollectionLabel) -> str:
        dir = lidvid_to_dirpath(LIDVID(collection_label.collection_lidvid))
        return fs.path.relpath(fs.path.join(dir, collection_label.basename))

    return (label_to_filepath(collection_label), collection_label.md5_hash)


def make_collection_inventory_pair(
    collection_inventory: CollectionInventory, lidvid_to_dirpath: _LTD
) -> Tuple[str, str]:
    def inventory_to_filepath(collection_inventory: CollectionInventory) -> str:
        dir = lidvid_to_dirpath(LIDVID(collection_inventory.collection_lidvid))
        return fs.path.relpath(fs.path.join(dir, collection_inventory.basename))

    return (inventory_to_filepath(collection_inventory), collection_inventory.md5_hash)


def make_product_label_pair(
    product_label: ProductLabel, lidvid_to_dirpath: _LTD
) -> Tuple[str, str]:
    def label_to_filepath(product_label: ProductLabel) -> str:
        dir = lidvid_to_dirpath(LIDVID(product_label.product_lidvid))
        return fs.path.relpath(fs.path.join(dir, product_label.basename))

    return (label_to_filepath(product_label), product_label.md5_hash)


def make_checksum_pair(file: File, lidvid_to_dirpath: _LTD) -> Tuple[str, str]:
    def file_to_filepath(file: File) -> str:
        dir = lidvid_to_dirpath(LIDVID(file.product_lidvid))
        return fs.path.relpath(fs.path.join(dir, file.basename))

    return (file_to_filepath(file), file.md5_hash)
