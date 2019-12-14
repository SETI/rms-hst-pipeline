import fs.path
from typing import TYPE_CHECKING

from pdart.new_db.BundleDB import BundleDB
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Callable, List, Tuple
    from pdart.new_db.SqlAlchTables import BundleLabel, CollectionInventory, \
        CollectionLabel, File, ProductLabel

    _LTD = Callable[[LIDVID], unicode]


def make_checksum_manifest(bundle_db, lidvid_to_dirpath):
    # type: (BundleDB, _LTD) -> str
    files = []  # type: List[File]

    bundle = bundle_db.get_bundle()
    bundle_lidvid = str(bundle.lidvid)
    label_pairs = [make_bundle_label_pair(bundle_db.get_bundle_label(
        bundle_lidvid), lidvid_to_dirpath)]

    for collection in bundle_db.get_bundle_collections(bundle_lidvid):
        collection_lidvid = str(collection.lidvid)
        label_pairs.append(
            make_collection_label_pair(
                bundle_db.get_collection_label(
                    collection_lidvid), lidvid_to_dirpath))
        label_pairs.append(
            make_collection_inventory_pair(
                bundle_db.get_collection_inventory(
                    collection_lidvid), lidvid_to_dirpath))
        for product \
                in bundle_db.get_collection_products(collection_lidvid):
            product_lidvid = str(product.lidvid)
            label_pairs.append(
                make_product_label_pair(
                    bundle_db.get_product_label(
                        product_lidvid), lidvid_to_dirpath))
            files.extend(bundle_db.get_product_files(product_lidvid))

    file_pairs = [make_checksum_pair(file, lidvid_to_dirpath)
                  for file in files]

    sorted_pairs = sorted(file_pairs + label_pairs)
    return ''.join('%s  %s\n' % (hash, path)
                   for (path, hash) in sorted_pairs)


def make_bundle_label_pair(bundle_label, lidvid_to_dirpath):
    # type: (BundleLabel, _LTD) -> Tuple[unicode, str]
    def label_to_filepath(bundle_label):
        # type: (BundleLabel) -> unicode
        dir = lidvid_to_dirpath(LIDVID(bundle_label.bundle_lidvid))
        return fs.path.relpath(fs.path.join(dir, bundle_label.basename))

    return (label_to_filepath(bundle_label), bundle_label.md5_hash)


def make_collection_label_pair(collection_label, lidvid_to_dirpath):
    # type: (CollectionLabel, _LTD) -> Tuple[unicode, str]
    def label_to_filepath(collection_label):
        # type: (CollectionLabel) -> unicode
        dir = lidvid_to_dirpath(LIDVID(collection_label.collection_lidvid))
        return fs.path.relpath(fs.path.join(dir, collection_label.basename))

    return (label_to_filepath(collection_label), collection_label.md5_hash)


def make_collection_inventory_pair(collection_inventory, lidvid_to_dirpath):
    # type: (CollectionInventory, _LTD) -> Tuple[unicode, str]
    def inventory_to_filepath(collection_inventory):
        # type: (CollectionInventory) -> unicode
        dir = lidvid_to_dirpath(LIDVID(collection_inventory.collection_lidvid))
        return fs.path.relpath(fs.path.join(dir,
                                            collection_inventory.basename))

    return (inventory_to_filepath(collection_inventory),
            collection_inventory.md5_hash)


def make_product_label_pair(product_label, lidvid_to_dirpath):
    # type: (ProductLabel, _LTD) -> Tuple[unicode, str]
    def label_to_filepath(product_label):
        # type: (ProductLabel) -> unicode
        dir = lidvid_to_dirpath(LIDVID(product_label.product_lidvid))
        return fs.path.relpath(fs.path.join(dir, product_label.basename))

    return (label_to_filepath(product_label), product_label.md5_hash)


def make_checksum_pair(file, lidvid_to_dirpath):
    # type: (File, _LTD) -> Tuple[unicode, str]
    def file_to_filepath(file):
        # type: (File) -> unicode
        dir = lidvid_to_dirpath(LIDVID(file.product_lidvid))
        return fs.path.relpath(fs.path.join(dir, file.basename))

    return (file_to_filepath(file), file.md5_hash)
