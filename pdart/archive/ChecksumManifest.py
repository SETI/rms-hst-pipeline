from typing import TYPE_CHECKING

import fs.path

from pdart.new_db.BundleDB import BundleDB
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Callable, List, Tuple
    from pdart.new_db.SqlTables import File


def make_checksum_manifest(bundle_db, lidvid_to_dirpath):
    # type: (BundleDB, Callable[[LIDVID], unicode]) -> str
    files = []  # type: List[File]
    bundle = bundle_db.get_bundle()
    for collection in bundle_db.get_bundle_collections(str(bundle.lidvid)):
        for product \
                in bundle_db.get_collection_products(str(collection.lidvid)):
            files.extend(bundle_db.get_product_files(str(product.lidvid)))

    sorted_pairs = sorted(make_checksum_pair(file, lidvid_to_dirpath)
                          for file in files)
    return ''.join('%s  %s\n' % (hash, path)
                   for (path, hash) in sorted_pairs)


def make_checksum_pair(file, lidvid_to_dirpath):
    # type: (File, Callable[[LIDVID], unicode]) -> Tuple[unicode, str]
    def file_to_filepath(file):
        # type: (File) -> unicode
        dir = lidvid_to_dirpath(LIDVID(file.product_lidvid))
        return fs.path.relpath(fs.path.join(dir, file.basename))

    return (file_to_filepath(file), file.md5_hash)
