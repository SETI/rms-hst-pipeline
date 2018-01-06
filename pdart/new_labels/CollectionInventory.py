"""
Functionality to build a collection inventory using a SQLite database.
"""

from typing import TYPE_CHECKING

from pdart.new_labels.Utils import lidvid_to_lid

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_collection_inventory(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    products = bundle_db.get_collection_products(
        collection_lidvid)
    inventory_lines = [u'P,%s\r\n' % lidvid_to_lid(product.lidvid)
                       for product in products]
    return ''.join(inventory_lines)
