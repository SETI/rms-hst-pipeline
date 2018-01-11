"""
Functionality to build a collection inventory using a SQLite database.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_collection_inventory(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    products = bundle_db.get_collection_products(
        collection_lidvid)
    inventory_lines = [u'P,%s\r\n' % product.lidvid
                       for product in products]
    return ''.join(inventory_lines)
