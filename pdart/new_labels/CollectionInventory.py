"""
Functionality to build a collection inventory using a SQLite database.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def get_collection_inventory_name(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    collection = bundle_db.get_collection(collection_lidvid)
    try:
        inventory_name = 'collection_%s.csv' % collection.prefix
    except Exception:
        # Document collections won't have prefixes.
        inventory_name = 'collection.csv'
    return inventory_name


def make_collection_inventory(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_collection_products(
        collection_lidvid)
    inventory_lines = [u'P,%s\r\n' % product.lidvid
                       for product in products]
    return ''.join(inventory_lines)
