"""
Functionality to build a collection inventory using a SQLite database.
"""

from typing import cast, List

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import DocumentCollection, NonDocumentCollection


def get_collection_inventory_name(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> str
    collection = bundle_db.get_collection(collection_lidvid)
    if isinstance(collection, DocumentCollection):
        return "collection.csv"
    else:
        return "collection_%s.csv" % cast(NonDocumentCollection, collection).prefix


def make_collection_inventory(bundle_db: BundleDB, collection_lidvid: str) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_collection_products(collection_lidvid)
    inventory_lines: List[str] = [f"P,{product.lidvid}\r\n" for product in products]
    res: str = "".join(inventory_lines)
    return res.encode()
