"""
Functionality to build a collection inventory using a SQLite database.
"""

from typing import List, cast

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    Collection,
    DocumentCollection,
    OtherCollection,
    switch_on_collection_subtype,
)


def get_collection_inventory_name(bundle_db: BundleDB, collection_lidvid: str) -> str:
    # We have to jump through some hoops to apply
    # switch_on_collection_type().
    def get_context_collection_inventory_name(collection: Collection) -> str:
        return "collection_context.csv"

    def get_document_collection_inventory_name(collection: Collection) -> str:
        return "collection.csv"

    def get_schema_collection_inventory_name(collection: Collection) -> str:
        return "collection_schema.csv"

    def get_other_collection_inventory_name(collection: Collection) -> str:
        collection_obj = cast(OtherCollection, collection)
        prefix = collection_obj.prefix
        instrument = collection_obj.instrument
        suffix = collection_obj.suffix
        return f"collection_{prefix}_{instrument}_{suffix}.csv"

    collection: Collection = bundle_db.get_collection(collection_lidvid)
    return switch_on_collection_subtype(
        collection,
        get_context_collection_inventory_name,
        get_document_collection_inventory_name,
        get_schema_collection_inventory_name,
        get_other_collection_inventory_name,
    )(collection)


def make_collection_inventory(bundle_db: BundleDB, collection_lidvid: str) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    collection = bundle_db.get_collection(collection_lidvid)
    return switch_on_collection_subtype(
        collection,
        make_context_collection_inventory,
        make_document_collection_inventory,
        make_schema_collection_inventory,
        make_other_collection_inventory,
    )(bundle_db, collection_lidvid)


def make_context_collection_inventory(
    bundle_db: BundleDB, collection_lidvid: str
) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_context_products()
    inventory_lines: List[str] = [f"S,{product.lidvid}\r\n" for product in products]
    res: str = "".join(inventory_lines)
    return res.encode()


def make_schema_collection_inventory(
    bundle_db: BundleDB, collection_lidvid: str
) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_schema_products()
    inventory_lines: List[str] = [f"S,{product.lidvid}\r\n" for product in products]
    res: str = "".join(inventory_lines)
    return res.encode()


def make_document_collection_inventory(
    bundle_db: BundleDB, collection_lidvid: str
) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_collection_products(collection_lidvid)
    inventory_lines: List[str] = [f"P,{product.lidvid}\r\n" for product in products]
    # Include handbooks in the document collection csv
    inst_list = bundle_db.get_instruments_of_the_bundle()
    for inst in inst_list:
        data_handbook_lid = f"S,urn:nasa:pds:hst-support:document:{inst}-dhb\r\n"
        inst_handbook_lid = f"S,urn:nasa:pds:hst-support:document:{inst}-ihb\r\n"
        inventory_lines.append(data_handbook_lid)
        inventory_lines.append(inst_handbook_lid)
    res: str = "".join(inventory_lines)
    return res.encode()


def make_other_collection_inventory(
    bundle_db: BundleDB, collection_lidvid: str
) -> bytes:
    """
    Create the inventory text for the collection having this LIDVID
    using the bundle database.
    """
    products = bundle_db.get_collection_products(collection_lidvid)
    inventory_lines: List[str] = [f"P,{product.lidvid}\r\n" for product in products]
    res: str = "".join(inventory_lines)
    return res.encode()
