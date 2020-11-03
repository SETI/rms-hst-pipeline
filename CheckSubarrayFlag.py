from collections import ChainMap
from typing import List
import os
import os.path
from pdart.db.BundleDB import (
    _BUNDLE_DB_NAME,
    BundleDB,
    create_bundle_db_from_os_filepath,
)
from pdart.db.SqlAlchTables import Card, switch_on_collection_subtype
from pdart.pds4.LIDVID import LIDVID

TWD = os.environ["TMP_WORKING_DIR"]


def has_subarray(db: BundleDB) -> bool:
    subarray_cards: List[Card] = (
        db.session.query(Card).filter(Card.keyword == "SUBARRAY").all()
    )
    return len(subarray_cards) != 0


for bundle_dir in os.listdir(TWD):
    db_path = os.path.join(TWD, bundle_dir, _BUNDLE_DB_NAME)
    if os.path.isfile(db_path):
        db = create_bundle_db_from_os_filepath(db_path)
        bundle = db.get_bundle()
        if has_subarray(db):
            print(f"# {LIDVID(bundle.lidvid).lid().bundle_id}:", flush=True)
            for collection in db.get_bundle_collections(bundle.lidvid):
                is_other = switch_on_collection_subtype(
                    collection, False, False, False, True
                )
                if is_other:
                    for product in db.get_collection_products(collection.lidvid):
                        product_id = LIDVID(product.lidvid).lid().product_id
                        assert product_id
                        instr = product_id[0]
                        if instr != "u":
                            for product_file in db.get_product_files(product.lidvid):
                                dicts = db.get_card_dictionaries(
                                    product.lidvid, product_file.basename
                                )
                                dict = ChainMap(*dicts)
                                if "SUBARRAY" in dict:
                                    print(
                                        f"{product.lidvid[13:-5]}/{product_file.basename}: {dict.get('DETECTOR')} SUBARRAY={dict.get('SUBARRAY')}, {dict.get('NAXIS1')} x {dict.get('NAXIS2')}",
                                        flush=True,
                                    )
