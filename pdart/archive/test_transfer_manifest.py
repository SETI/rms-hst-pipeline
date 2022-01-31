import unittest

import fs.path

from pdart.archive.test_utils import lid_to_dir
from pdart.archive.transfer_manifest import make_transfer_manifest
from pdart.db.bundle_db import create_bundle_db_in_memory
from pdart.pds4.lidvid import LIDVID

_BUNDLE_LIDVID: str = "urn:nasa:pds:hst_00001::1.0"
_COLLECTION_LIDVID: str = "urn:nasa:pds:hst_00001:data_wfpc2_raw::1.0"
_COLLECTION2_LIDVID: str = "urn:nasa:pds:hst_00001:document::1.0"
_PRODUCT_LIDVID: str = "urn:nasa:pds:hst_00001:data_wfpc2_raw:u2no0401t::1.0"
_PRODUCT2_LIDVID: str = "urn:nasa:pds:hst_00001:document:phase2::1.0"


def _lidvid_to_dirpath(lidvid: LIDVID) -> str:
    lid = lidvid.lid()
    return fs.path.relpath(lid_to_dir(lid))


class testTransferManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_db = create_bundle_db_in_memory()
        self.bundle_db.create_tables()
        self.bundle_db.create_bundle(_BUNDLE_LIDVID)

    def test_empty_db(self) -> None:
        expected = "urn:nasa:pds:hst_00001::1.0 hst_00001/bundle.xml\n"
        manifest = make_transfer_manifest(
            self.bundle_db, _BUNDLE_LIDVID, _lidvid_to_dirpath
        )
        self.assertEqual(expected, manifest)

    def test_filled_db(self) -> None:
        self.bundle_db.create_other_collection(_COLLECTION_LIDVID, _BUNDLE_LIDVID)
        self.bundle_db.create_fits_product(_PRODUCT_LIDVID, _COLLECTION_LIDVID)

        expected = """\
urn:nasa:pds:hst_00001::1.0                          hst_00001/bundle.xml
urn:nasa:pds:hst_00001:data_wfpc2_raw::1.0           \
hst_00001/data_wfpc2_raw/collection_data_wfpc2_raw.xml
urn:nasa:pds:hst_00001:data_wfpc2_raw:u2no0401t::1.0 \
hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t.xml
"""
        manifest = make_transfer_manifest(
            self.bundle_db, _BUNDLE_LIDVID, _lidvid_to_dirpath
        )
        self.assertEqual(expected, manifest)

        self.bundle_db.create_document_collection(_COLLECTION2_LIDVID, _BUNDLE_LIDVID)

        self.bundle_db.create_document_product(_PRODUCT2_LIDVID, _COLLECTION2_LIDVID)

        expected = """\
urn:nasa:pds:hst_00001::1.0                          hst_00001/bundle.xml
urn:nasa:pds:hst_00001:data_wfpc2_raw::1.0           \
hst_00001/data_wfpc2_raw/collection_data_wfpc2_raw.xml
urn:nasa:pds:hst_00001:data_wfpc2_raw:u2no0401t::1.0 \
hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t.xml
urn:nasa:pds:hst_00001:document::1.0                 \
hst_00001/document/collection.xml
urn:nasa:pds:hst_00001:document:phase2::1.0          \
hst_00001/document/phase2/phase2.xml
"""
        manifest = make_transfer_manifest(
            self.bundle_db, _BUNDLE_LIDVID, _lidvid_to_dirpath
        )

        self.assertEqual(expected, manifest)
