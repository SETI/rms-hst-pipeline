import shutil
import tempfile
import unittest

import fs.path
from typing import TYPE_CHECKING

from pdart.archive.ChecksumManifest import make_checksum_manifest
from pdart.fs.DirUtils import lid_to_dir
from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.CollectionInventory import get_collection_inventory_name
from pdart.new_labels.CollectionLabel import get_collection_label_name

if TYPE_CHECKING:
    from pdart.pds4.LIDVID import LIDVID

_BUNDLE_LIDVID = 'urn:nasa:pds:hst_00001::1.0'
_COLLECTION_LIDVID = 'urn:nasa:pds:hst_00001:data_wfpc2_raw::1.0'
_COLLECTION2_LIDVID = 'urn:nasa:pds:hst_00001:document::1.0'
_PRODUCT_LIDVID = 'urn:nasa:pds:hst_00001:data_wfpc2_raw:u2no0401t::1.0'
_PRODUCT_BASENAME = 'u2no0401t_raw.fits'
_PRODUCT_CONTENTS = "I am mascarading as a raw FITS file.\n"
_HDU_COUNT = 1
_PRODUCT2_LIDVID = 'urn:nasa:pds:hst_00001:document:phase2::1.0'
_PRODUCT2_BASENAME = 'phase2.pdf'
_PRODUCT2_CONTENTS = "I am mascarading as a PDF file."


def _lidvid_to_dirpath(lidvid):
    # type: (LIDVID) -> unicode
    lid = lidvid.lid()
    return fs.path.relpath(lid_to_dir(lid))


class test_ChecksumManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bundle_db = create_bundle_db_in_memory()
        self.bundle_db.create_tables()
        self.bundle_db.create_bundle(_BUNDLE_LIDVID)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_empty_db(self):
        os_filepath = fs.path.join(self.tmpdir, 'bundle.xml')
        with open(os_filepath, 'w') as f:
            f.write("I'm neither XML nor a label, sadly.")
        self.bundle_db.create_bundle_label(os_filepath, 'bundle.xml',
                                           _BUNDLE_LIDVID)
        self.assertEqual(
            'e2309513113b550428af0cf476f1fb67  hst_00001/bundle.xml\n',
            make_checksum_manifest(self.bundle_db, _lidvid_to_dirpath))

    def test_minimal_db(self):
        # create stuff (top-down)
        self.bundle_db.create_non_document_collection(_COLLECTION_LIDVID,
                                                      _BUNDLE_LIDVID)
        self.bundle_db.create_fits_product(_PRODUCT_LIDVID,
                                           _COLLECTION_LIDVID)
        os_filepath = fs.path.join(self.tmpdir, _PRODUCT_BASENAME)
        with open(os_filepath, 'w') as f:
            f.write(_PRODUCT_CONTENTS)

        self.bundle_db.create_fits_file(os_filepath, _PRODUCT_BASENAME,
                                        _PRODUCT_LIDVID, _HDU_COUNT)

        # create labels (bottom-up)
        self.bundle_db.create_product_label(os_filepath, 'u2no0401t.xml',
                                            _PRODUCT_LIDVID)
        collection_label_name = get_collection_label_name(self.bundle_db,
                                                          _COLLECTION_LIDVID)
        self.bundle_db.create_collection_label(os_filepath,
                                               collection_label_name,
                                               _COLLECTION_LIDVID)
        collection_inventory_name = get_collection_inventory_name(
            self.bundle_db, _COLLECTION_LIDVID)
        self.bundle_db.create_collection_inventory(os_filepath,
                                                   collection_inventory_name,
                                                   _COLLECTION_LIDVID)
        self.bundle_db.create_bundle_label(os_filepath, 'bundle.xml',
                                           _BUNDLE_LIDVID)

        manifest = make_checksum_manifest(self.bundle_db, _lidvid_to_dirpath)

        self.assertEqual(
            'ba8a714e47d3c7606c0a2d438f9e4811  hst_00001/bundle.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/collection_data.csv\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/collection_data.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t_raw.fits\n',
            manifest)

        # create stuff (top-down)
        self.bundle_db.create_document_collection(_COLLECTION2_LIDVID,
                                                  _BUNDLE_LIDVID)

        self.bundle_db.create_document_product(_PRODUCT2_LIDVID,
                                               _COLLECTION2_LIDVID)

        os_filepath = fs.path.join(self.tmpdir, _PRODUCT2_BASENAME)
        with open(os_filepath, 'w') as f:
            f.write(_PRODUCT2_CONTENTS)

        self.bundle_db.create_document_file(os_filepath,
                                            _PRODUCT2_BASENAME,
                                            _PRODUCT2_LIDVID)

        # create labels (bottom-up)
        self.bundle_db.create_product_label(os_filepath, 'phase2.xml',
                                            _PRODUCT2_LIDVID)
        collection_label_name = get_collection_label_name(self.bundle_db,
                                                          _COLLECTION2_LIDVID)
        self.bundle_db.create_collection_label(os_filepath,
                                               collection_label_name,
                                               _COLLECTION2_LIDVID)

        collection_inventory_name = get_collection_inventory_name(
            self.bundle_db, _COLLECTION2_LIDVID)
        self.bundle_db.create_collection_inventory(os_filepath,
                                                   collection_inventory_name,
                                                   _COLLECTION2_LIDVID)

        expected = (
            'ba8a714e47d3c7606c0a2d438f9e4811  hst_00001/bundle.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/collection_data.csv\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/collection_data.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t.xml\n'
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t_raw.fits\n'
            '64d11a5e59de03ce7ee7acf905c67aee  '
            'hst_00001/document/collection.csv\n'
            '64d11a5e59de03ce7ee7acf905c67aee  '
            'hst_00001/document/collection.xml\n'
            '64d11a5e59de03ce7ee7acf905c67aee  '
            'hst_00001/document/phase2/phase2.pdf\n'
            '64d11a5e59de03ce7ee7acf905c67aee  '
            'hst_00001/document/phase2/phase2.xml\n'
            )

        manifest = make_checksum_manifest(self.bundle_db, _lidvid_to_dirpath)
        self.assertEqual(expected, manifest)
