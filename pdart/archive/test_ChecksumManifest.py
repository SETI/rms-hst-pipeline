import shutil
import tempfile
from typing import TYPE_CHECKING
import unittest

import fs.path

from pdart.archive.ChecksumManifest import make_checksum_manifest
from pdart.fs.DirUtils import lid_to_dir
from pdart.new_db.BundleDB import create_bundle_db_in_memory

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


def _lidvid_to_filepath(lidvid):
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
        self.assertEqual('', make_checksum_manifest(self.bundle_db,
                                                    _lidvid_to_filepath))

    def test_minimal_db(self):
        self.bundle_db.create_non_document_collection(_COLLECTION_LIDVID,
                                                      _BUNDLE_LIDVID)
        self.bundle_db.create_fits_product(_PRODUCT_LIDVID,
                                           _COLLECTION_LIDVID)
        os_filepath = fs.path.join(self.tmpdir, _PRODUCT_BASENAME)
        with open(os_filepath, 'w') as f:
            f.write(_PRODUCT_CONTENTS)

        self.bundle_db.create_fits_file(os_filepath, _PRODUCT_BASENAME,
                                        _PRODUCT_LIDVID, _HDU_COUNT)

        manifest = make_checksum_manifest(self.bundle_db, _lidvid_to_filepath)

        self.assertEqual(
            'ba8a714e47d3c7606c0a2d438f9e4811  '
            'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t_raw.fits\n',
            manifest)

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

        expected = ('ba8a714e47d3c7606c0a2d438f9e4811  '
                    'hst_00001/data_wfpc2_raw/u2no0401t/u2no0401t_raw.fits\n'
                    '64d11a5e59de03ce7ee7acf905c67aee  '
                    'hst_00001/document/phase2/phase2.pdf\n')

        manifest = make_checksum_manifest(self.bundle_db, _lidvid_to_filepath)
        self.assertEqual(expected, manifest)
