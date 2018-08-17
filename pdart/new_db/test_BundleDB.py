from typing import TYPE_CHECKING
import unittest

from pdart.new_db.BundleDB import BundleDB, create_bundle_db_in_memory
from pdart.new_db.SqlAlchTables import Base, Bundle, NonDocumentCollection

if TYPE_CHECKING:
    from typing import Set

_TABLES = {'bundles',
           'collections', 'document_collections', 'non_document_collections',
           'products', 'browse_products', 'document_products', 'fits_products',
           'files', 'fits_files', 'bad_fits_files', 'browse_files',
           'document_files',
           'hdus', 'cards'}  # type: Set[str]


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_BundleDB(self):
        # type: () -> None
        db = self.db
        self.assertTrue(db.is_open())
        metadata = Base.metadata
        self.assertEqual(set(metadata.tables.keys()), _TABLES)
        db.close()
        self.assertFalse(db.is_open())

    ############################################################

    def test_create_bundle(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 0)
        self.assertFalse(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 1)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))

    def test_bundle_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.assertFalse(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))

    def test_get_bundle(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        bundle = self.db.get_bundle()
        self.assertEqual(bundle_lidvid, bundle.lidvid)
        self.assertEqual(99999, bundle.proposal_id)

    def test_get_bundle_collections(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        non_doc_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.1'
        doc_collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.1'

        # test empty
        self.db.create_bundle(bundle_lidvid)
        self.assertFalse(self.db.get_bundle_collections(bundle_lidvid))

        # test inserting one
        self.db.create_non_document_collection(non_doc_collection_lidvid,
                                               bundle_lidvid)
        self.assertEqual({non_doc_collection_lidvid},
                         {c.lidvid
                          for c
                          in self.db.get_bundle_collections(bundle_lidvid)})

        # test inserting a different kind
        self.db.create_document_collection(doc_collection_lidvid,
                                           bundle_lidvid)
        self.assertEqual({non_doc_collection_lidvid, doc_collection_lidvid},
                         {c.lidvid
                          for c
                          in self.db.get_bundle_collections(bundle_lidvid)})

    ############################################################

    def test_create_document_collection(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.assertFalse(
            self.db.document_collection_exists(collection_lidvid))

        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)
        self.assertTrue(
            self.db.document_collection_exists(collection_lidvid))

        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)
        self.assertTrue(
            self.db.document_collection_exists(collection_lidvid))

    def test_create_non_document_collection(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.assertFalse(
            self.db.non_document_collection_exists(collection_lidvid))

        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        self.assertTrue(
            self.db.non_document_collection_exists(collection_lidvid))

        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        self.assertTrue(
            self.db.non_document_collection_exists(collection_lidvid))

    def test_collection_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.1'
        self.db.create_bundle(bundle_lidvid)
        self.assertFalse(self.db.collection_exists(collection_lidvid))
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        self.assertTrue(self.db.collection_exists(collection_lidvid))

    def test_document_collection_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        non_doc_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.1'
        doc_collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.1'

        self.db.create_non_document_collection(non_doc_collection_lidvid,
                                               bundle_lidvid)
        self.assertFalse(
            self.db.document_collection_exists(
                doc_collection_lidvid))

        self.db.create_document_collection(doc_collection_lidvid,
                                           bundle_lidvid)

        self.assertTrue(
            self.db.document_collection_exists(
                doc_collection_lidvid))

    def test_non_document_collection_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        non_doc_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.1'
        doc_collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.1'

        self.db.create_document_collection(doc_collection_lidvid,
                                           bundle_lidvid)

        self.assertFalse(
            self.db.non_document_collection_exists(
                non_doc_collection_lidvid))

        self.db.create_non_document_collection(non_doc_collection_lidvid,
                                               bundle_lidvid)
        self.assertTrue(
            self.db.non_document_collection_exists(
                non_doc_collection_lidvid))

    def test_get_collection(self):
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        collection = self.db.get_collection(collection_lidvid)
        self.assertEqual(NonDocumentCollection, type(collection))
        self.assertEqual('acs', collection.instrument)
        self.assertEqual('raw', collection.suffix)

    def test_get_collection_products(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        p1_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:j6gp01lzq::1.8'
        p2_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:j6gp02lzq::1.8'
        p3_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:j6gp03lzq::1.8'

        self.db.create_bundle(bundle_lidvid)
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        self.assertFalse(self.db.get_collection_products(
            collection_lidvid))

        self.db.create_fits_product(p1_lidvid,
                                    collection_lidvid)

        self.assertEqual({p1_lidvid},
                         {p.lidvid
                          for p
                          in self.db.get_collection_products(
                             collection_lidvid)})

        self.db.create_fits_product(p2_lidvid,
                                    collection_lidvid)

        self.assertEqual({p1_lidvid, p2_lidvid},
                         {p.lidvid
                          for p
                          in self.db.get_collection_products(
                             collection_lidvid)})

        self.db.create_fits_product(p3_lidvid,
                                    collection_lidvid)

        self.assertEqual({p1_lidvid, p2_lidvid, p3_lidvid},
                         {p.lidvid
                          for p
                          in self.db.get_collection_products(
                             collection_lidvid)})

    ############################################################

    def test_create_browse_product(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        data_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(data_collection_lidvid,
                                               bundle_lidvid)

        browse_collection_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        self.db.create_non_document_collection(browse_collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        browse_product_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw:p::8.1'

        with self.assertRaises(Exception):
            self.db.create_browse_product(browse_product_lidvid,
                                          fits_product_lidvid,
                                          browse_collection_lidvid)

        self.db.create_fits_product(fits_product_lidvid,
                                    data_collection_lidvid)

        self.assertFalse(self.db.browse_product_exists(browse_product_lidvid))

        self.db.create_browse_product(browse_product_lidvid,
                                      fits_product_lidvid,
                                      browse_collection_lidvid)
        self.assertTrue(self.db.browse_product_exists(browse_product_lidvid))

        self.db.create_browse_product(browse_product_lidvid,
                                      fits_product_lidvid,
                                      browse_collection_lidvid)
        self.assertTrue(self.db.browse_product_exists(browse_product_lidvid))

    def test_create_document_product(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)
        product_lidvid = 'urn:nasa:pds:hst_99999:document:important_stuff::1.8'
        self.assertFalse(
            self.db.document_product_exists(product_lidvid))

        self.db.create_document_product(product_lidvid,
                                        collection_lidvid)
        self.assertTrue(
            self.db.document_product_exists(product_lidvid))

    def test_create_fits_product(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.assertFalse(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

    def test_product_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        data_coll_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        browse_coll_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        doc_coll_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_non_document_collection(data_coll_lidvid,
                                               bundle_lidvid)
        self.db.create_non_document_collection(browse_coll_lidvid,
                                               bundle_lidvid)
        self.db.create_document_collection(doc_coll_lidvid,
                                           bundle_lidvid)

        fits_prod_lidvid = \
            'urn:nasa:pds:hst_99999:data_acs_raw:p::1.8'
        browse_prod_lidvid = \
            'urn:nasa:pds:hst_99999:browse_acs_raw:p::1.8'
        doc_prod_lidvid = \
            'urn:nasa:pds:hst_99999:document:specs::1.8'

        self.assertFalse(self.db.product_exists(fits_prod_lidvid))
        self.assertFalse(self.db.product_exists(browse_prod_lidvid))
        self.assertFalse(self.db.product_exists(doc_prod_lidvid))

        self.db.create_fits_product(fits_prod_lidvid,
                                    data_coll_lidvid)

        self.assertTrue(self.db.product_exists(fits_prod_lidvid))
        self.assertFalse(self.db.product_exists(browse_prod_lidvid))
        self.assertFalse(self.db.product_exists(doc_prod_lidvid))

        self.db.create_browse_product(browse_prod_lidvid,
                                      fits_prod_lidvid,
                                      browse_coll_lidvid)

        self.assertTrue(self.db.product_exists(fits_prod_lidvid))
        self.assertTrue(self.db.product_exists(browse_prod_lidvid))
        self.assertFalse(self.db.product_exists(doc_prod_lidvid))

        self.db.create_document_product(doc_prod_lidvid,
                                        doc_coll_lidvid)

        self.assertTrue(self.db.product_exists(fits_prod_lidvid))
        self.assertTrue(self.db.product_exists(browse_prod_lidvid))
        self.assertTrue(self.db.product_exists(doc_prod_lidvid))

    def test_browse_product_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        collection_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        fits_prod_lidvid = \
            'urn:nasa:pds:hst_99999:data_acs_raw:p::1.8'
        product_lidvid = \
            'urn:nasa:pds:hst_99999:browse_acs_raw:p::1.8'

        self.assertFalse(self.db.browse_product_exists(product_lidvid))

        self.db.create_fits_product(fits_prod_lidvid,
                                    collection_lidvid)
        self.db.create_browse_product(product_lidvid,
                                      fits_prod_lidvid,
                                      collection_lidvid)
        self.assertTrue(self.db.browse_product_exists(product_lidvid))

    def test_document_product_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)
        product_lidvid = 'urn:nasa:pds:hst_99999:document:important_stuff::1.8'

        self.assertFalse(self.db.document_product_exists(product_lidvid))
        self.db.create_document_product(product_lidvid,
                                        collection_lidvid)
        self.assertTrue(self.db.document_product_exists(product_lidvid))

    def test_fits_product_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::1.8'

        self.assertFalse(self.db.fits_product_exists(product_lidvid))
        self.db.create_fits_product(product_lidvid,
                                    collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

    def test_get_product(self):
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        product = self.db.get_product(product_lidvid)
        self.assertEqual(product_lidvid, product.lidvid)
        self.assertEqual(collection_lidvid, product.collection_lidvid)

    def test_get_product_files(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        self.assertFalse(self.db.get_product_files(product_lidvid))

        self.db.create_fits_file('p_raw.fits', product_lidvid, 1)
        self.assertEqual(1, len(self.db.get_product_files(product_lidvid)))

    ############################################################

    def test_create_browse_file(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        data_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(data_collection_lidvid,
                                               bundle_lidvid)

        browse_collection_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        self.db.create_non_document_collection(browse_collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(fits_product_lidvid,
                                    data_collection_lidvid)
        browse_product_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw:p::8.1'
        self.db.create_browse_product(browse_product_lidvid,
                                      fits_product_lidvid,
                                      browse_collection_lidvid)

        basename = 'file.jpg'
        self.assertFalse(
            self.db.browse_file_exists(basename, browse_product_lidvid))

        self.db.create_browse_file(basename, browse_product_lidvid, 1)
        self.assertTrue(
            self.db.browse_file_exists(basename, browse_product_lidvid))

        self.db.create_browse_file(basename, browse_product_lidvid, 1)
        self.assertTrue(
            self.db.browse_file_exists(basename, browse_product_lidvid))

    def test_create_document_file(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:document:p::8.1'
        self.db.create_document_product(product_lidvid, collection_lidvid)

        filename = 'stuff.pdf'
        self.assertFalse(self.db.document_file_exists(filename,
                                                      product_lidvid))

        self.db.create_document_file(filename, product_lidvid)
        self.assertTrue(self.db.document_file_exists(filename,
                                                     product_lidvid))

    def test_create_fits_file(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        basename = 'file.fits'
        self.assertFalse(self.db.fits_file_exists(basename, product_lidvid))

        self.db.create_fits_file(basename, product_lidvid, 1)
        self.assertTrue(self.db.fits_file_exists(basename, product_lidvid))

        self.db.create_fits_file(basename, product_lidvid, 1)
        self.assertTrue(self.db.fits_file_exists(basename, product_lidvid))

    def test_file_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        raw_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        browse_collection_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        doc_collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_non_document_collection(raw_collection_lidvid,
                                               bundle_lidvid)
        self.db.create_non_document_collection(browse_collection_lidvid,
                                               bundle_lidvid)
        self.db.create_document_collection(doc_collection_lidvid,
                                           bundle_lidvid)
        bad_fits_prod_lidvid = \
            'urn:nasa:pds:hst_99999:data_acs_raw:x::1.8'
        fits_prod_lidvid = \
            'urn:nasa:pds:hst_99999:data_acs_raw:p::1.8'
        browse_prod_lidvid = \
            'urn:nasa:pds:hst_99999:browse_acs_raw:p::1.8'
        doc_prod_lidvid = \
            'urn:nasa:pds:hst_99999:document:specs::1.8'
        self.db.create_fits_product(bad_fits_prod_lidvid,
                                    raw_collection_lidvid)
        self.db.create_fits_product(fits_prod_lidvid,
                                    raw_collection_lidvid)
        self.db.create_browse_product(browse_prod_lidvid,
                                      fits_prod_lidvid,
                                      browse_collection_lidvid)
        self.db.create_document_product(doc_prod_lidvid,
                                        doc_collection_lidvid)

        bad_fits_filename = 'x_raw.fits'
        fits_filename = 'p_raw.fits'
        browse_filename = 'p_raw.jpg'
        doc_filename = 'specs.pdf'
        doc_filename_2 = 'specs.txt'

        self.assertFalse(self.db.file_exists(bad_fits_filename,
                                             bad_fits_prod_lidvid))
        self.assertFalse(self.db.file_exists(fits_filename,
                                             fits_prod_lidvid))
        self.assertFalse(self.db.file_exists(browse_filename,
                                             browse_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename,
                                             doc_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename_2,
                                             doc_prod_lidvid))

        self.db.create_bad_fits_file(bad_fits_filename,
                                     bad_fits_prod_lidvid,
                                     'kablooey!')

        self.assertTrue(self.db.file_exists(bad_fits_filename,
                                            bad_fits_prod_lidvid))
        self.assertFalse(self.db.file_exists(fits_filename,
                                             fits_prod_lidvid))
        self.assertFalse(self.db.file_exists(browse_filename,
                                             browse_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename,
                                             doc_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename_2,
                                             doc_prod_lidvid))

        self.db.create_fits_file(fits_filename,
                                 fits_prod_lidvid,
                                 666)

        self.assertTrue(self.db.file_exists(bad_fits_filename,
                                            bad_fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(fits_filename,
                                            fits_prod_lidvid))
        self.assertFalse(self.db.file_exists(browse_filename,
                                             browse_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename,
                                             doc_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename_2,
                                             doc_prod_lidvid))

        self.db.create_browse_file(browse_filename,
                                   browse_prod_lidvid,
                                   666)

        self.assertTrue(self.db.file_exists(bad_fits_filename,
                                            bad_fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(fits_filename,
                                            fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(browse_filename,
                                            browse_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename,
                                             doc_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename_2,
                                             doc_prod_lidvid))

        self.db.create_document_file(doc_filename,
                                     doc_prod_lidvid)

        self.assertTrue(self.db.file_exists(bad_fits_filename,
                                            bad_fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(fits_filename,
                                            fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(browse_filename,
                                            browse_prod_lidvid))
        self.assertTrue(self.db.file_exists(doc_filename,
                                            doc_prod_lidvid))
        self.assertFalse(self.db.file_exists(doc_filename_2,
                                             doc_prod_lidvid))

        self.db.create_document_file(doc_filename_2,
                                     doc_prod_lidvid)

        self.assertTrue(self.db.file_exists(bad_fits_filename,
                                            bad_fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(fits_filename,
                                            fits_prod_lidvid))
        self.assertTrue(self.db.file_exists(browse_filename,
                                            browse_prod_lidvid))
        self.assertTrue(self.db.file_exists(doc_filename,
                                            doc_prod_lidvid))
        self.assertTrue(self.db.file_exists(doc_filename_2,
                                            doc_prod_lidvid))

    def test_bad_fits_file_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        filename = 'p_raw.fits'
        self.assertFalse(self.db.bad_fits_file_exists(filename,
                                                      product_lidvid))

        self.db.create_bad_fits_file(filename, product_lidvid, "kablooey!")
        self.assertTrue(self.db.bad_fits_file_exists(filename,
                                                     product_lidvid))

    def test_browse_file_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        raw_collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        collection_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw::1.8'
        self.db.create_non_document_collection(raw_collection_lidvid,
                                               bundle_lidvid)
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        fits_product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::1.8'
        product_lidvid = 'urn:nasa:pds:hst_99999:browse_acs_raw:p::1.8'
        self.db.create_fits_product(fits_product_lidvid,
                                    raw_collection_lidvid)
        self.db.create_browse_product(product_lidvid,
                                      fits_product_lidvid,
                                      collection_lidvid)
        basename = 'p.jpg'
        self.assertFalse(self.db.browse_file_exists(basename,
                                                    product_lidvid))
        self.db.create_browse_file(basename, product_lidvid, 1024)
        self.assertTrue(self.db.browse_file_exists(basename,
                                                   product_lidvid))

    def test_document_file_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:document::1.8'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:document:p::8.1'
        self.db.create_document_product(product_lidvid, collection_lidvid)

        filename = 'stuff.pdf'
        self.assertFalse(self.db.document_file_exists(filename,
                                                      product_lidvid))

        self.db.create_document_file(filename, product_lidvid)
        self.assertTrue(self.db.document_file_exists(filename,
                                                     product_lidvid))

    def test_fits_file_exists(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        filename = 'p_raw.fits'
        self.assertFalse(self.db.fits_file_exists(filename,
                                                  product_lidvid))

        self.db.create_fits_file(filename, product_lidvid, 1)
        self.assertTrue(self.db.fits_file_exists(filename,
                                                 product_lidvid))

    def test_get_file(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        filename = 'p_raw.fits'
        with self.assertRaises(Exception):
            self.db.get_file(filename,
                             product_lidvid)

        self.db.create_fits_file(filename, product_lidvid, 1)
        self.assertTrue(self.db.get_file(filename,
                                         product_lidvid))

    ############################################################

    def test_get_card_dictionaries(self):
        # type: () -> None

        # We don't test the functionality of get_card_dictionaries()
        # here, since we'd first need to populate the database with
        # the contents of the FITS file and that machinery is in
        # FitsFileDB.  So the real testing happens there.  I leave
        # this empty test as the equivalent of a "this page left
        # intentionally blank" message.
        pass

    def test_exploratory(self):
        # type: () -> None
        # what happens if you create tables twice?
        self.db.create_tables()
        # no exception, at least
