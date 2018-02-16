import unittest

from pdart.pds4.LID import *


class TestLID(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        # sanity-check
        with self.assertRaises(Exception):
            LID(None)

        # test segments
        with self.assertRaises(Exception):
            LID('urn:nasa')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds')
        LID('urn:nasa:pds:bundle')
        LID('urn:nasa:pds:bundle:container')
        LID('urn:nasa:pds:bundle:container:product')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:bundle:container:product:ingredient')

        # test prefix
        with self.assertRaises(Exception):
            LID('urn:nasa:pdddddds:bundle')

        # test length
        LID('urn:nasa:pds:%s' % ('a' * 200))
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:%s' % ('a' * 250))

        # test characters
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:foo&bar')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:fooBAR')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds::foobar')

        # test fields
        lid = LID('urn:nasa:pds:bundle')
        self.assertEquals('bundle', lid.bundle_id)
        self.assertIsNone(lid.collection_id)
        self.assertIsNone(lid.product_id)
        self.assertEquals('urn:nasa:pds:bundle', lid.lid)

        lid = LID('urn:nasa:pds:bundle:collection')
        self.assertEquals('bundle', lid.bundle_id)
        self.assertEquals('collection', lid.collection_id)
        self.assertIsNone(lid.product_id)
        self.assertEquals('urn:nasa:pds:bundle:collection', lid.lid)

        lid = LID('urn:nasa:pds:bundle:collection:product')
        self.assertEquals('bundle', lid.bundle_id)
        self.assertEquals('collection', lid.collection_id)
        self.assertEquals('product', lid.product_id)
        self.assertEquals('urn:nasa:pds:bundle:collection:product', lid.lid)

    def test_eq(self):
        # type: () -> None
        self.assertTrue(LID('urn:nasa:pds:bundle:collection:product') ==
                        LID('urn:nasa:pds:bundle:collection:product'))
        self.assertFalse(LID('urn:nasa:pds:bundle:collection:product') !=
                         LID('urn:nasa:pds:bundle:collection:product'))
        self.assertFalse(LID('urn:nasa:pds:bundle:collection:product') ==
                         LID('urn:nasa:pds:bundle:collection:produit'))
        self.assertTrue(LID('urn:nasa:pds:bundle:collection:product') !=
                        LID('urn:nasa:pds:bundle:collection:produit'))

    def test_str(self):
        # type: () -> None
        self.assertEquals('urn:nasa:pds:bundle:collection:product',
                          str(LID('urn:nasa:pds:bundle:collection:product')))

    def test_repr(self):
        # type: () -> None
        self.assertEquals("LID('urn:nasa:pds:bundle:collection:product')",
                          repr(LID('urn:nasa:pds:bundle:collection:product')))

    def test_to_browse_lid(self):
        # type: () -> None
        data_coll_lid = LID('urn:nasa:pds:bundle:data_collection_raw')
        browse_coll_lid = LID('urn:nasa:pds:bundle:browse_collection_raw')
        self.assertEquals(browse_coll_lid, data_coll_lid.to_browse_lid())

        data_prod_lid = LID(
            'urn:nasa:pds:bundle:data_collection_raw:data_product')
        browse_prod_lid = LID(
            'urn:nasa:pds:bundle:browse_collection_raw:data_product')
        self.assertEquals(browse_prod_lid, data_prod_lid.to_browse_lid())

        # TODO Write tests for is_bundle_id, etc.

    def test_create_lid_from_parts(self):
        # type: () -> None
        parts = []
        # type: List[str]
        with self.assertRaises(AssertionError):
            LID.create_from_parts(parts)

        parts = ['b']
        self.assertEqual(LID('urn:nasa:pds:b'), LID.create_from_parts(parts))

        parts = ['b', 'c']
        self.assertEqual(LID('urn:nasa:pds:b:c'), LID.create_from_parts(parts))

        parts = ['b', 'c', 'p']
        self.assertEqual(LID('urn:nasa:pds:b:c:p'),
                         LID.create_from_parts(parts))

        parts = ['b', 'c', 'p', 'x']
        with self.assertRaises(AssertionError):
            LID.create_from_parts(parts)
