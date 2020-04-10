from typing import TYPE_CHECKING
import unittest
from hypothesis import assume, given
import hypothesis.strategies as st
from pdart.pds4.LID import LID

if TYPE_CHECKING:
    from typing import List

def lid_segments():
    '''A Hypothesis strategy to generate a legal LID segment string.'''
    return st.from_regex(r'[a-z0-9][a-z0-9._-]*', fullmatch=True)

def lid_strings():
    '''A Hypothesis strategy to generate LID strings.'''
    def segments_to_lid(segments):
        segments = ['urn', 'nasa', 'pds'] + segments
        res = ':'.join(segments)
        assume(len(res) <= 255)
        return res

    return st.lists(lid_segments(),
                    min_size=1,
                    max_size=3).map(segments_to_lid)

def lids():
    '''A Hypothesis strategy to generate LIDs.'''
    return st.builds(LID, lid_strings())

class TestLID(unittest.TestCase):
    def test_init(self):
        # type: () -> None

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

    @given(lid_strings(), lid_strings())
    def test_eq_property(self, lhs, rhs):
        # two LIDs are equal iff their strings are equal
        self.assertEqual(lhs == rhs, LID(lhs) == LID(rhs))

    def test_str(self):
        # type: () -> None
        self.assertEquals('urn:nasa:pds:bundle:collection:product',
                          str(LID('urn:nasa:pds:bundle:collection:product')))

    @given(lid_strings())
    def test_str_roundtrip_property(self, lid_str):
        '''
        Creating a LID from a string and turning it back into a string
        should result in the same string.
        '''
        self.assertEquals(lid_str, str(LID(lid_str)))

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

    def test_to_shm_lid(self):
        # type: () -> None
        data_coll_lid = LID('urn:nasa:pds:bundle:data_collection_raw:product')
        shm_coll_lid = LID('urn:nasa:pds:bundle:data_collection_shm:product')
        self.assertEquals(shm_coll_lid, data_coll_lid.to_shm_lid())

    def test_create_lid_from_parts(self):
        # type: () -> None
        parts = []  # type: List[str]
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

    @given(lids())
    def test_is_xxx_lid_property(self, lid):
        # LIDs must be either for bundles, collections, or products.
        if lid.is_bundle_lid():
            self.assertIsNotNone(lid.bundle_id)
            self.assertIsNone(lid.collection_id)
            self.assertIsNone(lid.product_id)
            self.assertFalse(lid.is_collection_lid())
            self.assertFalse(lid.is_product_lid())
        elif lid.is_collection_lid():
            self.assertIsNotNone(lid.bundle_id)
            self.assertIsNotNone(lid.collection_id)
            self.assertIsNone(lid.product_id)
            self.assertFalse(lid.is_bundle_lid())
            self.assertFalse(lid.is_product_lid())
        elif lid.is_product_lid():
            self.assertIsNotNone(lid.bundle_id)
            self.assertIsNotNone(lid.collection_id)
            self.assertIsNotNone(lid.product_id)
            self.assertFalse(lid.is_bundle_lid())
            self.assertFalse(lid.is_collection_lid())
        else:
            self.fail(
                'One of is_bundle_lid(), is_collection_lid(), '
                'or is_product_lid() must hold')
