import re
import unittest


class LID(object):
    def __init__(self, str):
        ids = str.split(':')

        # Check requirements
        assert len(str) <= 255
        assert len(ids) in [4, 5, 6]
        assert ids[0] == 'urn'
        assert ids[1] == 'nasa'
        assert ids[2] == 'pds'
        allowedCharsRE = '\\A[-._a-z0-9]+\\Z'
        for id in ids:
            assert re.match(allowedCharsRE, id)

        # Assign the Id fields
        self.LID = str
        self.bundleId = ids[3]

        # Now we modify ids to include possibly missing Ids...
        while len(ids) < 6:
            ids.append(None)

        # ...so this indexing of ids is safe
        self.collectionId = ids[4]
        self.productId = ids[5]

    def __eq__(self, other):
        return self.LID == other.LID

    def __ne__(self, other):
        return self.LID != other.LID

    def __str__(self):
        return self.LID

    def __repr__(self):
        return 'LID(%s)' % repr(self.LID)

    def isProductLID(self):
        return self.productId is not None

    def isCollectionLID(self):
        return self.collectionId is not None and self.productId is None

    def isBundleLID(self):
        return self.bundleId is not None and self.collectionId is None

############################################################


class TestLID(unittest.TestCase):
    def testInit(self):
        # sanity-check
        with self.assertRaises(Exception):
            LID(null)

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
        LID('urn:nasa:pds:%s' % ('a'*200))
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:%s' % ('a'*250))

        # test characters
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:foo&bar')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds:fooBAR')
        with self.assertRaises(Exception):
            LID('urn:nasa:pds::foobar')

        # test fields
        lid = LID('urn:nasa:pds:bundle')
        self.assertEquals('bundle', lid.bundleId)
        self.assertIsNone(lid.collectionId)
        self.assertIsNone(lid.productId)
        self.assertEquals('urn:nasa:pds:bundle', lid.LID)

        lid = LID('urn:nasa:pds:bundle:collection')
        self.assertEquals('bundle', lid.bundleId)
        self.assertEquals('collection', lid.collectionId)
        self.assertIsNone(lid.productId)
        self.assertEquals('urn:nasa:pds:bundle:collection', lid.LID)

        lid = LID('urn:nasa:pds:bundle:collection:product')
        self.assertEquals('bundle', lid.bundleId)
        self.assertEquals('collection', lid.collectionId)
        self.assertEquals('product', lid.productId)
        self.assertEquals('urn:nasa:pds:bundle:collection:product', lid.LID)

    def testEq(self):
        self.assertTrue(LID('urn:nasa:pds:bundle:collection:product') ==
                        LID('urn:nasa:pds:bundle:collection:product'))
        self.assertFalse(LID('urn:nasa:pds:bundle:collection:product') !=
                         LID('urn:nasa:pds:bundle:collection:product'))
        self.assertFalse(LID('urn:nasa:pds:bundle:collection:product') ==
                         LID('urn:nasa:pds:bundle:collection:produit'))
        self.assertTrue(LID('urn:nasa:pds:bundle:collection:product') !=
                        LID('urn:nasa:pds:bundle:collection:produit'))

    def testStr(self):
        self.assertEquals('urn:nasa:pds:bundle:collection:product',
                          str(LID('urn:nasa:pds:bundle:collection:product')))

    def testRepr(self):
        self.assertEquals("LID('urn:nasa:pds:bundle:collection:product')",
                          repr(LID('urn:nasa:pds:bundle:collection:product')))

    # TODO Write tests for isBundleId, etc.

if __name__ == '__main__':
    unittest.main()
