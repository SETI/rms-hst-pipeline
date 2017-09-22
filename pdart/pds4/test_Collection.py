import unittest

from fs.path import join

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Bundle import *
from pdart.pds4.Collection import *
from pdart.pds4.LID import LID


class TestCollection(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection')
        c = Collection(arch, lid)
        self.assertEquals(lid, c.lid)

        # check that creation of bundle fails with product LID
        lid = LID('urn:nasa:pds:bundle:collection:product')
        try:
            Collection(arch, lid)
            self.assertTrue(False)
        except Exception:
            pass

    def test_absolute_filepath(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection')
        c = Collection(arch, lid)
        self.assertEquals(join(arch.root, 'bundle', 'collection'),
                          c.absolute_filepath())

    def test_bundle(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection')
        c = Collection(arch, lid)
        self.assertEquals(Bundle(arch, LID('urn:nasa:pds:bundle')),
                          c.bundle())

    def test_products(self):
        # type: () -> None
        arch = get_any_archive()
        bundle = list(arch.bundles())[0]
        collection = list(bundle.collections())[0]
        for p in collection.products():
            self.assertEquals(collection, p.collection())

    def test_prefix_instrument_and_suffix(self):
        # type: () -> None
        arch = get_any_archive()
        bundle = list(arch.bundles())[0]
        collection = list(bundle.collections())[0]
        triple = (collection.prefix(),
                  collection.instrument(),
                  collection.suffix())
        self.assertEquals(collection.lid.collection_id,
                          '%s_%s_%s' % triple, str(triple))
