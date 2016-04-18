import os.path
import unittest

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Bundle import *
from pdart.pds4.LID import LID


class TestBundle(unittest.TestCase):
    def test_init(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle')
        b = Bundle(arch, lid)
        self.assertEquals(lid, b.lid)

        # check that creation of bundle fails with collection LID
        lid = LID('urn:nasa:pds:bundle:collection')
        try:
            b = Bundle(arch, lid)
            self.assertTrue(False)
        except Exception:
            pass

    def test_absolute_filepath(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle')
        b = Bundle(arch, lid)
        self.assertEquals(os.path.join(arch.root, 'bundle'),
                          b.absolute_filepath())

    def test_collections(self):
        arch = get_any_archive()
        bundle = list(arch.bundles())[0]
        for c in bundle.collections():
            self.assertEquals(bundle, c.bundle())

    def test_products(self):
        arch = get_any_archive()
        bundle = list(arch.bundles())[0]
        for c in bundle.products():
            self.assertEquals(bundle, c.bundle())

    def test_proposal_id(self):
        arch = get_any_archive()
        bundle = list(arch.bundles())[0]
        self.assertEquals(bundle.lid.bundle_id,
                          'hst_%05d' % bundle.proposal_id())
