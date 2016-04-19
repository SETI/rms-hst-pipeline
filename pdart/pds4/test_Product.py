import os.path
import unittest

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Bundle import *
from pdart.pds4.Collection import *
from pdart.pds4.HstFilename import *
from pdart.pds4.Product import *
from pdart.pds4.LID import LID


class TestProduct(unittest.TestCase):
    def test_init(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(lid, p.lid)

        # check that creation of bundle fails with collection LID
        lid = LID('urn:nasa:pds:bundle:collection')
        try:
            Collection(arch, lid)
            self.assertTrue(False)
        except Exception:
            pass

    def test_absolute_filepath(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        visit = HstFilename('uproduct.fits').visit()
        self.assertEquals(os.path.join(arch.root, 'bundle',
                                       'collection', 'visit_' + visit,
                                       'uproduct.fits'),
                          p.absolute_filepath())

    def test_bundle(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(Bundle(arch, LID('urn:nasa:pds:bundle')),
                          p.bundle())

    def test_collection(self):
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(Collection(arch,
                                     LID('urn:nasa:pds:bundle:collection')),
                          p.collection())
